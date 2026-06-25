"""Main organization pipeline."""
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image

from .settings import load_settings, save_settings, resolve_dest_dir, PROJECT_DIR
from .parallel import (
    cpu_count,
    estimate_eta,
    get_io_executor,
    is_free_threaded,
    run_parallel,
)
from .formats import (
    STATIC_EXTENSIONS, ANIMATED_EXTENSIONS, PLUGIN_EXTENSIONS,
    WALLPAPERS_DIR,
    supported_image_files,
)
from . import formats as _formats_mod
from . import categories as c
from . import tags as t
from .duplicates import (
    load_hash_cache, save_hash_cache,
    find_duplicate_groups,
    scan_and_hash, scan_and_hash_perceptual,
    find_md5_duplicate_groups,
)
from .quality import laplacian_variance
from .classify import classify
from .analyzers import get_analyzer as _analyzer_factory

LOW_QUALITY_FOLDER = "Low-Quality"
NSFW_FOLDER = "NSFW"
DISCARDED_FOLDER = "Discarded"
MAX_FILE_SIZE = 100 * 1024 * 1024
LAPLACIAN_MIN = 80.0
QUALITY_MIN_DEFAULT = 0.0
SKIP_DIRS: Set[str] = {".git", "__pycache__", "Duplicates",
                        LOW_QUALITY_FOLDER, NSFW_FOLDER, DISCARDED_FOLDER}


def get_analyzer(mode: str, settings: dict):
    """Factory: returns the appropriate analyzer for the given mode.

    Thin re-export of `wallpaper_analyzer.analyzers.get_analyzer` kept
    here for backwards compatibility with existing callers (and tests).
    """
    return _analyzer_factory(mode, settings)


def _extract_frame_if_video(path: str, max_size: int = 1024) -> Tuple[str, bool]:
    """If the path is a video or animated image, extract a single frame.

    Returns (path_to_use, is_temporary). If a temp file is created,
    the caller is responsible for deleting it.
    """
    ext = os.path.splitext(path)[1].lower()
    video_exts = {".mp4", ".m4v", ".m2v", ".webm", ".mkv", ".avi", ".mov",
                  ".flv", ".mpg", ".mpeg", ".mpe", ".mpv", ".3gp", ".3gpp",
                  ".ogv", ".wmv", ".asf", ".ts", ".m2ts", ".mts", ".vob",
                  ".rm", ".rmvb", ".ogm"}
    animated_image_exts = {".gif", ".apng", ".mng", ".fli", ".flc"}
    needs_extract = ext in video_exts or ext in animated_image_exts or ext == ".webp"
    if not needs_extract:
        return path, False
    if ext in video_exts and shutil.which("ffmpeg") is None:
        # Cannot handle without ffmpeg
        return path, False
    try:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        if ext in video_exts:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                 "-vframes", "1", "-vf", f"scale='min({max_size},iw)':-1", tmp],
                capture_output=True, timeout=15,
            )
        else:
            # Animated image: open and save first frame
            im = Image.open(path)
            if getattr(im, "is_animated", False) or getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size), Image.LANCZOS)
            im.save(tmp, "PNG", optimize=True)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            return tmp, True
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
    return path, False


def _extract_suggested_tags(profile: dict) -> Set[str]:
    """Extract free-form tags from a profile, not limited to the registry.

    Combines heuristic-derived tags with any AI-suggested tags found
    in the profile (from Ollama description or analysis).
    """
    from .classify import derive_tags_from_profile
    from .tags import _tags_flat
    valid_tags = _tags_flat

    # Get heuristic tags
    suggested = derive_tags_from_profile(profile)

    # Extract keywords from Ollama description if available
    desc = profile.get("ollama_description", "")
    if desc:
        import re
        # Extract meaningful words (3+ chars, alpha only)
        words = re.findall(r"[a-z][a-z\-]{2,20}", desc.lower())
        # Match against valid tags
        for w in words:
            if w in valid_tags:
                suggested.add(w)
            else:
                # Check if any valid tag is a substring/contains
                for tag in valid_tags:
                    if len(tag) >= 3 and (w in tag or tag in w):
                        suggested.add(tag)
                        break

    return suggested


def _detect_tags_for_file(fpath: str, profile: dict) -> Tuple[List[str], Optional[str]]:
    """Return (top_tags, main_subject) for the file.

    Sources (in priority order):
      1. Ollama structured analysis tags (`ollama_all_tags` + `ollama_subject`)
      2. Heuristic content tags (using tag_suggester)
    Module-level so it is picklable for ProcessPoolExecutor.
    """
    tags = list(profile.get("ollama_all_tags") or [])
    subject = profile.get("ollama_subject") or profile.get("main_subject")
    if not tags and not subject:
        try:
            from .tag_suggester import suggest_tags_for_category
            guessed = suggest_tags_for_category(
                profile.get("_current_category") or "default",
                profile, max_tags=8,
            )
            if guessed:
                tags = list(guessed)[:8]
        except Exception:
            pass
    # Final fallback: top 3 color names so we always produce something
    if not tags:
        weights = profile.get("weights") or {}
        top_colors = sorted(weights.items(), key=lambda kv: -kv[1])[:3]
        tags = [k.lower() for k, _ in top_colors if k]
    return tags[:8], (subject or None)


def _learn_tags_for_category(category: str, profile: dict) -> int:
    """Add AI-suggested tags to the category's tag list.

    Returns the number of new tags added. Updates the .category.json
    so categories accumulate learned tags over time.

    For NSFW categories, applies the default NSFW tag set (from settings)
    so the description-based tag gap from censored models is filled with
    the configured defaults.
    """
    if not category or category in (LOW_QUALITY_FOLDER, DISCARDED_FOLDER, "Uncategorized"):
        return 0
    if not os.path.isdir(os.path.join(c.CATEGORIES_DIR, category)):
        return 0

    suggested = _extract_suggested_tags(profile)
    if not suggested:
        suggested = set()

    # NSFW: prepend default tags from settings (always included)
    if category == NSFW_FOLDER:
        try:
            from .settings import load_settings
            defaults = load_settings().get("nsfw_default_tags", [])
        except Exception:
            defaults = ["nsfw", "figure", "human", "skin", "portrait", "person", "body"]
        for t in defaults:
            suggested.add(t.lower())

    # Also use the profile's nsfw_default_tags if present
    for t in profile.get("nsfw_default_tags", []):
        suggested.add(t.lower())

    if not suggested:
        return 0

    cfg = c.get_category_config(category)
    existing = set(t.lower() for t in cfg.get("tags", []))
    new_tags = suggested - existing
    if not new_tags:
        return 0

    cfg["tags"] = sorted(existing | new_tags)
    c.write_category_config(category, cfg)
    return len(new_tags)


def flatten_all(include_category_dirs: bool = False):
    """Move all image files from subdirectories back into the source root."""
    global WALLPAPERS_DIR
    WALLPAPERS_DIR = _formats_mod.WALLPAPERS_DIR
    print("Flattening directory structure...")
    skip_dirs = set(SKIP_DIRS)
    if not include_category_dirs:
        skip_dirs |= set(c.CATEGORIES)
    moved = 0
    collisions = 0
    for dirpath, dirnames, filenames in os.walk(WALLPAPERS_DIR):
        rel = os.path.relpath(dirpath, WALLPAPERS_DIR)
        if rel == ".": continue
        parts = rel.split(os.sep)
        if any(p.startswith(".") or p in skip_dirs for p in parts):
            continue
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (STATIC_EXTENSIONS | ANIMATED_EXTENSIONS | PLUGIN_EXTENSIONS):
                continue
            src = os.path.join(dirpath, fname)
            dst = os.path.join(WALLPAPERS_DIR, fname)
            if os.path.exists(dst):
                parent_name = os.path.basename(os.path.dirname(src))
                name_part, ext_part = os.path.splitext(fname)
                candidate = f"{parent_name}_{name_part}{ext_part}"
                if os.path.exists(os.path.join(WALLPAPERS_DIR, candidate)):
                    name2, ext2 = os.path.splitext(candidate)
                    candidate = f"{name2}_1{ext2}"
                dst = os.path.join(WALLPAPERS_DIR, candidate)
                collisions += 1
            shutil.move(src, dst)
            moved += 1
        try:
            os.rmdir(dirpath)
        except OSError:
            pass
    print(f"  Moved: {moved} files")
    if collisions:
        print(f"  Renamed (collisions): {collisions}")


# ---------------------------------------------------------------------------
# Worker function (module-level so it's picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _classify_worker(payload: Dict) -> Dict:
    """Worker entry point: analyse + classify a single file.

    `payload` contains everything we need (settings snapshot, file path,
    thresholds, parallel flags) so this function is fully picklable and
    can run in a separate Python process via ProcessPoolExecutor.

    Returns a dict with: {fname, fpath, category, extra_note,
    tags, subject, error?}. The caller (main process) aggregates results.
    """
    fpath = payload["fpath"]
    fname = payload["fname"]
    orig_i = payload["orig_i"]
    total_files = payload["total_files"]
    mode = payload["mode"]
    quality_min = float(payload.get("quality_min") or 0.0)
    rename_strategy = payload.get("rename_strategy") or "none"
    rename_use_tags = bool(payload.get("rename_use_tags"))
    rename_max_tags = int(payload.get("rename_max_tags") or 3)
    settings = payload["settings"]

    # Re-import inside the worker (each process has its own interpreter
    # state and the imports here are cheap once numpy/PIL are loaded).
    from PIL import Image  # noqa: F401  (already imported at top)
    from . import categories as c
    from .classify import classify, classify_with_confidence
    from .lowlevel.category_profile import score_against_pattern  # noqa: F401

    category: Optional[str] = None
    extra_note = ""
    temp_frame: Optional[str] = None
    file_tags: List[str] = []
    file_subject: Optional[str] = None
    error: Optional[str] = None

    try:
        fsize = os.path.getsize(fpath)
        if fsize > MAX_FILE_SIZE:
            category = DISCARDED_FOLDER
            extra_note = f"{fsize/(1024*1024):.1f} MB > 100 MB"

        if category is None:
            analyze_path, is_temp = _extract_frame_if_video(fpath)
            if is_temp:
                temp_frame = analyze_path
            analyzer = get_analyzer(mode, settings)
            profile = analyzer.analyze(analyze_path)

            sharpness = profile.get("sharpness", 0.0)
            if sharpness < LAPLACIAN_MIN:
                category = LOW_QUALITY_FOLDER
                extra_note = f"sharpness={sharpness:.1f}"
            elif quality_min > 0 and profile.get("aesthetic", 0.0) < quality_min:
                category = LOW_QUALITY_FOLDER
                extra_note = f"aesthetic={profile.get('aesthetic', 0.0):.2f}"

            if category is None:
                cv_nsfw = profile.get("nsfw", 0.0)
                ollama_nsfw = profile.get("ollama_nsfw")
                use_ollama_nsfw = settings.get("ollama_nsfw_use", True)
                cv_threshold = settings.get("nsfw_threshold", 0.5)
                ollama_threshold = settings.get("ollama_nsfw_threshold", 0.70)
                uncertain_ceiling = settings.get("ollama_nsfw_uncertain_ceiling", 0.65)

                cv_trigger = float(cv_nsfw) >= cv_threshold
                ollama_trigger = False
                if (
                    use_ollama_nsfw
                    and ollama_nsfw is not None
                    and float(ollama_nsfw) >= uncertain_ceiling
                ):
                    ollama_trigger = float(ollama_nsfw) >= ollama_threshold

                if ollama_trigger or cv_trigger:
                    category = NSFW_FOLDER
                    if ollama_trigger and cv_trigger:
                        extra_note = f"nsfw=cv{cv_nsfw:.2f}+ollama{ollama_nsfw:.2f}"
                    elif ollama_trigger:
                        extra_note = f"nsfw=ollama{ollama_nsfw:.2f}"
                    else:
                        extra_note = f"nsfw=cv{cv_nsfw:.2f}"

            if category is None:
                cat = None
                classify_method = "heuristic"
                if hasattr(analyzer, "classify_direct"):
                    try:
                        cat = analyzer.classify_direct(analyze_path)
                        if cat:
                            classify_method = f"mode={mode}"
                    except Exception as exc:
                        extra_note += f" direct_err:{type(exc).__name__}"
                if not cat:
                    try:
                        cat = analyzer.classify(profile)
                        if cat:
                            classify_method = f"mode={mode}"
                    except Exception as exc:
                        extra_note += f" err:{type(exc).__name__}"
                if cat:
                    category = cat
                    extra_note = (extra_note + " " + classify_method).strip()
                else:
                    category = classify(profile)
                    extra_note = (extra_note + " heuristic").strip()

            # Collect tags + subject for tag-based rename
            if rename_strategy and rename_use_tags and category:
                profile["_current_category"] = category
                file_tags, file_subject = _detect_tags_for_file(fpath, profile)
                if file_tags:
                    extra_note += f" tags={','.join(file_tags[:rename_max_tags])}"

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        category = c.CATEGORIES[0] if c.CATEGORIES else "Uncategorized"

    finally:
        if temp_frame and os.path.exists(temp_frame):
            try:
                os.remove(temp_frame)
            except Exception:
                pass

    return {
        "orig_i": orig_i,
        "fname": fname,
        "fpath": fpath,
        "category": category,
        "extra_note": extra_note,
        "tags": file_tags,
        "subject": file_subject,
        "error": error,
    }


# ---------------------------------------------------------------------------
# organise() pipeline
# ---------------------------------------------------------------------------


def organize(mode: str = "lowlevel",
             dry_run: bool = False,
             dedupe: bool = True,
             video_ffmpeg: bool = True,
             quality_min: float = QUALITY_MIN_DEFAULT,
             by_resolution: bool = False,
             resolution_bins: Optional[List[Dict]] = None,
             parallel: int = 1,
             progress_callback=None,
             should_cancel=None,
             rename_strategy: str = "none",
             rename_category_prefix: bool = True,
             rename_use_tags: bool = True,
             rename_max_tags: int = 3):
    """Main organization pipeline with pluggable analyzer modes."""
    global WALLPAPERS_DIR
    WALLPAPERS_DIR = _formats_mod.WALLPAPERS_DIR
    s = load_settings()
    s["organize_mode"] = mode
    save_settings(s)

    dest_dir = resolve_dest_dir(s)
    os.makedirs(dest_dir, exist_ok=True)
    c.discover_categories(dest_dir)

    analyzer = get_analyzer(mode, s)

    files = supported_image_files(WALLPAPERS_DIR, include_animations=True)
    print(f"Mode: {mode.upper()}", flush=True)
    print(f"Destination: {dest_dir}", flush=True)
    print(f"Source files: {len(files)}", flush=True)
    print(flush=True)

    if not files:
        print("No files to organize.")
        return

    total_files = len(files)

    if dedupe:
        cache = load_hash_cache()
        all_files: List[str] = []
        # Scan destination (already organized files)
        if os.path.isdir(dest_dir):
            for d, _, fns in os.walk(dest_dir):
                rel = os.path.relpath(d, dest_dir)
                parts = rel.split(os.sep)
                # Skip hidden dirs and special folders, but NOT the root
                if rel != "." and any(p.startswith(".") or p in SKIP_DIRS for p in parts):
                    continue
                for fn in fns:
                    if os.path.splitext(fn)[1].lower() in (STATIC_EXTENSIONS | ANIMATED_EXTENSIONS):
                        all_files.append(os.path.join(d, fn))
        # Scan source (files to be organized)
        for d, _, fns in os.walk(WALLPAPERS_DIR):
            rel = os.path.relpath(d, WALLPAPERS_DIR)
            parts = rel.split(os.sep)
            if rel != "." and any(p.startswith(".") or p in SKIP_DIRS for p in parts):
                continue
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in (STATIC_EXTENSIONS | ANIMATED_EXTENSIONS):
                    all_files.append(os.path.join(d, fn))
        print(f"[DEDUPE] Hashing {len(all_files)} files...", flush=True)
        import time as _time
        _dedupe_t0 = _time.time()
        _last_save = [0]
        _last_emit = [0]
        def _dedupe_progress(cur, total, fn, st):
            # Always forward to GUI progress signal so the progress bar
            # advances even when stdout buffering would delay the print.
            if progress_callback and (cur - _last_emit[0] >= 25 or cur == total):
                try:
                    progress_callback("progress", cur, total,
                                      os.path.basename(fn), "dedupe")
                except Exception:
                    pass
                _last_emit[0] = cur
            if cur % 50 == 0 or cur == total:
                elapsed = _time.time() - _dedupe_t0
                rate = cur / max(elapsed, 0.1)
                eta = (total - cur) / max(rate, 0.1)
                print(f"  [{cur}/{total}] {os.path.basename(fn)} ({rate:.0f}/s, ~{eta:.0f}s left)", flush=True)
            # Save cache incrementally every 100 files
            if cur - _last_save[0] >= 100:
                save_hash_cache(cache)
                _last_save[0] = cur

        # ---- Fast MD5 pre-pass (full-file, cached) ----
        # Catches byte-identical duplicates cheaply (just I/O, no image
        # decoding). The MD5 is stored on the cache entry so repeat
        # runs of `organize` are essentially free.
        _md5_t0 = _time.time()
        print(f"[DEDUPE-MD5] Computing full-file MD5 for {len(all_files)} files...", flush=True)
        _md5_last_save = [0]
        _md5_last_emit = [0]
        def _md5_progress(cur, total, fn, st):
            if progress_callback and (cur - _md5_last_emit[0] >= 25 or cur == total):
                try:
                    progress_callback("progress", cur, total,
                                      os.path.basename(fn), "dedupe-md5")
                except Exception:
                    pass
                _md5_last_emit[0] = cur
            if cur % 50 == 0 or cur == total:
                elapsed = _time.time() - _md5_t0
                rate = cur / max(elapsed, 0.1)
                eta = (total - cur) / max(rate, 0.1)
                print(f"  [{cur}/{total}] {os.path.basename(fn)} ({rate:.0f}/s, ~{eta:.0f}s left)", flush=True)
            if cur - _md5_last_save[0] >= 200:
                save_hash_cache(cache)
                _md5_last_save[0] = cur
        scan_and_hash(
            all_files, cache,
            progress_callback=_md5_progress,
            save_every=200,
            parallel=max(1, parallel),
        )
        # Extract MD5s from the now-populated cache.
        md5s = {f: cache[f]["md5"] for f in all_files
                if f in cache and cache[f].get("md5")}
        md5_groups = find_md5_duplicate_groups(all_files, md5s)
        if md5_groups:
            dest_abs = os.path.abspath(dest_dir)
            source_abs = os.path.abspath(WALLPAPERS_DIR)
            dest_set = {
                os.path.abspath(f) for f in all_files
                if os.path.abspath(f).startswith(dest_abs)
            }
            source_set = {
                os.path.abspath(f) for f in all_files
                if os.path.abspath(f).startswith(source_abs)
            }
            md5_to_move: List[str] = []
            for g in md5_groups:
                group_abs = [os.path.abspath(f) for f in g["files"]]
                has_dest = any(f in dest_set for f in group_abs)
                has_source = any(f in source_set for f in group_abs)
                if not has_source:
                    continue
                if has_dest:
                    # Keep one destination copy, drop the rest of source.
                    keep = next(
                        (f for f in g["files"]
                         if os.path.abspath(f) in dest_set),
                        g["files"][0],
                    )
                    for f in g["files"]:
                        if f == keep:
                            continue
                        if os.path.abspath(f) in source_set:
                            md5_to_move.append(f)
                else:
                    # Duplicates only in source: keep the largest, drop rest.
                    for f in g["files"][1:]:
                        md5_to_move.append(f)
            if md5_to_move:
                total_dupes = sum(
                    len(g["files"]) - 1 for g in md5_groups
                )
                print(
                    f"[DEDUPE-MD5] Found {len(md5_groups)} exact-duplicate "
                    f"groups ({total_dupes} files)",
                    flush=True,
                )
                if dry_run:
                    print(
                        f"[DRY] Would remove {len(md5_to_move)} "
                        f"exact duplicate(s) from source.",
                        flush=True,
                    )
                else:
                    from .duplicates import move_to_duplicates as _move_to_dupes
                    moved = _move_to_dupes(
                        md5_to_move,
                        dest_dir if os.path.isdir(dest_dir)
                        else WALLPAPERS_DIR,
                    )
                    print(
                        f"[DEDUPE-MD5] Moved {moved} exact duplicate(s) "
                        f"to Duplicates/ (saved from "
                        f"{_time.time() - _md5_t0:.1f}s of perceptual hashing)",
                        flush=True,
                    )
                    files = supported_image_files(
                        WALLPAPERS_DIR, include_animations=True,
                    )
                    # Rebuild all_files to reflect the removed duplicates
                    # so the perceptual pass below doesn't waste work on them.
                    all_files = [
                        f for f in all_files if os.path.exists(f)
                    ]
                    print(flush=True)
            else:
                print("[DEDUPE-MD5] No source-side duplicates found", flush=True)
        else:
            print("[DEDUPE-MD5] No exact duplicates found", flush=True)

        # ---- Perceptual pass (image-content-aware) ----
        # Catches "same image, different bytes" duplicates (re-encoded,
        # re-compressed, resized, different container). Uses multi-perceptual
        # hashes (dHash, pHash, aHash, color hash, RGB histogram) bucketed
        # via LSH on the 16-bit coarse dHash so we only verify candidates.
        s_for_dedupe = s
        perceptual_enabled = bool(s_for_dedupe.get("dedupe_perceptual", True))
        if perceptual_enabled:
            _perc_t0 = _time.time()
            print(
                f"[DEDUPE-PERC] Computing perceptual hashes for "
                f"{len(all_files)} files...",
                flush=True,
            )
            _perc_last_save = [0]
            _perc_last_emit = [0]
            def _perc_progress(cur, total, fn, st):
                if progress_callback and (cur - _perc_last_emit[0] >= 25 or cur == total):
                    try:
                        progress_callback("progress", cur, total,
                                          os.path.basename(fn), "dedupe-perceptual")
                    except Exception:
                        pass
                    _perc_last_emit[0] = cur
                if cur % 50 == 0 or cur == total:
                    elapsed = _time.time() - _perc_t0
                    rate = cur / max(elapsed, 0.1)
                    eta = (total - cur) / max(rate, 0.1)
                    print(
                        f"  [{cur}/{total}] {os.path.basename(fn)} "
                        f"({rate:.0f}/s, ~{eta:.0f}s left)",
                        flush=True,
                    )
                if cur - _perc_last_save[0] >= 200:
                    save_hash_cache(cache)
                    _perc_last_save[0] = cur
            cache = scan_and_hash_perceptual(
                all_files, cache,
                progress_callback=_perc_progress,
                save_every=200,
                parallel=max(1, parallel),
            )
            print(
                f"[DEDUPE-PERC] Perceptual pass done in "
                f"{_time.time() - _perc_t0:.1f}s",
                flush=True,
            )

        dedupe_min_tier = s_for_dedupe.get("dedupe_min_tier", "reencode")
        groups = find_duplicate_groups(
            all_files, cache,
            mode='soft',
            min_tier=dedupe_min_tier,
        )
        save_hash_cache(cache)
        # Separate: duplicates within destination vs cross-directory
        dest_abs = os.path.abspath(dest_dir)
        source_abs = os.path.abspath(WALLPAPERS_DIR)
        dest_files = set(os.path.abspath(f) for f in all_files if os.path.abspath(f).startswith(dest_abs))
        source_files = set(os.path.abspath(f) for f in all_files if os.path.abspath(f).startswith(source_abs))
        to_move_to_dupes: List[str] = []
        for g in groups:
            group_files = [os.path.abspath(f) for f in g["files"]]
            has_dest = any(f in dest_files for f in group_files)
            has_source = any(f in source_files for f in group_files)
            if has_source:
                if has_dest:
                    # Source file is duplicate of something already organized
                    # Keep destination copy, move source copy to Duplicates/
                    for f in g["files"][1:]:  # skip highest-res (first)
                        if os.path.abspath(f) in source_files:
                            to_move_to_dupes.append(f)
                elif len(group_files) > 1:
                    # Duplicates within source only
                    for f in g["files"][1:]:
                        if os.path.abspath(f) in source_files:
                            to_move_to_dupes.append(f)
        if to_move_to_dupes:
            if dry_run:
                print(f"[DRY] Would remove {len(to_move_to_dupes)} duplicate(s) from source.", flush=True)
            else:
                from .duplicates import move_to_duplicates as _move_to_dupes
                moved = _move_to_dupes(to_move_to_dupes, dest_dir if os.path.isdir(dest_dir) else WALLPAPERS_DIR)
                print(f"[DEDUPE] Moved {moved} duplicate(s) to Duplicates/", flush=True)
                files = supported_image_files(WALLPAPERS_DIR, include_animations=True)
                print(flush=True)
        else:
            print("[DEDUPE] No duplicates found", flush=True)

    stats: Dict[str, int] = {}
    classified: List[Tuple[str, str, Dict]] = []

    def _emit(stage, cur=0, total=None, filename="", extra=""):
        if progress_callback:
            try:
                progress_callback(stage, cur, total or total_files, filename, extra)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Parallel classification
    # ------------------------------------------------------------------
    # Strategy: build a flat list of "payloads" (one per file) that
    # contain everything `_classify_worker` needs. Run them through
    # `run_parallel` which uses ProcessPoolExecutor (true parallelism,
    # no GIL contention) when `parallel > 1` and there's more than one
    # file. Aggregate results back into `stats` and `classified`.
    # ------------------------------------------------------------------

    _emit("start", 0, "", f"mode={mode}")

    settings_snapshot = dict(s)  # shallow copy for pickling

    entries: List[Tuple[int, str, str]] = []
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(WALLPAPERS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        entries.append((i, fname, fpath))

    payloads = [
        {
            "orig_i": i,
            "fname": fname,
            "fpath": fpath,
            "total_files": total_files,
            "mode": mode,
            "quality_min": quality_min,
            "rename_strategy": rename_strategy,
            "rename_use_tags": rename_use_tags,
            "rename_max_tags": rename_max_tags,
            "settings": settings_snapshot,
        }
        for (i, fname, fpath) in entries
    ]

    n_workers = min(parallel, len(payloads)) if parallel > 1 else 1
    use_processes = n_workers > 1 and len(payloads) > 1

    # CLIP / fusion mode special case: spawning N processes would each
    # load the full CLIP model from disk into RAM (RN50x64 is 2GB!),
    # which means 4 workers = 8GB of model weights + 4x the load time.
    # The CPU forward pass is GIL-friendly enough that threads share
    # the model efficiently, so for AI modes we force a single worker
    # in the main process. The model load already happened above so
    # the user sees immediate file-by-file progress.
    if mode in ("clip", "fusion"):
        n_workers = 1
        use_processes = False

    # Tell the user what we're about to do so the GUI log isn't silent
    # while workers load models.
    if mode in ("clip", "fusion"):
        try:
            cmodel = s.get("clip_model", "ViT-B/32")
        except Exception:
            cmodel = "ViT-B/32"
        print(
            f"[AI] mode={mode}, workers={n_workers}, model={cmodel}. "
            f"Loading model + classifying {len(payloads)} files...",
            flush=True,
        )
        # Pre-load the engine in the main process so the model load
        # is visible in the GUI log (rather than 4 silent workers)
        # and so the single-worker run below doesn't block on the
        # first image.
        try:
            from .clip_client import get_engine
            t_load = time.monotonic()
            eng = get_engine(cmodel)
            _ = eng.device  # triggers lazy load
            print(
                f"[AI] Model {cmodel} ready on {eng.device} "
                f"({time.monotonic() - t_load:.1f}s)",
                flush=True,
            )
        except Exception as exc:
            print(f"[AI] WARN: CLIP preload failed: {exc}", flush=True)
    else:
        print(
            f"[AI] mode={mode}, workers={n_workers}, "
            f"classifying {len(payloads)} files...",
            flush=True,
        )

    # Show immediate progress so the bar starts moving right away.
    _emit("progress", 0, total_files, "", "starting")

    run_t0 = time.monotonic()

    def _on_progress(done: int, total: int, item, result):
        # Stream progress to stdout (CLI mode) AND forward to GUI callback.
        if result is None:
            return
        category = (result.get("category")
                    or (c.CATEGORIES[0] if c.CATEGORIES else "Uncategorized"))
        fname = result.get("fname", item["fname"])
        orig_i = result.get("orig_i", item["orig_i"])
        extra_note = result.get("extra_note", "")
        file_tags = result.get("tags") or []
        file_subject = result.get("subject")
        icon = f"[{category[:3].upper()}]" if category else "[???]"
        note = f"  ({extra_note})" if extra_note else ""
        print(f"  [{orig_i}/{total_files}] {fname}... {icon} {category}{note}",
              flush=True)
        # Update main-process aggregates incrementally
        classified.append((fname, category, {"tags": file_tags, "subject": file_subject}))
        stats[category] = stats.get(category, 0) + 1
        if category not in (LOW_QUALITY_FOLDER, DISCARDED_FOLDER, NSFW_FOLDER, "Uncategorized"):
            try:
                _learn_tags_for_category(category, {"ollama_all_tags": file_tags})
            except Exception:
                pass
        # Persist Ollama tags into the hash cache so the Reorder page's
        # AIRenamer can read them and produce renames that mirror the
        # classification log format (ABBR_Category_tag1-tag2-...).
        if file_tags and mode in ("ollama", "fusion"):
            try:
                _cache_path = item.get("fpath") or os.path.join(WALLPAPERS_DIR, fname)
                _cache_entry = cache.get(_cache_path) or {"sig": "", "md5": ""}
                _cache_entry["ollama_all_tags"] = list(file_tags)
                if file_subject:
                    _cache_entry["ollama_subject"] = file_subject
                cache[_cache_path] = _cache_entry
                # Throttle saves so we don't hammer the disk.
                if total_files <= 50 or stats.get(category, 0) % 25 == 0:
                    save_hash_cache(cache)
            except Exception:
                pass
        if progress_callback:
            try:
                progress_callback("progress", done, total, fname, category)
            except Exception:
                pass

    def _on_error(item, exc):
        fname = item["fname"]
        orig_i = item["orig_i"]
        print(f"  [{orig_i}/{total_files}] {fname}... [ERR] {exc}", flush=True)
        # Still count the failed file so the aggregate stays consistent
        fallback_cat = c.CATEGORIES[0] if c.CATEGORIES else "Uncategorized"
        classified.append((fname, fallback_cat, {"tags": [], "subject": None}))
        stats[fallback_cat] = stats.get(fallback_cat, 0) + 1

    try:
        results = run_parallel(
            _classify_worker,
            payloads,
            max_workers=n_workers,
            on_progress=_on_progress,
            on_error=_on_error,
            use_processes=use_processes,
        )
    except Exception as exc:
        err(f"Parallel classification crashed: {exc}")
        raise

    elapsed = time.monotonic() - run_t0
    if elapsed > 0:
        rate = len(classified) / max(elapsed, 0.01)
        print(
            f"  -> Classified {len(classified)} files in {elapsed:.1f}s "
            f"({rate:.1f}/s, free-threaded={is_free_threaded()}, "
            f"workers={n_workers}, processes={use_processes})",
            flush=True,
        )

    print(f"\n{'=' * 60}")
    print(f"CLASSIFICATION SUMMARY (mode: {mode}):\n")
    for cat in sorted(set(list(c.CATEGORIES) + list(stats.keys()))):
        count = stats.get(cat, 0)
        if count > 0:
            print(f"  {cat:20s} {count:4d}  {'#' * min(count, 40)}")
    print(f"\n  {'Total':20s} {len(classified):d}\n")

    if dry_run:
        print("DRY RUN -- No files moved.")
        return

    # Optionally rename files as they're moved
    from .rename import build_renames, TAG_BASED_STRATEGIES
    rename_pairs: list = []
    if rename_strategy and rename_strategy != "none":
        is_tag_strategy = rename_strategy in TAG_BASED_STRATEGIES

        # Build an AIRenamer once so the ai_classification strategy
        # can enrich Ollama colour-only tags with content tags from
        # CLIP / analyzer / suggest_tags. The renamer reads cached
        # Ollama tags first (so the classification log format is
        # preserved) and only adds new tags when needed.
        ai_renamer_for_build = None
        if rename_strategy == "ai_classification":
            try:
                from .rename import AIRenamer
                # 'auto' backend cascade: CLIP (if installed) →
                # Ollama → Analyzer. The cached Ollama tags get the
                # highest priority in _combined_classify.
                ai_renamer_for_build = AIRenamer(
                    backend="auto",
                    mode=mode,  # use the same mode as the classification
                    max_tags=max(rename_max_tags * 2, 8),
                    force_reprocess=False,
                )
            except Exception:
                ai_renamer_for_build = None

        # Build rename pairs for all classified files
        all_files = []
        cat_by_file = {}
        tags_by_file = {}
        subject_by_file = {}
        for fname, cat, extra in classified:
            src = os.path.join(WALLPAPERS_DIR, fname)
            if os.path.exists(src):
                all_files.append(src)
                cat_by_file[src] = cat
                if is_tag_strategy:
                    tags_by_file[src] = (extra or {}).get("tags") or []
                    subject_by_file[src] = (extra or {}).get("subject")

        # Group by category and rename within each
        by_cat: dict = {}
        for src in all_files:
            by_cat.setdefault(cat_by_file[src], []).append(src)
        for cat, files in by_cat.items():
            cat_for_rename = cat if rename_category_prefix else ""
            try:
                kwargs = {
                    "strategy": rename_strategy,
                    "category": cat_for_rename,
                    "start": 1, "pad": 3,
                }
                if is_tag_strategy:
                    kwargs["tags_by_file"] = tags_by_file
                    kwargs["subject_by_file"] = subject_by_file
                    kwargs["max_tags"] = rename_max_tags
                if ai_renamer_for_build is not None:
                    kwargs["ai_renamer"] = ai_renamer_for_build
                pairs = build_renames(files, **kwargs)
                rename_pairs.extend(pairs)
            except Exception as e:
                print(f"  [WARN] Rename build failed for {cat}: {e}")
        # Release the AIRenamer (closes its Ollama HTTP client if any).
        if ai_renamer_for_build is not None:
            try:
                ai_renamer_for_build.close()
            except Exception:
                pass
        if rename_pairs:
            tag_msg = f", max {rename_max_tags} tags" if is_tag_strategy else ""
            print(f"Rename: {sum(1 for o, n in rename_pairs if o != n)} files will be renamed ({rename_strategy}{tag_msg})")

    print("Moving files...")
    for fname, cat, _ in classified:
        src = os.path.join(WALLPAPERS_DIR, fname)
        if not os.path.exists(src):
            continue
        dst_dir = os.path.join(dest_dir, cat)
        os.makedirs(dst_dir, exist_ok=True)

        # Check if there's a rename for this file
        new_fname = fname
        for o, n in rename_pairs:
            if o == src:
                new_fname = os.path.basename(n)
                break
        dst = os.path.join(dst_dir, new_fname)
        if os.path.exists(dst):
            name, ext = os.path.splitext(new_fname)
            n = 1
            while os.path.exists(os.path.join(dst_dir, f"{name}_{n}{ext}")):
                n += 1
            dst = os.path.join(dst_dir, f"{name}_{n}{ext}")
        try:
            shutil.move(src, dst)
        except Exception as exc:
            print(f"  [ERR] Could not move {fname}: {exc}")

    print("\nOrganization complete.")

    # Auto-build heuristic patterns for categories that received files
    built_cats = set(cat for _, cat, _ in classified)
    if built_cats and not dry_run:
        print("\nBuilding heuristic patterns for updated categories...")
        from .lowlevel.category_profile import build_category_profile
        for cat in sorted(built_cats):
            cat_dir = os.path.join(dest_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            try:
                pattern = build_category_profile(cat_dir, max_samples=10)
                if pattern:
                    cfg = c.get_category_config(cat)
                    cfg["heuristic_pattern"] = pattern
                    c.write_category_config(cat, cfg)
                    print(f"  [OK] {cat}: {pattern.get('used_samples', 0)} samples")
            except Exception as exc:
                print(f"  [WARN] {cat}: {exc}")

    _emit("done", len(classified), "", "complete")
