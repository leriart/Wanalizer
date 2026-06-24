"""
File rename strategies for wallpapers.

All strategies preserve the file extension. Each function takes
a list of (old_path, new_name) pairs to apply renames atomically.

Available strategies:
  - sequential:    001, 002, 003...
  - category:      <Category>_001, <Category>_002...
  - date_prefix:   2024-12-19_<original>...
  - sanitize:      removes special chars, replaces spaces
  - slugify:       lowercase, dashes, no accents
  - lowercase:     all lowercase
  - uppercase:     all UPPERCASE
  - no_spaces:     spaces -> underscores
  - hash8:         <8-char MD5>.<ext>
  - date_hash:     2024-12-19_<8-char MD5>.<ext>
  - title_case:    Title Case With Spaces
  - underscore:    all_words_to_underscores
  - reverse:       reverses the basename
  - truncate:      shortens to N chars (default 32)
  - zero_pad:      zero-pads numeric prefixes
  - timestamp:     1703001234_<original>.<ext>
  - tags:          <tag1>_<tag2>_<tag3> (from detected tags)
  - category_tags: <Category>_<tag1>_<tag2>_<tag3>
  - subject_tags:  <subject>_<tag1>_<tag2>_<tag3>
  - date_tags:     2024-12-19_<tag1>_<tag2>_<tag3>
"""
import os
import re
import hashlib
import unicodedata
from datetime import datetime
from typing import List, Tuple, Callable, Dict, Optional


def split_ext(filename: str) -> Tuple[str, str]:
    """Split filename into (base, ext) preserving multi-char extensions."""
    base, ext = os.path.splitext(filename)
    if ext.lower() in (".tar",):
        base2, ext2 = os.path.splitext(base)
        if ext2:
            return base2, ext2 + ext
    return base, ext


def sanitize(name: str) -> str:
    """Remove or replace unsafe characters."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s.-]", "_", name)
    name = re.sub(r"[\s_]+", "_", name)
    return name.strip("._-")


def slugify(name: str) -> str:
    """Convert to URL-friendly slug (lowercase, dashes, no special chars)."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    return name.strip("-._")


def md5_8(text: str) -> str:
    """First 8 chars of MD5 hash of text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def file_md5_8(path: str) -> str:
    """First 8 chars of MD5 hash of file contents."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except Exception:
        return "00000000"


def sanitize_tag(tag: str) -> str:
    """Sanitize a single tag for inclusion in a filename.

    Strips non-ASCII / non-word chars, lowercases, falls back to 'tag' if
    the result is empty.
    """
    if not tag:
        return ""
    t = unicodedata.normalize("NFKD", str(tag)).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^\w\s-]", "", t).strip().lower()
    t = re.sub(r"[\s_]+", "-", t).strip("-")
    return t


def _tag_components(tags: List[str], max_tags: int = 3) -> List[str]:
    """Return up to `max_tags` sanitised, de-duplicated tag components
    ready to join into a filename."""
    out: List[str] = []
    seen: set = set()
    for t in tags or []:
        s = sanitize_tag(t)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_tags:
            break
    return out


def build_renames(
    files: List[str],
    strategy: str,
    category: str = "",
    start: int = 1,
    pad: int = 3,
    truncate_len: int = 32,
    tags_by_file: Optional[Dict[str, List[str]]] = None,
    subject_by_file: Optional[Dict[str, Optional[str]]] = None,
    max_tags: int = 3,
) -> List[Tuple[str, str]]:
    """Build a list of (old_path, new_path) for the chosen strategy.

    Args:
        files: list of file paths to rename
        strategy: one of the strategy names below
        category: category name (used by 'category' and 'category_tags' strategies)
        start: starting number for sequential strategies
        pad: zero-pad width for sequential
        truncate_len: max length for truncate strategy
        tags_by_file: per-file tag lists for tag-based strategies
        subject_by_file: per-file main-subject strings for subject_tags
        max_tags: max number of tags to include in tag-based strategies

    Returns:
        List of (old_path, new_path) tuples (new_path is full path)
    """
    files = sorted(files)
    pairs: List[Tuple[str, str]] = []
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = str(int(datetime.now().timestamp()))

    tags_by_file = tags_by_file or {}
    subject_by_file = subject_by_file or {}

    for i, old_path in enumerate(files):
        if not os.path.isfile(old_path):
            continue
        directory = os.path.dirname(old_path)
        filename = os.path.basename(old_path)
        base, ext = split_ext(filename)

        n = i + start
        if strategy == "sequential":
            new_base = f"{n:0{pad}d}"
        elif strategy == "category":
            cat = category or os.path.basename(directory) or "file"
            cat = sanitize(cat) or "file"
            new_base = f"{cat}_{n:0{pad}d}"
        elif strategy == "date_prefix":
            new_base = f"{today}_{base}"
        elif strategy == "sanitize":
            new_base = sanitize(base)
        elif strategy == "slugify":
            new_base = slugify(base)
        elif strategy == "lowercase":
            new_base = base.lower()
        elif strategy == "uppercase":
            new_base = base.upper()
        elif strategy == "no_spaces":
            new_base = re.sub(r"\s+", "_", base).strip("_")
        elif strategy == "hash8":
            new_base = file_md5_8(old_path)
        elif strategy == "date_hash":
            new_base = f"{today}_{file_md5_8(old_path)}"
        elif strategy == "title_case":
            new_base = base.replace("_", " ").replace("-", " ").title().replace(" ", "_")
        elif strategy == "underscore":
            new_base = re.sub(r"[\s\-]+", "_", base).strip("_")
        elif strategy == "reverse":
            new_base = base[::-1]
        elif strategy == "truncate":
            new_base = base[:truncate_len]
        elif strategy == "zero_pad":
            m = re.match(r"^(\d+)(.*)$", base)
            if m:
                num, rest = m.group(1), m.group(2)
                new_base = f"{int(num):08d}{rest}"
            else:
                new_base = base
        elif strategy == "timestamp":
            new_base = f"{timestamp}_{base}"
        elif strategy == "tags":
            file_tags = tags_by_file.get(old_path) or []
            comps = _tag_components(file_tags, max_tags=max_tags)
            new_base = "_".join(comps) if comps else f"untagged_{n:0{pad}d}"
        elif strategy == "category_tags":
            cat = sanitize(category or os.path.basename(directory) or "file") or "file"
            file_tags = tags_by_file.get(old_path) or []
            comps = _tag_components(file_tags, max_tags=max_tags)
            new_base = "_".join([cat] + comps) if comps else f"{cat}_{n:0{pad}d}"
        elif strategy == "subject_tags":
            subject = sanitize_tag(subject_by_file.get(old_path) or "")
            file_tags = tags_by_file.get(old_path) or []
            comps = _tag_components(file_tags, max_tags=max_tags)
            parts: List[str] = []
            if subject:
                parts.append(subject)
            for c in comps:
                if c and c not in parts:
                    parts.append(c)
            new_base = "_".join(parts) if parts else f"file_{n:0{pad}d}"
        elif strategy == "date_tags":
            file_tags = tags_by_file.get(old_path) or []
            comps = _tag_components(file_tags, max_tags=max_tags)
            new_base = "_".join([today] + comps) if comps else f"{today}_untagged_{n:0{pad}d}"
        else:
            new_base = base

        if not new_base:
            new_base = f"file_{n:0{pad}d}"

        # Enforce a sensible max length (keeps Windows happy at 255 chars
        # and avoids ridiculous tag lists blowing up the path).
        if len(new_base) > 120:
            new_base = new_base[:120].rstrip("_-.")

        new_filename = f"{new_base}{ext}"
        new_path = os.path.join(directory, new_filename)
        if new_path != old_path and new_path in [p[1] for p in pairs]:
            new_filename = f"{new_base}_{n}{ext}"
            new_path = os.path.join(directory, new_filename)
        pairs.append((old_path, new_path))

    return pairs


def apply_renames(pairs: List[Tuple[str, str]], dry_run: bool = False) -> Dict:
    """Apply renames. Returns a summary dict with stats and any errors."""
    stats = {"renamed": 0, "skipped": 0, "errors": 0, "error_list": []}
    temp_pairs: List[Tuple[str, str]] = []
    final_pairs: List[Tuple[str, str]] = []

    if not dry_run:
        for old, new in pairs:
            if old == new:
                stats["skipped"] += 1
                continue
            if not os.path.exists(old):
                stats["skipped"] += 1
                continue
            temp_path = old + ".__rename_tmp__"
            try:
                os.rename(old, temp_path)
                temp_pairs.append((temp_path, new))
            except Exception as e:
                stats["errors"] += 1
                stats["error_list"].append(f"{old}: {e}")
        for temp, new in temp_pairs:
            try:
                if os.path.exists(new):
                    base, ext = split_ext(new)
                    counter = 1
                    while os.path.exists(f"{base}_{counter}{ext}"):
                        counter += 1
                    new = f"{base}_{counter}{ext}"
                os.rename(temp, new)
                stats["renamed"] += 1
            except Exception as e:
                stats["errors"] += 1
                stats["error_list"].append(f"{temp}: {e}")
                try:
                    os.rename(temp, temp.replace(".__rename_tmp__", ""))
                except Exception:
                    pass

    return stats


RENAME_STRATEGIES = [
    ("sequential",    "Sequential (001, 002, ...)",       "Zero-padded numbers"),
    ("category",      "By category (<Cat>_001)",          "Category prefix + number"),
    ("date_prefix",   "Date prefix (2024-12-19_...)",     "Today's date + original"),
    ("sanitize",      "Sanitize (remove special chars)",  "Replace spaces/specials with _"),
    ("slugify",       "Slugify (lowercase-dashes)",       "URL-friendly lowercase-dash"),
    ("lowercase",     "Lowercase",                        "Convert to lowercase"),
    ("uppercase",     "UPPERCASE",                        "Convert to UPPERCASE"),
    ("no_spaces",     "No spaces (use _)",                "Replace spaces with underscores"),
    ("title_case",    "Title Case",                       "Capitalize Each Word"),
    ("underscore",    "Underscore style",                 "spaces/hyphens -> underscores"),
    ("hash8",         "MD5 hash (8 chars)",               "Short content hash"),
    ("date_hash",     "Date + MD5 hash",                  "Date prefix + content hash"),
    ("timestamp",     "Unix timestamp prefix",            "Numeric timestamp + original"),
    ("truncate",      "Truncate (32 chars max)",          "Cut long names to 32 chars"),
    ("zero_pad",      "Zero-pad leading number",          "Pad 7 -> 00000007"),
    ("reverse",       "Reverse",                          "Reverse the basename"),
    # Tag-based strategies
    ("tags",          "Tags (tag1_tag2_tag3)",            "Top detected tags (max 3)"),
    ("category_tags", "Category + tags",                   "Cat_tag1_tag2_tag3"),
    ("subject_tags",  "Subject + tags",                   "Subject_tag1_tag2_tag3"),
    ("date_tags",     "Date + tags",                      "2024-12-19_tag1_tag2_tag3"),
]

# Strategies that need per-file tag info from the caller.
TAG_BASED_STRATEGIES = {"tags", "category_tags", "subject_tags", "date_tags"}


# ---------------------------------------------------------------------------
# Per-file tag detection (used by rename dialog + reorganize "rename on move")
# ---------------------------------------------------------------------------

_PROFILE_TAGS_CACHE: Dict[str, Tuple[float, int, Tuple[List[str], Optional[str]]]] = {}


def _profile_cache_key(path: str) -> Tuple[str, float, int]:
    try:
        st = os.stat(path)
        return path, st.st_mtime, st.st_size
    except OSError:
        return path, 0.0, 0


def get_tags_for_file(
    path: str,
    category: Optional[str] = None,
    max_tags: int = 8,
    use_cache: bool = True,
) -> Tuple[List[str], Optional[str]]:
    """Compute (tags, subject) for a single file using the heuristic CV pipeline.

    Pure local CPU work; no AI/Ollama calls. Results are cached in-memory by
    (path, mtime, size) so successive Rename-dialog or "rename on move" calls
    reuse previous results and stay fast on large libraries.

    Args:
        path: absolute path to a supported image file.
        category: target category name (boosts tags that match the category).
        max_tags: cap on the returned tag list.
        use_cache: set to False to force a fresh recomputation.

    Returns:
        (tags, subject). `subject` is the strongest heuristic main-subject
        hint, or None when none could be inferred. Always returns a list —
        if all detectors fail, falls back to top-3 dominant color names.
    """
    if not path or not os.path.isfile(path):
        return [], None

    key = _profile_cache_key(path)
    if use_cache and key in _PROFILE_TAGS_CACHE:
        cached_mtime, cached_size, cached_result = _PROFILE_TAGS_CACHE[key]
        if cached_mtime == key[1] and cached_size == key[2]:
            return cached_result

    tags: List[str] = []
    subject: Optional[str] = None
    try:
        from .profile import get_image_profile
        profile = get_image_profile(path)
        from .tag_suggester import suggest_tags_for_category
        cat = category or profile.get("_current_category") or "default"
        guessed = suggest_tags_for_category(cat, profile, max_tags=max_tags)
        if guessed:
            tags = list(guessed)[:max_tags]
        # Best-effort main subject: the most "informative" (rarest) tag
        if tags:
            subject = tags[0]
    except Exception:
        pass

    if not tags:
        # Fallback: re-derive from a minimal profile scan (color weights only).
        try:
            from PIL import Image
            from .profile import extract_color_weights
            img = Image.open(path).convert("RGB")
            img.thumbnail((128, 128), Image.LANCZOS)
            weights, _ = extract_color_weights(img)
            tags = [k.lower() for k, _ in
                    sorted(weights.items(), key=lambda kv: -kv[1])[:3]]
        except Exception:
            tags = []

    result = (tags, subject)
    if use_cache:
        _PROFILE_TAGS_CACHE[key] = (key[1], key[2], result)
    return result


def compute_renames(
    files: List[str],
    strategy: str,
    category: Optional[str] = None,
    max_tags: int = 3,
    pad: int = 3,
    start: int = 1,
    truncate_len: int = 32,
) -> List[Tuple[str, str]]:
    """Build rename pairs, automatically detecting tags for tag-based strategies.

    Convenience wrapper around `build_renames` that pre-computes tags and
    subjects via `get_tags_for_file` only when the chosen strategy needs
    them. For non-tag strategies, behaviour is identical to `build_renames`.
    """
    needs_tags = strategy in TAG_BASED_STRATEGIES
    tags_by_file: Dict[str, List[str]] = {}
    subject_by_file: Dict[str, Optional[str]] = {}
    if needs_tags:
        for p in files:
            tags, subject = get_tags_for_file(p, category=category, max_tags=max_tags)
            tags_by_file[p] = tags
            subject_by_file[p] = subject
    return build_renames(
        files,
        strategy=strategy,
        category=category or "",
        start=start,
        pad=pad,
        truncate_len=truncate_len,
        tags_by_file=tags_by_file,
        subject_by_file=subject_by_file,
        max_tags=max_tags,
    )


def clear_tags_cache() -> None:
    """Drop the in-memory profile/tags cache."""
    _PROFILE_TAGS_CACHE.clear()
