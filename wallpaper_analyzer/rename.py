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
from collections import OrderedDict
from datetime import datetime
from typing import FrozenSet, List, Tuple, Callable, Dict, Optional


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


# Aspect-ratio buckets used as a last-resort descriptor so the
# filename always says *something* meaningful about the image even when
# no AI tag backend responded.
_ASPECT_BUCKETS = [
    ((9, 21),  "ultrawide"),
    ((9, 16),  "wide"),
    ((3, 4),   "portrait"),
    ((1, 1),   "square"),
    ((4, 3),   "standard"),
    ((16, 10), "wide"),
    ((16, 9),  "widescreen"),
    ((21, 9),  "ultrawide"),
]


def _aspect_token(path: str) -> str:
    """Return a short token describing the aspect-ratio bucket of `path`.

    Falls back to "image" if the file can't be opened.
    """
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            return "image"
        ratio = w / h
        # Pre-compute (ratio, name) pairs once.
        ratios = [(nw / nh, name) for (nw, nh), name in _ASPECT_BUCKETS]
        best_name = min(ratios, key=lambda r: abs(r[0] - ratio))[1]
        return best_name
    except Exception:
        return "image"


def _color_tokens(path: str, max_colors: int = 3) -> List[str]:
    """Return a list of dominant colour tokens for `path`.

    Always returns at least one token (or an empty list if the file
    can't be opened). Tokens are sanitised for use as filename parts.
    """
    try:
        from PIL import Image as _PILImage
        from .profile import extract_color_weights
        with _PILImage.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((128, 128), _PILImage.LANCZOS)
            weights, _ = extract_color_weights(im)
        # Take the top-N colours by weight; rename them with human words.
        items = sorted(weights.items(), key=lambda kv: -kv[1])[:max_colors]
        out: List[str] = []
        seen: set = set()
        # Friendly labels for the most common dominant-colour buckets.
        LABELS = {
            "black": "black", "white": "white", "gray": "gray",
            "red": "red", "green": "green", "blue": "blue",
            "yellow": "yellow", "orange": "orange", "purple": "purple",
            "pink": "pink", "brown": "brown", "cyan": "cyan", "magenta": "magenta",
        }
        for color, _w in items:
            tok = sanitize_tag(LABELS.get(color.lower(), color))
            if tok and tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out
    except Exception:
        return []


def _brightness_token(path: str) -> str:
    """Return 'dark' / 'mid' / 'bright' based on the image's mean luminance."""
    try:
        from PIL import Image as _PILImage, ImageStat as _ImageStat
        with _PILImage.open(path) as im:
            im = im.convert("L")
            im.thumbnail((64, 64), _PILImage.LANCZOS)
            mean = _ImageStat.Stat(im).mean[0]
        if mean < 64:
            return "dark"
        if mean > 192:
            return "bright"
        return "mid"
    except Exception:
        return ""


def _fallback_tags(path: str, primary: List[str], max_tags: int = 4) -> List[str]:
    """Always-return-something tag list.

    Combines the AI-supplied `primary` tags with deterministic
    fallback tokens derived from the image itself:
      1. The top-N dominant colour names (e.g. ['red', 'orange']).
      2. The image's aspect-ratio bucket (e.g. 'widescreen').
      3. The image's brightness bucket (e.g. 'dark').

    The fallback tokens are only added if there is room left after the
    primary tags. The function never returns an empty list — the caller
    always gets at least one descriptive token, which means
    ``build_renames`` will never produce an ``untagged_NNN`` filename.
    """
    out: List[str] = []
    seen: set = set()
    for t in primary or []:
        s = sanitize_tag(t)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_tags:
            return out
    # Augment with deterministic per-image descriptors.
    for c in _color_tokens(path, max_colors=max(1, max_tags - len(out))):
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= max_tags:
            return out
    asp = _aspect_token(path)
    if asp and asp not in seen:
        out.append(asp)
        seen.add(asp)
    if len(out) < max_tags:
        br = _brightness_token(path)
        if br and br not in seen:
            out.append(br)
    # Last-ditch: short content hash so we still get a unique-ish token.
    if not out:
        out.append(file_md5_8(path))
    return out[:max_tags]


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
    ai_renamer: Optional["AIRenamer"] = None,
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
        ai_renamer: optional AIRenamer used by `ai_classification` to
            enrich Ollama colour tags with CLIP / analyzer content tags.
            When provided, the strategy ignores `tags_by_file` for
            per-file detection and lets AIRenamer pick the best signals
            (Ollama cache → CLIP → analyzer → suggest_tags).

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
            comps = _fallback_tags(old_path, file_tags, max_tags=max_tags)
            new_base = "_".join(comps)
        elif strategy == "category_tags":
            cat = sanitize(category or os.path.basename(directory) or "file") or "file"
            file_tags = tags_by_file.get(old_path) or []
            comps = _fallback_tags(old_path, file_tags, max_tags=max_tags)
            # Category first, then as many tag tokens as fit (subject +
            # colors + aspect) so the filename stays in max_tags total
            # components when joined.
            tag_budget = max(1, max_tags)
            new_base = "_".join([cat] + comps[:tag_budget])
        elif strategy == "subject_tags":
            subject = sanitize_tag(subject_by_file.get(old_path) or "")
            file_tags = tags_by_file.get(old_path) or []
            comps = _fallback_tags(old_path, file_tags, max_tags=max_tags)
            parts: List[str] = []
            if subject:
                parts.append(subject)
            for c in comps:
                if c and c not in parts:
                    parts.append(c)
            new_base = "_".join(parts) if parts else "_".join(
                _fallback_tags(old_path, [], max_tags=max_tags)
            )
        elif strategy == "date_tags":
            file_tags = tags_by_file.get(old_path) or []
            comps = _fallback_tags(old_path, file_tags, max_tags=max_tags)
            new_base = "_".join([today] + comps)
        elif strategy == "ai_classification":
            # Mirror the AI classification log format in the filename:
            #   <3-letter-abbr>_<Category>_<tag1-tag2-...>.jpg
            # e.g. for the log line
            #   [3/15] 20260623_222915.jpg... [ANI] Anime  (mode=ollama tags=cartoon,anime,female)
            # the rename produces:
            #   20260623_222915_ANI_Anime_cartoon-anime-female.jpg
            #
            # When an AIRenamer is provided we let IT pick the tags
            # (Ollama cache → CLIP → analyzer → suggest_tags) so the
            # filename actually uses AI content analysis, not just the
            # colour tokens the Ollama vision model may have returned.
            cat = sanitize(category or os.path.basename(directory) or "file") or "file"
            abbr = cat[:3].upper()
            if ai_renamer is not None:
                try:
                    ai_tags, _ = ai_renamer.detect_tags(old_path, category=category)
                    file_tags = list(ai_tags)
                except Exception:
                    file_tags = list(tags_by_file.get(old_path) or [])
            else:
                file_tags = list(tags_by_file.get(old_path) or [])
            comps = _fallback_tags(old_path, file_tags, max_tags=max_tags)
            tag_part = "-".join(comps[:max(1, max_tags)]) if comps else "untagged"
            new_base = f"{abbr}_{cat}_{tag_part}"
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
    ("ai_classification",
                     "AI classification",               "ABBR_Category_tag1-tag2-... (mirrors the [AI] classify log)"),
]

# Strategies that need per-file tag info from the caller.
TAG_BASED_STRATEGIES = {
    "tags", "category_tags", "subject_tags", "date_tags", "ai_classification",
}


# ---------------------------------------------------------------------------
# Per-file tag detection (used by rename dialog + reorganize "rename on move")
# ---------------------------------------------------------------------------

# Bounded LRU cache keyed by (path, mtime, size). The OrderedDict moves
# recently-used entries to the end; when the cache is full we drop the
# oldest. This avoids the unbounded-growth leak the previous plain dict
# had: 10,000 files × ~2 KB per entry used to balloon past 20 MB and
# never reclaim memory until process exit.
_PROFILE_TAGS_CACHE: "OrderedDict[Tuple[str, float, int], Tuple[List[str], Optional[str]]]" = OrderedDict()
_PROFILE_TAGS_CACHE_MAX = 4096


def _profile_cache_key(path: str) -> Tuple[str, float, int]:
    try:
        st = os.stat(path)
        return path, st.st_mtime, st.st_size
    except OSError:
        return path, 0.0, 0


def _profile_cache_get(key: Tuple[str, float, int]) -> Optional[Tuple[List[str], Optional[str]]]:
    entry = _PROFILE_TAGS_CACHE.get(key)
    if entry is None:
        return None
    # Refresh LRU position.
    _PROFILE_TAGS_CACHE.move_to_end(key)
    return entry


def _profile_cache_put(key: Tuple[str, float, int],
                      value: Tuple[List[str], Optional[str]]) -> None:
    _PROFILE_TAGS_CACHE[key] = value
    _PROFILE_TAGS_CACHE.move_to_end(key)
    while len(_PROFILE_TAGS_CACHE) > _PROFILE_TAGS_CACHE_MAX:
        _PROFILE_TAGS_CACHE.popitem(last=False)


def clear_tags_cache() -> None:
    """Drop the in-memory profile/tags cache."""
    _PROFILE_TAGS_CACHE.clear()


def get_tags_for_file(
    path: str,
    category: Optional[str] = None,
    max_tags: int = 8,
    use_cache: bool = True,
) -> Tuple[List[str], Optional[str]]:
    """Compute (tags, subject) for a single file using the heuristic CV pipeline.

    Pure local CPU work; no AI/Ollama calls. Results are cached in-memory
    in a bounded LRU (size cap = ``_PROFILE_TAGS_CACHE_MAX``) keyed by
    (path, mtime, size) so successive Rename-dialog or "rename on move"
    calls reuse previous results and stay fast on large libraries, while
    also bounding memory so 50k-file libraries don't crash the process.

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
    if use_cache:
        cached = _profile_cache_get(key)
        if cached is not None:
            return cached

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
        _profile_cache_put(key, result)
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


# ---------------------------------------------------------------------------
# AI-powered tag detection (CLIP / Ollama)
# ---------------------------------------------------------------------------

# Curated tag vocabulary used for CLIP-based ranking. These are short,
# generic, descriptive words that compose well into filenames like
# "moonlit_neon_kimono.jpg". Curated from the same registry used by
# Ollama's prompt; filtered for word-shape compatibility.
_CLIP_TAG_VOCAB: Optional[List[str]] = None


def _clip_tag_vocab() -> List[str]:
    """Return the cached CLIP tag vocabulary, building it lazily.

    The full `_tags_flat` registry can hold many tags. We prefer
    short, generally-descriptive tags (2-20 chars, no spaces) and
    keep the full list — the user explicitly wants AI not to be
    limited. Memory is still bounded because we encode in batches
    inside ``_clip_detect_tags`` and explicitly drop the tensors.
    """
    global _CLIP_TAG_VOCAB
    # Rebuild when missing OR empty (an empty cache must always rebuild
    # so tests that explicitly clear() it don't lock in a permanent zero
    # state across processes).
    if _CLIP_TAG_VOCAB:
        return _CLIP_TAG_VOCAB
    try:
        from .tags import _tags_flat
        vocab = sorted(t for t in _tags_flat
                       if 1 < len(t) <= 20 and " " not in t)
        _CLIP_TAG_VOCAB = vocab
    except Exception:
        _CLIP_TAG_VOCAB = []
    return _CLIP_TAG_VOCAB


# Backend identifiers accepted by ai_detect_tags() / AIRenamer
AI_TAG_BACKENDS = ("auto", "heuristic", "clip", "ollama", "organize")

# Analyzer modes the AI renamer can run on top of. Each mode is the
# SAME analyzer the organize pipeline uses for category assignment,
# so the tags come from the same content analysis that picked the
# category. Default ("auto") picks the configured organize_mode.
AI_TAG_MODES = ("auto", "lowlevel", "fusion", "clip", "ollama")

# Organize-style AI option defaults (mirrors the Organize page).
ORGANIZE_AI_OPTIONS = {
    "ollama_nsfw_enabled": True,
    "ollama_describe_enabled": False,
    "ollama_classify_enabled": False,
    "ollama_classify_method": "tags",
}


# ---------------------------------------------------------------------------
# AIRenamer — stateful, batch-friendly AI rename engine.
#
# This class encapsulates everything that was previously scattered
# across top-level functions:
#   * Reusable Ollama HTTP client (one per batch instead of per file)
#   * Cached CLIP engine + text-feature cache (the engine itself
#     caches text embeddings; we never re-encode the vocab each call)
#   * Bounded per-file tag-result cache so the same file is never
#     re-processed within a batch
#   * Force-reprocess mode to invalidate the cache mid-batch
#   * Per-file error isolation: a single bad file produces an empty
#     tag list and a warning in the log; the batch keeps running
#   * Rich progress callback: file_started, file_done, file_failed,
#     batch_done
#
# The standalone ``ai_detect_tags`` / ``ai_compute_renames`` functions
# are kept as thin wrappers around this class so existing callers
# continue to work without modification.
# ---------------------------------------------------------------------------


class AIRenamer:
    """Stateful AI rename engine. Use one instance per batch.

    Tag detection is now analyzer-driven: by default the same
    `BaseAnalyzer` (lowlevel / fusion / clip / ollama) that the
    organize pipeline uses for category assignment also picks the
    descriptive tags for renaming. This means the rename output is
    always consistent with the category the user assigned — "anime"
    gets anime-style content tags, "nature" gets nature-style tags,
    etc. — instead of the previous colour-biased heuristic that
    couldn't tell a forest from an anime beach.

    Args:
        backend: tag-source backend ("auto", "heuristic", "clip",
            "ollama"). Determines WHERE the tags come from:
              - "heuristic" → analyzer-driven content analysis
                (this is now the recommended default; picks up CLIP
                scores + fingerprint + content + theme signals when
                the chosen analyzer exposes them).
              - "clip" → curated CLIP vocabulary (legacy).
              - "ollama" → local vision LLM (legacy).
              - "auto" → cascade analyzer → ollama → clip → heuristic.
        mode: analyzer mode ("auto", "lowlevel", "fusion", "clip",
            "ollama"). Defaults to "auto" which uses the user's
            configured `organize_mode` from settings.
        model: optional model override (for CLIP / Ollama backends).
        max_tags: cap on the returned tag list per file.
        cache_size: in-memory cache size for the AIRenamer's own LRU.
        force_reprocess: bypass the per-file cache and re-detect
            every image even if it was already processed in this batch.
    """

    def __init__(self,
                 backend: str = "auto",
                 mode: str = "auto",
                 model: Optional[str] = None,
                 max_tags: int = 8,
                 cache_size: int = 4096,
                 force_reprocess: bool = False,
                 organize_options: Optional[Dict] = None):
        self.backend = backend if backend in AI_TAG_BACKENDS else "heuristic"
        self.mode = mode if mode in AI_TAG_MODES else "auto"
        self.model = model or None
        self.max_tags = max_tags
        self._cache_size = cache_size
        self._force_reprocess = force_reprocess
        # Organize-style AI options (NSFW, describe, classify method).
        # When set, these override the settings passed to get_analyzer()
        # so the rename dialog can mirror the Organize page exactly.
        self._organize_options = dict(organize_options or {})
        # Bounded LRU of per-file tag results, keyed by path.
        # Lets a single batch reuse the result for the same path
        # without re-running the model. Bypassed when force_reprocess.
        self._tag_cache: "OrderedDict[str, Tuple[List[str], Optional[str]]]" = OrderedDict()
        self._ollama_client = None  # lazily opened, reused across files
        self._ollama_url = None
        self._ollama_model = None
        self._analyzer = None          # lazily built analyzer for content-aware path
        self._analyzer_mode_resolved: Optional[str] = None  # for logging
        # Counters (read by callers / status bars).
        self.processed = 0
        self.failed = 0
        self.cached_hits = 0
        # Backing log — appended to by _emit().
        self.log_lines: List[str] = []

    # ---------------- cache ----------------

    def clear_cache(self) -> None:
        """Drop the per-file tag cache (forces the next call to recompute)."""
        self._tag_cache.clear()

    def _cache_get(self, path: str) -> Optional[Tuple[List[str], Optional[str]]]:
        if self._force_reprocess:
            return None
        v = self._tag_cache.get(path)
        if v is not None:
            self._tag_cache.move_to_end(path)
            self.cached_hits += 1
        return v

    def _cache_put(self, path: str,
                   value: Tuple[List[str], Optional[str]]) -> None:
        self._tag_cache[path] = value
        self._tag_cache.move_to_end(path)
        while len(self._tag_cache) > self._cache_size:
            self._tag_cache.popitem(last=False)

    # ---------------- Ollama lifecycle ----------------

    def _ensure_ollama(self):
        """Lazily build the Ollama HTTP client. Reuses it across files
        so we don't pay the TCP+TLS handshake per image."""
        if self._ollama_client is not None:
            return self._ollama_client
        try:
            from . import settings as _settings
            from .ollama_client import OllamaClient
            cfg = _settings.load_settings()
            self._ollama_url = cfg.get("ollama_url", "http://localhost:11434")
            self._ollama_model = self.model or cfg.get("ollama_model", "llava:7b")
            self._ollama_client = OllamaClient(
                base_url=self._ollama_url,
                model=self._ollama_model,
                timeout=int(cfg.get("ollama_timeout", 60)),
            )
            self._emit("ollama_init",
                       f"Ollama client ready ({self._ollama_url}, {self._ollama_model})")
        except Exception as e:
            self._emit("ollama_init", f"Could not init Ollama client: {e}")
            self._ollama_client = None
        return self._ollama_client

    def close(self) -> None:
        """Release the Ollama HTTP client. Call when the batch is done."""
        if self._ollama_client is not None:
            try:
                self._ollama_client.close()
            except Exception:
                pass
            self._ollama_client = None

    # ---------------- per-backend tag detection ----------------

    def _ollama_detect(self, path: str) -> Tuple[List[str], Optional[str]]:
        client = self._ensure_ollama()
        if client is None:
            return [], None
        try:
            tags = client.detect_tags(path, max_tags=self.max_tags) or []
            subject = client.detect_main_subject(path)
            return tags, subject
        except Exception as e:
            self._emit("ollama_err", f"Ollama call failed for {os.path.basename(path)}: {e}")
            return [], None

    def _clip_detect(self, path: str) -> Tuple[List[str], Optional[str]]:
        try:
            from .clip_client import get_engine
            engine = get_engine(model_name=self.model)
            if not engine.available:
                return [], None
            vocab = _clip_tag_vocab()
            if not vocab:
                return [], None
            img_feat = engine.encode_image(path)
            if img_feat is None:
                return [], None
            import torch
            import numpy as np
            BATCH = 128
            sims_chunks: List["torch.Tensor"] = []
            valid_vocab: List[str] = []
            for i in range(0, len(vocab), BATCH):
                chunk = vocab[i:i + BATCH]
                chunk_feats_list = engine.encode_texts(chunk)
                if not chunk_feats_list or any(t is None for t in chunk_feats_list):
                    continue
                chunk_feats = torch.stack([t for t in chunk_feats_list], dim=0)
                chunk_sims = (
                    img_feat.to(chunk_feats.device) @ chunk_feats.T
                ).squeeze(0)
                sims_chunks.append(chunk_sims.detach().cpu())
                valid_vocab.extend(chunk)
                del chunk_feats, chunk_sims
            if not sims_chunks:
                return [], None
            sims = torch.cat(sims_chunks, dim=0)
            scores = sims.numpy()
            order = np.argsort(-scores)
            top: List[str] = []
            for idx in order[:self.max_tags]:
                if float(scores[idx]) <= 0:
                    break
                top.append(valid_vocab[int(idx)])
            del sims, scores, sims_chunks
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return top, (top[0] if top else None)
        except Exception as e:
            self._emit("clip_err", f"CLIP call failed for {os.path.basename(path)}: {e}")
            return [], None

    def _heuristic_detect(self, path: str, category: Optional[str]) -> Tuple[List[str], Optional[str]]:
        """Content-aware tag detection via the configured analyzer.

        This is the same analyzer pipeline the organize pass uses for
        category assignment (lowlevel / fusion / clip / ollama, picked
        by `self.mode`). It runs `analyzer.analyze(path)` to build the
        full feature profile (low-level CV + content detectors + theme
        + fingerprint + CLIP scores if available) and then derives the
        tag list from `classify_with_confidence(profile)["tags"]` — the
        content-aware set built by `tag_suggester.suggest_tags`.

        Result: tags describe the image's CONTENT (anime, cyberpunk,
        space, neon, portrait, pixel-art, minimalist, ...) — not just
        colours.
        """
        analyzer = self._ensure_analyzer()
        if analyzer is None:
            # No analyzer available — fall back to the legacy heuristic
            # so the rename still produces something.
            return get_tags_for_file(path, category=category, max_tags=self.max_tags)
        try:
            tags, subject = analyzer.detect_tags(path, max_tags=self.max_tags)
        except Exception as e:
            self._emit("analyzer_err", f"{os.path.basename(path)}: {e}")
            return [], None
        return tags, subject

    def _organize_detect(self, path: str, category: Optional[str]) -> Tuple[List[str], Optional[str]]:
        """Detect tags using the EXACT same pipeline as the Organize page.

        Steps mirror `_classify_worker` in `organize.py`:
          1. Extract a video/animated frame if needed.
          2. Build the analyzer for the selected mode (lowlevel/clip/fusion/ollama)
             with the Organize-style AI options (NSFW, describe, classify, ...).
          3. Run `analyzer.analyze()` to produce the full profile.
          4. Call `_detect_tags_for_file()` to extract the same (tags, subject)
             tuple that the organize pipeline uses for tag-based rename.
        """
        try:
            from .organize import _detect_tags_for_file, _extract_frame_if_video
        except Exception as import_err:
            self._emit("organize_err", f"Could not import organize helpers: {import_err}")
            return [], None

        analyzer = self._ensure_analyzer()
        if analyzer is None:
            return get_tags_for_file(path, category=category, max_tags=self.max_tags)

        analyze_path, is_temp = _extract_frame_if_video(path)
        try:
            profile = analyzer.analyze(analyze_path)
        except Exception as e:
            self._emit("organize_err", f"{os.path.basename(path)}: {e}")
            return [], None
        finally:
            if is_temp and os.path.exists(analyze_path):
                try:
                    os.remove(analyze_path)
                except Exception:
                    pass

        if category:
            profile["_current_category"] = category
        tags, subject = _detect_tags_for_file(path, profile)
        return tags, subject

    # ---------------- combined registry-aware classifier ----------------

    # Tag tokens that describe *colour only* — they don't carry any
    # content information. When the result has both content and colour
    # tags we want to surface the content tags first; colour tokens are
    # only kept as a last resort (no content detected) or as trailing
    # descriptors when there's room left after the content tags.
    _COLOUR_ONLY_TOKENS: FrozenSet[str] = frozenset({
        "red", "green", "blue", "yellow", "black", "white", "gray",
        "grey", "orange", "purple", "pink", "brown", "cyan", "magenta",
        "teal", "maroon", "navy", "olive", "lime", "aqua", "fuchsia",
        "silver", "gold", "beige", "tan", "khaki", "coral", "salmon",
        "ivory", "crimson", "turquoise", "lavender", "indigo", "violet",
        "rose", "amber", "mint", "peach", "plum", "ruby", "sapphire",
        "scarlet", "azure", "bronze", "cherry", "copper", "emerald",
    })

    @classmethod
    def _is_colour_token(cls, tag: str) -> bool:
        return tag.lower() in cls._COLOUR_ONLY_TOKENS

    def _combined_classify(self, path: str,
                           category: Optional[str] = None
                           ) -> Tuple[List[str], Optional[str]]:
        """Combine CLIP semantic + analyzer content + suggest_tags.

        Three independent signals are merged, each contributing what it's
        best at:

          1. **CLIP semantic match** (when installed) — encodes the
             image once, scores against the FULL tag registry
             (~1.5k tags), returns top-K by cosine similarity. Best at
             generic semantic recognition ("tokyo-night", "elf",
             "cyberpunk-era", ...).
          2. **Analyzer content heuristics** — runs the configured
             analyzer pipeline (same one the organize pass uses for
             category assignment) to get `classify_with_confidence`
             tags. Best at project-specific content signals driven by
             the curated registry (anime / cyberpunk / portrait /
             nature / ...).
          3. **suggest_tags from the profile** — runs the full heuristic
             tag suggester (anime_score, cyberpunk_score, neon_score,
             content detectors, palette, composition). Best at the
             curated content tags the user has defined.

        The merged list is **content-first**: every pure-colour token
        is moved to the back so content tags always surface first.
        The result always has at least one tag when any of the three
        signals produced anything — even on solid-colour images where
        CLIP legitimately returns colour tokens.
        """
        seen: set = set()
        ordered: List[str] = []  # preserves priority: ollama > CLIP > analyzer > suggest_tags
        sources_used: List[str] = []

        # 0. Cached Ollama tags from a prior classification. When the
        #    user already ran `organize --mode=ollama`, the tags are
        #    stored in the hash cache as `ollama_all_tags`. We surface
        #    them with the highest priority so the rename mirrors the
        #    classification log exactly:
        #      [AI] mode=ollama tags=cartoon,anime,female
        #      -> ANI_Anime_cartoon-anime-female.jpg
        try:
            from .duplicates import load_hash_cache
            cache = load_hash_cache()
            entry = cache.get(path) or {}
            ollama_tags = entry.get("ollama_all_tags") or entry.get("ollama_tags") or []
            for t in ollama_tags:
                tl = str(t).lower().strip()
                if tl and tl not in seen:
                    seen.add(tl)
                    ordered.append(tl)
            if ollama_tags:
                sources_used.append("ollama")
        except Exception:
            pass

        # 1. CLIP semantic match (priority 1).
        try:
            clip_tags, _ = self._clip_detect(path)
            for t in clip_tags:
                tl = t.lower().strip()
                if tl and tl not in seen:
                    seen.add(tl)
                    ordered.append(tl)
            if clip_tags:
                sources_used.append("clip")
        except Exception:
            pass

        # 2. Analyzer content signals (priority 2).
        analyzer = self._ensure_analyzer()
        if analyzer is not None:
            try:
                info = analyzer.classify_with_confidence(
                    analyzer.analyze(path)
                ) if hasattr(analyzer, "classify_with_confidence") else {}
                analyzer_tags = info.get("tags") or set()
                # suggest_tags returns a set with no order — sort for
                # determinism (longer content words tend to be more
                # specific, but we keep the registry's natural order).
                for t in sorted(analyzer_tags):
                    tl = t.lower().strip()
                    if tl and tl not in seen:
                        seen.add(tl)
                        ordered.append(tl)
                if analyzer_tags:
                    sources_used.append("analyzer")
            except Exception:
                pass

        # 3. Full suggest_tags (priority 3, fills gaps).
        try:
            from .profile import get_image_profile
            from .tag_suggester import suggest_tags, suggest_tags_for_category
            profile = get_image_profile(path)
            cat = category or profile.get("_current_category") or "default"
            base = suggest_tags(profile, max_tags=self.max_tags * 3)
            base = set(t.lower() for t in base)
            for t in sorted(base):
                tl = t.lower().strip()
                if tl and tl not in seen:
                    seen.add(tl)
                    ordered.append(tl)
            if base:
                sources_used.append("suggest_tags")
            # Also pick up the category itself when present in the
            # registry — useful as a content hint for known categories.
            if cat and cat.lower() in (t.lower() for t in __import__(
                    "wallpaper_analyzer.tags", fromlist=["_tags_flat"]
            )._tags_flat):
                if cat.lower() not in seen:
                    seen.add(cat.lower())
                    ordered.append(cat.lower())
        except Exception:
            pass

        if not ordered:
            return [], None

        # Re-order so content tags always come first. This is the key
        # guarantee for the user: the rename output describes the
        # image's CONTENT, not just its colour. Pure-colour tokens are
        # pushed to the end (and dropped if we exceed max_tags).
        content_first = [t for t in ordered if not self._is_colour_token(t)]
        colour_last = [t for t in ordered if self._is_colour_token(t)]
        merged = content_first + colour_last
        trimmed = merged[:self.max_tags]
        if sources_used:
            self._emit("combined", f"sources=[{','.join(sources_used)}] "
                                   f"content={len(content_first)} "
                                   f"colour={len(colour_last)}")
        subject = trimmed[0] if trimmed else None
        return trimmed, subject

    # ---------------- analyzer lifecycle ----------------

    # ---------------- analyzer lifecycle ----------------

    def _organize_settings(self) -> Dict:
        """Build a settings snapshot for the analyzer.

        Merges the user's saved settings with any Organize-style AI
        options overridden by the caller (NSFW, describe, classify, ...)
        and the explicitly selected model.
        """
        from . import settings as _settings
        cfg = _settings.load_settings()
        cfg.update(self._organize_options)
        requested = self.mode if self.mode != "auto" else cfg.get(
            "organize_mode", "lowlevel")
        requested = requested if requested in AI_TAG_MODES else "lowlevel"
        if self.model:
            if requested == "ollama":
                cfg["ollama_model"] = self.model
            elif requested == "clip":
                cfg["clip_model"] = self.model
        return cfg, requested

    def _ensure_analyzer(self):
        """Lazily build the analyzer used for content-aware tag detection.

        Resolves the effective mode:
          - "auto" → uses the user's configured `organize_mode` from settings.
          - explicit mode → uses that mode directly.
        """
        if self._analyzer is not None:
            return self._analyzer
        try:
            from .analyzers import get_analyzer
            cfg, requested = self._organize_settings()
            self._analyzer = get_analyzer(requested, cfg)
            self._analyzer_mode_resolved = requested
            self._emit("analyzer_init", f"Analyzer ready ({requested})")
        except Exception as e:
            self._emit("analyzer_init", f"Could not init analyzer: {e}")
            self._analyzer = None
        return self._analyzer

    # ---------------- single-file entry point ----------------

    def detect_tags(self,
                    path: str,
                    category: Optional[str] = None
                    ) -> Tuple[List[str], Optional[str]]:
        """Detect tags for one file using this renamer's backend/model.

        Returns (tags, subject) and never raises — failures yield an
        empty list which the deterministic fallback then fills with
        colour/aspect tokens.
        """
        if not path:
            return [], None
        if not os.path.isfile(path):
            self._emit("missing", f"file not found: {path}")
            self.failed += 1
            return [], None
        cached = self._cache_get(path)
        if cached is not None:
            return cached

        self._emit("file_started", os.path.basename(path))
        try:
            backend = self.backend
            tags: List[str] = []
            subject: Optional[str] = None

            if backend == "ollama":
                tags, subject = self._ollama_detect(path)
                if not tags and not subject:
                    self._emit("fallback", "Ollama returned no tags")
            elif backend == "clip":
                tags, subject = self._clip_detect(path)
                if not tags and not subject:
                    self._emit("fallback", "CLIP returned no tags")
            elif backend == "organize":
                # Exact Organize-page pipeline: analyzer + _detect_tags_for_file.
                tags, subject = self._organize_detect(path, category)
                if not tags and not subject:
                    self._emit("fallback", "Organize pipeline returned no tags")
            elif backend == "heuristic":
                # Heuristic = analyzer content heuristics (no AI model).
                # The combined classifier is the "all project tags" path
                # which is what the user actually wants — try that first
                # when AI is available.
                tags, subject = self._combined_classify(path, category)
                if not tags and not subject:
                    tags, subject = self._heuristic_detect(path, category)
            else:  # auto — combined AI + analyzer for the most reliable result
                # The combined classifier merges CLIP semantic + analyzer
                # content + suggest_tags into a single content-first tag
                # list. When no AI is installed, it falls back to the
                # analyzer heuristics via _heuristic_detect.
                tags, subject = self._combined_classify(path, category)
                if not tags and not subject:
                    self._emit("auto_fallback",
                               "Combined classifier empty, trying Ollama directly")
                    tags, subject = self._ollama_detect(path)
                if not tags and not subject:
                    self._emit("auto_fallback",
                               "No tags from any backend, using colour fallback")
                    tags, subject = [], None

            # Always wrap with deterministic fallback so the result
            # is never empty — augments AI tags with colour/aspect tokens.
            result = _fallback_tags(path, tags, max_tags=self.max_tags), subject
            self._cache_put(path, result)
            self.processed += 1
            self._emit("file_done", f"{os.path.basename(path)} → {result[0]}")
            return result
        except Exception as e:
            # One bad file must not break the batch.
            self.failed += 1
            self._emit("file_failed",
                       f"{os.path.basename(path)}: {type(e).__name__}: {e}")
            return [], None

    # ---------------- batch entry point ----------------

    def detect_tags_batch(self,
                          files: List[str],
                          category: Optional[str] = None
                          ) -> Dict[str, Tuple[List[str], Optional[str]]]:
        """Run ``detect_tags`` on every file. Returns a dict path → result.

        Per-file errors are isolated: a failing file yields ``[]`` and
        increments ``self.failed``. The batch never aborts early.
        """
        out: Dict[str, Tuple[List[str], Optional[str]]] = {}
        total = len(files)
        for i, p in enumerate(files, 1):
            self._emit("progress", f"file {i}/{total}: {os.path.basename(p)}")
            out[p] = self.detect_tags(p, category=category)
        self._emit("batch_done",
                   f"processed={self.processed} cached={self.cached_hits} "
                   f"failed={self.failed}")
        return out

    # ---------------- logging ----------------

    def _emit(self, stage: str, msg: str) -> None:
        line = f"[{stage}] {msg}"
        self.log_lines.append(line)
        # Also forward to the optional global progress_cb so legacy
        # callers keep working without modification.
        cb = globals().get("_AIRENAMER_GLOBAL_CB")
        if cb:
            try:
                cb(stage, msg)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Standalone function wrappers (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def ai_detect_tags(
    image_path: str,
    backend: str = "auto",
    category: Optional[str] = None,
    max_tags: int = 8,
    model: Optional[str] = None,
    progress_cb=None,
    mode: str = "auto",
    organize_options: Optional[Dict] = None,
) -> Tuple[List[str], Optional[str]]:
    """Single-file AI tag detection (uses a temporary AIRenamer).

    New code should construct an AIRenamer directly so the Ollama
    HTTP client and per-file cache are reused across the batch.

    Args:
        mode: analyzer mode ("auto"/"lowlevel"/"fusion"/"clip"/"ollama").
            Default "auto" picks up the user's configured `organize_mode`
            so the rename uses the same analyzer that picked the category.
        organize_options: Organize-style AI option overrides (NSFW,
            describe, classify, classify_method) used when backend="organize".
    """
    global _AIRENAMER_GLOBAL_CB
    _AIRENAMER_GLOBAL_CB = progress_cb
    try:
        ren = AIRenamer(
            backend=backend, mode=mode, model=model, max_tags=max_tags,
            organize_options=organize_options,
        )
        try:
            return ren.detect_tags(image_path, category=category)
        finally:
            ren.close()
    finally:
        _AIRENAMER_GLOBAL_CB = None


def ai_compute_renames(
    files: List[str],
    strategy: str,
    backend: str = "auto",
    category: Optional[str] = None,
    max_tags: int = 3,
    pad: int = 3,
    start: int = 1,
    truncate_len: int = 32,
    model: Optional[str] = None,
    progress_cb=None,
    force_reprocess: bool = False,
    mode: str = "auto",
    organize_options: Optional[Dict] = None,
) -> List[Tuple[str, str]]:
    """Build rename pairs using AI-detected tags.

    Constructs a single AIRenamer for the batch so the Ollama HTTP
    client and per-file cache are reused. For non-tag strategies the
    result is identical to ``build_renames``.

    Args:
        ... (see AIRenamer for the rest)
        mode: analyzer mode ("auto"/"lowlevel"/"fusion"/"clip"/"ollama").
            Default "auto" picks up the user's configured `organize_mode`
            so the rename uses the same analyzer that picked the category.
        force_reprocess: bypass the per-file cache and re-detect every
            image even if it was already processed in this batch.
        organize_options: Organize-style AI option overrides (NSFW,
            describe, classify, classify_method) used when backend="organize".
    """
    needs_tags = strategy in TAG_BASED_STRATEGIES
    if not needs_tags:
        return build_renames(
            files, strategy=strategy, category=category or "",
            start=start, pad=pad, truncate_len=truncate_len,
        )

    ren = AIRenamer(
        backend=backend, mode=mode, model=model, max_tags=max_tags,
        force_reprocess=force_reprocess,
        organize_options=organize_options,
    )
    try:
        tags_by_file: Dict[str, List[str]] = {}
        subject_by_file: Dict[str, Optional[str]] = {}
        total = len(files)
        for i, p in enumerate(files, 1):
            tags, subject = ren.detect_tags(p, category=category)
            tags_by_file[p] = tags
            subject_by_file[p] = subject
            if progress_cb:
                try:
                    progress_cb("progress", i, total, p)
                except Exception:
                    pass
        # Surface the renamer log to the global callback too.
        if progress_cb:
            for line in ren.log_lines:
                try:
                    progress_cb("log", line)
                except Exception:
                    pass
        return build_renames(
            files, strategy=strategy, category=category or "",
            start=start, pad=pad, truncate_len=truncate_len,
            tags_by_file=tags_by_file,
            subject_by_file=subject_by_file,
            max_tags=max_tags,
        )
    finally:
        ren.close()
