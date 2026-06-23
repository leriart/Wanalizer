"""Per-category "expected content" configuration.

Each category gets a declarative `expected` block in `.category.json`
that tells the classifier / organiser / re-organiser what kinds of
images belong in this category:

    {
      "expected": {
        "aspect_ratios": ["horizontal", "vertical", "square"],
        "file_kinds":     ["image", "video", "animated"],
        "min_resolution": [1920, 1080],
        "color_palette":  ["dark", "warm", "cool"],
        "style_keywords": ["anime", "illustration"],
        "exclude_keywords": ["anime", "photograph"],
        "source":         "user"  # or "ai" / "ai-edited"
      }
    }

The config is set via one of three paths:

  * **Interactive questionnaire** (`ask_questionnaire`) - presents the
    user with a series of Q&A pages in `CategoryConfigDialog`.
  * **AI generator** (`generate_config_from_samples`) - looks at
    10-20 sample images in the folder, runs CLIP + CV profile, and
    proposes a config. The user can review and tweak.
  * **Manual JSON edit** - power-user escape hatch.

After the config exists it can be queried cheaply via
`get_expected(category)` (cached per process) and used as a soft
filter / tie-breaker everywhere a category is involved:

  * `classify.py` boosts scores when an image's CV features match
    the category's `expected.color_palette` / `style_keywords`.
  * `organize.py` refuses to put a clearly-out-of-spec image into a
    category (with a warning instead of a silent misfile).
  * `reorganize.py` shows the expected spec beside each category in
    the sidebar so the user can see at a glance what each folder is
    "for".
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional

from . import categories as c


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

ALL_ASPECT_RATIOS = ("horizontal", "vertical", "square", "any")
ALL_FILE_KINDS = ("image", "video", "animated")
ALL_PALETTES = ("dark", "warm", "cool", "neon", "pastel", "vibrant", "muted", "monochrome")
ALL_STYLE_KEYWORDS = (
    "anime", "illustration", "digital-art", "painting", "photograph",
    "cyberpunk", "sci-fi", "space", "nature", "landscape",
    "minimalist", "abstract", "vintage", "retro", "pixel-art",
    "voxel", "minecraft", "neon", "monochrome", "gradient",
    "portrait", "character", "fantasy", "dark", "light",
)

# Sensible default for a brand-new category: accept anything.
DEFAULT_EXPECTED = {
    "aspect_ratios": ["horizontal", "vertical", "square"],
    "file_kinds": ["image", "video", "animated"],
    "min_resolution": [0, 0],
    "color_palette": [],
    "style_keywords": [],
    "exclude_keywords": [],
    "source": "default",
}


# ---------------------------------------------------------------------------
# Public schema helpers
# ---------------------------------------------------------------------------

def get_expected(category: str) -> Dict:
    """Return the `expected` block for `category`, with defaults filled in.

    Cached per-process so multiple callers don't repeatedly read JSON.
    """
    cache_attr = f"_expected_cache_{category}"
    cached = getattr(get_expected, cache_attr, None)
    if cached is not None:
        return cached
    cfg = c.get_category_config(category)
    expected = dict(DEFAULT_EXPECTED)
    expected.update(cfg.get("expected") or {})
    setattr(get_expected, cache_attr, expected)
    return expected


def write_expected(category: str, expected: Dict, *,
                  merge: bool = False) -> Dict:
    """Persist `expected` into the category's `.category.json`.

    When `merge=True`, the new fields are merged into the existing
    `expected` block; otherwise the whole block is replaced.
    Returns the resulting `expected` dict.
    """
    cfg = c.get_category_config(category)
    if merge and cfg.get("expected"):
        merged = dict(cfg["expected"])
        merged.update(expected)
        expected = merged
    cfg["expected"] = dict(expected)
    cfg["expected"]["source"] = expected.get("source", "user")
    c.write_category_config(category, cfg)
    setattr(get_expected, f"_expected_cache_{category}", None)
    return cfg["expected"]


def invalidate_cache() -> None:
    """Drop all cached `expected` blocks (call after write_expected)."""
    for attr in list(vars(get_expected)):
        if attr.startswith("_expected_cache_"):
            delattr(get_expected, attr)


# ---------------------------------------------------------------------------
# Image-feature -> expected compatibility
# ---------------------------------------------------------------------------

def aspect_ratio_class(w: int, h: int) -> str:
    """Bucket an image's (w, h) into horizontal / vertical / square.

    Threshold: square if w/h within 1.15x of each other.
    """
    if w <= 0 or h <= 0:
        return "any"
    if abs(w - h) <= 0.15 * max(w, h):
        return "square"
    return "horizontal" if w > h else "vertical"


def file_kind_for(path: str) -> str:
    """Return 'image' / 'video' / 'animated' for `path` based on its
    extension. Falls back to 'image' for unknown formats."""
    from . import formats as f
    ext = os.path.splitext(path)[1].lower()
    if ext in f.STATIC_EXTENSIONS:
        return "image"
    if ext in {".gif", ".apng", ".mng", ".fli", ".flc"}:
        return "animated"
    if ext in f.ANIMATED_EXTENSIONS:
        return "video"
    return "image"


def palette_match(palette_weights: Dict[str, float], expected: Dict) -> float:
    """Return a 0..1 score for how well `palette_weights` matches the
    category's expected palette. Higher = better match.

    The score is the fraction of expected palette colours that are
    present in `palette_weights` at >=5% of the total. Empty expected
    palette returns 0.5 (neutral).
    """
    expected_palettes = set(expected.get("color_palette") or [])
    if not expected_palettes:
        return 0.5
    if not palette_weights:
        return 0.0
    total = sum(palette_weights.values()) or 1.0
    matched = sum(
        1 for p in expected_palettes
        if palette_weights.get(p, 0) / total >= 0.05
    )
    return matched / len(expected_palettes)


def keyword_match(img_keywords: List[str], expected: Dict) -> float:
    """Return a 0..1 score for how well an image's keywords match the
    category's expected style keywords, penalised by exclude_keywords.

    Uses F1 between (matched expected) and (matched exclude).
    """
    expected_kw = set(k.lower() for k in expected.get("style_keywords") or [])
    exclude_kw = set(k.lower() for k in expected.get("exclude_keywords") or [])
    if not expected_kw and not exclude_kw:
        return 0.5
    img_set = set(k.lower() for k in img_keywords)
    pos = len(expected_kw & img_set) / max(len(expected_kw), 1)
    neg = len(exclude_kw & img_set) / max(len(exclude_kw), 1)
    if expected_kw and exclude_kw:
        return max(0.0, pos - neg)
    if expected_kw:
        return pos
    # Only exclude list: invert so the score is "how few excludes match"
    return 1.0 - neg


def config_match_score(profile: dict, expected: Dict) -> float:
    """Composite 0..1 score: how well an image's profile matches a
    category's `expected` config. Used as a soft tie-breaker in the
    classifier.

    Combines palette match, keyword match, aspect-ratio match, and a
    resolution threshold. Each component contributes equally.
    """
    if not expected or expected.get("source") == "default":
        return 0.5
    palette = palette_match(profile.get("weights") or {}, expected)
    keywords = keyword_match(
        list((profile.get("_tags") or [])),  # type: ignore[arg-type]
        expected,
    )
    w, h = profile.get("size", (0, 0))
    ar = aspect_ratio_class(w, h)
    expected_ars = expected.get("aspect_ratios") or ["any"]
    ar_score = 1.0 if ("any" in expected_ars or ar in expected_ars) else 0.0
    min_w, min_h = expected.get("min_resolution") or [0, 0]
    res_ok = (w >= min_w and h >= min_h) if (min_w and min_h) else True
    res_score = 1.0 if res_ok else 0.3
    # Equal weight average
    return (palette + keywords + ar_score + res_score) / 4.0


# ---------------------------------------------------------------------------
# AI generator: infer `expected` from sample images in the category
# ---------------------------------------------------------------------------

def _profile_palette_buckets(palette: Dict[str, float]) -> List[str]:
    """Map a palette dict to the high-level buckets used by `expected`."""
    if not palette:
        return []
    total = sum(palette.values()) or 1.0
    pcts = {c: v / total for c, v in palette.items()}
    out = []
    # dark: black + dark blue + purple dominate
    dark = pcts.get("Black", 0) + pcts.get("Purple", 0) * 0.5 + pcts.get("Blue", 0) * 0.3
    if dark > 0.30:
        out.append("dark")
    # warm: red/orange/yellow
    warm = pcts.get("Red", 0) + pcts.get("Orange", 0) + pcts.get("Yellow", 0)
    if warm > 0.35:
        out.append("warm")
    # cool: blue/teal/green/purple
    cool = pcts.get("Blue", 0) + pcts.get("Teal-Cyan", 0) + pcts.get("Green", 0) * 0.5
    if cool > 0.30:
        out.append("cool")
    # neon: high saturation + bright + dark
    if pcts.get("Pink-Magenta", 0) + pcts.get("Teal-Cyan", 0) > 0.25:
        out.append("neon")
    # pastel: high white + low sat overall
    if pcts.get("White", 0) > 0.40 and max(pcts.values()) < 0.50:
        out.append("pastel")
    # monochrome: dominated by neutrals
    neutral = pcts.get("Black", 0) + pcts.get("White", 0) + pcts.get("Gray", 0)
    if neutral > 0.85:
        out.append("monochrome")
    return out


def _infer_style_keywords(profiles: List[dict]) -> List[str]:
    """Infer dominant style keywords from a batch of CV profiles.

    Aggregates the relevant CV scores across all samples and keeps the
    ones that consistently fire above a threshold. Avoids generic
    keywords that fire on every image.
    """
    if not profiles:
        return []
    keys = ("anime_score", "cyberpunk_score", "space_score",
            "nature_score", "neon_score", "pastel_score",
            "pixel_art_score", "minimalist_score", "minecraft_score")
    agg = {k: 0.0 for k in keys}
    for p in profiles:
        for k in keys:
            agg[k] += float(p.get(k, 0) or 0)
    n = len(profiles)
    out = []
    # Style-score to keyword mapping
    style_kw_map = {
        "anime_score": "anime",
        "cyberpunk_score": "cyberpunk",
        "space_score": "space",
        "nature_score": "nature",
        "neon_score": "neon",
        "pastel_score": "pastel",
        "pixel_art_score": "pixel-art",
        "minimalist_score": "minimalist",
        "minecraft_score": "minecraft",
    }
    for k, kw in style_kw_map.items():
        avg = agg[k] / n
        if avg >= 0.20:
            out.append(kw)
    return out


def generate_config_from_samples(
    category: str,
    max_samples: int = 20,
    profile_cache: Optional[Dict[str, dict]] = None,
    use_clip: bool = True,
) -> Dict:
    """Inspect up to `max_samples` images in `category` and propose a
    config dict that can be reviewed and saved.

    Strategy:
      1. Pick the first `max_samples` supported images.
      2. Run `get_image_profile` on each (or use `profile_cache`).
      3. Optionally run CLIP to add semantic keywords.
      4. Aggregate aspect ratios, palette buckets, style scores into
         the `expected` schema.
      5. Mark source="ai" so the user knows to review.
    """
    from . import formats as f
    from .profile import get_image_profile

    cat_dir = os.path.join(c.resolve_dest_dir(), category)
    if not os.path.isdir(cat_dir):
        return dict(DEFAULT_EXPECTED)

    exts = f.STATIC_EXTENSIONS
    files = sorted(
        os.path.join(cat_dir, fn) for fn in os.listdir(cat_dir)
        if not fn.startswith(".")
        and os.path.splitext(fn)[1].lower() in exts
        and os.path.isfile(os.path.join(cat_dir, fn))
    )[:max_samples]

    if not files:
        return dict(DEFAULT_EXPECTED)

    profiles: List[dict] = []
    full_sizes: List[Tuple[int, int]] = []
    for fp in files:
        if profile_cache is not None and fp in profile_cache:
            profiles.append(profile_cache[fp])
        else:
            try:
                profiles.append(get_image_profile(fp))
            except Exception:
                continue
        # The CV profile only knows the THUMBNAIL dimensions (200x200)
        # because it processes the resized copy. For the resolution
        # recommendation we need the REAL image dimensions, so we open
        # the file separately to read them.
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(fp) as im:
                full_sizes.append(im.size)
        except Exception:
            pass

    if not profiles:
        return dict(DEFAULT_EXPECTED)

    # 1. Aspect ratios - keep all that appear (not just the dominant one).
    ar_counter: Counter = Counter()
    for p in profiles:
        w, h = p.get("size", (0, 0))
        ar_counter[aspect_ratio_class(w, h)] += 1
    # Keep aspect ratios that appeared in at least 20% of samples, plus
    # 'any' as a safe fallback if distribution is wide.
    threshold = max(1, len(profiles) * 0.20)
    aspect_ratios = [ar for ar, n in ar_counter.items() if n >= threshold]
    if not aspect_ratios or len(aspect_ratios) == 3:
        aspect_ratios = ["any"]
    elif "any" not in aspect_ratios:
        aspect_ratios.append("any")  # so unrelated images still pass

    # 2. File kinds - check what's in the folder, not just samples.
    folder_exts = {os.path.splitext(fn)[1].lower()
                   for fn in os.listdir(cat_dir)
                   if not fn.startswith(".")
                   and os.path.isfile(os.path.join(cat_dir, fn))}
    image_exts = folder_exts & set(f.STATIC_EXTENSIONS)
    animated_exts = folder_exts & {".gif", ".apng", ".mng", ".fli", ".flc"}
    video_exts = folder_exts & f.ANIMATED_EXTENSIONS
    file_kinds: List[str] = []
    if image_exts: file_kinds.append("image")
    if animated_exts: file_kinds.append("animated")
    if video_exts: file_kinds.append("video")
    if not file_kinds: file_kinds = ["image"]

    # 3. Resolution - use the 10th percentile of sample widths as the
    # minimum so most images in the folder satisfy the threshold. We
    # use the REAL image dimensions (read separately from PIL) rather
    # than the thumbnail dimensions in the CV profile.
    real_widths = sorted(w for w, _ in full_sizes if w > 0)
    real_heights = sorted(h for _, h in full_sizes if h > 0)
    min_w = int(real_widths[len(real_widths) // 10]) if real_widths else 0
    min_h = int(real_heights[len(real_heights) // 10]) if real_heights else 0

    # 4. Palette buckets
    palette_counter: Counter = Counter()
    for pr in profiles:
        for bucket in _profile_palette_buckets(pr.get("weights") or {}):
            palette_counter[bucket] += 1
    color_palette = [p for p, _ in palette_counter.most_common(4)]

    # 5. Style keywords from CV profile + optional CLIP
    style_keywords = _infer_style_keywords(profiles)
    if use_clip:
        try:
            from .clip_client import get_engine
            engine = get_engine()
            if engine.available:
                from .clip_client import score_image
                keyword_votes: Counter = Counter()
                for fp in files[:10]:
                    scores = score_image(fp)
                    if not scores:
                        continue
                    # Map top CLIP category -> keyword
                    top_cat = max(scores, key=scores.get)
                    kw_map = {
                        "Anime": "anime", "Pixel-Art": "pixel-art",
                        "Cyberpunk": "cyberpunk", "Space": "space",
                        "Minecraft": "minecraft", "Landscape": "nature",
                        "Minimalist": "minimalist", "Monochrome": "monochrome",
                        "Neon": "neon",
                    }
                    if top_cat in kw_map:
                        keyword_votes[kw_map[top_cat]] += 1
                # CLIP keywords that appear in >=30% of samples
                for kw, n in keyword_votes.items():
                    if n >= max(1, len(files[:10]) * 0.30):
                        if kw not in style_keywords:
                            style_keywords.append(kw)
        except Exception:
            pass

    return {
        "aspect_ratios": aspect_ratios,
        "file_kinds": file_kinds,
        "min_resolution": [min_w, min_h],
        "color_palette": color_palette,
        "style_keywords": style_keywords,
        "exclude_keywords": [],
        "source": "ai",
    }
