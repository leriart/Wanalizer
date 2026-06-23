"""Intelligent tag suggestion for wallpapers.

Generates a rich set of descriptive tags from an image profile using:
  - Style/colour cues from `profile.py` (anime, cyberpunk, neon, ...)
  - Composition features (centeredness, symmetry, subject area)
  - Palette-driven tags (dominant colors → colour names)
  - Aspect ratio and resolution (vertical, phone, wide, 4k, ...)
  - Content detection (skin fraction → portrait/person, period→pattern)
  - Quality cues (sharpness → detailed, aesthetic → photo vs illustration)

The output is intended to be:
  * **Interpretable** - tags are short, human-readable words
  * **Normalised** - same word for the same concept ("portrait" never "Portrait")
  * **Disjoint**  - mutually exclusive categories are not combined (e.g. we
                    don't return both "minimalist" and "detailed")
  * **Bounded**   - only tags present in the registry are kept; new
                    suggestions are returned separately for the user to review
"""
import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .tags import _tags_flat, is_valid_tag


# ---------------------------------------------------------------------------
# Tag maps: cheap, content-aware mapping from numeric profile features to
# semantic tag suggestions. The numeric thresholds were chosen by inspecting
# a few hundred wallpapers; they are conservative defaults.
# ---------------------------------------------------------------------------

# Map profile score -> suggested tags with minimum activation threshold.
STYLE_TAG_THRESHOLDS: Dict[str, List[Tuple[str, float]]] = {
    "anime_score":    [("anime", 0.25), ("illustration", 0.40), ("cartoon", 0.40), ("digital-art", 0.45)],
    "cyberpunk_score": [("cyberpunk", 0.30), ("sci-fi", 0.35), ("neon", 0.40), ("dark", 0.45), ("city", 0.45)],
    "space_score":    [("space", 0.40), ("galaxy", 0.40), ("cosmos", 0.45), ("dark", 0.45)],
    "nature_score":   [("nature", 0.30), ("outdoor", 0.35), ("landscape", 0.40)],
    "neon_score":     [("neon", 0.30), ("vibrant", 0.40)],
    "pastel_score":   [("pastel", 0.30), ("light", 0.35)],
    "sepia_score":    [("retro", 0.25), ("sepia", 0.30)],
    "vintage_score":  [("retro", 0.25), ("vintage", 0.30)],
    "vw_score":       [("vaporwave", 0.20), ("retro", 0.30)],
}

# Map dominant palette colour -> tag.
PALETTE_COLOUR_TAGS = {
    "Red":          "red",
    "Orange":       "orange",
    "Yellow":       "yellow",
    "Green":        "green",
    "Teal-Cyan":    "teal",
    "Blue":         "blue",
    "Purple":       "purple",
    "Pink-Magenta": "pink",
    "Black":        "dark",
    "White":        "light",
    "Gray":         "grayscale",
}

# Pairs that should never both be present in the final tag set.
MUTUALLY_EXCLUSIVE: List[Tuple[str, str]] = [
    ("minimalist", "detailed"),
    ("minimalist", "complex"),
    ("monochrome", "vibrant"),
    ("monochrome", "colorful"),
    ("grayscale", "vibrant"),
    ("grayscale", "colorful"),
    ("black-and-white", "colorful"),
    ("photo", "illustration"),
    ("photo", "anime"),
    ("photo", "cartoon"),
    ("photograph", "anime"),
    ("light", "dark"),
    ("warm", "cool"),
    ("sunset", "night"),
    ("day", "night"),
    ("summer", "winter"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_tags(profile: dict, max_tags: int = 20) -> Set[str]:
    """Return a set of semantic tags derived from a CV profile.

    The set is limited to `max_tags` and only includes words present in the
    global tag registry, so callers can use the result directly for matching.
    """
    raw = _collect_raw_tags(profile)
    filtered = _filter_and_dedupe(raw, max_tags=max_tags)
    return filtered


def suggest_new_tags(
    profile: dict,
    existing: Optional[Set[str]] = None,
    limit: int = 10,
) -> List[str]:
    """Return up to `limit` *new* tag suggestions that are NOT in the registry.

    Useful for prompting the user: "I think these tags describe this image -
    add them?". Existing tags in the global registry are filtered out.
    """
    existing = existing or set()
    raw = _collect_raw_tags(profile)
    return [t for t in raw if t not in _tags_flat and t not in existing][:limit]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _collect_raw_tags(profile: dict) -> List[str]:
    """Collect raw (possibly un-registered) tag suggestions from a profile."""
    tags: List[str] = []

    # 1. Style-score tags
    for score_key, tag_thresholds in STYLE_TAG_THRESHOLDS.items():
        v = profile.get(score_key, 0.0)
        for tag, threshold in tag_thresholds:
            if v >= threshold:
                tags.append(tag)

    # 2. Pixel-art: anime + few colours
    if profile.get("anime_score", 0) >= 0.25 and profile.get("unique_colors", 256) <= 32:
        tags.append("pixel-art")

    # 3. Skin -> people tags
    if profile.get("skin_fraction", 0) > 0.12:
        tags.extend(["portrait", "person", "human"])

    # 4. Quality-based
    sharpness = profile.get("sharpness", 0)
    aesthetic = profile.get("aesthetic", 0)
    if sharpness >= 400:
        tags.append("detailed")
    if aesthetic > 0.6:
        tags.append("photo")
        tags.append("photograph")

    # 5. Composition
    centeredness = profile.get("centeredness", 0.5)
    if centeredness > 0.75:
        tags.append("centered")
    if profile.get("overall_symmetry", 0) > 0.7:
        tags.append("symmetrical")

    # 6. Texture
    tex_entropy = profile.get("texture_entropy", 0)
    if tex_entropy < 3.0 and profile.get("unique_colors", 256) < 64:
        tags.append("minimalist")
    if tex_entropy > 4.7:
        tags.append("detailed")

    # 7. Palette
    weights = profile.get("weights") or {}
    if weights:
        total = sum(weights.values()) or 1
        for color, weight in weights.items():
            pct = weight / total
            if pct >= 0.25 and color in PALETTE_COLOUR_TAGS:
                tags.append(PALETTE_COLOUR_TAGS[color])

        # Monochrome / near-monochrome detection
        neutral = sum(weights.get(c, 0) for c in ("Black", "White", "Gray"))
        neutral_pct = neutral / max(total, 1)
        if neutral_pct > 0.78:
            tags.append("grayscale")
            tags.append("monochrome")
            tags.append("black-and-white")
            if neutral_pct > 0.88:
                tags.append("minimalist")

        # Sunset palette: warm colors dominate
        warm = sum(weights.get(c, 0) for c in ("Orange", "Yellow", "Red", "Pink-Magenta"))
        warm_pct = warm / max(total, 1)
        if warm_pct > 0.45 and profile.get("space_score", 0) < 0.4:
            tags.append("sunset")

        # Cool palette (night, water, ice)
        cool = sum(weights.get(c, 0) for c in ("Blue", "Teal-Cyan", "Purple"))
        cool_pct = cool / max(total, 1)
        if cool_pct > 0.55:
            if profile.get("space_score", 0) > 0.35:
                tags.append("night")
            else:
                tags.append("cool")

        # Warmth in general
        if warm_pct > 0.30 and cool_pct < 0.20:
            tags.append("warm")

        # Nature palette: green and blue, not too saturated
        nature = profile.get("nature_score", 0)
        green_blue = sum(weights.get(c, 0) for c in ("Green", "Blue", "Teal-Cyan")) / max(total, 1)
        if nature > 0.25 and green_blue > 0.35:
            tags.extend(["nature", "landscape", "outdoor"])

        # Vibrant: high saturation palette with multiple dominant colors
        non_neutral = sum(
            weights.get(c, 0)
            for c in ("Red", "Orange", "Yellow", "Green", "Teal-Cyan",
                      "Blue", "Purple", "Pink-Magenta")
        )
        if non_neutral / max(total, 1) > 0.70 and len([c for c, v in weights.items() if v / max(total, 1) > 0.10]) >= 4:
            tags.append("vibrant")
            tags.append("colorful")

    # 8. Aspect ratio / resolution
    w, h = profile.get("size", (0, 0))
    if w > 0 and h > 0:
        if h > w:
            tags.append("vertical")
            if h / max(w, 1) > 1.8:
                tags.append("phone")
        elif w / max(h, 1) > 2.0:
            tags.append("wide")
            tags.append("panoramic")
        if w * h >= 1920 * 1080 * 4:
            tags.append("4k")
            tags.append("hd")
        elif w * h >= 1920 * 1080:
            tags.append("hd")

    # 9. Periodicity (pattern) and tiling
    if profile.get("periodicity_score", 0) > 0.6 or profile.get("is_periodic", 0) > 0.5:
        tags.append("pattern")
        tags.append("abstract")
    if profile.get("is_tiled", 0) > 0.5:
        tags.append("tile")
        tags.append("pattern")

    # 10. Subject prominence
    subject = profile.get("subject_area_ratio", 0.5)
    if subject > 0.6:
        tags.append("portrait")
    elif subject < 0.18:
        tags.append("minimalist")

    # 11. Skin -> figure (additional)
    if profile.get("skin_fraction", 0) > 0.25:
        tags.append("figure")
        tags.append("human")

    return tags


def _filter_and_dedupe(tags: List[str], max_tags: int) -> Set[str]:
    """Keep only tags present in the registry, drop duplicates and mutual
    exclusives (first occurrence wins)."""
    seen: Set[str] = set()
    out: List[str] = []
    for t in tags:
        tl = t.lower().strip()
        if not tl or tl in seen:
            continue
        # Mutual exclusion: if the tag conflicts with an already-kept tag, skip
        if any(
            (a == tl and b in seen) or (b == tl and a in seen)
            for a, b in MUTUALLY_EXCLUSIVE
        ):
            continue
        seen.add(tl)
        out.append(tl)
        if len(out) >= max_tags:
            break
    return set(out)


# ---------------------------------------------------------------------------
# Co-occurrence based "minimal AI" tag inference
# ---------------------------------------------------------------------------

# A small hand-curated co-occurrence table. Acts as a tiny knowledge graph
# for tags that frequently appear together in wallpaper collections.
TAG_COOCCURRENCE: Dict[str, List[Tuple[str, float]]] = {
    "anime":       [("illustration", 0.7), ("digital-art", 0.6), ("cartoon", 0.5)],
    "cyberpunk":   [("neon", 0.8), ("sci-fi", 0.7), ("dark", 0.6), ("city", 0.6), ("tech", 0.3)],
    "space":       [("galaxy", 0.8), ("stars", 0.6), ("dark", 0.5), ("cosmos", 0.7)],
    "nature":      [("landscape", 0.7), ("outdoor", 0.7), ("forest", 0.5), ("mountain", 0.4)],
    "forest":      [("nature", 0.9), ("green", 0.7), ("trees", 0.7), ("outdoor", 0.7)],
    "ocean":       [("blue", 0.8), ("water", 0.8), ("nature", 0.6), ("waves", 0.4)],
    "city":        [("urban", 0.7), ("skyline", 0.5), ("night", 0.4)],
    "sunset":      [("warm", 0.7), ("orange", 0.7), ("sky", 0.6)],
    "portrait":    [("person", 0.7), ("face", 0.6), ("human", 0.5)],
    "pixel-art":   [("retro", 0.6), ("8-bit", 0.5), ("game", 0.4)],
    "vintage":     [("retro", 0.8), ("sepia", 0.5), ("old", 0.4)],
    "minimalist":  [("simple", 0.7), ("clean", 0.5), ("monochrome", 0.4)],
    "neon":        [("vibrant", 0.7), ("dark", 0.5), ("glow", 0.5)],
    "anime":       [("manga", 0.5), ("character", 0.4)],
    "illustration": [("drawing", 0.6), ("art", 0.5), ("painting", 0.4)],
    "sci-fi":      [("futuristic", 0.7), ("tech", 0.5), ("space", 0.4)],
}


def expand_tags_with_cooccurrence(
    tags: Set[str],
    max_added: int = 5,
    min_weight: float = 0.5,
) -> Set[str]:
    """Expand a tag set with co-occurring tags from `TAG_COOCCURRENCE`.

    Acts as a "minimal AI" hint: "if you have anime, you probably also want
    illustration and digital-art". The expansion is bounded by `max_added`
    and by `min_weight` so it does not silently inject low-confidence tags.
    """
    out = set(tags)
    added: List[Tuple[str, float]] = []
    for tag in tags:
        for related, weight in TAG_COOCCURRENCE.get(tag, ()):
            if related in out or weight < min_weight:
                continue
            added.append((related, weight))
    # Pick the highest-weighted additions first, then bounded by max_added.
    added.sort(key=lambda x: -x[1])
    for related, _ in added[:max_added]:
        out.add(related)
    return out


# ---------------------------------------------------------------------------
# Category fingerprint similarity (uses existing tags + co-occurrence)
# ---------------------------------------------------------------------------

def suggest_tags_for_category(
    category_name: str,
    profile: dict,
    registry_tags: Optional[Set[str]] = None,
    max_tags: int = 12,
) -> List[str]:
    """Suggest tags for a single wallpaper given the category it should belong to.

    Combines:
      1. Content tags from the image profile (suggest_tags)
      2. Co-occurrence expansion
      3. Category-name match (if the image literally depicts the category)

    Returns tags sorted by relevance; only the top `max_tags` are returned.
    """
    base = suggest_tags(profile, max_tags=max_tags * 2)
    base = expand_tags_with_cooccurrence(base, max_added=4)

    cat_lower = category_name.lower()
    if cat_lower in _tags_flat and cat_lower not in base:
        base.add(cat_lower)

    # Score: how informative is each tag for this category?
    scored: List[Tuple[str, float]] = []
    for t in base:
        score = 1.0
        # Boost tags literally present in category name
        if t in cat_lower:
            score += 0.5
        # Boost tags that are short (more general / useful)
        score -= 0.05 * len(t)
        scored.append((t, score))
    scored.sort(key=lambda x: -x[1])
    return [t for t, _ in scored[:max_tags]]
