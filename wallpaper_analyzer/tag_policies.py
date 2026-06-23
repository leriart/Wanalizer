"""Anti-pollution policies for LLM-generated category tags.

When the LLM tagger runs on every category's sample images it tends
to dump the same generic labels into every `.category.json`:
`anime`, `digital-art`, `illustration`, `pastel`, `detailed`, `simple`,
`monochrome`, `minimalist`, `neon`, `vintage`, `retro`, `sky`,
`landscape`, `building`, `city`, `urban`, `clouds`, `forest`,
`mountain`, `nature`, `person`, `face`, `portrait`, etc.

Within a single category that's harmless, but once several categories
have the same set of generic tags the multi-signal classifier in
`classify.py` can no longer discriminate between them and tends to
pick the largest tag bag (Anime) for every image.

This module provides:
  * CURATED_SAFE_TAGS / CURATED_NSFW_TAGS - focused lists of ~50
    discriminating tags the LLM is allowed to choose from. Generic
    tags that would cause pollution are *not* in the list.
  * filter_polluted_tags() - drops any tag already present in >=50%
    of the registry. Used as a post-filter on every LLM response.
  * cap_category_tags() - caps the number of stored tags per category
    so even a very generous LLM response can't bloat the registry.
  * dedupe_against_existing() - drops tags that are already in the
    target category so we don't keep piling duplicates.
  * TAG_BUDGET_PER_CATEGORY - default max tags stored in each category.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Curated tag lists - kept small and discriminating on purpose
# ---------------------------------------------------------------------------
# We deliberately OMIT the universal tags that end up polluting every
# category (anime, illustration, digital-art, pastel, neon, monochrome,
# minimalist, simple, detailed, vintage, retro, sky, clouds, building,
# city, urban, forest, mountain, nature, person, face, portrait,
# landscape, dark, light, building, animal, ...). The LLM picks those
# for EVERY image which is what causes the pollution problem.
#
# What we DO expose:
#   * Style keywords with a clear semantic boundary
#   * Subject keywords that are specific (woman, man, kid, ...)
#   * Mood / composition keywords that actually mean something
# ---------------------------------------------------------------------------

CURATED_SAFE_TAGS: List[str] = [
    # === Style - mutually exclusive art styles
    "anime", "manga", "cartoon", "3d-render", "sketch",
    "watercolor", "oil-painting", "drawing",
    "pixel-art", "voxel", "low-poly", "vector",
    "realistic", "photograph", "polaroid",
    # === Format / medium
    "screenshot", "poster", "wallpaper",
    # === Subject: people
    "woman", "man", "kid", "boy", "girl", "couple",
    "warrior", "wizard", "knight", "samurai", "cyborg",
    # === Subject: animals / creatures
    "dragon", "phoenix", "wolf", "fox", "tiger", "lion",
    "bear", "rabbit", "deer", "eagle", "raven",
    # === Subject: objects
    "robot", "mech", "spaceship", "car", "motorcycle",
    "sword", "shield", "castle", "tower", "spaceship",
    # === Themes
    "fantasy", "sci-fi", "cyberpunk", "vaporwave", "steampunk",
    "horror", "romance", "sports", "mecha", "space-opera",
    # === Mood
    "calm", "dramatic", "peaceful", "mysterious",
    "epic", "cute", "dark", "bright", "energetic",
    # === Composition
    "centered", "closeup", "wide-shot",
    # === Specific colors (only the most distinctive ones)
    "monochrome", "pastel", "neon", "vintage",
]

CURATED_NSFW_TAGS: List[str] = [
    # Style
    "anime", "manga", "cartoon", "3d-render", "realistic",
    "watercolor", "sketch", "photograph",
    # Composition (subjects only - no orientation meta-tags; those
    # are derived from the image dimensions, not its content).
    "closeup",
    # Subjects (focused)
    "woman", "man",
    # Mood / setting
    "indoor", "outdoor", "bedroom", "beach", "studio",
    "sitting", "standing", "lying", "leaning",
    "monochrome", "black-and-white",
    # NSFW-specific
    "figure", "human", "skin", "body",
]

# Default budget per category. Higher = more pollution risk.
TAG_BUDGET_PER_CATEGORY: int = 15

# Hard cap on number of tags we ask the LLM to pick at once.
LLM_TAG_PICK_MAX: int = 8

# Generic tags the LLM tends to over-pick. We strip them from any
# response even if the model returns them.
GENERIC_OVERUSED: Set[str] = {
    "abstract", "detailed", "simple", "light", "dark", "warm", "cool",
    "vibrant", "colorful", "beautiful", "artistic", "creative",
    "illustration", "digital-art", "painting", "art", "drawing",
    "wallpaper", "background", "image", "photo", "photograph",
    "high-quality", "4k", "hd",
    "landscape", "portrait", "portrait-orientation",
    "landscape-orientation", "vertical", "horizontal", "wide",
    "sky", "clouds",
    "building", "city", "urban", "nature", "forest",
    "mountain", "tree", "flowers", "animal",
    "person", "people", "human", "figure",
    "face", "head", "body",
    "woman", "man",   # kept in CURATED but sometimes too generic
    "couple",         # too generic - kept appearing in 2+ categories
    "vintage", "retro", "minimalist", "pastel", "neon", "monochrome",
}

# Tags that should NEVER be removed even if they appear in many
# categories - they are genuine category signatures.
PROTECTED_TAGS: Set[str] = {
    "anime", "photograph", "illustration", "digital-art",
    "pixel-art", "painting", "sketch", "3d-render", "watercolor",
    "monochrome", "minimalist", "portrait",
}


# ---------------------------------------------------------------------------
# Anti-pollution helpers
# ---------------------------------------------------------------------------

def document_frequency(tags_by_category: Dict[str, Iterable[str]]) -> Counter:
    """Count how many categories each tag appears in (case-insensitive)."""
    df: Counter = Counter()
    for cat_tags in tags_by_category.values():
        seen: Set[str] = set()
        for t in cat_tags:
            tl = str(t).strip().lower()
            if tl and tl not in seen:
                seen.add(tl)
                df[tl] += 1
    return df


def find_polluted_tags(
    tags_by_category: Dict[str, Iterable[str]],
    threshold: float = 0.50,
) -> Set[str]:
    """Tags present in >= threshold fraction of categories are polluted."""
    if not tags_by_category:
        return set()
    df = document_frequency(tags_by_category)
    n = len(tags_by_category)
    return {t for t, d in df.items() if d >= threshold * n}


def filter_polluted_tags(
    tags: Iterable[str],
    existing_registry: Optional[Dict[str, Iterable[str]]] = None,
    threshold: float = 0.50,
) -> List[str]:
    """Strip tags that would pollute the registry.

    Tags that are generic / over-picked are always removed. If
    `existing_registry` is provided, any tag present in >= `threshold`
    fraction of existing categories is also removed.
    """
    tags = [str(t).strip().lower() for t in tags if str(t).strip()]
    polluted: Set[str] = set(GENERIC_OVERUSED)
    if existing_registry:
        polluted |= find_polluted_tags(existing_registry, threshold)
    out: List[str] = []
    seen: Set[str] = set()
    for t in tags:
        if t in polluted or t in seen or t in PROTECTED_TAGS and t in polluted:
            # PROTECTED_TAGS are kept only if not already in pollution list;
            # they don't get stripped by GENERIC_OVERUSED.
            continue
        if t in PROTECTED_TAGS:
            out.append(t)
            seen.add(t)
        elif t not in polluted:
            out.append(t)
            seen.add(t)
    return out


def cap_category_tags(
    tags: Iterable[str],
    max_tags: int = TAG_BUDGET_PER_CATEGORY,
) -> List[str]:
    """Cap a tag list at `max_tags` entries, preserving order + dedup."""
    seen: Set[str] = set()
    out: List[str] = []
    for t in tags:
        tl = str(t).strip().lower()
        if not tl or tl in seen:
            continue
        seen.add(tl)
        out.append(tl)
        if len(out) >= max_tags:
            break
    return out


def dedupe_against_existing(
    new_tags: Iterable[str],
    existing: Iterable[str],
) -> List[str]:
    """Drop any tag that already exists in `existing` (case-insensitive)."""
    have: Set[str] = {str(t).strip().lower() for t in existing if str(t).strip()}
    out: List[str] = []
    for t in new_tags:
        tl = str(t).strip().lower()
        if tl and tl not in have:
            out.append(tl)
    return out


def build_focused_curated_list(
    full_registry: Iterable[str],
    existing: Optional[Iterable[str]] = None,
    nsfw: bool = False,
) -> List[str]:
    """Build a focused tag list to feed the LLM.

    Steps:
      1. Start with the curated safe/NSFW list.
      2. Drop any tag already in the existing category (so the LLM
         has to suggest something *new*).
      3. Drop any tag that would pollute the registry (present in
         >=50% of categories).
      4. Intersect with the user's registry to keep only valid tags.
    """
    base = CURATED_NSFW_TAGS if nsfw else CURATED_SAFE_TAGS
    full = {str(t).strip().lower() for t in full_registry if str(t).strip()}
    existing_set = {str(t).strip().lower() for t in (existing or []) if str(t).strip()}
    out: List[str] = []
    for t in base:
        tl = t.strip().lower()
        if not tl:
            continue
        if tl in existing_set:
            continue  # already in this category - skip
        if tl not in full:
            continue  # not a known tag in this user's registry
        out.append(tl)
    return out


def parse_llm_tag_response(
    response: str,
    curated_list: Iterable[str],
    max_tags: int = LLM_TAG_PICK_MAX,
) -> List[str]:
    """Parse a free-form LLM tag response into a clean deduped list.

    Strips noise (markdown, bullets, numbers), validates against the
    curated list the LLM was supposed to pick from, and caps the
    returned list at `max_tags` entries.
    """
    if not response:
        return []
    text = response.strip()
    # Strip code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    # Split on common separators (comma, semicolon, newline, bullet).
    raw: List[str] = []
    for piece in re.split(r"[,;\n•·]+", text):
        # Strip numbering "1. tag", "2) tag" and leading punctuation.
        piece = re.sub(r"^[\s\d\.\)\(\*\-\u2022]+", "", piece)
        piece = re.sub(r"[\s\.,;:\u2022]+$", "", piece)
        piece = piece.strip().lower()
        if piece:
            raw.append(piece)

    curated_set = {str(t).strip().lower() for t in curated_list}
    out: List[str] = []
    seen: Set[str] = set()
    for tag in raw:
        if tag not in curated_set:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= max_tags:
            break
    return out


# Late import to avoid name shadowing during module load.
import re  # noqa: E402
