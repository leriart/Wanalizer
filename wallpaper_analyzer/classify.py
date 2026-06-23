"""Classification logic: multi-signal scoring for category assignment.

Two complementary strategies are supported:

  * `classify(profile)` - the legacy entry point used by the organize pipeline.
    It chains: tag-match -> prompt-match -> palette-fallback.

  * `classify_with_confidence(profile)` - the richer scorer used by the GUI
    and the analyser modes. It combines:
      - TF-IDF-style tag matching against category tags
      - Multi-signal heuristic scoring (palette, content, composition, style)
      - Z-score similarity against each category's CV fingerprint
      - Mini-AI tag prediction (when available)

The output always includes the chosen category, a confidence value in
[0, 1], and the per-category score breakdown so the UI can show the user
why a given category was chosen.
"""
import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from . import categories as c
from .categories import (
    CATEGORIES,
    get_category_config,
    get_category_prompt,
    get_category_tags,
    get_palette_weights,
)
from .tags import _tags_flat
from .tag_suggester import (
    MUTUALLY_EXCLUSIVE,
    PALETTE_COLOUR_TAGS,
    STYLE_TAG_THRESHOLDS,
    expand_tags_with_cooccurrence,
    suggest_tags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) or 1
    return {c: v / total for c, v in weights.items()}


def _size_signal(profile: dict) -> Optional[str]:
    """Map aspect-ratio / resolution to a hard bucket, if any."""
    w, h = profile.get("size", (0, 0))
    if not w or not h:
        return None
    if h > w:
        if h / max(w, 1) > 1.8:
            return "phone"
        return "vertical"
    if w / max(h, 1) > 2.0:
        return "wide"
    return "landscape"


# ---------------------------------------------------------------------------
# Tag-based classifier (TF-IDF style)
# ---------------------------------------------------------------------------

def derive_tags_from_profile(profile: dict) -> Set[str]:
    """Derive a set of descriptive tags from a CV profile (delegates to
    `tag_suggester`). The returned set is filtered to the global tag
    registry, so it can be matched directly against category tag sets."""
    return suggest_tags(profile, max_tags=24)


def classify_by_tags(image_tags: Set[str]) -> Optional[str]:
    """Match image tags against category tags.

    Uses two complementary signals:
      * Tag-set overlap (count of shared tags)
      * TF-IDF-style weighting (rarer tags are more informative)

    Returns the category with the highest weighted overlap, or None if
    nothing matched.
    """
    if not image_tags:
        return None

    # Document frequency: how many categories use each tag.
    df: Counter = Counter()
    cat_tags_map: Dict[str, Set[str]] = {}
    for cat in CATEGORIES:
        ctags = set(t.lower() for t in get_category_tags(cat))
        cat_tags_map[cat] = ctags
        for t in ctags:
            df[t] += 1
    n_cats = max(len(CATEGORIES), 1)

    # IDF: rarer tags are more informative
    idf = {t: math.log((1 + n_cats) / (1 + d)) + 1 for t, d in df.items()}

    best_cat: Optional[str] = None
    best_score = 0.0
    best_breakdown: Dict[str, float] = {}
    for cat, cat_tags in cat_tags_map.items():
        if not cat_tags:
            continue
        overlap = image_tags & cat_tags
        if not overlap:
            continue
        score = sum(idf.get(t, 1.0) for t in overlap)
        # Boost exact category-name match
        if cat.lower() in image_tags:
            score += 3.0
        if score > best_score:
            best_score = score
            best_cat = cat
            best_breakdown = {t: idf.get(t, 1.0) for t in overlap}
    return best_cat


# ---------------------------------------------------------------------------
# Prompt text matching (Ollama description vs. category prompt)
# ---------------------------------------------------------------------------

def _build_prompt_idf() -> Dict[str, float]:
    """Build inverse-document-frequency over category prompts.

    Words that appear in many prompts ("wallpaper", "the", "in") get
    weight ~0; distinctive words ("cyberpunk", "voxel", "monochrome")
    get the highest weight. Cached for the process lifetime.
    """
    global _PROMPT_IDF_CACHE
    if _PROMPT_IDF_CACHE is not None:
        return _PROMPT_IDF_CACHE
    df: Counter = Counter()
    n = 0
    for cat in CATEGORIES:
        prompt = get_category_prompt(cat)
        if not prompt:
            continue
        n += 1
        for w in set(prompt.lower().split()):
            df[w] += 1
    if n == 0:
        _PROMPT_IDF_CACHE = {}
        return _PROMPT_IDF_CACHE
    # Smoothed IDF: rare words get high weight, common words get ~1.
    _PROMPT_IDF_CACHE = {
        w: math.log((1 + n) / (1 + d)) + 1.0 for w, d in df.items()
    }
    return _PROMPT_IDF_CACHE


def classify_by_prompt_text(text: str) -> Optional[str]:
    """Match text against category prompts using TF-IDF cosine similarity.

    Replaces the previous binary word-overlap scoring which was dominated
    by common words like "wallpaper" and "in" (every prompt has them).
    TF-IDF downweights common words and rewards distinctive vocabulary
    that only appears in one or two categories ("cyberpunk", "monochrome",
    "voxel", "vaporwave", etc.). A category wins only when its prompt
    is clearly more similar than the runner-up.
    """
    if not text:
        return None
    text_lower = text.lower()
    text_counts = Counter(text_lower.split())
    idf = _build_prompt_idf()
    if not idf:
        return None
    text_vec = {w: c * idf.get(w, 1.0) for w, c in text_counts.items()}
    text_norm = math.sqrt(sum(v * v for v in text_vec.values())) or 1.0

    scores: Dict[str, float] = {}
    for cat in CATEGORIES:
        prompt = get_category_prompt(cat)
        if not prompt:
            continue
        p_counts = Counter(prompt.lower().split())
        p_vec = {w: c * idf.get(w, 1.0) for w, c in p_counts.items()}
        p_norm = math.sqrt(sum(v * v for v in p_vec.values())) or 1.0
        common = set(text_vec) & set(p_vec)
        if not common:
            continue
        dot = sum(text_vec[w] * p_vec[w] for w in common)
        scores[cat] = dot / (text_norm * p_norm)
    if not scores:
        return None
    sorted_scores = sorted(scores.values(), reverse=True)
    # Need both an absolute threshold AND a clear winner.
    best_cat = max(scores, key=scores.get)
    if sorted_scores[0] < 0.10:
        return None
    if len(sorted_scores) >= 2 and sorted_scores[0] < 1.4 * sorted_scores[1]:
        return None  # too ambiguous
    return best_cat


# ---------------------------------------------------------------------------
# Palette-only fallback
# ---------------------------------------------------------------------------

def palette_fallback(profile: dict) -> Optional[str]:
    weights = profile.get("weights", {})
    if not weights:
        return CATEGORIES[0] if CATEGORIES else None
    pcts = _normalise_weights(weights)
    best_cat = CATEGORIES[0] if CATEGORIES else None
    best_score = -1.0
    for cat in CATEGORIES:
        pw = get_palette_weights(cat)
        if not pw:
            continue
        score = sum(pcts.get(color, 0.0) * w for color, w in pw.items())
        if score > best_score:
            best_score = score
            best_cat = cat
    if best_cat is None:
        return None
    return best_cat if best_score > 0.15 else (CATEGORIES[0] if CATEGORIES else None)


# ---------------------------------------------------------------------------
# Multi-signal scorer
# ---------------------------------------------------------------------------

# Per-signal weight. The fingerprint signal is the most reliable
# (it compares the image's CV profile against the actual feature
# distribution of each category) so it gets the largest weight.
# Tag/style signals can be polluted when categories were generated
# by an over-permissive LLM, so we keep their weights moderate.
SIGNAL_WEIGHTS: Dict[str, float] = {
    # Per-signal weight. The fingerprint signal is the most reliable
    # (it compares the image's CV profile against the actual feature
    # distribution of each category) so it gets the largest weight.
    # Theme and content signals are auxiliary: they only fire for the
    # matching category so they don't penalise other categories when
    # they're absent.
    "tags":         0.10,
    "palette":      0.10,
    "style":        0.08,
    "content":      0.05,
    "theme":        0.07,
    "composition":  0.05,
    "quality":      0.03,
    "pattern":      0.04,
    "size":         0.03,
    "prompt":       0.08,
    "clip":         0.25,
    "clip_nsfw":    0.05,
    "fingerprint":  0.20,
}


def _palette_signal(profile: dict) -> Dict[str, float]:
    """Score categories by palette match."""
    weights = profile.get("weights", {})
    if not weights:
        return {}
    pcts = _normalise_weights(weights)
    scores: Dict[str, float] = {}
    for cat in CATEGORIES:
        pw = get_palette_weights(cat)
        if not pw:
            continue
        s = sum(pcts.get(color, 0.0) * w for color, w in pw.items())
        scores[cat] = min(max(s, 0.0), 1.0)
    return scores


def _ramp_above(v: float, threshold: float, ceiling: float, ramp: float = 0.20) -> float:
    """Linearly scale from 0 (at `threshold`) to `ceiling` (at threshold+ramp).

    A *continuous* replacement for the old binary threshold bonus. With the
    defaults, an `anime_score` of `threshold` contributes 0, of
    `threshold + ramp` contributes `ceiling`, and anything above is capped.
    """
    if v <= threshold:
        return 0.0
    if ramp <= 0:
        return ceiling if v > threshold else 0.0
    excess = (v - threshold) / ramp
    return min(ceiling, ceiling * excess)


def _style_signal(profile: dict) -> Dict[str, float]:
    """Score categories by their explicit tag hints.

    The previous version used BINARY thresholds (fire +1.2 if above, 0 if
    below) so a "very anime" image and a "barely anime" image scored the
    same on the Anime category. This version is CONTINUOUS: each bonus
    ramps from 0 at the threshold to its ceiling at threshold + ramp.

    Anti-pollution still applies: when too many categories share a tag
    (e.g. 'sci-fi' ended up in 8 of 17 categories), the tag is dropped
    from contributing here so the more specific tags can win.
    """
    cat_tags_map = {cat: set(t.lower() for t in get_category_tags(cat)) for cat in CATEGORIES}
    # Document frequency per style tag - filter out tags that appear in
    # too many categories (they stop being discriminative).
    style_df: Counter = Counter()
    for ctags in cat_tags_map.values():
        for key, tag_thresholds in STYLE_TAG_THRESHOLDS.items():
            for tag, _ in tag_thresholds:
                if tag in ctags:
                    style_df[tag] += 1
    n = max(len(CATEGORIES), 1)
    informative_style_tags = {
        tag for tag, d in style_df.items()
        if math.log((1 + n) / (1 + d)) + 1 >= 1.10
    }

    scores: Dict[str, float] = {}
    for cat, ctags in cat_tags_map.items():
        if not ctags:
            continue
        score = 0.0
        # Per STYLE_TAG_THRESHOLDS, each match now scales with magnitude.
        for key, tag_thresholds in STYLE_TAG_THRESHOLDS.items():
            v = float(profile.get(key, 0.0))
            for tag, threshold in tag_thresholds:
                if (tag in ctags
                        and tag in informative_style_tags
                        and v >= threshold):
                    # Continuous: 0 at threshold, 1.0 at threshold + 0.20.
                    score += _ramp_above(v, threshold, 1.0, ramp=0.20)

        # Continuous categorical bonuses (old +1.2 → ramped 0..1.2)
        if "portrait" in ctags:
            score += _ramp_above(
                float(profile.get("skin_fraction", 0.0)),
                threshold=0.10, ceiling=1.5, ramp=0.15,
            )
        if {"anime", "illustration"} & ctags:
            score += _ramp_above(
                float(profile.get("anime_score", 0.0)),
                threshold=0.25, ceiling=1.2, ramp=0.25,
            )
        if "minimalist" in ctags:
            # Both conditions must be met; ramp when both "low" enough.
            tex = float(profile.get("texture_entropy", 6.0))
            uniq = float(profile.get("unique_colors", 256))
            # combined score: lower texture + fewer unique colors → higher bonus
            tex_bonus = max(0.0, (4.0 - tex) / 1.5) if tex < 4.0 else 0.0
            uniq_bonus = max(0.0, (96 - uniq) / 64) if uniq < 96 else 0.0
            if tex_bonus > 0 and uniq_bonus > 0:
                score += 1.2 * min(tex_bonus, uniq_bonus)
        if {"photo", "photograph"} & ctags:
            aest = float(profile.get("aesthetic", 0.0))
            sharp = float(profile.get("sharpness", 0))
            a_bonus = _ramp_above(aest, threshold=0.5, ceiling=0.7, ramp=0.3)
            s_bonus = _ramp_above(sharp, threshold=200.0, ceiling=0.5, ramp=300.0)
            if a_bonus > 0 and s_bonus > 0:
                score += a_bonus + s_bonus
        if "pattern" in ctags:
            score += _ramp_above(
                float(profile.get("periodicity_score", 0.0)),
                threshold=0.4, ceiling=1.2, ramp=0.3,
            )
        if "abstract" in ctags:
            score += _ramp_above(
                float(profile.get("gradient_complexity", 0.0)),
                threshold=0.4, ceiling=0.6, ramp=0.3,
            )
        scores[cat] = min(score / 3.0, 1.0)
    return scores


def _composition_signal(profile: dict) -> Dict[str, float]:
    cat_tags_map = {cat: set(t.lower() for t in get_category_tags(cat)) for cat in CATEGORIES}
    scores: Dict[str, float] = {}
    for cat, ctags in cat_tags_map.items():
        s = 0.0
        if {"portrait", "person", "human"} & ctags:
            if 0.2 < profile.get("subject_area_ratio", 0) < 0.6:
                s += 1.0
        if {"landscape", "nature", "outdoor"} & ctags:
            if profile.get("subject_area_ratio", 0.5) < 0.3:
                s += 1.0
        if {"minimalist", "simple"} & ctags:
            if profile.get("subject_area_ratio", 0.5) < 0.2:
                s += 0.8
        if {"centered", "symmetrical"} & ctags:
            if profile.get("centeredness", 0) > 0.7 or profile.get("overall_symmetry", 0) > 0.7:
                s += 0.8
        scores[cat] = min(s / 2.0, 1.0)
    return scores


def _quality_signal(profile: dict) -> Dict[str, float]:
    cat_tags_map = {cat: set(t.lower() for t in get_category_tags(cat)) for cat in CATEGORIES}
    scores: Dict[str, float] = {}
    ten = profile.get("tenengrad", 0)
    sharp = profile.get("sharpness", 0)
    aesthetic = profile.get("aesthetic", 0)
    for cat, ctags in cat_tags_map.items():
        s = 0.0
        if {"photo", "photograph"} & ctags and ten > 100:
            s += 1.0
        if {"digital-art", "illustration"} & ctags and ten > 50:
            s += 0.7
        if "detailed" in ctags and sharp > 400:
            s += 0.7
        if "high-quality" in ctags and aesthetic > 0.7:
            s += 0.8
        scores[cat] = min(s / 2.0, 1.0)
    return scores


def _pattern_signal(profile: dict) -> Dict[str, float]:
    cat_tags_map = {cat: set(t.lower() for t in get_category_tags(cat)) for cat in CATEGORIES}
    scores: Dict[str, float] = {}
    for cat, ctags in cat_tags_map.items():
        s = 0.0
        if "pattern" in ctags:
            s += max(profile.get("periodicity_score", 0), profile.get("is_periodic", 0))
        if "abstract" in ctags and profile.get("gradient_complexity", 0) > 0.5:
            s += 0.6
        if {"minimalist", "simple"} & ctags and profile.get("texture_complexity", 0) < 0.3:
            s += 0.6
        scores[cat] = min(s, 1.0)
    return scores


def _content_signal(profile: dict) -> Dict[str, float]:
    """Score categories using the new content detectors added in
    `profile.py`: `minecraft_score`, `pixel_art_score`, `minimalist_score`,
    `portrait_score`, `nsfw_score`, `gradient_score`.

    These detectors fire on VERY specific visual signatures that the old
    tag-based heuristics could only catch with binary thresholds. Each
    is gated by a tag match so we only credit categories whose tags
    actually describe the content.
    """
    cat_tags_map = {cat: set(t.lower() for t in get_category_tags(cat)) for cat in CATEGORIES}
    scores: Dict[str, float] = {}
    for cat, ctags in cat_tags_map.items():
        s = 0.0
        # Minecraft/voxel/block - require voxel/minecraft/game tag
        if {"voxel", "minecraft", "block", "lego", "game", "8-bit", "pixel-art"} & ctags:
            s += float(profile.get("minecraft_score", 0)) * 1.5
        # Pixel-art - require pixel-art tag
        if "pixel-art" in ctags or "8-bit" in ctags or "retro" in ctags:
            s += float(profile.get("pixel_art_score", 0)) * 1.5
        # Minimalist - require minimalist tag
        if {"minimalist", "simple", "clean", "monochrome"} & ctags:
            s += float(profile.get("minimalist_score", 0)) * 1.4
        # Portrait - require portrait/person tag (not anime which uses portrait
        # as composition shorthand)
        if {"portrait", "person", "human", "face", "figure"} & ctags - {"anime"}:
            s += float(profile.get("portrait_score", 0)) * 1.2
        # NSFW - require nsfw/explicit tag
        if {"nsfw", "18+", "explicit", "ecchi", "hentai", "mature"} & ctags:
            s += float(profile.get("nsfw_score", 0)) * 1.6
        # Gradient/abstract backgrounds
        if {"gradient", "abstract"} & ctags:
            s += float(profile.get("gradient_score", 0)) * 0.8
        scores[cat] = min(s / 2.0, 1.0)
    return scores



def _prompt_signal(profile: dict) -> Dict[str, float]:
    """TF-IDF cosine similarity between an image's generated prompt and
    each category's stored prompt.

    The image's prompt is built on the fly from the profile via
    `prompt_generator.generate_prompt(profile, "concise")` so the
    classifier can use prompt similarity without needing Ollama/CLIP.
    Common prompt words ("wallpaper", "in", "the") are down-weighted
    by the IDF so distinctive vocabulary ("cyberpunk", "voxel",
    "monochrome", "pixel-art", "vaporwave") drives the match.

    Returns per-category scores in [0, 1]. Categories whose prompts share
    no distinctive vocabulary with the image's prompt score 0.
    """
    try:
        from .prompt_generator import generate_prompt
        text = generate_prompt(profile, style="concise")
    except Exception:
        return {}
    return _classify_by_prompt_scores(text)


def _classify_by_prompt_scores(text: str) -> Dict[str, float]:
    """Compute TF-IDF cosine similarity between `text` and every category prompt.

    Returns an empty dict if no category prompt has overlapping distinctive
    vocabulary (e.g. when the image's prompt is too generic). This is
    the per-category scorer extracted so both `_prompt_signal` (which
    generates from the profile) and `classify_by_prompt_text` (which is
    called from the Ollama path with a free-text description) share the
    same scoring logic.
    """
    if not text:
        return {}
    text_lower = text.lower()
    text_counts = Counter(text_lower.split())
    idf = _build_prompt_idf()
    if not idf:
        return {}
    text_vec = {w: c * idf.get(w, 1.0) for w, c in text_counts.items()}
    text_norm = math.sqrt(sum(v * v for v in text_vec.values())) or 1.0

    scores: Dict[str, float] = {}
    for cat in CATEGORIES:
        prompt = get_category_prompt(cat)
        if not prompt:
            continue
        p_counts = Counter(prompt.lower().split())
        p_vec = {w: c * idf.get(w, 1.0) for w, c in p_counts.items()}
        p_norm = math.sqrt(sum(v * v for v in p_vec.values())) or 1.0
        common = set(text_vec) & set(p_vec)
        if not common:
            continue
        dot = sum(text_vec[w] * p_vec[w] for w in common)
        scores[cat] = dot / (text_norm * p_norm)
    return scores


def _theme_signal(profile: dict) -> Dict[str, float]:
    """Score categories by palette match to named themes (Catppuccin,
    Dracula, TokyoNight, Nord, Gruvbox, Everforest, Monochrome, Neon).

    Each theme has a pre-computed palette fingerprint in `profile.py`;
    this signal is the per-category cosine similarity scaled to [0, 1].
    Only categories whose NAME matches a known theme participate (e.g.
    "Catppuccin" -> theme_catppuccin_score). This is the primary
    discriminator for the colour-themed folders.
    """
    theme_keys = {
        "catppuccin": "theme_catppuccin_score",
        "dracula":    "theme_dracula_score",
        "tokyonight": "theme_tokyonight_score",
        "nord":       "theme_nord_score",
        "gruvbox":    "theme_gruvbox_score",
        "everforest": "theme_everforest_score",
        "monochrome": "theme_monochrome_score",
        "neon":       "theme_neon_score",
    }
    scores: Dict[str, float] = {}
    for cat in CATEGORIES:
        cat_lower = cat.lower()
        for theme_name, profile_key in theme_keys.items():
            if theme_name in cat_lower:
                scores[cat] = float(profile.get(profile_key, 0))
                break
        else:
            scores[cat] = 0.0
    return scores


def _clip_signal(profile: dict) -> Dict[str, float]:
    """Score categories using CLIP zero-shot classification (if available).

    CLIP scores are produced by `clip_client.CLIPAnalyzer.analyze` and
    stored on the profile as `clip_score_<cat>` (softmax probabilities,
    already calibrated). When present, this signal is the single
    most discriminative one for "what is this an image OF" because
    CLIP has seen millions of image-text pairs.

    If CLIP hasn't been run (e.g. the default lowlevel pipeline which
    doesn't include CLIP) or the scores are missing, this signal
    returns zeros so it doesn't penalise any category.

    The per-category scores are calibrated softmax probabilities
    (each in [0, 1], sum ~= 1), so the magnitude is comparable to the
    other signals. The downstream `classify_with_confidence` weighs
    this signal with the same total budget as the other signals.
    """
    scores: Dict[str, float] = {}
    for cat in CATEGORIES:
        key = f"clip_score_{cat}"
        if key in profile:
            try:
                scores[cat] = float(profile[key])
            except (TypeError, ValueError):
                scores[cat] = 0.0
        else:
            scores[cat] = 0.0
    # If no CLIP scores at all, return empty so the signal weight
    # does not artificially inflate uniform-zero baselines.
    if not any(v > 0 for v in scores.values()):
        return {}
    # Mild spread-stretching: when CLIP is confident but flat (e.g.
    # top-2 within 0.05) we leave it alone; when very confident we
    # already have a sharp distribution from softmax so nothing to do.
    return scores


def _clip_nsfw_signal(profile: dict) -> Dict[str, float]:
    """NSFW-only signal from CLIP (if CLIP has been run).

    If `clip_nsfw` is on the profile and the image's true label is
    one of the NSFW categories, boost that category; otherwise no
    effect. This avoids the problem of CLIP misclassifying NSFW
    images as Anime (because they're both anime-styled) when the
    only correct answer is NSFW.
    """
    if "clip_nsfw" not in profile:
        return {}
    try:
        nsfw = float(profile["clip_nsfw"])
    except (TypeError, ValueError):
        return {}
    if nsfw < 0.30:
        return {}  # below this threshold CLIP isn't saying anything useful
    nsfw_cats = {"nsfw", "18+", "explicit", "ecchi", "hentai", "mature"}
    scores: Dict[str, float] = {}
    for cat in CATEGORIES:
        if cat.lower() in nsfw_cats:
            scores[cat] = nsfw
    return scores


def _size_signal_scores(profile: dict) -> Dict[str, float]:
    sig = _size_signal(profile)
    if sig is None:
        return {}
    cat_tags_map = {cat: set(t.lower() for t in get_category_tags(cat)) for cat in CATEGORIES}
    scores: Dict[str, float] = {}
    for cat, ctags in cat_tags_map.items():
        s = 0.0
        if sig == "phone" and "phone" in ctags:
            s += 1.0
        if sig == "vertical" and "vertical" in ctags:
            s += 0.6
        if sig == "wide" and {"wide", "panoramic"} & ctags:
            s += 1.0
        if "landscape" in ctags and sig == "landscape" and profile.get("subject_area_ratio", 0.5) < 0.3:
            s += 0.5
        scores[cat] = min(s, 1.0)
    return scores


# Lazily-built TF-IDF table for category-prompt matching (see
# `_build_prompt_idf` and the `_prompt_signal` / `classify_by_prompt_text`
# helpers).
_PROMPT_IDF_CACHE: Optional[Dict[str, float]] = None

# Module-level cache for the pooled feature stats. Refreshed by
# `classify_with_confidence` (the cache lives for the process lifetime
# unless explicitly invalidated).
_GLOBAL_STATS_CACHE: Dict = {"stats": None, "dir": None}


def _get_global_stats() -> Dict:
    """Load pooled feature stats from `<dest>/.heuristic_global_stats.json`.

    Cached per-process. Invalidated automatically when the configured
    categories directory changes (so different wallpaper libraries don't
    share a stale cache). Call `invalidate_global_stats_cache()` after
    rebuilding patterns in the SAME process to pick up the fresh cache.
    """
    current_dir = c.CATEGORIES_DIR or ""
    if _GLOBAL_STATS_CACHE["stats"] is not None and _GLOBAL_STATS_CACHE["dir"] == current_dir:
        return _GLOBAL_STATS_CACHE["stats"]
    try:
        from .lowlevel.category_profile import load_global_stats
        stats = load_global_stats(current_dir)
    except Exception:
        stats = {"features": {}}
    _GLOBAL_STATS_CACHE["stats"] = stats
    _GLOBAL_STATS_CACHE["dir"] = current_dir
    return stats


def invalidate_global_stats_cache() -> None:
    """Drop the cached global stats. Call after rebuilding patterns."""
    _GLOBAL_STATS_CACHE["stats"] = None
    _GLOBAL_STATS_CACHE["dir"] = None


def _fingerprint_signal(profile: dict) -> Dict[str, float]:
    """Z-score similarity against each category's CV fingerprint.

    The scoring is *discriminative* when global stats are available:
    features that DISTINGUISH this category from the rest get more weight
    (e.g. `cyberpunk_score` matters a lot for the Cyberpunk folder, but
    `hsv_V_mean` matters little because every category has a similar
    brightness distribution).
    """
    try:
        from .lowlevel.category_profile import score_against_pattern
    except ImportError:
        return {}
    global_stats = _get_global_stats() if c.CATEGORIES_DIR else {"features": {}}
    # If the cache file doesn't exist (e.g. before rebuilding patterns)
    # `global_stats` will be empty and `score_against_pattern` falls back
    # to the un-discriminative behaviour automatically.
    scores: Dict[str, float] = {}
    for cat in CATEGORIES:
        cfg = get_category_config(cat)
        pattern = cfg.get("heuristic_pattern") or {}
        if not pattern:
            continue
        try:
            s = score_against_pattern(profile, pattern, global_stats=global_stats)
        except Exception:
            continue
        scores[cat] = float(s)
    return scores


def _tags_signal(profile: dict) -> Dict[str, float]:
    """TF-IDF style tag matching, returned as a normalised score per cat.

    Four layers of pollution defence (the user's category registry on
    this kind of corpus tends to be over-polluted by an over-generous
    LLM tag generator that drops 'anime', 'pastel', 'blue', etc. into
    every category):
      1. Stop-word filter: tags present in >= 35% of categories are
         excluded entirely (the worst offenders like 'anime',
         'sci-fi', 'black', 'detailed'). Stricter than the
         previous 50% threshold because the new content/theme signals
         pick up the slack, so we can afford to be more aggressive
         about dropping uninformative tags here.
      2. Average-IDF penalty: when most matches are still common tags
         the registry is polluted for those tags and the whole score
         is scaled down.
      3. Best-score normalisation keeps the signal in [0, 1].
      4. Global pollution discount: if >60% of categories score above
         zero on this image the registry is too polluted for this
         signal to be useful; we return zeros so fingerprint can win.
    """
    img_tags = suggest_tags(profile, max_tags=20)
    if not img_tags:
        return {}
    cat_tags_map: Dict[str, set] = {
        cat: set(t.lower() for t in get_category_tags(cat)) for cat in CATEGORIES
    }
    df: Counter = Counter()
    for ctags in cat_tags_map.values():
        for t in ctags:
            df[t] += 1
    n = max(len(CATEGORIES), 1)
    idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}

    stop_words = {t for t, d in df.items() if d >= 0.35 * n}
    informative_tags = {t for t in img_tags if t not in stop_words}
    if not informative_tags:
        informative_tags = img_tags

    raw_scores: Dict[str, float] = {}
    for cat, ctags in cat_tags_map.items():
        ctags_inf = ctags - stop_words
        if not ctags_inf:
            continue
        overlap = informative_tags & ctags_inf
        if not overlap:
            continue
        idf_sum = sum(idf.get(t, 1.0) for t in overlap)
        avg_idf = idf_sum / max(len(overlap), 1)
        raw_scores[cat] = idf_sum * min(avg_idf / 2.5, 1.0)
        cat_lower = cat.lower()
        if cat_lower in informative_tags and idf.get(cat_lower, 1.0) >= 1.5:
            raw_scores[cat] += 2.0

    if not raw_scores:
        return {}

    # Global pollution check: if half (or more) of the categories score
    # above zero on this image the registry is too polluted for any
    # signal here to be useful. Skip the signal entirely so the
    # fingerprint/CV signals can win.
    n_active = sum(1 for v in raw_scores.values() if v > 0)
    if n_active >= len(cat_tags_map) * 0.6:
        return {}

    best = max(raw_scores.values()) or 1.0
    return {c: v / best for c, v in raw_scores.items()}


def _combine_signals(profile: dict) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute every per-category signal and combine them.

    Returns:
        combined:  dict cat -> combined score in [0, 1]
        breakdown: dict cat -> per-signal score
    """
    breakdown: Dict[str, Dict[str, float]] = {}
    for cat in CATEGORIES:
        breakdown[cat] = {}

    for name, weight in SIGNAL_WEIGHTS.items():
        if name == "tags":
            sig = _tags_signal(profile)
        elif name == "palette":
            sig = _palette_signal(profile)
        elif name == "style":
            sig = _style_signal(profile)
        elif name == "composition":
            sig = _composition_signal(profile)
        elif name == "quality":
            sig = _quality_signal(profile)
        elif name == "pattern":
            sig = _pattern_signal(profile)
        elif name == "content":
            sig = _content_signal(profile)
        elif name == "prompt":
            sig = _prompt_signal(profile)
        elif name == "clip":
            sig = _clip_signal(profile)
        elif name == "clip_nsfw":
            sig = _clip_nsfw_signal(profile)
        elif name == "theme":
            sig = _theme_signal(profile)
        elif name == "size":
            sig = _size_signal_scores(profile)
        elif name == "fingerprint":
            sig = _fingerprint_signal(profile)
        else:
            continue
        for cat, v in sig.items():
            breakdown.setdefault(cat, {})[name] = float(v) * weight

    # Sum up the contributions
    combined: Dict[str, float] = {}
    for cat, signals in breakdown.items():
        combined[cat] = sum(signals.values())
    return combined, breakdown


# ---------------------------------------------------------------------------
# Pollution detection: warn the user when the registry is contaminated
# ---------------------------------------------------------------------------

_POLLUTION_WARNING_CACHE: Dict[str, bool] = {}


def _check_pollution_warn(categories_dir: Optional[str] = None,
                          force: bool = False) -> bool:
    """Print a one-shot warning when the category registry is polluted.

    Returns True if pollution was detected (and warned about), False
    otherwise. The warning is cached so it appears at most once per
    process invocation (pass force=True to print again).
    """
    cache_key = categories_dir or "<default>"
    if not force and _POLLUTION_WARNING_CACHE.get(cache_key):
        return False
    try:
        from . import categories as _cat_mod
        target = categories_dir or _cat_mod.CATEGORIES_DIR
        if not target:
            return False
        polluted, df = _analyse_pollution_quick(target)
    except Exception:
        return False
    n = len(_cat_mod.CATEGORIES)
    if not polluted or n < 3:
        _POLLUTION_WARNING_CACHE[cache_key] = True
        return False
    top = sorted(polluted, key=lambda t: -df[t])[:6]
    print(
        f"\n[warn] Detected {len(polluted)} tag(s) present in >=50% of your "
        f"{n} categories. This pollutes the multi-signal classifier so the "
        f"wrong category (often Anime) wins by tag-overlap.\n"
        f"  Most polluted: {', '.join(top)}\n"
        f"  Fix: run `python -m wallpaper_analyzer.clean_tags --clean` "
        f"or open Categories > Clean tags in the GUI.\n",
        flush=True,
    )
    _POLLUTION_WARNING_CACHE[cache_key] = True
    return True


def _analyse_pollution_quick(target_dir: str) -> Tuple[Set[str], Dict[str, int]]:
    """Cheap pollution check used for the runtime warning.

    Counts how many `.category.json` files each tag appears in, but
    does not load the full CATEGORIES list (avoiding side effects).
    """
    df: Counter = Counter()
    polluted: Set[str] = set()
    if not os.path.isdir(target_dir):
        return polluted, dict(df)
    files_seen = 0
    for entry in os.listdir(target_dir):
        cfg_path = os.path.join(target_dir, entry, ".category.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        files_seen += 1
        seen: Set[str] = set()
        for t in (cfg.get("tags") or []):
            tl = str(t).strip().lower()
            if tl and tl not in seen:
                seen.add(tl)
                df[tl] += 1
    if files_seen == 0:
        return polluted, dict(df)
    polluted = {t for t, d in df.items() if d >= 0.5 * files_seen}
    return polluted, dict(df)


# ---------------------------------------------------------------------------
# Public classification entry points
# ---------------------------------------------------------------------------

def classify(profile: dict) -> str:
    """Legacy entry point: returns the best category name or the default."""
    first_cat = CATEGORIES[0] if CATEGORIES else "Uncategorized"
    w, h = profile.get("size", (0, 0))
    if h > w and "Phone" in CATEGORIES:
        return "Phone"

    if any(get_category_tags(cat) for cat in CATEGORIES):
        img_tags = derive_tags_from_profile(profile)
        tag_cat = classify_by_tags(img_tags)
        if tag_cat and tag_cat in CATEGORIES:
            return tag_cat

    desc = profile.get("ollama_description", "")
    if desc:
        prompt_cat = classify_by_prompt_text(desc)
        if prompt_cat and prompt_cat in CATEGORIES:
            return prompt_cat

    fb = palette_fallback(profile)
    if fb:
        return fb
    return first_cat


def classify_with_confidence(profile: dict) -> Dict:
    """Return a full recommendation with confidence and per-category scores.

    Output::

        {
          "category":   "Nature",
          "confidence": 0.71,
          "scores":     {"Nature": 0.71, "Anime": 0.12, ...},
          "signals":    {"Nature": {"tags": 0.8, "palette": 0.4, ...}, ...},
          "tags":       { "nature", "landscape", ... },
        }
    """
    # One-shot pollution warning so the user knows when their registry
    # is dragging the classifier towards the wrong category.
    _check_pollution_warn()

    if not CATEGORIES:
        return {"category": None, "confidence": 0.0, "scores": {}, "signals": {}, "tags": set()}

    combined, breakdown = _combine_signals(profile)
    if not combined:
        fb = palette_fallback(profile)
        return {
            "category": fb,
            "confidence": 0.0,
            "scores": {fb: 1.0} if fb else {},
            "signals": {},
            "tags": suggest_tags(profile, max_tags=12),
        }

    # Softmax normalisation (temperature = 0.5 keeps winner sharp)
    norm = {c: max(0.0, s) for c, s in combined.items()}
    z = sum(norm.values()) or 1e-9
    probs = {c: v / z for c, v in norm.items()}

    # Specificity bonus: when one category scores much higher than the
    # next best, boost its probability. This rewards CLEAN wins (where
    # one category is clearly right) over ambiguous ties (where several
    # categories score similarly).
    sorted_probs = sorted(probs.items(), key=lambda kv: -kv[1])
    if len(sorted_probs) >= 2 and sorted_probs[0][1] > 0:
        gap = sorted_probs[0][1] / max(sorted_probs[1][1], 0.01)
        if gap > 2.0:
            # Sharpen the distribution: take a power of 0.6 of each score.
            probs = {c: v ** 0.6 for c, v in probs.items()}
            z = sum(probs.values()) or 1e-9
            probs = {c: v / z for c, v in probs.items()}
        elif gap < 1.2:
            # Ambiguous match: flatten the distribution to avoid false
            # confidence when the image is genuinely hard to classify.
            probs = {c: v ** 1.4 for c, v in probs.items()}
            z = sum(probs.values()) or 1e-9
            probs = {c: v / z for c, v in probs.items()}

    best = max(probs, key=probs.get)
    return {
        "category": best,
        "confidence": round(float(probs[best]), 3),
        "scores": {c: round(p, 3) for c, p in probs.items()},
        "signals": breakdown,
        "tags": suggest_tags(profile, max_tags=12),
    }


def explain_classification(profile: dict, top_n: int = 3) -> List[str]:
    """Return human-readable explanations for the top categories.

    Useful for the GUI to show "Why this category?" beside the result.
    """
    out: List[str] = []
    info = classify_with_confidence(profile)
    ranked = sorted(info["scores"].items(), key=lambda kv: -kv[1])[:top_n]
    for cat, score in ranked:
        signals = info["signals"].get(cat, {})
        triggered = sorted(
            [(name, w * signals.get(name, 0.0))
             for name, w in SIGNAL_WEIGHTS.items()
             if signals.get(name, 0.0) > 0.05],
            key=lambda kv: -kv[1],
        )
        top_signals = ", ".join(f"{n}={v:.2f}" for n, v in triggered[:3])
        out.append(f"{cat} ({score:.0%}): {top_signals}")
    return out
