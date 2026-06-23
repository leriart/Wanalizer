"""Prompt generation from image profiles.

Builds natural-language descriptions of a wallpaper collection that are
useful for:
  * Matching against category prompts (CLIP / Ollama semantic similarity)
  * Showing the user what makes a category unique
  * Generating a default `.category.json` prompt when the user adds a
    new style folder

The generated text deliberately stays concise (2-4 sentences) and uses
vocabulary likely to appear in stock image search engines, so similarity
scores with CLIP-style embeddings stay meaningful.

Two prompt styles are supported:
  * `concise` (default) - single sentence, 8-15 words
  * `detailed`           - 2-3 sentences covering palette, style and content
  * `vivid`              - one evocative sentence suitable for image-gen AIs
"""
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Phrase banks for each signal. Keep vocabulary tight to maximise the chance
# that generated tokens overlap with category prompts.

_PALETTE_PHRASES = {
    "Red":          ("warm red tones", "crimson highlights"),
    "Orange":       ("warm orange tones", "amber glow"),
    "Yellow":       ("warm yellow tones", "golden light"),
    "Green":        ("lush green tones", "verdant foliage"),
    "Teal-Cyan":    ("cool teal hues", "cyan accents"),
    "Blue":         ("cool blue tones", "deep blue palette"),
    "Purple":       ("rich purple tones", "violet shadows"),
    "Pink-Magenta": ("vibrant pink hues", "magenta accents"),
    "Black":        ("deep blacks", "inky shadows"),
    "White":        ("clean whites", "bright highlights"),
    "Gray":         ("neutral grays", "monochrome tones"),
}

_STYLE_PHRASES = [
    ("anime_score",  0.30, "anime illustration", "stylised anime artwork"),
    ("anime_score",  0.50, "vibrant anime art",   "detailed anime illustration"),
    ("cyberpunk_score", 0.30, "cyberpunk cityscape", "neon-lit sci-fi scene"),
    ("space_score", 0.35, "deep space vista", "cosmic galaxy scene"),
    ("nature_score", 0.30, "natural landscape", "outdoor nature scene"),
    ("neon_score",  0.30, "neon-glow aesthetic", "vivid neon colour palette"),
    ("pastel_score", 0.30, "soft pastel palette", "gentle pastel hues"),
    ("vw_score",    0.20, "vaporwave aesthetic", "retro vaporwave style"),
    ("sepia_score", 0.25, "vintage sepia tone", "aged sepia photograph"),
    ("vintage_score", 0.25, "vintage photograph look", "antique film grain"),
]

_CONTENT_PHRASES = [
    # (profile_key, threshold, phrase, alternative)
    ("skin_fraction", 0.15, "featuring a person", "with a human subject"),
    ("skin_fraction", 0.30, "portrait of a person", "figure-focused composition"),
    ("subject_area_ratio", 0.6, "with a central subject", "subject-focused"),
    ("subject_area_ratio", 0.2, "minimalist composition", "minimal subject"),
    ("centeredness", 0.75, "centred composition", "symmetric framing"),
    ("overall_symmetry", 0.7, "bilateral symmetry", "mirror composition"),
    ("periodicity_score", 0.6, "repeating pattern", "tiled motif"),
    ("is_periodic", 0.5, "repeating pattern", "tiled motif"),
    ("is_tiled", 0.5, "tiled pattern", "seamless texture"),
    ("unique_colors", 24, "limited colour palette", "pixel-art aesthetic"),
    ("texture_entropy", 4.7, "highly detailed", "rich texture"),
    ("texture_entropy", 2.8, "minimal texture", "clean minimalism"),
    ("aesthetic", 0.7, "high aesthetic quality", "polished visual"),
    ("sharpness", 400, "sharp high-resolution detail", "crisp imagery"),
]

_MOOD_PHRASES = {
    "warm":     "warm atmosphere",
    "cool":     "cool atmosphere",
    "dark":     "moody dark atmosphere",
    "light":    "bright airy atmosphere",
    "sunset":   "sunset warmth",
    "night":    "nighttime calm",
    "vibrant":  "vibrant energy",
    "pastel":   "soft pastel calm",
    "monochrome": "minimalist monochrome feel",
    "minimalist": "clean minimalist aesthetic",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _palette_phrase(weights: Dict[str, float]) -> Tuple[str, str]:
    """Return (short, long) palette phrase from the colour weights."""
    if not weights:
        return ("varied colour palette", "rich varied colours")
    total = sum(weights.values()) or 1
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    dominant = ranked[0][0]
    secondary = ranked[1][0] if len(ranked) > 1 and ranked[1][1] / total > 0.12 else None

    short_parts = []
    long_parts = []
    s_short, s_long = _PALETTE_PHRASES.get(dominant, ("varied palette", "mixed colours"))
    short_parts.append(s_short)
    long_parts.append(s_long)
    if secondary and secondary != dominant:
        ss_short, ss_long = _PALETTE_PHRASES.get(secondary, ("", ""))
        if ss_short:
            short_parts.append(ss_short)
            long_parts.append(ss_long)
    return ", ".join(short_parts), "with " + " and ".join(long_parts)


def _style_phrase(profile: dict) -> Optional[Tuple[str, str]]:
    """Return (short, long) style phrase or None if no style signal is strong."""
    for key, threshold, short, long in _STYLE_PHRASES:
        if profile.get(key, 0.0) >= threshold:
            return short, long
    return None


def _content_phrase(profile: dict) -> List[str]:
    """Return a list of short content phrases triggered by the profile."""
    out = []
    for key, threshold, short, long in _CONTENT_PHRASES:
        v = profile.get(key, 0)
        if v is None:
            continue
        # Some keys are categorical (is_periodic, is_tiled) - threshold is just >threshold
        if v >= threshold:
            out.append(short)
    return out


def _mood_phrase(profile: dict) -> Optional[str]:
    """Detect dominant mood from palette + style signals."""
    weights = profile.get("weights") or {}
    if not weights:
        return None
    total = sum(weights.values()) or 1
    pcts = {c: v / total for c, v in weights.items()}

    # Warm vs cool
    warm = sum(pcts.get(c, 0) for c in ("Orange", "Yellow", "Red", "Pink-Magenta"))
    cool = sum(pcts.get(c, 0) for c in ("Blue", "Teal-Cyan", "Purple"))
    light = sum(pcts.get(c, 0) for c in ("White", "Yellow"))
    dark = sum(pcts.get(c, 0) for c in ("Black", "Blue", "Purple"))

    if warm > 0.40:
        if pcts.get("Orange", 0) > 0.18 or pcts.get("Pink-Magenta", 0) > 0.18:
            return _MOOD_PHRASES["sunset"]
        return _MOOD_PHRASES["warm"]
    if cool > 0.45:
        if dark > 0.40 or profile.get("space_score", 0) > 0.30:
            return _MOOD_PHRASES["night"]
        return _MOOD_PHRASES["cool"]
    if light > 0.55:
        return _MOOD_PHRASES["light"]
    if dark > 0.45:
        return _MOOD_PHRASES["dark"]

    if profile.get("vintage_score", 0) > 0.25 or profile.get("sepia_score", 0) > 0.25:
        return _MOOD_PHRASES["monochrome"]

    if profile.get("pastel_score", 0) > 0.30:
        return _MOOD_PHRASES["pastel"]
    if profile.get("neon_score", 0) > 0.30:
        return _MOOD_PHRASES["vibrant"]
    if profile.get("unique_colors", 256) < 32 and profile.get("anime_score", 0) > 0.25:
        return _MOOD_PHRASES["pastel"]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_prompt(profile: dict, style: str = "concise") -> str:
    """Generate a natural-language description of a single wallpaper profile.

    `style`:
      * "concise" - single short sentence (default)
      * "detailed" - 2-3 sentences, palette + style + content
      * "vivid" - one evocative sentence suitable for image-gen AI prompts
    """
    palette_short, palette_long = _palette_phrase(profile.get("weights") or {})
    style_pair = _style_phrase(profile)
    content_phrases = _content_phrase(profile)
    mood = _mood_phrase(profile)

    # W, H
    w, h = profile.get("size", (0, 0))

    if style == "concise":
        parts = []
        if style_pair:
            parts.append(style_pair[0])
        else:
            parts.append(palette_short)
        if content_phrases:
            parts.append(content_phrases[0])
        if mood:
            parts.append(mood)
        if w and h:
            parts.append(_orientation_phrase(w, h))
        return ", ".join(parts) + "."

    if style == "detailed":
        sentences = []
        s_palette, _ = _palette_phrase(profile.get("weights") or {})
        first = f"A wallpaper {s_palette}"
        if style_pair:
            first += f" in a {style_pair[1]} style"
        first += "."
        sentences.append(first)

        if content_phrases:
            sentences.append(
                "The composition is " + ", ".join(content_phrases[:2]) + "."
            )
        if mood:
            sentences.append(f"It evokes a {mood}.")
        return " ".join(sentences)

    if style == "vivid":
        # Prompt-engineering friendly: comma-separated, evocative.
        bits = []
        bits.append(palette_short)
        if style_pair:
            bits.append(style_pair[0])
        if mood:
            bits.append(mood)
        bits.extend(content_phrases[:3])
        if w and h:
            bits.append(_orientation_phrase(w, h))
        return ", ".join(bits) + ", high quality, detailed"

    raise ValueError(f"Unknown style: {style!r}")


def _orientation_phrase(w: int, h: int) -> str:
    if h > w:
        if h / max(w, 1) > 1.8:
            return "vertical phone wallpaper"
        return "vertical wallpaper"
    if w / max(h, 1) > 2.0:
        return "wide panoramic wallpaper"
    if w * h >= 3840 * 2160:
        return "4K resolution"
    return "high resolution"


# ---------------------------------------------------------------------------
# Aggregate prompt for a whole folder of wallpapers
# ---------------------------------------------------------------------------

def generate_category_prompt(
    profiles: List[dict],
    style: str = "concise",
    category_name: str = "",
) -> str:
    """Build a prompt that describes an entire category by averaging the
    most-likely signals across all `profiles`.

    The result is meant to be stored in `category.json` so it can be matched
    against single-image profiles.
    """
    if not profiles:
        return ""

    # Aggregate scores
    score_keys = (
        "anime_score", "cyberpunk_score", "space_score", "nature_score",
        "neon_score", "pastel_score", "vw_score", "sepia_score",
        "vintage_score", "sharpness", "aesthetic", "skin_fraction",
        "subject_area_ratio", "centeredness", "overall_symmetry",
        "texture_entropy", "periodicity_score", "unique_colors",
    )
    agg = {k: 0.0 for k in score_keys}
    for p in profiles:
        for k in score_keys:
            agg[k] += float(p.get(k, 0) or 0)
    n = max(len(profiles), 1)
    for k in score_keys:
        agg[k] /= n

    # Aggregate palette
    palette_agg: Counter = Counter()
    for p in profiles:
        for color, weight in (p.get("weights") or {}).items():
            palette_agg[color] += weight
    agg_palette = dict(palette_agg)
    total = sum(agg_palette.values()) or 1
    agg["weights"] = {c: v / total for c, v in agg_palette.items()}

    # Aggregate size
    widths = [p.get("size", (0, 0))[0] for p in profiles if p.get("size")]
    heights = [p.get("size", (0, 0))[1] for p in profiles if p.get("size")]
    if widths:
        agg["size"] = (
            sum(widths) // len(widths),
            sum(heights) // len(heights),
        )

    prompt = generate_prompt(agg, style=style)

    if category_name:
        # Anchor with the category name for stronger CLIP match.
        prompt = f"{category_name.replace('_', ' ')} style: {prompt}"
    return prompt


def suggest_palette_weights(profile: dict) -> Dict[str, float]:
    """Convert profile colour weights into palette weights suitable for
    `.category.json` 'palette_weights'. The result emphasises the
    *strongest* signals so subsequent matching rewards dominant tones."""
    weights = profile.get("weights") or {}
    if not weights:
        return {}
    total = sum(weights.values()) or 1
    out = {}
    for color, v in weights.items():
        pct = v / total
        # Square-root amplification: medium-weight colours get more
        # influence than under linear weighting.
        out[color] = round(pct ** 0.5, 4)
    return out
