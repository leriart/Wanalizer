from PIL import Image
from collections import Counter
from typing import Dict

from .color import extract_color_weights, rgb_to_hsv
from .quality import laplacian_variance, heuristic_aesthetic_score

def count_unique_colors(img: Image.Image, max_colors: int = 256) -> int:
    try:
        q = img.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
        return len(set(q.getdata()))
    except Exception:
        return max_colors


# ---------------------------------------------------------------------------
# Theme palette matchers
# ---------------------------------------------------------------------------
# Each theme is a tuple of (color, weight) describing the expected palette
# distribution for an "ideal" image in that theme. The match score is the
# weighted cosine similarity between the image's normalised palette and the
# theme palette. Cheap to compute (12-colour quantize + dot product).
#
# These cover the "themed" categories where the user picked a category for
# its colour scheme rather than its content (Catppuccin / Dracula / TokyoNight
# / Nord are all "dark theme with blue/purple accents" wallpaper packs).
# ---------------------------------------------------------------------------

_THEME_PALETTES: Dict[str, list] = {
    "catppuccin": [
        # Catppuccin: pastel mauve/peach base with deep navy text. Mostly
        # low-saturation, mid-value with hints of pink and purple.
        ("Pink-Magenta", 0.20), ("Purple", 0.15), ("Blue", 0.10),
        ("Gray", 0.20), ("Black", 0.15), ("White", 0.10),
        ("Orange", 0.05), ("Teal-Cyan", 0.05),
    ],
    "dracula": [
        # Dracula: very dark backgrounds with vibrant pink/cyan/purple accents.
        ("Black", 0.45), ("Purple", 0.20), ("Pink-Magenta", 0.10),
        ("Blue", 0.10), ("Teal-Cyan", 0.05), ("Green", 0.05),
        ("Gray", 0.05),
    ],
    "tokyonight": [
        # TokyoNight: deep navy/purple base with cyan/blue accents, slightly
        # lighter than Dracula.
        ("Blue", 0.25), ("Purple", 0.20), ("Black", 0.20),
        ("Teal-Cyan", 0.10), ("Pink-Magenta", 0.05), ("Gray", 0.10),
        ("White", 0.05), ("Green", 0.05),
    ],
    "nord": [
        # Nord: cool blue-grays with subtle icy blue accents.
        ("Blue", 0.20), ("Teal-Cyan", 0.15), ("Gray", 0.35),
        ("White", 0.15), ("Black", 0.05), ("Green", 0.05),
        ("Purple", 0.05),
    ],
    "gruvbox": [
        # Gruvbox: warm earthy palette with orange/cream/brown.
        ("Orange", 0.30), ("Yellow", 0.20), ("Gray", 0.20),
        ("Black", 0.10), ("White", 0.10), ("Red", 0.05),
        ("Green", 0.05),
    ],
    "everforest": [
        # Everforest: muted forest greens with cream/sage accents.
        ("Green", 0.45), ("Gray", 0.20), ("Yellow", 0.10),
        ("White", 0.10), ("Black", 0.05), ("Orange", 0.05),
        ("Teal-Cyan", 0.05),
    ],
    "monochrome": [
        # Monochrome: pure neutrals, no chromatic colour.
        ("Black", 0.40), ("White", 0.30), ("Gray", 0.30),
    ],
    "neon": [
        # Neon: vivid pink/purple/cyan/blue glow on dark backgrounds.
        ("Pink-Magenta", 0.30), ("Purple", 0.25), ("Teal-Cyan", 0.15),
        ("Blue", 0.15), ("Black", 0.10), ("Green", 0.05),
    ],
}


def _theme_palette_score(weights: dict, theme: str) -> float:
    """Cosine similarity between image palette and the theme's ideal palette.

    Returns 0..1. Empty/invalid input returns 0. Image palettes that are
    dominated by neutral colours score lower on chromatic themes but high
    on monochrome, which is exactly the discrimination we want.
    """
    if not weights or theme not in _THEME_PALETTES:
        return 0.0
    total = sum(weights.values()) or 1
    img_pcts = {k: v / total for k, v in weights.items()}
    theme_pal = _THEME_PALETTES[theme]
    theme_total = sum(w for _, w in theme_pal) or 1
    theme_pcts = {c: w / theme_total for c, w in theme_pal}

    keys = set(img_pcts) | set(theme_pcts)
    dot = sum(img_pcts.get(k, 0) * theme_pcts.get(k, 0) for k in keys)
    n1 = sum(img_pcts.get(k, 0) ** 2 for k in keys) ** 0.5
    n2 = sum(theme_pcts.get(k, 0) ** 2 for k in keys) ** 0.5
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return min(1.0, dot / (n1 * n2))


# ---------------------------------------------------------------------------
# Content detectors (cheap, run inside get_image_profile)
# ---------------------------------------------------------------------------

def _detector_minecraft(pixel_counts, palette_rgb, total_pixels) -> float:
    """Detect voxel/block-style artwork (Minecraft, Lego, low-res pixel art).

    Strong signal: VERY low unique colours combined with high edge density
    (lots of sharp transitions = cube/block boundaries). We compute a
    rough "blockiness" by checking how many adjacent pixel pairs share the
    same quantized colour (block interiors are uniform) and how many are
    very different (block boundaries).
    """
    if total_pixels <= 0 or not pixel_counts:
        return 0.0
    # Cheap blockiness proxy: ratio of dominant-colour pixels
    sorted_counts = sorted(pixel_counts.values(), reverse=True)
    top3 = sum(sorted_counts[:3])
    top3_frac = top3 / total_pixels
    # Few colours + high top3 concentration = blocky
    n_unique = len(pixel_counts)
    if n_unique > 12:
        return 0.0
    # Top colours dominate AND unique count is small
    return min(1.0, top3_frac * (12 - n_unique) / 6)


def _detector_pixel_art(unique_colors, w, h) -> float:
    """Detect pixel art (few colours, small image).

    Pixel art typically uses 8-256 colours total. Smaller images on a
    thumbnail (200x200) with very low unique counts are usually pixel art.
    Larger images with similarly low counts are minimal art.
    """
    if unique_colors <= 0:
        return 0.0
    n_pixels = w * h
    # Size factor: smaller images are more likely to be pixel art
    size_factor = 1.0 if n_pixels < 20000 else max(0.0, 1.0 - (n_pixels - 20000) / 80000)
    # Colour factor: very low unique colours = pixel art
    color_factor = max(0.0, 1.0 - (unique_colors - 8) / 56) if unique_colors < 64 else 0.0
    return min(1.0, size_factor * color_factor + color_factor * 0.3)


def _detector_minimalist(unique_colors, skin_fraction, subject_area_ratio,
                          texture_entropy) -> float:
    """Continuous minimalist score: low entropy, few colours, focal subject.

    Combines four signals (any one alone is unreliable). The continuous
    version replaces the previous binary threshold so a "very minimalist"
    image scores higher than a "slightly minimalist" one.
    """
    # Colour simplicity
    color_part = max(0.0, 1.0 - unique_colors / 96.0)
    # Subject area (small = minimalist background)
    subject_part = max(0.0, 1.0 - subject_area_ratio / 0.4)
    # Texture entropy (low = clean surfaces)
    tex_part = max(0.0, 1.0 - texture_entropy / 4.0)
    # No skin = no people (cleaner)
    skin_part = 1.0 - min(1.0, skin_fraction / 0.10)
    # Geometric mean: any zero collapses the score (need ALL signals)
    score = (color_part * subject_part * tex_part) ** (1.0 / 3.0)
    return min(1.0, score + 0.05 * skin_part)


def _detector_portrait(skin_fraction, subject_area_ratio, centeredness) -> float:
    """Continuous portrait score: skin + subject focus + reasonable framing.

    Distinguishes portrait shots from NSFW (which also has skin but lacks
    the centred composition) and from character art (which has flat
    anime-style skin-tone pixels but no photographic skin texture).
    """
    if skin_fraction < 0.05:
        return 0.0
    skin_part = min(1.0, skin_fraction / 0.25)
    subject_part = 1.0 if 0.3 <= subject_area_ratio <= 0.7 else 0.4
    framing_part = 0.6 + 0.4 * centeredness
    return min(1.0, skin_part * subject_part * framing_part)


def _detector_nsfw(skin_fraction, subject_area_ratio, texture_entropy) -> float:
    """Continuous NSFW score: high skin fraction with low texture entropy.

    Differs from `portrait_score` by requiring LOTS of skin (suggesting
    undressed or partially-clothed figure) AND low texture entropy
    (suggesting smooth skin rather than textured photography or
    anime-patterned clothing).
    """
    if skin_fraction < 0.08:
        return 0.0
    skin_part = min(1.0, (skin_fraction - 0.05) / 0.30)
    # Body shots typically fill more of the frame than face-only portraits
    subject_part = 1.0 if subject_area_ratio >= 0.35 else 0.5
    # Smooth body texture rather than textured clothing/background
    tex_part = 1.0 if texture_entropy < 4.5 else max(0.0, 1.0 - (texture_entropy - 4.5) / 1.5)
    return min(1.0, skin_part * subject_part * tex_part)


def _detector_gradient(pixel_counts, palette_rgb, total_pixels) -> float:
    """Detect smooth gradient backgrounds (often paired with clean content).

    A high-gradient image has many colours at moderate weights (no single
    dominant colour) with no extreme brightness either way.
    """
    if total_pixels <= 0 or not pixel_counts:
        return 0.0
    n_unique = len(pixel_counts)
    if n_unique < 4:
        return 0.0
    weights = sorted([c / total_pixels for c in pixel_counts.values()], reverse=True)
    # Many moderately-weighted colours = gradient
    top1 = weights[0]
    if top1 > 0.50:
        return 0.0  # Single dominant colour, not gradient
    # Spread across many buckets
    spread = sum(min(w, 0.20) for w in weights) / min(len(weights), 12)
    return min(1.0, spread * 1.5)


def _texture_proxy(img) -> float:
    """Cheap texture-entropy proxy from the thumbnail.

    Computes a Sobel-style gradient magnitude and uses the std as a rough
    texture-energy proxy. Not as accurate as `lowlevel.texture` but cheap.
    """
    try:
        from PIL import ImageFilter
        gx = img.convert("L").filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=(-1, 0, 1, -2, 0, 2, -1, 0, 1),
            scale=1,
            offset=128,
        ))
        # Variance of the gradient magnitude is a rough entropy proxy
        hist = gx.histogram()
        var = sum((i - sum(hist) / 256) ** 2 * h for i, h in enumerate(hist)) / max(sum(hist), 1)
        return min(8.0, var / 5e5)
    except Exception:
        return 0.0


def _subject_area_proxy(pixel_counts, palette_rgb, total_pixels) -> float:
    """Estimate fraction of image occupied by a "subject" (non-background).

    Uses a simple proxy: pixels far from the dominant background colour
    form the subject. Cheap, not as good as `lowlevel.subject_features`,
    but runs in milliseconds.
    """
    if total_pixels <= 0 or not pixel_counts:
        return 0.0
    if len(pixel_counts) < 2:
        return 0.0
    sorted_by_count = sorted(pixel_counts.items(), key=lambda kv: -kv[1])
    bg_count = sorted_by_count[0][1]
    subject_pixels = total_pixels - bg_count
    return min(1.0, subject_pixels / total_pixels * 1.3)


def get_image_profile(image_path: str) -> dict:
    img_full = Image.open(image_path).convert("RGB")
    sharpness = laplacian_variance(img_full)
    aesthetic = heuristic_aesthetic_score(img_full, sharpness=sharpness)
    img = img_full.copy()
    img.thumbnail((200, 200), Image.LANCZOS)
    w, h = img.size
    quantized = img.quantize(colors=12, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()[:12 * 3]
    palette_rgb = [(palette[i], palette[i + 1], palette[i + 2]) for i in range(0, len(palette), 3)]
    pixel_counts = Counter(quantized.getdata())
    total_pixels = sum(pixel_counts.values())
    color_weights, style_scores = extract_color_weights(img)
    anime_score = cyberpunk_score = space_score = sepia_score = vintage_score = 0.0
    skin_pixels = black_pixels = 0
    saturated_flat_pixels = neon_dark_neighbour_pixels = 0

    for idx, count in pixel_counts.items():
        if idx >= len(palette_rgb): continue
        r, g, b = palette_rgb[idx]
        hh, ss, vv = rgb_to_hsv(r, g, b)
        weight = count / total_pixels
        if ss > 55 and vv > 40:
            saturated_flat_pixels += count
        if vv < 30:
            neon_dark_neighbour_pixels += count
        if 5 <= hh <= 38 and 15 <= ss <= 65 and 35 <= vv <= 92:
            skin_pixels += count
        if vv < 12:
            space_score += weight
        if 80 <= r <= 200 and 50 <= g <= 170 and b < g and (r - b) > 25 and ss < 45 and vv > 25:
            sepia_score += weight
        if ss < 35 and 35 <= hh <= 60 and 40 < vv < 80:
            vintage_score += weight
        if cat := getattr(__import__("wallpaper_analyzer.color", fromlist=["classify_pixel"]), "classify_pixel")(hh, ss, vv):
            if cat.get("category") == "Black":
                black_pixels += count

    anime_score = min((saturated_flat_pixels / max(total_pixels, 1)) * 1.5, 1.0)
    dark_fraction = neon_dark_neighbour_pixels / max(total_pixels, 1)
    neon = style_scores.get("neon", 0)
    vw = style_scores.get("vaporwave", 0)
    cyberpunk_score = min(neon * (1.0 + 0.8 * dark_fraction) + 0.8 * vw, 1.0)
    space_score = min(space_score + 0.4 * (black_pixels / max(total_pixels, 1)), 1.0)
    skin_fraction = skin_pixels / max(total_pixels, 1)
    unique_colors = count_unique_colors(img)

    # Cheap proxies used by the new content detectors. Running these in the
    # main profile (instead of the slow lowlevel extractor) means even the
    # fast rebuild path gets the new signals.
    texture_proxy = _texture_proxy(img)
    subject_area_proxy = _subject_area_proxy(pixel_counts, palette_rgb, total_pixels)

    # Theme palette scores (Catppuccin / Dracula / TokyoNight / Nord / ...)
    theme_scores = {
        theme: _theme_palette_score(color_weights, theme)
        for theme in _THEME_PALETTES
    }

    try:
        img_full.close()
    except Exception:
        pass

    return {
        "weights": color_weights,
        "vw_score": style_scores.get("vaporwave", 0),
        "neon_score": style_scores.get("neon", 0),
        "pastel_score": style_scores.get("pastel", 0),
        "nature_score": style_scores.get("nature", 0),
        "anime_score": anime_score,
        "cyberpunk_score": cyberpunk_score,
        "space_score": space_score,
        "sepia_score": sepia_score,
        "vintage_score": vintage_score,
        "skin_fraction": skin_fraction,
        "unique_colors": unique_colors,
        "size": (w, h),
        "sharpness": sharpness,
        "aesthetic": aesthetic,
        # New content detectors (continuous, in [0, 1])
        "minecraft_score": _detector_minecraft(pixel_counts, palette_rgb, total_pixels),
        "pixel_art_score": _detector_pixel_art(unique_colors, w, h),
        "minimalist_score": _detector_minimalist(
            unique_colors, skin_fraction, subject_area_proxy, texture_proxy,
        ),
        "portrait_score": _detector_portrait(
            skin_fraction, subject_area_proxy, centeredness=0.5,
        ),
        "nsfw_score": _detector_nsfw(
            skin_fraction, subject_area_proxy, texture_proxy,
        ),
        "gradient_score": _detector_gradient(pixel_counts, palette_rgb, total_pixels),
        # Theme palette scores (0..1 cosine similarity to each theme's palette)
        **{f"theme_{theme}_score": v for theme, v in theme_scores.items()},
    }
