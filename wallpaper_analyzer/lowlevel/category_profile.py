"""
Category heuristic profile: builds a CV fingerprint from sample images.

Each category gets a statistical profile computed from its actual images:
color weights, edge density, texture entropy, style scores, composition,
symmetry, pattern features, etc. These are stored in .category.json as
'heuristic_pattern' and used during classification to score new images
against the expected pattern of each category.

The fingerprint combines two complementary views of the category:
  * **Numeric fingerprint** - per-feature mean/std/min/max for everything
    in `FEATURE_KEYS` (used by `score_against_pattern` via z-score distance).
  * **Palette fingerprint** - average dominant-colour distribution over the
    sample images, exposed both as a normalised distribution and as
    ready-to-use `palette_weights` for `.category.json`.

The scoring (`score_against_pattern`) is *discriminative*: it cross-references
the category's feature stats against `compute_global_stats` (the pooled
mean/std across every category) and gives more weight to features that
actually DISTINGUISH this category from the rest. A Cyberpunk category with
`cyberpunk_score` mean=0.02 gets that feature weighted much more heavily
than `hsv_V_mean` which is generic across all categories. Combined with
Bayesian shrinkage on the std (small samples pool toward the global std),
the same scoring function works whether the category has 30 sample images
or 3.

A "minimal AI" `MiniClassifier` instance is also fitted automatically so the
GUI can later ask "given a new image, which category?" with full per-category
probability scores.
"""
import math
import os
import json
import statistics
from collections import Counter
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .. import categories as c
from .. import formats as f
from ..profile import get_image_profile
from ..lowlevel.edges import edge_features
from ..lowlevel.texture import texture_features
from ..lowlevel.silhouettes import silhouette_features
from ..lowlevel.contours import contour_features
from ..lowlevel.fourier import frequency_features
from ..lowlevel.hog import hog_features
from ..lowlevel.features import feature_features
from ..lowlevel.color_advanced import color_features
from ..lowlevel.composition import composition_features
from ..lowlevel.quality_advanced import quality_features
from ..lowlevel.subject import subject_features
from ..lowlevel.pattern import pattern_features
from ..lowlevel.symmetry_advanced import symmetry_features

# Features we aggregate (float values only)
FEATURE_KEYS = [
    # Color
    "colorfulness",
    "hsv_S_mean", "hsv_S_std", "hsv_V_mean", "hsv_V_std",
    "lab_warm_bias", "lab_chroma_mean",
    "sat_low_fraction", "sat_high_fraction", "sat_variance", "sat_bimodal",
    # Edge
    "edge_density", "mean_edge_magnitude",
    # Texture
    "texture_entropy", "texture_energy", "texture_contrast",
    "lbp_entropy", "lbp_variance",
    # Silhouette
    "silhouette_ratio", "silhouette_coverage",
    # Contour
    "contour_symmetry", "contour_complexity",
    # HOG
    "hog_entropy", "hog_energy",
    # Frequency
    "high_low_ratio", "radial_std",
    # Style
    "anime_score", "cyberpunk_score", "space_score", "nature_score",
    "neon_score", "pastel_score", "vw_score", "sepia_score", "vintage_score",
    # Quality
    "tenengrad", "sharpness", "aesthetic", "noise_estimate",
    "bris_quality_score", "dct_low_energy", "dct_high_energy",
    # Composition
    "centeredness", "thirds_score", "diagonal_strength",
    "depth_score", "depth_gradient",
    "saliency_concentration",
    # Subject
    "subject_area_ratio", "fg_bg_contrast",
    # Pattern
    "periodicity_score", "tile_similarity", "texture_complexity",
    "gradient_complexity", "local_variance", "entropy",
    # Symmetry
    "h_symmetry", "v_symmetry", "point_symmetry",
    "main_diagonal_symmetry", "anti_diagonal_symmetry",
    "overall_symmetry",
    # General
    # Content detectors (cheap, run from get_image_profile)
    "minecraft_score", "pixel_art_score", "minimalist_score",
    "portrait_score", "nsfw_score", "gradient_score",
    # Theme palette scores (Catppuccin / Dracula / TokyoNight / Nord / ...)
    "theme_catppuccin_score", "theme_dracula_score",
    "theme_tokyonight_score", "theme_nord_score",
    "theme_gruvbox_score", "theme_everforest_score",
    "theme_monochrome_score", "theme_neon_score",
    # General
    "skin_fraction", "unique_colors",
]

# Features that also aggregate per-color weights
COLOR_WEIGHTS_KEY = "color_weights_distribution"

# Canonical colour list used to aggregate palette weights.
COLOR_KEYS = [
    "Black", "White", "Gray", "Red", "Orange", "Yellow",
    "Green", "Teal-Cyan", "Blue", "Purple", "Pink-Magenta",
]


def _extract_feature_vector(img_path: str) -> Dict:
    """Extract all CV features from a single image as a flat dict of floats.

    Returns a dict that also contains the colour weights (under
    `weights`) and original size under `size`, in addition to the floats
    in `FEATURE_KEYS`. Callers can drop non-numeric keys before passing
    the vector into the aggregator.
    """
    return _extract_feature_vector_full(img_path)


def _extract_feature_vector_full(img_path: str) -> Dict:
    """Extract the FULL set of CV features (slow but rich).

    Includes every per-pixel, frequency-domain, and structural metric.
    Takes ~1-2 seconds per image on a typical desktop. Use this when
    you have a small library or need the most discriminative fingerprint.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return {}

    profile = {}
    try:
        profile = get_image_profile(img_path)
        profile.update(edge_features(img))
        profile.update(texture_features(img))
        profile.update(silhouette_features(img))
        profile.update(contour_features(img))
        profile.update(frequency_features(img))
        profile.update(hog_features(img))
        profile.update(feature_features(img))
        profile.update(color_features(img))
        profile.update(composition_features(img))
        profile.update(quality_features(img))
        profile.update(subject_features(img))
        profile.update(pattern_features(img))
        profile.update(symmetry_features(img))
    except Exception:
        # Corrupt image / unsupported codec / etc. - return whatever we
        # got so far (may be empty if PIL failed before get_image_profile).
        pass
    finally:
        try:
            img.close()
        except Exception:
            pass

    return profile


def _extract_feature_vector_fast(img_path: str) -> Dict:
    """Extract just the CHEAP, high-signal features (fast, ~250ms/image).

    Uses the lightweight `get_image_profile`. This includes the entire set
    of style scores (anime, cyberpunk, space, neon, pastel, vaporwave,
    sepia, vintage, nature) which are exactly the features that drive
    discriminative scoring between the user's curated categories. The
    fingerprint is less rich than the full extractor (no edge density, no
    texture entropy, no composition metrics) but the rebuild finishes in
    ~1-2 minutes instead of 15+.

    Features included: weights + all style scores + skin_fraction +
    unique_colors + sharpness + aesthetic + size.
    """
    try:
        return get_image_profile(img_path)
    except Exception:
        return {}


def _extract_numeric_vector(profile: Dict) -> Dict:
    feats = {}
    for key in FEATURE_KEYS:
        val = profile.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            feats[key] = float(val)
    return feats


def _aggregate_feature_vectors(vectors: List[Dict]) -> Dict:
    """Compute mean/std/min/max for each numeric feature."""
    if not vectors:
        return {}

    agg = {}
    for key in FEATURE_KEYS:
        values = [v[key] for v in vectors if key in v and isinstance(v[key], (int, float))]
        if len(values) >= 2:
            agg[key] = {
                "mean": round(statistics.mean(values), 6),
                "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "samples": len(values),
            }
        elif len(values) == 1:
            v = values[0]
            agg[key] = {
                "mean": round(v, 6),
                "std": 0.0,
                "min": round(v, 6),
                "max": round(v, 6),
                "samples": 1,
            }
    return agg
def _aggregate_color_weights(weight_vectors: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate the per-image palette distributions into one.

    Returns a normalised dict {color: mean_fraction}. If no samples are
    provided the result is empty.
    """
    if not weight_vectors:
        return {}
    sums: Counter = Counter()
    for w in weight_vectors:
        if not w:
            continue
        for color, weight in w.items():
            if color in COLOR_KEYS:
                sums[color] += float(weight)
    n = sum(sums.values())
    if n <= 0:
        return {}
    return {color: round(weight / n, 6) for color, weight in sums.items()}


def _smart_sample(images: List[str], max_samples: int) -> List[str]:
    """Pick a representative subset that covers the distribution extremes.

    Naive uniform sampling (every Nth image) is biased by sort order
    (alphabetical) and misses the outliers. This sampler instead picks:

      * the smallest and largest files by size (different content, different
        palette complexity - the two extremes of the category)
      * N-2 evenly-spaced picks from the middle by file size

    Result: the fingerprint captures the category's full range instead of
    a single alphabetical slice.
    """
    if len(images) <= max_samples:
        return list(images)
    try:
        by_size = sorted(images, key=lambda p: os.path.getsize(p))
    except OSError:
        by_size = list(images)
    if len(by_size) <= max_samples:
        return list(by_size)

    picks: List[str] = []
    seen = set()
    for path in (by_size[0], by_size[-1]):
        if path not in seen:
            picks.append(path)
            seen.add(path)

    middle = by_size[1:-1]
    if middle and max_samples > 2:
        n_extra = max_samples - len(picks)
        step = len(middle) / max(n_extra, 1)
        for i in range(n_extra):
            idx = int(i * step)
            if idx >= len(middle):
                idx = len(middle) - 1
            cand = middle[idx]
            if cand not in seen:
                picks.append(cand)
                seen.add(cand)
                if len(picks) >= max_samples:
                    break
    return picks[:max_samples]


def _build_buckets(images: List[str], n_buckets: int = 4) -> List[List[str]]:
    """Split `images` into `n_buckets` buckets by file size.

    Used for stratified sampling: when a category is dominated by a single
    "kind" of image (e.g. all tiny GIFs and a handful of large PNGs),
    bucketing lets us sample one from each bucket instead of one cluster.
    """
    if not images:
        return []
    try:
        sorted_imgs = sorted(images, key=lambda p: os.path.getsize(p))
    except OSError:
        return [list(images)]
    if len(sorted_imgs) <= n_buckets:
        return [[im] for im in sorted_imgs]
    buckets: List[List[str]] = [[] for _ in range(n_buckets)]
    for i, im in enumerate(sorted_imgs):
        b = min(int(i / len(sorted_imgs) * n_buckets), n_buckets - 1)
        buckets[b].append(im)
    return [b for b in buckets if b]


def _stratified_sample(images: List[str], max_samples: int) -> List[str]:
    """Sample N images by taking one (or two) from each size-bucket.

    Falls back to `_smart_sample` if bucketing doesn't help (very small
    libraries).
    """
    if len(images) <= max_samples:
        return list(images)
    n_buckets = min(max_samples, 6)
    buckets = _build_buckets(images, n_buckets=n_buckets)
    if not buckets or len(buckets) < 2:
        return _smart_sample(images, max_samples)
    picks: List[str] = []
    seen = set()
    # Round-robin: take first from each bucket, then second, etc.
    i = 0
    while len(picks) < max_samples:
        any_added = False
        for b in buckets:
            if i < len(b) and b[i] not in seen:
                picks.append(b[i])
                seen.add(b[i])
                any_added = True
                if len(picks) >= max_samples:
                    break
        if not any_added:
            break
        i += 1
    return picks[:max_samples]


def build_category_profile(
    category_dir: str,
    max_samples: int = 20,
    progress_callback=None,
    category_name: Optional[str] = None,
    sampling: str = "stratified",
    fast: bool = False,
) -> Optional[Dict]:
    """Build a heuristic profile from sample images in the category.

    Returns a dict suitable for .category.json's 'heuristic_pattern' field
    or None if no valid images found.

    Sampling strategies (controlled by `sampling`):
      * "stratified" - one (or two) images from each size-bucket. Best for
        categories that mix tiny icons with high-res wallpapers.
      * "smart"      - smallest + largest + evenly-spaced by size. Good
        general-purpose default.
      * "uniform"    - every Nth image alphabetically (legacy behaviour,
        kept for reproducibility).

    If `fast=True`, uses the cheap feature extractor (~250ms/image instead
    of ~5s). The resulting pattern is less rich but the rebuild finishes
    in seconds, which is the right trade-off when you're rebuilding every
    category at once.

    Video files (.mp4, .webm, ...) are filtered out upfront since the PIL-
    based feature extractors can't decode them. The original .category.json
    patterns included video frames via ffmpeg; if you want that, call
    `build_category_profile(include_videos=True)` explicitly.
    """
    if not os.path.isdir(category_dir):
        return None

    # Only image extensions that PIL can read directly. Video formats are
    # filtered out because PIL raises UnidentifiedImageError on them and
    # the fingerprint signal they would contribute (via a ffmpeg frame
    # extract) is essentially the same as a still image.
    image_exts = f.STATIC_EXTENSIONS | {"gif", ".webp", ".bmp", ".tiff"}
    images = sorted([
        os.path.join(category_dir, fn)
        for fn in os.listdir(category_dir)
        if os.path.isfile(os.path.join(category_dir, fn))
        and not fn.startswith(".")
        and os.path.splitext(fn)[1].lower() in image_exts
    ])
    if not images:
        return None

    # Choose a representative subset
    if sampling == "stratified":
        samples = _stratified_sample(images, max_samples)
    elif sampling == "smart":
        samples = _smart_sample(images, max_samples)
    else:
        # Legacy uniform
        n = len(images)
        if n <= max_samples:
            samples = images
        else:
            step = n / max_samples
            samples = [images[int(i * step)] for i in range(max_samples)]



    total = len(samples)
    numeric_vectors: List[Dict] = []
    weight_vectors: List[Dict[str, float]] = []
    full_profiles: List[Dict] = []
    extractor = _extract_feature_vector_fast if fast else _extract_feature_vector_full
    for i, img_path in enumerate(samples):
        profile = extractor(img_path)
        if not profile:
            if progress_callback:
                progress_callback(i + 1, total, os.path.basename(img_path))
            continue
        full_profiles.append(profile)
        weight_vectors.append(profile.get("weights") or {})
        numeric_vectors.append(_extract_numeric_vector(profile))
        if progress_callback:
            progress_callback(i + 1, total, os.path.basename(img_path))

    if not numeric_vectors:
        return None

    feat_agg = _aggregate_feature_vectors(numeric_vectors)
    palette_avg = _aggregate_color_weights(weight_vectors)

    # Auto-suggest palette_weights for .category.json using the average
    # palette plus the strongest single-image signal.
    palette_for_cfg = palette_avg

    result: Dict = {
        "features": feat_agg,
        "total_samples": len(images),
        "used_samples": len(numeric_vectors),
        "sample_size": len(samples),
        "sampling": sampling,
        "extractor": "fast" if fast else "full",
        "palette_avg": palette_avg,
        "palette_for_config": palette_for_cfg,
        "category": category_name or os.path.basename(category_dir.rstrip("/\\")),
    }
    return result


# Per-feature hand-tuned base weights. Style scores (anime, cyberpunk, ...)
# get the highest base weight because they encode the most semantically
# discriminative signal for the categories the user typically curates.
_FEATURE_BASE_WEIGHTS: Dict[str, float] = {
    # Colour
    "colorfulness": 1.5, "hsv_S_mean": 1.0, "hsv_V_mean": 1.0,
    "lab_warm_bias": 1.0, "lab_chroma_mean": 1.0,
    "sat_low_fraction": 1.0, "sat_high_fraction": 1.0,
    # Edges / texture
    "edge_density": 1.5, "mean_edge_magnitude": 1.0,
    "texture_entropy": 1.5, "texture_energy": 1.0,
    "lbp_entropy": 1.0, "lbp_variance": 1.0,
    # Structure
    "silhouette_ratio": 1.0, "silhouette_coverage": 1.0,
    "contour_symmetry": 1.0, "contour_complexity": 1.0,
    "high_low_ratio": 1.0, "radial_std": 0.5,
    "hog_entropy": 1.0, "hog_energy": 1.0,
    # Style scores - the most discriminative per-category
    "anime_score": 2.0, "cyberpunk_score": 2.0, "space_score": 2.0,
    "nature_score": 2.0, "neon_score": 1.8, "pastel_score": 1.8,
    "vw_score": 1.5, "sepia_score": 1.2, "vintage_score": 1.2,
    # Quality
    "tenengrad": 1.0, "sharpness": 1.0, "aesthetic": 1.0,
    "noise_estimate": 0.5, "bris_quality_score": 0.8,
    # Composition
    "centeredness": 1.0, "thirds_score": 1.0,
    "diagonal_strength": 0.8, "depth_score": 1.0,
    "depth_gradient": 0.8, "saliency_concentration": 1.0,
    # Subject
    "subject_area_ratio": 1.5, "fg_bg_contrast": 1.0,
    # Pattern
    "periodicity_score": 1.2, "tile_similarity": 1.0,
    "texture_complexity": 1.2, "gradient_complexity": 1.0,
    "local_variance": 0.8, "entropy": 0.8,
    # Symmetry
    "h_symmetry": 0.6, "v_symmetry": 0.6, "point_symmetry": 0.6,
    "main_diagonal_symmetry": 0.5, "anti_diagonal_symmetry": 0.5,
    "overall_symmetry": 1.5,
    # Other
    "skin_fraction": 1.5, "unique_colors": 1.2,
    # Content detectors - these are highly category-specific when they
    # fire (minecraft/pixel-art are unmistakable), but they only fire
    # on genuinely matching images so the base weight is moderate.
    "minecraft_score": 1.5, "pixel_art_score": 1.5, "minimalist_score": 1.5,
    "portrait_score": 1.2, "nsfw_score": 1.5, "gradient_score": 0.8,
    # Theme palette scores - palette is the primary signal for these
    "theme_catppuccin_score": 1.5, "theme_dracula_score": 1.5,
    "theme_tokyonight_score": 1.5, "theme_nord_score": 1.5,
    "theme_gruvbox_score": 1.5, "theme_everforest_score": 1.5,
    "theme_monochrome_score": 1.5, "theme_neon_score": 1.5,
}


def _shrink_std(cat_std: float, g_std: float, n_samples: int) -> float:
    """Bayesian-shrink a category std toward the global std.

    Small samples (<=5) pool strongly toward the global std, which prevents
    the "huge z-score for everything" problem caused by a noisy within-category
    estimate. Larger samples trust their own std.

    Returned value is bounded below by 0.05 so division never explodes.
    """
    if cat_std <= 0:
        return max(g_std, 0.05)
    if g_std <= 0:
        return max(cat_std, 0.05)
    # Effective sample count: trust more when we have more samples.
    k = 5.0
    w = k / (k + max(n_samples, 0))
    shrunk_sq = (1.0 - w) * (cat_std ** 2) + w * (g_std ** 2)
    return max(math.sqrt(shrunk_sq), 0.05)


def compute_global_stats(
    dest_dir: str,
    category_names: Optional[List[str]] = None,
) -> Dict:
    """Aggregate the per-feature stats across every category pattern.

    Returns a dict shaped like a single category's pattern, but with pooled
    `mean` (sample-weighted) and `std` (between-category spread). The
    `score_against_pattern` function uses this to:

      1. Apply Bayesian shrinkage to category stds (so tiny libraries still
         produce sensible z-scores).
      2. Detect which features are *discriminative* for each category
         (high `|cat_mean - g_mean| / g_std`) and weight them more.

    If `dest_dir` has no patterns yet, returns an empty `{"features": {}}`
    and the scoring falls back gracefully.
    """
    if category_names is None:
        try:
            c.discover_categories(dest_dir)
            category_names = list(c.CATEGORIES)
        except Exception:
            category_names = []

    per_key_means: Dict[str, List[Tuple[float, float, int]]] = {}
    for cat in category_names:
        cfg_path = os.path.join(dest_dir, cat, ".category.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        feats = cfg.get("heuristic_pattern", {}).get("features", {}) or {}
        for key, stats in feats.items():
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean")
            std = stats.get("std", 0.1)
            n = stats.get("samples", 1)
            if not isinstance(mean, (int, float)):
                continue
            per_key_means.setdefault(key, []).append((float(mean), float(std), int(n)))

    pooled: Dict[str, Dict[str, float]] = {}
    for key, entries in per_key_means.items():
        total_n = sum(max(n, 0) for _, _, n in entries)
        if total_n <= 0:
            continue
        # Sample-weighted mean (each category contributes its mean weighted
        # by how many samples produced it).
        wmean = sum(m * max(n, 0) for m, _, n in entries) / total_n
        # Between-category spread: variance of the category means,
        # weighted by their sample counts.
        var_btw = sum(max(n, 0) * (m - wmean) ** 2 for m, _, n in entries) / total_n
        std_btw = math.sqrt(max(var_btw, 1e-6))
        # Pooled within-category std (RMS of within-category stds weighted
        # by sample count). Used as the shrinkage target.
        var_win = sum(max(n, 0) * (s ** 2) for _, s, n in entries) / total_n
        std_win = math.sqrt(max(var_win, 1e-6))
        pooled[key] = {
            "mean": float(wmean),
            "std": float(std_btw),         # between-category spread (discriminativeness)
            "within_std": float(std_win),  # within-category variance (shrinkage target)
            "samples": int(total_n),
        }
    return {"features": pooled, "n_categories": len(category_names)}


def score_against_pattern(
    profile: dict,
    pattern: dict,
    global_stats: Optional[Dict] = None,
) -> float:
    """Score an image profile against a category's heuristic pattern.

    Returns a similarity in [0, 1]. Higher = more likely to belong to this
    category.

    The scoring has three layers:

      1. **Z-score similarity** to the category mean (using a std that is
         Bayesian-shrunk toward the global std when the category has few
         samples, so tiny libraries don't blow up).
      2. **Discriminativeness weighting** — features that DISTINGUISH this
         category from the rest (e.g. `cyberpunk_score` for a Cyberpunk
         folder) get up to 2.5x more weight than features that are generic
         across every category (e.g. `hsv_V_mean`).
      3. **Directional match** — if this category has an UNUSUALLY HIGH
         `cyberpunk_score` (vs. the global average) and the image has a
         LOWER-than-average `cyberpunk_score`, we penalise the match even
         if the image is "within the category's std". This stops a random
         dark image from matching a Cyberpunk folder whose pattern just
         happens to have low `cyberpunk_score` mean.

      A palette cosine similarity is blended in (30%) as before.

    If `global_stats` is None, the function falls back to the original
    non-discriminative behaviour so old `.category.json` files keep working.
    """
    features = pattern.get("features", {})
    if not features:
        return 0.0

    g_features = (global_stats or {}).get("features", {}) or {}

    total_weight = 0.0
    weighted_score = 0.0

    for key, stats in features.items():
        val = profile.get(key)
        if val is None or not isinstance(val, (int, float)):
            continue

        c_mean = float(stats.get("mean", val))
        c_std_raw = float(stats.get("std", 0.01))
        c_n = int(stats.get("samples", 1))

        g_stats = g_features.get(key, {}) if g_features else {}
        g_mean = float(g_stats.get("mean", c_mean)) if g_stats else c_mean
        g_std = float(g_stats.get("std", 0.0)) if g_stats else 0.0
        g_within = float(g_stats.get("within_std", c_std_raw)) if g_stats else c_std_raw

        # Layer 1: z-score similarity with Bayesian-shrunk std.
        if g_stats and g_within > 0:
            used_std = _shrink_std(c_std_raw, g_within, c_n)
        else:
            # No global stats available - use raw std with a floor.
            used_std = max(c_std_raw, 0.05)
        z_cat = abs(val - c_mean) / used_std
        sim_cat = max(0.0, 1.0 - z_cat / 3.0)

        # Layer 2: discriminativeness weight. How far is this category's
        # mean from the global mean, relative to the between-category std?
        # High = this feature is what makes the category special.
        base_w = _FEATURE_BASE_WEIGHTS.get(key, 1.0)
        if g_stats and g_std > 0.001:
            uniqueness = abs(c_mean - g_mean) / g_std
            # Floor at 0.4x, scale up to 2.5x for very distinctive features.
            d_weight = min(max(0.4, 0.4 + uniqueness * 0.6), 2.5)
        else:
            d_weight = 1.0

        # Layer 3: directional match. If both category and image deviate
        # from the global mean but in opposite directions, that's a strong
        # signal AGAINST a match.
        dir_match = 1.0
        if g_stats and abs(c_mean - g_mean) > 0.01 and abs(val - g_mean) > 0.01:
            if (c_mean - g_mean) * (val - g_mean) < 0:
                # Penalise proportional to how far they disagree.
                disagree = min(abs(val - g_mean) / max(g_std, 0.01), 2.0)
                dir_match = max(0.25, 1.0 - 0.35 * disagree)

        w = base_w * d_weight
        weighted_score += sim_cat * dir_match * w
        total_weight += w

    if total_weight == 0:
        return 0.0

    base_score = weighted_score / total_weight

    # Palette cosine similarity (unchanged). When global stats are available,
    # also blend a "palette uniqueness" component: how far is this category's
    # palette from the global average palette.
    palette_avg = pattern.get("palette_avg") or {}
    if palette_avg:
        p_palette = profile.get("weights") or {}
        if p_palette:
            total = sum(p_palette.values()) or 1
            pcts = {k: v / total for k, v in p_palette.items()}
            keys = set(palette_avg) | set(pcts)
            dot = sum(palette_avg.get(k, 0) * pcts.get(k, 0) for k in keys)
            n1 = sum(palette_avg.get(k, 0) ** 2 for k in keys) ** 0.5
            n2 = sum(pcts.get(k, 0) ** 2 for k in keys) ** 0.5
            if n1 > 0 and n2 > 0:
                sim = dot / (n1 * n2)
                base_score = 0.7 * base_score + 0.3 * sim

    return max(0.0, min(1.0, base_score))


def build_all_category_profiles(
    dest_dir: str,
    max_samples: int = 20,
    progress_callback=None,
    sampling: str = "stratified",
    fast: bool = False,
) -> Dict[str, Dict]:
    """Build profiles for all categories and save them to .category.json.

    Also fits a `MiniClassifier` from the sample profiles so the GUI can
    later do "category recommendation" calls with sub-millisecond latency.

    The global stats (used by `score_against_pattern` for discriminative
    scoring) are computed once across every rebuilt category and stored
    in the `global_stats` field of the first rebuilt category so subsequent
    scoring calls can re-use them without re-scanning every .category.json.
    """
    results = {}
    mini_classifier = None
    try:
        from ..minimal_ai import MiniClassifier
        mini_classifier = MiniClassifier()
    except Exception:
        mini_classifier = None

    c.discover_categories(dest_dir)
    cat_list = list(c.CATEGORIES)

    for cat in cat_list:
        cat_dir = os.path.join(dest_dir, cat) if dest_dir else os.path.join(
            c.CATEGORIES_DIR, cat)
        pattern = build_category_profile(
            cat_dir, max_samples=max_samples,
            progress_callback=progress_callback,
            category_name=cat,
            sampling=sampling,
            fast=fast,
        )
        if pattern:
            cfg = c.get_category_config(cat)
            cfg["heuristic_pattern"] = pattern
            # Auto-fill palette_weights if the user hasn't set any.
            if not cfg.get("palette_weights") and pattern.get("palette_for_config"):
                cfg["palette_weights"] = pattern["palette_for_config"]
            c.write_category_config(cat, cfg)
            results[cat] = pattern

            if mini_classifier is not None:
                # Re-extract the sample profiles (cheap; we already paid for them)
                # so the AI can learn the per-category fingerprint.
                mini_classifier.fit_category(
                    cat,
                    _re_extract_sample_profiles(cat_dir, max_samples=max_samples),
                    tags=cfg.get("tags") or [],
                )

    # Save the global stats to a small JSON so subsequent scoring runs
    # don't have to re-scan every .category.json.
    try:
        gs = compute_global_stats(dest_dir, category_names=cat_list)
        gs_path = os.path.join(dest_dir, ".heuristic_global_stats.json")
        with open(gs_path, "w", encoding="utf-8") as f:
            json.dump(gs, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if mini_classifier is not None:
        try:
            from .. import minimal_ai as _mini_ai
            _mini_ai._last_classifier = mini_classifier
        except Exception:
            pass

    return results


def load_global_stats(dest_dir: str) -> Dict:
    """Load the cached `.heuristic_global_stats.json` next to the patterns.

    Returns an empty `{"features": {}}` if the cache is missing or broken.
    The cache is refreshed by `build_all_category_profiles` and consumed
    by `_fingerprint_signal` in `classify.py`.
    """
    if not dest_dir:
        return {"features": {}}
    path = os.path.join(dest_dir, ".heuristic_global_stats.json")
    if not os.path.isfile(path):
        return {"features": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("features"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"features": {}}


def _re_extract_sample_profiles(category_dir: str, max_samples: int = 20) -> List[Dict]:
    """Re-extract numeric profiles for the sample images of a category.

    Used by `MiniClassifier.fit_category` so the AI classifier can build
    its per-category fingerprints.
    """
    if not os.path.isdir(category_dir):
        return []
    exts = f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS
    images = sorted([
        os.path.join(category_dir, fn)
        for fn in os.listdir(category_dir)
        if os.path.isfile(os.path.join(category_dir, fn))
        and not fn.startswith(".")
        and os.path.splitext(fn)[1].lower() in exts
    ])
    if not images:
        return []
    # Mirror the new sampling strategy so the AI sees what the fingerprint sees.
    images = _stratified_sample(images, max_samples)
    out: List[Dict] = []
    for p in images:
        profile = _extract_feature_vector(p)
        if profile:
            out.append(profile)
    return out

