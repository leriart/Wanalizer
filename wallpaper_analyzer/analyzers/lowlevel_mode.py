"""
Advanced Low-Level Computer Vision Analyzer Mode.

Classifies wallpapers using classical CV techniques without ML models:
  - Edge analysis (Canny, Sobel, orientation)
  - Silhouette extraction (Otsu, adaptive thresholding)
  - Contour/shape analysis (Hu moments, symmetry, compactness)
  - Texture analysis (LBP, GLCM contrast, Gabor filters)
  - HOG features
  - Frequency analysis (FFT radial/angular distribution)
  - Feature detection (ORB, FAST corners)
  - Color layout and histogram analysis
  - Advanced color (moments, colorfulness, harmony, LAB/HSV stats)
  - Composition (rule of thirds, depth, saliency, diagonal)
  - Quality (Tenengrad, BRISQUE-like, perceptual hash, noise)
  - Subject (largest component, position, foreground/background)
  - Pattern (periodicity, tile detection, complexity)
  - Symmetry (bilateral, rotational, diagonal)

`classify()` defers to `classify_with_confidence` from `classify.py`, which
combines multiple signals (tags, palette, style, composition, fingerprint)
into a single confidence-weighted recommendation. The analyzer wrapper
keeps the previous behaviour for the organize pipeline (returns the best
category name as a string).
"""
from typing import Optional, Dict, List
import numpy as np
from PIL import Image

from .base import BaseAnalyzer
from ..categories import CATEGORIES, get_category_tags, get_palette_weights, get_category_prompt, get_category_config
from ..color import extract_color_weights, rgb_to_hsv, dominant_colors_hex
from ..profile import get_image_profile
from ..lowlevel.edges import edge_features
from ..lowlevel.silhouettes import silhouette_features
from ..lowlevel.contours import contour_features
from ..lowlevel.texture import texture_features
from ..lowlevel.features import feature_features
from ..lowlevel.fourier import frequency_features
from ..lowlevel.hog import hog_features
from ..lowlevel.color_advanced import color_features
from ..lowlevel.composition import composition_features
from ..lowlevel.quality_advanced import quality_features
from ..lowlevel.subject import subject_features
from ..lowlevel.pattern import pattern_features
from ..lowlevel.symmetry_advanced import symmetry_features
from .. import classify as _classify


class LowLevelAnalyzer(BaseAnalyzer):
    """
    Analyzer that uses only classical computer vision algorithms.
    No ML models required. Fast, deterministic, works on any hardware.
    """

    name = "lowlevel"

    def __init__(self, settings: dict = None):
        self.settings = settings or {}

    def analyze(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")

        # Core profile (always computed)
        profile = get_image_profile(image_path)

        # Original CV features
        edge_feat = edge_features(img)
        sil_feat = silhouette_features(img)
        contour_feat = contour_features(img)
        tex_feat = texture_features(img)
        feat_feat = feature_features(img)
        freq_feat = frequency_features(img)
        hog_feat = hog_features(img)

        # Color layout descriptor
        color_weights, style_scores = extract_color_weights(img)
        dominant = dominant_colors_hex(img, 5)

        # NEW: Advanced algorithms
        adv_color = color_features(img)
        composition = composition_features(img)
        quality = quality_features(img)
        subject = subject_features(img)
        pattern = pattern_features(img)
        symmetry = symmetry_features(img)

        profile.update({
            **edge_feat,
            **sil_feat,
            **contour_feat,
            **tex_feat,
            **feat_feat,
            **freq_feat,
            **hog_feat,
            **adv_color,
            **composition,
            **quality,
            **subject,
            **pattern,
            **symmetry,
            "dominant_colors": dominant,
            "mode": "lowlevel",
        })

        img.close()
        return profile

    def classify(self, profile: dict) -> Optional[str]:
        """Classify using a weighted combination of multi-signal heuristics.

        Delegates to `classify.classify_with_confidence`, which combines:
          * TF-IDF tag matching
          * Palette, style, composition, quality, pattern and size signals
          * Z-score similarity against each category's CV fingerprint

        Falls back to a palette-only decision when the multi-signal score
        is too low. Returns None only when no categories exist.
        """
        valid_cats = set(CATEGORIES)
        if not valid_cats:
            return None

        info = _classify.classify_with_confidence(profile)
        if info.get("category") in valid_cats:
            return info["category"]

        # Fallback to palette-only classification
        weights = profile.get("weights", {})
        if weights:
            total_w = sum(weights.values()) or 1
            pcts = {k: v / total_w for k, v in weights.items()}
            best_fb = None
            best_fb_score = -1.0
            for cat in valid_cats:
                pw = get_palette_weights(cat)
                if not pw:
                    continue
                s = sum(pcts.get(c, 0.0) * w for c, w in pw.items())
                if s > best_fb_score:
                    best_fb_score = s
                    best_fb = cat
            if best_fb and best_fb_score > 0.15:
                return best_fb

        return CATEGORIES[0] if CATEGORIES else None

    def classify_with_confidence(self, profile: dict) -> Dict:
        """Expose the full multi-signal classification result."""
        return _classify.classify_with_confidence(profile)
