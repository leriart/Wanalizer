"""Fusion analyzer: CLIP + Low-Level CV working together.

The two approaches have complementary strengths:

  * CLIP ("what is this an image OF?") excels at semantic content
    recognition. It knows that Minecraft is a voxel game, that
    Catppuccin is a colour theme, that an anime girl illustration is
    anime. CV feature histograms struggle with all of these.

  * LowLevel CV ("what does this image LOOK LIKE?") excels at
    statistical structure - colour palettes, edge distributions,
    texture complexity, symmetry, periodicity. CLIP ignores all of
    this - it's all about semantic matching.

The fusion runs both pipelines and lets `classify._combine_signals`
combine the signals (the existing CLIP signal + the existing
fingerprint / tag / palette signals) using the same multi-signal
combination as every other analyzer.

CLIP is optional - when it isn't loaded (no torch, no clip, load
failure) the analyzer falls back to the pure-LowLevel pipeline so
the fusion mode is a strict superset of lowlevel_mode.
"""
from __future__ import annotations

from typing import Dict, Optional

from PIL import Image

from .base import BaseAnalyzer
from ..categories import CATEGORIES
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


class FusionAnalyzer(BaseAnalyzer):
    """Combine CLIP zero-shot classification with classical CV features.

    Produces a profile that has BOTH the full low-level CV features
    AND the per-category CLIP scores + NSFW sub-score. The existing
    `classify._clip_signal` picks up the CLIP scores automatically
    because they're stored under the `clip_score_<cat>` keys that
    `_combine_signals` already knows about.

    If CLIP isn't installed (no torch, no `clip` package, model load
    failure) the fusion gracefully degrades to the same pipeline as
    `LowLevelAnalyzer` so this mode is always usable.
    """

    name = "fusion"

    def __init__(self, settings: dict = None):
        self.settings = settings or {}
        # Try to grab the CLIP engine. Don't load eagerly - the engine
        # will load on first encode_image call. The model name comes
        # from `settings["clip_model"]` (set by the AI Models page).
        self._engine = None
        try:
            from ..clip_client import get_engine
            self._engine = get_engine(self.settings.get("clip_model"))
        except Exception:
            self._engine = None

    def _clip_available(self) -> bool:
        return self._engine is not None and self._engine.available

    def analyze(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")

        # ---- Low-Level CV features (same as LowLevelAnalyzer) ----
        profile = get_image_profile(image_path)
        edge_feat = edge_features(img)
        sil_feat = silhouette_features(img)
        contour_feat = contour_features(img)
        tex_feat = texture_features(img)
        feat_feat = feature_features(img)
        freq_feat = frequency_features(img)
        hog_feat = hog_features(img)
        adv_color = color_features(img)
        composition = composition_features(img)
        quality = quality_features(img)
        subject = subject_features(img)
        pattern = pattern_features(img)
        symmetry = symmetry_features(img)
        profile.update({
            **edge_feat, **sil_feat, **contour_feat, **tex_feat,
            **feat_feat, **freq_feat, **hog_feat,
            **adv_color, **composition, **quality, **subject,
            **pattern, **symmetry,
            "mode": "fusion",
        })

        # ---- CLIP zero-shot scores ----
        if self._clip_available():
            try:
                from ..clip_client import score_image, nsfw_score
                scores = score_image(image_path)
                if scores:
                    for cat, p in scores.items():
                        profile[f"clip_score_{cat}"] = p
                ns = nsfw_score(image_path)
                if ns is not None:
                    profile["clip_nsfw"] = ns
                profile["clip_available"] = True
            except Exception:
                profile["clip_available"] = False
        else:
            profile["clip_available"] = False

        img.close()
        return profile

    def classify(self, profile: dict) -> Optional[str]:
        valid = set(CATEGORIES)
        if not valid:
            return None
        info = _classify.classify_with_confidence(profile)
        cat = info.get("category")
        return cat if cat in valid else None

    def classify_with_confidence(self, profile: dict) -> Dict:
        return _classify.classify_with_confidence(profile)

    def diagnostics(self, profile: dict) -> Dict:
        """Return per-signal diagnostics so the GUI can show the user
        whether CLIP actually contributed to this image's classification.
        """
        info = _classify.classify_with_confidence(profile)
        diag = dict(info)
        diag["clip_available"] = bool(profile.get("clip_available"))
        if diag["clip_available"]:
            top_clip = max(
                (k, v) for k, v in profile.items()
                if k.startswith("clip_score_")
            )
            diag["clip_top"] = (top_clip[0].replace("clip_score_", ""), top_clip[1])
        return diag
