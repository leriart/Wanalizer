"""
Advanced color analysis: moments, histograms, colorfulness, LAB stats.

Goes beyond simple color weights to provide statistical color features
useful for wallpaper classification.
"""
import numpy as np
from PIL import Image


def _to_array(img: Image.Image) -> np.ndarray:
    """Convert image to RGB uint8 numpy array."""
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _to_float(img: Image.Image) -> np.ndarray:
    """Convert image to RGB float32 [0,1] numpy array."""
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def color_moments(img: Image.Image) -> dict:
    """Color moments: mean, std, skewness per channel.

    Useful as a compact color signature. Skewness indicates color
    distribution asymmetry (e.g., warm vs cool bias).
    """
    arr = _to_float(img)
    moments = {}
    channel_names = ["R", "G", "B"]
    for i, ch in enumerate(channel_names):
        c = arr[:, :, i].flatten()
        mean = float(np.mean(c))
        std = float(np.std(c))
        # Skewness = E[(x-mu)^3] / sigma^3
        if std > 1e-6:
            skew = float(np.mean((c - mean) ** 3) / (std ** 3))
        else:
            skew = 0.0
        moments[f"cm_mean_{ch}"] = mean
        moments[f"cm_std_{ch}"] = std
        moments[f"cm_skew_{ch}"] = skew
    return moments


def color_histogram_features(img: Image.Image, bins=16) -> dict:
    """Color histogram in RGB space with bins per channel.

    Returns summary statistics and a flat histogram vector.
    """
    arr = _to_array(img)
    hists = []
    moments = {}
    for i, ch in enumerate(["R", "G", "B"]):
        h, _ = np.histogram(arr[:, :, i], bins=bins, range=(0, 256))
        h = h / max(h.sum(), 1)
        hists.append(h)
        moments[f"hist_{ch}_max"] = float(h.max())
        moments[f"hist_{ch}_min"] = float(h.min())
        moments[f"hist_{ch}_entropy"] = float(-np.sum(h * np.log2(h + 1e-10)))
    # Histogram spread (how concentrated the colors are)
    moments["hist_overall_entropy"] = float(np.mean([
        moments["hist_R_entropy"], moments["hist_G_entropy"], moments["hist_B_entropy"]
    ]))
    moments["hist_concentration"] = float(np.mean([h.max() for h in hists]))
    return moments


def colorfulness(img: Image.Image) -> float:
    """Hasler-Süsstrunk colorfulness metric (2003).

    Higher = more colorful/saturated. Returns 0..100+ typically.
    """
    arr = _to_float(img)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    # Mean and std of opponent color differences
    rg_mean, rg_std = float(np.mean(rg)), float(np.std(rg))
    yb_mean, yb_std = float(np.mean(yb)), float(np.std(yb))
    mean_root = np.sqrt(rg_mean ** 2 + yb_mean ** 2)
    std_root = np.sqrt(rg_std ** 2 + yb_std ** 2)
    return float(mean_root + std_root)


def lab_statistics(img: Image.Image) -> dict:
    """Statistics in CIELAB color space.

    a/b channels correlate with color hue. Useful for warm/cool bias.
    """
    try:
        lab = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # Use cv2 for fast RGB->LAB if available
        try:
            import cv2
            lab_arr = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB).astype(np.float32)
        except ImportError:
            # Manual approximation
            r, g, b = lab[:, :, 0] / 255.0, lab[:, :, 1] / 255.0, lab[:, :, 2] / 255.0
            # Simplified RGB->LAB
            L = 0.2126 * r + 0.7152 * g + 0.0722 * b
            a = (r - g) * 1.0
            b_ = (g - b) * 0.5
            lab_arr = np.stack([L, a, b_], axis=2)
        L, A, B = lab_arr[:, :, 0], lab_arr[:, :, 1], lab_arr[:, :, 2]
        return {
            "lab_L_mean": float(np.mean(L)),
            "lab_L_std": float(np.std(L)),
            "lab_a_mean": float(np.mean(A)),
            "lab_a_std": float(np.std(A)),
            "lab_b_mean": float(np.mean(B)),
            "lab_b_std": float(np.std(B)),
            # Warm-cool bias: positive b = warm (yellow), positive a = warm (red)
            "lab_warm_bias": float((np.mean(A) + np.mean(B)) / 2.0),
            "lab_chroma_mean": float(np.mean(np.sqrt(A ** 2 + B ** 2))),
        }
    except Exception:
        return {"lab_L_mean": 0, "lab_L_std": 0, "lab_a_mean": 0, "lab_a_std": 0,
                "lab_b_mean": 0, "lab_b_std": 0, "lab_warm_bias": 0, "lab_chroma_mean": 0}


def hsv_statistics(img: Image.Image) -> dict:
    """Statistics in HSV color space."""
    try:
        hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    except Exception:
        # PIL doesn't always handle HSV well
        import colorsys
        arr = _to_array(img)
        hsv = np.zeros_like(arr, dtype=np.uint8)
        # Skip manual conversion if PIL fails
        return {
            "hsv_H_mean": 0, "hsv_S_mean": 0, "hsv_V_mean": 0,
            "hsv_H_std": 0, "hsv_S_std": 0, "hsv_V_std": 0,
            "hsv_saturation_mean": 0, "hsv_brightness_mean": 0,
        }
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return {
        "hsv_H_mean": float(np.mean(H)),
        "hsv_H_std": float(np.std(H)),
        "hsv_S_mean": float(np.mean(S)),
        "hsv_S_std": float(np.std(S)),
        "hsv_V_mean": float(np.mean(V)),
        "hsv_V_std": float(np.std(V)),
        "hsv_saturation_mean": float(np.mean(S) / 255.0),
        "hsv_brightness_mean": float(np.mean(V) / 255.0),
    }


def saturation_distribution(img: Image.Image) -> dict:
    """Analyze saturation distribution for style hints.

    Bimodal saturation = high contrast (vivid+gray). Unimodal low = pastel.
    """
    try:
        hsv = np.asarray(img.convert("HSV"))
        S = hsv[:, :, 1].flatten() / 255.0
        # Split into low/high saturation
        low_sat = float(np.mean(S < 0.3))
        high_sat = float(np.mean(S > 0.7))
        # Variance of saturation
        sat_var = float(np.var(S))
        # Binarize: bimodal = high variance
        return {
            "sat_low_fraction": low_sat,
            "sat_high_fraction": high_sat,
            "sat_variance": sat_var,
            "sat_bimodal": float(low_sat > 0.2 and high_sat > 0.2),
        }
    except Exception:
        return {"sat_low_fraction": 0, "sat_high_fraction": 0,
                "sat_variance": 0, "sat_bimodal": 0}


def color_harmony(img: Image.Image, n_colors=6) -> dict:
    """Detect color harmony type from dominant colors.

    Types: monochrome (all same hue), complementary (2 opposite),
    triadic (3 evenly spaced), analogous (adjacent hues), polychromatic.
    """
    try:
        import cv2
        # Quantize to find dominant colors
        arr = np.asarray(img.convert("RGB"))
        pixels = arr.reshape(-1, 3).astype(np.float32)
        # K-means with small k
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, n_colors, None, crit, 3,
                                            cv2.KMEANS_PP_CENTERS)
        except Exception:
            return {"harmony_type": "unknown", "n_dominant": 0, "hue_spread": 0}
        # Convert centers to HSV hue
        centers_u8 = np.uint8(centers.reshape(1, -1, 3))
        hsv = cv2.cvtColor(centers_u8, cv2.COLOR_RGB2HSV)
        hues = hsv[0, :, 0].astype(np.float32) * 2  # convert to 0..360
        # Get the most common cluster
        unique, counts = np.unique(labels, return_counts=True)
        top_idx = unique[np.argmax(counts)]
        primary_hue = hues[top_idx]
        # Count hues in same/adjacent sector
        diffs = np.abs(hues - primary_hue)
        diffs = np.minimum(diffs, 360 - diffs)
        same_hue = int(np.sum(diffs < 20))
        opposite = int(np.sum(diffs > 150))
        triadic = int(np.sum(np.abs(diffs - 120) < 20))
        analogous = int(np.sum(diffs < 30))
        # Determine harmony
        if opposite >= 1 and same_hue >= 1:
            harmony = "complementary"
        elif triadic >= 2:
            harmony = "triadic"
        elif analogous >= 3:
            harmony = "analogous"
        elif same_hue >= 3:
            harmony = "monochrome"
        else:
            harmony = "polychromatic"
        return {
            "harmony_type": harmony,
            "n_dominant": int(len(unique)),
            "hue_spread": float(np.max(diffs)),
        }
    except Exception:
        return {"harmony_type": "unknown", "n_dominant": 0, "hue_spread": 0}


def color_features(img: Image.Image) -> dict:
    """Aggregate all advanced color features."""
    feats = {}
    try:
        feats.update(color_moments(img))
    except Exception:
        pass
    try:
        feats["colorfulness"] = colorfulness(img)
    except Exception:
        feats["colorfulness"] = 0.0
    try:
        feats.update(lab_statistics(img))
    except Exception:
        pass
    try:
        feats.update(hsv_statistics(img))
    except Exception:
        pass
    try:
        feats.update(saturation_distribution(img))
    except Exception:
        pass
    try:
        feats.update(color_harmony(img))
    except Exception:
        pass
    return feats
