"""
Pattern detection: tile/repeat detection, periodicity, texture complexity.

Wallpapers often have repeating patterns (mandala, geometric, fabric).
Detecting these is a strong style signal.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def autocorrelation_periodicity(img: Image.Image, max_lag=64) -> dict:
    """Detect periodicity via autocorrelation.

    If the image has a repeating pattern, autocorrelation will have
    peaks at multiples of the period.
    """
    if not HAS_CV2:
        return {"is_periodic": 0, "period_x": 0, "period_y": 0, "periodicity_score": 0}
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        # Resize to speed up
        small = cv2.resize(arr, (128, 128))
        # Normalize
        small = (small - small.mean()) / (small.std() + 1e-6)
        # Compute autocorrelation via FFT
        f = np.fft.fft2(small)
        acf = np.fft.ifft2(f * np.conj(f)).real
        acf = np.fft.fftshift(acf)
        acf /= acf.max() + 1e-6
        h, w = acf.shape
        cy, cx = h // 2, w // 2
        # Look for peaks in horizontal and vertical lines through center
        # Horizontal autocorrelation
        horiz = acf[cy, :]
        vert = acf[:, cx]
        # Find first peak (excluding center)
        def find_first_peak(line, max_lag):
            line = np.abs(line)
            if line[0] == 0:
                return 0, 0
            for lag in range(1, min(max_lag, len(line) // 2)):
                if lag + 1 < len(line) and line[lag] > line[lag - 1] and line[lag] > line[lag + 1]:
                    if line[lag] > 0.2:  # significant peak
                        return lag, float(line[lag])
            return 0, 0
        period_x, peak_x = find_first_peak(horiz, max_lag)
        period_y, peak_y = find_first_peak(vert, max_lag)
        periodicity_score = max(peak_x, peak_y)
        is_periodic = 1.0 if periodicity_score > 0.3 else 0.0
        return {
            "is_periodic": is_periodic,
            "period_x": int(period_x),
            "period_y": int(period_y),
            "periodicity_score": float(periodicity_score),
        }
    except Exception:
        return {"is_periodic": 0, "period_x": 0, "period_y": 0, "periodicity_score": 0}


def tile_detection(img: Image.Image) -> dict:
    """Detect if image is composed of repeating tiles.

    Compares different regions of the image - if they're similar,
    it's likely tiled.
    """
    if not HAS_CV2:
        return {"is_tiled": 0, "tile_similarity": 0}
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        # Sample 4 quadrants
        tile_h, tile_w = h // 4, w // 4
        quadrants = [
            arr[:tile_h, :tile_w],
            arr[:tile_h, w - tile_w:],
            arr[h - tile_h:, :tile_w],
            arr[h - tile_h:, w - tile_w:],
        ]
        # Normalize
        norm_quads = [(q - q.mean()) / (q.std() + 1e-6) for q in quadrants]
        # Compute pairwise normalized cross-correlation
        similarities = []
        for i in range(len(norm_quads)):
            for j in range(i + 1, len(norm_quads)):
                a, b = norm_quads[i], norm_quads[j]
                # Resize to same size
                if a.shape != b.shape:
                    try:
                        b = cv2.resize(b, (a.shape[1], a.shape[0]))
                    except Exception:
                        continue
                sim = float((a * b).mean() / (np.sqrt((a ** 2).mean() * (b ** 2).mean()) + 1e-6))
                similarities.append(sim)
        avg_sim = np.mean(similarities) if similarities else 0
        is_tiled = 1.0 if avg_sim > 0.7 else 0.0
        return {
            "is_tiled": is_tiled,
            "tile_similarity": float(avg_sim),
        }
    except Exception:
        return {"is_tiled": 0, "tile_similarity": 0}


def texture_complexity(img: Image.Image) -> dict:
    """Measure texture complexity (low = smooth, high = detailed).

    Uses edge density, local variance, and entropy.
    """
    if not HAS_CV2:
        return {"texture_complexity": 0.5, "local_variance": 0}
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        # Local variance via box filter
        kernel_size = 15
        if HAS_CV2:
            mean = cv2.boxFilter(arr, -1, (kernel_size, kernel_size))
            sqr_mean = cv2.boxFilter(arr * arr, -1, (kernel_size, kernel_size))
            local_var = sqr_mean - mean ** 2
        else:
            local_var = np.zeros_like(arr)
        # Edge density
        edges = cv2.Canny(arr.astype(np.uint8), 50, 150)
        edge_density = float(edges.mean() / 255.0)
        # Entropy
        hist, _ = np.histogram(arr.astype(np.uint8), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-6)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        # Combined score
        complexity = float((edge_density + local_var.mean() / 1000.0 + entropy / 8.0) / 3.0)
        return {
            "texture_complexity": min(complexity, 1.0),
            "local_variance": float(local_var.mean()),
            "entropy": entropy,
        }
    except Exception:
        return {"texture_complexity": 0.5, "local_variance": 0, "entropy": 0}


def gradient_complexity(img: Image.Image) -> float:
    """Complexity of the gradient field (smooth gradients vs chaotic).

    Low = smooth gradients (vector art, simple backgrounds).
    High = chaotic gradients (photographs, noise).
    """
    if not HAS_CV2:
        return 0.5
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        # Local gradient variance (high = chaotic, low = smooth)
        mean = cv2.boxFilter(mag, -1, (15, 15))
        sqr_mean = cv2.boxFilter(mag * mag, -1, (15, 15))
        local_var = sqr_mean - mean ** 2
        return float(np.clip(local_var.mean() / 1000.0, 0, 1))
    except Exception:
        return 0.5


def pattern_features(img: Image.Image) -> dict:
    """Aggregate all pattern features."""
    feats = {}
    try:
        feats.update(autocorrelation_periodicity(img))
    except Exception:
        feats["is_periodic"] = 0
        feats["period_x"] = 0
        feats["period_y"] = 0
        feats["periodicity_score"] = 0
    try:
        feats.update(tile_detection(img))
    except Exception:
        feats["is_tiled"] = 0
        feats["tile_similarity"] = 0
    try:
        feats.update(texture_complexity(img))
    except Exception:
        pass
    try:
        feats["gradient_complexity"] = gradient_complexity(img)
    except Exception:
        feats["gradient_complexity"] = 0.5
    return feats
