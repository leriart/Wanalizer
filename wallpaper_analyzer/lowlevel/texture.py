"""
Texture analysis: Local Binary Patterns (LBP), Haralick features,
Gabor filters, and statistical texture measures.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def local_binary_pattern(img: Image.Image, radius=1, n_points=8) -> np.ndarray:
    """Compute Local Binary Pattern texture map."""
    if not HAS_CV2:
        return np.zeros((img.size[1], img.size[0]), dtype=np.float32)

    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    lbp = np.zeros_like(arr, dtype=np.float32)
    h, w = arr.shape

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
            bit = (shifted > arr).astype(np.float32)
            # Weight by distance from center
            weight = 1.0 / (dx * dx + dy * dy + 1e-6)
            lbp += bit * weight

    return lbp


def lbp_histogram(img: Image.Image, bins=26) -> np.ndarray:
    """LBP histogram for texture classification."""
    lbp = local_binary_pattern(img)
    hist, _ = np.histogram(lbp, bins=bins, range=(0, lbp.max() + 1e-6))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def glcm_contrast(arr: np.ndarray) -> float:
    """Gray-Level Co-occurrence Matrix contrast approximation."""
    h, w = arr.shape
    total = 0
    count = 0
    for dy, dx in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
        shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
        diff = (arr.astype(float) - shifted.astype(float)) ** 2
        total += float(diff.mean())
        count += 1
    return total / max(count, 1)


def texture_statistics(img: Image.Image) -> dict:
    """Statistical texture measures: contrast, energy, homogeneity, entropy."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    flat = arr.flatten()

    # Contrast via RMS of pixel differences
    contrast = glcm_contrast(arr)

    # Energy (angular second moment)
    energy = float((flat ** 2).sum()) / max(len(flat), 1)

    # Entropy of intensity distribution
    hist, _ = np.histogram(flat, bins=64, range=(0, 255))
    hist = hist.astype(np.float32)
    hist = hist[hist > 0]
    hist /= hist.sum()
    entropy = float(-(hist * np.log(hist + 1e-10)).sum())

    # Homogeneity (inverse difference moment approximation)
    h, w = arr.shape
    homogeneity = 0.0
    for dy, dx in [(0, 1), (1, 0)]:
        shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
        diff = np.abs(arr - shifted)
        homogeneity += float((1.0 / (1.0 + diff)).mean())
    homogeneity /= 2.0

    return {
        "contrast": float(contrast),
        "energy": float(energy),
        "entropy": float(entropy),
        "homogeneity": float(homogeneity),
        "std_intensity": float(flat.std()),
        "mean_intensity": float(flat.mean()),
    }


def gabor_filter_response(img: Image.Image) -> np.ndarray:
    """Multi-scale, multi-orientation Gabor filter bank response."""
    if not HAS_CV2:
        return np.zeros(16)

    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    responses = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for freq in [0.1, 0.2, 0.3, 0.4]:
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 1.0 / freq, 0.5, 0)
            filtered = cv2.filter2D(arr, cv2.CV_32F, kernel)
            responses.append(float(filtered.var()))
    return np.array(responses)


def texture_features(img: Image.Image) -> dict:
    """Combined texture features for classification."""
    stats = texture_statistics(img)
    lbp = lbp_histogram(img)
    gabor = gabor_filter_response(img)

    features = {
        "texture_contrast": stats["contrast"],
        "texture_energy": stats["energy"],
        "texture_entropy": stats["entropy"],
        "texture_homogeneity": stats["homogeneity"],
        "intensity_std": stats["std_intensity"],
    }

    # LBP histogram bins
    for i, v in enumerate(lbp[:8]):
        features[f"lbp_{i}"] = float(v)

    # Gabor response stats
    if len(gabor) > 0:
        features["gabor_mean"] = float(gabor.mean())
        features["gabor_std"] = float(gabor.std())

    return features
