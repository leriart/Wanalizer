"""
Edge detection algorithms: Canny, Sobel, Scharr, Laplacian.
Edge-based features for wallpaper classification.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def canny_edges(img: Image.Image, low=50, high=150) -> np.ndarray:
    """Canny edge detection. Returns edge magnitude map."""
    if not HAS_CV2:
        return _edge_fallback(img)
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return cv2.Canny(arr, low, high)


def sobel_magnitude(img: Image.Image) -> np.ndarray:
    """Sobel gradient magnitude (combined X and Y)."""
    if not HAS_CV2:
        return _edge_fallback(img)
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def scharr_magnitude(img: Image.Image) -> np.ndarray:
    """Scharr gradient magnitude (more accurate than Sobel)."""
    if not HAS_CV2:
        return _edge_fallback(img)
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    gx = cv2.Scharr(arr, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(arr, cv2.CV_32F, 0, 1)
    return np.sqrt(gx ** 2 + gy ** 2)


def _edge_fallback(img: Image.Image) -> np.ndarray:
    """Pillow-only edge detection fallback."""
    from PIL import ImageFilter
    g = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return np.asarray(g, dtype=np.float32)


def edge_density(img: Image.Image, method="canny") -> float:
    """Fraction of pixels identified as edges (0..1)."""
    if method == "canny":
        edges = canny_edges(img)
    elif method == "sobel":
        edges = sobel_magnitude(img) > 50
    elif method == "scharr":
        edges = scharr_magnitude(img) > 50
    else:
        edges = _edge_fallback(img) > 30
    return float(np.mean(edges > 0))


def edge_histogram(img: Image.Image, bins=16) -> np.ndarray:
    """Edge orientation histogram (HOG-light)."""
    if not HAS_CV2:
        return np.zeros(bins)
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang = (ang + 180) % 180
    hist, _ = np.histogram(ang, bins=bins, range=(0, 180), weights=mag)
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def edge_features(img: Image.Image) -> dict:
    """Extract edge-based features for classification."""
    if not HAS_CV2:
        return {"edge_density": float(_edge_fallback(img).mean() > 0.1), "dominant_orientation": 0}

    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    edges = cv2.Canny(arr, 50, 150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    density = edge_pixels / max(total_pixels, 1)

    gx = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang = (ang + 180) % 180

    mask = mag > (mag.mean() * 0.5)
    if mask.sum() > 0:
        dominant = float(np.average(ang, weights=mask * mag))
    else:
        dominant = 0.0

    # Horizontal vs vertical edge ratio
    horizontal = np.sum((ang > 80) & (ang < 100) & mask)
    vertical = np.sum(((ang > 170) | (ang < 10)) & mask)
    hv_ratio = horizontal / max(vertical, 1)

    return {
        "edge_density": float(density),
        "dominant_orientation": float(dominant),
        "horizontal_vertical_ratio": float(hv_ratio),
        "mean_edge_magnitude": float(mag[mask].mean()) if mask.sum() > 0 else 0.0,
    }
