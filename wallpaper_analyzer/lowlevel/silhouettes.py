"""
Silhouette extraction and analysis using thresholding, segmentation,
and morphological operations.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def extract_silhouette_otsu(img: Image.Image) -> np.ndarray:
    """Extract foreground silhouette using Otsu thresholding."""
    if not HAS_CV2:
        return _silhouette_fallback(img)
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_silhouette_adaptive(img: Image.Image, block_size=31, c=10) -> np.ndarray:
    """Extract silhouette using adaptive thresholding."""
    if not HAS_CV2:
        return _silhouette_fallback(img)
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block_size, c)


def extract_silhouette_kmeans(img: Image.Image, k=3) -> np.ndarray:
    """Extract silhouette via K-means clustering in grayscale."""
    if not HAS_CV2:
        return _silhouette_fallback(img)
    arr = np.asarray(img.convert("L"), dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(arr, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels.reshape(img.size[1], img.size[0]).astype(np.uint8) * (255 // (k - 1))


def _silhouette_fallback(img: Image.Image) -> np.ndarray:
    """Fallback using Pillow's point operation."""
    from PIL import ImageFilter
    g = img.convert("L").filter(ImageFilter.SMOOTH)
    arr = np.asarray(g, dtype=np.uint8)
    mean = arr.mean()
    return np.where(arr > mean, 255, 0).astype(np.uint8)


def silhouette_ratio(img: Image.Image) -> float:
    """Ratio of foreground to total area."""
    sil = extract_silhouette_otsu(img)
    foreground = np.sum(sil > 0)
    return foreground / max(sil.size, 1)


def centroid_offset(img: Image.Image) -> tuple:
    """(dx, dy) of silhouette centroid from image center, normalized to [-1, 1]."""
    sil = extract_silhouette_otsu(img)
    h, w = sil.shape
    ys, xs = np.where(sil > 0)
    if len(ys) == 0:
        return (0.0, 0.0)
    cy, cx = float(ys.mean()), float(xs.mean())
    dx = (cx - w / 2) / (w / 2)
    dy = (cy - h / 2) / (h / 2)
    return (dx, dy)


def silhouette_features(img: Image.Image) -> dict:
    """Comprehensive silhouette features."""
    sil = extract_silhouette_otsu(img)
    h, w = sil.shape
    fg_pixels = np.sum(sil > 0)

    # Silhouette compactness: perimeter^2 / (4 * pi * area)
    if not HAS_CV2 or fg_pixels == 0:
        return {
            "silhouette_ratio": fg_pixels / max(h * w, 1),
            "compactness": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
        }

    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"silhouette_ratio": 0.0, "compactness": 0.0, "centroid_x": 0.0, "centroid_y": 0.0}

    # Merge all external contours
    all_pts = np.vstack(contours)
    area = cv2.contourArea(all_pts)
    perimeter = cv2.arcLength(all_pts, True)
    compactness = (perimeter ** 2) / max(4 * np.pi * area, 1e-6) if area > 0 else 0.0

    M = cv2.moments(all_pts)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = w / 2, h / 2

    return {
        "silhouette_ratio": fg_pixels / max(h * w, 1),
        "compactness": float(min(compactness, 10.0)),
        "centroid_x": float((cx - w / 2) / (w / 2)),
        "centroid_y": float((cy - h / 2) / (h / 2)),
        "num_contours": len(contours),
    }
