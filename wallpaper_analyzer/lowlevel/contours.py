"""
Contour analysis: shape descriptors, Hu moments, and geometric features.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def hu_moments(img: Image.Image) -> np.ndarray:
    """Calculate Hu invariant moments (7 values, log-scaled)."""
    if not HAS_CV2:
        return np.zeros(7)
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments)
    # Log scale for numerical stability
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.flatten()


def contour_complexity(img: Image.Image) -> dict:
    """Analyze contour structure complexity."""
    if not HAS_CV2:
        return {"num_contours": 0, "avg_area": 0.0, "max_area_ratio": 0.0}

    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"num_contours": 0, "avg_area": 0.0, "max_area_ratio": 0.0, "avg_solidity": 0.0}

    areas = [cv2.contourArea(c) for c in contours]
    total_area = sum(areas)
    img_area = thresh.size
    hulls = [cv2.convexHull(c) for c in contours]
    hull_areas = [cv2.contourArea(h) for h in hulls]
    solidities = [a / max(ha, 1e-6) for a, ha in zip(areas, hull_areas)] if hull_areas else [0]

    return {
        "num_contours": len(contours),
        "avg_area": float(np.mean(areas)) if areas else 0.0,
        "max_area_ratio": float(max(areas) / max(img_area, 1)) if areas else 0.0,
        "avg_solidity": float(np.mean(solidities)) if solidities else 0.0,
        "total_contour_area_ratio": float(total_area / max(img_area, 1)),
    }


def shape_symmetry(img: Image.Image) -> float:
    """Measure shape symmetry (0 = asymmetric, 1 = perfectly symmetric)."""
    if not HAS_CV2:
        return 0.5
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = arr.shape
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    left = thresh[:, :w // 2]
    right = np.fliplr(thresh[:, -w // 2:])
    min_h = min(left.shape[0], right.shape[0])
    min_w = min(left.shape[1], right.shape[1])
    if min_h == 0 or min_w == 0:
        return 0.5
    diff = np.abs(left[:min_h, :min_w].astype(float) - right[:min_h, :min_w].astype(float))
    return float(1.0 - diff.mean() / 255.0)


def contour_features(img: Image.Image) -> dict:
    """Combined contour and shape features."""
    hu = hu_moments(img)
    complexity = contour_complexity(img)
    symmetry = shape_symmetry(img)
    return {
        "hu_moment_1": float(hu[0]) if len(hu) > 0 else 0.0,
        "hu_moment_2": float(hu[1]) if len(hu) > 1 else 0.0,
        "hu_moment_3": float(hu[2]) if len(hu) > 2 else 0.0,
        "symmetry": symmetry,
        **complexity,
    }
