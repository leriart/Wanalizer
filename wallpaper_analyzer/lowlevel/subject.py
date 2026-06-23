"""
Subject / foreground detection: largest connected component, bbox, position.

Identifies the main subject in the wallpaper and provides geometric
features about it (size, position, aspect, area).
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def largest_connected_component(img: Image.Image) -> dict:
    """Find the largest connected component (likely subject).

    Returns bbox, area ratio, and position features.
    """
    if not HAS_CV2:
        return {"subject_area_ratio": 0.5, "subject_position": "center",
                "subject_x": 0.5, "subject_y": 0.5, "subject_w": 0.5, "subject_h": 0.5}
    try:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        h, w = arr.shape
        # Otsu threshold
        try:
            _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            binary = (arr > arr.mean()).astype(np.uint8) * 255
        # Find contours
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return {"subject_area_ratio": 0.5, "subject_position": "center"}
        if not contours:
            return {"subject_area_ratio": 0.5, "subject_position": "center"}
        # Get largest
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            return {"subject_area_ratio": 0.0, "subject_position": "none"}
        # Bounding box
        x, y, bw, bh = cv2.boundingRect(largest)
        # Convert to normalized
        nx, ny = x / w, y / h
        nw, nh = bw / w, bh / h
        # Center of subject
        cx = nx + nw / 2
        cy = ny + nh / 2
        # Position: 9-zone classification
        col = 0 if cx < 0.33 else (1 if cx < 0.67 else 2)
        row = 0 if cy < 0.33 else (1 if cy < 0.67 else 2)
        positions = ["top-left", "top", "top-right",
                     "left", "center", "right",
                     "bottom-left", "bottom", "bottom-right"]
        position = positions[row * 3 + col]
        return {
            "subject_area_ratio": float(area / (w * h)),
            "subject_position": position,
            "subject_x": float(nx),
            "subject_y": float(ny),
            "subject_w": float(nw),
            "subject_h": float(nh),
            "subject_cx": float(cx),
            "subject_cy": float(cy),
            "subject_aspect": float(nw / max(nh, 1e-6)),
        }
    except Exception:
        return {"subject_area_ratio": 0.5, "subject_position": "center",
                "subject_x": 0.5, "subject_y": 0.5, "subject_w": 0.5, "subject_h": 0.5,
                "subject_cx": 0.5, "subject_cy": 0.5, "subject_aspect": 1.0}


def foreground_background_ratio(img: Image.Image) -> dict:
    """Ratio and contrast between foreground and background."""
    if not HAS_CV2:
        return {"fg_bg_ratio": 1.0, "fg_bg_contrast": 0.0}
    try:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape
        # Convert to grayscale for thresholding
        gray = arr.mean(axis=2)
        # Use Otsu
        try:
            _, binary = cv2.threshold(gray.astype(np.uint8), 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            binary = (gray > gray.mean()).astype(np.uint8) * 255
        # Compute mean color in fg and bg
        fg_mask = binary > 0
        bg_mask = ~fg_mask
        if fg_mask.sum() < 10 or bg_mask.sum() < 10:
            return {"fg_bg_ratio": 1.0, "fg_bg_contrast": 0.0}
        fg_mean = arr[fg_mask].mean(axis=0)
        bg_mean = arr[bg_mask].mean(axis=0)
        # Contrast: euclidean distance
        contrast = float(np.linalg.norm(fg_mean - bg_mean))
        # Ratio
        fg_area = float(fg_mask.sum())
        bg_area = float(bg_mask.sum())
        return {
            "fg_bg_ratio": float(fg_area / max(bg_area, 1)),
            "fg_bg_contrast": contrast,
        }
    except Exception:
        return {"fg_bg_ratio": 1.0, "fg_bg_contrast": 0.0}


def subject_features(img: Image.Image) -> dict:
    """Aggregate all subject features."""
    feats = {}
    try:
        feats.update(largest_connected_component(img))
    except Exception:
        pass
    try:
        feats.update(foreground_background_ratio(img))
    except Exception:
        feats["fg_bg_ratio"] = 1.0
        feats["fg_bg_contrast"] = 0.0
    return feats
