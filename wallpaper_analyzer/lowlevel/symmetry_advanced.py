"""
Advanced symmetry analysis: bilateral, quadrant, rotational, point.

Beyond simple shape symmetry, detects reflection and rotational
symmetry in the image content.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def horizontal_symmetry(img: Image.Image) -> float:
    """Vertical axis symmetry (left half mirrors right half).

    Returns 0..1 where 1 = perfectly symmetric.
    """
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        # Split and flip
        left = arr[:, :w // 2]
        right = np.flip(arr[:, w - w // 2:], axis=1)
        # Resize to same shape
        if left.shape != right.shape:
            right = np.flip(arr[:, w // 2:], axis=1)
            min_w = min(left.shape[1], right.shape[1])
            left = left[:, :min_w]
            right = right[:, :min_w]
        if left.size == 0:
            return 0.0
        # Normalize
        left_n = (left - left.mean()) / (left.std() + 1e-6)
        right_n = (right - right.mean()) / (right.std() + 1e-6)
        # Cross-correlation
        sim = float((left_n * right_n).mean() /
                    (np.sqrt((left_n ** 2).mean() * (right_n ** 2).mean()) + 1e-6))
        return max(0.0, sim)
    except Exception:
        return 0.0


def vertical_symmetry(img: Image.Image) -> float:
    """Horizontal axis symmetry (top half mirrors bottom half).

    Returns 0..1 where 1 = perfectly symmetric.
    """
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        top = arr[:h // 2, :]
        bottom = np.flip(arr[h - h // 2:, :], axis=0)
        if top.shape != bottom.shape:
            bottom = np.flip(arr[h // 2:, :], axis=0)
            min_h = min(top.shape[0], bottom.shape[0])
            top = top[:min_h, :]
            bottom = bottom[:min_h, :]
        if top.size == 0:
            return 0.0
        top_n = (top - top.mean()) / (top.std() + 1e-6)
        bottom_n = (bottom - bottom.mean()) / (bottom.std() + 1e-6)
        sim = float((top_n * bottom_n).mean() /
                    (np.sqrt((top_n ** 2).mean() * (bottom_n ** 2).mean()) + 1e-6))
        return max(0.0, sim)
    except Exception:
        return 0.0


def point_symmetry(img: Image.Image) -> float:
    """Point symmetry (180 degree rotational symmetry).

    Returns 0..1.
    """
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        # 180-degree rotation
        rotated = np.flip(arr, axis=(0, 1))
        if rotated.shape != arr.shape:
            min_h, min_w = min(h, rotated.shape[0]), min(w, rotated.shape[1])
            a = arr[:min_h, :min_w]
            b = rotated[:min_h, :min_w]
        else:
            a, b = arr, rotated
        if a.size == 0:
            return 0.0
        a_n = (a - a.mean()) / (a.std() + 1e-6)
        b_n = (b - b.mean()) / (b.std() + 1e-6)
        sim = float((a_n * b_n).mean() /
                    (np.sqrt((a_n ** 2).mean() * (b_n ** 2).mean()) + 1e-6))
        return max(0.0, sim)
    except Exception:
        return 0.0


def diagonal_symmetry(img: Image.Image) -> dict:
    """Diagonal symmetry (main and anti-diagonal).

    Returns scores for both diagonals.
    """
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        # Main diagonal: arr[i,j] vs arr[j,i]
        if h == w:
            transpose = arr.T
            a_n = (arr - arr.mean()) / (arr.std() + 1e-6)
            b_n = (transpose - transpose.mean()) / (transpose.std() + 1e-6)
            main_diag = float((a_n * b_n).mean() /
                              (np.sqrt((a_n ** 2).mean() * (b_n ** 2).mean()) + 1e-6))
            main_diag = max(0.0, main_diag)
        else:
            main_diag = 0.0
        # Anti-diagonal: arr[i,j] vs arr[h-1-j, w-1-i]
        flipped = np.flip(arr, axis=(0, 1))
        if flipped.shape == arr.shape:
            a_n = (arr - arr.mean()) / (arr.std() + 1e-6)
            b_n = (flipped - flipped.mean()) / (flipped.std() + 1e-6)
            anti_diag = float((a_n * b_n).mean() /
                              (np.sqrt((a_n ** 2).mean() * (b_n ** 2).mean()) + 1e-6))
            anti_diag = max(0.0, anti_diag)
        else:
            anti_diag = 0.0
        return {
            "main_diagonal_symmetry": main_diag,
            "anti_diagonal_symmetry": anti_diag,
        }
    except Exception:
        return {"main_diagonal_symmetry": 0.0, "anti_diagonal_symmetry": 0.0}


def symmetry_features(img: Image.Image) -> dict:
    """Aggregate all symmetry features."""
    feats = {}
    try:
        feats["h_symmetry"] = horizontal_symmetry(img)
    except Exception:
        feats["h_symmetry"] = 0.0
    try:
        feats["v_symmetry"] = vertical_symmetry(img)
    except Exception:
        feats["v_symmetry"] = 0.0
    try:
        feats["point_symmetry"] = point_symmetry(img)
    except Exception:
        feats["point_symmetry"] = 0.0
    try:
        feats.update(diagonal_symmetry(img))
    except Exception:
        feats["main_diagonal_symmetry"] = 0.0
        feats["anti_diagonal_symmetry"] = 0.0
    # Composite symmetry score (max of all)
    syms = [feats["h_symmetry"], feats["v_symmetry"], feats["point_symmetry"],
            feats["main_diagonal_symmetry"], feats["anti_diagonal_symmetry"]]
    feats["overall_symmetry"] = float(max(syms))
    feats["is_symmetric"] = float(feats["overall_symmetry"] > 0.7)
    return feats
