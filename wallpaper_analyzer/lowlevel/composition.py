"""
Composition analysis: rule of thirds, depth, diagonal, saliency, etc.

Wallpapers have distinct composition patterns: centered, thirds, diagonal,
symmetric, asymmetric. These are strong signals for categorization.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def rule_of_thirds_score(img: Image.Image) -> dict:
    """Measure how strongly the image follows rule of thirds.

    High score = subject/interest points align with thirds intersections.
    Low score = centered or random.
    """
    if not HAS_CV2:
        return {"thirds_score": 0.0, "interest_in_thirds": 0.0}
    try:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        h, w = arr.shape
        # Edge/interest map via Sobel
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        # Thirds intersections: (w/3, h/3), (2w/3, h/3), (w/3, 2h/3), (2w/3, 2h/3)
        thirds_x = [w // 3, 2 * w // 3]
        thirds_y = [h // 3, 2 * h // 3]
        # Sample interest at thirds lines
        thirds_interest = 0.0
        n_samples = 0
        radius = min(h, w) // 20
        for tx in thirds_x:
            y_start = max(0, h // 3 - radius)
            y_end = min(h, 2 * h // 3 + radius)
            thirds_interest += float(mag[y_start:y_end, max(0, tx - radius):min(w, tx + radius)].sum())
            n_samples += 1
        for ty in thirds_y:
            x_start = max(0, w // 3 - radius)
            x_end = min(w, 2 * w // 3 + radius)
            thirds_interest += float(mag[max(0, ty - radius):min(h, ty + radius), x_start:x_end].sum())
            n_samples += 1
        # Normalize
        thirds_interest /= max(n_samples, 1)
        total_interest = float(mag.sum())
        # Score: how much of the interest is at the thirds lines
        score = thirds_interest / max(total_interest, 1)
        return {"thirds_score": float(score), "interest_in_thirds": float(thirds_interest)}
    except Exception:
        return {"thirds_score": 0.0, "interest_in_thirds": 0.0}


def centeredness(img: Image.Image) -> dict:
    """How centered the main subject is.

    Returns the offset of the visual center from the geometric center.
    0 = perfectly centered, 1 = at corner.
    """
    if not HAS_CV2:
        return {"centeredness": 0.5, "x_offset": 0, "y_offset": 0}
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        # Brightness-weighted centroid
        total = arr.sum()
        if total == 0:
            return {"centeredness": 0.5, "x_offset": 0, "y_offset": 0}
        # Use high-brightness pixels (likely subject)
        threshold = arr.mean() + arr.std()
        if threshold <= 0:
            return {"centeredness": 0.5, "x_offset": 0, "y_offset": 0}
        bright = arr > threshold
        if bright.sum() == 0:
            bright = arr > arr.mean()
        if bright.sum() == 0:
            return {"centeredness": 0.5, "x_offset": 0, "y_offset": 0}
        ys, xs = np.where(bright)
        if xs.size == 0 or ys.size == 0:
            return {"centeredness": 0.5, "x_offset": 0, "y_offset": 0}
        cx, cy = float(xs.mean()), float(ys.mean())
        # Normalize to 0..1
        nx, ny = cx / w, cy / h
        # Offset from center (0.5, 0.5)
        x_off = abs(nx - 0.5) * 2
        y_off = abs(ny - 0.5) * 2
        # Centeredness: 0 at center, 1 at corner
        centered_val = float(np.sqrt(x_off ** 2 + y_off ** 2) / np.sqrt(2))
        return {
            "centeredness": 1.0 - centered_val,  # invert: 1 = centered
            "x_offset": float(x_off),
            "y_offset": float(y_off),
        }
    except Exception:
        return {"centeredness": 0.5, "x_offset": 0, "y_offset": 0}


def diagonal_strength(img: Image.Image) -> float:
    """Strength of diagonal lines (main diagonal vs anti-diagonal).

    High = strong diagonal composition (often used in dynamic wallpapers).
    """
    if not HAS_CV2:
        return 0.0
    try:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        h, w = arr.shape
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = np.arctan2(gy, gx) * 180 / np.pi
        # 45 degrees and 135 degrees (main and anti-diagonal)
        diag_mask = ((ang > 30) & (ang < 60)) | ((ang > 120) & (ang < 150))
        return float(mag[diag_mask].sum() / max(mag.sum(), 1))
    except Exception:
        return 0.0


def depth_estimate(img: Image.Image) -> dict:
    """Mono depth estimate using defocus and perspective cues.

    Returns a relative depth complexity score and an estimate of
    whether the image has a strong depth gradient (foreground/background).
    """
    if not HAS_CV2:
        return {"depth_score": 0.5, "depth_gradient": 0.0}
    try:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        h, w = arr.shape
        # Compute sharpness gradient: top vs bottom
        top_sharp = float(cv2.Laplacian(arr[: h // 2], cv2.CV_64F).var())
        bottom_sharp = float(cv2.Laplacian(arr[h // 2:], cv2.CV_64F).var())
        # Left vs right
        left_sharp = float(cv2.Laplacian(arr[:, : w // 2], cv2.CV_64F).var())
        right_sharp = float(cv2.Laplacian(arr[:, w // 2:], cv2.CV_64F).var())
        # Vertical and horizontal depth gradients
        v_grad = abs(top_sharp - bottom_sharp) / max(top_sharp + bottom_sharp, 1)
        h_grad = abs(left_sharp - right_sharp) / max(left_sharp + right_sharp, 1)
        # Overall sharpness (low = far away, high = close)
        overall = float(cv2.Laplacian(arr, cv2.CV_64F).var())
        return {
            "depth_score": float(min(overall / 500.0, 1.0)),
            "depth_gradient": float((v_grad + h_grad) / 2),
            "vertical_depth_diff": float(v_grad),
            "horizontal_depth_diff": float(h_grad),
        }
    except Exception:
        return {"depth_score": 0.5, "depth_gradient": 0.0,
                "vertical_depth_diff": 0, "horizontal_depth_diff": 0}


def saliency_estimate(img: Image.Image) -> dict:
    """Estimate visual saliency (where the eye is drawn).

    Uses a simple center-surround + edge-based approach.
    """
    if not HAS_CV2:
        return {"saliency_concentration": 0.5, "saliency_centroid_x": 0.5,
                "saliency_centroid_y": 0.5}
    try:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape
        # Difference from mean (color contrast = saliency proxy)
        mean = arr.reshape(-1, 3).mean(axis=0)
        diff = np.linalg.norm(arr - mean, axis=2)
        # Downsample for speed
        try:
            small = cv2.resize(diff, (w // 4, h // 4))
        except Exception:
            small = diff
        # Saliency map normalized
        if small.max() > 0:
            sal = small / small.max()
        else:
            sal = small
        # Find centroid
        total = sal.sum()
        if total > 0:
            ys, xs = np.mgrid[0:sal.shape[0], 0:sal.shape[1]]
            cy = float((ys * sal).sum() / total) / sal.shape[0]
            cx = float((xs * sal).sum() / total) / sal.shape[1]
        else:
            cx, cy = 0.5, 0.5
        # Concentration: how peaked the saliency is (high = strong subject)
        sorted_sal = np.sort(sal.flatten())[::-1]
        top_10_pct = sorted_sal[:max(1, len(sorted_sal) // 10)].sum()
        concentration = float(top_10_pct / max(sorted_sal.sum(), 1))
        return {
            "saliency_concentration": concentration,
            "saliency_centroid_x": float(cx),
            "saliency_centroid_y": float(cy),
        }
    except Exception:
        return {"saliency_concentration": 0.5, "saliency_centroid_x": 0.5,
                "saliency_centroid_y": 0.5}


def aspect_ratio_features(img: Image.Image) -> dict:
    """Aspect ratio and related geometric features."""
    w, h = img.size
    ar = w / max(h, 1)
    return {
        "aspect_ratio": float(ar),
        "is_landscape": float(ar > 1.0),
        "is_portrait": float(ar < 1.0),
        "is_square": float(0.9 <= ar <= 1.1),
        "is_ultrawide": float(ar > 2.0),
        "is_panoramic": float(ar > 2.5),
    }


def composition_features(img: Image.Image) -> dict:
    """Aggregate all composition features."""
    feats = {}
    try:
        feats.update(rule_of_thirds_score(img))
    except Exception:
        feats["thirds_score"] = 0.0
        feats["interest_in_thirds"] = 0.0
    try:
        feats.update(centeredness(img))
    except Exception:
        feats["centeredness"] = 0.5
        feats["x_offset"] = 0
        feats["y_offset"] = 0
    try:
        feats["diagonal_strength"] = diagonal_strength(img)
    except Exception:
        feats["diagonal_strength"] = 0.0
    try:
        feats.update(depth_estimate(img))
    except Exception:
        feats["depth_score"] = 0.5
        feats["depth_gradient"] = 0.0
    try:
        feats.update(saliency_estimate(img))
    except Exception:
        feats["saliency_concentration"] = 0.5
        feats["saliency_centroid_x"] = 0.5
        feats["saliency_centroid_y"] = 0.5
    try:
        feats.update(aspect_ratio_features(img))
    except Exception:
        feats["aspect_ratio"] = 1.0
    return feats
