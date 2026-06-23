"""
Feature detection and matching: ORB, FAST, corner detection.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def orb_features(img: Image.Image, max_features=500) -> tuple:
    """Extract ORB keypoints and descriptors."""
    if not HAS_CV2:
        return ([], None)
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp, des = orb.detectAndCompute(arr, None)
    return (kp, des)


def fast_corners(img: Image.Image, threshold=30) -> int:
    """Count FAST corner features."""
    if not HAS_CV2:
        return 0
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    kp = fast.detect(arr, None)
    return len(kp)


def good_features_to_track(img: Image.Image, max_corners=200) -> int:
    """Shi-Tomasi corner detection."""
    if not HAS_CV2:
        return 0
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    corners = cv2.goodFeaturesToTrack(arr, max_corners, 0.01, 10)
    if corners is None:
        return 0
    return len(corners)


def feature_features(img: Image.Image) -> dict:
    """Extract feature-based descriptors for classification."""
    kp, des = orb_features(img)

    if not kp:
        return {
            "num_keypoints": 0,
            "keypoint_density": 0.0,
            "mean_response": 0.0,
            "num_corners": 0,
        }

    responses = [k.response for k in kp]
    h, w = img.size[1], img.size[0]
    density = len(kp) / max(h * w, 1)

    # Spatial distribution of keypoints
    xs = np.array([k.pt[0] for k in kp])
    ys = np.array([k.pt[1] for k in kp])
    spatial_std = (float(xs.std()) + float(ys.std())) / 2.0 if len(kp) > 1 else 0.0

    return {
        "num_keypoints": len(kp),
        "keypoint_density": float(density * 10000),
        "mean_response": float(np.mean(responses)),
        "spatial_spread": float(spatial_std / max(np.sqrt(w * h), 1) * 100),
        "num_corners": good_features_to_track(img),
    }
