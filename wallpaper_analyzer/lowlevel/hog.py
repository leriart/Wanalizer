"""
Histogram of Oriented Gradients (HOG) features.
Used for object detection and style classification.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def hog_descriptor(img: Image.Image, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), orientations=9) -> np.ndarray:
    """Compute HOG feature vector."""
    if not HAS_CV2:
        return _hog_pillow_fallback(img, orientations)

    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    # Resize to standard size for consistency
    arr = cv2.resize(arr, (128, 256))

    hog = cv2.HOGDescriptor(
        _winSize=(128, 256),
        _blockSize=(cells_per_block[0] * pixels_per_cell[0],
                     cells_per_block[1] * pixels_per_cell[1]),
        _blockStride=(pixels_per_cell[0], pixels_per_cell[1]),
        _cellSize=pixels_per_cell,
        _nbins=orientations,
    )
    features = hog.compute(arr)
    if features is not None:
        return features.flatten()
    return np.zeros(orientations * 105)  # typical HOG size


def _hog_pillow_fallback(img: Image.Image, orientations=9) -> np.ndarray:
    """Simplified HOG using numpy only."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = arr.shape
    arr = arr[::2, ::2]  # downsample

    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]

    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang = (ang + 180) % 180

    cell_h, cell_w = arr.shape[0] // 8, arr.shape[1] // 8
    hist = np.zeros((cell_h, cell_w, orientations))

    for i in range(cell_h):
        for j in range(cell_w):
            block_mag = mag[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            block_ang = ang[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            for o in range(orientations):
                lower = o * (180 / orientations)
                upper = (o + 1) * (180 / orientations)
                mask = (block_ang >= lower) & (block_ang < upper)
                hist[i, j, o] = block_mag[mask].sum()

    # Normalize blocks
    for i in range(cell_h - 1):
        for j in range(cell_w - 1):
            block = hist[i:i + 2, j:j + 2].flatten()
            norm = np.sqrt(block @ block + 1e-6)
            if norm > 0:
                hist[i:i + 2, j:j + 2] /= norm

    return hist.flatten()


def hog_features(img: Image.Image) -> dict:
    """Extract HOG-based features for classification."""
    hog_vec = hog_descriptor(img)

    if len(hog_vec) == 0:
        return {"hog_mean": 0.0, "hog_std": 0.0, "hog_energy": 0.0}

    return {
        "hog_mean": float(hog_vec.mean()),
        "hog_std": float(hog_vec.std()),
        "hog_energy": float((hog_vec ** 2).sum()),
        "hog_max": float(hog_vec.max()),
        "hog_sparsity": float(np.sum(hog_vec > hog_vec.mean() * 0.1) / max(len(hog_vec), 1)),
    }
