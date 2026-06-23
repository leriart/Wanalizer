"""
Advanced quality and sharpness metrics: Tenengrad, BRISQUE-like, perceptual.

Goes beyond basic Laplacian variance to provide multiple quality
indicators and perceptual hashes.
"""
import numpy as np
import hashlib
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def tenengrad_focus(img: Image.Image) -> float:
    """Tenengrad focus measure.

    Variance of Sobel gradient magnitude. Higher = sharper.
    Considered one of the best focus measures in the literature.
    """
    if not HAS_CV2:
        return 0.0
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        # Threshold and compute mean of squared gradient
        return float((mag ** 2).mean())
    except Exception:
        return 0.0


def brisque_like(img: Image.Image) -> dict:
    """BRISQUE-like no-reference quality metric (simplified).

    BRISQUE uses natural scene statistics (NSS) to estimate quality
    without a reference. This is a simplified version that captures
    the core idea: deviation from natural image statistics.
    """
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        # Local mean subtraction (MSCN coefficients)
        if HAS_CV2:
            kernel = cv2.getGaussianKernel(7, 7 / 6)
            kernel = kernel @ kernel.T
            mu = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REFLECT)
            mu_sq = mu * mu
            sigma = cv2.filter2D(arr * arr, -1, kernel, borderType=cv2.BORDER_REFLECT)
            sigma = np.sqrt(np.abs(sigma - mu_sq))
        else:
            # Simple box filter fallback
            from scipy.ndimage import uniform_filter
            mu = uniform_filter(arr, size=7, mode="reflect")
            mu_sq = mu * mu
            sigma_sq = uniform_filter(arr * arr, size=7, mode="reflect")
            sigma = np.sqrt(np.abs(sigma_sq - mu_sq))
        # MSCN: mean-subtracted contrast-normalized
        mscn = (arr - mu) / (sigma + 1.0)
        # Statistics
        mscn_flat = mscn.flatten()
        # Skewness and kurtosis of MSCN
        std = mscn_flat.std()
        if std > 1e-6:
            skew = float(np.mean((mscn_flat - mscn_flat.mean()) ** 3) / std ** 3)
            kurt = float(np.mean((mscn_flat - mscn_flat.mean()) ** 4) / std ** 4) - 3
        else:
            skew, kurt = 0.0, 0.0
        # Natural images have MSCN near-Gaussian; deviations = quality issues
        return {
            "brisque_mscn_var": float(mscn_flat.var()),
            "brisque_mscn_skew": skew,
            "brisque_mscn_kurt": kurt,
            # Quality score: higher kurtosis = less natural = lower quality
            # Use a softer normalization (kurt of natural images is typically 3-8)
            "bris_quality_score": float(max(0, 1.0 - max(0, abs(kurt) - 3) / 30.0)),
        }
    except Exception:
        return {"brisque_mscn_var": 0, "brisque_mscn_skew": 0,
                "brisque_mscn_kurt": 0, "bris_quality_score": 0.5}


def perceptual_hash(img: Image.Image, hash_size=8) -> str:
    """Average hash (aHash) - simple perceptual hash.

    Robust to small changes. Returns hex string.
    """
    try:
        # Resize to small grayscale
        small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        # Compare each pixel to mean
        mean = arr.mean()
        bits = (arr > mean).flatten()
        # Convert to hash
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return f"{h:016x}"
    except Exception:
        return ""


def perceptual_hash_difference(img1: Image.Image, img2: Image.Image) -> int:
    """Hamming distance between two perceptual hashes.

    0 = identical, larger = more different. Useful for near-duplicate detection.
    """
    h1 = perceptual_hash(img1)
    h2 = perceptual_hash(img2)
    if not h1 or not h2:
        return -1
    # Convert hex to int and XOR
    n1, n2 = int(h1, 16), int(h2, 16)
    xor = n1 ^ n2
    # Count bits
    return bin(xor).count("1")


def dct_energy(img: Image.Image) -> dict:
    """DCT-based energy features (JPEG-like).

    High-frequency energy = noise/detail. Low-frequency = smooth regions.
    """
    if not HAS_CV2:
        return {"dct_low_energy": 0, "dct_high_energy": 0, "dct_ratio": 0}
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        # Resize to standard 8x8 blocks
        arr = cv2.resize(arr, (256, 256))
        # Compute DCT in 8x8 blocks
        h, w = arr.shape
        dct = cv2.dct(arr / 255.0)
        # Low-freq (top-left quadrant) vs high-freq (rest)
        cutoff = 16
        low = np.abs(dct[:cutoff, :cutoff]).sum()
        high = np.abs(dct[cutoff:, cutoff:]).sum()
        total = low + high + 1e-6
        return {
            "dct_low_energy": float(low / total),
            "dct_high_energy": float(high / total),
            "dct_ratio": float(low / max(high, 1e-6)),
        }
    except Exception:
        return {"dct_low_energy": 0, "dct_high_energy": 0, "dct_ratio": 0}


def noise_estimate(img: Image.Image) -> float:
    """Estimate image noise level.

    Uses median absolute deviation of Laplacian (Immerkaer's method).
    Higher = noisier image.
    """
    if not HAS_CV2:
        return 0.0
    try:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        conv = cv2.filter2D(arr, -1, M, borderType=cv2.BORDER_REFLECT)
        # Median absolute deviation
        sigma = float(np.median(np.abs(conv)) / 0.6745)
        return sigma
    except Exception:
        return 0.0


def quality_features(img: Image.Image) -> dict:
    """Aggregate all quality features."""
    feats = {}
    try:
        feats["tenengrad"] = tenengrad_focus(img)
    except Exception:
        feats["tenengrad"] = 0.0
    try:
        feats.update(brisque_like(img))
    except Exception:
        pass
    try:
        feats["perceptual_hash"] = perceptual_hash(img)
    except Exception:
        feats["perceptual_hash"] = ""
    try:
        feats.update(dct_energy(img))
    except Exception:
        pass
    try:
        feats["noise_estimate"] = noise_estimate(img)
    except Exception:
        feats["noise_estimate"] = 0.0
    return feats
