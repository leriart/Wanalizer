"""
Fourier/frequency domain analysis: FFT-based features, frequency distribution,
and periodic pattern detection.
"""
import numpy as np
from PIL import Image


def fft_magnitude_spectrum(img: Image.Image) -> np.ndarray:
    """Compute 2D FFT magnitude spectrum (log-scaled, centered)."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude


def radial_frequency_distribution(img: Image.Image, rings=10) -> np.ndarray:
    """Compute energy distribution across radial frequency bands."""
    mag = fft_magnitude_spectrum(img)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    ring_width = max_radius / rings
    distribution = np.zeros(rings)
    for i in range(rings):
        mask = (dist >= i * ring_width) & (dist < (i + 1) * ring_width)
        distribution[i] = mag[mask].sum()
    total = distribution.sum()
    if total > 0:
        distribution /= total
    return distribution


def angular_frequency_distribution(img: Image.Image, sectors=8) -> np.ndarray:
    """Energy distribution across angular sectors (directional frequency)."""
    mag = fft_magnitude_spectrum(img)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    angles = np.arctan2(Y - cy, X - cx) + np.pi
    sector_width = 2 * np.pi / sectors
    distribution = np.zeros(sectors)
    for i in range(sectors):
        mask = (angles >= i * sector_width) & (angles < (i + 1) * sector_width)
        distribution[i] = mag[mask].sum()
    total = distribution.sum()
    if total > 0:
        distribution /= total
    return distribution


def frequency_features(img: Image.Image) -> dict:
    """Extract frequency-domain features."""
    radial = radial_frequency_distribution(img)
    angular = angular_frequency_distribution(img)
    mag = fft_magnitude_spectrum(img)

    # Low-frequency vs high-frequency energy ratio
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = min(cy, cx)
    low_freq = mag[dist < max_r * 0.2].sum()
    high_freq = mag[dist > max_r * 0.6].sum()

    return {
        "low_freq_energy": float(low_freq),
        "high_freq_energy": float(high_freq),
        "high_low_ratio": float(high_freq / max(low_freq, 1e-6)),
        "dominant_radial_band": int(np.argmax(radial)),
        "dominant_angular_sector": int(np.argmax(angular)),
        "frequency_entropy": float(-(radial * np.log(radial + 1e-10)).sum()),
    }
