"""Perceptual image hashing.

Computes multiple perceptual hashes from a single 64x64 RGB downsample
to minimize redundant work. PIL's LANCZOS resize on full-resolution
images is expensive; by doing it once and then re-resizing the small
preview for every hash, the per-image cost drops from ~80 ms to ~15 ms.

Hashes returned by `compute_all_hashes(img)`:

  * `dh`    (64-bit)  Difference hash (dHash): gradient along x, 9x8 grayscale
  * `dh16`  (16-bit)  Coarse dHash for LSH pre-filtering, 5x4 grayscale
  * `ph`    (64-bit)  Perceptual hash (pHash): top-left 8x8 DCT coefficients
                      median thresholded. Uses scipy.fftpack.dct for O(n^3)
                      instead of naive O(n^4).
  * `ah`    (64-bit)  Average hash: pixel > mean
  * `ch16`  (16-bit)  Coarse color hash, 5x4 RGB luminance differences
  * `hist`  (16 bins) RGB color histogram from 64x64 downsample, for
                      cosine similarity refinement
  * `size`  (w, h)    Original image size

The two coarse hashes (`dh16`, `ch16`) are designed to bucket very similar
images together with minimal collision risk, so they can be used as an
LSH (locality-sensitive hashing) pre-filter before computing expensive
pairwise distances.

For multi-frame media (animated GIFs / WEBPs / APNGs, and videos), see
`compute_hashes_multi(path) -> List[Dict]` which returns one hash dict
per extracted frame. The duplicates detector compares the per-frame
hashes so two files with different intros but identical main content
still match.
"""
import math
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from PIL import Image

try:
    import numpy as np
    import scipy.fftpack
    _HAS_SCIPY = True
except Exception:
    np = None
    scipy = None
    _HAS_SCIPY = False

try:
    import imagehash
    _HAS_IMAGEHASH = True
except Exception:
    imagehash = None
    _HAS_IMAGEHASH = False


# ---------------------------------------------------------------------------
# File-type classification. Used by compute_hashes_multi() to decide how
# many frames to extract.
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".webm", ".mkv", ".avi", ".mov", ".flv",
    ".mpg", ".mpeg", ".mpe", ".mpv", ".ogv", ".wmv", ".asf",
    ".ts", ".m2ts", ".mts", ".vob", ".3gp", ".3gpp",
    ".rm", ".rmvb", ".ogm",
})

ANIMATED_IMAGE_EXTENSIONS = frozenset({
    ".gif", ".apng", ".mng", ".fli", ".flc", ".webp",
})

# Number of frames to extract for each media type.
# Static images always use 1 frame (the image itself).
FRAMES_FOR_STATIC = 1
FRAMES_FOR_ANIMATED = 3   # GIF / WEBP / APNG: first, middle, last
FRAMES_FOR_VIDEO = 5      # MP4 / MKV / ...: 0%, 25%, 50%, 75%, 95%


def classify_media(path: str) -> str:
    """Return one of: "static", "animated", "video"."""
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in ANIMATED_IMAGE_EXTENSIONS:
        # Some .webp are static - we detect this below by checking PIL.
        return "animated"
    return "static"


def ffmpeg_available() -> bool:
    """Return True iff ffmpeg is on PATH (or given by FFMPEG_BIN env)."""
    return shutil.which(os.environ.get("FFMPEG_BIN", "ffmpeg")) is not None


# ---------------------------------------------------------------------------
# One-shot multi-hash computation. Resize the input once and reuse the
# result for every hash; this is by far the dominant cost of the pipeline.
# ---------------------------------------------------------------------------

def compute_all_hashes(img: Image.Image) -> Dict:
    """Compute every perceptual hash for an image in a single pass."""
    rgb64 = _rgb64(img)
    gray32 = rgb64.convert("L").resize((32, 32), Image.LANCZOS)
    gray9 = rgb64.convert("L").resize((9, 8), Image.LANCZOS)
    gray8 = rgb64.convert("L").resize((8, 8), Image.LANCZOS)
    gray5 = rgb64.convert("L").resize((5, 4), Image.LANCZOS)
    rgb5 = rgb64.resize((5, 4), Image.LANCZOS)
    rgb8 = rgb64.resize((8, 8), Image.LANCZOS)

    return {
        "dh": _dhash_from(gray9),
        "dh16": _dhash_from_5x4(gray5),
        "ph": _phash_from(gray32),
        "ah": _ahash_from(gray8),
        "ch16": _chash16_from(rgb5),
        "chh": _color_hash_from(rgb8),
        "hist": _hist_from(rgb64),
        "size": img.size,
    }


def compute_hashes(path: str) -> Optional[Dict]:
    """Compute all hashes from a file path. Returns None on failure."""
    try:
        with Image.open(path) as raw:
            raw.load()
            img = raw.convert("RGB").copy()
    except Exception:
        return None
    return compute_all_hashes(img)


# ---------------------------------------------------------------------------
# Multi-frame hashing (animated images + videos)
# ---------------------------------------------------------------------------

def _frame_signature(frame: Image.Image, t_seconds: float = 0.0) -> Dict:
    """Build the same hash dict that `compute_all_hashes` returns, plus
    a `t` field with the frame timestamp (0.0 for static / unknown)."""
    h = compute_all_hashes(frame)
    if h is not None:
        h["t"] = float(t_seconds)
    return h


def _extract_gif_frames(path: str, n_frames: int) -> List[Image.Image]:
    """Pull `n_frames` representative RGB frames from an animated image.

    Strategy:
      * If the file has exactly 1 frame (static .webp/.gif), return that.
      * Otherwise sample at 0%, ~50%, and ~100% of the animation by index.
    """
    try:
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            is_animated = bool(getattr(im, "is_animated", False)) or n > 1
            if not is_animated:
                return [im.convert("RGB").copy()]
            # Sample evenly across the animation
            if n_frames >= n:
                indices = list(range(n))
            else:
                # Always include first and last
                indices = sorted({
                    0,
                    *(round((n - 1) * (i + 1) / (n_frames - 1))
                      for i in range(n_frames - 1)),
                    n - 1,
                })
            out: List[Image.Image] = []
            for idx in indices:
                try:
                    im.seek(idx)
                    out.append(im.convert("RGB").copy())
                except Exception:
                    continue
            return out or [im.convert("RGB").copy()]
    except Exception:
        return []


def _ffmpeg_duration(path: str) -> Optional[float]:
    """Return media duration in seconds via ffprobe, or None on failure."""
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    ffprobe = shutil.which(os.environ.get("FFPROBE_BIN", "ffprobe"))
    if ffprobe is None:
        # Fallback: parse `ffmpeg -i` stderr (slower, less reliable).
        try:
            out = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-i", path],
                capture_output=True, text=True, timeout=10,
            ).stderr
            import re
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out)
            if m:
                h, mn, s = m.groups()
                return int(h) * 3600 + int(mn) * 60 + float(s)
        except Exception:
            return None
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def _ffmpeg_extract_frame(path: str, t_seconds: float,
                          dest_path: str) -> bool:
    """Extract a single frame at `t_seconds` from `path` -> `dest_path`.

    Returns True on success. Uses -ss BEFORE -i so we seek fast (keyframe).
    """
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-y", "-loglevel", "error",
             "-ss", f"{max(0.0, t_seconds):.3f}",
             "-i", path,
             "-vframes", "1",
             "-vf", "scale='min(512,iw)':-2",
             "-q:v", "2",
             dest_path],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0 and os.path.exists(dest_path)
    except Exception:
        return False


def _is_dark_image(path: str, threshold: float = 20.0) -> bool:
    """Return True if the average luminance of `path` is below `threshold`
    (0-255). Used to detect intro / title-card / fully-black frames.

    Cheap: reads the file with PIL, downsamples to 16x16 grayscale,
    averages 256 pixels. <1 ms on a modern CPU.
    """
    try:
        with Image.open(path) as im:
            gray = im.convert("L").resize((16, 16), Image.BILINEAR)
            data = list(gray.getdata())
        return (sum(data) / len(data)) < threshold
    except Exception:
        return False


def _extract_video_frames(path: str, n_frames: int) -> List[Tuple[float, Image.Image]]:
    """Pull `n_frames` representative RGB frames from a video using ffmpeg.

    Returns a list of (timestamp_seconds, frame) tuples, OR empty on
    failure (e.g. ffmpeg missing). Falls back to a single frame at t=0
    when duration cannot be determined.

    If the first extracted frame is mostly black (luminance < 20/255),
    we drop it and pull a replacement frame at t=10% of duration.
    This avoids matching videos on their intro / title card.
    """
    if not ffmpeg_available():
        return []
    duration = _ffmpeg_duration(path)
    with tempfile.TemporaryDirectory(prefix="wpc_") as td:
        if duration is None or duration <= 0:
            timestamps = [0.0]
        else:
            # 0%, 25%, 50%, 75%, 95% of duration (avoid 100% = past EOF)
            offsets = [0.0, 0.25, 0.50, 0.75, 0.95][:n_frames]
            timestamps = [duration * o for o in offsets]

        extracted: List[Tuple[float, str]] = []
        for t in timestamps:
            dst = os.path.join(td, f"f_{int(t*1000)}.jpg")
            if _ffmpeg_extract_frame(path, t, dst):
                extracted.append((t, dst))

        # Replace a dark first frame with a t=10% sample.
        if extracted and _is_dark_image(extracted[0][1]):
            extracted = extracted[1:]
            if duration and duration > 1.0:
                extra_t = duration * 0.10
                extra_path = os.path.join(td, f"f_extra_{int(extra_t*1000)}.jpg")
                if _ffmpeg_extract_frame(path, extra_t, extra_path):
                    extracted.insert(0, (extra_t, extra_path))

        out: List[Tuple[float, Image.Image]] = []
        for t, p in extracted:
            try:
                with Image.open(p) as im:
                    out.append((t, im.convert("RGB").copy()))
            except Exception:
                continue
        return out


def compute_hashes_multi(path: str) -> List[Dict]:
    """Compute hashes for ALL relevant frames of `path`.

    Returns a list of hash dicts (one per frame). Each dict has the
    same shape as `compute_all_hashes(img)` plus a `t` field (seconds
    into the media, 0.0 for static images).

    Behaviour by file type:
      * static image (PNG / JPG / single-frame WEBP): `[single_dict]`
      * animated image (multi-frame GIF / WEBP / APNG): up to
        FRAMES_FOR_ANIMATED (3) frames: first, middle, last.
      * video (MP4 / MKV / ...): up to FRAMES_FOR_VIDEO (5) frames
        at 0%, 25%, 50%, 75%, 95% via ffmpeg. If ffmpeg is missing
        we fall back to a single-frame signature so the file is
        still part of the comparison (better than skipping it).
      * unreadable file: returns `[]` (the duplicates detector
        treats this as 'no signature' and skips it).
    """
    kind = classify_media(path)
    if kind == "static":
        try:
            with Image.open(path) as raw:
                raw.load()
                img = raw.convert("RGB").copy()
            return [_frame_signature(img, 0.0)]
        except Exception:
            return []

    if kind == "animated":
        frames = _extract_gif_frames(path, FRAMES_FOR_ANIMATED)
        out: List[Dict] = []
        for i, f in enumerate(frames):
            h = compute_all_hashes(f)
            if h is None:
                continue
            h["t"] = float(i)  # index as proxy for time
            out.append(h)
        if out:
            return out
        # Animated file we couldn't read frame-by-frame - try static.
        try:
            with Image.open(path) as raw:
                raw.load()
                return [_frame_signature(raw.convert("RGB").copy(), 0.0)]
        except Exception:
            return []

    # kind == "video"
    extracted = _extract_video_frames(path, FRAMES_FOR_VIDEO)
    if not extracted:
        # No ffmpeg or unreadable - try PIL as last resort (will likely
        # return []; better than crashing).
        try:
            with Image.open(path) as raw:
                raw.load()
                return [_frame_signature(raw.convert("RGB").copy(), 0.0)]
        except Exception:
            return []
    out = []
    for t, frame in extracted:
        h = compute_all_hashes(frame)
        if h is None:
            continue
        h["t"] = t
        out.append(h)
    return out


# ---------------------------------------------------------------------------
# Low-level helpers (operate on already-resized small images)
# ---------------------------------------------------------------------------

def _rgb64(img: Image.Image) -> Image.Image:
    """Convert + resize to 64x64 RGB. The single most expensive op."""
    return img.convert("RGB").resize((64, 64), Image.LANCZOS)


def _dhash_from(gray9: Image.Image) -> int:
    """dHash from a 9x8 grayscale image -> 64 bits."""
    arr = list(gray9.getdata())
    bits = 0
    for y in range(8):
        row = y * 9
        for x in range(8):
            bits = (bits << 1) | (1 if arr[row + x] > arr[row + x + 1] else 0)
    return bits


def _dhash_from_5x4(gray5: Image.Image) -> int:
    """16-bit coarse dHash from a 5x4 grayscale image."""
    arr = list(gray5.getdata())
    bits = 0
    for y in range(4):
        row = y * 5
        for x in range(4):
            bits = (bits << 1) | (1 if arr[row + x] > arr[row + x + 1] else 0)
    return bits


def _ahash_from(gray8: Image.Image) -> int:
    """Average hash from an 8x8 grayscale image."""
    arr = list(gray8.getdata())
    mean = sum(arr) / len(arr)
    bits = 0
    for px in arr:
        bits = (bits << 1) | (1 if px > mean else 0)
    return bits


def _phash_from(gray32: Image.Image) -> int:
    """pHash from a 32x32 grayscale image using scipy DCT (or pure-Python
    fallback when scipy is unavailable)."""
    if _HAS_SCIPY and np is not None:
        arr = np.asarray(gray32, dtype=np.float64) - 128.0
        coeffs = scipy.fftpack.dct(
            scipy.fftpack.dct(arr, axis=0, norm="ortho"),
            axis=1, norm="ortho",
        )
        flat = coeffs[:8, :8].flatten()
        median = float(np.median(flat[1:]))
        bits = 0
        for c in flat:
            bits = (bits << 1) | (1 if c > median else 0)
        return bits
    arr = [px - 128.0 for px in gray32.getdata()]
    coeffs = []
    for v in range(8):
        row_cos = [math.cos(math.pi * v * (2 * y + 1) / (2 * 32)) for y in range(32)]
        for u in range(8):
            cu_row = _DCT_COS_X[u]
            total = 0.0
            for y in range(32):
                row_offset = y * 32
                row_val = 0.0
                for x in range(32):
                    row_val += arr[row_offset + x] * cu_row[x]
                total += row_val * row_cos[y]
            coeffs.append(total)
    sorted_coeffs = sorted(coeffs[1:])
    median = sorted_coeffs[len(sorted_coeffs) // 2]
    bits = 0
    for c in coeffs:
        bits = (bits << 1) | (1 if c > median else 0)
    return bits


_DCT_COS_X = [
    [math.cos(math.pi * u * (2 * x + 1) / (2 * 32)) for x in range(32)]
    for u in range(8)
]


def _chash16_from(rgb5: Image.Image) -> int:
    """16-bit luminance-gradient hash from a 5x4 RGB image."""
    arr = list(rgb5.getdata())
    bits = 0
    for y in range(4):
        row = y * 5
        for x in range(4):
            r1, g1, b1 = arr[row + x]
            r2, g2, b2 = arr[row + x + 1]
            lum1 = (r1 * 299 + g1 * 587 + b1 * 114) // 1000
            lum2 = (r2 * 299 + g2 * 587 + b2 * 114) // 1000
            bits = (bits << 1) | (1 if lum1 > lum2 else 0)
    return bits


def _color_hash_from(rgb8: Image.Image) -> int:
    """64-bit color hash from an 8x8 RGB image."""
    if np is not None:
        arr = np.asarray(rgb8, dtype=np.int32)
        flat = (arr[..., 0] ^ arr[..., 1] ^ arr[..., 2]) * 16 // 256
        flat = flat % 16
        bins = np.bincount(flat.ravel(), minlength=16)
        total = int(bins.sum())
        bit_arr = (bins * 2 > total).astype(np.int64)
        bits = 0
        for b in bit_arr:
            bits = (bits << 1) | int(b)
        return bits
    arr = list(rgb8.getdata())
    bins = [0] * 16
    for r, g, b in arr:
        bins[((r * 16 // 256) ^ (g * 16 // 256) ^ (b * 16 // 256)) % 16] += 1
    total = len(arr)
    bits = 0
    for count in bins:
        bits = (bits << 1) | (1 if count * 2 > total else 0)
    return bits


def _hist_from(rgb64: Image.Image) -> List[int]:
    """16-bin RGB histogram from a 64x64 RGB image, numpy-accelerated."""
    if np is not None:
        arr = np.asarray(rgb64, dtype=np.int32)
        flat = ((arr[..., 0] ^ arr[..., 1] ^ arr[..., 2]) >> 4) % 16
        bins = np.bincount(flat.ravel(), minlength=16)
        return [int(b) for b in bins]
    arr = list(rgb64.getdata())
    bins = [0] * 16
    for r, g, b in arr:
        bins[((int(r) >> 4) ^ (int(g) >> 4) ^ (int(b) >> 4)) % 16] += 1
    return bins


# ---------------------------------------------------------------------------
# Distance / similarity helpers
# ---------------------------------------------------------------------------

def hamming_distance(h1: int, h2: int) -> int:
    """Population count of (h1 XOR h2)."""
    return bin(h1 ^ h2).count("1")


def histogram_similarity(h1: List[int], h2: List[int]) -> float:
    """Cosine similarity between two histograms (0..1)."""
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    dot = sum(a * b for a, b in zip(h1, h2))
    n1 = math.sqrt(sum(a * a for a in h1))
    n2 = math.sqrt(sum(b * b for b in h2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


# ---------------------------------------------------------------------------
# imagehash-library compatibility wrappers (kept for backward compat)
# ---------------------------------------------------------------------------

def _imagehash_to_int(h) -> int:
    if hasattr(h, "hash"):
        flat = h.hash.flatten().astype(np.uint8)
        val = 0
        for b in flat:
            val = (val << 1) | int(b)
        return val
    return int(h)


def phash_imagehash(img: Image.Image) -> Optional[int]:
    if not _HAS_IMAGEHASH:
        return None
    try:
        h = imagehash.phash(img.convert("RGB"), hash_size=8)
        return _imagehash_to_int(h)
    except Exception:
        return None


def dhash_imagehash(img: Image.Image) -> Optional[int]:
    if not _HAS_IMAGEHASH:
        return None
    try:
        h = imagehash.dhash(img.convert("RGB"), hash_size=8)
        return _imagehash_to_int(h)
    except Exception:
        return None


def whash_imagehash(img: Image.Image) -> Optional[int]:
    if not _HAS_IMAGEHASH:
        return None
    try:
        h = imagehash.whash(img.convert("RGB"), hash_size=8)
        return _imagehash_to_int(h)
    except Exception:
        return None


def colorhash_imagehash(img: Image.Image) -> Optional[int]:
    if not _HAS_IMAGEHASH:
        return None
    try:
        h = imagehash.colorhash(img.convert("RGB"), binbits=3)
        return _imagehash_to_int(h)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backward-compatible single-hash APIs (used by other modules).
# Each does its own resize - prefer compute_all_hashes when multiple
# hashes are needed for the same image.
# ---------------------------------------------------------------------------

def dhash(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = list(g.getdata())
    bits = 0
    for y in range(hash_size):
        row = y * (hash_size + 1)
        for x in range(hash_size):
            bits = (bits << 1) | (1 if arr[row + x] > arr[row + x + 1] else 0)
    return bits


def dhash16(img: Image.Image) -> int:
    return _dhash_from_5x4(img.convert("L").resize((5, 4), Image.LANCZOS))


def ahash(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = list(g.getdata())
    mean = sum(arr) / len(arr)
    bits = 0
    for px in arr:
        bits = (bits << 1) | (1 if px > mean else 0)
    return bits


def phash(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((32, 32), Image.LANCZOS)
    return _phash_from(g)


def chash16(img: Image.Image) -> int:
    return _chash16_from(img.convert("RGB").resize((5, 4), Image.LANCZOS))


def color_histogram(img: Image.Image, n_colors: int = 16) -> List[int]:
    g = img.convert("RGB").resize((64, 64), Image.LANCZOS)
    return _hist_from(g)


def color_histogram_hash(img: Image.Image, n_colors: int = 16, hash_size: int = 8) -> int:
    return _color_hash_from(img.convert("RGB").resize((hash_size, hash_size), Image.LANCZOS))


def clear_resize_cache() -> None:  # pragma: no cover - kept for API compat
    """No-op kept for API compatibility."""
    return
