import numpy as np
from PIL import Image, ImageFilter

def laplacian_variance(img: Image.Image) -> float:
    try:
        g = img.convert("L").copy()
        g.thumbnail((512, 512), Image.LANCZOS)
        arr = np.asarray(g, dtype=np.float32)
        lap = np.asarray(g.filter(ImageFilter.Kernel(
            (3, 3),
            [0, 1, 0, 1, -4, 1, 0, 1, 0],
            scale=1, offset=128,
        )), dtype=np.float32)
        return float(lap.var())
    except Exception:
        return 0.0

def heuristic_aesthetic_score(img: Image.Image, sharpness=None) -> float:
    try:
        w, h = img.size
        pixels = max(w * h, 1)
        res_score = min(pixels / (3840 * 2160), 1.0)
        if sharpness is None:
            sharpness = laplacian_variance(img)
        sharp_score = min(sharpness / 600.0, 1.0)
        g = img.convert("L").copy()
        g.thumbnail((128, 128), Image.LANCZOS)
        arr = np.asarray(g, dtype=np.float32)
        mean_luma = float(arr.mean()) / 255.0
        luma_score = max(0.0, min(1.0 - abs(mean_luma - 0.55) * 1.6, 1.0))
        small = img.convert("RGB").copy()
        small.thumbnail((128, 128), Image.LANCZOS)
        rgb = np.asarray(small, dtype=np.float32) / 255.0
        r, gr, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx = rgb.max(axis=-1)
        mn = rgb.min(axis=-1)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        mean_sat = float(sat.mean())
        sat_score = max(0.0, min(1.0 - abs(mean_sat - 0.30) * 1.5, 1.0))
        return 0.35 * res_score + 0.30 * sharp_score + 0.20 * luma_score + 0.15 * sat_score
    except Exception:
        return 0.5
