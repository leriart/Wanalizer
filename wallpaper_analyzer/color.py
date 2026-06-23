import math
import struct
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageFilter

_HSV_HUE_TABLE: List[str] = []

def _build_hue_table():
    global _HSV_HUE_TABLE
    ranges = [
        (0, 15, "Red"), (15, 40, "Orange"), (40, 65, "Yellow"),
        (65, 150, "Green"), (150, 200, "Teal-Cyan"),
        (200, 260, "Blue"), (260, 300, "Purple"),
        (300, 345, "Pink-Magenta"), (345, 360, "Red"),
    ]
    table = []
    for start, end, name in ranges:
        table.extend([name] * (end - start))
    _HSV_HUE_TABLE = table

_build_hue_table()

_SIN_TABLE = {i: math.sin(math.radians(i)) for i in range(360)}
_COS_TABLE = {i: math.cos(math.radians(i)) for i in range(360)}

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(rn, gn, bn), min(rn, gn, bn)
    delta = cmax - cmin
    if delta == 0:
        h = 0.0
    elif cmax == rn:
        h = 60.0 * (((gn - bn) / delta) % 6)
    elif cmax == gn:
        h = 60.0 * (((bn - rn) / delta) + 2)
    else:
        h = 60.0 * (((rn - gn) / delta) + 4)
    s = 0.0 if cmax == 0 else (delta / cmax)
    return h, s * 100.0, cmax * 100.0

def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    def linearise(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    rl = linearise(r / 255.0)
    gl = linearise(g / 255.0)
    bl = linearise(b / 255.0)
    x = (rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375) / 0.95047
    y = (rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750) / 1.00000
    z = (rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041) / 1.08883
    def f(t):
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + (16.0 / 116.0)
    fx, fy, fz = f(x), f(y), f(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)

def delta_e76(lab1, lab2):
    return math.sqrt((lab1[0] - lab2[0]) ** 2 +
                     (lab1[1] - lab2[1]) ** 2 +
                     (lab1[2] - lab2[2]) ** 2)

def classify_pixel(h: float, s: float, v: float) -> Dict:
    is_light = v > 85 and s < 20
    is_dark = v < 18 or (v < 25 and s < 25)
    is_pastel = (20 <= s <= 45 and v > 55 and v < 95 and not is_light)
    is_neon = (s > 75 and v > 50 and v < 95)
    is_vaporwave = ((280 <= h < 340 or 170 <= h < 200) and s > 50 and v > 40 and v < 90)
    is_nature = False
    if is_light:
        category = "White"
    elif is_dark:
        category = "Black"
    elif s < 12:
        category = "Gray"
    else:
        idx = int(h) % 360
        category = _HSV_HUE_TABLE[idx]
        if 65 <= h < 150 and 30 < v < 80:
            is_nature = (15 < s < 60)
        elif h < 200:
            is_nature = True
    return {"category": category, "vaporwave": is_vaporwave, "neon": is_neon,
            "pastel": is_pastel, "nature": is_nature}

def extract_color_weights(img: Image.Image) -> Tuple[Dict[str, float], Dict[str, float]]:
    quantized = img.quantize(colors=12, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()[:12 * 3]
    palette_rgb = [(palette[i], palette[i + 1], palette[i + 2])
                   for i in range(0, len(palette), 3)]
    pixel_counts = __import__("collections").Counter(quantized.getdata())
    total_pixels = sum(pixel_counts.values())
    color_weights: Dict[str, float] = {}
    style_scores: Dict[str, float] = {}
    vw_score = neon_score = pastel_score = nature_score = 0.0
    for idx, count in pixel_counts.items():
        if idx >= len(palette_rgb): continue
        r, g, b = palette_rgb[idx]
        hh, ss, vv = rgb_to_hsv(r, g, b)
        result = classify_pixel(hh, ss, vv)
        weight = count / total_pixels
        cat = result["category"]
        color_weights[cat] = color_weights.get(cat, 0.0) + weight
        if result["vaporwave"]: vw_score += weight
        if result["neon"]: neon_score += weight
        if result["pastel"]: pastel_score += weight
        if result["nature"]: nature_score += weight
    style_scores["vaporwave"] = vw_score
    style_scores["neon"] = neon_score
    style_scores["pastel"] = pastel_score
    style_scores["nature"] = nature_score
    return color_weights, style_scores

def dominant_colors_hex(img: Image.Image, n: int = 5) -> List[str]:
    quantized = img.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()[:n * 3]
    return [f"#{palette[i]:02x}{palette[i+1]:02x}{palette[i+2]:02x}"
            for i in range(0, len(palette), 3)]
