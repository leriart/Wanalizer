import os
import warnings
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

from .settings import PROJECT_DIR

WALLPAPERS_DIR = PROJECT_DIR

# Wallpapers are often very high resolution (8K, multi-monitor
# panoramas). PIL's default decompression-bomb limit (~89 megapixels)
# fires spurious warnings on legitimate files. Bump it to ~500 MP
# (about 22000x22000) and silence the warning so it doesn't pollute
# the GUI log.
try:
    Image.MAX_IMAGE_PIXELS = 500_000_000
    warnings.filterwarnings(
        "ignore", category=Image.DecompressionBombWarning
    )
except AttributeError:
    pass

STATIC_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".tif", ".ico",
    ".hdr",
    ".ppm", ".pgm", ".pbm",
    ".jp2", ".j2k", ".j2c", ".jpx",
    ".tga", ".icb", ".vda", ".vst",
    ".pcx", ".sgi", ".bw", ".spi", ".dds",
    ".fits", ".fit", ".xbm", ".icns", ".cur",
    ".im", ".msp", ".fpx", ".psp", ".blp",
}

ANIMATED_EXTENSIONS: Set[str] = {
    ".gif", ".apng", ".mng", ".fli", ".flc",
    ".mp4", ".m4v", ".m2v", ".webm", ".mkv",
    ".avi", ".mov", ".flv",
    ".mpg", ".mpeg", ".mpe", ".mpv",
    ".3gp", ".3gpp", ".ogv", ".wmv", ".asf",
    ".ts", ".m2ts", ".mts", ".vob", ".rm", ".rmvb", ".ogm",
}

PLUGIN_EXTENSIONS: Set[str] = {
    ".avif", ".heic", ".heif", ".jxl", ".qoi",
    ".psd", ".exr", ".svg", ".pdf",
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf",
    ".rw2", ".pef", ".nrw", ".srw", ".x3f",
}

MAGIC_SIGNATURES: List[Tuple[int, bytes, str]] = [
    (0, b"\xff\xd8\xff", "JPEG"),
    (0, b"\x89PNG\r\n\x1a\n", "PNG"),
    (0, b"GIF87a", "GIF"),
    (0, b"GIF89a", "GIF"),
    (0, b"BM", "BMP"),
    (0, b"II*\x00", "TIFF-LE"),
    (0, b"MM\x00*", "TIFF-BE"),
    (0, b"\x76\x2f\x31\x01", "EXR"),
    (0, b"8BPS", "PSD"),
    (0, b"8BPB", "PSB"),
    (0, b"DDS ", "DDS"),
    (0, b"qoif", "QOI"),
    (0, b"P1", "PBM-ASCII"),
    (0, b"P2", "PGM-ASCII"),
    (0, b"P3", "PPM-ASCII"),
    (0, b"P4", "PBM"),
    (0, b"P5", "PGM"),
    (0, b"P6", "PPM"),
    (0, b"\x8aMNG\r\n\x1a\n", "MNG"),
    (0, b"\x8bMNG\r\n\x1a\n", "MNG"),
    (0, b"\x8bJNG\r\n\x1a\n", "JNG"),
    (4, b"\x11\xaf", "FLI"),
    (4, b"\x12\xaf", "FLC"),
    (8, b"VP8 ", "WEBP"),
    (8, b"VP8L", "WEBP"),
    (8, b"VP8X", "WEBP"),
    (4, b"ftypheic", "HEIC"),
    (4, b"ftypheix", "HEIC"),
    (4, b"ftypheim", "HEIC"),
    (4, b"ftypheis", "HEIC"),
    (4, b"ftypmif1", "HEIF"),
    (4, b"ftypmsf1", "HEIF"),
    (4, b"ftypavif", "AVIF"),
    (4, b"ftypavis", "AVIF"),
    (4, b"ftypjxl", "JPEG-XL"),
    (4, b"ftypqt  ", "MOV"),
    (4, b"ftypmp42", "MP4"),
    (4, b"ftypmp41", "MP4"),
    (4, b"ftypisom", "MP4"),
    (4, b"ftypM4V ", "M4V"),
    (4, b"ftypM4A ", "M4A"),
    (4, b"ftypdash", "DASH"),
    (0, b"\x1a\x45\xdf\xa3", "WEBM/MKV"),
    (8, b"AVI ", "AVI"),
    (0, b"FLV\x01", "FLV"),
    (0, b"FWS", "SWF"),
    (0, b"CWS", "SWF"),
    (0, b"\x00\x00\x01\xba", "MPEG-PS"),
    (0, b"\x00\x00\x01\xb3", "MPEG-Video"),
    (0, b"OggS", "OGG"),
    (0, b"\x47", "MPEG-TS"),
]

def detect_format(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as fp:
            head = fp.read(32)
    except Exception:
        return None
    if not head:
        return None
    if head[:4] == b"RIFF" and len(head) >= 12:
        kind = head[8:12]
        if kind == b"WEBP": return "WEBP"
        if kind == b"AVI ": return "AVI"
        if kind == b"WAVE": return "WAV"
    for offset, sig, name in MAGIC_SIGNATURES:
        end = offset + len(sig)
        if len(head) >= end and head[offset:end] == sig:
            return name
    if head[:8] == b"\x00\x00\x00\x0cjP  " or head[:8] == b"\x00\x00\x00\x0cjp2 ":
        return "JPEG-2000"
    if head[:4] == b"\xff\x4f\xff\x51":
        return "JPEG-2000"
    if len(head) >= 12 and head[4:8] == b"JXL ":
        return "JPEG-XL"
    if head[:2] == b"\xff\x0a":
        return "JPEG-XL"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"heim", b"heis"): return "HEIC"
        if brand in (b"mif1", b"msf1"): return "HEIF"
        if brand in (b"avif", b"avis"): return "AVIF"
        if brand[:3] == b"jxl": return "JPEG-XL"
        if brand == b"qt  ": return "MOV"
        mp4_like = {b"isom", b"iso2", b"iso3", b"iso4", b"iso5",
                    b"iso6", b"mp41", b"mp42", b"mp71", b"3gp4",
                    b"3gp5", b"3gp6", b"3g2a", b"f4v ", b"MNV2",
                    b"M4VH", b"dash", b"M4V ", b"M4A ", b"M4B ", b"M4P "}
        if brand in mp4_like: return "MP4"
        return "MP4"
    if head[:4] == b"OggS": return "OGG"
    return None

def supported_image_files(directory: str, include_animations: bool = True) -> List[str]:
    out: List[str] = []
    skip_files = {"README.md", "LICENSE", "requirements.txt", "run.sh", ".gitignore"}
    for name in sorted(os.listdir(directory)):
        if name in skip_files or name.startswith("."):
            continue
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in STATIC_EXTENSIONS:
            out.append(name)
        elif include_animations and ext in ANIMATED_EXTENSIONS:
            out.append(name)
        elif ext in PLUGIN_EXTENSIONS:
            out.append(name)
    return out
