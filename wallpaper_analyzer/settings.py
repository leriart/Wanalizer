import os
import json

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_DIR, ".wallpaper_analyzer.json")
TAGS_PATH = os.path.join(PROJECT_DIR, "tags.json")

SETTINGS_DEFAULTS = {
    "theme": "dark",
    "organize_mode": "lowlevel",
    "source_dir": "",
    "dest_dir": "WP",
    "dedupe": True,
    "dedupe_perceptual": True,
    "dedupe_min_tier": "reencode",
    "video_ffmpeg": True,
    "quality_gate": True,
    "quality_min": 0.0,
    "by_resolution": False,
    "resolution_bins": [
        {"name": "SD", "max_pixels": 480000},
        {"name": "720p", "max_pixels": 921600},
        {"name": "1080p", "max_pixels": 2073600},
        {"name": "1440p", "max_pixels": 3686400},
        {"name": "4K", "max_pixels": 8294400},
        {"name": "5K+", "max_pixels": 999999999},
    ],
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llava:7b",
    "ollama_nsfw_enabled": True,
    "ollama_nsfw_threshold": 0.70,
    "ollama_nsfw_use": True,
    "ollama_nsfw_uncertain_floor": 0.40,
    "ollama_nsfw_uncertain_ceiling": 0.65,
    "ollama_describe_enabled": False,
    "ollama_characters_enabled": False,
    "ollama_classify_enabled": False,
    "ollama_classify_method": "tags",
    "ollama_tags_enabled": True,
    "ollama_timeout": 60,
    "ollama_description_max_tokens": 80,
    "ollama_image_max_size": 512,
    "clip_model": "ViT-B/32",
    "clip_nsfw_threshold": 0.5,
    "nsfw_threshold": 0.5,
    "nsfw_default_tags": [
        "nsfw", "figure", "human", "skin", "portrait", "person", "body",
    ],
    "laplacian_min": 80,
    "lowlevel_texture_weight": 0.25,
    "lowlevel_edge_weight": 0.20,
    "lowlevel_shape_weight": 0.25,
    "lowlevel_color_weight": 0.30,
    "active_model": "",
    "ai_backend": "lowlevel",
    # Vision+Text regeneration pipeline (Categories page)
    "regen_vision_model": "llava-phi3:3.8b",
    "regen_text_model": "qwen2.5:3b",
    "regen_samples": 3,
    "regen_vision_timeout": 240,
    "regen_text_timeout": 120,
}


def resolve_dest_dir(settings=None):
    """Resolve the destination directory path from settings.
    
    Returns an absolute path. If dest_dir is relative, it's relative to PROJECT_DIR.
    """
    if settings is None:
        settings = load_settings()
    dest = settings.get("dest_dir", "WP")
    if os.path.isabs(dest):
        return dest
    return os.path.join(PROJECT_DIR, dest)


def resolve_hash_cache_path(settings=None):
    """Hash cache lives inside the destination directory."""
    dest = resolve_dest_dir(settings)
    return os.path.join(dest, ".wallpaper_analyzer_hashes.json")


def load_settings():
    """Load settings from disk merged with environment overrides.

    Order of precedence (highest first):
      1. Environment variables (`WANALIZER_*`, plus a few documented
         shortcuts like `OLLAMA_URL`).
      2. `.wallpaper_analyzer.json` in the project directory.
      3. Built-in defaults.
    """
    defaults = SETTINGS_DEFAULTS.copy()

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                defaults.update(data)
        except Exception:
            pass

    env_overrides = {
        "organize_mode":    os.environ.get("WANALIZER_MODE"),
        "dest_dir":         os.environ.get("WANALIZER_DEST"),
        "ollama_url":       os.environ.get("OLLAMA_URL")
                            or os.environ.get("WANALIZER_OLLAMA_URL"),
        "ollama_model":     os.environ.get("OLLAMA_MODEL")
                            or os.environ.get("WANALIZER_OLLAMA_MODEL"),
        "clip_model":       os.environ.get("WANALIZER_CLIP_MODEL"),
        "theme":            os.environ.get("WANALIZER_THEME"),
    }
    for key, value in env_overrides.items():
        if value:
            defaults[key] = value

    return defaults


def save_settings(settings):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
            json.dump(settings, fp, indent=2)
    except Exception as exc:
        print(f"  [WARN] Could not save settings: {exc}")
