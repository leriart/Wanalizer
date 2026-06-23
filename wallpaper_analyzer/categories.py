import os
import json
from typing import Dict, List, Optional, Set

from .settings import resolve_dest_dir, load_settings

SPECIAL_FOLDERS: Set[str] = {"Duplicates", "Low-Quality", "Discarded", "Uncategorized"}
CATEGORIES_DIR: str = ""
CATEGORIES: List[str] = []
_CATEGORY_TAGS: Dict[str, Set[str]] = {}
_CATEGORY_PROMPTS: Dict[str, str] = {}
_PALETTE_WEIGHTS_CACHE: Dict[str, Dict[str, float]] = {}
CATEGORY_ICONS: Dict[str, str] = {}


def _read_category_config(folder_name: str) -> Optional[Dict]:
    path = os.path.join(CATEGORIES_DIR, folder_name, ".category.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_category_config(folder_name: str, config: Dict) -> bool:
    path = os.path.join(CATEGORIES_DIR, folder_name, ".category.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False


def _is_category_folder(folder_name: str) -> bool:
    if folder_name.startswith(".") or folder_name.startswith("_"):
        return False
    if folder_name in SPECIAL_FOLDERS:
        return False
    return os.path.isdir(os.path.join(CATEGORIES_DIR, folder_name))


def _build_category_icons():
    CATEGORY_ICONS.clear()
    for cat in CATEGORIES:
        CATEGORY_ICONS[cat] = f"[{cat[:3].upper():<3}]"
    for name in ["Black", "White", "Gray", "Other", "Multicolor"]:
        if name not in CATEGORY_ICONS:
            CATEGORY_ICONS[name] = f"[{name[:3].upper()}]"


def discover_categories(categories_dir: Optional[str] = None):
    global CATEGORIES, CATEGORIES_DIR, _PALETTE_WEIGHTS_CACHE, _CATEGORY_TAGS, _CATEGORY_PROMPTS
    if categories_dir is not None:
        CATEGORIES_DIR = os.path.abspath(categories_dir)
    else:
        CATEGORIES_DIR = resolve_dest_dir(load_settings())
    CATEGORIES.clear()
    _PALETTE_WEIGHTS_CACHE.clear()
    _CATEGORY_TAGS.clear()
    _CATEGORY_PROMPTS.clear()
    if not os.path.isdir(CATEGORIES_DIR):
        return
    for entry in sorted(os.listdir(CATEGORIES_DIR)):
        if not _is_category_folder(entry):
            continue
        cfg = _read_category_config(entry)
        if cfg is None:
            continue
        CATEGORIES.append(entry)
        pw = cfg.get("palette_weights")
        if pw and isinstance(pw, dict):
            _PALETTE_WEIGHTS_CACHE[entry] = {str(k): float(v) for k, v in pw.items()}
        cat_tags = cfg.get("tags", [])
        if cat_tags:
            _CATEGORY_TAGS[entry] = {str(t).lower() for t in cat_tags}
        prompt = cfg.get("prompt", "")
        if prompt:
            _CATEGORY_PROMPTS[entry] = str(prompt)
    _build_category_icons()


def get_category_tags(category: str) -> Set[str]:
    return _CATEGORY_TAGS.get(category, set())


def get_category_prompt(category: str) -> str:
    return _CATEGORY_PROMPTS.get(category, "")


def get_palette_weights(category: str) -> Dict[str, float]:
    return _PALETTE_WEIGHTS_CACHE.get(category, {})


def get_category_config(category: str) -> Dict:
    cfg = _read_category_config(category)
    if cfg is None:
        return {"name": category, "tags": [], "prompt": "", "palette_weights": {}}
    return cfg


def create_category(name: str) -> bool:
    name = name.strip().replace(" ", "_")
    fld = os.path.join(CATEGORIES_DIR, name)
    if os.path.exists(fld):
        return False
    os.makedirs(fld)
    return write_category_config(name, {"name": name, "tags": [], "prompt": "", "palette_weights": {}})


def delete_category(name: str) -> bool:
    import shutil
    fld = os.path.join(CATEGORIES_DIR, name)
    if not os.path.isdir(fld):
        return False
    shutil.rmtree(fld)
    return True


discover_categories()
