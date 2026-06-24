import os
import json
from typing import Dict, List, Optional, Set

from .settings import resolve_dest_dir, load_settings

# Folders the *system* creates automatically that should NOT appear in the
# user-facing categories list (Duplicates, Low-Quality, Discarded,
# Uncategorized). NSFW is treated as a regular category so the user can
# see/browse/rename the NSFW wallpapers — see is_managed_category().
SPECIAL_FOLDERS: Set[str] = {"Duplicates", "Low-Quality", "Discarded", "Uncategorized"}
NSFW_FOLDER: str = "NSFW"
LOW_QUALITY_FOLDER: str = "Low-Quality"
DISCARDED_FOLDER: str = "Discarded"
DUPLICATES_FOLDER: str = "Duplicates"
UNCATEGORIZED_FOLDER: str = "Uncategorized"
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
    """Return True for any folder that should be exposed to the user.

    NSFW is included here (it's a real, browsable category with NSFW
    wallpapers). SPECIAL_FOLDERS contains only folders the system creates
    and manages entirely (Duplicates / Low-Quality / Discarded /
    Uncategorized) — those don't represent user-facing categories.
    """
    if folder_name.startswith(".") or folder_name.startswith("_"):
        return False
    if folder_name in SPECIAL_FOLDERS:
        return False
    return os.path.isdir(os.path.join(CATEGORIES_DIR, folder_name))


def is_managed_category(name: str) -> bool:
    """True for categories whose contents are auto-managed by the system.

    NSFW is the only one right now — wallpapers end up there based on the
    NSFW classifier, not because the user dragged them in. UI layers use
    this to render a special badge and to prevent NSFW from appearing as
    a target in the "import images" dialog.
    """
    return name == NSFW_FOLDER


def list_category_folders(dest: Optional[str] = None,
                          include_unconfigured: bool = True) -> List[str]:
    """Return category folder names under `dest`.

    Defaults to CATEGORIES_DIR when no path is given. By default the
    list contains every subfolder of `dest` that is not in
    SPECIAL_FOLDERS, including folders without a `.category.json`
    (so a freshly created NSFW folder still shows up).
    Set `include_unconfigured=False` to restrict to folders that have a
    `.category.json` (i.e. already discovered).
    """
    base = dest if dest is not None else CATEGORIES_DIR
    out: List[str] = []
    if not base or not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        if name.startswith(".") or name in SPECIAL_FOLDERS:
            continue
        p = os.path.join(base, name)
        if not os.path.isdir(p):
            continue
        if not include_unconfigured:
            if name == NSFW_FOLDER:
                # NSFW is always listed even without .category.json
                out.append(name)
                continue
            if not os.path.isfile(os.path.join(p, ".category.json")):
                continue
        out.append(name)
    return out


def count_media_in(folder: str) -> int:
    """Count media files directly inside `folder` (non-recursive)."""
    from .formats import STATIC_EXTENSIONS, ANIMATED_EXTENSIONS
    if not os.path.isdir(folder):
        return 0
    exts = STATIC_EXTENSIONS | ANIMATED_EXTENSIONS
    return sum(1 for f in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, f))
               and os.path.splitext(f)[1].lower() in exts
               and not f.startswith("."))


def _build_category_icons():
    CATEGORY_ICONS.clear()
    for cat in CATEGORIES:
        if cat == NSFW_FOLDER:
            CATEGORY_ICONS[cat] = "[NSFW]"
        else:
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
        # NSFW is always a category, even without a .category.json.
        if cfg is None and entry != NSFW_FOLDER:
            continue
        CATEGORIES.append(entry)
        if cfg is None:
            continue
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
        # NSFW folder may exist without a .category.json — provide a
        # sensible default that still allows the UI to render it.
        defaults = {"name": category, "tags": [], "prompt": "",
                     "palette_weights": {}}
        if category == NSFW_FOLDER:
            defaults["managed"] = True
            defaults["prompt"] = "Auto-managed NSFW folder. Wallpapers end up here based on the NSFW classifier; you can still browse, rename, and move them out manually."
        return defaults
    if category == NSFW_FOLDER and "managed" not in cfg:
        cfg["managed"] = True
    return cfg


def create_category(name: str) -> bool:
    name = name.strip().replace(" ", "_")
    fld = os.path.join(CATEGORIES_DIR, name)
    if os.path.exists(fld):
        return False
    os.makedirs(fld)
    cfg = {"name": name, "tags": [], "prompt": "", "palette_weights": {}}
    if name == NSFW_FOLDER:
        cfg["managed"] = True
    return write_category_config(name, cfg)


def delete_category(name: str) -> bool:
    if name == NSFW_FOLDER:
        # Refuse to delete the auto-managed NSFW folder via the UI.
        return False
    import shutil
    fld = os.path.join(CATEGORIES_DIR, name)
    if not os.path.isdir(fld):
        return False
    shutil.rmtree(fld)
    return True


discover_categories()
