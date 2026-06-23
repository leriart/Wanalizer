"""Tag registry loader.

Tags are read from `tags.json` in this priority order:

  1. `WANALIZER_TAGS` environment variable (explicit override).
  2. `<PROJECT_DIR>/tags.json` (the editable copy shipped with the repo).
  3. The bundled copy shipped inside the package (`wallpaper_analyzer/data/tags.json`).
     This is what `pip install .` provides.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Set

from .settings import TAGS_PATH


def _bundled_tags_path() -> str:
    """Best-effort path to the tags.json shipped inside the package."""
    try:
        from importlib.resources import files

        candidate = files("wallpaper_analyzer").joinpath("data", "tags.json")
        if hasattr(candidate, "is_file") and candidate.is_file():
            return str(candidate)
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "tags.json")


def _resolve_tags_path() -> str:
    env = os.environ.get("WANALIZER_TAGS")
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(TAGS_PATH):
        return TAGS_PATH
    return _bundled_tags_path()


_tags_data: Dict = {}
_tags_flat: Set[str] = set()


def load_tags():
    global _tags_data, _tags_flat
    path = _resolve_tags_path()
    if not os.path.isfile(path):
        _tags_data = {}
        _tags_flat = set()
        return _tags_data
    try:
        with open(path, "r", encoding="utf-8") as fp:
            _tags_data = json.load(fp)
    except (json.JSONDecodeError, OSError):
        _tags_data = {}
    _tags_flat = set()
    for group in _tags_data.get("groups", {}).values():
        for tag in group.get("tags", []):
            _tags_flat.add(str(tag).lower())
    return _tags_data


def save_tags(data: Dict):
    """Save the editable tag registry to the project-level `tags.json`.

    If the bundled-data copy is being used (i.e. running from an
    installed wheel), we fall back to writing next to the package so
    the user's edits are still persisted somewhere.
    """
    global _tags_data, _tags_flat
    _tags_data = data
    _tags_flat = set()
    for group in data.get("groups", {}).values():
        for tag in group.get("tags", []):
            _tags_flat.add(str(tag).lower())
    target = TAGS_PATH if os.path.isdir(os.path.dirname(TAGS_PATH)) else _bundled_tags_path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"  [WARN] Could not save tags: {exc}")


def get_all_tags() -> Set[str]:
    return _tags_flat.copy()


def get_tag_groups() -> Dict:
    return _tags_data.get("groups", {})


def is_valid_tag(tag: str) -> bool:
    return tag.lower() in _tags_flat


def get_tags_path() -> str:
    return _resolve_tags_path()


load_tags()