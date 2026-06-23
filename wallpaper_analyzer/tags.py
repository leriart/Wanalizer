import os
import json
from typing import Dict, Set, List

from .settings import TAGS_PATH

_tags_data: Dict = {}
_tags_flat: Set[str] = set()

def load_tags():
    global _tags_data, _tags_flat
    if not os.path.isfile(TAGS_PATH):
        return {}
    try:
        with open(TAGS_PATH, "r", encoding="utf-8") as f:
            _tags_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        _tags_data = {}
    _tags_flat = set()
    for group in _tags_data.get("groups", {}).values():
        for tag in group.get("tags", []):
            _tags_flat.add(str(tag).lower())
    return _tags_data

def save_tags(data: Dict):
    global _tags_data, _tags_flat
    _tags_data = data
    _tags_flat = set()
    for group in data.get("groups", {}).values():
        for tag in group.get("tags", []):
            _tags_flat.add(str(tag).lower())
    try:
        with open(TAGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"  [WARN] Could not save tags: {exc}")

def get_all_tags() -> Set[str]:
    return _tags_flat.copy()

def get_tag_groups() -> Dict:
    return _tags_data.get("groups", {})

def is_valid_tag(tag: str) -> bool:
    return tag.lower() in _tags_flat

load_tags()
