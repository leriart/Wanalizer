"""Regenerate tags + prompts for every category folder.

This is a one-shot utility that:
  * Walks every category in the Wallpapers-Lerit library
  * Picks N representative images per category (well-spread by file size)
  * Asks the local Ollama vision model (llava-phi3) for structured info:
      - tags (validated against the curated `tag_policies` list)
      - palette + style signature
      - a 1-sentence visual description
  * Asks a text model (qwen2.5:3b) to write a 2-3 sentence *unified
    aesthetic prompt* that captures the actual visual identity of the
    category (not "the image features..." boilerplate)
  * Filters every tag through `tag_policies.filter_polluted_tags` and
    caps at `TAG_BUDGET_PER_CATEGORY`
  * Saves into `.category.json`, replacing the old polluted tags/prompt
  * Cleans up the legacy `.category.json.bak` files

Run from the repo root:
    .venv/bin/python -m wallpaper_analyzer.tools.regenerate_categories
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Set

from ..ollama_client import OllamaClient
from ..tag_policies import (
    CURATED_SAFE_TAGS,
    CURATED_NSFW_TAGS,
    GENERIC_OVERUSED,
    LLM_TAG_PICK_MAX,
    PROTECTED_TAGS,
    TAG_BUDGET_PER_CATEGORY,
    build_focused_curated_list,
    cap_category_tags,
    dedupe_against_existing,
    filter_polluted_tags,
    parse_llm_tag_response,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIBRARY_ROOT = "/home/leo/Imágenes/Wallpapers-Lerit"
SAMPLES_PER_CATEGORY = 3
VISION_MODEL = "llava-phi3:3.8b"
TEXT_MODEL = "qwen2.5:3b"
VISION_TIMEOUT = 240
TEXT_TIMEOUT = 120

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

log = logging.getLogger("regenerate_categories")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_images(category_dir: str) -> List[str]:
    """Return all image files in `category_dir` (one level deep)."""
    out = []
    for name in sorted(os.listdir(category_dir)):
        if name.startswith("."):
            continue
        full = os.path.join(category_dir, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            out.append(full)
    return out


def pick_samples(files: List[str], n: int) -> List[str]:
    """Pick N representative samples from `files`.

    Strategy: pick the smallest, the largest, and the rest evenly
    spaced. Falls back to random sampling for small lists.
    """
    if len(files) <= n:
        return files
    files_sorted = sorted(files, key=lambda p: os.path.getsize(p))
    pick = [files_sorted[0], files_sorted[-1]]
    if n >= 3:
        step = max(1, len(files_sorted) // (n - 2))
        for i in range(step, len(files_sorted) - 1, step):
            pick.append(files_sorted[i])
    return list(dict.fromkeys(pick))[:n]


def load_existing_cfg(category_dir: str) -> dict:
    cfg_path = os.path.join(category_dir, ".category.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Vision analysis (per image)
# ---------------------------------------------------------------------------

VISION_PROMPT = """You are tagging wallpaper images for a personal library.

Look at the image carefully and return STRICT JSON in this exact shape:
{{
  "subject": "<1-3 word description of the main subject, lowercase>",
  "style":  "<visual style: e.g. anime, photograph, pixel-art, 3d-render, watercolor, illustration, digital-art, sketch>",
  "tags":   [<up to 8 discriminating tags from the list below, lowercase, single words or hyphenated>],
  "palette": "<1-3 word colour description, lowercase, e.g. pastel-pink, deep-blue, monochrome>",
  "mood":   "<1-2 word mood/atmosphere, lowercase, e.g. dark-mood, cheerful, serene, dramatic>"
}}

Rules for `tags`:
  * ONLY pick tags from the provided list.
  * Skip generic overused words: do NOT pick `illustration`, `digital-art`,
    `pastel`, `neon`, `vintage`, `retro`, `monochrome`, `minimalist`,
    `simple`, `detailed`, `sky`, `landscape`, `building`, `city`, `urban`,
    `nature`, `abstract`, `vibrant`, `colorful`, `warm`, `cool`,
    `landscape-orientation`, `portrait-orientation`, `wide`, `vertical`,
    `horizontal`, `centered`, `aerial`, `closeup`, `wide-shot`, `night`,
    `day`, `dark`, `light` unless they are genuinely the most
    distinctive feature.
  * Prefer specific subjects (e.g. `dragon`, `samurai`, `forest`, `temple`)
    and specific styles (e.g. `pixel-art`, `cyberpunk`, `watercolor`,
    `low-poly`, `voxel`).
  * Maximum 8 tags.

Allowed tags (pick from this list):
{tag_list}

Respond with JSON only, no prose."""


def analyse_image_with_vision(client: OllamaClient, img_path: str,
                              focused_tags: List[str]) -> Dict:
    """Use vision model to get tags + style + palette + mood + subject.

    Falls back to a heuristic palette/style signature when the model
    is unreachable or returns invalid JSON, so we never lose a sample.
    """
    tag_list = ", ".join(focused_tags)
    prompt = VISION_PROMPT.format(tag_list=tag_list)
    b64 = client._img_to_b64(img_path)
    if not b64:
        return heuristic_signature(img_path, focused_tags)
    try:
        data = client.generate_structured(
            prompt=prompt,
            schema_hint=(
                '{"subject": "", "style": "", "tags": [], '
                '"palette": "", "mood": ""}'
            ),
            images=[b64],
            temperature=0.05,
            max_tokens=200,
            default={},
            timeout=VISION_TIMEOUT,
        )
    except Exception as e:
        log.warning(f"    vision error: {e}")
        return heuristic_signature(img_path, focused_tags)
    if not data or not isinstance(data, dict) or not data.get("tags"):
        return heuristic_signature(img_path, focused_tags)
    return data


def heuristic_signature(img_path: str, focused_tags: List[str]) -> Dict:
    """Cheap CV-only signature: dominant palette + style guess.

    Used as a fallback when the vision LLM is slow/down. Doesn't
    produce tags because heuristic tag generation is what produced the
    pollution problem in the first place - the category-level prompt
    can still be derived from palette + style alone.
    """
    from collections import Counter
    from PIL import Image
    try:
        im = Image.open(img_path).convert("RGB")
        im.thumbnail((128, 128))
        pixels = list(im.getdata())
    except Exception:
        return {}
    r = sum(p[0] for p in pixels) // len(pixels)
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    palette_bits = []
    if r + g + b < 200:
        palette_bits.append("dark")
    elif r + g + b > 600:
        palette_bits.append("light")
    if abs(r - g) < 20 and abs(g - b) < 20:
        palette_bits.append("monochrome")
    if r > g and r > b:
        palette_bits.append("warm")
    if b > r and b > g:
        palette_bits.append("cool")
    palette_bits.append(["red", "green", "blue"][
        [(r, 0), (g, 1), (b, 2)].index(max((r, 0), (g, 1), (b, 2)))
    ] + "-accent")
    bucket = Counter()
    for p in pixels:
        bucket[(p[0] // 64, p[1] // 64, p[2] // 64)] += 1
    top = bucket.most_common(1)[0][0]
    palette_bits.append(f"{top[0]*64}-{top[1]*64}-{top[2]*64}")
    return {
        "subject": "",
        "style": "heuristic",
        "tags": [],
        "palette": " ".join(palette_bits),
        "mood": "unknown",
    }


def clean_structured_tags(raw_tags, focused_set: Set[str]) -> List[str]:
    """Sanitize tags from the vision model output."""
    if not raw_tags:
        return []
    if isinstance(raw_tags, str):
        raw_tags = parse_llm_tag_response(raw_tags, focused_set,
                                           max_tags=LLM_TAG_PICK_MAX)
    out = []
    seen = set()
    for t in raw_tags:
        tt = str(t).strip().lower().rstrip(".,;:!?\"'")
        if not tt:
            continue
        if tt in seen:
            continue
        if tt in GENERIC_OVERUSED:
            continue
        if tt not in focused_set:
            continue
        out.append(tt)
        seen.add(tt)
        if len(out) >= 8:
            break
    return out


# ---------------------------------------------------------------------------
# Prompt synthesis (per category)
# ---------------------------------------------------------------------------

PROMPT_SYNTHESIS_PROMPT = """You are writing a unified aesthetic prompt that
captures the visual identity of a wallpaper CATEGORY.

You will be given:
  * the category name
  * 5-8 short visual descriptions of individual wallpapers from that category
  * the most common style, palette and mood signatures

Write a 2-3 sentence PROMPT that:
  * describes what the wallpapers look like as a unified set
  * mentions the dominant palette (with a concrete adjective, not just
    "vibrant" or "minimalist")
  * mentions the dominant subject / scene type if one is obvious
  * mentions the dominant artistic style (e.g. anime, pixel-art, minimalist
    illustration, photograph)
  * does NOT start with "The image features..." or "A unified aesthetic..."
  * is written as a stable description that would still fit new wallpapers
    added later

Category: {category}
Sample descriptions:
{descriptions}
Dominant style: {style}
Dominant palette: {palette}
Dominant mood: {mood}

Respond with ONLY the prompt (2-3 sentences), no preamble, no bullet list."""


def synthesise_prompt(client: OllamaClient, category: str,
                      descriptions: List[str],
                      style: str, palette: str, mood: str) -> Optional[str]:
    """Use text model to generate a unified-aesthetic prompt."""
    desc_block = "\n".join(f"  - {d}" for d in descriptions if d)
    if not desc_block:
        return None
    prompt = PROMPT_SYNTHESIS_PROMPT.format(
        category=category,
        descriptions=desc_block,
        style=style or "unspecified",
        palette=palette or "unspecified",
        mood=mood or "unspecified",
    )
    out = client.generate(
        prompt=prompt,
        temperature=0.3,
        max_tokens=220,
        timeout=TEXT_TIMEOUT,
    )
    if not out:
        return None
    out = out.strip().strip('"').strip("'")
    bad_prefixes = (
        "the image features", "a unified aesthetic can be achieved",
        "a unified aesthetic combines", "the unified aesthetic",
        "this category", "category:",
    )
    lower = out.lower()
    for bp in bad_prefixes:
        if lower.startswith(bp):
            for sep in (".", "\n"):
                idx = out.find(sep)
                if idx != -1 and idx < 200:
                    out = out[idx + 1:].strip()
                    break
            break
    if len(out) > 600:
        out = out[:600].rsplit(".", 1)[0] + "."
    return out or None


# ---------------------------------------------------------------------------
# Per-category orchestration
# ---------------------------------------------------------------------------

def regenerate_category(category_dir: str, vision_client: OllamaClient,
                        text_client: OllamaClient) -> bool:
    """Regenerate tags + prompt for a single category. Returns True on success."""
    category = os.path.basename(category_dir.rstrip("/"))
    cfg_path = os.path.join(category_dir, ".category.json")
    existing = load_existing_cfg(category_dir)

    nsfw = (category.lower() == "nsfw")
    curated = CURATED_NSFW_TAGS if nsfw else CURATED_SAFE_TAGS

    focused = build_focused_curated_list(
        full_registry=curated,
        existing=existing.get("tags", []) + list(PROTECTED_TAGS),
        nsfw=nsfw,
    )
    focused = focused[:60]
    focused_set = set(focused)

    images = list_images(category_dir)
    if not images:
        log.warning("  no images found, skipping")
        return False
    samples = pick_samples(images, SAMPLES_PER_CATEGORY)

    all_tags: Set[str] = set()
    descriptions: List[str] = []
    styles: List[str] = []
    palettes: List[str] = []
    moods: List[str] = []
    subjects: List[str] = []

    log.info(f"  analysing {len(samples)} samples with {VISION_MODEL}...")
    for i, img in enumerate(samples, 1):
        try:
            data = analyse_image_with_vision(vision_client, img, focused)
        except Exception as e:
            log.warning(f"    [{i}/{len(samples)}] {os.path.basename(img)}: error {e}")
            continue

        clean_tags = clean_structured_tags(data.get("tags", []), focused_set)
        all_tags.update(clean_tags)

        for key, lst in (("style", styles), ("palette", palettes),
                         ("mood", moods), ("subject", subjects)):
            v = (data.get(key) or "").strip().lower()
            if v and len(v) < 40 and v != "unspecified":
                lst.append(v)

        sub = (data.get("subject") or "").strip()
        style = (data.get("style") or "").strip()
        palette = (data.get("palette") or "").strip()
        mood = (data.get("mood") or "").strip()
        bits = []
        if sub:
            bits.append(sub)
        if style:
            bits.append(f"in {style} style")
        if palette:
            bits.append(f"{palette} palette")
        if mood:
            bits.append(f"{mood} mood")
        desc = ", ".join(bits) if bits else ""
        if desc:
            descriptions.append(desc)
        log.info(f"    [{i}/{len(samples)}] {os.path.basename(img)}: "
                 f"{len(clean_tags)} tags, {desc or '(empty)'}")

    if not all_tags and not descriptions:
        log.warning(f"  no usable vision output for {category}")
        return False

    all_tags.update(t for t in PROTECTED_TAGS
                    if t in focused_set and t.lower() not in GENERIC_OVERUSED)
    all_tags.add(category.lower())

    final_tags = filter_polluted_tags(all_tags)
    final_tags = cap_category_tags(final_tags, max_tags=TAG_BUDGET_PER_CATEGORY)
    final_tags = dedupe_against_existing(final_tags,
                                         existing.get("tags", []))
    final_tags = list(final_tags)[:TAG_BUDGET_PER_CATEGORY]

    def _majority(lst: List[str]) -> str:
        if not lst:
            return ""
        counts: Dict[str, int] = {}
        for x in lst:
            counts[x] = counts.get(x, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return best[0][0] if best else ""

    dom_style = _majority(styles)
    dom_palette = _majority(palettes)
    dom_mood = _majority(moods)

    log.info(f"  synthesising unified prompt with {TEXT_MODEL}...")
    new_prompt = synthesise_prompt(
        text_client, category,
        descriptions,
        dom_style, dom_palette, dom_mood,
    )
    if not new_prompt:
        new_prompt = (
            f"{category} wallpapers: {dom_style or 'distinctive'} style, "
            f"{dom_palette or 'characteristic'} palette, "
            f"{dom_mood or 'consistent'} mood."
        )
    log.info(f"  prompt: {new_prompt[:120]!r}")
    log.info(f"  final tags ({len(final_tags)}): {final_tags}")

    cfg = dict(existing)
    cfg["category"] = category
    cfg["tags"] = final_tags
    cfg["prompt"] = new_prompt
    cfg["style"] = dom_style
    cfg["palette"] = dom_palette
    cfg["mood"] = dom_mood
    cfg["samples_analysed"] = len(samples)
    cfg["regenerated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg["regenerated_with"] = {
        "vision_model": VISION_MODEL,
        "text_model": TEXT_MODEL,
        "tag_policies": True,
    }

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    log.info(f"  wrote {cfg_path}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cleanup_bak_files() -> int:
    """Delete every .category.json.bak left by previous runs."""
    removed = 0
    for root, _dirs, files in os.walk(LIBRARY_ROOT):
        for name in files:
            if name.endswith(".category.json.bak"):
                full = os.path.join(root, name)
                try:
                    os.remove(full)
                    removed += 1
                    log.info(f"removed backup: {full}")
                except OSError as e:
                    log.warning(f"could not remove {full}: {e}")
    return removed


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.path.isdir(LIBRARY_ROOT):
        log.error(f"library not found: {LIBRARY_ROOT}")
        return 1

    vision_client = OllamaClient(model=VISION_MODEL, timeout=VISION_TIMEOUT)
    text_client = OllamaClient(model=TEXT_MODEL, timeout=TEXT_TIMEOUT)
    if not vision_client.check():
        log.error("Ollama is not reachable at http://localhost:11434")
        return 1
    if not vision_client._supports_vision(VISION_MODEL):
        log.error(f"vision model {VISION_MODEL} doesn't support vision")
        return 1

    categories = sorted(
        d for d in os.listdir(LIBRARY_ROOT)
        if os.path.isdir(os.path.join(LIBRARY_ROOT, d))
        and not d.startswith(".")
        and d not in {"Duplicates"}
    )
    log.info(f"Found {len(categories)} categories: {categories}")
    log.info(f"Vision model: {VISION_MODEL} | Text model: {TEXT_MODEL}")
    log.info(f"Samples per category: {SAMPLES_PER_CATEGORY}")

    ok = 0
    for cat in categories:
        cat_dir = os.path.join(LIBRARY_ROOT, cat)
        log.info(f"\n=== {cat} ===")
        try:
            if regenerate_category(cat_dir, vision_client, text_client):
                ok += 1
        except Exception as e:
            log.exception(f"  failed: {e}")

    log.info("\n--- cleaning up .bak files ---")
    removed = cleanup_bak_files()
    log.info(f"removed {removed} backup files")

    vision_client.close()
    text_client.close()

    log.info(f"\nDONE: regenerated {ok}/{len(categories)} categories, "
             f"removed {removed} .bak files")
    return 0 if ok == len(categories) else 2


if __name__ == "__main__":
    sys.exit(main())