#!/usr/bin/env python3
"""CLI entry point for Wanalizer."""
import os
import sys
import argparse
import json

from .settings import load_settings, save_settings, resolve_dest_dir, PROJECT_DIR
from .categories import discover_categories, CATEGORIES
from .formats import STATIC_EXTENSIONS, ANIMATED_EXTENSIONS, PLUGIN_EXTENSIONS
from .organize import organize, flatten_all, get_analyzer
from .duplicates import load_hash_cache, find_duplicate_groups, move_duplicates
from .tags import load_tags, get_all_tags
from .parallel import cpu_count

OPTIONAL_DEPS: dict = {
    "numpy": "array operations",
    "cv2": "OpenCV (edge detection, features)",
    "imagehash": "perceptual hashing",
    "sklearn": "KMeans palette extraction",
    "clip": "OpenAI CLIP",
    "torch": "PyTorch",
}

HAS_DEPS = {}
for mod in OPTIONAL_DEPS:
    try:
        __import__(mod)
        HAS_DEPS[mod] = True
    except Exception:
        HAS_DEPS[mod] = False


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="wanalyzer",
        description="Wanalizer - Intelligent Wallpaper Organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", "-m",
                        choices=["lowlevel", "clip", "fusion", "ollama"],
                        default="lowlevel",
                        help="Classification mode (def: lowlevel). "
                             "'fusion' runs CLIP + LowLevel CV together (recommended "
                             "when CLIP is installed).")
    parser.add_argument("--dest", "-d", metavar="DIR", default=None,
                        help="Destination directory for organized wallpapers (def: WP/)")
    parser.add_argument("--dry", action="store_true",
                        help="Dry run: analyze only, don't move files")
    parser.add_argument("--full", "-f", action="store_true",
                        help="Full reset: flatten + reclassify all")
    parser.add_argument("--flatten", action="store_true",
                        help="Only flatten subdirectories")
    parser.add_argument("--find-duplicates", action="store_true",
                        help="List duplicates without moving")
    parser.add_argument("--dedupe", action="store_true", default=True,
                        help="Move duplicates to Duplicates/ (def: on)")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                        help="Disable duplicate removal")
    parser.add_argument("--parallel", type=int, default=None, metavar="N",
                        help="Use N worker processes (def: number of CPU cores, "
                             "auto-capped by analyzer mode)")
    parser.add_argument("--check-deps", action="store_true",
                        help="Show optional dependency status")
    parser.add_argument("--list-modes", action="store_true",
                        help="List available analysis modes")
    parser.add_argument("--set-dest", metavar="DIR",
                        help="Set the destination folder and save to config")
    parser.add_argument("--show-config", action="store_true",
                        help="Show current configuration")
    parser.add_argument("--report", metavar="FILE",
                        help="Save report to JSON file")
    return parser


def _check_deps():
    print("Optional dependencies:\n")
    for mod, desc in OPTIONAL_DEPS.items():
        status = "available" if HAS_DEPS.get(mod) else "missing"
        print(f"  [{status:9s}] {mod:14s} {desc}")
    print()


def _list_modes():
    print("Available analysis modes:\n")
    modes = [
        ("lowlevel", "Advanced CV algorithms (edges, silhouettes, textures, HOG, Fourier, features)\n"
         "             No ML models required. Fast, deterministic.\n"
         "             Best for: speed, reliability, no external dependencies."),
        ("clip",     "OpenAI CLIP zero-shot vision-language model.\n"
         "             Requires: clip, torch, torchvision (~150-350MB download)\n"
         "             Best for: semantic understanding, tag matching."),
        ("fusion",   "CLIP + LowLevel CV combined: multi-signal scorer that uses CLIP\n"
         "             semantic scores AND classical CV statistics. Recommended when\n"
         "             CLIP is installed. Gracefully degrades to lowlevel if CLIP\n"
         "             cannot be loaded."),
        ("ollama",   "Local vision LLMs via Ollama (LLaVA, MiniCPM-V, Moondream, etc.)\n"
         "             Requires: Ollama server running with vision models\n"
         "             Best for: detailed descriptions, character recognition, NSFW."),
    ]
    for name, desc in modes:
        print(f"  {name:10s}{desc}\n")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Handle config commands first
    if args.set_dest:
        s = load_settings()
        s["dest_dir"] = os.path.abspath(args.set_dest)
        save_settings(s)
        print(f"Destination folder set to: {s['dest_dir']}")
        return

    if args.show_config:
        s = load_settings()
        print("Current configuration:")
        print(f"  Destination: {resolve_dest_dir(s)}")
        print(f"  Mode: {s.get('organize_mode', 'lowlevel')}")
        print(f"  Config file: {os.path.join(PROJECT_DIR, '.wallpaper_analyzer.json')}")
        print(f"  Ollama: {s.get('ollama_url', 'http://localhost:11434')} / {s.get('ollama_model', 'llava:7b')}")
        return

    if args.dest:
        s = load_settings()
        s["dest_dir"] = os.path.abspath(args.dest)
        save_settings(s)

    if args.check_deps:
        _check_deps()
        return

    if args.list_modes:
        _list_modes()
        return

    discover_categories()

    if args.find_duplicates:
        from .duplicates import scan_and_hash as _scan_and_hash
        from .duplicates import save_hash_cache as shc
        scan_root = resolve_dest_dir(load_settings())
        all_files = []
        for d, _, fns in os.walk(scan_root):
            rel = os.path.relpath(d, scan_root)
            parts = rel.split(os.sep)
            # Always include the root itself; for nested dirs, skip hidden/special
            if rel != "." and any(
                p.startswith(".") or p in (".git", "__pycache__", "Duplicates")
                for p in parts
            ):
                continue
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in (STATIC_EXTENSIONS | ANIMATED_EXTENSIONS):
                    all_files.append(os.path.join(d, fn))
        if not all_files:
            print("No image files found in destination.")
            return
        # Use absolute paths for cache lookups; display relative paths in output
        abs_files = [os.path.abspath(p) for p in all_files]
        workers = max(1, cpu_count())
        if args.parallel is not None and args.parallel > 0:
            workers = args.parallel
        print(f"Hashing {len(abs_files)} files ({workers} workers)...")
        cache = load_hash_cache()
        cache = _scan_and_hash(
            abs_files, cache,
            parallel=workers,
        )
        shc(cache)
        groups = find_duplicate_groups(abs_files, cache)
        if not groups:
            print("No duplicates found.")
            return
        print(f"Found {len(groups)} duplicate group(s):\n")
        for i, group in enumerate(groups, 1):
            tier = group.get("tier", "")
            print(f"  Group {i} [{tier}] score={group['score']:.3f} ({len(group['files'])} files):")
            for f in group["files"]:
                try:
                    print(f"    - {os.path.relpath(f, scan_root)}")
                except ValueError:
                    print(f"    - {f}")
            print()
        return

    if args.flatten:
        flatten_all(include_category_dirs=args.full)
        return

    dest_dir = resolve_dest_dir(load_settings())
    parallel = args.parallel if args.parallel is not None else max(1, cpu_count())
    print(f"Wanalizer v3.0")
    print(f"Mode: {args.mode.upper()}")
    print(f"Destination: {dest_dir}")
    print(f"Directory: {os.getcwd()}")
    print(f"Workers: {parallel}\n")

    if args.dry:
        print("DRY RUN MODE\n")

    if args.full:
        print("Full reset: flattening all directories...\n")
        flatten_all(include_category_dirs=True)

    os.makedirs(dest_dir, exist_ok=True)
    root_files = [f for f in os.listdir(".")
                  if os.path.isfile(f) and not f.startswith(".")
                  and os.path.splitext(f)[1].lower() in (STATIC_EXTENSIONS | ANIMATED_EXTENSIONS)]

    if root_files:
        organize(
            mode=args.mode,
            dry_run=args.dry,
            dedupe=args.dedupe,
            parallel=parallel,
        )
    else:
        print("No files at root. Flattening structure first...\n")
        flatten_all(include_category_dirs=args.full)
        organize(
            mode=args.mode,
            dry_run=args.dry,
            dedupe=args.dedupe,
            parallel=parallel,
        )


if __name__ == "__main__":
    main()
