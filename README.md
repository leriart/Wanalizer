<div align="center">

# Wanalizer

### Intelligent wallpaper organization that just works.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-e01020.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-e01020.svg?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-e01020.svg?style=for-the-badge)](#-installation)
[![Version 3.0](https://img.shields.io/badge/version-3.0.0-e01020.svg?style=for-the-badge)](#)
[![Free-threaded](https://img.shields.io/badge/free--threaded-Python%203.14t-e01020.svg?style=for-the-badge)](https://py-free-threading.github.io/)

**Four analysis pipelines. One-click dedupe. A PySide6 desktop UI. CLI, too.**

Wanalizer takes a folder full of unsorted wallpapers and splits it into
named category folders using either classical computer vision, OpenAI
CLIP, a CLIP+CV fusion ensemble, or a local Ollama vision LLM.

</div>

---

## Table of contents

- [Why Wanalizer?](#-why-wanalizer)
- [Highlights](#-highlights)
- [Screenshots & UI overview](#-screenshots--ui-overview)
- [Quick start](#-quick-start)
- [Analysis modes](#-analysis-modes)
- [How classification works](#-how-classification-works)
- [Multi-signal combiner](#-multi-signal-combiner)
- [Per-category configuration](#-per-category-configuration)
- [Duplicate detection](#-duplicate-detection)
- [Project structure](#-project-structure)
- [Installation](#-installation)
- [CLI reference](#-cli-reference)
- [Tips & recipes](#-tips--recipes)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## Why Wanalizer?

Other wallpaper organizers either:

- need a **remote API** (privacy + latency),
- require a **GPU** (CLIP, LLaVA, ...),
- ship as a **black box** (you can't tune it),
- or only sort by **filename heuristics**.

Wanalizer gives you **four interchangeable pipelines** that run locally,
a 13-signal multi-scorer that you can inspect and tune, MD5-based
duplicate detection with caching, an interactive per-category
configuration wizard (Q&A **or** AI-suggested), and a full
**PySide6 desktop UI** — all from the same codebase.

| What                       | Wanalizer                                 |
| -------------------------- | ----------------------------------------- |
| Offline / no network       | ✅  LowLevel CV runs on any CPU            |
| Semantic understanding     | ✅  CLIP + Fusion + Ollama                 |
| Detects duplicates         | ✅  MD5 with incremental cache             |
| Per-category tuning        | ✅  Q&A wizard + AI-suggest + JSON edit    |
| Visual browser             | ✅  Reorganize page with thumbnails       |
| Multi-signal classifier    | ✅  13 weighted signals, pollution-aware  |
| True parallel CPU work     | ✅  ProcessPoolExecutor + optional 3.14t   |
| GUI + CLI from one package | ✅  Shared pipeline                       |

---

## Highlights

- **Four classification modes**: Low-Level CV, CLIP, CLIP+LowLevel
  fusion (recommended), and Ollama vision LLMs (LLaVA, MiniCPM-V,
  Llama 3.2 Vision, Moondream, ...).
- **13-signal multi-scorer** with anti-pollution defences so a noisy
  tag registry cannot hijack the result.
- **MD5 duplicate detection** with a per-folder cache so the second
  scan is essentially free; per-group keep/move/delete decisions.
- **Per-category "expected" configuration**: Q&A wizard **or** AI
  suggestion that samples your images and proposes `expected` rules
  (palette, aspect ratio, min resolution, style keywords, exclusions).
- **Reorganize tab**: visual browser with Move / Delete, aspect-ratio
  filter, category-side tooltip showing each folder's expected spec,
  integrated rename dialog.
- **Categories tab**: visual editor of `.category.json` files with
  *AI Suggest* button that pulls configuration from sample images.
- **50+ image formats** supported (JPEG, PNG, WebP, AVIF, HEIC, PSD,
  TIFF, RAW, SVG, PDF, …) plus a single-frame extraction for videos
  and animated images via `ffmpeg`.
- **True parallel hashing/classification** via
  `ProcessPoolExecutor` (always scales with cores, no GIL) and an
  **optional free-threaded Python 3.14t** runtime for true parallel
  threading.
- **Qt6 desktop UI** plus a complete **CLI**, both backed by the same
  pipeline. No feature in the CLI is missing from the GUI or vice versa.

---

## Screenshots & UI overview

The GUI ships with **9 pages** wired into a single `QStackedWidget`:

| # | Page          | What it does                                                              |
|---|---------------|---------------------------------------------------------------------------|
| 0 | Dashboard     | Library overview, mode picker, AI health, recent activity log            |
| 1 | Organize      | Run the classification pipeline; choose source/destination + workers      |
| 2 | Reorganize    | Visual browser: Move/Delete with aspect-ratio filter + rename dialog      |
| 3 | AI Models     | Manage the CLIP engine and the Ollama server connection                   |
| 4 | Categories    | Create/edit/configure categories; AI Suggest from sample images          |
| 5 | Tags          | Edit the global tag registry                                              |
| 6 | Duplicates    | MD5 duplicate scan, review groups, Move/Delete actions                    |
| 7 | Dependencies  | Install optional packages (OpenCV, scikit-learn, PySide6, …)              |
| 8 | Settings      | Application configuration (thresholds, theme, destination folder)         |

A typical session looks like:

```
┌─ Dashboard ─────────────────────────────────────────────────────────────┐
│  Total Files: 12 348     Categories: 23     Duplicates: 87             │
│  Mode: [x] Fusion (CLIP + LowLevel)  [ ] CLIP  [ ] LowLevel  [ ] Ollama │
│  CLIP: Ready   Ollama: Connected (4 models)                             │
│  [ Open Organizer ]   [ AI Models ]   [ Refresh ]                       │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ Organize ──────────────────────────────────────────────────────────────┐
│  Source: /home/.../Incoming                                               │
│  Destination: /home/.../WP                                                │
│  Mode: Fusion · Workers: 8 · Min quality: 0.0                             │
│  [x] Dry run   [x] Find duplicates   [ ] Full reset                       │
│  ▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱  67 %   image_0432.jpg → Cyberpunk                   │
│  [ Organize Now ]   [ Dry Run ]   [ Stop ]                                │
└─────────────────────────────────────────────────────────────────────────┘
```

> Tip: hit **Categories → AI Suggest** on any folder and Wanalizer will
> sample N images, run CLIP + CV, and propose a `.category.json` you
> can review before saving.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/leo/Wanalizer.git
cd Wanalizer

# 2. Launch the GUI (auto-creates .venv, installs PySide6, etc.)
./run.sh

# 3. Or run the CLI
./run.sh --cli --help
./run.sh --cli --mode fusion --dry          # preview only
./run.sh --cli --mode fusion --full        # flatten + classify

# 4. Or, after `pip install .`
wanalyzer --mode fusion --dry
wanalyzer-gui
```

If you prefer, you can run the launcher in **free-threaded mode** for
true parallel threading (Python 3.14t, no GIL). It downloads the
runtime automatically on first run:

```bash
./run.sh --cli --ft --mode fusion --dry
```

---

## Analysis modes

| Mode          | What it uses                                                   | Strength                                                       | Dependencies                            | Speed                         |
|---------------|----------------------------------------------------------------|----------------------------------------------------------------|-----------------------------------------|-------------------------------|
| **Low-Level CV** | Canny/Sobel/Scharr edges, Otsu + adaptive silhouettes, Hu moments, LBP/GLCM/Gabor texture, ORB/FAST keypoints, FFT, HOG, colour moments, composition, symmetry, pattern detection | Statistical structure; no ML; runs on any CPU                  | Pillow, numpy, scipy                    | Fast                          |
| **CLIP**      | OpenAI CLIP zero-shot vision-language model (ViT-B/32 by default) | Semantic understanding ("this is Anime, not Gruvbox")          | torch, clip                             | Slow first image, cached after|
| **CLIP + LowLevel (fusion)** | Runs LowLevel CV and CLIP in parallel, then fuses all 13 signals     | Best of both: semantic content + style statistics              | torch, clip, Pillow, numpy              | Slower than CLIP alone        |
| **Ollama**    | Local vision LLM via Ollama (LLaVA, MiniCPM-V, Llama 3.2 Vision, Moondream) | Natural-language descriptions; character recognition; NSFW      | requests + running Ollama server        | Medium                        |

**Fusion** is the recommended default when CLIP is installed. It is a
strict superset of the LowLevel pipeline (gracefully degrades if CLIP
cannot load).

---

## How classification works

1. **Source files** are scanned from the chosen source folder
   (`PROJECT_DIR` by default, configurable via `--dest` / `--set-dest`).
2. If **dedupe is enabled**, MD5s are computed and exact duplicates are
   moved to a `Duplicates/` folder (cached for next time).
3. Each remaining file is analysed by the selected mode and turned into
   a **profile dict** with weights, style scores, theme scores,
   content detectors, and (where available) CLIP scores.
4. The **multi-signal combiner** scores every category against the
   profile and picks the best match, optionally weighted by the
   category's `expected` spec.
5. The file is **moved into the winning category**, optionally renamed
   according to the rename strategy in the Organize tab.

The full pipeline lives in `wallpaper_analyzer/organize.py:405` and is
the same code path used by both the CLI and the GUI.

---

## Multi-signal combiner

Every analyzer returns a profile dict. `classify.py` combines **13
weighted signals** into a single confidence score per category:

```
tags         0.10
palette      0.10
style        0.08
content      0.05
theme        0.07
composition  0.05
quality      0.03
pattern      0.04
size         0.03
prompt       0.08
clip         0.25
clip_nsfw    0.05
fingerprint  0.20
```

| Signal      | What it measures                                                                 |
|-------------|----------------------------------------------------------------------------------|
| tags        | TF-IDF weighted overlap between the image tags and each category's tags          |
| palette     | Cosine similarity between the image's colour distribution and the category palette|
| style       | Continuous bonuses for `anime_score`, `skin_fraction`, `periodicity_score`, ...   |
| content     | Specific detectors: `minecraft_score`, `pixel_art_score`, `minimalist_score`, ...|
| theme       | Match against named themes (Catppuccin, Dracula, TokyoNight, Nord, Gruvbox, ...)  |
| composition | Rule-of-thirds + subject-area heuristics                                          |
| quality     | Tenengrad + aesthetic + sharpness for `photo` / `illustration` categories         |
| pattern     | Periodicity and tile detection                                                   |
| size        | Aspect-ratio bucket (phone, vertical, wide, landscape)                           |
| prompt      | TF-IDF cosine between the generated prompt and the category prompt               |
| clip        | CLIP softmax probabilities (only present if CLIP ran)                            |
| clip_nsfw   | CLIP-driven NSFW boost for `nsfw`/`18+`/`ecchi`/... categories                   |
| fingerprint | Z-score similarity against each category's learned CV fingerprint                 |

The **fingerprint** signal is the most reliable because it compares
the image's CV profile against the actual feature distribution of
each category. The **clip** signal is the most discriminative when
CLIP has been run. The tag-based signals have explicit **anti-pollution
defences** so a noisy registry (the typical LLM-generated mess) cannot
hijack the result.

---

## Per-category configuration

Each category folder contains a `.category.json` describing what belongs
there. Example for a Cyberpunk folder:

```json
{
  "name": "Cyberpunk",
  "tags": ["cyberpunk", "neon", "dark", "city"],
  "prompt": "Dark futuristic city with neon purple and teal lights",
  "palette_weights": {
    "Purple": 1.0,
    "Pink-Magenta": 0.5,
    "Black": 0.3
  },
  "expected": {
    "aspect_ratios": ["horizontal", "square"],
    "file_kinds": ["image"],
    "min_resolution": [1920, 1080],
    "color_palette": ["dark", "cool"],
    "style_keywords": ["neon", "illustration"],
    "exclude_keywords": ["anime", "photograph"],
    "source": "user"
  }
}
```

Three ways to create it:

1. **Interactive Q&A wizard** — *Categories → Configure (Q&A)*.
2. **AI Suggest** — *Categories → AI Suggest*. Samples images, runs
   CLIP + CV, proposes an `expected` block you can review and save.
3. **Manual JSON edit**.

After configuration, the Reorganize tab shows the expected spec in the
sidebar tooltip, and `classify.config_match_score` uses the spec as a
soft tie-breaker when classifying.

---

## Duplicate detection

The Duplicates tab detects exact duplicates via **MD5 hashing**. The
hash cache lives in `.wallpaper_analyzer_hashes.json` inside the
destination folder, so the second and subsequent scans only hash new
files.

For each duplicate group:

- The **largest file** is kept (heuristic for highest quality).
- The rest can be **moved** to `Duplicates/` or **deleted** permanently.
- Per-group decisions or one-click "Move ALL" / "Delete ALL".

There is also a CLI shortcut:

```bash
./run.sh --cli --find-duplicates
```

---

## Project structure

```
Wanalizer/
├── run.sh                              Entry point (GUI or CLI)
├── requirements.txt                    Python dependencies
├── setup.py                            Legacy setup script
├── pyproject.toml                      Modern build config
├── tags.json                           Global tag registry
├── wallpaper_analyzer/                 Core package (Python module name kept for stability)
│   ├── __init__.py                     Project metadata (__project__, __version__)
│   ├── analyzers/
│   │   ├── base.py                     Abstract analyzer interface
│   │   ├── lowlevel_mode.py            Low-Level CV analyzer
│   │   ├── fusion_mode.py              CLIP + LowLevel fusion analyzer
│   │   └── __init__.py
│   ├── lowlevel/                       Classical CV algorithms
│   │   ├── edges.py                    Canny, Sobel, Scharr, orientation
│   │   ├── silhouettes.py              Otsu, adaptive thresholding
│   │   ├── contours.py                 Hu moments, symmetry
│   │   ├── texture.py                  LBP, GLCM, Gabor filters
│   │   ├── features.py                 ORB, FAST, Shi-Tomasi
│   │   ├── fourier.py                  FFT, frequency distribution
│   │   ├── hog.py                      Histogram of Oriented Gradients
│   │   ├── color_advanced.py           Colour moments, harmony, LAB/HSV stats
│   │   ├── composition.py              Rule of thirds, depth, saliency
│   │   ├── quality_advanced.py         Tenengrad, BRISQUE-like, pHash, noise
│   │   ├── subject.py                  Largest component, foreground/background
│   │   ├── pattern.py                  Periodicity, tile detection, complexity
│   │   ├── symmetry_advanced.py        Bilateral, rotational, diagonal
│   │   └── category_profile.py         Per-category CV fingerprints + scoring
│   ├── gui/
│   │   ├── __init__.py                 Main window + theme
│   │   ├── __main__.py                 (python -m wallpaper_analyzer.gui)
│   │   ├── theme.py                    Red/black/white QSS stylesheet
│   │   ├── widgets.py                  Custom widgets (table buttons)
│   │   ├── workers.py                  Background QThread workers
│   │   ├── rename_dialog.py            Batch rename dialog
│   │   ├── category_config_dialog.py   Per-category Q&A wizard + AI review
│   │   └── pages/
│   │       ├── dashboard.py            Overview, mode selection, health
│   │       ├── organize.py             Classification pipeline runner
│   │       ├── reorganize.py           Visual file browser, filter, rename
│   │       ├── ai_models.py            CLIP and Ollama model management
│   │       ├── categories.py           Category folder management
│   │       ├── tags.py                 Tag registry editor
│   │       ├── duplicates.py           Duplicate scan UI
│   │       ├── dependencies.py         Package installer
│   │       └── settings.py             Configuration panel
│   ├── tools/
│   │   └── regenerate_categories.py    One-shot tool to rebuild every category config via Ollama
│   ├── clip_client.py                  CLIP engine and CLIPAnalyzer
│   ├── ollama_client.py                Ollama vision LLM analyzer
│   ├── categories.py                   Category management
│   ├── category_config.py              Per-category expected-configuration system
│   ├── classify.py                     Multi-signal classification combiner
│   ├── clean_tags.py                   Tag cleanup utilities
│   ├── color.py                        HSV / LAB / RGB helpers
│   ├── duplicates.py                   MD5-based duplicate detection
│   ├── formats.py                      Format detection (50+)
│   ├── hashing.py                      Perceptual hashing helpers
│   ├── minimal_ai.py                   Minimal AI tag/prompt generator
│   ├── organize.py                     Main classification pipeline + factory
│   ├── parallel.py                     Parallel processing helpers
│   ├── profile.py                      Image profile builder
│   ├── prompt_generator.py             AI prompt generation
│   ├── quality.py                      Sharpness, aesthetic scoring
│   ├── rename.py                       Rename strategies
│   ├── settings.py                     Configuration
│   ├── tag_policies.py                 Tag policy enforcement
│   ├── tag_suggester.py                AI tag suggestion
│   ├── tags.py                         Tag registry
│   └── cli.py                          Command-line interface
```

---

## Installation

### Option A — one-shot launcher (recommended)

The launcher auto-creates `.venv`, installs PySide6, picks the right
Python interpreter, and falls back to a free-threaded runtime when
asked:

```bash
./run.sh                  # GUI
./run.sh --cli --help     # CLI
./run.sh --bootstrap      # install everything upfront
```

### Option B — manual pip install

```bash
# Create venv and install core deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: full CV support (adds OpenCV, perceptual hashing, KMeans)
pip install opencv-python-headless imagehash scikit-learn

# Optional: PySide6 for the GUI
pip install PySide6

# Optional: CLIP + PyTorch (CPU build; ~2 GB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git

# Or install everything from pyproject.toml extras
pip install ".[all]"
```

After `pip install .`, two new console scripts are available:

```bash
wanalyzer       # CLI
wanalyzer-gui   # GUI
```

### External tools

| Tool      | Used for                           | Required?       |
|-----------|------------------------------------|-----------------|
| `ffmpeg`  | Single-frame extraction for videos | Optional (graceful skip if missing) |

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## CLI reference

```text
wanalyzer [-h] [--mode {lowlevel,clip,fusion,ollama}] [--dest DIR]
          [--dry] [--full] [--flatten] [--find-duplicates]
          [--dedupe | --no-dedupe] [--parallel N]
          [--check-deps] [--list-modes]
          [--set-dest DIR] [--show-config] [--report FILE]
```

| Flag                      | What it does                                                   |
|---------------------------|----------------------------------------------------------------|
| `--mode`, `-m`            | `lowlevel` (default), `clip`, `fusion`, `ollama`               |
| `--dest`, `-d`            | Destination folder (default `WP/`)                             |
| `--dry`                   | Preview only — don't move files                                |
| `--full`, `-f`            | Flatten everything first, then reclassify                      |
| `--flatten`               | Just flatten subdirectories                                    |
| `--find-duplicates`       | List duplicates without moving                                 |
| `--dedupe` / `--no-dedupe`| Enable/disable duplicate removal (default: on)                 |
| `--parallel N`            | Worker count (default: number of CPU cores)                    |
| `--check-deps`            | Show optional dependency status                                |
| `--list-modes`            | List analysis modes with descriptions                          |
| `--set-dest DIR`          | Set destination folder and save to config                      |
| `--show-config`           | Show current configuration                                     |
| `--report FILE`           | Save a JSON report                                             |

### `run.sh` flags

```text
./run.sh                     Launch GUI
./run.sh --cli [--ft]        CLI mode (optionally free-threaded)
./run.sh --ft                Force free-threaded Python (CLI fallback if no PySide6)
./run.sh --bootstrap         Install all deps into .venv and .venv-t
./run.sh --bootstrap-ft      Install only free-threaded deps into .venv-t
./run.sh --version           Print launcher / project version
./run.sh --help              Show launcher help
```

---

## Tips & recipes

**Preview without moving anything**

```bash
./run.sh --cli --mode fusion --dry
```

**Force a full re-classification (flatten first)**

```bash
./run.sh --cli --full --mode fusion
```

**Use all your cores for hashing + classification**

```bash
./run.sh --cli --mode lowlevel --parallel 16
```

(The fusion/CLIP/Ollama modes automatically cap workers at 1 to avoid
loading the model four times into RAM.)

**Point Wanalizer at a different library**

```bash
./run.sh --cli --set-dest /media/wallpapers/Main
```

**Run on a remote Ollama server**

```bash
OLLAMA_URL=http://gpu-box:11434 ./run.sh --cli --mode ollama
```

**Clean a polluted tag registry**

```bash
./run.sh --cli --mode lowlevel --dry   # trigger the runtime warning
python -c "from wallpaper_analyzer.clean_tags import clean_all; clean_all()"
```

---

## Roadmap

- [x] MD5-only duplicate detection with cache (v3.0)
- [x] CLIP + LowLevel fusion mode (v3.0)
- [x] Free-threaded Python bootstrap (v3.0)
- [x] Per-category AI Suggest + Q&A wizard (v3.0)
- [ ] Optional perceptual-hash duplicate tier (pHash/dHash) as opt-in
- [ ] Plugin system for third-party analyzers
- [ ] Multi-library profiles (separate WP, Anime, Photos libraries)
- [ ] Optional GPU acceleration for CLIP via `--device cuda`

---

## License

[MIT](LICENSE) — see `LICENSE` for the full text.

---

<div align="center">

**Wanalizer** — *Because sorting ten thousand wallpapers by hand isn't a personality trait.*

</div>