<div align="center">

<h1>WANALIZER</h1>

<p>Intelligent wallpaper organization that runs entirely on your machine.</p>

<p>
  <img src="assets/collage.png" alt="Wanalizer collage preview" width="720">
</p>

<p>
  <img src="https://img.shields.io/badge/version-3.0.0-7aa2f7?style=for-the-badge&logo=none" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-9ece6a?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-e0af68?style=for-the-badge&logo=none" alt="License">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-7dcfff?style=for-the-badge&logo=none" alt="Platform">
</p>

<p>
  <img src="https://img.shields.io/badge/modes-4%20pipelines-f7768e?style=for-the-badge&logo=none" alt="Modes">
  <img src="https://img.shields.io/badge/signals-13%20weighted-bb9af7?style=for-the-badge&logo=none" alt="Signals">
  <img src="https://img.shields.io/badge/formats-50%2B%20image%20types-ff9e64?style=for-the-badge&logo=none" alt="Formats">
</p>

</div>

---

<div align="center">

## Four ways to understand an image

</div>

<table align="center">
  <tr>
    <td align="center">
      <img src="https://img.shields.io/badge/Low--Level_CV-c0caf5?style=flat-square&logo=none" alt="Low-Level CV"><br>
      <sub>Classical CV, no ML, any CPU</sub>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/CLIP-7aa2f7?style=flat-square&logo=none" alt="CLIP"><br>
      <sub>OpenAI zero-shot vision-language</sub>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Fusion-ff007c?style=flat-square&logo=none" alt="Fusion"><br>
      <sub>CLIP + CV combined, recommended</sub>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Ollama-9ece6a?style=flat-square&logo=none" alt="Ollama"><br>
      <sub>Local vision LLMs</sub>
    </td>
  </tr>
</table>

---

## Table of contents

1. [Highlights](#highlights)
2. [Why Wanalizer](#why-wanalizer)
3. [Quick start](#quick-start)
4. [Analysis modes](#analysis-modes)
5. [How classification works](#how-classification-works)
6. [Multi-signal combiner](#multi-signal-combiner)
7. [Per-category configuration](#per-category-configuration)
8. [Duplicate detection](#duplicate-detection)
9. [GUI overview](#gui-overview)
10. [Project structure](#project-structure)
11. [Installation](#installation)
12. [CLI reference](#cli-reference)
13. [Tips and recipes](#tips-and-recipes)
14. [Roadmap](#roadmap)
15. [License](#license)

---

<div align="center">

## Highlights

</div>

- **Four interchangeable analysis modes** sharing a single pipeline abstraction.
- **13-signal multi-scorer** with anti-pollution defences so noisy tag registries cannot hijack the result.
- **MD5 duplicate detection** with a per-folder cache. The second scan is essentially free; per-group keep/move/delete decisions.
- **Per-category configuration** via an interactive Q&A wizard or AI-suggested rules sampled from your own images.
- **Reorganize tab**: visual browser with thumbnails, aspect-ratio filter, rename dialog, and per-folder "expected" specifications.
- **50+ image formats** supported (JPEG, PNG, WebP, AVIF, HEIC, PSD, TIFF, RAW, SVG, PDF, ...) plus single-frame extraction for videos and animated images via `ffmpeg`.
- **True parallel hashing and classification** via `ProcessPoolExecutor` and an optional **free-threaded Python 3.14t** runtime for genuine no-GIL threading.
- **GUI and CLI from one package**. After `pip install .` two console scripts are available: `wanalyzer` and `wanalyzer-gui`.

---

<div align="center">

## Why Wanalizer

</div>

Other wallpaper organizers usually have one or more of these limitations:

- Require a remote API (privacy + latency).
- Require a GPU.
- Ship as a black box with no tuning knobs.
- Sort by filename heuristics only.

Wanalizer takes a different approach:

| Concern                    | Wanalizer                                                    |
|----------------------------|--------------------------------------------------------------|
| Offline / no network       | Low-Level CV runs on any CPU.                                |
| Semantic understanding     | CLIP, Fusion, and Ollama modes.                              |
| Detects duplicates         | MD5 with incremental cache.                                  |
| Per-category tuning        | Q&A wizard, AI Suggest, or hand-edited JSON.                 |
| Visual browser             | Reorganize page with thumbnails and filters.                 |
| Multi-signal classifier    | 13 weighted signals, pollution-aware.                        |
| True parallel CPU work     | ProcessPoolExecutor + optional free-threaded Python 3.14t.   |
| GUI and CLI from one code  | Shared pipeline, identical behaviour.                        |

---

<div align="center">

## Quick start

</div>

```bash
git clone https://github.com/leo/Wanalizer.git
cd Wanalizer

./run.sh                       # launch the GUI (auto-installs PySide6)
./run.sh --cli --help          # CLI help
./run.sh --cli --mode fusion --dry   # preview classification
./run.sh --cli --mode fusion --full  # flatten + classify for real
```

If you prefer to install via pip:

```bash
pip install .
wanalyzer --help
wanalyzer-gui
```

Free-threaded mode (true parallel threads, no GIL):

```bash
./run.sh --cli --ft --mode fusion --dry
```

The first run downloads Python 3.14t (~30 MB) automatically.

---

<div align="center">

## Analysis modes

</div>

| Mode                | What it uses                                                    | Strength                                                        | Dependencies                          | Speed                         |
|---------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|---------------------------------------|-------------------------------|
| **Low-Level CV**    | Canny / Sobel / Scharr edges, Otsu silhouettes, Hu moments, LBP, GLCM, Gabor, ORB / FAST, FFT, HOG, colour moments, composition, symmetry, pattern detection | Pure statistics, no ML, any CPU                                 | Pillow, numpy, scipy                  | Fast                          |
| **CLIP**            | OpenAI CLIP zero-shot (ViT-B/32 by default)                     | Semantic understanding ("this is Anime, not Gruvbox")           | torch, clip                           | Slow first image, cached      |
| **Fusion**          | Low-Level CV + CLIP in parallel, 13-signal fusion                | Best of both worlds; recommended when CLIP is available         | torch, clip, Pillow, numpy            | Slower than CLIP alone        |
| **Ollama**          | Local vision LLM via Ollama (LLaVA, MiniCPM-V, Llama 3.2 Vision, Moondream) | Natural-language descriptions, character recognition, NSFW      | requests + running Ollama server      | Medium                        |

Fusion is a strict superset of Low-Level: it gracefully degrades to Low-Level when CLIP cannot be loaded.

---

<div align="center">

## How classification works

</div>

1. **Source files** are scanned from the current directory (or any folder you pass via `--dest` / `--set-dest`).
2. If **dedupe is enabled**, MD5s are computed and exact duplicates are moved to a `Duplicates/` folder (cached for next time).
3. Each remaining file is analysed by the selected mode and turned into a **profile dict** with weights, style scores, theme scores, content detectors, and (where available) CLIP scores.
4. The **multi-signal combiner** scores every category against the profile and picks the best match, optionally weighted by the category's `expected` specification.
5. The file is **moved into the winning category**, optionally renamed according to the strategy in the Organize tab.

The full pipeline lives in `wallpaper_analyzer/organize.py` and is the same code path used by both the CLI and the GUI.

---

<div align="center">

## Multi-signal combiner

</div>

Every analyzer returns a profile dict. `wallpaper_analyzer/classify.py` combines **13 weighted signals** into a single confidence score per category:

| Signal      | Weight | What it measures                                                                |
|-------------|--------|---------------------------------------------------------------------------------|
| tags        | 0.10   | TF-IDF overlap between the image tags and each category's tags                  |
| palette     | 0.10   | Cosine similarity between colour distribution and category palette               |
| style       | 0.08   | Continuous bonuses for `anime_score`, `skin_fraction`, `periodicity_score`, ...  |
| content     | 0.05   | Specific detectors: `minecraft_score`, `pixel_art_score`, `minimalist_score`, ...|
| theme       | 0.07   | Named themes (Catppuccin, Dracula, TokyoNight, Nord, Gruvbox, ...)              |
| composition | 0.05   | Rule-of-thirds and subject-area heuristics                                      |
| quality     | 0.03   | Tenengrad + aesthetic + sharpness for `photo` / `illustration` categories       |
| pattern     | 0.04   | Periodicity and tile detection                                                  |
| size        | 0.03   | Aspect-ratio bucket (phone, vertical, wide, landscape)                          |
| prompt      | 0.08   | TF-IDF cosine between the generated prompt and the category prompt             |
| clip        | 0.25   | CLIP softmax probabilities (only when CLIP has run)                             |
| clip_nsfw   | 0.05   | CLIP-driven NSFW boost for `nsfw` / `18+` / `ecchi` categories                  |
| fingerprint | 0.20   | Z-score similarity against each category's learned CV fingerprint               |

The **fingerprint** signal is the most reliable because it compares the image's CV profile against the actual feature distribution of each category. The **clip** signal is the most discriminative when CLIP has run. Tag-based signals have explicit **anti-pollution defences** so a noisy registry cannot hijack the result.

---

<div align="center">

## Per-category configuration

</div>

Each category folder contains a `.category.json` describing what belongs there. Example for a Cyberpunk folder:

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

Three ways to create one:

1. **Interactive Q&A wizard** - Categories page, Configure (Q&A).
2. **AI Suggest** - Categories page, AI Suggest. Samples images, runs CLIP + CV, proposes an `expected` block you can review and save.
3. **Manual JSON edit**.

After configuration, the Reorganize tab shows the expected spec in the sidebar tooltip, and `classify.config_match_score` uses the spec as a soft tie-breaker when classifying.

---

<div align="center">

## Duplicate detection

</div>

The Duplicates tab detects exact duplicates via **MD5 hashing**. The hash cache lives in `.wallpaper_analyzer_hashes.json` inside the destination folder, so the second and subsequent scans only hash new files.

For each duplicate group:

- The **largest file** is kept (heuristic for highest quality).
- The rest can be **moved** to `Duplicates/` or **deleted** permanently.
- Per-group decisions or one-click "Move ALL" / "Delete ALL".

CLI shortcut:

```bash
./run.sh --cli --find-duplicates
```

For visually-similar but byte-different images, the underlying perceptual-hash primitives are still available in `wallpaper_analyzer.hashing` for advanced callers.

---

<div align="center">

## GUI overview

</div>

The GUI ships with **9 pages** wired into a single `QStackedWidget`:

| #  | Page          | What it does                                                          |
|----|---------------|-----------------------------------------------------------------------|
| 0  | Dashboard     | Library overview, mode picker, AI health, recent activity log         |
| 1  | Organize      | Run the classification pipeline; choose source/destination + workers  |
| 2  | Reorganize    | Visual browser: Move / Delete with aspect-ratio filter + rename dialog|
| 3  | AI Models     | Manage the CLIP engine and the Ollama server connection               |
| 4  | Categories    | Create / edit / configure categories; AI Suggest from sample images   |
| 5  | Tags          | Edit the global tag registry                                          |
| 6  | Duplicates    | MD5 duplicate scan, review groups, Move / Delete actions              |
| 7  | Dependencies  | Install optional packages (OpenCV, scikit-learn, PySide6, ...)        |
| 8  | Settings      | Application configuration (thresholds, theme, destination folder)     |

A typical session:

```
+-- Dashboard ---------------------------------------------------+
|  Total Files: 12 348     Categories: 23     Duplicates: 87     |
|  Mode: [x] Fusion (CLIP + LowLevel)  [ ] CLIP  ...              |
|  CLIP: Ready   Ollama: Connected (4 models)                     |
|  [ Open Organizer ]   [ AI Models ]   [ Refresh ]               |
+--------------------------------------------------------------+
        |
        v
+-- Organize ----------------------------------------------------+
|  Source: /home/.../Incoming                                    |
|  Destination: /home/.../WP                                     |
|  Mode: Fusion  Workers: 8  Min quality: 0.0                    |
|  [x] Dry run   [x] Find duplicates   [ ] Full reset             |
|  [#############-----]  67 %   image_0432.jpg -> Cyberpunk      |
|  [ Organize Now ]   [ Dry Run ]   [ Stop ]                      |
+--------------------------------------------------------------+
```

---

<div align="center">

## Project structure

</div>

```
Wanalizer/
  run.sh                              Entry point (GUI or CLI)
  pyproject.toml                      Build config + tool settings (ruff, pytest)
  requirements.txt                    pip-installable runtime hints
  CHANGELOG.md                        Release notes
  CONTRIBUTING.md                     Development guide
  tags.json                           Global tag registry (editable copy)
  wallpaper_analyzer/                 Core package
    __init__.py                       Project metadata
    py.typed                          Type-checker marker
    analyzers/                        Pluggable analyzer modes
      base.py                         Abstract BaseAnalyzer
      lowlevel_mode.py                Low-Level CV analyzer
      fusion_mode.py                  CLIP + LowLevel fusion
    lowlevel/                         Classical CV primitives
      edges.py                        Canny, Sobel, Scharr, orientation
      silhouettes.py                  Otsu, adaptive thresholding
      contours.py                     Hu moments, symmetry
      texture.py                      LBP, GLCM, Gabor filters
      features.py                     ORB, FAST, Shi-Tomasi
      fourier.py                      FFT, frequency distribution
      hog.py                          Histogram of Oriented Gradients
      color_advanced.py               Colour moments, harmony, LAB / HSV stats
      composition.py                  Rule of thirds, depth, saliency
      quality_advanced.py             Tenengrad, BRISQUE-like, pHash, noise
      subject.py                      Largest component, foreground / background
      pattern.py                      Periodicity, tile detection, complexity
      symmetry_advanced.py            Bilateral, rotational, diagonal
      category_profile.py             Per-category CV fingerprints + scoring
    gui/                              PySide6 desktop UI
      __init__.py                     Main window + theme
      __main__.py                     `python -m wallpaper_analyzer.gui`
      theme.py                        Red / black / white QSS stylesheet
      widgets.py                      Custom widgets
      workers.py                      Background QThread workers
      rename_dialog.py                Batch rename dialog
      category_config_dialog.py       Per-category Q&A wizard + AI review
      pages/
        dashboard.py                  Overview, mode selection, health
        organize.py                   Classification pipeline runner
        reorganize.py                 Visual file browser, filter, rename
        ai_models.py                  CLIP and Ollama model management
        categories.py                 Category folder management
        tags.py                       Tag registry editor
        duplicates.py                 Duplicate scan UI
        dependencies.py               Package installer
        settings.py                   Configuration panel
    tools/                            One-off maintenance scripts
      regenerate_categories.py       Rebuild every category config via Ollama
    data/                             Bundled non-code assets
      tags.json                       Default tag registry shipped with the wheel
    clip_client.py                    CLIP engine and CLIPAnalyzer
    ollama_client.py                  Ollama vision LLM analyzer
    categories.py                     Category management
    category_config.py                Per-category expected-configuration system
    classify.py                       Multi-signal classification combiner
    clean_tags.py                     Tag cleanup utilities
    color.py                          HSV / LAB / RGB helpers
    duplicates.py                     MD5-based duplicate detection
    formats.py                        Format detection (50+)
    hashing.py                        Perceptual hashing helpers
    minimal_ai.py                     Minimal AI tag / prompt generator
    organize.py                       Main classification pipeline + factory
    parallel.py                       Parallel processing helpers
    profile.py                        Image profile builder
    prompt_generator.py               AI prompt generation
    quality.py                        Sharpness, aesthetic scoring
    rename.py                         Rename strategies
    settings.py                       Configuration
    tag_policies.py                   Tag policy enforcement
    tag_suggester.py                  AI tag suggestion
    tags.py                           Tag registry
    cli.py                            Command-line interface
  tests/                              Pytest smoke tests
```

---

<div align="center">

## Installation

</div>

### Option A - launcher (recommended)

The launcher auto-creates `.venv`, installs PySide6, picks the right Python interpreter, and falls back to a free-threaded runtime when asked:

```bash
./run.sh                  # GUI
./run.sh --cli --help     # CLI
./run.sh --bootstrap      # install everything upfront
./run.sh --bootstrap-ft   # install only free-threaded deps into .venv-t
```

### Option B - manual pip install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: full CV support
pip install opencv-python-headless imagehash scikit-learn

# Optional: GUI
pip install PySide6

# Optional: CLIP + PyTorch (CPU build, ~2 GB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git

# Or install everything from the extras
pip install ".[all]"
```

After `pip install .`, two console scripts are available:

```bash
wanalyzer       # CLI
wanalyzer-gui   # GUI
```

### External tools

| Tool      | Used for                           | Required?                                  |
|-----------|------------------------------------|--------------------------------------------|
| `ffmpeg`  | Single-frame extraction for videos | Optional - graceful skip if missing        |

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Environment variables

Settings can be overridden via environment variables. They take precedence over `.wallpaper_analyzer.json` but are overridden by explicit CLI flags.

| Variable                  | Equivalent setting key                  |
|---------------------------|------------------------------------------|
| `WANALIZER_DEST`          | `dest_dir`                               |
| `WANALIZER_MODE`          | `organize_mode`                          |
| `WANALIZER_THEME`         | `theme`                                  |
| `OLLAMA_URL`              | `ollama_url`                             |
| `WANALIZER_OLLAMA_URL`    | `ollama_url`                             |
| `OLLAMA_MODEL`            | `ollama_model`                           |
| `WANALIZER_OLLAMA_MODEL`  | `ollama_model`                           |
| `WANALIZER_CLIP_MODEL`    | `clip_model`                             |
| `WANALIZER_TAGS`          | path to a custom `tags.json` file        |
| `QT_QPA_PLATFORM`         | Qt platform plugin (wayland / xcb / offscreen) |

---

<div align="center">

## CLI reference

</div>

```
wanalyzer [-h] [--mode {lowlevel,clip,fusion,ollama}] [--dest DIR]
          [--dry] [--full] [--flatten] [--find-duplicates]
          [--dedupe | --no-dedupe] [--parallel N]
          [--check-deps] [--list-modes]
          [--set-dest DIR] [--show-config] [--report FILE]
```

| Flag                         | Description                                                              |
|------------------------------|--------------------------------------------------------------------------|
| `--mode`, `-m`               | `lowlevel` (default), `clip`, `fusion`, `ollama`                         |
| `--dest`, `-d`               | Destination folder (default `WP/`)                                       |
| `--dry`                      | Preview only - do not move files                                         |
| `--full`, `-f`               | Flatten everything first, then reclassify                                |
| `--flatten`                  | Just flatten subdirectories                                              |
| `--find-duplicates`          | List duplicates without moving                                           |
| `--dedupe` / `--no-dedupe`   | Enable / disable duplicate removal (default: on)                         |
| `--parallel N`               | Worker count (default: number of CPU cores)                              |
| `--check-deps`               | Show optional dependency status                                          |
| `--list-modes`               | List analysis modes with descriptions                                    |
| `--set-dest DIR`             | Set destination folder and save to config                                |
| `--show-config`              | Show current configuration                                               |
| `--report FILE`              | Save a JSON report                                                       |

### `run.sh` flags

```
./run.sh                     Launch GUI
./run.sh --cli [--ft]        CLI mode (optionally free-threaded)
./run.sh --ft                Force free-threaded Python (CLI fallback if no PySide6)
./run.sh --bootstrap         Install all deps into .venv and .venv-t
./run.sh --bootstrap-ft      Install only free-threaded deps into .venv-t
./run.sh --version           Print launcher / project version
./run.sh --help              Show launcher help
```

---

<div align="center">

## Tips and recipes

</div>

Preview without moving anything:

```bash
./run.sh --cli --mode fusion --dry
```

Force a full re-classification (flatten first):

```bash
./run.sh --cli --full --mode fusion
```

Use all your cores for hashing and classification:

```bash
./run.sh --cli --mode lowlevel --parallel 16
```

Fusion / CLIP / Ollama modes automatically cap workers at 1 to avoid loading the model four times into RAM.

Point Wanalizer at a different library:

```bash
./run.sh --cli --set-dest /media/wallpapers/Main
```

Run on a remote Ollama server:

```bash
OLLAMA_URL=http://gpu-box:11434 ./run.sh --cli --mode ollama
```

Clean a polluted tag registry:

```bash
./run.sh --cli --mode lowlevel --dry
python -c "from wallpaper_analyzer.clean_tags import clean_all; clean_all()"
```

Run the smoke tests after a change:

```bash
pip install ".[dev]"
pytest -q
```

Lint the codebase:

```bash
pip install ".[dev]"
ruff check .
```

---

<div align="center">

## Roadmap

</div>

- [x] MD5-only duplicate detection with cache (v3.0)
- [x] CLIP + LowLevel fusion mode (v3.0)
- [x] Free-threaded Python bootstrap (v3.0)
- [x] Per-category AI Suggest + Q&A wizard (v3.0)
- [x] Bundled `data/tags.json` and `py.typed` marker (v3.1)
- [x] Unified `wallpaper_analyzer.analyzers.get_analyzer` factory (v3.1)
- [x] Environment-variable overrides for settings (v3.1)
- [ ] Optional perceptual-hash duplicate tier (pHash / dHash) as opt-in
- [ ] Plugin system for third-party analyzers
- [ ] Multi-library profiles (separate WP, Anime, Photos libraries)
- [ ] Optional GPU acceleration for CLIP via `--device cuda`

---

<div align="center">

## License

</div>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-9ece6a?style=for-the-badge&logo=none" alt="MIT License"></a>
</p>

<p align="center">
  Made for organized desktops everywhere.
</p>
