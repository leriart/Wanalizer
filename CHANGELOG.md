# Changelog

All notable changes to Wanalizer are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `py.typed` marker so downstream consumers get type information.
- `data/tags.json` shipped inside the package so the tag registry is
  available when the project is installed via `pip install .`.
- `tests/` directory with smoke tests covering public APIs and the
  analyzer factory.
- Environment variable overrides for settings (`WANALIZER_MODE`,
  `WANALIZER_DEST`, `WANALIZER_OLLAMA_URL`, `OLLAMA_URL`,
  `WANALIZER_OLLAMA_MODEL`, `OLLAMA_MODEL`, `WANALIZER_CLIP_MODEL`,
  `WANALIZER_THEME`, `WANALIZER_TAGS`).
- `pyproject.toml` `[tool.ruff]` and `[tool.pytest.ini_options]`
  configuration; `[dev]` extra with `pytest`, `pytest-cov`, `ruff`.
- `wallpaper_analyzer.analyzers.get_analyzer()` factory unifying all
  four modes (`lowlevel`, `fusion`, `clip`, `ollama`).

### Changed
- `__version__` is now read from installed-package metadata via
  `importlib.metadata.version`, with the literal fallback used only
  when running from a source checkout.
- `run.sh --version` works without a pre-existing virtualenv and on
  Windows.
- `.gitignore` extended with Python tooling caches (mypy, pytest,
  ruff, coverage) and IDE files.

### Removed
- Redundant `setup.py` (everything is declared in `pyproject.toml`).

### Fixed
- Unused imports in `cli.py` and `organize.py`.
- `parallel.py` now imports `pickle` at module top instead of a
  misleading "late import" at the bottom.

## [3.0.0] - 2026-06-23

### Added
- **Fusion mode**: CLIP + LowLevel CV combined through a 13-signal
  multi-scorer with anti-pollution defences.
- **MD5-only duplicate detection** with per-folder cache, scalable via
  `ProcessPoolExecutor`.
- **Free-threaded Python 3.14t bootstrap** via `run.sh --ft`.
- **Per-category AI Suggest** and an interactive Q&A wizard for
  authoring `.category.json` files.
- 9-page Qt6 desktop UI: Dashboard, Organize, Reorganize, AI Models,
  Categories, Tags, Duplicates, Dependencies, Settings.

### Changed
- Cleaner separation between analyzers (`analyzers/`), low-level CV
  primitives (`lowlevel/`), GUI (`gui/`), and tools (`tools/`).
- True parallel hashing and classification regardless of GIL.

### Removed
- Multi-tier duplicate detection (reencode / resize / similar) in
  favour of fast MD5-only matching. Perceptual hashes are still
  available via `hashing.py` for callers that need them.