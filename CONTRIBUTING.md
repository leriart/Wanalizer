# Contributing to Wanalizer

Thanks for your interest in improving Wanalizer. This document covers
the basics of working with the codebase.

## Project layout

```
Wanalizer/
  wallpaper_analyzer/        Core Python package
    analyzers/               Pluggable analyzer modes (factory entry point)
    lowlevel/                Classical CV primitives (edges, texture, HOG, ...)
    gui/                     PySide6 desktop UI (9 pages)
    tools/                   One-off maintenance scripts
    data/                    Bundled non-code assets (tags.json)
  tests/                     Pytest smoke + functional tests
  pyproject.toml             PEP 621 packaging + tool config
  requirements.txt           pip-installable runtime hints
  run.sh                     Bash launcher (auto-creates .venv, PySide6, ...)
```

## Development setup

The launcher handles most of the setup:

```bash
./run.sh --bootstrap   # create .venv + .venv-t and install every dep
./run.sh --cli --help  # verify the CLI works
```

For an editable install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Coding conventions

- Python 3.10+ syntax. No walrus-in-default-argument tricks.
- Type hints on every new public function.
- No emojis in code, commits, or documentation.
- Docstrings for every module and public function.
- Keep `organize.py`, `classify.py`, and `gui/workers.py` slim: if a
  helper grows past ~150 lines, move it into its own module.

## Linting and tests

```bash
ruff check .
pytest -q
```

Both are configured in `pyproject.toml`.

## Adding a new analyzer mode

1. Subclass `wallpaper_analyzer.analyzers.base.BaseAnalyzer`.
2. Register the mode in `wallpaper_analyzer/analyzers/__init__.py`:
   add it to `AVAILABLE_MODES` and wire it into `get_analyzer()`.
3. Add a CLI description in `wallpaper_analyzer/cli.py::_list_modes`.
4. Add a smoke test under `tests/`.

## Reporting issues

Please open an issue at <https://github.com/leo/Wanalizer/issues>
with:

- Reproduction steps (CLI flags, sample image, expected vs. actual).
- Output of `./run.sh --cli --check-deps`.
- The relevant entries from the `Duplicates/` folder log when
  classification misbehaves.