"""Tests for the redesigned Reorder AI-rename system.

The Reorder page's "_RenameJob" worker must use the AI combined
classifier for tag-based strategies (same as the AI Rename dialog),
so the "Rename on move" / "Rename only" buttons produce content-aware
filenames, not colour-only ones.

These tests target the pure-function `_execute_rename_job` helper
extracted from `_RenameJob.run()` so they don't need Qt threads and
won't race with the rest of the suite's QApplication state.
"""
import inspect
import os


# ---------------------------------------------------------------------------
# Signature / API
# ---------------------------------------------------------------------------

def test_rename_job_has_ai_backend_parameter():
    """_RenameJob must accept ai_backend, ai_mode, ai_model parameters
    so the Reorder page can pass the user's AI backend selection."""
    from wallpaper_analyzer.gui.pages.reorganize import _RenameJob
    sig = inspect.signature(_RenameJob.__init__)
    params = sig.parameters
    assert "ai_backend" in params
    assert "ai_mode" in params
    assert "ai_model" in params
    # Defaults must be sensible.
    assert params["ai_backend"].default == "auto"
    assert params["ai_mode"].default == "auto"
    assert params["ai_model"].default is None


def test_execute_rename_job_is_pure_function():
    """`_execute_rename_job` is the pure-function core of the worker,
    testable without Qt threads."""
    from wallpaper_analyzer.gui.pages.reorganize import _execute_rename_job
    sig = inspect.signature(_execute_rename_job)
    assert "paths" in sig.parameters
    assert "strategy" in sig.parameters
    assert "ai_backend" in sig.parameters
    # Must accept an `on_progress` callback (used by the QThread wrapper).
    assert "on_progress" in sig.parameters


def test_rename_job_ai_defaults_are_ai_first():
    """Default backend must be 'auto' (CLIP → Ollama → Analyzer cascade).

    This is the user's stated requirement: AI should drive the rename,
    not the legacy colour-only heuristic.
    """
    from wallpaper_analyzer.gui.pages.reorganize import _RenameJob
    sig = inspect.signature(_RenameJob.__init__)
    assert sig.parameters["ai_backend"].default == "auto"


def test_ai_backend_options_include_ai_first():
    """The AI backend selector must include 'auto' as the first option."""
    # Mirrors the option list in reorganize.py — keep this in sync.
    _AI_BACKEND_OPTIONS = (
        ("auto",      "Auto (CLIP → Ollama → Analyzer)"),
        ("clip",      "CLIP"),
        ("ollama",    "Ollama"),
        ("heuristic", "Analyzer"),
    )
    assert _AI_BACKEND_OPTIONS[0][0] == "auto"
    # Auto must include both CLIP and Ollama in the cascade (AI-first).
    assert "CLIP" in _AI_BACKEND_OPTIONS[0][1]
    assert "Ollama" in _AI_BACKEND_OPTIONS[0][1]


# ---------------------------------------------------------------------------
# AIRenamer wiring inside the rename worker
# ---------------------------------------------------------------------------

def test_rename_job_uses_airenamer_for_tag_strategies(tmp_path, monkeypatch):
    """For TAG_BASED_STRATEGIES, the rename worker must build an AIRenamer
    and call detect_tags per file."""
    from wallpaper_analyzer.gui.pages.reorganize import _execute_rename_job
    from wallpaper_analyzer.rename import AIRenamer, TAG_BASED_STRATEGIES

    # Track AIRenamer construction + detect_tags calls.
    constructed = []
    detected = []

    class TrackingRenamer(AIRenamer):
        def __init__(self, **kw):
            constructed.append(kw.get("backend"))
            super().__init__(**kw)
        def detect_tags(self, path, category=None):
            detected.append((path, category))
            return ["anime", "cyberpunk"], "anime"

    monkeypatch.setattr(
        "wallpaper_analyzer.rename.AIRenamer", TrackingRenamer,
    )

    # Create test files.
    from PIL import Image
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    Image.new("RGB", (100, 100), (200, 50, 100)).save(p1, "JPEG")
    Image.new("RGB", (100, 100), (50, 200, 100)).save(p2, "JPEG")

    # Pick a TAG_BASED strategy.
    tag_strategy = next(iter(TAG_BASED_STRATEGIES))

    result = _execute_rename_job(
        paths=[str(p1), str(p2)],
        target_dir=None,
        strategy=tag_strategy,
        category="Anime",
        max_tags=3,
        ai_backend="auto",
        ai_mode="auto",
        ai_model=None,
    )

    # AIRenamer was constructed with the user's ai_backend.
    assert constructed == ["auto"]
    # detect_tags was called once per file with the user's category.
    assert len(detected) == 2
    for path, cat in detected:
        assert cat == "Anime"
    # And the result includes the AI log.
    assert result["ai_backend"] == "auto"
    assert "ai_log" in result
    assert len(result["ai_log"]) == 2


def test_rename_job_skips_ai_for_non_tag_strategies(tmp_path, monkeypatch):
    """For non-tag strategies, the rename worker must NOT instantiate an
    AIRenamer (no AI work needed)."""
    from wallpaper_analyzer.gui.pages.reorganize import _execute_rename_job
    from wallpaper_analyzer.rename import AIRenamer

    constructed = []

    class TrackingRenamer(AIRenamer):
        def __init__(self, **kw):
            constructed.append("x")
            super().__init__(**kw)

    monkeypatch.setattr(
        "wallpaper_analyzer.rename.AIRenamer", TrackingRenamer,
    )

    from PIL import Image
    p = tmp_path / "a.jpg"
    Image.new("RGB", (50, 50), (200, 100, 50)).save(p, "JPEG")

    # "sequential" is non-tag.
    result = _execute_rename_job(
        paths=[str(p)],
        target_dir=None,
        strategy="sequential",
        category="",
        max_tags=3,
        ai_backend="auto",
    )

    # No AIRenamer was constructed.
    assert constructed == []
    # And the result doesn't claim to have used AI.
    assert result.get("ai_backend") is None


# ---------------------------------------------------------------------------
# Content-aware tags (the user's main requirement)
# ---------------------------------------------------------------------------

def test_rename_job_returns_ai_content_tags_not_colours(tmp_path):
    """For tag-based strategies, the rename must produce filenames
    that include AI-detected CONTENT tags (e.g. anime, cyberpunk,
    tokyo-night), not just colour tokens (red, blue, widescreen)."""
    from wallpaper_analyzer.gui.pages.reorganize import _execute_rename_job
    from PIL import Image

    # Solid-colour image — would previously have produced only colour
    # tags via the legacy heuristic path.
    p = tmp_path / "test.jpg"
    Image.new("RGB", (200, 200), (180, 100, 220)).save(p, "JPEG")

    result = _execute_rename_job(
        paths=[str(p)],
        target_dir=None,
        strategy="category_tags",  # tag-based
        category="Anime",
        max_tags=3,
        ai_backend="auto",
    )

    assert "pairs" in result
    assert len(result["pairs"]) == 1
    old, new = result["pairs"][0]
    new_base = os.path.basename(new)
    # The new name must include "Anime" (category prefix) AND at least
    # one of the AI-detected content tags.
    new_lower = new_base.lower()
    assert "anime" in new_lower, f"missing category prefix: {new_base}"
    # The AI log should record the content tags detected.
    ai_log = result.get("ai_log") or []
    assert ai_log
    log_line = ai_log[0]
    # Must include content-aware tokens like 'anime' or 'cyberpunk'
    # — the AI combined classifier must surface them.
    content_tokens = {"anime", "cyberpunk", "illustration", "portrait",
                      "tokyo", "neon", "nature", "landscape", "minimalist"}
    assert any(tok in log_line for tok in content_tokens), (
        f"AI log doesn't include any content token: {log_line}"
    )


def test_rename_job_progress_callback_fires(tmp_path):
    """The on_progress callback must be called so the QThread wrapper
    can forward progress events to the UI."""
    from wallpaper_analyzer.gui.pages.reorganize import _execute_rename_job
    from PIL import Image

    events = []

    def cb(cur, total):
        events.append((cur, total))

    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    Image.new("RGB", (50, 50), (10, 10, 10)).save(p1, "JPEG")
    Image.new("RGB", (50, 50), (20, 20, 20)).save(p2, "JPEG")

    _execute_rename_job(
        paths=[str(p1), str(p2)],
        target_dir=None,
        strategy="sequential",
        category="",
        max_tags=3,
        on_progress=cb,
    )
    # Progress events fired for both tag detection AND rename apply.
    assert events, "on_progress was never called"
    # The final progress must reach `total` files.
    last = events[-1]
    assert last[0] == last[1], f"progress never reached total: {events}"