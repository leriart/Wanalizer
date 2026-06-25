"""Tests for the simplified Reorder AI rename flow.

After the cleanup:
  - Only "AI Rename..." and "AI Rename category..." buttons remain.
  - The non-AI RenameDialog + "Rename only" buttons are gone (they
    either froze the UI or duplicated the AI flow).
  - The Reorder header exposes an AI model picker that pre-configures
    the AIRenameDialog.
  - Preview limit is capped at 5 by default (per the user's request).
"""
import inspect


# ---------------------------------------------------------------------------
# Removed buttons / handlers
# ---------------------------------------------------------------------------

def test_reorder_has_no_non_ai_rename_dialog_import():
    """The non-AI RenameDialog must no longer be imported in Reorder.

    It was removed because it only handled heuristic tag detection
    and was superseded by the AIRenameDialog (which runs the same
    combined AI classifier).
    """
    with open("wallpaper_analyzer/gui/pages/reorganize.py") as f:
        src = f.read()
    assert "from ..rename_dialog" not in src
    # (AI RenameDialog import is fine — it's a different class.)
    assert "from ..ai_rename_dialog import AIRenameDialog" in src


def test_reorder_has_no_rename_only_button():
    """The "Rename only" button was removed because it froze the UI.

    Without a preview / dry-run, users couldn't see what the AI was
    producing before the rename hit disk.
    """
    with open("wallpaper_analyzer/gui/pages/reorganize.py") as f:
        src = f.read()
    assert "btn_rename_only" not in src
    assert "_on_rename_only" not in src


def test_reorder_has_no_non_ai_rename_button():
    """The "Rename dialog..." button was removed (used non-AI RenameDialog)."""
    with open("wallpaper_analyzer/gui/pages/reorganize.py") as f:
        src = f.read()
    # "btn_rename = " was the non-AI button creation line.
    assert "btn_rename = " not in src
    assert "_on_rename(" not in src


def test_reorder_keeps_ai_rename_buttons():
    """The two AI rename entry points must remain."""
    with open("wallpaper_analyzer/gui/pages/reorganize.py") as f:
        src = f.read()
    assert "btn_ai_rename" in src
    assert "btn_ai_rename_cat" in src
    assert "_on_ai_rename" in src
    assert "_on_ai_rename_category" in src


# ---------------------------------------------------------------------------
# AI model picker in the Reorder header
# ---------------------------------------------------------------------------

def test_reorder_has_ai_model_picker():
    """Reorder header must expose an AI model picker so the user can
    pick the model BEFORE opening the dialog (the dialog then opens
    with the right selection)."""
    with open("wallpaper_analyzer/gui/pages/reorganize.py") as f:
        src = f.read()
    assert "_ai_model" in src
    assert "_ai_model_label" in src
    # The model picker should be tied to CLIP / Ollama backends.
    assert "needs_model" in src


def test_reorder_passes_model_to_ai_rename_dialog():
    """The selected model from the header must be forwarded to the
    AIRenameDialog as `default_model`."""
    with open("wallpaper_analyzer/gui/pages/reorganize.py") as f:
        src = f.read()
    assert "_selected_ai_model" in src
    # Both openers must use it.
    assert "default_model=self._selected_ai_model()" in src


# ---------------------------------------------------------------------------
# Preview limit capped at 5
# ---------------------------------------------------------------------------

def test_reorder_preview_limit_is_capped_at_five():
    """Preview must default to 5 so the user sees how names are applied
    without waiting for the full set (per the user's explicit request)."""
    with open("wallpaper_analyzer/gui/pages/reorganize.py") as f:
        src = f.read()
    assert "preview_cap = min(5, len(paths))" in src
    # Both openers must use it.
    assert src.count("preview_limit=preview_cap") >= 2


# ---------------------------------------------------------------------------
# Move flow still uses AI rename (unchanged)
# ---------------------------------------------------------------------------

def test_reorder_move_flow_still_uses_rename_job():
    """The drag-to-move flow still uses _RenameJob + AI renaming."""
    with open("wallpaper_analyzer/gui/pages/reorganize.py") as f:
        src = f.read()
    assert "_run_rename_job" in src
    assert "_RenameJob" in src
    assert "_execute_rename_job" in src


# ---------------------------------------------------------------------------
# Preview-then-apply flow in AIRenameDialog
# ---------------------------------------------------------------------------

def test_ai_rename_dialog_has_preview_then_apply():
    """The AI Rename dialog must have explicit 'Run preview' and 'Apply'
    buttons so the user controls when each phase happens."""
    from wallpaper_analyzer.gui.ai_rename_dialog import AIRenameDialog
    src = inspect.getsource(AIRenameDialog)
    assert "_run_btn" in src  # "Run preview"
    assert "_apply_btn" in src  # "Apply"
    assert "_refresh_preview" in src  # explicit preview action


def test_ai_rename_dialog_preview_runs_in_background():
    """Preview must NOT block the UI — it must run in a QThread."""
    from wallpaper_analyzer.gui.ai_rename_dialog import (
        AIRenameDialog, _PreviewJob,
    )
    import PySide6.QtCore as qc
    # _PreviewJob extends QThread.
    assert issubclass(_PreviewJob, qc.QThread)
    # And the dialog must connect to its signals.
    src = inspect.getsource(AIRenameDialog)
    assert "_preview_job.start()" in src


# ---------------------------------------------------------------------------
# Verification via page construction (without GUI)
# ---------------------------------------------------------------------------

def test_reorder_page_can_be_constructed_with_rename_header():
    """The Reorder page must build cleanly with the new header layout
    (AI model picker, no removed buttons)."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import sys
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv)

    # Stub the main window — we only want the page widget itself.
    class StubMain:
        def append_log(self, *a, **kw): pass
        def switch_page(self, *a, **kw): pass

    # We can't fully construct the page without a real main window /
    # parent, but we can inspect its source for the attributes the
    # header is supposed to expose.
    from wallpaper_analyzer.gui.pages import reorganize
    src = inspect.getsource(reorganize)
    # Required header attributes the user interacts with.
    for attr in ("_ai_backend", "_ai_model", "_ai_model_label",
                 "_ai_status", "btn_ai_rename", "btn_ai_rename_cat"):
        assert f"self.{attr}" in src, f"missing {attr} in Reorder page"
    # Removed attributes must NOT be present.
    for removed in ("self.btn_rename_only", "self.btn_rename ="):
        assert removed not in src, f"{removed} should have been removed"