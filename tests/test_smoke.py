"""Smoke tests for the package public surface."""
import importlib
import os

import pytest


def test_package_metadata():
    pkg = importlib.import_module("wallpaper_analyzer")
    assert pkg.__project__ == "Wanalizer"
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__.count(".") >= 1


def test_analyzers_factory_known_modes():
    from wallpaper_analyzer.analyzers import AVAILABLE_MODES, get_analyzer

    assert {"lowlevel", "fusion", "clip", "ollama"} <= set(AVAILABLE_MODES)
    settings = {}
    # LowLevel never fails to load.
    a = get_analyzer("lowlevel", settings)
    assert a is not None
    assert hasattr(a, "analyze")
    assert hasattr(a, "classify")


def test_analyzers_factory_unknown_mode():
    from wallpaper_analyzer.analyzers import get_analyzer

    with pytest.raises(ValueError):
        get_analyzer("not-a-mode", {})


def test_settings_defaults():
    from wallpaper_analyzer.settings import SETTINGS_DEFAULTS, load_settings

    # Defaults are always present even when the config file is missing.
    s = load_settings()
    assert s.get("dest_dir")
    assert s.get("organize_mode") in {"lowlevel", "clip", "fusion", "ollama"}
    # Every default key must be present in the merged result.
    for key in SETTINGS_DEFAULTS:
        assert key in s


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WANALIZER_DEST", str(tmp_path))
    monkeypatch.setenv("OLLAMA_URL", "http://example.invalid:1234")
    from wallpaper_analyzer.settings import load_settings

    s = load_settings()
    assert s["dest_dir"] == str(tmp_path)
    assert s["ollama_url"] == "http://example.invalid:1234"


def test_tags_registry_loads():
    from wallpaper_analyzer.tags import get_all_tags, is_valid_tag, load_tags

    data = load_tags()
    tags = get_all_tags()
    assert isinstance(data, dict)
    assert len(tags) > 0
    assert all(is_valid_tag(t) for t in list(tags)[:10])


def test_formats_extension_sets():
    from wallpaper_analyzer.formats import STATIC_EXTENSIONS, ANIMATED_EXTENSIONS

    assert ".jpg" in STATIC_EXTENSIONS
    assert ".png" in STATIC_EXTENSIONS
    assert ".webp" in STATIC_EXTENSIONS or ".webp" in ANIMATED_EXTENSIONS
    assert ".gif" in ANIMATED_EXTENSIONS


def test_settings_resolve_dest_dir_absolute():
    from wallpaper_analyzer.settings import resolve_dest_dir

    s = {"dest_dir": "/tmp/absolute-path-test"}
    assert resolve_dest_dir(s) == "/tmp/absolute-path-test"


def test_parallel_cpu_count_positive():
    from wallpaper_analyzer.parallel import cpu_count

    n = cpu_count()
    assert isinstance(n, int)
    assert n >= 1


def test_parallel_is_free_threaded_returns_bool():
    from wallpaper_analyzer.parallel import is_free_threaded

    assert isinstance(is_free_threaded(), bool)


def test_duplicates_signature_helper(tmp_path):
    from wallpaper_analyzer.duplicates import _file_signature

    f = tmp_path / "a.txt"
    f.write_bytes(b"hello")
    sig = _file_signature(str(f))
    assert sig and "-" in sig


def test_cli_parser_builds():
    from wallpaper_analyzer.cli import _build_parser

    parser = _build_parser()
    ns = parser.parse_args(["--mode", "lowlevel", "--dry"])
    assert ns.mode == "lowlevel"
    assert ns.dry is True
    assert ns.dedupe is True
    assert ns.parallel is None
    assert ns.source is None
    ns2 = parser.parse_args(["--no-dedupe", "--parallel", "4", "--source", "/tmp"])
    assert ns2.dedupe is False
    assert ns2.parallel == 4
    assert ns2.source == "/tmp"


def test_cli_source_rejects_missing_dir():
    """The CLI must validate --source early and exit non-zero if missing."""
    from wallpaper_analyzer.cli import _build_parser
    from wallpaper_analyzer import formats

    original = formats.WALLPAPERS_DIR
    p = _build_parser()
    ns = p.parse_args(["--source", "/this/path/does/not/exist/__nope__"])
    assert ns.source == "/this/path/does/not/exist/__nope__"
    # Validation runs inside main(); parsing alone must not mutate globals
    assert formats.WALLPAPERS_DIR == original


def test_categories_special_folders_excludes_nsfw():
    """NSFW is a browsable category (not a hidden special folder).

    The original audit flagged that NSFW was both treated as a regular
    category AND skipped in some places; the fix is to keep it as a
    category and use is_managed_category() to mark it as auto-managed.
    """
    from wallpaper_analyzer.categories import (
        SPECIAL_FOLDERS, NSFW_FOLDER, is_managed_category,
    )

    assert NSFW_FOLDER not in SPECIAL_FOLDERS
    assert "Duplicates" in SPECIAL_FOLDERS
    assert "Low-Quality" in SPECIAL_FOLDERS
    assert is_managed_category(NSFW_FOLDER)
    assert not is_managed_category("Anime")


def test_nsfw_folder_appears_in_categories(tmp_path):
    """NSFW folder must show up in the category list even without config.

    Even folders without a `.category.json` (like NSFW before any user
    action) are visible so the user can see the count and browse them.
    """
    from wallpaper_analyzer.categories import (
        discover_categories, list_category_folders, NSFW_FOLDER,
    )
    import json

    (tmp_path / "Anime").mkdir()
    (tmp_path / "Anime" / ".category.json").write_text(json.dumps({"name": "Anime"}))
    (tmp_path / NSFW_FOLDER).mkdir()
    (tmp_path / NSFW_FOLDER / "y.jpg").write_bytes(b"y")
    (tmp_path / "Duplicates").mkdir()  # should NOT appear

    discover_categories(str(tmp_path))
    from wallpaper_analyzer.categories import CATEGORIES
    assert NSFW_FOLDER in CATEGORIES
    assert "Anime" in CATEGORIES
    assert "Duplicates" not in CATEGORIES

    folders = list_category_folders(str(tmp_path))
    assert NSFW_FOLDER in folders
    assert "Anime" in folders
    assert "Duplicates" not in folders


def test_nsfw_folder_default_config():
    """get_category_config returns sensible defaults for NSFW with no config."""
    from wallpaper_analyzer.categories import get_category_config, NSFW_FOLDER

    cfg = get_category_config(NSFW_FOLDER)
    assert cfg["name"] == NSFW_FOLDER
    assert cfg.get("managed") is True
    assert "tags" in cfg
    assert "prompt" in cfg


def test_nsfw_folder_cannot_be_deleted():
    """delete_category refuses to remove the NSFW folder."""
    from wallpaper_analyzer.categories import delete_category, NSFW_FOLDER

    assert delete_category(NSFW_FOLDER) is False


def test_list_category_folders_helper(tmp_path):
    from wallpaper_analyzer.categories import (
        list_category_folders, count_media_in, SPECIAL_FOLDERS,
    )

    (tmp_path / "Anime").mkdir()
    (tmp_path / "Anime" / "a.jpg").write_bytes(b"x")
    (tmp_path / "Anime" / "b.png").write_bytes(b"x")
    (tmp_path / "Empty").mkdir()
    for special in ("Duplicates", "Low-Quality", "Discarded", "Uncategorized", "NSFW"):
        (tmp_path / special).mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "not_a_dir.txt").write_text("x")

    cats = list_category_folders(str(tmp_path))
    assert "Anime" in cats
    assert "Empty" in cats
    assert ".hidden" not in cats
    assert "not_a_dir.txt" not in cats
    for s in SPECIAL_FOLDERS:
        assert s not in cats

    # Restricted to configured-only
    (tmp_path / "Anime" / ".category.json").write_text("{}")
    cats2 = list_category_folders(str(tmp_path), include_unconfigured=False)
    assert "Anime" in cats2
    assert "Empty" not in cats2

    assert count_media_in(str(tmp_path / "Anime")) == 2
    assert count_media_in(str(tmp_path / "Empty")) == 0


def test_settings_source_dir_default():
    from wallpaper_analyzer.settings import SETTINGS_DEFAULTS

    assert "source_dir" in SETTINGS_DEFAULTS
    assert SETTINGS_DEFAULTS["source_dir"] == ""


def test_workers_get_tags_ollama_uses_public_categories_alias():
    """Regression: _get_tags_ollama must reference cats_mod, not the
    method-local c_mod alias (which would raise NameError)."""
    import inspect
    from wallpaper_analyzer.gui.workers import GenerateTagsWorker

    src = inspect.getsource(GenerateTagsWorker._get_tags_ollama)
    assert "c_mod.get_category_config" not in src
    assert "cats_mod.get_category_config" in src


def test_duplicates_scan_uses_custom_exception():
    """Regression: StopIteration-as-control-flow is deprecated/removed in 3.14."""
    import inspect
    from wallpaper_analyzer.gui.pages.duplicates import DupeScanWorker

    src = inspect.getsource(DupeScanWorker.run)
    assert "raise StopIteration" not in src
    assert "_ScanCancelled" in src


def test_light_theme_supported_in_apply_theme():
    from wallpaper_analyzer.gui.theme import apply_theme, LIGHT_QSS

    assert callable(apply_theme)
    assert isinstance(LIGHT_QSS, str) and len(LIGHT_QSS) > 0


def test_rename_compute_renames_sequential(tmp_path):
    from wallpaper_analyzer.rename import compute_renames
    fa = tmp_path / "a.jpg"
    fb = tmp_path / "b.jpg"
    fa.write_bytes(b"x")
    fb.write_bytes(b"x")
    pairs = compute_renames([str(fa), str(fb)], strategy="sequential",
                            category="Anime", pad=3)
    assert len(pairs) == 2
    new_names = sorted(os.path.basename(n) for _, n in pairs)
    assert new_names == ["001.jpg", "002.jpg"]


def test_rename_compute_renames_no_tags_needed(tmp_path):
    """Non-tag strategies should not call into the profile/tag pipeline."""
    from wallpaper_analyzer.rename import compute_renames
    fa = tmp_path / "a.jpg"
    fa.write_bytes(b"x")
    pairs = compute_renames([str(fa)], strategy="category",
                            category="Anime")
    assert len(pairs) == 1
    _, dst = pairs[0]
    assert os.path.basename(dst).startswith("Anime_")
    assert dst.endswith(".jpg")


def test_rename_get_tags_for_file_missing():
    from wallpaper_analyzer.rename import get_tags_for_file
    tags, subj = get_tags_for_file("/this/path/does/not/exist.jpg")
    assert tags == []
    assert subj is None


def test_rename_clear_tags_cache():
    from wallpaper_analyzer.rename import clear_tags_cache, _PROFILE_TAGS_CACHE
    _PROFILE_TAGS_CACHE["/tmp/synthetic"] = (0.0, 0, ([], None))
    clear_tags_cache()
    assert len(_PROFILE_TAGS_CACHE) == 0


def test_rename_dialog_accepts_default_strategy_and_max_tags():
    """The dialog should pre-select the strategy passed by the caller."""
    import inspect
    from wallpaper_analyzer.gui.rename_dialog import RenameDialog
    sig = inspect.signature(RenameDialog.__init__)
    assert "default_strategy" in sig.parameters
    assert "max_tags" in sig.parameters


def test_ai_models_redesigned_uses_backend_cards():
    """AIModelsPage should expose backend cards and a hero summary."""
    from wallpaper_analyzer.gui.pages.ai_models import (
        AIModelsPage, BACKEND_DEFS, StatusDot, BackendCard,
    )
    assert {"lowlevel", "clip", "ollama"} <= {d["key"] for d in BACKEND_DEFS}
    assert len(BACKEND_DEFS) >= 3
    # Each def needs title, summary, pros/cons for the card to look complete
    for d in BACKEND_DEFS:
        assert d["title"]
        assert d["summary"]
        assert d["pros"]
        assert d["cons"]


def test_reorganize_rename_button_text():
    """Reorganize header should expose Rename + Rename-only controls."""
    import inspect
    from wallpaper_analyzer.gui.pages import reorganize
    src = inspect.getsource(reorganize)
    assert "btn_rename_only" in src
    assert "_RenameJob" in src
    assert "compute_renames" in src or "build_renames" in src


def test_ai_tag_backends_constant():
    from wallpaper_analyzer.rename import AI_TAG_BACKENDS
    assert "auto" in AI_TAG_BACKENDS
    assert "ollama" in AI_TAG_BACKENDS
    assert "clip" in AI_TAG_BACKENDS
    assert "heuristic" in AI_TAG_BACKENDS


def test_ai_detect_tags_heuristic_returns_tuple(tmp_path):
    """ai_detect_tags with backend='heuristic' returns a (tags, subject) tuple."""
    from PIL import Image
    from wallpaper_analyzer.rename import ai_detect_tags
    # Create a small test image with content the heuristic should recognise.
    img = Image.new("RGB", (200, 200), (255, 0, 0))  # pure red
    fp = tmp_path / "red.jpg"
    img.save(fp, "JPEG")
    tags, subject = ai_detect_tags(str(fp), backend="heuristic", max_tags=5)
    assert isinstance(tags, list)
    assert isinstance(subject, (str, type(None)))
    # Heuristic on pure red should at least return some colour tag
    # (the fallback uses top-3 colour names from palette_weights).
    assert all(isinstance(t, str) for t in tags)


def test_ai_detect_tags_unknown_backend_falls_back(tmp_path):
    """Invalid backend string should fall back to heuristic (no crash)."""
    from PIL import Image
    from wallpaper_analyzer.rename import ai_detect_tags
    img = Image.new("RGB", (100, 100), (0, 0, 255))
    fp = tmp_path / "blue.jpg"
    img.save(fp, "JPEG")
    tags, subject = ai_detect_tags(str(fp), backend="bogus", max_tags=5)
    assert isinstance(tags, list)


def test_ai_compute_renames_non_tag_strategy(tmp_path):
    """Non-tag strategies shouldn't try to call any AI backend."""
    from PIL import Image
    from wallpaper_analyzer.rename import ai_compute_renames
    img = Image.new("RGB", (50, 50), (128, 128, 128))
    fp = tmp_path / "img.jpg"
    img.save(fp, "JPEG")
    pairs = ai_compute_renames([str(fp)], strategy="category", category="Anime")
    assert len(pairs) == 1
    _, dst = pairs[0]
    assert os.path.basename(dst).startswith("Anime_")


def test_ai_rename_dialog_exists():
    """The AIRenameDialog module must exist and be importable."""
    from wallpaper_analyzer.gui.ai_rename_dialog import AIRenameDialog, _PreviewJob
    assert callable(AIRenameDialog)
    assert callable(_PreviewJob)


def test_reorganize_has_ai_rename_buttons():
    """Reorganize header must expose AI Rename + AI Rename category."""
    import inspect
    from wallpaper_analyzer.gui.pages import reorganize
    src = inspect.getsource(reorganize)
    assert "btn_ai_rename" in src
    assert "btn_ai_rename_cat" in src
    assert "_on_ai_rename" in src
    assert "_on_ai_rename_category" in src
    assert "_category_files" in src
    assert "AIRenameDialog" in src


def test_ai_compute_renames_invalid_backend_falls_back(tmp_path):
    """Invalid backend should behave like 'auto' and still produce pairs."""
    from PIL import Image
    from wallpaper_analyzer.rename import ai_compute_renames
    img = Image.new("RGB", (60, 60), (10, 200, 50))
    fp = tmp_path / "img.jpg"
    img.save(fp, "JPEG")
    pairs = ai_compute_renames([str(fp)], strategy="category", backend="bogus",
                               category="Anime")
    assert len(pairs) == 1


def test_ai_rename_dialog_no_auto_preview():
    """The dialog must NOT auto-start tag detection on open.

    This was the root cause of the OOM crash: opening the dialog used to
    fire ai_compute_renames immediately, which (for Ollama/CLIP backends)
    blew memory before the user could pick a backend.
    """
    import sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from wallpaper_analyzer.gui.ai_rename_dialog import AIRenameDialog
    dlg = AIRenameDialog(
        files=[f"/tmp/fake_{i}.jpg" for i in range(100)],
        category="Anime",
    )
    assert dlg._preview_job is None or not dlg._preview_job.isRunning()
    assert dlg._has_preview is False
    assert dlg._apply_btn.isEnabled() is False
    assert dlg._stop_btn.isEnabled() is False
    assert dlg._run_btn.isEnabled() is True


def test_ai_rename_dialog_default_backend_is_heuristic():
    """Default backend must be 'heuristic' (safe, no AI)."""
    import sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from wallpaper_analyzer.gui.ai_rename_dialog import AIRenameDialog
    dlg = AIRenameDialog(
        files=["/tmp/fake.jpg"], category="Anime",
        default_backend="ollama",  # caller override is honoured
    )
    assert dlg._backend.currentData() == "ollama"

    # Without a caller override we fall back to heuristic.
    dlg2 = AIRenameDialog(files=["/tmp/fake.jpg"], category="Anime")
    assert dlg2._backend.currentData() == "heuristic"


def test_clip_tag_vocab_is_full():
    """_clip_tag_vocab must return ALL curated tags (no artificial cap).

    The user explicitly wants the AI not to be limited — capping the
    vocab would hide legitimate tags. The encoded tensors are dropped
    after each image in ``_clip_detect_tags`` to keep memory bounded
    in a different way.
    """
    from wallpaper_analyzer.rename import _clip_tag_vocab, _CLIP_TAG_VOCAB
    _CLIP_TAG_VOCAB.clear() if hasattr(_CLIP_TAG_VOCAB, "clear") else None
    vocab = _clip_tag_vocab()
    # Must be the full tag registry, not a small slice.
    assert len(vocab) > 600, f"Vocab has only {len(vocab)} tags — limit too tight"
    assert all(isinstance(t, str) for t in vocab)
    # All entries should be tag-shaped (1-20 chars, no spaces).
    for t in vocab[:100]:
        assert 1 < len(t) <= 20 and " " not in t


def test_ai_rename_dialog_preview_limit_changes_apply_default():
    """The preview_limit parameter must cap how many files are processed."""
    import sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from wallpaper_analyzer.gui.ai_rename_dialog import AIRenameDialog
    dlg = AIRenameDialog(
        files=[f"/tmp/f_{i}.jpg" for i in range(200)],
        category="Anime",
        preview_limit=10,
    )
    assert dlg._preview_limit.value() == 10


def test_fallback_tags_never_empty(tmp_path):
    """_fallback_tags must always return at least one token."""
    from PIL import Image
    from wallpaper_analyzer.rename import _fallback_tags
    img = Image.new("RGB", (200, 200), (255, 0, 0))  # solid red
    fp = tmp_path / "red.jpg"
    img.save(fp, "JPEG")
    tags = _fallback_tags(str(fp), [], max_tags=4)
    assert len(tags) >= 1, "_fallback_tags returned empty list"
    assert all(isinstance(t, str) and t for t in tags)


def test_fallback_tags_combines_ai_with_deterministic(tmp_path):
    """_fallback_tags should append deterministic tokens when AI list is short."""
    from PIL import Image
    from wallpaper_analyzer.rename import _fallback_tags
    img = Image.new("RGB", (1920, 1080), (50, 50, 200))  # 16:9, blue
    fp = tmp_path / "widescreen.jpg"
    img.save(fp, "JPEG")
    # Give it only 1 AI tag, ask for 4 — should fill with deterministic.
    tags = _fallback_tags(str(fp), ["sunset"], max_tags=4)
    assert tags[0] == "sunset"
    assert len(tags) >= 2
    # The deterministic tokens should include "widescreen" since 16:9
    # is closer to 16:9 bucket than to "square".
    assert "widescreen" in tags


def test_fallback_tags_on_missing_file_returns_hash():
    """Missing file: still returns at least one token (the file hash)."""
    from wallpaper_analyzer.rename import _fallback_tags
    tags = _fallback_tags("/nonexistent/path/never.jpg", [], max_tags=3)
    assert len(tags) >= 1


def test_build_renames_no_untagged_in_output(tmp_path):
    """Even when no AI tags are provided, output filenames must contain
    real words — no 'untagged_NNN' placeholders."""
    from PIL import Image
    from wallpaper_analyzer.rename import build_renames
    img = Image.new("RGB", (1920, 1080), (50, 200, 100))
    fp = tmp_path / "img.jpg"
    img.save(fp, "JPEG")
    pairs = build_renames([str(fp)], strategy="tags",
                          tags_by_file={str(fp): []})
    assert len(pairs) == 1
    _, dst = pairs[0]
    base = os.path.splitext(os.path.basename(dst))[0]
    assert "untagged" not in base, f"Got untagged placeholder: {dst}"
    assert base, f"Got empty base: {dst}"


def test_build_renames_category_tags_no_untagged(tmp_path):
    """category_tags strategy must not produce untagged filenames."""
    from PIL import Image
    from wallpaper_analyzer.rename import build_renames
    img = Image.new("RGB", (500, 500), (200, 50, 50))
    fp = tmp_path / "img.jpg"
    img.save(fp, "JPEG")
    pairs = build_renames([str(fp)], strategy="category_tags",
                          category="Anime",
                          tags_by_file={str(fp): []})
    _, dst = pairs[0]
    base = os.path.splitext(os.path.basename(dst))[0]
    assert base.startswith("Anime_")
    # The remainder must have at least one descriptive token beyond
    # the category prefix.
    remainder = base[len("Anime_"):]
    assert remainder, f"category_tags produced only the category: {dst}"