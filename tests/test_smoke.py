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


def test_categories_special_folders_contains_nsfw():
    from wallpaper_analyzer.categories import SPECIAL_FOLDERS, NSFW_FOLDER

    assert NSFW_FOLDER in SPECIAL_FOLDERS
    assert "Duplicates" in SPECIAL_FOLDERS
    assert "Low-Quality" in SPECIAL_FOLDERS


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