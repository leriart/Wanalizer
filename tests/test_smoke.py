"""Smoke tests for the package public surface."""
import importlib

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
    ns2 = parser.parse_args(["--no-dedupe", "--parallel", "4"])
    assert ns2.dedupe is False
    assert ns2.parallel == 4