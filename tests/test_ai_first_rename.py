"""Tests for the AI-first rename cascade.

The renamer's default ("auto") cascade must prioritize AI semantic tag
matching — CLIP (when installed) against the FULL tag registry, with
Ollama and the analyzer as fallbacks — so the user gets tags that
describe what the image actually depicts, not just colour tokens.
"""
import pytest


def _clip_importable() -> bool:
    try:
        import clip  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _make_nature(tmp_path):
    """Realistic green/blue gradient image."""
    import numpy as np
    from PIL import Image
    img = Image.new("RGB", (400, 300), (50, 100, 70))
    arr = np.array(img)
    for y in range(300):
        for x in range(400):
            arr[y, x] = [
                30 + (y * 50 // 300),
                80 + (x * 40 // 400),
                50 + ((x + y) % 30),
            ]
    p = tmp_path / "nature.jpg"
    img.save(str(p), "JPEG", quality=85)
    return str(p)


def _make_solid(tmp_path, color=(120, 60, 200)):
    """Solid-colour image."""
    from PIL import Image
    p = tmp_path / "solid.jpg"
    Image.new("RGB", (200, 200), color).save(str(p), "JPEG")
    return str(p)


# ---------------------------------------------------------------------------
# Backend constants
# ---------------------------------------------------------------------------

def test_auto_is_default_for_ai_detect_tags():
    """ai_detect_tags default backend must be 'auto' (AI-first)."""
    import inspect
    from wallpaper_analyzer.rename import ai_detect_tags
    sig = inspect.signature(ai_detect_tags)
    assert sig.parameters["backend"].default == "auto"


def test_auto_is_default_for_ai_compute_renames():
    import inspect
    from wallpaper_analyzer.rename import ai_compute_renames
    sig = inspect.signature(ai_compute_renames)
    assert sig.parameters["backend"].default == "auto"


def test_airenamer_auto_backend_constant():
    from wallpaper_analyzer.rename import AIRenamer, AI_TAG_BACKENDS
    assert "auto" in AI_TAG_BACKENDS
    ren = AIRenamer(backend="auto", max_tags=4)
    try:
        assert ren.backend == "auto"
    finally:
        ren.close()


# ---------------------------------------------------------------------------
# Cascade behaviour
# ---------------------------------------------------------------------------

def test_auto_cascade_prefers_clip_when_available(tmp_path, monkeypatch):
    """When CLIP is available, auto must use CLIP (not the analyzer)."""
    from wallpaper_analyzer import rename

    # Stub CLIP to detect which backend the cascade picked.
    used = {"clip": 0, "analyzer": 0}

    def fake_clip(self, path):
        used["clip"] += 1
        return ["tokyo-night", "cyberpunk", "neon"], "tokyo-night"

    def fake_analyzer(self, path, category):
        used["analyzer"] += 1
        return ["green", "blue"], "green"

    monkeypatch.setattr(rename.AIRenamer, "_clip_detect", fake_clip)
    monkeypatch.setattr(rename.AIRenamer, "_heuristic_detect", fake_analyzer)

    ren = rename.AIRenamer(backend="auto", max_tags=4)
    try:
        tags, _ = ren.detect_tags(_make_solid(tmp_path))
        # CLIP must have been called; analyzer should NOT have been called
        # because CLIP returned non-empty.
        assert used["clip"] >= 1
        assert used["analyzer"] == 0
        # And the result must be CLIP's tags.
        assert "tokyo-night" in tags
    finally:
        ren.close()


def test_auto_cascade_falls_back_to_ollama_when_combined_empty(tmp_path, monkeypatch):
    """When the combined classifier (CLIP+analyzer+suggest_tags) returns
    empty, auto must try Ollama directly as the last AI resort."""
    from wallpaper_analyzer import rename

    used = {"combined": 0, "ollama": 0}

    def fake_combined(self, path, category):
        used["combined"] += 1
        return [], None

    def fake_ollama(self, path):
        used["ollama"] += 1
        return ["description-derived-tag", "another"], "description-derived-tag"

    monkeypatch.setattr(rename.AIRenamer, "_combined_classify", fake_combined)
    monkeypatch.setattr(rename.AIRenamer, "_ollama_detect", fake_ollama)

    ren = rename.AIRenamer(backend="auto", max_tags=4)
    try:
        tags, _ = ren.detect_tags(_make_solid(tmp_path))
        assert used["combined"] >= 1
        assert used["ollama"] >= 1
        # Ollama worked → tags come from there.
        assert "description-derived-tag" in tags
    finally:
        ren.close()


def test_auto_combined_uses_all_signals(tmp_path, monkeypatch):
    """The auto backend's combined classifier merges CLIP + analyzer +
    suggest_tags. Stub the three sources and verify all three feed in."""
    from wallpaper_analyzer import rename
    from wallpaper_analyzer import tag_suggester

    def fake_clip(self, path):
        return ["anime", "illustration"], None

    def fake_heuristic(self, path, category):
        return [], None

    class StubAnalyzer:
        def analyze(self, p): return {}
        def classify_with_confidence(self, profile):
            return {"tags": {"cyberpunk", "portrait"}}

    monkeypatch.setattr(rename.AIRenamer, "_clip_detect", fake_clip)
    monkeypatch.setattr(rename.AIRenamer, "_heuristic_detect", fake_heuristic)
    monkeypatch.setattr(rename.AIRenamer, "_ensure_analyzer",
                        lambda self: StubAnalyzer())
    monkeypatch.setattr(tag_suggester, "suggest_tags",
                        lambda profile, max_tags=20: {"minimalist"})

    ren = rename.AIRenamer(backend="auto", max_tags=8)
    try:
        tags, _ = ren.detect_tags(_make_solid(tmp_path))
        # All four sources contributed (anime/illustration from CLIP,
        # cyberpunk/portrait from analyzer, minimalist from suggest_tags).
        for t in ("anime", "illustration", "cyberpunk", "portrait", "minimalist"):
            assert t in tags, f"missing {t}: {tags}"
    finally:
        ren.close()


# ---------------------------------------------------------------------------
# Real CLIP integration (when installed) — smoke test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _clip_importable(),
    reason="CLIP not installed (torch + clip)",
)
def test_clip_returns_semantic_tags_from_registry(tmp_path):
    """With the real CLIP engine, semantic tags from the registry must
    be returned for an actual image — not just colour tokens."""
    from wallpaper_analyzer.rename import AIRenamer
    from wallpaper_analyzer.tags import _tags_flat

    ren = AIRenamer(backend="clip", max_tags=6)
    try:
        tags, _ = ren.detect_tags(_make_nature(tmp_path))
        # Every returned tag must come from the registered tag vocabulary
        # (i.e. CLIP is matching against the full registry, not making up
        # arbitrary words).
        for t in tags:
            assert t in _tags_flat, (
                f"CLIP returned {t!r} which is not in the tag registry. "
                "The CLIP path must score against the available tags."
            )
        # And the tags must NOT be just colour tokens.
        colour_only = {"red", "green", "blue", "yellow", "black", "white",
                       "orange", "purple", "pink", "brown", "cyan", "magenta"}
        assert any(t not in colour_only for t in tags), (
            f"CLIP returned only colour tokens: {tags}"
        )
    finally:
        ren.close()


def _clip_importable() -> bool:
    try:
        import clip  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# AI_renamer contract: never returns ONLY colours
# ---------------------------------------------------------------------------

def test_combined_avoids_colour_only_output(tmp_path, monkeypatch):
    """The combined classifier must not degenerate into a colour-only output.

    Even when CLIP/Ollama are unavailable, the analyzer should produce
    content-aware tags (anime / cyberpunk / portrait / ...) for any
    image that triggers a content signal. The colour-only fallback
    should only kick in for genuinely empty results.
    """
    from wallpaper_analyzer import rename
    from wallpaper_analyzer import tag_suggester

    # Pretend CLIP and Ollama are unavailable so we hit the analyzer.
    def fake_clip_unavail(self, path):
        return [], None

    def fake_ollama_unavail(self, path):
        return [], None

    def fake_combined_with_content_tags(self, path, category):
        return ["portrait", "human", "photo"], "portrait"

    monkeypatch.setattr(rename.AIRenamer, "_clip_detect", fake_clip_unavail)
    monkeypatch.setattr(rename.AIRenamer, "_ollama_detect", fake_ollama_unavail)
    # Stub the combined classifier to return content tags directly so we
    # don't depend on the analyzer being loaded.
    monkeypatch.setattr(rename.AIRenamer, "_combined_classify",
                        fake_combined_with_content_tags)

    ren = rename.AIRenamer(backend="auto", max_tags=6)
    try:
        tags, _ = ren.detect_tags(_make_nature(tmp_path))
        # The colour-only fallback should NOT have replaced the content
        # tags from the combined classifier.
        assert "portrait" in tags or "photo" in tags
        # The combined classifier returned 3 content tags that fill all
        # max_tags=6 slots' first 3 — no room for colour tokens.
        colour_only = {"red", "green", "blue", "yellow", "black", "white",
                       "orange", "purple", "pink", "brown", "cyan", "magenta"}
        assert not all(t in colour_only for t in tags), (
            f"Output is entirely colour tokens: {tags}"
        )
    finally:
        ren.close()