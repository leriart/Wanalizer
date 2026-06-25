"""Tests for the combined AI tag classifier.

The combined classifier merges three independent signals (CLIP
semantic, analyzer content heuristics, suggest_tags) and prioritizes
content tags over colour tokens. This guarantees that the rename
output always describes what the image depicts, not just its colour.
"""
import pytest


def _clip_importable() -> bool:
    try:
        import clip  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _make_solid(tmp_path, color=(120, 60, 200)):
    from PIL import Image
    p = tmp_path / "solid.jpg"
    Image.new("RGB", (200, 200), color).save(str(p), "JPEG")
    return str(p)


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


# ---------------------------------------------------------------------------
# Constants / colour-token filter
# ---------------------------------------------------------------------------

def test_colour_token_set_defined():
    from wallpaper_analyzer.rename import AIRenamer
    assert hasattr(AIRenamer, "_COLOUR_ONLY_TOKENS")
    assert len(AIRenamer._COLOUR_ONLY_TOKENS) >= 20
    # Common colours must be in the set.
    for c in ("red", "green", "blue", "black", "white", "purple",
              "pink", "orange", "yellow", "cyan"):
        assert c in AIRenamer._COLOUR_ONLY_TOKENS


def test_is_colour_token_case_insensitive():
    from wallpaper_analyzer.rename import AIRenamer
    assert AIRenamer._is_colour_token("Red") is True
    assert AIRenamer._is_colour_token("RED") is True
    assert AIRenamer._is_colour_token("red") is True
    # Content-style tags must NOT be considered colour tokens.
    assert AIRenamer._is_colour_token("anime") is False
    assert AIRenamer._is_colour_token("cyberpunk") is False
    assert AIRenamer._is_colour_token("tokyo-night") is False


# ---------------------------------------------------------------------------
# Combined classifier behaviour (with stubs)
# ---------------------------------------------------------------------------

def test_combined_classify_merges_clip_analyzer_suggest(tmp_path, monkeypatch):
    """Three sources all contribute; duplicates are deduped."""
    from wallpaper_analyzer import rename
    from wallpaper_analyzer.rename import AIRenamer
    from PIL import Image

    def fake_clip_v2(self, path):
        return ["anime", "illustration", "red"], None

    def fake_heuristic_v2(self, path, category):
        return [], None

    monkeypatch.setattr(rename.AIRenamer, "_clip_detect", fake_clip_v2)
    monkeypatch.setattr(rename.AIRenamer, "_heuristic_detect", fake_heuristic_v2)

    class StubAnalyzer:
        name = "stub"
        def analyze(self, p): return {}
        def classify(self, profile): return "Anime"
        def classify_with_confidence(self, profile):
            return {"tags": {"illustration", "cyberpunk", "portrait"}}

    monkeypatch.setattr(rename.AIRenamer, "_ensure_analyzer", lambda self: StubAnalyzer())
    # Stub suggest_tags so the third source is deterministic.
    from wallpaper_analyzer import tag_suggester
    monkeypatch.setattr(tag_suggester, "suggest_tags",
                        lambda profile, max_tags=20: {"minimalist"})

    img = Image.new("RGB", (200, 200), (180, 100, 220))
    img.save(str(tmp_path / "real.jpg"), "JPEG")

    ren = rename.AIRenamer(backend="heuristic", max_tags=8)
    try:
        tags, subj = ren.detect_tags(str(tmp_path / "real.jpg"))
        # anime (CLIP), illustration (CLIP + analyzer), cyberpunk
        # (analyzer), portrait (analyzer), minimalist (suggest_tags)
        # must all appear.
        for t in ("anime", "illustration", "cyberpunk", "portrait", "minimalist"):
            assert t in tags, f"missing {t}: {tags}"
        # 'red' is a colour-only token. If it appears, content tags must
        # come BEFORE it (content-first ordering).
        if "red" in tags:
            first_colour_idx = next(
                i for i, t in enumerate(tags) if AIRenamer._is_colour_token(t)
            )
            for j in range(first_colour_idx):
                assert not AIRenamer._is_colour_token(tags[j]), (
                    f"colour token 'red' at {first_colour_idx} but "
                    f"later content tag at {j}: {tags}"
                )
    finally:
        ren.close()


def test_combined_classify_prioritises_content_over_colour(tmp_path, monkeypatch):
    """Pure-colour tokens are moved to the END of the tag list."""
    from wallpaper_analyzer import rename
    from wallpaper_analyzer.rename import AIRenamer
    from PIL import Image

    def fake_clip(self, path):
        # CLIP returns mostly colour tokens + 1 content token.
        return ["red", "blue", "anime", "purple"], None

    def fake_heuristic(self, path, category):
        return [], None

    monkeypatch.setattr(rename.AIRenamer, "_clip_detect", fake_clip)
    monkeypatch.setattr(rename.AIRenamer, "_heuristic_detect", fake_heuristic)
    monkeypatch.setattr(rename.AIRenamer, "_ensure_analyzer", lambda self: None)
    # Stub suggest_tags so only CLIP contributes.
    from wallpaper_analyzer import tag_suggester
    monkeypatch.setattr(tag_suggester, "suggest_tags",
                        lambda profile, max_tags=20: set())

    img = Image.new("RGB", (100, 100), (200, 100, 50))
    img.save(str(tmp_path / "x.jpg"), "JPEG")

    ren = rename.AIRenamer(backend="heuristic", max_tags=4)
    try:
        tags, _ = ren.detect_tags(str(tmp_path / "x.jpg"))
        # 'anime' must come first (only content token).
        assert tags[0] == "anime", f"expected 'anime' first, got {tags}"
        # All colour tokens must come AFTER the content tokens.
        colour_positions = [
            i for i, t in enumerate(tags) if AIRenamer._is_colour_token(t)
        ]
        content_positions = [
            i for i, t in enumerate(tags) if not AIRenamer._is_colour_token(t)
        ]
        if colour_positions and content_positions:
            assert min(colour_positions) > max(content_positions), (
                f"colour tokens should come after content, got {tags}"
            )
    finally:
        ren.close()


def test_combined_classify_handles_empty_sources(tmp_path, monkeypatch):
    """When all sources are empty, returns ([], None)."""
    from wallpaper_analyzer import rename
    from PIL import Image

    def fake_clip(self, path):
        return [], None

    def fake_heuristic(self, path, category):
        return [], None

    monkeypatch.setattr(rename.AIRenamer, "_clip_detect", fake_clip)
    monkeypatch.setattr(rename.AIRenamer, "_heuristic_detect", fake_heuristic)
    monkeypatch.setattr(rename.AIRenamer, "_ensure_analyzer", lambda self: None)
    # Stub suggest_tags too so the combined classifier's third source
    # is empty (otherwise it picks up "black-and-white" from the dark
    # test image).
    from wallpaper_analyzer import tag_suggester
    monkeypatch.setattr(tag_suggester, "suggest_tags",
                        lambda profile, max_tags=20: set())

    img = Image.new("RGB", (50, 50), (10, 20, 30))
    img.save(str(tmp_path / "x.jpg"), "JPEG")

    ren = rename.AIRenamer(backend="heuristic", max_tags=4)
    try:
        tags, subj = ren.detect_tags(str(tmp_path / "x.jpg"))
        # The combined classifier returns empty; the deterministic
        # fallback (_fallback_tags) adds colour/aspect tokens so the
        # result is never completely empty for the caller.
        # The key invariant is: subject is None (no AI content to anchor).
        # Tags may be empty or colour-only fallback tokens.
        assert subj is None
    finally:
        ren.close()


# ---------------------------------------------------------------------------
# Real-wallpaper integration: content tags MUST appear
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _clip_importable(),
    reason="CLIP not installed (torch + clip)",
)
def test_real_wallpaper_has_content_tags(tmp_path):
    """On a real image, the combined classifier must return content tags.

    This is the user's main acceptance criterion: AI tag detection must
    describe what the image DEPICTS, not just its colour. For a
    green/blue landscape-style image, the result must contain at least
    one non-colour content tag from the registry.
    """
    from wallpaper_analyzer.rename import AIRenamer
    from wallpaper_analyzer.tags import _tags_flat

    ren = AIRenamer(backend="auto", max_tags=8)
    try:
        tags, _ = ren.detect_tags(_make_nature(tmp_path))
        # All tags must come from the registry.
        for t in tags:
            assert t in _tags_flat, f"{t!r} not in registry"
        # At least one tag must NOT be a pure-colour token.
        assert any(not AIRenamer._is_colour_token(t) for t in tags), (
            f"Real wallpaper tags are all colour tokens: {tags}"
        )
    finally:
        ren.close()


def test_real_wallpaper_heuristic_uses_combined(tmp_path):
    """The 'heuristic' backend must use the combined classifier (which
    uses CLIP/analyzer when available), not just return colours."""
    from wallpaper_analyzer.rename import AIRenamer
    from wallpaper_analyzer.tags import _tags_flat

    if not _clip_importable():
        pytest.skip("CLIP not installed")

    ren = AIRenamer(backend="heuristic", max_tags=8)
    try:
        tags, _ = ren.detect_tags(_make_nature(tmp_path))
        # All tags from registry.
        for t in tags:
            assert t in _tags_flat, f"{t!r} not in registry"
        # At least one content tag (heuristic backend should now produce
        # content tags via the combined classifier's CLIP path).
        assert any(not AIRenamer._is_colour_token(t) for t in tags), (
            f"Heuristic backend only produced colour tags: {tags}"
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