"""Tests for the analyzer-driven AI renamer.

The renamer's "heuristic" backend must use the same analyzer pipeline
the organize pass uses for category assignment, so the tags describe
the image content (anime / cyberpunk / portrait / nature / ...) and
not just colours.
"""
import numpy as np
from PIL import Image


def _save(path, factory):
    img = factory()
    img.save(str(path), "JPEG", quality=90)


def _nature_img():
    """Realistic green/blue gradient with some texture and clouds."""
    img = Image.new("RGB", (400, 300), (50, 100, 70))
    arr = np.array(img)
    for y in range(300):
        for x in range(400):
            arr[y, x] = [
                30 + (y * 50 // 300),
                80 + (x * 40 // 400),
                50 + ((x + y) % 30),
            ]
    for cx in (100, 250, 350):
        for y in range(50, 100):
            for x in range(cx - 30, cx + 30):
                if 0 <= x < 400:
                    arr[y, x] = [200, 210, 220]
    return Image.fromarray(arr)


def _portrait_img():
    """Skin-tone dominant with darker hair + eyes."""
    img = Image.new("RGB", (300, 400), (200, 150, 120))
    arr = np.array(img)
    arr[0:80, :] = [40, 30, 25]
    arr[150:170, 100:130] = [30, 20, 15]
    arr[150:170, 170:200] = [30, 20, 15]
    noise = np.random.RandomState(42).randint(-15, 15, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _cyberpunk_img():
    """Dark base with neon pink/cyan accents."""
    img = Image.new("RGB", (300, 300), (5, 5, 20))
    for x in range(50, 250, 10):
        img.paste((255, 0, 200), (x, 50, x + 3, 250))
    for y in range(50, 250, 10):
        img.paste((0, 255, 255), (50, y, 250, y + 3))
    return img


# ---------------------------------------------------------------------------
# BaseAnalyzer.detect_tags
# ---------------------------------------------------------------------------

def test_base_analyzer_detect_tags_returns_content_aware(tmp_path):
    """BaseAnalyzer.detect_tags must return content-aware tags, not just colours."""
    from wallpaper_analyzer.analyzers import get_analyzer
    from wallpaper_analyzer import settings as s

    a = get_analyzer("lowlevel", s.load_settings())

    p = tmp_path / "nature.jpg"
    _save(p, _nature_img)

    tags, subject = a.detect_tags(str(p), max_tags=8)
    assert isinstance(tags, list)
    assert len(tags) > 0
    # Must NOT be only colour names — at least one of the natural-content
    # tags should appear for a green/blue image.
    colour_only = {"red", "green", "blue", "yellow", "black", "white",
                   "orange", "purple", "pink", "brown", "cyan", "magenta"}
    content_tags = [t for t in tags if t not in colour_only]
    assert content_tags, (
        f"detect_tags returned only colour tokens: {tags}. "
        "Expected at least one content tag (nature/landscape/outdoor/...)"
    )


def test_base_analyzer_detect_tags_portrait(tmp_path):
    """A skin-tone-heavy image should surface portrait/human/person tags."""
    from wallpaper_analyzer.analyzers import get_analyzer
    from wallpaper_analyzer import settings as s

    a = get_analyzer("lowlevel", s.load_settings())

    p = tmp_path / "portrait.jpg"
    _save(p, _portrait_img)

    tags, subject = a.detect_tags(str(p), max_tags=10)
    portrait_tags = {"portrait", "person", "human", "face", "figure"}
    assert any(t in portrait_tags for t in tags), (
        f"Expected at least one portrait-style tag, got: {tags}"
    )


def test_base_analyzer_detect_tags_batch_isolates_errors(tmp_path):
    """detect_tags_batch must never raise on a bad path."""
    from wallpaper_analyzer.analyzers import get_analyzer
    from wallpaper_analyzer import settings as s

    a = get_analyzer("lowlevel", s.load_settings())
    good = tmp_path / "good.jpg"
    _save(good, _nature_img)
    bad = tmp_path / "does_not_exist.jpg"

    out = a.detect_tags_batch([str(good), str(bad)], max_tags=4)
    assert str(good) in out
    assert str(bad) in out
    # Bad path yielded an empty list, never raised.
    assert out[str(bad)] == ([], None)


def test_base_analyzer_detect_tags_consistent_with_category(tmp_path):
    """Tags returned by detect_tags must come from the SAME pipeline the
    classifier uses for category assignment — the user's main request."""
    from wallpaper_analyzer.analyzers import get_analyzer
    from wallpaper_analyzer import settings as s

    a = get_analyzer("lowlevel", s.load_settings())
    p = tmp_path / "img.jpg"
    _save(p, _nature_img)
    path = str(p)

    profile = a.analyze(path)
    cat = a.classify(profile)
    info = a.classify_with_confidence(profile) if hasattr(a, "classify_with_confidence") else {}
    classifier_tags = set(info.get("tags") or [])

    tags, _ = a.detect_tags(path, max_tags=10)
    # The rename tag list must overlap with the classifier tag set
    # (same profile → same tags, modulo ordering / cap).
    assert set(tags) & classifier_tags, (
        f"detect_tags={tags} is disjoint from classifier tags={classifier_tags}. "
        "The rename pipeline and category pipeline must share the tag source."
    )


# ---------------------------------------------------------------------------
# AIRenamer wiring
# ---------------------------------------------------------------------------

def test_airenamer_heuristic_uses_analyzer(tmp_path):
    """AIRenamer(backend='heuristic') must run the analyzer pipeline."""
    from wallpaper_analyzer.rename import AIRenamer

    ren = AIRenamer(backend="heuristic", mode="lowlevel", max_tags=8)
    try:
        p = tmp_path / "nature.jpg"
        _save(p, _nature_img)

        tags, _ = ren.detect_tags(str(p))
        assert isinstance(tags, list) and tags
        # Should NOT be just colour tokens.
        colour_only = {"red", "green", "blue", "yellow", "black", "white",
                       "orange", "purple", "pink", "brown", "cyan", "magenta"}
        non_colour = [t for t in tags if t not in colour_only]
        assert non_colour, f"Only colours returned: {tags}"
    finally:
        ren.close()


def test_airenamer_heuristic_per_file_cache(tmp_path):
    """Same file processed twice with cache should hit on second call."""
    from wallpaper_analyzer.rename import AIRenamer

    ren = AIRenamer(backend="heuristic", mode="lowlevel", max_tags=8)
    try:
        p = tmp_path / "x.jpg"
        _save(p, _portrait_img)

        t1, _ = ren.detect_tags(str(p))
        first_processed = ren.processed
        t2, _ = ren.detect_tags(str(p))
        # Same result returned, processed counter did not advance.
        assert t1 == t2
        assert ren.processed == first_processed
        assert ren.cached_hits >= 1
    finally:
        ren.close()


def test_airenamer_force_reprocess_bypasses_cache(tmp_path):
    from wallpaper_analyzer.rename import AIRenamer

    ren = AIRenamer(backend="heuristic", mode="lowlevel", max_tags=8,
                    force_reprocess=True)
    try:
        p = tmp_path / "x.jpg"
        _save(p, _portrait_img)

        ren.detect_tags(str(p))
        before = ren.processed
        ren.detect_tags(str(p))
        # force_reprocess=True means every call re-detects.
        assert ren.processed == before + 1
        assert ren.cached_hits == 0
    finally:
        ren.close()


def test_airenamer_handles_missing_file(tmp_path):
    from wallpaper_analyzer.rename import AIRenamer

    ren = AIRenamer(backend="heuristic", mode="lowlevel", max_tags=8)
    try:
        tags, subj = ren.detect_tags(str(tmp_path / "missing.jpg"))
        assert tags == []
        assert subj is None
        assert ren.failed >= 1
    finally:
        ren.close()


def test_airenamer_batch_isolates_failures(tmp_path):
    """detect_tags_batch never aborts on a bad file."""
    from wallpaper_analyzer.rename import AIRenamer

    ren = AIRenamer(backend="heuristic", mode="lowlevel", max_tags=6)
    try:
        good = tmp_path / "good.jpg"
        _save(good, _nature_img)
        bad = tmp_path / "missing.jpg"

        out = ren.detect_tags_batch([str(good), str(bad)])
        assert str(good) in out
        assert str(bad) in out
        assert out[str(good)][0]  # real file produced tags
        assert out[str(bad)][0] == []  # missing file produced empty
    finally:
        ren.close()


def test_ai_detect_tags_accepts_mode_parameter(tmp_path):
    """ai_detect_tags wrapper must accept mode= and forward it."""
    from wallpaper_analyzer.rename import ai_detect_tags

    p = tmp_path / "x.jpg"
    _save(p, _nature_img)
    tags, subj = ai_detect_tags(str(p), backend="heuristic", mode="lowlevel", max_tags=6)
    assert isinstance(tags, list)
    assert tags


def test_ai_compute_renames_uses_analyzer_pipeline(tmp_path):
    """ai_compute_renames with category_tags must use the analyzer-driven tags."""
    from wallpaper_analyzer.rename import ai_compute_renames

    p = tmp_path / "img.jpg"
    _save(p, _nature_img)
    pairs = ai_compute_renames(
        [str(p)], strategy="category_tags",
        backend="heuristic", mode="lowlevel",
        category="Nature", max_tags=4,
    )
    assert len(pairs) == 1
    old, new = pairs[0]
    # The new basename must include at least one content token (not just colours).
    new_base = new.rsplit(".", 1)[0]
    colour_only = {"red", "green", "blue", "yellow", "black", "white",
                   "orange", "purple", "pink", "brown", "cyan", "magenta"}
    parts = new_base.split("_")
    non_colour = [p for p in parts if p.lower() not in colour_only and p != "nature"]
    # We expect at least one non-colour, non-category token.
    assert non_colour, f"category_tags only has colour/category tokens: {new_base}"


def test_airenamer_modes_resolve(tmp_path):
    """AIRenamer(mode='auto') must resolve to a valid analyzer (default = organise mode)."""
    from wallpaper_analyzer.rename import AIRenamer

    ren = AIRenamer(backend="heuristic", mode="auto", max_tags=4)
    try:
        p = tmp_path / "x.jpg"
        _save(p, _nature_img)
        # Doesn't raise → analyzer resolved.
        tags, _ = ren.detect_tags(str(p))
        assert isinstance(tags, list)
    finally:
        ren.close()


def test_ai_tag_modes_constant_includes_auto():
    from wallpaper_analyzer.rename import AI_TAG_MODES

    for m in ("auto", "lowlevel", "fusion", "clip", "ollama"):
        assert m in AI_TAG_MODES