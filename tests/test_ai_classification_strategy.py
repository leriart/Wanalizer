"""Tests for the ai_classification rename strategy.

The `ai_classification` strategy mirrors the AI classification log
format in the filename:

  Classification log:
    [3/15] 20260623_222915.jpg... [ANI] Anime  (mode=ollama tags=cartoon,anime,female)

  Rename output:
    ANI_Anime_cartoon-anime-female.jpg

The user wanted the rename to be "equal to how it classifies" so the
filename encodes the same info as the log line: category abbreviation,
category name, and AI-detected tags.
"""
import os

from PIL import Image


def _save(path, color=(150, 60, 200)):
    Image.new("RGB", (200, 200), color).save(str(path), "JPEG")


# ---------------------------------------------------------------------------
# Strategy registration
# ---------------------------------------------------------------------------

def test_ai_classification_in_strategy_list():
    """`ai_classification` must be a registered strategy."""
    from wallpaper_analyzer.rename import RENAME_STRATEGIES
    keys = [k for k, _, _ in RENAME_STRATEGIES]
    assert "ai_classification" in keys


def test_ai_classification_is_tag_based():
    """`ai_classification` must be in TAG_BASED_STRATEGIES so the
    rename pipeline knows to feed it per-file tags."""
    from wallpaper_analyzer.rename import TAG_BASED_STRATEGIES
    assert "ai_classification" in TAG_BASED_STRATEGIES


# ---------------------------------------------------------------------------
# Format mirrors the classification log
# ---------------------------------------------------------------------------

def test_ai_classification_matches_classification_log(tmp_path):
    """For an Anime image with Ollama tags `cartoon,anime,female`, the
    rename must produce `ANI_Anime_cartoon-anime-female.jpg`, mirroring
    the classification log line `[ANI] Anime (mode=ollama tags=...)`."""
    from wallpaper_analyzer.rename import build_renames
    p = tmp_path / "20260623_222915.jpg"
    _save(p)
    pairs = build_renames(
        [str(p)],
        strategy="ai_classification",
        category="Anime",
        tags_by_file={str(p): ["cartoon", "anime", "female"]},
        subject_by_file={str(p): "cartoon"},
        max_tags=3,
    )
    assert len(pairs) == 1
    old, new = pairs[0]
    new_base = os.path.basename(new)
    # 3-letter abbreviation, full category, hyphen-joined tags.
    assert new_base == "ANI_Anime_cartoon-anime-female.jpg", (
        f"expected 'ANI_Anime_cartoon-anime-female.jpg', got {new_base!r}"
    )


def test_ai_classification_for_nsfw(tmp_path):
    """NSFW images: abbreviation `NSF`, category `NSFW`."""
    from wallpaper_analyzer.rename import build_renames
    p = tmp_path / "img.jpg"
    _save(p)
    pairs = build_renames(
        [str(p)],
        strategy="ai_classification",
        category="NSFW",
        tags_by_file={str(p): ["bikini", "sunglasses", "cool"]},
        max_tags=3,
    )
    new_base = os.path.basename(pairs[0][1])
    assert new_base == "NSF_NSFW_bikini-sunglasses-cool.jpg"


def test_ai_classification_short_category(tmp_path):
    """Categories with <3 chars: abbreviation is the full upper-case name."""
    from wallpaper_analyzer.rename import build_renames
    p = tmp_path / "img.jpg"
    _save(p)
    pairs = build_renames(
        [str(p)],
        strategy="ai_classification",
        category="4K",
        tags_by_file={str(p): ["hd", "widescreen"]},
        max_tags=3,
    )
    new_base = os.path.basename(pairs[0][1])
    # Python's `[:3]` on "4K" returns "4K" (less than 3 chars is OK).
    assert new_base.startswith("4K_4K_")


def test_ai_classification_fallback_when_no_tags(tmp_path):
    """When no tags are available, _fallback_tags adds colour/aspect
    tokens so the filename is still meaningful — never the literal
    'untagged' placeholder."""
    from wallpaper_analyzer.rename import build_renames
    p = tmp_path / "img.jpg"
    _save(p)
    pairs = build_renames(
        [str(p)],
        strategy="ai_classification",
        category="Anime",
        tags_by_file={str(p): []},
        max_tags=3,
    )
    new_base = os.path.basename(pairs[0][1])
    # Abbreviation + category are always there.
    assert new_base.startswith("ANI_Anime_")
    # The fallback produces SOMETHING (colour/aspect tokens) so the
    # filename is never just "ANI_Anime.jpg".
    assert len(new_base) > len("ANI_Anime_.jpg")


def test_ai_classification_respects_max_tags(tmp_path):
    """max_tags must cap the number of tags in the filename."""
    from wallpaper_analyzer.rename import build_renames
    p = tmp_path / "img.jpg"
    _save(p)
    pairs = build_renames(
        [str(p)],
        strategy="ai_classification",
        category="Anime",
        tags_by_file={str(p): ["a", "b", "c", "d", "e"]},
        max_tags=2,
    )
    new_base = os.path.basename(pairs[0][1])
    # Only first 2 tags should appear (joined with `-`).
    assert new_base == "ANI_Anime_a-b.jpg"


def test_ai_classification_sanitizes_category(tmp_path):
    """Special chars in category are sanitized for filenames."""
    from wallpaper_analyzer.rename import build_renames
    p = tmp_path / "img.jpg"
    _save(p)
    pairs = build_renames(
        [str(p)],
        strategy="ai_classification",
        category="Cat/With/Slashes",
        tags_by_file={str(p): ["tag1"]},
        max_tags=3,
    )
    new_base = os.path.basename(pairs[0][1])
    # Slashes must NOT appear in the basename.
    assert "/" not in new_base
    # Abbreviation is the first 3 chars of the sanitized category.
    assert new_base.startswith("CAT_")


def test_ai_classification_uses_renamer_when_provided(tmp_path, monkeypatch):
    """When an AIRenamer is passed to build_renames, the
    ai_classification strategy must call detect_tags() so the filename
    actually uses AI content analysis (CLIP / analyzer / cache),
    not just the colour tokens from Ollama.

    This is the user's main bug: Ollama returned `black,red,white`
    (only colours) for an image, the rename produced
    `ANI_Anime_black-red-white.jpg`, and the user reported "the AI
    isn't being used". After this fix, the renamer runs the
    combined classifier and surfaces content tags instead.
    """
    from PIL import Image
    from wallpaper_analyzer.rename import AIRenamer, build_renames

    p = tmp_path / "img.jpg"
    Image.new("RGB", (200, 200), (180, 30, 30)).save(p, "JPEG")

    # Stub the renamer to return content tags (not colours) so we can
    # verify the strategy picks them up.
    class StubRenamer(AIRenamer):
        def detect_tags(self, path, category=None):
            return ["illustration", "anime-character", "vibrant"], "illustration"

    ren = StubRenamer(backend="auto", max_tags=4)
    try:
        pairs = build_renames(
            [str(p)],
            strategy="ai_classification",
            category="Anime",
            tags_by_file={str(p): ["black", "red", "white"]},  # ollama tags
            max_tags=3,
            ai_renamer=ren,
        )
        new_base = os.path.basename(pairs[0][1])
        # Content tags from the renamer must appear, NOT the Ollama
        # colour tokens.
        assert "illustration" in new_base, (
            f"renamer tags not used: {new_base!r}"
        )
        assert "black" not in new_base, (
            f"colour-only Ollama tags leaked: {new_base!r}"
        )
    finally:
        ren.close()


def test_ai_classification_renamer_called_per_file(tmp_path, monkeypatch):
    """The AIRenamer must be invoked once per file so the cache can
    amortise work across the batch."""
    from PIL import Image
    from wallpaper_analyzer.rename import AIRenamer, build_renames

    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    p3 = tmp_path / "c.jpg"
    _save(p1)
    _save(p2)
    _save(p3)

    called = []

    class TrackingRenamer(AIRenamer):
        def detect_tags(self, path, category=None):
            called.append(path)
            return ["content-tag"], "content-tag"

    ren = TrackingRenamer(backend="auto", max_tags=3)
    try:
        build_renames(
            [str(p1), str(p2), str(p3)],
            strategy="ai_classification",
            category="Anime",
            tags_by_file={str(p): [] for p in (p1, p2, p3)},
            max_tags=3,
            ai_renamer=ren,
        )
        assert len(called) == 3, f"expected 3 renamer calls, got {len(called)}"
    finally:
        ren.close()


def test_ai_classification_renamer_failure_falls_back_to_passed_tags(tmp_path, monkeypatch):
    """If the renamer raises, the strategy must fall back to the
    passed tags_by_file so the rename never crashes."""
    from PIL import Image
    from wallpaper_analyzer.rename import AIRenamer, build_renames

    p = tmp_path / "img.jpg"
    _save(p)

    class FailingRenamer(AIRenamer):
        def detect_tags(self, path, category=None):
            raise RuntimeError("simulated AI failure")

    ren = FailingRenamer(backend="auto", max_tags=3)
    try:
        pairs = build_renames(
            [str(p)],
            strategy="ai_classification",
            category="Anime",
            tags_by_file={str(p): ["anime-tag"]},
            max_tags=3,
            ai_renamer=ren,
        )
        new_base = os.path.basename(pairs[0][1])
        # Falls back to the passed tags.
        assert "anime-tag" in new_base
    finally:
        ren.close()


def test_ai_classification_non_tag_strategies_ignore_renamer(tmp_path, monkeypatch):
    """Non-tag strategies must NOT call the renamer (no AI work needed)."""
    from PIL import Image
    from wallpaper_analyzer.rename import AIRenamer, build_renames

    p = tmp_path / "img.jpg"
    _save(p)

    called = []

    class TrackingRenamer(AIRenamer):
        def detect_tags(self, path, category=None):
            called.append("called")
            return ["x"], None

    ren = TrackingRenamer(backend="auto", max_tags=3)
    try:
        build_renames(
            [str(p)],
            strategy="sequential",  # not tag-based
            category="Anime",
            tags_by_file={str(p): ["x"]},
            max_tags=3,
            ai_renamer=ren,
        )
        assert called == [], (
            f"renamer called for non-tag strategy: {called}"
        )
    finally:
        ren.close()


def test_ai_classification_does_not_collide_on_disk(tmp_path):
    """Two files with the same classification output must NOT collide
    on disk — build_renames auto-suffixes the second one."""
    from wallpaper_analyzer.rename import build_renames
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    _save(p1)
    _save(p2)
    pairs = build_renames(
        [str(p1), str(p2)],
        strategy="ai_classification",
        category="Anime",
        tags_by_file={str(p1): ["a", "b"], str(p2): ["a", "b"]},
        max_tags=2,
    )
    targets = [os.path.basename(n) for _, n in pairs]
    assert len(set(targets)) == 2, f"collision: {targets}"


# ---------------------------------------------------------------------------
# _combined_classify reads cached Ollama tags
# ---------------------------------------------------------------------------

def test_combined_classify_prefers_cached_ollama_tags(tmp_path, monkeypatch):
    """When the hash cache has Ollama tags from a prior classification,
    _combined_classify must surface them at the top of the result so
    the rename mirrors the classification log."""
    from wallpaper_analyzer import rename
    from PIL import Image

    p = tmp_path / "img.jpg"
    Image.new("RGB", (50, 50), (200, 100, 50)).save(p, "JPEG")

    # Stub the cache loader so we can inject Ollama tags without
    # actually running the pipeline.
    fake_cache = {
        str(p): {
            "sig": "fake",
            "md5": "fake",
            "ollama_all_tags": ["cartoon", "anime", "female"],
        },
    }
    monkeypatch.setattr(
        "wallpaper_analyzer.duplicates.load_hash_cache",
        lambda: fake_cache,
    )

    # Stub CLIP and analyzer so they don't override the Ollama tags.
    monkeypatch.setattr(rename.AIRenamer, "_clip_detect",
                        lambda self, path: ([], None))
    monkeypatch.setattr(rename.AIRenamer, "_ensure_analyzer",
                        lambda self: None)
    # Stub suggest_tags so the 3rd source doesn't add noise.
    from wallpaper_analyzer import tag_suggester
    monkeypatch.setattr(tag_suggester, "suggest_tags",
                        lambda profile, max_tags=20: set())

    ren = rename.AIRenamer(backend="auto", max_tags=5)
    try:
        tags, subj = ren.detect_tags(str(p), category="Anime")
        # Ollama tags must come first.
        assert tags[:3] == ["cartoon", "anime", "female"], (
            f"expected Ollama tags first, got {tags}"
        )
    finally:
        ren.close()


def test_combined_classify_logs_ollama_source(tmp_path, monkeypatch):
    """When Ollama tags are surfaced, the renamer log mentions 'ollama'
    as a source so the user can confirm the rename mirrors the
    classification."""
    from wallpaper_analyzer import rename
    from PIL import Image

    p = tmp_path / "img.jpg"
    Image.new("RGB", (50, 50), (200, 100, 50)).save(p, "JPEG")

    monkeypatch.setattr(
        "wallpaper_analyzer.duplicates.load_hash_cache",
        lambda: {str(p): {"sig": "", "md5": "", "ollama_all_tags": ["anime"]}},
    )
    monkeypatch.setattr(rename.AIRenamer, "_clip_detect",
                        lambda self, path: ([], None))
    monkeypatch.setattr(rename.AIRenamer, "_ensure_analyzer",
                        lambda self: None)
    from wallpaper_analyzer import tag_suggester
    monkeypatch.setattr(tag_suggester, "suggest_tags",
                        lambda profile, max_tags=20: set())

    ren = rename.AIRenamer(backend="auto", max_tags=4)
    try:
        ren.detect_tags(str(p))
        # Log must mention 'ollama' as a source.
        assert any("ollama" in line for line in ren.log_lines), (
            f"Ollama source not mentioned in log: {ren.log_lines}"
        )
    finally:
        ren.close()