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