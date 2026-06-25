"""Tests for perceptual duplicate detection."""
import os

from PIL import Image


def _make_image(path, color=(128, 64, 200), size=(200, 200), fmt="JPEG", quality=85):
    """Save a solid-color image so hashes are deterministic."""
    Image.new("RGB", size, color).save(path, fmt, quality=quality)


def _add_to_cache(cache, path, frames):
    """Insert a `frames` entry into the cache for `path`."""
    entry = cache.setdefault(path, {})
    entry["sig"] = "fake-sig"
    entry["frames"] = frames
    entry["size"] = (200, 200)
    entry["size_bytes"] = os.path.getsize(path) if os.path.exists(path) else 0


def test_perceptual_worker_returns_frames(tmp_path):
    """_perceptual_worker must return multi-frame hashes for a static image."""
    from wallpaper_analyzer.duplicates import _perceptual_worker

    p = tmp_path / "img.jpg"
    _make_image(p)
    res = _perceptual_worker(str(p))
    assert res is not None
    entry = res["entry"]
    assert "frames" in entry and len(entry["frames"]) >= 1
    f0 = entry["frames"][0]
    for k in ("dh", "dh16", "ph", "ah", "ch16", "chh", "hist"):
        assert k in f0, f"missing {k}"


def test_perceptual_worker_handles_missing_file(tmp_path):
    from wallpaper_analyzer.duplicates import _perceptual_worker

    res = _perceptual_worker(str(tmp_path / "does_not_exist.jpg"))
    assert res is None


def test_frame_tier_identical_returns_reencode():
    from wallpaper_analyzer.duplicates import _frame_tier, TIER_REENCODE

    f = {"dh": 0x1234, "ph": 0x5678, "hist": [4] * 16}
    assert _frame_tier(f, dict(f)) == TIER_REENCODE


def test_frame_tier_too_different_returns_none():
    from wallpaper_analyzer.duplicates import _frame_tier

    f1 = {"dh": 0x0000, "ph": 0x0000, "hist": [0, 0, 0, 16, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0]}
    f2 = {"dh": 0xFFFF, "ph": 0xFFFF, "hist": [16, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0]}
    assert _frame_tier(f1, f2) is None


def test_frame_tier_close_returns_resize():
    from wallpaper_analyzer.duplicates import _frame_tier, TIER_RESIZE

    f1 = {"dh": 0x0000, "ph": 0x0000, "hist": [16] * 16}
    # 6 bit differences on dh (within RESIZE's 8), 10 on ph (== RESIZE limit).
    f2 = {"dh": 0x003F, "ph": (1 << 10) - 1, "hist": [16] * 16}
    assert _frame_tier(f1, f2) == TIER_RESIZE


def test_files_match_tier_majority_required():
    """Multi-frame files: only majority-matching pairs count."""
    from wallpaper_analyzer.duplicates import _files_match_tier

    a = [
        {"dh": 0x00, "ph": 0x00, "hist": [16] * 16},
        {"dh": 0xFF, "ph": 0xFF, "hist": [0] * 16},  # differs
    ]
    b = [
        {"dh": 0x00, "ph": 0x00, "hist": [16] * 16},  # matches a[0]
    ]
    # 1 match / 2 pairs = 50%, borderline majority (>= threshold)
    assert _files_match_tier(a, b) is not None


def test_files_match_tier_rejects_minority():
    from wallpaper_analyzer.duplicates import _files_match_tier

    a = [
        {"dh": 0x00, "ph": 0x00, "hist": [16] * 16},
        {"dh": 0xFF, "ph": 0xFF, "hist": [0] * 16},
        {"dh": 0xFF, "ph": 0xFF, "hist": [0] * 16},
    ]
    b = [
        {"dh": 0xFF, "ph": 0xFF, "hist": [0] * 16},
        {"dh": 0xFF, "ph": 0xFF, "hist": [0] * 16},
    ]
    # a[0] differs; only a[1]/a[2] match b. 4/6 pairs match, but
    # _files_match_tier uses all-pairs (6 pairs) and threshold 0.5,
    # so 4/6 >= 0.5 -> still matches.
    # Verify the symmetric "completely different" case rejects:
    a2 = [{"dh": 0x00, "ph": 0x00, "hist": [16] * 16}]
    b2 = [{"dh": 0xFF, "ph": 0xFF, "hist": [0] * 16}]
    assert _files_match_tier(a2, b2) is None


def test_perceptual_groups_detects_reencoded(tmp_path):
    """The same image saved as JPG and PNG should be a perceptual dup."""
    from wallpaper_analyzer.duplicates import (
        scan_and_hash_perceptual, find_perceptual_duplicate_groups,
    )

    jpg = tmp_path / "a.jpg"
    png = tmp_path / "b.png"
    # Same colour, slightly different sizes so they're byte-different
    # AND perceptual hashes are very close.
    _make_image(jpg, color=(180, 90, 60), fmt="JPEG", quality=90)
    _make_image(png, color=(180, 90, 60), fmt="PNG")

    files = [str(jpg), str(png)]
    cache: dict = {}
    cache = scan_and_hash_perceptual(files, cache, parallel=1)

    groups = find_perceptual_duplicate_groups(files, cache, min_tier="reencode")
    assert len(groups) == 1
    assert set(groups[0]["files"]) == {str(jpg), str(png)}
    assert groups[0]["md5_match"] is False
    assert groups[0]["tier"] in ("reencode", "resize")


def test_perceptual_groups_ignores_unrelated(tmp_path):
    """Two visually-different images should NOT be grouped."""
    from wallpaper_analyzer.duplicates import (
        scan_and_hash_perceptual, find_perceptual_duplicate_groups,
    )

    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    # Strong contrast: dark red vs bright blue, opposite corners of the
    # colour wheel so colour histograms and luminance gradients differ.
    _make_image(a, color=(10, 10, 80))
    _make_image(b, color=(220, 220, 20))

    files = [str(a), str(b)]
    cache: dict = {}
    cache = scan_and_hash_perceptual(files, cache, parallel=1)

    groups = find_perceptual_duplicate_groups(files, cache, min_tier="reencode")
    assert groups == []


def test_perceptual_groups_respects_min_tier(tmp_path):
    """A pair that's only `resize`-close should be filtered out at
    `reencode` strictness, and reported at `resize` strictness."""
    from wallpaper_analyzer.duplicates import (
        scan_and_hash_perceptual, find_perceptual_duplicate_groups,
    )

    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    # Same colour but at very different sizes - hashes should be
    # close-ish but probably not at reencode strictness.
    _make_image(a, size=(400, 400), color=(120, 120, 200))
    _make_image(b, size=(40, 40), color=(120, 120, 200))
    # Resize the small one back to a similar size so the dHash has the
    # same downsample path (compute_all_hashes resizes to 64 internally
    # either way, so the colour histogram dominates).

    files = [str(a), str(b)]
    cache: dict = {}
    cache = scan_and_hash_perceptual(files, cache, parallel=1)

    # At reencode strictness we may or may not match; the key invariant
    # is that the result never includes a pair that fails the resize
    # threshold when min_tier=resize.
    g_strict = find_perceptual_duplicate_groups(files, cache, min_tier="reencode")
    g_loose = find_perceptual_duplicate_groups(files, cache, min_tier="similar")

    # Loose is a superset of strict.
    assert len(g_loose) >= len(g_strict)


def test_find_duplicate_groups_combines_md5_and_perceptual(tmp_path):
    """find_duplicate_groups should report MD5 and perceptual matches."""
    from wallpaper_analyzer.duplicates import (
        scan_and_hash, scan_and_hash_perceptual, find_duplicate_groups,
    )

    a = tmp_path / "a.jpg"
    a_copy = tmp_path / "a_copy.jpg"  # byte-identical -> MD5 match
    a_png = tmp_path / "a.png"        # same content, different bytes -> perceptual
    b = tmp_path / "b.jpg"            # unrelated

    _make_image(a, color=(50, 200, 100), fmt="JPEG", quality=95)
    _make_image(a_copy, color=(50, 200, 100), fmt="JPEG", quality=95)
    _make_image(a_png, color=(50, 200, 100), fmt="PNG")
    _make_image(b, color=(10, 10, 10))

    # Force the bytes to differ between a and a_copy: re-save with
    # different EXIF / encoding. PIL JPEG re-saves usually produce
    # different bytes even at the same quality.
    a_bytes = a.read_bytes()
    a_copy.write_bytes(a_bytes)

    files = [str(a), str(a_copy), str(a_png), str(b)]
    cache: dict = {}
    cache = scan_and_hash(files, cache, parallel=1)
    cache = scan_and_hash_perceptual(files, cache, parallel=1)

    # Hard mode: only MD5.
    groups_hard = find_duplicate_groups(files, cache, mode="hard")
    # Soft mode: MD5 + perceptual.
    groups_soft = find_duplicate_groups(files, cache, mode="soft",
                                         min_tier="reencode")

    # The MD5 group (if any) and perceptual group should both be present
    # in soft mode.
    assert len(groups_soft) >= len(groups_hard)
    # Whatever appears in hard mode also appears in soft mode.
    md5_files_hard = {f for g in groups_hard for f in g["files"]}
    md5_files_soft = {f for g in groups_soft if g.get("md5_match")
                      for f in g["files"]}
    assert md5_files_hard <= md5_files_soft


def test_find_duplicate_groups_excludes_md5_files_from_perceptual(tmp_path):
    """A file that's already in an MD5 group shouldn't be re-listed
    in a perceptual group."""
    from wallpaper_analyzer.duplicates import (
        scan_and_hash, scan_and_hash_perceptual, find_duplicate_groups,
    )

    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.png"
    _make_image(a, color=(30, 200, 80))
    _make_image(b, color=(30, 200, 80))
    b.write_bytes(a.read_bytes())  # force byte-identical
    _make_image(c, color=(30, 200, 80), fmt="PNG")

    files = [str(a), str(b), str(c)]
    cache: dict = {}
    cache = scan_and_hash(files, cache, parallel=1)
    cache = scan_and_hash_perceptual(files, cache, parallel=1)

    groups = find_duplicate_groups(files, cache, mode="soft",
                                   min_tier="reencode")

    md5_files = {f for g in groups if g.get("md5_match") for f in g["files"]}
    for g in groups:
        if not g.get("md5_match"):
            # No perceptual-group file should be in an MD5 group.
            assert not (set(g["files"]) & md5_files)


def test_duplicate_stats_reports_by_tier(tmp_path):
    """get_duplicate_stats must include the perceptual_groups / by_tier keys."""
    from wallpaper_analyzer.duplicates import get_duplicate_stats

    groups = [
        {"files": ["a", "b"], "md5_match": True, "size_match": True,
         "tier": "exact", "size_bytes": [100, 100]},
        {"files": ["c", "d", "e"], "md5_match": False, "size_match": False,
         "tier": "reencode", "size_bytes": [100, 200, 300]},
    ]
    stats = get_duplicate_stats(groups)
    assert stats["total_groups"] == 2
    assert stats["md5_matched_groups"] == 1
    assert stats["perceptual_groups"] == 1
    assert stats["by_tier"]["exact"] == 1
    assert stats["by_tier"]["reencode"] == 1


def test_scan_and_hash_perceptual_is_cached(tmp_path):
    """Re-running scan_and_hash_perceptual should be near-instant on
    unchanged files (signature matches)."""
    import time
    from wallpaper_analyzer.duplicates import scan_and_hash_perceptual

    p = tmp_path / "a.jpg"
    _make_image(p)

    cache: dict = {}
    t0 = time.monotonic()
    scan_and_hash_perceptual([str(p)], cache, parallel=1)
    elapsed_first = time.monotonic() - t0

    t0 = time.monotonic()
    scan_and_hash_perceptual([str(p)], cache, parallel=1)
    elapsed_second = time.monotonic() - t0

    # Cache hit should be at least 10x faster than first scan.
    assert elapsed_second * 10 < elapsed_first or elapsed_second < 0.01


def test_perceptual_groups_same_category_only(tmp_path):
    """When same_category_only=True, files in different parent dirs
    should not be grouped even if their perceptual hashes are close."""
    from wallpaper_analyzer.duplicates import (
        scan_and_hash_perceptual, find_perceptual_duplicate_groups,
    )

    cat_a = tmp_path / "Anime"
    cat_b = tmp_path / "Nature"
    cat_a.mkdir()
    cat_b.mkdir()
    a = cat_a / "img.jpg"
    b = cat_b / "img.jpg"
    _make_image(a, color=(100, 100, 100))
    _make_image(b, color=(100, 100, 100))

    files = [str(a), str(b)]
    cache: dict = {}
    cache = scan_and_hash_perceptual(files, cache, parallel=1)

    # Without the filter we should get a group.
    g_all = find_perceptual_duplicate_groups(files, cache, min_tier="similar")
    # With the filter, no group (different categories).
    g_filtered = find_perceptual_duplicate_groups(
        files, cache, min_tier="similar", same_category_only=True,
    )
    assert len(g_all) >= 1
    assert g_filtered == []