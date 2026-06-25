"""Duplicate detection for wallpapers - MD5 + perceptual.

Two-pass detection:

  Pass 1 - MD5 (cheap, byte-identical):
    Two files are byte-identical duplicates iff they share the same
    full-file MD5 hash. Catches copies / re-downloads cheaply (pure I/O).

  Pass 2 - Perceptual (image-content-aware):
    Catches the case the user actually cares about: same image but
    different bytes (re-encoded JPG vs PNG, re-compressed, resized,
    re-saved, different container, etc.). Uses multi-perceptual hashes
    from `hashing.compute_hashes_multi` (dHash, pHash, aHash, color
    hash, RGB histogram) bucketed via LSH on the coarse 16-bit dHash.

Tiers (loose -> strict, defaults from `TIER_RULES`):

  TIER_REENCODE  max dh 2, max ph 2, hist sim >= 0.97
                 Same image, different format / re-compressed.
  TIER_RESIZE    max dh 8, max ph 10, hist sim >= 0.85
                 Same image at a different resolution.
  TIER_SIMILAR   max dh 14, max ph 16, hist sim >= 0.75
                 Visually similar (broader; use with care).
  TIER_EXACT     MD5 match (byte-identical).

Each tier is a strict subset of the looser ones - if a pair matches
`reencode`, it's also `resize`+`similar`+`exact`-ish. The tier reported
in the result is the strictest one that all three thresholds agree on,
which is the safest signal.

For multi-frame content (animated GIF / WEBP / videos), `frames` from
`compute_hashes_multi` are compared pairwise. A pair matches iff a
majority of frame pairs match (>= MAJORITY_THRESHOLD). This avoids
false positives when only the first frame is identical.

Implementation notes:

  * The cache is a flat dict `{path: {"sig": ..., "md5": ..., "frames": [...],
    "size_bytes": ..., "size": (w, h)}}`. Old MD5-only entries keep working;
    missing `frames` just means the perceptual pass will recompute.
  * `scan_and_hash` adds MD5; `scan_and_hash_perceptual` adds `frames`.
    They share the same (size, mtime) signature so a re-run is cheap.
  * Hashing runs in a process pool (`parallel` argument) so multi-core
    machines scan thousands of files in seconds.
"""
import hashlib
import json
import os
import shutil
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .settings import (
    load_settings,
    resolve_dest_dir,
    resolve_hash_cache_path,
)
from . import hashing as _hashing
from .hashing import (
    hamming_distance,
    histogram_similarity,
)


# ---------------------------------------------------------------------------
# Public tier constants and thresholds
# ---------------------------------------------------------------------------

TIER_EXACT = "exact"            # MD5 match
TIER_REENCODE = "reencode"      # same content, different format/quality
TIER_RESIZE = "resize"          # same content, different resolution
TIER_SIMILAR = "similar"        # visually similar (broader)

# (max_dh, max_ph, min_hist_sim) - thresholds to enter each tier.
# Strictest first (reencode is most strict), so iterating in order
# gives us the strictest tier that both frames pass.
TIER_RULES = {
    TIER_REENCODE: (2, 2, 0.97),
    TIER_RESIZE:   (8, 10, 0.85),
    TIER_SIMILAR: (14, 16, 0.75),
}

TIER_LABELS = {
    TIER_EXACT:    "Exact (same MD5)",
    TIER_REENCODE: "Re-encoded (same image, different bytes)",
    TIER_RESIZE:   "Re-sized (same image, different resolution)",
    TIER_SIMILAR:  "Similar (visually close)",
}

# Order from most-strict to loosest, used when picking the strictest
# tier a pair satisfies.
_TIER_ORDER = [TIER_REENCODE, TIER_RESIZE, TIER_SIMILAR]

# For multi-frame content: pair matches if this fraction of frames match.
MAJORITY_THRESHOLD = 0.5

# LSH: hash buckets are formed on `dh16`. We also probe adjacent
# buckets (Hamming distance <= LSH_NEIGHBOURS) to catch near-collisions.
LSH_NEIGHBOURS = 1

DEFAULT_TIER = TIER_EXACT
MODE_SOFT = "soft"
MODE_HARD = "hard"

# In "soft" mode we report any tier >= dedupe_min_tier (default: reencode).
# In "hard" mode we only report TIER_EXACT (byte-identical).
MODE_DEFAULT_MIN_TIER = {
    MODE_HARD: TIER_EXACT,
    MODE_SOFT: TIER_REENCODE,
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _hash_cache_path() -> str:
    return resolve_hash_cache_path(load_settings())


def _dest_dir() -> str:
    return resolve_dest_dir(load_settings())


def get_hash_cache_path() -> str:
    return _hash_cache_path()


def get_dest_dir() -> str:
    return _dest_dir()


def _file_signature(path: str) -> str:
    """size-mtime signature used to detect changes."""
    try:
        st = os.stat(path)
        return f"{st.st_size}-{int(st.st_mtime)}"
    except Exception:
        return ""


def _md5_of_file(path: str) -> str:
    """Full-file MD5.

    Reads the file in 1 MB chunks so multi-GB videos don't allocate a
    single big buffer. Returns "" on read failure.
    """
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _image_dimensions(path: str) -> Tuple[int, int]:
    """Cheap dimension probe via PIL. Returns (0, 0) on failure."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# Cache load/save
# ---------------------------------------------------------------------------
# Cache entry shape:
#   {
#     "sig":         "size-mtime" signature for change detection,
#     "md5":         full-file MD5 hex digest,
#     "size_bytes":  total file size,
#     "size":        (width, height) of the first frame / image,
#     "frames":      [  # 1+ perceptual hash dicts (multi-frame aware)
#       {
#         "dh":    64-bit dHash int,
#         "dh16":  16-bit coarse dHash int,
#         "ph":    64-bit pHash int,
#         "ah":    64-bit average hash int,
#         "ch16":  16-bit coarse color hash int,
#         "chh":   64-bit color hash int,
#         "hist":  [16 ints] RGB histogram,
#         "t":     timestamp in seconds (0.0 for static images),
#       },
#       ...
#     ],
#   }
#
# Old MD5-only entries are still readable - missing `frames` just means
# the perceptual pass will recompute on next scan.


def load_hash_cache() -> Dict:
    path = _hash_cache_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def save_hash_cache(cache: Dict) -> None:
    path = _hash_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(cache, fp, indent=2)
    except Exception as exc:
        print(f"  [WARN] Could not save hash cache: {exc}")


def _serialize_frames_for_cache(frames: List[Dict]) -> List[Dict]:
    """Convert in-memory frame hashes (numpy ints etc.) to JSON-safe types."""
    out = []
    for f in frames:
        out.append({
            "dh":   int(f.get("dh", 0)),
            "dh16": int(f.get("dh16", 0)),
            "ph":   int(f.get("ph", 0)),
            "ah":   int(f.get("ah", 0)),
            "ch16": int(f.get("ch16", 0)),
            "chh":  int(f.get("chh", 0)),
            "hist": [int(x) for x in f.get("hist", [])],
            "t":    float(f.get("t", 0.0)),
            "size": list(f.get("size", (0, 0))),
        })
    return out


def _deserialize_frames_from_cache(frames: List[Dict]) -> List[Dict]:
    """Inverse of `_serialize_frames_for_cache`."""
    out = []
    for f in frames:
        size = f.get("size") or [0, 0]
        out.append({
            "dh":   int(f.get("dh", 0)),
            "dh16": int(f.get("dh16", 0)),
            "ph":   int(f.get("ph", 0)),
            "ah":   int(f.get("ah", 0)),
            "ch16": int(f.get("ch16", 0)),
            "chh":  int(f.get("chh", 0)),
            "hist": [int(x) for x in f.get("hist", [])],
            "t":    float(f.get("t", 0.0)),
            "size": tuple(size) if isinstance(size, list) else size,
        })
    return out


# ---------------------------------------------------------------------------
# Workers - picklable so they can run in a process pool
# ---------------------------------------------------------------------------

def _md5_worker(path: str) -> Optional[Dict]:
    """Picklable worker: returns {"path": path, "entry": {sig, md5, ...}}."""
    try:
        sig = _file_signature(path)
    except Exception:
        return None
    md5 = _md5_of_file(path)
    if not md5:
        return None
    try:
        size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    except OSError:
        size_bytes = 0
    return {
        "path": path,
        "entry": {
            "sig": sig,
            "md5": md5,
            "size_bytes": size_bytes,
            "size": _image_dimensions(path),
        },
    }


def _perceptual_worker(path: str) -> Optional[Dict]:
    """Picklable worker: returns perceptual hash frames for `path`.

    Computes multi-frame hashes via `compute_hashes_multi` (handles
    static images, animated GIF/WEBP, and videos via ffmpeg). Returns
    None on failure (corrupt file, IO error, missing ffmpeg for video).
    """
    try:
        sig = _file_signature(path)
    except Exception:
        return None
    try:
        frames = _hashing.compute_hashes_multi(path)
    except Exception:
        frames = []
    if not frames:
        return None
    try:
        size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    except OSError:
        size_bytes = 0
    size = frames[0].get("size", (0, 0)) if frames else _image_dimensions(path)
    return {
        "path": path,
        "entry": {
            "sig": sig,
            "size_bytes": size_bytes,
            "size": size,
            "frames": _serialize_frames_for_cache(frames),
        },
    }


# ---------------------------------------------------------------------------
# Public scan + group API
# ---------------------------------------------------------------------------

def scan_and_hash(
    files: List[str],
    cache: Dict,
    progress_callback: Optional[Callable] = None,
    save_every: int = 100,
    parallel: int = 1,
) -> Dict:
    """Update the cache with MD5 hashes for every file in `files`.

    Files whose (size, mtime) signature matches the cache entry are
    skipped (assumed unchanged). When `parallel > 1`, the actual
    hashing runs in a `ProcessPoolExecutor` so multi-core machines
    can scan thousands of files in seconds.

    The cache is updated in place and saved to disk every `save_every`
    files. Returns the cache for convenience.
    """
    from .parallel import run_parallel

    total = len(files)
    if total == 0:
        save_hash_cache(cache)
        return cache

    todo = []
    for path in files:
        try:
            sig = _file_signature(path)
        except Exception:
            continue
        entry = cache.get(path) or {}
        if entry.get("sig") != sig or not entry.get("md5"):
            todo.append((path, sig))

    done = total - len(todo)
    if progress_callback and done:
        for path in files:
            if all(path != t[0] for t in todo):
                try:
                    progress_callback(done, total, path, "cached")
                except Exception:
                    pass

    if not todo:
        if total:
            save_hash_cache(cache)
        return cache

    paths = [t[0] for t in todo]

    if parallel > 1 and len(paths) > 1:

        def _on_progress(_done, _total, _item, _res):
            if progress_callback:
                try:
                    progress_callback(done + _done, total, _item, "ok")
                except Exception:
                    pass

        def _on_error(item, exc):
            if progress_callback:
                try:
                    progress_callback(done, total, item, f"err: {exc}")
                except Exception:
                    pass

        results = run_parallel(
            _md5_worker, paths,
            max_workers=parallel,
            on_progress=_on_progress,
            on_error=_on_error,
            use_processes=True,
        )
        for r in results:
            if r is None:
                continue
            cache[r["path"]] = r["entry"]
            done += 1
            if save_every and done % save_every == 0:
                save_hash_cache(cache)
    else:
        for i, path in enumerate(paths):
            res = _md5_worker(path)
            if res is not None:
                cache[res["path"]] = res["entry"]
            done += 1
            if progress_callback and (i % 10 == 0 or i == len(paths) - 1):
                try:
                    progress_callback(done, total, path, "ok")
                except Exception:
                    pass
            if save_every and done % save_every == 0:
                save_hash_cache(cache)

    save_hash_cache(cache)
    return cache


def scan_and_hash_perceptual(
    files: List[str],
    cache: Dict,
    progress_callback: Optional[Callable] = None,
    save_every: int = 100,
    parallel: int = 1,
) -> Dict:
    """Add perceptual hashes (`frames`) to the cache for every file in `files`.

    Files whose (size, mtime) signature matches AND that already have
    `frames` in the cache are skipped (assumed unchanged). New or
    changed files get hashed via `compute_hashes_multi` so animated /
    video content is handled correctly.

    Designed to be called AFTER `scan_and_hash` (which adds `md5`) so
    the cache has both signatures for the same files. The two passes
    are independent - missing `frames` for a file just means it won't
    participate in perceptual grouping.

    The cache is updated in place and saved to disk every `save_every`
    files. Returns the cache for convenience.
    """
    from .parallel import run_parallel

    total = len(files)
    if total == 0:
        save_hash_cache(cache)
        return cache

    todo = []
    for path in files:
        try:
            sig = _file_signature(path)
        except Exception:
            continue
        entry = cache.get(path) or {}
        if entry.get("sig") != sig or not entry.get("frames"):
            todo.append((path, sig))

    done = total - len(todo)
    if progress_callback and done:
        try:
            progress_callback(done, total, files[0], "cached")
        except Exception:
            pass

    if not todo:
        if total:
            save_hash_cache(cache)
        return cache

    paths = [t[0] for t in todo]

    if parallel > 1 and len(paths) > 1:

        def _on_progress(_done, _total, _item, _res):
            if progress_callback:
                try:
                    progress_callback(done + _done, total, _item, "ok")
                except Exception:
                    pass

        def _on_error(item, exc):
            if progress_callback:
                try:
                    progress_callback(done, total, item, f"err: {exc}")
                except Exception:
                    pass

        results = run_parallel(
            _perceptual_worker, paths,
            max_workers=parallel,
            on_progress=_on_progress,
            on_error=_on_error,
            use_processes=True,
        )
        for r in results:
            if r is None:
                continue
            path = r["path"]
            existing = cache.get(path) or {}
            existing.update(r["entry"])
            cache[path] = existing
            done += 1
            if save_every and done % save_every == 0:
                save_hash_cache(cache)
    else:
        for i, path in enumerate(paths):
            res = _perceptual_worker(path)
            if res is not None:
                path2 = res["path"]
                existing = cache.get(path2) or {}
                existing.update(res["entry"])
                cache[path2] = existing
            done += 1
            if progress_callback and (i % 10 == 0 or i == len(paths) - 1):
                try:
                    progress_callback(done, total, path, "ok")
                except Exception:
                    pass
            if save_every and done % save_every == 0:
                save_hash_cache(cache)

    save_hash_cache(cache)
    return cache


# ---------------------------------------------------------------------------
# Perceptual comparison primitives
# ---------------------------------------------------------------------------

def _frame_tier(f1: Dict, f2: Dict) -> Optional[str]:
    """Return the strictest tier (TIER_REENCODE / TIER_RESIZE / TIER_SIMILAR)
    that both frames satisfy, or None if they don't match at any tier.

    The strictest tier that satisfies all three thresholds (dh, ph, hist)
    is reported - this is the safest signal because the reencode tier
    rules are a subset of resize rules which are a subset of similar
    rules.
    """
    if not f1 or not f2:
        return None
    hist1 = f1.get("hist") or []
    hist2 = f2.get("hist") or []
    hist_sim = histogram_similarity(hist1, hist2) if hist1 and hist2 else 0.0
    dh = hamming_distance(int(f1.get("dh", 0)), int(f2.get("dh", 0)))
    ph = hamming_distance(int(f1.get("ph", 0)), int(f2.get("ph", 0)))

    # Strictest -> loosest: pick the strictest that ALL three pass.
    for tier in _TIER_ORDER:
        max_dh, max_ph, min_hist = TIER_RULES[tier]
        if dh <= max_dh and ph <= max_ph and hist_sim >= min_hist:
            return tier
    return None


def _files_match_tier(frames1: List[Dict], frames2: List[Dict]) -> Optional[str]:
    """Decide if two files (each with 1+ frames) match, and at what tier.

    Returns the strictest tier whose threshold is satisfied by a
    majority of frame pairs, or None if no tier passes. Returns
    TIER_EXACT for byte-identical files (caller checks MD5 separately).
    """
    if not frames1 or not frames2:
        return None
    # Try tiers strictest first; pick the strictest whose threshold is
    # met by a majority of frame pairs.
    n1, n2 = len(frames1), len(frames2)
    if n1 == 0 or n2 == 0:
        return None
    # Greedy bipartite matching between frames, ordered by tier score.
    # For static images (1 frame each) this is a single comparison.
    for tier in _TIER_ORDER:
        max_dh, max_ph, min_hist = TIER_RULES[tier]
        matches = 0
        total_pairs = 0
        for f1 in frames1:
            for f2 in frames2:
                total_pairs += 1
                if (hamming_distance(int(f1.get("dh", 0)), int(f2.get("dh", 0))) <= max_dh
                        and hamming_distance(int(f1.get("ph", 0)), int(f2.get("ph", 0))) <= max_ph):
                    hist_sim = histogram_similarity(f1.get("hist") or [], f2.get("hist") or [])
                    if hist_sim >= min_hist:
                        matches += 1
        if total_pairs > 0 and matches / total_pairs >= MAJORITY_THRESHOLD:
            return tier
    return None


# ---------------------------------------------------------------------------
# LSH bucketing + union-find for grouping
# ---------------------------------------------------------------------------

class _UnionFind:
    """Disjoint-set forest with path compression + union by rank."""

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _lsh_neighbors(bucket_key: int, max_dist: int) -> List[int]:
    """Yield all 16-bit ints within Hamming distance <= max_dist of
    `bucket_key`. With max_dist=1 we generate 17 keys (the bucket itself
    plus 16 single-bit flips); with max_dist=2 we'd generate 137 keys.

    Kept small (LSH_NEIGHBOURS=1) so the per-bucket work stays linear.
    """
    out = [bucket_key]
    if max_dist <= 0:
        return out
    for bit in range(16):
        flipped = bucket_key ^ (1 << bit)
        if flipped not in out:
            out.append(flipped)
    return out


def find_perceptual_duplicate_groups(
    files: List[str],
    cache: Dict,
    min_tier: Optional[str] = None,
    same_category_only: bool = False,
    progress_callback: Optional[Callable] = None,
) -> List[Dict]:
    """Group files by perceptual similarity using multi-hash LSH.

    Files that already share the same MD5 are NOT grouped here - that's
    handled by `find_md5_duplicate_groups` and merged by the caller. We
    only consider files whose MD5s are distinct (otherwise the
    MD5-match is reported at a stricter tier).

    Args:
        files: list of absolute paths to consider.
        cache: dict from `scan_and_hash_perceptual` (or merged with
            MD5 entries); entries must have `frames` to participate.
        min_tier: strictest tier to include in the output (default:
            TIER_REENCODE). Set to TIER_SIMILAR for the broadest matches.
        same_category_only: when True, only group files that share the
            same parent directory basename (category).

    Returns:
        List of groups: `[{"files": [...], "tier": str, "score": float,
        "md5_match": False, "frame_count_total": int, ...}]`. Only
        groups with 2+ members are returned. Files within each group
        are sorted largest-first.
    """
    if min_tier is None:
        min_tier = TIER_REENCODE

    # Index by path -> frames. Skip files without `frames` or whose
    # MD5 already matched (those are handled by find_md5_duplicate_groups
    # at a stricter tier).
    md5_by_path: Dict[str, str] = {}
    for f in files:
        entry = cache.get(f) or {}
        m = entry.get("md5")
        if m:
            md5_by_path[f] = m

    by_md5: Dict[str, List[str]] = defaultdict(list)
    for f, m in md5_by_path.items():
        by_md5[m].append(f)
    already_grouped = {f for group in by_md5.values() if len(group) > 1 for f in group}

    candidate_paths = [
        f for f in files
        if f not in already_grouped
        and (cache.get(f) or {}).get("frames")
    ]
    if len(candidate_paths) < 2:
        return []

    # Deserialize frames once.
    frames_by_path: Dict[str, List[Dict]] = {}
    for f in candidate_paths:
        frames_by_path[f] = _deserialize_frames_from_cache(
            (cache.get(f) or {}).get("frames") or []
        )

    if same_category_only:
        # Restrict pairwise comparisons to same-category files.
        cat_by_path = {f: os.path.basename(os.path.dirname(f)) for f in candidate_paths}

    # LSH bucket on dh16 of the FIRST frame. Multi-frame content has a
    # single bucket anchor (its primary frame) - good enough for
    # initial filtering; the verification step handles frame-level
    # matching.
    bucket_paths: Dict[int, List[str]] = defaultdict(list)
    for f in candidate_paths:
        fr = frames_by_path[f]
        if not fr:
            continue
        key = int(fr[0].get("dh16", 0)) & 0xFFFF
        bucket_paths[key].append(f)

    # Decide which bucket pairs to verify. With LSH_NEIGHBOURS=1 we
    # verify each bucket plus its 16 single-bit-flip neighbours.
    # Union-find merges all matching files into groups.
    uf = _UnionFind(candidate_paths)

    bucket_keys = sorted(bucket_paths.keys())
    verified_pairs = 0
    total_pairs_target = sum(len(v) for v in bucket_paths.values()) ** 2 // 2

    if progress_callback:
        try:
            progress_callback(0, max(total_pairs_target, 1), "", "verify")
        except Exception:
            pass

    # Pre-group files by hash for fast candidate lookups within a bucket.
    for key in bucket_keys:
        members = bucket_paths[key]
        if len(members) < 2:
            continue
        # Compare every pair within this bucket.
        for i in range(len(members)):
            a = members[i]
            frames_a = frames_by_path.get(a) or []
            if not frames_a:
                continue
            for j in range(i + 1, len(members)):
                b = members[j]
                if same_category_only and cat_by_path.get(a) != cat_by_path.get(b):
                    continue
                frames_b = frames_by_path.get(b) or []
                if not frames_b:
                    continue
                tier = _files_match_tier(frames_a, frames_b)
                if tier is None:
                    continue
                # Is this tier >= min_tier (in strictness order)?
                tier_rank = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else -1
                min_rank = _TIER_ORDER.index(min_tier) if min_tier in _TIER_ORDER else 0
                if tier_rank > min_rank:
                    continue
                uf.union(a, b)
                verified_pairs += 1

        # Also probe adjacent buckets (Hamming <= LSH_NEIGHBOURS) to
        # catch near-collisions.
        if LSH_NEIGHBOURS > 0:
            for nb_key in _lsh_neighbors(key, LSH_NEIGHBOURS):
                if nb_key == key or nb_key not in bucket_paths:
                    continue
                other = bucket_paths[nb_key]
                if not other:
                    continue
                for a in members:
                    frames_a = frames_by_path.get(a) or []
                    if not frames_a:
                        continue
                    for b in other:
                        if same_category_only and cat_by_path.get(a) != cat_by_path.get(b):
                            continue
                        frames_b = frames_by_path.get(b) or []
                        if not frames_b:
                            continue
                        tier = _files_match_tier(frames_a, frames_b)
                        if tier is None:
                            continue
                        tier_rank = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else -1
                        min_rank = _TIER_ORDER.index(min_tier) if min_tier in _TIER_ORDER else 0
                        if tier_rank > min_rank:
                            continue
                        uf.union(a, b)
                        verified_pairs += 1

    if progress_callback:
        try:
            progress_callback(verified_pairs, max(verified_pairs, 1), "", "verify-done")
        except Exception:
            pass

    # Collect union-find groups with 2+ members.
    members_by_root: Dict[str, List[str]] = defaultdict(list)
    for f in candidate_paths:
        members_by_root[uf.find(f)].append(f)

    def _by_size_desc(group: List[str]) -> List[str]:
        return sorted(
            group,
            key=lambda f: -os.path.getsize(f) if os.path.exists(f) else 0,
        )

    out: List[Dict] = []
    for root, members in members_by_root.items():
        if len(members) < 2:
            continue
        ordered = _by_size_desc(members)
        # Pick the strictest tier that all pairs within the group agree on.
        # Use the strictest pairwise match as the group tier - this is
        # the safest label for the GUI ("exact-ish" / "re-encoded" etc.).
        group_tier = TIER_SIMILAR
        # Default to looser if nothing stronger applies
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                t = _files_match_tier(frames_by_path[ordered[i]], frames_by_path[ordered[j]])
                if t is None:
                    continue
                if _TIER_ORDER.index(t) < _TIER_ORDER.index(group_tier):
                    group_tier = t
        sizes = set()
        size_bytes = []
        total_frames = 0
        for f in ordered:
            sz = (cache.get(f) or {}).get("size")
            if sz:
                sizes.add(tuple(sz))
            try:
                size_bytes.append(os.path.getsize(f))
            except OSError:
                pass
            total_frames += len(frames_by_path.get(f) or [])
        tier_rank = _TIER_ORDER.index(group_tier) if group_tier in _TIER_ORDER else len(_TIER_ORDER)
        out.append({
            "files": ordered,
            "md5": None,
            "md5_match": False,
            "size_match": len(sizes) == 1,
            "tier": group_tier,
            "score": 1.0 - (tier_rank / max(len(_TIER_ORDER), 1)),
            "avg_similarity": 1.0,
            "frame_count_total": total_frames,
            "matched_frame_count_total": total_frames,
            "size_bytes": size_bytes,
        })

    out.sort(key=lambda g: (-len(g["files"]), g["tier"]))
    return out


# ---------------------------------------------------------------------------
# MD5-only grouping (cheap first pass)
# ---------------------------------------------------------------------------

def find_md5_duplicate_groups(
    files: List[str],
    md5s: Optional[Dict[str, str]] = None,
    cache: Optional[Dict] = None,
    same_category_only: bool = False,
) -> List[Dict]:
    """Group files that share the same MD5 hash (byte-identical).

    Either pass an explicit `md5s` dict (from `compute_md5s`) OR a
    `cache` dict that already has `md5` entries from `scan_and_hash`.

    Returns a list of `{"files": [...sorted largest first...], "md5": "...",
    "tier": "exact", ...}` only for groups with more than one member.
    """
    # Resolve md5s from either parameter.
    resolved: Dict[str, str] = {}
    if md5s is not None:
        resolved.update(md5s)
    if cache is not None:
        for f in files:
            entry = cache.get(f) or {}
            m = entry.get("md5")
            if m and f not in resolved:
                resolved[f] = m

    def _by_size_desc(group: List[str]) -> List[str]:
        return sorted(
            group,
            key=lambda f: -os.path.getsize(f) if os.path.exists(f) else 0,
        )

    by_md5: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        m = resolved.get(f)
        if not m:
            continue
        by_md5[m].append(f)

    groups: List[Dict] = []
    for md, group in by_md5.items():
        if same_category_only:
            by_cat: Dict[str, List[str]] = defaultdict(list)
            for f in group:
                by_cat[os.path.basename(os.path.dirname(f))].append(f)
            for cat_group in by_cat.values():
                if len(cat_group) > 1:
                    ordered = _by_size_desc(cat_group)
                    groups.append(_make_group_dict(md, ordered, cache))
        else:
            if len(group) > 1:
                ordered = _by_size_desc(group)
                groups.append(_make_group_dict(md, ordered, cache))

    groups.sort(key=lambda g: (-len(g["files"]), g["md5"]))
    return groups


def _make_group_dict(md5: str, ordered_files: List[str], cache: Optional[Dict]) -> Dict:
    """Build a group dict in the legacy schema so the GUI keeps working."""
    sizes = set()
    size_bytes = []
    for f in ordered_files:
        if cache is not None:
            sz = (cache.get(f) or {}).get("size")
            if sz:
                sizes.add(tuple(sz))
        try:
            size_bytes.append(os.path.getsize(f))
        except OSError:
            pass
    return {
        "files": ordered_files,
        "md5": md5,
        "md5_match": True,
        "size_match": len(sizes) == 1,
        "score": 1.0,
        "tier": TIER_EXACT,
        "avg_similarity": 1.0,
        "frame_coverage": 1.0,
        "frame_count_total": len(ordered_files),
        "matched_frame_count_total": len(ordered_files),
        "size_bytes": size_bytes,
    }


# ---------------------------------------------------------------------------
# Combined public API
# ---------------------------------------------------------------------------

def find_duplicate_groups(
    files: List[str],
    cache: Dict,
    mode: str = MODE_SOFT,
    min_tier: Optional[str] = None,
    same_category_only: bool = False,
    progress_callback: Optional[Callable] = None,
    include_perceptual: Optional[bool] = None,
) -> List[Dict]:
    """Find groups of duplicate files (MD5 + perceptual).

    Returns MD5 groups first (strictest tier), then perceptual groups,
    de-duplicated so files that appear in an MD5 group don't appear in
    a perceptual one. Within each tier, groups are sorted by size
    (largest first) so the user keeps the highest-resolution copy.

    Args:
        files: list of file paths.
        cache: dict from `scan_and_hash` / `scan_and_hash_perceptual`.
        mode: "hard" -> MD5-only; "soft" -> MD5 + perceptual (any tier
            >= `min_tier`). Default: "soft".
        min_tier: strictest tier to include in perceptual pass. Default
            for soft mode is `TIER_REENCODE`. Ignored in hard mode.
        same_category_only: when True, only group files that share the
            same parent directory basename.
        include_perceptual: explicit override; defaults to (mode == "soft").
    """
    md5_groups = find_md5_duplicate_groups(
        files,
        cache=cache,
        same_category_only=same_category_only,
    )

    if include_perceptual is None:
        include_perceptual = (mode == MODE_SOFT)
    if not include_perceptual:
        return md5_groups

    if min_tier is None:
        min_tier = MODE_DEFAULT_MIN_TIER.get(mode, TIER_REENCODE)

    perceptual_groups = find_perceptual_duplicate_groups(
        files, cache,
        min_tier=min_tier,
        same_category_only=same_category_only,
        progress_callback=progress_callback,
    )

    # Files that are already in an MD5 group don't need a perceptual
    # group - the byte-identical match is reported at a stricter tier.
    md5_files = {f for g in md5_groups for f in g["files"]}
    filtered_perceptual = []
    for g in perceptual_groups:
        remaining = [f for f in g["files"] if f not in md5_files]
        if len(remaining) >= 2:
            g2 = dict(g)
            g2["files"] = sorted(
                remaining,
                key=lambda f: -os.path.getsize(f) if os.path.exists(f) else 0,
            )
            filtered_perceptual.append(g2)

    return md5_groups + filtered_perceptual


# ---------------------------------------------------------------------------
# File operations (delete / move to Duplicates/)
# ---------------------------------------------------------------------------

def delete_files(paths: List[str]) -> int:
    deleted = 0
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
                deleted += 1
        except Exception as e:
            print(f"  [ERR] Could not delete {p}: {e}")
    return deleted


def move_to_duplicates(paths: List[str], base_dir: str = "") -> int:
    if not base_dir:
        base_dir = _dest_dir()
    dup_dir = os.path.join(base_dir, "Duplicates")
    os.makedirs(dup_dir, exist_ok=True)
    moved = 0
    for p in paths:
        try:
            if not os.path.exists(p):
                continue
            dst = os.path.join(dup_dir, os.path.basename(p))
            if os.path.exists(dst):
                name, ext = os.path.splitext(os.path.basename(dst))
                n = 1
                while os.path.exists(os.path.join(dup_dir, f"{name}_{n}{ext}")):
                    n += 1
                dst = os.path.join(dup_dir, f"{name}_{n}{ext}")
            shutil.move(p, dst)
            moved += 1
        except Exception as e:
            print(f"  [ERR] Could not move {p}: {e}")
    return moved


def move_duplicates(
    groups: List[List[str]],
    base_dir: str = "",
    keep_first: bool = True,
) -> int:
    """Backwards-compatible helper: groups is a list of lists."""
    if not base_dir:
        base_dir = _dest_dir()
    dup_dir = os.path.join(base_dir, "Duplicates")
    os.makedirs(dup_dir, exist_ok=True)
    moved = 0
    for group in groups:
        if keep_first:
            order = sorted(
                group,
                key=lambda f: -os.path.getsize(os.path.join(base_dir, f))
                if os.path.exists(os.path.join(base_dir, f)) else 0,
            )
            to_move = order[1:]
        else:
            to_move = group
        for f in group:
            src = os.path.join(base_dir, f) if not os.path.isabs(f) else f
            if not os.path.exists(src):
                continue
            dst = os.path.join(dup_dir, os.path.basename(src))
            if os.path.exists(dst):
                name, ext = os.path.splitext(os.path.basename(dst))
                dst = os.path.join(dup_dir, f"{name}_dup{ext}")
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as exc:
                print(f"  [ERR] Could not move {src}: {exc}")
    return moved


def get_duplicate_stats(groups: List[Dict]) -> Dict:
    total_dupes = sum(len(g["files"]) - 1 for g in groups)
    total_groups = len(groups)
    same_size = sum(1 for g in groups if g.get("size_match"))
    by_tier: Dict[str, int] = defaultdict(int)
    by_kind: Dict[str, int] = defaultdict(int)
    wasted_bytes = 0
    for g in groups:
        files = g["files"]
        if len(files) < 2:
            continue
        for f in files[1:]:
            try:
                wasted_bytes += os.path.getsize(f)
            except OSError:
                pass
        by_tier[g.get("tier", TIER_EXACT)] += 1
        kinds = {os.path.splitext(f)[1].lower() for f in files}
        for k in kinds:
            by_kind[k] += 1
    return {
        "total_groups": total_groups,
        "total_duplicates": total_dupes,
        "size_matched_groups": same_size,
        "md5_matched_groups": sum(1 for g in groups if g.get("md5_match")),
        "perceptual_groups": sum(1 for g in groups if not g.get("md5_match")),
        "by_tier": dict(by_tier),
        "by_kind": dict(by_kind),
        "wasted_bytes": wasted_bytes,
        "wasted_mb": wasted_bytes / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Backwards-compat constants (older code referenced these)
# ---------------------------------------------------------------------------

DUPE_DHASH_THRESHOLD = TIER_RULES[TIER_SIMILAR][0]
DUPE_PHASH_THRESHOLD = TIER_RULES[TIER_SIMILAR][1]
DUPE_HIST_THRESHOLD = 1.0 - TIER_RULES[TIER_SIMILAR][2]
DUPE_NEIGHBORS = LSH_NEIGHBOURS
DUPE_MODE_SOFT = MODE_SOFT
DUPE_MODE_HARD = MODE_HARD