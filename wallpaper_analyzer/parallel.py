"""Parallel execution helpers.

The CPU-heavy parts of wallpaper analysis (image hashing, CV feature
extraction, classification) are bottlenecked by Python's GIL when run
in threads. To get true parallelism we need to:

  * Run CPU work in `concurrent.futures.ProcessPoolExecutor` so each
    worker has its own interpreter and runs on a separate core.
  * Keep I/O-bound work (Ollama network calls, file reading) on
    threads, since the GIL is released during socket reads.
  * Stream progress back to the main process via a multiprocessing-safe
    queue so the UI never freezes.

This module also exposes a `is_free_threaded()` probe that the GUI can
use to show a "Free-threaded Python detected" badge when the no-GIL
build is in use.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def is_free_threaded() -> bool:
    """True if the current Python interpreter was built without the GIL
    (PEP 703 / `Py_GIL_DISABLED=1`)."""
    gil = getattr(sys.flags, "gil", None)
    return gil is None or gil == 0


def cpu_count() -> int:
    """Return the number of CPUs available for CPU-bound work.

    Honours the standard affinity / container env vars (OMP_NUM_THREADS
    etc. are not honoured here, only scheduling affinity).
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return mp.cpu_count() or 1


# ---------------------------------------------------------------------------
# Thread pool (I/O) - never blocks the UI
# ---------------------------------------------------------------------------

_io_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_io_executor_lock = threading.Lock()


def get_io_executor(max_workers: int = 8) -> concurrent.futures.ThreadPoolExecutor:
    """Lazy module-level ThreadPoolExecutor for I/O-bound tasks.

    Reusing one pool avoids the per-call overhead of creating/destroying
    worker threads (especially for repeated Ollama API calls).
    """
    global _io_executor
    with _io_executor_lock:
        if _io_executor is None or _io_executor._max_workers != max_workers:
            if _io_executor is not None:
                _io_executor.shutdown(wait=False, cancel_futures=True)
            _io_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="wa-io",
            )
        return _io_executor


def submit_io(fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
    """Submit a function to the shared I/O thread pool."""
    return get_io_executor().submit(fn, *args, **kwargs)


def shutdown_io_executor(wait: bool = False) -> None:
    global _io_executor
    with _io_executor_lock:
        if _io_executor is not None:
            _io_executor.shutdown(wait=wait, cancel_futures=not wait)
            _io_executor = None


# ---------------------------------------------------------------------------
# Process pool (CPU) - real parallelism regardless of GIL
# ---------------------------------------------------------------------------

def run_parallel(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, Any, Any], None]] = None,
    on_error: Optional[Callable[[Any, BaseException], None]] = None,
    use_processes: bool = True,
    initializer: Optional[Callable] = None,
    initargs: Tuple = (),
) -> List[Any]:
    """Run `fn(item)` in parallel over `items`.

    Uses `ProcessPoolExecutor` by default (true parallelism, no GIL) so
    CPU-bound work scales linearly with core count even on the regular
    CPython build. If `fn` is not picklable or the caller passes
    `use_processes=False`, falls back to a thread pool.

    Args:
        fn: callable applied to each item; must return a result.
        items: iterable of work items.
        max_workers: max number of workers (defaults to cpu_count()).
        on_progress: optional callback(done, total, item, result).
        on_error: optional callback(item, exc); if not provided, errors
            are re-raised after all items finish.
        use_processes: prefer processes (default) or threads.

    Returns:
        List of results in the order items were submitted.
    """
    items = list(items)
    total = len(items)
    if total == 0:
        return []
    if total == 1 or max_workers == 1:
        return [_run_one(fn, item, on_progress, i + 1, total) for i, item in enumerate(items)]

    n_workers = max(1, min(max_workers or cpu_count(), total))
    results: List[Any] = [None] * total  # type: ignore

    executor_cls = (
        concurrent.futures.ProcessPoolExecutor
        if use_processes
        else concurrent.futures.ThreadPoolExecutor
    )
    # Each worker process initialises once (loads numpy/Pillow, sets the
    # thread count, etc.) so we don't pay the import cost per item.
    initializer = initializer
    if executor_cls is concurrent.futures.ProcessPoolExecutor and initializer is None:
        # Default: clamp BLAS/OpenMP threads inside workers so they don't
        # oversubscribe (would defeat the purpose of multi-process).
        def _init_worker():
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
        initializer = _init_worker

    try:
        with executor_cls(
            max_workers=n_workers,
            initializer=initializer,
            initargs=initargs,
        ) as pool:
            fut_to_idx = {
                pool.submit(fn, item): (i, item) for i, item in enumerate(items)
            }
            for fut in concurrent.futures.as_completed(fut_to_idx):
                idx, item = fut_to_idx[fut]
                try:
                    res = fut.result()
                    results[idx] = res
                    if on_progress is not None:
                        on_progress(idx + 1, total, item, res)
                except Exception as exc:
                    if on_error is not None:
                        on_error(item, exc)
                        results[idx] = None
                    else:
                        raise
    except (pickle.PicklingError, AttributeError, TypeError):
        if not use_processes:
            raise
        # fn wasn't picklable - retry with threads as a safety net.
        return run_parallel(
            fn, items, max_workers=max_workers, on_progress=on_progress,
            on_error=on_error, use_processes=False,
            initializer=initializer, initargs=initargs,
        )
    return results


def _run_one(fn, item, on_progress, done, total):
    res = fn(item)
    if on_progress is not None:
        on_progress(done, total, item, res)
    return res


# Late import so the module can still be imported even if pickle is missing.
import pickle  # noqa: E402


# ---------------------------------------------------------------------------
# Asynchronous progress reporter (used by long-running workers)
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Multiprocessing-safe progress reporter.

    Workers call `update(done, total, *info)` from inside a process and
    the main process can poll `latest()` from any thread (e.g. the Qt
    UI thread) without blocking.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._done = 0
        self._total = 0
        self._latest: Tuple[int, int, Any] = (0, 0, None)
        self._started = time.monotonic()

    def update(self, done: int, total: int, info: Any = None) -> None:
        with self._lock:
            self._done = done
            self._total = total
            self._latest = (done, total, info)

    def snapshot(self) -> Tuple[int, int, Any, float]:
        with self._lock:
            return (self._done, self._total, self._latest[2],
                    time.monotonic() - self._started)


def estimate_eta(done: int, total: int, elapsed: float) -> Optional[float]:
    """Return a wall-clock ETA in seconds, or None if not enough data."""
    if done <= 0 or elapsed <= 0:
        return None
    if done >= total:
        return 0.0
    rate = done / elapsed
    remaining = (total - done) / max(rate, 1e-9)
    return remaining
