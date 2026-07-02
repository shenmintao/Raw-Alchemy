"""Phase 7 review M1/M2/M3/m6: CachedImage output-slot locking.

Two processor QThreads share one ImageCacheManager (main + baseline), and the
GUI thread triggers lock-held field-level eviction via ``set_limit_mb``. The
CachedImage entry itself therefore needs its own lock:

- ``store_output``/``get_output`` (worker threads, no manager lock) mutate the
  ``_output_slots`` OrderedDict while ``update_size`` (manager lock held)
  iterates it -> RuntimeError "OrderedDict mutated during iteration" (M1a/M3).
- ``get_output`` re-read the primary fields after the null check, so a
  concurrent ``drop_stage('output')`` could produce a torn ``(None, ev, size)``
  hit (M2), and interleaved ``store_output`` calls could pair one key's image
  with another key's EV/source size (M1c/m6).
- ``drop_stage`` returned ``before - after`` and the manager skipped negative
  deltas, so unaccounted in-place growth discovered during eviction left
  ``current_memory_mb`` permanently out of sync with the entries (M1b).

These tests hammer the same interleavings from plain threads; they are
probabilistic repros pre-fix (they failed reliably before the entry lock was
added) and deterministic guards post-fix.
"""
import sys
import threading

import numpy as np
import pytest

from raw_alchemy.pipeline import cache_manager as cache_manager_module
from raw_alchemy.pipeline.cache_manager import (
    OUTPUT_SLOT_LIMIT,
    CachedImage,
    ImageCacheManager,
)


class _FakeVirtualMemory:
    def __init__(self, available):
        self.available = available


@pytest.fixture
def huge_available_ram(monkeypatch):
    """Make the relative quota term huge so the absolute cap dominates."""
    monkeypatch.setattr(
        cache_manager_module.psutil,
        "virtual_memory",
        lambda: _FakeVirtualMemory(available=1 << 50),
    )


@pytest.fixture
def fast_thread_switching():
    """Shrink the GIL switch interval so race windows are hit quickly."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    yield
    sys.setswitchinterval(old)


@pytest.fixture
def quiet_cache_logs():
    """The eviction hammer would otherwise emit thousands of log lines."""
    from loguru import logger
    logger.disable("raw_alchemy.pipeline.cache_manager")
    yield
    logger.enable("raw_alchemy.pipeline.cache_manager")


def _mb_array(mb: float, dtype=np.float32) -> np.ndarray:
    """An ndarray whose logical size is `mb` MB without physical allocation."""
    count = int(mb * 1024 * 1024 // np.dtype(dtype).itemsize)
    return np.broadcast_to(np.zeros(1, dtype=dtype), (count,))


def _run_threads(threads):
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# =====================================================================
# M1a/M3: store/get on worker threads vs lock-held update_size iteration
# =====================================================================


def test_concurrent_store_get_vs_update_size_does_not_crash(fast_thread_switching):
    """update_size iterates _output_slots while another thread inserts/pops.

    Pre-fix this raised ``RuntimeError: OrderedDict mutated during iteration``
    within a few hundred iterations (even a get_output hit mutates the dict
    via move_to_end).
    """
    item = CachedImage("p", np.zeros((8, 8, 3), np.float32), None, None)
    errors = []
    stop = threading.Event()

    def storer():
        try:
            for i in range(6000):
                key = ("pipe", i % (OUTPUT_SLOT_LIMIT + 2))
                item.store_output(key, np.full((16, 16, 3), i % 251, np.uint8),
                                  float(i), (16, 16))
                item.get_output(("pipe", (i + 1) % (OUTPUT_SLOT_LIMIT + 2)))
        except Exception as e:  # noqa: BLE001 - the test asserts on this
            errors.append(e)
        finally:
            stop.set()

    def accountant():
        # Stands in for manager.notify_entry_updated / drop_stage, which call
        # update_size with the manager lock held (a lock the workers' direct
        # entry access never takes).
        try:
            while not stop.is_set():
                item.update_size()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    _run_threads([threading.Thread(target=storer),
                  threading.Thread(target=accountant)])

    assert errors == []


# =====================================================================
# M2: get_output primary-slot TOCTOU vs concurrent drop_stage('output')
# =====================================================================


def test_get_output_hit_never_returns_none_image(fast_thread_switching):
    """A hit must never be the torn tuple ``(None, ev, size)``.

    Pre-fix ``get_output`` re-read ``output_uint8`` after the null check, so
    an eviction between the check and the tuple construction produced a hit
    whose image is None -> AttributeError in on_process_result on the GUI
    thread.
    """
    item = CachedImage("p", np.zeros((8, 8, 3), np.float32), None, None)
    key = ("pipe", "fit")
    torn = []
    errors = []
    stop = threading.Event()

    def churner():
        try:
            for _ in range(12000):
                item.store_output(key, np.full((4, 4, 3), 7, np.uint8),
                                  1.25, (8, 8))
                item.drop_stage("output")
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                out = item.get_output(key)
                if out is not None and out[0] is None:
                    torn.append(out)
                    return
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    _run_threads([threading.Thread(target=churner),
                  threading.Thread(target=reader)])

    assert errors == []
    assert torn == []


class _EvictionBetweenReads(CachedImage):
    """CachedImage whose primary image vanishes right after it is read.

    Models a concurrent ``drop_stage('output')`` (GUI-thread cap change)
    winning the race immediately after any single read of ``output_uint8``:
    with unsynchronized readers, any read may be the last one to observe the
    value, so a correct ``get_output`` must read it exactly once and return
    that local reference.
    """

    _evict_after_read = False

    @property
    def output_uint8(self):
        val = self._output_uint8_backing
        if self._evict_after_read and val is not None:
            self._output_uint8_backing = None
        return val

    @output_uint8.setter
    def output_uint8(self, value):
        self._output_uint8_backing = value


def test_get_output_reads_primary_image_once():
    """Deterministic M2 repro: no check-then-re-read of the primary slot.

    Pre-fix ``get_output`` re-read ``output_uint8`` when building the result
    tuple after the null check, so the simulated eviction made it return the
    torn ``(None, ev, size)`` hit; post-fix it snapshots the reference under
    the entry lock and returns it.
    """
    item = _EvictionBetweenReads("p", np.zeros((8, 8, 3), np.float32), None, None)
    key = ("pipe", "fit")
    item.store_output(key, np.full((4, 4, 3), 7, np.uint8), 1.25, (8, 8))

    item._evict_after_read = True
    out = item.get_output(key)

    assert out is not None
    img, ev, size = out
    assert img is not None, "torn hit: image evicted between check and return"
    assert img[0, 0, 0] == 7
    assert ev == 1.25
    assert size == (8, 8)


# =====================================================================
# M1c/m6: torn pairing of primary-slot fields across two worker threads
# =====================================================================


def test_concurrent_stores_never_serve_mismatched_image_ev_pairs(
    fast_thread_switching,
):
    """A hit for key K must return the image/EV that were stored under K.

    Pre-fix ``store_output`` wrote the five primary fields one by one without
    a lock, so two workers (main + baseline processors render the same path
    concurrently) could interleave and pair one view's image with another
    view's EV / key.
    """
    item = CachedImage("p", np.zeros((8, 8, 3), np.float32), None, None)
    torn = []
    errors = []

    def worker(tag):
        try:
            for i in range(4000):
                value = (i % 3) * 10 + tag
                key = ("pipe", value)
                item.store_output(
                    key, np.full((4, 4, 3), value, np.uint8),
                    float(value), (value, value),
                )
                out = item.get_output(key)
                if out is None:
                    continue  # evicted by the other worker's stores: fine
                img, ev, size = out
                if img is None or img[0, 0, 0] != value or ev != float(value) \
                        or size != (value, value):
                    torn.append((key, None if img is None else int(img[0, 0, 0]),
                                 ev, size))
                    return
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    _run_threads([threading.Thread(target=worker, args=(1,)),
                  threading.Thread(target=worker, args=(2,))])

    assert errors == []
    assert torn == []


# =====================================================================
# M1b: eviction accounting must survive unaccounted in-place growth
# =====================================================================


def test_eviction_accounting_survives_unaccounted_growth(huge_available_ram):
    """current_memory_mb must equal the sum of entry sizes after eviction.

    A worker grows an entry in place (store_output / corrected write-back)
    and only afterwards calls notify_entry_updated; a GUI-side set_limit_mb
    eviction can land in between. drop_stage's re-account then discovers the
    growth: pre-fix the manager skipped non-positive deltas (`if freed > 0`),
    so the running counter drifted from the entries permanently.
    """
    manager = ImageCacheManager(limit_mb=1000)
    item = CachedImage("p", _mb_array(1), None, None)
    manager.put("p", item)
    assert manager.current_memory_mb == pytest.approx(1.0, rel=0.01)

    # Unaccounted growth: notify_entry_updated has NOT run yet.
    item.store_output(("k",), _mb_array(0.5, dtype=np.uint8), 0.0, (8, 8))
    item.corrected_data = _mb_array(8)

    # GUI-side cap change fires the field-level eviction on the stale sizes.
    manager.set_limit_mb(0.25)

    assert manager.current_memory_mb == pytest.approx(
        sum(it.size_mb for it in manager.cache.values()), abs=1e-6
    )


def test_manager_hammer_accounting_matches_contents(
    huge_available_ram, fast_thread_switching, quiet_cache_logs
):
    """Full-stack hammer: two workers store/notify/get while the GUI thread
    flips the cache cap (lock-held drop_stage/update_size). No exceptions,
    and once quiescent the running counter matches the actual contents.
    """
    manager = ImageCacheManager(limit_mb=10000)
    item = CachedImage("p", np.zeros((32, 32, 3), np.float32), None, None)
    manager.put("p", item)
    errors = []
    done = threading.Event()

    def worker(tag):
        try:
            for i in range(1500):
                key = ("pipe", tag, i % 3)
                item.store_output(key, np.full((8, 8, 3), tag, np.uint8),
                                  float(tag), (8, 8))
                manager.notify_entry_updated("p")
                item.get_output(key)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def gui():
        try:
            while not done.is_set():
                manager.set_limit_mb(0.0005)  # force field-level eviction
                manager.set_limit_mb(10000)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    workers = [threading.Thread(target=worker, args=(t,)) for t in (1, 2)]
    gui_thread = threading.Thread(target=gui)
    gui_thread.start()
    _run_threads(workers)
    done.set()
    gui_thread.join()

    assert errors == []

    # Quiesce and check the invariant the eviction logic depends on.
    manager.notify_entry_updated("p")
    assert manager.current_memory_mb == pytest.approx(
        sum(it.size_mb for it in manager.cache.values()), abs=1e-6
    )
