"""T7.6: host memory governance for the decoded-image cache.

Covers the absolute quota cap, the eviction threshold (at quota, not 80%),
field-level eviction order (corrected -> output -> denoise -> whole entry),
alias-aware size accounting, and in-place update re-accounting.
"""
import numpy as np
import pytest

from raw_alchemy.pipeline import cache_manager as cache_manager_module
from raw_alchemy.pipeline.cache_manager import CachedImage, ImageCacheManager


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


def _mb_array(mb: float, dtype=np.float32) -> np.ndarray:
    """An ndarray whose logical size is `mb` MB without physical allocation."""
    count = int(mb * 1024 * 1024 // np.dtype(dtype).itemsize)
    return np.broadcast_to(np.zeros(1, dtype=dtype), (count,))


def test_mb_array_reports_logical_size_without_allocation():
    arr = _mb_array(512)
    assert arr.nbytes == 512 * 1024 * 1024


def test_quota_is_capped_by_absolute_limit(huge_available_ram):
    manager = ImageCacheManager(memory_fraction=0.5, limit_mb=6144)
    assert manager._get_quota_mb() == 6144


def test_quota_relative_term_still_applies_when_smaller(monkeypatch):
    monkeypatch.setattr(
        cache_manager_module.psutil,
        "virtual_memory",
        lambda: _FakeVirtualMemory(available=4096 * 1024 * 1024),
    )
    manager = ImageCacheManager(memory_fraction=0.5, limit_mb=6144)
    assert manager._get_quota_mb() == pytest.approx(2048)


def test_cached_image_alias_not_double_counted():
    linear = _mb_array(100)
    item = CachedImage("p", linear, None, None, corrected_data=linear)
    assert item.size_mb == pytest.approx(100, rel=0.01)
    assert item.field_sizes_mb["corrected"] == 0.0

    item.corrected_data = _mb_array(100)
    item.update_size()
    assert item.size_mb == pytest.approx(200, rel=0.01)
    assert item.field_sizes_mb["corrected"] == pytest.approx(100, rel=0.01)


def test_no_eviction_below_quota(huge_available_ram):
    # The old manager started evicting at 80% of quota; T7.6 keeps everything
    # until the quota itself is exceeded.
    manager = ImageCacheManager(limit_mb=1000)
    for i in range(3):
        manager.put(f"p{i}", CachedImage(f"p{i}", _mb_array(300), None, None))

    assert len(manager.cache) == 3
    assert manager.current_memory_mb == pytest.approx(900, rel=0.01)


def test_gb_entries_capped_and_corrected_evicted_before_linear(huge_available_ram):
    limit = 6144
    manager = ImageCacheManager(limit_mb=limit)

    for i in range(20):
        item = CachedImage(
            f"img{i}",
            _mb_array(512),
            None,
            (True, None, False),
            corrected_data=_mb_array(512),
        )
        item.output_uint8 = _mb_array(24, dtype=np.uint8)
        item.output_key = ("k",)
        item.update_size()
        manager.put(f"img{i}", item)
        assert manager.current_memory_mb <= limit

    # Total usage bounded by the absolute cap, accounting consistent.
    assert manager.current_memory_mb <= limit
    assert manager.current_memory_mb == pytest.approx(
        sum(it.size_mb for it in manager.cache.values())
    )

    # Whole entries were evicted, so every remaining entry must already have
    # had its corrected field dropped first (corrected goes before linear).
    assert 1 <= len(manager.cache) < 20
    for it in manager.cache.values():
        assert it.linear_data is not None
        assert it.corrected_data is None
        assert it.lens_key is None

    # LRU: the most recently added entry survives.
    assert "img19" in manager.cache


def test_field_eviction_order_corrected_then_output_then_denoise_then_entry(huge_available_ram):
    manager = ImageCacheManager(limit_mb=1000)

    item = CachedImage(
        "p", _mb_array(400), None, (True, None, False), corrected_data=_mb_array(400)
    )
    item.output_uint8 = _mb_array(50, dtype=np.uint8)
    item.output_key = ("k",)
    item.denoise_full = _mb_array(100)
    item.denoise_key = ("p", "denoise")
    item.update_size()
    manager.put("p", item)  # 950 <= 1000: nothing evicted
    assert item.corrected_data is not None

    manager.put("q", CachedImage("q", _mb_array(200), None, None))  # 1150
    # Stage 1: only the corrected field of the LRU entry is dropped.
    assert item.corrected_data is None
    assert item.lens_key is None
    assert item.output_uint8 is not None
    assert item.denoise_full is not None
    assert manager.current_memory_mb <= 1000

    manager.put("r", CachedImage("r", _mb_array(300), None, None))  # 1050
    # Stage 2: output goes next, denoise still kept.
    assert item.output_uint8 is None
    assert item.output_key is None
    assert item.denoise_full is not None
    assert manager.current_memory_mb <= 1000

    manager.put("s", CachedImage("s", _mb_array(150), None, None))  # 1150
    # Stage 3 frees the denoise buffers; still over quota, so the whole LRU
    # entry (linear last) is finally evicted.
    assert "p" not in manager.cache
    assert set(manager.cache) == {"q", "r", "s"}
    assert manager.current_memory_mb <= 1000
    assert manager.current_memory_mb == pytest.approx(650, rel=0.01)


def test_notify_entry_updated_reaccounts_and_evicts_lru_first(huge_available_ram):
    manager = ImageCacheManager(limit_mb=1000)
    a = CachedImage("a", _mb_array(300), None, None)
    b = CachedImage("b", _mb_array(300), None, None)
    manager.put("a", a)
    manager.put("b", b)

    # Simulate a lens-correction write-back growing entry "a" in place.
    a.corrected_data = _mb_array(300)
    a.lens_key = (True, None, False)
    manager.notify_entry_updated("a")
    assert manager.current_memory_mb == pytest.approx(900, rel=0.01)
    assert a.corrected_data is not None

    # Growing "b" pushes the cache over quota. notify_entry_updated marks the
    # updated entry as most recently used, so "a" is now the LRU entry and
    # its corrected field is dropped first while "b" keeps the fresh one.
    b.corrected_data = _mb_array(300)
    b.lens_key = (True, None, False)
    manager.notify_entry_updated("b")
    assert manager.current_memory_mb <= 1000
    assert a.corrected_data is None      # LRU entry degraded first
    assert b.corrected_data is not None  # freshly updated entry kept
    assert manager.current_memory_mb == pytest.approx(900, rel=0.01)


def test_set_limit_mb_reapplies_eviction(huge_available_ram):
    manager = ImageCacheManager(limit_mb=2000)
    for i in range(3):
        item = CachedImage(
            f"p{i}", _mb_array(300), None, (True, None, False),
            corrected_data=_mb_array(300),
        )
        manager.put(f"p{i}", item)
    assert manager.current_memory_mb == pytest.approx(1800, rel=0.01)

    manager.set_limit_mb(1000)
    assert manager.current_memory_mb <= 1000


def test_stats_reports_per_field_totals(huge_available_ram):
    manager = ImageCacheManager(limit_mb=10000)
    item = CachedImage(
        "p", _mb_array(400), None, (True, None, False),
        corrected_data=_mb_array(300), proxy_linear=_mb_array(30),
    )
    item.output_uint8 = _mb_array(20, dtype=np.uint8)
    item.update_size()
    manager.put("p", item)

    stats = manager.stats()
    assert stats["items"] == 1
    assert stats["linear"] == pytest.approx(400, rel=0.01)
    assert stats["corrected"] == pytest.approx(300, rel=0.01)
    assert stats["proxy"] == pytest.approx(30, rel=0.01)
    assert stats["output"] == pytest.approx(20, rel=0.01)
    assert stats["total"] == pytest.approx(750, rel=0.01)
    assert stats["total"] == pytest.approx(manager.current_memory_mb)
def test_distortion_map_cache_is_byte_bounded(monkeypatch):
    from raw_alchemy import config
    from raw_alchemy import lensfun_wrapper as lensfun

    monkeypatch.setattr(config, "DISTORTION_MAP_CACHE_LIMIT_MB", 1)
    lensfun.clear_distortion_map_cache()
    # ~625 KiB per entry: only the most recent one fits under a 1 MiB cap.
    entry_a = (
        np.zeros((160, 160, 3, 2), np.float32),
        np.zeros((160, 160), bool),
    )
    entry_b = (
        np.ones((160, 160, 3, 2), np.float32),
        np.zeros((160, 160), bool),
    )
    assert lensfun._put_distortion_map("a", entry_a) is True
    assert lensfun._put_distortion_map("b", entry_b) is True
    assert lensfun._get_distortion_map("a") is None
    assert lensfun._get_distortion_map("b") is entry_b
    assert lensfun._distortion_map_cache_bytes <= 1024 * 1024
    lensfun.clear_distortion_map_cache()


def test_lens_database_cache_is_entry_bounded(monkeypatch):
    from raw_alchemy import config
    from raw_alchemy import lensfun_wrapper as lensfun

    created = []

    class FakeDatabase:
        def __init__(self, custom_db_path=None):
            self.path = custom_db_path
            created.append(custom_db_path)

    monkeypatch.setattr(config, "LENSFUN_DB_CACHE_ENTRIES", 2)
    monkeypatch.setattr(lensfun, "LensfunDatabase", FakeDatabase)
    lensfun.clear_lensfun_database_cache()
    try:
        first = lensfun._get_or_create_database("db-a.xml")
        lensfun._get_or_create_database("db-b.xml")
        lensfun._get_or_create_database("db-c.xml")

        assert len(lensfun._global_db_cache) == 2
        assert lensfun._database_cache_key("db-a.xml") not in lensfun._global_db_cache
        assert lensfun._get_or_create_database("db-a.xml") is not first
        assert created == ["db-a.xml", "db-b.xml", "db-c.xml", "db-a.xml"]
    finally:
        lensfun.clear_lensfun_database_cache()
