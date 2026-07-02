"""T7.8 — persistent thumbnail cache (mtime keyed, LRU pruned, switchable)."""

import os
import sys
import time
import types


def _ensure_qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_test_image(width=450, height=300, color=(200, 30, 30)):
    from PySide6.QtGui import QColor, QImage

    img = QImage(width, height, QImage.Format_RGB888)
    img.fill(QColor(*color))
    return img


def _make_jpeg_bytes(width=1200, height=800):
    """JPEG bytes of a solid green image, used as a fake embedded thumbnail."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QImage

    img = QImage(width, height, QImage.Format_RGB888)
    img.fill(QColor(20, 180, 20))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPG", 95)
    buf.close()
    return bytes(data)


def _bump_mtime(path, seconds=10):
    st = os.stat(path)
    os.utime(path, (st.st_atime + seconds, st.st_mtime + seconds))


def _entry_file(cache, path):
    return cache._entry_path(cache.key_for(path))


# --- disk layer -------------------------------------------------------------


def test_roundtrip_and_mtime_invalidation(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache

    src = tmp_path / "photo.raf"
    src.write_bytes(b"raw-bytes")
    cache = ThumbnailCache(cache_dir=str(tmp_path / "thumbs"))

    assert cache.get(str(src)) is None  # cold miss
    assert cache.put(str(src), _make_test_image()) is True
    assert os.path.isfile(_entry_file(cache, str(src)))

    hit = cache.get(str(src))
    assert hit is not None and not hit.isNull()
    assert (hit.width(), hit.height()) == (450, 300)
    color = hit.pixelColor(200, 150)
    assert abs(color.red() - 200) <= 6 and color.blue() <= 40  # JPEG-lossy match

    # mtime change invalidates only via the key: the get misses.
    _bump_mtime(str(src))
    assert cache.get(str(src)) is None

    # size change also invalidates
    src.write_bytes(b"raw-bytes-different-length")
    assert cache.get(str(src)) is None

    # missing source file: no crash, plain miss
    assert cache.get(str(tmp_path / "gone.raf")) is None


def test_corrupt_entry_is_dropped_and_overwritten(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache

    src = tmp_path / "photo.raf"
    src.write_bytes(b"raw-bytes")
    cache_dir = str(tmp_path / "thumbs")

    ThumbnailCache(cache_dir=cache_dir).put(str(src), _make_test_image())

    # New instance = new session (empty memory layer); corrupt the disk entry.
    cache = ThumbnailCache(cache_dir=cache_dir)
    entry = _entry_file(cache, str(src))
    with open(entry, "wb") as f:
        f.write(b"not a jpeg at all")

    assert cache.get(str(src)) is None  # no crash
    assert not os.path.exists(entry)  # corrupt entry removed

    # Re-extraction overwrites and the next get hits again.
    assert cache.put(str(src), _make_test_image()) is True
    assert cache.get(str(src)) is not None


def test_memory_layer_serves_same_session_reopen(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache

    src = tmp_path / "photo.raf"
    src.write_bytes(b"raw-bytes")
    cache = ThumbnailCache(cache_dir=str(tmp_path / "thumbs"))
    cache.put(str(src), _make_test_image())

    # Even with the disk entry gone, the session memory layer still hits.
    os.unlink(_entry_file(cache, str(src)))
    hit = cache.get(str(src))
    assert hit is not None and hit.height() == 300

    # ...but a modified source file misses (key includes mtime).
    _bump_mtime(str(src))
    assert cache.get(str(src)) is None


def test_disabled_cache_is_a_noop_and_reenable_works(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache

    src = tmp_path / "photo.raf"
    src.write_bytes(b"raw-bytes")
    cache_dir = tmp_path / "thumbs"
    cache = ThumbnailCache(cache_dir=str(cache_dir), enabled=False)

    assert cache.put(str(src), _make_test_image()) is False
    assert cache.get(str(src)) is None
    assert not cache_dir.exists()  # nothing was written anywhere
    assert cache.prune_async() is None

    cache.set_enabled(True)
    assert cache.put(str(src), _make_test_image()) is True
    assert cache.get(str(src)) is not None

    # Disabling drops the session memory layer too: after deleting the disk
    # entry and re-enabling, behavior is back to a plain miss (= status quo).
    cache.set_enabled(False)
    assert cache.get(str(src)) is None
    os.unlink(_entry_file(cache, str(src)))
    cache.set_enabled(True)
    assert cache.get(str(src)) is None


def test_clear_removes_all_entries(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache

    cache = ThumbnailCache(cache_dir=str(tmp_path / "thumbs"))
    sources = []
    for i in range(4):
        src = tmp_path / f"photo{i}.raf"
        src.write_bytes(b"raw" + bytes([i]))
        cache.put(str(src), _make_test_image())
        sources.append(str(src))

    assert cache.clear() == 4
    assert os.listdir(str(tmp_path / "thumbs")) == []
    for src in sources:  # memory layer cleared as well
        assert cache.get(src) is None


# --- capacity governance ------------------------------------------------------


def test_prune_lru_by_file_count(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache

    cache_dir = tmp_path / "thumbs"
    cache = ThumbnailCache(cache_dir=str(cache_dir))
    entries = []
    now = time.time()
    for i in range(6):
        src = tmp_path / f"photo{i}.raf"
        src.write_bytes(b"raw" + bytes([i]))
        cache.put(str(src), _make_test_image())
        entry = _entry_file(cache, str(src))
        # Stagger LRU stamps: photo0 oldest ... photo5 newest.
        os.utime(entry, (now - 600 + i * 60, now - 600 + i * 60))
        entries.append(entry)

    assert cache.prune(max_bytes=10**9, max_files=3) == 3
    assert [os.path.exists(e) for e in entries] == [
        False, False, False, True, True, True
    ]
    # Under budget now: a second prune removes nothing.
    assert cache.prune(max_bytes=10**9, max_files=3) == 0


def test_prune_lru_by_total_bytes_and_read_hit_refreshes_stamp(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from raw_alchemy import thumb_cache
    from raw_alchemy.thumb_cache import ThumbnailCache

    cache_dir = tmp_path / "thumbs"
    cache_dir.mkdir()
    now = time.time()
    paths = []
    for i in range(5):
        p = cache_dir / f"{i:040x}.jpg"
        p.write_bytes(b"x" * 10_000)
        os.utime(str(p), (now - 500 + i * 60, now - 500 + i * 60))
        paths.append(str(p))

    cache = ThumbnailCache(cache_dir=str(cache_dir))
    # 50KB total, budget 25KB -> the 3 oldest go, the 2 newest stay.
    assert cache.prune(max_bytes=25_000, max_files=10**6) == 3
    assert [os.path.exists(p) for p in paths] == [False, False, False, True, True]

    # A read hit bumps the entry's timestamps so LRU protects it.
    src = tmp_path / "photo.raf"
    src.write_bytes(b"raw-bytes")
    cache.put(str(src), _make_test_image())
    entry = _entry_file(cache, str(src))
    os.utime(entry, (now - 10_000, now - 10_000))
    fresh_session = ThumbnailCache(cache_dir=str(cache_dir))
    assert fresh_session.get(str(src)) is not None
    st = os.stat(entry)
    assert max(st.st_atime, st.st_mtime) > now - 60

    # Default budgets come from the constructor.
    assert thumb_cache.DEFAULT_MAX_BYTES == 500 * 1024 * 1024
    assert thumb_cache.DEFAULT_MAX_FILES == 20000
    assert cache.max_bytes == thumb_cache.DEFAULT_MAX_BYTES
    assert cache.max_files == thumb_cache.DEFAULT_MAX_FILES


def test_prune_async_runs_in_background(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache

    cache_dir = tmp_path / "thumbs"
    cache_dir.mkdir()
    now = time.time()
    for i in range(4):
        p = cache_dir / f"{i:040x}.jpg"
        p.write_bytes(b"x" * 1000)
        os.utime(str(p), (now - 500 + i * 60, now - 500 + i * 60))

    cache = ThumbnailCache(cache_dir=str(cache_dir), max_bytes=10**9, max_files=2)
    thread = cache.prune_async()
    assert thread is not None
    thread.join(5.0)
    assert len(os.listdir(str(cache_dir))) == 2


# --- extract_thumbnail integration -------------------------------------------


class _FakeThumbFormat:
    JPEG = "jpeg"
    BITMAP = "bitmap"


class _FakeSizes:
    def __init__(self, flip):
        self.flip = flip


class _FakeRaw:
    def __init__(self, thumb_data):
        self._thumb_data = thumb_data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @property
    def sizes(self):
        return _FakeSizes(0)

    def extract_thumb(self):
        return types.SimpleNamespace(format=_FakeThumbFormat.JPEG, data=self._thumb_data)


def _install_fake_rawpy(monkeypatch, jpeg_bytes, calls):
    def imread(path):
        calls.append(path)
        return _FakeRaw(jpeg_bytes)

    fake = types.SimpleNamespace(
        imread=imread,
        ThumbFormat=_FakeThumbFormat,
        ColorSpace=types.SimpleNamespace(sRGB=1),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    return calls


def test_extract_thumbnail_uses_cache_and_reextracts_on_mtime_change(
    monkeypatch, tmp_path
):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    src = tmp_path / "photo.raf"
    src.write_bytes(b"raw-bytes")
    cache = ThumbnailCache(cache_dir=str(tmp_path / "thumbs"))
    calls = _install_fake_rawpy(monkeypatch, _make_jpeg_bytes(), [])

    # Miss -> decode via rawpy -> write-back.
    first = ThumbnailWorker.extract_thumbnail(str(src), cache=cache)
    assert first is not None and first.height() == ThumbnailWorker.THUMB_HEIGHT
    assert len(calls) == 1
    assert os.path.isfile(_entry_file(cache, str(src)))

    # Hit -> rawpy is not touched at all.
    second = ThumbnailWorker.extract_thumbnail(str(src), cache=cache)
    assert second is not None
    assert (second.width(), second.height()) == (first.width(), first.height())
    assert len(calls) == 1

    # mtime change -> only this file re-extracts.
    _bump_mtime(str(src))
    third = ThumbnailWorker.extract_thumbnail(str(src), cache=cache)
    assert third is not None
    assert len(calls) == 2

    # Without a cache the legacy behavior is unchanged (decodes every time).
    ThumbnailWorker.extract_thumbnail(str(src))
    ThumbnailWorker.extract_thumbnail(str(src))
    assert len(calls) == 4


def test_worker_second_scan_serves_thumbnails_from_cache(monkeypatch, tmp_path):
    """Acceptance: reopening the same folder emits every thumbnail from the
    cache without a single RAW decode, and the scan triggers pruning."""
    _ensure_qapp(monkeypatch)
    from raw_alchemy.thumb_cache import ThumbnailCache
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    folder = tmp_path / "shoot"
    folder.mkdir()
    files = []
    for i in range(6):
        p = folder / f"img{i}.raf"
        p.write_bytes(b"raw" + bytes([i]))
        files.append(str(p))

    cache = ThumbnailCache(cache_dir=str(tmp_path / "thumbs"))
    calls = _install_fake_rawpy(monkeypatch, _make_jpeg_bytes(), [])

    prunes = []
    monkeypatch.setattr(cache, "prune_async", lambda: prunes.append(True))

    worker = ThumbnailWorker(str(folder), max_workers=1, disk_cache=cache)
    ready = []
    worker.thumbnail_ready.connect(lambda path, img: ready.append(path))
    worker.run()
    assert sorted(ready) == sorted(files)
    assert len(calls) == len(files)  # first scan decodes everything
    assert prunes == [True]

    # Second scan, fresh session: rawpy must never be called again.
    fresh_cache = ThumbnailCache(cache_dir=str(tmp_path / "thumbs"))
    monkeypatch.setattr(fresh_cache, "prune_async", lambda: prunes.append(True))

    def exploding_imread(path):
        raise AssertionError(f"RAW decode attempted for {path}")

    monkeypatch.setitem(
        sys.modules,
        "rawpy",
        types.SimpleNamespace(
            imread=exploding_imread,
            ThumbFormat=_FakeThumbFormat,
            ColorSpace=types.SimpleNamespace(sRGB=1),
        ),
    )

    worker2 = ThumbnailWorker(str(folder), max_workers=1, disk_cache=fresh_cache)
    ready2 = []
    worker2.thumbnail_ready.connect(lambda path, img: ready2.append(path))
    worker2.run()
    assert sorted(ready2) == sorted(files)
    assert prunes == [True, True]


def test_start_thumbnail_scan_passes_shared_cache_to_worker(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QListWidget

    from raw_alchemy.thumb_cache import ThumbnailCache
    from raw_alchemy.ui.library_controller import LibraryControllerMixin
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    class _FakeLabel:
        def setText(self, text):
            pass

        def show(self):
            pass

        def hide(self):
            pass

    class _Harness(QObject, LibraryControllerMixin):
        def __init__(self):
            super().__init__()
            self.thumb_worker = None
            self._retired_thumb_workers = []
            self.loading_label = _FakeLabel()
            self.gallery_list = QListWidget()
            self.gallery_items_by_path = {}
            self.marked_files = set()

    monkeypatch.setattr(ThumbnailWorker, "start", lambda self, *a, **k: None)

    harness = _Harness()
    shared = ThumbnailCache(cache_dir=str(tmp_path / "thumbs"))
    harness.thumb_cache = shared
    harness.start_thumbnail_scan(str(tmp_path))
    assert harness.thumb_worker.disk_cache is shared

    # Without a MainWindow-provided cache the worker gets none (legacy path).
    plain = _Harness()
    plain.start_thumbnail_scan(str(tmp_path))
    assert plain.thumb_worker.disk_cache is None
