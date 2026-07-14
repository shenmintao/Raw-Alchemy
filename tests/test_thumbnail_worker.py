import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np


def _ensure_qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_jpeg_bytes(width=1200, height=800, patch=200):
    """Landscape JPEG: blue background, red patch in the top-left corner."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QImage, QPainter

    img = QImage(width, height, QImage.Format_RGB888)
    img.fill(QColor(0, 0, 255))
    painter = QPainter(img)
    painter.fillRect(0, 0, patch, patch, QColor(255, 0, 0))
    painter.end()

    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPG", 95)
    buf.close()
    return bytes(data)


class _FakeThumbFormat:
    JPEG = "jpeg"
    BITMAP = "bitmap"


class _FakeColorSpace:
    sRGB = 1


class _FakeThumb:
    def __init__(self, fmt, data):
        self.format = fmt
        self.data = data


class _FakeSizes:
    def __init__(self, flip):
        self.flip = flip


class _FakeRaw:
    def __init__(self, thumb=None, flip=0, postprocess_result=None, calls=None):
        self._thumb = thumb
        self._flip = flip
        self._postprocess_result = postprocess_result
        self._calls = calls if calls is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @property
    def sizes(self):
        return _FakeSizes(self._flip)

    def extract_thumb(self):
        self._calls.append("extract_thumb")
        if self._thumb is None:
            raise RuntimeError("no thumbnail")
        return self._thumb

    def postprocess(self, **kwargs):
        self._calls.append("postprocess")
        if self._postprocess_result is None:
            raise RuntimeError("no postprocess")
        return self._postprocess_result


def _install_fake_rawpy(monkeypatch, raw_factory, calls=None):
    calls = calls if calls is not None else []

    def imread(path):
        calls.append(("imread", path))
        return raw_factory()

    fake = types.SimpleNamespace(
        imread=imread,
        ThumbFormat=_FakeThumbFormat,
        ColorSpace=_FakeColorSpace,
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    return calls


def test_embedded_jpeg_decoded_at_thumb_size_and_rotated_after_scaling(monkeypatch):
    """Portrait orientation (flip=6, 90CW): the JPEG is decoded scaled and the
    rotation is applied after scaling; final height is THUMB_HEIGHT and the
    corner patch ends up in the rotated position."""
    _ensure_qapp(monkeypatch)
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    jpeg = _make_jpeg_bytes(1200, 800, patch=300)
    thumb = _FakeThumb(_FakeThumbFormat.JPEG, jpeg)
    _install_fake_rawpy(monkeypatch, lambda: _FakeRaw(thumb=thumb, flip=6))

    image = ThumbnailWorker.extract_thumbnail("/fake/portrait.raf")
    assert image is not None and not image.isNull()

    # 1200x800 scaled so that post-rotation height == 300 -> 300x200 decode,
    # then rotated 90CW -> 200 wide x 300 high.
    assert image.height() == ThumbnailWorker.THUMB_HEIGHT
    assert image.width() == 200

    # Red patch was at the top-left; after 90CW rotation it is at the top-right.
    top_right = image.pixelColor(image.width() - 5, 5)
    assert top_right.red() > 150 and top_right.blue() < 100
    top_left = image.pixelColor(5, 5)
    assert top_left.blue() > 150 and top_left.red() < 100


def test_embedded_jpeg_flip5_rotates_ccw(monkeypatch):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    jpeg = _make_jpeg_bytes(1200, 800, patch=300)
    thumb = _FakeThumb(_FakeThumbFormat.JPEG, jpeg)
    _install_fake_rawpy(monkeypatch, lambda: _FakeRaw(thumb=thumb, flip=5))

    image = ThumbnailWorker.extract_thumbnail("/fake/portrait.raf")
    assert image is not None and not image.isNull()
    assert image.height() == ThumbnailWorker.THUMB_HEIGHT
    assert image.width() == 200

    # Red patch was at the top-left; after 90CCW rotation it is bottom-left.
    bottom_left = image.pixelColor(5, image.height() - 5)
    assert bottom_left.red() > 150 and bottom_left.blue() < 100


def test_landscape_no_flip_keeps_orientation_and_height(monkeypatch):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    jpeg = _make_jpeg_bytes(1200, 800, patch=300)
    thumb = _FakeThumb(_FakeThumbFormat.JPEG, jpeg)
    _install_fake_rawpy(monkeypatch, lambda: _FakeRaw(thumb=thumb, flip=0))

    image = ThumbnailWorker.extract_thumbnail("/fake/landscape.raf")
    assert image is not None and not image.isNull()
    assert image.height() == ThumbnailWorker.THUMB_HEIGHT
    assert image.width() == 450
    top_left = image.pixelColor(5, 5)
    assert top_left.red() > 150 and top_left.blue() < 100


def test_fallback_postprocess_resizes_then_srgb_encodes(monkeypatch):
    """No embedded thumb -> half_size fallback. Output must be 300px high and
    the sRGB encode must still be correct after the pre-resize refactor."""
    _ensure_qapp(monkeypatch)
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    linear = 0.2
    rgb16 = np.full((1600, 2400, 3), round(linear * 65535), dtype=np.uint16)
    _install_fake_rawpy(
        monkeypatch,
        lambda: _FakeRaw(thumb=None, flip=0, postprocess_result=rgb16),
    )

    image = ThumbnailWorker.extract_thumbnail("/fake/nothumb.raf")
    assert image is not None and not image.isNull()
    assert image.height() == ThumbnailWorker.THUMB_HEIGHT
    assert image.width() == 450

    expected = (1.055 * linear ** (1.0 / 2.4) - 0.055) * 255  # ~123.5
    center = image.pixelColor(image.width() // 2, image.height() // 2)
    for channel in (center.red(), center.green(), center.blue()):
        assert abs(channel - expected) <= 4


def test_stop_check_skips_decode_entirely(monkeypatch):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    calls = _install_fake_rawpy(monkeypatch, lambda: _FakeRaw())

    result = ThumbnailWorker.extract_thumbnail(
        "/fake/file.raf", stop_check=lambda: True
    )
    assert result is None
    assert calls == []  # rawpy was never touched


def test_stop_check_skips_fallback_after_failed_embedded(monkeypatch):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    calls = []
    flags = {"stopped": False}

    def raw_factory():
        # After the embedded attempt fails, flip the stop flag: the expensive
        # postprocess fallback must then be skipped.
        flags["stopped"] = True
        return _FakeRaw(thumb=None, calls=calls)

    _install_fake_rawpy(monkeypatch, raw_factory, calls)

    result = ThumbnailWorker.extract_thumbnail(
        "/fake/file.raf", stop_check=lambda: flags["stopped"]
    )
    assert result is None
    assert "postprocess" not in calls


def test_default_worker_count_leaves_cores_for_ui(monkeypatch):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.config import THUMBNAIL_MAX_WORKERS
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    worker = ThumbnailWorker("/nonexistent")
    assert worker.max_workers == max(1, min(THUMBNAIL_MAX_WORKERS, os.cpu_count() or 1))
    # explicit value still honored
    assert ThumbnailWorker("/nonexistent", max_workers=3).max_workers == 3


def test_priority_paths_are_extracted_first(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from PySide6.QtGui import QImage
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    files = []
    for i in range(8):
        p = tmp_path / f"img{i}.raf"
        p.write_bytes(b"")
        files.append(str(p))

    order = []
    lock = threading.Lock()

    def fake_extract(full_path, stop_check=None):
        with lock:
            order.append(full_path)
        return QImage(10, 10, QImage.Format_RGB888)

    monkeypatch.setattr(
        ThumbnailWorker, "extract_thumbnail", staticmethod(fake_extract)
    )

    worker = ThumbnailWorker(str(tmp_path), max_workers=1)

    placeholders = []
    worker.placeholders_ready.connect(lambda paths, img: placeholders.extend(paths))
    ready = []
    worker.thumbnail_ready.connect(lambda path, img: ready.append(path))
    finished = []
    worker.finished_scanning.connect(lambda: finished.append(True))

    worker.set_priority_paths([files[5], files[6]])
    worker.run()  # synchronous: max_workers=1 makes execution order deterministic

    assert placeholders == files  # sorted placeholders for the whole folder
    assert order[:2] == [files[5], files[6]]  # visible items first
    assert sorted(order) == sorted(files)  # everything still extracted
    assert sorted(ready) == sorted(files)
    assert finished == [True]


def test_worker_stop_aborts_quickly_mid_scan(monkeypatch, tmp_path):
    _ensure_qapp(monkeypatch)
    from PySide6.QtGui import QImage
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    for i in range(50):
        (tmp_path / f"img{i:02d}.raf").write_bytes(b"")

    worker = ThumbnailWorker(str(tmp_path), max_workers=1)
    count = {"n": 0}

    def fake_extract(full_path, stop_check=None):
        count["n"] += 1
        if count["n"] == 3:
            worker.stop()  # simulate a folder switch mid-scan
        if stop_check is not None and stop_check():
            return None
        return QImage(10, 10, QImage.Format_RGB888)

    monkeypatch.setattr(
        ThumbnailWorker, "extract_thumbnail", staticmethod(fake_extract)
    )

    worker.run()
    # in-flight futures may still run their (cheap) stop-checked bodies,
    # but the remaining ~45 files must not be processed
    assert count["n"] <= 6


class _FakeLoadingLabel:
    def __init__(self):
        self.text = ""
        self.visible = False

    def setText(self, text):
        self.text = text

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False


def _make_scan_harness(monkeypatch):
    _ensure_qapp(monkeypatch)
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QListWidget

    from raw_alchemy.ui.library_controller import LibraryControllerMixin

    class _ScanHarness(QObject, LibraryControllerMixin):
        def __init__(self):
            super().__init__()
            self.thumb_worker = None
            self._retired_thumb_workers = []
            self.loading_label = _FakeLoadingLabel()
            self.gallery_list = QListWidget()
            self.gallery_items_by_path = {}
            self.marked_files = set()
            self.preloaded = []

        def _preload_neighbors(self, index, count=2):
            self.preloaded.append((index, count))

    return _ScanHarness()


def test_folder_switch_does_not_block_and_ignores_stale_signals(monkeypatch, tmp_path):
    app = _ensure_qapp(monkeypatch)
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    harness = _make_scan_harness(monkeypatch)

    release = threading.Event()

    class _BlockingWorker(ThumbnailWorker):
        def run(self):
            release.wait(5.0)

    old_worker = _BlockingWorker(str(tmp_path))
    harness.thumb_worker = old_worker
    old_worker.start()
    time.sleep(0.05)
    assert old_worker.isRunning()

    # New workers must not actually spin up threads in this test.
    started = []
    monkeypatch.setattr(
        ThumbnailWorker, "start", lambda self, *a, **k: started.append(self)
    )

    t0 = time.monotonic()
    harness.start_thumbnail_scan(str(tmp_path))
    elapsed = time.monotonic() - t0

    # The old worker is still running: switching folders did not wait() on it.
    assert old_worker.isRunning()
    assert elapsed < 1.0
    assert old_worker.stopped is True
    assert harness.thumb_worker is not old_worker
    assert old_worker in harness._retired_thumb_workers
    assert started == [harness.thumb_worker]

    # Stale signals from the retired worker must not touch the gallery.
    placeholder = ThumbnailWorker._make_placeholder()
    old_worker.placeholders_ready.emit(["/stale/a.raf"], placeholder)
    app.processEvents()
    assert harness.gallery_list.count() == 0
    assert "/stale/a.raf" not in harness.gallery_items_by_path

    old_worker.finished_scanning.emit()
    app.processEvents()
    assert harness.loading_label.visible is True  # not hidden by stale worker

    # Signals from the current worker are accepted (batched placeholders).
    new_worker = harness.thumb_worker
    new_worker.placeholders_ready.emit(["/new/a.raf", "/new/b.raf"], placeholder)
    app.processEvents()
    assert harness.gallery_list.count() == 2
    from PySide6.QtCore import Qt

    assert (
        harness.gallery_list.item(0).data(Qt.ItemDataRole.UserRole) == "/new/a.raf"
    )

    from PySide6.QtGui import QImage

    thumb = QImage(20, 30, QImage.Format_RGB888)
    new_worker.thumbnail_ready.emit("/new/b.raf", thumb)
    app.processEvents()
    assert not harness.gallery_items_by_path["/new/b.raf"].icon().isNull()

    new_worker.finished_scanning.emit()
    app.processEvents()
    assert harness.loading_label.visible is False
    assert harness.preloaded == [(0, 2)]

    # Once the retired worker finishes it is reaped from the retired list.
    release.set()
    assert old_worker.wait(2000)
    app.processEvents()
    assert old_worker not in harness._retired_thumb_workers

    harness.thumb_worker = None  # patched start() never ran the new worker


def test_update_thumbnail_priority_forwards_visible_paths(monkeypatch):
    _ensure_qapp(monkeypatch)
    harness = _make_scan_harness(monkeypatch)

    class _RecordingWorker:
        def __init__(self):
            self.priorities = []

        def set_priority_paths(self, paths):
            self.priorities.append(list(paths))

    worker = _RecordingWorker()
    harness.thumb_worker = worker
    monkeypatch.setattr(
        harness, "_visible_gallery_paths", lambda: ["/x/b.raf", "/x/c.raf"]
    )

    harness._update_thumbnail_priority()
    assert worker.priorities == [["/x/b.raf", "/x/c.raf"]]

    harness.thumb_worker = None
    harness._update_thumbnail_priority()  # no worker -> no crash
    assert worker.priorities == [["/x/b.raf", "/x/c.raf"]]


# =====================================================================
# m1/m5 — shutdown join must never leave a running QThread behind
# =====================================================================


def test_stop_and_join_outwaits_uninterruptible_decode(monkeypatch, tmp_path):
    """closeEvent used a bare wait(2000): a rawpy half_size postprocess in
    flight cannot observe stop() and can overrun the grace period, after
    which the still-running QThread would be destroyed with the window —
    a Qt qFatal abort on exit. stop_and_join must keep waiting past the
    grace until the thread has actually finished."""
    _ensure_qapp(monkeypatch)
    from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker

    (tmp_path / "big.raf").write_bytes(b"raw")

    decode_started = threading.Event()

    def slow_extract(full_path, stop_check=None):
        # Models the Method-2 fallback: inside the C decode, stop_check
        # cannot run, so the job outlives any bounded grace period.
        decode_started.set()
        time.sleep(1.0)
        return None

    monkeypatch.setattr(
        ThumbnailWorker, "extract_thumbnail", staticmethod(slow_extract)
    )

    worker = ThumbnailWorker(str(tmp_path), max_workers=1)
    worker.start()
    assert decode_started.wait(5.0)

    joined_in_grace = worker.stop_and_join(grace_ms=100)

    # The in-flight decode overran the grace period (the old code path
    # would have returned here with the QThread still running) ...
    assert joined_in_grace is False
    # ... but the thread was joined to completion regardless: destroying
    # a running QThread aborts the whole process with a Qt fatal error.
    assert not worker.isRunning()
    assert worker.isFinished()
