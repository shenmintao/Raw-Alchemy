from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize, Signal, QThread
from PySide6.QtGui import QImage, QImageReader, QImageIOHandler, QColor
import os
import threading
import numpy as np
import concurrent.futures
from loguru import logger

from raw_alchemy.config import SUPPORTED_RAW_EXTENSIONS, THUMBNAIL_MAX_WORKERS


def _default_max_workers():
    """Bound decode concurrency by memory as well as CPU count.

    On a 32-core machine the previous ``cpu_count - 2`` policy launched 30
    rawpy jobs; fallback decodes can each hold a large native buffer and swamp
    both RAM and the interactive pipeline. Embedded-JPEG extraction is fast
    enough that four low-priority workers keep the viewport fed.
    """
    return max(1, min(int(THUMBNAIL_MAX_WORKERS), os.cpu_count() or 1))


def _lower_thread_priority():
    """Best-effort: drop the calling thread's scheduling priority.

    On Linux each thread is its own scheduling entity, so setpriority(0)
    only affects the calling (pool) thread. On Windows we use
    SetThreadPriority. Failures are silently ignored.
    """
    try:
        if os.name == "nt":
            import ctypes

            THREAD_PRIORITY_BELOW_NORMAL = -1
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority(
                kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL
            )
        else:
            os.setpriority(os.PRIO_PROCESS, 0, 10)
    except Exception:
        pass


def _apply_flip_rotation(image: QImage, flip: int) -> QImage:
    """Apply rawpy flip code to a QImage.

    flip values: 0=none, 3=180°, 5=90°CCW, 6=90°CW
    """
    if flip == 0:
        return image
    from PySide6.QtGui import QTransform
    if flip == 3:
        return image.transformed(QTransform().rotate(180))
    elif flip == 5:
        return image.transformed(QTransform().rotate(-90))
    elif flip == 6:
        return image.transformed(QTransform().rotate(90))
    return image


def _decode_jpeg_scaled(data, target_height: int):
    """Decode embedded JPEG bytes at ~thumbnail size via QImageReader.

    Returns ``(QImage | None, exif_oriented)``. ``exif_oriented`` is True when
    the embedded JPEG carried its own EXIF orientation tag — setAutoTransform
    has then already rotated the image to its correct display orientation, so
    the caller must NOT re-apply the rawpy flip on top (doing both rotates the
    image twice, the Fuji RAF portrait bug: flip=6 + EXIF Rotate90 = +180°).

    setScaledSize lets libjpeg do DCT-domain scaled decoding: ~8x faster and a
    tiny fraction of the peak memory of a full-resolution decode. We decode to
    ~2x target height by the longest *stored* edge and leave exact sizing to
    the caller, sidestepping any pre/post-transform ambiguity in size().
    """
    byte_array = QByteArray(data)
    buffer = QBuffer(byte_array)
    if not buffer.open(QIODevice.OpenModeFlag.ReadOnly):
        return None, False
    reader = QImageReader(buffer, QByteArray(b"jpg"))
    # QImage.loadFromData applies EXIF orientation by default; keep parity.
    reader.setAutoTransform(True)
    exif_oriented = (
        reader.transformation()
        != QImageIOHandler.Transformation.TransformationNone
    )
    size = reader.size()  # header-only, no pixel decode
    if size.isValid() and size.width() > 0 and size.height() > 0:
        longest = max(size.width(), size.height())
        cap = max(1, target_height * 2)
        if longest > cap:
            scale = cap / longest
            reader.setScaledSize(QSize(
                max(1, round(size.width() * scale)),
                max(1, round(size.height() * scale)),
            ))
    image = reader.read()
    if image.isNull():
        return None, False
    return image, exif_oriented


class ThumbnailWorker(QThread):
    """
    Scan folder and generate thumbnails.
    1. Emit placeholders sorted by filename (instant)
    2. Load actual thumbnails in parallel, emit as ready

    Scheduling: paths passed to set_priority_paths() (the gallery's visible
    viewport) are extracted first; everything else follows in sorted order.
    Only a small number of futures is kept in flight, so a viewport change
    takes effect almost immediately and off-screen work is deferred.

    disk_cache (T7.8): optional ThumbnailCache. Hits skip the RAW decode
    entirely; misses are extracted as before and written back on the pool
    thread. After a completed scan the cache prunes itself in the background.
    """
    thumbnail_ready = Signal(str, QImage)
    placeholders_ready = Signal(object, QImage)  # batched sorted placeholders
    progress_update = Signal(int, int)
    finished_scanning = Signal()

    THUMB_HEIGHT = 300

    def __init__(self, folder_path, max_workers=None, disk_cache=None):
        super().__init__()
        self.folder_path = folder_path
        self.stopped = False
        self.max_workers = max_workers if max_workers else _default_max_workers()
        self.disk_cache = disk_cache
        self._priority_lock = threading.Lock()
        self._priority_paths = []

    def set_priority_paths(self, paths):
        """Thread-safe: called from the UI thread when the visible viewport
        changes. Pending extraction jobs for these paths jump the queue."""
        with self._priority_lock:
            self._priority_paths = list(paths)

    @staticmethod
    def _make_placeholder(width=200, height=300) -> QImage:
        """Create a gray placeholder QImage."""
        img = QImage(width, height, QImage.Format_RGB888)
        img.fill(QColor(40, 40, 40))
        return img

    @staticmethod
    def extract_thumbnail(full_path, stop_check=None, cache=None):
        """
        Extract thumbnail — tries the disk cache first (T7.8, ~5ms), then the
        embedded JPEG (decoded directly at thumbnail size, ~50ms), and finally
        falls back to rawpy half_size postprocess.

        stop_check: optional callable; when it returns True the (remaining)
        expensive decode work is skipped and None is returned, so worker
        shutdown never waits behind a full RAW decode.

        cache: optional ThumbnailCache; extracted thumbnails are written back
        so the next visit of this folder skips the decode entirely.
        """
        try:
            ext = os.path.splitext(full_path)[1].lower()
            if ext not in SUPPORTED_RAW_EXTENSIONS:
                return None
            if stop_check is not None and stop_check():
                return None

            if cache is not None:
                try:
                    cached = cache.get(full_path)
                except Exception:
                    cached = None
                if cached is not None and not cached.isNull():
                    return cached

            target_height = ThumbnailWorker.THUMB_HEIGHT
            image = None
            flip = 0
            exif_oriented = False

            # Method 1: Extract embedded JPEG thumbnail (fastest, ~50ms)
            try:
                import rawpy
                with rawpy.imread(full_path) as raw:
                    flip = raw.sizes.flip
                    thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    image, exif_oriented = _decode_jpeg_scaled(
                        thumb.data, target_height
                    )
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    h, w = thumb.data.shape[:2]
                    thumb_data = np.ascontiguousarray(thumb.data)
                    image = QImage(
                        thumb_data.data, w, h, 3 * w,
                        QImage.Format_RGB888
                    ).copy()
            except Exception:
                image = None
                exif_oriented = False

            # Orient the embedded thumbnail. When the JPEG carried its own EXIF
            # orientation, setAutoTransform already applied it — re-applying the
            # rawpy flip double-rotates (Fuji RAF portrait: flip=6 + Rotate90 =
            # +180°). Only apply the rawpy flip when the JPEG had no orientation
            # of its own (e.g. Canon CR2 embedded thumbnails are unrotated).
            # Scale *before* rotating so the transform runs on a ~300px image
            # instead of allocating a rotated full-size copy.
            if image is not None and not image.isNull():
                if exif_oriented:
                    if image.height() != target_height and image.height() > 0:
                        image = image.scaledToHeight(target_height)
                else:
                    ref = image.width() if flip in (5, 6) else image.height()
                    if ref != target_height and ref > 0:
                        if flip in (5, 6):
                            image = image.scaledToWidth(target_height)
                        else:
                            image = image.scaledToHeight(target_height)
                    image = _apply_flip_rotation(image, flip)
                if not image.isNull():
                    if cache is not None:
                        try:
                            cache.put(full_path, image)
                        except Exception:
                            pass
                    return image
                image = None

            if stop_check is not None and stop_check():
                return None

            # Method 2: Fallback to rawpy half_size postprocess
            # rawpy.postprocess() applies orientation automatically, no extra rotation needed.
            try:
                import rawpy
                with rawpy.imread(full_path) as raw:
                    rgb16 = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=True,
                        use_auto_wb=False,
                        output_bps=16,
                        output_color=rawpy.ColorSpace.sRGB,
                        half_size=True,
                    )
                if stop_check is not None and stop_check():
                    return None

                if rgb16.ndim == 2:
                    rgb16 = rgb16[:, :, None]
                # Downscale in uint16 first so the float32 conversion and the
                # sRGB encode below run on a ~600px image (a few MB) instead
                # of the full half-size frame (hundreds of MB in temporaries).
                inter_height = target_height * 2
                if rgb16.shape[0] > inter_height:
                    try:
                        import cv2
                        new_w = max(1, round(
                            rgb16.shape[1] * inter_height / rgb16.shape[0]
                        ))
                        rgb16 = cv2.resize(
                            rgb16, (new_w, inter_height),
                            interpolation=cv2.INTER_AREA,
                        )
                        if rgb16.ndim == 2:
                            rgb16 = rgb16[:, :, None]
                    except Exception:
                        pass

                rgb = rgb16.astype(np.float32) / 65535.0
                del rgb16
                if rgb.ndim == 3 and rgb.shape[2] == 1:
                    rgb = np.repeat(rgb, 3, axis=2)

                np.clip(rgb, 0.0, 1.0, out=rgb)
                mask = rgb <= 0.0031308
                rgb[mask] *= 12.92
                rgb[~mask] = 1.055 * np.power(rgb[~mask], 1.0 / 2.4) - 0.055

                thumb_uint8 = (rgb * 255).astype(np.uint8)
                thumb_uint8 = np.ascontiguousarray(thumb_uint8)
                image = QImage(
                    thumb_uint8.data,
                    rgb.shape[1], rgb.shape[0],
                    3 * rgb.shape[1],
                    QImage.Format_RGB888
                ).copy()
            except Exception:
                image = None

            if image is not None and not image.isNull():
                if image.height() != target_height:
                    image = image.scaledToHeight(target_height)
                if cache is not None:
                    try:
                        cache.put(full_path, image)
                    except Exception:
                        pass
                return image
            return None

        except Exception:
            return None

    def run(self):
        # 1. Scan and sort by filename
        files = []
        try:
            with os.scandir(self.folder_path) as entries:
                for entry in entries:
                    if self.stopped:
                        return
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in SUPPORTED_RAW_EXTENSIONS:
                        files.append(entry.path)
        except Exception as e:
            logger.error(f"Failed to scan directory: {e}")
            self.finished_scanning.emit()
            return

        files.sort(key=lambda p: os.path.basename(p).lower())
        total = len(files)

        # 2. Emit placeholders in sorted order. Batch them to avoid thousands of
        # queued Qt events and repeated QListWidget relayouts on large folders.
        placeholder = self._make_placeholder()
        chunk_size = 200
        for i in range(0, total, chunk_size):
            if self.stopped:
                return
            self.placeholders_ready.emit(files[i:i + chunk_size], placeholder)

        # 3. Load actual thumbnails in parallel, emit as ready. Keep only a small
        # number of futures in flight so huge folders do not create huge queues
        # and viewport-priority changes take effect quickly.
        pending = dict.fromkeys(files)  # ordered set, sorted order

        def next_file():
            with self._priority_lock:
                priority = self._priority_paths
            for p in priority:
                if p in pending:
                    del pending[p]
                    return p
            if pending:
                p = next(iter(pending))
                del pending[p]
                return p
            return None

        processed_count = 0

        def stop_check():
            return self.stopped

        # Only pass the cache through when present so injected extractors
        # with the plain (path, stop_check) signature keep working.
        extract_kwargs = {}
        if self.disk_cache is not None:
            extract_kwargs["cache"] = self.disk_cache

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="thumb",
            initializer=_lower_thread_priority,
        ) as executor:
            futures = {}
            max_in_flight = max(1, self.max_workers * 2)

            def submit_next():
                next_path = next_file()
                if next_path is None:
                    return False
                futures[executor.submit(
                    self.extract_thumbnail, next_path, stop_check, **extract_kwargs
                )] = next_path
                return True

            for _ in range(min(max_in_flight, total)):
                submit_next()

            while futures:
                if self.stopped:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return

                done, _pending = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    f_path = futures.pop(future)
                    try:
                        qimg = future.result()
                        if qimg:
                            self.thumbnail_ready.emit(f_path, qimg)
                    except Exception as e:
                        logger.error(f"Worker exception for {f_path}: {e}")

                    processed_count += 1
                    if processed_count % 5 == 0 or processed_count == total:
                        self.progress_update.emit(processed_count, total)

                    if not self.stopped:
                        submit_next()

        # Capacity governance (T7.8): after a completed scan, trim the disk
        # cache in the background if it grew past its budget.
        if self.disk_cache is not None and not self.stopped:
            try:
                self.disk_cache.prune_async()
            except Exception:
                pass

        self.finished_scanning.emit()

    def stop(self):
        self.stopped = True

    def stop_and_join(self, grace_ms=2000):
        """Stop the worker and join it, never leaving a running QThread.

        ``stop()`` is checked before each decode stage, so the thread
        normally exits within the grace period. But an in-flight rawpy
        decode (a Method-2 half_size postprocess on a large RAW can take
        seconds) is not interruptible, so the bounded wait can time out.
        A QThread that is destroyed while still running makes Qt abort
        with a qFatal ("QThread: Destroyed while thread is still
        running"), so on timeout this falls back to a blocking join —
        bounded by the decode already in flight (m1/m5).

        Returns True when the thread finished within the grace period.
        """
        self.stop()
        if not self.isRunning():
            return True
        self.quit()
        if self.wait(int(grace_ms)):
            return True
        logger.warning(
            "[Thumbs] worker still decoding after "
            f"{int(grace_ms)}ms grace; blocking until it finishes"
        )
        self.wait()
        return False
