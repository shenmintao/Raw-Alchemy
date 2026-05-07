from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QImage, QColor
import os
import numpy as np
import concurrent.futures
from loguru import logger

from raw_alchemy.config import SUPPORTED_RAW_EXTENSIONS


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


class ThumbnailWorker(QThread):
    """
    Scan folder and generate thumbnails.
    1. Emit placeholders sorted by filename (instant)
    2. Load actual thumbnails in parallel, emit as ready
    """
    thumbnail_ready = Signal(str, QImage)
    placeholder_ready = Signal(str, QImage)  # sorted placeholders
    progress_update = Signal(int, int)
    finished_scanning = Signal()

    THUMB_HEIGHT = 300

    def __init__(self, folder_path, max_workers=8):
        super().__init__()
        self.folder_path = folder_path
        self.stopped = False
        self.max_workers = max_workers

    @staticmethod
    def _make_placeholder(width=200, height=300) -> QImage:
        """Create a gray placeholder QImage."""
        img = QImage(width, height, QImage.Format_RGB888)
        img.fill(QColor(40, 40, 40))
        return img

    @staticmethod
    def extract_thumbnail(full_path):
        """
        Extract thumbnail — tries embedded JPEG first (~50ms),
        falls back to rawpy half_size postprocess (~300ms).
        """
        try:
            ext = os.path.splitext(full_path)[1].lower()
            if ext not in SUPPORTED_RAW_EXTENSIONS:
                return None

            image = None
            used_embedded = False
            flip = 0

            # Method 1: Extract embedded JPEG thumbnail (fastest, ~50ms)
            try:
                import rawpy
                with rawpy.imread(full_path) as raw:
                    flip = raw.sizes.flip
                    thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    image = QImage()
                    image.loadFromData(thumb.data)
                    used_embedded = True
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    h, w = thumb.data.shape[:2]
                    thumb_data = np.ascontiguousarray(thumb.data)
                    image = QImage(
                        thumb_data.data, w, h, 3 * w,
                        QImage.Format_RGB888
                    ).copy()
                    used_embedded = True
            except Exception:
                pass

            # Embedded thumbnails are typically unrotated; apply camera orientation.
            if used_embedded and image and not image.isNull():
                image = _apply_flip_rotation(image, flip)

            # Method 2: Fallback to rawpy half_size postprocess
            # rawpy.postprocess() applies orientation automatically, no extra rotation needed.
            if image is None or image.isNull():
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
                    rgb = rgb16.astype(np.float32) / 65535.0
                    del rgb16

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
                    pass

            if image and not image.isNull():
                return image.scaledToHeight(ThumbnailWorker.THUMB_HEIGHT)
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

        # 2. Emit placeholders in sorted order (instant, UI can lay out immediately)
        placeholder = self._make_placeholder()
        for f in files:
            if self.stopped:
                return
            self.placeholder_ready.emit(f, placeholder)

        # 3. Load actual thumbnails in parallel, emit as ready
        processed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.extract_thumbnail, f): f for f in files}

            for future in concurrent.futures.as_completed(futures):
                if self.stopped:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return

                f_path = futures[future]
                try:
                    qimg = future.result()
                    if qimg:
                        self.thumbnail_ready.emit(f_path, qimg)
                except Exception as e:
                    logger.error(f"Worker exception for {f_path}: {e}")

                processed_count += 1
                if processed_count % 5 == 0 or processed_count == total:
                    self.progress_update.emit(processed_count, total)

        self.finished_scanning.emit()

    def stop(self):
        self.stopped = True
