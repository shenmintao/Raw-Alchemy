from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtGui import QImage, QTransform
import os
import numpy as np
import concurrent.futures
from loguru import logger

from raw_alchemy.config import SUPPORTED_RAW_EXTENSIONS

# Module-level decoder shared across thumbnail worker threads
_thumb_decoder = None
_thumb_decoder_lock = None


def _get_thumb_decoder():
    """Get or create the shared RawSpeed decoder for thumbnails."""
    global _thumb_decoder, _thumb_decoder_lock
    import threading
    if _thumb_decoder_lock is None:
        _thumb_decoder_lock = threading.Lock()
    with _thumb_decoder_lock:
        if _thumb_decoder is None:
            from raw_alchemy.rawspeed_binding import RawSpeedDecoder
            _thumb_decoder = RawSpeedDecoder()
        return _thumb_decoder


class ThumbnailWorker(QThread):
    """
    Scan folder and generate thumbnails - 优化版本使用线程池
    """
    # Define signals
    thumbnail_ready = Signal(str, QImage)
    progress_update = Signal(int, int)
    finished_scanning = Signal()

    def __init__(self, folder_path, max_workers=4):
        super().__init__()
        self.folder_path = folder_path
        self.stopped = False
        self.max_workers = max_workers

    @staticmethod
    def extract_thumbnail(full_path):
        """
        静态方法用于线程池并行处理。
        Uses RawSpeed for decoding + simple bilinear demosaic for speed.
        """
        try:
            # 1. 快速检查文件扩展名
            ext = os.path.splitext(full_path)[1].lower()
            if ext not in SUPPORTED_RAW_EXTENSIONS:
                return None

            image = None
            orientation = 0

            # 2. Read EXIF orientation (best effort)
            try:
                import pyexiv2
                with pyexiv2.Image(full_path) as exif_img:
                    exif_data = exif_img.read_exif() or {}
                    exif_orient = int(exif_data.get('Exif.Image.Orientation', 1))
                    # Map EXIF orientation to flip codes:
                    # EXIF 1=normal(0), 3=180(3), 6=90CW(6), 8=90CCW(5)
                    _exif_to_flip = {1: 0, 2: 0, 3: 3, 4: 0, 5: 5, 6: 6, 7: 6, 8: 5}
                    orientation = _exif_to_flip.get(exif_orient, 0)
            except Exception:
                pass

            # 3. Decode with RawSpeed and generate thumbnail via downscaled demosaic
            try:
                decoder = _get_thumb_decoder()
                result = decoder.decode(full_path)

                # Simple half-size bilinear demosaic for speed (thumbnails don't need RCD quality)
                bayer = result.normalize()
                h, w = bayer.shape
                h2, w2 = h // 2 * 2, w // 2 * 2
                bayer = bayer[:h2, :w2]

                # Quick 2x2 average demosaic (just average the Bayer quad)
                r = bayer[0::2, 0::2]
                g1 = bayer[0::2, 1::2]
                g2 = bayer[1::2, 0::2]
                b = bayer[1::2, 1::2]
                thumb_h, thumb_w = r.shape

                rgb = np.stack([r, (g1 + g2) * 0.5, b], axis=-1)

                # Apply white balance
                wb = result.wb_coeffs
                g = wb[1] if wb[1] > 0 else 1.0
                rgb[:, :, 0] *= wb[0] / g
                rgb[:, :, 2] *= wb[2] / g

                # Simple sRGB-ish gamma for display (skip full color pipeline for speed)
                np.clip(rgb, 0.0, 1.0, out=rgb)
                rgb = np.power(rgb, 1.0 / 2.2)

                # Convert to uint8
                thumb_uint8 = (rgb * 255).astype(np.uint8)
                thumb_uint8 = np.ascontiguousarray(thumb_uint8)

                image = QImage(
                    thumb_uint8.data,
                    thumb_w,
                    thumb_h,
                    3 * thumb_w,
                    QImage.Format_RGB888
                ).copy()

            except Exception:
                pass

            # 4. 统一缩放 & 旋转
            if image and not image.isNull():
                # Apply rotation based on orientation
                if orientation == 3:
                    image = image.transformed(QTransform().rotate(180))
                elif orientation == 5:
                    image = image.transformed(QTransform().rotate(-90))
                elif orientation == 6:
                    image = image.transformed(QTransform().rotate(90))

                # 统一缩放为 300px 高度，保持比例
                return image.scaledToHeight(300)

            return None

        except Exception as e:
            # logger.error(f"Error extracting thumbnail for {full_path}: {e}")
            return None

    def run(self):
        # 1. Scan folder
        valid_extensions = SUPPORTED_RAW_EXTENSIONS
        files = []
        try:
            with os.scandir(self.folder_path) as entries:
                for entry in entries:
                    if self.stopped:
                        return
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_extensions:
                        files.append(entry.path)
        except Exception as e:
            logger.error(f"Failed to scan directory: {e}")
            self.finished_scanning.emit()
            return

        total = len(files)
        # self.progress_update.emit(0, total) # Optional startup signal

        # 2. Parallel processing using ThreadPoolExecutor
        # 注意: QThread.run 是在独立线程中，我们可以开启一个在这里wait的线程池
        # 或者直接使用 max_workers 限制并发
        processed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(self.extract_thumbnail, f): f for f in files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                if self.stopped:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return

                f_path = future_to_file[future]
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
