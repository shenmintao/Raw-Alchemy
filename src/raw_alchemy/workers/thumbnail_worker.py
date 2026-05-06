from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtGui import QImage, QTransform
import os
import numpy as np
import concurrent.futures
from loguru import logger

from raw_alchemy.config import SUPPORTED_RAW_EXTENSIONS



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

            # 2. Decode with rawpy and generate thumbnail via downscaled demosaic
            try:
                import rawpy
                with rawpy.imread(full_path) as raw:
                    # Read raw sensor data (NO postprocess)
                    bayer = raw.raw_image_visible.astype(np.float32)
                    bl = float(raw.black_level_per_channel[0])
                    wl = float(raw.white_level)
                    wb = np.array(raw.camera_whitebalance, dtype=np.float32)
                    flip = raw.sizes.flip

                # Black level subtract + normalize
                bayer = np.maximum(bayer - bl, 0) / (wl - bl)
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
                g = wb[1] if wb[1] > 0 else 1.0
                rgb[:, :, 0] *= wb[0] / g
                rgb[:, :, 2] *= wb[2] / g

                # Apply orientation
                from raw_alchemy.onnx.denoiser import _apply_flip
                rgb = np.ascontiguousarray(_apply_flip(rgb, flip))
                thumb_h, thumb_w = rgb.shape[0], rgb.shape[1]

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

            # 3. 统一缩放
            if image and not image.isNull():
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
