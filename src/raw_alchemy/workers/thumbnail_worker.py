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

            # 2. Use rawpy half_size postprocess (fast, accurate colors)
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
                thumb_h, thumb_w = rgb.shape[0], rgb.shape[1]

                # sRGB OETF
                np.clip(rgb, 0.0, 1.0, out=rgb)
                mask = rgb <= 0.0031308
                rgb[mask] *= 12.92
                rgb[~mask] = 1.055 * np.power(rgb[~mask], 1.0 / 2.4) - 0.055

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
