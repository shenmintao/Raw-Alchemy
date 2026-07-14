from typing import Optional

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap


class ImageState:
    """
    Unified state for a single image.
    Replaces the mess of: original_pixmap_raw, original_pixmap_scaled,
    last_processed_pixmap, _last_processed_pixmap_full, etc.

    Three states total:
    - original: RAW decoded image
    - current: processed with current params
    - baseline: saved baseline (optional)
    """

    def __init__(self):
        # Do not retain a full-resolution QPixmap. The OpenGL viewport already
        # owns the presentation texture; keeping a Qt raster copy alongside
        # uint8 state costs another 4 bytes/pixel (244 MiB at 61MP).
        self.full = None  # compatibility attribute; presence is uint8_data
        self.display: Optional[QPixmap] = None
        self.float_data: Optional[np.ndarray] = None
        self.uint8_data: Optional[np.ndarray] = None
        # Bounded whole-frame view used when switching histogram/waveform.
        # It aliases uint8_data (or a strided view), never a second frame.
        self.scope_uint8_data: Optional[np.ndarray] = None
        self.source_size: Optional[tuple[int, int]] = None

    def update_full(
        self,
        pixmap: QPixmap,
        float_data: Optional[np.ndarray] = None,
        uint8_data: Optional[np.ndarray] = None,
        source_size: Optional[tuple[int, int]] = None,
    ):
        """Update the full-size image and clear cached display version."""
        self.full = None
        self.float_data = float_data
        self.uint8_data = uint8_data
        self.scope_uint8_data = uint8_data
        self.source_size = source_size
        self.display = None

    def get_display(self, size: QSize) -> Optional[QPixmap]:
        """Get display-sized version, caching the result."""
        if self.uint8_data is None:
            return None

        if self.display is None:
            h, w, c = self.uint8_data.shape
            image = QImage(
                self.uint8_data.data,
                w,
                h,
                c * w,
                QImage.Format.Format_RGB888,
            )
            self.display = QPixmap.fromImage(
                image.scaled(
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        return self.display

    def clear(self):
        """Clear all cached data."""
        self.full = None
        self.display = None
        self.float_data = None
        self.uint8_data = None
        self.scope_uint8_data = None
        self.source_size = None
