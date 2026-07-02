from typing import Optional

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap


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
        self.full: Optional[QPixmap] = None
        self.display: Optional[QPixmap] = None
        self.float_data: Optional[np.ndarray] = None
        self.uint8_data: Optional[np.ndarray] = None
        self.source_size: Optional[tuple[int, int]] = None

    def update_full(
        self,
        pixmap: QPixmap,
        float_data: Optional[np.ndarray] = None,
        uint8_data: Optional[np.ndarray] = None,
        source_size: Optional[tuple[int, int]] = None,
    ):
        """Update the full-size image and clear cached display version."""
        self.full = pixmap
        self.float_data = float_data
        self.uint8_data = uint8_data
        self.source_size = source_size
        self.display = None

    def get_display(self, size: QSize) -> Optional[QPixmap]:
        """Get display-sized version, caching the result."""
        if not self.full:
            return None

        if self.display is None:
            self.display = self.full.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return self.display

    def clear(self):
        """Clear all cached data."""
        self.full = None
        self.display = None
        self.float_data = None
        self.uint8_data = None
        self.source_size = None
