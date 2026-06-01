from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QImage
import numpy as np
from loguru import logger
from raw_alchemy import utils


class _WaveformWorker(QThread):
    result_ready = Signal(object, int)

    def __init__(self, img_array, generation, parent=None):
        super().__init__(parent)
        self.img_array = img_array
        self.generation = generation

    def run(self):
        result = None
        try:
            if self.img_array is not None and self.img_array.size != 0:
                result = utils.compute_waveform_fast(self.img_array, bins=150, sample_rate=8)
        except (RuntimeError, ValueError, TypeError, OSError, SystemError) as e:
            logger.warning(f"Waveform update error: {type(e).__name__}: {e}")
        except BaseException as e:
            logger.error(f"Waveform update error: {type(e).__name__}: {e}")
        self.result_ready.emit(result, self.generation)


class WaveformWidget(QWidget):
    """Waveform scope widget for image luminance distribution."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.waveform_data = None
        self._waveform_image = None  # Pre-rendered QImage buffer
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Debounce rapid preview updates before starting background work.
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(25)
        self.update_timer.timeout.connect(self._do_update)
        self.pending_data = None
        self._worker = None
        self._generation = 0

    def update_data(self, img_array):
        """Schedule a waveform update without blocking the UI thread.

        Accepts both uint8 [0,255] and float32 [0,1] arrays.
        """
        if img_array is None:
            return
        self.pending_data = img_array  # Store reference only (no copy)
        self.update_timer.start()

    def _do_update(self):
        """Run waveform computation off the UI thread."""
        if self.pending_data is None:
            return

        if self._worker is not None and self._worker.isRunning():
            self.update_timer.start()
            return

        data = self.pending_data
        self.pending_data = None
        self._generation += 1

        worker = _WaveformWorker(data, self._generation)
        self._worker = worker
        worker.result_ready.connect(self._on_worker_result)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        worker.start()

    def _on_worker_result(self, waveform_result, generation):
        if generation != self._generation or waveform_result is None:
            return
        self.waveform_data = waveform_result
        self._render_waveform_image()
        self.update()

    def _on_worker_finished(self, worker):
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()
        if self.pending_data is not None:
            self.update_timer.start()

    def shutdown_worker(self):
        self.update_timer.stop()
        self.pending_data = None
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(1500)
            self._worker = None

    def _render_waveform_image(self):
        """Pre-render waveform data to a QImage buffer using numpy (no Python loops)."""
        waveform = self.waveform_data
        if waveform is None:
            self._waveform_image = None
            return

        num_cols, num_bins = waveform.shape
        if num_cols == 0:
            self._waveform_image = None
            return

        # Build RGBA image: shape [num_bins, num_cols, 4] (height x width)
        # Flip vertically: bin 0 (low IRE) at bottom, bin N-1 (high IRE) at top
        # waveform[col, bin] -> image pixel at (row=num_bins-1-bin, col)
        img_data = waveform.T[::-1, :]  # shape [num_bins, num_cols], flipped vertically

        # Non-linear density mapping (gamma 0.6) for visibility
        mask = img_data > 0
        enhanced = np.zeros_like(img_data)
        enhanced[mask] = np.power(img_data[mask], 0.6)

        # Brightness: enhanced * 150 + 180, clamped to 255
        brightness = np.clip((enhanced * 150 + 180), 0, 255).astype(np.uint8)
        # Alpha: enhanced * 200 + 150, clamped to 255; 0 where no data
        alpha = np.zeros_like(brightness)
        alpha[mask] = np.clip((enhanced[mask] * 200 + 150).astype(np.int32), 0, 255).astype(np.uint8)

        # Assemble RGBA buffer (Format_RGBA8888)
        rgba = np.zeros((num_bins, num_cols, 4), dtype=np.uint8)
        rgba[:, :, 0] = brightness  # R
        rgba[:, :, 1] = brightness  # G
        rgba[:, :, 2] = brightness  # B
        rgba[:, :, 3] = alpha

        # Zero out pixels with no data (fully transparent)
        rgba[~mask, :] = 0

        # Create QImage from buffer (must keep reference to data)
        rgba_contiguous = np.ascontiguousarray(rgba)
        self._waveform_rgba = rgba_contiguous  # prevent GC
        self._waveform_image = QImage(
            rgba_contiguous.data, num_cols, num_bins,
            num_cols * 4, QImage.Format.Format_RGBA8888
        )

    def paintEvent(self, event):
        if self.waveform_data is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Dark background.
        painter.fillRect(self.rect(), QColor(10, 10, 10))

        # Draw waveform image scaled to widget size.
        if self._waveform_image is not None:
            painter.drawImage(self.rect(), self._waveform_image)

        # Draw reference IRE grid lines.
        try:
            line_color = QColor(0, 255, 0, 180)

            def ire_to_y(ire_value):
                normalized = (ire_value - (-4.0)) / 113.0
                return h - (normalized * h)

            # 109% IRE - dashed.
            painter.setPen(QPen(line_color, 0.5, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(ire_to_y(109)), w, int(ire_to_y(109)))

            # 100% IRE - solid.
            painter.setPen(QPen(line_color, 1.0, Qt.PenStyle.SolidLine))
            painter.drawLine(0, int(ire_to_y(100)), w, int(ire_to_y(100)))

            # 50% IRE - solid.
            painter.drawLine(0, int(ire_to_y(50)), w, int(ire_to_y(50)))

            # 0% IRE - solid.
            painter.drawLine(0, int(ire_to_y(0)), w, int(ire_to_y(0)))

            # -4% IRE - dashed.
            painter.setPen(QPen(line_color, 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(ire_to_y(-4)), w, int(ire_to_y(-4)))

            # Dashed guides every 10% between 0 and 100%.
            painter.setPen(QPen(line_color, 0.5, Qt.PenStyle.DashLine))
            for ire in [10, 20, 30, 40, 60, 70, 80, 90]:
                y_ire = ire_to_y(ire)
                painter.drawLine(0, int(y_ire), w, int(y_ire))

        except Exception as e:
            logger.error(f"Error drawing grid: {e}")
