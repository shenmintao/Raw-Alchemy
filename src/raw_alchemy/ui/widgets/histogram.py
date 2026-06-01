from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QPointF, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPolygonF, QImage
import numpy as np
from loguru import logger
from raw_alchemy import utils


class _HistogramWorker(QThread):
    result_ready = Signal(object, int)

    def __init__(self, img_array, generation, parent=None):
        super().__init__(parent)
        self.img_array = img_array
        self.generation = generation

    def run(self):
        result = None
        try:
            if self.img_array is not None and self.img_array.size != 0:
                result = utils.compute_histogram_fast(self.img_array, bins=256, sample_rate=4)
        except BaseException as e:
            try:
                logger.error(f"Histogram update error: {type(e).__name__}: {e}")
            except Exception:
                pass
        self.result_ready.emit(result, self.generation)


class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.hist_data = None
        self._hist_image = None  # Pre-rendered QImage cache
        self._hist_image_size = None  # Cached size for invalidation
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
        """Schedule a histogram update without blocking the UI thread.

        Accepts both uint8 [0,255] and float32 [0,1] arrays.
        """
        if img_array is None:
            return
        self.pending_data = img_array  # Store reference only (no copy)
        self.update_timer.start()

    def _do_update(self):
        """Run histogram computation off the UI thread."""
        if self.pending_data is None:
            return

        if self._worker is not None and self._worker.isRunning():
            self.update_timer.start()
            return

        data = self.pending_data
        self.pending_data = None
        self._generation += 1

        worker = _HistogramWorker(data, self._generation)
        self._worker = worker
        worker.result_ready.connect(self._on_worker_result)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        worker.start()

    def _on_worker_result(self, hist_data, generation):
        if generation != self._generation or hist_data is None:
            return
        self.hist_data = hist_data
        self._hist_image = None  # Invalidate cache
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

    def _render_hist_image(self, w, h):
        """Pre-render histogram to a QImage with additive blending."""
        if not self.hist_data:
            return

        img = QImage(w, h, QImage.Format.Format_RGB32)
        img.fill(QColor(20, 20, 20))

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        try:
            # Ignore endpoint spikes and use log scaling.
            display_max_vals = []
            for hist in self.hist_data:
                if len(hist) > 2:
                    inner_max = np.max(hist[1:-1])
                    display_max_vals.append(inner_max if inner_max > 0 else 1)
                else:
                    display_max_vals.append(np.max(hist) if len(hist) > 0 else 1)

            display_max = max(display_max_vals) if display_max_vals else 1
            if display_max == 0 or display_max < 1e-10:
                display_max = 1

            log_max_height = np.log1p(display_max)
        except Exception as e:
            logger.error(f"Error computing display_max: {e}")
            painter.end()
            return

        colors = [
            QColor(255, 0, 0, 160),
            QColor(0, 255, 0, 160),
            QColor(0, 0, 255, 160)
        ]

        # Additive blending.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        for i, hist in enumerate(self.hist_data):
            if len(hist) == 0:
                continue

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colors[i])

            bin_w = w / len(hist)

            # Vectorized ratio computation with numpy
            hist_arr = np.asarray(hist, dtype=np.float64)
            ratios = np.clip(np.log1p(hist_arr) / log_max_height, 0.0, 1.0)
            xs = np.arange(len(hist)) * bin_w
            ys = h - (ratios * h)

            points = [QPointF(0, h)]
            for j in range(len(hist)):
                points.append(QPointF(xs[j], ys[j]))
            points.append(QPointF(w, h))
            painter.drawPolygon(QPolygonF(points))

        painter.end()

        self._hist_image = img
        self._hist_image_size = (w, h)

    def paintEvent(self, event):
        if not self.hist_data:
            return

        w = self.width()
        h = self.height()

        # Re-render cache if data changed or size changed
        if self._hist_image is None or self._hist_image_size != (w, h):
            self._render_hist_image(w, h)

        if self._hist_image is not None:
            painter = QPainter(self)
            painter.drawImage(0, 0, self._hist_image)
            painter.end()
