from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QPolygonF, QImage
import numpy as np
from loguru import logger
from raw_alchemy import utils

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.hist_data = None
        self._hist_image = None  # Pre-rendered QImage cache
        self._hist_image_size = None  # Cached size for invalidation
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # 优化: 添加更新定时器防抖
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(25)  # 25ms防抖
        self.update_timer.timeout.connect(self._do_update)
        self.pending_data = None

    def update_data(self, img_array):
        """异步更新直方图数据 - 使用防抖避免频繁计算

        Accepts both uint8 [0,255] and float32 [0,1] arrays.
        Copy and conversion are deferred to _do_update to avoid blocking UI.
        """
        if img_array is None:
            return
        self.pending_data = img_array  # Store reference only (no copy)
        self.update_timer.start()

    def _do_update(self):
        """实际执行直方图计算"""
        if self.pending_data is None:
            return

        data = self.pending_data
        self.pending_data = None

        try:
            if data is None or data.size == 0:
                return

            # 使用utils中的快速计算函数
            self.hist_data = utils.compute_histogram_fast(data, bins=256, sample_rate=4)
            self._hist_image = None  # Invalidate cache
            self.update()
        except BaseException as e:
            try:
                logger.error(f"Histogram update error: {type(e).__name__}: {e}")
            except Exception:
                pass

    def _render_hist_image(self, w, h):
        """Pre-render histogram to a QImage with additive blending."""
        if not self.hist_data:
            return

        img = QImage(w, h, QImage.Format.Format_RGB32)
        img.fill(QColor(20, 20, 20))

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        try:
            # 忽略两端极值 + 对数缩放
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

        # 加色混合模式
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
