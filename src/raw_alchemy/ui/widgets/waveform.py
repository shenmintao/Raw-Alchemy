from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QImage
import numpy as np
from loguru import logger
from raw_alchemy import utils

class WaveformWidget(QWidget):
    """示波器组件 - 显示图像的亮度分布"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.waveform_data = None
        self._waveform_image = None  # Pre-rendered QImage buffer
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # 优化: 添加更新定时器防抖
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(25)  # 25ms防抖
        self.update_timer.timeout.connect(self._do_update)
        self.pending_data = None

    def update_data(self, img_array):
        """异步更新示波器数据 - 使用防抖避免频繁计算

        Accepts both uint8 [0,255] and float32 [0,1] arrays.
        Copy and conversion are deferred to _do_update to avoid blocking UI.
        """
        if img_array is None:
            return
        self.pending_data = img_array  # Store reference only (no copy)
        self.update_timer.start()

    def _do_update(self):
        """实际执行示波器计算"""
        if self.pending_data is None:
            return

        data = self.pending_data
        self.pending_data = None

        try:
            if data is None or data.size == 0:
                return

            # 使用utils中的快速计算函数 - 增加bins数量以提高垂直分辨率
            waveform_result = utils.compute_waveform_fast(data, bins=150, sample_rate=8)

            # 检查结果是否有效
            if waveform_result is not None:
                self.waveform_data = waveform_result
                self._render_waveform_image()
                self.update()
        except (RuntimeError, ValueError, TypeError, OSError, SystemError) as e:
            logger.warning(f"Waveform update error: {type(e).__name__}: {e}")
        except BaseException as e:
            logger.error(f"Waveform update error: {type(e).__name__}: {e}")

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
        # waveform[col, bin] → image pixel at (row=num_bins-1-bin, col)
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

        # 填充深色背景
        painter.fillRect(self.rect(), QColor(10, 10, 10))

        # 绘制波形图像（缩放到widget尺寸）
        if self._waveform_image is not None:
            painter.drawImage(self.rect(), self._waveform_image)

        # 绘制专业网格线（达芬奇风格）
        try:
            line_color = QColor(0, 255, 0, 180)

            def ire_to_y(ire_value):
                normalized = (ire_value - (-4.0)) / 113.0
                return h - (normalized * h)

            # 109% IRE - 虚线
            painter.setPen(QPen(line_color, 0.5, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(ire_to_y(109)), w, int(ire_to_y(109)))

            # 100% IRE - 实线
            painter.setPen(QPen(line_color, 1.0, Qt.PenStyle.SolidLine))
            painter.drawLine(0, int(ire_to_y(100)), w, int(ire_to_y(100)))

            # 50% IRE - 实线
            painter.drawLine(0, int(ire_to_y(50)), w, int(ire_to_y(50)))

            # 0% IRE - 实线
            painter.drawLine(0, int(ire_to_y(0)), w, int(ire_to_y(0)))

            # -4% IRE - 虚线
            painter.setPen(QPen(line_color, 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(ire_to_y(-4)), w, int(ire_to_y(-4)))

            # 0-100%之间每10%画虚线
            painter.setPen(QPen(line_color, 0.5, Qt.PenStyle.DashLine))
            for ire in [10, 20, 30, 40, 60, 70, 80, 90]:
                y_ire = ire_to_y(ire)
                painter.drawLine(0, int(y_ire), w, int(y_ire))

        except Exception as e:
            logger.error(f"Error drawing grid: {e}")
