"""Preview zoom controls; percentages refer to source pixels on the display."""
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import DoubleSpinBox, PushButton

from raw_alchemy.i18n import tr


class ZoomControls(QWidget):
    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        self.fit_button = PushButton(tr('zoom_fit'))
        self.native_button = PushButton('100%')
        self.native_button.setToolTip(tr('zoom_native_hint'))
        self.percent = DoubleSpinBox()
        self.percent.setRange(0.1, 1600.0)
        self.percent.setDecimals(1)
        self.percent.setSingleStep(25.0)
        self.percent.setSuffix('%')
        self.percent.setFixedWidth(128)
        self.percent.setAccessibleName(tr('zoom_percentage'))
        self.percent.setToolTip(tr('zoom_percentage'))
        self.percent.setKeyboardTracking(False)
        for widget in (self.fit_button, self.native_button, self.percent):
            layout.addWidget(widget)
        self.fit_button.clicked.connect(viewport.fit_to_view)
        self.native_button.clicked.connect(viewport.zoom_to_100)
        self.percent.valueChanged.connect(viewport.set_zoom_percent)
        viewport.zoom_changed.connect(self.sync)
        viewport.image_changed.connect(self.sync)
        viewport.viewport_resized.connect(self.sync)
        self.sync()

    def sync(self, *args):
        percent = self.viewport.zoom_percent()
        self.setEnabled(percent is not None)
        blocked = self.percent.blockSignals(True)
        try:
            self.percent.setMaximum(max(1600.0, percent or 0.0))
            self.percent.setValue(percent if percent is not None else 100.0)
        finally:
            self.percent.blockSignals(blocked)
