"""Thumbnail grid for the library panel."""

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QListWidget

# Every thumbnail is letterboxed into this aspect ratio, so portrait and
# landscape frames occupy the same icon box and the file names line up.
THUMB_ASPECT = 1.3


def boxed_icon(image, aspect=THUMB_ASPECT):
    """QIcon of ``image`` centred on a transparent ``aspect`` canvas."""
    width, height = image.width(), image.height()
    if width <= 0 or height <= 0:
        return QIcon(QPixmap.fromImage(image))
    box_h = max(height, round(width / aspect))
    box_w = max(width, round(box_h * aspect))
    canvas = QPixmap(box_w, box_h)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.drawImage((box_w - width) // 2, (box_h - height) // 2, image)
    painter.end()
    return QIcon(canvas)


class GalleryListWidget(QListWidget):
    """Icon grid whose columns always share the full width evenly.

    A fixed grid size left a ragged gap on the right and never used a wider
    panel; here the column count follows the width and the thumbnails grow
    with their cells.
    """

    CELL_MIN_WIDTH = 150
    TEXT_HEIGHT = 34
    CELL_PADDING = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setUniformItemSizes(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)  # keep the extension
        # A permanent scrollbar keeps the width stable, so the column count
        # cannot oscillate as the scrollbar appears and disappears.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.viewport().installEventFilter(self)
        self._fit_grid()

    def eventFilter(self, obj, event):
        if obj is self.viewport() and event.type() == QEvent.Type.Resize:
            self._fit_grid()
        return super().eventFilter(obj, event)

    @classmethod
    def grid_for_width(cls, width):
        """(cell size, icon size) for a viewport ``width`` in pixels."""
        # IconMode wraps unless the row is strictly narrower than the
        # viewport, so keep a 2 px margin or an exact fit drops a column.
        usable = max(1, width - 2)
        columns = max(1, usable // cls.CELL_MIN_WIDTH)
        cell_w = max(cls.CELL_MIN_WIDTH, usable // columns)
        icon_w = cell_w - cls.CELL_PADDING
        icon_h = round(icon_w / THUMB_ASPECT)
        return QSize(cell_w, icon_h + cls.TEXT_HEIGHT + cls.CELL_PADDING // 2), QSize(icon_w, icon_h)

    def _fit_grid(self):
        cell, icon = self.grid_for_width(self.viewport().width())
        if cell != self.gridSize():
            self.setIconSize(icon)
            self.setGridSize(cell)
