"""Zoom uses original sensor dimensions; export marks survive thumbnail work."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_zoom_controls_follow_source_pixels_hidpi_and_wheel_state(app, monkeypatch):
    from raw_alchemy.ui.viewport_gl import ImageViewportGL
    from raw_alchemy.ui.widgets.zoom_controls import ZoomControls

    viewport = ImageViewportGL()
    viewport.resize(1000, 800)
    monkeypatch.setattr(viewport, 'devicePixelRatioF', lambda: 1.5)
    controls = ZoomControls(viewport)
    assert not controls.isEnabled()
    viewport.set_image(np.zeros((60, 80, 3), np.uint8), source_size=(8000, 6000))
    assert controls.isEnabled()
    assert controls.percent.value() == pytest.approx(18.8)  # fit = 1500 / 8000
    controls.percent.setValue(200.0)
    assert viewport.zoom_percent() == pytest.approx(200.0)
    assert viewport.preview_zoom() == pytest.approx(2.0 * 8000 / 1500)
    controls.native_button.click()
    assert viewport.zoom_percent() == pytest.approx(100.0)
    assert controls.percent.value() == pytest.approx(100.0)
    viewport.set_zoom_percent(75.0)  # wheel / viewport change synchronizes back
    assert controls.percent.value() == pytest.approx(75.0)
    controls.fit_button.click()
    assert viewport.preview_zoom() == pytest.approx(1.0)
    viewport.clear_image()
    assert not controls.isEnabled()
    controls.close()
    viewport.close()


def test_refreshing_zoom_controls_does_not_submit_another_zoom_request(app):
    from raw_alchemy.ui.viewport_gl import ImageViewportGL
    from raw_alchemy.ui.widgets.zoom_controls import ZoomControls

    viewport = ImageViewportGL()
    controls = ZoomControls(viewport)
    seen = []
    viewport.zoom_changed.connect(seen.append)
    viewport.set_image(np.zeros((60, 80, 3), np.uint8), source_size=(8000, 6000))
    controls.sync()
    assert seen == []  # receiving each render must not create a render loop
    controls.close()
    viewport.close()


def test_mark_badge_survives_replaced_thumbnail_and_disappears_on_unmark(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage
    from raw_alchemy.ui.library_controller import LibraryControllerMixin
    from raw_alchemy.ui.widgets.gallery_list import GalleryListWidget, MARKED_ROLE

    gallery = GalleryListWidget()
    gallery.resize(330, 220)
    image = QImage(130, 100, QImage.Format.Format_RGB32)
    image.fill(QColor('#474747'))
    window = SimpleNamespace(gallery_list=gallery, gallery_items_by_path={}, marked_files={'a.nef'})
    LibraryControllerMixin.add_gallery_items(window, ['a.nef', 'b.nef'], image)
    gallery.show()
    app.processEvents()
    item = window.gallery_items_by_path['a.nef']
    assert item.data(Qt.ItemDataRole.UserRole) == 'a.nef'
    assert item.data(MARKED_ROLE)

    def green_pixels():
        rendered = gallery.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(rendered.bits(), np.uint8).reshape(rendered.height(), -1, 4)
        return np.all(pixels[:, :rendered.width(), :3] == [54, 214, 134], axis=2).sum()

    assert green_pixels() > 100
    image.fill(QColor('#565656'))
    LibraryControllerMixin.update_gallery_item(window, 'a.nef', image)
    app.processEvents()
    assert green_pixels() > 100
    window.marked_files.clear()
    LibraryControllerMixin.update_gallery_item_mark_indicator(window, 'a.nef')
    app.processEvents()
    assert green_pixels() == 0
    gallery.close()
