"""Single-image export resolves format suffixes inside the save dialog (#27)."""
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('name_filter,suffix', [
    ('JPEG (*.jpg)', '.jpg'), ('HEIF (*.heif)', '.heif'),
    ('HDR HEIF (*.hdr.heif)', '.hdr.heif'), ('TIFF (*.tif)', '.tif'),
    ('DNG (*.dng)', '.dng'),
])
def test_export_uses_selected_format_without_typing_suffix(app, monkeypatch, tmp_path, name_filter, suffix):
    from PySide6.QtWidgets import QFileDialog, QWidget
    from raw_alchemy.ui.export_controller import ExportControllerMixin, ExportFileDialog, InfoBar

    class Window(QWidget, ExportControllerMixin):
        pass

    window = Window()
    window.current_raw_path = str(tmp_path / 'input.RAF')
    window.last_export_path = str(tmp_path)
    window.last_folder_path = ''
    window.processor = SimpleNamespace(get_cached_for_export=lambda params: None)
    window.right_panel = SimpleNamespace(get_params=lambda: {})
    window.btn_export_curr = SimpleNamespace(setEnabled=lambda enabled: None)
    exports = []
    window.run_export = lambda source, output, **kwargs: exports.append(output)
    monkeypatch.setattr(InfoBar, 'info', lambda *a, **kw: None)

    def choose(dialog):
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.selectNameFilter(name_filter)
        dialog.filterSelected.emit(name_filter)
        dialog.selectFile(str(tmp_path / 'picture'))
        assert not dialog.testOption(QFileDialog.Option.DontConfirmOverwrite)
        # Qt sees the final filename while the picker is still open.
        assert dialog.selectedFiles()[0].endswith(suffix)
        return QFileDialog.DialogCode.Accepted

    monkeypatch.setattr(ExportFileDialog, 'exec', choose)
    window.export_current()
    assert exports == [str(tmp_path / ('picture' + suffix)).replace('\\', '/')]
    window.close()


def test_existing_explicit_extension_is_preserved(app, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    from raw_alchemy.ui.export_controller import ExportFileDialog
    dialog = ExportFileDialog(None, str(tmp_path / 'picture.jpeg'))
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
    assert Path(dialog.selectedFiles()[0]).name == 'picture.jpeg'
    dialog.close()


def test_cancelling_export_does_not_submit_work(app, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QWidget
    from raw_alchemy.ui.export_controller import ExportControllerMixin, ExportFileDialog
    class Window(QWidget, ExportControllerMixin):
        pass
    window = Window()
    window.current_raw_path = str(tmp_path / 'input.RAF')
    window.last_export_path = str(tmp_path)
    window.last_folder_path = ''
    window.run_export = lambda *a, **kw: pytest.fail('cancel must not export')
    monkeypatch.setattr(ExportFileDialog, 'exec', lambda self: 0)
    window.export_current()
    window.close()
