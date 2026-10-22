"""CUDA settings offer local import or download only when dependencies are absent."""
import time
from unittest.mock import Mock

import onnxruntime as ort
import pytest

from raw_alchemy.onnx import gpu_runtime as runtime


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from raw_alchemy.ui.widgets import settings_panel as ui

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(runtime, 'get_cuda_runtime_dir', lambda: tmp_path / 'cuda_runtime')
    monkeypatch.setattr(runtime, '_system_cuda_search_dirs', lambda: [])
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(runtime, 'cuda_provider_bundled', lambda: True)
    monkeypatch.setattr(runtime, 'detect_gpu_vendor', lambda: {
        'vendor': 'nvidia', 'name': 'NVIDIA GeForce RTX 3050 Ti', 'cuda_compatible': True,
    })
    monkeypatch.setattr(ui, 'InfoBar', Mock())
    panel = ui.SettingsPanel()
    yield panel, ui, app
    panel.shutdown_worker()
    panel.close()
    panel.deleteLater()
    app.processEvents()


def _install_libraries(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for pattern, required in runtime.CUDA_LIBS['Windows']:
        if required:
            (directory / pattern.replace('*', '2')).write_bytes(b'fixture')


def test_existing_system_runtime_hides_install_buttons_and_never_downloads(settings, monkeypatch, tmp_path):
    panel, ui, _ = settings
    system = tmp_path / 'system'
    _install_libraries(system)
    monkeypatch.setattr(runtime, '_system_cuda_search_dirs', lambda: [system])
    start = Mock()
    monkeypatch.setattr(panel, '_start_cuda_install', start)
    panel._update_cuda_status()
    assert ui.tr('cuda_system_installed') in panel.cuda_status_label.text()
    assert panel.cuda_download_btn.isHidden()
    assert panel.cuda_import_btn.isHidden()
    assert panel.cuda_remove_btn.isHidden()
    assert str(system) in panel.cuda_details_label.text()
    panel.on_cuda_download_clicked()
    start.assert_not_called()


@pytest.mark.parametrize('system_runtime_installed', [False, True])
def test_directml_build_explains_cuda_is_unnecessary(settings, monkeypatch, tmp_path, system_runtime_installed):
    panel, ui, _ = settings
    monkeypatch.setattr(runtime, 'cuda_provider_bundled', lambda: False)
    monkeypatch.setattr(ort, 'get_available_providers', lambda: ['DmlExecutionProvider', 'CPUExecutionProvider'])
    monkeypatch.delenv('RAW_ALCHEMY_CPU_ONLY', raising=False)
    session = Mock(side_effect=AssertionError('displaying backend status must not load a model'))
    monkeypatch.setattr(ort, 'InferenceSession', session)
    if system_runtime_installed:
        system = tmp_path / 'system'
        _install_libraries(system)
        monkeypatch.setattr(runtime, '_system_cuda_search_dirs', lambda: [system])
    panel._update_cuda_status()
    assert panel.backend_status_label.text() == ui.tr(
        'inference_backend', backend=ui.tr('inference_backend_gpu', backend='DirectML'),
    )
    assert panel.backend_hint_label.text() == ui.tr('inference_backend_gpu_hint')
    assert panel.cuda_status_label.text() == ui.tr('cuda_not_required')
    assert panel.cuda_download_btn.isHidden()
    assert panel.cuda_import_btn.isHidden()
    assert ui.tr('cuda_provider_missing') in panel.cuda_details_label.text()
    session.assert_not_called()


def test_backend_refresh_respects_cpu_only_configuration(settings, monkeypatch):
    panel, ui, _ = settings
    monkeypatch.setattr(ort, 'get_available_providers', lambda: ['DmlExecutionProvider', 'CPUExecutionProvider'])
    monkeypatch.setenv('RAW_ALCHEMY_CPU_ONLY', '1')
    panel._update_cuda_status()
    assert panel.backend_status_label.text() == ui.tr('inference_backend', backend='CPU')
    assert panel.backend_hint_label.text() == ui.tr('inference_backend_cpu_hint')
    monkeypatch.delenv('RAW_ALCHEMY_CPU_ONLY')
    panel._update_cuda_status()
    assert 'DirectML' in panel.backend_status_label.text()


def test_backend_detection_failure_does_not_claim_cpu_or_hide_cuda_status(settings, monkeypatch):
    panel, ui, _ = settings
    monkeypatch.delenv('RAW_ALCHEMY_CPU_ONLY', raising=False)
    monkeypatch.setattr(ort, 'get_available_providers', Mock(side_effect=RuntimeError('runtime unavailable')))
    monkeypatch.setattr(runtime, 'cuda_provider_bundled', lambda: False)
    panel._update_cuda_status()
    assert panel.backend_status_label.text() == ui.tr('inference_backend_unknown')
    assert panel.backend_hint_label.text() == ''
    assert panel.cuda_status_label.text() == ui.tr('cuda_not_required')


def test_incomplete_toolkit_lists_missing_cudnn(settings):
    panel, ui, _ = settings
    local = runtime.get_cuda_runtime_dir()
    local.mkdir()
    (local / 'cudart64_12.dll').write_bytes(b'fixture')
    panel._update_cuda_status()
    assert ui.tr('cuda_incomplete') in panel.cuda_status_label.text()
    assert 'cudnn64_9.dll' in panel.cuda_details_label.text()
    assert not panel.cuda_import_btn.isHidden()
    assert not panel.cuda_download_btn.isHidden()


def test_import_button_installs_archive_offline_and_refreshes_status(settings, monkeypatch, tmp_path):
    import zipfile

    panel, ui, app = settings
    archive_path = tmp_path / 'runtime.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for pattern, required in runtime.CUDA_LIBS['Windows']:
            if required:
                archive.writestr('runtime/bin/' + pattern.replace('*', '2'), b'fixture')
    monkeypatch.setattr(ui.QFileDialog, 'getOpenFileName', lambda *a: (str(archive_path), ''))
    network = Mock(side_effect=AssertionError('local import must stay offline'))
    monkeypatch.setattr('urllib.request.urlopen', network)
    panel.on_cuda_import_clicked()
    assert not panel.cuda_import_btn.isEnabled()
    deadline = time.monotonic() + 10
    while panel._download_thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert panel._download_thread is None
    assert runtime.is_cuda_runtime_installed()
    assert ui.tr('cuda_installed') in panel.cuda_status_label.text()
    assert panel.cuda_download_btn.isHidden()
    assert panel.cuda_import_btn.isHidden()
    assert panel.cuda_progress.isHidden()
    ui.InfoBar.success.assert_called_once()
    network.assert_not_called()


def test_large_package_progress_does_not_overflow_qt_int(settings):
    panel, ui, _ = settings
    worker = ui.CudaDownloadWorker()
    worker.progress.connect(panel._on_download_progress)
    worker.progress.emit(3 * 1024**3, 4 * 1024**3)
    assert panel.cuda_progress.value() == 75


def test_invalid_import_restores_controls_and_shows_reason(settings, monkeypatch, tmp_path):
    panel, ui, app = settings
    archive_path = tmp_path / 'bad.zip'
    archive_path.write_bytes(b'not a CUDA archive')
    monkeypatch.setattr(ui.QFileDialog, 'getOpenFileName', lambda *a: (str(archive_path), ''))
    panel.on_cuda_import_clicked()
    deadline = time.monotonic() + 10
    while panel._download_thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert panel._download_thread is None
    assert not runtime.is_cuda_runtime_installed()
    assert panel.cuda_import_btn.isEnabled()
    assert panel.cuda_download_btn.isEnabled()
    assert panel.cuda_detect_btn.isEnabled()
    assert panel.cuda_progress.isHidden()
    ui.InfoBar.error.assert_called_once()
    assert ui.InfoBar.error.call_args.kwargs['content']
