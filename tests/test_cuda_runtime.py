"""Offline CUDA discovery and transactional imports do not require an NVIDIA GPU."""
import io
import tarfile
import zipfile
from unittest.mock import Mock

import pytest

from raw_alchemy.onnx import gpu_runtime as runtime


def _libraries(system='Windows'):
    return {
        pattern.replace('*', '2'): b'library fixture'
        for pattern, required in runtime.CUDA_LIBS[system] if required
    }


def _write_libraries(directory, libraries):
    directory.mkdir(parents=True, exist_ok=True)
    for name, contents in libraries.items():
        (directory / name).write_bytes(contents)


def _zip(path, libraries, prefix=''):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, contents in libraries.items():
            archive.writestr(prefix + name, contents)
    return path


@pytest.fixture
def cuda_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(runtime, '_system_cuda_search_dirs', lambda: [])
    local = tmp_path / 'managed' / 'cuda_runtime'
    monkeypatch.setattr(runtime, 'get_cuda_runtime_dir', lambda: local)
    return local


def test_separate_toolkit_and_cudnn_are_detected_and_loaded(monkeypatch, tmp_path, cuda_environment):
    toolkit, cudnn = tmp_path / 'toolkit', tmp_path / 'cudnn'
    libraries = _libraries()
    _write_libraries(toolkit, {n: c for n, c in libraries.items() if not n.startswith('cudnn')})
    _write_libraries(cudnn, {n: c for n, c in libraries.items() if n.startswith('cudnn')})
    monkeypatch.setattr(runtime, '_system_cuda_search_dirs', lambda: [toolkit, cudnn])
    status = runtime.detect_cuda_runtime()
    assert status['installed'] and status['source'] == 'system'
    assert set(status['library_dirs']) == {str(toolkit), str(cudnn)}
    assert not status['local_present']
    activate = Mock(return_value=True)
    monkeypatch.setattr(runtime, '_activate_cuda_dirs', activate)
    assert runtime.setup_cuda_dll_paths()
    assert set(activate.call_args.args[0]) == {toolkit, cudnn}


def test_toolkit_without_cudnn_reports_missing_components(monkeypatch, tmp_path, cuda_environment):
    toolkit = tmp_path / 'toolkit'
    _write_libraries(toolkit, {n: c for n, c in _libraries().items() if not n.startswith('cudnn')})
    monkeypatch.setattr(runtime, '_system_cuda_search_dirs', lambda: [toolkit])
    status = runtime.detect_cuda_runtime()
    assert status['detected'] and not status['installed']
    assert 'cudnn64_9.dll' in status['missing_libraries']
    assert not runtime.setup_cuda_dll_paths()


def test_unversioned_manual_install_is_detected(cuda_environment):
    _write_libraries(cuda_environment, _libraries())
    assert runtime.is_cuda_runtime_installed()
    assert runtime.detect_cuda_runtime()['source'] == 'local'
    assert not runtime.is_cuda_update_available()


def test_directory_named_like_a_dll_does_not_count_as_installed(cuda_environment):
    _write_libraries(cuda_environment, _libraries())
    library = cuda_environment / 'cudnn64_9.dll'
    library.unlink()
    library.mkdir()
    assert not runtime.is_cuda_runtime_installed()


@pytest.mark.parametrize('nested', ['', 'cuda-runtime-windows-x64/bin/'])
def test_zip_import_flattens_libraries_and_reports_progress(tmp_path, cuda_environment, nested):
    archive = _zip(tmp_path / 'runtime.zip', _libraries(), nested)
    progress = Mock()
    assert runtime.import_cuda_runtime(archive, progress)
    assert runtime.detect_cuda_runtime()['local_installed']
    assert {p.name for p in cuda_environment.iterdir()} == set(_libraries())
    assert progress.call_args.args[0] == progress.call_args.args[1] > 0
    assert archive.exists()
    assert runtime.get_installed_cuda_version() is None


def test_linux_tar_import_materializes_library_links(monkeypatch, tmp_path, cuda_environment):
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    archive = tmp_path / 'runtime.tar.gz'
    libraries = _libraries('Linux')
    with tarfile.open(archive, 'w:gz') as package:
        for name, contents in libraries.items():
            info = tarfile.TarInfo('runtime/lib/' + name)
            info.size = len(contents)
            package.addfile(info, io.BytesIO(contents))
        link = tarfile.TarInfo('runtime/lib/libcudart.so')
        link.type = tarfile.SYMTYPE
        link.linkname = 'libcudart.so.12'
        package.addfile(link)
    assert runtime.import_cuda_runtime(archive)
    assert runtime.is_cuda_runtime_installed()
    assert (cuda_environment / 'libcudart.so').read_bytes() == libraries['libcudart.so.12']
    assert not (cuda_environment / 'libcudart.so').is_symlink()


@pytest.mark.parametrize('failure', ['incomplete', 'cancelled', 'unsafe', 'duplicate', 'corrupt'])
def test_failed_import_preserves_existing_runtime(tmp_path, cuda_environment, failure):
    _write_libraries(cuda_environment, {'keep.dll': b'original', 'version.txt': b'previous'})
    archive = tmp_path / 'runtime.zip'
    libraries = _libraries()
    if failure == 'incomplete':
        del libraries['cudnn64_9.dll']
    elif failure == 'unsafe':
        libraries['../outside.dll'] = b'bad path'
    elif failure == 'duplicate':
        libraries['other/cudnn64_9.dll'] = b'duplicate'
    _zip(archive, libraries)
    if failure == 'corrupt':
        archive.write_bytes(b'not an archive')
    if failure == 'cancelled':
        # Cancel after copying has started, not just before opening the archive.
        checks = iter([False, False, True])
        assert not runtime.import_cuda_runtime(archive, cancel_callback=lambda: next(checks, True))
    else:
        with pytest.raises((ValueError, tarfile.ReadError)):
            runtime.import_cuda_runtime(archive)
    assert {p.name: p.read_bytes() for p in cuda_environment.iterdir()} == {
        'keep.dll': b'original', 'version.txt': b'previous',
    }
    assert not (tmp_path / 'outside.dll').exists()
    assert not list(cuda_environment.parent.glob('.cuda-import-*'))


def test_download_is_skipped_when_system_cuda_is_complete(monkeypatch, tmp_path, cuda_environment):
    system = tmp_path / 'system'
    _write_libraries(system, _libraries())
    monkeypatch.setattr(runtime, '_system_cuda_search_dirs', lambda: [system])
    network = Mock(side_effect=AssertionError('already installed; must not download'))
    monkeypatch.setattr('urllib.request.urlopen', network)
    assert runtime.download_cuda_runtime()
    network.assert_not_called()
    assert not cuda_environment.exists()


@pytest.mark.parametrize('complete', [True, False])
def test_download_validates_before_reporting_success(monkeypatch, tmp_path, cuda_environment, complete):
    libraries = _libraries()
    if not complete:
        del libraries['cudnn64_9.dll']
    archive = _zip(tmp_path / 'download.zip', libraries)
    response = io.BytesIO(archive.read_bytes())
    response.headers = {'content-length': str(archive.stat().st_size)}
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: response)
    assert runtime.download_cuda_runtime() is complete
    assert runtime.is_cuda_runtime_installed() is complete
    assert runtime.get_installed_cuda_version() == (runtime.CUDA_RUNTIME_VERSION if complete else None)
    assert not list(cuda_environment.parent.glob('cuda-download-*'))


def test_search_finds_cuda_path_and_separate_cudnn_on_path(monkeypatch, tmp_path):
    import site

    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Windows')
    local = tmp_path / 'managed'
    toolkit, cudnn = tmp_path / 'CUDA' / 'bin', tmp_path / 'cuDNN'
    _write_libraries(toolkit, {n: c for n, c in _libraries().items() if not n.startswith('cudnn')})
    _write_libraries(cudnn, {n: c for n, c in _libraries().items() if n.startswith('cudnn')})
    monkeypatch.setattr(runtime, 'get_cuda_runtime_dir', lambda: local)
    for variable in ('CUDA_HOME', 'CUDNN_HOME', 'CUDNN_PATH'):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv('CUDA_PATH', str(toolkit.parent))
    monkeypatch.setenv('PATH', str(cudnn))
    monkeypatch.setenv('ProgramFiles', str(tmp_path / 'Program Files'))
    monkeypatch.setattr(site, 'getsitepackages', lambda: [])
    monkeypatch.setattr(site, 'getusersitepackages', lambda: str(tmp_path / 'user-site'))
    assert set(runtime.detect_cuda_runtime()['library_dirs']) == {str(toolkit), str(cudnn)}
