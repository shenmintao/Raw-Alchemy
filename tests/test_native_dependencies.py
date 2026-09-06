"""Pinned binary architecture and complete-file verification are build contracts."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'build_support.py'
spec = importlib.util.spec_from_file_location('build_support', path)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


@pytest.mark.parametrize('system,machine', [('Windows', 'AMD64'), ('Linux', 'x86_64'), ('Darwin', 'arm64')])
def test_supported_binaries_are_pinned_and_verified(system, machine):
    release, asset = build.asset_for(system, machine)
    assert release != 'latest'
    assert len(asset['sha256']) == len(asset['tree_sha256']) == 64


@pytest.mark.parametrize('system,machine', [('Windows', 'ARM64'), ('Linux', 'aarch64'), ('Darwin', 'x86_64')])
def test_unavailable_architecture_does_not_package_wrong_binary(system, machine):
    with pytest.raises(RuntimeError, match='No verified Lensfun'):
        build.asset_for(system, machine)


def test_local_archive_corruption_preserves_generated_runtime(tmp_path, monkeypatch):
    archive = tmp_path / 'lensfun.zip'
    archive.write_bytes(b'corrupt download')
    vendor = tmp_path / 'vendor'
    vendor.mkdir()
    previous = vendor / 'runtime'
    previous.write_bytes(b'previous')
    monkeypatch.setenv('RAWALCHEMY_LENSFUN_ARCHIVE', str(archive))
    monkeypatch.setattr(build, 'asset_for', lambda: ('pinned', {'archive': 'lensfun.zip', 'sha256': '0'*64, 'tree_sha256': '0'*64}))
    with pytest.raises(RuntimeError, match='checksum'):
        build.ensure_lensfun(vendor)
    assert previous.read_bytes() == b'previous'


def test_frozen_bundle_excludes_unverified_local_lensfun(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'lensfun').mkdir(parents=True)
    (source / 'lensfun' / 'wrong-platform.dll').write_bytes(b'stale')
    (source / 'model.onnx').write_bytes(b'model')
    nested = source / 'other'
    nested.mkdir()
    (nested / 'asset').write_bytes(b'asset')
    generated = tmp_path / 'build'
    def prepare(path):
        path.mkdir(parents=True)
        (path / 'verified-runtime').write_bytes(b'verified')
    monkeypatch.setattr(build, 'ensure_lensfun', prepare)
    datas = build.pyinstaller_vendor_datas(source, generated)
    assert (str(source / 'model.onnx'), 'vendor') in datas
    assert (str(nested), str(Path('vendor') / 'other')) in datas
    assert (str(generated / 'lensfun'), 'vendor/lensfun') in datas
    assert not any(Path(path) == source / 'lensfun' for path, _ in datas)
    assert (source / 'lensfun' / 'wrong-platform.dll').read_bytes() == b'stale'


def test_failed_source_build_preserves_previous_runtime(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    import zipfile
    source = tmp_path / 'locked-source'
    source.mkdir()
    (source / 'CMakeLists.txt').write_text('project(locked)')
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.write(source / 'CMakeLists.txt', 'source/CMakeLists.txt')
    asset = dict(build='cmake', archive='source.zip', source_root='source',
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 tree_sha256=build.tree_digest(source))
    monkeypatch.setattr(build, 'asset_for', lambda: ('pinned', asset))
    monkeypatch.setenv('RAWALCHEMY_LENSFUN_ARCHIVE', str(archive))
    vendor = tmp_path / 'vendor'
    vendor.mkdir()
    (vendor / 'previous').write_bytes(b'working')
    def failed_compile(verified, work):
        assert build.tree_digest(verified) == asset['tree_sha256']
        raise subprocess.CalledProcessError(1, 'cmake')
    monkeypatch.setattr(build, '_build_lensfun', failed_compile)
    with pytest.raises(subprocess.CalledProcessError):
        build.ensure_lensfun(vendor)
    assert (vendor / 'previous').read_bytes() == b'working'


def test_built_runtime_must_load_before_publication(tmp_path, monkeypatch):
    import ctypes
    output = tmp_path / 'installed'
    (output / 'lib').mkdir(parents=True)
    (output / 'lib/liblensfun.so').write_bytes(b'incompatible runtime')
    db = output / 'share/lensfun/version_2'
    db.mkdir(parents=True)
    (db / 'camera.xml').write_text('<lensdatabase/>')
    monkeypatch.setattr(build.shutil, 'which', lambda name: 'cmake')
    monkeypatch.setattr(build.subprocess, 'run', lambda *a, **kw: None)
    def incompatible(path):
        raise OSError('GLIBC version unavailable')
    monkeypatch.setattr(ctypes, 'CDLL', incompatible)
    with pytest.raises(OSError, match='GLIBC'):
        build._build_lensfun(tmp_path / 'source', tmp_path)
