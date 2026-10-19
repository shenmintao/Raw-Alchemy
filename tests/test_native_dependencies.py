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
    database = asset['database']
    assert len(database['commit']) == 40
    assert len(database['sha256']) == len(database['tree_sha256']) == 64
    if asset.get('build') == 'cmake':
        assert asset['commit'] == database['commit']
        assert asset['sha256'] == database['sha256']
    else:
        assert len(asset['installed_tree_sha256']) == 64


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


@pytest.fixture
def database_archives(tmp_path, monkeypatch):
    import hashlib
    import json
    import zipfile

    runtime = tmp_path / 'runtime'
    (runtime / 'lib').mkdir(parents=True)
    (runtime / 'lib/lensfun.dll').write_bytes(b'verified runtime')
    old_db = runtime / 'share/lensfun/version_2'
    old_db.mkdir(parents=True)
    (old_db / 'stale.xml').write_bytes(b'<lensdatabase/>')
    source = tmp_path / 'source'
    source_db = source / 'data/db'
    source_db.mkdir(parents=True)
    (source_db / 'new.xml').write_bytes(b'<lensdatabase version="2"><camera/></lensdatabase>')
    (source_db / 'timestamp.txt').write_bytes(b'1234567890\n')
    (source / 'data/COPYING.CC_BY-SA_3.0').write_bytes(b'database licence fixture')

    def archive(root, name, prefix=''):
        path = tmp_path / name
        with zipfile.ZipFile(path, 'w') as bundle:
            for item in root.rglob('*'):
                if item.is_file():
                    bundle.write(item, prefix + item.relative_to(root).as_posix())
        return dict(archive=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    tree_sha256=build.tree_digest(root))

    asset = archive(runtime, 'runtime.zip')
    database = archive(source, 'source.zip', 'source/')
    database.update(source_root='source', commit='f' * 40, url='https://example.invalid/pinned')
    asset['database'] = database
    provenance = dict(source_commit=database['commit'], source_url=database['url'],
                      database_tree_sha256=build.tree_digest(source_db))
    expected = tmp_path / 'expected'
    expected_files = {
        'lib/lensfun.dll': b'verified runtime',
        'share/lensfun/version_2/new.xml': (source_db / 'new.xml').read_bytes(),
        'share/lensfun/version_2/timestamp.txt': b'1234567890\n',
        'share/lensfun/COPYING.CC_BY-SA_3.0': b'database licence fixture',
        'share/lensfun/rawalchemy-database.json':
            (json.dumps(provenance, sort_keys=True, separators=(',', ':')) + '\n').encode(),
    }
    for name, content in expected_files.items():
        target = expected / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    asset['installed_tree_sha256'] = build.tree_digest(expected)
    monkeypatch.setattr(build, 'asset_for', lambda: ('pinned', asset))
    monkeypatch.setenv('RAWALCHEMY_LENSFUN_ARCHIVE', str(tmp_path / 'runtime.zip'))
    monkeypatch.setenv('RAWALCHEMY_LENSFUN_SOURCE_ARCHIVE', str(tmp_path / 'source.zip'))
    return asset, expected


def test_development_database_replaces_stale_profiles_and_preserves_runtime(tmp_path, monkeypatch, database_archives):
    asset, expected = database_archives
    output = tmp_path / 'installed'
    build.ensure_lensfun(output)
    assert build.tree_digest(output) == build.tree_digest(expected)
    assert (output / 'lib/lensfun.dll').read_bytes() == b'verified runtime'
    assert not (output / 'share/lensfun/version_2/stale.xml').exists()
    # A complete runtime/database match can be reused without another download.
    def unexpected_download(*args, **kwargs):
        pytest.fail('verified installed files should be reused')
    monkeypatch.setattr(build, '_verified_archive', unexpected_download)
    build.ensure_lensfun(output)
    assert build.tree_digest(output) == asset['installed_tree_sha256']


@pytest.mark.parametrize('corrupt', ['source_checksum', 'installed_checksum'])
def test_database_verification_failure_preserves_previous_runtime(tmp_path, database_archives, corrupt):
    asset, _ = database_archives
    if corrupt == 'source_checksum':
        asset['database']['sha256'] = '0' * 64
    else:
        asset['installed_tree_sha256'] = '0' * 64
    output = tmp_path / 'installed'
    output.mkdir()
    (output / 'working-runtime').write_bytes(b'previous working installation')
    before = build.tree_digest(output)
    with pytest.raises(RuntimeError, match='checksum|do not match'):
        build.ensure_lensfun(output)
    assert build.tree_digest(output) == before
