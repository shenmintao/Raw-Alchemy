"""Verified Lensfun assets for generated build output, never the source tree."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile


LOCK_PATH = Path(__file__).with_name('native-dependencies.json')


def asset_for(system=None, machine=None):
    lock = json.loads(LOCK_PATH.read_text())
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    asset = lock['platforms'].get(system)
    if asset is None or machine not in asset['architectures']:
        raise RuntimeError(f'No verified Lensfun binary for {system}/{machine}')
    if lock.get('database'):
        asset = dict(asset, database=lock['database'])
    return lock['release'], asset


def tree_digest(root):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*') if p.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def ensure_lensfun(build_vendor):
    """Replace only generated build output after complete archive validation."""
    release, asset = asset_for()
    build_vendor = Path(build_vendor)
    from_source = asset.get('build') == 'cmake'
    expected = asset.get('installed_tree_sha256', asset['tree_sha256'])
    if not from_source and build_vendor.is_dir() and tree_digest(build_vendor) == expected:
        return
    with tempfile.TemporaryDirectory(prefix='rawalchemy-lensfun-') as work:
        verified = _verified_archive(release, asset, Path(work) / 'runtime')
        database = asset.get('database')
        database_source = None
        if database:
            if from_source and asset['sha256'] == database['sha256']:
                database_source = verified
            else:
                database_source = _verified_archive(
                    release, database, Path(work) / 'database',
                    local_env='RAWALCHEMY_LENSFUN_SOURCE_ARCHIVE',
                )
        extracted = verified
        if from_source:
            extracted = _build_lensfun(verified, Path(work))
        if database:
            _install_database(extracted, database_source, database)
            if not from_source and tree_digest(extracted) != expected:
                raise RuntimeError('Lensfun runtime/database files do not match the lock')
        if build_vendor.exists():
            shutil.rmtree(build_vendor)
        shutil.copytree(extracted, build_vendor, symlinks=True)


def _verified_archive(release, asset, work, *, local_env='RAWALCHEMY_LENSFUN_ARCHIVE'):
    """Verify an entire native/source archive before any build or installation."""
    work.mkdir(parents=True)
    archive = work / asset['archive']
    local = os.environ.get(local_env)
    if local:
        shutil.copyfile(local, archive)
    else:
        url = asset.get('url') or f"https://github.com/shenmintao/lensfun/releases/download/{release}/{asset['archive']}"
        with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != asset['sha256']:
        raise RuntimeError('Lensfun archive checksum mismatch')
    extracted = work / 'extracted'
    extracted.mkdir()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(extracted, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
    verified = extracted / asset.get('source_root', '')
    if tree_digest(verified) != asset['tree_sha256']:
        raise RuntimeError('Lensfun extracted files do not match the lock')
    return verified


def _install_database(runtime, source, asset):
    """Use one verified development database with every supported native ABI."""
    source_db = source / 'data/db'
    if not any(source_db.glob('*.xml')):
        raise RuntimeError('Lensfun source is missing its camera database')
    destination = runtime / 'share/lensfun/version_2'
    destination.mkdir(parents=True, exist_ok=True)
    for old in destination.iterdir():
        if not old.is_file() or old.is_symlink():
            raise RuntimeError('Unexpected directory or link in Lensfun database')
        old.unlink()
    for item in source_db.iterdir():
        if item.is_file():
            shutil.copyfile(item, destination / item.name)
    shutil.copyfile(source / 'data/COPYING.CC_BY-SA_3.0', destination.parent / 'COPYING.CC_BY-SA_3.0')
    provenance = dict(source_commit=asset['commit'], source_url=asset['url'],
                      database_tree_sha256=tree_digest(source_db))
    (destination.parent / 'rawalchemy-database.json').write_text(
        json.dumps(provenance, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n',
    )


def _build_lensfun(source, work):
    """Build against the host ABI; never ship a newer distro's glibc by accident.

    The source tree is verified before CMake executes. Every source build is
    rebuilt with the current toolchain; a generated runtime is not a binary lock.
    """
    import ctypes
    cmake = shutil.which('cmake')
    if cmake is None:
        raise RuntimeError('Linux Lensfun requires CMake, a C++17 compiler, pkg-config and GLib development headers')
    output, build = work / 'installed', work / 'cmake-build'
    subprocess.run([
        cmake, '-S', str(source), '-B', str(build),
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_INSTALL_PREFIX=' + str(output),
        '-DCMAKE_INSTALL_LIBDIR=lib', '-DBUILD_TESTS=OFF', '-DBUILD_DOC=OFF',
        '-DBUILD_LENSTOOL=OFF', '-DINSTALL_PYTHON_MODULE=OFF',
        '-DINSTALL_HELPER_SCRIPTS=OFF',
    ], check=True)
    subprocess.run([cmake, '--build', str(build), '--parallel', '2'], check=True)
    subprocess.run([cmake, '--install', str(build)], check=True)
    runtime = output / 'lib/liblensfun.so'
    database = output / 'share/lensfun/version_2'
    if not runtime.is_file() or not any(database.glob('*.xml')):
        raise RuntimeError('Lensfun build is missing its runtime or camera database')
    # A successful link does not prove the packaged library can load locally.
    ctypes.CDLL(str(runtime))
    return output


def pyinstaller_vendor_datas(source_vendor, build_dir):
    """Bundle the same verified runtime as wheels, ignoring local Lensfun files."""
    verified = Path(build_dir) / 'lensfun'
    ensure_lensfun(verified)
    datas = []
    for item in sorted(Path(source_vendor).iterdir()):
        if item.name == 'lensfun':
            continue
        destination = str(Path('vendor') / item.name) if item.is_dir() else 'vendor'
        datas.append((str(item), destination))
    return datas + [(str(verified), 'vendor/lensfun')]
