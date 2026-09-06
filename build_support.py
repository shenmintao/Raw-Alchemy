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
    if not from_source and build_vendor.is_dir() and tree_digest(build_vendor) == asset['tree_sha256']:
        return
    with tempfile.TemporaryDirectory(prefix='rawalchemy-lensfun-') as work:
        archive = Path(work) / asset['archive']
        local = os.environ.get('RAWALCHEMY_LENSFUN_ARCHIVE')
        if local:
            shutil.copyfile(local, archive)
        else:
            url = asset.get('url') or f"https://github.com/shenmintao/lensfun/releases/download/{release}/{asset['archive']}"
            with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as output:
                shutil.copyfileobj(response, output)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != asset['sha256']:
            raise RuntimeError('Lensfun archive checksum mismatch')
        extracted = Path(work) / 'extracted'
        extracted.mkdir()
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
        else:
            with tarfile.open(archive) as bundle:
                bundle.extractall(extracted, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
        verified = extracted / asset['source_root'] if from_source else extracted
        if tree_digest(verified) != asset['tree_sha256']:
            raise RuntimeError('Lensfun extracted files do not match the lock')
        if from_source:
            extracted = _build_lensfun(verified, Path(work))
        if build_vendor.exists():
            shutil.rmtree(build_vendor)
        shutil.copytree(extracted, build_vendor, symlinks=True)


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
