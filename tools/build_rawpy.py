"""Build/install rawpy with a pinned, corrected Nikon HE/HE* LibRaw decoder.

Run before installing the application: python tools/build_rawpy.py --install.
Only wheels produced by this command contain the additional decoder; the
ordinary PyPI rawpy wheel remains usable for other RAW formats.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'raw-dependencies.json'


@contextmanager
def build_directory(cache):
    # Retain sources and link metadata for review and native build diagnosis.
    directory = tempfile.mkdtemp(prefix='build-', dir=cache)
    try:
        yield directory
    finally:
        print('Native sources and build output:', directory, flush=True)


def download(asset, cache):
    """Validate cached files as well as fresh downloads before extraction."""
    archive = cache / (asset['sha256'] + '.tar.gz')
    if not archive.is_file():
        with urllib.request.urlopen(asset['url'], timeout=60) as response:
            body = response.read()
        if hashlib.sha256(body).hexdigest() != asset['sha256']:
            raise RuntimeError(f"RAW source checksum mismatch: {asset['url']}")
        archive.write_bytes(body)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != asset['sha256']:
        raise RuntimeError(f'RAW source checksum mismatch: {archive}')
    return archive


def unpack(archive, asset, work):
    with tarfile.open(archive) as bundle:
        bundle.extractall(work, filter='data')
    source = work / asset['source_root']
    if not source.is_dir():
        raise RuntimeError(f'Missing source directory: {source}')
    return source


def run(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def configure_build(source, build, prefix, options, env):
    cmake = shutil.which('cmake')
    if cmake is None:
        raise RuntimeError('Building the HE decoder requires CMake and a C++ compiler')
    args = [cmake, '-S', source, '-B', build,
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_POLICY_VERSION_MINIMUM=3.12',
            '-DCMAKE_INSTALL_PREFIX=' + str(prefix), '-DCMAKE_INSTALL_LIBDIR=lib',
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON']
    if sys.platform == 'win32':
        # VS discovers MSVC without needing a Developer Command Prompt.
        generator = env.get('CMAKE_GENERATOR') or 'Visual Studio 17 2022'
        args += ['-G', generator]
        if generator.startswith('Visual Studio'):
            args += ['-A', 'ARM64' if platform.machine().lower() == 'arm64' else 'x64']
            # Global vcpkg MSBuild integration injects wildcard import libs,
            # which can override our pinned static codecs with unrelated DLLs.
            args += ['-DCMAKE_VS_GLOBALS=VcpkgEnabled=false']
    run(args + options, env=env)
    run([cmake, '--build', build, '--config', 'Release', '--parallel', '2'], env=env)
    run([cmake, '--install', build, '--config', 'Release'], env=env)


def static_library(prefix, *names):
    for name in names:
        path = prefix / 'lib' / name
        if path.is_file():
            return path
    raise RuntimeError(f'Static codec library missing: {names}')


def prepare_wrapper(source, prefix, lock):
    """Use our already built LibRaw, rather than rawpy's bundled upstream copy."""
    setup = source / 'setup.py'
    body = setup.read_text(encoding='utf-8')
    for function in ('windows_libraw_compile', 'unix_libraw_compile'):
        signature = f'def {function}():'
        if body.count(signature) != 1:
            raise RuntimeError('rawpy build hooks changed; review the source pin')
        body = body.replace(signature, signature + '\n    return  # Built by Raw Alchemy before the extension.')
    setup.write_text(body, encoding='utf-8')
    (source / 'rawpy/_version.py').write_text(
        '__version__ = ' + repr(lock['wheel_version']) + '\n', encoding='utf-8',
    )
    (source / 'rawpy/_rawalchemy_build.py').write_text(
        '# Native source provenance; also identifies persistent decode caches.\n'
        'BUILD = ' + repr(lock) + '\n', encoding='utf-8',
    )
    package = source / 'rawpy'
    if sys.platform == 'win32':
        libraries = list((prefix / 'bin').glob('raw_r.dll'))
    else:
        pattern = 'libraw_r*.dylib' if sys.platform == 'darwin' else 'libraw_r.so*'
        libraries = list((prefix / 'lib').glob(pattern))
    if not libraries:
        raise RuntimeError('The enhanced LibRaw shared library was not built')
    for library in libraries:
        shutil.copyfile(library, package / library.name)


def build_wheel(output, cache):
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    archives = {name: download(asset, cache) for name, asset in lock['sources'].items()}
    # All generated sources and compiler output belong to this new directory;
    # no neighbouring rawpy/LibRaw checkouts are modified.
    with build_directory(cache) as directory:
        work = Path(directory)
        sources = {name: unpack(archives[name], asset, work)
                   for name, asset in lock['sources'].items()}
        source = sources['rawpy']
        native = source / 'external/LibRaw-cmake/build'
        prefix = native / 'install'
        env = os.environ.copy()
        env.pop('RAWPY_USE_SYSTEM_LIBRAW', None)
        env['RAWPY_BUILD_GPL_CODE'] = '0'
        configure_build(sources['zlib'], work / 'zlib-build', prefix, [
            '-DZLIB_BUILD_TESTING=OFF', '-DZLIB_BUILD_SHARED=OFF',
        ], env)
        configure_build(sources['jpeg'], work / 'jpeg-build', prefix, [
            '-DENABLE_SHARED=OFF', '-DENABLE_STATIC=ON', '-DWITH_TURBOJPEG=OFF',
            '-DWITH_TOOLS=OFF', '-DWITH_TESTS=OFF', '-DWITH_SIMD=OFF',
            '-DWITH_JPEG8=ON', '-DWITH_CRT_DLL=ON',
        ], env)
        zlib = static_library(prefix, 'zlibstatic.lib', 'zs.lib', 'libz.a', 'libzs.a')
        jpeg = static_library(prefix, 'jpeg-static.lib', 'libjpeg.a')
        options = [
            '-DLIBRAW_PATH=' + str(sources['libraw']), '-DBUILD_SHARED_LIBS=ON',
            '-DENABLE_EXAMPLES=OFF', '-DENABLE_RAWSPEED=OFF',
            '-DENABLE_X3FTOOLS=ON', '-DENABLE_6BY9RPI=ON',
            '-DENABLE_OPENMP=' + ('ON' if sys.platform.startswith('linux') else 'OFF'),
            '-DZLIB_LIBRARY=' + str(zlib), '-DZLIB_INCLUDE_DIR=' + str(prefix / 'include'),
            '-DZLIB_LIBRARY_RELEASE=' + str(zlib), '-DZLIB_LIBRARY_DEBUG=' + str(zlib),
            '-DZLIB_USE_STATIC_LIBS=ON',
            '-DJPEG_LIBRARY=' + str(jpeg), '-DJPEG_INCLUDE_DIR=' + str(prefix / 'include'),
            '-DJPEG_LIBRARY_RELEASE=' + str(jpeg), '-DJPEG_LIBRARY_DEBUG=' + str(jpeg),
        ]
        if sys.platform == 'darwin':
            options += ['-DCMAKE_INSTALL_NAME_DIR=@rpath']
        configure_build(source / 'external/LibRaw-cmake', native, prefix, options, env)
        config = (prefix / 'include/libraw/libraw_config.h').read_text(encoding='utf-8')
        for flag in ('LIBRAW_USE_DNGDEFLATECODEC', 'LIBRAW_USE_DNGLOSSYCODEC'):
            if '#define ' + flag + ' 1' not in config:
                raise RuntimeError('The native build lost codec support: ' + flag)
        prepare_wrapper(source, prefix, lock)
        # Carry the shared decoder and statically linked codec notices.
        notices = [sources['libraw'] / name for name in ('COPYRIGHT', 'LICENSE.LGPL', 'REDLAMP.md')]
        notices += [sources['zlib'] / 'LICENSE', sources['jpeg'] / 'LICENSE.md',
                    sources['jpeg'] / 'README.ijg']
        (source / 'LICENSE.LibRaw').write_text('\n\n'.join(
            path.read_text(encoding='utf-8') for path in notices
        ), encoding='utf-8')
        run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
             '--wheel-dir', output, source], env=env)
    wheels = list(output.glob('rawpy-' + lock['wheel_version'] + '-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(f'Expected one enhanced rawpy wheel, found {len(wheels)}')
    # Check the actual wheel before replacing an existing working decoder.
    with tempfile.TemporaryDirectory(prefix='rawalchemy-he-probe-') as directory:
        with zipfile.ZipFile(wheels[0]) as bundle:
            bundle.extractall(directory)
        run([sys.executable, '-I', '-c',
             'import sys; sys.path.insert(0, ' + repr(directory) + '); ' + installed_probe(lock)])
    return wheels[0]


def installed_probe(expected):
    return (
        'import rawpy; from rawpy._rawalchemy_build import BUILD; '
        f'assert rawpy.__version__ == {expected["wheel_version"]!r}; '
        f'assert BUILD == {expected!r}; '
        'assert rawpy.flags["DNGDEFLATECODEC"]; '
        'assert rawpy.flags["DNGLOSSYCODEC"]; '
        'print("Enhanced rawpy", rawpy.__version__, "LibRaw", rawpy.libraw_version, "HE/HE* build verified")'
    )


def verify_installed():
    """Verify in a fresh interpreter, including native codec feature flags."""
    expected = json.loads(LOCK.read_text(encoding='utf-8'))
    run([sys.executable, '-I', '-c', installed_probe(expected)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='Install the newly built wheel')
    parser.add_argument('--verify', action='store_true', help='Check the installed build without compiling')
    parser.add_argument('--output', type=Path, default=ROOT / '.test-output/rawpy-wheels')
    parser.add_argument('--cache', type=Path, default=ROOT / '.test-output/rawpy-downloads')
    args = parser.parse_args()
    if args.verify:
        verify_installed()
        return
    wheel = build_wheel(args.output.resolve(), args.cache.resolve())
    print('Built:', wheel, flush=True)
    if args.install:
        run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', wheel])
        verify_installed()


if __name__ == '__main__':
    main()
