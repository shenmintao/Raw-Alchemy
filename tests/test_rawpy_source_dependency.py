"""The pip source dependency is reproducible and defers native compilation."""
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


@pytest.fixture
def sdist_builder(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[1] / 'tools'
    monkeypatch.syspath_prepend(str(tools))
    builder = importlib.import_module('build_rawpy_sdist')
    files = {
        'rawpy': {
            'setup.py': 'def windows_libraw_compile():\n    raise RuntimeError("upstream")\n'
                        'def unix_libraw_compile():\n    raise RuntimeError("upstream")\n',
            'rawpy/__init__.py': '', 'rawpy/_version.py': '__version__="upstream"\n',
            'MANIFEST.in': '', 'rawpy.egg-info/PKG-INFO': 'Version: upstream\n',
        },
        'libraw': {name: name for name in ('COPYRIGHT', 'LICENSE.LGPL', 'REDLAMP.md')},
        'zlib': {'LICENSE': 'zlib licence'},
        'jpeg': {'LICENSE.md': 'libjpeg-turbo licence', 'README.ijg': 'IJG licence'},
    }
    lock = dict(wheel_version='0.27.1+rawalchemy.he1', decoder_revision='fixed', sources={})
    for name, content in files.items():
        archive = tmp_path / (name + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as bundle:
            for filename, body in content.items():
                data = body.encode()
                info = tarfile.TarInfo(name + '/' + filename)
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
        lock['sources'][name] = dict(url=archive.as_uri(), source_root=name,
                                    sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    pin = tmp_path / 'raw-dependencies.json'
    pin.write_text(json.dumps(lock))
    monkeypatch.setattr(builder.native, 'LOCK', pin)
    return builder, lock


def test_sdist_bundles_all_checked_sources_and_is_reproducible(sdist_builder, tmp_path):
    builder, lock = sdist_builder
    archive = builder.create_sdist(tmp_path / 'out', tmp_path / 'cache')
    expected = archive.read_bytes()
    builder.create_sdist(tmp_path / 'out', tmp_path / 'cache')
    assert archive.read_bytes() == expected
    prefix = 'rawpy-' + lock['wheel_version'] + '/'
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert not any('.egg-info' in name for name in names)
        assert prefix + 'rawalchemy_native.py' in names
        for asset in lock['sources'].values():
            payload = bundle.extractfile(prefix + 'external/rawalchemy-sources/' + asset['sha256'] + '.tar.gz').read()
            assert hashlib.sha256(payload).hexdigest() == asset['sha256']
        assert json.loads(bundle.extractfile(prefix + 'raw-dependencies.json').read()) == lock


@pytest.mark.parametrize('hook', ['windows_libraw_compile', 'unix_libraw_compile'])
def test_metadata_defers_build_and_native_hook_loads_outside_cwd(sdist_builder, tmp_path, hook):
    builder, _ = sdist_builder
    archive = builder.create_sdist(tmp_path / 'out', tmp_path / 'cache')
    unpacked = tmp_path / 'unpacked'
    with tarfile.open(archive) as bundle:
        bundle.extractall(unpacked, filter='data')
    source = next(unpacked.iterdir())
    setup = source / 'setup.py'
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    # Metadata evaluation must not start compilers or install into a build env.
    subprocess.run([sys.executable, '-I', '-c',
                    f'import runpy,sys; runpy.run_path({str(setup)!r}); '
                    'assert "rawalchemy_native" not in sys.modules'],
                   cwd=unrelated, check=True)
    (source / 'rawalchemy_native.py').write_text('def build_bundled():\n    return "native-hook-called"\n')
    result = subprocess.check_output([sys.executable, '-I', '-c',
                                     f'import runpy; ns=runpy.run_path({str(setup)!r}); '
                                     f'print(ns[{hook!r}]())'], cwd=unrelated, text=True)
    assert result.strip() == 'native-hook-called'
