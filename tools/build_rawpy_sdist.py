"""Create the pinned HE-enabled rawpy source dependency consumed by pip.

All four upstream source archives are included after checksum validation.
Building this sdist uses pip's normal build isolation and installs rawpy in
the destination environment; no post-install hooks or nested pip installs.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import build_rawpy as native


ROOT = Path(__file__).resolve().parents[1]


def create_sdist(output, cache):
    lock = json.loads(native.LOCK.read_text(encoding='utf-8'))
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    archives = {name: native.download(asset, cache) for name, asset in lock['sources'].items()}
    name = 'rawpy-' + lock['wheel_version']
    target = output / (name + '.tar.gz')
    with tempfile.TemporaryDirectory(prefix='rawalchemy-rawpy-sdist-', dir=cache) as directory:
        work = Path(directory)
        sources = {key: native.unpack(archives[key], asset, work)
                   for key, asset in lock['sources'].items()}
        source = sources['rawpy']
        native.prepare_wrapper(source, None, lock, build_from_source=True)
        # The helper remains at the sdist root so its source lock is local.
        helper = Path(native.__file__).read_text(encoding='utf-8').replace(
            'ROOT = Path(__file__).resolve().parents[1]',
            'ROOT = Path(__file__).resolve().parent',
        )
        (source / 'rawalchemy_native.py').write_text(helper, encoding='utf-8')
        (source / 'raw-dependencies.json').write_text(json.dumps(lock, indent=2) + '\n', encoding='utf-8')
        bundled = source / 'external/rawalchemy-sources'
        bundled.mkdir(parents=True)
        for archive in archives.values():
            shutil.copyfile(archive, bundled / archive.name)
        notices = [sources['libraw'] / f for f in ('COPYRIGHT', 'LICENSE.LGPL', 'REDLAMP.md')]
        notices += [sources['zlib'] / 'LICENSE', sources['jpeg'] / 'LICENSE.md',
                    sources['jpeg'] / 'README.ijg']
        (source / 'LICENSE.LibRaw').write_text(
            '\n\n'.join(path.read_text(encoding='utf-8') for path in notices), encoding='utf-8',
        )
        with (source / 'MANIFEST.in').open('a', encoding='utf-8') as manifest:
            manifest.write('\ninclude rawalchemy_native.py raw-dependencies.json\n'
                           'recursive-include external/rawalchemy-sources *.tar.gz\n')
        # Remove metadata from the upstream version; setuptools regenerates
        # it for the enhanced version. Archive timestamps/owners are stable.
        for metadata in source.glob('*.egg-info'):
            shutil.rmtree(metadata)
        with target.open('wb') as stream, gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as bundle:
                for path in sorted(source.rglob('*')):
                    if not path.is_file():
                        continue
                    info = bundle.gettarinfo(str(path), arcname=str(Path(name) / path.relative_to(source)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    info.mode = 0o644
                    info.pax_headers = {}
                    with path.open('rb') as content:
                        bundle.addfile(info, content)
    print('Source dependency:', target)
    print('SHA256:', hashlib.sha256(target.read_bytes()).hexdigest())
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.test-output/rawpy-sdists')
    parser.add_argument('--cache', type=Path, default=ROOT / '.test-output/rawpy-downloads')
    args = parser.parse_args()
    create_sdist(args.output.resolve(), args.cache.resolve())


if __name__ == '__main__':
    main()
