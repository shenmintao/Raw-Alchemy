"""Check real HE/HE* mosaics against the pinned decoder's reference hashes.

Samples are CC0 images from raw.pixls.us, recorded by the decoder's upstream
bench at 3b20539d3faa766a370e93a529a3b87edffbf0f6. Downloads are opt-in and are
kept outside version control. Missing samples fail instead of silently skipping.
"""
import argparse
import hashlib
from pathlib import Path
import urllib.request

import numpy as np
import rawpy


SAMPLES = [
    ('Z8_3x2_HE_82df0415', 6616, 'Z%208', '3:2',
     '82df041542b8d328738443bb5ff3373b396e406ecc8cbe7dc3b35252790c53e7',
     'a273a25f3b3f99be4c4ded39a8c8646464b92d29e98ceffcb47a4e0dd7b46ae1', (5520, 8280)),
    ('Z8_3x2_HEstar_ea8b093c', 6618, 'Z%208', '3:2',
     'ea8b093cd11c946c5eeb501b775859e8e617ce1d79577abc41b233c339e4004c',
     'e4c31118fad3aa09ecf221fdaf1e9e3cb5c9a58e9a6859b902af8ac704bc8ab4', (5520, 8280)),
    ('Z50_2_1x1_HE_99b2a569', 7767, 'Z50_2', '1:1',
     '99b2a569356fb084d6946fa6e0b501b014e6e004be0a8e8d564a7b9dab91857e',
     'da094caa3aca1da0920e4134a76f761445423090bcc9a965f8fb9032a077a56f', (3728, 3728)),
    ('Z50_2_1x1_HEstar_605f903e', 7766, 'Z50_2', '1:1',
     '605f903e4c07a534c812288bfe3b0ce00716cfb0905c134f7e418e228f455040',
     'e055639a8e37af49ddf7b80cb8b390903857b43ced770d47ffa9853515c779c9', (3728, 3728)),
]


def check_samples(directory, download=False):
    directory.mkdir(parents=True, exist_ok=True)
    for name, number, model, crop, file_hash, mosaic_hash, shape in SAMPLES:
        path = directory / (name + '.nef')
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f'{path} is missing; use --download to fetch the sample')
            bits = '8bit%208bit' if model == 'Z%208' else '8bit'
            url = (f'https://raw.pixls.us/getfile.php/{number}/nice/'
                   f'Nikon%20-%20{model}%20-%20{bits}%20compressed%20%28{crop}%29.NEF')
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read()
            if hashlib.sha256(body).hexdigest() != file_hash:
                raise RuntimeError(f'Sample download checksum mismatch: {name}')
            path.write_bytes(body)
        if hashlib.sha256(path.read_bytes()).hexdigest() != file_hash:
            raise RuntimeError(f'Sample file checksum mismatch: {name}')
        with rawpy.imread(str(path)) as raw:
            mosaic = np.ascontiguousarray(raw.raw_image, dtype='<u2')
            if mosaic.shape != shape:
                raise RuntimeError(f'{name}: unexpected sensor dimensions {mosaic.shape}')
            if hashlib.sha256(mosaic.tobytes()).hexdigest() != mosaic_hash:
                raise RuntimeError(f'{name}: decoded mosaic differs from the reference')
            if raw.white_level != 16383:
                raise RuntimeError(f'{name}: unexpected white level {raw.white_level}')
        print(f'PASS {name}: {shape[1]}x{shape[0]}, every sensor sample matches', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--sample-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / '.test-output/nikon-he')
    args = parser.parse_args()
    check_samples(args.sample_dir, args.download)


if __name__ == '__main__':
    main()
