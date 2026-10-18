"""DNG profiles missing cam_xyz must survive both RAW decode entry points (#34)."""
from types import SimpleNamespace

import numpy as np
import pytest

from raw_alchemy import config
from raw_alchemy.colorspace_matrices import (
    SRGB_TO_XYZ_D65, cam_to_working_space_matrix, working_rgb_to_xyz_d65,
)
from raw_alchemy.raw_color import camera_to_working


class ProfileRaw:
    rgb_xyz_matrix = np.zeros((4, 3), np.float32)
    color_matrix = np.zeros((3, 4), np.float32)
    num_colors = 3
    raw_pattern = np.array([[0, 1], [3, 2]], np.uint8)
    raw_image_visible = np.full((8, 10), 2000, np.uint16)
    black_level_per_channel = [0, 0, 0, 0]
    white_level = 16383
    camera_whitebalance = [2.0, 1.0, 3.0, 1.0]
    sizes = SimpleNamespace(flip=6)

    def __init__(self):
        self.calls = []
        self.rgb16 = np.arange(8 * 10 * 3, dtype=np.uint16).reshape(8, 10, 3) + 3000

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def postprocess(self, **kwargs):
        self.calls.append(kwargs)
        return self.rgb16.copy()


def test_embedded_profile_recovers_zero_cam_xyz():
    raw = ProfileRaw()
    raw.color_matrix = np.column_stack((
        np.array([[1.5, -0.3, -0.2], [-0.1, 1.3, -0.2], [0.1, -0.2, 1.1]]),
        np.zeros(3),
    ))
    actual = camera_to_working(raw)
    # Check in XYZ space: cmatrix maps camera RGB to linear sRGB, whereas
    # cam_xyz has the opposite direction. Transposing it would pass an
    # existence test but break the camera's colours.
    np.testing.assert_allclose(
        working_rgb_to_xyz_d65() @ actual,
        SRGB_TO_XYZ_D65 @ raw.color_matrix[:, :3], atol=1e-12,
    )
    assert raw.calls == []


def test_normal_camera_matrix_keeps_existing_conversion():
    raw = ProfileRaw()
    raw.rgb_xyz_matrix = np.vstack((np.eye(3), np.zeros(3)))
    np.testing.assert_array_equal(
        camera_to_working(raw), cam_to_working_space_matrix(raw.rgb_xyz_matrix),
    )


@pytest.mark.parametrize('matrix', [
    np.zeros((4, 3)), np.ones((4, 3)), np.full((4, 3), np.nan),
    np.full((4, 3), np.inf), np.eye(4), np.zeros(3),
])
def test_unusable_camera_matrix_raises_instead_of_producing_nan(matrix):
    with pytest.raises(ValueError):
        cam_to_working_space_matrix(matrix)


@pytest.mark.parametrize('variant', ['canonical', 'preload'])
@pytest.mark.parametrize('space', [config.WORKING_SPACE_PROPHOTO, config.WORKING_SPACE_ACESCG])
def test_missing_matrices_use_libraw_colour_once(monkeypatch, variant, space):
    import rawpy
    from raw_alchemy import core, exif, native_decode, rawspeed
    from raw_alchemy.onnx import rcd_demosaic
    from raw_alchemy.onnx.denoiser import _apply_flip

    raw = ProfileRaw()
    monkeypatch.setattr(config, 'WORKING_SPACE', space)
    monkeypatch.setattr(rawpy, 'imread', lambda path: raw)
    # RawSpeed can return an all-zero DNG matrix instead of reporting that
    # its profile is unavailable. That must also reach the LibRaw path.
    monkeypatch.setattr(rawspeed, 'try_decode', lambda path: SimpleNamespace(
        is_xtrans=False, is_bayer=True, color_matrix=np.zeros((3, 3)),
    ))
    monkeypatch.setattr(exif, 'extract_lens_exif', lambda *a: ({}, None))

    def unexpected_demosaic(*args, **kwargs):
        pytest.fail('a missing colour matrix must be resolved before GPU demosaic')

    monkeypatch.setattr(rcd_demosaic, 'rcd_demosaic', unexpected_demosaic)
    if variant == 'canonical':
        actual = core._rawpy_decode_to_prophoto('embedded-profile.dng')
    else:
        actual, _, _ = native_decode.decode_cpu('embedded-profile.dng')
    expected = raw.rgb16.astype(np.float32) / 65535.0
    expected = expected @ np.linalg.solve(working_rgb_to_xyz_d65(), SRGB_TO_XYZ_D65).T
    expected = _apply_flip(expected, raw.sizes.flip)
    if variant == 'preload':
        expected = np.clip(expected, 0.0, 1.0)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert len(raw.calls) == 1
    assert raw.calls[0]['use_camera_wb'] is True
    assert 'user_wb' not in raw.calls[0]
    assert raw.calls[0]['output_color'] == rawpy.ColorSpace.sRGB
    assert raw.calls[0]['user_flip'] == 0
    assert raw.calls[0]['gamma'] == (1, 1)


@pytest.mark.parametrize('make,model', [
    ('NIKON CORPORATION', 'NIKON Z50_2'), ('Nikon', 'Z50_2'),
])
def test_he_camera_uses_same_bundled_profile_as_lossless(monkeypatch, make, model):
    from raw_alchemy import exif
    monkeypatch.setattr(exif, 'extract_lens_exif', lambda path, _: (
        {'camera_maker': make, 'camera_model': model}, None,
    ))
    raw = ProfileRaw()
    xyz = np.array([[11640, -4829, -1079], [-5107, 13006, 2325], [-972, 1711, 7380]]) / 10000.0
    np.testing.assert_allclose(camera_to_working(raw, 'Z50_2.nef'), cam_to_working_space_matrix(xyz))
    assert raw.calls == []


def test_camera_profile_changes_invalidate_content_cache(tmp_path, monkeypatch):
    from raw_alchemy import rawspeed
    monkeypatch.setattr(rawspeed, '_vendor_dir', lambda: tmp_path)
    xml = '''<Cameras><Camera make="MAKE" model="MODEL"><ColorMatrices><ColorMatrix>
    <ColorMatrixRow plane="0">10000 0 0</ColorMatrixRow>
    <ColorMatrixRow plane="1">0 10000 0</ColorMatrixRow>
    <ColorMatrixRow plane="2">0 0 VALUE</ColorMatrixRow>
    </ColorMatrix></ColorMatrices></Camera></Cameras>'''
    (tmp_path / 'cameras.xml').write_text(xml.replace('VALUE', '10000'))
    first = rawspeed.camera_matrix('MAKE', 'MODEL')
    (tmp_path / 'cameras.xml').write_text(xml.replace('VALUE', '20000'))
    second = rawspeed.camera_matrix('MAKE', 'MODEL')
    assert first[2, 2] == 1.0
    assert second[2, 2] == 2.0
