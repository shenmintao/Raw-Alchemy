"""Missing compact-camera LensModel must not disable its fixed lens (#22)."""
from types import SimpleNamespace

import numpy as np
import pytest

from raw_alchemy import lensfun_wrapper as lf, utils


@pytest.mark.parametrize('lens_model', ['', None, 'unreported compact lens'])
def test_zv1_resolves_its_unique_private_mount_profile(lens_model):
    if lf._lensfun is None:
        pytest.skip('native Lensfun unavailable')
    db = lf.LensfunDatabase()
    camera = db.find_camera('Sony', 'ZV-1')
    assert camera
    lens = db.find_lens(camera, None, lens_model)
    assert lens and lens.contents.Model == b'ZV-1 & compatibles'


def test_missing_interchangeable_lens_is_not_guessed_from_camera_name():
    if lf._lensfun is None:
        pytest.skip('native Lensfun unavailable')
    db = lf.LensfunDatabase()
    camera = db.find_camera('Sony', 'ILCE-7M4')
    assert camera
    # A fuzzy ILCE-7M4 body-name search can match a Helios-44M-4 lens.
    assert db.find_lens(camera, None, '') is None


@pytest.mark.parametrize('budget_mb', [0, 64])
def test_missing_lens_exif_reaches_real_compact_correction(monkeypatch, budget_mb):
    if lf._lensfun is None:
        pytest.skip('native Lensfun unavailable')
    from raw_alchemy import config
    monkeypatch.setattr(config, 'DISTORTION_MAP_CACHE_LIMIT_MB', budget_mb)
    original_modifier = lf.LensfunModifier
    calibrated_crops, enabled = [], []
    class ObservedModifier(original_modifier):
        def __init__(self, lens, focal, crop, *args, **kwargs):
            calibrated_crops.append(crop)
            super().__init__(lens, focal, crop, *args, **kwargs)
        def enable_distortion_correction(self):
            flags = super().enable_distortion_correction()
            enabled.append(flags)
            return flags
    monkeypatch.setattr(lf, 'LensfunModifier', ObservedModifier)
    x, y = np.meshgrid(np.linspace(0.1, 0.9, 128), np.linspace(0.1, 0.9, 96))
    source = np.stack((x, y, x * y), axis=-1).astype(np.float32)
    result = utils.apply_lens_correction(source.copy(), {
        'camera_maker': 'Sony', 'camera_model': 'ZV-1', 'lens_model': '',
        # Matches the reporter's RAW metadata. This 35mm-equivalent estimate
        # is below Lensfun's calibrated 2.70 and otherwise disables correction.
        'focal_length': 9.4, 'aperture': 1.8, 'crop_factor': 2.55,
    }, correct_vignetting=False)
    assert result.shape == source.shape and result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.max(np.abs(result - source)) > 1e-3
    assert calibrated_crops == pytest.approx([2.7])
    assert all(flags & lf.LF_MODIFY_DISTORTION for flags in enabled)


def test_gpu_map_path_uses_fixed_lens_calibration_crop():
    if lf._lensfun is None:
        pytest.skip('native Lensfun unavailable')
    source = np.full((96, 128, 3), 0.25, np.float32)
    result = lf.compute_lens_distortion_map(
        source, 'Sony', 'ZV-1', None, '', 9.4, 1.8,
        crop_factor=2.55, correct_vignetting=False,
    )
    assert result is not None
    coords, mask, _ = result
    assert coords.shape == (96, 128, 3, 2) and np.isfinite(coords).all()
    assert not mask.any()


def test_ambiguous_private_mount_does_not_pick_first_profile(monkeypatch):
    profiles = [object(), object(), None]
    database = lf.LensfunDatabase.__new__(lf.LensfunDatabase)
    database.db = None
    camera = SimpleNamespace(contents=SimpleNamespace(Mount=b'privateCompact'))
    monkeypatch.setattr(lf, '_lensfun', SimpleNamespace(
        lf_db_find_mount=lambda *args: None,
        lf_db_find_lenses=lambda *args: profiles,
    ))
    assert database.find_lens(camera, None, '') is None
