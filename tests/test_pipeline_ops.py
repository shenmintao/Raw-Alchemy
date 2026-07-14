from pathlib import Path

import colour
import numpy as np

from raw_alchemy import core, utils
from raw_alchemy.exporting import resolve_export_exposure
from raw_alchemy.math_ops import (
    apply_highlight_shadow_inplace,
    apply_lut_inplace,
    apply_matrix_inplace,
    apply_saturation_contrast_inplace,
    compute_perspective_matrix,
    perspective_warp_kernel,
)
from raw_alchemy.metering import get_metering_strategy
from raw_alchemy.pipeline.executor import ExportExecutor
from raw_alchemy.pipeline.ops import build_op_list


GOLDEN = Path(__file__).with_name("golden") / "pipeline_ops.npz"


def test_cached_export_avoids_redundant_source_copy(monkeypatch):
    source = np.linspace(0.0, 1.0, 9 * 11 * 3, dtype=np.float32).reshape(9, 11, 3)
    original = source.copy()
    captured = {}

    def fake_run(src, params, metering_source=None, lens_corrector=None):
        captured["source"] = src
        captured["metering_source"] = metering_source
        return src.copy()

    monkeypatch.setattr(core, "_run_export_executor", fake_run)
    monkeypatch.setattr(core, "save_image", lambda *args, **kwargs: True)

    core.export_from_cache(
        cached_img=source,
        output_path="out.tif",
        exif_data={},
        exif_metadata=None,
        log_space="None",
        lut_path=None,
        exposure=0.0,
    )

    assert captured["source"] is source
    assert captured["metering_source"] is source
    np.testing.assert_array_equal(source, original)


def test_large_one_shot_lens_correction_uses_striped_path(monkeypatch):
    from raw_alchemy import config

    source = np.zeros((12, 16, 3), np.float32)
    exif = {
        "camera_maker": "Maker",
        "camera_model": "Model",
        "lens_maker": "Lens",
        "lens_model": "Prime",
        "focal_length": 50.0,
        "aperture": 2.8,
    }
    calls = []
    monkeypatch.setattr(config, "DISTORTION_MAP_CACHE_LIMIT_MB", 0)
    monkeypatch.setattr(
        utils.lf,
        "apply_lens_correction_tiled",
        lambda image, **kwargs: calls.append("striped") or (image + 0.5),
    )
    monkeypatch.setattr(
        utils.lf,
        "apply_lens_correction",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("oversized one-shot correction must not allocate a full map")
        ),
    )

    corrected = utils.apply_lens_correction(source, exif)
    assert calls == ["striped"]
    np.testing.assert_array_equal(corrected, source + 0.5)


def test_subtract_black_level_golden():
    golden = np.load(GOLDEN)

    actual = core.subtract_black_level(
        golden["sensor_raw"],
        golden["black_levels"],
        float(golden["white_level"]),
        golden["cfa_pattern"],
    )

    np.testing.assert_allclose(actual, golden["black_level_expected"], rtol=1e-7, atol=1e-7)


def test_fix_hot_pixels_golden():
    golden = np.load(GOLDEN)
    actual = golden["hot_pixels_input"].copy()

    core.fix_hot_pixels(actual, golden["cfa_pattern"], threshold=1.0)

    np.testing.assert_allclose(actual, golden["hot_pixels_expected"], rtol=1e-7, atol=1e-7)


def test_metering_strategies_golden():
    golden = np.load(GOLDEN)
    source_cs = colour.RGB_COLOURSPACES["sRGB"]

    for mode in golden["metering_modes"]:
        mode = str(mode)
        actual = get_metering_strategy(mode).calculate_gain(
            golden["metering_img"].copy(),
            source_cs,
            target_gray=0.18,
        )
        np.testing.assert_allclose(
            actual,
            float(golden[f"metering_{mode.replace('-', '_')}"]),
            rtol=1e-7,
            atol=1e-7,
        )


def test_math_ops_golden_values():
    golden = np.load(GOLDEN)

    img = golden["math_img"].copy()
    apply_matrix_inplace(img, golden["matrix"])
    np.testing.assert_allclose(img, golden["matrix_expected"], rtol=1e-6, atol=1e-6)

    img = golden["math_img"].copy()
    apply_saturation_contrast_inplace(
        img,
        float(golden["saturation"]),
        float(golden["contrast"]),
        float(golden["pivot"]),
        golden["luma_coeffs"],
    )
    np.testing.assert_allclose(img, golden["sat_contrast_expected"], rtol=1e-6, atol=1e-6)

    img = golden["math_img"].copy()
    apply_highlight_shadow_inplace(
        img,
        float(golden["highlight"]),
        float(golden["shadow"]),
        golden["luma_coeffs"],
    )
    np.testing.assert_allclose(img, golden["highlight_shadow_expected"], rtol=1e-6, atol=1e-6)

    img = golden["lut_img"].copy()
    apply_lut_inplace(img, golden["lut_table"], golden["lut_domain_min"], golden["lut_domain_max"])
    np.testing.assert_allclose(img, golden["lut_expected"], rtol=1e-6, atol=1e-6)


def test_export_from_cache_applies_perspective_before_crop(monkeypatch):
    captured = {}

    monkeypatch.setattr(core, "save_image", lambda img, *args, **kwargs: captured.setdefault("img", img.copy()))

    src = np.linspace(0.0, 1.0, 10 * 12 * 3, dtype=np.float32).reshape(10, 12, 3)
    corners = ((0.1, 0.0), (0.95, 0.08), (0.9, 0.95), (0.05, 0.9))
    crop = (0.1, 0.2, 0.7, 0.6)

    core.export_from_cache(
        cached_img=src,
        output_path="out.tif",
        exif_data={},
        exif_metadata=None,
        log_space="None",
        lut_path=None,
        exposure=0.0,
        saturation=1.0,
        contrast=1.0,
        perspective_corners=corners,
        crop=crop,
    )

    _, m_inv = compute_perspective_matrix(corners, src.shape[1], src.shape[0])
    preview_warp = np.zeros_like(src)
    perspective_warp_kernel(np.ascontiguousarray(src.copy()), preview_warp, m_inv)
    expected_params = {
        "lens_correct": False,
        "exposure_mode": "Manual",
        "exposure": 0.0,
        "metering_mode": "hybrid",
        "wb_temp": 0.0,
        "wb_tint": 0.0,
        "highlight": 0.0,
        "shadow": 0.0,
        "saturation": 1.0,
        "contrast": 1.0,
        "log_space": "None",
        "lut_path": None,
        "rotation": 0,
        "flip_horizontal": False,
        "flip_vertical": False,
        "perspective_corners": None,
        "crop": (0.0, 0.0, 1.0, 1.0),
        "sharpen_strength": 0.0,
    }
    expected = ExportExecutor().run(
        build_op_list(expected_params),
        utils.apply_crop(preview_warp, crop),
    )
    np.testing.assert_allclose(captured["img"], expected, rtol=1e-6, atol=1e-6)


def test_default_perspective_corners_are_noop_for_export(monkeypatch):
    captured = []

    monkeypatch.setattr(core, "save_image", lambda img, *args, **kwargs: captured.append(img.copy()))

    src = np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    default_corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    for corners in (None, default_corners):
        core.export_from_cache(
            cached_img=src,
            output_path="out.tif",
            exif_data={},
            exif_metadata=None,
            log_space="None",
            lut_path=None,
            exposure=0.0,
            saturation=1.0,
            contrast=1.0,
            perspective_corners=corners,
        )

    np.testing.assert_allclose(captured[1], captured[0], rtol=0.0, atol=0.0)


def test_process_image_entry_uses_export_executor(monkeypatch):
    captured = {}
    src = np.linspace(0.02, 0.6, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)

    monkeypatch.setattr(core, "_rawpy_decode_to_prophoto", lambda raw_path: src.copy())
    monkeypatch.setattr("raw_alchemy.exif.extract_lens_exif", lambda raw_path, raw: ({}, None))
    monkeypatch.setattr(core, "save_image", lambda img, *args, **kwargs: captured.setdefault("img", img.copy()))

    core.process_image(
        raw_path="synthetic.dng",
        output_path="out.tif",
        log_space="None",
        lut_path=None,
        exposure=0.25,
        lens_correct=False,
        saturation=1.1,
        contrast=0.95,
    )

    params = {
        "lens_correct": False,
        "exposure_mode": "Manual",
        "exposure": 0.25,
        "metering_mode": "hybrid",
        "wb_temp": 0.0,
        "wb_tint": 0.0,
        "highlight": 0.0,
        "shadow": 0.0,
        "saturation": 1.1,
        "contrast": 0.95,
        "log_space": "None",
        "lut_path": None,
        "rotation": 0,
        "flip_horizontal": False,
        "flip_vertical": False,
        "perspective_corners": None,
        "crop": (0.0, 0.0, 1.0, 1.0),
        "sharpen_strength": 0.0,
    }
    expected = ExportExecutor().run(build_op_list(params), src.copy())
    np.testing.assert_allclose(captured["img"], expected, rtol=1e-6, atol=1e-6)


def test_resolve_export_exposure_auto_uses_preview_ev_and_full_path_auto():
    params = {"exposure_mode": "Auto", "exposure": -2.0}

    fast_path, full_path = resolve_export_exposure(params, last_applied_ev=0.75)

    assert fast_path == 0.75
    assert full_path is None


def test_resolve_export_exposure_manual_uses_slider_value_for_both_paths():
    params = {"exposure_mode": "Manual", "exposure": -0.5}

    assert resolve_export_exposure(params, last_applied_ev=0.75) == (-0.5, -0.5)
