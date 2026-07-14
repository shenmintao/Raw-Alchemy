from pathlib import Path

import colour
import numpy as np
import pytest

from raw_alchemy import config
from raw_alchemy.colorspace_matrices import (
    ACESCG_TO_XYZ_D65,
    PROPHOTO_TO_XYZ_D65,
    cam_to_prophoto_matrix,
    cam_to_working_matrix,
    cam_to_working_space_matrix,
    working_rgb_to_xyz_d65,
)
from raw_alchemy.math_ops import (
    log_encode_gpu,
    pq_encode_inplace,
    white_balance_matrix,
    working_space_adaptation_matrix,
)


GOLDEN = Path(__file__).with_name("golden") / "color_math.npz"


def test_pq_encode_inplace_matches_colour_reference():
    src = np.linspace(0.0, 1.25, 37 * 29 * 3, dtype=np.float32).reshape(37, 29, 3)
    actual = src.copy()

    pq_encode_inplace(actual, peak_nits=1000.0, mastering_nits=10000.0)
    expected = colour.cctf_encoding(
        np.clip(src, 0.0, None).astype(np.float64) * 1000.0,
        function="ST 2084",
    ).astype(np.float32)
    np.clip(expected, 0.0, 1.0, out=expected)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("log_space", list(config.LOG_TO_WORKING_SPACE))
def test_log_encode_gpu_matches_colour_reference(log_space):
    curve = config.LOG_ENCODING_MAP.get(log_space, log_space)
    src = np.linspace(0.001, 0.9, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    actual = src.copy()

    assert log_encode_gpu(actual, curve)

    expected = colour.cctf_encoding(src.astype(np.float64), function=curve).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_cam_to_prophoto_matrix_golden_values():
    golden = np.load(GOLDEN)

    for name in golden["case_names"]:
        case = str(name)
        actual = cam_to_prophoto_matrix(golden[f"{case}_xyz_to_cam"])
        np.testing.assert_allclose(
            actual,
            golden[f"{case}_cam_to_prophoto"],
            rtol=1e-10,
            atol=1e-10,
        )


def test_working_space_default_matches_prophoto_wrapper():
    golden = np.load(GOLDEN)

    for name in golden["case_names"]:
        xyz_to_cam = golden[f"{str(name)}_xyz_to_cam"]
        np.testing.assert_allclose(
            cam_to_working_space_matrix(xyz_to_cam),
            cam_to_prophoto_matrix(xyz_to_cam),
            rtol=0,
            atol=0,
        )


def test_acescg_to_xyz_d65_matches_colour_bradford_reference():
    from colour.adaptation import matrix_chromatic_adaptation_VonKries

    acescg = colour.RGB_COLOURSPACES[config.WORKING_SPACE_ACESCG]
    d65_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    cat = matrix_chromatic_adaptation_VonKries(
        colour.xy_to_XYZ(acescg.whitepoint),
        colour.xy_to_XYZ(d65_xy),
        transform="Bradford",
    )
    expected = cat @ acescg.matrix_RGB_to_XYZ

    np.testing.assert_allclose(ACESCG_TO_XYZ_D65, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "working_matrix",
    [PROPHOTO_TO_XYZ_D65, ACESCG_TO_XYZ_D65],
)
def test_cam_to_working_matrix_locks_neutral_white_point(working_matrix):
    golden = np.load(GOLDEN)
    neutral_cam = np.ones(3, dtype=np.float64)

    for name in golden["case_names"]:
        cam_to_rgb, _daylight_mul = cam_to_working_matrix(
            golden[f"{str(name)}_xyz_to_cam"],
            working_matrix,
        )
        actual = cam_to_rgb @ neutral_cam
        np.testing.assert_allclose(actual, np.ones(3), rtol=1e-10, atol=1e-10)


def test_working_rgb_to_xyz_d65_dispatches_configured_spaces():
    np.testing.assert_allclose(
        working_rgb_to_xyz_d65(config.WORKING_SPACE_PROPHOTO),
        PROPHOTO_TO_XYZ_D65,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        working_rgb_to_xyz_d65(config.WORKING_SPACE_ACESCG),
        ACESCG_TO_XYZ_D65,
        rtol=0,
        atol=0,
    )


def test_white_balance_matrix_zero_adjustment_is_identity():
    np.testing.assert_allclose(
        white_balance_matrix(0.0, 0.0),
        np.eye(3, dtype=np.float64),
        rtol=0,
        atol=0,
    )


def test_working_space_adaptation_matrix_matches_colour_bradford_reference():
    from colour.adaptation import matrix_chromatic_adaptation_VonKries

    observer = "CIE 1931 2 Degree Standard Observer"
    source_xy = colour.CCS_ILLUMINANTS[observer]["D65"]
    target_xy = colour.CCS_ILLUMINANTS[observer]["A"]
    rgb_to_xyz = working_rgb_to_xyz_d65(config.WORKING_SPACE)
    cat = matrix_chromatic_adaptation_VonKries(
        colour.xy_to_XYZ(source_xy),
        colour.xy_to_XYZ(target_xy),
        transform="Bradford",
    )
    expected = np.linalg.inv(rgb_to_xyz) @ cat @ rgb_to_xyz

    actual = working_space_adaptation_matrix(
        tuple(float(v) for v in source_xy),
        tuple(float(v) for v in target_xy),
        config.WORKING_SPACE,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
