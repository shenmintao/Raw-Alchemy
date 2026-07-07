"""ONNX demosaic runtime modules (RCD Bayer / Markesteijn X-Trans).

Algorithm parity vs the retired Taichi ports was established offline
(RCD max |delta| 2.1e-7; X-Trans bit-exact vs fast_math=False reference).
These tests pin the runtime wrappers: session reuse, mask construction,
tiling consistency, and the flat-zero-region NaN regression (the dynamo
exporter's rewritten graph produced NaN over black shadow areas — the
shipped torchscript export must not).
"""
import numpy as np
import pytest

from raw_alchemy.onnx import rcd_demosaic as R
from raw_alchemy.onnx import xtrans_demosaic as X

RGGB = np.array([[0, 1], [3, 2]])


def test_rcd_zero_regions_produce_no_nan():
    rng = np.random.default_rng(0)
    bayer = rng.random((128, 160), dtype=np.float32) * 0.01
    bayer[16:80, 16:80] = 0.0  # clipped black shadow block
    bayer[90:110, 100:140] = 2.3  # highlight-inpaint can exceed 1.0
    rgb = R.rcd_demosaic(bayer, RGGB)
    assert rgb.shape == (128, 160, 3)
    assert np.isfinite(rgb).all()


def test_rcd_session_singleton():
    bayer = np.random.default_rng(1).random((64, 64), dtype=np.float32)
    R.rcd_demosaic(bayer, RGGB)
    s1 = R._get_session()
    R.rcd_demosaic(np.random.default_rng(2).random((128, 96), dtype=np.float32), RGGB)
    assert R._get_session() is s1  # 单一 tile 形状,任何尺寸共用一个 session


def test_rcd_tiled_matches_single_tile_interior():
    rng = np.random.default_rng(9)
    bayer = rng.random((300, 420), dtype=np.float32)
    full = R.rcd_demosaic(bayer, RGGB)
    old_tile, old_ov = R.TILE, R.OVERLAP
    try:
        R.TILE, R.OVERLAP = 150, 24
        tiled = R.rcd_demosaic(bayer, RGGB)
    finally:
        R.TILE, R.OVERLAP = old_tile, old_ov
    d = np.abs(tiled - full)[16:-16, 16:-16]
    assert float(d.max()) < 1e-6


def test_rcd_neutral_field_stays_neutral():
    """On a constant mosaic every channel must reconstruct that constant."""
    bayer = np.full((96, 96), 0.25, np.float32)
    rgb = R.rcd_demosaic(bayer, RGGB)
    np.testing.assert_allclose(rgb[8:-8, 8:-8], 0.25, atol=1e-5)


def test_rcd_rejects_bad_input():
    with pytest.raises(ValueError):
        R.rcd_demosaic(np.zeros((65, 64), np.float32), RGGB)  # odd size
    with pytest.raises(ValueError):
        R.rcd_demosaic(np.zeros((64, 64, 3), np.float32), RGGB)


def test_xtrans_masks_shape_and_partition():
    m = X._build_masks(X.CANONICAL_PATTERN)
    assert m.shape == (15, 6, 6)
    np.testing.assert_array_equal(m[0] + m[1] + m[2], np.ones((6, 6)))  # RGB partition
    np.testing.assert_array_equal(sum(m[6:]), np.ones((6, 6)))  # 9 phase masks


def test_xtrans_canonical_roll_detection():
    assert X._canonical_roll(X.CANONICAL_PATTERN) == (0, 0)
    rolled = np.roll(np.roll(X.CANONICAL_PATTERN, 2, 0), 1, 1)
    dr, dc = X._canonical_roll(rolled)
    assert (dr, dc) != (0, 0)
    with pytest.raises(ValueError):
        X._canonical_roll(np.zeros((6, 6), int))


def test_xtrans_demosaic_shapes_and_finite():
    rng = np.random.default_rng(2)
    raw = rng.random((132, 180), dtype=np.float32)  # multiples of 6
    rgb = X.xtrans_markesteijn_demosaic(raw, X.CANONICAL_PATTERN)
    assert rgb.shape == (132, 180, 3)
    assert np.isfinite(rgb).all()
    # non-multiple-of-6 sizes must also work (module pads internally)
    rgb2 = X.xtrans_markesteijn_demosaic(np.ascontiguousarray(raw[:130, :177]),
                                         X.CANONICAL_PATTERN)
    assert rgb2.shape == (130, 177, 3)
    assert np.isfinite(rgb2).all()


def test_xtrans_tiled_matches_single_tile_interior():
    rng = np.random.default_rng(3)
    raw = rng.random((300, 432), dtype=np.float32)
    full = X.xtrans_markesteijn_demosaic(raw, X.CANONICAL_PATTERN)
    old_tile, old_ov = X.TILE, X.OVERLAP
    try:
        X.TILE, X.OVERLAP = 150, 24
        tiled = X.xtrans_markesteijn_demosaic(raw, X.CANONICAL_PATTERN)
    finally:
        X.TILE, X.OVERLAP = old_tile, old_ov
    d = np.abs(tiled - full)[16:-16, 16:-16]
    assert float(d.max()) == 0.0
