"""Sub-black noise must survive decoding: clipping it at black turns zero-mean
read noise into a positive bias that white balance scales per channel (the
magenta night sky)."""

import numpy as np


def test_black_subtraction_can_keep_sub_black_readings():
    from raw_alchemy import core

    pattern = np.array([[0, 1], [1, 2]])
    raw = np.array([[1000, 1030], [990, 1024]], np.float32)
    kept = core.subtract_black_level(raw.copy(), [1024] * 4, 2048.0, pattern, clip_negative=False)
    clipped = core.subtract_black_level(raw.copy(), [1024] * 4, 2048.0, pattern)
    assert kept.min() < 0.0
    assert clipped.min() == 0.0  # default unchanged for the other callers
    np.testing.assert_allclose(kept[kept > 0], clipped[kept > 0])


def test_lift_and_unlift_are_exact_including_highlights_above_one():
    from raw_alchemy import core

    rng = np.random.default_rng(0)
    mosaic = rng.normal(0.0, 0.01, (64, 64)).astype(np.float32)
    mosaic[:8, :8] = 1.35  # reconstructed highlight above the clip level
    original = mosaic.copy()

    lift, scale = core._lift_for_demosaic(mosaic)
    assert lift > 0.0 and scale >= 1.35 + lift
    assert mosaic.min() >= 0.0 and mosaic.max() <= 1.0

    # Stand-in demosaic: the lifted value in all three channels.
    rgb = np.repeat(mosaic[:, :, None], 3, axis=2)
    wb3 = np.array([2.0, 1.0, 1.5], np.float32)
    cam = np.array([[1.2, -0.1, -0.1], [-0.2, 1.3, -0.1], [0.0, -0.3, 1.3]], np.float32)
    out = core._unlift_to_working_space(rgb, lift, scale, wb3, cam, rows_per_chunk=7)

    expected = np.einsum("ij,hwj->hwi", cam, np.repeat(original[:, :, None], 3, axis=2) * wb3)
    expected = np.minimum(expected, 1.0)
    unclipped = original >= 0.0  # values the demosaic guard had to raise
    np.testing.assert_allclose(out[unclipped], expected[unclipped], atol=2e-5)
    assert (out < 0.0).any()  # sub-black noise is still there after WB + matrix


def test_zero_mean_noise_has_no_channel_bias_after_white_balance():
    """The bug itself: black-level noise must not turn magenta."""
    from raw_alchemy import core

    rng = np.random.default_rng(1)
    pattern = np.array([[0, 1], [1, 2]])
    raw = (1024 + rng.normal(0, 40, (256, 256))).astype(np.float32)
    wb3 = np.array([1.7, 1.0, 3.0], np.float32)
    means = {}
    for clip in (True, False):
        norm = core.subtract_black_level(raw.copy(), [1024] * 4, 16383.0, pattern, clip_negative=clip)
        plane_means = [norm[pattern_mask(pattern, c, norm.shape)].mean() * wb3[c] for c in range(3)]
        means[clip] = np.array(plane_means)
    # Clipped: R and B come out several times brighter than G (magenta).
    assert means[True][2] > 2.0 * means[True][1]
    # Kept: every channel stays at the true black level.
    assert np.all(np.abs(means[False]) < 2e-4)


def pattern_mask(pattern, colour, shape):
    tiled = np.tile(pattern, (shape[0] // 2, shape[1] // 2))
    return tiled == colour


def test_metering_view_is_area_averaged():
    from raw_alchemy import utils

    rng = np.random.default_rng(2)
    img = (0.001 + rng.normal(0, 0.01, (2048, 3072, 3))).astype(np.float32)
    view = utils.get_subsampled_view(img, target_size=512)
    assert max(view.shape[:2]) == 512
    # Averaging, not striding: noise shrinks and the mean stays unbiased.
    assert view.std() < 0.2 * img.std()
    np.testing.assert_allclose(view.mean(), 0.001, atol=2e-4)
