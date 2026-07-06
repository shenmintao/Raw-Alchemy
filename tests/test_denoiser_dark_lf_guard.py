"""Dark low-frequency guard + seam-free tile feathering.

On high-ISO night sky (near-black flat regions) v14's low-frequency output is
unreliable: an olive veil (G lifted, B crushed), low-frequency chroma blobs,
and per-tile DC offsets that print a grid once exposure is pushed. The guard
replaces the output's low-frequency component with the input's below a
brightness ramp; the tile window must enter at ~zero weight so tile DC
disagreement cannot print a sharp seam line.
"""
import numpy as np

from raw_alchemy.onnx.denoiser import (
    _apply_dark_lf_guard,
    _dark_lf_guard_config,
    _tile_weight,
)


def _cfg(**over):
    cfg = _dark_lf_guard_config()
    cfg.update(over)
    return cfg


def test_dark_veil_is_removed():
    """G lifted 9x / B crushed in near-black sky must return to input levels."""
    rng = np.random.default_rng(0)
    inp = np.full((256, 256, 4), 0.001, np.float32)
    inp += rng.normal(0, 0.0008, inp.shape).astype(np.float32)
    inp = np.clip(inp, 0, 1)
    out = np.full_like(inp, 0.001)
    out[..., 1] = 0.009  # G veil
    out[..., 3] = 0.009
    out[..., 2] = 0.0002  # B floor

    g = _apply_dark_lf_guard(out, inp, _cfg())
    m = g[64:192, 64:192].reshape(-1, 4).mean(0)
    mi = inp[64:192, 64:192].reshape(-1, 4).mean(0)
    np.testing.assert_allclose(m, mi, atol=2e-4)


def test_tile_dc_grid_is_flattened():
    """Neighbouring tiles settling on different DC levels in dark sky."""
    inp = np.full((256, 256, 4), 0.002, np.float32)
    out = np.full_like(inp, 0.002)
    out[:, 128:] += 0.003  # right tile sits on a different DC level

    g = _apply_dark_lf_guard(out, inp, _cfg())
    left = float(g[:, :96].mean())
    right = float(g[:, 160:].mean())
    assert abs(left - right) < 3e-4  # was 3e-3 pre-guard


def test_midtones_untouched():
    rng = np.random.default_rng(1)
    inp = rng.uniform(0.05, 0.6, (128, 128, 4)).astype(np.float32)
    out = np.clip(inp + rng.normal(0, 0.01, inp.shape).astype(np.float32), 0, 1)
    g = _apply_dark_lf_guard(out, inp, _cfg())
    np.testing.assert_allclose(g, out, atol=1e-6)


def test_denoised_detail_survives():
    """The guard only moves low frequencies: per-pixel detail the model
    produced in a dark region must keep its amplitude."""
    inp = np.full((256, 256, 4), 0.002, np.float32)
    out = np.full_like(inp, 0.005)  # wrong DC (will be corrected)...
    checker = ((np.indices((256, 256)).sum(0) % 2) * 0.002).astype(np.float32)
    out += checker[..., None]  # ...carrying fine detail (must survive)

    g = _apply_dark_lf_guard(out, inp, _cfg())
    hf = g - g.mean()  # remove global DC, checker amplitude remains
    got = float(np.abs(np.diff(g[128, 64:192, 0])).mean())
    assert abs(got - 0.002) < 1e-4


def test_disabled_is_identity():
    rng = np.random.default_rng(2)
    inp = rng.uniform(0, 0.01, (64, 64, 4)).astype(np.float32)
    out = rng.uniform(0, 0.01, (64, 64, 4)).astype(np.float32)
    g = _apply_dark_lf_guard(out, inp, _cfg(enabled=False))
    np.testing.assert_array_equal(g, out)


def test_xtrans_nine_channels():
    inp = np.full((128, 128, 9), 0.001, np.float32)
    out = np.full_like(inp, 0.001)
    out[..., 4] = 0.008
    g = _apply_dark_lf_guard(out, inp, _cfg())
    assert g.shape == (128, 128, 9)
    assert abs(float(g[32:96, 32:96, 4].mean()) - 0.001) < 2e-4


def test_tile_window_enters_at_near_zero_weight():
    """A tile whose edge overlaps a neighbour must fade in from ~0, otherwise
    tile DC disagreement prints a sharp seam line across flat sky."""
    w = _tile_weight(768, 768, 64, at_top=False, at_bottom=False,
                     at_left=False, at_right=False)[0]
    assert w[0, 384] < 2e-3  # first feather row ~0 (was 1/65 ≈ 0.015)
    assert w[384, 0] < 2e-3
    assert w[384, 384] == 1.0  # interior plateau intact


def test_tile_window_keeps_image_borders_at_one():
    w = _tile_weight(768, 768, 64, at_top=True, at_bottom=False,
                     at_left=True, at_right=False)[0]
    assert w[0, 0] == 1.0
    assert w[0, 767] < 2e-3


def test_blend_of_disagreeing_tiles_has_no_sharp_step():
    """Two constant tiles differing by delta, blended with the window over a
    64px overlap: the per-pixel step must stay far below the raw delta."""
    delta = 0.003
    tile_a = np.zeros((1, 128, 128), np.float32)
    tile_b = np.full((1, 128, 128), delta, np.float32)
    # tile A covers x=[0,128), tile B x=[64,192): overlap 64
    acc = np.zeros((1, 128, 192), np.float32)
    wsum = np.zeros_like(acc)
    wa = _tile_weight(128, 128, 64, at_top=True, at_bottom=True,
                      at_left=True, at_right=False)
    wb = _tile_weight(128, 128, 64, at_top=True, at_bottom=True,
                      at_left=False, at_right=True)
    acc[:, :, :128] += tile_a * wa
    wsum[:, :, :128] += wa
    acc[:, :, 64:] += tile_b * wb
    wsum[:, :, 64:] += wb
    blended = acc / np.maximum(wsum, 1e-8)
    steps = np.abs(np.diff(blended[0, 64], axis=-1))
    assert steps.max() < delta / 15  # linear-ramp entry jumped delta/65*... ~delta/33
