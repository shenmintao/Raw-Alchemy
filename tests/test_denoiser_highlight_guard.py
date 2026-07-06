"""Saturated-spot halo dilation in the highlight guard.

The model paints a magenta/purple ring around small clipped light sources
(street lamps at night). The ring sits at *mid* brightness — below the
guard's blend_lo ramp — so the brightness ramp alone never protected it.
The fix dilates the near-clip mask (sat_thr/sat_dilate) so the halo gets
full brightness weight, while the chroma-divergence gate still decides
whether anything is pulled back: ordinary mid-tones inside the dilated
ring keep their full denoise.
"""
import numpy as np

from raw_alchemy.onnx.denoiser import _apply_highlight_guard, _highlight_guard_config


def _chroma_dev(x, sl):
    """Max channel imbalance |ch/mean - 1| over a region (0 == neutral)."""
    v = x[sl]
    m = v.mean(axis=-1, keepdims=True) + 1e-6
    return float(np.abs(v / m - 1).max())


def _lamp_scene():
    """Neutral mid-tone field + clipped lamp core + model-painted magenta ring."""
    inp = np.full((128, 128, 4), 0.25, np.float32)
    inp[60:68, 60:68] = 1.0  # clipped lamp core
    out = inp.copy()
    out[50:78, 50:78, 2] *= 0.3  # chroma shift around the lamp (the ring)
    out[10:20, 10:20] = 0.22  # legit denoise change far from any lamp
    return inp, out


RING = (slice(52, 58), slice(60, 68))  # inside the ring, outside the core
FAR = (slice(10, 20), slice(10, 20))


def test_halo_dilation_neutralises_lamp_ring_chroma():
    inp, out = _lamp_scene()
    g = _apply_highlight_guard(out, inp, _highlight_guard_config())
    assert _chroma_dev(out, RING) > 0.3  # the artifact is real pre-guard
    assert _chroma_dev(g, RING) < 0.02  # ...and neutral post-guard


def test_halo_restores_luma_not_just_chroma():
    """The model crushes luma around lamps too: a chroma-only fix turns
    bright pink lamp cores into dim violet. Inside the halo the full pixel
    must return to the input."""
    inp, out = _lamp_scene()
    g = _apply_highlight_guard(out, inp, _highlight_guard_config())
    np.testing.assert_allclose(g[RING], inp[RING], atol=0.02)


def test_halo_dilation_keeps_denoise_far_from_lamps():
    inp, out = _lamp_scene()
    g = _apply_highlight_guard(out, inp, _highlight_guard_config())
    assert abs(float(g[FAR].mean()) - float(out[FAR].mean())) < 0.005


def test_ring_was_unprotected_without_dilation():
    """Regression witness: the old ramp-only guard misses the mid-bright ring."""
    inp, out = _lamp_scene()
    cfg = dict(_highlight_guard_config(), sat_dilate=0)
    g = _apply_highlight_guard(out, inp, cfg)
    assert _chroma_dev(g, RING) > 0.3


def test_halo_leaves_dark_sky_around_lamps_denoised():
    """The dilated disc around a lamp crosses night sky. Reverting the sky
    to the noisy input paints a visible dark noise bubble around every lamp,
    so the local-brightness gate must keep the model's output there."""
    rng = np.random.default_rng(3)
    inp = np.full((128, 128, 4), 0.02, np.float32)  # night sky
    inp += rng.normal(0, 0.01, inp.shape).astype(np.float32)  # sensor noise
    inp = np.clip(inp, 0, 1)
    inp[60:68, 60:68] = 1.0  # clipped lamp core
    out = np.full_like(inp, 0.02)  # model: smooth, denoised sky
    out[60:68, 60:68] = 1.0

    g = _apply_highlight_guard(out, inp, _highlight_guard_config())
    sky_ring = (slice(44, 52), slice(60, 68))  # inside dilation, dark sky
    np.testing.assert_allclose(g[sky_ring], out[sky_ring], atol=0.005)


def test_no_saturated_pixels_means_no_behaviour_change():
    """Scenes without clipped spots must be untouched by the new branch."""
    rng = np.random.default_rng(7)
    inp = rng.uniform(0.05, 0.5, (64, 64, 4)).astype(np.float32)
    out = np.clip(inp + rng.normal(0, 0.01, inp.shape).astype(np.float32), 0, 1)
    cfg = _highlight_guard_config()
    g_with = _apply_highlight_guard(out, inp, cfg)
    g_without = _apply_highlight_guard(out, inp, dict(cfg, sat_dilate=0))
    np.testing.assert_array_equal(g_with, g_without)
