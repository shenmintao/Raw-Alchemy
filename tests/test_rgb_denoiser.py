"""SCUNet RGB denoiser wrapper: gain/gamma round-trip, tiling, passthrough.

The ONNX model itself is exercised with a stub session (identity / constant),
so these tests cover the wrapper's contract without loading the 99MB model.
"""
import numpy as np
import pytest

from raw_alchemy.onnx import rgb_denoiser as R


class _StubSession:
    """Stands in for an ort.InferenceSession; applies `fn` per tile (NCHW)."""

    def __init__(self, fn):
        self._fn = fn

    def get_inputs(self):
        class _I:  # minimal shim
            name = "x"
        return [_I()]

    def run(self, _outs, feeds):
        x = feeds["x"]
        return [self._fn(x)]


@pytest.fixture
def identity_session(monkeypatch):
    monkeypatch.setattr(R, "_get_session", lambda: _StubSession(lambda x: x))


def test_identity_model_round_trips_the_image(identity_session):
    rng = np.random.default_rng(0)
    img = rng.uniform(0.0, 0.02, (600, 900, 3)).astype(np.float32)  # dark image
    out = R.denoise_rgb_linear(img)
    assert out.shape == img.shape and out.dtype == np.float32
    # gain*gamma encode/decode with an identity model must return the input
    np.testing.assert_allclose(out, img, atol=2e-6)


def test_saturated_highlights_pass_through(monkeypatch):
    # model zeroes everything: without passthrough, clipped lamps would go black
    monkeypatch.setattr(R, "_get_session", lambda: _StubSession(np.zeros_like))
    img = np.full((520, 520, 3), 0.001, np.float32)  # gain will hit GAIN_MAX
    img[10:20, 10:20] = 1.0  # saturated highlight
    out = R.denoise_rgb_linear(img)
    np.testing.assert_array_equal(out[12:18, 12:18], img[12:18, 12:18])
    assert float(out[100:200, 100:200].max()) == 0.0  # rest followed the model


def test_smaller_than_tile_image_is_padded_and_cropped(identity_session):
    img = np.random.default_rng(1).uniform(0, 0.5, (200, 300, 3)).astype(np.float32)
    out = R.denoise_rgb_linear(img)
    assert out.shape == (200, 300, 3)
    np.testing.assert_allclose(out, img, atol=2e-6)


def test_denoising_effect_survives_decoding(monkeypatch):
    """A model that smooths must yield a smoothed linear image (not just an
    encoded-space effect that the round trip cancels)."""
    def box3(x):  # NCHW box blur via numpy roll
        acc = np.zeros_like(x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(x, dy, axis=2), dx, axis=3)
        return acc / 9.0
    monkeypatch.setattr(R, "_get_session", lambda: _StubSession(box3))
    rng = np.random.default_rng(2)
    img = np.clip(0.05 + rng.normal(0, 0.02, (512, 512, 3)), 0, 1).astype(np.float32)
    out = R.denoise_rgb_linear(img)
    assert float(out[8:-8, 8:-8].std()) < float(img[8:-8, 8:-8].std()) * 0.7


def test_compute_gain_bounds():
    dark = np.full((8, 8, 3), 1e-5, np.float32)
    assert R.compute_gain(dark) == R.GAIN_MAX
    bright = np.full((8, 8, 3), 0.5, np.float32)
    assert R.compute_gain(bright) == 1.0  # never darkens
    mid = np.full((8, 8, 3), 0.09, np.float32)
    assert R.compute_gain(mid) == pytest.approx(2.0)
    assert R.compute_gain(np.zeros((8, 8, 3), np.float32)) == 1.0


def test_rejects_non_rgb_input(identity_session):
    with pytest.raises(ValueError):
        R.denoise_rgb_linear(np.zeros((64, 64, 4), np.float32))
