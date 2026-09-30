"""Fixed CoreML pixel tiles preserve pixel order and colour parameters."""

import numpy as np
import pytest

from raw_alchemy.onnx import grade


@pytest.mark.parametrize("shape", [(3, 4, 3), (10, 12, 3), (19, 23, 3)])
def test_coreml_pixel_tiles_do_not_mix_pixels_or_padding(monkeypatch, shape):
    monkeypatch.setattr(grade, "COREML_PIXEL_TILE", (10, 12))
    rng = np.random.default_rng(27)
    img = rng.random(shape, dtype=np.float32)
    gain = np.float32(1.37)
    matrix = np.array([[1.2, 0.1, -0.1], [0.02, 0.9, 0.05], [0.1, 0.0, 1.1]], np.float32)
    seen = []

    class Session:
        _rawalchemy_pixel_tile = (10, 12)

        def get_providers(self):
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        def run(self, output_names, feeds):
            seen.append(feeds["img"].shape)
            assert feeds["mat"] is matrix
            return [(feeds["img"] * feeds["gain"]) @ feeds["mat"].T]

    result = grade._run_strips(Session(), {"img": img, "gain": gain, "mat": matrix})

    assert seen == [(10, 12, 3)] * (-(-shape[0] * shape[1] // 120))
    assert result.shape == shape
    np.testing.assert_allclose(result, (img * gain) @ matrix.T, atol=1e-7)
    np.testing.assert_array_equal(img, rng_image(shape))


def rng_image(shape):
    return np.random.default_rng(27).random(shape, dtype=np.float32)


@pytest.mark.parametrize("initial_provider", ["CPUExecutionProvider", "CoreMLExecutionProvider"])
def test_fixed_shape_survives_constructor_and_inference_cpu_fallback(initial_provider):
    class Session:
        _rawalchemy_pixel_tile = (10, 12)

        def __init__(self):
            self.provider = initial_provider
            self.calls = 0

        def get_providers(self):
            return [self.provider]

        def run(self, output_names, feeds):
            assert feeds["img"].shape == (10, 12, 3)
            self.provider = "CPUExecutionProvider"
            self.calls += 1
            return [feeds["img"] * np.float32(1.25)]

    session = Session()
    img = rng_image((19, 23, 3))
    for _ in range(2):
        np.testing.assert_array_equal(grade._run_strips(session, {"img": img}), img * np.float32(1.25))
    assert session.calls == 8
