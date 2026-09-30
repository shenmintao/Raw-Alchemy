"""Standard demosaicing must not apply heuristic sensor-defect removal."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from raw_alchemy.onnx.xtrans_demosaic import CANONICAL_PATTERN


@pytest.mark.parametrize("pattern", [np.array([[0, 1], [3, 2]]), CANONICAL_PATTERN],
                         ids=["bayer", "xtrans"])
def test_canonical_decode_preserves_fine_sensor_detail(monkeypatch, pattern):
    from raw_alchemy import core, rawspeed
    from raw_alchemy.onnx import rcd_demosaic, xtrans_demosaic

    source = np.full((120, 120), 16000, np.uint16)
    source[48:51, 48:51] = 32000  # small bright petal, not a sensor defect
    source[66:69, 66:69] = 2000   # small dark scene feature

    class Raw:
        raw_pattern = pattern
        raw_image_visible = source
        black_level_per_channel = [0] * 4
        white_level = 65535
        camera_whitebalance = [1, 1, 1, 1]
        rgb_xyz_matrix = np.eye(4, 3)
        sizes = SimpleNamespace(flip=0)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(rawspeed, "try_decode", lambda _path: None)
    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=lambda _path: Raw()))

    def forbidden_correction(*_args, **_kwargs):
        raise AssertionError("normal decode must preserve the sensor's fine detail")

    seen = []

    def demosaic(raw, cfa):
        seen.append(raw.copy())
        np.testing.assert_array_equal(cfa, pattern)
        return np.repeat(raw[..., None], 3, axis=2)

    monkeypatch.setattr(core, "fix_hot_pixels", forbidden_correction)
    monkeypatch.setattr(rcd_demosaic, "rcd_demosaic", demosaic)
    monkeypatch.setattr(xtrans_demosaic, "xtrans_markesteijn_demosaic", demosaic)

    core._rawpy_decode_to_prophoto("fine-detail.raw")

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], source.astype(np.float32) / 65535.0)
