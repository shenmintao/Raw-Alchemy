"""The application decode path must use the optional native-DLL guard."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_core_uses_safe_rawspeed_wrapper_before_rawpy_fallback(monkeypatch):
    from raw_alchemy import core, rawspeed

    calls = []
    class FallbackReached(Exception):
        pass

    def safe_decode(path):
        calls.append(path)
        return None  # wrapper reports missing native DLL without constructing RawSpeed

    def rawpy_fallback(path):
        raise FallbackReached(path)

    monkeypatch.setattr(rawspeed, "try_decode", safe_decode)
    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=rawpy_fallback))
    with pytest.raises(FallbackReached):
        core._rawpy_decode_to_prophoto("missing-native-library.raw")
    assert calls == ["missing-native-library.raw"]


# Layouts of RawSpeed's uncropped frame, verified pixel-for-pixel against
# LibRaw on real files (the X-S10 in both compressed and uncompressed mode).
X_T10 = np.array([
    [0, 2, 1, 2, 0, 1],
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
], dtype=np.uint8)
X_S10 = np.roll(X_T10, -1, axis=0)  # the old hard-coded pattern


def _xtrans_mosaic(pattern, blocks=12, seed=0):
    rng = np.random.default_rng(seed)
    level = np.array([0.30, 0.55, 0.22])[np.tile(pattern, (blocks, blocks))]
    return (1023 + 8000 * level + rng.normal(0, 40, level.shape)).astype(np.uint16)


class _XTransResult:
    is_bayer = False
    is_xtrans = True
    filters = 9
    color_matrix = np.eye(3)
    black_levels = [1023, 1023, 1023, 1023]
    white_level = 16383
    wb_coeffs = [526.0, 302.0, 569.0, 0.0]

    def __init__(self, make, model, bayer):
        self.make, self.model, self.bayer = make, model, bayer


def test_xtrans_layouts_come_from_cameras_xml_per_model():
    from raw_alchemy import rawspeed

    result = lambda model: SimpleNamespace(make="FUJIFILM", model=model)  # noqa: E731
    assert np.array_equal(rawspeed.xtrans_pattern(result("X-T10")), X_T10)
    assert np.array_equal(rawspeed.xtrans_pattern(result("X-S10")), X_S10)
    assert rawspeed.xtrans_pattern(result("NOT-A-CAMERA")) is None


def test_green_check_rejects_a_phase_shifted_layout():
    from raw_alchemy import rawspeed

    mosaic = _xtrans_mosaic(X_T10)
    assert rawspeed.xtrans_greens_consistent(mosaic, X_T10)
    # The X-T10 bug: the right layout one row off.
    assert not rawspeed.xtrans_greens_consistent(mosaic, X_S10)


def test_unknown_xtrans_model_is_decoded_by_libraw(monkeypatch):
    from raw_alchemy import core, rawspeed

    class FallbackReached(Exception):
        pass

    class Unknown(_XTransResult):
        @property
        def bayer(self):
            raise AssertionError("an unverified X-Trans frame must not be demosaiced")

    monkeypatch.setattr(rawspeed, "try_decode", lambda _path: Unknown("FUJIFILM", "X-NEW", None))
    monkeypatch.setattr(core, "_read_orientation_flip", lambda _path: 0)

    def rawpy_imread(path):
        raise FallbackReached(path)

    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=rawpy_imread))
    with pytest.raises(FallbackReached):
        core._rawpy_decode_to_prophoto("x-new.raf")


def test_known_xtrans_model_uses_rawspeed_with_its_own_layout(monkeypatch):
    from raw_alchemy import core, rawspeed
    from raw_alchemy.onnx import xtrans_demosaic

    mosaic = _xtrans_mosaic(X_T10)
    monkeypatch.setattr(
        rawspeed, "try_decode", lambda _path: _XTransResult("FUJIFILM", "X-T10", mosaic)
    )
    monkeypatch.setattr(core, "_read_orientation_flip", lambda _path: 0)
    seen = []

    def fake_demosaic(raw_norm, pattern):
        seen.append(np.asarray(pattern).copy())
        return np.zeros(raw_norm.shape + (3,), np.float32)

    monkeypatch.setattr(xtrans_demosaic, "xtrans_markesteijn_demosaic", fake_demosaic)
    core._rawpy_decode_to_prophoto("x-t10.raf")
    assert len(seen) == 1
    assert np.array_equal(seen[0], X_T10)


def test_frozen_rawspeed_uses_collected_native_assets(tmp_path, monkeypatch):
    from raw_alchemy import rawspeed
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / rawspeed._dll_name()).write_bytes(b"native placeholder")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert rawspeed._vendor_dir() == vendor
