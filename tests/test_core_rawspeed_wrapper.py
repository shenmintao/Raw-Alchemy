"""The application decode path must use the optional native-DLL guard."""
import sys
from types import SimpleNamespace

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


def test_frozen_rawspeed_uses_collected_native_assets(tmp_path, monkeypatch):
    from raw_alchemy import rawspeed
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / rawspeed._dll_name()).write_bytes(b"native placeholder")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert rawspeed._vendor_dir() == vendor
