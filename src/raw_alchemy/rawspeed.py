"""RawSpeed integration helpers.

The rawspeedpy wheel/source install used by the app does not always include
its vendor DLL and cameras.xml data. Raw Alchemy ships those files itself, so
configure rawspeedpy to load them from this package before constructing the
decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


XTRANS_PATTERN = np.array([
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
    [0, 2, 1, 2, 0, 1],
], dtype=np.uint8)


_decoder = None
_disabled = False
_binding_patched = False


def _vendor_dir() -> Path:
    package_vendor = Path(__file__).resolve().parent / "vendor"
    # PyInstaller specs collect native assets at _MEIPASS/vendor.
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled_vendor = Path(sys._MEIPASS) / "vendor"
        if (bundled_vendor / _dll_name()).is_file():
            return bundled_vendor
    return package_vendor


def _dll_name() -> str:
    if sys.platform == "win32":
        return "rawspeed_capi.dll"
    if sys.platform == "darwin":
        return "librawspeed_capi.dylib"
    return "librawspeed_capi.so"


def _patch_binding(binding) -> Optional[str]:
    """Point rawspeedpy at Raw Alchemy's packaged RawSpeed runtime files."""
    global _binding_patched

    vendor = _vendor_dir()
    dll_path = vendor / _dll_name()
    cameras_xml = vendor / "cameras.xml"

    if not dll_path.is_file() or not cameras_xml.is_file():
        logger.debug(
            f"RawSpeed runtime files missing: dll={dll_path}, cameras={cameras_xml}"
        )
        return None

    if not _binding_patched:
        original_close = binding.RawSpeedDecoder.close

        def safe_close(self):
            if not hasattr(self, "_handle"):
                return
            original_close(self)

        def safe_del(self):
            try:
                safe_close(self)
            except Exception:
                pass

        binding.RawSpeedDecoder.close = safe_close
        binding.RawSpeedDecoder.__del__ = safe_del
        binding._find_dll = lambda: str(dll_path)
        _binding_patched = True

    return str(cameras_xml)


def try_decode(path: str):
    """Decode a RAW via RawSpeed when available, otherwise return None."""
    global _decoder, _disabled

    if _disabled:
        return None

    try:
        import rawspeedpy._binding as binding

        cameras_xml = _patch_binding(binding)
        if cameras_xml is None:
            _disabled = True
            return None

        if _decoder is None:
            _decoder = binding.RawSpeedDecoder(cameras_xml=cameras_xml)

        return _decoder.decode(path)
    except Exception as exc:
        if _decoder is None:
            _disabled = True
            logger.debug(f"RawSpeed unavailable, using rawpy fallback: {exc}")
        else:
            logger.debug(f"RawSpeed decode failed, using rawpy fallback: {exc}")
        return None
