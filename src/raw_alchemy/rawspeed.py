"""RawSpeed integration helpers.

The rawspeedpy wheel/source install used by the app does not always include
its vendor DLL and cameras.xml data. Raw Alchemy ships those files itself, so
configure rawspeedpy to load them from this package before constructing the
decoder.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


_CFA_COLOURS = {"R": 0, "G": 1, "B": 2}

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


@lru_cache(maxsize=1)
def _xtrans_cfa_table() -> dict:
    """(make, model) -> list of 6x6 CFAs declared in the bundled cameras.xml."""
    table: dict = {}
    try:
        root = ET.parse(_vendor_dir() / "cameras.xml").getroot()
    except (OSError, ET.ParseError) as exc:
        logger.debug(f"cameras.xml unreadable, X-Trans falls back to LibRaw: {exc}")
        return table
    for camera in root.iter("Camera"):
        cfa = camera.find("CFA2")
        if cfa is None or cfa.get("width") != "6" or cfa.get("height") != "6":
            continue
        rows = sorted(cfa.findall("ColorRow"), key=lambda row: int(row.get("y", "0")))
        try:
            pattern = np.array(
                [[_CFA_COLOURS[c] for c in (row.text or "").strip()] for row in rows],
                dtype=np.uint8,
            )
        except KeyError:
            continue
        if pattern.shape == (6, 6):
            table.setdefault((camera.get("make"), camera.get("model")), []).append(pattern)
    return table


def xtrans_pattern(result) -> Optional[np.ndarray]:
    """6x6 CFA of an X-Trans frame returned by :func:`try_decode`, or None.

    The C API decodes with applyCrop=false and exports only the dcraw filter
    code, which is always 9 for X-Trans, so the layout comes from the same
    cameras.xml RawSpeed decoded with. For the uncropped frame the CFA2 rows
    apply as listed (checked pixel-for-pixel against LibRaw on the X-T10 and
    the X-S10, compressed and uncompressed). The binding does not report the
    mode, so every mode of the camera must declare the same layout.
    """
    patterns = _xtrans_cfa_table().get((result.make, result.model))
    if not patterns or any(not np.array_equal(p, patterns[0]) for p in patterns):
        return None
    return patterns[0].copy()


def xtrans_greens_consistent(mosaic: np.ndarray, pattern: np.ndarray) -> bool:
    """Cheap guard that ``pattern`` puts its greens on the green photosites.

    Green sites are the brightest raw channel in ordinary scenes, so among all
    36 shifts of the layout the right phase maximizes mean(G) - mean(R, B).
    A row or column offset (the X-T10 failure) scores lower; an R/B swap
    cannot be seen here, which is why the layout itself must come from
    cameras.xml. Dark or strongly red scenes may fail the check, which only
    costs the LibRaw fallback.
    """
    height, width = mosaic.shape[:2]
    rows, cols = height // 6, width // 6
    if rows < 4 or cols < 4:
        return False
    step = max(1, rows // 64)  # ~64 block-rows is plenty for the means
    blocks = mosaic[: rows * 6, : cols * 6].reshape(rows, 6, cols, 6)[::step]
    site_means = blocks.mean(axis=(0, 2), dtype=np.float64)  # (6, 6)
    green = pattern == 1

    def score(mask):
        return site_means[mask].mean() - site_means[~mask].mean()

    own = score(green)
    best = max(
        score(np.roll(np.roll(green, dy, 0), dx, 1))
        for dy in range(6) for dx in range(6)
    )
    return own > 0 and own >= best - 1e-9 * max(1.0, abs(best))


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
