"""
RawSpeed Python binding via ctypes.

Wraps the rawspeed3_capi DLL to decode RAW files without rawpy.
Provides: raw sensor data, CFA pattern, black/white levels, WB coefficients,
color matrix, and camera metadata.

Usage:
    from raw_alchemy.rawspeed_binding import RawSpeedDecoder

    decoder = RawSpeedDecoder()  # loads cameras.xml
    result = decoder.decode("photo.ARW")

    print(result.width, result.height)
    print(result.bayer)          # (H, W) uint16 sensor data
    print(result.filters)        # dcraw CFA filter code
    print(result.black_levels)   # [R, G, B, G2] per-channel
    print(result.white_level)
    print(result.wb_coeffs)      # [R, G, B, G2] camera WB
    print(result.color_matrix)   # 3x3 or 4x3 XYZ→Camera
    print(result.make, result.model)
    print(result.iso_speed)
"""

import os
import sys
import ctypes
import ctypes.wintypes
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ---------------------------------------------------------------------------
# C API structures (must match rawspeed3_capi.h)
# ---------------------------------------------------------------------------

class _RSResult(ctypes.Structure):
    """Mirrors rs_result_t from rawspeed_capi.h."""
    _fields_ = [
        ("status", ctypes.c_int),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("bpp", ctypes.c_uint32),
        ("cpp", ctypes.c_uint32),
        ("pitch", ctypes.c_uint32),
        ("filters", ctypes.c_uint32),
        ("pixeldata", ctypes.c_void_p),
        ("black_level", ctypes.c_int32),
        ("black_level_separate", ctypes.c_int32 * 4),
        ("white_point", ctypes.c_int32),
        ("has_wb", ctypes.c_int),
        ("wb_coeffs", ctypes.c_float * 4),
        ("color_matrix_elems", ctypes.c_int),
        ("color_matrix", ctypes.c_float * 12),
        ("make", ctypes.c_char * 128),
        ("model", ctypes.c_char * 128),
        ("mode", ctypes.c_char * 64),
        ("iso_speed", ctypes.c_int),
    ]


@dataclass
class RawDecodeResult:
    """Decoded RAW image data."""
    width: int
    height: int
    bayer: np.ndarray           # (H, W) uint16 raw sensor data
    filters: int                # dcraw-style CFA filter code
    bpp: int                    # bytes per pixel
    cpp: int                    # components per pixel
    black_levels: list          # [R, G, B, G2] per-channel black levels
    white_level: int            # white point / saturation
    wb_coeffs: list             # [R, G, B, G2] camera white balance
    color_matrix: Optional[np.ndarray]  # XYZ→Camera matrix (4x3 or 3x3)
    iso_speed: int
    make: str
    model: str

    @property
    def is_xtrans(self) -> bool:
        return self.filters == 9

    @property
    def is_bayer(self) -> bool:
        return not self.is_xtrans and self.filters != 0

    def normalize(self) -> np.ndarray:
        """Black-level subtract and normalize to [0, 1] float32.

        Returns (H, W) float32 normalized Bayer data.
        """
        img = self.bayer.astype(np.float32)
        bl = float(self.black_levels[0])  # use first channel as reference
        wl = float(self.white_level)
        return np.maximum(img - bl, 0) / (wl - bl)


# ---------------------------------------------------------------------------
# DLL loader
# ---------------------------------------------------------------------------

_dll = None
_dll_path = None


def _find_dll() -> str:
    """Find the rawspeed3 DLL."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "vendor", "rawspeed_capi.dll"),
    ]

    for p in candidates:
        np_ = os.path.normpath(p)
        if os.path.isfile(np_):
            return np_

    raise FileNotFoundError(
        f"rawspeed3 DLL not found. Searched: {candidates}")


def _load_dll():
    """Load the rawspeed3 DLL and set up function signatures."""
    global _dll, _dll_path
    if _dll is not None:
        return _dll

    _dll_path = _find_dll()
    logger.info(f"Loading RawSpeed DLL from: {_dll_path}")

    # Add DLL directory to search path
    dll_dir = os.path.dirname(_dll_path)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(dll_dir)
    os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

    _dll = ctypes.CDLL(_dll_path)

    # Set up function signatures
    _dll.rs_init.argtypes = [ctypes.c_char_p]
    _dll.rs_init.restype = ctypes.c_void_p

    _dll.rs_decode.argtypes = [
        ctypes.c_void_p,              # handle
        ctypes.POINTER(_RSResult),    # result
        ctypes.c_void_p,              # data
        ctypes.c_size_t,              # size
        ctypes.c_int,                 # allow_unknown
    ]
    _dll.rs_decode.restype = ctypes.c_int

    _dll.rs_release.argtypes = [ctypes.c_void_p]
    _dll.rs_release.restype = None

    _dll.rs_close.argtypes = [ctypes.c_void_p]
    _dll.rs_close.restype = None

    return _dll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RawSpeedDecoder:
    """High-level Python wrapper for RawSpeed RAW decoder."""

    def __init__(self, cameras_xml: Optional[str] = None):
        """Initialize RawSpeed with camera definitions.

        Args:
            cameras_xml: Path to cameras.xml. If None, uses bundled default.
        """
        dll = _load_dll()

        if not cameras_xml:
            cameras_xml = os.path.join(
                os.path.dirname(__file__), "vendor", "cameras.xml")

        if not os.path.isfile(cameras_xml):
            raise FileNotFoundError(f"cameras.xml not found: {cameras_xml}")

        self._handle = dll.rs_init(cameras_xml.encode('utf-8'))
        if not self._handle:
            raise RuntimeError("Failed to initialize RawSpeed")

        logger.info(f"RawSpeed decoder initialized (cameras.xml: {cameras_xml})")

    def decode(self, raw_path: str, allow_unknown: bool = True) -> RawDecodeResult:
        """Decode a RAW file.

        Args:
            raw_path: Path to RAW file (.ARW, .CR3, .NEF, .DNG, .RAF, etc.)
            allow_unknown: Allow decoding of cameras not in cameras.xml

        Returns:
            RawDecodeResult with sensor data and metadata
        """
        dll = _load_dll()

        # Read file into memory (RawSpeed decodes from memory)
        with open(raw_path, 'rb') as f:
            data = f.read()

        result = _RSResult()

        status = dll.rs_decode(
            self._handle,
            ctypes.byref(result),
            data,
            len(data),
            1 if allow_unknown else 0,
        )

        if status < 0:
            raise ValueError(f"Invalid parameters for RawSpeed decode: {raw_path}")
        if status >= 2:
            raise RuntimeError(
                f"RawSpeed decode failed (status={status}): {raw_path}")

        # Extract pixel data
        w, h = result.width, result.height
        bpp = result.bpp
        cpp = result.cpp
        pitch = result.pitch

        if not result.pixeldata:
            raise RuntimeError("RawSpeed returned null pixel data")

        # Copy pixel data to numpy (RawSpeed owns the buffer, we must copy)
        if bpp == 2 and cpp == 1:
            # Most common: 16-bit Bayer
            row_pixels = pitch // 2
            raw_buf = (ctypes.c_uint16 * (h * row_pixels)).from_address(
                result.pixeldata)
            bayer = np.ctypeslib.as_array(raw_buf).reshape(h, row_pixels)[:, :w].copy()
        elif bpp == 4 and cpp == 1:
            # Float32
            row_pixels = pitch // 4
            raw_buf = (ctypes.c_float * (h * row_pixels)).from_address(
                result.pixeldata)
            bayer = np.ctypeslib.as_array(raw_buf).reshape(h, row_pixels)[:, :w].copy()
        else:
            raise RuntimeError(f"Unsupported pixel format: bpp={bpp}, cpp={cpp}")

        # Extract metadata
        black_levels = list(result.black_level_separate)
        if all(bl == 0 for bl in black_levels) and result.black_level > 0:
            black_levels = [result.black_level] * 4

        wb = list(result.wb_coeffs) if result.has_wb else [1.0, 1.0, 1.0, 1.0]
        cm_elems = result.color_matrix_elems
        if cm_elems > 0:
            cm_flat = list(result.color_matrix[:cm_elems])
            rows = cm_elems // 3
            color_matrix = np.array(cm_flat).reshape(rows, 3)
        else:
            color_matrix = None

        decode_result = RawDecodeResult(
            width=w,
            height=h,
            bayer=bayer,
            filters=result.filters,
            bpp=bpp,
            cpp=cpp,
            black_levels=black_levels,
            white_level=result.white_point if result.white_point > 0 else 65535,
            wb_coeffs=wb,
            color_matrix=color_matrix,
            iso_speed=result.iso_speed,
            make=result.make.decode('utf-8', errors='replace').strip('\x00'),
            model=result.model.decode('utf-8', errors='replace').strip('\x00'),
        )

        # Release internal buffer (but keep handle alive)
        dll.rs_release(self._handle)

        logger.debug(
            f"Decoded {os.path.basename(raw_path)}: {w}x{h} "
            f"filters=0x{result.filters:08x} BL={black_levels} WL={decode_result.white_level}")

        return decode_result

    def close(self):
        """Release all resources."""
        if self._handle:
            dll = _load_dll()
            dll.rs_close(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# X-Trans 6x6 CFA pattern (universal across all Fuji X-Trans sensors)
XTRANS_PATTERN = np.array([
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
    [0, 2, 1, 2, 0, 1],
], dtype=np.uint8)

_decoder = None

def try_decode(path: str) -> Optional[RawDecodeResult]:
    global _decoder
    try:
        if _decoder is None:
            _decoder = RawSpeedDecoder()
        return _decoder.decode(path)
    except Exception:
        return None
