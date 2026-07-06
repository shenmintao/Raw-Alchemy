"""CFA pattern/filter-code helpers (pure Python, no GPU runtime).

Moved out of the retired Taichi demosaic module; the demosaic algorithms
themselves now run on the ONNX runtime (onnx/rcd_demosaic.py,
onnx/xtrans_demosaic.py).
"""

import numpy as np


def get_dcraw_filters(cfa_pattern: np.ndarray) -> int:
    """Convert a 2x2 CFA pattern array to dcraw-style 32-bit filter code.

    Args:
        cfa_pattern: (2, 2) array with values 0=R, 1=G, 2=B, 3=G2

    Returns:
        32-bit dcraw filter code
    """
    color_map = {0: 0, 1: 1, 2: 2, 3: 1}
    filters = 0
    for row in range(8):
        for col in range(2):
            color = color_map[int(cfa_pattern[row % 2, col % 2])]
            bit_pos = (((row << 1) & 14) + (col & 1)) << 1
            filters |= (color << bit_pos)
    return filters


def get_cfa_pattern_from_filters(filters: int) -> np.ndarray:
    """Convert dcraw-style 32-bit filter code to 2x2 CFA pattern array."""
    pattern = np.zeros((2, 2), dtype=np.uint8)
    for r in range(2):
        for c in range(2):
            pattern[r, c] = (filters >> (((r << 1 & 14) | (c & 1)) << 1)) & 3
    return pattern


# Common CFA filter codes
FILTERS_RGGB = 0x94949494  # Sony, Canon, Nikon (most common)
FILTERS_BGGR = 0x16161616
FILTERS_GRBG = 0x61616161
FILTERS_GBRG = 0x49494949
