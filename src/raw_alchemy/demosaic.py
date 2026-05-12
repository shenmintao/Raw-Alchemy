"""
RCD (Ratio Corrected Demosaicing) — Taichi GPU implementation.

Translated from darktable's OpenCL kernels (demosaic_rcd.cl).
Original algorithm by Luis Sanz Rodríguez, darktable implementation
by the darktable developers (GPLv3).

This implementation is independent and licensed under AGPLv3+ as part
of Raw Alchemy Studio.

Input:  (H, W) float32 Bayer mosaic, black-level subtracted, normalized to [0, 1]
Output: (H, W, 3) float32 RGB

Usage:
    rgb = rcd_demosaic(bayer, filters)
"""

import taichi as ti
import numpy as np
from loguru import logger

EPS = 1e-5
EPSSQ = 1e-10


# ---------------------------------------------------------------------------
# CFA pattern helper
# ---------------------------------------------------------------------------

@ti.func
def FC(row: ti.i32, col: ti.i32, filters: ti.u32) -> ti.i32:
    """Get CFA color at (row, col). Returns 0=RED, 1=GREEN, 2=BLUE."""
    return ti.cast((filters >> ti.cast(((((row) << 1) & 14) + ((col) & 1)) << 1, ti.u32)) & 3, ti.i32)


@ti.func
def clipf(x: ti.f32) -> ti.f32:
    return ti.min(ti.max(x, 0.0), 1.0)


@ti.func
def fsquare(x: ti.f32) -> ti.f32:
    return x * x


# ---------------------------------------------------------------------------
# Step 0: Populate CFA and RGB buffers
# ---------------------------------------------------------------------------

@ti.kernel
def _rcd_populate(
    bayer: ti.types.ndarray(dtype=ti.f32, ndim=2),
    cfa: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb0: ti.types.ndarray(dtype=ti.f32, ndim=2),  # RED channel
    rgb1: ti.types.ndarray(dtype=ti.f32, ndim=2),  # GREEN channel
    rgb2: ti.types.ndarray(dtype=ti.f32, ndim=2),  # BLUE channel
    filters: ti.u32,
):
    for row, col in ti.ndrange(bayer.shape[0], bayer.shape[1]):
        val = ti.max(0.0, bayer[row, col])
        color = FC(row, col, filters)
        cfa[row, col] = val
        if color == 0:
            rgb0[row, col] = val
        elif color == 1:
            rgb1[row, col] = val
        else:
            rgb2[row, col] = val


# ---------------------------------------------------------------------------
# Step 1: Vertical/Horizontal directional discrimination
# ---------------------------------------------------------------------------

@ti.kernel
def _rcd_step1(
    cfa: ti.types.ndarray(dtype=ti.f32, ndim=2),
    VH_dir: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.i32, h: ti.i32,
):
    """Compute VH directional discrimination from high-pass filters."""
    for row, col in ti.ndrange((4, h - 4), (4, w - 4)):
        # Vertical high-pass (3-point neighborhood)
        V_Stat = EPSSQ
        for dr in range(-1, 2):
            r = row + dr
            v = (cfa[r - 3, col] - cfa[r - 1, col] - cfa[r + 1, col] + cfa[r + 3, col]
                 - 3.0 * (cfa[r - 2, col] + cfa[r + 2, col]) + 6.0 * cfa[r, col])
            V_Stat += v * v
        V_Stat = ti.max(EPSSQ, V_Stat)

        # Horizontal high-pass (3-point neighborhood)
        H_Stat = EPSSQ
        for dc in range(-1, 2):
            c = col + dc
            hv = (cfa[row, c - 3] - cfa[row, c - 1] - cfa[row, c + 1] + cfa[row, c + 3]
                  - 3.0 * (cfa[row, c - 2] + cfa[row, c + 2]) + 6.0 * cfa[row, c])
            H_Stat += hv * hv
        H_Stat = ti.max(EPSSQ, H_Stat)

        VH_dir[row, col] = V_Stat / (V_Stat + H_Stat)


# ---------------------------------------------------------------------------
# Step 2: Low-pass filter at R/B positions
# ---------------------------------------------------------------------------

@ti.kernel
def _rcd_step2(
    cfa: ti.types.ndarray(dtype=ti.f32, ndim=2),
    lpf: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.i32, h: ti.i32,
    filters: ti.u32,
):
    """Low-pass filter at non-green CFA positions."""
    for row, col in ti.ndrange((2, h - 2), (2, w - 2)):
        # Only process R/B positions
        if FC(row, col, filters) != 1:  # Not green
            lpf[row, col] = (cfa[row, col]
                + 0.5 * (cfa[row - 1, col] + cfa[row + 1, col]
                         + cfa[row, col - 1] + cfa[row, col + 1])
                + 0.25 * (cfa[row - 1, col - 1] + cfa[row - 1, col + 1]
                          + cfa[row + 1, col - 1] + cfa[row + 1, col + 1]))


# ---------------------------------------------------------------------------
# Step 3: Green interpolation at R/B positions
# ---------------------------------------------------------------------------

@ti.kernel
def _rcd_step3(
    cfa: ti.types.ndarray(dtype=ti.f32, ndim=2),
    lpf: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb1: ti.types.ndarray(dtype=ti.f32, ndim=2),  # GREEN
    VH_dir: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.i32, h: ti.i32,
    filters: ti.u32,
):
    """Interpolate Green channel at Red/Blue CFA positions."""
    for row, col in ti.ndrange((4, h - 4), (4, w - 4)):
        if FC(row, col, filters) == 1:  # Skip green positions
            continue

        # Refined VH discrimination
        VH_Central = VH_dir[row, col]
        VH_Neigh = 0.25 * (VH_dir[row - 1, col - 1] + VH_dir[row - 1, col + 1]
                          + VH_dir[row + 1, col - 1] + VH_dir[row + 1, col + 1])
        VH_Disc = VH_Neigh if ti.abs(0.5 - VH_Central) < ti.abs(0.5 - VH_Neigh) else VH_Central

        cfai = cfa[row, col]

        # Cardinal gradients
        N_Grad = (EPS + ti.abs(cfa[row - 1, col] - cfa[row + 1, col])
                  + ti.abs(cfai - cfa[row - 2, col])
                  + ti.abs(cfa[row - 1, col] - cfa[row - 3, col])
                  + ti.abs(cfa[row - 2, col] - cfa[row - 4, col]))
        S_Grad = (EPS + ti.abs(cfa[row + 1, col] - cfa[row - 1, col])
                  + ti.abs(cfai - cfa[row + 2, col])
                  + ti.abs(cfa[row + 1, col] - cfa[row + 3, col])
                  + ti.abs(cfa[row + 2, col] - cfa[row + 4, col]))
        W_Grad = (EPS + ti.abs(cfa[row, col - 1] - cfa[row, col + 1])
                  + ti.abs(cfai - cfa[row, col - 2])
                  + ti.abs(cfa[row, col - 1] - cfa[row, col - 3])
                  + ti.abs(cfa[row, col - 2] - cfa[row, col - 4]))
        E_Grad = (EPS + ti.abs(cfa[row, col + 1] - cfa[row, col - 1])
                  + ti.abs(cfai - cfa[row, col + 2])
                  + ti.abs(cfa[row, col + 1] - cfa[row, col + 3])
                  + ti.abs(cfa[row, col + 2] - cfa[row, col + 4]))

        # Ratio-corrected estimations
        lfpi = lpf[row, col]
        N_Est = cfa[row - 1, col] * (2.0 * lfpi) / (EPS + lfpi + lpf[row - 2, col])
        S_Est = cfa[row + 1, col] * (2.0 * lfpi) / (EPS + lfpi + lpf[row + 2, col])
        W_Est = cfa[row, col - 1] * (2.0 * lfpi) / (EPS + lfpi + lpf[row, col - 2])
        E_Est = cfa[row, col + 1] * (2.0 * lfpi) / (EPS + lfpi + lpf[row, col + 2])

        V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad)
        H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad)

        # Interpolate
        d = clipf(VH_Disc)
        rgb1[row, col] = d * (H_Est - V_Est) + V_Est


# ---------------------------------------------------------------------------
# Step 4.0: P/Q diagonal high-pass filters
# ---------------------------------------------------------------------------

@ti.kernel
def _rcd_step4_0(
    cfa: ti.types.ndarray(dtype=ti.f32, ndim=2),
    p_diff: ti.types.ndarray(dtype=ti.f32, ndim=2),
    q_diff: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.i32, h: ti.i32,
):
    for row, col in ti.ndrange((3, h - 3), (3, w - 3)):
        p_diff[row, col] = fsquare(
            cfa[row - 3, col - 3] - cfa[row - 1, col - 1]
            - cfa[row + 1, col + 1] + cfa[row + 3, col + 3]
            - 3.0 * (cfa[row - 2, col - 2] + cfa[row + 2, col + 2])
            + 6.0 * cfa[row, col])
        q_diff[row, col] = fsquare(
            cfa[row - 3, col + 3] - cfa[row - 1, col + 1]
            - cfa[row + 1, col - 1] + cfa[row + 3, col - 3]
            - 3.0 * (cfa[row - 2, col + 2] + cfa[row + 2, col - 2])
            + 6.0 * cfa[row, col])


# ---------------------------------------------------------------------------
# Step 4.1: P/Q directional discrimination
# ---------------------------------------------------------------------------

@ti.kernel
def _rcd_step4_1(
    p_diff: ti.types.ndarray(dtype=ti.f32, ndim=2),
    q_diff: ti.types.ndarray(dtype=ti.f32, ndim=2),
    PQ_dir: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.i32, h: ti.i32,
    filters: ti.u32,
):
    for row, col in ti.ndrange((4, h - 4), (4, w - 4)):
        if FC(row, col, filters) == 1:
            continue
        P_Stat = ti.max(EPSSQ,
            p_diff[row - 1, col - 1] + p_diff[row, col] + p_diff[row + 1, col + 1])
        Q_Stat = ti.max(EPSSQ,
            q_diff[row - 1, col + 1] + q_diff[row, col] + q_diff[row + 1, col - 1])
        PQ_dir[row, col] = P_Stat / (P_Stat + Q_Stat)


# ---------------------------------------------------------------------------
# Step 4.2: R/B at R/B positions (opposite color)
# ---------------------------------------------------------------------------

@ti.func
def _rcd_step4_2_color(
    rgbc: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    PQ_dir: ti.types.ndarray(dtype=ti.f32, ndim=2),
    row: ti.i32, col: ti.i32,
):
    """Interpolate one color channel at an R/B position (opposite color)."""
    PQ_Central = PQ_dir[row, col]
    PQ_Neigh = 0.25 * (PQ_dir[row - 1, col - 1] + PQ_dir[row - 1, col + 1]
                      + PQ_dir[row + 1, col - 1] + PQ_dir[row + 1, col + 1])
    PQ_Disc = PQ_Neigh if ti.abs(0.5 - PQ_Central) < ti.abs(0.5 - PQ_Neigh) else PQ_Central

    g = rgb1[row, col]
    NW = rgbc[row - 1, col - 1]; NE = rgbc[row - 1, col + 1]
    SW = rgbc[row + 1, col - 1]; SE = rgbc[row + 1, col + 1]

    NW_Grad = EPS + ti.abs(NW - SE) + ti.abs(NW - rgbc[row - 3, col - 3]) + ti.abs(g - rgb1[row - 2, col - 2])
    NE_Grad = EPS + ti.abs(NE - SW) + ti.abs(NE - rgbc[row - 3, col + 3]) + ti.abs(g - rgb1[row - 2, col + 2])
    SW_Grad = EPS + ti.abs(NE - SW) + ti.abs(SW - rgbc[row + 3, col - 3]) + ti.abs(g - rgb1[row + 2, col - 2])
    SE_Grad = EPS + ti.abs(NW - SE) + ti.abs(SE - rgbc[row + 3, col + 3]) + ti.abs(g - rgb1[row + 2, col + 2])

    NW_Est = NW - rgb1[row - 1, col - 1]
    NE_Est = NE - rgb1[row - 1, col + 1]
    SW_Est = SW - rgb1[row + 1, col - 1]
    SE_Est = SE - rgb1[row + 1, col + 1]

    P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad)
    Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad)

    d = clipf(PQ_Disc)
    return g + d * (Q_Est - P_Est) + P_Est


@ti.kernel
def _rcd_step4_2(
    rgb0: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    PQ_dir: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.i32, h: ti.i32,
    filters: ti.u32,
):
    for row, col in ti.ndrange((4, h - 4), (4, w - 4)):
        fc = FC(row, col, filters)
        if fc == 1:
            continue
        color = 2 - fc  # opposite
        if color == 0:
            rgb0[row, col] = _rcd_step4_2_color(rgb0, rgb1, PQ_dir, row, col)
        else:
            rgb2[row, col] = _rcd_step4_2_color(rgb2, rgb1, PQ_dir, row, col)


# ---------------------------------------------------------------------------
# Step 4.3: R/B at Green positions
# ---------------------------------------------------------------------------

@ti.func
def _rcd_step4_3_color(
    rgbc: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    row: ti.i32, col: ti.i32,
    d: ti.f32, rgbi1: ti.f32,
    N1: ti.f32, S1: ti.f32, W1: ti.f32, E1: ti.f32,
    rgb1mw: ti.f32, rgb1pw: ti.f32, rgb1m1: ti.f32, rgb1p1: ti.f32,
):
    cn = rgbc[row - 1, col]; cs = rgbc[row + 1, col]
    cw = rgbc[row, col - 1]; ce = rgbc[row, col + 1]

    SNabs = ti.abs(cn - cs)
    EWabs = ti.abs(cw - ce)

    N_Grad = N1 + SNabs + ti.abs(cn - rgbc[row - 3, col])
    S_Grad = S1 + SNabs + ti.abs(cs - rgbc[row + 3, col])
    W_Grad = W1 + EWabs + ti.abs(cw - rgbc[row, col - 3])
    E_Grad = E1 + EWabs + ti.abs(ce - rgbc[row, col + 3])

    N_Est = cn - rgb1mw
    S_Est = cs - rgb1pw
    W_Est = cw - rgb1m1
    E_Est = ce - rgb1p1

    V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad)
    H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad)

    return rgbi1 + d * (H_Est - V_Est) + V_Est


@ti.kernel
def _rcd_step4_3(
    rgb0: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    VH_dir: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.i32, h: ti.i32,
    filters: ti.u32,
):
    for row, col in ti.ndrange((4, h - 4), (4, w - 4)):
        if FC(row, col, filters) != 1:
            continue

        VH_Central = VH_dir[row, col]
        VH_Neigh = 0.25 * (VH_dir[row - 1, col - 1] + VH_dir[row - 1, col + 1]
                          + VH_dir[row + 1, col - 1] + VH_dir[row + 1, col + 1])
        VH_Disc = VH_Neigh if ti.abs(0.5 - VH_Central) < ti.abs(0.5 - VH_Neigh) else VH_Central

        rgbi1 = rgb1[row, col]
        N1 = EPS + ti.abs(rgbi1 - rgb1[row - 2, col])
        S1 = EPS + ti.abs(rgbi1 - rgb1[row + 2, col])
        W1 = EPS + ti.abs(rgbi1 - rgb1[row, col - 2])
        E1 = EPS + ti.abs(rgbi1 - rgb1[row, col + 2])

        rgb1mw = rgb1[row - 1, col]
        rgb1pw = rgb1[row + 1, col]
        rgb1m1 = rgb1[row, col - 1]
        rgb1p1 = rgb1[row, col + 1]

        d = clipf(VH_Disc)

        rgb0[row, col] = _rcd_step4_3_color(
            rgb0, rgb1, row, col, d, rgbi1, N1, S1, W1, E1, rgb1mw, rgb1pw, rgb1m1, rgb1p1)
        rgb2[row, col] = _rcd_step4_3_color(
            rgb2, rgb1, row, col, d, rgbi1, N1, S1, W1, E1, rgb1mw, rgb1pw, rgb1m1, rgb1p1)


# ---------------------------------------------------------------------------
# Step 5: Write output
# ---------------------------------------------------------------------------

@ti.kernel
def _rcd_write_output(
    rgb0: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    border: ti.i32,
):
    h, w = rgb0.shape[0], rgb0.shape[1]
    for row, col in ti.ndrange((border, h - border), (border, w - border)):
        out[row, col, 0] = ti.max(0.0, rgb0[row, col])
        out[row, col, 1] = ti.max(0.0, rgb1[row, col])
        out[row, col, 2] = ti.max(0.0, rgb2[row, col])


# ---------------------------------------------------------------------------
# Simple bilinear fallback for border pixels
# ---------------------------------------------------------------------------

@ti.kernel
def _border_interpolate(
    bayer: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    border: ti.i32,
    filters: ti.u32,
):
    """Simple bilinear interpolation for border pixels only."""
    h, w = bayer.shape[0], bayer.shape[1]
    for row, col in ti.ndrange(h, w):
        # Only process border region
        if row >= border and row < h - border and col >= border and col < w - border:
            continue

        sums = ti.Vector([0.0, 0.0, 0.0])
        counts = ti.Vector([0.0, 0.0, 0.0])

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y = row + dy
                x = col + dx
                if 0 <= y < h and 0 <= x < w:
                    c = FC(y, x, filters)
                    val = ti.max(0.0, bayer[y, x])
                    sums[c] += val
                    counts[c] += 1.0

        for c in ti.static(range(3)):
            if counts[c] > 0.0:
                out[row, col, c] = sums[c] / counts[c]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dcraw_filters(cfa_pattern: np.ndarray) -> int:
    """Convert a 2x2 CFA pattern array to dcraw-style 32-bit filter code.

    Args:
        cfa_pattern: (2, 2) array with values 0=R, 1=G, 2=B, 3=G2

    Returns:
        32-bit dcraw filter code
    """
    # Map: 0=R, 1=G, 2=B, 3=G (treat G2 as G)
    color_map = {0: 0, 1: 1, 2: 2, 3: 1}

    # Build 8x2 repeating pattern
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


def rcd_demosaic(
    bayer: np.ndarray,
    filters: int = FILTERS_RGGB,
) -> np.ndarray:
    """Demosaic a Bayer image using the RCD algorithm (GPU-accelerated).

    Args:
        bayer: (H, W) float32, black-level subtracted, normalized to [0, 1]
        filters: dcraw-style 32-bit CFA filter code (default: RGGB)

    Returns:
        (H, W, 3) float32 RGB in [0, 1]
    """
    import time
    t0 = time.time()

    h, w = bayer.shape
    assert h >= 10 and w >= 10, f"Image too small for RCD: {w}x{h}"

    filters_u32 = np.uint32(filters)

    # Allocate GPU buffers — release early to minimize peak VRAM.
    # Peak: 7 × HW float32 + 1 × HWx3 float32
    # For 42MP: 7×170MB + 510MB ≈ 1.7GB
    cfa = ti.ndarray(dtype=ti.f32, shape=(h, w))
    rgb0 = ti.ndarray(dtype=ti.f32, shape=(h, w))
    rgb1 = ti.ndarray(dtype=ti.f32, shape=(h, w))
    rgb2 = ti.ndarray(dtype=ti.f32, shape=(h, w))
    VH_dir = ti.ndarray(dtype=ti.f32, shape=(h, w))
    lpf = ti.ndarray(dtype=ti.f32, shape=(h, w))
    out = ti.ndarray(dtype=ti.f32, shape=(h, w, 3))

    cfa.from_numpy(np.ascontiguousarray(bayer, dtype=np.float32))

    # Step 0-3: Green channel interpolation
    _rcd_populate(cfa, cfa, rgb0, rgb1, rgb2, filters_u32)
    _rcd_step1(cfa, VH_dir, w, h)
    _rcd_step2(cfa, lpf, w, h, filters_u32)
    _rcd_step3(cfa, lpf, rgb1, VH_dir, w, h, filters_u32)

    # Free lpf, allocate p_diff and q_diff (reuse lpf slot)
    del lpf
    p_diff = ti.ndarray(dtype=ti.f32, shape=(h, w))
    q_diff = ti.ndarray(dtype=ti.f32, shape=(h, w))

    # Step 4.0-4.1: Diagonal discrimination
    _rcd_step4_0(cfa, p_diff, q_diff, w, h)

    # Free cfa (no longer needed), allocate PQ_dir
    del cfa
    PQ_dir = ti.ndarray(dtype=ti.f32, shape=(h, w))
    _rcd_step4_1(p_diff, q_diff, PQ_dir, w, h, filters_u32)

    # Free p_diff, q_diff
    del p_diff, q_diff

    # Step 4.2-4.3: R/B interpolation
    _rcd_step4_2(rgb0, rgb1, rgb2, PQ_dir, w, h, filters_u32)
    del PQ_dir
    _rcd_step4_3(rgb0, rgb1, rgb2, VH_dir, w, h, filters_u32)
    del VH_dir

    # Step 5: Write output (skip 4-pixel border)
    _rcd_write_output(rgb0, rgb1, rgb2, out, 4)
    del rgb0, rgb1, rgb2

    # Border: simple bilinear — reuse rgb0 slot for bayer upload
    bayer_arr = ti.ndarray(dtype=ti.f32, shape=(h, w))
    bayer_arr.from_numpy(np.ascontiguousarray(bayer, dtype=np.float32))
    _border_interpolate(bayer_arr, out, 4, filters_u32)
    del bayer_arr

    result = out.to_numpy()
    del out

    # Force GPU memory release
    import gc; gc.collect()

    elapsed = time.time() - t0
    logger.info(f"RCD demosaic: {w}x{h} in {elapsed*1000:.0f}ms (GPU)")

    return result
