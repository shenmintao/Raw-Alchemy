"""
X-Trans Markesteijn Demosaicing — Taichi GPU implementation.

Ported from darktable's xtrans.c (Frank Markesteijn's algorithm, 1-pass).
Original darktable implementation GPLv3, this port AGPLv3+ as part of
Raw Alchemy Studio.

Input:  (H, W) float32 X-Trans mosaic, black-level subtracted, normalized to [0, 1]
Output: (H, W, 3) float32 RGB

Usage:
    rgb = xtrans_markesteijn_demosaic(raw, xtrans_pattern)
"""

import taichi as ti
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BORDER = 12  # 1-pass border width
NDIR = 4     # number of interpolation directions (1-pass)


# ---------------------------------------------------------------------------
# Helper ti.func
# ---------------------------------------------------------------------------

@ti.func
def FCxt(row: ti.i32, col: ti.i32,
         xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2)) -> ti.i32:
    return xtrans[(row + 600) % 6, (col + 600) % 6]


@ti.func
def clampf(x: ti.f32, lo: ti.f32, hi: ti.f32) -> ti.f32:
    return ti.min(ti.max(x, lo), hi)


@ti.func
def sqrf(x: ti.f32) -> ti.f32:
    return x * x


@ti.func
def ahdr(allhex_dr: ti.types.ndarray(dtype=ti.i32, ndim=1),
         r3: ti.i32, c3: ti.i32, entry: ti.i32) -> ti.i32:
    return allhex_dr[r3 * 24 + c3 * 8 + entry]


@ti.func
def ahdc(allhex_dc: ti.types.ndarray(dtype=ti.i32, ndim=1),
         r3: ti.i32, c3: ti.i32, entry: ti.i32) -> ti.i32:
    return allhex_dc[r3 * 24 + c3 * 8 + entry]


# ---------------------------------------------------------------------------
# allhex precomputation (CPU)
# ---------------------------------------------------------------------------

def _build_allhex(xtrans_pattern: np.ndarray):
    """Build hexagonal neighbor lookup and find solitary green position.

    Returns:
        allhex_dr: (72,) int32 — flattened [3][3][8] delta-row offsets
        allhex_dc: (72,) int32 — flattened [3][3][8] delta-col offsets
        sgrow: int — solitary-green row offset (0-2)
        sgcol: int — solitary-green col offset (0-2)
    """
    orth = [1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1]
    patt = [
        [0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0],
        [0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1],
    ]

    allhex_dr = np.zeros(72, dtype=np.int32)
    allhex_dc = np.zeros(72, dtype=np.int32)
    sgrow, sgcol = 0, 0
    xt = xtrans_pattern

    def _fc(r, c):
        return int(xt[(r + 600) % 6, (c + 600) % 6])

    for row in range(3):
        for col in range(3):
            ng = 0
            for d_idx in range(0, 10, 2):
                g = 1 if _fc(row, col) == 1 else 0
                if _fc(row + orth[d_idx], col + orth[d_idx + 2]) == 1:
                    ng = 0
                else:
                    ng += 1
                if ng == 4:
                    sgrow, sgcol = row, col
                if ng == g + 1:
                    for c in range(8):
                        v = orth[d_idx] * patt[g][c * 2] + orth[d_idx + 1] * patt[g][c * 2 + 1]
                        h = orth[d_idx + 2] * patt[g][c * 2] + orth[d_idx + 3] * patt[g][c * 2 + 1]
                        idx = c ^ (g * 2 & d_idx)
                        flat = row * 24 + col * 8 + idx
                        allhex_dr[flat] = v
                        allhex_dc[flat] = h

    return allhex_dr, allhex_dc, sgrow, sgcol


# ---------------------------------------------------------------------------
# Kernel 1: Populate — copy raw to 4 direction buffers
# ---------------------------------------------------------------------------

@ti.kernel
def _xtm_populate(
    raw: ti.types.ndarray(dtype=ti.f32, ndim=2),
    rgb: ti.types.ndarray(dtype=ti.f32, ndim=3),
    xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2),
):
    """Copy raw CFA values into (H,W,3) buffer — only the CFA-color channel is set."""
    for row, col in ti.ndrange(raw.shape[0], raw.shape[1]):
        val = ti.max(0.0, raw[row, col])
        f = FCxt(row, col, xtrans)
        rgb[row, col, 0] = 0.0
        rgb[row, col, 1] = 0.0
        rgb[row, col, 2] = 0.0
        rgb[row, col, f] = val


@ti.kernel
def _xtm_copy_rgb(
    src: ti.types.ndarray(dtype=ti.f32, ndim=3),
    dst: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    for row, col, ch in ti.ndrange(src.shape[0], src.shape[1], 3):
        dst[row, col, ch] = src[row, col, ch]


# ---------------------------------------------------------------------------
# Kernel 2: gmin / gmax — green bounds from hexagonal neighbors
# ---------------------------------------------------------------------------

@ti.kernel
def _xtm_gminmax(
    rgb: ti.types.ndarray(dtype=ti.f32, ndim=3),
    gmin_buf: ti.types.ndarray(dtype=ti.f32, ndim=2),
    gmax_buf: ti.types.ndarray(dtype=ti.f32, ndim=2),
    allhex_dr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    allhex_dc: ti.types.ndarray(dtype=ti.i32, ndim=1),
    xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2),
    h: ti.i32, w: ti.i32,
):
    pad = 3
    for row, col in ti.ndrange((pad, h - pad), (pad, w - pad)):
        if FCxt(row, col, xtrans) == 1:
            continue
        r3 = row % 3
        c3 = col % 3
        mn = ti.cast(1e30, ti.f32)
        mx = ti.cast(0.0, ti.f32)
        for k in ti.static(range(6)):
            dr = ahdr(allhex_dr, r3, c3, k)
            dc = ahdc(allhex_dc, r3, c3, k)
            val = rgb[row + dr, col + dc, 1]
            mn = ti.min(mn, val)
            mx = ti.max(mx, val)
        gmin_buf[row, col] = mn
        gmax_buf[row, col] = mx


# ---------------------------------------------------------------------------
# Kernel 3: Green interpolation — 4 directional estimates
# ---------------------------------------------------------------------------

@ti.func
def _green_interp_one(
    rgb: ti.types.ndarray(dtype=ti.f32, ndim=3),
    allhex_dr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    allhex_dc: ti.types.ndarray(dtype=ti.i32, ndim=1),
    xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2),
    row: ti.i32, col: ti.i32,
    color_idx: ti.i32,
) -> ti.f32:
    """Compute one directional green estimate (color_idx 0..3)."""
    f = FCxt(row, col, xtrans)
    r3 = row % 3
    c3 = col % 3

    result = ti.cast(0.0, ti.f32)
    if color_idx == 0:
        # color[0] = 0.6796875 * (pix[hex[1]][1] + pix[hex[0]][1])
        #          - 0.1796875 * (pix[2*hex[1]][1] + pix[2*hex[0]][1])
        dr0 = ahdr(allhex_dr, r3, c3, 0)
        dc0 = ahdc(allhex_dc, r3, c3, 0)
        dr1 = ahdr(allhex_dr, r3, c3, 1)
        dc1 = ahdc(allhex_dc, r3, c3, 1)
        result = (0.6796875 * (rgb[row + dr1, col + dc1, 1] + rgb[row + dr0, col + dc0, 1])
                  - 0.1796875 * (rgb[row + 2 * dr1, col + 2 * dc1, 1]
                                 + rgb[row + 2 * dr0, col + 2 * dc0, 1]))
    elif color_idx == 1:
        # color[1] = 0.87109375 * pix[hex[3]][1] + pix[hex[2]][1] * 0.13
        #          + 0.359375 * (pix[0][f] - pix[-hex[2]][f])
        dr2 = ahdr(allhex_dr, r3, c3, 2)
        dc2 = ahdc(allhex_dc, r3, c3, 2)
        dr3 = ahdr(allhex_dr, r3, c3, 3)
        dc3 = ahdc(allhex_dc, r3, c3, 3)
        result = (0.87109375 * rgb[row + dr3, col + dc3, 1]
                  + 0.13 * rgb[row + dr2, col + dc2, 1]
                  + 0.359375 * (rgb[row, col, f] - rgb[row - dr2, col - dc2, f]))
    else:
        # color[2+c] = 0.640625 * pix[hex[4+c]][1] + 0.359375 * pix[-2*hex[4+c]][1]
        #            + 0.12890625 * (2*pix[0][f] - pix[3*hex[4+c]][f] - pix[-3*hex[4+c]][f])
        k = 4 + (color_idx - 2)  # hex entry 4 or 5
        drk = ahdr(allhex_dr, r3, c3, k)
        dck = ahdc(allhex_dc, r3, c3, k)
        result = (0.640625 * rgb[row + drk, col + dck, 1]
                  + 0.359375 * rgb[row - 2 * drk, col - 2 * dck, 1]
                  + 0.12890625 * (2.0 * rgb[row, col, f]
                                  - rgb[row + 3 * drk, col + 3 * dck, f]
                                  - rgb[row - 3 * drk, col - 3 * dck, f]))
    return result


@ti.kernel
def _xtm_green_interp(
    rgb_d0: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d1: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d2: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d3: ti.types.ndarray(dtype=ti.f32, ndim=3),
    gmin_buf: ti.types.ndarray(dtype=ti.f32, ndim=2),
    gmax_buf: ti.types.ndarray(dtype=ti.f32, ndim=2),
    allhex_dr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    allhex_dc: ti.types.ndarray(dtype=ti.i32, ndim=1),
    xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2),
    sgrow: ti.i32, H: ti.i32, W: ti.i32,
):
    pad = 3
    for row, col in ti.ndrange((pad, H - pad), (pad, W - pad)):
        if FCxt(row, col, xtrans) == 1:
            continue

        mn = gmin_buf[row, col]
        mx = gmax_buf[row, col]

        c0 = _green_interp_one(rgb_d0, allhex_dr, allhex_dc, xtrans, row, col, 0)
        c1 = _green_interp_one(rgb_d0, allhex_dr, allhex_dc, xtrans, row, col, 1)
        c2 = _green_interp_one(rgb_d0, allhex_dr, allhex_dc, xtrans, row, col, 2)
        c3 = _green_interp_one(rgb_d0, allhex_dr, allhex_dc, xtrans, row, col, 3)

        # Direction assignment: c ^ !((row-sgrow)%3)
        # When (row-sgrow)%3 == 0: flip = 1, else flip = 0
        flip = 1 if (row - sgrow) % 3 == 0 else 0

        # c=0 ^ flip, c=1 ^ flip, c=2 ^ flip, c=3 ^ flip
        # This assigns color estimates to direction buffers
        i0 = 0 ^ flip
        i1 = 1 ^ flip
        i2 = 2 ^ flip
        i3 = 3 ^ flip

        vals = ti.Vector([clampf(c0, mn, mx),
                          clampf(c1, mn, mx),
                          clampf(c2, mn, mx),
                          clampf(c3, mn, mx)])

        # Write to the correct direction buffers
        # i0 maps estimate 0 → direction buffer i0, etc.
        # We need: for each direction buffer d, find which estimate maps to it
        # c ^ flip = d → c = d ^ flip
        rgb_d0[row, col, 1] = vals[0 ^ flip]
        rgb_d1[row, col, 1] = vals[1 ^ flip]
        rgb_d2[row, col, 1] = vals[2 ^ flip]
        rgb_d3[row, col, 1] = vals[3 ^ flip]


# ---------------------------------------------------------------------------
# Kernel 4: R/B at solitary green pixels
# ---------------------------------------------------------------------------

@ti.func
def _rb_solitary_green_for_dir(
    rgb: ti.types.ndarray(dtype=ti.f32, ndim=3),
    row: ti.i32, col: ti.i32,
    h0: ti.i32,
    mode: ti.i32,  # 0=horiz only, 1=vert only, 2=best of both
):
    """Compute R and B at a solitary green pixel for one direction buffer.

    mode: 0 = use horizontal estimate
          1 = use vertical estimate
          2 = pick best of horizontal/vertical based on diff
    Returns (R, B) tuple.
    """
    g_pixel = rgb[row, col, 1]

    # Horizontal estimates
    # c=0 (shift=1): g = 2*G - G(col+1) - G(col-1); color = g + C(col+1,h) + C(col-1,h)
    # c=1 (shift=2): g = 2*G - G(col+2) - G(col-2); color = g + C(col+2,h^2) + C(col-2,h^2)
    h_first = h0       # first color channel in horiz pass
    h_second = h0 ^ 2  # second color channel

    g_h1 = 2.0 * g_pixel - rgb[row, col + 1, 1] - rgb[row, col - 1, 1]
    color_h_c0 = g_h1 + rgb[row, col + 1, h_first] + rgb[row, col - 1, h_first]

    g_h2 = 2.0 * g_pixel - rgb[row, col + 2, 1] - rgb[row, col - 2, 1]
    color_h_c1 = g_h2 + rgb[row, col + 2, h_second] + rgb[row, col - 2, h_second]

    # Vertical estimates
    # For vert pass, h starts as h0^2 (due to outer h^=2 toggle)
    v_first = h0 ^ 2
    v_second = h0

    g_v1 = 2.0 * g_pixel - rgb[row + 1, col, 1] - rgb[row - 1, col, 1]
    color_v_c0 = g_v1 + rgb[row + 1, col, v_first] + rgb[row - 1, col, v_first]

    g_v2 = 2.0 * g_pixel - rgb[row + 2, col, 1] - rgb[row - 2, col, 1]
    color_v_c1 = g_v2 + rgb[row + 2, col, v_second] + rgb[row - 2, col, v_second]

    # Assemble color[0][*] (red) and color[1][*] (blue)
    # Horiz pass: c=0 writes color[h_first!=0], c=1 writes color[h_second!=0]
    # h_first = h0. If h0=0 → color[0]; if h0=2 → color[1]
    # h_second = h0^2. If h0^2=2 → color[1]; if h0^2=0 → color[0]

    # For the horiz pass estimate:
    horiz_r = ti.cast(0.0, ti.f32)
    horiz_b = ti.cast(0.0, ti.f32)
    if h_first == 0:
        horiz_r = color_h_c0
        horiz_b = color_h_c1
    else:
        horiz_b = color_h_c0
        horiz_r = color_h_c1

    # For the vert pass estimate:
    vert_r = ti.cast(0.0, ti.f32)
    vert_b = ti.cast(0.0, ti.f32)
    if v_first == 0:
        vert_r = color_v_c0
        vert_b = color_v_c1
    else:
        vert_b = color_v_c0
        vert_r = color_v_c1

    out_r = ti.cast(0.0, ti.f32)
    out_b = ti.cast(0.0, ti.f32)

    if mode == 0:
        out_r = horiz_r / 2.0
        out_b = horiz_b / 2.0
    elif mode == 1:
        out_r = vert_r / 2.0
        out_b = vert_b / 2.0
    else:
        # Compute diff for horiz and vert
        h_diff = ti.cast(0.0, ti.f32)
        v_diff = ti.cast(0.0, ti.f32)

        # Horiz diff (c=0)
        h_diff += sqrf(rgb[row, col + 1, 1] - rgb[row, col - 1, 1]
                       - rgb[row, col + 1, h_first] + rgb[row, col - 1, h_first]) + sqrf(g_h1)
        # Horiz diff (c=1)
        h_diff += sqrf(rgb[row, col + 2, 1] - rgb[row, col - 2, 1]
                       - rgb[row, col + 2, h_second] + rgb[row, col - 2, h_second]) + sqrf(g_h2)

        # Vert diff (c=0)
        v_diff += sqrf(rgb[row + 1, col, 1] - rgb[row - 1, col, 1]
                       - rgb[row + 1, col, v_first] + rgb[row - 1, col, v_first]) + sqrf(g_v1)
        # Vert diff (c=1)
        v_diff += sqrf(rgb[row + 2, col, 1] - rgb[row - 2, col, 1]
                       - rgb[row + 2, col, v_second] + rgb[row - 2, col, v_second]) + sqrf(g_v2)

        if h_diff <= v_diff:
            out_r = horiz_r / 2.0
            out_b = horiz_b / 2.0
        else:
            out_r = vert_r / 2.0
            out_b = vert_b / 2.0

    return out_r, out_b


@ti.kernel
def _xtm_rb_at_green(
    rgb_d0: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d1: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d2: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d3: ti.types.ndarray(dtype=ti.f32, ndim=3),
    xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2),
    sgrow: ti.i32, sgcol: ti.i32,
    H: ti.i32, W: ti.i32,
):
    pad = 6
    for row, col in ti.ndrange((pad, H - pad), (pad, W - pad)):
        if (row - sgrow) % 3 != 0 or (col - sgcol) % 3 != 0:
            continue
        if FCxt(row, col, xtrans) != 1:
            continue

        h0 = FCxt(row, col + 1, xtrans)

        # Dir 0: horizontal only
        r0, b0 = _rb_solitary_green_for_dir(rgb_d0, row, col, h0, 0)
        rgb_d0[row, col, 0] = r0
        rgb_d0[row, col, 2] = b0

        # Dir 1: vertical only
        r1, b1 = _rb_solitary_green_for_dir(rgb_d1, row, col, h0, 1)
        rgb_d1[row, col, 0] = r1
        rgb_d1[row, col, 2] = b1

        # Dir 2: best of horiz/vert
        r2, b2 = _rb_solitary_green_for_dir(rgb_d2, row, col, h0, 2)
        rgb_d2[row, col, 0] = r2
        rgb_d2[row, col, 2] = b2

        # Dir 3: best of horiz/vert
        r3, b3 = _rb_solitary_green_for_dir(rgb_d3, row, col, h0, 2)
        rgb_d3[row, col, 0] = r3
        rgb_d3[row, col, 2] = b3


# ---------------------------------------------------------------------------
# Kernel 5: R at B pixels, B at R pixels (cross-color interpolation)
# ---------------------------------------------------------------------------

@ti.func
def _rb_cross_for_dir(
    rgb: ti.types.ndarray(dtype=ti.f32, ndim=3),
    row: ti.i32, col: ti.i32,
    f: ti.i32,        # target channel (0=R or 2=B)
    sgrow: ti.i32,
    d: ti.i32,
) -> ti.f32:
    """Interpolate the opposite color at a R or B pixel for one direction."""
    # c = TS (vert) when (row-sgrow)%3 != 0, else 1 (horiz)
    # h = 3*(c^TS^1) → when c is horiz, h is vert-3; when c is vert, h is horiz-3
    is_vert_primary = (row - sgrow) % 3 != 0

    # In our flat-image model:
    # c_direction: (dr, dc) for primary direction
    # h_direction: (dr, dc) for secondary direction (3x the perpendicular)
    c_dr = ti.cast(0, ti.i32)
    c_dc = ti.cast(0, ti.i32)
    h_dr = ti.cast(0, ti.i32)
    h_dc = ti.cast(0, ti.i32)
    c_val = ti.cast(0, ti.i32)  # original c value (1 or TS) for logic

    if is_vert_primary:
        # c = TS → vertical, distance 1
        c_dr = 1; c_dc = 0
        # h = 3 * (TS ^ TS ^ 1) = 3 * 1 = 3 → horizontal, distance 3
        h_dr = 0; h_dc = 3
        c_val = 122  # TS
    else:
        # c = 1 → horizontal, distance 1
        c_dr = 0; c_dc = 1
        # h = 3 * (1 ^ TS ^ 1) = 3 * TS → vertical, distance 3
        h_dr = 3; h_dc = 0
        c_val = 1

    # Decision: use c or h direction?
    # Original: i = d>1 || ((d^c)&1) || (gradient_c < 2*gradient_h) ? c : h
    use_c = False
    if d > 1:
        use_c = True
    elif (d ^ c_val) & 1:
        use_c = True
    else:
        grad_c = (ti.abs(rgb[row, col, 1] - rgb[row + c_dr, col + c_dc, 1])
                  + ti.abs(rgb[row, col, 1] - rgb[row - c_dr, col - c_dc, 1]))
        grad_h = (ti.abs(rgb[row, col, 1] - rgb[row + h_dr, col + h_dc, 1])
                  + ti.abs(rgb[row, col, 1] - rgb[row - h_dr, col - h_dc, 1]))
        if grad_c < 2.0 * grad_h:
            use_c = True

    i_dr = c_dr if use_c else h_dr
    i_dc = c_dc if use_c else h_dc

    return (rgb[row + i_dr, col + i_dc, f]
            + rgb[row - i_dr, col - i_dc, f]
            + 2.0 * rgb[row, col, 1]
            - rgb[row + i_dr, col + i_dc, 1]
            - rgb[row - i_dr, col - i_dc, 1]) / 2.0


@ti.kernel
def _xtm_rb_interp(
    rgb_d0: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d1: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d2: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d3: ti.types.ndarray(dtype=ti.f32, ndim=3),
    xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2),
    sgrow: ti.i32,
    H: ti.i32, W: ti.i32,
):
    pad = 6
    for row, col in ti.ndrange((pad, H - pad), (pad, W - pad)):
        f = 2 - FCxt(row, col, xtrans)
        if f == 1:
            continue

        rgb_d0[row, col, f] = _rb_cross_for_dir(rgb_d0, row, col, f, sgrow, 0)
        rgb_d1[row, col, f] = _rb_cross_for_dir(rgb_d1, row, col, f, sgrow, 1)
        rgb_d2[row, col, f] = _rb_cross_for_dir(rgb_d2, row, col, f, sgrow, 2)
        rgb_d3[row, col, f] = _rb_cross_for_dir(rgb_d3, row, col, f, sgrow, 3)


# ---------------------------------------------------------------------------
# Kernel 6: Fill R/B at 2x2 green blocks
# ---------------------------------------------------------------------------

@ti.func
def _fill_g22_for_dir(
    rgb: ti.types.ndarray(dtype=ti.f32, ndim=3),
    allhex_dr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    allhex_dc: ti.types.ndarray(dtype=ti.i32, ndim=1),
    row: ti.i32, col: ti.i32,
    hex_entry_a: ti.i32,
    hex_entry_b: ti.i32,
):
    """Fill R and B at a 2x2 green block pixel using two hex neighbors."""
    r3 = row % 3
    c3 = col % 3
    dra = ahdr(allhex_dr, r3, c3, hex_entry_a)
    dca = ahdc(allhex_dc, r3, c3, hex_entry_a)
    drb = ahdr(allhex_dr, r3, c3, hex_entry_b)
    dcb = ahdc(allhex_dc, r3, c3, hex_entry_b)

    # Check if hex[a] + hex[b] != 0 (in 1D offset terms, check if combined offset is nonzero)
    sum_nonzero = (dra + drb != 0) or (dca + dcb != 0)

    if sum_nonzero:
        g = (3.0 * rgb[row, col, 1]
             - 2.0 * rgb[row + dra, col + dca, 1]
             - rgb[row + drb, col + dcb, 1])
        for ch in ti.static(range(0, 3, 2)):  # ch = 0, 2 (R and B)
            rgb[row, col, ch] = (g + 2.0 * rgb[row + dra, col + dca, ch]
                                 + rgb[row + drb, col + dcb, ch]) / 3.0
    else:
        g = (2.0 * rgb[row, col, 1]
             - rgb[row + dra, col + dca, 1]
             - rgb[row + drb, col + dcb, 1])
        for ch in ti.static(range(0, 3, 2)):
            rgb[row, col, ch] = (g + rgb[row + dra, col + dca, ch]
                                 + rgb[row + drb, col + dcb, ch]) / 2.0


@ti.kernel
def _xtm_fill_green22(
    rgb_d0: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d1: ti.types.ndarray(dtype=ti.f32, ndim=3),
    allhex_dr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    allhex_dc: ti.types.ndarray(dtype=ti.i32, ndim=1),
    xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2),
    sgrow: ti.i32, sgcol: ti.i32,
    H: ti.i32, W: ti.i32,
):
    """Fill R/B for 2x2 green blocks. Only processes directions 0 and 1 (1-pass)."""
    pad = 8
    for row, col in ti.ndrange((pad, H - pad), (pad, W - pad)):
        if (row - sgrow) % 3 == 0 or (col - sgcol) % 3 == 0:
            continue

        # d=0: hex[0], hex[1] → direction 0
        _fill_g22_for_dir(rgb_d0, allhex_dr, allhex_dc, row, col, 0, 1)
        # d=2: hex[2], hex[3] → direction 1
        _fill_g22_for_dir(rgb_d1, allhex_dr, allhex_dc, row, col, 2, 3)


# ---------------------------------------------------------------------------
# Kernel 7: YPbPr conversion + directional derivatives
# ---------------------------------------------------------------------------

@ti.kernel
def _xtm_yuv_derivatives(
    rgb_d0: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d1: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d2: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d3: ti.types.ndarray(dtype=ti.f32, ndim=3),
    drv0: ti.types.ndarray(dtype=ti.f32, ndim=2),
    drv1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    drv2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    drv3: ti.types.ndarray(dtype=ti.f32, ndim=2),
    H: ti.i32, W: ti.i32,
):
    """Convert to YPbPr (BT.2020) and compute 2nd-order directional derivatives.

    Direction d uses derivative direction dir[d]:
      d=0: horizontal   (0, +1)
      d=1: vertical     (+1, 0)
      d=2: diagonal     (+1, +1)
      d=3: anti-diag    (+1, -1)
    """
    pad = 9
    for row, col in ti.ndrange((pad, H - pad), (pad, W - pad)):
        # Direction 0 — horizontal derivative
        # YPbPr at (row, col), (row, col+1), (row, col-1)
        for d_idx in ti.static(range(4)):
            dr = ti.cast(0, ti.i32)
            dc = ti.cast(0, ti.i32)
            if d_idx == 0:
                dr = 0; dc = 1
            elif d_idx == 1:
                dr = 1; dc = 0
            elif d_idx == 2:
                dr = 1; dc = 1
            else:
                dr = 1; dc = -1

            # Select direction buffer
            rx0 = ti.cast(0.0, ti.f32)
            rx1 = ti.cast(0.0, ti.f32)
            rx2 = ti.cast(0.0, ti.f32)
            rxp0 = ti.cast(0.0, ti.f32)
            rxp1 = ti.cast(0.0, ti.f32)
            rxp2 = ti.cast(0.0, ti.f32)
            rxn0 = ti.cast(0.0, ti.f32)
            rxn1 = ti.cast(0.0, ti.f32)
            rxn2 = ti.cast(0.0, ti.f32)

            if d_idx == 0:
                rx0 = rgb_d0[row, col, 0]; rx1 = rgb_d0[row, col, 1]; rx2 = rgb_d0[row, col, 2]
                rxp0 = rgb_d0[row+dr, col+dc, 0]; rxp1 = rgb_d0[row+dr, col+dc, 1]; rxp2 = rgb_d0[row+dr, col+dc, 2]
                rxn0 = rgb_d0[row-dr, col-dc, 0]; rxn1 = rgb_d0[row-dr, col-dc, 1]; rxn2 = rgb_d0[row-dr, col-dc, 2]
            elif d_idx == 1:
                rx0 = rgb_d1[row, col, 0]; rx1 = rgb_d1[row, col, 1]; rx2 = rgb_d1[row, col, 2]
                rxp0 = rgb_d1[row+dr, col+dc, 0]; rxp1 = rgb_d1[row+dr, col+dc, 1]; rxp2 = rgb_d1[row+dr, col+dc, 2]
                rxn0 = rgb_d1[row-dr, col-dc, 0]; rxn1 = rgb_d1[row-dr, col-dc, 1]; rxn2 = rgb_d1[row-dr, col-dc, 2]
            elif d_idx == 2:
                rx0 = rgb_d2[row, col, 0]; rx1 = rgb_d2[row, col, 1]; rx2 = rgb_d2[row, col, 2]
                rxp0 = rgb_d2[row+dr, col+dc, 0]; rxp1 = rgb_d2[row+dr, col+dc, 1]; rxp2 = rgb_d2[row+dr, col+dc, 2]
                rxn0 = rgb_d2[row-dr, col-dc, 0]; rxn1 = rgb_d2[row-dr, col-dc, 1]; rxn2 = rgb_d2[row-dr, col-dc, 2]
            else:
                rx0 = rgb_d3[row, col, 0]; rx1 = rgb_d3[row, col, 1]; rx2 = rgb_d3[row, col, 2]
                rxp0 = rgb_d3[row+dr, col+dc, 0]; rxp1 = rgb_d3[row+dr, col+dc, 1]; rxp2 = rgb_d3[row+dr, col+dc, 2]
                rxn0 = rgb_d3[row-dr, col-dc, 0]; rxn1 = rgb_d3[row-dr, col-dc, 1]; rxn2 = rgb_d3[row-dr, col-dc, 2]

            # BT.2020 YPbPr
            y  = 0.2627 * rx0 + 0.6780 * rx1 + 0.0593 * rx2
            yp = 0.2627 * rxp0 + 0.6780 * rxp1 + 0.0593 * rxp2
            yn = 0.2627 * rxn0 + 0.6780 * rxn1 + 0.0593 * rxn2

            pb  = (rx2 - y) * 0.56433
            pbp = (rxp2 - yp) * 0.56433
            pbn = (rxn2 - yn) * 0.56433

            pr  = (rx0 - y) * 0.67815
            prp = (rxp0 - yp) * 0.67815
            prn = (rxn0 - yn) * 0.67815

            deriv = (sqrf(2.0 * y - yp - yn)
                     + sqrf(2.0 * pb - pbp - pbn)
                     + sqrf(2.0 * pr - prp - prn))

            if d_idx == 0:
                drv0[row, col] = deriv
            elif d_idx == 1:
                drv1[row, col] = deriv
            elif d_idx == 2:
                drv2[row, col] = deriv
            else:
                drv3[row, col] = deriv


# ---------------------------------------------------------------------------
# Kernel 8: Homogeneity analysis + weighted merge
# ---------------------------------------------------------------------------

@ti.kernel
def _xtm_homo_merge(
    rgb_d0: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d1: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d2: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rgb_d3: ti.types.ndarray(dtype=ti.f32, ndim=3),
    drv0: ti.types.ndarray(dtype=ti.f32, ndim=2),
    drv1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    drv2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    drv3: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    H: ti.i32, W: ti.i32,
):
    """Fused homogeneity computation and directional averaging."""
    pad = BORDER
    for row, col in ti.ndrange((pad, H - pad), (pad, W - pad)):
        # Phase 1: Build homogeneity counts (3x3 neighborhood check against 8*min)
        # Phase 2: Sum in 5x5 for each direction
        # Phase 3: Average most homogeneous

        homosum = ti.Vector([0, 0, 0, 0])  # 5x5 summed homogeneity per direction

        for vv in range(-2, 3):
            for hh in range(-2, 3):
                r2 = row + vv
                c2 = col + hh
                # Find min derivative across 4 directions at this position
                d0v = drv0[r2, c2]
                d1v = drv1[r2, c2]
                d2v = drv2[r2, c2]
                d3v = drv3[r2, c2]
                tr = ti.min(ti.min(d0v, d1v), ti.min(d2v, d3v))
                tr *= 8.0

                # Count homogeneous cells in 3x3 around (r2, c2)
                for di in ti.static(range(4)):
                    cnt = 0
                    for v3 in range(-1, 2):
                        for h3 in range(-1, 2):
                            r3 = r2 + v3
                            c3 = c2 + h3
                            dval = ti.cast(0.0, ti.f32)
                            if di == 0:
                                dval = drv0[r3, c3]
                            elif di == 1:
                                dval = drv1[r3, c3]
                            elif di == 2:
                                dval = drv2[r3, c3]
                            else:
                                dval = drv3[r3, c3]
                            if dval <= tr:
                                cnt += 1
                    homosum[di] += cnt

        # Phase 3: Average directions with highest homogeneity
        maxval = ti.max(ti.max(homosum[0], homosum[1]),
                        ti.max(homosum[2], homosum[3]))
        threshold = maxval - (maxval >> 3)

        # Compare direction pairs (d vs d+ndir/2 doesn't apply for ndir=4 in 1-pass)
        # In darktable for ndir=4: ndir-4=0, so the pair-comparison loop runs 0 times
        # Just average all directions above threshold

        avg_r = ti.cast(0.0, ti.f32)
        avg_g = ti.cast(0.0, ti.f32)
        avg_b = ti.cast(0.0, ti.f32)
        cnt = ti.cast(0.0, ti.f32)

        if homosum[0] >= threshold:
            avg_r += rgb_d0[row, col, 0]
            avg_g += rgb_d0[row, col, 1]
            avg_b += rgb_d0[row, col, 2]
            cnt += 1.0
        if homosum[1] >= threshold:
            avg_r += rgb_d1[row, col, 0]
            avg_g += rgb_d1[row, col, 1]
            avg_b += rgb_d1[row, col, 2]
            cnt += 1.0
        if homosum[2] >= threshold:
            avg_r += rgb_d2[row, col, 0]
            avg_g += rgb_d2[row, col, 1]
            avg_b += rgb_d2[row, col, 2]
            cnt += 1.0
        if homosum[3] >= threshold:
            avg_r += rgb_d3[row, col, 0]
            avg_g += rgb_d3[row, col, 1]
            avg_b += rgb_d3[row, col, 2]
            cnt += 1.0

        if cnt > 0.0:
            out[row, col, 0] = ti.max(0.0, avg_r / cnt)
            out[row, col, 1] = ti.max(0.0, avg_g / cnt)
            out[row, col, 2] = ti.max(0.0, avg_b / cnt)
        else:
            out[row, col, 0] = ti.max(0.0, rgb_d0[row, col, 0])
            out[row, col, 1] = ti.max(0.0, rgb_d0[row, col, 1])
            out[row, col, 2] = ti.max(0.0, rgb_d0[row, col, 2])


# ---------------------------------------------------------------------------
# Kernel 9: Border bilinear interpolation
# ---------------------------------------------------------------------------

@ti.kernel
def _xtm_border_interpolate(
    raw: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    xtrans: ti.types.ndarray(dtype=ti.i32, ndim=2),
    border: ti.i32,
):
    h, w = raw.shape[0], raw.shape[1]
    for row, col in ti.ndrange(h, w):
        if row >= border and row < h - border and col >= border and col < w - border:
            continue

        sums = ti.Vector([0.0, 0.0, 0.0])
        counts = ti.Vector([0.0, 0.0, 0.0])

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y = row + dy
                x = col + dx
                if 0 <= y < h and 0 <= x < w:
                    c = FCxt(y, x, xtrans)
                    val = ti.max(0.0, raw[y, x])
                    sums[c] += val
                    counts[c] += 1.0

        for c in ti.static(range(3)):
            if counts[c] > 0.0:
                out[row, col, c] = sums[c] / counts[c]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def xtrans_markesteijn_demosaic(
    raw: np.ndarray,
    xtrans_pattern: np.ndarray,
) -> np.ndarray:
    """Demosaic an X-Trans image using Markesteijn 1-pass (GPU-accelerated).

    Args:
        raw: (H, W) float32, black-level subtracted, normalized to [0, 1]
        xtrans_pattern: (6, 6) array with values 0=R, 1=G, 2=B

    Returns:
        (H, W, 3) float32 RGB in [0, 1]
    """
    import time
    from raw_alchemy.math_ops import init_taichi
    init_taichi()

    t0 = time.time()

    h, w = raw.shape
    assert h >= 25 and w >= 25, f"Image too small for X-Trans demosaic: {w}x{h}"
    assert xtrans_pattern.shape == (6, 6), f"xtrans_pattern must be (6,6), got {xtrans_pattern.shape}"

    # --- CPU precomputation ---
    allhex_dr, allhex_dc, sgrow, sgcol = _build_allhex(xtrans_pattern)
    logger.debug(f"X-Trans allhex built: sgrow={sgrow}, sgcol={sgcol}")

    # --- GPU buffer allocation ---
    xt_gpu = ti.ndarray(dtype=ti.i32, shape=(6, 6))
    xt_gpu.from_numpy(xtrans_pattern.astype(np.int32))

    ah_dr_gpu = ti.ndarray(dtype=ti.i32, shape=(72,))
    ah_dr_gpu.from_numpy(allhex_dr)
    ah_dc_gpu = ti.ndarray(dtype=ti.i32, shape=(72,))
    ah_dc_gpu.from_numpy(allhex_dc)

    raw_gpu = ti.ndarray(dtype=ti.f32, shape=(h, w))
    raw_gpu.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))

    # 4 direction buffers, each (H, W, 3)
    rgb_d0 = ti.ndarray(dtype=ti.f32, shape=(h, w, 3))
    rgb_d1 = ti.ndarray(dtype=ti.f32, shape=(h, w, 3))
    rgb_d2 = ti.ndarray(dtype=ti.f32, shape=(h, w, 3))
    rgb_d3 = ti.ndarray(dtype=ti.f32, shape=(h, w, 3))

    # --- Kernel 1: Populate ---
    _xtm_populate(raw_gpu, rgb_d0, xt_gpu)
    _xtm_copy_rgb(rgb_d0, rgb_d1)
    _xtm_copy_rgb(rgb_d0, rgb_d2)
    _xtm_copy_rgb(rgb_d0, rgb_d3)

    # --- Kernel 2: gmin/gmax ---
    gmin_buf = ti.ndarray(dtype=ti.f32, shape=(h, w))
    gmax_buf = ti.ndarray(dtype=ti.f32, shape=(h, w))
    del raw_gpu

    _xtm_gminmax(rgb_d0, gmin_buf, gmax_buf, ah_dr_gpu, ah_dc_gpu, xt_gpu, h, w)

    # --- Kernel 3: Green interpolation ---
    _xtm_green_interp(rgb_d0, rgb_d1, rgb_d2, rgb_d3,
                       gmin_buf, gmax_buf, ah_dr_gpu, ah_dc_gpu, xt_gpu,
                       sgrow, h, w)

    del gmin_buf, gmax_buf

    # --- Kernel 4: R/B at solitary green ---
    _xtm_rb_at_green(rgb_d0, rgb_d1, rgb_d2, rgb_d3, xt_gpu, sgrow, sgcol, h, w)

    # --- Kernel 5: R/B cross-interpolation ---
    _xtm_rb_interp(rgb_d0, rgb_d1, rgb_d2, rgb_d3, xt_gpu, sgrow, h, w)

    # --- Kernel 6: Fill 2x2 green blocks ---
    _xtm_fill_green22(rgb_d0, rgb_d1, ah_dr_gpu, ah_dc_gpu, xt_gpu, sgrow, sgcol, h, w)

    del ah_dr_gpu, ah_dc_gpu

    # --- Kernel 7: Derivatives ---
    drv0 = ti.ndarray(dtype=ti.f32, shape=(h, w))
    drv1 = ti.ndarray(dtype=ti.f32, shape=(h, w))
    drv2 = ti.ndarray(dtype=ti.f32, shape=(h, w))
    drv3 = ti.ndarray(dtype=ti.f32, shape=(h, w))

    _xtm_yuv_derivatives(rgb_d0, rgb_d1, rgb_d2, rgb_d3,
                          drv0, drv1, drv2, drv3, h, w)

    # --- Kernel 8: Homogeneity + merge ---
    out = ti.ndarray(dtype=ti.f32, shape=(h, w, 3))

    _xtm_homo_merge(rgb_d0, rgb_d1, rgb_d2, rgb_d3,
                     drv0, drv1, drv2, drv3, out, h, w)

    del rgb_d0, rgb_d1, rgb_d2, rgb_d3
    del drv0, drv1, drv2, drv3

    # --- Kernel 9: Border ---
    raw_gpu2 = ti.ndarray(dtype=ti.f32, shape=(h, w))
    raw_gpu2.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))
    _xtm_border_interpolate(raw_gpu2, out, xt_gpu, BORDER)
    del raw_gpu2, xt_gpu

    result = out.to_numpy()
    del out

    import gc; gc.collect()

    elapsed = time.time() - t0
    logger.info(f"X-Trans Markesteijn demosaic: {w}x{h} in {elapsed*1000:.0f}ms (GPU)")

    return result
