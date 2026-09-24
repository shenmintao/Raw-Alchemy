"""Region-restricted highlight reconstruction must match the full-frame pass."""

import cv2
import numpy as np
import pytest


def compute_hl_refavg(raw_data, color_map, wb_gains, raw_clips):
    """The previous full-frame reference average, kept verbatim as the oracle."""
    h, w = raw_data.shape
    cbrt_sum = np.zeros((h, w), dtype=np.float32)
    own_cbrt = np.empty((h, w), dtype=np.float32)
    stripe_rows = max(64, min(1024, 1_000_000 // max(w, 1)))
    for c in range(3):
        gain = np.float32(wb_gains[c])
        for y in range(0, h, stripe_rows):
            y2 = min(y + stripe_rows, h)
            halo0, halo1 = max(0, y - 1), min(h, y2 + 1)
            mask = color_map[halo0:halo1] == c
            values = np.maximum(raw_data[halo0:halo1], np.float32(0.0))
            values *= mask
            counts = mask.astype(np.float32)
            sums = cv2.boxFilter(values, -1, (3, 3), normalize=False,
                                 borderType=cv2.BORDER_REPLICATE)
            cnts = cv2.boxFilter(counts, -1, (3, 3), normalize=False,
                                 borderType=cv2.BORDER_REPLICATE)
            np.divide(sums, cnts, out=sums, where=cnts > 0)
            sums[cnts <= 0] = 0.0
            sums *= gain
            np.cbrt(sums, out=sums)
            core = sums[y - halo0:y - halo0 + (y2 - y)]
            cbrt_sum[y:y2] += core
            core_mask = color_map[y:y2] == c
            own = own_cbrt[y:y2]
            own[core_mask] = core[core_mask]
    cbrt_sum -= own_cbrt
    cbrt_sum *= np.float32(0.5)
    np.power(cbrt_sum, np.float32(3.0), out=cbrt_sum)
    for c in range(3):
        gain = float(wb_gains[c])
        if gain > 1e-6:
            cbrt_sum[color_map == c] /= np.float32(gain)
    clipped = np.empty((h, w), dtype=bool)
    for c in range(3):
        mask = color_map == c
        clipped[mask] = raw_data[mask] >= np.float32(raw_clips[c])
    return cbrt_sum, clipped


def _reference_highlight_inpaint(raw_data, cfa_pattern, wb):
    """The previous full-frame implementation, kept verbatim as the oracle."""
    H, W = raw_data.shape
    pat_size = cfa_pattern.shape[0]
    g = max(float(wb[1]), 1e-6)
    color_map = np.tile(cfa_pattern, ((H + pat_size - 1) // pat_size,
                                      (W + pat_size - 1) // pat_size))[:H, :W]
    color_map = np.where(color_map >= 3, 1, color_map).astype(np.uint8)
    wb_gains = np.array([wb[0] / g, 1.0, wb[2] / g], dtype=np.float32)
    raw_clips = np.array([0.987 / max(wg, 1e-6) for wg in wb_gains], dtype=np.float32)
    if float(raw_data.max()) < float(raw_clips.min()):
        return
    refavg, clipped = compute_hl_refavg(raw_data, color_map, wb_gains, raw_clips)
    if not np.any(clipped):
        return
    diff = raw_data - refavg
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for c in range(3):
        clipped_c = clipped & (color_map == c)
        if not np.any(clipped_c):
            continue
        closed = cv2.morphologyEx(clipped_c.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        n, labels = cv2.connectedComponents(closed, connectivity=8)
        if n - 1 == 0:
            continue
        expanded = cv2.dilate(labels.astype(np.float32), kernel).astype(np.int32)
        unclipped_valid = (color_map == c) & ~clipped & (raw_data > raw_clips[c] * 0.2)
        border = (expanded > 0) & (labels == 0) & unclipped_valid
        seg_sum = np.bincount(expanded[border], weights=diff[border], minlength=n)
        seg_cnt = np.bincount(expanded[border], minlength=n)
        global_chroma = 0.0
        if seg_cnt[1:].sum() > 100:
            global_chroma = seg_sum[1:].sum() / seg_cnt[1:].sum()
        seg_chroma = np.where(seg_cnt > 10, seg_sum / np.maximum(seg_cnt, 1),
                              global_chroma).astype(np.float32)
        target = clipped_c & (labels > 0)
        raw_data[target] = np.maximum(
            raw_data[target], raw_data[target] - diff[target] + seg_chroma[labels[target]]
        )


BAYER = np.array([[0, 1], [3, 2]])
XTRANS = np.array([
    [0, 2, 1, 2, 0, 1], [1, 1, 0, 1, 1, 2], [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1], [1, 1, 2, 1, 1, 0], [1, 1, 0, 1, 1, 2],
])
WB = np.array([2.1, 1.0, 1.6, 0.0], np.float32)


def _scene(shape, blobs, seed=0):
    rng = np.random.default_rng(seed)
    raw = (0.25 + 0.2 * rng.random(shape) + rng.normal(0, 0.01, shape)).astype(np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for cy, cx, r in blobs:
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        halo = (yy - cy) ** 2 + (xx - cx) ** 2 <= (r + 6) ** 2
        raw[halo] = np.maximum(raw[halo], 0.42)
        raw[disk] = 1.0 + 0.1 * rng.random(int(disk.sum()))
    return raw


LAYOUTS = {
    "spanning_tiles": [(300, 400, 90)],
    "scattered_lamps": [(40, 60, 3), (500, 700, 4), (200, 150, 2), (450, 90, 5)],
    "image_border": [(2, 3, 12), (595, 790, 10)],
    "two_close_clusters": [(150, 150, 20), (150, 420, 20)],
    "dense_fallback": [(y, x, 30) for y in range(60, 600, 120) for x in range(60, 800, 120)],
}


@pytest.mark.parametrize("pattern", [BAYER, XTRANS], ids=["bayer", "xtrans"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_region_version_matches_full_frame(pattern, layout):
    from raw_alchemy import core

    shape = (600, 792)
    raw = _scene(shape, LAYOUTS[layout])
    expected = raw.copy()
    _reference_highlight_inpaint(expected, pattern, WB)
    actual = raw.copy()
    core.highlight_inpaint_opposed(actual, pattern, WB)

    assert not np.array_equal(expected, raw), "scene must exercise reconstruction"
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)


def test_frames_without_clipping_are_untouched():
    from raw_alchemy import core

    raw = _scene((300, 400), [])
    before = raw.copy()
    core.highlight_inpaint_opposed(raw, BAYER, WB)
    np.testing.assert_array_equal(raw, before)
