"""X-Trans Markesteijn demosaic on ONNX Runtime (GPU via DirectML/CUDA).

Replaces the retired Taichi port. The 0.47MB dynamic-shape graph is a
vectorized translation of the same 1-pass Markesteijn (bit-exact against the
Taichi reference compiled without fast_math; see the port report). The graph
is built for the canonical Fuji pattern — mosaics on a different phase are
rolled onto it first.

Full-frame execution exhausts VRAM above ~30MP (direction-stacked
intermediates), so inference is tiled: 1560px tiles, 24px overlap (the
algorithm's neighbourhood influence is <=12px + 8px border logic), one fixed
tile shape so DirectML compiles the graph once. Measured on RX 9070 XT:
26MP ~1.2s, 40.2MP ~2.0s.

Input:  (H, W) float32 mosaic, black-subtracted, normalized [0, 1]
Output: (H, W, 3) float32 RGB
"""

import threading
import time

import numpy as np
from loguru import logger

from .denoiser import (
    _configure_providers,
    _find_model,
    _get_providers,
    _make_session_options,
)

MODEL_FILE = "xtrans_markesteijn_dyn.onnx"
TILE = 1560  # multiple of 6
OVERLAP = 24  # multiple of 6, > max neighbourhood influence (12px) + border (8px)

CANONICAL_PATTERN = np.array([
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
    [0, 2, 1, 2, 0, 1],
], dtype=np.int32)

_sessions = {}
_session_lock = threading.Lock()
_session_provider = None
_masks = None


def _build_allhex(xt: np.ndarray):
    """darktable's hex-neighbour table precompute (numpy, port-verified)."""
    orth = [1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1]
    patt = [
        [0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0],
        [0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1],
    ]
    sgrow = sgcol = 0

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
    return sgrow, sgcol


def _build_masks(xt: np.ndarray) -> np.ndarray:
    """(15, 6, 6) float32 mask stack the ONNX graph expects."""
    sgrow, sgcol = _build_allhex(xt)
    fc = np.asarray(xt, dtype=np.int32)
    r = np.arange(6)[:, None]
    c = np.arange(6)[None, :]
    ms = [
        fc == 0, fc == 1, fc == 2,
        np.broadcast_to((r - sgrow) % 3 == 0, (6, 6)),
        np.broadcast_to((c - sgcol) % 3 == 0, (6, 6)),
        fc[np.arange(6)][:, (np.arange(6) + 1) % 6] == 0,
    ]
    for r3 in range(3):
        for c3 in range(3):
            ms.append((r % 3 == r3) & (c % 3 == c3))
    return np.stack([np.asarray(m, np.float32) for m in ms], axis=0)


def _get_session():
    global _session_provider, _masks
    session = _sessions.get(TILE)
    if session is not None:
        return session
    with _session_lock:
        session = _sessions.get(TILE)
        if session is not None:
            return session
        import onnxruntime as ort

        model_path = _find_model(MODEL_FILE)
        so = _make_session_options(ort)
        so.add_free_dimension_override_by_name("h", TILE)
        so.add_free_dimension_override_by_name("w", TILE)
        if _cpu_fallback:
            providers = ["CPUExecutionProvider"]
        else:
            providers = _get_providers()
        session = ort.InferenceSession(
            model_path, so, providers=_configure_providers(
                providers, model_path, variant=f"xtrans:h={TILE},w={TILE}"
            ),
        )
        _sessions[TILE] = session
        _session_provider = session.get_providers()[0]
        _masks = _build_masks(CANONICAL_PATTERN)
        logger.info(f"X-Trans demosaic session (tile {TILE}): {_session_provider}")
    return session


def clear_session() -> None:
    global _session_provider
    with _session_lock:
        _sessions.clear()
        _session_provider = None
    import gc
    gc.collect()


_cpu_fallback = False


def _run_graph(session, feeds):
    global _cpu_fallback
    # A previous tile may have rebuilt the process-global session on CPU.
    # Always adopt the current session instead of retrying a stale GPU object
    # for every remaining tile.
    session = _get_session()
    try:
        return session.run(None, feeds)[0]
    except Exception as e:
        logger.warning(
            f"X-Trans tile failed on {_session_provider} "
            f"({type(e).__name__}: {str(e)[:80]}); rebuilding on CPU EP")
        _cpu_fallback = True
        clear_session()
        return _get_session().run(None, feeds)[0]


def _canonical_roll(pattern: np.ndarray):
    """(dr, dc) so that pattern[(r+dr)%6, (c+dc)%6] == CANONICAL_PATTERN."""
    pat = np.where(np.asarray(pattern) >= 3, 1, np.asarray(pattern))
    for dr in range(6):
        for dc in range(6):
            if np.array_equal(np.roll(np.roll(pat, -dr, 0), -dc, 1), CANONICAL_PATTERN):
                return dr, dc
    raise ValueError(f"not an X-Trans pattern phase: {pattern.tolist()}")


def xtrans_markesteijn_demosaic(raw_norm: np.ndarray, xtrans_pattern: np.ndarray) -> np.ndarray:
    """Demosaic an X-Trans mosaic with Markesteijn on the ONNX runtime.

    Args:
        raw_norm: (H, W) float32, black-subtracted, normalized to [0, 1]
        xtrans_pattern: (6, 6) ints, 0=R 1=G 2=B

    Returns:
        (H, W, 3) float32 RGB
    """
    if raw_norm.ndim != 2:
        raise ValueError(f"expected (H, W) mosaic, got {raw_norm.shape}")
    t0 = time.time()
    session = _get_session()
    H, W = raw_norm.shape

    # Phase-align onto the canonical pattern by cropping (dr, dc) at top/left;
    # restore with edge-replication afterwards (only 0-5 border pixels).
    dr, dc = _canonical_roll(xtrans_pattern)
    work = raw_norm[dr:, dc:]
    wh, ww = work.shape
    # Pad bottom/right to multiples of 6 (reflect keeps CFA phase via period-6
    # aware padding: reflect by 6-aligned amount using edge rows repeated).
    ph = (6 - wh % 6) % 6
    pw = (6 - ww % 6) % 6
    if ph or pw:
        work = np.pad(work, ((0, ph), (0, pw)), mode="reflect")

    out = _demosaic_tiled(session, np.ascontiguousarray(work, np.float32))
    out = out[:wh, :ww]

    if dr or dc:
        full = np.empty((H, W, 3), np.float32)
        full[dr:, dc:] = out
        if dr:
            full[:dr, dc:] = out[0:1]
        if dc:
            full[:, :dc] = full[:, dc:dc + 1]
        out = full

    logger.info(
        f"X-Trans demosaic: {W}x{H} in {(time.time() - t0) * 1000:.0f}ms "
        f"({_session_provider})"
    )
    return out


def _demosaic_tiled(session, raw: np.ndarray) -> np.ndarray:
    H, W = raw.shape
    if H <= TILE and W <= TILE:
        ph, pw = TILE - H, TILE - W
        patch = np.pad(raw, ((0, ph), (0, pw)), mode="reflect") if (ph or pw) else raw
        rgb = _run_graph(session, {"raw": patch, "masks": _masks})
        return rgb[:H, :W]

    out = np.zeros((H, W, 3), np.float32)
    step = TILE - 2 * OVERLAP
    for y in range(0, H, step):
        for x in range(0, W, step):
            y0 = max(0, min(y - OVERLAP, H - TILE))
            x0 = max(0, min(x - OVERLAP, W - TILE))
            # keep the tile origin on the 6px CFA grid
            y0 -= y0 % 6
            x0 -= x0 % 6
            y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
            patch = raw[y0:y1, x0:x1]
            th, tw = patch.shape
            if th < TILE or tw < TILE:
                patch = np.pad(patch, ((0, TILE - th), (0, TILE - tw)), mode="reflect")
            rgb = _run_graph(session, {"raw": np.ascontiguousarray(patch), "masks": _masks})
            iy0, ix0 = y, x
            iy1, ix1 = min(H, y + step), min(W, x + step)
            out[iy0:iy1, ix0:ix1] = rgb[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    return out
