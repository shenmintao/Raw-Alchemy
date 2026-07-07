"""RCD Bayer demosaic on ONNX Runtime (GPU via DirectML/CUDA, CPU fallback).

Replaces the retired Taichi port. The graph is a vectorized translation of
the same algorithm (validated pixel-exact against the Taichi reference,
max |delta| ~2e-7 on real mosaics) exported with dynamic H/W.

Inference is TILED at a fixed 1536px tile (overlap 24 >> the algorithm's
~8px neighbourhood + 4px border logic):
- DirectML compiles the graph ONCE for the single tile shape — no per-sensor
  4-6s dimension-freeze, no pathological dynamic-shape partitions;
- VRAM is bounded by the tile working set (~0.4GB) instead of a full-frame
  arena (a frozen 42MP graph pinned multiple GB per sensor size).

Runs fp32 — the algorithm's EPSSQ=1e-10 discriminators underflow in fp16.

Input:  (H, W) float32 Bayer mosaic, black-subtracted, normalized [0, 1]
Output: (H, W, 3) float32 RGB
"""

import threading
import time

import numpy as np
from loguru import logger

from .denoiser import _find_model, _get_providers

MODEL_FILE = "rcd_demosaic_dyn.onnx"
TILE = 1536   # even (CFA phase), single compiled shape
OVERLAP = 24  # even; > border(4) + neighbourhood influence (~8px)

_sessions: dict = {}
_session_lock = threading.Lock()
_session_provider = None


def _get_session():
    global _session_provider
    sess = _sessions.get(TILE)
    if sess is not None:
        return sess
    with _session_lock:
        sess = _sessions.get(TILE)
        if sess is not None:
            return sess
        import onnxruntime as ort

        model_path = _find_model(MODEL_FILE)
        # Freeze the (constant) tile dims into the graph: DirectML compiles a
        # static graph in <1s, vs ~11s JIT for the dynamic one at first use.
        model_bytes = None
        try:
            import onnx

            m = onnx.load(model_path)
            for d in m.graph.input[0].type.tensor_type.shape.dim:
                d.ClearField("dim_param")
                d.dim_value = TILE
            m = onnx.shape_inference.infer_shapes(m)
            model_bytes = m.SerializeToString()
        except Exception as e:
            logger.warning(f"RCD: tile-dim freeze unavailable ({e}); dynamic graph")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if _cpu_fallback:
            providers = ["CPUExecutionProvider"]
        else:
            providers = _get_providers()
        provider_options = [
            ({"device_id": 0} if p in ("CUDAExecutionProvider", "DmlExecutionProvider") else {})
            for p in providers
        ]
        sess = ort.InferenceSession(
            model_bytes if model_bytes is not None else model_path,
            so, providers=providers, provider_options=provider_options,
        )
        _sessions[TILE] = sess
        _session_provider = sess.get_providers()[0]
        logger.info(f"RCD demosaic session (tile {TILE}): {_session_provider}")
    return sess


def clear_session() -> None:
    global _session_provider
    with _session_lock:
        _sessions.clear()
        _session_provider = None
    import gc
    gc.collect()


def _phase_masks(cfa_pattern: np.ndarray) -> np.ndarray:
    """(3, 2, 2) R/G/B position masks from a 2x2 CFA pattern (G2 -> G)."""
    m2 = np.zeros((3, 2, 2), np.float32)
    for r in range(2):
        for c in range(2):
            color = int(cfa_pattern[r, c])
            if color == 3:
                color = 1
            m2[color, r, c] = 1.0
    return m2


_cpu_fallback = False


def _run_tile(session, patch: np.ndarray, m2: np.ndarray) -> np.ndarray:
    global _cpu_fallback
    feeds = {
        "bayer": np.ascontiguousarray(patch, dtype=np.float32),
        "mr2": m2[0], "mg2": m2[1], "mb2": m2[2],
    }
    try:
        return session.run(None, feeds)[0]
    except Exception as e:
        # DML 运行时故障(常见:显存耗尽;Windows 本地化错误文本还会被
        # pybind 以 utf-8 解码炸成 UnicodeDecodeError,掩盖真实原因)。
        # 重建 CPU-EP 会话兜底:慢,但绝不让解码整体失败。
        logger.warning(
            f"RCD tile failed on {_session_provider} "
            f"({type(e).__name__}: {str(e)[:80]}); rebuilding on CPU EP")
        _cpu_fallback = True
        clear_session()
        return _get_session().run(None, feeds)[0]


def rcd_demosaic(bayer: np.ndarray, cfa_pattern: np.ndarray) -> np.ndarray:
    """Demosaic a Bayer mosaic with RCD on the ONNX runtime (tiled).

    Args:
        bayer: (H, W) float32, black-level subtracted, normalized to [0, 1]
        cfa_pattern: (2, 2) ints, 0=R 1=G 2=B 3=G2

    Returns:
        (H, W, 3) float32 RGB in [0, 1]
    """
    if bayer.ndim != 2:
        raise ValueError(f"expected (H, W) mosaic, got {bayer.shape}")
    H, W = bayer.shape
    if H < 10 or W < 10 or H % 2 or W % 2:
        raise ValueError(f"unsupported mosaic size {W}x{H}")

    t0 = time.time()
    session = _get_session()
    m2 = _phase_masks(np.asarray(cfa_pattern))

    if H <= TILE and W <= TILE:
        ph, pw = TILE - H, TILE - W
        patch = (np.pad(bayer, ((0, ph), (0, pw)), mode="reflect")
                 if (ph or pw) else bayer)
        rgb = _run_tile(session, patch, m2)[:H, :W]
    else:
        rgb = np.zeros((H, W, 3), np.float32)
        step = TILE - 2 * OVERLAP
        for y in range(0, H, step):
            for x in range(0, W, step):
                y0 = max(0, min(y - OVERLAP, H - TILE))
                x0 = max(0, min(x - OVERLAP, W - TILE))
                y0 -= y0 % 2  # keep CFA phase
                x0 -= x0 % 2
                y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
                patch = bayer[y0:y1, x0:x1]
                th, tw = patch.shape
                if th < TILE or tw < TILE:
                    patch = np.pad(patch, ((0, TILE - th), (0, TILE - tw)),
                                   mode="reflect")
                out = _run_tile(session, patch, m2)
                iy0, ix0 = y, x
                iy1, ix1 = min(H, y + step), min(W, x + step)
                rgb[iy0:iy1, ix0:ix1] = out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]

    logger.info(
        f"RCD demosaic: {W}x{H} in {(time.time() - t0) * 1000:.0f}ms "
        f"({_session_provider})"
    )
    return rgb
