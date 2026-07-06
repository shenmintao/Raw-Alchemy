"""RCD Bayer demosaic on ONNX Runtime (GPU via DirectML/CUDA, CPU fallback).

Replaces the retired Taichi port of darktable's RCD. The graph is a
vectorized translation of the same algorithm (validated pixel-exact against
the Taichi reference, max |delta| ~2e-7 on real mosaics) exported with
dynamic H/W, so one 448KB model serves every sensor size.

Runs fp32 — the algorithm's EPSSQ=1e-10 discriminators underflow in fp16.
Measured on RX 9070 XT (DirectML): 24MP ~0.31s, 42.6MP ~0.53s per frame.

Input:  (H, W) float32 Bayer mosaic, black-subtracted, normalized [0, 1]
Output: (H, W, 3) float32 RGB
"""

import threading
import time

import numpy as np
from loguru import logger

from .denoiser import _find_model, _get_providers

MODEL_FILE = "rcd_demosaic_dyn.onnx"

# DirectML compiles/partitions dynamic-shape graphs poorly for some sizes
# (24MP ran 20x slower than 42.6MP on the same graph), so the runtime
# freezes the dynamic model's H/W to the actual sensor size before creating
# a session — one session per distinct size, cached (cameras have one size).
_sessions: dict = {}
_session_lock = threading.Lock()
_session_provider = None


def _get_session(h: int, w: int):
    global _session_provider
    key = (h, w)
    sess = _sessions.get(key)
    if sess is not None:
        return sess
    with _session_lock:
        sess = _sessions.get(key)
        if sess is not None:
            return sess
        import onnxruntime as ort

        model_path = _find_model(MODEL_FILE)
        model_bytes = None
        try:
            import onnx

            m = onnx.load(model_path)
            for d, v in zip(m.graph.input[0].type.tensor_type.shape.dim, (h, w)):
                d.ClearField("dim_param")
                d.dim_value = int(v)
            m = onnx.shape_inference.infer_shapes(m)
            model_bytes = m.SerializeToString()
        except Exception as e:
            logger.warning(f"RCD: dim freeze unavailable ({e}); using dynamic graph")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _get_providers()
        provider_options = [
            ({"device_id": 0} if p in ("CUDAExecutionProvider", "DmlExecutionProvider") else {})
            for p in providers
        ]
        sess = ort.InferenceSession(
            model_bytes if model_bytes is not None else model_path,
            so, providers=providers, provider_options=provider_options,
        )
        _sessions[key] = sess
        _session_provider = sess.get_providers()[0]
        logger.info(f"RCD demosaic session ({w}x{h}): {_session_provider}")
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


def rcd_demosaic(bayer: np.ndarray, cfa_pattern: np.ndarray) -> np.ndarray:
    """Demosaic a Bayer mosaic with RCD on the ONNX runtime.

    Args:
        bayer: (H, W) float32, black-level subtracted, normalized to [0, 1]
        cfa_pattern: (2, 2) ints, 0=R 1=G 2=B 3=G2

    Returns:
        (H, W, 3) float32 RGB in [0, 1]
    """
    if bayer.ndim != 2:
        raise ValueError(f"expected (H, W) mosaic, got {bayer.shape}")
    h, w = bayer.shape
    if h < 10 or w < 10 or h % 2 or w % 2:
        raise ValueError(f"unsupported mosaic size {w}x{h}")

    t0 = time.time()
    session = _get_session(h, w)
    m2 = _phase_masks(np.asarray(cfa_pattern))
    feeds = {
        "bayer": np.ascontiguousarray(bayer, dtype=np.float32),
        "mr2": m2[0], "mg2": m2[1], "mb2": m2[2],
    }
    rgb = session.run(None, feeds)[0]
    logger.info(
        f"RCD demosaic: {w}x{h} in {(time.time() - t0) * 1000:.0f}ms "
        f"({_session_provider})"
    )
    return rgb
