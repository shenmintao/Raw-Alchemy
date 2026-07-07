"""Fused colour-grade graph on ONNX Runtime (GPU via DirectML/CUDA).

One 5KB dynamic-shape graph applies the whole interactive colour tail —
gain -> WB matrix -> highlight/shadow -> saturation/contrast -> output
matrix -> sRGB OETF -> clip — in a single GPU call with the parameters fed
as graph inputs, so slider changes never rebuild anything. The math is a
line-by-line port of the math_ops kernels (max |delta| ~8e-7).

Measured on RX 9070 XT (DirectML): 3MP proxy 16ms, 8.3MP ROI 45ms,
18.7MP ROI 99ms — the numpy fallback path takes 0.5-1.2s at ROI sizes.
Elementwise-only graphs do not exhibit DirectML's dynamic-shape pathology
(verified across sizes), so no dimension freezing is needed here.

Disable with RAWALCHEMY_GRADE_GPU=0 (falls back to the per-op numpy path).
"""

import os
import threading

import numpy as np
from loguru import logger

from .denoiser import _find_model, _get_providers

MODEL_FILE = "grade_dyn.onnx"
MODEL_FILE_LOG = "grade_log_dyn.onnx"   # ...→log 矩阵→max→1D LUT→[3D LUT]
MODEL_FILE_LUT = "grade_lut_dyn.onnx"   # ...→3D LUT→sRGB 矩阵→OETF

_sessions: dict = {}
_session_lock = threading.Lock()
_session_provider = None

# 3D-LUT 直通用的最小恒等表(S=2):四面体插值在恒等格点上重建输入本身。
_IDENTITY_LUT3 = np.array(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32,
)


def is_enabled() -> bool:
    if os.environ.get("RAWALCHEMY_GRADE_GPU", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return False
    try:
        _find_model(MODEL_FILE)
        return True
    except FileNotFoundError:
        return False


def _get_session(model_file: str):
    global _session_provider
    sess = _sessions.get(model_file)
    if sess is not None:
        return sess
    with _session_lock:
        sess = _sessions.get(model_file)
        if sess is not None:
            return sess
        import onnxruntime as ort

        model_path = _find_model(model_file)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _get_providers()
        provider_options = [
            ({"device_id": 0} if p in ("CUDAExecutionProvider", "DmlExecutionProvider") else {})
            for p in providers
        ]
        sess = ort.InferenceSession(
            model_path, so, providers=providers, provider_options=provider_options,
        )
        _sessions[model_file] = sess
        _session_provider = sess.get_providers()[0]
        logger.info(f"Grade session ({model_file}): {_session_provider}")
    return sess


def clear_session() -> None:
    global _session_provider
    with _session_lock:
        _sessions.clear()
        _session_provider = None
    import gc
    gc.collect()


_STRIP_PIXELS = 6_000_000  # 单次喂图上限:约束 DML arena 增长(逐像素链,条带切分数学恒等)


def _run_strips(session, feeds, img_key="img"):
    img = feeds[img_key]
    h, w = img.shape[:2]
    if h * w <= _STRIP_PIXELS:
        return session.run(None, feeds)[0]
    rows = max(1, _STRIP_PIXELS // max(w, 1))
    out = np.empty((h, w, 3), np.float32)
    for y in range(0, h, rows):
        part = dict(feeds)
        part[img_key] = np.ascontiguousarray(img[y:y + rows])
        out[y:y + rows] = session.run(None, part)[0]
    return out


def apply_grade(
    img: np.ndarray,
    *,
    gain: float,
    mat_a: np.ndarray,
    highlight: float,
    shadow: float,
    saturation: float,
    contrast: float,
    pivot: float,
    luma: np.ndarray,
    mat_b: np.ndarray,
    srgb_encode: bool,
) -> np.ndarray:
    """Run the fused grade on HWC float32; returns clipped [0,1] float32."""
    session = _get_session(MODEL_FILE)
    feeds = {
        "img": np.ascontiguousarray(img, dtype=np.float32),
        "gain": np.array(gain, np.float32),
        "mat_a": np.ascontiguousarray(mat_a, dtype=np.float32),
        "hl": np.array(highlight, np.float32),
        "sh": np.array(shadow, np.float32),
        "sat": np.array(saturation, np.float32),
        "con": np.array(contrast, np.float32),
        "pivot": np.array(pivot, np.float32),
        "luma": np.ascontiguousarray(luma, dtype=np.float32),
        "mat_b": np.ascontiguousarray(mat_b, dtype=np.float32),
        "srgb_flag": np.array(1.0 if srgb_encode else 0.0, np.float32),
    }
    return _run_strips(session, feeds)


def _core_feeds(img, gain, mat_a, highlight, shadow, saturation, contrast,
                pivot, luma):
    return {
        "img": np.ascontiguousarray(img, dtype=np.float32),
        "gain": np.array(gain, np.float32),
        "mat_a": np.ascontiguousarray(mat_a, dtype=np.float32),
        "hl": np.array(highlight, np.float32),
        "sh": np.array(shadow, np.float32),
        "sat": np.array(saturation, np.float32),
        "con": np.array(contrast, np.float32),
        "pivot": np.array(pivot, np.float32),
        "luma": np.ascontiguousarray(luma, dtype=np.float32),
    }


def apply_grade_log(
    img: np.ndarray, *, gain, mat_a, highlight, shadow, saturation, contrast,
    pivot, luma, mat_log, lut1d, d1_min, d1_max,
    lut3d_flat=None, lut3d_size=0, d3_min=None, d3_max=None,
) -> np.ndarray:
    """Fused grade ending in a log encode (matrix -> max -> 1D LUT [-> 3D])."""
    session = _get_session(MODEL_FILE_LOG)
    feeds = _core_feeds(img, gain, mat_a, highlight, shadow, saturation,
                        contrast, pivot, luma)
    use3 = lut3d_flat is not None
    feeds.update({
        "mat_b": np.ascontiguousarray(mat_log, dtype=np.float32),
        "lut1d": np.ascontiguousarray(lut1d, dtype=np.float32),
        "d1_min": np.array(d1_min, np.float32),
        "d1_max": np.array(d1_max, np.float32),
        "lut3d_flat": (np.ascontiguousarray(lut3d_flat, dtype=np.float32)
                       if use3 else _IDENTITY_LUT3),
        "lut3d_size": np.array(int(lut3d_size) if use3 else 2, np.int64),
        "d3_min": (np.ascontiguousarray(d3_min, dtype=np.float32)
                   if use3 else np.zeros(3, np.float32)),
        "d3_max": (np.ascontiguousarray(d3_max, dtype=np.float32)
                   if use3 else np.ones(3, np.float32)),
        "use_lut3d": np.array(1.0 if use3 else 0.0, np.float32),
    })
    return _run_strips(session, feeds)


def apply_grade_lut(
    img: np.ndarray, *, gain, mat_a, highlight, shadow, saturation, contrast,
    pivot, luma, lut3d_flat, lut3d_size, d3_min, d3_max, mat_b,
) -> np.ndarray:
    """Fused grade with a 3D LUT in working space, then sRGB out."""
    session = _get_session(MODEL_FILE_LUT)
    feeds = _core_feeds(img, gain, mat_a, highlight, shadow, saturation,
                        contrast, pivot, luma)
    feeds.update({
        "lut3d_flat": np.ascontiguousarray(lut3d_flat, dtype=np.float32),
        "lut3d_size": np.array(int(lut3d_size), np.int64),
        "d3_min": np.ascontiguousarray(d3_min, dtype=np.float32),
        "d3_max": np.ascontiguousarray(d3_max, dtype=np.float32),
        "mat_b": np.ascontiguousarray(mat_b, dtype=np.float32),
    })
    return _run_strips(session, feeds)
