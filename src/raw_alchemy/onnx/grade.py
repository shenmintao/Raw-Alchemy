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

_session = None
_session_lock = threading.Lock()
_session_provider = None
_INPUT_NAMES = (
    "img", "gain", "mat_a", "hl", "sh", "sat", "con", "pivot",
    "luma", "mat_b", "srgb_flag",
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


def _get_session():
    global _session, _session_provider
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        import onnxruntime as ort

        model_path = _find_model(MODEL_FILE)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _get_providers()
        provider_options = [
            ({"device_id": 0} if p in ("CUDAExecutionProvider", "DmlExecutionProvider") else {})
            for p in providers
        ]
        _session = ort.InferenceSession(
            model_path, so, providers=providers, provider_options=provider_options,
        )
        _session_provider = _session.get_providers()[0]
        logger.info(f"Grade session: {_session_provider}")
    return _session


def clear_session() -> None:
    global _session, _session_provider
    with _session_lock:
        _session = None
        _session_provider = None
    import gc
    gc.collect()


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
    session = _get_session()
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
    return session.run(None, feeds)[0]
