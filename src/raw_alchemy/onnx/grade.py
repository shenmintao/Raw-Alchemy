"""Fused colour-grade graph on ONNX Runtime (stage-selected GPU or CPU).

One 5KB dynamic-shape graph applies the whole interactive colour tail —
gain -> WB matrix -> highlight/shadow -> saturation/contrast -> output
matrix -> sRGB OETF -> clip — in a single ORT call with the parameters fed
as graph inputs, so slider changes never rebuild anything. The math is a
line-by-line port of the math_ops kernels (max |delta| ~8e-7).

Measured on RX 9070 XT (DirectML): 3MP proxy 16ms, 8.3MP ROI 45ms,
18.7MP ROI 99ms — the numpy fallback path takes 0.5-1.2s at ROI sizes.
Elementwise-only graphs do not exhibit DirectML's dynamic-shape pathology
(verified across sizes), so no dimension freezing is needed here.

Disable with RAWALCHEMY_GRADE_GPU=0 (falls back to the per-op numpy path).
Apple Silicon uses MLProgram on fixed pixel tiles, reusing compiled GPU
graphs across image sizes. RAWALCHEMY_COREML_GRADE=cpu selects fused CPU ORT.
"""

import os
import threading

import numpy as np
from loguru import logger

from .denoiser import (
    _configure_providers,
    _find_model,
    _get_providers,
    _make_session_options,
)

from .session_policy import configuration_token, create_session, provider_names, stage_providers
from raw_alchemy.pipeline.resources import checkpoint

MODEL_FILE = "grade_dyn.onnx"
MODEL_FILE_LOG = "grade_log_dyn.onnx"   # ...→log 矩阵→max→1D LUT→[3D LUT]
MODEL_FILE_LUT = "grade_lut_dyn.onnx"   # ...→3D LUT→sRGB 矩阵→OETF
COREML_PIXEL_TILE = (1024, 3072)
# Gather-heavy tetrahedral LUT graphs need smaller working sets. Reuse one
# compiled shape per graph while bounding CoreML and CPU partition buffers.
COREML_LUT_PIXEL_TILE = (512, 1536)

_sessions: dict = {}
_session_lock = threading.Lock()
_session_provider = None
_session_token = None

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
    global _session_provider, _session_token
    token = configuration_token("grade")
    sess = _sessions.get(model_file)
    if sess is not None and _session_token == token:
        return sess
    with _session_lock:
        if _session_token != token:
            _sessions.clear()
            _session_token = token
        sess = _sessions.get(model_file)
        if sess is not None:
            return sess
        import onnxruntime as ort

        model_path = _find_model(model_file)
        preferred = _get_providers()
        use_coreml = "CoreMLExecutionProvider" in provider_names(stage_providers(preferred, "grade"))
        h, w = COREML_PIXEL_TILE if model_file == MODEL_FILE else COREML_LUT_PIXEL_TILE
        variant = f"grade:h={h},w={w}" if use_coreml else "grade"
        providers = _configure_providers(preferred, model_path, variant=variant)

        def options():
            so = _make_session_options(ort)
            if use_coreml:
                so.add_free_dimension_override_by_name("h", h)
                so.add_free_dimension_override_by_name("w", w)
            return so

        sess = create_session(
            ort, model_path, options, providers,
            variant=variant,
        )
        # The CPU retry retains these dimension overrides. Tile selection
        # must follow the session shape even after its provider changes.
        sess._rawalchemy_pixel_tile = (h, w) if use_coreml else None
        _sessions[model_file] = sess
        _session_provider = sess.get_providers()[0]
    return sess


def clear_session() -> None:
    global _session_provider, _session_token
    with _session_lock:
        _sessions.clear()
        _session_provider = None
        _session_token = None
    import gc
    gc.collect()


_STRIP_PIXELS = 6_000_000  # 单次喂图上限:约束 DML arena 增长(逐像素链,条带切分数学恒等)


def _run_strips(session, feeds, img_key="img"):
    if getattr(session, "_rawalchemy_pixel_tile", None) is not None:
        return _run_coreml_pixel_tiles(session, feeds, img_key)
    img = feeds[img_key]
    h, w = img.shape[:2]
    if h * w <= _STRIP_PIXELS:
        checkpoint()
        return session.run(None, feeds)[0]
    rows = max(1, _STRIP_PIXELS // max(w, 1))
    out = np.empty((h, w, 3), np.float32)
    for y in range(0, h, rows):
        checkpoint()
        part = dict(feeds)
        part[img_key] = np.ascontiguousarray(img[y:y + rows])
        out[y:y + rows] = session.run(None, part)[0]
    return out


def _run_coreml_pixel_tiles(session, feeds, img_key="img"):
    """Reuse one fixed CoreML shape across crops, zoom and image dimensions.

    Grade graphs have no spatial operators: only the last RGB axis matters.
    Flattening pixels into fixed tiles preserves the math and avoids a new
    compiled model for every ROI. Padding is discarded after inference.
    """
    img = feeds[img_key]
    flat = img.reshape(-1, 3)
    th, tw = session._rawalchemy_pixel_tile
    capacity = th * tw
    out = np.empty_like(img)
    dest = out.reshape(-1, 3)
    scratch = None
    for start in range(0, len(flat), capacity):
        checkpoint()
        count = min(capacity, len(flat) - start)
        if count == capacity:
            tile = flat[start:start + count].reshape(th, tw, 3)
        else:
            if scratch is None:
                scratch = np.zeros((th, tw, 3), np.float32)
            scratch.reshape(-1, 3)[:count] = flat[start:start + count]
            tile = scratch
        part = dict(feeds)
        part[img_key] = tile
        result = session.run(None, part)[0]
        dest[start:start + count] = result.reshape(-1, 3)[:count]
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
