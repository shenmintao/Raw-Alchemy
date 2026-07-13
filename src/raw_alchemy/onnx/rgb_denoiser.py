"""FastDenoise v4 RGB denoiser (ONNX) — the app's denoise engine.

自研 DML 亲和架构(纯密集卷积,主干 1/4 分辨率,6.1M/12MB fp16),
训练数据与本管线逐比特对齐(合成标定噪声 + SID/RawNIND 真实配对 +
SCUNet 蒸馏)。RX 9070 XT 实测 2.2ms/tile,42.6MP ≈ 0.5s(SCUNet 42s)。
噪声强度 σ 为条件输入 → UI 降噪强度滑块(默认 0.25)。
蒸馏容器原则:将来任何更强 teacher 都可经蒸馏管线注入本模型升级画质。

Runs on the demosaiced linear ProPhoto RGB image (HWC float32 [0,1]), so the
pipeline contract is unchanged for everything downstream: WB/matrix/edits all
operate on linear ProPhoto exactly as before.

Encoding round-trip: SCUNet (scunet_color_real_psnr, Apache-2.0) is trained on
display-referred sRGB photographs, so the linear image is auto-gained to a
mid-grey target and gamma-encoded before inference, then decoded and un-gained
after. The gain makes night shots (linear mean ~0.005) look to the network
like the ordinarily-exposed photos it was trained on. Pixels the gain would
clip (gain * lin >= 1) are returned unchanged — they are saturated highlights
carrying no recoverable noise.

Model: vendor/scunet_real_512_fp16.onnx — 3ch in/out, fixed 512x512 tiles,
overlap feathered with the same raised-cosine window as the old raw engine.
"""

import os
import time
from typing import Callable, Optional

import numpy as np
from loguru import logger

from .denoiser import _find_model, _get_providers, _tile_weight

MODEL_FILE = "fastdenoise_v4_512_fp16.onnx"
MODEL_TILE = 512
DEFAULT_OVERLAP = 64

GAMMA = 2.2
# Auto-gain: scale so the (luma) mean lands at mid-grey, within sane bounds.
GAIN_TARGET = 0.18
GAIN_MAX = 64.0

_session = None
_session_provider = None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_session():
    global _session, _session_provider
    if _session is not None:
        return _session
    import onnxruntime as ort

    model_path = _find_model(MODEL_FILE)
    logger.info(f"Loading FastDenoise v4 from: {model_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = _get_providers()
    provider_options = [
        ({"device_id": 0} if p in ("CUDAExecutionProvider", "DmlExecutionProvider") else {})
        for p in providers
    ]
    _session = ort.InferenceSession(
        model_path, sess_options,
        providers=providers, provider_options=provider_options,
    )
    _session_provider = _session.get_providers()[0]
    logger.info(f"FastDenoise session: {_session_provider}")
    return _session


def is_available() -> bool:
    """True if the model file is present (session not necessarily created)."""
    try:
        _find_model(MODEL_FILE)
        return True
    except FileNotFoundError:
        return False


def compute_gain(linear_rgb: np.ndarray) -> float:
    """Exposure gain that brings the image mean to mid-grey (clamped)."""
    mean = float(linear_rgb.mean())
    if not np.isfinite(mean) or mean <= 0:
        return 1.0
    return float(np.clip(GAIN_TARGET / mean, 1.0, GAIN_MAX))


def denoise_rgb_linear(
    linear_rgb: np.ndarray,
    strength: float = 0.25,
    tile_overlap: int = DEFAULT_OVERLAP,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Denoise linear ProPhoto RGB (HWC float32 [0,1]) -> same space/shape."""
    if linear_rgb.ndim != 3 or linear_rgb.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB, got {linear_rgb.shape}")
    t0 = time.time()
    session = _get_session()

    # 上限 0.5:σ 扫描实测(scratch sigma_cast_sweep)σ 超过 0.5 后中性灰
    # R/G、B/G 漂移超 -5%(偏绿),两种曝光/噪声水平下单调恶化;0.30-0.45
    # 是最干净带。旧 sidecar 里 >0.5 的值在此一并夹回。
    strength = float(np.clip(strength, 0.01, 0.5))
    lin = np.clip(linear_rgb.astype(np.float32, copy=False), 0.0, 1.0)
    gain = compute_gain(lin)
    gained = lin * gain
    clipped = gained >= 1.0  # saturated after gain: passthrough at the end
    enc = np.clip(gained, 0.0, 1.0) ** (1.0 / GAMMA)

    H, W = enc.shape[:2]
    tile = MODEL_TILE
    overlap = int(np.clip(tile_overlap, 0, tile - 1))
    step = tile - overlap

    pad_h = max(tile - H, 0)
    pad_w = max(tile - W, 0)
    if pad_h or pad_w:
        enc = np.pad(enc, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    PH, PW = enc.shape[:2]

    ys = list(range(0, PH - tile + 1, step))
    xs = list(range(0, PW - tile + 1, step))
    if ys[-1] + tile < PH:
        ys.append(PH - tile)
    if xs[-1] + tile < PW:
        xs.append(PW - tile)
    total = len(ys) * len(xs)

    accum = np.zeros((PH, PW, 3), np.float32)
    weight = np.zeros((PH, PW, 1), np.float32)
    done = 0
    for y in ys:
        for x in xs:
            patch = enc[y:y + tile, x:x + tile]
            chw = np.ascontiguousarray(patch.transpose(2, 0, 1))[np.newaxis]
            sig = np.full((1, 1, tile, tile), strength, np.float32)
            pred = session.run(None, {"rgb": chw, "sigma": sig})[0][0].transpose(1, 2, 0)
            wt = _tile_weight(
                tile, tile, overlap,
                at_top=(y == 0), at_bottom=(y + tile >= PH),
                at_left=(x == 0), at_right=(x + tile >= PW),
            )[0][..., np.newaxis]
            accum[y:y + tile, x:x + tile] += pred * wt
            weight[y:y + tile, x:x + tile] += wt
            done += 1
            if progress_callback:
                progress_callback(done, total)

    out_enc = (accum / np.maximum(weight, 1e-8))[:H, :W]
    out_lin = np.clip(out_enc, 0.0, 1.0) ** GAMMA / gain
    out_lin = np.where(clipped, lin, out_lin)
    logger.info(
        f"FastDenoise v4 (s={strength:.2f}) done in {time.time() - t0:.1f}s "
        f"({total} tiles, gain {gain:.1f}x, {_session_provider})"
    )
    return np.clip(out_lin, 0.0, 1.0).astype(np.float32)


def warmup() -> None:
    """Create the session ahead of first use (optional)."""
    try:
        _get_session()
    except Exception as e:
        logger.warning(f"FastDenoise warmup failed: {e}")


def clear_session() -> None:
    """Release the ONNX session (frees GPU memory between edits).

    The DirectML provider holds its D3D12 allocations until the session
    object is actually destroyed, so collect immediately — pybind objects
    routinely sit in reference cycles that plain refcounting won't clear.
    """
    global _session, _session_provider
    _session = None
    _session_provider = None
    import gc
    gc.collect()
