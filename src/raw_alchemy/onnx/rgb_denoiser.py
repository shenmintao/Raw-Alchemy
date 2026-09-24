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

Level re-anchoring: the v4 weights shift levels by themselves (a noise-free
flat grey comes back +5..9% brighter per channel in the encoded domain, and
dark levels turn green: G up, R/B down, up to 1.7x/0.3x in ISO 12800 night
skies). The output's local mean is therefore re-anchored to the input's, per
luminance band (a bilateral grid on the denoised luminance), so a dark sky
never borrows the correction of a lit building next to it. Only the level is
touched; the denoised detail and noise reduction stay the network's.
"""

from raw_alchemy.pipeline.resources import checkpoint
import os
import time
import threading
from typing import Callable, Optional

import numpy as np
from loguru import logger

from .denoiser import (
    _configure_providers,
    _find_model,
    _get_providers,
    _make_session_options,
    _tile_weight,
)

from .session_policy import configuration_token, create_session, registered_provider_names

MODEL_FILE = "fastdenoise_v4_512_fp16.onnx"
MODEL_TILE = 512
DEFAULT_OVERLAP = 64

GAMMA = 2.2
# Auto-gain: scale so the (luma) mean lands at mid-grey, within sane bounds.
GAIN_TARGET = 0.18
GAIN_MAX = 64.0

# Level re-anchoring (see module docstring). Statistics live on a grid of
# LEVEL_GRID_CELL px cells x LEVEL_STEP_STOPS luminance bands, blurred over
# LEVEL_SIGMA_PX; the ratio map is sliced at 1/LEVEL_SLICE resolution. On 8
# real frames this brought mid-tone and shadow levels to 1.00 +/- 0.01 of the
# input with no halo at 1:1 (a plain 96 px blur left a green rim round lit
# buildings), for ~0.6 s at 42 MP.
LEVEL_SIGMA_PX = 96
LEVEL_SLICE = 4
LEVEL_GRID_CELL = 32
LEVEL_STEP_STOPS = 1.0
LEVEL_RATIO_RANGE = (0.2, 5.0)
LEVEL_PRIOR = 0.05  # sparse bands lean on the band's frame-wide ratio
LEVEL_MIN_SIDE = 64
_PROPHOTO_Y = np.array([0.2880402, 0.7118741, 0.0000857], np.float32)

_session = None
_session_provider = None
_session_token = None
_session_lock = threading.Lock()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_session():
    global _session, _session_provider, _session_token
    token = (MODEL_FILE, configuration_token("rgb-denoiser"))
    if _session is not None and _session_token == token:
        return _session
    with _session_lock:
        if _session is not None and _session_token == token:
            return _session
        import onnxruntime as ort
        model_path = _find_model(MODEL_FILE)
        providers = _configure_providers(
            _get_providers(), model_path, variant="rgb-denoiser"
        )
        session = create_session(
            ort, model_path, lambda: _make_session_options(ort), providers,
            variant="rgb-denoiser",
        )
        _session = session
        _session_token = token
        _session_provider = session.get_providers()[0]
        return session


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


def _area(img: np.ndarray, factor: int) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(np.ascontiguousarray(img), (max(1, w // factor), max(1, h // factor)),
                      interpolation=cv2.INTER_AREA)


def level_ratio_map(source: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """Per-pixel ratio (at 1/LEVEL_SLICE) that restores ``source``'s local level.

    Bands are log2 luminance of the *denoised* image: a clean guide, so noise
    in ``source`` cannot bias which band a pixel lands in. Membership is soft
    (linear between neighbouring bands) so the ratio is continuous in level.
    """
    import cv2
    a = _area(source, LEVEL_SLICE)
    b = _area(denoised, LEVEL_SLICE)
    h, w = b.shape[:2]
    g = max(1, LEVEL_GRID_CELL // LEVEL_SLICE)
    gh, gw = -(-h // g), -(-w // g)
    lum = cv2.transform(b, _PROPHOTO_Y[None, :])
    L = np.log2(np.maximum(lum, np.float32(2.0 ** -24)))
    lmin, lmax = np.percentile(L[::2, ::2], [0.5, 99.9])
    K = int(np.floor((lmax - lmin) / LEVEL_STEP_STOPS)) + 2
    t = np.clip((L - np.float32(lmin)) / np.float32(LEVEL_STEP_STOPS), 0, K - 1 - 1e-4)
    i0 = t.astype(np.int32)
    fr = (t - i0).astype(np.float32)
    del L, t, lum

    # Splat: two bands per pixel, box-summed into grid cells.
    cell = (np.arange(h, dtype=np.int64)[:, None] // g) * gw + np.arange(w, dtype=np.int64)[None, :] // g
    idx0 = (i0.astype(np.int64) * (gh * gw) + cell).ravel()
    idx1 = idx0 + gh * gw
    w1 = fr.ravel()
    w0 = 1.0 - w1
    n = K * gh * gw
    sums = []
    for img in (a, b):
        chans = [np.bincount(idx0, img[..., c].ravel() * w0, n)
                 + np.bincount(idx1, img[..., c].ravel() * w1, n) for c in range(3)]
        sums.append(np.stack(chans, -1).reshape(K, gh, gw, 3).astype(np.float32))
    del cell, idx0, idx1, w0, w1
    sa, sb = sums
    tot_a = sa.reshape(K, -1, 3).sum(1)
    tot_b = sb.reshape(K, -1, 3).sum(1)
    band_ratio = np.ones((K, 3), np.float32)
    filled = np.flatnonzero(tot_b.min(1) > 0)
    if len(filled):
        for c in range(3):
            band_ratio[:, c] = np.interp(np.arange(K), filled, tot_a[filled, c] / tot_b[filled, c])

    lo, hi = LEVEL_RATIO_RANGE
    sigma = LEVEL_SIGMA_PX / (LEVEL_SLICE * g)
    # Bands side by side with a replicated 1-cell border, so one bilinear
    # remap per neighbouring band slices the grid without bleeding.
    grid = np.empty((gh + 2, K * (gw + 2), 3), np.float32)
    for k in range(K):
        ga = cv2.GaussianBlur(sa[k], (0, 0), sigma)
        gb = cv2.GaussianBlur(sb[k], (0, 0), sigma)
        prior = np.float32(LEVEL_PRIOR) * np.maximum(gb.reshape(-1, 3).mean(0), np.float32(1e-12))
        ratio = np.clip((ga + prior * band_ratio[k]) / (gb + prior), lo, hi)
        grid[:, k * (gw + 2):(k + 1) * (gw + 2)] = cv2.copyMakeBorder(ratio, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    gx = np.clip((np.arange(w, dtype=np.float32) + 0.5) / g - 0.5, 0, gw - 1) + 1
    gy = np.clip((np.arange(h, dtype=np.float32) + 0.5) / g - 0.5, 0, gh - 1) + 1
    map_y = np.ascontiguousarray(np.broadcast_to(gy[:, None], (h, w)))
    map_x = i0.astype(np.float32) * np.float32(gw + 2) + gx[None, :]
    r0 = cv2.remap(grid, map_x, map_y, cv2.INTER_LINEAR)
    map_x += np.float32(gw + 2)
    r1 = cv2.remap(grid, map_x, map_y, cv2.INTER_LINEAR)
    r1 -= r0
    r1 *= fr[..., None]
    r0 += r1
    return r0


def _passthrough_saturated(out_flat, src_flat, gain, start, stop, scratch):
    src_chunk = scratch[: stop - start]
    np.clip(src_flat[start:stop], 0.0, 1.0, out=src_chunk)
    saturated = src_chunk * np.float32(gain) >= np.float32(1.0)
    np.copyto(out_flat[start:stop], src_chunk, where=saturated)


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
    source = linear_rgb.astype(np.float32, copy=False)
    gain = compute_gain(source)

    H, W = source.shape[:2]
    tile = MODEL_TILE
    overlap = int(np.clip(tile_overlap, 0, tile - 1))
    step = tile - overlap

    PH, PW = max(tile, H), max(tile, W)

    ys = list(range(0, PH - tile + 1, step))
    xs = list(range(0, PW - tile + 1, step))
    if ys[-1] + tile < PH:
        ys.append(PH - tile)
    if xs[-1] + tile < PW:
        xs.append(PW - tile)
    total = len(ys) * len(xs)

    # The accumulator becomes the returned output in place.  The old path
    # also materialised full-frame lin/gained/clipped/encoded/out_enc/out_lin
    # arrays, pushing a 61MP denoise several gigabytes above the source image.
    accum = np.zeros((PH, PW, 3), np.float32)
    weight = np.zeros((PH, PW, 1), np.float32)
    chw = np.empty((3, tile, tile), np.float32)
    sigma = np.full((1, 1, tile, tile), strength, np.float32)
    weight_cache = {}
    done = 0
    for y in ys:
        for x in xs:
            checkpoint()
            patch = source[y:min(y + tile, H), x:min(x + tile, W)]
            ph, pw = patch.shape[:2]
            if ph < tile or pw < tile:
                patch = np.pad(
                    patch,
                    ((0, tile - ph), (0, tile - pw), (0, 0)),
                    mode="reflect",
                )
            np.copyto(chw, patch.transpose(2, 0, 1))
            np.clip(chw, np.float32(0.0), np.float32(1.0), out=chw)
            chw *= np.float32(gain)
            np.clip(chw, np.float32(0.0), np.float32(1.0), out=chw)
            np.power(chw, np.float32(1.0 / GAMMA), out=chw)

            pred = session.run(
                None, {"rgb": chw[np.newaxis], "sigma": sigma}
            )[0][0].transpose(1, 2, 0)
            edge_key = (y == 0, y + tile >= PH, x == 0, x + tile >= PW)
            wt = weight_cache.get(edge_key)
            if wt is None:
                wt = _tile_weight(
                    tile, tile, overlap,
                    at_top=edge_key[0], at_bottom=edge_key[1],
                    at_left=edge_key[2], at_right=edge_key[3],
                )[0][..., np.newaxis]
                weight_cache[edge_key] = wt
            # ORT outputs are disposable. Weight in place to avoid another
            # full tile temporary before overlap-add.
            pred *= wt
            accum[y:y + tile, x:x + tile] += pred
            weight[y:y + tile, x:x + tile] += wt
            done += 1
            if progress_callback:
                progress_callback(done, total)

    np.maximum(weight, np.float32(1e-8), out=weight)
    np.divide(accum, weight, out=accum)
    out_lin = accum[:H, :W]
    if not out_lin.flags["C_CONTIGUOUS"]:
        out_lin = np.ascontiguousarray(out_lin)
    np.clip(out_lin, np.float32(0.0), np.float32(1.0), out=out_lin)
    np.power(out_lin, np.float32(GAMMA), out=out_lin)
    out_lin /= np.float32(gain)

    # Saturated-after-gain pixels are passthrough, evaluated in bounded
    # chunks instead of retaining a full-frame boolean mask.
    out_flat = out_lin.reshape(-1, 3)
    src_flat = source.reshape(-1, 3)
    rows = max(1, 1_000_000 // W)
    scratch = np.empty((min(rows * W, src_flat.shape[0]), 3), np.float32)
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        _passthrough_saturated(out_flat, src_flat, gain, y0 * W, y1 * W, scratch)

    t_level = time.time()
    if min(H, W) >= LEVEL_MIN_SIDE:
        import cv2
        # Statistics see the passthrough pixels (ratio 1), not the model's
        # output for clipped input, which would brighten their whole band.
        checkpoint()
        ratio = level_ratio_map(source, out_lin)
        checkpoint()
        rh, rw = ratio.shape[:2]
        map_x = (np.arange(W, dtype=np.float32) + 0.5) * np.float32(rw / W) - 0.5
        map_x = np.ascontiguousarray(np.broadcast_to(map_x, (rows, W)))
        for y0 in range(0, H, rows):
            y1 = min(H, y0 + rows)
            map_y = (np.arange(y0, y1, dtype=np.float32) + 0.5) * np.float32(rh / H) - 0.5
            map_y = np.ascontiguousarray(np.broadcast_to(map_y[:, None], (y1 - y0, W)))
            out_lin[y0:y1] *= cv2.remap(ratio, map_x[: y1 - y0], map_y, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
            _passthrough_saturated(out_flat, src_flat, gain, y0 * W, y1 * W, scratch)
        del ratio
    np.clip(out_lin, np.float32(0.0), np.float32(1.0), out=out_lin)
    logger.info(
        f"FastDenoise v4 (s={strength:.2f}) done in {time.time() - t0:.1f}s "
        f"(level re-anchor {time.time() - t_level:.2f}s, "
        f"{total} tiles, gain {gain:.1f}x, registered={registered_provider_names(session)})"
    )
    return np.ascontiguousarray(out_lin, dtype=np.float32)


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
    global _session, _session_provider, _session_token
    with _session_lock:
        _session = None
        _session_provider = None
        _session_token = None
    import gc
    gc.collect()
