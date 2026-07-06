"""
CANS raw-main v14 (cfa-v11) denoiser via ONNX Runtime.

Production path is CANS raw-main v14 (arch cfa-v11), a *raw-to-raw* denoiser:
  - cans_raw_v11_v14_ep10_bayer_fp16.onnx    (4ch Bayer packed RAW in/out, 768x768)
  - cans_raw_v11_v14_ep10_xtrans_fp16.onnx   (9ch X-Trans packed RAW in/out, 512x512)

Unlike the old CANS V2 (which demosaiced internally and emitted ProPhoto RGB),
v14 denoises *packed sensor-space RAW* and returns RAW of the same shape. The
denoised packed RAW is then unpacked, demosaiced by Raw-Alchemy's existing
Bayer/X-Trans pipeline, white-balanced and mapped camera->ProPhoto exactly like
the rawpy decode path.

Model I/O contract (from the export manifest):
  raw    : NCHW packed sensor-space RAW, normalized after black-level subtraction
  wb_iso : [R/G, 1.0, B/G, log10(ISO)]   (clean-frame WB convention)
  output : NCHW denoised packed sensor-space RAW (absolute mode)

The public `denoise_raw(raw_path, exposure_ratio, tile_size, tile_overlap,
progress_callback) -> (H, W, 3) float32 ProPhoto Linear RGB` wrapper keeps the
signature/return contract the rest of Raw-Alchemy depends on
(core.py, workers/image_processor.py).

Tile-based inference with feathered blending for seamless large-image
processing.

Supports:
  - Windows: CPU, CUDA or DirectML (if available)
  - macOS: CPU, CoreML (if available)
  - Linux: CPU, CUDA or ROCm (if available)
"""

import os
import sys
import platform
import numpy as np
from typing import Optional, Callable
from loguru import logger


# ---------------------------------------------------------------------------
# CUDA setup (must run before importing onnxruntime)
# ---------------------------------------------------------------------------

def _setup_cuda_paths():
    """Setup CUDA library paths for onnxruntime-gpu."""
    try:
        from . import gpu_runtime
        if gpu_runtime.setup_cuda_dll_paths():
            logger.debug("Using locally installed CUDA runtime")
            return
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to setup local CUDA runtime: {e}")

    if platform.system() != 'Windows':
        return

    try:
        import site
        site_packages = site.getsitepackages()
        if hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        ):
            venv_site = os.path.join(sys.prefix, 'Lib', 'site-packages')
            if venv_site not in site_packages:
                site_packages.insert(0, venv_site)

        nvidia_paths = []
        for sp in site_packages:
            nvidia_base = os.path.join(sp, 'nvidia')
            if os.path.isdir(nvidia_base):
                for subdir in os.listdir(nvidia_base):
                    bin_path = os.path.join(nvidia_base, subdir, 'bin')
                    if os.path.isdir(bin_path):
                        nvidia_paths.append(bin_path)

        if hasattr(os, 'add_dll_directory'):
            for path in nvidia_paths:
                try:
                    os.add_dll_directory(path)
                except Exception:
                    pass

        if nvidia_paths:
            os.environ['PATH'] = (
                os.pathsep.join(nvidia_paths) + os.pathsep + os.environ.get('PATH', '')
            )
    except Exception as e:
        logger.debug(f"Failed to setup CUDA paths: {e}")


_setup_cuda_paths()


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_session_bayer = None
_session_xtrans = None
_session_provider = None

BAYER_MODEL = "cans_raw_v11_v14_ep10_bayer_fp16.onnx"
XTRANS_MODEL = "cans_raw_v11_v14_ep10_xtrans_fp16.onnx"
MANIFEST = "cans_raw_v11_v14_ep10_manifest.json"

# v14 ONNX graphs have *fixed* spatial input dims (H==W==tile) and a fixed
# batch of 1 on the output. Tiles must be exactly these sizes; edge tiles are
# reflect-padded to fill. Overlap is feathered on blending.
MODEL_TILE_BAYER = 768
MODEL_TILE_XTRANS = 512
DEFAULT_TILE_OVERLAP_BAYER = 64
DEFAULT_TILE_OVERLAP_XTRANS = 48


# Canonical X-Trans 6x6 CFA (rawpy colour indices 0=R, 1=G, 2=B) that the
# training packer (ProPhotoDenoiser/preprocess_raw.py, SID convention) assumes
# at the visible-image origin. Any Fuji body / crop whose visible origin sits at
# a different phase is rolled onto this layout before packing (see bug #1 fix).
_XTRANS_CANONICAL = np.array(
    [
        [0, 2, 1, 2, 0, 1],
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
    ],
    dtype=np.int32,
)

# Canonical RGGB 2x2 (greens unified to colour-class 1).
_BAYER_CANONICAL = np.array([[0, 1], [1, 2]], dtype=np.int32)


# ---------------------------------------------------------------------------
# Robust WB (bug #2): the noisy frame's as-shot AWB is unreliable on very dark
# / short-exposure frames (Fuji fallback values). v14 was trained on clean-frame
# WB. We trust the as-shot AWB only when both R/G and B/G land in the training
# distribution, otherwise fall back to a daylight prior.
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Plausible band for as-shot WB ratios (matches v14 training distribution).
_WB_RATIO_MIN = _env_float("RAWALCHEMY_DENOISE_WB_MIN", 1.0)
_WB_RATIO_MAX = _env_float("RAWALCHEMY_DENOISE_WB_MAX", 5.5)
# Daylight prior fallback [R/G, 1.0, B/G] used when AWB is untrustworthy.
_DAYLIGHT_WB3 = np.array([2.0, 1.0, 1.6], dtype=np.float32)


def _robust_wb3(cam_wb) -> tuple[np.ndarray, bool]:
    """Return a G-normalized, trustworthy [R/G, 1.0, B/G] and a trusted flag.

    Falls back to a daylight prior when the as-shot AWB ratios fall outside the
    plausible range (bug #2). The same WB is used for the model condition and
    for the render so noise/target stay consistent.
    """
    cam_wb = np.asarray(cam_wb, dtype=np.float32)
    g = float(cam_wb[1]) if cam_wb.size > 1 and cam_wb[1] > 0 else 1.0
    rg = float(cam_wb[0]) / g
    bg = float(cam_wb[2]) / g if cam_wb.size > 2 else 1.0
    trusted = (
        np.isfinite(rg)
        and np.isfinite(bg)
        and _WB_RATIO_MIN <= rg <= _WB_RATIO_MAX
        and _WB_RATIO_MIN <= bg <= _WB_RATIO_MAX
    )
    if trusted:
        return np.array([rg, 1.0, bg], dtype=np.float32), True
    logger.warning(
        f"CANS v14: as-shot WB out of plausible range (R/G={rg:.3f} B/G={bg:.3f}, "
        f"band=[{_WB_RATIO_MIN:g},{_WB_RATIO_MAX:g}]); using daylight prior "
        f"[R/G={_DAYLIGHT_WB3[0]:g} B/G={_DAYLIGHT_WB3[2]:g}]"
    )
    return _DAYLIGHT_WB3.copy(), False


def _build_wb_iso(wb3: np.ndarray, iso: float) -> np.ndarray:
    """Build the FiLM condition [R/G, 1.0, B/G, log10(ISO)] from a validated wb3."""
    iso = float(iso) if iso > 0 else 100.0
    return np.array(
        [float(wb3[0]), 1.0, float(wb3[2]), np.log10(max(iso, 1.0))],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# CFA phase alignment (bug #1)
# ---------------------------------------------------------------------------

def _detect_sensor(raw) -> str:
    """Detect sensor type from rawpy object using raw_pattern."""
    pattern = raw.raw_pattern
    if pattern is not None and pattern.shape == (6, 6):
        return 'xtrans'
    elif pattern is not None and pattern.shape == (2, 2):
        return 'bayer'
    else:
        raise ValueError(
            f"Unknown CFA pattern shape: {None if pattern is None else pattern.shape}"
        )


def _visible_colors(raw) -> Optional[np.ndarray]:
    """Colour index of every visible pixel (0=R,1=G,2=B[,3=G2]), or None.

    Prefers ``raw_colors_visible`` (aligned to ``raw_image_visible``) which is
    the source of truth for the phase at the visible origin. Falls back to
    ``raw_pattern`` (raw-image origin) when unavailable.
    """
    rc = getattr(raw, 'raw_colors_visible', None)
    if rc is not None:
        return np.asarray(rc)
    pat = getattr(raw, 'raw_pattern', None)
    return None if pat is None else np.asarray(pat)


def _cfa_phase_offset(raw, sensor: str) -> tuple[int, int]:
    """Find (dr, dc) in [0, period) rolling the visible origin onto the training
    convention (canonical X-Trans / RGGB Bayer).

    Raises ValueError when no offset aligns — better a hard error than a silently
    mis-packed, colour-scrambled output (bug #1).
    """
    if sensor == 'xtrans':
        period, target = 6, _XTRANS_CANONICAL
    else:
        period, target = 2, _BAYER_CANONICAL

    colors = _visible_colors(raw)
    if colors is None:
        raise ValueError("CANS v14: RAW exposes no CFA colour map; cannot align phase")

    corner = np.asarray(colors[: 2 * period, : 2 * period]).astype(np.int32).copy()
    if corner.shape[0] < period or corner.shape[1] < period:
        raise ValueError(f"CANS v14: CFA colour map too small to align ({corner.shape})")
    if sensor == 'bayer':
        corner[corner == 3] = 1  # unify the two green sites into one colour class

    for dr in range(period):
        for dc in range(period):
            win = corner[dr:dr + period, dc:dc + period]
            if win.shape == (period, period) and np.array_equal(win, target):
                return dr, dc

    raise ValueError(
        f"CANS v14: unsupported {sensor} CFA phase; visible corner "
        f"{corner[:period, :period].tolist()} does not match the training "
        f"convention. Refusing to emit a colour-scrambled image."
    )


def _normalize_raw_image(raw) -> np.ndarray:
    """Black-level subtract and normalize to match training (single black level).

    ProPhotoDenoiser/preprocess_raw.py uses ``black_level_per_channel[0]`` and
    ``(white_level - bl)`` for the whole frame; we mirror that so the model sees
    the exact input distribution it was trained on. Values are not clipped to 1
    (highlights above white level, if any, pass through as in training).
    """
    im = raw.raw_image_visible.astype(np.float32)
    bl = float(raw.black_level_per_channel[0])
    wl = float(raw.white_level)
    return np.maximum(im - bl, 0.0) / max(wl - bl, 1.0)


def _pack_bayer(raw) -> np.ndarray:
    """Pack Bayer RAW to (H/2, W/2, 4) float32 after phase-aligning to RGGB.

    Channel order [R, G1, B, G2] at positions (0,0), (0,1), (1,1), (1,0),
    matching ProPhotoDenoiser/preprocess_raw.py::pack_bayer.
    """
    dr, dc = _cfa_phase_offset(raw, 'bayer')
    im = _normalize_raw_image(raw)[dr:, dc:]

    H, W = im.shape
    H = H // 2 * 2
    W = W // 2 * 2
    im = im[:H, :W]

    return np.stack([
        im[0::2, 0::2],  # R
        im[0::2, 1::2],  # G1
        im[1::2, 1::2],  # B
        im[1::2, 0::2],  # G2
    ], axis=-1)


def _pack_xtrans(raw) -> np.ndarray:
    """Pack X-Trans RAW to (H/3, W/3, 9) float32, SID convention.

    Bug #1: the visible origin is first rolled onto the canonical X-Trans phase
    so the fixed positional slicing lands on the same colours the model was
    trained on for *any* Fuji body (X-S10 / X-T5 / X-T2 ...). Mirrors
    ProPhotoDenoiser/preprocess_raw.py::pack_xtrans after alignment.
    """
    dr, dc = _cfa_phase_offset(raw, 'xtrans')
    im = _normalize_raw_image(raw)[dr:, dc:]

    H = (im.shape[0] // 6) * 6
    W = (im.shape[1] // 6) * 6
    im = im[:H, :W]

    out = np.zeros((H // 3, W // 3, 9), dtype=np.float32)

    # Channel 0: R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # Channel 1: G (subset 1)
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # Channel 2: B (subset 1)
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # Channel 3: R (subset 2)
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # Channel 4: B (subset 2)
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    # Channels 5-8: remaining greens in the 3x3 sub-blocks
    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]

    return out


# ---------------------------------------------------------------------------
# Output guards (bug #3): DEFAULT OFF.
# v14 output is validated (best_val_psnr ~40.2); low-freq matching / channel
# stat matching / input blending distort a good result. Kept behind env vars
# purely for diagnostics.
# ---------------------------------------------------------------------------

def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _output_guard_config() -> dict:
    return {
        "match_low_freq": _env_bool("RAWALCHEMY_DENOISE_MATCH_LOWFREQ"),
        "low_freq_kernel": int(os.environ.get("RAWALCHEMY_DENOISE_LOWFREQ_KERNEL", "129")),
        "match_channel_stats": _env_bool("RAWALCHEMY_DENOISE_MATCH_CHANNEL_STATS"),
        "blend": _env_float("RAWALCHEMY_DENOISE_BLEND", 0.0),
    }


def _guards_enabled(cfg: dict) -> bool:
    return bool(cfg["match_low_freq"] or cfg["match_channel_stats"] or cfg["blend"] > 0)


def _box_blur_hwc(img: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur (reflect padding) — diagnostic low-freq estimate."""
    k = max(1, int(k) | 1)
    if k <= 1:
        return img.astype(np.float32, copy=True)
    r = k // 2
    out = np.pad(img.astype(np.float32), ((r, r), (r, r), (0, 0)), mode="reflect")
    csum = np.cumsum(out, axis=0)
    out = (csum[k - 1:, :, :] - np.vstack([np.zeros((1,) + csum.shape[1:], np.float32), csum[:-k, :, :]])) / k
    csum = np.cumsum(out, axis=1)
    out = (csum[:, k - 1:, :] - np.hstack([np.zeros((csum.shape[0], 1, csum.shape[2]), np.float32), csum[:, :-k, :]])) / k
    return out[: img.shape[0], : img.shape[1], :]


def _apply_output_guards(out_hwc: np.ndarray, in_hwc: np.ndarray, cfg: dict) -> np.ndarray:
    """Optional, default-off post-guards on the packed output (bug #3)."""
    if not _guards_enabled(cfg):
        return out_hwc
    logger.warning(f"CANS v14: output guards ENABLED (diagnostic) -> {cfg}")
    guarded = out_hwc.astype(np.float32, copy=True)
    if cfg["match_channel_stats"]:
        for c in range(guarded.shape[-1]):
            om = float(guarded[..., c].mean()); osd = float(guarded[..., c].std()) + 1e-6
            im = float(in_hwc[..., c].mean()); isd = float(in_hwc[..., c].std()) + 1e-6
            guarded[..., c] = (guarded[..., c] - om) / osd * isd + im
    if cfg["match_low_freq"]:
        k = cfg["low_freq_kernel"]
        guarded = guarded - _box_blur_hwc(guarded, k) + _box_blur_hwc(in_hwc, k)
    if cfg["blend"] > 0:
        a = min(1.0, cfg["blend"])
        guarded = (1.0 - a) * guarded + a * in_hwc.astype(np.float32)
    return guarded


# Highlight-protective guard: DEFAULT ON.
# v14 is out-of-distribution on bright, near-saturated content (illuminated
# night architecture): it pulls the brightest packed channel down more than the
# others, which the strong blue WB gain renders as magenta blotches, and it
# emits isolated colour spikes (green/blue dots). Both live only in bright/
# high-contrast regions that carry little recoverable noise, so a cheap post-
# guard removes them without touching the denoised shadows/mid-tones. Disable
# with CANS_HIGHLIGHT_GUARD=0; tune via CANS_HL_BLEND_LO/HI, CANS_DESPECKLE_TAU.
# ---------------------------------------------------------------------------

def _highlight_guard_config() -> dict:
    on = os.environ.get("CANS_HIGHLIGHT_GUARD", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )
    return {
        "enabled": on,
        "blend_lo": _env_float("CANS_HL_BLEND_LO", 0.45),
        "blend_hi": _env_float("CANS_HL_BLEND_HI", 0.75),
        "chroma_lo": _env_float("CANS_HL_CHROMA_LO", 0.08),
        "chroma_hi": _env_float("CANS_HL_CHROMA_HI", 0.25),
        "despeckle_tau": _env_float("CANS_DESPECKLE_TAU", 0.12),
    }


def _median3_hwc(img: np.ndarray) -> Optional[np.ndarray]:
    """Per-channel 3x3 median via OpenCV (float32). None if cv2 unavailable."""
    try:
        import cv2
    except Exception:
        return None
    src = np.ascontiguousarray(img, dtype=np.float32)
    out = np.empty_like(src)
    for c in range(src.shape[-1]):
        out[..., c] = cv2.medianBlur(src[..., c], 3)
    return out


def _apply_highlight_guard(out_hwc: np.ndarray, in_hwc: np.ndarray, cfg: dict) -> np.ndarray:
    """Default-on guard fixing the model's OOD behaviour in bright regions.

    1. Despeckle: replace pixels that deviate from their 3x3 median by more than
       ``despeckle_tau`` with the median (kills isolated colour spikes).
    2. Chroma-aware highlight guard: only where a pixel is bright AND the model
       shifted its channel balance away from the input (the magenta artifact),
       pull the *chroma* back toward the input while KEEPING the denoised luma.
       Faithfully-reproduced bright colour (neon, lanterns) has near-zero chroma
       divergence, so it is left untouched; shadows/mid-tones stay fully
       denoised. Blending only chroma (not luma) preserves highlight detail.
    """
    if not cfg["enabled"]:
        return out_hwc
    guarded = out_hwc.astype(np.float32, copy=True)
    inp = in_hwc.astype(np.float32, copy=False)
    tau = float(cfg["despeckle_tau"])
    if tau > 0:
        med = _median3_hwc(guarded)
        if med is not None:
            spike = np.abs(guarded - med) > tau
            guarded[spike] = med[spike]
    lo, hi = float(cfg["blend_lo"]), float(cfg["blend_hi"])
    clo, chi = float(cfg["chroma_lo"]), float(cfg["chroma_hi"])
    if hi > lo:
        eps = 1e-4
        brightness = inp.max(axis=-1, keepdims=True)
        a_bright = np.clip((brightness - lo) / (hi - lo), 0.0, 1.0)
        # per-pixel colour signature = channels normalised by their own luma
        luma_out = guarded.mean(axis=-1, keepdims=True) + eps
        chroma_out = guarded / luma_out
        chroma_in = inp / (inp.mean(axis=-1, keepdims=True) + eps)
        cdiff = np.abs(chroma_out - chroma_in).mean(axis=-1, keepdims=True)
        a_chroma = np.clip((cdiff - clo) / max(chi - clo, 1e-6), 0.0, 1.0)
        a = a_bright * a_chroma
        chroma_fixed = (1.0 - a) * chroma_out + a * chroma_in
        guarded = luma_out * chroma_fixed
    return np.clip(guarded, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _get_base_path() -> str:
    """Get resource base path (dev, PyInstaller, Nuitka)."""
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    executable_dir = os.path.dirname(sys.executable)
    internal = os.path.join(executable_dir, '_internal')
    if getattr(sys, 'frozen', False) and os.path.isdir(internal):
        return internal
    return os.path.dirname(os.path.abspath(__file__))


def _find_model(model_filename: str) -> str:
    """Find the ONNX model file."""
    base = _get_base_path()
    candidates = [
        os.path.join(base, "vendor", model_filename),
        os.path.join(base, "raw_alchemy", "vendor", model_filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", model_filename),
    ]
    for p in candidates:
        np_ = os.path.normpath(p)
        if os.path.exists(np_):
            return np_
    raise FileNotFoundError(
        f"ONNX model '{model_filename}' not found. Searched: {candidates}"
    )


def _get_providers() -> list:
    """Get execution providers (CUDA/ROCm/DirectML/CoreML > CPU)."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

    available = ort.get_available_providers()
    providers = []

    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
        logger.info("Using CUDA execution provider")
    elif 'ROCMExecutionProvider' in available:
        providers.append('ROCMExecutionProvider')
        logger.info("Using ROCm execution provider")
    elif 'DmlExecutionProvider' in available:
        providers.append('DmlExecutionProvider')
        logger.info("Using DirectML execution provider")
    elif 'CoreMLExecutionProvider' in available and platform.system() == 'Darwin':
        providers.append('CoreMLExecutionProvider')
        logger.info("Using CoreML execution provider (macOS)")

    providers.append('CPUExecutionProvider')
    if len(providers) == 1:
        logger.info("Using CPU execution provider (no GPU acceleration)")
    return providers


def _get_session(sensor: str):
    """Get or create ONNX session for the given sensor type."""
    global _session_bayer, _session_xtrans, _session_provider

    if sensor == 'bayer' and _session_bayer is not None:
        return _session_bayer
    if sensor == 'xtrans' and _session_xtrans is not None:
        return _session_xtrans

    import onnxruntime as ort

    model_file = BAYER_MODEL if sensor == 'bayer' else XTRANS_MODEL
    model_path = _find_model(model_file)
    providers = _get_providers()

    logger.info(f"Loading CANS raw-main v14 ({sensor}) from: {model_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True

    provider_options = []
    for p in providers:
        if p == 'CUDAExecutionProvider':
            provider_options.append((p, {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
                'cudnn_conv_use_max_workspace': True,
            }))
        else:
            provider_options.append(p)

    session = ort.InferenceSession(model_path, sess_options, providers=provider_options)
    _session_provider = session.get_providers()[0]
    logger.info(f"CANS session ({sensor}): {_session_provider}")

    if sensor == 'bayer':
        _session_bayer = session
    else:
        _session_xtrans = session
    return session


# ---------------------------------------------------------------------------
# Orientation (match rawpy.postprocess behavior)
# ---------------------------------------------------------------------------

def _apply_flip(img: np.ndarray, flip_code: int) -> np.ndarray:
    """Apply rawpy orientation to HWC image.

    rawpy.sizes.flip values (matches libraw):
        0 = no rotation
        3 = 180deg
        5 = 90deg CCW (270deg CW)
        6 = 90deg CW
    """
    if flip_code == 0:
        return img
    elif flip_code == 3:
        return np.rot90(img, k=2)
    elif flip_code == 5:
        return np.rot90(img, k=1)  # 90 CCW
    elif flip_code == 6:
        return np.rot90(img, k=3)  # 90 CW (= 270 CCW)
    else:
        logger.warning(f"Unknown flip code {flip_code}, skipping rotation")
        return img


# ---------------------------------------------------------------------------
# Tile-based inference
# ---------------------------------------------------------------------------

def _has_wb_input(session) -> bool:
    """Check if the ONNX model accepts a wb_iso input."""
    return len(session.get_inputs()) >= 2


def _ort_input_dtype(input_info) -> np.dtype:
    if "float16" in input_info.type:
        return np.float16
    return np.float32


def _process_batch(session, tiles_chw: list[np.ndarray],
                   wb_iso: Optional[np.ndarray] = None) -> list[np.ndarray]:
    """Run a batch of packed tiles through the ONNX model.

    v14 graphs pin the output batch to 1, so callers must submit one tile at a
    time; this helper still accepts a list for API symmetry.
    """
    if not tiles_chw:
        return []

    raw_dtype = _ort_input_dtype(session.get_inputs()[0])

    if len(tiles_chw) == 1:
        batch = tiles_chw[0][np.newaxis, ...].astype(raw_dtype, copy=False)
    else:
        batch = np.stack(tiles_chw, axis=0).astype(raw_dtype, copy=False)

    feeds = {session.get_inputs()[0].name: batch}

    if wb_iso is not None and _has_wb_input(session):
        B = batch.shape[0]
        wb_dtype = _ort_input_dtype(session.get_inputs()[1])
        wb_batch = np.tile(wb_iso.astype(wb_dtype)[np.newaxis, :], (B, 1))
        feeds[session.get_inputs()[1].name] = wb_batch

    output = session.run(None, feeds)[0]
    return [output[i].astype(np.float32) for i in range(output.shape[0])]


def _tile_weight(
    h: int,
    w: int,
    overlap: int,
    *,
    at_top: bool,
    at_bottom: bool,
    at_left: bool,
    at_right: bool,
) -> np.ndarray:
    """Linear feathering window that keeps outer image borders at weight 1."""
    feather = max(0, min(overlap, h // 2, w // 2))
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    if feather > 0:
        ramp = np.linspace(1.0 / (feather + 1), 1.0, feather, dtype=np.float32)
        if not at_top:
            wy[:feather] = ramp
        if not at_bottom:
            wy[-feather:] = ramp[::-1]
        if not at_left:
            wx[:feather] = ramp
        if not at_right:
            wx[-feather:] = ramp[::-1]
    return np.outer(wy, wx)[np.newaxis, :, :]


def _unpack_bayer_rggb(packed: np.ndarray) -> np.ndarray:
    h, w, c = packed.shape
    if c != 4:
        raise ValueError(f"Bayer packed RAW must have 4 channels, got {c}")
    raw = np.zeros((h * 2, w * 2), dtype=np.float32)
    raw[0::2, 0::2] = packed[..., 0]
    raw[0::2, 1::2] = packed[..., 1]
    raw[1::2, 1::2] = packed[..., 2]
    raw[1::2, 0::2] = packed[..., 3]
    return raw


def _assign_xtrans(
    raw: np.ndarray,
    packed: np.ndarray,
    channel: int,
    *,
    top: int,
    left: int,
    offsets: dict[tuple[int, int], tuple[int, int]],
) -> None:
    h, w, _ = packed.shape
    yy, xx = np.indices((h, w), dtype=np.int32)
    parity_y = (yy + top) & 1
    parity_x = (xx + left) & 1
    for py in (0, 1):
        for px in (0, 1):
            mask = (parity_y == py) & (parity_x == px)
            if not np.any(mask):
                continue
            dy, dx = offsets[(py, px)]
            raw[yy[mask] * 3 + dy, xx[mask] * 3 + dx] = packed[..., channel][mask]


def _unpack_xtrans_sid(packed: np.ndarray, *, top: int = 0, left: int = 0) -> np.ndarray:
    h, w, c = packed.shape
    if c != 9:
        raise ValueError(f"X-Trans packed RAW must have 9 channels, got {c}")
    raw = np.zeros((h * 3, w * 3), dtype=np.float32)

    _assign_xtrans(
        raw,
        packed,
        0,
        top=top,
        left=left,
        offsets={(0, 0): (0, 0), (0, 1): (0, 1), (1, 0): (0, 1), (1, 1): (0, 0)},
    )
    raw[0::3, 2::3] = packed[..., 1]
    _assign_xtrans(
        raw,
        packed,
        2,
        top=top,
        left=left,
        offsets={(0, 0): (0, 1), (0, 1): (0, 0), (1, 0): (0, 0), (1, 1): (0, 1)},
    )
    _assign_xtrans(
        raw,
        packed,
        3,
        top=top,
        left=left,
        offsets={(0, 0): (1, 2), (0, 1): (2, 2), (1, 0): (2, 2), (1, 1): (1, 2)},
    )
    _assign_xtrans(
        raw,
        packed,
        4,
        top=top,
        left=left,
        offsets={(0, 0): (2, 2), (0, 1): (1, 2), (1, 0): (1, 2), (1, 1): (2, 2)},
    )

    raw[1::3, 0::3] = packed[..., 5]
    raw[1::3, 1::3] = packed[..., 6]
    raw[2::3, 0::3] = packed[..., 7]
    raw[2::3, 1::3] = packed[..., 8]
    return raw


def _render_packed_raw_to_prophoto(result: dict) -> np.ndarray:
    from raw_alchemy.colorspace_matrices import cam_to_prophoto_matrix
    from raw_alchemy.demosaic import FILTERS_RGGB, rcd_demosaic
    from raw_alchemy.math_ops import apply_matrix_inplace

    packed = np.clip(result["packed"], 0.0, 1.0).astype(np.float32, copy=False)
    sensor = result["sensor"]

    if sensor == "bayer":
        mosaic = _unpack_bayer_rggb(packed)
        rgb = rcd_demosaic(mosaic, FILTERS_RGGB)
    elif sensor == "xtrans":
        from raw_alchemy.xtrans_demosaic import xtrans_markesteijn_demosaic

        # The packed RAW was phase-aligned to the canonical X-Trans layout, so
        # the reconstructed mosaic is in canonical phase -> demosaic with the
        # canonical pattern (NOT raw.raw_pattern, which may sit at a different
        # phase for non-X-T2 bodies). This closes the loop on the bug #1 fix.
        mosaic = _unpack_xtrans_sid(packed, top=0, left=0)
        pattern = np.ascontiguousarray(result["cfa_pattern"].astype(np.int32))
        rgb = xtrans_markesteijn_demosaic(mosaic, pattern)
    else:
        raise ValueError(f"unknown sensor {sensor!r}")

    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    # Apply the validated white balance (bug #2): trusted as-shot AWB on normal
    # frames, daylight prior on untrustworthy ones.
    wb3 = result["wb3"]
    rgb[:, :, 0] *= float(wb3[0])
    rgb[:, :, 2] *= float(wb3[2])

    apply_matrix_inplace(rgb, cam_to_prophoto_matrix(result["xyz_to_cam"]))
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return _apply_flip(rgb, int(result["flip_code"]))


def _read_iso_tiff_fallback(raw_path: str) -> float:
    """Minimal TIFF/EXIF walker: read tag 0x8827 (ISOSpeedRatings) directly.

    pyexiv2 hard-fails on some files instead of degrading (e.g. Sony-converted
    DNGs raise "Directory Sony2 ... considered invalid" over a makernote it
    cannot parse), which used to silently default the FiLM ISO condition to
    100 and wreck the denoise output on high-ISO shots. The standard ISO tag
    in IFD0/ExifIFD/SubIFDs is intact in those files, so walk it ourselves.
    Returns 0.0 when the file is not TIFF-based or the tag is absent.
    """
    import struct

    try:
        with open(raw_path, "rb") as f:
            data = f.read(4 * 1024 * 1024)
        if len(data) < 8 or data[:2] not in (b"II", b"MM"):
            return 0.0
        bo = "<" if data[:2] == b"II" else ">"

        def u16(off: int) -> int:
            return struct.unpack_from(bo + "H", data, off)[0]

        def u32(off: int) -> int:
            return struct.unpack_from(bo + "I", data, off)[0]

        visited: set[int] = set()

        def walk_ifd(off: int, depth: int) -> float:
            if off <= 0 or off + 2 > len(data) or off in visited or depth > 4:
                return 0.0
            visited.add(off)
            n = u16(off)
            if n > 4096 or off + 2 + n * 12 > len(data):
                return 0.0
            sub_offsets = []
            for i in range(n):
                e = off + 2 + i * 12
                tag, typ, cnt = u16(e), u16(e + 2), u32(e + 4)
                if tag == 0x8827 and typ in (3, 4) and cnt >= 1:
                    # count>1 (multi-ISO) stores values behind an offset
                    voff = u32(e + 8) if (cnt * (2 if typ == 3 else 4)) > 4 else e + 8
                    if voff + 4 <= len(data):
                        return float(u16(voff) if typ == 3 else u32(voff))
                if tag == 0x8769:  # ExifIFD pointer
                    sub_offsets.append(u32(e + 8))
                if tag == 0x014A and typ == 4:  # SubIFDs
                    base = u32(e + 8)
                    if cnt == 1:
                        sub_offsets.append(base)
                    elif base + cnt * 4 <= len(data):
                        sub_offsets.extend(u32(base + j * 4) for j in range(min(cnt, 8)))
            for sub in sub_offsets:
                iso = walk_ifd(sub, depth + 1)
                if iso > 0:
                    return iso
            # chained next-IFD pointer (IFD1...)
            next_off = off + 2 + n * 12
            if next_off + 4 <= len(data):
                return walk_ifd(u32(next_off), depth + 1)
            return 0.0

        return walk_ifd(u32(4), 0)
    except Exception:
        return 0.0


def _read_iso(raw_path: str) -> float:
    """Best-effort ISO read from EXIF (rawpy on this build has no iso_speed)."""
    try:
        import pyexiv2

        with pyexiv2.Image(raw_path) as exif_img:
            exif_data = exif_img.read_exif() or {}
            iso_str = exif_data.get('Exif.Photo.ISOSpeedRatings') or exif_data.get('Exif.Photo.ISOSpeed', '')
            if iso_str:
                return float(str(iso_str).split()[0])
    except Exception:
        pass
    iso = _read_iso_tiff_fallback(raw_path)
    if iso > 0:
        logger.info(f"CANS: ISO {iso:.0f} recovered via TIFF fallback (pyexiv2 failed) for {os.path.basename(raw_path)}")
        return iso
    logger.warning(f"CANS: ISO unreadable for {os.path.basename(raw_path)}, defaulting to 100 — denoise strength may be wrong")
    return 100.0


def _model_tile(sensor: str) -> int:
    return MODEL_TILE_XTRANS if sensor == "xtrans" else MODEL_TILE_BAYER


def _default_overlap(sensor: str) -> int:
    if sensor == "xtrans":
        return DEFAULT_TILE_OVERLAP_XTRANS
    return DEFAULT_TILE_OVERLAP_BAYER


def denoise_raw_packed(
    raw_path: str,
    exposure_ratio: float = 1.0,
    tile_size: Optional[int] = None,
    tile_overlap: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Denoise a RAW file to packed sensor-space RAW using CANS raw-main v14.

    Args:
        raw_path: Path to the RAW file (.ARW, .CR3, .NEF, .RAF, .DNG, etc.)
        exposure_ratio: Source pre-gain multiplier applied to the packed RAW
            before denoising (source-mode convention). Also scales the condition
            ISO. Default 1.0 (no change).
        tile_size: Advisory only — v14 ONNX pins the tile to a fixed size, so
            the model's required size is always used.
        tile_overlap: Feather overlap in packed pixels. Defaults by sensor.
        progress_callback: Optional (current, total) callback.

    Returns:
        Dict with denoised HWC packed RAW plus metadata needed for rendering.
    """
    import time
    import rawpy

    t0 = time.time()

    with rawpy.imread(raw_path) as raw:
        sensor = _detect_sensor(raw)
        if sensor == 'bayer':
            packed = _pack_bayer(raw)
            cfa_pattern = _BAYER_CANONICAL.copy()
        else:
            packed = _pack_xtrans(raw)
            cfa_pattern = _XTRANS_CANONICAL.copy()

        cam_wb = np.array(raw.camera_whitebalance, dtype=np.float32)
        flip_code = raw.sizes.flip
        xyz_to_cam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)

    # Bug #2: validate the as-shot AWB before trusting it.
    wb3, wb_trusted = _robust_wb3(cam_wb)

    iso = _read_iso(raw_path)
    # source-mode condition ISO scales with the applied pre-gain (== exposure_ratio).
    iso_cond = iso * max(float(exposure_ratio), 1e-6)
    wb_iso = _build_wb_iso(wb3, iso_cond)

    packed_input = packed  # keep the model input for optional guards
    if exposure_ratio != 1.0:
        packed = np.clip(packed * float(exposure_ratio), 0.0, 1.0)
        packed_input = packed

    # v14 has fixed spatial tile dims; honour that regardless of caller override.
    model_tile = _model_tile(sensor)
    if tile_size is not None and int(tile_size) != model_tile:
        logger.debug(
            f"CANS v14 ({sensor}): ignoring tile_size={tile_size}, model requires {model_tile}"
        )
    tile_size = model_tile
    if tile_overlap is None:
        tile_overlap = _default_overlap(sensor)
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap must be in [0, tile_size), got {tile_overlap}/{tile_size}")

    pack_h, pack_w, in_ch = packed.shape

    logger.info(
        f"CANS raw-main v14: {os.path.basename(raw_path)} ({sensor}), "
        f"packed {pack_w}x{pack_h}, tile={tile_size}/{tile_overlap}, "
        f"WB=[{wb_iso[0]:.2f}, {wb_iso[2]:.2f}]{'' if wb_trusted else ' (daylight-fallback)'} "
        f"ISO={iso:.0f} flip={flip_code}"
    )

    session = _get_session(sensor)
    packed_chw = np.ascontiguousarray(packed.transpose(2, 0, 1), dtype=np.float16)
    guard_cfg = _output_guard_config()
    highlight_cfg = _highlight_guard_config()

    step = tile_size - tile_overlap
    ys = list(range(0, max(pack_h - tile_size, 0) + 1, step))
    xs = list(range(0, max(pack_w - tile_size, 0) + 1, step))
    if ys[-1] + tile_size < pack_h:
        ys.append(pack_h - tile_size)
    if xs[-1] + tile_size < pack_w:
        xs.append(pack_w - tile_size)

    # v14 output batch is fixed to 1.
    batch_size = 1

    if pack_h <= tile_size and pack_w <= tile_size:
        if progress_callback:
            progress_callback(0, 1)
        pad_h = max(tile_size - pack_h, 0)
        pad_w = max(tile_size - pack_w, 0)
        if pad_h > 0 or pad_w > 0:
            tile = np.pad(packed_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        else:
            tile = packed_chw
        result = _process_batch(session, [tile.astype(np.float16)], wb_iso)[0]
        result = result[:, :pack_h, :pack_w]
        if progress_callback:
            progress_callback(1, 1)
        elapsed = time.time() - t0
        logger.info(f"CANS raw-main v14 done in {elapsed:.1f}s (1 tile, {_session_provider})")
        packed_out = np.clip(result.transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)
        packed_out = _apply_output_guards(packed_out, packed_input, guard_cfg)
        packed_out = _apply_highlight_guard(packed_out, packed_input, highlight_cfg)
        return {
            "packed": np.clip(packed_out, 0.0, 1.0).astype(np.float32),
            "sensor": sensor,
            "wb3": wb3,
            "wb_trusted": wb_trusted,
            "cfa_pattern": cfa_pattern,
            "xyz_to_cam": xyz_to_cam,
            "flip_code": flip_code,
        }

    tile_infos = []
    for y in ys:
        y2 = min(y + tile_size, pack_h)
        y = max(y2 - tile_size, 0)
        for x in xs:
            x2 = min(x + tile_size, pack_w)
            x = max(x2 - tile_size, 0)
            tile_infos.append((y, x, y2, x2))

    total_tiles = len(tile_infos)
    accum = np.zeros((in_ch, pack_h, pack_w), dtype=np.float32)
    weight = np.zeros((1, pack_h, pack_w), dtype=np.float32)

    current = 0
    for batch_start in range(0, total_tiles, batch_size):
        batch_infos = tile_infos[batch_start:batch_start + batch_size]

        tiles = []
        for y, x, y2, x2 in batch_infos:
            tile = packed_chw[:, y:y2, x:x2].copy()
            th, tw = tile.shape[1], tile.shape[2]
            if th < tile_size or tw < tile_size:
                tile = np.pad(
                    tile,
                    ((0, 0), (0, tile_size - th), (0, tile_size - tw)),
                    mode='reflect',
                )
            tiles.append(tile.astype(np.float16))

        preds = _process_batch(session, tiles, wb_iso)

        for idx, (y, x, y2, x2) in enumerate(batch_infos):
            pred = preds[idx]
            oph = min(pred.shape[1], y2 - y)
            opw = min(pred.shape[2], x2 - x)
            pred = pred[:, :oph, :opw]
            wt = _tile_weight(
                oph,
                opw,
                tile_overlap,
                at_top=(y == 0),
                at_bottom=(y2 >= pack_h),
                at_left=(x == 0),
                at_right=(x2 >= pack_w),
            )

            accum[:, y:y + oph, x:x + opw] += pred * wt
            weight[:, y:y + oph, x:x + opw] += wt

            current += 1
            if progress_callback:
                progress_callback(current, total_tiles)

    weight = np.maximum(weight, 1e-8)
    result = accum / weight

    elapsed = time.time() - t0
    n_batches = (total_tiles + batch_size - 1) // batch_size
    logger.info(
        f"CANS raw-main v14 done in {elapsed:.1f}s "
        f"({total_tiles} tiles, {n_batches} batches of {batch_size}, {_session_provider})"
    )

    packed_out = np.clip(result.transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)
    packed_out = _apply_output_guards(packed_out, packed_input, guard_cfg)
    packed_out = _apply_highlight_guard(packed_out, packed_input, highlight_cfg)
    return {
        "packed": np.clip(packed_out, 0.0, 1.0).astype(np.float32),
        "sensor": sensor,
        "wb3": wb3,
        "wb_trusted": wb_trusted,
        "cfa_pattern": cfa_pattern,
        "xyz_to_cam": xyz_to_cam,
        "flip_code": flip_code,
    }


def denoise_raw(
    raw_path: str,
    exposure_ratio: float = 1.0,
    tile_size: Optional[int] = None,
    tile_overlap: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Denoise RAW and return HWC float32 ProPhoto Linear RGB in [0, 1].

    Public API / contract (unchanged): denoise packed sensor-space RAW with
    CANS raw-main v14, then unpack -> demosaic -> WB -> camera->ProPhoto.
    """
    result = denoise_raw_packed(
        raw_path,
        exposure_ratio=exposure_ratio,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        progress_callback=progress_callback,
    )
    return _render_packed_raw_to_prophoto(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Check if the CANS denoiser is available (models + ONNX Runtime)."""
    try:
        import onnxruntime  # noqa: F401
        _find_model(BAYER_MODEL)
        _find_model(XTRANS_MODEL)
        return True
    except (ImportError, FileNotFoundError):
        return False


def get_provider_info() -> dict:
    """Get information about the current execution provider."""
    try:
        session = _get_session('bayer')
        return {
            "available": True,
            "provider": session.get_providers()[0],
            "all_providers": session.get_providers(),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def clear_session():
    """Clear cached sessions and release GPU memory."""
    global _session_bayer, _session_xtrans, _session_provider
    for name, sess in [('bayer', _session_bayer), ('xtrans', _session_xtrans)]:
        if sess is not None:
            del sess
            logger.info(f"CANS session ({name}) cleared")
    _session_bayer = None
    _session_xtrans = None
    _session_provider = None
    import gc
    gc.collect()
