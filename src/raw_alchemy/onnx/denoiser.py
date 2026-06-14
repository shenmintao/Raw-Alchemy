"""
CANS RAW denoiser via ONNX Runtime.

The current production path is CANSRawV8 raw-main v5:
  - cans_raw_v8_rawmain_v5_bayer_fp16.onnx   (4ch Bayer packed RAW in/out)
  - cans_raw_v8_rawmain_v5_xtrans_fp16.onnx  (9ch X-Trans packed RAW in/out)

The denoised packed RAW is demosaiced by Raw-Alchemy's existing Bayer/X-Trans
pipeline, then WB and camera-to-ProPhoto are applied as before.

Tile-based inference with Hann window blending for seamless large-image processing.

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

BAYER_MODEL = "cans_raw_v8_rawmain_v5_bayer_fp16.onnx"
XTRANS_MODEL = "cans_raw_v8_rawmain_v5_xtrans_fp16.onnx"

# Tile size at packed resolution.
DEFAULT_TILE_SIZE_BAYER = 768
DEFAULT_TILE_OVERLAP_BAYER = 64
DEFAULT_TILE_SIZE_XTRANS = 512
DEFAULT_TILE_OVERLAP_XTRANS = 48
ENABLE_RAWMAIN_V5_XTRANS = os.environ.get("RAW_ALCHEMY_ENABLE_CANS_V5_XTRANS", "0") == "1"


# ---------------------------------------------------------------------------
# RAW packing (from preprocess_raw.py)
# ---------------------------------------------------------------------------

def _detect_sensor(raw) -> str:
    """Detect sensor type from rawpy object using raw_pattern."""
    pattern = raw.raw_pattern
    if pattern is not None and pattern.shape == (6, 6):
        return 'xtrans'
    elif pattern is not None and pattern.shape == (2, 2):
        return 'bayer'
    else:
        raise ValueError(f"Unknown CFA pattern shape: {None if pattern is None else pattern.shape}")


def _normalize_raw_image(raw) -> np.ndarray:
    """Black-level subtract and normalize per CFA color channel."""
    im = raw.raw_image_visible.astype(np.float32)
    pattern = raw.raw_pattern
    bl = np.array(raw.black_level_per_channel, dtype=np.float32)
    wl = float(raw.white_level)
    if pattern is None:
        base = float(bl[0])
        return np.maximum(im - base, 0) / max(wl - base, 1.0)

    pat_size = pattern.shape[0]
    out = np.empty_like(im, dtype=np.float32)
    for r in range(pat_size):
        for c in range(pat_size):
            color = int(pattern[r, c])
            bl_c = float(bl[min(color, len(bl) - 1)])
            out[r::pat_size, c::pat_size] = np.maximum(
                im[r::pat_size, c::pat_size] - bl_c,
                0,
            ) / max(wl - bl_c, 1.0)
    return out


def _is_rggb_pattern(pattern: np.ndarray) -> bool:
    if pattern is None or pattern.shape != (2, 2):
        return False
    # rawpy may encode the two green sites as either 1/1 or 1/3.
    return (
        int(pattern[0, 0]) == 0
        and int(pattern[0, 1]) in (1, 3)
        and int(pattern[1, 0]) in (1, 3)
        and int(pattern[1, 1]) == 2
    )


def _pack_bayer(raw) -> np.ndarray:
    """Pack RGGB Bayer raw to (H/2, W/2, 4) float32.

    Channel order: [R, G1, B, G2] at positions (0,0), (0,1), (1,1), (1,0).
    Uses rawpy raw_image_visible directly.
    """
    if not _is_rggb_pattern(raw.raw_pattern):
        raise ValueError(f"CANS raw-main v5 currently supports RGGB Bayer only, got pattern={raw.raw_pattern}")
    im = _normalize_raw_image(raw)

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
    """Pack X-Trans raw to (H/3, W/3, 9) float32, SID convention.

    Uses rawpy raw_image_visible directly.
    """
    im = _normalize_raw_image(raw)

    H = (im.shape[0] // 6) * 6
    W = (im.shape[1] // 6) * 6
    im = im[:H, :W]

    out = np.zeros((H // 3, W // 3, 9), dtype=np.float32)

    # Channels 0-4: R, G, B subsets (SID convention)
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]

    return out


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

    logger.info(f"Loading CANS raw-main v5 ({sensor}) from: {model_path}")

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
# VRAM query
# ---------------------------------------------------------------------------

def _get_free_vram_mb() -> float:
    """Query free GPU VRAM in MB. Returns 0 if unavailable."""
    # Try pynvml (most reliable, ships with onnxruntime-gpu/CUDA toolkit)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.free / (1024 * 1024)
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except Exception:
        pass

    return 0.0


def _estimate_tile_vram_mb(in_ch: int, out_ch: int, tile_size: int, upscale: int = 1) -> float:
    """Estimate VRAM needed per tile in MB (input + output + intermediate).

    Conservative estimate: ~4x the raw input+output size for intermediate
    activations and workspace.
    """
    in_bytes = in_ch * tile_size * tile_size * 2  # fp16
    out_bytes = out_ch * (tile_size * upscale) ** 2 * 2  # fp16 output
    # Model intermediate activations roughly 4x input
    return (in_bytes + out_bytes) * 4 / (1024 * 1024)


def _compute_batch_size(in_ch: int, tile_size: int, upscale: int = 1,
                        out_ch: Optional[int] = None,
                        vram_reserve_mb: float = 512) -> int:
    """Compute optimal batch size based on available VRAM.

    Args:
        in_ch: Number of input channels (4 for Bayer, 9 for X-Trans)
        tile_size: Tile size at packed resolution
        upscale: 2 for Bayer, 3 for X-Trans
        vram_reserve_mb: VRAM to keep free for other GPU tasks (Taichi, display)

    Returns:
        Batch size (>= 1)
    """
    free_mb = _get_free_vram_mb()
    if free_mb <= 0:
        # Can't query VRAM — conservative default
        return 1

    out_ch = in_ch if out_ch is None else out_ch
    per_tile_mb = _estimate_tile_vram_mb(in_ch, out_ch, tile_size, upscale)
    if per_tile_mb <= 0:
        return 1

    available = max(free_mb - vram_reserve_mb, per_tile_mb)
    batch_size = max(1, min(8, int(available // per_tile_mb)))

    logger.info(
        f"VRAM: {free_mb:.0f} MB free, {per_tile_mb:.1f} MB/tile, "
        f"reserve {vram_reserve_mb:.0f} MB -> batch_size={batch_size}"
    )
    return batch_size


# ---------------------------------------------------------------------------
# Orientation (match rawpy.postprocess behavior)
# ---------------------------------------------------------------------------

def _apply_flip(img: np.ndarray, flip_code: int) -> np.ndarray:
    """Apply rawpy orientation to HWC image.

    rawpy.sizes.flip values (matches libraw):
        0 = no rotation
        3 = 180°
        5 = 90° CCW (270° CW)
        6 = 90° CW
    """
    if flip_code == 0:
        return img
    elif flip_code == 3:
        return np.rot90(img, k=2)
    elif flip_code == 5:
        return np.rot90(img, k=1)  # 90° CCW
    elif flip_code == 6:
        return np.rot90(img, k=3)  # 90° CW (= 270° CCW)
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


def _build_wb_iso(cam_wb: np.ndarray, iso: float) -> np.ndarray:
    """Build WB+ISO condition vector.

    Args:
        cam_wb: (4,) camera white balance [R, G, B, G2]
        iso: ISO speed value

    Returns (4,) float32: [R/G, 1.0, B/G, log10(ISO)]
    """
    g = cam_wb[1] if cam_wb[1] > 0 else 1.0
    wb3 = np.array([cam_wb[0] / g, 1.0, cam_wb[2] / g], dtype=np.float32)

    iso = float(iso) if iso > 0 else 100.0

    return np.concatenate([wb3, [np.log10(max(iso, 1.0))]], dtype=np.float32)


def _process_batch(session, tiles_chw: list[np.ndarray],
                   wb_iso: Optional[np.ndarray] = None) -> list[np.ndarray]:
    """Run a batch of packed tiles through the ONNX model.

    Args:
        tiles_chw: List of (C, H, W) float16 packed RAW tiles (same spatial size)
        wb_iso: Optional (4,) float32 WB+ISO condition [R/G, 1.0, B/G, log10(ISO)]
    Returns:
        List of CHW float32 model outputs.
    """
    if not tiles_chw:
        return []

    raw_dtype = _ort_input_dtype(session.get_inputs()[0])

    if len(tiles_chw) == 1:
        batch = tiles_chw[0][np.newaxis, ...].astype(raw_dtype, copy=False)
    else:
        batch = np.stack(tiles_chw, axis=0).astype(raw_dtype, copy=False)

    feeds = {session.get_inputs()[0].name: batch}

    # Add WB+ISO condition if model supports it
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

        mosaic = _unpack_xtrans_sid(packed, top=0, left=0)
        pattern = np.ascontiguousarray(result["cfa_pattern"].astype(np.int32))
        rgb = xtrans_markesteijn_demosaic(mosaic, pattern)
    else:
        raise ValueError(f"unknown sensor {sensor!r}")

    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    cam_wb = result["camera_wb"]
    g = float(cam_wb[1]) if float(cam_wb[1]) > 0 else 1.0
    rgb[:, :, 0] *= float(cam_wb[0]) / g
    rgb[:, :, 2] *= float(cam_wb[2]) / g

    apply_matrix_inplace(rgb, cam_to_prophoto_matrix(result["xyz_to_cam"]))
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return _apply_flip(rgb, int(result["flip_code"]))


def _read_iso(raw_path: str) -> float:
    """Best-effort ISO read from EXIF."""
    try:
        import pyexiv2

        with pyexiv2.Image(raw_path) as exif_img:
            exif_data = exif_img.read_exif() or {}
            iso_str = exif_data.get('Exif.Photo.ISOSpeedRatings') or exif_data.get('Exif.Photo.ISOSpeed', '')
            if iso_str:
                return float(iso_str)
    except Exception:
        pass
    return 100.0


def _default_tile_params(sensor: str) -> tuple[int, int]:
    if sensor == "xtrans":
        return DEFAULT_TILE_SIZE_XTRANS, DEFAULT_TILE_OVERLAP_XTRANS
    return DEFAULT_TILE_SIZE_BAYER, DEFAULT_TILE_OVERLAP_BAYER


def denoise_raw_packed(
    raw_path: str,
    exposure_ratio: float = 1.0,
    tile_size: Optional[int] = None,
    tile_overlap: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Denoise a RAW file to packed sensor-space RAW using CANS raw-main v5.

    Args:
        raw_path: Path to the RAW file (.ARW, .CR3, .NEF, .RAF, .DNG, etc.)
        exposure_ratio: Multiply packed RAW by this ratio before denoising.
        tile_size: Tile size at packed resolution. Defaults by sensor.
        tile_overlap: Overlap in packed pixels. Defaults by sensor.
        progress_callback: Optional (current, total) callback

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
        else:
            if not ENABLE_RAWMAIN_V5_XTRANS:
                raise RuntimeError(
                    "CANS raw-main v5 X-Trans is disabled by default; "
                    "set RAW_ALCHEMY_ENABLE_CANS_V5_XTRANS=1 for experimental testing"
                )
            packed = _pack_xtrans(raw)

        cam_wb = np.array(raw.camera_whitebalance, dtype=np.float32)
        flip_code = raw.sizes.flip
        cfa_pattern = raw.raw_pattern.copy().astype(np.int32)
        xyz_to_cam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)

    iso = _read_iso(raw_path)
    wb_iso = _build_wb_iso(cam_wb, iso)

    if exposure_ratio != 1.0:
        packed = np.clip(packed * exposure_ratio, 0.0, 1.0)

    pack_h, pack_w, in_ch = packed.shape
    if tile_size is None or tile_overlap is None:
        default_tile, default_overlap = _default_tile_params(sensor)
        tile_size = default_tile if tile_size is None else tile_size
        tile_overlap = default_overlap if tile_overlap is None else tile_overlap

    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap must be in [0, tile_size), got {tile_overlap}/{tile_size}")
    if sensor == "xtrans" and (tile_size % 2 or tile_overlap % 2):
        raise ValueError("X-Trans tile_size and tile_overlap must be even in packed coordinates")

    logger.info(
        f"CANS raw-main v5: {os.path.basename(raw_path)} ({sensor}), "
        f"packed {pack_w}x{pack_h}, tile={tile_size}/{tile_overlap}, "
        f"WB=[{wb_iso[0]:.2f}, {wb_iso[2]:.2f}] ISO={iso:.0f} flip={flip_code}"
    )

    session = _get_session(sensor)
    packed_chw = np.ascontiguousarray(packed.transpose(2, 0, 1), dtype=np.float16)

    step = tile_size - tile_overlap
    ys = list(range(0, max(pack_h - tile_size, 0) + 1, step))
    xs = list(range(0, max(pack_w - tile_size, 0) + 1, step))
    if ys[-1] + tile_size < pack_h:
        ys.append(pack_h - tile_size)
    if xs[-1] + tile_size < pack_w:
        xs.append(pack_w - tile_size)

    batch_size = _compute_batch_size(in_ch, tile_size, upscale=1, out_ch=in_ch)

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
        logger.info(f"CANS raw-main v5 done in {elapsed:.1f}s (1 tile, {_session_provider})")
        return {
            "packed": np.clip(result.transpose(1, 2, 0), 0.0, 1.0).astype(np.float32),
            "sensor": sensor,
            "camera_wb": cam_wb,
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

        # Extract and pad tiles for this batch
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

        # Batch inference
        preds = _process_batch(session, tiles, wb_iso)

        # Accumulate results
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
        f"CANS raw-main v5 done in {elapsed:.1f}s "
        f"({total_tiles} tiles, {n_batches} batches of {batch_size}, {_session_provider})"
    )

    return {
        "packed": np.clip(result.transpose(1, 2, 0), 0.0, 1.0).astype(np.float32),
        "sensor": sensor,
        "camera_wb": cam_wb,
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

    This public API remains controlled by the caller. Raw-Alchemy keeps
    denoise disabled by default; the UI switch decides whether this function
    is called.
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
        import onnxruntime
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
