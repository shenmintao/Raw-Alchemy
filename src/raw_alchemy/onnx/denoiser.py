"""
CANS RAW V2 denoiser — packed RAW → ProPhoto Linear RGB via ONNX Runtime.

Two ONNX models (auto-selected by CFA pattern):
  - cans_raw_v2_bayer_fp16.onnx   (4ch Bayer → 3ch ProPhoto RGB, 2× upscale)
  - cans_raw_v2_xtrans_fp16.onnx  (9ch X-Trans → 3ch ProPhoto RGB, 3× upscale)

Tile-based inference with Hann window blending for seamless large-image processing.

Supports:
  - Windows: CPU, CUDA (if available)
  - macOS: CPU, CoreML (if available)
  - Linux: CPU, CUDA (if available)
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
_rs_decoder = None

BAYER_MODEL = "cans_raw_v2_bayer_fp16.onnx"
XTRANS_MODEL = "cans_raw_v2_xtrans_fp16.onnx"

# Tile size at packed resolution (before upscale)
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_OVERLAP = 32


# ---------------------------------------------------------------------------
# RAW packing (from preprocess_raw.py)
# ---------------------------------------------------------------------------

def _detect_sensor_from_result(result) -> str:
    """Detect sensor type from RawSpeed decode result."""
    if result.is_xtrans:
        return 'xtrans'
    elif result.is_bayer:
        return 'bayer'
    else:
        raise ValueError(f"Unknown CFA filter code: 0x{result.filters:08x}")


def _pack_bayer(result) -> np.ndarray:
    """Pack Bayer raw to (H/2, W/2, 4) float32, black-level subtracted and normalized.

    Channel order: [R, G1, B, G2] at positions (0,0), (0,1), (1,1), (1,0).
    Uses RawSpeed decode result instead of rawpy.
    """
    im = result.bayer.astype(np.float32)
    bl = float(result.black_levels[0])
    wl = float(result.white_level)
    im = np.maximum(im - bl, 0) / (wl - bl)

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


def _pack_xtrans(result) -> np.ndarray:
    """Pack X-Trans raw to (H/3, W/3, 9) float32, SID convention.

    Uses RawSpeed decode result instead of rawpy.
    """
    im = result.bayer.astype(np.float32)
    bl = float(result.black_levels[0])
    wl = float(result.white_level)
    im = np.maximum(im - bl, 0) / (wl - bl)

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
    """Get execution providers (CUDA > CoreML > CPU). DirectML excluded."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

    available = ort.get_available_providers()
    providers = []

    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
        logger.info("Using CUDA execution provider")
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

    logger.info(f"Loading CANS RAW V2 ({sensor}) from: {model_path}")

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


def _estimate_tile_vram_mb(in_ch: int, tile_size: int, upscale: int) -> float:
    """Estimate VRAM needed per tile in MB (input + output + intermediate).

    Conservative estimate: ~4x the raw input+output size for intermediate
    activations and workspace.
    """
    in_bytes = in_ch * tile_size * tile_size * 2  # fp16
    out_bytes = 3 * (tile_size * upscale) ** 2 * 2  # fp16 output
    # Model intermediate activations roughly 4x input
    return (in_bytes + out_bytes) * 4 / (1024 * 1024)


def _compute_batch_size(in_ch: int, tile_size: int, upscale: int,
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

    per_tile_mb = _estimate_tile_vram_mb(in_ch, tile_size, upscale)
    if per_tile_mb <= 0:
        return 1

    available = max(free_mb - vram_reserve_mb, per_tile_mb)
    # CANS RAW V2 ONNX has ScatterElements ops that require batch=1
    batch_size = 1

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


def _build_wb_iso(result) -> np.ndarray:
    """Build WB+ISO condition vector from RawSpeed decode result.

    Returns (4,) float32: [R/G, 1.0, B/G, log10(ISO)]
    """
    cam_wb = np.array(result.wb_coeffs, dtype=np.float32)
    g = cam_wb[1] if cam_wb[1] > 0 else 1.0
    wb3 = np.array([cam_wb[0] / g, 1.0, cam_wb[2] / g], dtype=np.float32)

    iso = float(result.iso_speed) if result.iso_speed > 0 else 100.0

    return np.concatenate([wb3, [np.log10(max(iso, 1.0))]], dtype=np.float32)


def _process_batch(session, tiles_chw: list[np.ndarray],
                   wb_iso: Optional[np.ndarray] = None) -> list[np.ndarray]:
    """Run a batch of packed tiles through the ONNX model.

    Args:
        tiles_chw: List of (C, H, W) float16 packed RAW tiles (same spatial size)
        wb_iso: Optional (4,) float32 WB+ISO condition [R/G, 1.0, B/G, log10(ISO)]
    Returns:
        List of (3, H*upscale, W*upscale) float32 ProPhoto Linear RGB
    """
    if not tiles_chw:
        return []

    if len(tiles_chw) == 1:
        batch = tiles_chw[0][np.newaxis, ...]
    else:
        batch = np.stack(tiles_chw, axis=0)

    feeds = {session.get_inputs()[0].name: batch}

    # Add WB+ISO condition if model supports it
    if wb_iso is not None and _has_wb_input(session):
        B = batch.shape[0]
        wb_batch = np.tile(wb_iso.astype(np.float16)[np.newaxis, :], (B, 1))
        feeds[session.get_inputs()[1].name] = wb_batch

    output = session.run(None, feeds)[0]
    return [output[i].astype(np.float32) for i in range(output.shape[0])]


def denoise_raw(
    raw_path: str,
    exposure_ratio: float = 1.0,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Denoise a RAW file using CANS RAW V2 (packed RAW → ProPhoto Linear RGB).

    This replaces rawpy's demosaicing entirely — the model performs both
    denoising and demosaicing in a single pass.

    Args:
        raw_path: Path to the RAW file (.ARW, .CR3, .NEF, .RAF, .DNG, etc.)
        exposure_ratio: Multiply packed RAW by this ratio (for exposure matching)
        tile_size: Tile size at packed resolution (default 256)
        tile_overlap: Overlap in packed pixels (default 32)
        progress_callback: Optional (current, total) callback

    Returns:
        (H, W, 3) float32 ProPhoto Linear RGB in [0, 1]
    """
    import time
    t0 = time.time()

    # Open RAW via RawSpeed, pack, extract WB+ISO and orientation
    from raw_alchemy.rawspeed_binding import RawSpeedDecoder

    # Use a module-level decoder to avoid repeated init
    global _rs_decoder
    if '_rs_decoder' not in globals() or _rs_decoder is None:
        _rs_decoder = RawSpeedDecoder()

    result = _rs_decoder.decode(raw_path)
    sensor = _detect_sensor_from_result(result)

    if sensor == 'bayer':
        packed = _pack_bayer(result)
        upscale = 2
    else:
        packed = _pack_xtrans(result)
        upscale = 3

    # WB+ISO condition
    cam_wb = np.array(result.wb_coeffs, dtype=np.float32)
    g = cam_wb[1] if cam_wb[1] > 0 else 1.0
    wb3 = np.array([cam_wb[0] / g, 1.0, cam_wb[2] / g], dtype=np.float32)

    iso = float(result.iso_speed) if result.iso_speed > 0 else 100.0

    # Orientation: RawSpeed doesn't provide EXIF orientation.
    # Read it from EXIF via pyexiv2 (best effort).
    flip_code = 0
    try:
        import pyexiv2
        with pyexiv2.Image(raw_path) as exif_img:
            exif_data = exif_img.read_exif() or {}
            orientation = int(exif_data.get('Exif.Image.Orientation', 1))
            # Map EXIF orientation to rawpy flip codes:
            # EXIF 1=normal(0), 3=180(3), 6=90CW(6), 8=90CCW(5)
            _exif_to_flip = {1: 0, 2: 0, 3: 3, 4: 0, 5: 5, 6: 6, 7: 6, 8: 5}
            flip_code = _exif_to_flip.get(orientation, 0)
    except Exception:
        pass

    wb_iso = np.concatenate([wb3, [np.log10(max(iso, 1.0))]], dtype=np.float32)

    # Apply exposure ratio
    if exposure_ratio != 1.0:
        packed = np.clip(packed * exposure_ratio, 0.0, 1.0)

    pack_h, pack_w, in_ch = packed.shape
    out_h, out_w = pack_h * upscale, pack_w * upscale

    logger.info(
        f"CANS denoise: {os.path.basename(raw_path)} ({sensor}), "
        f"packed {pack_w}x{pack_h} -> output {out_w}x{out_h}, "
        f"WB=[{wb3[0]:.2f}, {wb3[2]:.2f}] ISO={iso:.0f} flip={flip_code}"
    )

    # Get ONNX session
    session = _get_session(sensor)

    # Convert to CHW float16
    packed_chw = np.ascontiguousarray(packed.transpose(2, 0, 1), dtype=np.float16)

    # Tile positions (at packed resolution)
    step = tile_size - tile_overlap
    ys = list(range(0, max(pack_h - tile_size, 0) + 1, step))
    xs = list(range(0, max(pack_w - tile_size, 0) + 1, step))
    if ys[-1] + tile_size < pack_h:
        ys.append(pack_h - tile_size)
    if xs[-1] + tile_size < pack_w:
        xs.append(pack_w - tile_size)

    total_tiles = len(ys) * len(xs)

    # Compute batch size from available VRAM
    batch_size = _compute_batch_size(in_ch, tile_size, upscale)

    # Small image: single tile
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
        result = result[:, :out_h, :out_w]
        if progress_callback:
            progress_callback(1, 1)
        elapsed = time.time() - t0
        logger.info(f"CANS denoise done in {elapsed:.1f}s (1 tile, {_session_provider})")
        out = np.clip(result.transpose(1, 2, 0), 0.0, 1.0)
        return _apply_flip(out, flip_code)

    # Collect all tile coordinates
    tile_infos = []
    for y in ys:
        y2 = min(y + tile_size, pack_h)
        y = max(y2 - tile_size, 0)
        for x in xs:
            x2 = min(x + tile_size, pack_w)
            x = max(x2 - tile_size, 0)
            tile_infos.append((y, x, y2, x2))

    total_tiles = len(tile_infos)

    # Tile-based processing with Hann window blending
    out_tile_size = tile_size * upscale
    ramp = np.hanning(out_tile_size).astype(np.float32)
    w2d = np.outer(ramp, ramp)[np.newaxis, :, :]  # (1, tile_out, tile_out)

    accum = np.zeros((3, out_h, out_w), dtype=np.float32)
    weight = np.zeros((1, out_h, out_w), dtype=np.float32)

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
            oph = min(pred.shape[1], (y2 - y) * upscale)
            opw = min(pred.shape[2], (x2 - x) * upscale)
            pred = pred[:, :oph, :opw]
            wt = w2d[:, :oph, :opw]

            oy, ox = y * upscale, x * upscale
            accum[:, oy:oy + oph, ox:ox + opw] += pred * wt
            weight[:, oy:oy + oph, ox:ox + opw] += wt

            current += 1
            if progress_callback:
                progress_callback(current, total_tiles)

    weight = np.maximum(weight, 1e-8)
    result = accum / weight

    elapsed = time.time() - t0
    n_batches = (total_tiles + batch_size - 1) // batch_size
    logger.info(
        f"CANS denoise done in {elapsed:.1f}s "
        f"({total_tiles} tiles, {n_batches} batches of {batch_size}, {_session_provider})"
    )

    out = np.clip(result.transpose(1, 2, 0), 0.0, 1.0)
    return _apply_flip(out, flip_code)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Check if the CANS denoiser is available (models + ONNX Runtime)."""
    try:
        import onnxruntime
        _find_model(BAYER_MODEL)
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
