"""
Denoiser module using ONNX Runtime for cross-platform image denoising.

Supports:
- Windows: CPU, CUDA (if available)
- macOS: CPU, CoreML (if available)
- Linux: CPU, CUDA (if available)

Note: DirectML is explicitly disabled due to compatibility issues.
"""
import os
import sys
import platform
import numpy as np
from typing import Optional, Callable, Tuple
from loguru import logger


def _setup_cuda_paths():
    """
    Setup CUDA library paths for onnxruntime-gpu.
    
    Checks two locations:
    1. Local downloaded CUDA runtime (~/.raw_alchemy/cuda_runtime/)
    2. nvidia packages in site-packages (pip install nvidia-cublas-cu12 etc.)
    """
    if platform.system() != 'Windows':
        return
    
    # First, try to use locally downloaded CUDA runtime
    try:
        from . import gpu_runtime
        if gpu_runtime.setup_cuda_dll_paths():
            logger.debug("Using locally installed CUDA runtime")
            return
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to setup local CUDA runtime: {e}")
    
    # Fallback: Check for nvidia packages in site-packages
    try:
        import site
        site_packages = site.getsitepackages()
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # Virtual environment
            venv_site = os.path.join(sys.prefix, 'Lib', 'site-packages')
            if venv_site not in site_packages:
                site_packages.insert(0, venv_site)
        
        nvidia_paths = []
        for sp in site_packages:
            nvidia_base = os.path.join(sp, 'nvidia')
            if os.path.isdir(nvidia_base):
                # Look for bin directories containing DLLs
                for subdir in os.listdir(nvidia_base):
                    bin_path = os.path.join(nvidia_base, subdir, 'bin')
                    if os.path.isdir(bin_path):
                        nvidia_paths.append(bin_path)
        
        # Add paths using os.add_dll_directory (Python 3.8+)
        if hasattr(os, 'add_dll_directory'):
            for path in nvidia_paths:
                try:
                    os.add_dll_directory(path)
                    logger.debug(f"Added DLL directory: {path}")
                except Exception as e:
                    logger.debug(f"Failed to add DLL directory {path}: {e}")
        
        # Also add to PATH for older Python versions
        if nvidia_paths:
            os.environ['PATH'] = os.pathsep.join(nvidia_paths) + os.pathsep + os.environ.get('PATH', '')
            
    except Exception as e:
        logger.debug(f"Failed to setup CUDA paths: {e}")


# Setup CUDA paths before importing onnxruntime
_setup_cuda_paths()

# Global session cache
_session = None
_session_provider = None

# Model filename - using NIND UtNet FP16 model
# This model uses FP16 input/output with fixed 504x504 tile size
MODEL_FILENAME = "nind_utnet_fp16.onnx"


def _get_base_path() -> str:
    """
    获取资源的基础路径。
    兼容：开发环境、PyInstaller (单文件/文件夹)、Nuitka (单文件/文件夹)。
    """
    # 1. 优先处理 PyInstaller One-file 模式
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS

    # 2. 处理 PyInstaller One-dir 模式 (检查是否存在 _internal 文件夹)
    # PyInstaller 6+ 默认将依赖放在 _internal 中
    executable_dir = os.path.dirname(sys.executable)
    pyinstaller_internal_path = os.path.join(executable_dir, '_internal')
    
    # 只有当处于 frozen 状态 且 _internal 文件夹确实存在时，才使用它
    if getattr(sys, 'frozen', False) and os.path.isdir(pyinstaller_internal_path):
        return pyinstaller_internal_path

    # 3. Nuitka (单文件/文件夹) 以及 普通 Python 脚本
    # Nuitka 会修正 __file__ 指向正确的运行时位置（无论是临时目录还是 dist 目录）
    return os.path.dirname(os.path.abspath(__file__))


def _get_model_path() -> str:
    """Get the path to the ONNX model file."""
    base_path = _get_base_path()
    
    # 可能的模型路径
    possible_paths = [
        # vendor 目录（模型现在存放在 vendor 中）
        os.path.join(base_path, "vendor", MODEL_FILENAME),
        # PyInstaller/Nuitka 打包后的路径
        os.path.join(base_path, "raw_alchemy", "vendor", MODEL_FILENAME),
        # 开发环境：相对于 onnx 目录向上一级
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", MODEL_FILENAME),
        # 旧路径（兼容）
        os.path.join(base_path, MODEL_FILENAME),
        os.path.join(base_path, "raw_alchemy", "onnx", MODEL_FILENAME),
    ]
    
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            logger.debug(f"Found ONNX model at: {normalized_path}")
            return normalized_path
    
    raise FileNotFoundError(
        f"ONNX model '{MODEL_FILENAME}' not found. "
        f"Searched paths: {possible_paths}"
    )


def _get_providers() -> list:
    """
    Get the list of execution providers based on the platform.
    DirectML is explicitly excluded due to compatibility issues.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
    
    available = ort.get_available_providers()
    providers = []
    system = platform.system()
    
    logger.debug(f"Available ONNX Runtime providers: {available}")
    logger.debug(f"Platform: {system}")
    
    # CUDA for Windows/Linux with NVIDIA GPU
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
        logger.info("Using CUDA execution provider")
    
    # CoreML for macOS (Apple Silicon and Intel)
    elif 'CoreMLExecutionProvider' in available and system == 'Darwin':
        providers.append('CoreMLExecutionProvider')
        logger.info("Using CoreML execution provider (macOS)")
    
    # Note: DirectML is explicitly NOT used due to compatibility issues
    # elif 'DmlExecutionProvider' in available and system == 'Windows':
    #     providers.append('DmlExecutionProvider')
    
    # Always add CPU as fallback
    providers.append('CPUExecutionProvider')
    
    if len(providers) == 1:
        logger.info("Using CPU execution provider (no GPU acceleration available)")
    
    return providers


def _get_session():
    """Get or create the ONNX Runtime inference session (cached)."""
    global _session, _session_provider
    
    if _session is not None:
        return _session
    
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
    
    model_path = _get_model_path()
    providers = _get_providers()
    
    logger.info(f"Loading ONNX model from: {model_path}")
    
    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Enable memory pattern optimization
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    
    # Configure CUDA provider options for dynamic shapes
    provider_options = []
    for provider in providers:
        if provider == 'CUDAExecutionProvider':
            cuda_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',  # Better for dynamic shapes
                'cudnn_conv_algo_search': 'HEURISTIC',  # Faster algo selection for dynamic shapes
                'do_copy_in_default_stream': True,
                'cudnn_conv_use_max_workspace': True,  # Use more workspace for faster convs
            }
            provider_options.append((provider, cuda_options))
        else:
            provider_options.append(provider)
    
    try:
        _session = ort.InferenceSession(model_path, sess_options, providers=provider_options)
        actual_providers = _session.get_providers()
        _session_provider = actual_providers[0]
        logger.info(f"ONNX session created with provider: {_session_provider}")
        logger.info(f"All active providers: {actual_providers}")
        
        # Warn if CUDA was requested but not used
        if 'CUDAExecutionProvider' in [p[0] if isinstance(p, tuple) else p for p in provider_options]:
            if 'CUDAExecutionProvider' not in actual_providers:
                logger.warning("⚠️ CUDA was requested but not available! Falling back to CPU.")
                logger.warning("Check: 1) onnxruntime-gpu installed? 2) CUDA/cuDNN installed?")
        
        return _session
    except Exception as e:
        logger.error(f"Failed to create ONNX session: {e}")
        raise


# Tile size for the UtNet model (504 required by U-Net architecture)
MIN_TILE_SIZE = 504


def _pad_to_min_size(image: np.ndarray, min_size: int = MIN_TILE_SIZE, multiple: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to at least min_size and ensure dimensions are multiples of the given value.
    Returns padded image and original dimensions.
    
    UtNet requires fixed 512x512 input to match the ONNX model export dimensions.
    """
    h, w = image.shape[:2]
    orig_h, orig_w = h, w
    
    # First, ensure minimum size
    target_h = max(h, min_size)
    target_w = max(w, min_size)
    
    # Then, ensure multiple of 8
    target_h = ((target_h + multiple - 1) // multiple) * multiple
    target_w = ((target_w + multiple - 1) // multiple) * multiple
    
    pad_h = target_h - h
    pad_w = target_w - w
    
    if pad_h == 0 and pad_w == 0:
        return image, (orig_h, orig_w)
    
    if len(image.shape) == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    return padded, (orig_h, orig_w)


def _process_tile(session, tile: np.ndarray) -> np.ndarray:
    """
    Process a single tile through the model.
    
    Args:
        session: ONNX Runtime session
        tile: Input tile in HWC format, float32, range [0, 1]
    
    Returns:
        Denoised tile in HWC format (float32)
    """
    # Convert HWC to NCHW and ensure contiguous memory (FP16 input)
    tile_nchw = np.ascontiguousarray(tile.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float16)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tile_nchw})[0]
    
    # Convert NCHW back to HWC (return float32 for consistency)
    result = output[0].transpose(1, 2, 0).astype(np.float32)
    
    return result


def _process_tiles_batch(session, tiles: list) -> list:
    """
    Process multiple tiles in a batch for better GPU utilization.
    
    Args:
        session: ONNX Runtime session
        tiles: List of input tiles in HWC format, float32, range [0, 1]
    
    Returns:
        List of denoised tiles in HWC format (float32)
    """
    if not tiles:
        return []
    
    # Stack tiles into batch: list of HWC -> NCHW batch (FP16 input)
    batch = np.ascontiguousarray(
        np.stack([tile.transpose(2, 0, 1) for tile in tiles], axis=0),
        dtype=np.float16
    )
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: batch})[0]
    
    # Convert NCHW batch back to list of HWC (return float32 for consistency)
    results = [output[i].transpose(1, 2, 0).astype(np.float32) for i in range(output.shape[0])]
    
    return results


def denoise(
    image: np.ndarray,
    strength: float = 1.0,
    tile_size: int = 504,  # UtNet requires 504 (U-Net skip connections)
    tile_overlap: int = 64,  # Increased from 32 for better blending
    batch_size: int = 4,  # Process 4 tiles at once for best GPU utilization
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> np.ndarray:
    """
    Denoise an image using the SCUNet model.
    
    Args:
        image: Input image in HWC format, float32, range [0, 1]
        strength: Denoising strength (0.0 = no effect, 1.0 = full effect)
        tile_size: Size of tiles for processing (576 for ~3-4GB VRAM)
        tile_overlap: Overlap between tiles to avoid seams
        batch_size: Number of tiles to process in a batch (1 = dynamic model)
        progress_callback: Optional callback function(current, total) for progress updates
    
    Returns:
        Denoised image in HWC format, float32, range [0, 1]
    """
    if strength <= 0:
        return image.copy()
    
    # Clamp strength to valid range
    strength = min(max(strength, 0.0), 1.0)
    
    # Ensure image is float32 and in valid range
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Clip to [0, 1] range
    image = np.clip(image, 0.0, 1.0)
    
    # Get session and log provider info
    import time
    start_time = time.time()
    session = _get_session()
    session_time = time.time() - start_time
    
    h, w, c = image.shape
    logger.info(f"Denoising image: {w}x{h} ({w*h/1e6:.1f}MP), provider: {_session_provider}, session load: {session_time:.2f}s")
    
    # Ensure tile_size is at least MIN_TILE_SIZE
    tile_size = max(tile_size, MIN_TILE_SIZE)
    
    # For small images, process directly without tiling
    if h <= tile_size and w <= tile_size:
        # Pad to minimum size (256) and multiple of 8
        padded, (orig_h, orig_w) = _pad_to_min_size(image, MIN_TILE_SIZE, 8)
        
        if progress_callback:
            progress_callback(0, 1)
        
        denoised = _process_tile(session, padded)
        
        # Crop back to original size
        denoised = denoised[:orig_h, :orig_w, :]
        
        if progress_callback:
            progress_callback(1, 1)
        
        # Blend with original based on strength
        if strength < 1.0:
            denoised = image * (1 - strength) + denoised * strength
        
        # Release GPU memory after processing
        clear_session()
        
        return np.clip(denoised, 0.0, 1.0)
    
    # Tiled processing for large images
    step = tile_size - tile_overlap
    
    # Calculate number of tiles
    n_tiles_h = max(1, (h - tile_overlap + step - 1) // step)
    n_tiles_w = max(1, (w - tile_overlap + step - 1) // step)
    total_tiles = n_tiles_h * n_tiles_w
    
    n_batches = (total_tiles + batch_size - 1) // batch_size
    logger.debug(f"Processing {total_tiles} tiles ({n_tiles_h}x{n_tiles_w}) in {n_batches} batches (batch_size={batch_size})")
    
    # Output buffer and weight buffer for blending
    output = np.zeros_like(image)
    weights = np.zeros((h, w, 1), dtype=np.float32)
    
    # Create blending weights (smooth cosine ramp at edges for seamless blending)
    def create_blend_weights(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
        """Create smooth blending weights for a tile using cosine interpolation."""
        weight = np.ones((tile_h, tile_w, 1), dtype=np.float32)
        
        if overlap > 0:
            # Create smooth cosine ramps for edges (smoother than linear)
            # Cosine ramp: 0.5 * (1 - cos(pi * x)) for x in [0, 1]
            t = np.linspace(0, 1, overlap, dtype=np.float32)
            ramp = 0.5 * (1 - np.cos(np.pi * t))
            
            # Top edge
            weight[:overlap, :, 0] *= ramp[:, np.newaxis]
            # Bottom edge
            weight[-overlap:, :, 0] *= ramp[::-1, np.newaxis]
            # Left edge
            weight[:, :overlap, 0] *= ramp[np.newaxis, :]
            # Right edge
            weight[:, -overlap:, 0] *= ramp[::-1][np.newaxis, :]
        
        return weight
    
    current_tile = 0
    
    # Collect all tile info first
    tile_infos = []
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Calculate tile boundaries
            y_start = i * step
            x_start = j * step
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            
            # Adjust start if we're at the edge
            if y_end == h:
                y_start = max(0, h - tile_size)
            if x_end == w:
                x_start = max(0, w - tile_size)
            
            tile_infos.append((y_start, x_start, y_end, x_end))
    
    # Process tiles in batches
    for batch_start in range(0, len(tile_infos), batch_size):
        batch_end = min(batch_start + batch_size, len(tile_infos))
        batch_infos = tile_infos[batch_start:batch_end]
        
        # Prepare batch of tiles
        padded_tiles = []
        orig_sizes = []
        tile_sizes = []
        
        for y_start, x_start, y_end, x_end in batch_infos:
            # Extract tile
            tile = image[y_start:y_end, x_start:x_end, :].copy()
            tile_h, tile_w = tile.shape[:2]
            tile_sizes.append((tile_h, tile_w))
            
            # Pad tile to minimum size (256) and multiple of 8
            padded_tile, (orig_tile_h, orig_tile_w) = _pad_to_min_size(tile, MIN_TILE_SIZE, 8)
            padded_tiles.append(padded_tile)
            orig_sizes.append((orig_tile_h, orig_tile_w))
        
        # Process batch
        denoised_tiles = _process_tiles_batch(session, padded_tiles)
        
        # Apply results
        for idx, (y_start, x_start, y_end, x_end) in enumerate(batch_infos):
            orig_tile_h, orig_tile_w = orig_sizes[idx]
            tile_h, tile_w = tile_sizes[idx]
            
            # Crop back to original tile size
            denoised_tile = denoised_tiles[idx][:orig_tile_h, :orig_tile_w, :]
            
            # Create blend weights for this tile
            blend_weights = create_blend_weights(tile_h, tile_w, tile_overlap // 2)
            
            # Accumulate results
            output[y_start:y_end, x_start:x_end, :] += denoised_tile * blend_weights
            weights[y_start:y_end, x_start:x_end, :] += blend_weights
            
            current_tile += 1
            if progress_callback:
                progress_callback(current_tile, total_tiles)
    
    # Normalize by weights
    weights = np.maximum(weights, 1e-8)  # Avoid division by zero
    output = output / weights
    
    # Blend with original based on strength
    if strength < 1.0:
        output = image * (1 - strength) + output * strength
    
    # Release GPU memory after processing
    clear_session()
    
    return np.clip(output, 0.0, 1.0)


def is_available() -> bool:
    """Check if the denoiser is available (model exists and ONNX Runtime is installed)."""
    try:
        import onnxruntime
        _get_model_path()
        return True
    except (ImportError, FileNotFoundError):
        return False


def get_provider_info() -> dict:
    """Get information about the current execution provider."""
    try:
        session = _get_session()
        return {
            "available": True,
            "provider": session.get_providers()[0],
            "all_providers": session.get_providers(),
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def clear_session():
    """Clear the cached session and release GPU memory."""
    global _session, _session_provider
    if _session is not None:
        # Delete the session first
        del _session
        _session = None
        _session_provider = None
        
        # Force garbage collection to release GPU memory
        import gc
        gc.collect()
        
        logger.info("ONNX session cleared and GPU memory released")
    else:
        _session_provider = None
