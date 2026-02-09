"""
GPU Runtime Manager - On-demand CUDA library download and management.

This module handles:
1. Checking if CUDA libraries are available locally
2. Downloading CUDA runtime packages from GitHub releases
3. Setting up DLL paths for ONNX Runtime

CUDA packages are stored in ~/.raw_alchemy/cuda_runtime/
"""
import os
import sys
import platform
import zipfile
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable
from loguru import logger

# Version of the CUDA runtime package
# Update this when releasing new CUDA packages
CUDA_RUNTIME_VERSION = "1.0.0"

# GitHub repository for CUDA runtime releases
GITHUB_REPO = "shenmintao/raw-alchemy"

# Expected package names on GitHub releases
CUDA_PACKAGE_NAMES = {
    'Windows': 'cuda-runtime-windows-x64.zip',
    'Linux': 'cuda-runtime-linux-x64.tar.gz',
}

# Required DLLs that must exist for CUDA to work
REQUIRED_DLLS = {
    'Windows': [
        'cudart64_12.dll',
        'cublas64_12.dll',
        'cublasLt64_12.dll', 
        'cudnn64_9.dll',
    ],
    'Linux': [
        'libcudart.so.12',
        'libcublas.so.12',
        'libcublasLt.so.12',
        'libcudnn.so.9',
    ],
}


def detect_gpu_vendor() -> dict:
    """
    Detect the GPU vendor on the system.
    
    Returns:
        Dict with:
        - vendor: 'nvidia', 'amd', 'intel', or 'unknown'
        - name: GPU name string
        - cuda_compatible: True if NVIDIA GPU detected
    """
    system = platform.system()
    result = {
        'vendor': 'unknown',
        'name': '',
        'cuda_compatible': False,
    }
    
    if system == 'Windows':
        try:
            import subprocess
            # Use WMIC to get GPU info
            output = subprocess.check_output(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                text=True, stderr=subprocess.DEVNULL, timeout=5
            )
            lines = [l.strip() for l in output.strip().split('\n') if l.strip() and l.strip() != 'Name']
            if lines:
                gpu_name = lines[0]
                result['name'] = gpu_name
                gpu_lower = gpu_name.lower()
                
                if 'nvidia' in gpu_lower or 'geforce' in gpu_lower or 'quadro' in gpu_lower or 'rtx' in gpu_lower or 'gtx' in gpu_lower:
                    result['vendor'] = 'nvidia'
                    result['cuda_compatible'] = True
                elif 'amd' in gpu_lower or 'radeon' in gpu_lower:
                    result['vendor'] = 'amd'
                elif 'intel' in gpu_lower:
                    result['vendor'] = 'intel'
        except Exception as e:
            logger.debug(f"Failed to detect GPU via WMIC: {e}")
    
    elif system == 'Linux':
        try:
            import subprocess
            # Try lspci
            output = subprocess.check_output(
                ['lspci'], text=True, stderr=subprocess.DEVNULL, timeout=5
            )
            for line in output.split('\n'):
                if 'VGA' in line or '3D' in line:
                    result['name'] = line
                    line_lower = line.lower()
                    if 'nvidia' in line_lower:
                        result['vendor'] = 'nvidia'
                        result['cuda_compatible'] = True
                        break
                    elif 'amd' in line_lower or 'radeon' in line_lower:
                        result['vendor'] = 'amd'
                        break
                    elif 'intel' in line_lower:
                        result['vendor'] = 'intel'
                        break
        except Exception as e:
            logger.debug(f"Failed to detect GPU via lspci: {e}")
    
    elif system == 'Darwin':
        # macOS - check for Apple Silicon or discrete GPU
        try:
            import subprocess
            output = subprocess.check_output(
                ['system_profiler', 'SPDisplaysDataType'],
                text=True, stderr=subprocess.DEVNULL, timeout=10
            )
            result['name'] = 'Apple GPU'
            if 'Apple' in output:
                result['vendor'] = 'apple'
            # macOS uses CoreML, not CUDA
            result['cuda_compatible'] = False
        except Exception:
            pass
    
    return result


def get_cuda_runtime_dir() -> Path:
    """Get the directory where CUDA runtime libraries are stored."""
    return Path(os.path.expanduser('~/.raw_alchemy/cuda_runtime'))


def get_cuda_version_file() -> Path:
    """Get the path to the version file."""
    return get_cuda_runtime_dir() / 'version.txt'


def is_cuda_runtime_installed() -> bool:
    """Check if CUDA runtime libraries are installed locally."""
    system = platform.system()
    if system not in REQUIRED_DLLS:
        return False
    
    cuda_dir = get_cuda_runtime_dir()
    if not cuda_dir.exists():
        return False
    
    # Check if all required DLLs exist
    required = REQUIRED_DLLS[system]
    for dll_name in required:
        dll_path = cuda_dir / dll_name
        if not dll_path.exists():
            logger.debug(f"Missing CUDA library: {dll_name}")
            return False
    
    return True


def get_installed_cuda_version() -> Optional[str]:
    """Get the version of installed CUDA runtime, or None if not installed."""
    version_file = get_cuda_version_file()
    if version_file.exists():
        try:
            return version_file.read_text().strip()
        except Exception:
            pass
    return None


def is_cuda_update_available() -> bool:
    """Check if a newer CUDA runtime version is available."""
    installed = get_installed_cuda_version()
    if installed is None:
        return True  # Not installed, so "update" is available
    return installed != CUDA_RUNTIME_VERSION


def get_cuda_download_url() -> Optional[str]:
    """Get the download URL for the CUDA runtime package."""
    system = platform.system()
    if system not in CUDA_PACKAGE_NAMES:
        return None
    
    package_name = CUDA_PACKAGE_NAMES[system]
    # Use GitHub releases API to get the download URL
    return f"https://github.com/{GITHUB_REPO}/releases/download/cuda-runtime-v{CUDA_RUNTIME_VERSION}/{package_name}"


def get_cuda_package_size_mb() -> int:
    """Get the approximate download size in MB."""
    system = platform.system()
    if system == 'Windows':
        return 500  # Approximate compressed size
    elif system == 'Linux':
        return 400
    return 500


def download_cuda_runtime(
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> bool:
    """
    Download and install the CUDA runtime package.
    
    Args:
        progress_callback: Optional callback(downloaded_bytes, total_bytes)
    
    Returns:
        True if successful, False otherwise
    """
    import urllib.request
    import urllib.error
    
    system = platform.system()
    if system not in CUDA_PACKAGE_NAMES:
        logger.error(f"CUDA runtime not available for {system}")
        return False
    
    download_url = get_cuda_download_url()
    if not download_url:
        logger.error("Could not determine CUDA download URL")
        return False
    
    cuda_dir = get_cuda_runtime_dir()
    cuda_dir.mkdir(parents=True, exist_ok=True)
    
    package_name = CUDA_PACKAGE_NAMES[system]
    temp_file = cuda_dir / f"temp_{package_name}"
    
    try:
        logger.info(f"Downloading CUDA runtime from: {download_url}")
        
        # Download with progress
        req = urllib.request.Request(download_url)
        req.add_header('User-Agent', 'RawAlchemy')
        
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_file, 'wb') as f:
                while True:
                    chunk = response.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)
        
        logger.info(f"Download complete, extracting...")
        
        # Extract the package
        if package_name.endswith('.zip'):
            with zipfile.ZipFile(temp_file, 'r') as zf:
                zf.extractall(cuda_dir)
        elif package_name.endswith('.tar.gz'):
            with tarfile.open(temp_file, 'r:gz') as tf:
                tf.extractall(cuda_dir)
        
        # Write version file
        version_file = get_cuda_version_file()
        version_file.write_text(CUDA_RUNTIME_VERSION)
        
        logger.info(f"CUDA runtime installed successfully to {cuda_dir}")
        return True
        
    except urllib.error.URLError as e:
        logger.error(f"Failed to download CUDA runtime: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to install CUDA runtime: {e}")
        return False
    finally:
        # Clean up temp file
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


def setup_cuda_dll_paths() -> bool:
    """
    Setup DLL search paths to include locally installed CUDA runtime.
    
    This should be called before importing onnxruntime.
    
    Returns:
        True if CUDA paths were set up, False if CUDA is not available
    """
    if platform.system() != 'Windows':
        # On Linux, we need to set LD_LIBRARY_PATH before process starts
        # This is handled differently (environment variable or ldconfig)
        cuda_dir = get_cuda_runtime_dir()
        if cuda_dir.exists():
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if str(cuda_dir) not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{cuda_dir}:{current_ld_path}"
            return True
        return False
    
    # Windows: Use os.add_dll_directory
    cuda_dir = get_cuda_runtime_dir()
    if not cuda_dir.exists():
        return False
    
    if not is_cuda_runtime_installed():
        return False
    
    try:
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(str(cuda_dir))
            logger.debug(f"Added CUDA DLL directory: {cuda_dir}")
        
        # Also add to PATH for compatibility
        current_path = os.environ.get('PATH', '')
        if str(cuda_dir) not in current_path:
            os.environ['PATH'] = f"{cuda_dir}{os.pathsep}{current_path}"
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup CUDA DLL paths: {e}")
        return False


def get_cuda_status() -> dict:
    """
    Get the current CUDA runtime status.
    
    Returns:
        Dict with status information
    """
    return {
        'platform_supported': platform.system() in CUDA_PACKAGE_NAMES,
        'installed': is_cuda_runtime_installed(),
        'installed_version': get_installed_cuda_version(),
        'latest_version': CUDA_RUNTIME_VERSION,
        'update_available': is_cuda_update_available(),
        'cuda_dir': str(get_cuda_runtime_dir()),
        'download_size_mb': get_cuda_package_size_mb(),
    }


def remove_cuda_runtime() -> bool:
    """Remove the installed CUDA runtime to free disk space."""
    cuda_dir = get_cuda_runtime_dir()
    if cuda_dir.exists():
        try:
            shutil.rmtree(cuda_dir)
            logger.info(f"Removed CUDA runtime from {cuda_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove CUDA runtime: {e}")
            return False
    return True
