"""
Script to create CUDA runtime package from locally installed nvidia packages.
Run this on a machine with nvidia-cudnn-cu12, nvidia-cublas-cu12 installed.

Usage:
    pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
    python create_cuda_package.py
"""
import os
import sys
import site
import zipfile
import tarfile
from pathlib import Path

# DLLs to include (essential for ONNX Runtime CUDA)
# Windows uses names like cudart64_12.dll
# Linux uses names like libcudart.so.12
INCLUDE_PATTERNS_WINDOWS = [
    'cudart64_12',
    'cublas64_12',
    'cublasLt64_12', 
    'cudnn64_9',
    'cudnn_ops64_9',
    'cudnn_cnn64_9',
    'cudnn_graph64_9',
    'cudnn_adv64_9',
    'cufft64_11',
    'nvJitLink',
    'zlibwapi',
]

INCLUDE_PATTERNS_LINUX = [
    'libcudart',
    'libcublas',
    'libcublasLt',
    'libcudnn',
    'libcufft',
    'libnvJitLink',
]

# Large optional DLLs to EXCLUDE (reduces size by ~1GB)
EXCLUDE_PATTERNS = [
    'cudnn_engines_precompiled',
    'cudnn_heuristic',
]


def find_nvidia_base():
    """Find the nvidia package directory."""
    for sp in site.getsitepackages() + [site.getusersitepackages() or '']:
        potential = Path(sp) / 'nvidia'
        if potential.exists():
            return potential
    # Check venv
    venv_path = Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia'
    if venv_path.exists():
        return venv_path
    return None


def collect_dlls(nvidia_base: Path) -> list:
    """Collect required DLLs from nvidia package."""
    dlls = []
    
    if sys.platform == 'win32':
        pattern = '*.dll'
        include_patterns = INCLUDE_PATTERNS_WINDOWS
    else:
        pattern = '*.so*'
        include_patterns = INCLUDE_PATTERNS_LINUX
    
    # Recursive search for ALL DLLs in nvidia package
    print(f"Searching for DLLs in {nvidia_base}...")
    
    # Also look for zlibwapi.dll which is sometimes needed by cuDNN on Windows
    # It might be in a different location or not present in nvidia-* packages
    # For now we assume the user has a working environment and we try to find it
    
    for dll_path in nvidia_base.rglob(pattern):
        dll_name = dll_path.name
        dll_lower = dll_name.lower()
        
        # Skip excluded patterns
        is_excluded = False
        for excl in EXCLUDE_PATTERNS:
            if excl.lower() in dll_lower:
                is_excluded = True
                break
        if is_excluded:
            continue
            
        # Check if this file matches any include pattern
        is_included = False
        for inc in include_patterns:
            if inc.lower() in dll_lower:
                is_included = True
                break
        
        if is_included:
            dlls.append(dll_path)
            size_mb = dll_path.stat().st_size / 1024 / 1024
            print(f"  Including: {dll_name} ({size_mb:.1f} MB)")

    return dlls


def create_package(dlls: list, output_name: str):
    """Create the compressed package flattening directory structure."""
    if not dlls:
        print("No DLLs found to package!")
        return

    total_size = sum(d.stat().st_size for d in dlls)
    print(f"\nTotal uncompressed size: {total_size / 1024 / 1024:.1f} MB")
    
    if sys.platform == 'win32':
        output_file = f"{output_name}.zip"
        print(f"Creating {output_file}...")
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for dll_path in dlls:
                # Flatten: put all DLLs in root of zip
                zf.write(dll_path, dll_path.name)
    else:
        output_file = f"{output_name}.tar.gz"
        print(f"Creating {output_file}...")
        with tarfile.open(output_file, 'w:gz') as tf:
            for dll in dlls:
                tf.add(dll, arcname=dll.name)
    
    compressed_size = Path(output_file).stat().st_size
    print(f"Compressed size: {compressed_size / 1024 / 1024:.1f} MB")
    print(f"Created: {output_file}")


def main():
    print("CUDA Runtime Package Creator")
    print("=" * 40)
    
    nvidia_base = find_nvidia_base()
    if not nvidia_base:
        print("ERROR: nvidia package not found!")
        print("Install with: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12")
        sys.exit(1)
    
    print(f"Found nvidia at: {nvidia_base}")
    print("\nCollecting DLLs...")
    
    dlls = collect_dlls(nvidia_base)
    if not dlls:
        print("ERROR: No DLLs found!")
        sys.exit(1)
    
    print(f"\nCollected {len(dlls)} DLLs")
    
    # Create package
    platform = 'windows' if sys.platform == 'win32' else 'linux'
    output_name = f"cuda-runtime-{platform}-x64"
    create_package(dlls, output_name)
    
    print("\nDone! Upload this file to GitHub Releases with tag: cuda-runtime-v1.0.0")


if __name__ == '__main__':
    main()
