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
INCLUDE_DLLS = [
    'cudart64_12',
    'cublas64_12',
    'cublasLt64_12', 
    'cudnn64_9',
    'cudnn_ops64_9',
    'cudnn_cnn64_9',
    'cudnn_graph64_9',
    'cudnn_adv64_9',  # May be needed for some models
    'cufft64_11',
    'nvJitLink',
]

# Large optional DLLs to EXCLUDE (reduces size by ~500MB)
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
    pattern = '*.dll' if sys.platform == 'win32' else '*.so*'
    
    for dll_path in nvidia_base.rglob(pattern):
        dll_name = dll_path.stem.lower()
        
        # Skip excluded patterns
        if any(excl in dll_name for excl in EXCLUDE_PATTERNS):
            print(f"  Skipping (excluded): {dll_path.name}")
            continue
        
        # Check if this DLL is in our include list
        if any(inc.lower() in dll_name for inc in INCLUDE_DLLS):
            dlls.append(dll_path)
            print(f"  Including: {dll_path.name} ({dll_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    return dlls


def create_package(dlls: list, output_name: str):
    """Create the compressed package."""
    total_size = sum(d.stat().st_size for d in dlls)
    print(f"\nTotal uncompressed size: {total_size / 1024 / 1024:.1f} MB")
    
    if sys.platform == 'win32':
        output_file = f"{output_name}.zip"
        print(f"Creating {output_file}...")
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for dll in dlls:
                zf.write(dll, dll.name)
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
