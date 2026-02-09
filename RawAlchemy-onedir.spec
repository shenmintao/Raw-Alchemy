# -*- mode: python ; coding: utf-8 -*-
import sys


# --- Platform-specific settings ---
# Enable strip on Linux and macOS for a smaller executable.
# On Windows, stripping can sometimes cause issues with antivirus software
# or runtime behavior, so it's safer to leave it disabled.
strip_executable = True if sys.platform.startswith('linux') else False

# --- Platform-specific binaries ---
import os
import glob
from PyInstaller.utils.hooks import collect_all

binaries_list = []

# Add math_ops_ext compiled module (.pyd on Windows, .so on Linux/macOS)
# First check in src directory (development)
math_ops_ext_files = glob.glob(os.path.join('src', 'raw_alchemy', 'math_ops_ext*.pyd'))
if not math_ops_ext_files:
    math_ops_ext_files = glob.glob(os.path.join('src', 'raw_alchemy', 'math_ops_ext*.so'))
# If not found, check in installed package
if not math_ops_ext_files:
    try:
        import raw_alchemy
        pkg_dir = os.path.dirname(raw_alchemy.__file__)
        math_ops_ext_files = glob.glob(os.path.join(pkg_dir, 'math_ops_ext*.pyd'))
        if not math_ops_ext_files:
            math_ops_ext_files = glob.glob(os.path.join(pkg_dir, 'math_ops_ext*.so'))
    except ImportError:
        pass

for pyd_file in math_ops_ext_files:
    binaries_list.append((pyd_file, 'raw_alchemy'))

if sys.platform == 'darwin' or sys.platform.startswith('linux'):
    import rawpy

    # Find the path to libraw_r library within the rawpy package
    rawpy_path = os.path.dirname(rawpy.__file__)
    lib_file = None
    for f in os.listdir(rawpy_path):
        if f.startswith('libraw_r'):
            lib_file = os.path.join(rawpy_path, f)
            break
    if lib_file:
        binaries_list.append((lib_file, '.'))

if sys.platform == 'darwin':
    # List of libraries to manually bundle.
    # 这里的列表完全基于 otool -L 的输出结果整理
    libs_to_bundle = [
        # --- Brotli (必须同时包含 dec 和 common) ---
        '/opt/homebrew/opt/brotli/lib/libbrotlidec.1.dylib',
        '/opt/homebrew/opt/brotli/lib/libbrotlicommon.1.dylib',
        
        # --- Gettext ---
        '/opt/homebrew/opt/gettext/lib/libintl.8.dylib',

        # --- INIH (必须同时包含 Reader 和 Core) ---
        '/opt/homebrew/opt/inih/lib/libINIReader.0.dylib',
        '/opt/homebrew/opt/inih/lib/libinih.0.dylib',
    ]
    
    found_libs = set()
    
    for lib_path in libs_to_bundle:
        if os.path.exists(lib_path):
            lib_name = os.path.basename(lib_path)
            # Only add if we haven't added this lib name yet
            if lib_name not in found_libs:
                print(f"Found system library: {lib_path}")
                binaries_list.append((lib_path, '.')) 
                found_libs.add(lib_name)
        else:
            print(f"⚠️ WARNING: Library not found: {lib_path}")
            print("Please run: brew install brotli gettext inih")

pyexiv2_ret = collect_all('pyexiv2')
pyexiv2_datas = pyexiv2_ret[0]
pyexiv2_binaries = pyexiv2_ret[1]
pyexiv2_hiddenimports = pyexiv2_ret[2]
binaries_list.extend(pyexiv2_binaries)


# --- NVIDIA / CUDA Dependencies (Unified Build) ---
# Check for nvidia packages and bundle their DLLs if present.
# This allows the same spec to build a GPU-capable executable if built in a GPU environment.
def collect_nvidia_libs():
    libs = []
    # Subfolders within the nvidia package that contain CUDA/cuDNN binaries
    # Only include the essential packages for ONNX Runtime
    nvidia_subpackages = [
        'cudnn',
        'cublas',
        'cuda_runtime',
        'cufft',  # Required for some operations
        'nvjitlink',  # Required for JIT compilation
    ]
    
    # Large optional DLLs to exclude (saves ~500MB+)
    # These are precompiled kernels and heuristics that cuDNN can regenerate at runtime if needed
    exclude_patterns = [
        'cudnn_engines_precompiled',  # ~470MB of precompiled kernels
        'cudnn_heuristic',  # ~54MB of heuristic data
    ]
    
    import pathlib
    import site
    
    print("Checking for NVIDIA CUDA libraries...")
    
    # Find nvidia package in site-packages
    nvidia_base = None
    for sp in site.getsitepackages() + [site.getusersitepackages() or '']:
        potential = pathlib.Path(sp) / 'nvidia'
        if potential.exists():
            nvidia_base = potential
            break
    
    # Also check in the current venv
    if nvidia_base is None:
        venv_path = pathlib.Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia'
        if venv_path.exists():
            nvidia_base = venv_path
    
    if nvidia_base is None:
        print("  NVIDIA package not found in site-packages.")
        return libs
    
    print(f"  Found nvidia package at: {nvidia_base}")
    
    for pkg_name in nvidia_subpackages:
        pkg_path = nvidia_base / pkg_name
        if not pkg_path.exists():
            print(f"  Package nvidia.{pkg_name} not found.")
            continue
            
        print(f"  Found nvidia.{pkg_name} at {pkg_path}")
        
        # Collect all DLLs (Windows) or .so (Linux) from bin/ or lib/ subdirectories
        patterns = ['*.dll'] if sys.platform == 'win32' else ['*.so*']
        
        found_count = 0
        skipped_count = 0
        for pattern in patterns:
            for dll_path in pkg_path.rglob(pattern):
                # Check if this DLL should be excluded
                dll_name = dll_path.stem.lower()
                should_exclude = any(excl in dll_name for excl in exclude_patterns)
                
                if should_exclude:
                    skipped_count += 1
                    continue
                    
                # Add to binaries. 
                # We place them in the root ('.') so ONNX Runtime can find them easily.
                libs.append((str(dll_path), '.'))
                found_count += 1
        print(f"    Collected {found_count} binary files (skipped {skipped_count} optional large files).")

    return libs

nvidia_binaries = collect_nvidia_libs()
binaries_list.extend(nvidia_binaries)

# Ensure ONNX Runtime providers are collected
# Specifically look for onnxruntime_providers_*.dll
try:
    import onnxruntime
    ort_path = os.path.dirname(onnxruntime.__file__)
    print(f"Checking onnxruntime at {ort_path}")
    if sys.platform == 'win32':
        ort_providers = glob.glob(os.path.join(ort_path, 'onnxruntime_providers_*.dll'))
        for provider in ort_providers:
            print(f"  Found provider: {os.path.basename(provider)}")
            binaries_list.append((provider, '.'))
except ImportError:
    print("onnxruntime not installed, skipping provider collection.")




a = Analysis(
    ['src/raw_alchemy/main.py'],
    pathex=[],
    binaries=binaries_list,
    datas=[('src/raw_alchemy/vendor', 'vendor'),('src/raw_alchemy/locales', 'locales'), ('icon.ico', '.'), ('icon.png', '.')],
    hiddenimports=['tkinter', 'loguru', 'pyexiv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pandas',
        'IPython',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'qtpy',
        'test',
        'doctest',
        'distutils',
        'setuptools',
        'wheel',
        'pkg_resources',
        'Cython',
        'PyInstaller',
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RawAlchemy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=strip_executable,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=strip_executable,
    upx=False,
    upx_exclude=[],
    name='RawAlchemy',
)