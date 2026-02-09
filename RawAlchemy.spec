# -*- mode: python ; coding: utf-8 -*-
import sys

# --- Platform-specific settings ---
# Enable strip on Linux and macOS for a smaller executable.
# On Windows, stripping can sometimes cause issues with antivirus software
# or runtime behavior, so it's safer to leave it disabled.
# On macOS, stripping the Python shared library can lead to runtime errors,
# such as "Failed to load Python shared library". Disabling strip for macOS
# is a safer approach to ensure all necessary symbols are preserved.
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


# Determine ONNX Runtime package based on platform
ort_package = 'onnxruntime' # Default fallback
if sys.platform == 'win32' or sys.platform.startswith('linux'):
    ort_package = 'onnxruntime-gpu'
elif sys.platform == 'darwin':
    ort_package = 'onnxruntime-coreml'

print(f"Using ONNX Runtime package: {ort_package}")


a = Analysis(
    ['src/raw_alchemy/main.py'],
    pathex=[],
    binaries=binaries_list,
    datas=[('src/raw_alchemy/vendor', 'vendor'),('src/raw_alchemy/locales', 'locales'), ('icon.ico', '.'), ('icon.png', '.')],
    hiddenimports=['tkinter', 'loguru', 'pyexiv2', ort_package],
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

# On macOS, BUNDLE is used, which has its own icon parameter.
# The .ico format is for Windows, so we remove it from datas on macOS.
if sys.platform == 'darwin':
    a.datas = [item for item in a.datas if item[0] != 'icon.ico']

pyz = PYZ(a.pure)

# Platform-specific EXE and BUNDLE for macOS .app creation
# Create a one-file executable.
# Binaries and data are included directly in the EXE for a one-file build.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RawAlchemy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=strip_executable,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Set the icon based on the platform.
    icon='icon.icns' if sys.platform == 'darwin' else 'icon.ico',
)

# If on macOS, bundle the one-file executable into a .app directory.
# This is required for a proper GUI application on macOS.
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='RawAlchemy.app',
        icon='icon.icns',
        bundle_identifier=None,
    )
