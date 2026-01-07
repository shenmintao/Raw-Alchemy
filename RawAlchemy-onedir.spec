# -*- mode: python ; coding: utf-8 -*-
import sys


# --- Platform-specific settings ---
# Enable strip on Linux and macOS for a smaller executable.
# On Windows, stripping can sometimes cause issues with antivirus software
# or runtime behavior, so it's safer to leave it disabled.
strip_executable = True if sys.platform.startswith('linux') else False

# --- Platform-specific binaries ---
binaries_list = []
if sys.platform == 'darwin' or sys.platform.startswith('linux'):
    import os
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


a = Analysis(
    ['src/raw_alchemy/gui.py'],
    pathex=[],
    binaries=binaries_list,
    datas=[('src/raw_alchemy/vendor', 'vendor'), ('icon.ico', '.'), ('icon.png', '.')],
    hiddenimports=['tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pandas',
        'IPython',
        'PyQt5',
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