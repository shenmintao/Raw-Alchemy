import json
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

# Add src to path to import math_ops for AOT compilation
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from raw_alchemy.math_ops import cc
except ImportError:
    cc = None


class PlatformWheel(bdist_wheel):
    """Lensfun/RawSpeed are native libraries, so this is not an any-platform wheel."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        # Preserve wheel's minimum-OS scan of bundled Mach-O libraries.
        # Lensfun is arm64-only even if the interpreter is universal2.
        if sys.platform == "darwin" and plat.endswith("_universal2"):
            plat = plat.removesuffix("_universal2") + "_arm64"
        # Optional AOT extensions require their interpreter ABI. Otherwise
        # Python code loads the platform-specific runtime through ctypes.
        if self.distribution.has_ext_modules():
            return python, abi, plat
        return "py3", "none", plat


class CustomBuildPy(build_py):
    """Install verified native files into this platform's build output."""
    def run(self):
        super().run()
        from build_support import ensure_lensfun
        ensure_lensfun(Path(self.build_lib) / 'raw_alchemy' / 'vendor' / 'lensfun')


ext_modules = []
if cc:
    ext = cc.distutils_extension()
    ext.name = "raw_alchemy.math_ops_ext"
    ext_modules.append(ext)

setup(
    cmdclass={
        "build_py": CustomBuildPy,
        "bdist_wheel": PlatformWheel,
    },
    ext_modules=ext_modules,
)