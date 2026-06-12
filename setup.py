import json
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

# Add src to path to import math_ops for AOT compilation
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from raw_alchemy.math_ops import cc
except ImportError:
    cc = None


class CustomBuildPy(build_py):
    """Custom build command to download and set up Lensfun."""

    def run(self):
        self.download_and_extract_lensfun()
        super().run()

    def lensfun_vendor_complete(self, vendor_dir, system):
        """Return True when the vendored Lensfun runtime and database exist."""
        if system == "windows":
            runtime = vendor_dir / "lib" / "lensfun.dll"
        elif system == "linux":
            runtime = vendor_dir / "lib" / "liblensfun.so"
        elif system == "darwin":
            runtime = vendor_dir / "lib" / "liblensfun.dylib"
        else:
            return False

        db_dir = vendor_dir / "share" / "lensfun" / "version_2"
        return runtime.exists() and db_dir.is_dir() and any(db_dir.glob("*.xml"))

    def get_download_url(self, asset_name):
        """Gets the download URL for a given asset from the latest GitHub release."""
        api_url = "https://api.github.com/repos/shenmintao/lensfun/releases/latest"
        # Try both GITHUB_TOKEN and GH_TOKEN for compatibility
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"

        req = urllib.request.Request(api_url, headers=headers)
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                for asset in data["assets"]:
                    if asset["name"] == asset_name:
                        return asset["browser_download_url"]
        except Exception as e:
            print(f"Error fetching release info from GitHub: {e}", file=sys.stderr)
            return None
        return None

    def download_and_extract_lensfun(self):
        """Downloads and extracts the appropriate Lensfun library."""
        vendor_dir = Path("src/raw_alchemy/vendor/lensfun")
        vendor_dir.mkdir(parents=True, exist_ok=True)

        system = platform.system().lower()
        force_download = os.environ.get("RAW_ALCHEMY_FORCE_LENSFUN_DOWNLOAD") == "1"
        if not force_download and self.lensfun_vendor_complete(vendor_dir, system):
            print(f"Using existing vendored Lensfun for {system}.")
            return

        if system == "windows":
            asset_name = "lensfun-windows.zip"
        elif system == "linux":
            asset_name = "lensfun-linux.tar.gz"
        elif system == "darwin":
            asset_name = "lensfun-macos.tar.gz"
        else:
            print(f"Unsupported system: {system}", file=sys.stderr)
            sys.exit(1)

        download_url = self.get_download_url(asset_name)
        if not download_url:
            print(f"Could not find download URL for {asset_name}", file=sys.stderr)
            sys.exit(1)

        archive_path = asset_name
        try:
            print(f"Downloading Lensfun for {system} from {download_url}...")
            with urllib.request.urlopen(download_url) as response, open(
                archive_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)

            print("Extracting Lensfun...")
            shutil.unpack_archive(archive_path, vendor_dir)
            os.remove(archive_path)
            print("Lensfun setup complete.")

        except Exception as e:
            print(f"Error during Lensfun setup: {e}", file=sys.stderr)
            sys.exit(1)


ext_modules = []
if cc:
    ext = cc.distutils_extension()
    ext.name = "raw_alchemy.math_ops_ext"
    ext_modules.append(ext)

setup(
    cmdclass={
        "build_py": CustomBuildPy,
    },
    ext_modules=ext_modules,
)
