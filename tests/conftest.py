import os
from pathlib import Path


os.environ.setdefault(
    "RAW_ALCHEMY_LOG_DIR",
    str(Path.cwd() / ".test-output" / "logs"),
)
os.environ.setdefault("TI_OFFLINE_CACHE", "0")

_tmp_root = Path.cwd() / ".test-output" / "tmp"
_tmp_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_tmp_root))
os.environ.setdefault("TEMP", str(_tmp_root))
os.environ.setdefault("TMPDIR", str(_tmp_root))

import taichi as ti

from raw_alchemy.math_ops import init_taichi


def pytest_sessionstart(session):
    init_taichi(arch=ti.cpu)
