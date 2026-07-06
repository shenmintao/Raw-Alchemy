import os
from pathlib import Path


os.environ.setdefault(
    "RAW_ALCHEMY_LOG_DIR",
    str(Path.cwd() / ".test-output" / "logs"),
)
_tmp_root = Path.cwd() / ".test-output" / "tmp"
_tmp_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_tmp_root))
os.environ.setdefault("TEMP", str(_tmp_root))
os.environ.setdefault("TMPDIR", str(_tmp_root))


