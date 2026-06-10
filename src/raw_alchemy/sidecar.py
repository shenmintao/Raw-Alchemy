"""Read and write per-image Raw Alchemy sidecar files."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from raw_alchemy import config


SIDECAR_VERSION = 1
SIDECAR_SUFFIX = ".rwa.json"


@dataclass(frozen=True)
class SidecarData:
    params: dict[str, Any]
    marked: bool = False
    version: int = SIDECAR_VERSION


def sidecar_path(raw_path: str | os.PathLike[str]) -> Path:
    """Return the sidecar path for a RAW file."""
    return Path(f"{Path(raw_path)}{SIDECAR_SUFFIX}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def jsonable_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Convert UI params to JSON-native values."""
    if not params:
        return {}
    converted = _jsonable(params)
    return converted if isinstance(converted, dict) else {}


def write_sidecar(
    raw_path: str | os.PathLike[str],
    params: dict[str, Any] | None,
    marked: bool = False,
) -> Path:
    """Write a versioned sidecar file next to the RAW file."""
    target = sidecar_path(raw_path)
    payload = {
        "version": SIDECAR_VERSION,
        "params": jsonable_params(params),
        "marked": bool(marked),
    }

    tmp_path = target.with_name(f"{target.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_path, target)
    return target


def read_sidecar(raw_path: str | os.PathLike[str]) -> SidecarData | None:
    """Read a sidecar file, returning None for missing or invalid data."""
    path = sidecar_path(raw_path)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        logger.warning(f"Failed to read sidecar {path}: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    version = payload.get("version")
    params = payload.get("params")
    if version != SIDECAR_VERSION or not isinstance(params, dict):
        return None

    return SidecarData(
        params=jsonable_params(params),
        marked=bool(payload.get("marked", False)),
        version=version,
    )


def load_folder_sidecars(folder: str | os.PathLike[str]) -> dict[str, SidecarData]:
    """Load valid sidecars for supported RAW files in a folder."""
    loaded: dict[str, SidecarData] = {}

    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                if Path(entry.name).suffix.lower() not in config.SUPPORTED_RAW_EXTENSIONS:
                    continue
                data = read_sidecar(entry.path)
                if data is not None:
                    loaded[entry.path] = data
    except Exception as exc:
        logger.warning(f"Failed to scan sidecars in {folder}: {exc}")

    return loaded
