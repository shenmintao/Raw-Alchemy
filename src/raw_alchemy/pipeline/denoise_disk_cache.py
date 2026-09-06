"""Lossless float32 stage cache shared by preview and full export.

RADC1 float16 entries are deliberately not promoted to export-quality data.
Keys include RAW content and a caller-supplied complete stage identity.
Storage is bounded by RAWALCHEMY_DENOISE_CACHE_GB (default 20 GB).
All filesystem failures are cache misses, never processing failures.
"""

import hashlib
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zstandard
from loguru import logger

from .resources import checkpoint
from .executor import PipelineAborted

_MAGIC = b"RADC2\n"


def _cache_dir() -> Path:
    d = os.environ.get("RAWALCHEMY_DENOISE_CACHE_DIR")
    p = Path(d) if d else Path.home() / ".rawalchemy" / "denoise_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _limit_bytes() -> int:
    try:
        gb = float(os.environ.get("RAWALCHEMY_DENOISE_CACHE_GB", "20"))
    except ValueError:
        gb = 20.0
    if not math.isfinite(gb):
        gb = 20.0
    return max(0, int(gb * (1 << 30)))


def _key(raw_path: str, model_tag: str, source_token=None) -> Optional[str]:
    if model_tag is None:
        return None
    try:
        from .stage_identity import file_digest
        source_digest = source_token if source_token is not None else file_digest(raw_path)
    except OSError:
        return None
    h = hashlib.sha256(
        f"RADC2|{os.path.abspath(raw_path)}|{source_digest}|{model_tag}"
        .encode("utf-8", "replace")
    ).hexdigest()
    return h


def load(raw_path: str, model_tag: str, *, source_token=None) -> Optional[np.ndarray]:
    """命中返回 (H, W, 3) float32 线性工作空间;未命中返回 None。"""
    k = _key(raw_path, model_tag, source_token)
    if k is None or _limit_bytes() == 0:
        return None
    f = None
    try:
        f = _cache_dir() / f"{k}.radc"
        if not f.exists():
            return None
        t0 = time.time()
        with f.open("rb") as source:
            header = source.read(14)
            if len(header) != 14 or header[:6] != _MAGIC:
                raise ValueError("invalid cache header")
            h = int.from_bytes(header[6:10], "little")
            w = int.from_bytes(header[10:14], "little")
            if h <= 0 or w <= 0 or h * w > 200_000_000:
                raise ValueError("invalid cached image dimensions")

            if h * w * 12 > _limit_bytes():
                return None

            # Decompress directly into the final export-quality array.
            packed = np.empty((h, w, 3), dtype="<f4")
            target = memoryview(packed).cast("B")
            offset = 0
            with zstandard.ZstdDecompressor().stream_reader(
                source, closefd=False
            ) as reader:
                while offset < target.nbytes:
                    checkpoint()
                    read = reader.readinto(target[offset:offset + 4 * 1024 * 1024])
                    if not read:
                        raise ValueError("truncated cache payload")
                    offset += read
                if reader.read(1):
                    raise ValueError("oversized cache payload")
            arr = packed.astype(np.float32, copy=False)
        try:
            os.utime(f)
        except OSError:
            pass
        logger.info(f"[DenoiseCache] hit {os.path.basename(raw_path)} "
                    f"({time.time() - t0:.2f}s load)")
        return arr
    except (PipelineAborted, MemoryError):
        raise
    except Exception as e:
        logger.warning(f"[DenoiseCache] corrupt entry dropped: {e}")
        try:
            if f is not None:
                f.unlink()
        except OSError:
            pass
        return None


def save(raw_path: str, model_tag: str, denoised: np.ndarray, *, source_token=None) -> None:
    k = _key(raw_path, model_tag, source_token)
    if k is None:
        return
    tmp = None
    try:
        t0 = time.time()
        if denoised.ndim != 3 or denoised.shape[-1] != 3:
            raise ValueError("expected HWC RGB cache data")
        h, w = denoised.shape[:2]
        if denoised.nbytes > _limit_bytes():
            return
        packed = np.ascontiguousarray(denoised, dtype="<f4")
        f = _cache_dir() / f"{k}.radc"
        tmp = f.with_name(
            f"{f.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        header = _MAGIC + h.to_bytes(4, "little") + w.to_bytes(4, "little")
        with tmp.open("wb") as output:
            output.write(header)
            # No full-frame bytes copy or precision reduction.
            with zstandard.ZstdCompressor(level=1).stream_writer(
                output, closefd=False
            ) as writer:
                data = memoryview(packed).cast("B")
                for offset in range(0, len(data), 4 * 1024 * 1024):
                    checkpoint()
                    writer.write(data[offset:offset + 4 * 1024 * 1024])
        # Publication uses the identity captured BEFORE computation. A file
        # replacement during compression cannot bind old pixels to the new RAW.
        if source_token is not None:
            from .stage_identity import source_identity
            if source_identity(raw_path) != source_token:
                tmp.unlink()
                return
        tmp.replace(f)
        size = f.stat().st_size
        logger.info(f"[DenoiseCache] saved {os.path.basename(raw_path)} "
                    f"({size / 1e6:.0f}MB, {time.time() - t0:.2f}s)")
        _evict()
    except (PipelineAborted, MemoryError):
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    except Exception as e:
        logger.warning(f"[DenoiseCache] save failed: {e}")
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def _evict() -> None:
    limit = _limit_bytes()
    d = _cache_dir()
    files = sorted(d.glob("*.radc"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    while total > limit and files:
        f = files.pop(0)
        try:
            total -= f.stat().st_size
            f.unlink()
            logger.info(f"[DenoiseCache] evicted {f.name}")
        except OSError:
            break
