"""降噪结果磁盘缓存:全图降噪一次付费、跨会话免费。

键:RAW 路径 + 文件大小 + mtime + 降噪模型文件名(模型升级自动失效)。
存储:线性工作空间 float16 + zstd(≈150MB/42MP 张),LRU 按 atime 逐出。
配置:RAWALCHEMY_DENOISE_CACHE_DIR / RAWALCHEMY_DENOISE_CACHE_GB(默认 20)。
"""

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zstandard
from loguru import logger

_MAGIC = b"RADC1\n"


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
    return int(gb * (1 << 30))


def _key(raw_path: str, model_tag: str) -> Optional[str]:
    try:
        st = os.stat(raw_path)
    except OSError:
        return None
    h = hashlib.sha1(
        f"{os.path.abspath(raw_path)}|{st.st_size}|{int(st.st_mtime)}|{model_tag}"
        .encode("utf-8", "replace")
    ).hexdigest()
    return h


def load(raw_path: str, model_tag: str) -> Optional[np.ndarray]:
    """命中返回 (H, W, 3) float32 线性工作空间;未命中返回 None。"""
    k = _key(raw_path, model_tag)
    if k is None:
        return None
    f = _cache_dir() / f"{k}.radc"
    if not f.exists():
        return None
    try:
        t0 = time.time()
        with f.open("rb") as source:
            header = source.read(14)
            if len(header) != 14 or header[:6] != _MAGIC:
                raise ValueError("invalid cache header")
            h = int.from_bytes(header[6:10], "little")
            w = int.from_bytes(header[10:14], "little")
            if h <= 0 or w <= 0:
                raise ValueError("invalid cached image dimensions")

            # Stream into the final float16 storage instead of materializing
            # the compressed file and decompressed payload as two Python bytes
            # objects. Peak load memory is now float16 + float32 output.
            packed = np.empty((h, w, 3), dtype=np.float16)
            target = memoryview(packed).cast("B")
            offset = 0
            with zstandard.ZstdDecompressor().stream_reader(
                source, closefd=False
            ) as reader:
                while offset < target.nbytes:
                    read = reader.readinto(target[offset:])
                    if not read:
                        raise ValueError("truncated cache payload")
                    offset += read
            arr = packed.astype(np.float32)
        os.utime(f)  # LRU: 命中刷新 atime/mtime
        logger.info(f"[DenoiseCache] hit {os.path.basename(raw_path)} "
                    f"({time.time() - t0:.2f}s load)")
        return arr
    except Exception as e:
        logger.warning(f"[DenoiseCache] corrupt entry dropped: {e}")
        try:
            f.unlink()
        except OSError:
            pass
        return None


def save(raw_path: str, model_tag: str, denoised: np.ndarray) -> None:
    k = _key(raw_path, model_tag)
    if k is None:
        return
    tmp = None
    try:
        t0 = time.time()
        h, w = denoised.shape[:2]
        packed = np.ascontiguousarray(denoised, np.float32).astype(np.float16)
        f = _cache_dir() / f"{k}.radc"
        tmp = f.with_name(
            f"{f.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        header = _MAGIC + h.to_bytes(4, "little") + w.to_bytes(4, "little")
        with tmp.open("wb") as output:
            output.write(header)
            # Stream compressed bytes directly to disk. This avoids both the
            # full float16 .tobytes() copy and a second header+payload blob.
            with zstandard.ZstdCompressor(level=1).stream_writer(
                output, closefd=False
            ) as writer:
                writer.write(memoryview(packed).cast("B"))
        tmp.replace(f)
        size = f.stat().st_size
        logger.info(f"[DenoiseCache] saved {os.path.basename(raw_path)} "
                    f"({size / 1e6:.0f}MB, {time.time() - t0:.2f}s)")
        _evict()
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
