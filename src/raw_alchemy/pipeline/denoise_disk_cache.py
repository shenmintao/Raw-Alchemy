"""降噪结果磁盘缓存:全图降噪一次付费、跨会话免费。

键:RAW 路径 + 文件大小 + mtime + 降噪模型文件名(模型升级自动失效)。
存储:线性工作空间 float16 + zstd(≈150MB/42MP 张),LRU 按 atime 逐出。
配置:RAWALCHEMY_DENOISE_CACHE_DIR / RAWALCHEMY_DENOISE_CACHE_GB(默认 20)。
"""

import hashlib
import os
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
        blob = f.read_bytes()
        assert blob[:6] == _MAGIC
        h, w = int.from_bytes(blob[6:10], "little"), int.from_bytes(blob[10:14], "little")
        data = zstandard.ZstdDecompressor().decompress(blob[14:], max_output_size=h * w * 3 * 2)
        arr = np.frombuffer(data, np.float16).reshape(h, w, 3).astype(np.float32)
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
    try:
        t0 = time.time()
        h, w = denoised.shape[:2]
        payload = zstandard.ZstdCompressor(level=1).compress(
            np.ascontiguousarray(denoised, np.float32).astype(np.float16).tobytes())
        blob = _MAGIC + h.to_bytes(4, "little") + w.to_bytes(4, "little") + payload
        f = _cache_dir() / f"{k}.radc"
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(blob)
        tmp.replace(f)
        logger.info(f"[DenoiseCache] saved {os.path.basename(raw_path)} "
                    f"({len(blob) / 1e6:.0f}MB, {time.time() - t0:.2f}s)")
        _evict()
    except Exception as e:
        logger.warning(f"[DenoiseCache] save failed: {e}")


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
