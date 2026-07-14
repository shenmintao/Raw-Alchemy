"""
Buffer manager for the pipeline — numpy-backed since the Taichi runtime was
removed from the interactive pipeline.

GpuImage / NdarrayPool / GpuBufferPool keep their historical names and exact
interfaces (upload / to_numpy / copy_from / acquire / release / trim / byte
accounting), but the buffers are now plain host (RAM) numpy arrays. The
shape-keyed pool still recycles same-(dtype, shape) buffers so full-frame
allocations are reused instead of churning the allocator, and the byte
budget/trim semantics (T7.2/T7.6) are unchanged — the accounted bytes are
now RAM instead of VRAM.

Buffers returned by the pool are ``HostNdarray`` (a numpy subclass) carrying
the minimal ti.ndarray-compatible surface (``from_numpy`` / ``to_numpy`` /
``copy_from`` / ``fill``) so the legacy Taichi demosaic modules keep working
against pool buffers unchanged.
"""
import threading
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from raw_alchemy import config


# Taichi primitive dtype names (str(ti.f32) == "f32", etc.) — accepted for
# compatibility with legacy callers; normalized to numpy dtypes.
_TI_DTYPE_NAMES = {
    "f16": np.float16, "f32": np.float32, "f64": np.float64,
    "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64,
}


def _as_numpy_dtype(dtype) -> np.dtype:
    """Normalize numpy dtypes and taichi primitive dtypes to np.dtype."""
    mapped = _TI_DTYPE_NAMES.get(str(dtype))
    if mapped is not None:
        return np.dtype(mapped)
    try:
        return np.dtype(dtype)
    except TypeError:
        raise TypeError(f"Unsupported buffer dtype: {dtype!r}")


def _nbytes(dtype, shape) -> int:
    total = _as_numpy_dtype(dtype).itemsize
    for dim in shape:
        total *= int(dim)
    return total


class HostNdarray(np.ndarray):
    """numpy array with the small ti.ndarray-compatible method surface."""

    def from_numpy(self, src: np.ndarray):
        np.copyto(self, src)

    def to_numpy(self) -> np.ndarray:
        """Return an independent plain-ndarray copy of the buffer contents."""
        return np.array(self, copy=True)

    def copy_from(self, other: np.ndarray):
        np.copyto(self, other)


def ndarray(*, dtype, shape) -> HostNdarray:
    """Allocate a pool-compatible host buffer of (dtype, shape)."""
    np_dtype = _as_numpy_dtype(dtype)
    return np.empty(tuple(int(dim) for dim in shape), dtype=np_dtype).view(HostNdarray)


class NdarrayPool:
    """Shape-keyed pool of free host ndarrays (T7.2).

    acquire() pops a recycled buffer of the exact (dtype, shape) or allocates
    a fresh one; release() returns a buffer for reuse. Retained *free* bytes
    are capped (LRU eviction across keys) so the pool never hoards RAM.
    Thread-safe: the worker, export and preload threads share one instance.
    """

    def __init__(self, max_bytes: Optional[int] = None, max_per_key: Optional[int] = None):
        self._lock = threading.Lock()
        # key -> list of free ndarrays; OrderedDict gives LRU across keys.
        self._free: "OrderedDict[tuple, list]" = OrderedDict()
        self._free_bytes = 0
        self.max_bytes = (
            int(config.GPU_POOL_LIMIT_MB) * 1024 * 1024 if max_bytes is None else int(max_bytes)
        )
        self.max_per_key = (
            int(config.GPU_POOL_MAX_PER_KEY) if max_per_key is None else int(max_per_key)
        )

    @staticmethod
    def _key(dtype, shape) -> tuple:
        return (_as_numpy_dtype(dtype).name, tuple(int(dim) for dim in shape))

    def acquire(self, dtype, shape):
        """Return an ndarray of exactly (dtype, shape); pooled if possible."""
        key = self._key(dtype, shape)
        with self._lock:
            bucket = self._free.get(key)
            if bucket:
                arr = bucket.pop()
                if not bucket:
                    self._free.pop(key, None)
                else:
                    self._free.move_to_end(key)
                self._free_bytes -= _nbytes(dtype, shape)
                return arr
        return ndarray(dtype=dtype, shape=tuple(shape))

    def release(self, arr, dtype, shape) -> bool:
        """Return a buffer to the pool. The caller must be the sole owner.

        Returns True when the buffer was retained, False when it was dropped
        (budget/per-key cap exceeded) and left to the garbage collector.
        """
        if arr is None:
            return False
        size = _nbytes(dtype, shape)
        if size <= 0 or size > self.max_bytes:
            return False
        key = self._key(dtype, shape)
        with self._lock:
            bucket = self._free.get(key)
            if bucket is not None and len(bucket) >= self.max_per_key:
                return False
            if bucket is None:
                bucket = []
                self._free[key] = bucket
            bucket.append(arr)
            self._free.move_to_end(key)
            self._free_bytes += size
            self._evict_locked(self.max_bytes)
        return True

    def _evict_locked(self, budget: int):
        while self._free_bytes > budget and self._free:
            key, bucket = next(iter(self._free.items()))
            bucket.pop(0)
            self._free_bytes -= _nbytes_from_key(key)
            if not bucket:
                self._free.pop(key, None)

    def free_bytes(self) -> int:
        with self._lock:
            return self._free_bytes

    def trim(self, budget_bytes: int = 0) -> int:
        """Drop free buffers until retained bytes <= budget. Returns bytes freed."""
        with self._lock:
            before = self._free_bytes
            self._evict_locked(max(0, int(budget_bytes)))
            return before - self._free_bytes

    def clear(self) -> int:
        """Drop every free buffer (image switch / shutdown)."""
        return self.trim(0)


def _nbytes_from_key(key: tuple) -> int:
    name, shape = key
    return _nbytes(name, shape)


_pool = NdarrayPool()


def gpu_pool() -> NdarrayPool:
    """The process-wide buffer pool."""
    return _pool


def acquire_ndarray(dtype, shape):
    return _pool.acquire(dtype, shape)


def release_ndarray(arr, dtype, shape) -> bool:
    return _pool.release(arr, dtype, shape)


class GpuImage:
    """
    Image buffer backed by a pooled numpy array (float32, HxWxC).

    The historical GPU lifecycle names are preserved:
        1. upload(np_array)  — copies the source into the pooled buffer
        2. pixel ops         — mutate ``arr`` in place (numpy/cv2)
        3. to_numpy()        — independent host copy (display/export)

    Buffers come from the shape-keyed pool; clear() (and garbage collection
    of the GpuImage) returns them for reuse instead of freeing memory.
    """

    def __init__(self, height: int = 0, width: int = 0, channels: int = 3):
        self._arr: Optional[HostNdarray] = None
        self._height = height
        self._width = width
        self._channels = channels

        if height > 0 and width > 0:
            self._allocate(height, width, channels)

    def _allocate(self, height: int, width: int, channels: int = 3):
        """Acquire memory from the pool. Re-acquires only if size changes."""
        if (self._arr is not None
                and self._height == height
                and self._width == width
                and self._channels == channels):
            return

        self._release_to_pool()
        self._arr = acquire_ndarray(np.float32, (height, width, channels))
        self._height = height
        self._width = width
        self._channels = channels

    def _release_to_pool(self):
        if self._arr is not None:
            try:
                release_ndarray(self._arr, np.float32, (self._height, self._width, self._channels))
            except Exception:
                pass
            self._arr = None

    @property
    def arr(self) -> np.ndarray:
        """The underlying array. Mutated in place by the pixel ops."""
        return self._arr

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._height, self._width, self._channels)

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def valid(self) -> bool:
        return self._arr is not None and self._height > 0 and self._width > 0

    @property
    def nbytes(self) -> int:
        """Bytes held by this buffer (float32)."""
        if self._arr is None:
            return 0
        return self._height * self._width * self._channels * 4

    def upload(self, np_array: np.ndarray):
        """Copy a numpy array into the pooled buffer. Re-acquires if shape changed."""
        if np_array.dtype != np.float32:
            np_array = np_array.astype(np.float32)

        h, w = np_array.shape[:2]
        c = np_array.shape[2] if np_array.ndim == 3 else 1
        self._allocate(h, w, c)
        np.copyto(self._arr.reshape(np_array.shape), np_array)

    def adopt(self, np_array: np.ndarray):
        """Take ownership of a newly-created float32 image without copying it.

        This is for operation outputs whose producer has already allocated an
        independent, contiguous ndarray (for example ONNX Runtime).  Callers
        must not pass cache-owned/source arrays because subsequent pipeline
        operations mutate the adopted storage in place.
        """
        arr = np.asarray(np_array)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        if not arr.flags.writeable:
            arr = np.array(arr, copy=True, order="C")
        if arr.ndim not in (2, 3):
            raise ValueError(f"expected HxW or HxWxC image, got shape {arr.shape}")

        if arr is self._arr:
            return
        if self._arr is not None and np.shares_memory(arr, self._arr):
            # Never return storage to the pool while adopting an alias of it.
            arr = np.array(arr, copy=True, order="C")

        h, w = arr.shape[:2]
        c = arr.shape[2] if arr.ndim == 3 else 1
        self._release_to_pool()
        self._arr = arr if isinstance(arr, HostNdarray) else arr.view(HostNdarray)
        self._height = h
        self._width = w
        self._channels = c

    def to_numpy(self) -> np.ndarray:
        """Independent copy of the buffer contents.

        A copy (not a view) so callers keep a stable result even after this
        buffer is cleared back to the pool and reused — same semantics as the
        old device download.
        """
        if self._arr is None:
            return np.empty((0, 0, 3), dtype=np.float32)
        return self._arr.to_numpy()

    def copy_from(self, other: 'GpuImage'):
        """Buffer-to-buffer copy from another GpuImage."""
        if not other.valid:
            return
        self._allocate(other._height, other._width, other._channels)
        np.copyto(self._arr, other._arr)

    def clear(self):
        """Return the buffer to the pool immediately.

        Only call from *live* code paths when this GpuImage is the sole owner
        of its buffer (no other objects still referencing ``arr``). There is
        deliberately no ``__del__``-based pooling: unreferenced buffers that
        never get an explicit clear() are simply freed by the garbage
        collector.
        """
        self._release_to_pool()
        self._height = 0
        self._width = 0
        self._channels = 0

    def size_mb(self) -> float:
        """Approximate memory usage in MB."""
        if not self.valid:
            return 0.0
        return self.nbytes / (1024 * 1024)


class GpuBufferPool:
    """
    Pool of pre-allocated buffers for the processing pipeline.
    Avoids repeated allocation/deallocation.
    """

    def __init__(self):
        # Pipeline stage buffers
        self.linear = GpuImage()       # RAW decoded (ProPhoto Linear)
        self.corrected = GpuImage()    # After lens correction
        self.geometry = GpuImage()     # After rotation/flip
        self.perspective = GpuImage()  # After perspective correction
        self.cropped = GpuImage()      # After crop
        self.exposed = GpuImage()      # After exposure
        self.adjusted = GpuImage()     # After WB/HS/Sat/Con
        self.graded = GpuImage()       # After Log/LUT/sRGB

        # Viewport output (smaller, for display)
        self.viewport = GpuImage()

    def total_gpu_mb(self) -> float:
        """Total memory used by all buffers."""
        total = 0.0
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, GpuImage):
                total += attr.size_mb()
        return total

    def clear_all(self):
        """Release all buffers."""
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, GpuImage):
                attr.clear()
        logger.debug("[GpuBufferPool] All buffers released.")
