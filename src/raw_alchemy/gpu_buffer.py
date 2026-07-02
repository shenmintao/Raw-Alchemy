"""
GPU Buffer Manager using Taichi ndarray.

ti.ndarray is GPU-resident: when passed to @ti.kernel via ti.types.ndarray(),
NO CPU copy is made. Data stays on GPU between kernel calls.

Only from_numpy() / to_numpy() cross the CPU-GPU boundary.

T7.2: all GpuImage allocations are backed by a process-wide, shape-keyed
ndarray pool so repeated full-frame allocations (700MB-class per request)
recycle device memory instead of churning vkAllocate/vkFree.
"""
import threading
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from raw_alchemy import config
from raw_alchemy.backend import ndarray, ti


def _dtype_itemsize(dtype) -> int:
    """Best-effort byte size of a taichi primitive dtype."""
    name = str(dtype)
    if name.endswith(("64",)):
        return 8
    if name.endswith(("16",)):
        return 2
    if name.endswith(("8",)):
        return 1
    return 4  # f32 / i32 / u32 and unknowns


def _nbytes(dtype, shape) -> int:
    total = _dtype_itemsize(dtype)
    for dim in shape:
        total *= int(dim)
    return total


class NdarrayPool:
    """Shape-keyed pool of free backend ndarrays (T7.2).

    acquire() pops a recycled buffer of the exact (dtype, shape) or allocates
    a fresh one; release() returns a buffer for reuse. Retained *free* bytes
    are capped (LRU eviction across keys) so the pool never hoards VRAM.
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
        return (str(dtype), tuple(int(dim) for dim in shape))

    def acquire(self, dtype, shape):
        """Return a device ndarray of exactly (dtype, shape); pooled if possible."""
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
    """The process-wide GPU buffer pool."""
    return _pool


def acquire_ndarray(dtype, shape):
    return _pool.acquire(dtype, shape)


def release_ndarray(arr, dtype, shape) -> bool:
    return _pool.release(arr, dtype, shape)


class GpuImage:
    """
    GPU-resident image buffer backed by ti.ndarray.

    Lifecycle:
        1. upload(np_array)  — CPU → GPU (once, on RAW load)
        2. Taichi kernels    — GPU → GPU (zero-copy, many times)
        3. to_numpy()        — GPU → CPU (only for display/export)

    Buffers come from the shape-keyed pool; clear() (and garbage collection
    of the GpuImage) returns them for reuse instead of freeing device memory.
    """

    def __init__(self, height: int = 0, width: int = 0, channels: int = 3):
        self._arr: Optional[ti.ndarray] = None
        self._height = height
        self._width = width
        self._channels = channels

        if height > 0 and width > 0:
            self._allocate(height, width, channels)

    def _allocate(self, height: int, width: int, channels: int = 3):
        """Acquire device memory from the pool. Re-acquires only if size changes."""
        if (self._arr is not None
                and self._height == height
                and self._width == width
                and self._channels == channels):
            return

        self._release_to_pool()
        self._arr = acquire_ndarray(ti.f32, (height, width, channels))
        self._height = height
        self._width = width
        self._channels = channels

    def _release_to_pool(self):
        if self._arr is not None:
            try:
                release_ndarray(self._arr, ti.f32, (self._height, self._width, self._channels))
            except Exception:
                pass
            self._arr = None

    @property
    def arr(self) -> ti.ndarray:
        """The underlying ti.ndarray. Pass this to @ti.kernel params."""
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
        """Device bytes held by this buffer (float32)."""
        if self._arr is None:
            return 0
        return self._height * self._width * self._channels * 4

    def upload(self, np_array: np.ndarray):
        """
        Upload numpy array to GPU. Re-acquires if shape changed.
        This is the ONLY place data crosses CPU → GPU.
        """
        if np_array.dtype != np.float32:
            np_array = np_array.astype(np.float32)
        if not np_array.flags['C_CONTIGUOUS']:
            np_array = np.ascontiguousarray(np_array)

        h, w = np_array.shape[:2]
        c = np_array.shape[2] if np_array.ndim == 3 else 1
        self._allocate(h, w, c)
        self._arr.from_numpy(np_array)

    def to_numpy(self) -> np.ndarray:
        """
        Download GPU data to numpy. This crosses GPU → CPU.
        Use sparingly (only for display/export).
        """
        if self._arr is None:
            return np.empty((0, 0, 3), dtype=np.float32)
        return self._arr.to_numpy()

    def copy_from(self, other: 'GpuImage'):
        """GPU-to-GPU copy from another GpuImage."""
        if not other.valid:
            return
        self._allocate(other._height, other._width, other._channels)
        self._arr.copy_from(other._arr)

    def clear(self):
        """Return the device buffer to the pool immediately.

        Only call from *live* code paths when this GpuImage is the sole owner
        of its buffer (no kernels or other objects still referencing ``arr``).

        Note there is deliberately no ``__del__``-based pooling: resurrecting
        the ndarray from inside a garbage cycle is unsafe — PEP 442 runs all
        finalizers of cyclic garbage first, so taichi's ``ScalarNdarray.__del__``
        would free the native buffer while the resurrected Python object sits
        in the pool with a dangling handle. Unreferenced buffers that never
        get an explicit clear() are simply freed by taichi on collection.
        """
        self._release_to_pool()
        self._height = 0
        self._width = 0
        self._channels = 0

    def size_mb(self) -> float:
        """Approximate GPU memory usage in MB."""
        if not self.valid:
            return 0.0
        return self.nbytes / (1024 * 1024)


class GpuBufferPool:
    """
    Pool of pre-allocated GPU buffers for the processing pipeline.
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
        """Total GPU memory used by all buffers."""
        total = 0.0
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, GpuImage):
                total += attr.size_mb()
        return total

    def clear_all(self):
        """Release all GPU buffers."""
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, GpuImage):
                attr.clear()
        logger.debug("[GpuBufferPool] All GPU buffers released.")
