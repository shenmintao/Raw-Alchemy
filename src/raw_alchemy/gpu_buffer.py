"""
GPU Buffer Manager using Taichi ndarray.

ti.ndarray is GPU-resident: when passed to @ti.kernel via ti.types.ndarray(),
NO CPU copy is made. Data stays on GPU between kernel calls.

Only from_numpy() / to_numpy() cross the CPU-GPU boundary.
"""
import taichi as ti
import numpy as np
from loguru import logger
from typing import Optional, Tuple


class GpuImage:
    """
    GPU-resident image buffer backed by ti.ndarray.

    Lifecycle:
        1. upload(np_array)  — CPU → GPU (once, on RAW load)
        2. Taichi kernels    — GPU → GPU (zero-copy, many times)
        3. to_numpy()        — GPU → CPU (only for display/export)
    """

    def __init__(self, height: int = 0, width: int = 0, channels: int = 3):
        self._arr: Optional[ti.ndarray] = None
        self._height = height
        self._width = width
        self._channels = channels

        if height > 0 and width > 0:
            self._allocate(height, width, channels)

    def _allocate(self, height: int, width: int, channels: int = 3):
        """Allocate GPU memory. Re-allocates only if size changes."""
        if (self._arr is not None
                and self._height == height
                and self._width == width
                and self._channels == channels):
            return

        self._arr = ti.ndarray(dtype=ti.f32, shape=(height, width, channels))
        self._height = height
        self._width = width
        self._channels = channels

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

    def upload(self, np_array: np.ndarray):
        """
        Upload numpy array to GPU. Re-allocates if shape changed.
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
        """Release GPU memory immediately."""
        if self._arr is not None:
            try:
                del self._arr
            except Exception:
                pass
            self._arr = None
        self._height = 0
        self._width = 0
        self._channels = 0

    def size_mb(self) -> float:
        """Approximate GPU memory usage in MB."""
        if not self.valid:
            return 0.0
        return self._height * self._width * self._channels * 4 / (1024 * 1024)


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
