import collections
import threading
from typing import Optional, Any
import numpy as np
import psutil
from loguru import logger

class CachedImage:
    """
    Container for cached image data
    """
    def __init__(self, path: str, linear_data: np.ndarray, exif_data: Any, lens_key: Any, corrected_data: Optional[np.ndarray] = None, exif_metadata: Any = None):
        self.path = path
        self.linear_data = linear_data
        self.exif_data = exif_data
        self.exif_metadata = exif_metadata
        self.lens_key = lens_key
        self.corrected_data = corrected_data

        # Denoise Cache
        self.denoise_full = None
        self.denoise_original = None
        self.denoise_key = None

        # Sharpen Cache
        self.sharpened_data = None
        self.sharpen_key = None

        # Final output cache (uint8 sRGB, ~25MB for 45MP)
        self.output_uint8 = None
        self.output_key = None
        self.output_ev = 0.0
        self.output_source_size = None

        # Calculate approximate size in MB
        self.size_mb = linear_data.nbytes / (1024 * 1024)
        if corrected_data is not None:
             self.size_mb += corrected_data.nbytes / (1024 * 1024)

    def update_size(self):
        self.size_mb = self.linear_data.nbytes / (1024 * 1024)
        if self.corrected_data is not None:
            self.size_mb += self.corrected_data.nbytes / (1024 * 1024)
        if self.output_uint8 is not None:
            self.size_mb += self.output_uint8.nbytes / (1024 * 1024)

class ImageCacheManager:
    """
    Thread-safe LRU Cache Manager with dynamic memory quota.
    Uses 70% of available system memory. Eviction triggers at 80% of quota.
    """
    def __init__(self, memory_fraction: float = 0.7):
        self.memory_fraction = memory_fraction
        self.cache = collections.OrderedDict()
        self.lock = threading.Lock()
        self.current_memory_mb = 0.0

    def _get_quota_mb(self) -> float:
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        return (avail_mb + self.current_memory_mb) * self.memory_fraction

    def get(self, path: str) -> Optional[CachedImage]:
        with self.lock:
            if path in self.cache:
                self.cache.move_to_end(path)
                return self.cache[path]
            return None

    def put(self, path: str, item: CachedImage):
        with self.lock:
            if path in self.cache:
                old_item = self.cache.pop(path)
                self.current_memory_mb -= old_item.size_mb

            self.cache[path] = item
            self.current_memory_mb += item.size_mb

            self._evict_if_needed()

            quota = self._get_quota_mb()
            logger.debug(f"[Cache] Added {path}. Items: {len(self.cache)}, "
                         f"Mem: {self.current_memory_mb:.0f}MB / {quota:.0f}MB")

    def _evict_if_needed(self):
        quota = self._get_quota_mb()
        while self.current_memory_mb > quota * 0.8 and len(self.cache) > 1:
            path, item = self.cache.popitem(last=False)
            self.current_memory_mb -= item.size_mb
            logger.debug(f"[Cache] Evicted {path}. Mem: {self.current_memory_mb:.0f}MB")

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_memory_mb = 0.0
