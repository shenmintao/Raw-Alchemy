import collections
import threading
from typing import Optional, Any
import numpy as np
from loguru import logger

class CachedImage:
    """
    Container for cached image data
    """
    def __init__(self, path: str, linear_data: np.ndarray, exif_data: Any, lens_key: Any, corrected_data: Optional[np.ndarray] = None):
        self.path = path
        self.linear_data = linear_data
        self.exif_data = exif_data
        self.lens_key = lens_key
        self.corrected_data = corrected_data
        
        # Denoise Cache
        self.denoise_full = None
        self.denoise_original = None
        self.denoise_key = None
        
        # Sharpen Cache
        self.sharpened_data = None
        self.sharpen_key = None
        
        # Calculate approximate size in MB
        self.size_mb = linear_data.nbytes / (1024 * 1024)
        if corrected_data is not None:
             self.size_mb += corrected_data.nbytes / (1024 * 1024)

class ImageCacheManager:
    """
    Thread-safe LRU Cache Manager for processed images.
    """
    def __init__(self, max_items: int = 5, max_memory_mb: int = 2048):
        self.max_items = max_items
        self.max_memory_mb = max_memory_mb
        self.cache = collections.OrderedDict()
        self.lock = threading.Lock()
        self.current_memory_mb = 0.0

    def get(self, path: str) -> Optional[CachedImage]:
        with self.lock:
            if path in self.cache:
                # Move to end (mark as recently used)
                self.cache.move_to_end(path)
                return self.cache[path]
            return None

    def put(self, path: str, item: CachedImage):
        with self.lock:
            if path in self.cache:
                # Update existing item
                old_item = self.cache.pop(path)
                self.current_memory_mb -= old_item.size_mb
            
            # Add new item
            self.cache[path] = item
            self.current_memory_mb += item.size_mb
            
            # Enforce constraints
            self._evict_if_needed()
            
            logger.debug(f"[Cache] Added {path}. Items: {len(self.cache)}, Mem: {self.current_memory_mb:.1f}MB")

    def _evict_if_needed(self):
        # 1. Check item count
        while len(self.cache) > self.max_items:
            path, item = self.cache.popitem(last=False) # Pop first (LRU)
            self.current_memory_mb -= item.size_mb
            logger.debug(f"[Cache] Evicted (Count Update) {path}. Mem: {self.current_memory_mb:.1f}MB")

        # 2. Check memory usage
        while self.current_memory_mb > self.max_memory_mb and len(self.cache) > 0:
             # Always keep at least 1 item (the current one usually)
            if len(self.cache) <= 1:
                break
                
            path, item = self.cache.popitem(last=False)
            self.current_memory_mb -= item.size_mb
            logger.debug(f"[Cache] Evicted (Memory Limit) {path}. Mem: {self.current_memory_mb:.1f}MB")

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_memory_mb = 0.0
