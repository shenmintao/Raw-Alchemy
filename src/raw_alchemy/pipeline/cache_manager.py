import collections
import threading
from typing import Any, Optional

import numpy as np
import psutil
from loguru import logger

from raw_alchemy import config


class CachedImage:
    """
    Container for cached image data.

    Size accounting is field-aware (T7.6): each array is attributed to a
    category so the manager can degrade entries field-by-field instead of
    evicting whole entries, and aliased arrays (e.g. ``corrected_data`` is
    the same object as ``linear_data`` when lens correction is a passthrough)
    are only counted once.
    """

    # Field-level eviction stages, applied in order before a whole entry is
    # dropped: full-size corrected first (largest, recomputable), then the
    # final uint8 output, then denoise buffers. linear + proxy go last, with
    # the whole entry.
    EVICTION_STAGES = ("corrected", "output", "denoise")

    def __init__(
        self,
        path: str,
        linear_data: np.ndarray,
        exif_data: Any,
        lens_key: Any,
        corrected_data: Optional[np.ndarray] = None,
        exif_metadata: Any = None,
        proxy_linear: Optional[np.ndarray] = None,
        proxy_corrected_data: Optional[np.ndarray] = None,
        proxy_lens_key: Any = None,
    ):
        self.path = path
        self.linear_data = linear_data
        self.proxy_linear = proxy_linear
        self.exif_data = exif_data
        self.exif_metadata = exif_metadata
        self.lens_key = lens_key
        self.proxy_lens_key = proxy_lens_key
        self.corrected_data = corrected_data
        self.proxy_corrected_data = proxy_corrected_data

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

        # Approximate size in MB, per category and total.
        self.field_sizes_mb: dict[str, float] = {}
        self.size_mb = 0.0
        self.update_size()

    def update_size(self) -> float:
        """Recompute per-field and total sizes in MB.

        Arrays shared between fields (aliases) are charged only to the first
        category that sees them, so e.g. a passthrough ``corrected_data`` that
        is the same object as ``linear_data`` is not double-billed.
        """
        seen: set = set()

        def unique_mb(arr) -> float:
            if arr is None:
                return 0.0
            key = id(arr)
            if key in seen:
                return 0.0
            seen.add(key)
            return arr.nbytes / (1024 * 1024)

        self.field_sizes_mb = {
            'linear': unique_mb(self.linear_data),
            'proxy': unique_mb(self.proxy_linear) + unique_mb(self.proxy_corrected_data),
            'corrected': unique_mb(self.corrected_data),
            'denoise': (
                unique_mb(self.denoise_full)
                + unique_mb(self.denoise_original)
                + unique_mb(self.sharpened_data)
            ),
            'output': unique_mb(self.output_uint8),
        }
        self.size_mb = sum(self.field_sizes_mb.values())
        return self.size_mb

    def drop_stage(self, stage: str) -> float:
        """Drop one eviction stage's arrays (and their validity keys).

        Returns the freed size in MB (0.0 when the stage held nothing, or
        only aliases of arrays still referenced by other fields).
        """
        if stage == "corrected":
            if self.corrected_data is None:
                return 0.0
            self.corrected_data = None
            self.lens_key = None
        elif stage == "output":
            if self.output_uint8 is None:
                return 0.0
            self.output_uint8 = None
            self.output_key = None
        elif stage == "denoise":
            if (
                self.denoise_full is None
                and self.denoise_original is None
                and self.sharpened_data is None
            ):
                return 0.0
            self.denoise_full = None
            self.denoise_original = None
            self.denoise_key = None
            self.sharpened_data = None
            self.sharpen_key = None
        else:
            raise ValueError(f"Unknown eviction stage: {stage}")

        before = self.size_mb
        self.update_size()
        return before - self.size_mb


class ImageCacheManager:
    """
    Thread-safe LRU cache manager with an absolute memory cap (T7.6).

    Quota = min((available + current) * memory_fraction, limit_mb); eviction
    triggers as soon as usage exceeds the quota (no 80% slack). Entries are
    degraded field-by-field in LRU order (corrected -> output -> denoise)
    before whole entries (linear + proxy) are dropped.
    """

    def __init__(
        self,
        memory_fraction: Optional[float] = None,
        limit_mb: Optional[float] = None,
    ):
        self.memory_fraction = (
            config.CACHE_MEMORY_FRACTION if memory_fraction is None else memory_fraction
        )
        self.limit_mb = float(config.CACHE_LIMIT_MB if limit_mb is None else limit_mb)
        self.cache = collections.OrderedDict()
        self.lock = threading.Lock()
        self.current_memory_mb = 0.0

    def set_limit_mb(self, limit_mb: float):
        """Change the absolute cap (settings page) and re-apply eviction."""
        with self.lock:
            self.limit_mb = float(limit_mb)
            self._evict_if_needed()

    def _get_quota_mb(self) -> float:
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        relative = (avail_mb + self.current_memory_mb) * self.memory_fraction
        return min(relative, self.limit_mb)

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

            logger.debug(f"[Cache] Added {path}. Items: {len(self.cache)}, "
                         f"Mem: {self.current_memory_mb:.0f}MB / {self._get_quota_mb():.0f}MB")

    def notify_entry_updated(self, path: str):
        """Re-account an entry whose arrays were changed in place.

        Replaces the ad-hoc ``current_memory_mb`` adjustments callers used to
        do: recomputes the entry size under the lock, marks it most-recently
        used, and applies quota eviction (in-place growth such as a lens
        correction write-back can push the cache over quota).
        """
        with self.lock:
            item = self.cache.get(path)
            if item is None:
                return
            old_size = item.size_mb
            item.update_size()
            self.current_memory_mb += item.size_mb - old_size
            self.cache.move_to_end(path)
            self._evict_if_needed()

    def stats(self) -> dict:
        """Per-category memory usage in MB across all entries (observability)."""
        with self.lock:
            return self._stats_locked()

    def _stats_locked(self) -> dict:
        totals: dict = {'items': len(self.cache), 'total': self.current_memory_mb}
        for item in self.cache.values():
            for field, mb in item.field_sizes_mb.items():
                totals[field] = totals.get(field, 0.0) + mb
        return totals

    def _log_stats_locked(self, reason: str, quota_mb: float):
        s = self._stats_locked()
        fields = " ".join(
            f"{name}={s.get(name, 0.0):.0f}MB"
            for name in ('linear', 'proxy', 'corrected', 'denoise', 'output')
        )
        logger.info(
            f"[Cache] {reason}: items={s['items']} total={s['total']:.0f}MB "
            f"quota={quota_mb:.0f}MB limit={self.limit_mb:.0f}MB | {fields}"
        )

    def _evict_if_needed(self):
        quota = self._get_quota_mb()
        if self.current_memory_mb <= quota:
            return

        self._log_stats_locked("Over quota, evicting", quota)

        # Stage 1: field-level degradation in LRU order.
        dropped_fields = 0
        for stage in CachedImage.EVICTION_STAGES:
            for path, item in list(self.cache.items()):
                if self.current_memory_mb <= quota:
                    break
                freed = item.drop_stage(stage)
                if freed > 0.0:
                    self.current_memory_mb -= freed
                    dropped_fields += 1
                    logger.debug(
                        f"[Cache] Evicted field '{stage}' of {path} "
                        f"(-{freed:.0f}MB, mem={self.current_memory_mb:.0f}MB)"
                    )
            if self.current_memory_mb <= quota:
                break

        # Stage 2: whole-entry LRU eviction (linear + proxy last); always
        # keep the most recent entry, even when it alone exceeds the quota.
        dropped_entries = 0
        while self.current_memory_mb > quota and len(self.cache) > 1:
            path, item = self.cache.popitem(last=False)
            self.current_memory_mb -= item.size_mb
            dropped_entries += 1
            logger.debug(
                f"[Cache] Evicted entry {path} "
                f"(-{item.size_mb:.0f}MB, mem={self.current_memory_mb:.0f}MB)"
            )

        self._log_stats_locked(
            f"Eviction done (fields={dropped_fields}, entries={dropped_entries})",
            quota,
        )
        if self.current_memory_mb > quota:
            logger.warning(
                f"[Cache] Still over quota after eviction: "
                f"{self.current_memory_mb:.0f}MB > {quota:.0f}MB "
                f"(single newest entry is kept regardless)"
            )

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_memory_mb = 0.0
