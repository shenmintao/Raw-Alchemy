import collections
import threading
from typing import Any, Optional

import numpy as np
import psutil
from loguru import logger

from raw_alchemy import config


# Maximum number of final uint8 outputs kept per cached image (T7.4): the
# most recent output (primary slot) plus a few secondary slots so e.g. the
# fit view and the current zoom level each keep one — a fit<->100% round
# trip then costs zero recomputation.
OUTPUT_SLOT_LIMIT = 4


class CachedImage:
    """
    Container for cached image data.

    Size accounting is field-aware (T7.6): each array is attributed to a
    category so the manager can degrade entries field-by-field instead of
    evicting whole entries, and aliased arrays (e.g. ``corrected_data`` is
    the same object as ``linear_data`` when lens correction is a passthrough)
    are only counted once.

    Threading: entries are shared between two processor worker threads (the
    main and baseline processors share one manager) and are degraded in place
    by lock-held eviction (``set_limit_mb`` on the GUI thread). ``self.lock``
    guards the output slots, the cached-array fields and the size accounting.
    Lock order is ``manager.lock -> entry.lock`` (eviction and
    ``notify_entry_updated`` take the entry lock while holding the manager
    lock); entry methods must therefore never call back into the manager.
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
        # Entry-level lock (see class docstring). Reentrant because
        # drop_stage() re-accounts via update_size() while already holding it.
        self.lock = threading.RLock()

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

        # Final output cache (uint8 sRGB, ~25MB for 45MP). The most recent
        # output lives in these primary fields; older outputs for other
        # output keys (different zoom/viewport, T7.4) are kept in
        # ``_output_slots`` up to OUTPUT_SLOT_LIMIT total.
        self.output_uint8 = None
        self.output_key = None
        self.output_ev = 0.0
        self.output_source_size = None
        self._output_slots: collections.OrderedDict = collections.OrderedDict()

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

        with self.lock:
            self.field_sizes_mb = {
                'linear': unique_mb(self.linear_data),
                'proxy': unique_mb(self.proxy_linear) + unique_mb(self.proxy_corrected_data),
                'corrected': unique_mb(self.corrected_data),
                'denoise': (
                    unique_mb(self.denoise_full)
                    + unique_mb(self.denoise_original)
                    + unique_mb(self.sharpened_data)
                ),
                'output': unique_mb(self.output_uint8)
                + sum(unique_mb(img) for img, _ev, _size in self._output_slots.values()),
            }
            self.size_mb = sum(self.field_sizes_mb.values())
            return self.size_mb

    def get_output(self, key):
        """Look up a cached final output by its output key (T7.4).

        Returns ``(uint8_image, applied_ev, source_size)`` or None. Checks
        the primary (most recent) slot first, then the secondary slots;
        secondary hits are marked most-recently-used.

        Reads the primary fields exactly once, under the entry lock, and
        returns local references: a concurrent field-level eviction
        (settings-page cap change on another thread) can therefore never
        produce a torn ``(None, ev, size)`` hit or a key/image mismatch.
        """
        if key is None:
            return None
        with self.lock:
            img = self.output_uint8
            if self.output_key == key and img is not None:
                return (img, self.output_ev, self.output_source_size)
            slot = self._output_slots.get(key)
            if slot is not None:
                self._output_slots.move_to_end(key)
            return slot

    def store_output(self, key, img_uint8, applied_ev, source_size):
        """Store a final output, keeping up to OUTPUT_SLOT_LIMIT slots (T7.4).

        The new output becomes the primary slot; the previous primary (if it
        held a different key) is demoted to a secondary slot. Oldest slots
        are evicted LRU beyond the limit — so a fit view and the current
        zoom level can each keep their output alive at the same time.
        """
        with self.lock:
            if (
                self.output_uint8 is not None
                and self.output_key is not None
                and self.output_key != key
            ):
                self._output_slots[self.output_key] = (
                    self.output_uint8,
                    self.output_ev,
                    self.output_source_size,
                )
                self._output_slots.move_to_end(self.output_key)
            self._output_slots.pop(key, None)
            self.output_uint8 = img_uint8
            self.output_key = key
            self.output_ev = applied_ev
            self.output_source_size = source_size
            while len(self._output_slots) > OUTPUT_SLOT_LIMIT - 1:
                self._output_slots.popitem(last=False)

    def drop_stage(self, stage: str) -> float:
        """Drop one eviction stage's arrays (and their validity keys).

        Returns the entry-size delta in MB: usually the freed size, 0.0 when
        the stage held nothing (or only aliases of arrays still referenced by
        other fields), and possibly *negative* — the re-account runs over the
        whole entry, so it also picks up in-place growth a worker published
        but has not re-accounted via ``notify_entry_updated`` yet. Callers
        must apply the delta unconditionally to keep the running counter in
        sync with the entries.
        """
        with self.lock:
            if stage == "corrected":
                if self.corrected_data is None:
                    return 0.0
                self.corrected_data = None
                self.lens_key = None
            elif stage == "output":
                if self.output_uint8 is None and not self._output_slots:
                    return 0.0
                self.output_uint8 = None
                self.output_key = None
                self._output_slots.clear()
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
            # manager.lock -> entry.lock, the same order eviction uses.
            with item.lock:
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
                # Apply the delta unconditionally: drop_stage re-accounts the
                # whole entry, so a negative delta (unaccounted in-place
                # growth discovered before the worker's notify_entry_updated)
                # must also land in the counter, or it drifts from the
                # entries permanently.
                self.current_memory_mb -= freed
                if freed > 0.0:
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
