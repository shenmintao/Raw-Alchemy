"""Persistent thumbnail cache (T7.8).

Two layers, both keyed by a hash of (absolute path, mtime_ns, size, format
version), so a modified source file simply misses and is re-extracted:

- Session memory layer: encoded JPEG bytes kept in a small LRU dict, so
  re-entering a folder within the same session redisplays instantly without
  touching the disk.
- Disk layer: ~300px JPEG (quality 85) files under
  ``<QStandardPaths.CacheLocation>/raw_alchemy/thumbs/``. A hit costs one
  small file read (~5ms) instead of a full RAW decode.

Robustness: corrupt/unreadable entries are deleted and the caller falls back
to re-extraction (which overwrites the entry); all I/O failures degrade to a
cache miss, never an exception.

Capacity governance: :meth:`ThumbnailCache.prune` removes least-recently-used
entries once the directory exceeds the byte or file-count budget. Read hits
explicitly bump the entry's timestamps (``os.utime``) so the LRU order stays
correct even on ``noatime``/``relatime`` filesystems.

The whole cache can be disabled (Settings toggle); when disabled every
operation is a no-op and behavior falls back to plain re-extraction.
"""

import hashlib
import os
import threading
from collections import OrderedDict

from loguru import logger

# Bump when the on-disk format or the thumbnail geometry changes; old
# entries then simply miss and are garbage-collected by prune().
CACHE_FORMAT_VERSION = 1

DEFAULT_MAX_BYTES = 500 * 1024 * 1024  # 500 MB
DEFAULT_MAX_FILES = 20000
DEFAULT_MEMORY_ENTRIES = 2048  # encoded JPEGs, ~50KB each
JPEG_QUALITY = 85
_SUFFIX = ".jpg"


def default_cache_dir():
    """``<CacheLocation>/raw_alchemy/thumbs`` with a non-Qt fallback."""
    base = None
    try:
        from PySide6.QtCore import QStandardPaths

        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.CacheLocation
        )
    except Exception:
        base = None
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "raw_alchemy", "thumbs")


class ThumbnailCache:
    """mtime-keyed thumbnail cache with a memory layer and LRU pruning."""

    def __init__(
        self,
        cache_dir=None,
        max_bytes=DEFAULT_MAX_BYTES,
        max_files=DEFAULT_MAX_FILES,
        enabled=True,
        memory_entries=DEFAULT_MEMORY_ENTRIES,
    ):
        self.cache_dir = cache_dir or default_cache_dir()
        self.max_bytes = max_bytes
        self.max_files = max_files
        self._enabled = bool(enabled)
        self._memory_entries = memory_entries
        self._mem = OrderedDict()  # key -> encoded JPEG bytes
        self._lock = threading.Lock()
        self._prune_thread = None

    # -- state ---------------------------------------------------------

    @property
    def enabled(self):
        return self._enabled

    def set_enabled(self, enabled):
        """Toggle the cache. Disabling also drops the session memory layer
        so behavior fully falls back to plain re-extraction."""
        self._enabled = bool(enabled)
        if not self._enabled:
            with self._lock:
                self._mem.clear()

    # -- keys ------------------------------------------------------------

    @staticmethod
    def _stat_key(path, st):
        payload = "|".join(
            (
                os.path.abspath(path),
                str(st.st_mtime_ns),
                str(st.st_size),
                f"v{CACHE_FORMAT_VERSION}",
            )
        )
        return hashlib.sha1(payload.encode("utf-8", "surrogatepass")).hexdigest()

    def key_for(self, path):
        """Cache key for the source file's current identity, or None."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        return self._stat_key(path, st)

    def _entry_path(self, key):
        return os.path.join(self.cache_dir, key + _SUFFIX)

    # -- memory layer ------------------------------------------------------

    def _mem_get(self, key):
        with self._lock:
            data = self._mem.get(key)
            if data is not None:
                self._mem.move_to_end(key)
            return data

    def _mem_put(self, key, data):
        with self._lock:
            self._mem[key] = data
            self._mem.move_to_end(key)
            while len(self._mem) > self._memory_entries:
                self._mem.popitem(last=False)

    # -- public API --------------------------------------------------------

    def get(self, path):
        """Return the cached thumbnail QImage for ``path``, or None.

        Corrupt disk entries are deleted so the caller re-extracts and
        overwrites them. Never raises for I/O problems.
        """
        if not self._enabled:
            return None
        key = self.key_for(path)
        if key is None:
            return None

        from PySide6.QtGui import QImage

        data = self._mem_get(key)
        if data is not None:
            image = QImage.fromData(data, "JPG")
            if not image.isNull():
                return image
            with self._lock:
                self._mem.pop(key, None)

        entry = self._entry_path(key)
        try:
            with open(entry, "rb") as f:
                data = f.read()
        except OSError:
            return None
        image = QImage.fromData(data, "JPG")
        if image.isNull():
            # Corrupt entry: drop it; the caller re-extracts and overwrites.
            try:
                os.unlink(entry)
            except OSError:
                pass
            return None
        try:
            os.utime(entry, None)  # bump LRU stamp; robust to relatime mounts
        except OSError:
            pass
        self._mem_put(key, data)
        return image

    def put(self, path, image):
        """Store ``image`` for the current identity of ``path``. Returns
        True when the disk write succeeded."""
        if not self._enabled or image is None or image.isNull():
            return False
        key = self.key_for(path)
        if key is None:
            return False
        data = self._encode_jpeg(image)
        if data is None:
            return False
        self._mem_put(key, data)

        entry = self._entry_path(key)
        tmp = f"{entry}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, entry)  # atomic: readers never see partial files
        except OSError as e:
            logger.debug(f"[ThumbCache] write failed for {path}: {e}")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
        return True

    @staticmethod
    def _encode_jpeg(image):
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice

        array = QByteArray()
        buffer = QBuffer(array)
        if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
            return None
        ok = image.save(buffer, "JPG", JPEG_QUALITY)
        buffer.close()
        if not ok or array.isEmpty():
            return None
        return bytes(array)

    # -- capacity governance -------------------------------------------------

    def prune(self, max_bytes=None, max_files=None):
        """Delete least-recently-used entries until the directory is within
        both budgets. Returns the number of files removed."""
        max_bytes = self.max_bytes if max_bytes is None else max_bytes
        max_files = self.max_files if max_files is None else max_files

        entries = []
        total = 0
        try:
            with os.scandir(self.cache_dir) as it:
                for entry in it:
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    stamp = max(st.st_atime, st.st_mtime)
                    entries.append((stamp, entry.path, st.st_size))
                    total += st.st_size
        except OSError:
            return 0

        count = len(entries)
        if total <= max_bytes and count <= max_files:
            return 0

        entries.sort()  # oldest stamp first
        removed = 0
        for _stamp, path, size in entries:
            if total <= max_bytes and count <= max_files:
                break
            try:
                os.unlink(path)
            except FileNotFoundError:
                # Already deleted by a concurrent prune/clear: its bytes are
                # gone either way, so account for it — skipping the deduction
                # here made concurrent prunes evict up to 2x the excess (m2).
                total -= size
                count -= 1
                continue
            except OSError:
                continue
            total -= size
            count -= 1
            removed += 1
        if removed:
            logger.info(f"[ThumbCache] pruned {removed} entries from {self.cache_dir}")
        return removed

    def prune_async(self):
        """Run prune() on a low-cost daemon thread (at most one at a time)."""
        if not self._enabled:
            return None
        with self._lock:
            thread = self._prune_thread
            if thread is not None and thread.is_alive():
                return thread
            thread = threading.Thread(
                target=self._prune_guarded, name="thumb-cache-prune", daemon=True
            )
            self._prune_thread = thread
            # Start inside the lock: a not-yet-started thread reports
            # is_alive() == False, so publishing before start let a second
            # caller sneak in and launch a duplicate prune (m2 single-flight
            # race). prune() itself never takes self._lock, so the new
            # thread cannot deadlock against this critical section.
            thread.start()
        return thread

    def _prune_guarded(self):
        try:
            self.prune()
        except Exception as e:  # never propagate from the background thread
            logger.debug(f"[ThumbCache] prune failed: {e}")

    def clear(self):
        """Delete every cache entry (works even when disabled). Returns the
        number of files removed."""
        with self._lock:
            self._mem.clear()
        try:
            with os.scandir(self.cache_dir) as it:
                paths = [e.path for e in it if e.is_file(follow_symlinks=False)]
        except OSError:
            return 0
        removed = 0
        for path in paths:
            try:
                os.unlink(path)
                removed += 1
            except OSError:
                pass
        return removed
