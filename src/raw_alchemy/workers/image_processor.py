"""
GPU-resident image processing pipeline using Taichi.

All image data lives on GPU as ti.ndarray. Data crosses CPU-GPU boundary only:
  1. RAW decode (rawpy + RCD demosaic -> numpy -> GPU upload)  [CPU -> GPU, once per image]
  2. Lens correction (GPU -> CPU -> GPU)          [cv2, once per image]
  3. Display output (GPU -> numpy)                [GPU -> CPU, each frame]

Taichi has a single global instance per process - all kernel calls are
serialized. This QThread processes requests sequentially, which is safe.
"""
import math
import threading
import time
import os
import gc
import numpy as np
import colour
from typing import Optional
from PySide6.QtCore import QThread, Signal
from loguru import logger

from raw_alchemy import config, metering
from raw_alchemy.backend import ti
from raw_alchemy.math_ops import (
    apply_matrix_inplace,
    float_to_uint8_gpu,
    resize_float_to_uint8_gpu,
)
from raw_alchemy.pipeline.request import ProcessRequest, ProcessorParams
from raw_alchemy.pipeline.cache_manager import ImageCacheManager, CachedImage
from raw_alchemy.pipeline.executor import PipelineAborted, PreviewExecutor
from raw_alchemy.pipeline.ops import Op, _as_hashable, build_op_list
from raw_alchemy.gpu_buffer import GpuImage, acquire_ndarray, gpu_pool, release_ndarray
from raw_alchemy.onnx.denoiser import denoise_raw, clear_session as denoise_clear_session


PROXY_TARGET_PIXELS = 3_000_000
PROXY_MIN_SOURCE_PIXELS = 4_000_000
FULL_REFINE_IDLE_SECONDS = 1.0

# ===== zoom>fit ROI rendering (T7.5) =====
# Margin added around the visible region, per side, as a fraction of the
# visible span (0.25 per side => the ROI spans ~1.5x the viewport, so small
# pans stay inside the rendered region and cost zero pipeline work).
ROI_MARGIN_FRACTION = 0.25
# ROI edges snap outward to this pixel grid so tiny pan/zoom jitters map to
# the same ROI rect (stable op/prefix/output-cache keys).
ROI_ALIGN = 32
# When the snapped ROI would cover almost the whole frame there is nothing
# to save — render the plain full frame instead (legacy path).
ROI_FULL_COVERAGE_FRACTION = 0.9
# Safety cap for the ROI output texture (>= 4K viewport * 1.5 margin^2).
ROI_MAX_OUTPUT_PIXELS = 24_000_000


class RoiSourceSize(tuple):
    """(width, height) of the full preview frame, plus the ROI pixel rect.

    Behaves exactly like the plain ``(w, h)`` tuple emitted with full-frame
    results (indexing/equality/unpacking), so every existing consumer of
    ``result_ready``'s ``source_size`` keeps working. ROI-aware consumers
    read ``roi_rect`` = (x, y, w, h) in full-frame pixel coordinates.
    """

    def __new__(cls, size, roi_rect):
        obj = super().__new__(cls, (int(size[0]), int(size[1])))
        obj.roi_rect = tuple(int(v) for v in roi_rect)
        return obj


class ImageProcessor(QThread):
    """
    GPU-resident image processing worker.

    Processing pipeline (all on GPU via ti.ndarray):
      RAW decode -> upload -> lens -> geometry -> perspective -> crop ->
      exposure -> WB/HS/Sat/Con -> Log/Matrix -> LUT -> sRGB ->
      denoise -> sharpen -> display

    Emits:
      result_ready(img_uint8, path, request_id, applied_ev, source_size)
      load_complete(path, request_id)
    """
    result_ready = Signal(np.ndarray, str, int, float, object)
    load_complete = Signal(str, int)
    error_occurred = Signal(str)
    denoise_progress = Signal(int, int)
    denoise_started = Signal()
    denoise_finished = Signal()
    export_finished = Signal(bool, str)
    preview_source_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

        # Request management
        self.pending_request: Optional[ProcessRequest] = None
        self.current_request_id = 0
        self._preload_queue: list = []
        self._export_queue: list = []

        # LRU Cache (CPU-side, for RAW decode results)
        self.cache_manager = ImageCacheManager()
        self._warmed_up = False

        # ===== CPU stage caches =====
        self.cpu_linear: Optional[np.ndarray] = None
        self.cpu_proxy_linear: Optional[np.ndarray] = None
        self.cpu_corrected: Optional[np.ndarray] = None  # After lens correction
        self.cpu_proxy_corrected: Optional[np.ndarray] = None

        # GPU output buffer for float32 -> uint8 preview conversion.
        self.gpu_graded = GpuImage()

        # ===== Cache Keys =====
        self.cached_lens_key = None
        self.cached_proxy_lens_key = None

        # Metering Cache
        self.cached_auto_ev = 0.0
        self.last_metering_key = None

        self.exif_data = None
        self.exif_metadata = None  # Full metadata dict for export EXIF writing
        self.last_applied_ev = 0.0
        self.current_path = None

        # Denoising Cache
        self.cached_denoise_original = None
        self.cached_denoise_full = None
        self.last_denoise_key = None

        self._should_stop = False
        self._gpu_uint8 = None      # Pre-allocated GPU uint8 buffer (ti.ndarray)
        # Proxy and full previews each keep their own executor (T7.4), so a
        # zoom across 1.0 or an idle full refine never evicts the other
        # side's prefix/final caches. Keyed by source mode: 'proxy' / 'full'.
        self._preview_executors: dict[str, PreviewExecutor] = {}
        self._executor_path: Optional[str] = None
        self._executor_params: ProcessorParams = {}
        self._executor_using_proxy = False
        self._executor_corrected_source: Optional[np.ndarray] = None
        self._full_refine_request: Optional[ProcessRequest] = None
        self._full_refine_due = 0.0
        # Most recent user interaction (load/preview request) timestamp,
        # used to postpone the idle full refine (T7.3).
        self._last_interaction = 0.0
        self.last_preview_source = "full"

    def stop_and_cleanup(self):
        """Signal the worker to stop and wait for GPU cleanup."""
        self._should_stop = True
        if self.isRunning():
            self.wait(5000)  # Wait up to 5s for thread to finish

    def load_image(self, path: str):
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self._full_refine_request = None
            self._last_interaction = time.perf_counter()
            self.pending_request = ProcessRequest(path, {'_load': True}, request_id)
        if not self.isRunning():
            self.start()
        return request_id

    def preload_image(self, path: str):
        with self.lock:
            if not any(p.path == path for p in self._preload_queue):
                self._preload_queue.append(ProcessRequest(path, {'_preload': True}, -1))
        if not self.isRunning():
            self.start()

    def update_preview(self, path: str, params: ProcessorParams):
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self._full_refine_request = None
            self._last_interaction = time.perf_counter()
            self.pending_request = ProcessRequest(path, params, request_id)
        if not self.isRunning():
            self.start()
        return request_id

    def export_from_cache(self, path: str, export_params: dict):
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self._full_refine_request = None
            params = {'_export': True, 'export_kind': 'cache', 'export': export_params}
            self._export_queue.append(ProcessRequest(path, params, request_id))
        if not self.isRunning():
            self.start()
        return request_id

    def export_path(self, path: str, export_params: dict):
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self._full_refine_request = None
            params = {'_export': True, 'export_kind': 'path', 'export': export_params}
            self._export_queue.append(ProcessRequest(path, params, request_id))
        if not self.isRunning():
            self.start()
        return request_id

    def run(self):
        # Initialize Taichi on this thread so CUDA context is valid here (once only)
        from raw_alchemy.math_ops import init_taichi, warmup
        init_taichi()
        if not self._warmed_up:
            warmup()
            self._warmed_up = True

        # Permanent worker loop 鈥?thread stays alive until app closes
        while not self._should_stop:
            request = self._take_next_request()

            if not request:
                time.sleep(0.05)
                continue

            try:
                if '_export' in request.params:
                    self._do_export(request)
                elif '_preload' in request.params:
                    if self._interactive_abort_requested():
                        continue
                    self._do_preload(request)
                elif '_load' in request.params:
                    self._do_load(request)
                else:
                    self._do_process(request)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.error_occurred.emit(str(e))

        # Release GPU buffers on shutdown while CUDA context is still valid
        self._release_gpu_buffers()

    def _take_next_request(self) -> Optional[ProcessRequest]:
        """Pick the next request: user > export > preload > idle full refine.

        Full-refine trigger tightening (T7.3): a due refine is postponed —
        never dropped — while queued work exists or a user interaction
        happened within the idle window, so the expensive full-frame pass
        only starts once the worker is genuinely idle.
        """
        with self.lock:
            now = time.perf_counter()
            request = self.pending_request
            if request:
                self.pending_request = None
                self._preload_queue.clear()
                return request
            if self._full_refine_request is not None and (
                self._export_queue or self._preload_queue
            ):
                # Queued work runs first; restart the refine idle window so
                # it fires only after the queue drains and things calm down.
                self._full_refine_due = max(
                    self._full_refine_due, now + FULL_REFINE_IDLE_SECONDS
                )
            if self._export_queue:
                return self._export_queue.pop(0)
            if self._preload_queue:
                return self._preload_queue.pop(0)
            if self._full_refine_request is not None and now >= self._full_refine_due:
                if now - self._last_interaction < FULL_REFINE_IDLE_SECONDS:
                    self._full_refine_due = (
                        self._last_interaction + FULL_REFINE_IDLE_SECONDS
                    )
                    return None
                request = self._full_refine_request
                self._full_refine_request = None
                return request
        return None

    def _interactive_abort_requested(self) -> bool:
        """True when a newer user request is waiting (T7.3 abort predicate)
        or shutdown was requested (m7).

        Injected into PreviewExecutor.run_result as ``should_abort`` and
        polled at op boundaries; also checked between _do_preload stages.
        Including ``_should_stop`` lets stop_and_cleanup() cancel a long
        in-flight run at the next op boundary instead of timing out its
        wait() while kernels are still running (the abort surfaces as
        PipelineAborted, which _do_process already drops silently).
        """
        with self.lock:
            return self.pending_request is not None or self._should_stop

    # =================================================================
    # Loading
    # =================================================================

    @staticmethod
    def _get_flip(path: str) -> int:
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                return raw.sizes.flip
        except Exception:
            return 0

    def _rawpy_to_prophoto(self, path: str):
        """Decode RAW via RawSpeed (or rawpy fallback) + GPU demosaic -> ProPhoto Linear float32.

        Returns:
            (img, exif_data, exif_metadata) where img is (H, W, 3) float32 ProPhoto Linear.
        """
        import rawpy
        from raw_alchemy.demosaic import rcd_demosaic, get_dcraw_filters, get_cfa_pattern_from_filters
        from raw_alchemy.exif import extract_lens_exif
        from raw_alchemy.onnx.denoiser import _apply_flip
        from raw_alchemy.colorspace_matrices import cam_to_working_space_matrix
        from rawspeedpy import try_decode, XTRANS_PATTERN

        # Try RawSpeed first (faster decode for supported formats)
        rs = try_decode(path)

        if rs and (rs.is_bayer or rs.is_xtrans) and rs.color_matrix is not None:
            logger.info(f"  [RawSpeed] Decoded: {rs.width}x{rs.height}")
            sensor_raw = rs.bayer.astype(np.float32)
            bl = np.array(rs.black_levels, dtype=np.float32)
            wl = float(rs.white_level)
            wb = np.array(rs.wb_coeffs, dtype=np.float32)
            flip = self._get_flip(path)
            xyz_to_cam = rs.color_matrix.astype(np.float64) if rs.color_matrix is not None else None
            is_bayer = rs.is_bayer
            is_xtrans = rs.is_xtrans
            filters = rs.filters
            cfa_pattern = get_cfa_pattern_from_filters(filters) if is_bayer else XTRANS_PATTERN
            xtrans_pat = XTRANS_PATTERN if is_xtrans else None
        else:
            # rawpy fallback
            if rs:
                logger.info(f"  [RawSpeed] Non-Bayer/X-Trans sensor, falling back to rawpy")
            else:
                logger.info(f"  [RawSpeed] Unsupported format, falling back to rawpy")

            with rawpy.imread(path) as raw:
                cfa_pattern = raw.raw_pattern if raw.raw_pattern is not None else None
                is_bayer = cfa_pattern is not None and cfa_pattern.shape == (2, 2)
                is_xtrans = cfa_pattern is not None and cfa_pattern.shape == (6, 6)

                if is_bayer or is_xtrans:
                    sensor_raw = raw.raw_image_visible.astype(np.float32)
                    bl = np.array(raw.black_level_per_channel, dtype=np.float32)
                    wl = float(raw.white_level)
                    wb = np.array(raw.camera_whitebalance, dtype=np.float32)
                    flip = raw.sizes.flip
                    xyz_to_cam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)
                    if is_xtrans:
                        xtrans_pat = cfa_pattern.copy()
                    filters = None

        from raw_alchemy.core import highlight_inpaint_opposed, subtract_black_level, fix_hot_pixels

        if is_bayer:
            g = wb[1] if wb[1] > 0 else 1.0
            bayer_norm = subtract_black_level(sensor_raw, bl, wl, cfa_pattern)
            fix_hot_pixels(bayer_norm, cfa_pattern)
            highlight_inpaint_opposed(bayer_norm, cfa_pattern, wb)

            dcraw_filters = filters if filters else get_dcraw_filters(cfa_pattern)
            rgb = rcd_demosaic(bayer_norm, dcraw_filters)
            rgb = np.ascontiguousarray(_apply_flip(rgb, flip))

            # Apply white balance
            rgb[:, :, 0] *= wb[0] / g
            rgb[:, :, 2] *= wb[2] / g

            # Camera鈫扨roPhoto matrix, derived analytically (dcraw/darktable
            # algorithm). Matches the old lstsq fit to within 0.3% per cell.
            cam_to_working = cam_to_working_space_matrix(xyz_to_cam)
            apply_matrix_inplace(rgb, cam_to_working)

        elif is_xtrans:
            g = wb[1] if wb[1] > 0 else 1.0
            from raw_alchemy.xtrans_demosaic import xtrans_markesteijn_demosaic

            raw_norm = subtract_black_level(sensor_raw, bl, wl, xtrans_pat)
            fix_hot_pixels(raw_norm, xtrans_pat)
            highlight_inpaint_opposed(raw_norm, xtrans_pat, wb)

            rgb = xtrans_markesteijn_demosaic(raw_norm, xtrans_pat)
            rgb = np.ascontiguousarray(_apply_flip(rgb, flip))

            rgb[:, :, 0] *= wb[0] / g
            rgb[:, :, 2] *= wb[2] / g

            cam_to_working = cam_to_working_space_matrix(xyz_to_cam)
            apply_matrix_inplace(rgb, cam_to_working)

        else:
            # Non-Bayer/non-X-Trans sensor (Foveon, etc.): fallback to rawpy postprocess
            logger.info(f"  Non-Bayer sensor, using rawpy postprocess fallback")
            with rawpy.imread(path) as raw:
                rgb16 = raw.postprocess(
                    gamma=(1, 1), no_auto_bright=True, use_camera_wb=True,
                    use_auto_wb=False, output_bps=16,
                    output_color=rawpy.ColorSpace.ProPhoto,
                    bright=1.0, highlight_mode=2,
                )
            rgb = (rgb16 / 65535.0).astype(np.float32)
            del rgb16
            if rgb.ndim == 3 and rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)

        np.clip(rgb, 0.0, 1.0, out=rgb)

        # Extract EXIF via pyexiv2
        exif_data, exif_metadata = extract_lens_exif(path, None)

        return rgb, exif_data, exif_metadata

    @staticmethod
    def _make_proxy(linear_data: np.ndarray) -> Optional[np.ndarray]:
        """Create a 2-4MP linear proxy for interactive preview work."""
        h, w = linear_data.shape[:2]
        pixels = h * w
        if pixels <= PROXY_MIN_SOURCE_PIXELS:
            return None

        scale = (PROXY_TARGET_PIXELS / float(pixels)) ** 0.5
        if scale >= 0.95:
            return None

        target_w = max(1, int(round(w * scale)))
        target_h = max(1, int(round(h * scale)))
        try:
            import cv2

            proxy = cv2.resize(linear_data, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.debug(f"[Worker] Proxy generation skipped: {e}")
            return None

        if proxy.dtype != np.float32:
            proxy = proxy.astype(np.float32)
        if not proxy.flags['C_CONTIGUOUS']:
            proxy = np.ascontiguousarray(proxy)
        logger.info(
            f"[Worker] Proxy generated: {w}x{h} -> {target_w}x{target_h} "
            f"({target_w * target_h / 1_000_000:.1f}MP)"
        )
        return proxy

    def _do_preload(self, request: ProcessRequest):
        """Preload a neighbor RAW into the CPU cache: decode + proxy only.

        Preload slimming (T7.3): neighbor preloads no longer precompute the
        full-size lens correction — compute_lens_distortion_map plus three
        full-resolution INTER_CUBIC remaps cost seconds per image and pinned
        ~515MB of corrected data per entry (T7.6). The interactive path
        computes `corrected` on demand for the image actually being viewed.

        Stage boundaries (decode -> proxy) check for a pending user request;
        on hit the remaining stages are abandoned, but everything already
        computed still goes into the cache.
        """
        path = request.path
        if self.cache_manager.get(path):
            return

        try:
            # Stage 1: RAW decode + demosaic (the expensive part).
            img, exif_data, _exif_meta = self._rawpy_to_prophoto(path)

            # Stage 2: interactive proxy generation — skipped when a user
            # request arrived while decoding (the decode result is kept).
            proxy_img = None
            if not self._interactive_abort_requested():
                proxy_img = self._make_proxy(img)

            new_cache_item = CachedImage(
                path=path,
                linear_data=img,
                proxy_linear=proxy_img,
                exif_data=exif_data,
                lens_key=None,
                exif_metadata=_exif_meta,
            )
            self.cache_manager.put(path, new_cache_item)
            logger.info(
                f"[Worker] Preloaded: {os.path.basename(path)} "
                f"({img.shape[1]}x{img.shape[0]})"
                f"{'' if proxy_img is not None else ' (proxy skipped: superseded)'}"
            )

        except Exception as e:
            logger.warning(f"Preload failed for {path}: {e}")

    def _do_load(self, request: ProcessRequest):
        """Load RAW and upload to GPU."""
        path = request.path

        # Check CPU cache first
        cached_item = self.cache_manager.get(path)

        if cached_item:
            logger.info(f"[Worker] Cache Hit: {os.path.basename(path)}")
            # Snapshot under the entry lock so the correlated (data, key)
            # pairs stay consistent against a concurrent field-level eviction
            # (GUI-thread cap change) or the baseline processor's write-backs.
            with cached_item.lock:
                self.cpu_linear = cached_item.linear_data
                self.cpu_proxy_linear = cached_item.proxy_linear
                self.exif_data = cached_item.exif_data
                self.exif_metadata = cached_item.exif_metadata
                self.cached_lens_key = cached_item.lens_key
                self.cached_proxy_lens_key = cached_item.proxy_lens_key

                self.cpu_corrected = cached_item.corrected_data
                self.cpu_proxy_corrected = cached_item.proxy_corrected_data

                # Restore denoise cache
                self.cached_denoise_full = cached_item.denoise_full
                self.cached_denoise_original = cached_item.denoise_original
                self.last_denoise_key = cached_item.denoise_key

            # Aborted-preload backfill (T7.3/M6): a neighbor preload that was
            # superseded mid-decode caches the entry with proxy_linear=None.
            # Without a rebuild, _should_use_proxy_preview stays False for
            # this image forever and every fit-view interaction pays the
            # full-resolution pipeline. Rebuild the proxy once, on the load
            # that actually views the image (~100ms, vs seconds per slider
            # move saved), and publish it back to the entry.
            if self.cpu_proxy_linear is None and self.cpu_linear is not None:
                proxy_np = self._make_proxy(self.cpu_linear)
                if proxy_np is not None:
                    self.cpu_proxy_linear = proxy_np
                    with cached_item.lock:
                        cached_item.proxy_linear = proxy_np
                    # Re-account outside the entry lock (manager.lock ->
                    # entry.lock is the fixed lock order).
                    self.cache_manager.notify_entry_updated(path)

        else:
            logger.info(f"[Worker] Cache Miss - Loading: {os.path.basename(path)}")

            try:
                img_np, self.exif_data, self.exif_metadata = self._rawpy_to_prophoto(path)
                proxy_np = self._make_proxy(img_np)

                # Keep on CPU (GPU upload only when needed for processing)
                self.cpu_linear = img_np
                self.cpu_proxy_linear = proxy_np

                # Cache on CPU
                new_cache_item = CachedImage(
                    path=path,
                    linear_data=img_np,
                    proxy_linear=proxy_np,
                    exif_data=self.exif_data,
                    lens_key=None,
                    exif_metadata=self.exif_metadata,
                )
                self.cache_manager.put(path, new_cache_item)

                logger.info(f"[Worker] Loaded: {img_np.shape[1]}x{img_np.shape[0]}")

            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                self.error_occurred.emit(f"Failed to load image: {e}")
                return

        # Reset pipeline caches for new image
        if path != self.current_path:
            # Decide from the locked snapshot, not a re-read of the entry (a
            # concurrent eviction may have cleared the field since).
            if cached_item and self.cpu_corrected is not None:
                # Cache hit with lens correction: only invalidate downstream
                self.gpu_graded.clear()
                self._release_gpu_uint8()
                self._clear_all_executor_caches()
                self.last_metering_key = None
                self.cached_auto_ev = 0.0
            else:
                self._invalidate_pipeline_caches()
            self.current_path = path

        self.load_complete.emit(path, request.request_id)

    def _release_gpu_uint8(self):
        """Return the uint8 output buffer to the GPU pool (T7.2)."""
        if self._gpu_uint8 is not None:
            release_ndarray(self._gpu_uint8, ti.u8, tuple(self._gpu_uint8.shape))
            self._gpu_uint8 = None

    def _invalidate_pipeline_caches(self):
        """Invalidate all pipeline stage caches (but not denoise/sharpen)."""
        self.cached_lens_key = None
        self.cached_proxy_lens_key = None
        self.cpu_corrected = None
        self.cpu_proxy_corrected = None
        self.gpu_graded.clear()
        self._release_gpu_uint8()
        self._full_refine_request = None
        self._clear_all_executor_caches()
        self.cached_auto_ev = 0.0
        # Force immediate GPU memory release: collect dead GpuImages into the
        # pool first, then drop the pool's free buffers (image switch — T7.2
        # VRAM lifecycle: pooled sharpen/demosaic scratch is released here).
        gc.collect()
        gpu_pool().clear()
        self.last_metering_key = None

    def _release_gpu_buffers(self):
        """Release all GPU buffers while CUDA context is still valid (on worker thread)."""
        try:
            self.gpu_graded.clear()
            self._release_gpu_uint8()
            self._full_refine_request = None
            self._clear_all_executor_caches()
            gpu_pool().clear()
            logger.debug("[Worker] GPU buffers released.")
        except Exception as e:
            logger.debug(f"[Worker] GPU buffer release error (non-critical): {e}")

    @staticmethod
    def _normalize_lens_state_key(key) -> Optional[tuple]:
        """Canonical form of a (lens_correct, custom_db_path, denoise) key.

        The worker stores these keys with raw param values (True/False/None
        for lens_correct); normalizing the booleans lets state written by
        different code paths compare equal when semantically identical.
        """
        if key is None:
            return None
        lens_correct, custom_db_path, denoise_enabled = key
        return (bool(lens_correct), custom_db_path, bool(denoise_enabled))

    @classmethod
    def _expected_lens_state_key(cls, params: ProcessorParams) -> tuple:
        """The lens/denoise state the current UI params require of
        ``cpu_corrected`` (normalized)."""
        return (
            bool(params.get('lens_correct', False)),
            params.get('custom_db_path'),
            bool(params.get('denoise_enabled', False)),
        )

    def get_cached_for_export(self, params: Optional[ProcessorParams] = None) -> Optional[dict]:
        """Return cached data for the current image, for use by single-image export.

        Called from the main thread. The numpy arrays are read-only references
        so no copy is made here 鈥?the export thread should treat them as
        immutable (they are only ever replaced, never mutated in-place).

        When ``params`` (the current UI params) are given, ``cpu_corrected``
        is served only if its recorded lens/denoise state
        (``cached_lens_key``) matches what those params require. The T7.4
        multi-slot output cache made pure preview hits routine, and a hit
        returns before the executor re-syncs this state — so the export
        path must key-validate instead of trusting it (M5). On mismatch
        this returns None and the caller falls back to the full
        re-processing export path: slower, but always consistent with the
        preview. (Re-syncing on the hit path instead would mean re-running
        lens correction / denoise — seconds to tens of seconds — inside
        the interactive path, exactly the cost T7.4 exists to avoid.)
        """
        # Snapshot once: the worker thread only ever *replaces* these
        # references, so locals keep the validated pairing consistent.
        corrected = self.cpu_corrected
        if corrected is None:
            return None
        if params is not None:
            expected = self._expected_lens_state_key(params)
            if self._normalize_lens_state_key(self.cached_lens_key) != expected:
                return None
            if expected[2] and self.cached_denoise_full is None:
                # The key claims a denoised source but the denoise artifact
                # is gone (failed/aborted run, dropped cache) — cpu_corrected
                # cannot be trusted to actually be denoised.
                return None
        return {
            'corrected': corrected,
            'denoise_original': self.cached_denoise_original,
            'denoise_full': self.cached_denoise_full,
            'exif_data': self.exif_data,
            'exif_metadata': self.exif_metadata,
        }

    def _do_export(self, request: ProcessRequest):
        payload = request.params.get('export', {})
        export_kind = request.params.get('export_kind', 'cache')
        try:
            if export_kind == 'path':
                from raw_alchemy import orchestrator

                orchestrator.process_path(**payload)
            else:
                from raw_alchemy.core import export_from_cache

                export_from_cache(**payload)
            self.export_finished.emit(True, "")
        except Exception as e:
            logger.error(f"[Worker] Export failed: {e}")
            self.export_finished.emit(False, str(e))

    # =================================================================
    # Processing Pipeline (GPU-resident)
    # =================================================================

    @staticmethod
    def _make_pipeline_key(params):
        """Key over everything that changes the color pipeline result (T7.4).

        Covers the source identity (proxy/full, denoise, lens) and every op
        parameter — but *no* view state (zoom/viewport/DPR). Two requests
        with equal pipeline keys share the same GPU-resident pipeline
        output; only the final resampling to uint8 differs.
        """
        return _as_hashable((
            params.get('_preview_source', 'full'),
            params.get('denoise_enabled', False),
            params.get('lens_correct'), params.get('custom_db_path'),
            params.get('rotation', 0),
            params.get('flip_horizontal', False),
            params.get('flip_vertical', False),
            params.get('perspective_corners'),
            params.get('crop', (0.0, 0.0, 1.0, 1.0)),
            params.get('exposure_mode'), params.get('exposure', 0.0),
            params.get('metering_mode', 'matrix'),
            params.get('wb_temp', 0.0), params.get('wb_tint', 0.0),
            params.get('highlight', 0.0), params.get('shadow', 0.0),
            params.get('saturation', 1.0), params.get('contrast', 1.0),
            params.get('log_space'), params.get('lut_path'),
            params.get('sharpen_strength', 0.0),
        ))

    @staticmethod
    def _make_view_key(params):
        """Key over the output resampling state only (T7.4): zoom/viewport.

        The quantized ROI rect (T7.5) is view state too: panning beyond the
        rendered margin changes which pixels are output, but never which
        color pipeline produced them.
        """
        viewport_size = params.get('viewport_size')
        return (
            _as_hashable(viewport_size) if viewport_size else None,
            round(float(params.get('preview_zoom', 1.0) or 1.0), 3),
            round(float(params.get('device_pixel_ratio', 1.0) or 1.0), 2),
            _as_hashable(params.get('_preview_roi')),
        )

    @classmethod
    def _make_output_key(cls, params):
        """(pipeline key, view key) pair identifying one uint8 output slot."""
        return (cls._make_pipeline_key(params), cls._make_view_key(params))

    @staticmethod
    def _make_preview_target_size(src_w: int, src_h: int, params: ProcessorParams):
        viewport_size = params.get('viewport_size')
        if not viewport_size or src_w <= 0 or src_h <= 0:
            return src_w, src_h

        try:
            vp_w = max(1, int(viewport_size[0]))
            vp_h = max(1, int(viewport_size[1]))
        except (TypeError, ValueError, IndexError):
            return src_w, src_h

        dpr = max(1.0, float(params.get('device_pixel_ratio', 1.0) or 1.0))
        zoom = max(1.0, float(params.get('preview_zoom', 1.0) or 1.0))
        target_box_w = vp_w * dpr
        target_box_h = vp_h * dpr

        fit_scale = min(target_box_w / src_w, target_box_h / src_h, 1.0)
        target_scale = min(fit_scale * zoom, 1.0)

        target_w = max(1, int(round(src_w * target_scale)))
        target_h = max(1, int(round(src_h * target_scale)))

        max_preview_pixels = int(params.get('max_preview_pixels', 12_000_000) or 12_000_000)
        if max_preview_pixels > 0 and target_w * target_h > max_preview_pixels:
            scale = (max_preview_pixels / float(target_w * target_h)) ** 0.5
            target_w = max(1, int(target_w * scale))
            target_h = max(1, int(target_h * scale))

        return target_w, target_h

    @staticmethod
    def _pipeline_output_dims(src_w: int, src_h: int, params: ProcessorParams):
        """Dimensions of the pipeline output *before* the ROI crop (T7.5).

        Mirrors the shape-changing ops exactly: geometry (90/270 rotation
        swaps axes; flips keep shape), perspective (same shape) and the user
        crop (integer snapping replicated from apply_crop_gpu). Color ops
        never change shape.
        """
        w, h = int(src_w), int(src_h)
        rotation = int(params.get('rotation', 0) or 0) % 360
        if rotation in (90, 270):
            w, h = h, w

        crop = params.get('crop', (0.0, 0.0, 1.0, 1.0))
        if crop and tuple(crop) != (0.0, 0.0, 1.0, 1.0):
            x_norm, y_norm, w_norm, h_norm = (float(v) for v in crop)
            if w_norm > 0 and h_norm > 0:
                cx = max(0, min(int(x_norm * w), w - 1))
                cy = max(0, min(int(y_norm * h), h - 1))
                w = max(1, min(int(w_norm * w), w - cx))
                h = max(1, min(int(h_norm * h), h - cy))
        return w, h

    def _compute_preview_roi(self, params: ProcessorParams):
        """Resolve the zoom>fit visible-region ROI for this request (T7.5).

        Returns ``((x, y, w, h), (full_w, full_h))`` — the ROI pixel rect in
        pipeline-output coordinates plus the full pipeline-output size — or
        None when ROI rendering does not apply (fit view, no viewport info,
        idle full refine, or the ROI would cover ~the whole frame anyway).

        The visible rect comes from the viewport (normalized to the displayed
        frame); it is expanded by ROI_MARGIN_FRACTION per side so pans inside
        the margin need no pipeline work, then snapped outward to a ROI_ALIGN
        pixel grid so jittery pan/zoom values reuse the same cache keys.
        """
        if params.get('_force_full_preview', False):
            return None
        visible = params.get('preview_visible_rect')
        if not visible or not params.get('viewport_size'):
            return None
        zoom = float(params.get('preview_zoom', 1.0) or 1.0)
        if zoom <= 1.001:
            return None
        if self.cpu_linear is None:
            return None

        src_h, src_w = self.cpu_linear.shape[:2]
        full_w, full_h = self._pipeline_output_dims(src_w, src_h, params)

        try:
            x0, y0, x1, y1 = (float(v) for v in visible)
        except (TypeError, ValueError):
            return None
        x0, x1 = max(0.0, min(x0, 1.0)), max(0.0, min(x1, 1.0))
        y0, y1 = max(0.0, min(y0, 1.0)), max(0.0, min(y1, 1.0))
        if x1 <= x0 or y1 <= y0:
            return None

        margin_x = (x1 - x0) * ROI_MARGIN_FRACTION
        margin_y = (y1 - y0) * ROI_MARGIN_FRACTION
        x0 = max(0.0, x0 - margin_x)
        x1 = min(1.0, x1 + margin_x)
        y0 = max(0.0, y0 - margin_y)
        y1 = min(1.0, y1 + margin_y)

        # Snap outward to the alignment grid, clamped to the frame.
        px0 = max(0, (int(math.floor(x0 * full_w)) // ROI_ALIGN) * ROI_ALIGN)
        py0 = max(0, (int(math.floor(y0 * full_h)) // ROI_ALIGN) * ROI_ALIGN)
        px1 = min(full_w, ((int(math.ceil(x1 * full_w)) + ROI_ALIGN - 1) // ROI_ALIGN) * ROI_ALIGN)
        py1 = min(full_h, ((int(math.ceil(y1 * full_h)) + ROI_ALIGN - 1) // ROI_ALIGN) * ROI_ALIGN)
        roi_w = px1 - px0
        roi_h = py1 - py0
        if roi_w <= 0 or roi_h <= 0:
            return None
        if roi_w * roi_h >= ROI_FULL_COVERAGE_FRACTION * full_w * full_h:
            return None
        return (px0, py0, roi_w, roi_h), (full_w, full_h)

    @staticmethod
    def _insert_roi_op(ops: list, roi_rect) -> list:
        """Insert the ROI crop after the shape ops, before the color ops.

        The exposure op is always present and is the first color op, so the
        ROI goes right before it: denoise/lens/geometry/perspective/crop run
        (and prefix-cache) on the full frame — keeping their outputs valid
        across pans — while everything downstream runs on the small ROI.
        """
        index = next(
            (i for i, op in enumerate(ops) if op.name == "exposure"), len(ops)
        )
        ops = list(ops)
        ops.insert(index, Op("roi", tuple(int(v) for v in roi_rect)))
        return ops

    @staticmethod
    def _make_roi_target_size(
        roi_w: int, roi_h: int, full_w: int, full_h: int, params: ProcessorParams
    ):
        """Output size for a ROI render (T7.5).

        The display scale (screen pixels per source pixel) is derived from
        the *full* frame — the same fit*zoom the viewport uses — then applied
        to the ROI, capped at 1:1 native resolution. This is what lifts the
        12MP full-frame cap: at 100%+ the visible region is output
        pixel-for-pixel instead of a downscaled whole frame.
        """
        viewport_size = params.get('viewport_size')
        if not viewport_size or full_w <= 0 or full_h <= 0:
            return roi_w, roi_h
        try:
            vp_w = max(1, int(viewport_size[0]))
            vp_h = max(1, int(viewport_size[1]))
        except (TypeError, ValueError, IndexError):
            return roi_w, roi_h

        dpr = max(1.0, float(params.get('device_pixel_ratio', 1.0) or 1.0))
        zoom = max(1.0, float(params.get('preview_zoom', 1.0) or 1.0))
        fit_scale = min(vp_w * dpr / full_w, vp_h * dpr / full_h, 1.0)
        scale = min(fit_scale * zoom, 1.0)

        target_w = max(1, int(round(roi_w * scale)))
        target_h = max(1, int(round(roi_h * scale)))
        if target_w * target_h > ROI_MAX_OUTPUT_PIXELS:
            shrink = (ROI_MAX_OUTPUT_PIXELS / float(target_w * target_h)) ** 0.5
            target_w = max(1, int(target_w * shrink))
            target_h = max(1, int(target_h * shrink))
        return target_w, target_h

    def _should_use_proxy_preview(self, params: ProcessorParams) -> bool:
        if params.get('_force_full_preview', False):
            return False
        if self.cpu_proxy_linear is None:
            return False
        if params.get('denoise_enabled', False):
            return False
        if not params.get('viewport_size'):
            return False
        zoom = float(params.get('preview_zoom', 1.0) or 1.0)
        return zoom <= 1.0

    def _select_executor_source(self, use_proxy: bool) -> np.ndarray:
        if use_proxy and self.cpu_proxy_linear is not None:
            return self.cpu_proxy_linear
        return self.cpu_linear

    def _schedule_full_refinement(self, request: ProcessRequest, params: ProcessorParams):
        refine_params = params.copy()
        refine_params['_force_full_preview'] = True
        refine_params['_preview_source'] = 'full'
        with self.lock:
            self._full_refine_request = ProcessRequest(
                request.path,
                refine_params,
                request.request_id,
            )
            self._full_refine_due = time.perf_counter() + FULL_REFINE_IDLE_SECONDS

    def _executor_source_mode(self) -> str:
        return 'proxy' if self._executor_using_proxy else 'full'

    @property
    def preview_executor(self) -> Optional[PreviewExecutor]:
        """Executor for the current source mode (T7.4 proxy/full split)."""
        return self._preview_executors.get(self._executor_source_mode())

    def _clear_all_executor_caches(self):
        """Drop the pipeline caches of both the proxy and full executors."""
        for executor in self._preview_executors.values():
            executor.clear_cache()

    def _get_preview_executor(self) -> PreviewExecutor:
        mode = self._executor_source_mode()
        executor = self._preview_executors.get(mode)
        if executor is None:
            executor = PreviewExecutor(
                denoiser=self._executor_denoise,
                lens_corrector=self._executor_lens_correct,
                auto_gain_resolver=self._executor_auto_gain,
                round_exposure_gain=True,
            )
            self._preview_executors[mode] = executor
        return executor

    def _trim_executor_caches(self, budget_bytes: Optional[int] = None):
        """Apply the shared cache byte budget across both executors (T7.4).

        The proxy executor powers slider interactivity and its stages are
        small (<=4MP), so it gets first claim on the budget; the full
        executor absorbs whatever remains. With a single executor this
        degenerates to the T7.6 behavior (one trim with the full budget).
        """
        if budget_bytes is None:
            budget_bytes = int(config.EXECUTOR_CACHE_LIMIT_MB) * 1024 * 1024
        remaining = max(0, int(budget_bytes))
        proxy = self._preview_executors.get('proxy')
        if proxy is not None:
            proxy.trim(remaining)
            remaining = max(0, remaining - proxy.cache_bytes())
        full = self._preview_executors.get('full')
        if full is not None:
            full.trim(remaining)

    def _executor_denoise(self, src: np.ndarray) -> np.ndarray:
        path = self._executor_path
        if path is None:
            return src

        denoise_cache_key = (path, 'denoise')
        if denoise_cache_key != self.last_denoise_key or self.cached_denoise_full is None:
            try:
                self.denoise_started.emit()
                logger.info("[Worker] CANS RAW V2 denoise (replaces demosaicing)...")

                def progress_cb(cur, total):
                    self.denoise_progress.emit(cur, total)

                denoised = denoise_raw(
                    path,
                    exposure_ratio=1.0,
                    progress_callback=progress_cb,
                )

                self.cached_denoise_original = self.cpu_linear
                self.cached_denoise_full = denoised
                self.last_denoise_key = denoise_cache_key
                denoise_clear_session()
                self.denoise_finished.emit()
            except Exception as e:
                logger.error(f"[Worker] Denoising failed: {e}")
                self.denoise_finished.emit()
                self.cached_denoise_original = None
                self.cached_denoise_full = None
                self.last_denoise_key = None
                return src

        denoised = np.ascontiguousarray(self.cached_denoise_full)
        if not self._executor_params.get('lens_correct', False):
            self.cpu_corrected = denoised
            self.cached_lens_key = (
                False,
                self._executor_params.get('custom_db_path'),
                True,
            )
        return denoised

    def _executor_lens_correct(self, src: np.ndarray) -> np.ndarray:
        params = self._executor_params
        denoise_enabled = params.get('denoise_enabled', False)
        current_lens_key = (
            params.get('lens_correct'),
            params.get('custom_db_path'),
            denoise_enabled,
        )

        if self._executor_using_proxy:
            if current_lens_key == self.cached_proxy_lens_key and self.cpu_proxy_corrected is not None:
                self._executor_corrected_source = self.cpu_proxy_corrected
                return self.cpu_proxy_corrected
        elif current_lens_key == self.cached_lens_key and self.cpu_corrected is not None:
            self._executor_corrected_source = self.cpu_corrected
            return self.cpu_corrected

        if params.get('lens_correct') and self.exif_data:
            try:
                import cv2
                from raw_alchemy.lensfun_wrapper import compute_lens_distortion_map

                lf_params = {**self.exif_data}
                dist_result = compute_lens_distortion_map(
                    src,
                    camera_maker=lf_params.get('camera_maker'),
                    camera_model=lf_params.get('camera_model'),
                    lens_maker=lf_params.get('lens_maker'),
                    lens_model=lf_params.get('lens_model'),
                    focal_length=lf_params.get('focal_length', 0),
                    aperture=lf_params.get('aperture', 0),
                    crop_factor=lf_params.get('crop_factor'),
                    custom_db_path=params.get('custom_db_path'),
                )

                if dist_result is not None:
                    coords, oob_mask, vignette_img = dist_result
                    corrected = np.empty_like(vignette_img)
                    for ch in range(3):
                        corrected[:, :, ch] = cv2.remap(
                            vignette_img[:, :, ch],
                            coords[:, :, ch, 0],
                            coords[:, :, ch, 1],
                            cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
                    corrected[oob_mask] = 0
                else:
                    corrected = src
            except Exception as e:
                logger.error(f"[Worker] Lens correction failed: {e}")
                corrected = src
        else:
            corrected = src

        if self._executor_using_proxy:
            self.cpu_proxy_corrected = corrected
            self.cached_proxy_lens_key = current_lens_key
        else:
            self.cpu_corrected = corrected
            self.cached_lens_key = current_lens_key
        self.last_metering_key = None
        self._executor_corrected_source = corrected

        cached_item = self.cache_manager.get(self._executor_path) if self._executor_path else None
        if cached_item:
            # Compute the proxy resize outside the entry lock, then publish
            # the correlated (data, lens_key) pairs atomically so a lock-held
            # field-level eviction on another thread can never observe (or
            # produce) a torn pair.
            proxy_corrected = None
            if not self._executor_using_proxy and cached_item.proxy_linear is not None:
                proxy_corrected = self._make_proxy(corrected)
            with cached_item.lock:
                if self._executor_using_proxy:
                    cached_item.proxy_corrected_data = corrected
                    cached_item.proxy_lens_key = current_lens_key
                else:
                    cached_item.corrected_data = corrected
                    cached_item.lens_key = current_lens_key
                    if proxy_corrected is not None:
                        cached_item.proxy_corrected_data = proxy_corrected
                        cached_item.proxy_lens_key = current_lens_key
            # Re-account the entry and apply quota eviction (T7.6): a
            # full-size corrected write-back can push the cache over budget.
            # Must run outside the entry lock — it takes manager.lock then
            # entry.lock (the eviction order); nesting it inside the entry
            # lock would invert the order and deadlock.
            self.cache_manager.notify_entry_updated(self._executor_path)

        return corrected

    def _executor_auto_gain(self, _metering_img: np.ndarray, metering_mode: str) -> float:
        current_metering_key = (
            'proxy' if self._executor_using_proxy else 'full',
            self.cached_lens_key,
            self.cached_proxy_lens_key,
            metering_mode,
        )

        if current_metering_key == self.last_metering_key:
            return 2.0 ** self.cached_auto_ev

        logger.debug("[Worker] Calculating Auto Exposure...")
        source_cs = colour.RGB_COLOURSPACES[config.WORKING_SPACE]
        strategy = metering.get_metering_strategy(metering_mode)
        metering_data = self._executor_corrected_source
        if metering_data is None:
            metering_data = self.cpu_corrected if self.cpu_corrected is not None else self.cpu_linear
        gain = strategy.calculate_gain(metering_data, source_cs)
        self.cached_auto_ev = np.log2(gain)
        self.last_metering_key = current_metering_key
        return gain

    def _prepare_executor_source_state(self, params: ProcessorParams):
        self._executor_corrected_source = None
        denoise_enabled = params.get('denoise_enabled', False)
        if not denoise_enabled and self.last_denoise_key is not None:
            self.cached_denoise_original = None
            self.cached_denoise_full = None
            self.last_denoise_key = None

        if not params.get('lens_correct', False):
            if denoise_enabled and self.cached_denoise_full is not None:
                corrected_source = np.ascontiguousarray(self.cached_denoise_full)
            else:
                corrected_source = self._select_executor_source(self._executor_using_proxy)
            next_lens_key = (
                False,
                params.get('custom_db_path'),
                denoise_enabled,
            )
            current_key = self.cached_proxy_lens_key if self._executor_using_proxy else self.cached_lens_key
            if next_lens_key != current_key:
                self.last_metering_key = None
            if self._executor_using_proxy:
                self.cpu_proxy_corrected = corrected_source
                self.cached_proxy_lens_key = next_lens_key
            else:
                self.cpu_corrected = corrected_source
                self.cached_lens_key = next_lens_key
            self._executor_corrected_source = corrected_source

    def _sync_executor_export_source(self, params: ProcessorParams):
        if self._executor_using_proxy:
            return
        if params.get('lens_correct', False):
            return

        denoise_enabled = params.get('denoise_enabled', False)
        if denoise_enabled and self.cached_denoise_full is not None:
            self.cpu_corrected = np.ascontiguousarray(self.cached_denoise_full)
        else:
            self.cpu_corrected = self.cpu_linear

    def _restore_export_state_from_entry(
        self, cached_item: Optional[CachedImage], params: ProcessorParams
    ):
        """Cheap export-state resync for the output-cache hit path (M5).

        A hit emits and returns before the executor runs, so
        ``cpu_corrected`` / ``cached_lens_key`` may still describe an older
        parameter state (e.g. after a lens/denoise ON->OFF->ON round trip,
        which the T7.4 multi-slot cache turns into a pure hit). Resync
        without any recompute:

        - lens OFF: the corrected source is just the linear data (or the
          cached denoise result) — synthesize it, mirroring
          ``_prepare_executor_source_state`` / ``_sync_executor_export_source``.
        - lens ON: adopt the entry's full-size corrected write-back when its
          lens key matches the current params (a reference, no copy).

        When neither applies (e.g. the required denoise artifact was
        dropped), the stale fields are deliberately left as-is:
        ``get_cached_for_export`` key-validates them and refuses to serve a
        mismatched array, so the export takes the full re-processing path
        instead of silently diverging from the preview.
        """
        expected = self._expected_lens_state_key(params)
        if (
            self.cpu_corrected is not None
            and self._normalize_lens_state_key(self.cached_lens_key) == expected
        ):
            return

        denoise_enabled = bool(params.get('denoise_enabled', False))
        if not params.get('lens_correct', False):
            if denoise_enabled:
                if self.cached_denoise_full is None:
                    return  # nothing cheap to resync; export re-runs fully
                corrected = np.ascontiguousarray(self.cached_denoise_full)
            else:
                corrected = self.cpu_linear
            if corrected is None:
                return
            self.cpu_corrected = corrected
            self.cached_lens_key = (
                False,
                params.get('custom_db_path'),
                denoise_enabled,
            )
            return

        if cached_item is None:
            return
        with cached_item.lock:
            corrected = cached_item.corrected_data
            entry_key = cached_item.lens_key
        if (
            corrected is not None
            and self._normalize_lens_state_key(entry_key) == expected
        ):
            self.cpu_corrected = corrected
            self.cached_lens_key = entry_key

    def _invalidate_executor_prefix_if_needed(self, params: ProcessorParams):
        """Semantic invalidation of the executor's pipeline caches.

        Division of labor (T7.1): PreviewExecutor.set_source only clears its
        caches when the source *array object* actually changes, so that the
        prefix cache survives across requests that reuse the same source.
        This method owns the semantic invalidations the executor cannot see:
        lens/denoise state changes whose cached op outputs (denoise /
        lens_correct callbacks) depend on worker state not captured by the
        op parameters themselves.

        Since T7.4 proxy and full previews use separate executors, a
        proxy/full source-mode switch (zoom across 1.0, idle full refine)
        no longer clears anything — each executor's caches survive until *its*
        source or semantic state changes. Only the current mode's executor
        is checked here; the other mode re-validates on its next request.
        """
        executor = self.preview_executor  # current source mode's executor
        if executor is None:
            return

        denoise_enabled = params.get('denoise_enabled', False)
        desired_lens_key = (
            params.get('lens_correct'),
            params.get('custom_db_path'),
            denoise_enabled,
        )
        current_lens_key = self.cached_proxy_lens_key if self._executor_using_proxy else self.cached_lens_key

        needs_clear = False
        if denoise_enabled and self.cached_denoise_full is None:
            needs_clear = True
        if desired_lens_key != current_lens_key:
            needs_clear = True

        if needs_clear:
            executor.clear_cache()

    def _do_process(self, request: ProcessRequest):
        t_start = time.perf_counter()
        logger.debug(f"[Worker] _do_process: {os.path.basename(request.path)}, id={request.request_id}")

        try:
            if self.cpu_linear is None or self.current_path != request.path:
                self._do_load(ProcessRequest(request.path, {'_load': True}, request.request_id))
                if self.cpu_linear is None:
                    return

            params = request.params.copy()
            use_proxy = self._should_use_proxy_preview(params)
            preview_source = 'proxy' if use_proxy else 'full'
            params['_preview_source'] = preview_source
            # zoom>fit ROI rendering (T7.5): resolve the visible-region crop
            # before the output key so the view key covers the ROI rect.
            roi_info = None if use_proxy else self._compute_preview_roi(params)
            if roi_info is not None:
                params['_preview_roi'] = roi_info[0]
                params['_roi_full_size'] = roi_info[1]
            output_key = self._make_output_key(params)
            cached_item = self.cache_manager.get(request.path)
            cached_output = cached_item.get_output(output_key) if cached_item else None
            if cached_output is not None:
                logger.info(f"[Worker] Output Cache Hit: {os.path.basename(request.path)}")
                cached_uint8, cached_ev, cached_source_size = cached_output
                # The hit returns before the executor runs, so sync the
                # state downstream consumers read from the worker:
                # - last_applied_ev feeds the Auto-exposure export contract
                #   (resolve_export_exposure); without this line an export
                #   after a cross-image hit bakes the *other* image's EV
                #   while the preview shows this slot's EV (M4).
                # - cpu_corrected/cached_lens_key feed the single-image
                #   export fast path; resync them when it costs nothing (M5).
                self.last_applied_ev = float(cached_ev)
                self._restore_export_state_from_entry(cached_item, params)
                self.last_preview_source = preview_source
                self.preview_source_changed.emit(preview_source)
                if use_proxy:
                    self._schedule_full_refinement(request, params)
                self.result_ready.emit(
                    cached_uint8,
                    request.path,
                    request.request_id,
                    cached_ev,
                    cached_source_size,
                )
                return

            self._executor_path = request.path
            self._executor_params = params
            self._executor_using_proxy = use_proxy
            self._prepare_executor_source_state(params)
            self._invalidate_executor_prefix_if_needed(params)

            ops = build_op_list(params)
            if roi_info is not None:
                # The ROI crop is a preview-only op: it only narrows the
                # source region, so the color-pipeline parameter caches
                # (pipeline key, metering, lens/denoise) stay untouched.
                ops = self._insert_roi_op(ops, roi_info[0])
            executor_source = self._select_executor_source(use_proxy)
            executor = self._get_preview_executor()
            # Cooperative cancellation (T7.3): a newer user request aborts
            # this run at the next op boundary (completed prefixes stay
            # cached). Applies to interactive runs and idle full refines.
            result = executor.run_result(
                ops,
                source=executor_source,
                should_abort=self._interactive_abort_requested,
            )
            self._sync_executor_export_source(params)
            # Prefix-cache byte budget (T7.6): VRAM budget since T7.2 moved
            # pipeline residency to the GPU; shared across the proxy/full
            # executor pair since T7.4. Even if the trim evicts `result`
            # from the final cache, the local reference keeps its GPU buffer
            # alive for the output stage below.
            self._trim_executor_caches()

            applied_ev = float(result.applied_ev)
            self.last_applied_ev = applied_ev

        except PipelineAborted as e:
            # Superseded in flight: drop the run silently — no result_ready,
            # no error. The pending request re-renders from the kept prefixes.
            logger.debug(
                f"[Worker] Aborted in-flight run for "
                f"{os.path.basename(request.path)} (id={request.request_id}): {e}"
            )
            return
        except Exception as e:
            logger.error(f"[Worker] Error in unified pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Processing error: {str(e)}")
            return

        try:
            t_out = time.perf_counter()
            # T7.2: convert the executor's GPU-resident output straight to
            # uint8 on-device — only the final (viewport-sized) uint8 image
            # crosses back to the host, not the full-frame float32.
            if result.gpu is not None and result.gpu.valid:
                graded_arr = result.gpu.arr
                source_h, source_w = result.gpu.height, result.gpu.width
            else:
                # Fallback (results without a GPU buffer): legacy host round-trip.
                processed = result.image
                if processed.dtype != np.float32:
                    processed = processed.astype(np.float32)
                if not processed.flags['C_CONTIGUOUS']:
                    processed = np.ascontiguousarray(processed)
                self.gpu_graded.upload(processed)
                graded_arr = self.gpu_graded.arr
                source_h, source_w = processed.shape[:2]

            if roi_info is not None:
                # ROI render (T7.5): the executor output is the ROI at native
                # resolution; size it by the full-frame display scale and
                # report the *full* frame as source_size (plus the ROI rect)
                # so the viewport geometry stays stable.
                roi_rect, (full_w, full_h) = roi_info
                target_w, target_h = self._make_roi_target_size(
                    source_w, source_h, full_w, full_h, params
                )
                output_source_size = RoiSourceSize((full_w, full_h), roi_rect)
            else:
                target_w, target_h = self._make_preview_target_size(
                    source_w, source_h, params
                )
                output_source_size = (source_w, source_h)

            if self._gpu_uint8 is None or self._gpu_uint8.shape != (target_h, target_w, 3):
                self._release_gpu_uint8()
                self._gpu_uint8 = acquire_ndarray(ti.u8, (target_h, target_w, 3))

            if target_w == source_w and target_h == source_h:
                float_to_uint8_gpu(graded_arr, self._gpu_uint8)
            else:
                resize_float_to_uint8_gpu(graded_arr, self._gpu_uint8)

            img_uint8 = self._gpu_uint8.to_numpy()

            t_end = time.perf_counter()
            self.last_preview_source = preview_source
            self.preview_source_changed.emit(preview_source)
            logger.info(
                f"[Worker] Unified pipeline: {(t_end - t_start)*1000:.0f}ms total, "
                f"output: {(t_end - t_out)*1000:.0f}ms "
                f"source={preview_source}"
                f"{' roi=' + str(params['_preview_roi']) if roi_info is not None else ''} "
                f"({source_w}x{source_h} -> {target_w}x{target_h}, ops={len(ops)})"
            )

            cached_item = self.cache_manager.get(request.path)
            if cached_item:
                # Multi-slot output cache (T7.4): keeps a few recent outputs
                # keyed by (pipeline key, view key), so e.g. fit and the
                # current zoom level survive side by side.
                cached_item.store_output(
                    output_key, img_uint8.copy(), applied_ev, output_source_size
                )
                self.cache_manager.notify_entry_updated(request.path)

            self.result_ready.emit(
                img_uint8,
                request.path,
                request.request_id,
                applied_ev,
                output_source_size,
            )
            if use_proxy:
                self._schedule_full_refinement(request, params)

        except Exception as e:
            logger.error(f"[Worker] Error in unified output: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Output error: {str(e)}")
