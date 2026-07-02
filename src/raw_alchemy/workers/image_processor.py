"""
GPU-resident image processing pipeline using Taichi.

All image data lives on GPU as ti.ndarray. Data crosses CPU-GPU boundary only:
  1. RAW decode (rawpy + RCD demosaic -> numpy -> GPU upload)  [CPU -> GPU, once per image]
  2. Lens correction (GPU -> CPU -> GPU)          [cv2, once per image]
  3. Display output (GPU -> numpy)                [GPU -> CPU, each frame]

Taichi has a single global instance per process - all kernel calls are
serialized. This QThread processes requests sequentially, which is safe.
"""
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
from raw_alchemy.backend import ndarray as backend_ndarray, ti
from raw_alchemy.math_ops import (
    apply_matrix_inplace,
    float_to_uint8_gpu,
    resize_float_to_uint8_gpu,
)
from raw_alchemy.pipeline.request import ProcessRequest, ProcessorParams
from raw_alchemy.pipeline.cache_manager import ImageCacheManager, CachedImage
from raw_alchemy.pipeline.executor import PreviewExecutor
from raw_alchemy.pipeline.ops import build_op_list
from raw_alchemy.gpu_buffer import GpuImage
from raw_alchemy.onnx.denoiser import denoise_raw, clear_session as denoise_clear_session


PROXY_TARGET_PIXELS = 3_000_000
PROXY_MIN_SOURCE_PIXELS = 4_000_000
FULL_REFINE_IDLE_SECONDS = 1.0


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
        self.preview_executor: Optional[PreviewExecutor] = None
        self._executor_path: Optional[str] = None
        self._executor_params: ProcessorParams = {}
        self._executor_using_proxy = False
        self._executor_corrected_source: Optional[np.ndarray] = None
        self._last_executor_source_mode: Optional[str] = None
        self._full_refine_request: Optional[ProcessRequest] = None
        self._full_refine_due = 0.0
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
            request = None
            with self.lock:
                now = time.perf_counter()
                request = self.pending_request
                if request:
                    self.pending_request = None
                    self._preload_queue.clear()
                elif self._export_queue:
                    request = self._export_queue.pop(0)
                elif self._preload_queue:
                    request = self._preload_queue.pop(0)
                elif self._full_refine_request and now >= self._full_refine_due:
                    request = self._full_refine_request
                    self._full_refine_request = None

            if not request:
                time.sleep(0.05)
                continue

            try:
                if '_export' in request.params:
                    self._do_export(request)
                elif '_preload' in request.params:
                    with self.lock:
                        has_user_request = self.pending_request is not None
                    if has_user_request:
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
        """Preload RAW into CPU cache with lens correction."""
        path = request.path
        if self.cache_manager.get(path):
            return

        try:
            img, exif_data, _exif_meta = self._rawpy_to_prophoto(path)
            proxy_img = self._make_proxy(img)

            new_cache_item = CachedImage(
                path=path,
                linear_data=img,
                proxy_linear=proxy_img,
                exif_data=exif_data,
                lens_key=None,
                exif_metadata=_exif_meta,
            )

            # Precompute lens correction on CPU
            if exif_data:
                try:
                    import cv2
                    from raw_alchemy.lensfun_wrapper import compute_lens_distortion_map

                    dist_result = compute_lens_distortion_map(
                        img,
                        camera_maker=exif_data.get('camera_maker'),
                        camera_model=exif_data.get('camera_model'),
                        lens_maker=exif_data.get('lens_maker'),
                        lens_model=exif_data.get('lens_model'),
                        focal_length=exif_data.get('focal_length', 0),
                        aperture=exif_data.get('aperture', 0),
                        crop_factor=exif_data.get('crop_factor'),
                    )
                    if dist_result is not None:
                        coords, oob_mask, vignette_img = dist_result
                        corrected = np.empty_like(vignette_img)
                        for ch in range(3):
                            map_x = coords[:, :, ch, 0]
                            map_y = coords[:, :, ch, 1]
                            corrected[:, :, ch] = cv2.remap(
                                vignette_img[:, :, ch], map_x, map_y,
                                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        corrected[oob_mask] = 0
                        new_cache_item.corrected_data = corrected
                        new_cache_item.proxy_corrected_data = self._make_proxy(corrected)
                        new_cache_item.lens_key = (True, None, False)
                        new_cache_item.proxy_lens_key = (True, None, False)
                        new_cache_item.update_size()
                except Exception as e:
                    logger.debug(f"[Worker] Preload lens correction skipped: {e}")

            self.cache_manager.put(path, new_cache_item)
            logger.info(f"[Worker] Preloaded: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})"
                         f"{' +lens' if new_cache_item.corrected_data is not None else ''}")

        except Exception as e:
            logger.warning(f"Preload failed for {path}: {e}")

    def _do_load(self, request: ProcessRequest):
        """Load RAW and upload to GPU."""
        path = request.path

        # Check CPU cache first
        cached_item = self.cache_manager.get(path)

        if cached_item:
            logger.info(f"[Worker] Cache Hit: {os.path.basename(path)}")
            self.cpu_linear = cached_item.linear_data
            self.cpu_proxy_linear = cached_item.proxy_linear
            self.exif_data = cached_item.exif_data
            self.exif_metadata = cached_item.exif_metadata
            self.cached_lens_key = cached_item.lens_key
            self.cached_proxy_lens_key = cached_item.proxy_lens_key

            if cached_item.corrected_data is not None:
                self.cpu_corrected = cached_item.corrected_data
            else:
                self.cpu_corrected = None
            self.cpu_proxy_corrected = cached_item.proxy_corrected_data

            # Restore denoise cache
            self.cached_denoise_full = cached_item.denoise_full
            self.cached_denoise_original = cached_item.denoise_original
            self.last_denoise_key = cached_item.denoise_key

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
            if cached_item and cached_item.corrected_data is not None:
                # Cache hit with lens correction: only invalidate downstream
                self.gpu_graded.clear()
                self._gpu_uint8 = None
                if self.preview_executor is not None:
                    self.preview_executor.clear_cache()
                self.last_metering_key = None
                self.cached_auto_ev = 0.0
            else:
                self._invalidate_pipeline_caches()
            self.current_path = path

        self.load_complete.emit(path, request.request_id)

    def _invalidate_pipeline_caches(self):
        """Invalidate all pipeline stage caches (but not denoise/sharpen)."""
        self.cached_lens_key = None
        self.cached_proxy_lens_key = None
        self.cpu_corrected = None
        self.cpu_proxy_corrected = None
        self.gpu_graded.clear()
        self._gpu_uint8 = None
        self._full_refine_request = None
        if self.preview_executor is not None:
            self.preview_executor.clear_cache()
        self._last_executor_source_mode = None
        self.cached_auto_ev = 0.0
        # Force immediate GPU memory release
        gc.collect()
        self.last_metering_key = None

    def _release_gpu_buffers(self):
        """Release all GPU buffers while CUDA context is still valid (on worker thread)."""
        try:
            self.gpu_graded.clear()
            self._gpu_uint8 = None
            self._full_refine_request = None
            if self.preview_executor is not None:
                self.preview_executor.clear_cache()
            self._last_executor_source_mode = None
            logger.debug("[Worker] GPU buffers released.")
        except Exception as e:
            logger.debug(f"[Worker] GPU buffer release error (non-critical): {e}")

    def get_cached_for_export(self) -> Optional[dict]:
        """Return cached data for the current image, for use by single-image export.

        Called from the main thread. The numpy arrays are read-only references
        so no copy is made here 鈥?the export thread should treat them as
        immutable (they are only ever replaced, never mutated in-place).
        """
        if self.cpu_corrected is None:
            return None
        return {
            'corrected': self.cpu_corrected,
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
    def _make_output_key(params):
        return (
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
            params.get('viewport_size'),
            round(float(params.get('preview_zoom', 1.0) or 1.0), 3),
            round(float(params.get('device_pixel_ratio', 1.0) or 1.0), 2),
        )

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

    def _get_preview_executor(self) -> PreviewExecutor:
        if self.preview_executor is None:
            self.preview_executor = PreviewExecutor(
                denoiser=self._executor_denoise,
                lens_corrector=self._executor_lens_correct,
                auto_gain_resolver=self._executor_auto_gain,
                round_exposure_gain=True,
            )
        return self.preview_executor

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
            old_size = cached_item.size_mb
            if self._executor_using_proxy:
                cached_item.proxy_corrected_data = corrected
                cached_item.proxy_lens_key = current_lens_key
            else:
                cached_item.corrected_data = corrected
                cached_item.lens_key = current_lens_key
                if cached_item.proxy_linear is not None:
                    cached_item.proxy_corrected_data = self._make_proxy(corrected)
                    cached_item.proxy_lens_key = current_lens_key
            cached_item.update_size()
            with self.cache_manager.lock:
                self.cache_manager.current_memory_mb += cached_item.size_mb - old_size

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

    def _invalidate_executor_prefix_if_needed(self, params: ProcessorParams):
        """Semantic invalidation of the executor's pipeline caches.

        Division of labor (T7.1): PreviewExecutor.set_source only clears its
        caches when the source *array object* actually changes, so that the
        prefix cache survives across requests that reuse the same source.
        This method owns the semantic invalidations the executor cannot see:
        proxy/full source-mode switches, and lens/denoise state changes whose
        cached op outputs (denoise/lens_correct callbacks) depend on worker
        state not captured by the op parameters themselves.
        """
        source_mode = 'proxy' if self._executor_using_proxy else 'full'
        if self.preview_executor is None:
            # Executor not created yet (first request): nothing to clear, but
            # still record the mode its cache is about to be built from, so
            # the *second* request does not spuriously invalidate it.
            self._last_executor_source_mode = source_mode
            return

        denoise_enabled = params.get('denoise_enabled', False)
        desired_lens_key = (
            params.get('lens_correct'),
            params.get('custom_db_path'),
            denoise_enabled,
        )
        current_lens_key = self.cached_proxy_lens_key if self._executor_using_proxy else self.cached_lens_key

        needs_clear = False
        if source_mode != self._last_executor_source_mode:
            needs_clear = True
        if denoise_enabled and self.cached_denoise_full is None:
            needs_clear = True
        if desired_lens_key != current_lens_key:
            needs_clear = True

        if needs_clear:
            self.preview_executor.clear_cache()
        self._last_executor_source_mode = source_mode

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
            output_key = self._make_output_key(params)
            cached_item = self.cache_manager.get(request.path)
            if cached_item and cached_item.output_key == output_key and cached_item.output_uint8 is not None:
                logger.info(f"[Worker] Output Cache Hit: {os.path.basename(request.path)}")
                self.last_preview_source = preview_source
                self.preview_source_changed.emit(preview_source)
                if use_proxy:
                    self._schedule_full_refinement(request, params)
                self.result_ready.emit(
                    cached_item.output_uint8,
                    request.path,
                    request.request_id,
                    cached_item.output_ev,
                    cached_item.output_source_size,
                )
                return

            self._executor_path = request.path
            self._executor_params = params
            self._executor_using_proxy = use_proxy
            self._prepare_executor_source_state(params)
            self._invalidate_executor_prefix_if_needed(params)

            ops = build_op_list(params)
            executor_source = self._select_executor_source(use_proxy)
            result = self._get_preview_executor().run_result(ops, source=executor_source)
            self._sync_executor_export_source(params)

            processed = result.image
            if processed.dtype != np.float32:
                processed = processed.astype(np.float32)
            if not processed.flags['C_CONTIGUOUS']:
                processed = np.ascontiguousarray(processed)

            applied_ev = float(result.applied_ev)
            self.last_applied_ev = applied_ev

        except Exception as e:
            logger.error(f"[Worker] Error in unified pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Processing error: {str(e)}")
            return

        try:
            t_out = time.perf_counter()
            self.gpu_graded.upload(processed)
            source_h, source_w = processed.shape[:2]
            target_w, target_h = self._make_preview_target_size(source_w, source_h, params)

            if self._gpu_uint8 is None or self._gpu_uint8.shape != (target_h, target_w, 3):
                self._gpu_uint8 = backend_ndarray(dtype=ti.u8, shape=(target_h, target_w, 3))

            if target_w == source_w and target_h == source_h:
                float_to_uint8_gpu(self.gpu_graded.arr, self._gpu_uint8)
            else:
                resize_float_to_uint8_gpu(self.gpu_graded.arr, self._gpu_uint8)

            img_uint8 = self._gpu_uint8.to_numpy()
            output_source_size = (source_w, source_h)

            t_end = time.perf_counter()
            self.last_preview_source = preview_source
            self.preview_source_changed.emit(preview_source)
            logger.info(
                f"[Worker] Unified pipeline: {(t_end - t_start)*1000:.0f}ms total, "
                f"output: {(t_end - t_out)*1000:.0f}ms "
                f"source={preview_source} "
                f"({source_w}x{source_h} -> {target_w}x{target_h}, ops={len(ops)})"
            )

            cached_item = self.cache_manager.get(request.path)
            if cached_item:
                old_size = cached_item.size_mb
                cached_item.output_uint8 = img_uint8.copy()
                cached_item.output_key = output_key
                cached_item.output_ev = applied_ev
                cached_item.output_source_size = output_source_size
                cached_item.update_size()
                with self.cache_manager.lock:
                    self.cache_manager.current_memory_mb += cached_item.size_mb - old_size

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
