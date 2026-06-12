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
import taichi as ti
import numpy as np
import colour
from typing import Optional
from PySide6.QtCore import QThread, Signal
from loguru import logger

from raw_alchemy import utils, metering, config
from raw_alchemy.math_ops import (
    apply_matrix_inplace,
    apply_lut_inplace,
    apply_saturation_contrast_inplace,
    apply_white_balance_inplace,
    apply_highlight_shadow_inplace,
    apply_gain_inplace,
    linear_to_srgb_inplace,
    clip_inplace,
    max_inplace,
    apply_geometry_gpu,
    apply_crop_gpu,
    apply_crop_pixels_gpu,
    log_encode_gpu,
    float_to_uint8_gpu,
    resize_float_to_uint8_gpu,
    fused_expose_adjust_grade,
    downsample_gpu,
)
from raw_alchemy.pipeline.request import ProcessRequest, ProcessorParams
from raw_alchemy.pipeline.cache_manager import ImageCacheManager, CachedImage
from raw_alchemy.gpu_buffer import GpuImage
from raw_alchemy.onnx.denoiser import denoise_raw, clear_session as denoise_clear_session


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

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self._wake = threading.Condition(self.lock)

        # Request management
        self.pending_request: Optional[ProcessRequest] = None
        self.active_request: Optional[ProcessRequest] = None
        self.current_request_id = 0
        self._preload_queue: list = []

        # LRU Cache (CPU-side, for RAW decode results)
        self.cache_manager = ImageCacheManager()
        self._warmed_up = False

        # ===== CPU stage caches =====
        self.cpu_linear: Optional[np.ndarray] = None
        self.cpu_corrected: Optional[np.ndarray] = None  # After lens correction

        # ===== GPU Buffers (3 total — immutable source + temp + output) =====
        self.gpu_corrected = GpuImage()    # Lens-corrected, uploaded once
        self.gpu_cropped = GpuImage()      # After geo+crop (immutable ProPhoto Linear source)
        self.gpu_preview_source = GpuImage()  # Viewport-sized linear source for interactive grading
        self.gpu_graded = GpuImage()       # Output: always rebuilt from a clean linear source
        self.gpu_tile_source = GpuImage()   # Full-detail tiled preview source
        self.gpu_tile_graded = GpuImage()   # Full-detail tiled preview output

        # ===== Cache Keys =====
        self.cached_lens_key = None
        self.last_geo_crop_key = None   # Combined geometry+perspective+crop
        self.last_preview_source_key = None
        self.last_grading_key = None    # Combined exposure+adjust+grading

        # Metering Cache
        self.cached_auto_ev = 0.0
        self.last_metering_key = None

        self.exif_data = None
        self.exif_metadata = None  # Full metadata dict for export EXIF writing
        self.last_applied_ev = 0.0
        self.current_path = None

        # LUT Cache
        self.cached_lut_path = None
        self.cached_lut_table = None
        self.cached_lut_domain_min = None
        self.cached_lut_domain_max = None
        self.cached_lut_is_3d = False

        # Denoising Cache
        self.cached_denoise_original = None
        self.cached_denoise_full = None
        self.last_denoise_key = None

        # Sharpening Cache
        self.cached_graded_clean = None   # grading output before sharpening
        self.cached_sharpened = None
        self.last_sharpen_key = None
        self._graded_clean_pending = False
        self._gpu_graded_is_sharpened = False

        self._should_stop = False
        self._gpu_uint8 = None      # Pre-allocated GPU uint8 buffer (ti.ndarray)
        self._gpu_tile_uint8 = None

    def stop_and_cleanup(self):
        """Signal the worker to stop and wait for GPU cleanup."""
        with self.lock:
            self._should_stop = True
            self._wake.notify_all()
        if self.isRunning():
            self.wait(5000)  # Wait up to 5s for thread to finish

    def load_image(self, path: str):
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self.pending_request = ProcessRequest(path, {'_load': True}, request_id)
            self._wake.notify_all()
        if not self.isRunning():
            self.start()
        return request_id

    def preload_image(self, path: str):
        with self.lock:
            if not any(p.path == path for p in self._preload_queue):
                self._preload_queue.append(ProcessRequest(path, {'_preload': True}, -1))
                self._wake.notify_all()
        if not self.isRunning():
            self.start()

    @staticmethod
    def _freeze_param_value(value):
        if isinstance(value, dict):
            return tuple(sorted((k, ImageProcessor._freeze_param_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(ImageProcessor._freeze_param_value(v) for v in value)
        if isinstance(value, np.ndarray):
            return tuple(value.reshape(-1).tolist())
        return value

    @staticmethod
    def _dedupe_params(params: ProcessorParams):
        return tuple(sorted(
            (k, ImageProcessor._freeze_param_value(v))
            for k, v in params.items()
            if k not in ('_load', '_preload')
        ))

    def update_preview(self, path: str, params: ProcessorParams):
        with self.lock:
            incoming_params = self._dedupe_params(params)
            if self.pending_request is not None:
                pending_params = self._dedupe_params(self.pending_request.params)
                if self.pending_request.path == path and pending_params == incoming_params:
                    return self.pending_request.request_id
            if self.active_request is not None:
                active_params = self._dedupe_params(self.active_request.params)
                if self.active_request.path == path and active_params == incoming_params:
                    return self.active_request.request_id
            self.current_request_id += 1
            request_id = self.current_request_id
            self.pending_request = ProcessRequest(path, params, request_id)
            self._wake.notify_all()
        if not self.isRunning():
            self.start()
        return request_id

    def _has_newer_pending_request(self, request: ProcessRequest) -> bool:
        with self.lock:
            pending = self.pending_request
            return pending is not None and pending.request_id > request.request_id

    def run(self):
        # Initialize Taichi on this thread so CUDA context is valid here (once only)
        from raw_alchemy.math_ops import init_taichi, warmup
        init_taichi()
        if not self._warmed_up:
            warmup()
            self._warmed_up = True

        # Permanent worker loop — thread stays alive until app closes
        while True:
            with self.lock:
                while (self.pending_request is None and not self._preload_queue
                       and not self._should_stop):
                    self._wake.wait(timeout=0.5)

                if self._should_stop:
                    break

                request = self.pending_request
                if request:
                    self.pending_request = None
                    self._preload_queue.clear()
                elif self._preload_queue:
                    request = self._preload_queue.pop(0)
                else:
                    request = None

            if not request:
                continue

            try:
                with self.lock:
                    self.active_request = request
                if '_preload' in request.params:
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
            finally:
                with self.lock:
                    if self.active_request is request:
                        self.active_request = None

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
        from raw_alchemy.colorspace_matrices import cam_to_prophoto_matrix
        from raw_alchemy.rawspeed import try_decode, XTRANS_PATTERN

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

            # Camera→ProPhoto matrix, derived analytically (dcraw/darktable
            # algorithm). Matches the old lstsq fit to within 0.3% per cell.
            cam_to_prophoto = cam_to_prophoto_matrix(xyz_to_cam)
            apply_matrix_inplace(rgb, cam_to_prophoto)

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

            cam_to_prophoto = cam_to_prophoto_matrix(xyz_to_cam)
            apply_matrix_inplace(rgb, cam_to_prophoto)

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

    def _do_preload(self, request: ProcessRequest):
        """Preload RAW into CPU cache with lens correction."""
        path = request.path
        if self.cache_manager.get(path):
            return

        try:
            img, exif_data, _exif_meta = self._rawpy_to_prophoto(path)

            new_cache_item = CachedImage(
                path=path,
                linear_data=img,
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
                                cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0)
                        corrected[oob_mask] = 0
                        new_cache_item.corrected_data = corrected
                        new_cache_item.lens_key = (True, None, False)
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
            self.exif_data = cached_item.exif_data
            self.exif_metadata = cached_item.exif_metadata
            self.cached_lens_key = cached_item.lens_key

            if cached_item.corrected_data is not None:
                self.cpu_corrected = cached_item.corrected_data
                self.gpu_corrected.upload(cached_item.corrected_data)
            else:
                self.cpu_corrected = None
                self.gpu_corrected.clear()

            # Restore denoise/sharpen caches
            self.cached_denoise_full = cached_item.denoise_full
            self.cached_denoise_original = cached_item.denoise_original
            self.last_denoise_key = cached_item.denoise_key
            self.cached_sharpened = cached_item.sharpened_data
            self.last_sharpen_key = cached_item.sharpen_key

        else:
            logger.info(f"[Worker] Cache Miss - Loading: {os.path.basename(path)}")

            try:
                img_np, self.exif_data, self.exif_metadata = self._rawpy_to_prophoto(path)

                # Keep on CPU (GPU upload only when needed for processing)
                self.cpu_linear = img_np

                # Cache on CPU
                new_cache_item = CachedImage(
                    path=path,
                    linear_data=img_np,
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
                self.last_geo_crop_key = None
                self.gpu_cropped.clear()
                self.last_preview_source_key = None
                self.gpu_preview_source.clear()
                self.last_grading_key = None
                self.cached_graded_clean = None
                self._graded_clean_pending = False
                self._gpu_graded_is_sharpened = False
                self.gpu_graded.clear()
                self.gpu_tile_source.clear()
                self.gpu_tile_graded.clear()
                try:
                    ti.sync()
                except Exception:
                    pass
                self._gpu_uint8 = None
                self._gpu_tile_uint8 = None
                self.last_metering_key = None
                self.cached_auto_ev = 0.0
            else:
                self._invalidate_pipeline_caches()
            self.current_path = path

        self.load_complete.emit(path, request.request_id)

    def _invalidate_pipeline_caches(self):
        """Invalidate all pipeline stage caches (but not denoise/sharpen)."""
        self.cached_lens_key = None
        self.cpu_corrected = None
        self.gpu_corrected.clear()
        self.last_geo_crop_key = None
        self.gpu_cropped.clear()
        self.last_preview_source_key = None
        self.gpu_preview_source.clear()
        self.last_grading_key = None
        self.cached_graded_clean = None
        self._graded_clean_pending = False
        self._gpu_graded_is_sharpened = False
        self.gpu_graded.clear()
        self.gpu_tile_source.clear()
        self.gpu_tile_graded.clear()
        try:
            ti.sync()
        except Exception:
            pass
        self._gpu_uint8 = None
        self._gpu_tile_uint8 = None
        self.cached_auto_ev = 0.0
        # Force immediate GPU memory release
        gc.collect()
        self.last_metering_key = None

    def _release_gpu_buffers(self):
        """Release all GPU buffers while CUDA context is still valid (on worker thread)."""
        try:
            self.gpu_corrected.clear()
            self.gpu_cropped.clear()
            self.gpu_preview_source.clear()
            self.gpu_graded.clear()
            self.gpu_tile_source.clear()
            self.gpu_tile_graded.clear()
            try:
                ti.sync()
            except Exception:
                pass
            self._gpu_uint8 = None
            self._gpu_tile_uint8 = None
            logger.debug("[Worker] GPU buffers released.")
        except Exception as e:
            logger.debug(f"[Worker] GPU buffer release error (non-critical): {e}")

    def get_cached_for_export(self) -> Optional[dict]:
        """Return cached data for the current image, for use by single-image export.

        Called from the main thread. The numpy arrays are read-only references
        so no copy is made here — the export thread should treat them as
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

    # =================================================================
    # Processing Pipeline (GPU-resident)
    # =================================================================

    @staticmethod
    def _make_output_key(params):
        raw_max_preview_pixels = params.get('max_preview_pixels', 12_000_000)
        max_preview_pixels = (
            12_000_000 if raw_max_preview_pixels is None
            else int(raw_max_preview_pixels)
        )
        full_detail = bool(params.get('_detail_preview', False)) and max_preview_pixels <= 0
        viewport_key = None if full_detail else params.get('viewport_size')
        zoom_key = 1.0 if full_detail else round(float(params.get('preview_zoom', 1.0) or 1.0), 3)
        dpr_key = 1.0 if full_detail else round(float(params.get('device_pixel_ratio', 1.0) or 1.0), 2)
        return (
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
            viewport_key,
            zoom_key,
            dpr_key,
            max_preview_pixels,
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

        raw_max_preview_pixels = params.get('max_preview_pixels', 12_000_000)
        max_preview_pixels = 12_000_000 if raw_max_preview_pixels is None else int(raw_max_preview_pixels)
        if max_preview_pixels > 0 and target_w * target_h > max_preview_pixels:
            scale = (max_preview_pixels / float(target_w * target_h)) ** 0.5
            target_w = max(1, int(target_w * scale))
            target_h = max(1, int(target_h * scale))

        return target_w, target_h

    def _cache_output_result(self, path: str, img_uint8: np.ndarray, output_key,
                             applied_ev: float, output_source_size):
        cached_item = self.cache_manager.get(path)
        if not cached_item:
            return

        old_size = cached_item.size_mb
        cached_item.output_uint8 = img_uint8.copy()
        cached_item.output_key = output_key
        cached_item.output_ev = applied_ev
        cached_item.output_source_size = output_source_size
        cached_item.update_size()
        with self.cache_manager.lock:
            self.cache_manager.current_memory_mb += cached_item.size_mb - old_size

    def _grade_source_to(self, source_buf: GpuImage, dst_buf: GpuImage,
                         final_exposure_gain: float, params: ProcessorParams):
        """Apply exposure, color, log/LUT and output transform to a GPU source."""
        log_space = params.get('log_space')
        lut_path = params.get('lut_path')
        wb_temp = params.get('wb_temp', 0.0)
        wb_tint = params.get('wb_tint', 0.0)
        highlight = params.get('highlight', 0.0)
        shadow = params.get('shadow', 0.0)
        sat = params.get('saturation', 1.0)
        con = params.get('contrast', 1.0)

        use_fused = (not log_space or log_space == 'None') and not lut_path

        if use_fused:
            source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
            luma = utils.get_luminance_coeffs(source_cs).astype(np.float32)
            M_srgb = colour.matrix_RGB_to_RGB(
                source_cs, colour.RGB_COLOURSPACES['sRGB'])

            t_val = wb_temp * 0.005
            g_val = wb_tint * 0.005

            dst_buf._allocate(source_buf.height, source_buf.width, 3)

            fused_expose_adjust_grade(
                source_buf.arr, dst_buf.arr,
                gain=float(final_exposure_gain),
                wb_r=float(1.0 + t_val),
                wb_g=float(1.0 - g_val),
                wb_b=float(1.0 - t_val),
                highlight=float(highlight / 100.0),
                shadow=float(shadow / 100.0),
                saturation=float(sat),
                contrast=float(con),
                pivot=0.18,
                luma_coeffs=luma,
                matrix=M_srgb,
                apply_srgb=True,
            )
            return

        dst_buf.copy_from(source_buf)
        arr = dst_buf.arr

        apply_gain_inplace(arr, float(final_exposure_gain))

        if wb_temp != 0.0 or wb_tint != 0.0:
            t_val = wb_temp * 0.005
            g_val = wb_tint * 0.005
            apply_white_balance_inplace(arr,
                                        float(1.0 + t_val),
                                        float(1.0 - g_val),
                                        float(1.0 - t_val))

        source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
        luma = utils.get_luminance_coeffs(source_cs).astype(np.float32)

        if highlight != 0.0 or shadow != 0.0:
            apply_highlight_shadow_inplace(arr,
                                           float(highlight / 100.0),
                                           float(shadow / 100.0),
                                           luma)

        apply_saturation_contrast_inplace(arr,
                                          float(sat), float(con), 0.18,
                                          luma)

        if log_space and log_space != 'None':
            log_color_space = config.LOG_TO_WORKING_SPACE.get(log_space)
            log_curve = config.LOG_ENCODING_MAP.get(log_space, log_space)

            if log_color_space:
                M = colour.matrix_RGB_to_RGB(
                    colour.RGB_COLOURSPACES['ProPhoto RGB'],
                    colour.RGB_COLOURSPACES[log_color_space])
                apply_matrix_inplace(arr, M)
                max_inplace(arr, 1e-6)
                if not log_encode_gpu(arr, log_curve):
                    graded_np = dst_buf.to_numpy()
                    graded_np = colour.cctf_encoding(graded_np, function=log_curve).astype(np.float32)
                    dst_buf.upload(graded_np)
                    arr = dst_buf.arr

        if lut_path and os.path.exists(lut_path):
            try:
                if lut_path != self.cached_lut_path:
                    logger.info(f"[Worker] Loading LUT: {os.path.basename(lut_path)}")
                    lut = utils.load_lut_cached(lut_path)
                    self.cached_lut_is_3d = isinstance(lut, colour.LUT3D)
                    if self.cached_lut_is_3d:
                        lut_table = np.ascontiguousarray(lut.table.astype(np.float32))
                        self.cached_lut_table = lut_table
                        self.cached_lut_domain_min = np.ascontiguousarray(lut.domain[0].astype(np.float64))
                        self.cached_lut_domain_max = np.ascontiguousarray(lut.domain[1].astype(np.float64))
                    else:
                        self.cached_lut_table = lut
                    self.cached_lut_path = lut_path

                if self.cached_lut_is_3d:
                    apply_lut_inplace(arr,
                                      self.cached_lut_table,
                                      self.cached_lut_domain_min,
                                      self.cached_lut_domain_max)
                else:
                    graded_np = dst_buf.to_numpy()
                    graded_np = self.cached_lut_table.apply(graded_np).astype(np.float32)
                    dst_buf.upload(graded_np)
                    arr = dst_buf.arr
            except Exception as e:
                logger.error(f"[Worker] LUT error: {e}")
                self.cached_lut_path = None

        if not log_space or log_space == 'None':
            M_srgb = colour.matrix_RGB_to_RGB(
                colour.RGB_COLOURSPACES['ProPhoto RGB'],
                colour.RGB_COLOURSPACES['sRGB'])
            apply_matrix_inplace(arr, M_srgb)
            linear_to_srgb_inplace(arr)

        clip_inplace(arr)

    def _render_full_uint8_tiled(self, request: ProcessRequest,
                                 source_buf: GpuImage,
                                 final_exposure_gain: float,
                                 params: ProcessorParams) -> Optional[np.ndarray]:
        """Render a full-size preview in tiles to cap peak GPU memory."""
        h, w = source_buf.height, source_buf.width
        out = np.empty((h, w, 3), dtype=np.uint8)

        max_tile_pixels = int(params.get('tile_preview_pixels', 2_000_000) or 2_000_000)
        sharpen_strength = float(params.get('sharpen_strength', 0.0) or 0.0)
        overlap = int(params.get('tile_overlap_pixels', 32) or 32) if sharpen_strength > 0 else 0
        tile_w = min(w, 1536)
        tile_h = max(1, min(h, max_tile_pixels // max(1, tile_w)))
        logger.info(
            f"[Worker] Tiled detail preview: {w}x{h}, "
            f"tile={tile_w}x{tile_h}, overlap={overlap}"
        )

        sharpen_gpu_fn = None
        if sharpen_strength > 0:
            from raw_alchemy.math_ops import sharpen_gpu as sharpen_gpu_fn

        for y in range(0, h, tile_h):
            if self._has_newer_pending_request(request):
                return None
            th = min(tile_h, h - y)
            for x in range(0, w, tile_w):
                if self._has_newer_pending_request(request):
                    return None
                tw = min(tile_w, w - x)

                src_x = max(0, x - overlap)
                src_y = max(0, y - overlap)
                src_right = min(w, x + tw + overlap)
                src_bottom = min(h, y + th + overlap)
                ext_w = src_right - src_x
                ext_h = src_bottom - src_y
                inner_x = x - src_x
                inner_y = y - src_y

                apply_crop_pixels_gpu(source_buf, self.gpu_tile_source,
                                      src_x, src_y, ext_w, ext_h)
                self._grade_source_to(
                    self.gpu_tile_source,
                    self.gpu_tile_graded,
                    final_exposure_gain,
                    params,
                )
                if sharpen_gpu_fn is not None:
                    sharpen_gpu_fn(self.gpu_tile_graded,
                                   strength=sharpen_strength,
                                   sigma=1.0)

                output_tile = self.gpu_tile_graded
                if inner_x != 0 or inner_y != 0 or ext_w != tw or ext_h != th:
                    apply_crop_pixels_gpu(self.gpu_tile_graded, self.gpu_tile_source,
                                          inner_x, inner_y, tw, th)
                    output_tile = self.gpu_tile_source

                if (self._gpu_tile_uint8 is None
                        or self._gpu_tile_uint8.shape != (th, tw, 3)):
                    if self._gpu_tile_uint8 is not None:
                        try:
                            ti.sync()
                        except Exception:
                            pass
                        self._gpu_tile_uint8 = None
                    self._gpu_tile_uint8 = ti.ndarray(dtype=ti.u8, shape=(th, tw, 3))

                float_to_uint8_gpu(output_tile.arr, self._gpu_tile_uint8)
                if self._has_newer_pending_request(request):
                    return None
                out[y:y + th, x:x + tw, :] = self._gpu_tile_uint8.to_numpy()

        return out

    def _do_process(self, request: ProcessRequest):
        t_start = time.perf_counter()
        logger.debug(f"[Worker] _do_process: {os.path.basename(request.path)}, id={request.request_id}")

        try:
            # Ensure image is loaded
            if self.cpu_linear is None or self.current_path != request.path:
                self._do_load(ProcessRequest(request.path, {'_load': True}, request.request_id))
                if self.cpu_linear is None:
                    return
                if self._has_newer_pending_request(request):
                    return

            params = request.params

            # Fast path: check if cached output matches current params
            output_key = self._make_output_key(params)
            cached_item = self.cache_manager.get(request.path)
            if cached_item and cached_item.output_key == output_key and cached_item.output_uint8 is not None:
                logger.info(f"[Worker] Output Cache Hit: {os.path.basename(request.path)}")
                self.result_ready.emit(cached_item.output_uint8, request.path,
                                       request.request_id, cached_item.output_ev,
                                       cached_item.output_source_size)
                return

            # --- Stage 0: Denoising (CANS RAW V2, packed RAW → ProPhoto Linear) ---
            # When enabled, replaces rawpy demosaicing entirely.
            denoise_enabled = params.get('denoise_enabled', False)
            denoise_cache_key = (request.path, 'denoise')

            if denoise_enabled:
                if denoise_cache_key != self.last_denoise_key or self.cached_denoise_full is None:
                    try:
                        self.denoise_started.emit()

                        logger.info("[Worker] CANS RAW V2 denoise (replaces demosaicing)...")

                        def progress_cb(cur, total):
                            self.denoise_progress.emit(cur, total)

                        denoised = denoise_raw(
                            request.path,
                            exposure_ratio=1.0,
                            progress_callback=progress_cb
                        )

                        self.cached_denoise_original = self.cpu_linear
                        self.cached_denoise_full = denoised
                        self.last_denoise_key = denoise_cache_key
                        # Release ONNX session GPU memory immediately
                        denoise_clear_session()
                        self.denoise_finished.emit()

                    except Exception as e:
                        logger.error(f"[Worker] Denoising failed: {e}")
                        self.denoise_finished.emit()
                        self.cached_denoise_original = None
                        self.cached_denoise_full = None
                        self.last_denoise_key = None

            # Determine the linear source for lens correction
            if denoise_enabled and self.cached_denoise_full is not None:
                # Ensure contiguous memory layout (rotation may produce non-contiguous array)
                linear_source = np.ascontiguousarray(self.cached_denoise_full)
            else:
                linear_source = self.cpu_linear
                if self.last_denoise_key is not None:
                    # Denoise was just disabled, invalidate downstream
                    self.cached_denoise_original = None
                    self.cached_denoise_full = None
                    self.last_denoise_key = None

            # --- Stage 1: Lens Correction (CPU distortion map + GPU remap) ---
            current_lens_key = (params.get('lens_correct'), params.get('custom_db_path'),
                                denoise_enabled)

            if current_lens_key != self.cached_lens_key or not self.gpu_corrected.valid:
                logger.debug("[Worker] Lens correction...")
                if params.get('lens_correct') and self.exif_data:
                    import cv2
                    from raw_alchemy.lensfun_wrapper import compute_lens_distortion_map

                    lf_params = {**self.exif_data}
                    dist_result = compute_lens_distortion_map(
                        linear_source,
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
                                coords[:, :, ch, 0], coords[:, :, ch, 1],
                                cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0)
                        corrected[oob_mask] = 0
                        self.cpu_corrected = corrected
                        self.gpu_corrected.upload(corrected)
                    else:
                        self.cpu_corrected = linear_source
                        self.gpu_corrected.upload(self.cpu_corrected)
                else:
                    self.cpu_corrected = linear_source
                    self.gpu_corrected.upload(self.cpu_corrected)

                self.cached_lens_key = current_lens_key

                # Invalidate downstream caches
                self.last_geo_crop_key = None
                self.gpu_cropped.clear()
                self.last_preview_source_key = None
                self.gpu_preview_source.clear()
                self.last_grading_key = None

                # Update CPU cache
                cached_item = self.cache_manager.get(request.path)
                if cached_item:
                    cached_item.corrected_data = self.cpu_corrected
                    cached_item.lens_key = current_lens_key

            # --- Stage 2-4: Geometry + Perspective + Crop (GPU, cached to CPU) ---
            geo_crop_key = (
                self.cached_lens_key,
                params.get('rotation', 0),
                params.get('flip_horizontal', False),
                params.get('flip_vertical', False),
                params.get('perspective_corners'),
                params.get('crop', (0.0, 0.0, 1.0, 1.0))
            )

            if geo_crop_key != self.last_geo_crop_key or not self.gpu_cropped.valid:
                logger.debug("[Worker] Geometry+Perspective+Crop (GPU)...")

                # Geometry: gpu_corrected → gpu_graded (as temp)
                apply_geometry_gpu(
                    self.gpu_corrected, self.gpu_graded,
                    rotation=params.get('rotation', 0),
                    flip_h=params.get('flip_horizontal', False),
                    flip_v=params.get('flip_vertical', False)
                )

                # Perspective: gpu_graded (temp) → gpu_corrected reused as temp2
                corners = params.get('perspective_corners')
                default_corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
                if corners and corners != default_corners:
                    from raw_alchemy.math_ops import perspective_warp_kernel, compute_perspective_matrix
                    from raw_alchemy.gpu_buffer import GpuImage
                    h, w = self.gpu_graded.height, self.gpu_graded.width
                    _, M_inv = compute_perspective_matrix(corners, w, h)
                    gpu_persp = GpuImage()
                    gpu_persp._allocate(h, w, 3)
                    perspective_warp_kernel(self.gpu_graded.arr, gpu_persp.arr, M_inv)
                    # Crop: gpu_persp → gpu_cropped
                    crop_rect = params.get('crop', (0.0, 0.0, 1.0, 1.0))
                    apply_crop_gpu(gpu_persp, self.gpu_cropped, crop_rect)
                    gpu_persp.clear()
                else:
                    # No perspective — Crop: gpu_graded (temp) → gpu_cropped
                    crop_rect = params.get('crop', (0.0, 0.0, 1.0, 1.0))
                    apply_crop_gpu(self.gpu_graded, self.gpu_cropped, crop_rect)

                # GPU-resident: no CPU download needed
                self.last_geo_crop_key = geo_crop_key
                self.last_preview_source_key = None
                self.last_grading_key = None

            if self._has_newer_pending_request(request):
                return

            # Build/reuse a viewport-sized linear source before color grading.
            # Slider changes should not run exposure/color/LUT kernels over the
            # full RAW resolution when only a screen-sized preview is needed.
            full_source_h, full_source_w = self.gpu_cropped.height, self.gpu_cropped.width
            preview_target_w, preview_target_h = self._make_preview_target_size(
                full_source_w, full_source_h, params)
            preview_source_key = (
                self.last_geo_crop_key,
                preview_target_w,
                preview_target_h,
            )

            if preview_target_w == full_source_w and preview_target_h == full_source_h:
                grading_source = self.gpu_cropped
                self.last_preview_source_key = preview_source_key
                if self.gpu_preview_source.valid:
                    self.gpu_preview_source.clear()
            else:
                if (preview_source_key != self.last_preview_source_key
                        or not self.gpu_preview_source.valid):
                    logger.debug(
                        "[Worker] Downsampling preview source: "
                        f"{full_source_w}x{full_source_h} -> "
                        f"{preview_target_w}x{preview_target_h}"
                    )
                    self.gpu_preview_source._allocate(preview_target_h, preview_target_w, 3)
                    downsample_gpu(
                        self.gpu_cropped.arr,
                        self.gpu_preview_source.arr,
                        full_source_h,
                        full_source_w,
                        preview_target_h,
                        preview_target_w,
                    )
                    self.last_preview_source_key = preview_source_key
                grading_source = self.gpu_preview_source

            # --- Stage 5: Auto Exposure ---
            current_metering_key = (
                self.cached_lens_key,
                params.get('metering_mode', 'matrix')
            )

            final_exposure_gain = 0.0
            applied_ev = 0.0

            if params.get('exposure_mode') == 'Manual':
                applied_ev = params.get('exposure', 0.0)
                final_exposure_gain = 2.0 ** applied_ev
            else:
                if current_metering_key == self.last_metering_key:
                    applied_ev = self.cached_auto_ev
                    final_exposure_gain = 2.0 ** applied_ev
                else:
                    logger.debug("[Worker] Calculating Auto Exposure...")
                    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
                    mode = params.get('metering_mode', 'matrix')
                    strategy = metering.get_metering_strategy(mode)
                    if grading_source is self.gpu_preview_source and self.gpu_preview_source.valid:
                        metering_data = self.gpu_preview_source.to_numpy()
                    else:
                        metering_data = self.cpu_corrected if self.cpu_corrected is not None else self.cpu_linear
                    gain = strategy.calculate_gain(metering_data, source_cs)
                    self.cached_auto_ev = np.log2(gain)
                    self.last_metering_key = current_metering_key
                    applied_ev = self.cached_auto_ev
                    final_exposure_gain = gain

            final_exposure_gain = round(float(final_exposure_gain), 4)

            self.last_applied_ev = applied_ev

            # --- Stage 6-8: Exposure + Adjustments + Grading ---
            # Always start from a clean ProPhoto Linear source. For interactive
            # preview, that source is already downsampled to the viewport target.
            log_space = params.get('log_space')
            lut_path = params.get('lut_path')
            wb_temp = params.get('wb_temp', 0.0)
            wb_tint = params.get('wb_tint', 0.0)
            highlight = params.get('highlight', 0.0)
            shadow = params.get('shadow', 0.0)
            sat = params.get('saturation', 1.0)
            con = params.get('contrast', 1.0)

            grading_key = (
                preview_source_key,
                final_exposure_gain,
                wb_temp, wb_tint,
                highlight, shadow,
                sat, con,
                log_space, lut_path,
            )

            full_pixels = full_source_w * full_source_h
            tile_threshold = int(params.get('tile_preview_threshold', 12_000_000) or 12_000_000)
            sharpen_strength = params.get('sharpen_strength', 0.0)
            if (params.get('_detail_preview', False)
                    and preview_target_w == full_source_w
                    and preview_target_h == full_source_h
                    and full_pixels > tile_threshold):
                t_out = time.perf_counter()
                img_uint8 = self._render_full_uint8_tiled(
                    request,
                    self.gpu_cropped,
                    final_exposure_gain,
                    params,
                )
                if img_uint8 is None:
                    return

                t_end = time.perf_counter()
                logger.info(
                    f"[Worker] Tiled pipeline: {(t_end - t_start)*1000:.0f}ms total, "
                    f"output: {(t_end - t_out)*1000:.0f}ms "
                    f"({full_source_w}x{full_source_h})"
                )

                output_source_size = (full_source_w, full_source_h)
                self._cache_output_result(
                    request.path, img_uint8, output_key, applied_ev,
                    output_source_size)
                self.result_ready.emit(img_uint8, request.path,
                                       request.request_id, applied_ev,
                                       output_source_size)
                return

            if grading_key != self.last_grading_key:
                use_fused = (not log_space or log_space == 'None') and not lut_path

                if use_fused:
                    # Fused path: single kernel gpu_cropped → gpu_graded
                    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
                    luma = utils.get_luminance_coeffs(source_cs).astype(np.float32)
                    M_srgb = colour.matrix_RGB_to_RGB(
                        source_cs, colour.RGB_COLOURSPACES['sRGB'])

                    t_val = wb_temp * 0.005
                    g_val = wb_tint * 0.005

                    self.gpu_graded._allocate(
                        grading_source.height, grading_source.width, 3)

                    fused_expose_adjust_grade(
                        grading_source.arr, self.gpu_graded.arr,
                        gain=float(final_exposure_gain),
                        wb_r=float(1.0 + t_val),
                        wb_g=float(1.0 - g_val),
                        wb_b=float(1.0 - t_val),
                        highlight=float(highlight / 100.0),
                        shadow=float(shadow / 100.0),
                        saturation=float(sat),
                        contrast=float(con),
                        pivot=0.18,
                        luma_coeffs=luma,
                        matrix=M_srgb,
                        apply_srgb=True,
                    )

                else:
                    # Non-fused: copy from grading_source, apply each step in-place
                    self.gpu_graded.copy_from(grading_source)
                    arr = self.gpu_graded.arr

                    # Exposure
                    apply_gain_inplace(arr, float(final_exposure_gain))

                    # White Balance
                    if wb_temp != 0.0 or wb_tint != 0.0:
                        t_val = wb_temp * 0.005
                        g_val = wb_tint * 0.005
                        apply_white_balance_inplace(arr,
                                                    float(1.0 + t_val),
                                                    float(1.0 - g_val),
                                                    float(1.0 - t_val))

                    # Highlight / Shadow
                    if highlight != 0.0 or shadow != 0.0:
                        source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
                        luma = utils.get_luminance_coeffs(source_cs).astype(np.float32)
                        apply_highlight_shadow_inplace(arr,
                                                        float(highlight / 100.0),
                                                        float(shadow / 100.0),
                                                        luma)

                    # Saturation / Contrast
                    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
                    luma = utils.get_luminance_coeffs(source_cs).astype(np.float32)
                    apply_saturation_contrast_inplace(arr,
                                                       float(sat), float(con), 0.18,
                                                       luma)

                    # Log + LUT or sRGB grading
                    if log_space and log_space != 'None':
                        log_color_space = config.LOG_TO_WORKING_SPACE.get(log_space)
                        log_curve = config.LOG_ENCODING_MAP.get(log_space, log_space)

                        if log_color_space:
                            M = colour.matrix_RGB_to_RGB(
                                colour.RGB_COLOURSPACES['ProPhoto RGB'],
                                colour.RGB_COLOURSPACES[log_color_space])
                            apply_matrix_inplace(arr, M)
                            max_inplace(arr, 1e-6)
                            if not log_encode_gpu(arr, log_curve):
                                graded_np = self.gpu_graded.to_numpy()
                                graded_np = colour.cctf_encoding(graded_np, function=log_curve).astype(np.float32)
                                self.gpu_graded.upload(graded_np)
                                arr = self.gpu_graded.arr

                    if lut_path and os.path.exists(lut_path):
                        try:
                            if lut_path != self.cached_lut_path:
                                logger.info(f"[Worker] Loading LUT: {os.path.basename(lut_path)}")
                                lut = utils.load_lut_cached(lut_path)
                                self.cached_lut_is_3d = isinstance(lut, colour.LUT3D)
                                if self.cached_lut_is_3d:
                                    lut_table = np.ascontiguousarray(lut.table.astype(np.float32))
                                    self.cached_lut_table = lut_table
                                    self.cached_lut_domain_min = np.ascontiguousarray(lut.domain[0].astype(np.float64))
                                    self.cached_lut_domain_max = np.ascontiguousarray(lut.domain[1].astype(np.float64))
                                else:
                                    self.cached_lut_table = lut
                                self.cached_lut_path = lut_path

                            if self.cached_lut_is_3d:
                                apply_lut_inplace(arr,
                                                  self.cached_lut_table,
                                                  self.cached_lut_domain_min,
                                                  self.cached_lut_domain_max)
                            else:
                                graded_np = self.gpu_graded.to_numpy()
                                graded_np = self.cached_lut_table.apply(graded_np).astype(np.float32)
                                self.gpu_graded.upload(graded_np)
                        except Exception as e:
                            logger.error(f"[Worker] LUT error: {e}")
                            self.cached_lut_path = None

                    if not log_space or log_space == 'None':
                        M_srgb = colour.matrix_RGB_to_RGB(
                            colour.RGB_COLOURSPACES['ProPhoto RGB'],
                            colour.RGB_COLOURSPACES['sRGB'])
                        apply_matrix_inplace(arr, M_srgb)
                        linear_to_srgb_inplace(arr)

                    clip_inplace(arr)

                self.last_grading_key = grading_key
                self.cached_graded_clean = None
                self._graded_clean_pending = True
                self._gpu_graded_is_sharpened = False

        except Exception as e:
            logger.error(f"[Worker] Error in dev stages: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Processing error: {str(e)}")
            return

        try:

            if self._has_newer_pending_request(request):
                return

            # --- Stage 10: Sharpening (Taichi RL, GPU) ---
            sharpen_strength = params.get('sharpen_strength', 0.0)
            sharpen_key = (grading_key, sharpen_strength)

            if sharpen_strength <= 0 and self._gpu_graded_is_sharpened:
                if self.cached_graded_clean is not None:
                    self.gpu_graded.upload(self.cached_graded_clean)
                else:
                    logger.warning("[Worker] Missing clean grading cache while disabling sharpening")
                    self.last_grading_key = None
                self._gpu_graded_is_sharpened = False

            if sharpen_strength > 0:
                if sharpen_key != self.last_sharpen_key or self.cached_sharpened is None:
                    try:
                        from raw_alchemy.math_ops import sharpen_gpu
                        logger.info(f"[Worker] RL sharpening (strength={sharpen_strength:.2f})...")
                        # Lazy download of graded clean (deferred from fused kernel)
                        if getattr(self, '_graded_clean_pending', False):
                            self.cached_graded_clean = self.gpu_graded.to_numpy()
                            self._graded_clean_pending = False
                        # Restore clean grading data before sharpening (avoid double-sharpen)
                        if self.cached_graded_clean is not None:
                            self.gpu_graded.upload(self.cached_graded_clean)
                        sharpen_gpu(self.gpu_graded, strength=sharpen_strength, sigma=1.0)
                        self.cached_sharpened = self.gpu_graded.to_numpy()
                        self.last_sharpen_key = sharpen_key
                        self._gpu_graded_is_sharpened = True
                    except Exception as e:
                        logger.error(f"[Worker] Sharpening failed: {e}")
                elif self.cached_sharpened is not None:
                    self.gpu_graded.upload(self.cached_sharpened)
                    self._gpu_graded_is_sharpened = True

            # --- Stage 11: Preview Output ---
            # GPU float32 -> uint8 on GPU, optionally resized to preview target.
            t_out = time.perf_counter()
            source_h, source_w = self.gpu_graded.height, self.gpu_graded.width
            target_w, target_h = source_w, source_h

            # Pre-allocate GPU uint8 buffer (avoids Taichi's implicit upload of numpy dst)
            if self._gpu_uint8 is None or self._gpu_uint8.shape != (target_h, target_w, 3):
                if self._gpu_uint8 is not None:
                    try:
                        ti.sync()
                    except Exception:
                        pass
                    self._gpu_uint8 = None
                self._gpu_uint8 = ti.ndarray(dtype=ti.u8, shape=(target_h, target_w, 3))

            # GPU→GPU kernel (no CPU transfer)
            if target_w == source_w and target_h == source_h:
                float_to_uint8_gpu(self.gpu_graded.arr, self._gpu_uint8)
            else:
                resize_float_to_uint8_gpu(self.gpu_graded.arr, self._gpu_uint8)

            # Single GPU→CPU download of uint8 preview.
            if self._has_newer_pending_request(request):
                return

            img_uint8 = self._gpu_uint8.to_numpy()
            output_source_size = (full_source_w, full_source_h)

            t_end = time.perf_counter()
            logger.info(
                f"[Worker] Pipeline: {(t_end - t_start)*1000:.0f}ms total, "
                f"output: {(t_end - t_out)*1000:.0f}ms "
                f"({source_w}x{source_h} -> {target_w}x{target_h})"
            )

            # Cache final output for instant switching
            self._cache_output_result(
                request.path, img_uint8, output_key, applied_ev,
                output_source_size)

            self.result_ready.emit(img_uint8, request.path,
                                   request.request_id, applied_ev,
                                   output_source_size)

        except Exception as e:
            logger.error(f"[Worker] Error in grading/output: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Output error: {str(e)}")
