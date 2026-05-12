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
    log_encode_gpu,
    float_to_uint8_gpu,
    fused_expose_adjust_grade,
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
      result_ready(img_uint8, path, request_id, applied_ev)
      load_complete(path, request_id)
    """
    result_ready = Signal(np.ndarray, str, int, float)
    load_complete = Signal(str, int)
    error_occurred = Signal(str)
    denoise_progress = Signal(int, int)
    denoise_started = Signal()
    denoise_finished = Signal()

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

        # Request management
        self.pending_request: Optional[ProcessRequest] = None
        self.current_request_id = 0

        # LRU Cache (CPU-side, for RAW decode results)
        self.cache_manager = ImageCacheManager(max_items=5, max_memory_mb=1500)
        self._warmed_up = False

        # ===== CPU stage caches =====
        self.cpu_linear: Optional[np.ndarray] = None
        self.cpu_corrected: Optional[np.ndarray] = None  # After lens correction

        # ===== GPU Buffers (3 total — immutable source + temp + output) =====
        self.gpu_corrected = GpuImage()    # Lens-corrected, uploaded once
        self.gpu_cropped = GpuImage()      # After geo+crop (immutable ProPhoto Linear source)
        self.gpu_graded = GpuImage()       # Output: always rebuilt from gpu_cropped

        # ===== Cache Keys =====
        self.cached_lens_key = None
        self.last_geo_crop_key = None   # Combined geometry+perspective+crop
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

        self._should_stop = False
        self._gpu_uint8 = None      # Pre-allocated GPU uint8 buffer (ti.ndarray)

    def stop_and_cleanup(self):
        """Signal the worker to stop and wait for GPU cleanup."""
        self._should_stop = True
        if self.isRunning():
            self.wait(5000)  # Wait up to 5s for thread to finish

    def load_image(self, path: str):
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self.pending_request = ProcessRequest(path, {'_load': True}, request_id)
        if not self.isRunning():
            self.start()
        return request_id

    def preload_image(self, path: str):
        with self.lock:
            if self.pending_request is None:
                self.pending_request = ProcessRequest(path, {'_preload': True}, -1)
        if not self.isRunning():
            self.start()

    def update_preview(self, path: str, params: ProcessorParams):
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self.pending_request = ProcessRequest(path, params, request_id)
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

        # Permanent worker loop — thread stays alive until app closes
        while not self._should_stop:
            with self.lock:
                request = self.pending_request
                if request:
                    self.pending_request = None

            if not request:
                time.sleep(0.05)
                continue

            try:
                if '_preload' in request.params:
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

    def _rawpy_to_prophoto(self, path: str):
        """Decode RAW via rawpy + RCD demosaic -> ProPhoto Linear float32.

        rawpy is used ONLY for reading raw sensor data and metadata (no postprocess).

        Returns:
            (img, exif_data, exif_metadata) where img is (H, W, 3) float32 ProPhoto Linear.
        """
        import rawpy
        from raw_alchemy.demosaic import rcd_demosaic, get_dcraw_filters
        from raw_alchemy.exif import extract_lens_exif
        from raw_alchemy.onnx.denoiser import _apply_flip
        from raw_alchemy.colorspace_matrices import cam_to_prophoto_matrix

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
                # XYZ_D65 -> Camera matrix (4x3). Needed for analytic
                # cam->ProPhoto derivation; replaces the old approach of
                # calling postprocess() twice + lstsq fit (which was eating
                # 0.5-3s per image).
                xyz_to_cam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)
                if is_xtrans:
                    xtrans_pat = cfa_pattern.copy()

        g = wb[1] if wb[1] > 0 else 1.0

        from raw_alchemy.core import highlight_inpaint_opposed, subtract_black_level, fix_hot_pixels

        if is_bayer:
            bayer_norm = subtract_black_level(sensor_raw, bl, wl, cfa_pattern)
            fix_hot_pixels(bayer_norm, cfa_pattern)
            highlight_inpaint_opposed(bayer_norm, cfa_pattern, wb)

            dcraw_filters = get_dcraw_filters(cfa_pattern)
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

        np.clip(rgb, 0.0, 1.0, out=rgb)

        # Extract EXIF via pyexiv2
        exif_data, exif_metadata = extract_lens_exif(path, None)

        return rgb, exif_data, exif_metadata

    def _do_preload(self, request: ProcessRequest):
        """Preload RAW into CPU cache only."""
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
            self.cache_manager.put(path, new_cache_item)
            logger.info(f"[Worker] Preloaded: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

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
        self.last_grading_key = None
        self.cached_graded_clean = None
        self.gpu_graded.clear()
        self._gpu_uint8 = None
        self.cached_auto_ev = 0.0
        # Force immediate GPU memory release
        gc.collect()
        self.last_metering_key = None

    def _release_gpu_buffers(self):
        """Release all GPU buffers while CUDA context is still valid (on worker thread)."""
        try:
            self.gpu_corrected.clear()
            self.gpu_cropped.clear()
            self.gpu_graded.clear()
            self._gpu_uint8 = None
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

    def _do_process(self, request: ProcessRequest):
        t_start = time.perf_counter()
        logger.debug(f"[Worker] _do_process: {os.path.basename(request.path)}, id={request.request_id}")

        try:
            # Ensure image is loaded
            if self.cpu_linear is None or self.current_path != request.path:
                self._do_load(ProcessRequest(request.path, {'_load': True}, request.request_id))
                if self.cpu_linear is None:
                    return

            params = request.params

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
                    from raw_alchemy.lensfun_wrapper import compute_lens_distortion_map
                    from raw_alchemy.math_ops import lens_remap_gpu

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
                        coords, oob_mask, corrected_img = dist_result
                        # Upload source to GPU, then remap on GPU
                        self.gpu_corrected.upload(corrected_img)
                        # Use a temp buffer for remap output
                        from raw_alchemy.gpu_buffer import GpuImage
                        gpu_temp = GpuImage()
                        lens_remap_gpu(self.gpu_corrected, gpu_temp, coords, oob_mask)
                        # Swap: corrected = remapped result
                        self.gpu_corrected.copy_from(gpu_temp)
                        gpu_temp.clear()
                        self.cpu_corrected = self.gpu_corrected.to_numpy()
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
                self.last_grading_key = None

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
                    metering_data = self.cpu_corrected if self.cpu_corrected is not None else self.cpu_linear
                    gain = strategy.calculate_gain(metering_data, source_cs)
                    self.cached_auto_ev = np.log2(gain)
                    self.last_metering_key = current_metering_key
                    applied_ev = self.cached_auto_ev
                    final_exposure_gain = gain

            final_exposure_gain = round(float(final_exposure_gain), 4)

            self.last_applied_ev = applied_ev

            # --- Stage 6-8: Exposure + Adjustments + Grading ---
            # Always start from gpu_cropped (immutable ProPhoto Linear source).
            # Every parameter change re-processes from this clean source.
            log_space = params.get('log_space')
            lut_path = params.get('lut_path')
            wb_temp = params.get('wb_temp', 0.0)
            wb_tint = params.get('wb_tint', 0.0)
            highlight = params.get('highlight', 0.0)
            shadow = params.get('shadow', 0.0)
            sat = params.get('saturation', 1.0)
            con = params.get('contrast', 1.0)

            grading_key = (
                self.last_geo_crop_key,
                final_exposure_gain,
                wb_temp, wb_tint,
                highlight, shadow,
                sat, con,
                log_space, lut_path,
            )

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
                        self.gpu_cropped.height, self.gpu_cropped.width, 3)

                    fused_expose_adjust_grade(
                        self.gpu_cropped.arr, self.gpu_graded.arr,
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
                    # Non-fused: copy from gpu_cropped, apply each step in-place
                    self.gpu_graded.copy_from(self.gpu_cropped)
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

        except Exception as e:
            logger.error(f"[Worker] Error in dev stages: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Processing error: {str(e)}")
            return

        try:

            # --- Stage 10: Sharpening (Taichi RL, GPU) ---
            sharpen_strength = params.get('sharpen_strength', 0.0)
            sharpen_key = (grading_key, sharpen_strength)

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
                    except Exception as e:
                        logger.error(f"[Worker] Sharpening failed: {e}")
                elif self.cached_sharpened is not None:
                    self.gpu_graded.upload(self.cached_sharpened)

            # --- Stage 11: Output ---
            # GPU float32→uint8 on GPU, then single GPU→CPU download
            t_out = time.perf_counter()
            h, w = self.gpu_graded.height, self.gpu_graded.width

            # Pre-allocate GPU uint8 buffer (avoids Taichi's implicit upload of numpy dst)
            if self._gpu_uint8 is None or self._gpu_uint8.shape != (h, w, 3):
                self._gpu_uint8 = ti.ndarray(dtype=ti.u8, shape=(h, w, 3))

            # GPU→GPU kernel (no CPU transfer)
            float_to_uint8_gpu(self.gpu_graded.arr, self._gpu_uint8)

            # Single GPU→CPU download of uint8 (135MB vs 375MB float32)
            img_uint8 = self._gpu_uint8.to_numpy()

            t_end = time.perf_counter()
            logger.info(f"[Worker] Pipeline: {(t_end - t_start)*1000:.0f}ms total, output: {(t_end - t_out)*1000:.0f}ms")

            self.result_ready.emit(img_uint8, request.path,
                                   request.request_id, applied_ev)

        except Exception as e:
            logger.error(f"[Worker] Error in grading/output: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Output error: {str(e)}")
