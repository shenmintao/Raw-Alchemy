import threading
import time
import os
import numpy as np
import rawpy
import colour
from typing import Optional
from PySide6.QtCore import QThread, Signal
from loguru import logger

from raw_alchemy import utils, metering, config
from raw_alchemy.pipeline.request import ProcessRequest, ProcessorParams
from raw_alchemy.pipeline.cache_manager import ImageCacheManager, CachedImage
from raw_alchemy.onnx import denoiser

class ImageProcessor(QThread):
    """
    Image processing worker. Uses request queue pattern to eliminate race conditions.
    No more params_dirty, no more mode/running_mode confusion.
    """
    result_ready = Signal(np.ndarray, np.ndarray, str, int, float)  # img_uint8, img_float, path, request_id, applied_ev
    load_complete = Signal(str, int)  # path, request_id - signals RAW loading is done
    error_occurred = Signal(str)
    denoise_progress = Signal(int, int)  # current, total - for progress bar
    denoise_started = Signal()  # signals denoising has started
    denoise_finished = Signal()  # signals denoising has finished

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        
        # Request management
        self.pending_request: Optional[ProcessRequest] = None
        self.current_request_id = 0
        
        # LRU Cache Manager
        self.cache_manager = ImageCacheManager(max_items=5, max_memory_mb=1500)
        
        # Current State
        self.cached_linear = None
        self.cached_corrected = None
        self.cached_lens_key = None
        
        # Caching Layers (in processing order)
        
        # Layer 1: After Geometry (Rotation/Flip)
        self.cached_geometry = None
        self.last_geometry_key = None

        # Layer 3: After Perspective Correction
        self.cached_perspective = None
        self.last_perspective_key = None

        # Layer 4: After Crop
        self.cached_cropped = None
        self.last_crop_key = None

        # Layer 5: After Exposure Gain
        self.cached_exposed = None
        self.last_exposure_key = None
        
        # Layer 6: After WB/HS/Sat/Con
        self.cached_adjusted = None
        self.last_adjustment_key = None
        
        # Metering Cache (Logic)
        self.cached_auto_ev = 0.0
        self.last_metering_key = None
        
        self.exif_data = None
        self.last_applied_ev = 0.0
        self.current_path = None

        # LUT Cache
        self.cached_lut_path = None
        self.cached_lut_table = None
        self.cached_lut_domain_min = None
        self.cached_lut_domain_max = None
        self.cached_lut_is_3d = False
        
        # Denoising Cache (at the end, after LUT/sRGB)
        # Cache the fully denoised image (strength=1.0) and original for blending
        self.cached_denoise_original = None  # Original sRGB image before denoising
        self.cached_denoise_full = None      # Fully denoised image (strength=1.0)
        self.last_denoise_key = None         # Key based on all params affecting sRGB output
        
        # Sharpening Cache
        self.cached_sharpened = None
        self.last_sharpen_key = None

    def load_image(self, path: str):
        """Load RAW image - creates a special load request"""
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self.pending_request = ProcessRequest(path, {'_load': True}, request_id)
        
        if not self.isRunning():
            self.start()
        return request_id

    def preload_image(self, path: str):
        """Preload image into cache silently (low priority if possible, but queue based here)"""
        with self.lock:
            # Only preload if no other request is pending (simple priority)
            if self.pending_request is None:
                self.pending_request = ProcessRequest(path, {'_preload': True}, -1) # -1 ID for internal ops
        
        if not self.isRunning():
            self.start()

    def update_preview(self, path: str, params: ProcessorParams):
        """Process image with parameters"""
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            self.pending_request = ProcessRequest(path, params, request_id)
        
        if not self.isRunning():
            self.start()
        return request_id

    def run(self):
        """Keep thread alive to process requests without restart overhead"""
        idle_count = 0
        max_idle = 10  # Exit after 10 empty checks (1 second)
        
        while True:
            # Atomically get pending request
            with self.lock:
                request = self.pending_request
                if request:
                    self.pending_request = None
                    idle_count = 0
                else:
                    idle_count += 1
            
            if not request:
                if idle_count >= max_idle:
                    break  # Exit after idle timeout
                time.sleep(0.1)  # Sleep 100ms, check again
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
    
    def _do_preload(self, request: ProcessRequest):
        """Load image into cache ONLY. Do not update current state."""
        path = request.path
        if self.cache_manager.get(path):
            return # Already cached
        try:
            logger.info(f"[Worker] Preloading: {os.path.basename(path)}")
            import gc
            
            # Re-use loading logic but isolated
            with rawpy.imread(path) as raw:
                exif_data, _ = utils.extract_lens_exif(path, raw)
                raw_post = raw.postprocess(
                    gamma=(1, 1),
                    no_auto_bright=True,
                    use_camera_wb=True,
                    use_auto_wb=False,
                    output_bps=16,
                    output_color=rawpy.ColorSpace.ProPhoto,
                    bright=1.0,
                    highlight_mode=2,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
                    half_size=True
                )
                
                img = (raw_post / 65535.0).astype(np.float32)
                
                # Downsample
                h, w = img.shape[:2]
                max_dim = 2048
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    step = int(1.0 / scale)
                    if step > 1:
                        img = img[::step, ::step, :].copy() 
                
                del raw_post
                gc.collect() 
                
                # Add to Cache
                new_cache_item = CachedImage(
                    path=path,
                    linear_data=img, 
                    exif_data=exif_data,
                    lens_key=None
                )
                self.cache_manager.put(path, new_cache_item)
                
        except Exception as e:
            logger.warning(f"Preload failed for {path}: {e}")

    def _do_load(self, request: ProcessRequest):
        """Load and decode RAW file"""
        path = request.path
        
        # Check Cache First
        cached_item = self.cache_manager.get(path)
        
        if cached_item:
            logger.info(f"[Worker] Cache Hit: {os.path.basename(path)}")
            # Restore from cache
            self.cached_linear = cached_item.linear_data
            self.exif_data = cached_item.exif_data
            self.cached_lens_key = cached_item.lens_key
            self.cached_corrected = cached_item.corrected_data
            
            # Reset pipeline caches for new image context
            self.cached_geometry = None
            self.last_geometry_key = None
            self.cached_perspective = None
            self.last_perspective_key = None
            self.cached_cropped = None
            self.last_crop_key = None
            self.cached_exposed = None
            self.last_exposure_key = None
            self.cached_adjusted = None
            self.last_adjustment_key = None
            self.cached_adjusted = None
            self.last_adjustment_key = None
            
            # Restore Denoise Cache
            self.cached_denoise_full = cached_item.denoise_full
            self.cached_denoise_original = cached_item.denoise_original
            self.last_denoise_key = cached_item.denoise_key
            
            # Restore Sharpen Cache
            self.cached_sharpened = cached_item.sharpened_data
            self.last_sharpen_key = cached_item.sharpen_key

            self.cached_auto_ev = 0.0
            self.last_metering_key = None
            
            self.current_path = path
            self.load_complete.emit(path, request.request_id)
            return

        # Cache Miss - Full Load
        logger.info(f"[Worker] Cache Miss - Loading: {os.path.basename(path)}")
        
        # Invalidate current state
        if path != self.current_path:
            self.cached_linear = None
            self.cached_corrected = None
            self.cached_lens_key = None
            
            self.cached_geometry = None
            self.last_geometry_key = None
            self.cached_perspective = None
            self.last_perspective_key = None
            self.cached_cropped = None
            self.last_crop_key = None
            self.cached_exposed = None
            self.last_exposure_key = None
            self.cached_adjusted = None
            self.last_adjustment_key = None
            self.cached_denoised = None
            self.last_denoise_key = None
            self.cached_sharpened = None
            self.last_sharpen_key = None
            self.cached_auto_ev = 0.0
            self.last_metering_key = None
            
            self.exif_data = None
            self.current_path = path
            
            # Reset LUT cache if image changes? Actually LUT is global global param usually, 
            # but let's keep it persistent across images for now as it makes sense.
            # No need to reset self.cached_lut_path here.
        
        try:
            import gc
            
            with rawpy.imread(path) as raw:
                self.exif_data, _ = utils.extract_lens_exif(path, raw)
                raw_post = raw.postprocess(
                    gamma=(1, 1),
                    no_auto_bright=True,
                    use_camera_wb=True,
                    use_auto_wb=False,
                    output_bps=16,
                    output_color=rawpy.ColorSpace.ProPhoto,
                    bright=1.0,
                    highlight_mode=2,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
                    half_size=True
                )
                
                img = (raw_post / 65535.0).astype(np.float32)
                
                # Downsample if needed
                h, w = img.shape[:2]
                max_dim = 2048
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    step = int(1.0 / scale)
                    if step > 1:
                        # KEY FIX: Force Copy to detach from full raw_post buffer
                        img = img[::step, ::step, :].copy() 
                
                self.cached_linear = img
                
                # Explicitly cleanup
                del raw_post
                gc.collect() 
                
                # Add to Cache
                new_cache_item = CachedImage(
                    path=path,
                    linear_data=self.cached_linear, # This is now efficient
                    exif_data=self.exif_data,
                    lens_key=None
                )
                self.cache_manager.put(path, new_cache_item)
                
                # Signal that loading is complete
                self.load_complete.emit(path, request.request_id)
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            self.error_occurred.emit(f"Failed to load image: {e}")

    def _do_process(self, request: ProcessRequest):
        """Process image with parameters"""
        logger.debug(f"[Worker] _do_process called for: {os.path.basename(request.path)}, request_id={request.request_id}")
        
        try:
            # Ensure image is loaded and matches request path
            if self.cached_linear is None or self.current_path != request.path:
                logger.debug(f"[Worker] Need to load first")
                # Need to load first
                self._do_load(ProcessRequest(request.path, {'_load': True}, request.request_id))
                if self.cached_linear is None:
                    logger.error(f"[Worker] Failed to load image, aborting processing")
                    return
            
            params = request.params
            
            # --- Stage 2: Lens Correction ---
            current_lens_key = (params.get('lens_correct'), params.get('custom_db_path'))
            
            if current_lens_key != self.cached_lens_key or self.cached_corrected is None:
                logger.debug(f"[Worker] Applying lens correction...")
                temp = self.cached_linear.copy()
                if params.get('lens_correct') and self.exif_data:
                    temp = utils.apply_lens_correction(
                        temp, self.exif_data, custom_db_path=params.get('custom_db_path')
                    )
                self.cached_corrected = temp
                self.cached_lens_key = current_lens_key
                # Invalidate dev cache since base changed
                self.cached_lens_key = current_lens_key
                # Invalidate dev cache since base changed
                self.cached_geometry = None # Invalidate downstream
                self.last_geometry_key = None
                self.cached_cropped = None
                self.last_crop_key = None
                self.cached_exposed = None
                self.last_exposure_key = None
                self.cached_adjusted = None
                self.last_adjustment_key = None
                
                # Update Cache with Corrected Data for future speedups
                if self.cached_linear is not None:
                     cached_item = self.cache_manager.get(request.path)
                     if cached_item:
                         # 检查是否是首次添加 corrected_data（避免重复累加内存大小）
                         is_first_correction = (cached_item.corrected_data is None)
                         
                         # 如果之前已有 corrected_data，先减去旧的大小
                         if not is_first_correction:
                             old_corrected_size = cached_item.corrected_data.nbytes / (1024 * 1024)
                             cached_item.size_mb -= old_corrected_size
                             self.cache_manager.current_memory_mb -= old_corrected_size
                         
                         # 更新 corrected_data 和 lens_key
                         cached_item.corrected_data = self.cached_corrected
                         cached_item.lens_key = current_lens_key
                         
                         # 添加新的 corrected_data 大小
                         new_corrected_size = self.cached_corrected.nbytes / (1024 * 1024)
                         cached_item.size_mb += new_corrected_size
                         self.cache_manager.current_memory_mb += new_corrected_size
                         
                         # 检查是否需要驱逐缓存项
                         self.cache_manager._evict_if_needed()
             
            
            # --- Stage 3: Geometry (Rotation/Flip) ---
            # Note: Denoising is now done at the end (after LUT/sRGB) for JPG-style denoising
            geometry_key = (
                self.cached_lens_key,
                params.get('rotation', 0),
                params.get('flip_horizontal', False),
                params.get('flip_vertical', False)
            )

            if geometry_key == self.last_geometry_key and self.cached_geometry is not None:
                # Cache hit
                pass
            else:
                logger.debug(f"[Worker] Layer 3 (Geometry) Computing...")
                # Apply geometry to corrected image (denoising is now at the end)
                self.cached_geometry = utils.apply_geometry(
                    self.cached_corrected,
                    rotation=params.get('rotation', 0),
                    flip_h=params.get('flip_horizontal', False),
                    flip_v=params.get('flip_vertical', False)
                )
                self.last_geometry_key = geometry_key
                # Invalidate next layers
                self.cached_perspective = None
                self.last_perspective_key = None
                self.cached_cropped = None
                self.last_crop_key = None
                self.cached_exposed = None
                self.last_exposure_key = None
                self.cached_adjusted = None
                self.last_adjustment_key = None

            # --- Stage 2.55: Perspective Correction ---
            perspective_key = (
                self.last_geometry_key,
                params.get('perspective_corners')  # 4-tuple of corner coords
            )

            if perspective_key == self.last_perspective_key and self.cached_perspective is not None:
                # Cache hit
                pass
            else:
                logger.debug(f"[Worker] Layer 2.55 (Perspective) Computing...")
                self.cached_perspective = utils.apply_perspective(
                    self.cached_geometry,
                    params.get('perspective_corners')
                )
                self.last_perspective_key = perspective_key
                # Invalidate downstream caches
                self.cached_cropped = None
                self.last_crop_key = None
                self.cached_exposed = None
                self.last_exposure_key = None
                self.cached_adjusted = None
                self.last_adjustment_key = None

            # --- Stage 2.6: Crop (Layer 0.6) ---
            crop_key = (
                self.last_perspective_key,  # Now depends on perspective, not geometry
                params.get('crop', (0.0, 0.0, 1.0, 1.0))
            )
            
            if crop_key == self.last_crop_key and self.cached_cropped is not None:
                # Cache hit
                pass
            else:
                logger.debug(f"[Worker] Layer 2.6 (Crop) Computing... {params.get('crop')}")
                self.cached_cropped = utils.apply_crop(
                    self.cached_perspective,  # Use perspective-corrected image
                    params.get('crop', (0.0, 0.0, 1.0, 1.0))
                )
                self.last_crop_key = crop_key
                self.cached_exposed = None
                self.last_exposure_key = None
                self.cached_adjusted = None
                self.last_adjustment_key = None 

            # --- Stage 3: Exposure (Layer 1) ---
            # Define metering key to avoid re-calculating auto exposure when only sliders allow
            # We ONLY re-calc auto exposure if:
            # 1. Lens correction changed (implied by execution flow reaching here)
            # 2. Metering mode changed
            # 3. Image changed
            # 3. Image changed
            # 4. Geometry changed (REMOVED - Metering should be stable on rotation)
            # 5. Crop changed (REMOVED - Metering should be on full frame)
            current_metering_key = (
                self.cached_lens_key,
                # self.last_geometry_key, # Decouple from geometry
                # self.last_crop_key,     # Decouple from crop
                params.get('metering_mode', 'matrix')
            )
            
            # Determine Exposure Gain
            final_exposure_gain = 0.0
            applied_ev = 0.0
            
            if params.get('exposure_mode') == 'Manual':
                # Manual Mode: Direct calculation
                applied_ev = params.get('exposure', 0.0)
                final_exposure_gain = 2.0 ** applied_ev
            else:
                # Auto Mode: Check cache logic
                logger.debug(f"[Worker] Auto Exp Check: CurrentKey={current_metering_key}, LastKey={self.last_metering_key}")
                if current_metering_key == self.last_metering_key:
                    logger.debug(f"[Worker] Using cached Auto Exposure EV: {self.cached_auto_ev:.2f}")
                    applied_ev = self.cached_auto_ev
                    final_exposure_gain = 2.0 ** applied_ev
                else:
                    logger.debug(f"[Worker] Calculating Auto Exposure (Cache Miss)...")
                    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
                    mode = params.get('metering_mode', 'matrix')
                    # Use corrected (but un-cropped/un-rotated) image for metering to ensure stability
                    strategy = metering.get_metering_strategy(mode)
                    gain = strategy.calculate_gain(self.cached_corrected, source_cs)
                    
                    self.cached_auto_ev = np.log2(gain)
                    self.last_metering_key = current_metering_key
                    
                    applied_ev = self.cached_auto_ev
                    final_exposure_gain = gain

            # Round gain to avoid float mismatch in cache keys
            # Reduced precision to 4 decimal places to prevent cache misses due to micro-variations
            final_exposure_gain = round(float(final_exposure_gain), 4)
            
            # Exposure Key for Pixel Cache
            exposure_key = (
                self.cached_lens_key,
                self.last_geometry_key,
                self.last_crop_key,
                final_exposure_gain
            )
            
            if exposure_key == self.last_exposure_key and self.cached_exposed is not None:
                # logger.debug(f"[Worker] Layer 1 (Exposure) Cache Hit")
                img_exposed = self.cached_exposed # No copy needed yet, read-only
            else:
                logger.debug(f"[Worker] Layer 1 (Exposure) Computing...")
                img_exposed = self.cached_cropped.copy()
                utils.apply_gain_inplace(img_exposed, final_exposure_gain)
                
                self.cached_exposed = img_exposed
                self.last_exposure_key = exposure_key
                # Invalidate next layer
                self.cached_adjusted = None
                self.last_adjustment_key = None

            self.last_applied_ev = applied_ev

            # --- Stage 4: Adjustments (Layer 2) ---
            # WB, H/S, Sat, Con
            adjustment_key = (
                exposure_key,
                params.get('wb_temp', 0.0),
                params.get('wb_tint', 0.0),
                params.get('highlight', 0.0),
                params.get('shadow', 0.0),
                params.get('saturation', 1.0),
                params.get('contrast', 1.0)
            )
            
            if adjustment_key == self.last_adjustment_key and self.cached_adjusted is not None:
                # logger.debug(f"[Worker] Layer 2 (Adjustments) Cache Hit")
                img = self.cached_adjusted # Read-only for next stage
            else:
                logger.debug(f"[Worker] Layer 2 (Adjustments) Computing...")
                # Start from exposed image (COPY it!)
                img = self.cached_exposed.copy()
                
                # White Balance
                utils.apply_white_balance(img, params.get('wb_temp', 0.0), params.get('wb_tint', 0.0))
                
                # Highlight / Shadow
                utils.apply_highlight_shadow(img, params.get('highlight', 0.0), params.get('shadow', 0.0))
                
                # Saturation / Contrast
                utils.apply_saturation_and_contrast(img, saturation=params.get('saturation', 1.0), contrast=params.get('contrast', 1.0))
                
                self.cached_adjusted = img
                self.last_adjustment_key = adjustment_key
            
            # Prepare image for Grading (COPY!)
            img = self.cached_adjusted.copy()
            
            # --- Stage 5: Viewport Resize (Before Grading for speed) ---
            # Resize to viewport size BEFORE Log/LUT for 8x speedup
            viewport_size = params.get('viewport_size')  # (width, height) tuple
            if viewport_size and viewport_size[0] > 0 and viewport_size[1] > 0:
                src_h, src_w = img.shape[:2]
                dst_w, dst_h = viewport_size
                
                # Only resize if destination is smaller
                if dst_w < src_w or dst_h < src_h:
                    # Use uniform step to preserve aspect ratio
                    step = max(1, max(src_h // dst_h, src_w // dst_w))
                    img = img[::step, ::step, :].copy()
                    logger.debug(f"[Worker] Viewport resize: {src_w}x{src_h} -> {img.shape[1]}x{img.shape[0]}")
                
        except Exception as e:
            logger.error(f"[Worker] Error in _do_process (Dev Stage): {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Processing error: {str(e)}")
            return
        
        # --- Stage 4: Grading (Log, LUT, Display) ---
        # Log Transform
        log_space = params.get('log_space')
        if log_space and log_space != 'None':
            try:
                log_color_space = config.LOG_TO_WORKING_SPACE.get(log_space)
                log_curve = config.LOG_ENCODING_MAP.get(log_space, log_space)
                
                if log_color_space:
                    if not np.isfinite(img).all():
                        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    M = colour.matrix_RGB_to_RGB(
                        colour.RGB_COLOURSPACES['ProPhoto RGB'],
                        colour.RGB_COLOURSPACES[log_color_space]
                    )
                    
                    if not img.flags['C_CONTIGUOUS']:
                        img = np.ascontiguousarray(img)
                    utils.apply_matrix_inplace(img, M)
                    
                    np.maximum(img, 1e-6, out=img)
                    img = colour.cctf_encoding(img, function=log_curve)
                    
                    if not np.isfinite(img).all():
                        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            except Exception as e:
                logger.error(f"[Worker] Log transform failed: {e}")
        
        # LUT
        lut_path = params.get('lut_path')
        if lut_path and os.path.exists(lut_path):
            try:
                # Check cache
                if lut_path != self.cached_lut_path:
                    logger.info(f"[Worker] Loading LUT: {os.path.basename(lut_path)}")
                    lut = colour.read_LUT(lut_path)
                    
                    self.cached_lut_is_3d = isinstance(lut, colour.LUT3D)
                    if self.cached_lut_is_3d:
                        lut_table = lut.table.astype(np.float32)
                        if not lut_table.flags['C_CONTIGUOUS']:
                            lut_table = np.ascontiguousarray(lut_table)
                        self.cached_lut_table = lut_table
                        self.cached_lut_domain_min = np.ascontiguousarray(lut.domain[0].astype(np.float64))
                        self.cached_lut_domain_max = np.ascontiguousarray(lut.domain[1].astype(np.float64))
                    else:
                        # For 1D LUTs or others, we just store the object (less common in this pipeline but supported)
                        self.cached_lut_table = lut 
                        
                    self.cached_lut_path = lut_path
                
                # Apply Cached LUT
                if self.cached_lut_is_3d:
                    if not img.flags['C_CONTIGUOUS']:
                        img = np.ascontiguousarray(img)
                    if img.dtype != np.float32:
                        img = img.astype(np.float32)
                    logger.debug(f"[Worker] Use Cached LUT: {os.path.basename(self.cached_lut_path)}")
                    utils.apply_lut_inplace(
                        img, 
                        self.cached_lut_table, 
                        self.cached_lut_domain_min, 
                        self.cached_lut_domain_max
                    )
                else:
                    # Non-optimized path for non-3D LUTs (rare in this context)
                    img = self.cached_lut_table.apply(img)
                    
            except Exception as e:
                logger.error(f"[Worker] LUT application error: {e}")
                # Invalidate cache on error to force retry next time if fixed
                self.cached_lut_path = None 
                self.cached_lut_table = None
        else:
            # If no LUT path provided but we have one cached, just don't apply it.
            # We don't necessarily need to clear the cache, keeping it ready is fine.
            pass
        
        try:
            # Display transform - sRGB Standard
            if not log_space or log_space == 'None':
                utils.linear_to_srgb_inplace(img)
            
            if not np.isfinite(img).all():
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            
            img = np.clip(img, 0, 1)
            
            # --- Stage 6: Denoising (After LUT/sRGB, JPG-style denoising) ---
            # Cache key includes all parameters that affect the sRGB image
            # (excluding denoise_strength since we cache full denoising and blend)
            denoise_cache_key = (
                self.last_adjustment_key,  # All dev params up to adjustment
                params.get('viewport_size'),  # Viewport resize
                log_space,  # Log transform
                lut_path,  # LUT
            )
            
            denoise_strength = params.get('denoise_strength', 0.0)
            if denoise_strength > 0:
                # Check if we need to recompute denoising
                    if denoise_cache_key != self.last_denoise_key or self.cached_denoise_full is None:
                        logger.debug(f"[Worker] Denoise Cache Miss!")
                        if self.cached_denoise_full is None:
                            logger.debug(f"[Worker] Reason: cached_denoise_full is None")
                        if denoise_cache_key != self.last_denoise_key:
                            logger.debug(f"[Worker] Reason: Key Mismatch")
                            if self.last_denoise_key:
                                logger.debug(f"  Differences:")
                                # Compare elements
                                names = ["AdjustmentKey", "Viewport", "Log", "LUT"]
                                for i, (v1, v2) in enumerate(zip(denoise_cache_key, self.last_denoise_key)):
                                    if v1 != v2:
                                        logger.debug(f"    {names[i]}: {v1} != {v2}")
                        
                        # Cache miss - need to compute full denoising
                        try:
                            self.denoise_started.emit()
                            logger.info(f"[Worker] Computing full denoising (cache miss)...")
                        
                        # Save original for blending
                        self.cached_denoise_original = img.copy()
                        
                        # Progress callback for UI
                        def progress_callback(current, total):
                            self.denoise_progress.emit(current, total)
                        
                        # Apply full denoising (strength=1.0)
                        self.cached_denoise_full = denoiser.denoise(
                            img,
                            strength=1.0,  # Always compute full denoising
                            tile_size=504,  # Optimized for 4GB VRAM (~3-3.5GB usage)
                            tile_overlap=64,  # Increased from 32 for better blending
                            progress_callback=progress_callback
                        )
                        
                        self.last_denoise_key = denoise_cache_key
                        self.denoise_finished.emit()
                        logger.info(f"[Worker] Full denoising cached")
                        
                        # Update Cache Manager
                        cached_item = self.cache_manager.get(request.path)
                        if cached_item:
                            # Remove old size if exists
                            if cached_item.denoise_full is not None:
                                old_size = cached_item.denoise_full.nbytes / (1024 * 1024)
                                cached_item.size_mb -= old_size
                                self.cache_manager.current_memory_mb -= old_size

                            cached_item.denoise_full = self.cached_denoise_full
                            cached_item.denoise_original = self.cached_denoise_original
                            cached_item.denoise_key = denoise_cache_key
                            
                            # Add new size
                            new_size = self.cached_denoise_full.nbytes / (1024 * 1024)
                            cached_item.size_mb += new_size
                            self.cache_manager.current_memory_mb += new_size
                            
                            self.cache_manager._evict_if_needed()
                    except Exception as e:
                        logger.error(f"[Worker] Denoising failed: {e}")
                        self.denoise_finished.emit()
                        # Clear cache on error
                        self.cached_denoise_original = None
                        self.cached_denoise_full = None
                        self.last_denoise_key = None
                
                # Apply strength blending from cache
                if self.cached_denoise_full is not None and self.cached_denoise_original is not None:
                    if denoise_strength >= 1.0:
                        img = self.cached_denoise_full.copy()
                    else:
                        # Blend: result = original * (1 - strength) + denoised * strength
                        img = self.cached_denoise_original * (1.0 - denoise_strength) + \
                              self.cached_denoise_full * denoise_strength
                    logger.debug(f"[Worker] Applied denoise strength={denoise_strength:.2f} from cache")
            else:
                # Denoising disabled - invalidate cache if params changed
                if denoise_cache_key != self.last_denoise_key:
                    self.cached_denoise_original = None
                    self.cached_denoise_full = None
                    self.last_denoise_key = None
            
            # --- Stage 7: Sharpening (Richardson-Lucy, after denoise) ---
            sharpen_strength = params.get('sharpen_strength', 0.0)
            
            # Cache Key
            current_img_state_key = (denoise_cache_key, denoise_strength)
            sharpen_key = (current_img_state_key, sharpen_strength)
            
            if sharpen_strength > 0:
                if sharpen_key == self.last_sharpen_key and self.cached_sharpened is not None:
                     logger.debug(f"[Worker] Sharpen Cache Hit")
                     img = self.cached_sharpened
                else:
                    try:
                        from raw_alchemy.math_ops import sharpen
                        logger.info(f"[Worker] Applying RL sharpening (strength={sharpen_strength:.2f})...")
                        img = sharpen(img, strength=sharpen_strength, sigma=1.0)
                        logger.info(f"[Worker] Sharpening complete")
                        
                        # Update Local Cache
                        self.cached_sharpened = img
                        self.last_sharpen_key = sharpen_key
                        
                        # Update Global Cache (Persistence)
                        cached_item = self.cache_manager.get(request.path)
                        if cached_item:
                            if cached_item.sharpened_data is not None:
                                old_size = cached_item.sharpened_data.nbytes / (1024 * 1024)
                                cached_item.size_mb -= old_size
                                self.cache_manager.current_memory_mb -= old_size
                            
                            cached_item.sharpened_data = self.cached_sharpened
                            cached_item.sharpen_key = self.last_sharpen_key
                            
                            new_size = self.cached_sharpened.nbytes / (1024 * 1024)
                            cached_item.size_mb += new_size
                            self.cache_manager.current_memory_mb += new_size
                            
                            self.cache_manager._evict_if_needed()
                            
                    except Exception as e:
                        import traceback
                        logger.error(f"[Worker] Sharpening failed: {e}")
                        traceback.print_exc()
            
            img_float = img  # Shared buffer
            img_uint8 = (img * 255).astype(np.uint8)
            
            self.result_ready.emit(img_uint8, img_float, request.path, request.request_id, applied_ev)
            
        except Exception as e:
            logger.error(f"[Worker] Error in final output: {e}")
            self.error_occurred.emit(f"Output error: {str(e)}")
