import gc
import numpy as np
import colour
import os
from typing import Optional

from raw_alchemy import utils
from raw_alchemy.config import LOG_TO_WORKING_SPACE, LOG_ENCODING_MAP
from raw_alchemy.logger import create_logger
from raw_alchemy.metering import apply_auto_exposure
from raw_alchemy.file_io import save_image
from raw_alchemy.onnx.denoiser import denoise_raw
from raw_alchemy.math_ops import log_encode_gpu, apply_matrix_inplace


# ==========================================
#          RAW 预处理
# ==========================================

def subtract_black_level(sensor_raw, bl, wl, cfa_pattern):
    """Per-channel black level subtraction and normalization to [0, 1]."""
    pat_size = cfa_pattern.shape[0]
    result = np.empty_like(sensor_raw)
    for r in range(pat_size):
        for c in range(pat_size):
            color = cfa_pattern[r, c]
            bl_c = float(bl[min(color, len(bl) - 1)])
            result[r::pat_size, c::pat_size] = np.maximum(
                sensor_raw[r::pat_size, c::pat_size] - bl_c, 0
            ) / (wl - bl_c)
    return result


def fix_hot_pixels(raw_norm, cfa_pattern, threshold=4.0):
    """Detect and replace hot/dead pixels using per-channel median comparison."""
    import cv2
    pat_size = cfa_pattern.shape[0]
    for r in range(pat_size):
        for c in range(pat_size):
            plane = raw_norm[r::pat_size, c::pat_size]
            plane32 = plane.astype(np.float32) if plane.dtype != np.float32 else plane
            med = cv2.medianBlur(plane32, 3)
            diff = np.abs(plane - med)
            std = max(np.std(diff), 1e-6)
            hot = diff > threshold * std
            plane[hot] = med[hot]


# ==========================================
#     高光重建 (Segmentation Based, GPU)
# ==========================================

def highlight_inpaint_opposed(raw_data, cfa_pattern, wb):
    """Segmentation-based highlight reconstruction.

    Same semantics as the darktable ``DT_IOP_HIGHLIGHTS_SEGMENTS`` mode:
    per-pixel opposing-channel reference average + per-segment chroma
    correction.

    Implementation:
      * ``compute_hl_refavg`` runs on GPU (Taichi).
      * Morphology / CCL / max-filter run on CPU via OpenCV (SIMD,
        operating on packed uint8 with SSE2/AVX2).
    """
    import cv2
    from raw_alchemy.math_ops import compute_hl_refavg

    H, W = raw_data.shape
    pat_size = cfa_pattern.shape[0]
    g = max(float(wb[1]), 1e-6)

    color_map = np.tile(cfa_pattern,
                        ((H + pat_size - 1) // pat_size,
                         (W + pat_size - 1) // pat_size))[:H, :W]
    color_map = np.where(color_map >= 3, 1, color_map).astype(np.int32)

    wb_gains = np.array([wb[0] / g, 1.0, wb[2] / g], dtype=np.float32)
    CLIP = 0.987
    raw_clips = np.array([CLIP / max(wg, 1e-6) for wg in wb_gains],
                         dtype=np.float32)

    refavg, clipped = compute_hl_refavg(raw_data, color_map, wb_gains, raw_clips)
    if not np.any(clipped):
        return

    diff = raw_data - refavg
    # 7x7 square SE — closes gaps up to 6 px wide.
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for c in range(3):
        clipped_c = clipped & (color_map == c)
        if not np.any(clipped_c):
            continue

        # Morphological closing (SIMD-accelerated).
        cc_u8 = clipped_c.astype(np.uint8)
        closed = cv2.morphologyEx(cc_u8, cv2.MORPH_CLOSE, closing_kernel)

        # Connected components, 8-connectivity. Matches darktable's
        # segmentation choice; closing has already merged any 4-connected
        # clusters so the connectivity choice is near-equivalent here.
        num_seg_plus_one, labels = cv2.connectedComponents(closed, connectivity=8)
        num_seg = num_seg_plus_one - 1
        if num_seg == 0:
            continue

        # 7x7 max-filter on labels via cv2.dilate(float32) — used to
        # identify the segment-border zone for chroma estimation.
        expanded = cv2.dilate(labels.astype(np.float32), closing_kernel).astype(np.int32)

        lo = raw_clips[c] * 0.2
        unclipped_valid = (color_map == c) & ~clipped & (raw_data > lo)

        border = (expanded > 0) & (labels == 0) & unclipped_valid
        border_labels = expanded[border]
        border_diffs = diff[border]

        seg_sum = np.bincount(border_labels, weights=border_diffs,
                              minlength=num_seg + 1)
        seg_cnt = np.bincount(border_labels, minlength=num_seg + 1)

        global_chroma = 0.0
        total_cnt = seg_cnt[1:].sum()
        if total_cnt > 100:
            global_chroma = seg_sum[1:].sum() / total_cnt

        seg_chroma = np.where(seg_cnt > 10,
                              seg_sum / np.maximum(seg_cnt, 1),
                              global_chroma).astype(np.float32)

        target = clipped_c & (labels > 0)
        target_labels = labels[target]
        raw_data[target] = np.maximum(
            raw_data[target],
            refavg[target] + seg_chroma[target_labels]
        )


# ==========================================
#              核心处理函数
# ==========================================

def _rawpy_decode_to_prophoto(raw_path: str) -> np.ndarray:
    """Decode RAW via RawSpeed (or rawpy fallback) + GPU demosaic to ProPhoto Linear float32 (H, W, 3)."""
    import rawpy
    from raw_alchemy.demosaic import rcd_demosaic, get_dcraw_filters, get_cfa_pattern_from_filters
    from raw_alchemy.onnx.denoiser import _apply_flip
    from raw_alchemy.colorspace_matrices import cam_to_prophoto_matrix
    from raw_alchemy.rawspeed_binding import try_decode, XTRANS_PATTERN

    rs = try_decode(raw_path)

    if rs and (rs.is_bayer or rs.is_xtrans):
        sensor_raw = rs.bayer.astype(np.float32)
        bl = np.array(rs.black_levels, dtype=np.float32)
        wl = float(rs.white_level)
        wb = np.array(rs.wb_coeffs, dtype=np.float32)
        xyz_to_cam = rs.color_matrix.astype(np.float64) if rs.color_matrix is not None else None
        is_bayer = rs.is_bayer
        is_xtrans = rs.is_xtrans
        filters = rs.filters
        cfa_pattern = get_cfa_pattern_from_filters(filters) if is_bayer else XTRANS_PATTERN
        try:
            import rawpy as _rp
            with _rp.imread(raw_path) as _r:
                flip = _r.sizes.flip
        except Exception:
            flip = 0
    else:
        with rawpy.imread(raw_path) as raw:
            sensor_raw = raw.raw_image_visible.astype(np.float32)
            bl = np.array(raw.black_level_per_channel, dtype=np.float32)
            wl = float(raw.white_level)
            wb = np.array(raw.camera_whitebalance, dtype=np.float32)
            flip = raw.sizes.flip
            cfa_pattern = raw.raw_pattern.copy() if raw.raw_pattern is not None else None
            xyz_to_cam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)

        is_bayer = cfa_pattern is not None and cfa_pattern.shape == (2, 2)
        is_xtrans = cfa_pattern is not None and cfa_pattern.shape == (6, 6)
        filters = None

    if is_bayer:
        raw_norm = subtract_black_level(sensor_raw, bl, wl, cfa_pattern)
        fix_hot_pixels(raw_norm, cfa_pattern)
        highlight_inpaint_opposed(raw_norm, cfa_pattern, wb)
        dcraw_filters = filters if filters else get_dcraw_filters(cfa_pattern)
        rgb = rcd_demosaic(raw_norm, dcraw_filters)
    elif is_xtrans:
        xtrans_pat = cfa_pattern
        raw_norm = subtract_black_level(sensor_raw, bl, wl, xtrans_pat)
        fix_hot_pixels(raw_norm, xtrans_pat)
        highlight_inpaint_opposed(raw_norm, xtrans_pat, wb)
        from raw_alchemy.xtrans_demosaic import xtrans_markesteijn_demosaic
        rgb = xtrans_markesteijn_demosaic(raw_norm, xtrans_pat)
    else:
        import rawpy as _rawpy
        with _rawpy.imread(raw_path) as _raw:
            rgb16 = _raw.postprocess(
                gamma=(1, 1), no_auto_bright=True, use_camera_wb=True,
                use_auto_wb=False, output_bps=16,
                output_color=_rawpy.ColorSpace.ProPhoto,
                bright=1.0, highlight_mode=2,
            )
        rgb = (rgb16 / 65535.0).astype(np.float32)
        del rgb16
        if rgb.ndim == 3 and rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        np.maximum(rgb, 0.0, out=rgb)
        return rgb

    rgb = np.ascontiguousarray(_apply_flip(rgb, flip))

    g = wb[1] if wb[1] > 0 else 1.0
    rgb[:, :, 0] *= wb[0] / g
    rgb[:, :, 2] *= wb[2] / g

    cam_to_prophoto = cam_to_prophoto_matrix(xyz_to_cam)
    apply_matrix_inplace(rgb, cam_to_prophoto)

    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb


def process_image(
    raw_path: str,
    output_path: str,
    log_space: str,
    lut_path: Optional[str],
    exposure: Optional[float] = None,
    lens_correct: bool = True,
    metering_mode: str = 'hybrid',
    custom_db_path: Optional[str] = None,
    log_queue: Optional[object] = None,
    # New params
    wb_temp: float = 0.0,
    wb_tint: float = 0.0,
    saturation: float = 1.25,
    contrast: float = 1.1,
    highlight: float = 0.0,
    shadow: float = 0.0,
    # Geometry
    rotation: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    crop: Optional[tuple] = None,
    # Denoising
    denoise_enabled: bool = False,
    # Sharpening (Richardson-Lucy)
    sharpen_strength: float = 0.0,
):
    filename = os.path.basename(raw_path)
    
    # 创建统一的日志处理器
    logger = create_logger(log_queue, filename)
    
    logger.info(f"🧪 [Raw Alchemy] Processing: {raw_path}")

    # --- Step 1: 解码 RAW (rawpy + RCD demosaic -> ProPhoto RGB Linear) ---
    from raw_alchemy.exif import extract_lens_exif

    exif_data, exif_metadata = extract_lens_exif(raw_path, None)

    if denoise_enabled:
        # CANS RAW V2: packed RAW → ProPhoto Linear RGB (replaces demosaicing)
        logger.info(f"  🔹 [Step 1] CANS RAW V2 denoise + demosaic...")
        try:
            img = denoise_raw(raw_path, exposure_ratio=1.0)
            logger.info("  ✅ CANS denoise complete")
        except Exception as e:
            logger.error(f"  ❌ CANS denoise failed, falling back to rawpy+RCD: {e}")
            img = _rawpy_decode_to_prophoto(raw_path)
    else:
        logger.info(f"  🔹 [Step 1] Decoding RAW (rawpy + RCD)...")
        img = _rawpy_decode_to_prophoto(raw_path)

    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']

    # --- Step 2: 曝光控制 ---
    if exposure is not None:
        # 路径 A: 手动曝光
        logger.info(f"  🔹 [Step 2] Manual Exposure Override ({exposure:+.2f} stops)")
        gain = 2.0 ** exposure
        utils.apply_gain_inplace(img, gain)
    else:
        # 路径 B: 自动测光（使用策略模式）
        logger.info(f"  🔹 [Step 2] Auto Exposure ({metering_mode})")
        img, applied_gain = apply_auto_exposure(img, source_cs, metering_mode, target_gray=0.18)


    # --- Step 3: 基础校正 (WB, Lens, HL/SH) ---
    
    # 3.1 镜头校正
    if lens_correct:
        logger.info("  🔹 [Step 3.1] Lens Correction...")
        img = utils.apply_lens_correction(
            img,
            exif_data=exif_data,
            custom_db_path=custom_db_path
        )
            
    # 3.1.5 几何变换
    if rotation != 0 or flip_horizontal or flip_vertical:
        logger.info(f"  🔹 [Step 3.1.5] Geometry (Rot:{rotation}, FlipH:{flip_horizontal}, FlipV:{flip_vertical})...")
        img = utils.apply_geometry(img, rotation, flip_horizontal, flip_vertical)
    
    # 3.1.6 裁切
    if crop and crop != (0.0, 0.0, 1.0, 1.0):
        logger.info(f"  🔹 [Step 3.1.6] Cropping {crop}...")
        img = utils.apply_crop(img, crop)
    
    # 3.2 白平衡
    if wb_temp != 0.0 or wb_tint != 0.0:
        logger.info(f"  🔹 [Step 3.2] White Balance (T:{wb_temp}, t:{wb_tint})...")
        utils.apply_white_balance(img, wb_temp, wb_tint)

    # 3.3 高光/阴影
    if highlight != 0.0 or shadow != 0.0:
        logger.info(f"  🔹 [Step 3.3] Highlight/Shadow (H:{highlight}, S:{shadow})...")
        utils.apply_highlight_shadow(img, highlight, shadow, colourspace=source_cs)

    # 3.4 饱和度/对比度
    logger.info(f"  🔹 [Step 3.4] Saturation/Contrast (S:{saturation:.2f}, C:{contrast:.2f})...")
    img = utils.apply_saturation_and_contrast(img, saturation=saturation, contrast=contrast, colourspace=source_cs)

    # --- Step 4: 色彩空间转换 (ProPhoto Linear -> Log 或 sRGB Standard) ---
    if log_space and log_space != 'None':
        log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
        log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)
        
        if not log_color_space_name:
             raise ValueError(f"Unknown Log Space: {log_space}")

        logger.info(f"  🔹 [Step 4] Color Transform (ProPhoto -> {log_color_space_name} -> {log_curve_name})")

        # 4.1 Gamut 变换 (矩阵运算)
        M = colour.matrix_RGB_to_RGB(
            colour.RGB_COLOURSPACES['ProPhoto RGB'],
            colour.RGB_COLOURSPACES[log_color_space_name],
        )
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        utils.apply_matrix_inplace(img, M)
        
        # 4.2 Log 编码
        # Log 函数无法处理负值，需裁剪微小底噪
        np.maximum(img, 1e-6, out=img)
        if not log_encode_gpu(img, log_curve_name):
            img = colour.cctf_encoding(img, function=log_curve_name)
    else:
        logger.info("  🔹 [Step 4] Applying sRGB Standard Transform (ProPhoto -> sRGB -> sRGB OETF)")
        # 4.1 Gamut: ProPhoto RGB → sRGB
        M = colour.matrix_RGB_to_RGB(
            colour.RGB_COLOURSPACES['ProPhoto RGB'],
            colour.RGB_COLOURSPACES['sRGB'],
        )
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        utils.apply_matrix_inplace(img, M)
        # 4.2 sRGB OETF
        utils.linear_to_srgb_inplace(img)

    # --- Step 5: 应用 LUT ---
    if lut_path:
        logger.info(f"  🔹 [Step 5] Applying LUT {os.path.basename(lut_path)}...")
        try:
            lut = utils.load_lut_cached(lut_path)
            
            # 3D LUT 使用 Taichi 加速
            if isinstance(lut, colour.LUT3D):
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                
                # Ensure LUT table is float32 and C-contiguous
                lut_table = lut.table
                if lut_table.dtype != np.float32:
                    lut_table = lut_table.astype(np.float32)
                if not lut_table.flags['C_CONTIGUOUS']:
                    lut_table = np.ascontiguousarray(lut_table)
                
                # Ensure domains are float64 and C-contiguous
                domain_min = lut.domain[0].astype(np.float64)
                domain_max = lut.domain[1].astype(np.float64)
                if not domain_min.flags['C_CONTIGUOUS']:
                    domain_min = np.ascontiguousarray(domain_min)
                if not domain_max.flags['C_CONTIGUOUS']:
                    domain_max = np.ascontiguousarray(domain_max)
                
                utils.apply_lut_inplace(img, lut_table, domain_min, domain_max)
            else:
                # 1D LUT 使用 colour 库默认方法
                img = lut.apply(img)
            
        except Exception as e:
            logger.error(f"  ❌ applying LUT: {e}")

    # --- Step 5.6: Sharpening (Richardson-Lucy, after denoise) ---
    if sharpen_strength > 0:
        logger.info(f"  🔹 [Step 5.6] Applying RL sharpening (strength={sharpen_strength:.2f})...")
        try:
            from raw_alchemy.math_ops import sharpen
            img = sharpen(img, strength=sharpen_strength, sigma=1.0)
            logger.info("  ✅ Sharpening complete")
        except Exception as e:
            logger.error(f"  ❌ Sharpening failed: {e}")

    # --- Step 5.6: DNG 线性化处理 ---
    # 如果保存为 DNG，必须确保数据是线性的 (Linear Raw)。
    # 前面的 Log 变换或 sRGB 转换可能已经应用了 Gamma/Log 曲线，需要逆转。
    color_matrix = None
    if output_path.lower().endswith('.dng'):
        logger.info("  🔹 [Pre-Save] Converting to Linear for DNG export...")
        try:
            if log_space and log_space != 'None' and not lut_path:
                # Log 模式：逆转 Log 曲线
                log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
                color_matrix = colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
                log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)
                utils.srgb_to_linear_inplace(img)
            else:
                # 标准模式：逆转 sRGB 曲线 (sRGB EOTF)
                utils.srgb_to_linear_inplace(img)
                color_matrix = colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
        except Exception as e:
            logger.warning(f"  ⚠️  Linearization failed: {e}. Saving as is.")

    # --- Step 6: 保存（使用模块化的文件保存功能）---
    logger.info(f"  💾 Saving to {os.path.basename(output_path)}...")
    save_image(img, output_path, logger, exif_metadata=exif_metadata, exif_dict=exif_data, color_matrix=color_matrix)
    
    # --- 最终清理 ---
    del img
    gc.collect()


def export_from_cache(
    cached_img: np.ndarray,
    output_path: str,
    exif_data: dict,
    exif_metadata: Optional[dict],
    log_space: str,
    lut_path: Optional[str],
    exposure: Optional[float] = None,
    metering_mode: str = 'hybrid',
    log_queue: Optional[object] = None,
    wb_temp: float = 0.0,
    wb_tint: float = 0.0,
    saturation: float = 1.25,
    contrast: float = 1.1,
    highlight: float = 0.0,
    shadow: float = 0.0,
    rotation: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    crop: Optional[tuple] = None,
    sharpen_strength: float = 0.0,
):
    """Export using cached ProPhoto Linear data (after demosaic + lens correction + denoise).

    Skips RAW decode, demosaicing, lens correction and denoising — reuses the
    preview pipeline's cached result.  Everything from geometry onward is
    identical to process_image().
    """
    filename = os.path.basename(output_path)
    logger = create_logger(log_queue, filename)
    logger.info(f"🧪 [Raw Alchemy] Export from cache -> {output_path}")

    img = cached_img.copy()
    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']

    # --- Step 2: Exposure ---
    if exposure is not None:
        logger.info(f"  🔹 [Step 2] Manual Exposure Override ({exposure:+.2f} stops)")
        gain = 2.0 ** exposure
        utils.apply_gain_inplace(img, gain)
    else:
        logger.info(f"  🔹 [Step 2] Auto Exposure ({metering_mode})")
        img, _ = apply_auto_exposure(img, source_cs, metering_mode, target_gray=0.18)

    # --- Step 3.1.5: Geometry ---
    if rotation != 0 or flip_horizontal or flip_vertical:
        logger.info(f"  🔹 [Step 3.1.5] Geometry (Rot:{rotation}, FlipH:{flip_horizontal}, FlipV:{flip_vertical})...")
        img = utils.apply_geometry(img, rotation, flip_horizontal, flip_vertical)

    # --- Step 3.1.6: Crop ---
    if crop and crop != (0.0, 0.0, 1.0, 1.0):
        logger.info(f"  🔹 [Step 3.1.6] Cropping {crop}...")
        img = utils.apply_crop(img, crop)

    # --- Step 3.2: White Balance ---
    if wb_temp != 0.0 or wb_tint != 0.0:
        logger.info(f"  🔹 [Step 3.2] White Balance (T:{wb_temp}, t:{wb_tint})...")
        utils.apply_white_balance(img, wb_temp, wb_tint)

    # --- Step 3.3: Highlight / Shadow ---
    if highlight != 0.0 or shadow != 0.0:
        logger.info(f"  🔹 [Step 3.3] Highlight/Shadow (H:{highlight}, S:{shadow})...")
        utils.apply_highlight_shadow(img, highlight, shadow, colourspace=source_cs)

    # --- Step 3.4: Saturation / Contrast ---
    logger.info(f"  🔹 [Step 3.4] Saturation/Contrast (S:{saturation:.2f}, C:{contrast:.2f})...")
    img = utils.apply_saturation_and_contrast(img, saturation=saturation, contrast=contrast, colourspace=source_cs)

    # --- Step 4: Color Space Transform ---
    if log_space and log_space != 'None':
        log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
        log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)
        if not log_color_space_name:
            raise ValueError(f"Unknown Log Space: {log_space}")
        logger.info(f"  🔹 [Step 4] Color Transform (ProPhoto -> {log_color_space_name} -> {log_curve_name})")
        M = colour.matrix_RGB_to_RGB(
            colour.RGB_COLOURSPACES['ProPhoto RGB'],
            colour.RGB_COLOURSPACES[log_color_space_name],
        )
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        utils.apply_matrix_inplace(img, M)
        np.maximum(img, 1e-6, out=img)
        if not log_encode_gpu(img, log_curve_name):
            img = colour.cctf_encoding(img, function=log_curve_name)
    else:
        logger.info("  🔹 [Step 4] Applying sRGB Standard Transform (ProPhoto -> sRGB -> sRGB OETF)")
        # Gamut: ProPhoto RGB → sRGB
        M = colour.matrix_RGB_to_RGB(
            colour.RGB_COLOURSPACES['ProPhoto RGB'],
            colour.RGB_COLOURSPACES['sRGB'],
        )
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        utils.apply_matrix_inplace(img, M)
        # sRGB OETF
        utils.linear_to_srgb_inplace(img)

    # --- Step 5: LUT ---
    if lut_path:
        logger.info(f"  🔹 [Step 5] Applying LUT {os.path.basename(lut_path)}...")
        try:
            lut = utils.load_lut_cached(lut_path)
            if isinstance(lut, colour.LUT3D):
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                lut_table = lut.table
                if lut_table.dtype != np.float32:
                    lut_table = lut_table.astype(np.float32)
                if not lut_table.flags['C_CONTIGUOUS']:
                    lut_table = np.ascontiguousarray(lut_table)
                domain_min = lut.domain[0].astype(np.float64)
                domain_max = lut.domain[1].astype(np.float64)
                if not domain_min.flags['C_CONTIGUOUS']:
                    domain_min = np.ascontiguousarray(domain_min)
                if not domain_max.flags['C_CONTIGUOUS']:
                    domain_max = np.ascontiguousarray(domain_max)
                utils.apply_lut_inplace(img, lut_table, domain_min, domain_max)
            else:
                img = lut.apply(img)
        except Exception as e:
            logger.error(f"  ❌ applying LUT: {e}")

    # --- Step 5.6: Sharpening ---
    if sharpen_strength > 0:
        logger.info(f"  🔹 [Step 5.6] Applying RL sharpening (strength={sharpen_strength:.2f})...")
        try:
            from raw_alchemy.math_ops import sharpen
            img = sharpen(img, strength=sharpen_strength, sigma=1.0)
            logger.info("  ✅ Sharpening complete")
        except Exception as e:
            logger.error(f"  ❌ Sharpening failed: {e}")

    # --- DNG linearization ---
    color_matrix = None
    if output_path.lower().endswith('.dng'):
        logger.info("  🔹 [Pre-Save] Converting to Linear for DNG export...")
        try:
            if log_space and log_space != 'None' and not lut_path:
                color_matrix = colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
                utils.srgb_to_linear_inplace(img)
            else:
                utils.srgb_to_linear_inplace(img)
                color_matrix = colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
        except Exception as e:
            logger.warning(f"  ⚠️  Linearization failed: {e}. Saving as is.")

    # --- Save ---
    logger.info(f"  💾 Saving to {filename}...")
    save_image(img, output_path, logger, exif_metadata=exif_metadata, exif_dict=exif_data, color_matrix=color_matrix)

    del img
    gc.collect()