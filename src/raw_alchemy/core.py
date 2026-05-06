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
#              核心处理函数
# ==========================================

def _rawspeed_decode_to_prophoto(result) -> np.ndarray:
    """Decode RawSpeed result to ProPhoto Linear float32 (H, W, 3).

    Applies: RCD demosaic -> WB -> color matrix -> ProPhoto.
    """
    from raw_alchemy.demosaic import rcd_demosaic

    bayer_norm = result.normalize()
    rgb = rcd_demosaic(bayer_norm, result.filters)

    # White balance (normalize by G channel)
    wb = result.wb_coeffs
    g = wb[1] if wb[1] > 0 else 1.0
    rgb[:, :, 0] *= wb[0] / g
    rgb[:, :, 2] *= wb[2] / g

    # Color matrix: Camera -> XYZ -> ProPhoto
    if result.color_matrix is not None:
        cam_to_xyz = np.linalg.inv(result.color_matrix[:3, :3])
        prophoto_to_xyz = colour.RGB_COLOURSPACES['ProPhoto RGB'].matrix_RGB_to_XYZ
        xyz_to_prophoto = np.linalg.inv(prophoto_to_xyz)
        cam_to_prophoto = xyz_to_prophoto @ cam_to_xyz
        apply_matrix_inplace(rgb, cam_to_prophoto)

    np.maximum(rgb, 0.0, out=rgb)
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

    # --- Step 1: 解码 RAW (RawSpeed + RCD demosaic -> ProPhoto RGB Linear) ---
    from raw_alchemy.rawspeed_binding import RawSpeedDecoder
    from raw_alchemy.demosaic import rcd_demosaic
    from raw_alchemy.exif import extract_lens_exif

    _decoder = RawSpeedDecoder()
    _rs_result = _decoder.decode(raw_path)
    exif_data, exif_metadata = extract_lens_exif(raw_path, _rs_result)

    if denoise_enabled:
        # CANS RAW V2: packed RAW → ProPhoto Linear RGB (replaces demosaicing)
        logger.info(f"  🔹 [Step 1] CANS RAW V2 denoise + demosaic...")
        try:
            img = denoise_raw(raw_path, exposure_ratio=1.0)
            logger.info("  ✅ CANS denoise complete")
        except Exception as e:
            logger.error(f"  ❌ CANS denoise failed, falling back to RawSpeed+RCD: {e}")
            img = _rawspeed_decode_to_prophoto(_rs_result)
    else:
        logger.info(f"  🔹 [Step 1] Decoding RAW (RawSpeed + RCD)...")
        img = _rawspeed_decode_to_prophoto(_rs_result)

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