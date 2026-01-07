import gc
import rawpy
import numpy as np
import colour
import os
from typing import Optional

# 尝试导入同级目录下的模块，如果失败则尝试绝对导入 (方便不同运行环境调试)
from raw_alchemy import utils
from raw_alchemy.config import LOG_TO_WORKING_SPACE, LOG_ENCODING_MAP
from raw_alchemy.logger import create_logger
from raw_alchemy.metering import apply_auto_exposure
from raw_alchemy.file_io import save_image


# ==========================================
#              核心处理函数
# ==========================================

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
):
    filename = os.path.basename(raw_path)
    
    # 创建统一的日志处理器
    logger = create_logger(log_queue, filename)
    
    logger.info(f"🧪 [Raw Alchemy] Processing: {raw_path}")

    # --- Step 1: 解码 RAW (统一至 ProPhoto RGB / 16-bit Linear) ---
    logger.info(f"  🔹 [Step 1] Decoding RAW...")
    with rawpy.imread(raw_path) as raw:
        # 提取 EXIF (用于镜头校正)
        exif_data = utils.extract_lens_exif(raw, logger=logger.log)

        # 解码: 必须使用 16-bit 以保留 Log 转换所需的动态范围
        prophoto_linear = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.ProPhoto,
            bright=1.0,
            highlight_mode=2, # 2=Blend (防止高光死白)
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
        )
        # 转为 Float32 (0.0 - 1.0) 进行数学运算
        img = prophoto_linear.astype(np.float32) / 65535.0
        
        # 立即释放内存
        del prophoto_linear 
        gc.collect()

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
        img = apply_auto_exposure(img, source_cs, metering_mode, target_gray=0.18, logger=logger)

    # --- Step 3: 基础校正 (WB, Lens, HL/SH) ---
    
    # 3.1 镜头校正
    if lens_correct:
        logger.info("  🔹 [Step 3.1] Lens Correction...")
        img = utils.apply_lens_correction(
            img,
            exif_data=exif_data,
            custom_db_path=custom_db_path,
            logger=logger.log
        )
    
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

    # --- Step 4: 色彩空间转换 (ProPhoto Linear -> Log) ---
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
        img = colour.cctf_encoding(img, function=log_curve_name)
    else:
        logger.info("  🔹 [Step 4] Skipping Color Transform (Log Space is None)")

    # --- Step 5: 应用 LUT ---
    if lut_path:
        logger.info(f"  🔹 [Step 5] Applying LUT {os.path.basename(lut_path)}...")
        try:
            lut = colour.read_LUT(lut_path)
            
            # 3D LUT 使用 Numba 加速
            if isinstance(lut, colour.LUT3D):
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                if lut.table.dtype != np.float32:
                    lut.table = lut.table.astype(np.float32)
                utils.apply_lut_inplace(img, lut.table, lut.domain[0], lut.domain[1])
            else:
                # 1D LUT 使用 colour 库默认方法
                img = lut.apply(img)
            
        except Exception as e:
            logger.error(f"  ❌ applying LUT: {e}")

    # --- Step 6: 保存（使用模块化的文件保存功能）---
    logger.info(f"  💾 Saving to {os.path.basename(output_path)}...")
    save_image(img, output_path, logger)
    
    # --- 最终清理 ---
    del img
    gc.collect()