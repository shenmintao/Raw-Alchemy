import rawpy
import numpy as np
import colour
import tifffile
from typing import Optional

from . import utils

# 1. 映射：Log 空间名称 -> 对应的线性色域 (Linear Gamut)
LOG_TO_WORKING_SPACE = {
    'F-Log': 'F-Gamut',
    'F-Log2': 'F-Gamut',
    'F-Log2C': 'F-Gamut C',
    'V-Log': 'V-Gamut',
    'N-Log': 'N-Gamut',
    'Canon Log 2': 'Cinema Gamut',
    'Canon Log 3': 'Cinema Gamut',
    'S-Log3': 'S-Gamut3',
    'S-Log3.Cine': 'S-Gamut3.Cine',
    'Arri LogC3': 'ARRI Wide Gamut 3',
    'Arri LogC4': 'ARRI Wide Gamut 4',
    'Log3G10': 'RED Wide Gamut RGB',
}

# 2. 映射：复合名称 -> colour 库识别的 Log 编码函数名称
# 例如：S-Log3.Cine 使用的是 S-Gamut3.Cine 色域，但曲线依然是 S-Log3
LOG_ENCODING_MAP = {
    'S-Log3.Cine': 'S-Log3',
    'F-Log2C': 'F-Log2',
    # 其他名称如果跟 colour 库一致，可以在代码逻辑中直接 fallback
}

# 3. 映射：用户友好的 LUT 空间名 -> colour 库标准名称
LUT_SPACE_MAP = {
    "Rec.709": "ITU-R BT.709",
    "Rec.2020": "ITU-R BT.2020",
}

# 4. 测光模式选项
METERING_MODES = [
    'average',        # 几何平均 (默认)
    'center-weighted',# 中央重点
    'highlight-safe', # 高光保护 (ETTR)
    'hybrid',         # 混合 (平均 + 高光限制)
]

def process_image(
    raw_path: str,
    output_path: str,
    log_space: str,
    lut_path: Optional[str],
    exposure: Optional[float] = None, # 如果是 None 则自动，如果是数字则手动
    lens_correct: bool = True,
    metering_mode: str = 'hybrid',
):
    
    print(f"\n🧪 [Raw Alchemy] Processing: {raw_path}")

    # --- Step 1: 统一解码 (始终保持原始亮度) ---
    print(f"  🔹 [Step 1] Decoding RAW to Linear ProPhoto RGB...")
    with rawpy.imread(raw_path) as raw:
        # 关键修改：bright=1.0。无论手动自动，我们先拿最原始的数据。
        # 这样能保证起点一致。
        prophoto_linear = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.ProPhoto, 
            bright=1.0, 
            highlight_mode=2,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
        )
        img_linear = prophoto_linear.astype(np.float32) / 65535.0
        
    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']

    # --- Step 2: 曝光控制 (二选一) ---
    # 定义最终使用的增益 gain
    gain = 1.0

    if exposure is not None:
        # === 路径 A: 手动曝光 ===
        print(f"  🔹 [Step 2] Manual Exposure Override ({exposure:+.2f} stops)")
        gain = 2.0 ** exposure
        
        # 应用增益
        img_exposed = img_linear * gain

    else:
        # === 路径 B: 自动测光 ===
        print(f"  🔹 [Step 2] Auto Exposure ({metering_mode})")
        
        # 为了复用 utils 里的函数 (假设它们返回的是处理后的图)，我们直接调用
        if metering_mode == 'center-weighted':
            img_exposed = utils.auto_expose_center_weighted(img_linear, source_cs, target_gray=0.18)
        elif metering_mode == 'highlight-safe':
            img_exposed = utils.auto_expose_highlight_safe(img_linear, clip_threshold=1.0)
        elif metering_mode == 'average':
            img_exposed = utils.auto_expose_linear(img_linear, source_cs, target_gray=0.18)
        else:
            # 默认混合模式
            img_exposed = utils.auto_expose_hybrid(img_linear, source_cs, target_gray=0.18)

    # --- Step 3: 镜头校正 ---
    if lens_correct:
        print("  🔹 [Step 3] Applying Lens Correction...")
        img_exposed = utils.apply_lens_correction(img_exposed, raw_path)


    # 经验值：饱和度 1.15 ~ 1.25，对比度 1.0 ~ 1.1
    # 这会让你的 RAW 转换结果在过 LUT 之前就拥有足够的"底料"
    print("  🔹 [Step 3.5] Applying Camera-Match Boost...")
    img_exposed = utils.apply_saturation_and_contrast(img_exposed, saturation=1.25, contrast=1.1)

    # --- Step 4: 转换色彩空间 (Linear -> Log) ---
    log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
    log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)
    
    if not log_color_space_name:
         raise ValueError(f"Unknown Log Space: {log_space}")

    print(f"  🔹 [Step 4] Color Transform (ProPhoto -> {log_color_space_name} -> {log_curve_name})")

    # 4.1 Gamut 变换
    log_linear_image = colour.RGB_to_RGB(
        img_exposed,
        colour.RGB_COLOURSPACES['ProPhoto RGB'],
        colour.RGB_COLOURSPACES[log_color_space_name],
    )
    # Log 编码前必须裁剪负值
    log_linear_image = np.maximum(log_linear_image, 1e-6)

    # 4.2 Curve 编码
    log_image = colour.cctf_encoding(log_linear_image, function=log_curve_name)
    image_to_save = log_image

    # --- Step 5: LUT (可选) ---
    if lut_path:
        print(f"  🔹 [Step 5] Applying LUT {lut_path}...")
        try:
            lut = colour.read_LUT(lut_path)
            image_to_save = lut.apply(log_image)
            image_to_save = np.clip(image_to_save, 0.0, 1.0) # LUT 后防溢出
        except Exception as e:
            print(f"  ❌ [Error] applying LUT: {e}")

    # --- Step 6: 保存 ---
    print(f"  💾 Saving to {output_path}...")
    image_16bit = (image_to_save * 65535).astype(np.uint16)
    tifffile.imwrite(output_path, image_16bit)
    print("  ✅ Done.")