"""
文件输入输出模块
处理各种格式的图像保存
"""
import os
import numpy as np
import tifffile
from PIL import Image
import pillow_heif
from typing import List, Optional
from raw_alchemy.logger import Logger
import pyexiv2

def save_image(
    img: np.ndarray,
    output_path: str,
    logger: Optional[Logger] = None,
    exif_metadata: Optional[dict] = None,
    exif_dict: Optional[dict] = None,
    color_matrix = None
) -> bool:
    """
    保存图像到指定路径，根据扩展名自动选择格式

    Args:
        img: 图像数据 (float32, 0.0-1.0)
        output_path: 输出路径
        logger: 日志处理器
        exif_metadata: 完整的元数据字典 {'exif', 'iptc', 'xmp'} (代替之前的 pyexiv2.Image 对象)
        exif_dict: 从 rawpy 提取的 EXIF 数据字典，当 exif_metadata 写入失败或不可用时降级使用


    Returns:
        bool: 是否保存成功
    """
    if logger is None:
        from .logger import create_logger
        logger = create_logger()

    # 确保数据在有效范围内
    np.clip(img, 0.0, 1.0, out=img)

    file_ext = os.path.splitext(output_path)[1].lower()

    # 某些格式不适合用 pyexiv2 写入完整元数据：
    # - TIFF: 不复制 RAW 的完整 EXIF/IPTC/XMP。RAW 元数据里可能带有
    #   CFA/DNG 专用标签、MakerNote 偏移、缩略图引用、SubIFD/其他 TIFF 结构标签；
    #   这些内容移植到新生成的 RGB TIFF 后，可能破坏 IFD 结构，导致 Photoshop
    #   等严格解析器无法打开。所以 TIFF 仅保留基础 EXIF 的降级写入。
    # - DNG: 避免完整 EXIF 注入破坏 IFD 结构
    allow_full_exif_write = True

    try:
        if file_ext in ['.tif', '.tiff']:
            _save_tiff(img, output_path, logger)
            allow_full_exif_write = False
        elif file_ext == '.dng':
            _save_dng(img, output_path, color_matrix, logger)
            allow_full_exif_write = False
        elif file_ext in ['.heic', '.heif']:
            _save_heif(img, output_path, logger)
        else:
            _save_jpeg_or_other(img, output_path, file_ext, logger)

        exif_written = False
        if allow_full_exif_write and exif_metadata:
            exif_written = _write_exif(output_path, exif_metadata, logger)

        # 如果完整 EXIF 不适用或写入失败，则尝试降级写入基础 EXIF
        if not exif_written and exif_dict:
            _write_exif_from_dict(output_path, exif_dict, logger)

        logger.info(f"  ✅ Saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"  ❌ Failed to save file: {e}")
        import traceback
        traceback.print_exc()
        return False


def _save_tiff(img: np.ndarray, output_path: str, logger: Logger):
    """保存为 16-bit TIFF 格式"""
    logger.info("    Format: TIFF (16-bit, ZLIB Optimized)")
    output_image_uint16 = (img * 65535).astype(np.uint16)

    tifffile.imwrite(
        output_path,
        output_image_uint16,
        photometric='rgb',
        compression='zlib',
        predictor=2,  # 水平差分，提升压缩率
        compressionargs={'level': 8}  # 平衡速度和体积
    )


def _save_heif(img: np.ndarray, output_path: str, logger: Logger):
    """保存为 10-bit HEIF 格式"""
    logger.info("    Format: HEIF (10-bit, High Quality)")
    output_image_uint16 = (img * 65535).astype(np.uint16)

    heif_file = pillow_heif.from_bytes(
        mode='RGB;16',
        size=(output_image_uint16.shape[1], output_image_uint16.shape[0]),
        data=output_image_uint16.tobytes()
    )
    heif_file.save(output_path, quality=85, bit_depth=10)


def _save_dng(img: np.ndarray, output_path: str, color_matrix, logger: Logger):
    """保存为 16-bit DNG 格式 (Adobe Digital Negative)"""
    logger.info("    Format: DNG (16-bit, Linear Raw)")
    output_image_uint16 = (img * 65535).astype(np.uint16)

    # 34892 = LinearRaw (表示数据是线性的且已去马赛克)
    photometric = 34892

    # DNG 基础标签
    dng_version = [1, 4, 0, 0]
    dng_backward_version = [1, 2, 0, 0]
    camera_model = "Raw Alchemy"

    # WhiteLevel: 16-bit 满量程
    white_level = (1 << 16) - 1
    black_level = [0, 0, 0] # RGB 三通道的黑电平

    # 3. 颜色矩阵 (ColorMatrix1) - sRGB -> XYZ (D65) 转换矩阵
    if color_matrix is not None:
        matrix_xyz = color_matrix.flatten().tolist()
    else:
        matrix_xyz = [
            3.2404542, -1.5371385, -0.4985314,
            -0.9692660,  1.8760108,  0.0415560,
            0.0556434, -0.2040259,  1.0572252
        ]
    # 转换浮点矩阵为分数形式 (TIFF SRATIONAL 需要 numerator, denominator)
    # 扁平化列表: [num1, den1, num2, den2, ...]
    matrix_rational = []
    for v in matrix_xyz:
        matrix_rational.extend([int(v * 10000), 10000])

    calibration_illuminant1 = 21 # D65

    # TIFF Data Types (使用整数代码以兼容不同版本的 tifffile)
    TIFF_BYTE = 1
    TIFF_ASCII = 2
    TIFF_SHORT = 3
    TIFF_LONG = 4
    TIFF_RATIONAL = 5
    TIFF_SRATIONAL = 10
    TIFF_FLOAT = 11

    as_shot_neutral_values = [1, 1, 1, 1, 1, 1]
    baseline_exposure_value = [0, 1] # -1.25 EV
    baseline_exposure_offset_value = [0, 1] # 0.0 EV

    profile_name = "Linear Flat"

    # ProfileToneCurve: 定义一条直线 (0.0->0.0, 1.0->1.0)
    # 这会覆盖 Adobe 默认的强对比曲线
    # 格式: [Input1, Output1, Input2, Output2] (浮点数)
    profile_tone_curve = [0.0, 0.0, 1.0, 1.0]

    # ProfileEmbedPolicy: 允许复制和使用此配置
    profile_embed_policy = [0]

    # DefaultBlackRender: 0 = Auto, 1 = None (禁止 Adobe 自动渲染黑点)
    # 设为 1 可以防止 Adobe 擅自切掉暗部细节
    default_black_render = [1]

    # 4. 组装 Extra Tags
    # 注意: Tags 必须按 ID 升序排列
    extratags = [
        (254, TIFF_LONG, 1, 0),                        # NewSubfileType (Main Image)
        (274, TIFF_SHORT, 1, 1),                       # Orientation (TopLeft)
        (50706, TIFF_BYTE, 4, dng_version),            # DNGVersion
        (50707, TIFF_BYTE, 4, dng_backward_version),   # DNGBackwardVersion
        (50708, TIFF_ASCII, len(camera_model)+1, camera_model), # UniqueCameraModel
        (50714, TIFF_SHORT, 3, black_level),           # BlackLevel
        (50717, TIFF_SHORT, 1, white_level),           # WhiteLevel
        (50721, TIFF_SRATIONAL, 9, matrix_rational),   # ColorMatrix1
        (50728, TIFF_RATIONAL, 3, as_shot_neutral_values), # AsShotNeutral
        (50730, TIFF_SRATIONAL, 1, baseline_exposure_value), # BaselineExposure (-1.25 EV)
        (50778, TIFF_SHORT, 1, calibration_illuminant1),# CalibrationIlluminant1 (D50)
        (51109, TIFF_RATIONAL, 1, baseline_exposure_offset_value), # BaselineExposureOffset (0 EV)

                # --- 新增的 Profile 标签 ---
        (50933, TIFF_LONG, 1, profile_embed_policy),               # ProfileEmbedPolicy
        (50936, TIFF_ASCII, len(profile_name)+1, profile_name),    # ProfileName
        (50940, TIFF_FLOAT, 4, profile_tone_curve),                 # ProfileToneCurve (强制线性)
        (51110, TIFF_LONG, 1, default_black_render),               # DefaultBlackRender (禁止黑点修正)
    ]

    # 使用无损压缩以保持图像质量 (虽然用户建议 compression=None，但 dng 通常可以接受无损压缩，先保持基本兼容性)
    # 若要完全匹配用户建议，可移除 comparison
    # 考虑到文件体积，我们仍然不使用 ZLIB 压缩，或者使用 compression=None 确保兼容性

    tifffile.imwrite(
        output_path,
        output_image_uint16,
        # compression=52546, # DNG
        # predictor=2,
        photometric=photometric,
        planarconfig=1,
        extratags=extratags,
        description="Linear RGB DNG generated by Raw Alchemy",
        metadata=None
    )


def _save_jpeg_or_other(img: np.ndarray, output_path: str, file_ext: str, logger: Logger):
    """保存为 8-bit JPEG 或其他格式"""
    logger.info(f"    Format: {file_ext.upper()} (8-bit High Quality)")

    # 转换为 8-bit（img 已经在 save_image 中被 clip 过了）
    output_image_uint8 = (img * 255).astype(np.uint8)

    # JPEG 特殊优化参数
    save_params = {}
    if file_ext in ['.jpg', '.jpeg']:
        save_params = {
            'quality': 90,
            'subsampling': 2,
            'optimize': True
        }

    # 保存时暂时不写入 EXIF，通过 _write_exif 将 EXIF 注入文件
    Image.fromarray(output_image_uint8).save(output_path, **save_params)


def _write_exif(output_path: str, exif_metadata: dict, logger: Logger) -> bool:
    """
    将 EXIF 数据写入输出文件（排除旋转信息）
    Args:
        output_path: 输出文件路径
        exif_metadata: 包含 'exif', 'iptc', 'xmp' 字典的元数据对象
        logger: 日志器
    Returns: True if successful, False otherwise
    """
    try:
        if not exif_metadata:
            return False

        exif_data = exif_metadata.get('exif', {})
        iptc_data = exif_metadata.get('iptc', {})
        xmp_data = exif_metadata.get('xmp', {})

        if not exif_data and not iptc_data and not xmp_data:
            logger.info("    ℹ️  No EXIF data provided to write.")
            return False

        # 打开输出文件并写入 EXIF 数据
        output_img = pyexiv2.Image(output_path)

        # 写入 EXIF 数据，但排除旋转相关标签
        if exif_data:
            # 移除旋转相关的 EXIF 标签
            problematic_tags = ['Exif.Image.Orientation', 'Exif.Thumbnail.JPEGInterchangeFormat', 'Exif.Thumbnail.JPEGInterchangeFormatLength']

            for tag in problematic_tags:
                if tag in exif_data:
                    del exif_data[tag]

            output_img.modify_exif(exif_data)

        # 写入 IPTC 数据
        if iptc_data:
            output_img.modify_iptc(iptc_data)

        # 写入 XMP 数据
        if xmp_data:
            output_img.modify_xmp(xmp_data)

        output_img.close()
        logger.info("    ✅ EXIF data written successfully (rotation info excluded)")
        return True

    except Exception as e:
        logger.warning(f"    ⚠️  Failed to write full EXIF data: {e}")
        return False


def _write_exif_from_dict(output_path: str, exif_dict: dict, logger: Logger):
    """
    从 rawpy 提取的数据字典构造并写入基本 EXIF 标签

    Args:
        output_path: 输出文件路径
        exif_dict: 包含相机和镜头信息的字典 (从 utils.extract_lens_exif 返回)
        logger: 日志处理器
    """
    try:
        # 打开输出文件
        output_img = pyexiv2.Image(output_path)

        # 构造基本的 EXIF 数据
        basic_exif = {}

        # 相机信息
        if exif_dict.get('camera_maker'):
            basic_exif['Exif.Image.Make'] = exif_dict['camera_maker']

        if exif_dict.get('camera_model'):
            basic_exif['Exif.Image.Model'] = exif_dict['camera_model']

        # 镜头信息
        if exif_dict.get('lens_model'):
            basic_exif['Exif.Photo.LensModel'] = exif_dict['lens_model']

        if exif_dict.get('lens_maker'):
            basic_exif['Exif.Photo.LensMake'] = exif_dict['lens_maker']

        # 拍摄参数
        if exif_dict.get('focal_length'):
            # 将浮点数转换为分数格式（例如 50.0 -> "50/1"）
            focal_mm = exif_dict['focal_length']
            basic_exif['Exif.Photo.FocalLength'] = f"{int(focal_mm * 10)}/10"

        if exif_dict.get('aperture'):
            # 光圈值（例如 2.8 -> "28/10"）
            aperture = exif_dict['aperture']
            basic_exif['Exif.Photo.FNumber'] = f"{int(aperture * 10)}/10"

        # ISO
        if exif_dict.get('iso'):
            basic_exif['Exif.Photo.ISOSpeedRatings'] = str(exif_dict['iso'])

        # 曝光时间 (如果是字符串 "1/100" 直接写入，如果是 float 则转换)
        if exif_dict.get('exposure_time'):
            et = exif_dict['exposure_time']
            if isinstance(et, (float, int)):
                # Convert float seconds to fractional string approx
                if et >= 1:
                    basic_exif["Exif.Photo.ExposureTime"] = f"{int(et)}/1"
                else:
                    # e.g. 0.01 -> 1/100
                    denom = int(1.0 / et + 0.5)
                    basic_exif["Exif.Photo.ExposureTime"] = f"1/{denom}"
            else:
                basic_exif['Exif.Photo.ExposureTime'] = str(et)

        # 拍摄时间 "YYYY:MM:DD HH:MM:SS"
        if exif_dict.get('datetime'):
            basic_exif["Exif.Photo.DateTimeOriginal"] = str(exif_dict["datetime"])
            basic_exif["Exif.Image.DateTime"] = str(exif_dict["datetime"])

        # 数字化时间
        if exif_dict.get("datetime_digitized"):
            basic_exif["Exif.Photo.DateTimeDigitized"] = str(
                exif_dict["datetime_digitized"]
            )

        # 亚秒精度 (SubSec 标签与对应日期时间配对)
        if exif_dict.get("subsec_time_original"):
            basic_exif["Exif.Photo.SubSecTimeOriginal"] = str(
                exif_dict["subsec_time_original"]
            )
        if exif_dict.get("subsec_time"):
            basic_exif["Exif.Photo.SubSecTime"] = str(exif_dict["subsec_time"])
        if exif_dict.get("subsec_time_digitized"):
            basic_exif["Exif.Photo.SubSecTimeDigitized"] = str(
                exif_dict["subsec_time_digitized"]
            )

        # 35mm 等效焦距
        if exif_dict.get("focal_length_35mm"):
            basic_exif["Exif.Photo.FocalLengthIn35mmFilm"] = str(
                int(round(exif_dict["focal_length_35mm"]))
            )

        # 亮度值 (APEX SRATIONAL)
        if exif_dict.get("brightness_value") is not None:
            bv = exif_dict["brightness_value"]
            basic_exif["Exif.Photo.BrightnessValue"] = f"{int(bv * 100)}/100"

        # 曝光补偿 (APEX SRATIONAL)
        if exif_dict.get("exposure_bias") is not None:
            eb = exif_dict["exposure_bias"]
            basic_exif["Exif.Photo.ExposureBiasValue"] = f"{int(eb * 100)}/100"

        # 闪光灯
        if exif_dict.get("flash") is not None and exif_dict["flash"] != "":
            basic_exif["Exif.Photo.Flash"] = str(exif_dict["flash"])

        # 曝光程序
        if (
            exif_dict.get("exposure_program") is not None
            and exif_dict["exposure_program"] != ""
        ):
            basic_exif["Exif.Photo.ExposureProgram"] = str(
                exif_dict["exposure_program"]
            )

        # 测光模式
        if (
            exif_dict.get("metering_mode") is not None
            and exif_dict["metering_mode"] != ""
        ):
            basic_exif["Exif.Photo.MeteringMode"] = str(exif_dict["metering_mode"])

        # GPS 信息
        if (
            exif_dict.get("gps_latitude") is not None
            and exif_dict.get("gps_longitude") is not None
        ):

            def _decimal_to_dms_rational(decimal_degrees):
                """Convert decimal degrees to DMS rational string for pyexiv2."""
                d = abs(decimal_degrees)
                deg = int(d)
                mins_f = (d - deg) * 60
                mins = int(mins_f)
                secs_f = (mins_f - mins) * 60
                secs_num = int(round(secs_f * 1000))
                return f"{deg}/1 {mins}/1 {secs_num}/1000"

            lat = exif_dict["gps_latitude"]
            lon = exif_dict["gps_longitude"]
            basic_exif["Exif.GPSInfo.GPSLatitudeRef"] = "N" if lat >= 0 else "S"
            basic_exif["Exif.GPSInfo.GPSLatitude"] = _decimal_to_dms_rational(lat)
            basic_exif["Exif.GPSInfo.GPSLongitudeRef"] = "E" if lon >= 0 else "W"
            basic_exif["Exif.GPSInfo.GPSLongitude"] = _decimal_to_dms_rational(lon)

            if exif_dict.get("gps_altitude") is not None:
                alt = exif_dict["gps_altitude"]
                basic_exif["Exif.GPSInfo.GPSAltitudeRef"] = "1" if alt < 0 else "0"
                basic_exif["Exif.GPSInfo.GPSAltitude"] = f"{int(abs(alt) * 100)}/100"

        # 写入 EXIF 数据
        if basic_exif:
            output_img.modify_exif(basic_exif)
            output_img.close()
            logger.info(f"    ✅ Basic EXIF written from rawpy data ({len(basic_exif)} tags)")
        else:
            output_img.close()
            logger.warning("    ⚠️  No valid EXIF data to write from rawpy")

    except Exception as e:
        logger.warning(f"    ⚠️  Failed to write EXIF from dict: {e}")
