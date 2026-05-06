"""
EXIF 元数据提取与写入模块
"""

import time
import pyexiv2
from typing import Any, Dict, Optional, Tuple

from loguru import logger as _logger


# =========================================================
# 私有辅助函数
# =========================================================


def _parse_rational(s: str) -> Optional[float]:
    """将有理数字符串（如 '50/1' 或 '28/10'）解析为浮点数。失败时返回 None。"""
    s = str(s).strip()
    if not s:
        return None
    if "/" in s:
        try:
            num, denom = map(float, s.split("/", 1))
            return num / denom if denom != 0 else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_gps_rational(coord_str: str) -> float:
    """将 GPS DMS 有理数字符串（如 '51/1 30/1 0/1'）解析为十进制度数。"""
    parts = str(coord_str).split()
    values = []
    for part in parts:
        v = _parse_rational(part)
        if v is not None:
            values.append(v)
    deg = values[0] if len(values) > 0 else 0.0
    mins = values[1] / 60.0 if len(values) > 1 else 0.0
    secs = values[2] / 3600.0 if len(values) > 2 else 0.0
    return deg + mins + secs


def _decimal_to_dms_rational(decimal_degrees: float) -> str:
    """将十进制度数转换为 DMS 有理数字符串，用于 pyexiv2（例如 '51/1 30/1 0/1000'）。"""
    d = abs(decimal_degrees)
    deg = int(d)
    mins_f = (d - deg) * 60
    mins = int(mins_f)
    secs_num = int(round((mins_f - mins) * 60 * 1000))
    return f"{deg}/1 {mins}/1 {secs_num}/1000"


# =========================================================
# 提取
# =========================================================


def extract_lens_exif(raw_path: str, raw) -> Tuple[dict, Optional[Dict[str, dict]]]:
    """
    使用 pyexiv2 从 RAW 文件中提取 EXIF 和镜头信息。

    Returns:
        Tuple[dict, Optional[Dict[str, dict]]]:
            (镜头/拍摄参数字典, 完整元数据 {'exif', 'iptc', 'xmp'} 或 None)
        Always returns a 2-tuple; metadata is None when pyexiv2 fails.
    """
    result: dict = {}
    metadata: Optional[Dict[str, dict]] = None
    pyexiv2_failed = False

    try:
        # 使用 pyexiv2 读取 EXIF 数据
        # 使用 verify_supported=False 防止某些 raw 格式检查报错
        # 使用 ignore_xmp_decoding_errors 防止 XMP 解析错误
        with pyexiv2.Image(raw_path) as exif_img:
            exif_data = exif_img.read_exif() or {}
            iptc_data = exif_img.read_iptc() or {}
            xmp_data = exif_img.read_xmp() or {}

        metadata = {"exif": exif_data, "iptc": iptc_data, "xmp": xmp_data}

        # 提取镜头校正所需的信息
        # 相机制造商和型号
        result["camera_maker"] = exif_data.get("Exif.Image.Make", "").strip()
        result["camera_model"] = exif_data.get("Exif.Image.Model", "").strip()

        # 镜头信息 (不同厂商的标签可能不同)
        lens_model = (
            exif_data.get("Exif.Photo.LensModel")
            or exif_data.get("Exif.Canon.LensModel")
            or exif_data.get("Exif.Nikon3.Lens")
            or exif_data.get("Exif.Panasonic.LensType")
            or exif_data.get("Exif.OlympusEq.LensModel")
            or ""
        )
        result["lens_model"] = lens_model.strip() if lens_model else ""

        # 镜头制造商
        lens_maker = exif_data.get("Exif.Photo.LensMake", "").strip()
        result["lens_maker"] = lens_maker if lens_maker else ""

        # 焦距
        focal_length = _parse_rational(exif_data.get("Exif.Photo.FocalLength", ""))
        if focal_length is not None:
            result["focal_length"] = focal_length

        # 光圈
        aperture = _parse_rational(exif_data.get("Exif.Photo.FNumber", ""))
        if aperture is not None:
            result["aperture"] = aperture

        # --- 曝光 / 时间 ---
        result["iso"] = (
            exif_data.get("Exif.Photo.ISOSpeedRatings")
            or exif_data.get("Exif.Photo.ISOSpeed")
            or ""
        )
        # 保留原始字符串，如 "1/100"
        result["exposure_time"] = exif_data.get("Exif.Photo.ExposureTime", "")
        result["datetime"] = (
            exif_data.get("Exif.Photo.DateTimeOriginal")
            or exif_data.get("Exif.Image.DateTime")
            or ""
        )
        result["datetime_digitized"] = exif_data.get("Exif.Photo.DateTimeDigitized", "")
        # 亚秒精度 (SubSec 标签与对应日期时间配对)
        result["subsec_time_original"] = exif_data.get(
            "Exif.Photo.SubSecTimeOriginal", ""
        )
        result["subsec_time"] = exif_data.get("Exif.Photo.SubSecTime", "")
        result["subsec_time_digitized"] = exif_data.get(
            "Exif.Photo.SubSecTimeDigitized", ""
        )

        # 35mm等效焦距（用于计算裁切系数）
        focal_35mm = _parse_rational(
            exif_data.get("Exif.Photo.FocalLengthIn35mmFilm", "")
        )
        if focal_35mm is not None:
            result["focal_length_35mm"] = focal_35mm
            fl = result.get("focal_length")
            if fl and fl > 0:
                result["crop_factor"] = round(focal_35mm / fl, 2)

        # 亮度值 (APEX)
        # 注意: 0.0 是一个有效值，所以使用 _parse_rational 来区分缺失和零值
        bv = _parse_rational(exif_data.get("Exif.Photo.BrightnessValue", ""))
        if bv is not None:
            result["brightness_value"] = bv

        # 曝光补偿 (EV)
        eb = _parse_rational(exif_data.get("Exif.Photo.ExposureBiasValue", ""))
        if eb is not None:
            result["exposure_bias"] = eb

        # 拍摄模式标签
        result["flash"] = exif_data.get("Exif.Photo.Flash", "")
        result["exposure_program"] = exif_data.get("Exif.Photo.ExposureProgram", "")
        result["metering_mode"] = exif_data.get("Exif.Photo.MeteringMode", "")

        # GPS 信息 (经纬度和海拔)
        gps_lat_str = exif_data.get("Exif.GPSInfo.GPSLatitude", "")
        gps_lat_ref = exif_data.get("Exif.GPSInfo.GPSLatitudeRef", "")
        gps_lon_str = exif_data.get("Exif.GPSInfo.GPSLongitude", "")
        gps_lon_ref = exif_data.get("Exif.GPSInfo.GPSLongitudeRef", "")
        gps_alt_str = exif_data.get("Exif.GPSInfo.GPSAltitude", "")
        gps_alt_ref = exif_data.get("Exif.GPSInfo.GPSAltitudeRef", "0")

        if gps_lat_str and gps_lon_str:
            try:
                lat = _parse_gps_rational(gps_lat_str)
                lon = _parse_gps_rational(gps_lon_str)
                if gps_lat_ref.upper() == "S":
                    lat = -lat
                if gps_lon_ref.upper() == "W":
                    lon = -lon
                result["gps_latitude"] = lat
                result["gps_longitude"] = lon

                alt = _parse_rational(gps_alt_str) if gps_alt_str else None
                if alt is not None:
                    if str(gps_alt_ref) == "1":
                        alt = -alt
                    result["gps_altitude"] = alt
            except Exception:
                pass

    except Exception as e:
        pyexiv2_failed = True

        # Sony2 目录错误是已知的 exiv2 库限制，不影响其他 EXIF 数据读取
        _logger.error(f"  ❌ [EXIF Error] {e}")
        _logger.info("  ℹ️  Trying to extract basic info from RawSpeed metadata...")

    # 如果 pyexiv2 失败或数据不完整，尝试从 RawSpeed result 获取基本信息
    if pyexiv2_failed:
        try:
            # Check if this is a RawSpeed RawDecodeResult (has .make, .model, .iso_speed)
            if hasattr(raw, 'make') and hasattr(raw, 'model'):
                # RawSpeed RawDecodeResult
                result["camera_maker"] = raw.make
                result["camera_model"] = raw.model
                if raw.iso_speed and raw.iso_speed > 0:
                    result["iso"] = raw.iso_speed
            elif hasattr(raw, 'camera_params'):
                # Legacy rawpy object fallback
                result["camera_maker"] = raw.camera_params.make
                result["camera_model"] = raw.camera_params.model
                result["lens_maker"] = raw.lens_params.make
                result["lens_model"] = raw.lens_params.model
                result["focal_length"] = raw.other_params.focal_len
                result["aperture"] = raw.other_params.aperture
                result["iso"] = raw.other_params.iso_speed
                result["exposure_time"] = raw.other_params.shutter
                if raw.other_params.timestamp > 0:
                    result["datetime"] = time.strftime(
                        "%Y:%m:%d %H:%M:%S", time.localtime(raw.other_params.timestamp)
                    )
        except Exception as e:
            print(f"  ❌ [EXIF Error (Fallback)] {e}")

    # 过滤掉值为 "" 的键，但保留数值零（如 flash=0, exposure_bias=0.0），因为它们是有效的拍摄参数。
    result = {k: v for k, v in result.items() if not (isinstance(v, str) and not v)}

    return result, metadata


# =========================================================
# 写入
# =========================================================

# 这些标签可以直接从 extract_lens_exif 的结果传递到 EXIF 写入，而无需格式化
_PASSTHROUGH_TAGS: Dict[str, str] = {
    "camera_maker": "Exif.Image.Make",
    "camera_model": "Exif.Image.Model",
    "lens_model": "Exif.Photo.LensModel",
    "lens_maker": "Exif.Photo.LensMake",
    "iso": "Exif.Photo.ISOSpeedRatings",
    "datetime_digitized": "Exif.Photo.DateTimeDigitized",
    "subsec_time_original": "Exif.Photo.SubSecTimeOriginal",
    "subsec_time": "Exif.Photo.SubSecTime",
    "subsec_time_digitized": "Exif.Photo.SubSecTimeDigitized",
    "flash": "Exif.Photo.Flash",
    "exposure_program": "Exif.Photo.ExposureProgram",
    "metering_mode": "Exif.Photo.MeteringMode",
}

_ROTATION_TAGS = frozenset(
    {
        "Exif.Image.Orientation",
        "Exif.Thumbnail.JPEGInterchangeFormat",
        "Exif.Thumbnail.JPEGInterchangeFormatLength",
    }
)


def write_exif(output_path: str, exif_metadata: dict, logger: Any) -> bool:
    """
    将完整 EXIF/IPTC/XMP 数据写入输出文件（排除旋转相关标签）。

    Args:
        output_path: 输出文件路径
        exif_metadata: 包含 'exif', 'iptc', 'xmp' 字典的元数据对象
        logger: 日志器

    Returns:
        True on success, False otherwise.
    """
    if not exif_metadata:
        return False

    exif_data = exif_metadata.get("exif", {})
    iptc_data = exif_metadata.get("iptc", {})
    xmp_data = exif_metadata.get("xmp", {})

    if not exif_data and not iptc_data and not xmp_data:
        logger.info("    ℹ️  No EXIF data provided to write.")
        return False

    try:
        # 打开输出文件并写入 EXIF 数据
        output_img = pyexiv2.Image(output_path)
        try:
            # 写入 EXIF 数据，但排除旋转相关标签
            if exif_data:
                # 构建一个过滤后的副本；不要修改调用者的字典。
                filtered_exif = {
                    k: v for k, v in exif_data.items() if k not in _ROTATION_TAGS
                }
                output_img.modify_exif(filtered_exif)

            # 写入 IPTC 数据
            if iptc_data:
                output_img.modify_iptc(iptc_data)

            # 写入 XMP 数据
            if xmp_data:
                output_img.modify_xmp(xmp_data)
        finally:
            output_img.close()

        logger.info("    ✅ EXIF data written successfully (rotation info excluded)")
        return True

    except Exception as e:
        logger.warning(f"    ⚠️  Failed to write full EXIF data: {e}")
        return False


def write_exif_from_dict(output_path: str, exif_dict: dict, logger: Any) -> None:
    """
    从 extract_lens_exif 返回的数据字典构造并写入基本 EXIF 标签。

    Args:
        output_path: 输出文件路径
        exif_dict:   包含相机和镜头信息的字典 (从 extract_lens_exif 返回)
        logger:      日志处理器
    """
    try:
        # 构造基本的 EXIF 数据
        basic_exif: dict = {}

        # --- 直接字符串传递 ---
        for key, tag in _PASSTHROUGH_TAGS.items():
            val = exif_dict.get(key)
            if val is not None and str(val) != "":
                basic_exif[tag] = str(val)

        # --- 有理数 / 格式化值 ---
        if exif_dict.get("focal_length"):
            # 将浮点数转换为分数格式（例如 50.0 -> "50/1"）
            basic_exif["Exif.Photo.FocalLength"] = (
                f"{int(exif_dict['focal_length'] * 10)}/10"
            )

        if exif_dict.get("aperture"):
            # 光圈值（例如 2.8 -> "28/10"）
            basic_exif["Exif.Photo.FNumber"] = f"{int(exif_dict['aperture'] * 10)}/10"

        # 曝光时间 (如果是字符串 "1/100" 直接写入，如果是 float 则转换)
        if exif_dict.get("exposure_time"):
            et = exif_dict["exposure_time"]
            if isinstance(et, (float, int)):
                if et >= 1:
                    basic_exif["Exif.Photo.ExposureTime"] = f"{int(et)}/1"
                else:
                    denom = int(1.0 / et + 0.5)
                    basic_exif["Exif.Photo.ExposureTime"] = f"1/{denom}"
            else:
                basic_exif["Exif.Photo.ExposureTime"] = str(et)

        # 拍摄时间 "YYYY:MM:DD HH:MM:SS"
        if exif_dict.get("datetime"):
            dt = str(exif_dict["datetime"])
            basic_exif["Exif.Photo.DateTimeOriginal"] = dt
            basic_exif["Exif.Image.DateTime"] = dt

        # 35mm 等效焦距
        if exif_dict.get("focal_length_35mm"):
            basic_exif["Exif.Photo.FocalLengthIn35mmFilm"] = str(
                int(round(exif_dict["focal_length_35mm"]))
            )

        # 亮度值
        if exif_dict.get("brightness_value") is not None:
            bv = exif_dict["brightness_value"]
            basic_exif["Exif.Photo.BrightnessValue"] = f"{int(bv * 100)}/100"

        # 曝光补偿
        if exif_dict.get("exposure_bias") is not None:
            eb = exif_dict["exposure_bias"]
            basic_exif["Exif.Photo.ExposureBiasValue"] = f"{int(eb * 100)}/100"

        # GPS 信息
        if (
            exif_dict.get("gps_latitude") is not None
            and exif_dict.get("gps_longitude") is not None
        ):
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
        output_img = pyexiv2.Image(output_path)
        try:
            if basic_exif:
                output_img.modify_exif(basic_exif)
                logger.info(
                    f"    ✅ Basic EXIF written from metadata ({len(basic_exif)} tags)"
                )
            else:
                logger.warning("    ⚠️  No valid EXIF data to write")
        finally:
            output_img.close()

    except Exception as e:
        logger.warning(f"    ⚠️  Failed to write EXIF from dict: {e}")
