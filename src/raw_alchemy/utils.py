import os
import sys
from functools import lru_cache
from typing import Optional, Tuple, Dict
import numpy as np
import colour
from loguru import logger
from raw_alchemy import config, lensfun_wrapper as lf
from raw_alchemy.math_ops import (
    apply_matrix_inplace,
    apply_lut_inplace,
    apply_saturation_contrast_inplace,
    apply_highlight_shadow_inplace,
    apply_gain_inplace,
    linear_to_srgb_inplace,
    srgb_to_linear_inplace,
    perspective_warp_kernel,
    compute_perspective_matrix,
    white_balance_matrix,
)
import cv2

@lru_cache(maxsize=8)
def _load_lut_cached(resolved_path: str, mtime_ns: int, size: int):
    """Parse one immutable file revision and normalize hot-path arrays."""
    del mtime_ns, size  # values are part of the cache key
    logger.info(f"Parsing LUT: {os.path.basename(resolved_path)}")
    lut = colour.read_LUT(resolved_path)
    if isinstance(lut, colour.LUT3D):
        # The executor and ONNX graph both consume float32. Normalize once on
        # cache fill instead of re-casting the table/domain for every slider
        # request. colour's public setters coerce back to its default float64,
        # so keep explicit prepared views alongside the parsed object.
        lut._raw_alchemy_table32 = np.ascontiguousarray(lut.table, dtype=np.float32)
        lut._raw_alchemy_domain32 = np.ascontiguousarray(lut.domain, dtype=np.float32)
    return lut


def load_lut_cached(lut_path: str):
    """Cache parsed LUTs by file revision, reloading when a LUT is edited."""
    resolved, mtime_ns, size = lut_revision_token(lut_path)
    if mtime_ns is None:
        raise FileNotFoundError(resolved)
    return _load_lut_cached(resolved, mtime_ns, size)


def lut_revision_token(lut_path: str) -> tuple[str, Optional[int], Optional[int]]:
    """Hashable LUT identity used by parsed, pipeline and output caches."""
    resolved = os.path.abspath(os.path.expanduser(os.fspath(lut_path)))
    try:
        stat = os.stat(resolved)
    except OSError:
        return (resolved, None, None)
    return (resolved, int(stat.st_mtime_ns), int(stat.st_size))


def resource_path(relative_path):
    """
    鑾峰彇璧勬簮鐨勭粷瀵硅矾寰勶紝鍏煎 Dev, PyInstaller, 鍜?Nuitka (Onefile & Standalone).
    """
    # 1. 澶勭悊 PyInstaller (瀹冩妸璧勬簮瑙ｅ帇鍒?_MEIPASS)
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    
    # 2. 澶勭悊 Nuitka 鍜?鏅€?Python 鑴氭湰
    # Nuitka 浼氬阀濡欏湴澶勭悊 __file__锛屼娇鍏舵寚鍚戣В鍘嬪悗鐨勪复鏃剁洰褰?Onefile)鎴栧彂甯冪洰褰?Standalone)
    else:
        # 鑾峰彇褰撳墠鑴氭湰鎵€鍦ㄧ殑鐩綍
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

# =========================================================
# Taichi 鍔犻€熸牳鍑芥暟 (In-Place / 鏃犲唴瀛樺垎閰?
# =========================================================


def compute_histogram_fast(img_array, bins=100, sample_rate=4):
    """
    蹇€熻绠?RGB 涓夐€氶亾鐩存柟鍥?

    Args:
        img_array: HxWx3 numpy array, uint8 [0,255] or float32 [0,1]
        bins: number of histogram bins
        sample_rate: subsample rate (e.g., 4 means take every 4th pixel)

    Returns:
        list of 3 histogram arrays (R, G, B) as float arrays
    """
    try:
        # 鏁版嵁楠岃瘉
        if img_array is None or img_array.size == 0:
            return None

        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return None

        sample_rate = max(1, int(sample_rate))
        sample = img_array[::sample_rate, ::sample_rate, :]

        hist_data = []
        for channel in range(3):
            if sample.dtype == np.uint8:
                # Histogramming the native bytes with a [0,255] range is
                # exactly equivalent to the old uint8->float32 /255 path and
                # avoids a sampled RGB float frame plus three channel copies.
                channel_data = sample[:, :, channel]
                hist, _ = np.histogram(channel_data, bins=bins, range=(0, 255))
            else:
                channel_data = np.array(
                    sample[:, :, channel], dtype=np.float32, copy=True, order='C'
                )
                np.clip(channel_data, 0.0, 1.0, out=channel_data)
                hist, _ = np.histogram(channel_data, bins=bins, range=(0.0, 1.0))
            hist_data.append(hist.astype(np.float64))
        
        return hist_data
    except Exception as e:
        # 璁板綍閿欒浣嗕笉鎶涘嚭寮傚父
        logger.warning(f"Histogram computation failed: {type(e).__name__}: {e}")
        return None


def compute_waveform_fast(img_array, bins=100, sample_rate=4):
    """
    蹇€熻绠椾寒搴︽尝褰㈠浘鏁版嵁
    绫讳技杈捐姮濂囩殑娉㈠舰鍥撅紝鏄剧ず鍥惧儚鐨勪寒搴﹀垎甯?

    Args:
        img_array: HxWx3 numpy array, uint8 [0,255] or float32 [0,1]
        bins: number of vertical bins (浜害绾у埆)
        sample_rate: horizontal subsample rate (姘村钩閲囨牱鐜?

    Returns:
        numpy array of shape [sampled_width, bins] - 浜害娉㈠舰鏁版嵁
    """
    try:
        # 鏁版嵁楠岃瘉
        if img_array is None or img_array.size == 0:
            return None

        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return None

        _h, w, _c = img_array.shape
        sample_rate = max(1, int(sample_rate))

        # 姘村钩鏂瑰悜閲囨牱浠ユ彁楂樻€ц兘
        sampled_width = w // sample_rate
        if sampled_width == 0:
            sampled_width = 1

        # 鍨傜洿涔熷瓙閲囨牱锛堜笌histogram涓€鑷达級锛岄伩鍏嶅叏鍥捐绠?
        v_step = max(1, sample_rate // 2)
        # Only every ``sample_rate``-th column contributes to the result. The
        # retired implementation converted all columns to float/luma and then
        # discarded most of them inside a Python loop (24MP ~= 0.9s).
        img_sub = img_array[
            ::v_step,
            :sampled_width * sample_rate:sample_rate,
            :,
        ]

        # uint8 鈫?float32 [0,1]锛堜粎瀵瑰瓙閲囨牱鍚庣殑灏忔暟鎹級
        img_f = np.array(img_sub, dtype=np.float32, copy=True, order='C')
        if img_sub.dtype == np.uint8:
            img_f *= np.float32(1.0 / 255.0)
        np.clip(img_f, 0.0, 1.0, out=img_f)

        # 璁＄畻浜害锛堜娇鐢?Rec.709 绯绘暟锛?
        # Y = 0.2126*R + 0.7152*G + 0.0722*B
        luma = np.empty(img_f.shape[:2], dtype=np.float32)
        np.multiply(img_f[:, :, 0], np.float32(0.2126), out=luma)
        luma += img_f[:, :, 1] * np.float32(0.7152)
        luma += img_f[:, :, 2] * np.float32(0.0722)

        bin_indices = (luma * np.float32(bins)).astype(np.int32)
        np.clip(bin_indices, 0, bins - 1, out=bin_indices)
        offsets = np.arange(sampled_width, dtype=np.int32) * int(bins)
        bin_indices += offsets[None, :]
        waveform = np.bincount(
            bin_indices.ravel(), minlength=sampled_width * bins
        ).reshape(sampled_width, bins).astype(np.float32)
        
        # 褰掍竴鍖?
        max_val = np.max(waveform)
        if max_val > 0:
            waveform /= max_val
        
        return waveform
    except Exception as e:
        # 璁板綍閿欒浣嗕笉鎶涘嚭寮傚父
        logger.warning(f"Waveform computation failed: {type(e).__name__}: {e}")
        return None

# =========================================================
# 杈呭姪璁＄畻鍑芥暟 (鐢ㄤ簬娴嬪厜)
# =========================================================

def get_luminance_coeffs(colourspace):
    """Return RGB luminance coefficients from a colour-science RGB space."""
    return colourspace.matrix_RGB_to_XYZ[1, :]


def get_working_colourspace():
    return colour.RGB_COLOURSPACES[config.WORKING_SPACE]

def get_subsampled_view(img, target_size=1024):
    """
    鑾峰彇鍥惧儚鐨勪笅閲囨牱瑙嗗浘銆?
    瀵逛簬娴嬪厜鏉ヨ锛屽垎鏋?1000px 瀹界殑缂╃暐鍥惧拰鍒嗘瀽 8000px 鐨勫師鍥撅紝缁撴灉宸紓鍙拷鐣ヤ笉璁°€?
    """
    h, w, _ = img.shape
    # 璁＄畻姝ラ暱锛屼娇寰楅暱杈瑰ぇ绾︿负 target_size
    step = max(1, max(h, w) // target_size)
    # Numpy鍒囩墖鏄鍥?View)锛屼笉鍗犵敤鏂板唴瀛?
    return img[::step, ::step, :]

# =========================================================
# 涓氬姟閫昏緫鍑芥暟 (浼樺寲鐗?
# =========================================================

def apply_saturation_and_contrast(img_linear, saturation=1.25, contrast=1.10, colourspace=None):
    """
    In-Place 搴旂敤楗卞拰搴﹀拰瀵规瘮搴︺€?
    """
    import colour
    
    if colourspace is None:
        colourspace = get_working_colourspace()
    
    luma_coeffs = get_luminance_coeffs(colourspace).astype(np.float32)
    
    if not img_linear.flags['C_CONTIGUOUS']:
        img_linear = np.ascontiguousarray(img_linear)
        
    apply_saturation_contrast_inplace(
        img_linear,
        float(saturation),
        float(contrast),
        0.18,
        luma_coeffs
    )
    return img_linear

def apply_white_balance(img_linear, temp=0.0, tint=0.0):
    """
    Apply White Balance.
    temp: -100 to 100 (Blue <-> Amber)
    tint: -100 to 100 (Green <-> Magenta)
    """
    if not img_linear.flags['C_CONTIGUOUS']:
        img_linear = np.ascontiguousarray(img_linear)

    apply_matrix_inplace(img_linear, white_balance_matrix(temp, tint))
    return img_linear

def apply_highlight_shadow(img_linear, highlight=0.0, shadow=0.0, colourspace=None):
    """
    highlight: -100 to 100
    shadow: -100 to 100
    """
    import colour
    if colourspace is None:
        colourspace = get_working_colourspace()
    luma_coeffs = get_luminance_coeffs(colourspace).astype(np.float32)

    # Normalize inputs to -1.0 to 1.0 roughly
    h_val = highlight / 100.0
    s_val = shadow / 100.0
    
    if not img_linear.flags['C_CONTIGUOUS']:
        img_linear = np.ascontiguousarray(img_linear)

    apply_highlight_shadow_inplace(img_linear, float(h_val), float(s_val), luma_coeffs)
    return img_linear

# ----------------- 闀滃ご鏍℃ (淇濇寔閫昏緫锛屼紭鍖栨敞閲? -----------------

def apply_lens_correction(image: np.ndarray, exif_data: dict, custom_db_path: Optional[str] = None, **kwargs) -> np.ndarray:
    """
    闀滃ご鏍℃閫氬父闇€瑕佸嚑浣曞彉鎹紝寰堥毦瀹屽叏 In-Place銆?
    杩欐槸鏁翠釜娴佺▼涓皯鏁板嚑涓繀鐒朵細浜х敓鍐呭瓨鎷疯礉鐨勫湴鏂广€?
    """
    # exif_data is now passed directly
    
    # 绠€鍗曠殑瀛楀吀鍚堝苟
    params = {**exif_data, **kwargs}
    
    # 蹇呰鐨?key 妫€鏌?
    if not params.get('camera_model') or not params.get('lens_model'):
        logger.warning("  鈿狅笍  [Lens] Missing camera model info, skipping.")
        return image
    
    if not params.get('focal_length') or not params.get('aperture'):
        logger.warning("  鈿狅笍  [Lens] Missing optical info, skipping.")
        return image
    
    logger.info(f"  馃К [Lens] {params.get('camera_maker')} {params.get('camera_model')} + {params.get('lens_model')}")
    
    try:
        # lensfun_wrapper 鍐呴儴浼氳皟鐢?cv2.remap
        # 杩欏繀鐒惰繑鍥炴柊鍥惧儚
        lens_kwargs = dict(
            camera_maker=params.get('camera_maker'),
            camera_model=params.get('camera_model'),
            lens_maker=params.get('lens_maker'),
            lens_model=params.get('lens_model'),
            focal_length=params.get('focal_length'),
            aperture=params.get('aperture'),
            crop_factor=params.get('crop_factor'),
            correct_distortion=params.get('correct_distortion', True),
            correct_tca=params.get('correct_tca', True),
            correct_vignetting=params.get('correct_vignetting', True),
            distance=params.get('distance', 1000.0),
            custom_db_path=custom_db_path,
        )
        estimated_map_bytes = image.shape[0] * image.shape[1] * 25
        map_limit = int(config.DISTORTION_MAP_CACHE_LIMIT_MB) * 1024 * 1024
        if estimated_map_bytes > map_limit:
            corrected = lf.apply_lens_correction_tiled(image=image, **lens_kwargs)
            if corrected is None:
                corrected = image
        else:
            corrected = lf.apply_lens_correction(image=image, **lens_kwargs)
        
        # 鏄惧紡甯姪 GC (铏界劧 Python 浼氳嚜鍔ㄥ鐞嗭紝浣嗗湪澶у唴瀛樺帇鍔涗笅 explicit is better)
        # 杩欓噷鍘熸潵鐨?image 寮曠敤璁℃暟浼氬噺灏戯紝濡傛灉澶栭潰娌℃湁寮曠敤锛屾棫鍐呭瓨浼氳閲婃斁
        return corrected
        
    except Exception as e:
        logger.error(f"  鉂?[Lens Error] {e}")
        return image # 澶辫触鍒欒繑鍥炲師鍥?

def extract_lens_exif(raw_path: str, raw=None) -> Tuple[dict, Optional[Dict[str, dict]]]:
    """
    Extract EXIF and lens info from RAW file.

    Delegates to raw_alchemy.exif.extract_lens_exif which uses pyexiv2 for
    EXIF reading.

    Args:
        raw_path: Path to the RAW file
        raw: Optional rawpy object for fallback metadata extraction

    Returns:
        Tuple[dict, Optional[Dict[str, dict]]]: (lens correction params, full metadata dict or None)
    """
    from raw_alchemy.exif import extract_lens_exif as _extract
    return _extract(raw_path, raw)


def compute_denoise_normalization_gain(img_linear: np.ndarray,
                                        max_gain: float = 64.0,
                                        target_gray: float = 0.18) -> float:
    """
    鏍规嵁鍥惧儚瀹為檯浜害璁＄畻闄嶅櫔褰掍竴鍖栧鐩娿€?

    鐢ㄤ簬闄嶅櫔鍓嶇殑浜害褰掍竴鍖栵細鍏堝皢鏆楀浘鎻愪寒鍒版爣鍑嗘按骞筹紙18% gray锛夛紝
    闄嶅櫔鍚庡啀闄ゅ洖锛岃繖鏍锋爣鍑嗛檷鍣櫒灏辫兘姝ｅ父宸ヤ綔銆?

    鐩存帴鍩轰簬鍥惧儚鍐呭璁＄畻锛屼笉渚濊禆 EXIF锛圗XIF 鏃犳硶鎻愪緵鍦烘櫙浜害淇℃伅锛夈€?
    浣跨敤鍑犱綍骞冲潎浜害锛屼笌 metering.py 鐨?AverageMeteringStrategy 涓€鑷淬€?

    Args:
        img_linear: 绾挎€?ProPhoto RGB 鍥惧儚锛宖loat32 [0, 1]
        max_gain: 鏈€澶у鐩婂€嶆暟锛堥粯璁?64.0锛屽嵆 +6 妗ｏ級
        target_gray: 鐩爣鐏板害锛堥粯璁?0.18锛屾爣鍑?18% gray锛?

    Returns:
        褰掍竴鍖栧鐩婏紙>= 1.0锛夈€傝嫢鍥惧儚宸茶冻澶熶寒鍒欒繑鍥?1.0銆?
    """
    import math

    try:
        sample = get_subsampled_view(img_linear)
        source_cs = get_working_colourspace()
        coeffs = get_luminance_coeffs(source_cs)
        luminance = np.dot(sample, coeffs)

        luminance = np.maximum(luminance, 1e-10)
        avg_log_lum = np.mean(np.log(luminance + 1e-6))
        avg_lum = np.exp(avg_log_lum)

        if avg_lum < 1e-6:
            return 1.0

        gain = target_gray / avg_lum

        # 浠呮彁浜笉鍘嬫殫锛屼笖闄愬箙
        gain = max(1.0, min(gain, max_gain))

        if gain > 1.0:
            ev_stops = math.log2(gain)
            logger.info(f"  馃搻 [Denoise Norm] avg_lum={avg_lum:.6f}, gain={gain:.2f}x (+{ev_stops:.1f} stops)")

        return gain

    except Exception as e:
        logger.warning(f"  鈿狅笍 [Denoise Norm] Failed to compute gain: {e}")
        return 1.0


def get_version_info():
    """Get version and license information"""
    try:
        from raw_alchemy import __version__
        version = __version__
    except ImportError:
        version = "0.0.0"
    
    current_year = "2025"
    license_info = f"Copyright 漏 {current_year} MinQ.\nAGPL-V3 License."
    return version, license_info

def apply_geometry(img: np.ndarray, rotation: int = 0, flip_h: bool = False, flip_v: bool = False) -> np.ndarray:
    """
    搴旂敤鍑犱綍鍙樻崲锛堟棆杞拰缈昏浆锛?
    rotation: degrees clockwise. If divisible by 90, uses fast numpy.rot90, 
              otherwise uses higher quality interpolation.
    """
    if rotation == 0 and not flip_h and not flip_v:
        return img
    
    out = img
    
    # Rotation
    # Normalize rotation to [0, 360)
    rotation = rotation % 360
    
    # 1. Fast 90-degree steps rotation
    if rotation % 90 == 0:
        k = 0
        if rotation == 90:
            k = -1 # numpy rot90 is CCW
        elif rotation == 180:
            k = 2
        elif rotation == 270:
            k = 1
            
        if k != 0:
            out = np.rot90(out, k=k)
            
    # 2. Arbitrary angle rotation
    else:
        # cv2.getRotationMatrix2D uses CCW-positive angles. The caller's
        # ``rotation`` convention is CW-positive, so we negate.
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -rotation, 1.0)

        # reshape=True equivalent: compute the new bounding box and shift
        # the matrix so the rotated image fits exactly.
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        # Use ceil so we never crop a pixel at the rotated bbox edge.
        new_w = int(np.ceil(h * sin + w * cos))
        new_h = int(np.ceil(h * cos + w * sin))
        M[0, 2] += new_w / 2.0 - w / 2.0
        M[1, 2] += new_h / 2.0 - h / 2.0

        # Bilinear is sufficient for interactive preview rotation.
        out = cv2.warpAffine(
            out, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
    if flip_h:
        out = np.fliplr(out)
        
    if flip_v:
        out = np.flipud(out)
        
    # Ensure contiguous
    if not out.flags['C_CONTIGUOUS']:
        out = np.ascontiguousarray(out)
        
    return out


def apply_perspective(img: np.ndarray, corners: Tuple[Tuple[float, float], ...] = None) -> np.ndarray:
    """
    Apply perspective correction using 4 corner control points.
    
    Args:
        img: Input image HxWx3 float32
        corners: 4 points in normalized coords (0-1) as ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
                 Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
                 Default (None) or corners at image corners = no transformation
    
    Returns:
        Perspective-corrected image
    """
    # Default corners (no transformation)
    default_corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    
    if corners is None:
        return img

    corners = tuple(tuple(point) for point in corners)
    if corners == default_corners:
        return img
    
    # Validate corners
    if len(corners) != 4:
        logger.warning(f"[Perspective] Invalid corners count: {len(corners)}, expected 4")
        return img
    
    h, w = img.shape[:2]
    
    # Compute perspective matrix
    _, M_inv = compute_perspective_matrix(corners, w, h)
    
    # Ensure input is contiguous float32
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # Allocate output
    dst = np.zeros_like(img)
    
    # Apply perspective warp (Taichi accelerated)
    perspective_warp_kernel(img, dst, M_inv)
    
    return dst


def apply_crop(img: np.ndarray, crop_rect: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Apply crop to image using normalized coordinates.
    crop_rect: (x, y, w, h) where all values are 0.0-1.0
    """
    if not crop_rect or crop_rect == (0.0, 0.0, 1.0, 1.0):
        return img
        
    h, w = img.shape[:2]
    x_norm, y_norm, w_norm, h_norm = crop_rect
    
    # Validation
    if w_norm <= 0 or h_norm <= 0:
        return img
        
    x = int(x_norm * w)
    y = int(y_norm * h)
    cw = int(w_norm * w)
    ch = int(h_norm * h)
    
    # Boundary checks
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    cw = max(1, min(cw, w - x))
    ch = max(1, min(ch, h - y))
    
    # Slice
    # Force copy to allow previous large buffers to be GC'd if needed, 
    # but here we might want a view for speed? 
    # Actually for caching, a copy is safer to avoid keeping the huge rotated image alive if not needed.
    return img[y:y+ch, x:x+cw, :].copy() 

def get_rotated_bounds(w: int, h: int, angle: float) -> Tuple[int, int]:
    """Calculate the bounding box size of a rotated rectangle"""
    angle_rad = np.radians(angle)
    sin_a = abs(np.sin(angle_rad))
    cos_a = abs(np.cos(angle_rad))
    
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    return new_w, new_h

def get_max_rect_in_rotated(w: int, h: int, angle: float) -> Tuple[float, float, float, float]:
    """
    Calculate the maximum axis-aligned rectangle (in normalized coords) 
    that fits inside the rotated image (which itself has black corners).
    
    This is complex geometry. Simplified approach:
    For a given angle, we want to find the largest rect with aspect ratio w/h?
    Or just *a* valid rect.
    
    Actually, usually we just want to know "how much do I need to zoom in" to avoid black borders.
    """
    # This is a placeholder for the math. 
    # The UI will likely handle the interactive logic.
    # But strictly for a "Default Safe Crop", we can compute it.
    # Ref: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    
    if angle % 180 == 0:
        return (0.0, 0.0, 1.0, 1.0)
        
    w_rot, h_rot = get_rotated_bounds(w, h, angle)
    
    # We want to find x,y,w,h in the ROTATED image coordinate system (0..w_rot, 0..h_rot)
    # that represents the maximum area inscribed rectangle.
    # That is mathematically involved.
    # For now, let's return a safe full rect (0,0,1,1) effectively saying "User deals with it"
    # Or implement a simple heuristic if needed.
    return (0.0, 0.0, 1.0, 1.0)


