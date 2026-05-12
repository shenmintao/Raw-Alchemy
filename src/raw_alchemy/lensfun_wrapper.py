"""
Lensfun库的Python包装器
用于镜头畸变、色差和暗角校正
"""
import sys
import os
import platform
import ctypes
from typing import Optional
import numpy as np
from loguru import logger

def _get_base_path():
    """
    获取资源的基础路径。
    兼容：开发环境、PyInstaller (单文件/文件夹)、Nuitka (单文件/文件夹)。
    """
    # 1. 优先处理 PyInstaller One-file 模式
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS

    # 2. 处理 PyInstaller One-dir 模式 (检查是否存在 _internal 文件夹)
    # PyInstaller 6+ 默认将依赖放在 _internal 中
    executable_dir = os.path.dirname(sys.executable)
    pyinstaller_internal_path = os.path.join(executable_dir, '_internal')
    
    # 只有当处于 frozen 状态 且 _internal 文件夹确实存在时，才使用它
    if getattr(sys, 'frozen', False) and os.path.isdir(pyinstaller_internal_path):
        return pyinstaller_internal_path

    # 3. Nuitka (单文件/文件夹) 以及 普通 Python 脚本
    # Nuitka 会修正 __file__ 指向正确的运行时位置（无论是临时目录还是 dist 目录）
    return os.path.dirname(os.path.abspath(__file__))

# _load_lensfun_library 不需要大幅修改，但要注意 bin 目录处理
def _load_lensfun_library():
    """加载lensfun动态库"""
    system = platform.system()
    base_path = _get_base_path()
    
    # 确保路径拼接正确
    lensfun_dir = os.path.join(base_path, "vendor", "lensfun")
    lib_dir = os.path.join(lensfun_dir, "lib")
    bin_dir = os.path.join(lensfun_dir, "bin")

    lib_path = None
    if system == "Windows":
        lib_path = os.path.join(lib_dir, "lensfun.dll")
        # Windows DLL 加载关键：添加搜索路径
        if os.path.isdir(bin_dir) and hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass # 防止路径添加失败影响主流程
        # 很多时候 DLL 会依赖同目录下的其他 DLL，把 lib_dir 也加进去更保险
        if os.path.isdir(lib_dir) and hasattr(os, 'add_dll_directory'):
             try:
                os.add_dll_directory(lib_dir)
             except Exception:
                pass
                
    elif system == "Darwin":
        lib_path = os.path.join(lib_dir, "liblensfun.dylib")
    else:
        lib_path = os.path.join(lib_dir, "liblensfun.so")

    try:
        # 显式路径加载
        if lib_path and os.path.exists(lib_path):
            # Windows 下建议使用 LoadLibrary 或添加 path 后加载
            return ctypes.CDLL(lib_path)
        else:
            # 兜底：尝试系统路径
            name = "lensfun.dll" if system == "Windows" else ("liblensfun.dylib" if system == "Darwin" else "liblensfun.so")
            return ctypes.CDLL(name)
    except OSError as e:
        # 错误处理保持不变...
        raise RuntimeError(f"Could not load lensfun from {lib_path}. Error: {e}")


# 加载库
try:
    _lensfun = _load_lensfun_library()
except RuntimeError as e:
    _lensfun = None
    # 记录更详细的错误信息
    logger.warning(f"  ⚠️ [Lensfun] Warning: {e}")
    logger.warning("  ⚠️ [Lensfun] Lens correction will be disabled.")


# ============================================================================
# Lensfun 常量定义
# ============================================================================

# 像素格式
LF_PF_U8 = 0
LF_PF_U16 = 1
LF_PF_U32 = 2
LF_PF_F32 = 3
LF_PF_F64 = 4

# 校正标志
LF_MODIFY_TCA = 0x00000001          # 横向色差
LF_MODIFY_VIGNETTING = 0x00000002   # 暗角
LF_MODIFY_DISTORTION = 0x00000008   # 畸变
LF_MODIFY_GEOMETRY = 0x00000010     # 几何投影
LF_MODIFY_SCALE = 0x00000020        # 缩放
LF_MODIFY_ALL = ~0

# 镜头类型
LF_UNKNOWN = 0
LF_RECTILINEAR = 1
LF_FISHEYE = 2
LF_PANORAMIC = 3
LF_EQUIRECTANGULAR = 4
LF_FISHEYE_ORTHOGRAPHIC = 5
LF_FISHEYE_STEREOGRAPHIC = 6
LF_FISHEYE_EQUISOLID = 7
LF_FISHEYE_THOBY = 8

# 颜色组件角色
LF_CR_END = 0
LF_CR_NEXT = 1
LF_CR_UNKNOWN = 2
LF_CR_INTENSITY = 3
LF_CR_RED = 4
LF_CR_GREEN = 5
LF_CR_BLUE = 6

# 颜色组件宏
def LF_CR_3(a, b, c):
    """定义3个组件的像素格式 (RGB)"""
    return a | (b << 4) | (c << 8)

LF_CR_RGB = LF_CR_3(LF_CR_RED, LF_CR_GREEN, LF_CR_BLUE)


# ============================================================================
# C结构体定义
# ============================================================================

class lfDatabase(ctypes.Structure):
    """数据库对象 (不透明)"""
    pass

class lfCamera(ctypes.Structure):
    """相机对象
    
    根据lensfun.h定义的lfCamera结构体:
    - Maker: lfMLstr (char*)
    - Model: lfMLstr (char*)
    - Variant: lfMLstr (char*)
    - Mount: char*
    - CropFactor: float
    - Score: int
    """
    _fields_ = [
        ("Maker", ctypes.c_char_p),
        ("Model", ctypes.c_char_p),
        ("Variant", ctypes.c_char_p),
        ("Mount", ctypes.c_char_p),
        ("CropFactor", ctypes.c_float),
        ("Score", ctypes.c_int),
    ]

class lfLens(ctypes.Structure):
    """镜头对象
    
    根据lensfun.h定义的lfLens结构体:
    - Maker: lfMLstr (char*)
    - Model: lfMLstr (char*)
    - MinFocal: float
    - MaxFocal: float
    - MinAperture: float
    - MaxAperture: float
    - Mounts: char**
    - Type: lfLensType
    - CropFactor: float (已弃用)
    - AspectRatio: float
    - CenterX: float
    - CenterY: float
    - Score: int
    
    注意：这里只定义我们需要访问的前几个字段
    """
    _fields_ = [
        ("Maker", ctypes.c_char_p),
        ("Model", ctypes.c_char_p),
        ("MinFocal", ctypes.c_float),
        ("MaxFocal", ctypes.c_float),
        ("MinAperture", ctypes.c_float),
        ("MaxAperture", ctypes.c_float),
        # 其他字段暂不定义，因为我们主要需要 Maker 和 Model
    ]

class lfModifier(ctypes.Structure):
    """校正修改器对象 (不透明)"""
    pass


# ============================================================================
# 函数签名定义
# ============================================================================

if _lensfun:
    # 数据库函数
    _lensfun.lf_db_create.restype = ctypes.POINTER(lfDatabase)
    _lensfun.lf_db_create.argtypes = []
    
    _lensfun.lf_db_destroy.restype = None
    _lensfun.lf_db_destroy.argtypes = [ctypes.POINTER(lfDatabase)]
    
    _lensfun.lf_db_load.restype = ctypes.c_int
    _lensfun.lf_db_load.argtypes = [ctypes.POINTER(lfDatabase)]
    
    _lensfun.lf_db_load_path.restype = ctypes.c_int
    _lensfun.lf_db_load_path.argtypes = [ctypes.POINTER(lfDatabase), ctypes.c_char_p]

    _lensfun.lf_db_load_str.restype = ctypes.c_int
    _lensfun.lf_db_load_str.argtypes = [ctypes.POINTER(lfDatabase), ctypes.c_char_p, ctypes.c_size_t]
    
    _lensfun.lf_db_find_cameras_ext.restype = ctypes.POINTER(ctypes.POINTER(lfCamera))
    _lensfun.lf_db_find_cameras_ext.argtypes = [
        ctypes.POINTER(lfDatabase),
        ctypes.c_char_p,  # maker
        ctypes.c_char_p,  # model
        ctypes.c_int      # sflags
    ]
    
    _lensfun.lf_db_find_lenses.restype = ctypes.POINTER(ctypes.POINTER(lfLens))
    _lensfun.lf_db_find_lenses.argtypes = [
        ctypes.POINTER(lfDatabase),
        ctypes.POINTER(lfCamera),
        ctypes.c_char_p,  # maker
        ctypes.c_char_p,  # model
        ctypes.c_int      # sflags
    ]
    
    # 修改器函数
    _lensfun.lf_modifier_create.restype = ctypes.POINTER(lfModifier)
    _lensfun.lf_modifier_create.argtypes = [
        ctypes.POINTER(lfLens),
        ctypes.c_float,   # focal
        ctypes.c_float,   # crop
        ctypes.c_int,     # width
        ctypes.c_int,     # height
        ctypes.c_int,     # pixel_format
        ctypes.c_int      # reverse
    ]
    
    _lensfun.lf_modifier_destroy.restype = None
    _lensfun.lf_modifier_destroy.argtypes = [ctypes.POINTER(lfModifier)]
    
    _lensfun.lf_modifier_enable_distortion_correction.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_distortion_correction.argtypes = [ctypes.POINTER(lfModifier)]
    
    _lensfun.lf_modifier_enable_tca_correction.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_tca_correction.argtypes = [ctypes.POINTER(lfModifier)]
    
    _lensfun.lf_modifier_enable_vignetting_correction.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_vignetting_correction.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_float,  # aperture
        ctypes.c_float   # distance
    ]
    
    _lensfun.lf_modifier_enable_projection_transform.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_projection_transform.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_int  # target_projection
    ]
    
    _lensfun.lf_modifier_enable_scaling.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_scaling.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_float  # scale
    ]
    
    _lensfun.lf_modifier_apply_subpixel_geometry_distortion.restype = ctypes.c_int
    _lensfun.lf_modifier_apply_subpixel_geometry_distortion.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_float,                    # xu
        ctypes.c_float,                    # yu
        ctypes.c_int,                      # width
        ctypes.c_int,                      # height
        ctypes.POINTER(ctypes.c_float)     # res
    ]
    
    _lensfun.lf_modifier_apply_color_modification.restype = ctypes.c_int
    _lensfun.lf_modifier_apply_color_modification.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_void_p,  # pixels
        ctypes.c_float,   # x
        ctypes.c_float,   # y
        ctypes.c_int,     # width
        ctypes.c_int,     # height
        ctypes.c_int,     # comp_role
        ctypes.c_int      # row_stride
    ]
    
    _lensfun.lf_free.restype = None
    _lensfun.lf_free.argtypes = [ctypes.c_void_p]

    _lensfun.lf_modifier_get_auto_scale.restype = ctypes.c_float
    _lensfun.lf_modifier_get_auto_scale.argtypes = [ctypes.POINTER(lfModifier)]


# ============================================================================
# Python包装类
# ============================================================================

class LensfunDatabase:
    """Lensfun数据库包装器"""
    
    def __init__(self, custom_db_path: Optional[str] = None):
        if not _lensfun:
            raise RuntimeError("Lensfun library not loaded")
        self.db = _lensfun.lf_db_create()
        if not self.db:
            raise RuntimeError("Could not create lensfun database")
        
        # 检查本地数据库路径
        base_path = _get_base_path()
        db_path = os.path.join(base_path, "vendor", "lensfun", "share", "lensfun", "version_2")
        
        result = -1
        if os.path.isdir(db_path):
            logger.info(f"  ✨ [Lensfun] Found local database, loading from: {db_path}")
            result = _lensfun.lf_db_load_path(self.db, db_path.encode('utf-8'))
        else:
            logger.info(f"  ℹ️ [Lensfun] Local database not found, loading from system default paths.")
            result = _lensfun.lf_db_load(self.db)

        # Check loading result
        if result != 0:
            error_msg = f"Failed to load lensfun database, error code: {result}"
            if result == 2:  # LF_IO_ERROR
                error_msg += "\n  💡 [Hint] Database file not found or could not be read."
                error_msg += f"\n     - Check if the path is correct: {db_path if os.path.isdir(db_path) else 'System paths'}"
                error_msg += "\n     - Ensure file permissions are correct."
            raise RuntimeError(error_msg)
        
        # 加载用户自定义数据库
        if custom_db_path and os.path.exists(custom_db_path):
            logger.info(f"  ✨ [Lensfun] Loading custom database from: {custom_db_path}")
            try:
                with open(custom_db_path, 'rb') as f:
                    xml_data = f.read()
                
                if xml_data:
                    # lf_db_load_str用于从字符串加载XML数据
                    result = _lensfun.lf_db_load_str(self.db, xml_data, len(xml_data))
                    if result != 0:
                        error_msg = f"Failed to load custom lensfun database from file: {custom_db_path}, error code: {result}"
                        if result == 1:  # LF_WRONG_FORMAT
                            error_msg += "\n  💡 [Hint] The XML data has the wrong format. Please check if the file is a valid Lensfun database file."
                        elif result == 2:  # LF_NO_DATABASE
                            error_msg += "\n  💡 [Hint] No database could be loaded from the provided data. The file might be empty or corrupted."
                        raise RuntimeError(error_msg)
            except IOError as e:
                raise RuntimeError(f"Could not read custom database file: {custom_db_path}. Error: {e}")
    
    def __del__(self):
        if hasattr(self, 'db') and self.db and _lensfun is not None:
            _lensfun.lf_db_destroy(self.db)
    
    def find_camera(self, maker: Optional[str], model: str) -> Optional[ctypes.POINTER(lfCamera)]:
        """查找相机"""
        maker_b = maker.encode('utf-8') if maker else None
        model_b = model.encode('utf-8')
        
        cameras = _lensfun.lf_db_find_cameras_ext(self.db, maker_b, model_b, 1)
        if cameras and cameras[0]:
            return cameras[0]
        return None
    
    def find_lens(self, camera: Optional[ctypes.POINTER(lfCamera)], 
                  maker: Optional[str], model: str) -> Optional[ctypes.POINTER(lfLens)]:
        """查找镜头"""
        maker_b = maker.encode('utf-8') if maker else None
        model_b = model.encode('utf-8')
        
        lenses = _lensfun.lf_db_find_lenses(self.db, camera, maker_b, model_b, 1)
        if lenses and lenses[0]:
            return lenses[0]
        return None


class LensfunModifier:
    """Lensfun校正修改器包装器"""
    
    def __init__(self, lens: ctypes.POINTER(lfLens), focal: float, crop: float,
                 width: int, height: int, pixel_format: int = LF_PF_F32, reverse: bool = False):
        if not _lensfun:
            raise RuntimeError("Lensfun library not loaded")
        
        self.modifier = _lensfun.lf_modifier_create(
            lens, focal, crop, width, height, pixel_format, int(reverse)
        )
        if not self.modifier:
            raise RuntimeError("Could not create lensfun modifier")
        
        self.width = width
        self.height = height
    
    def __del__(self):
        if hasattr(self, 'modifier') and self.modifier and _lensfun is not None:
            _lensfun.lf_modifier_destroy(self.modifier)
    
    def enable_distortion_correction(self) -> int:
        """启用畸变校正"""
        return _lensfun.lf_modifier_enable_distortion_correction(self.modifier)
    
    def enable_tca_correction(self) -> int:
        """启用横向色差校正"""
        return _lensfun.lf_modifier_enable_tca_correction(self.modifier)
    
    def enable_vignetting_correction(self, aperture: float, distance: float = 1000.0) -> int:
        """启用暗角校正"""
        return _lensfun.lf_modifier_enable_vignetting_correction(
            self.modifier, aperture, distance
        )
    
    def enable_projection_transform(self, target_projection: int) -> int:
        """启用投影变换"""
        return _lensfun.lf_modifier_enable_projection_transform(
            self.modifier, target_projection
        )
    
    def enable_scaling(self, scale: float) -> int:
        """启用缩放"""
        return _lensfun.lf_modifier_enable_scaling(self.modifier, scale)

    def get_auto_scale(self) -> float:
        """获取自动缩放比例"""
        return _lensfun.lf_modifier_get_auto_scale(self.modifier)
    
    def apply_subpixel_geometry_distortion(self, xu: float, yu: float, 
                                           width: int, height: int) -> Optional[np.ndarray]:
        """应用子像素几何畸变校正
        
        返回: shape为 (height, width, 2, 3) 的数组，存储R/G/B三通道的(x,y)坐标
        """
        # 分配输出缓冲区: width * height * 2 * 3
        res_size = width * height * 2 * 3
        res = (ctypes.c_float * res_size)()
        
        result = _lensfun.lf_modifier_apply_subpixel_geometry_distortion(
            self.modifier, xu, yu, width, height, res
        )
        
        if result:
            # 转换为numpy数组并重塑
            arr = np.ctypeslib.as_array(res)
            return arr.reshape(height, width, 3, 2)  # (h, w, RGB, xy)
        return None
    
    def apply_color_modification(self, pixels: np.ndarray, x: float, y: float,
                                 width: int, height: int) -> bool:
        """应用颜色修改（暗角校正）
        
        参数:
            pixels: 像素数据，会被原地修改
        """
        # 确保数据类型正确
        if pixels.dtype != np.float32:
            raise ValueError("Pixel data must be of type float32")
        
        # 获取数据指针
        pixels_ptr = pixels.ctypes.data_as(ctypes.c_void_p)
        row_stride = width * pixels.shape[2] * pixels.itemsize
        
        result = _lensfun.lf_modifier_apply_color_modification(
            self.modifier, pixels_ptr, x, y, width, height, LF_CR_RGB, row_stride
        )
        
        return bool(result)


# ============================================================================
# 全局数据库缓存
# ============================================================================

# 全局数据库缓存，避免每次都重新加载
_global_db_cache = {}
_global_db_lock = None

def _get_or_create_database(custom_db_path: Optional[str] = None):
    """
    获取或创建Lensfun数据库（带缓存）
    
    参数:
        custom_db_path: 自定义数据库路径，None表示使用默认数据库
    
    返回:
        LensfunDatabase对象
    """
    global _global_db_cache, _global_db_lock
    
    # 初始化锁（线程安全）
    if _global_db_lock is None:
        import threading
        _global_db_lock = threading.Lock()
    
    # 使用custom_db_path作为缓存键
    cache_key = custom_db_path if custom_db_path else '__default__'
    
    with _global_db_lock:
        # 检查缓存
        if cache_key in _global_db_cache:
            return _global_db_cache[cache_key]
        
        # 创建新数据库并缓存
        try:
            db = LensfunDatabase(custom_db_path=custom_db_path)
            _global_db_cache[cache_key] = db
            return db
        except Exception as e:
            logger.error(f"  ❌ [Lensfun] Failed to create database: {e}")
            raise

def reload_lensfun_database(custom_db_path: Optional[str] = None):
    """
    强制重新加载Lensfun数据库（用于更新custom db时）
    
    参数:
        custom_db_path: 自定义数据库路径，None表示重新加载默认数据库
    """
    global _global_db_cache, _global_db_lock
    
    if _global_db_lock is None:
        import threading
        _global_db_lock = threading.Lock()
    
    cache_key = custom_db_path if custom_db_path else '__default__'
    
    with _global_db_lock:
        # 删除旧缓存
        if cache_key in _global_db_cache:
            old_db = _global_db_cache[cache_key]
            del _global_db_cache[cache_key]
            # 显式销毁旧数据库
            if hasattr(old_db, 'db') and old_db.db:
                _lensfun.lf_db_destroy(old_db.db)
                old_db.db = None
        
        # 创建新数据库
        try:
            db = LensfunDatabase(custom_db_path=custom_db_path)
            _global_db_cache[cache_key] = db
            logger.success(f"  ✅ [Lensfun] Database reloaded successfully")
            return db
        except Exception as e:
            logger.error(f"  ❌ [Lensfun] Failed to reload database: {e}")
            raise

# ============================================================================
# 便捷函数
# ============================================================================

def apply_lens_correction(
    image: np.ndarray,
    camera_maker: Optional[str],
    camera_model: str,
    lens_maker: Optional[str],
    lens_model: str,
    focal_length: float,
    aperture: float,
    crop_factor: Optional[float] = None,
    correct_distortion: bool = True,
    correct_tca: bool = True,
    correct_vignetting: bool = True,
    distance: float = 1000.0,
    custom_db_path: Optional[str] = None,
) -> np.ndarray:
    """应用镜头校正到图像
    
    参数:
        image: 输入图像，shape为 (height, width, 3)，范围0-1
        camera_maker: 相机制造商
        camera_model: 相机型号
        lens_maker: 镜头制造商
        lens_model: 镜头型号
        focal_length: 焦距 (mm)
        aperture: 光圈值 (f-number)
        crop_factor: 裁剪系数，如果为None则从相机信息获取
        correct_distortion: 是否校正畸变
        correct_tca: 是否校正横向色差
        correct_vignetting: 是否校正暗角
        distance: 对焦距离 (米)
        custom_db_path: 自定义数据库路径
    
    返回:
        校正后的图像（与输入相同dtype）
    """
    if not _lensfun:
        logger.warning("  ⚠️ [Lensfun] Library not loaded. Skipping lens correction.")
        return image
    
    # 记住原始dtype以便最后转换回去
    original_dtype = image.dtype
    
    # 转换为float32（如果不是的话）
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    height, width = image.shape[:2]
    
    # 使用缓存的数据库（避免每次都重新加载）
    db = _get_or_create_database(custom_db_path=custom_db_path)
    camera = db.find_camera(camera_maker, camera_model)
    lens = db.find_lens(camera, lens_maker, lens_model)
    
    if not lens:
        logger.warning(f"  ⚠️ [Lensfun] Lens not found: {lens_maker} {lens_model}. Skipping correction.")
        return image
    
    # 确定裁剪系数
    if crop_factor is None:
        # 优先从相机获取crop factor
        if camera:
            try:
                crop_factor = camera.contents.CropFactor
                if crop_factor > 0:  # 确保是有效值
                    logger.info(f"  📷 [Lensfun] Using camera crop factor: {crop_factor:.2f}")
                else:
                    raise ValueError("Invalid crop factor from camera")
            except (AttributeError, ValueError) as e:
                logger.warning(f"  ⚠️ [Lensfun] Could not read camera crop factor: {e}")
                crop_factor = None
        
        # 如果相机没有提供有效的crop factor，使用默认值
        # 注意：lfLens.CropFactor已弃用且结构体复杂，不建议直接访问
        if crop_factor is None:
            crop_factor = 1.0
            logger.info(f"  ℹ️ [Lensfun] Using default crop factor: {crop_factor}")
    
    # 创建修改器
    modifier = LensfunModifier(lens, focal_length, crop_factor, width, height, LF_PF_F32)
    
    # 启用所需的校正
    if correct_distortion:
        modifier.enable_distortion_correction()

    if correct_tca:
        modifier.enable_tca_correction()

    if correct_vignetting:
        modifier.enable_vignetting_correction(aperture, distance)

    # 创建输出图像
    output = np.zeros_like(image)

    # 步骤1: 应用颜色修改（暗角）
    # 这是原位操作，会直接修改 image 数组。
    # 后续的几何校正会从这个修改后的 image 中读取数据，所以这是期望的行为。
    if correct_vignetting:
        modifier.apply_color_modification(image, 0.0, 0.0, width, height)

    # 步骤2: 应用几何畸变和TCA校正
    if correct_distortion or correct_tca:
        coords = modifier.apply_subpixel_geometry_distortion(0.0, 0.0, width, height)

        if coords is not None:
            # 检查坐标是否超出图像范围，如果是则向中心缩放以消除黑边
            x_min = min(coords[:, :, ch, 0].min() for ch in range(3))
            x_max = max(coords[:, :, ch, 0].max() for ch in range(3))
            y_min = min(coords[:, :, ch, 1].min() for ch in range(3))
            y_max = max(coords[:, :, ch, 1].max() for ch in range(3))

            if x_min < 0 or y_min < 0 or x_max >= width or y_max >= height:
                # 计算将坐标范围压缩到 [0, width-1] x [0, height-1] 所需的缩放
                x_range = x_max - x_min
                y_range = y_max - y_min
                scale_x = (width - 1) / x_range if x_range > 0 else 1.0
                scale_y = (height - 1) / y_range if y_range > 0 else 1.0
                scale = min(scale_x, scale_y)

                cx, cy = width / 2.0, height / 2.0
                for ch in range(3):
                    coords[:, :, ch, 0] = cx + (coords[:, :, ch, 0] - cx) * scale
                    coords[:, :, ch, 1] = cy + (coords[:, :, ch, 1] - cy) * scale

                logger.info(f"  ⚖️ [Lensfun] Auto-crop scale: {scale:.4f} (src range: x=[{x_min:.1f},{x_max:.1f}] y=[{y_min:.1f},{y_max:.1f}])")

            # 构建越界掩码：任一通道越界的像素全部置黑，避免 R/G/B 独立 clamp 产生彩色伪影
            # order=3 三次插值需要 4x4 邻域，边界 2 像素内会采样到零填充值，
            # 不同通道坐标不同（TCA）导致混入量不一致产生彩色噪点，因此加 margin
            interp_margin = 2.0
            oob_mask = np.zeros((height, width), dtype=bool)
            for ch in range(3):
                oob_mask |= (coords[:, :, ch, 0] < interp_margin) | (coords[:, :, ch, 0] > width - 1 - interp_margin)
                oob_mask |= (coords[:, :, ch, 1] < interp_margin) | (coords[:, :, ch, 1] > height - 1 - interp_margin)

            # clamp 坐标以便 remap 不会读到非法地址
            for ch in range(3):
                np.clip(coords[:, :, ch, 0], 0, width - 1, out=coords[:, :, ch, 0])
                np.clip(coords[:, :, ch, 1], 0, height - 1, out=coords[:, :, ch, 1])

            import cv2

            for c in range(3):  # R, G, B
                # cv2.remap takes (map_x, map_y) as float32 ndarrays of
                # shape (out_H, out_W). coords[:,:,c,0] = x, [:,:,c,1] = y.
                map_x = np.ascontiguousarray(coords[:, :, c, 0], dtype=np.float32)
                map_y = np.ascontiguousarray(coords[:, :, c, 1], dtype=np.float32)
                output[:, :, c] = cv2.remap(
                    image[:, :, c],
                    map_x, map_y,
                    interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )

            # 将越界像素的所有通道统一置零（黑色）
            output[oob_mask] = 0.0
        else:
            output = image
    else:
        output = image
    
    # 转换回原始dtype
    if output.dtype != original_dtype:
        output = output.astype(original_dtype)
    
    return output


def compute_lens_distortion_map(
    image: np.ndarray,
    camera_maker: Optional[str],
    camera_model: str,
    lens_maker: Optional[str],
    lens_model: str,
    focal_length: float,
    aperture: float,
    crop_factor: Optional[float] = None,
    correct_distortion: bool = True,
    correct_tca: bool = True,
    correct_vignetting: bool = True,
    distance: float = 1000.0,
    custom_db_path: Optional[str] = None,
) -> Optional[tuple]:
    """Compute lens distortion map (CPU) for GPU remap.

    Returns (coords, oob_mask, vignette_corrected_image) or None if no correction needed.
    - coords: (H, W, 3, 2) float32 — per-channel (x, y) from lensfun
    - oob_mask: (H, W) bool — out-of-bounds mask
    - image: possibly vignette-corrected (in-place)
    """
    height, width = image.shape[:2]

    if not _lensfun:
        return None

    db = _get_or_create_database(custom_db_path=custom_db_path)
    camera = db.find_camera(camera_maker, camera_model)
    lens = db.find_lens(camera, lens_maker, lens_model)
    if not lens:
        return None

    if crop_factor is None:
        if camera:
            try:
                crop_factor = camera.contents.CropFactor
                if crop_factor <= 0:
                    crop_factor = 1.0
            except (AttributeError, ValueError):
                crop_factor = 1.0
        else:
            crop_factor = 1.0
    logger.info(f"  📷 [Lensfun] Using camera crop factor: {crop_factor:.2f}")

    modifier = LensfunModifier(lens, focal_length, crop_factor, width, height, LF_PF_F32)

    # Vignetting: apply in-place on CPU (per-pixel color modification, fast)
    if correct_vignetting:
        modifier.enable_vignetting_correction(aperture, distance)
        image_f32 = image if image.dtype == np.float32 else image.astype(np.float32)
        modifier.apply_color_modification(image_f32, 0.0, 0.0, width, height)
        if image.dtype != np.float32:
            np.copyto(image, image_f32.astype(image.dtype))

    # Geometry: compute distortion map (CPU, fast ~few ms)
    if not correct_distortion and not correct_tca:
        return None

    if correct_distortion:
        modifier.enable_distortion_correction()
    if correct_tca:
        modifier.enable_tca_correction()

    coords = modifier.apply_subpixel_geometry_distortion(0.0, 0.0, width, height)
    if coords is None:
        return None

    # Auto-crop scale (same logic as apply_lens_correction)
    x_min = min(coords[:, :, ch, 0].min() for ch in range(3))
    x_max = max(coords[:, :, ch, 0].max() for ch in range(3))
    y_min = min(coords[:, :, ch, 1].min() for ch in range(3))
    y_max = max(coords[:, :, ch, 1].max() for ch in range(3))

    if x_min < 0 or y_min < 0 or x_max >= width or y_max >= height:
        x_range = x_max - x_min
        y_range = y_max - y_min
        scale_x = (width - 1) / x_range if x_range > 0 else 1.0
        scale_y = (height - 1) / y_range if y_range > 0 else 1.0
        scale = min(scale_x, scale_y)
        cx, cy = width / 2.0, height / 2.0
        for ch in range(3):
            coords[:, :, ch, 0] = cx + (coords[:, :, ch, 0] - cx) * scale
            coords[:, :, ch, 1] = cy + (coords[:, :, ch, 1] - cy) * scale
        logger.info(f"  ⚖️ [Lensfun] Auto-crop scale: {scale:.4f} (src range: x=[{x_min:.1f},{x_max:.1f}] y=[{y_min:.1f},{y_max:.1f}])")

    # Build OOB mask
    interp_margin = 2.0
    oob_mask = np.zeros((height, width), dtype=bool)
    for ch in range(3):
        oob_mask |= (coords[:, :, ch, 0] < interp_margin) | (coords[:, :, ch, 0] > width - 1 - interp_margin)
        oob_mask |= (coords[:, :, ch, 1] < interp_margin) | (coords[:, :, ch, 1] > height - 1 - interp_margin)

    # Clamp coords
    for ch in range(3):
        np.clip(coords[:, :, ch, 0], 0, width - 1, out=coords[:, :, ch, 0])
        np.clip(coords[:, :, ch, 1], 0, height - 1, out=coords[:, :, ch, 1])

    return coords, oob_mask, image


def get_lens_info(
    camera_maker: Optional[str],
    camera_model: str,
    lens_maker: Optional[str],
    lens_model: str,
    custom_db_path: Optional[str] = None,
) -> Optional[dict]:
    """获取镜头信息（厂商和名称）
    
    参数:
        camera_maker: 相机制造商
        camera_model: 相机型号
        lens_maker: 镜头制造商
        lens_model: 镜头型号
        custom_db_path: 自定义数据库路径
    
    返回:
        包含镜头信息的字典，格式为:
        {
            'maker': str,           # 镜头厂商
            'model': str,           # 镜头名称
            'min_focal': float,     # 最小焦距
            'max_focal': float,     # 最大焦距
            'min_aperture': float,  # 最小光圈
            'max_aperture': float   # 最大光圈
        }
        如果未找到镜头则返回 None
    """
    if not _lensfun:
        logger.warning("  ⚠️ [Lensfun] Library not loaded. Cannot get lens info.")
        return None
    
    try:
        # 使用缓存的数据库
        db = _get_or_create_database(custom_db_path=custom_db_path)
        camera = db.find_camera(camera_maker, camera_model)
        lens = db.find_lens(camera, lens_maker, lens_model)
        
        if not lens:
            logger.warning(f"  ⚠️ [Lensfun] Lens not found: {lens_maker} {lens_model}")
            return None
        
        # 提取镜头信息
        lens_info = {
            'maker': lens.contents.Maker.decode('utf-8') if lens.contents.Maker else None,
            'model': lens.contents.Model.decode('utf-8') if lens.contents.Model else None,
            'min_focal': lens.contents.MinFocal,
            'max_focal': lens.contents.MaxFocal,
            'min_aperture': lens.contents.MinAperture,
            'max_aperture': lens.contents.MaxAperture,
        }
        
        return lens_info
        
    except Exception as e:
        logger.error(f"  ❌ [Lensfun] Error getting lens info: {e}")
        return None

