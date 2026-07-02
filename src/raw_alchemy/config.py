"""[config.py](src/raw_alchemy/config.py)
Raw Alchemy 閰嶇疆鏂囦欢
鍖呭惈 Log 绌洪棿鏄犲皠銆佺紪鐮佹槧灏勩€佹祴鍏夋ā寮忓畾涔夊拰 GUI 閰嶇疆
"""

# ==========================================
#           鏂囦欢绫诲瀷鏀寔
# ==========================================

# 缂╃暐鍥?鎵弿绛夊姛鑳芥敮鎸佺殑 RAW 鎵╁睍鍚嶏紙灏忓啓锛屽寘鍚偣鍙凤級
SUPPORTED_RAW_EXTENSIONS = {
'.dng', '.cr2', '.cr3', '.nef', '.arw', '.rw2', '.raf', '.orf', '.pef', '.srw', '.x3f', '.fff', '.3fr'
}

WORKING_SPACE_PROPHOTO = 'ProPhoto RGB'
WORKING_SPACE_ACESCG = 'ACEScg'
WORKING_SPACE = WORKING_SPACE_PROPHOTO
HDR_OUTPUT_COLOURSPACE = 'ITU-R BT.2020'
HDR_PQ_TRANSFER_FUNCTION = 'ST 2084'
HDR_PEAK_NITS = 1000.0
HDR_PQ_MASTERING_NITS = 10000.0

# ==========================================
#           鏍稿績澶勭悊閰嶇疆
# ==========================================

# 鏄犲皠锛歀og 绌洪棿鍚嶇О -> 瀵瑰簲鐨勭嚎鎬ц壊鍩?(Linear Gamut)
LOG_TO_WORKING_SPACE = {
    'F-Log': 'F-Gamut',
    'F-Log2': 'F-Gamut',
    'F-Log2C': 'F-Gamut C',
    'V-Log': 'V-Gamut',
    'N-Log': 'N-Gamut',
    'L-Log': 'ITU-R BT.2020',
    'Canon Log 2': 'Cinema Gamut',
    'Canon Log 3': 'Cinema Gamut',
    'S-Log3': 'S-Gamut3',
    'S-Log3.Cine': 'S-Gamut3.Cine',
    'Arri LogC3': 'ARRI Wide Gamut 3',
    'Arri LogC4': 'ARRI Wide Gamut 4',
    'Log3G10': 'REDWideGamutRGB',
    'D-Log': 'DJI D-Gamut',
}

# 鏄犲皠锛氬鍚堝悕绉?-> colour 搴撹瘑鍒殑 Log 缂栫爜鍑芥暟鍚嶇О
LOG_ENCODING_MAP = {
    'S-Log3.Cine': 'S-Log3',
    'F-Log2C': 'F-Log2',
}

# 娴嬪厜妯″紡閫夐」
METERING_MODES = [
    'average',        # 鍑犱綍骞冲潎 (榛樿)
    'center-weighted',# 涓ぎ閲嶇偣
    'highlight-safe', # 楂樺厜淇濇姢 (ETTR)
    'hybrid',         # 娣峰悎 (骞冲潎 + 楂樺厜闄愬埗)
    'matrix',         # 鐭╅樀/璇勪环娴嬪厜
]

# ==========================================
#           Host memory governance (T7.6)
# ==========================================

# Absolute cap (MB) for the decoded-image cache. Adjustable in Settings.
CACHE_LIMIT_MB = 6144
# Relative cap as a fraction of available memory; effective quota is
# min(relative, absolute).
CACHE_MEMORY_FRACTION = 0.5
# Byte budget (MB) for the preview pipeline prefix cache. Since T7.2 the
# prefix/final caches hold GPU-resident buffers, so this is a VRAM budget.
EXECUTOR_CACHE_LIMIT_MB = 4096
# Byte budget (MB) for *free* (recyclable) GPU buffers retained by the
# shape-keyed ndarray pool (T7.2). Released buffers above this budget are
# actually freed instead of pooled.
GPU_POOL_LIMIT_MB = 2048
# At most this many free buffers are retained per (dtype, shape) key.
# Sharpen holds 4 same-shape 2D scratch buffers and RCD demosaic up to 7;
# the global byte budget above is the primary bound.
GPU_POOL_MAX_PER_KEY = 8

# ==========================================
#           GUI 閰嶇疆
# ==========================================

# GUI 绐楀彛閰嶇疆
GUI_WINDOW_WIDTH = 1000
GUI_WINDOW_HEIGHT = 950
GUI_WINDOW_TITLE = "Raw Alchemy"

# GUI 鏇存柊闂撮殧锛堟绉掞級
GUI_QUEUE_UPDATE_INTERVAL = 50
GUI_INITIAL_UPDATE_INTERVAL = 100

# 榛樿鍊?
DEFAULT_CPU_THREADS = 4
DEFAULT_OUTPUT_FORMAT = 'tif'
DEFAULT_METERING_MODE = 'matrix'
DEFAULT_EXPOSURE_STOPS = 0.0
DEFAULT_LENS_CORRECTION = True

# 鏇濆厜璋冩暣鑼冨洿
EXPOSURE_MIN = -5.0
EXPOSURE_MAX = 5.0

# 鏃ュ織瀛椾綋
LOG_FONT_FAMILY = "Consolas"
LOG_FONT_SIZE = 9

# 杩涘害鏉￠厤缃?
PROGRESS_BAR_LENGTH = 400
PROGRESS_LABEL_WIDTH = 16
