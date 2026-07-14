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
# Keep this conservative: the active frame, executor working set, Qt/OpenGL
# presentation buffers and ONNX arenas live outside this cache.
CACHE_LIMIT_MB = 2048
# Relative cap as a fraction of available memory; effective quota is
# min(relative, absolute).
CACHE_MEMORY_FRACTION = 0.35
# Byte budget (MB) for the numpy-backed preview prefix/final cache. ONNX and
# OpenGL allocations have separate VRAM lifetimes and are not counted here.
EXECUTOR_CACHE_LIMIT_MB = 768
# Byte budget (MB) for free recyclable host ndarrays retained by the legacy-
# named GPU pool. Released buffers above this budget are freed, not pooled.
GPU_POOL_LIMIT_MB = 256
# At most this many free buffers are retained per (dtype, shape) key.
# Sharpen holds 3 same-shape 2D scratch buffers; the global byte budget above
# is the primary bound.
GPU_POOL_MAX_PER_KEY = 4

# Per-image cap for cached final uint8 preview frames.  ROI/pan keys can
# otherwise retain ten tens-of-megabytes frames before the outer cache gets a
# chance to evict the whole output category.
OUTPUT_CACHE_LIMIT_MB = 256

# Lensfun coordinate maps cost roughly 24 bytes/pixel (per-channel x/y) and
# previously lived in an unbounded module-global dictionary. Full-resolution
# maps larger than this cap are used once and discarded; proxy maps are LRU.
DISTORTION_MAP_CACHE_LIMIT_MB = 256

# Lens databases are much smaller than coordinate maps, but custom database
# paths used during a long session must still not grow a process-global cache
# without bound.  The default database plus a few recent custom databases is
# enough for normal interactive and batch workflows.
LENSFUN_DB_CACHE_ENTRIES = 4

# Presentation tiers use a bounded quality base map and native-resolution
# detail ROIs. The base texture is deliberately not a full
# 45/61MP frame; this keeps host copies, PBO staging and mipmapped texture VRAM
# bounded while preserving 1:1 detail through the ROI path.
QUALITY_BASE_MAX_SIDE = 4096
QUALITY_BASE_MAX_PIXELS = 16_000_000

# One large PBO upload is released after presentation instead of pinning its
# high-water allocation for the rest of the session.
PBO_RETAIN_LIMIT_MB = 64

# Thumbnail RAW/JPEG extraction is I/O heavy and each fallback decoder can
# carry a sizeable native buffer.  More workers hurt latency and memory on
# high-core-count machines.
THUMBNAIL_MAX_WORKERS = 4

# ONNX Runtime CUDA arena cap per session. DirectML ignores this option, but
# still benefits from tiled/strip execution and prompt session cleanup.
ONNX_GPU_MEMORY_LIMIT_MB = 2048

# CLI batch workers are separate processes, so every worker owns independent
# decode arrays and ONNX arenas. Multiple GPU processes multiply VRAM without
# improving a single-device queue; CPU-only jobs are additionally capped by
# available system memory.
GPU_BATCH_MAX_JOBS = 1
BATCH_MEMORY_PER_JOB_MB = 3072

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
