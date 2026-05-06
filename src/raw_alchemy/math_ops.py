import threading
import taichi as ti
import numpy as np
from loguru import logger

# =========================================================
# Taichi Initialization
# =========================================================
# Taichi supports only a single global instance per process.
# We initialize once here; all kernels share this runtime.
# Priority: CUDA > Vulkan > Metal > CPU

_ti_initialized = False
_ti_init_lock = threading.Lock()

def init_taichi(arch=None):
    """Initialize Taichi runtime. Thread-safe, no-op after first call."""
    global _ti_initialized
    if _ti_initialized:
        return

    with _ti_init_lock:
        if _ti_initialized:  # double-check after acquiring lock
            return

        if arch is not None:
            ti.init(arch=arch)
        else:
            ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

        _ti_initialized = True
        logger.info(f"  Taichi initialized.")


# NOTE: Do NOT auto-initialize here. Taichi's CUDA context is thread-local.
# init_taichi() must be called on the same thread that will run kernels
# (i.e. the QThread worker), otherwise CUDA_ERROR_INVALID_CONTEXT occurs.


# =========================================================
# Taichi Kernels (GPU-accelerated, parallel)
# =========================================================

@ti.kernel
def _apply_matrix_inplace_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3),
                                  m00: ti.f32, m01: ti.f32, m02: ti.f32,
                                  m10: ti.f32, m11: ti.f32, m12: ti.f32,
                                  m20: ti.f32, m21: ti.f32, m22: ti.f32):
    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        r_val = img[r, c, 0]
        g_val = img[r, c, 1]
        b_val = img[r, c, 2]

        img[r, c, 0] = r_val * m00 + g_val * m01 + b_val * m02
        img[r, c, 1] = r_val * m10 + g_val * m11 + b_val * m12
        img[r, c, 2] = r_val * m20 + g_val * m21 + b_val * m22


def _is_numpy(arr):
    return isinstance(arr, np.ndarray)


def apply_matrix_inplace(img, matrix):
    """Apply 3x3 color matrix to image in-place. img: HxWx3 float32, matrix: 3x3"""
    if _is_numpy(img) and not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    m = np.ascontiguousarray(matrix).astype(np.float64)
    _apply_matrix_inplace_kernel(img,
                                  float(m[0, 0]), float(m[0, 1]), float(m[0, 2]),
                                  float(m[1, 0]), float(m[1, 1]), float(m[1, 2]),
                                  float(m[2, 0]), float(m[2, 1]), float(m[2, 2]))


@ti.kernel
def _apply_lut_inplace_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3),
                               lut_table: ti.types.ndarray(dtype=ti.f32, ndim=4),
                               min_r: ti.f32, min_g: ti.f32, min_b: ti.f32,
                               scale_r: ti.f32, scale_g: ti.f32, scale_b: ti.f32,
                               size_minus_1: ti.i32):
    size_float = ti.cast(size_minus_1, ti.f32)

    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        in_r = img[r, c, 0]
        in_g = img[r, c, 1]
        in_b = img[r, c, 2]

        raw_idx_r = (in_r - min_r) * scale_r
        raw_idx_g = (in_g - min_g) * scale_g
        raw_idx_b = (in_b - min_b) * scale_b

        idx_r = ti.min(ti.max(raw_idx_r, 0.0), size_float)
        idx_g = ti.min(ti.max(raw_idx_g, 0.0), size_float)
        idx_b = ti.min(ti.max(raw_idx_b, 0.0), size_float)

        x0 = ti.cast(idx_r, ti.i32)
        y0 = ti.cast(idx_g, ti.i32)
        z0 = ti.cast(idx_b, ti.i32)

        x1 = x0 + 1
        if x0 == size_minus_1:
            x1 = x0
        y1 = y0 + 1
        if y0 == size_minus_1:
            y1 = y0
        z1 = z0 + 1
        if z0 == size_minus_1:
            z1 = z0

        dx = idx_r - ti.cast(x0, ti.f32)
        dy = idx_g - ti.cast(y0, ti.f32)
        dz = idx_b - ti.cast(z0, ti.f32)

        # Tetrahedral interpolation
        c_0 = ti.cast(0.0, ti.f32)
        c_1 = ti.cast(0.0, ti.f32)
        c_2 = ti.cast(0.0, ti.f32)

        if dx >= dy:
            if dy >= dz:
                # dx >= dy >= dz
                w0 = 1.0 - dx
                c_0 = lut_table[x0, y0, z0, 0] * w0
                c_1 = lut_table[x0, y0, z0, 1] * w0
                c_2 = lut_table[x0, y0, z0, 2] * w0

                w1 = dx - dy
                c_0 += lut_table[x1, y0, z0, 0] * w1
                c_1 += lut_table[x1, y0, z0, 1] * w1
                c_2 += lut_table[x1, y0, z0, 2] * w1

                w2 = dy - dz
                c_0 += lut_table[x1, y1, z0, 0] * w2
                c_1 += lut_table[x1, y1, z0, 1] * w2
                c_2 += lut_table[x1, y1, z0, 2] * w2

                c_0 += lut_table[x1, y1, z1, 0] * dz
                c_1 += lut_table[x1, y1, z1, 1] * dz
                c_2 += lut_table[x1, y1, z1, 2] * dz

            elif dx >= dz:
                # dx >= dz > dy
                w0 = 1.0 - dx
                c_0 = lut_table[x0, y0, z0, 0] * w0
                c_1 = lut_table[x0, y0, z0, 1] * w0
                c_2 = lut_table[x0, y0, z0, 2] * w0

                w1 = dx - dz
                c_0 += lut_table[x1, y0, z0, 0] * w1
                c_1 += lut_table[x1, y0, z0, 1] * w1
                c_2 += lut_table[x1, y0, z0, 2] * w1

                w2 = dz - dy
                c_0 += lut_table[x1, y0, z1, 0] * w2
                c_1 += lut_table[x1, y0, z1, 1] * w2
                c_2 += lut_table[x1, y0, z1, 2] * w2

                c_0 += lut_table[x1, y1, z1, 0] * dy
                c_1 += lut_table[x1, y1, z1, 1] * dy
                c_2 += lut_table[x1, y1, z1, 2] * dy

            else:
                # dz > dx >= dy
                w0 = 1.0 - dz
                c_0 = lut_table[x0, y0, z0, 0] * w0
                c_1 = lut_table[x0, y0, z0, 1] * w0
                c_2 = lut_table[x0, y0, z0, 2] * w0

                w1 = dz - dx
                c_0 += lut_table[x0, y0, z1, 0] * w1
                c_1 += lut_table[x0, y0, z1, 1] * w1
                c_2 += lut_table[x0, y0, z1, 2] * w1

                w2 = dx - dy
                c_0 += lut_table[x1, y0, z1, 0] * w2
                c_1 += lut_table[x1, y0, z1, 1] * w2
                c_2 += lut_table[x1, y0, z1, 2] * w2

                c_0 += lut_table[x1, y1, z1, 0] * dy
                c_1 += lut_table[x1, y1, z1, 1] * dy
                c_2 += lut_table[x1, y1, z1, 2] * dy
        else:
            # dy > dx
            if dz >= dy:
                # dz >= dy > dx
                w0 = 1.0 - dz
                c_0 = lut_table[x0, y0, z0, 0] * w0
                c_1 = lut_table[x0, y0, z0, 1] * w0
                c_2 = lut_table[x0, y0, z0, 2] * w0

                w1 = dz - dy
                c_0 += lut_table[x0, y0, z1, 0] * w1
                c_1 += lut_table[x0, y0, z1, 1] * w1
                c_2 += lut_table[x0, y0, z1, 2] * w1

                w2 = dy - dx
                c_0 += lut_table[x0, y1, z1, 0] * w2
                c_1 += lut_table[x0, y1, z1, 1] * w2
                c_2 += lut_table[x0, y1, z1, 2] * w2

                c_0 += lut_table[x1, y1, z1, 0] * dx
                c_1 += lut_table[x1, y1, z1, 1] * dx
                c_2 += lut_table[x1, y1, z1, 2] * dx

            elif dz >= dx:
                # dy > dz >= dx
                w0 = 1.0 - dy
                c_0 = lut_table[x0, y0, z0, 0] * w0
                c_1 = lut_table[x0, y0, z0, 1] * w0
                c_2 = lut_table[x0, y0, z0, 2] * w0

                w1 = dy - dz
                c_0 += lut_table[x0, y1, z0, 0] * w1
                c_1 += lut_table[x0, y1, z0, 1] * w1
                c_2 += lut_table[x0, y1, z0, 2] * w1

                w2 = dz - dx
                c_0 += lut_table[x0, y1, z1, 0] * w2
                c_1 += lut_table[x0, y1, z1, 1] * w2
                c_2 += lut_table[x0, y1, z1, 2] * w2

                c_0 += lut_table[x1, y1, z1, 0] * dx
                c_1 += lut_table[x1, y1, z1, 1] * dx
                c_2 += lut_table[x1, y1, z1, 2] * dx

            else:
                # dy > dx > dz
                w0 = 1.0 - dy
                c_0 = lut_table[x0, y0, z0, 0] * w0
                c_1 = lut_table[x0, y0, z0, 1] * w0
                c_2 = lut_table[x0, y0, z0, 2] * w0

                w1 = dy - dx
                c_0 += lut_table[x0, y1, z0, 0] * w1
                c_1 += lut_table[x0, y1, z0, 1] * w1
                c_2 += lut_table[x0, y1, z0, 2] * w1

                w2 = dx - dz
                c_0 += lut_table[x1, y1, z0, 0] * w2
                c_1 += lut_table[x1, y1, z0, 1] * w2
                c_2 += lut_table[x1, y1, z0, 2] * w2

                c_0 += lut_table[x1, y1, z1, 0] * dz
                c_1 += lut_table[x1, y1, z1, 1] * dz
                c_2 += lut_table[x1, y1, z1, 2] * dz

        img[r, c, 0] = c_0
        img[r, c, 1] = c_1
        img[r, c, 2] = c_2


def apply_lut_inplace(img, lut_table, domain_min, domain_max):
    """Apply 3D LUT with tetrahedral interpolation in-place."""
    if _is_numpy(img):
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
    if lut_table.dtype != np.float32:
        lut_table = lut_table.astype(np.float32)
    if not lut_table.flags['C_CONTIGUOUS']:
        lut_table = np.ascontiguousarray(lut_table)

    size = lut_table.shape[0]
    size_minus_1 = size - 1

    scale_r = float(size_minus_1 / (domain_max[0] - domain_min[0]))
    scale_g = float(size_minus_1 / (domain_max[1] - domain_min[1]))
    scale_b = float(size_minus_1 / (domain_max[2] - domain_min[2]))

    _apply_lut_inplace_kernel(img, lut_table,
                               float(domain_min[0]), float(domain_min[1]), float(domain_min[2]),
                               scale_r, scale_g, scale_b,
                               size_minus_1)


@ti.kernel
def _apply_saturation_contrast_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3),
                                       saturation: ti.f32, contrast: ti.f32,
                                       pivot: ti.f32,
                                       cr: ti.f32, cg: ti.f32, cb: ti.f32):
    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        r_val = img[r, c, 0]
        g_val = img[r, c, 1]
        b_val = img[r, c, 2]

        lum = r_val * cr + g_val * cg + b_val * cb

        r_sat = lum + (r_val - lum) * saturation
        g_sat = lum + (g_val - lum) * saturation
        b_sat = lum + (b_val - lum) * saturation

        r_fin = (r_sat - pivot) * contrast + pivot
        g_fin = (g_sat - pivot) * contrast + pivot
        b_fin = (b_sat - pivot) * contrast + pivot

        if r_fin < 0.0:
            r_fin = 0.0
        if g_fin < 0.0:
            g_fin = 0.0
        if b_fin < 0.0:
            b_fin = 0.0

        img[r, c, 0] = r_fin
        img[r, c, 1] = g_fin
        img[r, c, 2] = b_fin


def apply_saturation_contrast_inplace(img, saturation, contrast, pivot, luma_coeffs):
    """Apply saturation and contrast adjustment in-place."""
    _apply_saturation_contrast_kernel(img, float(saturation), float(contrast),
                                       float(pivot),
                                       float(luma_coeffs[0]), float(luma_coeffs[1]), float(luma_coeffs[2]))


@ti.kernel
def _apply_white_balance_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3),
                                 r_gain: ti.f32, g_gain: ti.f32, b_gain: ti.f32):
    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        img[r, c, 0] *= r_gain
        img[r, c, 1] *= g_gain
        img[r, c, 2] *= b_gain


def apply_white_balance_inplace(img, r_gain, g_gain, b_gain):
    """Apply per-channel white balance gains in-place."""
    _apply_white_balance_kernel(img, float(r_gain), float(g_gain), float(b_gain))


@ti.kernel
def _apply_highlight_shadow_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3),
                                    highlight: ti.f32, shadow: ti.f32,
                                    cr: ti.f32, cg: ti.f32, cb: ti.f32):
    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        r_v = img[r, c, 0]
        g_v = img[r, c, 1]
        b_v = img[r, c, 2]

        lum = r_v * cr + g_v * cg + b_v * cb

        if shadow != 0.0:
            mask = 1.0 - lum
            if mask > 0.0:
                factor = 1.0 + shadow * (mask * mask * mask)
                r_v *= factor
                g_v *= factor
                b_v *= factor

        if highlight != 0.0:
            mask_h = lum
            if mask_h < 0.0:
                mask_h = 0.0
            elif mask_h > 1.0:
                mask_h = 1.0
            t = 1.0 - mask_h
            factor = 1.0 + highlight * (1.0 - t * t * t)
            if factor < 0.0:
                factor = 0.0
            r_v *= factor
            g_v *= factor
            b_v *= factor

        if r_v < 0.0:
            r_v = 0.0
        if g_v < 0.0:
            g_v = 0.0
        if b_v < 0.0:
            b_v = 0.0

        img[r, c, 0] = r_v
        img[r, c, 1] = g_v
        img[r, c, 2] = b_v


def apply_highlight_shadow_inplace(img, highlight, shadow, luma_coeffs):
    """Apply highlight/shadow tone mapping in-place."""
    _apply_highlight_shadow_kernel(img, float(highlight), float(shadow),
                                    float(luma_coeffs[0]), float(luma_coeffs[1]), float(luma_coeffs[2]))


@ti.kernel
def _apply_gain_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3), gain: ti.f32):
    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        img[r, c, 0] *= gain
        img[r, c, 1] *= gain
        img[r, c, 2] *= gain


def apply_gain_inplace(img, gain):
    """Apply uniform exposure gain in-place."""
    _apply_gain_kernel(img, float(gain))


@ti.kernel
def _linear_to_srgb_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        for ch in ti.static(range(3)):
            linear = img[r, c, ch]
            result = ti.cast(0.0, ti.f32)
            if linear <= 0.0031308:
                result = linear * 12.92
            else:
                result = 1.055 * ti.pow(linear, 1.0 / 2.4) - 0.055
            img[r, c, ch] = result


def linear_to_srgb_inplace(img):
    """Convert linear RGB to sRGB gamma in-place."""
    if _is_numpy(img) and not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    _linear_to_srgb_kernel(img)


@ti.kernel
def _srgb_to_linear_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        for ch in ti.static(range(3)):
            srgb = img[r, c, ch]
            result = ti.cast(0.0, ti.f32)
            if srgb <= 0.04045:
                result = srgb / 12.92
            else:
                result = ti.pow((srgb + 0.055) / 1.055, 2.4)
            img[r, c, ch] = result


def srgb_to_linear_inplace(img):
    """Convert sRGB gamma to linear RGB in-place (inverse of linear_to_srgb)."""
    if _is_numpy(img) and not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    _srgb_to_linear_kernel(img)


@ti.kernel
def _bt709_to_srgb_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for r, c in ti.ndrange(img.shape[0], img.shape[1]):
        for ch in ti.static(range(3)):
            val = img[r, c, ch]

            # BT.709 EOTF inverse -> linear
            linear = ti.cast(0.0, ti.f32)
            if val < 0.081:
                linear = val / 4.5
            else:
                linear = ti.pow((val + 0.099) / 1.099, 1.0 / 0.45)

            # linear -> sRGB
            result = ti.cast(0.0, ti.f32)
            if linear <= 0.0031308:
                result = linear * 12.92
            else:
                result = 1.055 * ti.pow(linear, 1.0 / 2.4) - 0.055

            img[r, c, ch] = result


def bt709_to_srgb_inplace(img):
    """Convert BT.709 encoded data to sRGB in-place."""
    if _is_numpy(img) and not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    _bt709_to_srgb_kernel(img)


# =========================================================
# Histogram & Waveform (CPU-side, since reduction is tricky on GPU)
# =========================================================
# NOTE: Taichi atomic operations make GPU histograms possible,
# but for subsampled data the CPU path is fast enough.
# We use Taichi for the inner loop with ti.atomic_add.

@ti.kernel
def _compute_histogram_kernel(data: ti.types.ndarray(dtype=ti.f32, ndim=1),
                               hist: ti.types.ndarray(dtype=ti.i64, ndim=1),
                               bins: ti.i32,
                               min_val: ti.f32, max_val: ti.f32):
    bin_width = (max_val - min_val) / ti.cast(bins, ti.f32)
    for i in range(data.shape[0]):
        val = data[i]
        if min_val <= val <= max_val:
            bin_idx = ti.cast((val - min_val) / bin_width, ti.i32)
            if bin_idx >= bins:
                bin_idx = bins - 1
            ti.atomic_add(hist[bin_idx], 1)


def compute_histogram_channel(data, bins, min_val, max_val):
    """Compute histogram for a 1D channel array. Returns int64 array."""
    hist = np.zeros(bins, dtype=np.int64)
    if max_val <= min_val:
        return hist
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    _compute_histogram_kernel(data, hist, bins, float(min_val), float(max_val))
    return hist


@ti.kernel
def _compute_waveform_kernel(channel_data: ti.types.ndarray(dtype=ti.f32, ndim=2),
                              waveform_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
                              bins: ti.i32, sample_rate: ti.i32):
    h = channel_data.shape[0]
    w = channel_data.shape[1]
    sampled_width = w // sample_rate
    if sampled_width == 0:
        sampled_width = 1

    ire_min = -4.0
    ire_max = 109.0
    ire_range = ire_max - ire_min
    bins_minus_1 = bins - 1

    for i, row in ti.ndrange(sampled_width, h):
        col_idx = i * sample_rate
        val = channel_data[row, col_idx]
        ire_value = val * 100.0
        normalized_pos = (ire_value - ire_min) / ire_range
        bin_idx = ti.cast(normalized_pos * ti.cast(bins_minus_1, ti.f32), ti.i32)
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= bins:
            bin_idx = bins - 1
        ti.atomic_add(waveform_out[i, bin_idx], 1.0)


def compute_waveform_channel(channel_data, waveform_out, bins, sample_rate):
    """Compute waveform data for a single channel. Modifies waveform_out in-place."""
    if not channel_data.flags['C_CONTIGUOUS']:
        channel_data = np.ascontiguousarray(channel_data)
    _compute_waveform_kernel(channel_data, waveform_out, bins, sample_rate)


# =========================================================
# Perspective Warp
# =========================================================

@ti.kernel
def _perspective_warp_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                              dst: ti.types.ndarray(dtype=ti.f32, ndim=3),
                              m00: ti.f64, m01: ti.f64, m02: ti.f64,
                              m10: ti.f64, m11: ti.f64, m12: ti.f64,
                              m20: ti.f64, m21: ti.f64, m22: ti.f64):
    dst_h = dst.shape[0]
    dst_w = dst.shape[1]
    src_h = src.shape[0]
    src_w = src.shape[1]

    src_h_1 = ti.cast(src_h - 1, ti.f64)
    src_w_1 = ti.cast(src_w - 1, ti.f64)

    for y, x in ti.ndrange(dst_h, dst_w):
        xf = ti.cast(x, ti.f64)
        yf = ti.cast(y, ti.f64)
        denom = m20 * xf + m21 * yf + m22
        if denom == 0.0:
            denom = 1e-10

        src_x = (m00 * xf + m01 * yf + m02) / denom
        src_y = (m10 * xf + m11 * yf + m12) / denom

        # Clamp to image bounds (edge reflection)
        if src_x < 0.0:
            src_x = 0.0
        elif src_x > src_w_1:
            src_x = src_w_1
        if src_y < 0.0:
            src_y = 0.0
        elif src_y > src_h_1:
            src_y = src_h_1

        x0 = ti.cast(src_x, ti.i32)
        y0 = ti.cast(src_y, ti.i32)
        x1 = ti.min(x0 + 1, ti.cast(src_w_1, ti.i32))
        y1 = ti.min(y0 + 1, ti.cast(src_h_1, ti.i32))

        dx = ti.cast(src_x - ti.cast(x0, ti.f64), ti.f32)
        dy = ti.cast(src_y - ti.cast(y0, ti.f64), ti.f32)

        w00 = (1.0 - dx) * (1.0 - dy)
        w01 = dx * (1.0 - dy)
        w10 = (1.0 - dx) * dy
        w11 = dx * dy

        for ch in ti.static(range(3)):
            val = (src[y0, x0, ch] * w00 +
                   src[y0, x1, ch] * w01 +
                   src[y1, x0, ch] * w10 +
                   src[y1, x1, ch] * w11)
            dst[y, x, ch] = val


def perspective_warp_kernel(src, dst, M_inv):
    """Perspective warp using inverse mapping + bilinear interpolation."""
    M = np.ascontiguousarray(M_inv.astype(np.float64))
    _perspective_warp_kernel(src, dst,
                              M[0, 0], M[0, 1], M[0, 2],
                              M[1, 0], M[1, 1], M[1, 2],
                              M[2, 0], M[2, 1], M[2, 2])


def compute_perspective_matrix(src_corners, dst_width, dst_height):
    """
    Compute 3x3 perspective transform matrix from 4 source corners to destination rectangle.
    src_corners: 4 points in normalized coords [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                 Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    Returns: (H, H_inv) - forward and inverse 3x3 matrices
    """
    src = np.array([
        [src_corners[0][0] * dst_width, src_corners[0][1] * dst_height],
        [src_corners[1][0] * dst_width, src_corners[1][1] * dst_height],
        [src_corners[2][0] * dst_width, src_corners[2][1] * dst_height],
        [src_corners[3][0] * dst_width, src_corners[3][1] * dst_height],
    ], dtype=np.float64)

    dst = np.array([
        [0.0, 0.0],
        [dst_width - 1, 0.0],
        [dst_width - 1, dst_height - 1],
        [0.0, dst_height - 1],
    ], dtype=np.float64)

    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)

    for i in range(4):
        sx, sy = src[i, 0], src[i, 1]
        dx, dy = dst[i, 0], dst[i, 1]

        A[i * 2, 0] = sx
        A[i * 2, 1] = sy
        A[i * 2, 2] = 1.0
        A[i * 2, 6] = -dx * sx
        A[i * 2, 7] = -dx * sy
        b[i * 2] = dx

        A[i * 2 + 1, 3] = sx
        A[i * 2 + 1, 4] = sy
        A[i * 2 + 1, 5] = 1.0
        A[i * 2 + 1, 6] = -dy * sx
        A[i * 2 + 1, 7] = -dy * sy
        b[i * 2 + 1] = dy

    try:
        h = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.eye(3, dtype=np.float64)

    return H, H_inv


# =========================================================
# Richardson-Lucy Deconvolution Sharpening
# =========================================================

def _gaussian_kernel_1d(sigma: float, size: int = None) -> np.ndarray:
    """Create a normalized 1D Gaussian kernel."""
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1

    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _gaussian_kernel(sigma: float, size: int = None) -> np.ndarray:
    """Create a 2D Gaussian kernel for PSF."""
    k1d = _gaussian_kernel_1d(sigma, size)
    kernel_2d = np.outer(k1d, k1d)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d.astype(np.float32)


# --- Separable 1D convolution kernels (Taichi) ---

@ti.kernel
def _convolve_1d_h_reflect_kernel(channel: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                    kernel_1d: ti.types.ndarray(dtype=ti.f32, ndim=1),
                                    output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    h = channel.shape[0]
    w = channel.shape[1]
    k = kernel_1d.shape[0]
    k2 = k // 2

    for y, x in ti.ndrange(h, w):
        val = ti.cast(0.0, ti.f32)
        for kx in range(k):
            sx = x + kx - k2
            if sx < 0:
                sx = -sx
            elif sx >= w:
                sx = 2 * w - sx - 2
            if sx < 0:
                sx = 0
            if sx >= w:
                sx = w - 1
            val += channel[y, sx] * kernel_1d[kx]
        output[y, x] = val


@ti.kernel
def _convolve_1d_v_reflect_kernel(channel: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                    kernel_1d: ti.types.ndarray(dtype=ti.f32, ndim=1),
                                    output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    h = channel.shape[0]
    w = channel.shape[1]
    k = kernel_1d.shape[0]
    k2 = k // 2

    for y, x in ti.ndrange(h, w):
        val = ti.cast(0.0, ti.f32)
        for ky in range(k):
            sy = y + ky - k2
            if sy < 0:
                sy = -sy
            elif sy >= h:
                sy = 2 * h - sy - 2
            if sy < 0:
                sy = 0
            if sy >= h:
                sy = h - 1
            val += channel[sy, x] * kernel_1d[ky]
        output[y, x] = val


@ti.kernel
def _rl_update_ratio_kernel(channel: ti.types.ndarray(dtype=ti.f32, ndim=2),
                             blurred: ti.types.ndarray(dtype=ti.f32, ndim=2),
                             ratio: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    eps = 1e-8
    for y, x in ti.ndrange(channel.shape[0], channel.shape[1]):
        b = blurred[y, x]
        if b < eps:
            b = eps
        ratio[y, x] = channel[y, x] / b


@ti.kernel
def _rl_ratio_from_hwc_kernel(src_hwc: ti.types.ndarray(dtype=ti.f32, ndim=3),
                                blurred: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                ratio: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                ch: ti.i32):
    eps = 1e-8
    for y, x in ti.ndrange(src_hwc.shape[0], src_hwc.shape[1]):
        b = blurred[y, x]
        if b < eps:
            b = eps
        ratio[y, x] = src_hwc[y, x, ch] / b


@ti.kernel
def _rl_update_estimate_kernel(estimate: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                correction: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for y, x in ti.ndrange(estimate.shape[0], estimate.shape[1]):
        val = estimate[y, x] * correction[y, x]
        if val < 0.0:
            val = 0.0
        elif val > 2.0:
            val = 2.0
        estimate[y, x] = val


def _rl_iteration_sep(estimate, channel, kernel_1d, blurred, ratio, correction, temp):
    """Single RL iteration using separable convolution."""
    # Forward: blurred = estimate * PSF
    _convolve_1d_h_reflect_kernel(estimate, kernel_1d, temp)
    _convolve_1d_v_reflect_kernel(temp, kernel_1d, blurred)

    # Ratio: observed / blurred
    _rl_update_ratio_kernel(channel, blurred, ratio)

    # Backward: correction = ratio * PSF_mirror (symmetric for Gaussian)
    _convolve_1d_h_reflect_kernel(ratio, kernel_1d, temp)
    _convolve_1d_v_reflect_kernel(temp, kernel_1d, correction)

    # Update estimate
    _rl_update_estimate_kernel(estimate, correction)


def richardson_lucy_channel(
    channel: np.ndarray,
    kernel_1d: np.ndarray,
    iterations: int = 10,
    clip: bool = True
) -> np.ndarray:
    """Apply Richardson-Lucy deconvolution to a single channel using separable Gaussian."""
    h, w = channel.shape
    estimate = channel.copy().astype(np.float32)
    kernel_1d = np.ascontiguousarray(kernel_1d.astype(np.float32))

    blurred = np.empty_like(estimate)
    ratio = np.empty_like(estimate)
    correction = np.empty_like(estimate)
    temp = np.empty_like(estimate)

    for _ in range(iterations):
        _rl_iteration_sep(estimate, channel, kernel_1d, blurred, ratio, correction, temp)

    if clip:
        estimate = np.clip(estimate, 0, 1)

    return estimate


def richardson_lucy(
    image: np.ndarray,
    sigma: float = 1.0,
    iterations: int = 10,
    strength: float = 1.0
) -> np.ndarray:
    """Apply Richardson-Lucy deconvolution sharpening to an RGB image."""
    if strength <= 0 or iterations <= 0:
        return image

    logger.debug(f"RL sharpening: sigma={sigma}, iterations={iterations}, strength={strength}")

    kernel_1d = _gaussian_kernel_1d(sigma)
    h, w, c = image.shape
    result = np.empty_like(image)

    for i in range(c):
        result[:, :, i] = richardson_lucy_channel(
            np.ascontiguousarray(image[:, :, i].astype(np.float32)),
            kernel_1d,
            iterations
        )

    if strength < 1.0:
        result = image * (1 - strength) + result * strength

    return result.astype(np.float32)


def sharpen(
    image: np.ndarray,
    strength: float = 0.5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Simplified sharpening interface.

    Args:
        image: RGB image in HWC format, float32, range [0, 1]
        strength: Sharpening strength (0 = off, 1 = max). Maps to 1-10 RL iterations.
        sigma: PSF sigma (default 1.0)
    """
    if strength <= 0:
        return image

    iterations = max(1, int(strength * 10))
    return richardson_lucy(image, sigma=sigma, iterations=iterations, strength=1.0)


# =========================================================
# GPU-Resident RL Sharpening (per-channel 2D, memory efficient)
# =========================================================

@ti.kernel
def _extract_channel_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                              dst: ti.types.ndarray(dtype=ti.f32, ndim=2),
                              ch: ti.i32):
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        dst[y, x] = src[y, x, ch]


@ti.kernel
def _insert_channel_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=2),
                             dst: ti.types.ndarray(dtype=ti.f32, ndim=3),
                             ch: ti.i32):
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        dst[y, x, ch] = src[y, x]


@ti.kernel
def _blend_channel_kernel(original: ti.types.ndarray(dtype=ti.f32, ndim=3),
                            sharpened: ti.types.ndarray(dtype=ti.f32, ndim=2),
                            ch: ti.i32, alpha: ti.f32):
    for y, x in ti.ndrange(original.shape[0], original.shape[1]):
        sharpened[y, x] = original[y, x, ch] * (1.0 - alpha) + sharpened[y, x] * alpha


# Cache for GPU work buffers to avoid re-allocation
_sharpen_gpu_bufs = {}


def sharpen_gpu(gpu_image, strength: float = 0.5, sigma: float = 1.0):
    """
    GPU-resident RL sharpening. Operates on GpuImage in-place.
    Processes per-channel with 2D buffers to minimize GPU memory usage.
    Only 4 × (H×W×4) bytes extra GPU memory (~640MB for 42MP).
    """
    if strength <= 0 or not gpu_image.valid:
        return

    iterations = max(1, int(strength * 10))
    h, w = gpu_image.height, gpu_image.width
    shape2d = (h, w)

    # Kernel (tiny, created on CPU then uploaded once)
    kernel_1d_np = _gaussian_kernel_1d(sigma)
    kernel_1d = ti.ndarray(dtype=ti.f32, shape=(kernel_1d_np.shape[0],))
    kernel_1d.from_numpy(kernel_1d_np)

    # Allocate/reuse 2D GPU work buffers (4 buffers shared across channels)
    global _sharpen_gpu_bufs
    if _sharpen_gpu_bufs.get('shape') != shape2d:
        _sharpen_gpu_bufs = {
            'shape': shape2d,
            'estimate': ti.ndarray(dtype=ti.f32, shape=shape2d),
            'buf_a': ti.ndarray(dtype=ti.f32, shape=shape2d),  # blurred, then correction
            'buf_b': ti.ndarray(dtype=ti.f32, shape=shape2d),  # ratio
            'temp': ti.ndarray(dtype=ti.f32, shape=shape2d),
        }

    estimate = _sharpen_gpu_bufs['estimate']
    buf_a = _sharpen_gpu_bufs['buf_a']
    buf_b = _sharpen_gpu_bufs['buf_b']
    temp = _sharpen_gpu_bufs['temp']

    # Process each channel independently (reuses same 2D buffers)
    for ch in range(3):
        # Extract channel from HWC image → estimate (GPU→GPU)
        _extract_channel_kernel(gpu_image.arr, estimate, ch)

        # RL iterations using existing 2D kernels
        for _ in range(iterations):
            # Forward: blurred = estimate * PSF (separable)
            _convolve_1d_h_reflect_kernel(estimate, kernel_1d, temp)
            _convolve_1d_v_reflect_kernel(temp, kernel_1d, buf_a)  # buf_a = blurred
            # Ratio: observed / blurred (read observed from HWC source)
            _rl_ratio_from_hwc_kernel(gpu_image.arr, buf_a, buf_b, ch)  # buf_b = ratio
            # Backward: correction = ratio * PSF_mirror (symmetric Gaussian)
            _convolve_1d_h_reflect_kernel(buf_b, kernel_1d, temp)
            _convolve_1d_v_reflect_kernel(temp, kernel_1d, buf_a)  # buf_a = correction
            # Update estimate
            _rl_update_estimate_kernel(estimate, buf_a)

        # Blend with original if strength < 1, then insert back
        if strength < 1.0:
            _blend_channel_kernel(gpu_image.arr, estimate, ch, float(strength))
        _insert_channel_kernel(estimate, gpu_image.arr, ch)

    logger.debug(f"GPU RL sharpen: {iterations} iters, sigma={sigma}, strength={strength}")


# =========================================================
# 2D Convolution (kept for compatibility)
# =========================================================

@ti.kernel
def _convolve_2d_reflect_kernel(channel: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                 kernel: ti.types.ndarray(dtype=ti.f32, ndim=2),
                                 output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    h = channel.shape[0]
    w = channel.shape[1]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    kh2 = kh // 2
    kw2 = kw // 2

    for y, x in ti.ndrange(h, w):
        val = ti.cast(0.0, ti.f32)
        for ky in range(kh):
            for kx in range(kw):
                sy = y + ky - kh2
                sx = x + kx - kw2

                if sy < 0:
                    sy = -sy
                elif sy >= h:
                    sy = 2 * h - sy - 2
                if sx < 0:
                    sx = -sx
                elif sx >= w:
                    sx = 2 * w - sx - 2
                if sy < 0:
                    sy = 0
                if sy >= h:
                    sy = h - 1
                if sx < 0:
                    sx = 0
                if sx >= w:
                    sx = w - 1

                val += channel[sy, sx] * kernel[ky, kx]
        output[y, x] = val


def _convolve_2d_reflect(channel, kernel, output):
    """2D convolution with reflect boundary (Taichi accelerated)."""
    _convolve_2d_reflect_kernel(channel, kernel, output)


# Expose 1D convolution functions for compatibility
def _convolve_1d_h_reflect(channel, kernel_1d, output):
    _convolve_1d_h_reflect_kernel(channel, kernel_1d, output)

def _convolve_1d_v_reflect(channel, kernel_1d, output):
    _convolve_1d_v_reflect_kernel(channel, kernel_1d, output)


# =========================================================
# GPU Downsample (area-average for display)
# =========================================================

@ti.kernel
def _downsample_area_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                             dst: ti.types.ndarray(dtype=ti.f32, ndim=3),
                             src_h: ti.i32, src_w: ti.i32,
                             dst_h: ti.i32, dst_w: ti.i32):
    """Area-average downsample: each dst pixel = average of corresponding src region."""
    scale_y = ti.cast(src_h, ti.f32) / ti.cast(dst_h, ti.f32)
    scale_x = ti.cast(src_w, ti.f32) / ti.cast(dst_w, ti.f32)

    for dy, dx in ti.ndrange(dst_h, dst_w):
        # Source region for this destination pixel
        sy0 = ti.cast(ti.cast(dy, ti.f32) * scale_y, ti.i32)
        sy1 = ti.cast(ti.cast(dy + 1, ti.f32) * scale_y, ti.i32)
        sx0 = ti.cast(ti.cast(dx, ti.f32) * scale_x, ti.i32)
        sx1 = ti.cast(ti.cast(dx + 1, ti.f32) * scale_x, ti.i32)

        # Clamp
        if sy1 > src_h:
            sy1 = src_h
        if sx1 > src_w:
            sx1 = src_w
        if sy0 >= sy1:
            sy0 = sy1 - 1
        if sx0 >= sx1:
            sx0 = sx1 - 1

        count = ti.cast((sy1 - sy0) * (sx1 - sx0), ti.f32)
        if count < 1.0:
            count = 1.0

        acc_r = ti.cast(0.0, ti.f32)
        acc_g = ti.cast(0.0, ti.f32)
        acc_b = ti.cast(0.0, ti.f32)

        for sy in range(sy0, sy1):
            for sx in range(sx0, sx1):
                acc_r += src[sy, sx, 0]
                acc_g += src[sy, sx, 1]
                acc_b += src[sy, sx, 2]

        dst[dy, dx, 0] = acc_r / count
        dst[dy, dx, 1] = acc_g / count
        dst[dy, dx, 2] = acc_b / count


def downsample_gpu(src, dst, src_h, src_w, dst_h, dst_w):
    """GPU area-average downsample. Works with both numpy and ti.ndarray."""
    _downsample_area_kernel(src, dst, src_h, src_w, dst_h, dst_w)


@ti.kernel
def _float_to_uint8_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                            dst: ti.types.ndarray(dtype=ti.u8, ndim=3)):
    """Convert float32 [0,1] to uint8 [0,255] with clamp, on GPU."""
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        for ch in ti.static(range(3)):
            val = src[y, x, ch]
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            dst[y, x, ch] = ti.cast(val * 255.0 + 0.5, ti.u8)


def float_to_uint8_gpu(src, dst):
    """GPU float32→uint8 conversion. Works with both numpy and ti.ndarray."""
    _float_to_uint8_kernel(src, dst)


@ti.kernel
def _clip_inplace_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    """Clip image values to [0, 1] on GPU."""
    for y, x in ti.ndrange(img.shape[0], img.shape[1]):
        for ch in ti.static(range(3)):
            val = img[y, x, ch]
            if val < 0.0:
                img[y, x, ch] = 0.0
            elif val > 1.0:
                img[y, x, ch] = 1.0


def clip_inplace(img):
    """GPU clip to [0,1]. Works with both numpy and ti.ndarray."""
    _clip_inplace_kernel(img)


@ti.kernel
def _max_inplace_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3), min_val: ti.f32):
    """Clamp image values to >= min_val on GPU (like np.maximum)."""
    for y, x in ti.ndrange(img.shape[0], img.shape[1]):
        for ch in ti.static(range(3)):
            if img[y, x, ch] < min_val:
                img[y, x, ch] = min_val


def max_inplace(img, min_val):
    """GPU np.maximum equivalent. Works with both numpy and ti.ndarray."""
    _max_inplace_kernel(img, float(min_val))


# =========================================================
# GPU Lens Remap (replaces scipy.map_coordinates)
# =========================================================

@ti.func
def _bicubic_weight(t: ti.f32) -> ti.types.vector(4, ti.f32):
    """Mitchell-Netravali bicubic weights (B=0, C=0.5 = Catmull-Rom)."""
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2
    return ti.Vector([w0, w1, w2, w3])


@ti.func
def _sample_bicubic(
    src: ti.types.ndarray(dtype=ti.f32, ndim=2),
    x: ti.f32, y: ti.f32, H: ti.i32, W: ti.i32,
) -> ti.f32:
    """Bicubic interpolation of a 2D array at (x, y)."""
    ix = ti.cast(ti.floor(x), ti.i32)
    iy = ti.cast(ti.floor(y), ti.i32)
    fx = x - ti.cast(ix, ti.f32)
    fy = y - ti.cast(iy, ti.f32)

    wx = _bicubic_weight(fx)
    wy = _bicubic_weight(fy)

    val = ti.cast(0.0, ti.f32)
    for dy in ti.static(range(4)):
        for dx in ti.static(range(4)):
            sy = ti.min(ti.max(iy - 1 + dy, 0), H - 1)
            sx = ti.min(ti.max(ix - 1 + dx, 0), W - 1)
            val += wy[dy] * wx[dx] * src[sy, sx]
    return val


@ti.kernel
def _lens_remap_kernel(
    src: ti.types.ndarray(dtype=ti.f32, ndim=3),   # (H, W, 3) input
    dst: ti.types.ndarray(dtype=ti.f32, ndim=3),   # (H, W, 3) output
    coords_x: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (H, W, 3) x coords per channel
    coords_y: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (H, W, 3) y coords per channel
    oob_mask: ti.types.ndarray(dtype=ti.i32, ndim=2),   # (H, W) out-of-bounds mask
):
    """GPU bicubic remap with per-channel coordinates (for TCA correction)."""
    H = src.shape[0]
    W = src.shape[1]
    for row, col in ti.ndrange(dst.shape[0], dst.shape[1]):
        if oob_mask[row, col] != 0:
            dst[row, col, 0] = 0.0
            dst[row, col, 1] = 0.0
            dst[row, col, 2] = 0.0
        else:
            for ch in ti.static(range(3)):
                x = coords_x[row, col, ch]
                y = coords_y[row, col, ch]
                dst[row, col, ch] = _sample_bicubic(src[:, :, ch], x, y, H, W)


def lens_remap_gpu(src_gpu, dst_gpu, coords, oob_mask_np):
    """GPU lens distortion remap (replaces scipy.map_coordinates).

    Args:
        src_gpu: Source GpuImage (HxWx3 float32)
        dst_gpu: Destination GpuImage (HxWx3 float32)
        coords: (H, W, 3, 2) float32 numpy — per-channel (x, y) from lensfun
        oob_mask_np: (H, W) bool numpy — out-of-bounds mask
    """
    h, w = coords.shape[:2]
    dst_gpu._allocate(h, w, 3)

    # Split coords into x and y, upload to GPU
    coords_x = ti.ndarray(dtype=ti.f32, shape=(h, w, 3))
    coords_y = ti.ndarray(dtype=ti.f32, shape=(h, w, 3))
    oob = ti.ndarray(dtype=ti.i32, shape=(h, w))

    coords_x.from_numpy(np.ascontiguousarray(coords[:, :, :, 0], dtype=np.float32))
    coords_y.from_numpy(np.ascontiguousarray(coords[:, :, :, 1], dtype=np.float32))
    oob.from_numpy(oob_mask_np.astype(np.int32))

    _lens_remap_kernel(src_gpu.arr, dst_gpu.arr, coords_x, coords_y, oob)

    del coords_x, coords_y, oob


# =========================================================
# Fused Exposure + Adjustments + sRGB Grading (single pass)
# =========================================================

@ti.kernel
def _fused_expose_adjust_srgb_kernel(
    src: ti.types.ndarray(dtype=ti.f32, ndim=3),
    dst: ti.types.ndarray(dtype=ti.f32, ndim=3),
    # Exposure
    gain: ti.f32,
    # White balance
    wb_r: ti.f32, wb_g: ti.f32, wb_b: ti.f32,
    # Highlight / Shadow
    highlight: ti.f32, shadow: ti.f32,
    # Saturation / Contrast
    saturation: ti.f32, contrast: ti.f32, pivot: ti.f32,
    # Luma coefficients
    cr: ti.f32, cg: ti.f32, cb: ti.f32,
    # Color matrix (ProPhoto→sRGB or ProPhoto→Log gamut)
    m00: ti.f32, m01: ti.f32, m02: ti.f32,
    m10: ti.f32, m11: ti.f32, m12: ti.f32,
    m20: ti.f32, m21: ti.f32, m22: ti.f32,
    # sRGB OETF flag (1=apply sRGB, 0=skip for Log path)
    apply_srgb: ti.i32,
):
    for r, c in ti.ndrange(src.shape[0], src.shape[1]):
        # Read once from global memory
        rv = src[r, c, 0] * gain
        gv = src[r, c, 1] * gain
        bv = src[r, c, 2] * gain

        # White balance
        rv *= wb_r
        gv *= wb_g
        bv *= wb_b

        # Highlight / Shadow
        lum = rv * cr + gv * cg + bv * cb

        if shadow != 0.0:
            mask = 1.0 - lum
            if mask > 0.0:
                factor = 1.0 + shadow * (mask * mask * mask)
                rv *= factor
                gv *= factor
                bv *= factor

        if highlight != 0.0:
            mask_h = ti.min(ti.max(lum, 0.0), 1.0)
            t = 1.0 - mask_h
            factor = ti.max(1.0 + highlight * (1.0 - t * t * t), 0.0)
            rv *= factor
            gv *= factor
            bv *= factor

        rv = ti.max(rv, 0.0)
        gv = ti.max(gv, 0.0)
        bv = ti.max(bv, 0.0)

        # Saturation
        lum2 = rv * cr + gv * cg + bv * cb
        rv = lum2 + (rv - lum2) * saturation
        gv = lum2 + (gv - lum2) * saturation
        bv = lum2 + (bv - lum2) * saturation

        # Contrast
        rv = (rv - pivot) * contrast + pivot
        gv = (gv - pivot) * contrast + pivot
        bv = (bv - pivot) * contrast + pivot

        rv = ti.max(rv, 0.0)
        gv = ti.max(gv, 0.0)
        bv = ti.max(bv, 0.0)

        # Color matrix (ProPhoto→sRGB or gamut)
        out_r = rv * m00 + gv * m01 + bv * m02
        out_g = rv * m10 + gv * m11 + bv * m12
        out_b = rv * m20 + gv * m21 + bv * m22

        # sRGB OETF (or skip for Log path)
        if apply_srgb == 1:
            if out_r <= 0.0031308:
                out_r = out_r * 12.92
            else:
                out_r = 1.055 * ti.pow(ti.max(out_r, 0.0), 1.0 / 2.4) - 0.055
            if out_g <= 0.0031308:
                out_g = out_g * 12.92
            else:
                out_g = 1.055 * ti.pow(ti.max(out_g, 0.0), 1.0 / 2.4) - 0.055
            if out_b <= 0.0031308:
                out_b = out_b * 12.92
            else:
                out_b = 1.055 * ti.pow(ti.max(out_b, 0.0), 1.0 / 2.4) - 0.055

        # Clip
        dst[r, c, 0] = ti.min(ti.max(out_r, 0.0), 1.0)
        dst[r, c, 1] = ti.min(ti.max(out_g, 0.0), 1.0)
        dst[r, c, 2] = ti.min(ti.max(out_b, 0.0), 1.0)


def fused_expose_adjust_grade(src, dst, gain,
                               wb_r, wb_g, wb_b,
                               highlight, shadow,
                               saturation, contrast, pivot,
                               luma_coeffs, matrix, apply_srgb):
    """Fused GPU kernel: exposure → WB → HL/SH → sat/con → matrix → sRGB → clip.

    Single pass over the image. Eliminates 2 GPU-to-GPU copies.

    Args:
        src: Source GpuImage.arr (ti.ndarray, HxWx3 float32)
        dst: Destination GpuImage.arr (ti.ndarray, HxWx3 float32)
        gain: Exposure gain
        wb_r, wb_g, wb_b: White balance per-channel gains
        highlight, shadow: Tone mapping (-1..1)
        saturation, contrast: Color adjustments
        pivot: Contrast pivot (0.18)
        luma_coeffs: (3,) array of luminance coefficients
        matrix: 3x3 color matrix (numpy)
        apply_srgb: True to apply sRGB OETF, False for Log path
    """
    m = np.ascontiguousarray(matrix).astype(np.float64)
    _fused_expose_adjust_srgb_kernel(
        src, dst,
        float(gain),
        float(wb_r), float(wb_g), float(wb_b),
        float(highlight), float(shadow),
        float(saturation), float(contrast), float(pivot),
        float(luma_coeffs[0]), float(luma_coeffs[1]), float(luma_coeffs[2]),
        float(m[0, 0]), float(m[0, 1]), float(m[0, 2]),
        float(m[1, 0]), float(m[1, 1]), float(m[1, 2]),
        float(m[2, 0]), float(m[2, 1]), float(m[2, 2]),
        1 if apply_srgb else 0,
    )


# =========================================================
# GPU Geometry (Rotate / Flip)
# =========================================================

@ti.kernel
def _rotate90_cw_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                         dst: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    """Rotate 90° clockwise: dst[x, H-1-y] = src[y, x]"""
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        for ch in ti.static(range(3)):
            dst[x, src.shape[0] - 1 - y, ch] = src[y, x, ch]

@ti.kernel
def _rotate180_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                       dst: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        for ch in ti.static(range(3)):
            dst[src.shape[0] - 1 - y, src.shape[1] - 1 - x, ch] = src[y, x, ch]

@ti.kernel
def _rotate270_cw_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                          dst: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    """Rotate 270° clockwise (= 90° CCW): dst[W-1-x, y] = src[y, x]"""
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        for ch in ti.static(range(3)):
            dst[src.shape[1] - 1 - x, y, ch] = src[y, x, ch]

@ti.kernel
def _flip_h_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                    dst: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        for ch in ti.static(range(3)):
            dst[y, src.shape[1] - 1 - x, ch] = src[y, x, ch]

@ti.kernel
def _flip_v_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                    dst: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        for ch in ti.static(range(3)):
            dst[src.shape[0] - 1 - y, x, ch] = src[y, x, ch]

@ti.kernel
def _copy_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                  dst: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for y, x in ti.ndrange(src.shape[0], src.shape[1]):
        for ch in ti.static(range(3)):
            dst[y, x, ch] = src[y, x, ch]


def geometry_gpu(src, dst, rotation, flip_h, flip_v):
    """
    GPU geometry transform (rotate + flip).
    src/dst: ti.ndarray or numpy. dst must be pre-allocated with correct shape.
    rotation: 0, 90, 180, 270 (clockwise degrees)
    """
    rotation = rotation % 360

    # Step 1: Rotate
    if rotation == 90:
        _rotate90_cw_kernel(src, dst)
    elif rotation == 180:
        _rotate180_kernel(src, dst)
    elif rotation == 270:
        _rotate270_cw_kernel(src, dst)
    else:
        _copy_kernel(src, dst)

    # Step 2: Flip (in-place on dst)
    # For flips after rotation, we need a temp buffer or do it in-place
    # Since we can't easily flip in-place with Taichi, we'll handle combined ops
    if flip_h and flip_v:
        # flip_h + flip_v = rotate 180 of dst
        # We already wrote to dst, so we need to re-read it
        # Simpler: do the combined transform in the rotation step
        pass  # Handled below
    elif flip_h:
        pass
    elif flip_v:
        pass

    # If flips are needed, they must be part of the index calculation
    # Rewrite: combine rotation + flip into single kernel call
    return  # The simple kernels above only handle rotation


def geometry_gpu_shape(src_h, src_w, rotation):
    """Compute output shape after rotation."""
    rotation = rotation % 360
    if rotation == 90 or rotation == 270:
        return (src_w, src_h, 3)
    return (src_h, src_w, 3)


# Combined rotation+flip kernel for all cases
@ti.kernel
def _geometry_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                      dst: ti.types.ndarray(dtype=ti.f32, ndim=3),
                      rot: ti.i32, flip_h: ti.i32, flip_v: ti.i32):
    """Combined rotate + flip in a single pass."""
    src_h = src.shape[0]
    src_w = src.shape[1]
    dst_h = dst.shape[0]
    dst_w = dst.shape[1]

    for y, x in ti.ndrange(src_h, src_w):
        # Step 1: Apply rotation to get intermediate coordinates
        dy = 0
        dx = 0
        if rot == 0:
            dy = y
            dx = x
        elif rot == 90:
            dy = x
            dx = src_h - 1 - y
        elif rot == 180:
            dy = src_h - 1 - y
            dx = src_w - 1 - x
        elif rot == 270:
            dy = src_w - 1 - x
            dx = y

        # Step 2: Apply flips
        if flip_h == 1:
            dx = dst_w - 1 - dx
        if flip_v == 1:
            dy = dst_h - 1 - dy

        for ch in ti.static(range(3)):
            dst[dy, dx, ch] = src[y, x, ch]


def apply_geometry_gpu(src_buf, dst_buf, rotation=0, flip_h=False, flip_v=False):
    """
    GPU geometry transform. src_buf/dst_buf are GpuImage.
    dst_buf will be re-allocated if needed.
    """
    if rotation == 0 and not flip_h and not flip_v:
        dst_buf.copy_from(src_buf)
        return

    rotation = rotation % 360
    out_shape = geometry_gpu_shape(src_buf.height, src_buf.width, rotation)
    dst_buf._allocate(out_shape[0], out_shape[1], 3)
    _geometry_kernel(src_buf.arr, dst_buf.arr,
                      rotation, int(flip_h), int(flip_v))


# =========================================================
# GPU Crop
# =========================================================

@ti.kernel
def _crop_kernel(src: ti.types.ndarray(dtype=ti.f32, ndim=3),
                  dst: ti.types.ndarray(dtype=ti.f32, ndim=3),
                  src_x: ti.i32, src_y: ti.i32):
    """Copy a sub-region from src to dst."""
    for y, x in ti.ndrange(dst.shape[0], dst.shape[1]):
        for ch in ti.static(range(3)):
            dst[y, x, ch] = src[src_y + y, src_x + x, ch]


def apply_crop_gpu(src_buf, dst_buf, crop_rect):
    """
    GPU crop. crop_rect: (x, y, w, h) normalized 0-1.
    src_buf/dst_buf are GpuImage.
    """
    if not crop_rect or crop_rect == (0.0, 0.0, 1.0, 1.0):
        dst_buf.copy_from(src_buf)
        return

    h, w = src_buf.height, src_buf.width
    x_norm, y_norm, w_norm, h_norm = crop_rect
    if w_norm <= 0 or h_norm <= 0:
        dst_buf.copy_from(src_buf)
        return

    cx = max(0, min(int(x_norm * w), w - 1))
    cy = max(0, min(int(y_norm * h), h - 1))
    cw = max(1, min(int(w_norm * w), w - cx))
    ch = max(1, min(int(h_norm * h), h - cy))

    dst_buf._allocate(ch, cw, 3)
    _crop_kernel(src_buf.arr, dst_buf.arr, cx, cy)


# =========================================================
# GPU Log Encoding
# =========================================================
# Each log curve is a per-pixel transfer function.
# We implement the most common ones as Taichi kernels.

@ti.kernel
def _log_encode_kernel(img: ti.types.ndarray(dtype=ti.f32, ndim=3),
                        log_type: ti.i32):
    """
    GPU log encoding. log_type selects the curve:
      0 = S-Log3       4 = Canon Log 2    8 = F-Log2
      1 = V-Log        5 = Canon Log 3    9 = D-Log
      2 = N-Log        6 = Log3G10       10 = L-Log
      3 = Arri LogC3   7 = F-Log         11 = Arri LogC4
    """
    LOG10E = 0.4342944819032518  # 1 / ln(10), so log10(x) = log(x) * LOG10E
    LOG2E = 1.4426950408889634   # 1 / ln(2),  so log2(x)  = log(x) * LOG2E

    for y, x in ti.ndrange(img.shape[0], img.shape[1]):
        for ch in ti.static(range(3)):
            v = img[y, x, ch]
            result = v  # fallback

            if log_type == 0:
                # S-Log3
                if v >= 0.01125000:
                    result = (420.0 + ti.log(v * 261.5 + 0.01) * LOG10E * 261.5) / 1023.0
                else:
                    result = (v * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0

            elif log_type == 1:
                # V-Log
                b = 0.00873
                c = 0.241514
                d = 0.598206
                if v >= 0.01:
                    result = c * ti.log(v + 0.00873) * LOG10E + d
                else:
                    result = 5.6 * v + 0.125

            elif log_type == 2:
                # N-Log
                if v >= 328.0 / 65536.0:
                    result = (150.0 * ti.log(v) * LOG10E + 619.0) / 1023.0
                else:
                    result = 1000.0 / 17.0 * v

            elif log_type == 3:
                # ARRI LogC3 (EI 800)
                cut = 0.010591
                a = 5.555556
                b = 0.052272
                c_val = 0.247190
                d_val = 0.385537
                e_val = 5.367655
                f_val = 0.092809
                if v > cut:
                    result = c_val * ti.log(a * v + b) * LOG10E + d_val
                else:
                    result = e_val * v + f_val

            elif log_type == 4:
                # Canon Log 2
                if v >= -0.006:
                    result = 0.092864 * ti.log(v * 87.099375 + 1.0) * LOG10E + 0.5
                else:
                    result = 0.092864 * (v + 0.006) / 0.010797 + 0.5

            elif log_type == 5:
                # Canon Log 3
                if v >= -0.014:
                    result = 0.42889912 * ti.log(v * 14.98325 + 1.0) * LOG10E * 0.36726845 + 0.069886632
                else:
                    result = -0.42889912 * ti.log(-v * 14.98325 + 1.0) * LOG10E * 0.36726845 + 0.069886632

            elif log_type == 6:
                # Log3G10
                a_val = 0.224282
                b_val = 155.975327
                c_val = 0.01
                g = 15.1927
                if v >= -c_val:
                    result = a_val * ti.log((v + c_val) * b_val + 1.0) * LOG10E
                else:
                    result = (v + c_val) * g

            elif log_type == 7:
                # F-Log (Fujifilm)
                a_val = 0.555556
                b_val = 0.009468
                c_val = 0.344676
                d_val = 0.790453
                e_val = 8.735631
                f_val = 0.092864
                cut_val = 0.00089
                if v >= cut_val:
                    result = c_val * ti.log(a_val * v + b_val) * LOG10E + d_val
                else:
                    result = e_val * v + f_val

            elif log_type == 8:
                # F-Log2 (Fujifilm)
                a_val = 5.555556
                b_val = 0.064829
                c_val = 0.245281
                d_val = 0.384316
                e_val = 8.799461
                f_val = 0.092864
                cut_val = 0.000889
                if v >= cut_val:
                    result = c_val * ti.log(a_val * v + b_val) * LOG10E + d_val
                else:
                    result = e_val * v + f_val

            elif log_type == 9:
                # D-Log (DJI)
                if v <= 0.0078:
                    result = 6.025 * v + 0.0929
                else:
                    result = ti.log(v * 19.0 + 1.0) * LOG10E * 0.256663 + 0.584555

            elif log_type == 10:
                # L-Log (Leica)
                if v >= 0.006:
                    result = 0.432699 * ti.log(20.0 * v + 1.0) * LOG10E + 0.1
                else:
                    result = 7.5 * v + 0.055

            elif log_type == 11:
                # ARRI LogC4
                a_val = 2231.826309067688
                b_val = 64.0
                c_val = 0.0740718950408889
                s = (7.0 * ti.log(a_val) * LOG2E + 1.0) / 7.0
                # t derived from continuity
                t = (ti.pow(2.0, (0.0 - 1.0) / 7.0) - b_val) / a_val
                if v >= t:
                    result = (ti.log(a_val * v + b_val) * LOG2E + 1.0) / 14.0 + 0.5
                else:
                    result = (v - t) / c_val

            img[y, x, ch] = result


# Map log space names to log_type indices
_LOG_TYPE_MAP = {
    'S-Log3': 0,
    'V-Log': 1,
    'N-Log': 2,
    'Arri LogC3': 3,
    'Canon Log 2': 4,
    'Canon Log 3': 5,
    'Log3G10': 6,
    'F-Log': 7,
    'F-Log2': 8,
    'D-Log': 9,
    'L-Log': 10,
    'Arri LogC4': 11,
}


def log_encode_gpu(img, log_curve_name):
    """
    Apply log encoding on GPU. img: ti.ndarray or numpy.
    log_curve_name: one of the supported log curve names.
    Returns True if handled on GPU, False if fallback to CPU is needed.
    """
    log_type = _LOG_TYPE_MAP.get(log_curve_name, -1)
    if log_type < 0:
        return False
    _log_encode_kernel(img, log_type)
    return True


# =========================================================
# Warmup
# =========================================================

def warmup():
    """Pre-compile all Taichi kernels with dummy data."""
    logger.info("  Warming up Taichi kernels...")

    dummy_img = np.zeros((64, 64, 3), dtype=np.float32)
    dummy_coeffs = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    # 1. Gain
    apply_gain_inplace(dummy_img, 2.0)

    # 2. White Balance
    apply_white_balance_inplace(dummy_img, 1.1, 1.0, 1.2)

    # 3. Saturation / Contrast
    apply_saturation_contrast_inplace(dummy_img, 1.2, 1.1, 0.18, dummy_coeffs)

    # 4. Highlight / Shadow
    apply_highlight_shadow_inplace(dummy_img, -20.0, 20.0, dummy_coeffs)

    # 5. LUT (3D, 4x4x4)
    lut_size = 4
    lut_table = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float32)
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    apply_lut_inplace(dummy_img, lut_table, domain_min, domain_max)

    # 6. Matrix
    matrix = np.eye(3, dtype=np.float64)
    apply_matrix_inplace(dummy_img, matrix)

    # 7. sRGB / BT.709
    linear_to_srgb_inplace(dummy_img)
    bt709_to_srgb_inplace(dummy_img)

    # 8. Histogram
    flat_data = np.zeros(1000, dtype=np.float32)
    compute_histogram_channel(flat_data, 100, 0.0, 1.0)

    # 9. Waveform
    wf_in = np.zeros((100, 100), dtype=np.float32)
    wf_out = np.zeros((25, 100), dtype=np.float32)
    compute_waveform_channel(wf_in, wf_out, 100, 4)

    # 10. Perspective Warp
    persp_src = np.zeros((64, 64, 3), dtype=np.float32)
    persp_dst = np.zeros((64, 64, 3), dtype=np.float32)
    persp_matrix = np.eye(3, dtype=np.float64)
    perspective_warp_kernel(persp_src, persp_dst, persp_matrix)

    # 11. Richardson-Lucy Deconvolution
    rl_channel = np.ones((16, 16), dtype=np.float32) * 0.5
    rl_k1d = _gaussian_kernel_1d(1.0, size=3)
    rl_buf = np.empty_like(rl_channel)

    _convolve_1d_h_reflect(rl_channel, rl_k1d, rl_buf)
    _convolve_1d_v_reflect(rl_channel, rl_k1d, rl_buf)

    rl_blurred = np.empty_like(rl_channel)
    rl_ratio = np.empty_like(rl_channel)
    rl_correction = np.empty_like(rl_channel)
    rl_temp = np.empty_like(rl_channel)
    _rl_iteration_sep(rl_channel.copy(), rl_channel, rl_k1d,
                      rl_blurred, rl_ratio, rl_correction, rl_temp)

    rl_psf_2d = _gaussian_kernel(1.0, size=3)
    _convolve_2d_reflect(rl_channel, rl_psf_2d, rl_buf)

    # 12. GPU Downsample
    ds_src = np.zeros((64, 64, 3), dtype=np.float32)
    ds_dst = np.zeros((16, 16, 3), dtype=np.float32)
    downsample_gpu(ds_src, ds_dst, 64, 64, 16, 16)

    # 13. Float→Uint8
    u8_dst = np.zeros((16, 16, 3), dtype=np.uint8)
    float_to_uint8_gpu(ds_dst, u8_dst)

    # 14. Clip / Max
    clip_inplace(ds_dst)
    max_inplace(ds_dst, 1e-6)

    # 15. Geometry (rotate/flip)
    geom_src = np.zeros((16, 16, 3), dtype=np.float32)
    geom_dst = np.zeros((16, 16, 3), dtype=np.float32)
    _geometry_kernel(geom_src, geom_dst, 0, 0, 0)

    # 16. Crop
    crop_dst = np.zeros((8, 8, 3), dtype=np.float32)
    _crop_kernel(geom_src, crop_dst, 0, 0)

    # 17. Log encoding (warm up with S-Log3)
    _log_encode_kernel(geom_src, 0)

    # 18. Fused exposure+adjust+grade kernel warmup
    _fused_expose_adjust_srgb_kernel(
        geom_src, geom_src,
        1.0,  # gain
        1.0, 1.0, 1.0,  # WB r/g/b
        0.0, 0.0,  # highlight, shadow
        1.0, 1.0, 0.18,  # sat, con, pivot
        0.2126, 0.7152, 0.0722,  # luma
        1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,  # matrix (identity)
        1,  # apply sRGB
    )

    logger.info("  Taichi Warmup complete.")


if __name__ == '__main__':
    warmup()
