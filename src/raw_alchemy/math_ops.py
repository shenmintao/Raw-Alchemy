from numba import njit, prange, float32, float64, int64, void
import numpy as np
from loguru import logger

# =========================================================
# JIT Compiled Kernels (Parallel, FastMath, Cached)
# =========================================================

# Helper for parallel configuration
JIT_CONFIG = {
    'nopython': True,
    'parallel': True,
    'fastmath': True,
    'cache': True,
    'nogil': False
}

@njit(**JIT_CONFIG)
def apply_matrix_inplace(img, matrix):
    # Ensure matrix is contiguous for better access pattern
    m = np.ascontiguousarray(matrix)
    rows = img.shape[0]
    cols = img.shape[1]
    
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m10, m11, m12 = m[1, 0], m[1, 1], m[1, 2]
    m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]

    # Parallel loop over rows
    for r in prange(rows):
        for c in range(cols):
            r_val = img[r, c, 0]
            g_val = img[r, c, 1]
            b_val = img[r, c, 2]
            
            img[r, c, 0] = r_val * m00 + g_val * m01 + b_val * m02
            img[r, c, 1] = r_val * m10 + g_val * m11 + b_val * m12
            img[r, c, 2] = r_val * m20 + g_val * m21 + b_val * m22

@njit(**JIT_CONFIG)
def apply_lut_inplace(img, lut_table, domain_min, domain_max):
    rows = img.shape[0]
    cols = img.shape[1]
    
    size = lut_table.shape[0]
    size_minus_1 = size - 1
    size_float = float(size_minus_1)
    
    scale_r = size_minus_1 / (domain_max[0] - domain_min[0])
    scale_g = size_minus_1 / (domain_max[1] - domain_min[1])
    scale_b = size_minus_1 / (domain_max[2] - domain_min[2])
    
    min_r, min_g, min_b = domain_min[0], domain_min[1], domain_min[2]

    for r in prange(rows):
        for c in range(cols):
            in_r = img[r, c, 0]
            in_g = img[r, c, 1]
            in_b = img[r, c, 2]

            raw_idx_r = (in_r - min_r) * scale_r
            raw_idx_g = (in_g - min_g) * scale_g
            raw_idx_b = (in_b - min_b) * scale_b

            idx_r = min(max(raw_idx_r, 0.0), size_float)
            idx_g = min(max(raw_idx_g, 0.0), size_float)
            idx_b = min(max(raw_idx_b, 0.0), size_float)

            x0 = int(idx_r)
            y0 = int(idx_g)
            z0 = int(idx_b)

            x1 = x0 + 1
            if x0 == size_minus_1: x1 = x0
            
            y1 = y0 + 1
            if y0 == size_minus_1: y1 = y0
            
            z1 = z0 + 1
            if z0 == size_minus_1: z1 = z0

            dx = idx_r - x0
            dy = idx_g - y0
            dz = idx_b - z0

            # Tetrahedral interpolation
            if dx >= dy:
                if dy >= dz:
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
                    
                else: # dz > dx >= dy
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

            else: # dy > dx
                if dz >= dy:
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
                    
                else: # dy > dx > dz
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

@njit(**JIT_CONFIG)
def apply_saturation_contrast_inplace(img, saturation, contrast, pivot, luma_coeffs):
    rows = img.shape[0]
    cols = img.shape[1]
    cr, cg, cb = luma_coeffs[0], luma_coeffs[1], luma_coeffs[2]

    for r in prange(rows):
        for c in range(cols):
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

            # Clamp negative values
            if r_fin < 0.0: r_fin = 0.0
            if g_fin < 0.0: g_fin = 0.0
            if b_fin < 0.0: b_fin = 0.0

            img[r, c, 0] = r_fin
            img[r, c, 1] = g_fin
            img[r, c, 2] = b_fin

@njit(**JIT_CONFIG)
def apply_white_balance_inplace(img, r_gain, g_gain, b_gain):
    rows = img.shape[0]
    cols = img.shape[1]
    # Simple multiplication is very vectorizable
    for r in prange(rows):
        for c in range(cols):
            img[r, c, 0] *= r_gain
            img[r, c, 1] *= g_gain
            img[r, c, 2] *= b_gain

@njit(**JIT_CONFIG)
def apply_highlight_shadow_inplace(img, highlight, shadow, luma_coeffs):
    rows = img.shape[0]
    cols = img.shape[1]
    cr, cg, cb = luma_coeffs[0], luma_coeffs[1], luma_coeffs[2]

    for r in prange(rows):
        for c in range(cols):
            r_v = img[r, c, 0]
            g_v = img[r, c, 1]
            b_v = img[r, c, 2]
            
            lum = r_v * cr + g_v * cg + b_v * cb
            
            if shadow != 0.0:
                mask = (1.0 - lum)
                if mask > 0:
                    factor = 1.0 + shadow * (mask * mask * mask)
                    r_v *= factor
                    g_v *= factor
                    b_v *= factor
            
            if highlight != 0.0:
                mask = lum
                if mask < 0.0:
                    mask = 0.0
                elif mask > 1.0:
                    mask = 1.0
                t = 1.0 - mask
                factor = 1.0 + highlight * (1.0 - t * t * t)
                if factor < 0.0:
                    factor = 0.0
                r_v *= factor
                g_v *= factor
                b_v *= factor
            
            if r_v < 0.0: r_v = 0.0
            if g_v < 0.0: g_v = 0.0
            if b_v < 0.0: b_v = 0.0

            img[r, c, 0] = r_v
            img[r, c, 1] = g_v
            img[r, c, 2] = b_v

@njit(**JIT_CONFIG)
def apply_gain_inplace(img, gain):
    rows = img.shape[0]
    cols = img.shape[1]
    for r in prange(rows):
        for c in range(cols):
            img[r, c, 0] *= gain
            img[r, c, 1] *= gain
            img[r, c, 2] *= gain

@njit(**JIT_CONFIG)
def linear_to_srgb_inplace(img):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for r in prange(rows):
        for c in range(cols):
            for ch in range(3):
                linear = img[r, c, ch]
                
                if linear <= 0.0031308:
                    result = linear * 12.92
                else:
                    # Pow can be expensive, fastmath helps
                    result = 1.055 * (linear ** (1.0 / 2.4)) - 0.055
                
                img[r, c, ch] = result

@njit(**JIT_CONFIG)
def bt709_to_srgb_inplace(img):
    rows = img.shape[0]
    cols = img.shape[1]
    
    for r in prange(rows):
        for c in range(cols):
            for ch in range(3):
                val = img[r, c, ch]
                
                if val < 0.081:
                    linear = val / 4.5
                else:
                    linear = ((val + 0.099) / 1.099) ** (1.0 / 0.45)
                
                if linear <= 0.0031308:
                    result = linear * 12.92
                else:
                    result = 1.055 * (linear ** (1.0 / 2.4)) - 0.055
                
                img[r, c, ch] = result

@njit(**JIT_CONFIG)
def compute_histogram_channel(data, bins, min_val, max_val):
    hist = np.zeros(bins, dtype=np.int64)
    # Avoid div by zero
    if max_val <= min_val:
        return hist
        
    bin_width = (max_val - min_val) / bins
    n = data.shape[0]
    
    # Simple histogram logic. Parallelism for histogram reduction is complex in Numba
    # without race conditions. Given specific channel subsampling, this is fast enough.
    # We can stick to serial for correct reduction or use explicit atomic add if needed.
    # But for a subsampled array, serial is usually fine. Let's keep it simple JIT.
    for i in range(n):
        val = data[i]
        if min_val <= val <= max_val:
            bin_idx = int((val - min_val) / bin_width)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist[bin_idx] += 1
    
    return hist

@njit(**JIT_CONFIG)
def compute_waveform_channel(channel_data, waveform_out, bins, sample_rate):
    h = channel_data.shape[0]
    w = channel_data.shape[1]
    sampled_width = w // sample_rate
    if sampled_width == 0:
        sampled_width = 1
    
    ire_min = -4.0
    ire_max = 109.0
    ire_range = ire_max - ire_min
    bins_minus_1 = bins - 1
    
    # Parallel over columns (independent output memory locations)
    for i in prange(sampled_width):
        col_idx = i * sample_rate
        # if col_idx >= w: break  <-- Removed this check as it breaks parallelism and is redundant (loop bounds cover it)

        
        for row in range(h):
            val = channel_data[row, col_idx]
            
            ire_value = val * 100.0
            normalized_pos = (ire_value - ire_min) / ire_range
            bin_idx = int(normalized_pos * bins_minus_1)
            
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= bins:
                bin_idx = bins - 1
            
            waveform_out[i, bin_idx] += 1.0

@njit(**JIT_CONFIG)
def perspective_warp_kernel(src, dst, M_inv):
    """
    Perspective warp using inverse mapping + bilinear interpolation.
    src: Source image HxWx3
    dst: Destination image (pre-allocated, same size or different)
    M_inv: 3x3 inverse perspective matrix
    """
    dst_h = dst.shape[0]
    dst_w = dst.shape[1]
    src_h = src.shape[0]
    src_w = src.shape[1]
    
    # Extract matrix elements for speed
    m00, m01, m02 = M_inv[0, 0], M_inv[0, 1], M_inv[0, 2]
    m10, m11, m12 = M_inv[1, 0], M_inv[1, 1], M_inv[1, 2]
    m20, m21, m22 = M_inv[2, 0], M_inv[2, 1], M_inv[2, 2]
    
    src_h_1 = src_h - 1.0
    src_w_1 = src_w - 1.0
    
    for y in prange(dst_h):
        for x in range(dst_w):
            # Inverse perspective transform to find source coordinates
            denom = m20 * x + m21 * y + m22
            if denom == 0.0:
                denom = 1e-10  # Avoid division by zero
            
            src_x = (m00 * x + m01 * y + m02) / denom
            src_y = (m10 * x + m11 * y + m12) / denom
            
            # Bilinear interpolation
            if src_x < 0.0 or src_x > src_w_1 or src_y < 0.0 or src_y > src_h_1:
                # Out of bounds - use edge reflection
                src_x = max(0.0, min(src_x, src_w_1))
                src_y = max(0.0, min(src_y, src_h_1))
            
            x0 = int(src_x)
            y0 = int(src_y)
            x1 = min(x0 + 1, int(src_w_1))
            y1 = min(y0 + 1, int(src_h_1))
            
            dx = src_x - x0
            dy = src_y - y0
            
            w00 = (1.0 - dx) * (1.0 - dy)
            w01 = dx * (1.0 - dy)
            w10 = (1.0 - dx) * dy
            w11 = dx * dy
            
            for ch in range(3):
                val = (src[y0, x0, ch] * w00 +
                       src[y0, x1, ch] * w01 +
                       src[y1, x0, ch] * w10 +
                       src[y1, x1, ch] * w11)
                dst[y, x, ch] = val


def compute_perspective_matrix(src_corners, dst_width, dst_height):
    """
    Compute 3x3 perspective transform matrix from 4 source corners to destination rectangle.
    src_corners: 4 points in normalized coords [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                 Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    Returns: 3x3 transformation matrix M such that dst = M @ src
    """
    # Source points in pixel coordinates
    src = np.array([
        [src_corners[0][0] * dst_width, src_corners[0][1] * dst_height],  # TL
        [src_corners[1][0] * dst_width, src_corners[1][1] * dst_height],  # TR
        [src_corners[2][0] * dst_width, src_corners[2][1] * dst_height],  # BR
        [src_corners[3][0] * dst_width, src_corners[3][1] * dst_height],  # BL
    ], dtype=np.float64)
    
    # Destination is full image rectangle
    dst = np.array([
        [0.0, 0.0],                     # TL
        [dst_width - 1, 0.0],           # TR
        [dst_width - 1, dst_height - 1], # BR
        [0.0, dst_height - 1],          # BL
    ], dtype=np.float64)
    
    # Compute perspective transform using direct linear transform (DLT)
    # Solve for H in: dst = H @ src (homogeneous coordinates)
    # Build matrix A for Ah = 0
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    
    for i in range(4):
        sx, sy = src[i, 0], src[i, 1]
        dx, dy = dst[i, 0], dst[i, 1]
        
        A[i*2, 0] = sx
        A[i*2, 1] = sy
        A[i*2, 2] = 1.0
        A[i*2, 6] = -dx * sx
        A[i*2, 7] = -dx * sy
        b[i*2] = dx
        
        A[i*2+1, 3] = sx
        A[i*2+1, 4] = sy
        A[i*2+1, 5] = 1.0
        A[i*2+1, 6] = -dy * sx
        A[i*2+1, 7] = -dy * sy
        b[i*2+1] = dy
    
    # Solve linear system
    try:
        h = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Singular matrix - return identity
        return np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)
    
    # Inverse matrix for inverse mapping
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.eye(3, dtype=np.float64)
    
    return H, H_inv


# =========================================================
# Richardson-Lucy Deconvolution Sharpening
# =========================================================

def _gaussian_kernel(sigma: float, size: int = None) -> np.ndarray:
    """Create a 2D Gaussian kernel for PSF."""
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= kernel_2d.sum()
    
    return kernel_2d.astype(np.float32)


@njit(**JIT_CONFIG)
def _convolve_2d_reflect(channel, kernel, output):
    """JIT-compiled 2D convolution with reflect boundary."""
    h, w = channel.shape
    kh, kw = kernel.shape
    kh2, kw2 = kh // 2, kw // 2
    
    for y in prange(h):
        for x in range(w):
            val = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    # Reflect boundary
                    sy = y + ky - kh2
                    sx = x + kx - kw2
                    
                    # Reflect at boundaries
                    if sy < 0:
                        sy = -sy
                    elif sy >= h:
                        sy = 2 * h - sy - 2
                    
                    if sx < 0:
                        sx = -sx
                    elif sx >= w:
                        sx = 2 * w - sx - 2
                    
                    # Clamp to valid range
                    if sy < 0: sy = 0
                    if sy >= h: sy = h - 1
                    if sx < 0: sx = 0
                    if sx >= w: sx = w - 1
                    
                    val += channel[sy, sx] * kernel[ky, kx]
            
            output[y, x] = val


@njit(**JIT_CONFIG)
def _richardson_lucy_iteration(estimate, channel, psf, psf_mirror, blurred, ratio, correction):
    """Single RL iteration with JIT acceleration."""
    h, w = channel.shape
    eps = 1e-8
    
    # Forward convolution: blurred = estimate * psf
    _convolve_2d_reflect(estimate, psf, blurred)
    
    # Compute ratio and clip blurred to avoid division by zero
    for y in prange(h):
        for x in range(w):
            b = blurred[y, x]
            if b < eps:
                b = eps
            ratio[y, x] = channel[y, x] / b
    
    # Backward convolution: correction = ratio * psf_mirror
    _convolve_2d_reflect(ratio, psf_mirror, correction)
    
    # Update estimate
    for y in prange(h):
        for x in range(w):
            val = estimate[y, x] * correction[y, x]
            # Clip to prevent extreme values
            if val < 0.0:
                val = 0.0
            elif val > 2.0:
                val = 2.0
            estimate[y, x] = val


def richardson_lucy_channel(
    channel: np.ndarray,
    psf: np.ndarray,
    iterations: int = 10,
    clip: bool = True
) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution to a single channel.
    """
    h, w = channel.shape
    estimate = channel.copy().astype(np.float32)
    psf = np.ascontiguousarray(psf.astype(np.float32))
    psf_mirror = np.ascontiguousarray(psf[::-1, ::-1].copy())
    
    # Pre-allocate work buffers
    blurred = np.empty_like(estimate)
    ratio = np.empty_like(estimate)
    correction = np.empty_like(estimate)
    
    for _ in range(iterations):
        _richardson_lucy_iteration(estimate, channel, psf, psf_mirror, blurred, ratio, correction)
    
    if clip:
        estimate = np.clip(estimate, 0, 1)
    
    return estimate


def richardson_lucy(
    image: np.ndarray,
    sigma: float = 1.0,
    iterations: int = 10,
    strength: float = 1.0
) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution sharpening to an RGB image.
    """
    if strength <= 0 or iterations <= 0:
        return image
    
    logger.debug(f"RL sharpening: sigma={sigma}, iterations={iterations}, strength={strength}")
    
    psf = _gaussian_kernel(sigma)
    h, w, c = image.shape
    result = np.empty_like(image)
    
    for i in range(c):
        result[:, :, i] = richardson_lucy_channel(
            np.ascontiguousarray(image[:, :, i].astype(np.float32)),
            psf,
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
        strength: Sharpening strength (0 = off, 1 = max)
                  Maps to 1-10 RL iterations
        sigma: PSF sigma (default 1.0, same as NIND)
    
    Returns:
        Sharpened image
    """
    if strength <= 0:
        return image
    
    iterations = max(1, int(strength * 10))
    return richardson_lucy(image, sigma=sigma, iterations=iterations, strength=1.0)


def warmup():
    """Compiles all JIT functions with dummy data"""
    logger.info("  ♨️ Warming up JIT kernels...")
    
    # Small dummy image: 64x64
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
    
    # 5. LUT
    # 3D LUT: 4x4x4
    lut_size = 4
    lut_table = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float32)
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    # Prevent div by zero in kernel
    apply_lut_inplace(dummy_img, lut_table, domain_min, domain_max)
    
    # 6. Matrix
    matrix = np.eye(3, dtype=np.float64)
    apply_matrix_inplace(dummy_img, matrix)
    
    # 7. sRGB
    linear_to_srgb_inplace(dummy_img)
    bt709_to_srgb_inplace(dummy_img)

    # 8. Histogram
    flat_data = np.zeros(1000, dtype=np.float32)
    compute_histogram_channel(flat_data, 100, 0.0, 1.0)
    
    # 9. Waveform
    # Input HxW, Output W/sample x Bins
    wf_in = np.zeros((100, 100), dtype=np.float32)
    wf_out = np.zeros((25, 100), dtype=np.float32) # width=100, sample=4 -> 25
    compute_waveform_channel(wf_in, wf_out, 100, 4)
    
    # 10. Perspective Warp
    persp_src = np.zeros((64, 64, 3), dtype=np.float32)
    persp_dst = np.zeros((64, 64, 3), dtype=np.float32)
    persp_matrix = np.eye(3, dtype=np.float64)
    perspective_warp_kernel(persp_src, persp_dst, persp_matrix)
    
    logger.info("✅ JIT Warmup complete.")

if __name__ == '__main__':
    warmup()
