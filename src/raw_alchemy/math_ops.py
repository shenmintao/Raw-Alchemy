"""Pixel math for the interactive pipeline — numpy/cv2 implementations.

These functions were historically Taichi GPU kernels. The interactive
pipeline no longer uses the Taichi runtime: every active op is implemented
with numpy (SIMD) and OpenCV, while keeping the exact public signatures and
semantics (including in-place behaviour) the pipeline executor and worker
rely on. ``init_taichi``/``warmup`` remain as no-op compatibility stubs.

The only remaining Taichi consumers in the code base are the legacy
demosaic modules (``demosaic.py`` / ``xtrans_demosaic.py``), which import
the ONNX demosaic modules (onnx/rcd_demosaic, onnx/xtrans_demosaic)
lands.
"""
from functools import lru_cache

import colour
import cv2
import numpy as np
from loguru import logger

from raw_alchemy import config
from raw_alchemy.colorspace_matrices import working_rgb_to_xyz_d65

# =========================================================
# Compatibility stubs (Taichi runtime removed)
# =========================================================

_backend_logged = False

# Keep whole-frame elementwise/matrix temporaries bounded. One million RGB
# float32 pixels is about 12 MiB and stays friendly to CPU caches.
_PIXEL_CHUNK = 1_000_000


def init_taichi(arch=None):
    """Compatibility no-op. The interactive pipeline is numpy/cv2 now.

    Legacy callers (worker startup, export entry points, xtrans_demosaic)
    still invoke this; nothing needs to be initialized anymore. The legacy
    Taichi demosaic modules must initialize taichi themselves if used.
    """
    global _backend_logged
    if not _backend_logged:
        _backend_logged = True
        logger.info("  math_ops: numpy/cv2 backend active (Taichi runtime not used).")


def warmup():
    """Compatibility no-op. numpy/cv2 ops need no kernel pre-compilation."""
    logger.debug("  math_ops warmup: nothing to pre-compile (numpy/cv2 backend).")


# =========================================================
# Color matrix
# =========================================================

def apply_matrix_inplace(img, matrix):
    """Apply a 3x3 matrix in-place with a bounded scratch buffer.

    ``flat @ matrix.T`` creates another complete frame.  On a 61MP image that
    is roughly 700 MiB for a single colour transform, so process cache-sized
    chunks and reuse one small output slab instead.
    """
    m = np.ascontiguousarray(matrix, dtype=np.float32)
    flat = img.reshape(-1, img.shape[-1])
    if flat.shape[1] != 3:
        raise ValueError(f"expected RGB image, got shape {img.shape}")
    chunk = min(flat.shape[0], _PIXEL_CHUNK)
    scratch = np.empty((chunk, 3), dtype=np.float32)
    for start in range(0, flat.shape[0], chunk):
        stop = min(start + chunk, flat.shape[0])
        view = flat[start:stop]
        np.matmul(view, m.T, out=scratch[: stop - start])
        np.copyto(view, scratch[: stop - start])


D65_XY = tuple(float(v) for v in colour.CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["D65"])
D65_CCT = 6504.0
TEMP_MIRED_SHIFT_PER_STEP = 1.0
TINT_DUV_SHIFT_PER_STEP = -0.0005


@lru_cache(maxsize=128)
def _working_space_adaptation_matrix_cached(
    source_x: float,
    source_y: float,
    target_x: float,
    target_y: float,
    working_space: str,
    transform: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    from colour.adaptation import matrix_chromatic_adaptation_VonKries

    source_xyz = colour.xy_to_XYZ((source_x, source_y))
    target_xyz = colour.xy_to_XYZ((target_x, target_y))
    cat_xyz = matrix_chromatic_adaptation_VonKries(
        source_xyz,
        target_xyz,
        transform=transform,
    )
    rgb_to_xyz = working_rgb_to_xyz_d65(working_space)
    xyz_to_rgb = np.linalg.inv(rgb_to_xyz)
    matrix = xyz_to_rgb @ cat_xyz @ rgb_to_xyz
    return tuple(tuple(float(v) for v in row) for row in matrix)


def working_space_adaptation_matrix(
    source_xy,
    target_xy,
    working_space: str | None = None,
    transform: str = "Bradford",
) -> np.ndarray:
    """Return a working-RGB chromatic adaptation matrix."""
    working_space = working_space or config.WORKING_SPACE
    matrix = _working_space_adaptation_matrix_cached(
        round(float(source_xy[0]), 12),
        round(float(source_xy[1]), 12),
        round(float(target_xy[0]), 12),
        round(float(target_xy[1]), 12),
        working_space,
        transform,
    )
    return np.asarray(matrix, dtype=np.float64)


def _white_balance_target_xy(wb_temp: float, wb_tint: float) -> tuple[float, float]:
    base_mired = 1_000_000.0 / D65_CCT
    target_mired = np.clip(
        base_mired + float(wb_temp) * TEMP_MIRED_SHIFT_PER_STEP,
        1_000_000.0 / 25_000.0,
        1_000_000.0 / 1_500.0,
    )
    cct = 1_000_000.0 / target_mired
    duv = float(wb_tint) * TINT_DUV_SHIFT_PER_STEP
    uv = colour.CCT_to_uv(np.array([cct, duv], dtype=np.float64), method="Ohno 2013")
    xy = colour.UCS_uv_to_xy(uv)
    return float(xy[0]), float(xy[1])


@lru_cache(maxsize=256)
def _white_balance_matrix_cached(
    wb_temp: float,
    wb_tint: float,
    working_space: str,
    transform: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if float(wb_temp) == 0.0 and float(wb_tint) == 0.0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    target_xy = _white_balance_target_xy(wb_temp, wb_tint)
    matrix = working_space_adaptation_matrix(D65_XY, target_xy, working_space, transform)
    return tuple(tuple(float(v) for v in row) for row in matrix)


def white_balance_matrix(
    wb_temp: float,
    wb_tint: float,
    working_space: str | None = None,
    transform: str = "Bradford",
) -> np.ndarray:
    """Map UI temp/tint controls to a cached CAT matrix in working RGB."""
    working_space = working_space or config.WORKING_SPACE
    matrix = _white_balance_matrix_cached(
        round(float(wb_temp), 6),
        round(float(wb_tint), 6),
        working_space,
        transform,
    )
    return np.asarray(matrix, dtype=np.float64)


# =========================================================
# 3D LUT (tetrahedral interpolation)
# =========================================================

def apply_lut_inplace(img, lut_table, domain_min, domain_max):
    """Apply 3D LUT with tetrahedral interpolation in-place."""
    if lut_table.dtype != np.float32:
        lut_table = lut_table.astype(np.float32)

    size = lut_table.shape[0]
    n = size - 1

    scale = np.array(
        [n / (float(domain_max[i]) - float(domain_min[i])) for i in range(3)],
        dtype=np.float32,
    )
    dmin = np.array([float(v) for v in domain_min[:3]], dtype=np.float32)

    flat = img.reshape(-1, 3)
    # Tetrahedral indexing creates several integer/mask/vertex temporaries;
    # use a smaller block than simple elementwise ops to hold peak RSS down.
    lut_chunk = 250_000
    T = lut_table
    for start in range(0, flat.shape[0], lut_chunk):
        block = flat[start:start + lut_chunk]
        idx = (block - dmin) * scale
        np.clip(idx, 0.0, np.float32(n), out=idx)

        i0 = idx.astype(np.int32)  # floor: idx >= 0
        d = idx - i0
        i1 = np.minimum(i0 + 1, n)

        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

        cxy = dx >= dy
        cyz = dy >= dz
        cxz = dx >= dz
        czy = dz >= dy
        czx = dz >= dx

        out = np.empty_like(block)

        # (mask, weights per vertex, vertex index triples) for the 6
        # tetrahedra — branch structure identical to the retired kernel.
        cases = (
            (cxy & cyz,
             (lambda m: 1.0 - dx[m], lambda m: dx[m] - dy[m],
              lambda m: dy[m] - dz[m], lambda m: dz[m]),
             ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1))),
            (cxy & ~cyz & cxz,
             (lambda m: 1.0 - dx[m], lambda m: dx[m] - dz[m],
              lambda m: dz[m] - dy[m], lambda m: dy[m]),
             ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x1, y1, z1))),
            (cxy & ~cyz & ~cxz,
             (lambda m: 1.0 - dz[m], lambda m: dz[m] - dx[m],
              lambda m: dx[m] - dy[m], lambda m: dy[m]),
             ((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1))),
            (~cxy & czy,
             (lambda m: 1.0 - dz[m], lambda m: dz[m] - dy[m],
              lambda m: dy[m] - dx[m], lambda m: dx[m]),
             ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x1, y1, z1))),
            (~cxy & ~czy & czx,
             (lambda m: 1.0 - dy[m], lambda m: dy[m] - dz[m],
              lambda m: dz[m] - dx[m], lambda m: dx[m]),
             ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1))),
            (~cxy & ~czy & ~czx,
             (lambda m: 1.0 - dy[m], lambda m: dy[m] - dx[m],
              lambda m: dx[m] - dz[m], lambda m: dz[m]),
             ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y1, z1))),
        )

        for mask, weights, verts in cases:
            if not np.any(mask):
                continue
            acc = None
            for weight_fn, (vx, vy, vz) in zip(weights, verts):
                weight = weight_fn(mask).astype(np.float32)[..., None]
                term = weight * T[vx[mask], vy[mask], vz[mask]]
                acc = term if acc is None else acc + term
            out[mask] = acc

        block[...] = out


# =========================================================
# Per-pixel color adjustments (in-place)
# =========================================================

def apply_saturation_contrast_inplace(img, saturation, contrast, pivot, luma_coeffs):
    """Apply saturation and contrast adjustment in-place."""
    saturation = np.float32(saturation)
    contrast = np.float32(contrast)
    pivot = np.float32(pivot)
    cr, cg, cb = (np.float32(luma_coeffs[i]) for i in range(3))

    flat = img.reshape(-1, 3)
    for start in range(0, flat.shape[0], _PIXEL_CHUNK):
        block = flat[start:start + _PIXEL_CHUNK]
        lum = block[:, 0] * cr + block[:, 1] * cg + block[:, 2] * cb
        block -= lum[:, None]
        block *= saturation
        block += lum[:, None]
        block -= pivot
        block *= contrast
        block += pivot
        np.maximum(block, np.float32(0.0), out=block)


def apply_white_balance_inplace(img, r_gain, g_gain, b_gain):
    """Apply per-channel white balance gains in-place."""
    img[..., 0] *= np.float32(r_gain)
    img[..., 1] *= np.float32(g_gain)
    img[..., 2] *= np.float32(b_gain)


def apply_highlight_shadow_inplace(img, highlight, shadow, luma_coeffs):
    """Apply highlight/shadow tone mapping in-place."""
    highlight = float(highlight)
    shadow = float(shadow)
    cr, cg, cb = (np.float32(luma_coeffs[i]) for i in range(3))

    flat = img.reshape(-1, 3)
    for start in range(0, flat.shape[0], _PIXEL_CHUNK):
        block = flat[start:start + _PIXEL_CHUNK]
        lum = block[:, 0] * cr + block[:, 1] * cg + block[:, 2] * cb

        if shadow != 0.0:
            mask = np.float32(1.0) - lum
            factor = np.where(
                mask > 0.0,
                np.float32(1.0) + np.float32(shadow) * mask * mask * mask,
                np.float32(1.0),
            )
            block *= factor[:, None]

        if highlight != 0.0:
            t = np.float32(1.0) - np.clip(
                lum, np.float32(0.0), np.float32(1.0)
            )
            factor = np.float32(1.0) + np.float32(highlight) * (
                np.float32(1.0) - t * t * t
            )
            np.maximum(factor, np.float32(0.0), out=factor)
            block *= factor[:, None]

        np.maximum(block, np.float32(0.0), out=block)


def apply_gain_inplace(img, gain):
    """Apply uniform exposure gain in-place."""
    img *= np.float32(gain)


# =========================================================
# Transfer functions (in-place)
# =========================================================

def linear_to_srgb_inplace(img):
    """Convert linear RGB to sRGB gamma in-place."""
    flat = img.reshape(-1)
    chunk = _PIXEL_CHUNK * 3
    for start in range(0, flat.size, chunk):
        block = flat[start:start + chunk]
        mask = block > np.float32(0.0031308)
        high = np.float32(1.055) * np.power(
            block[mask], np.float32(1.0 / 2.4)
        ) - np.float32(0.055)
        block *= np.float32(12.92)
        block[mask] = high


def srgb_to_linear_inplace(img):
    """Convert sRGB gamma to linear RGB in-place (inverse of linear_to_srgb)."""
    flat = img.reshape(-1)
    chunk = _PIXEL_CHUNK * 3
    for start in range(0, flat.size, chunk):
        block = flat[start:start + chunk]
        mask = block > np.float32(0.04045)
        high = np.power(
            (block[mask] + np.float32(0.055)) / np.float32(1.055),
            np.float32(2.4),
        )
        block /= np.float32(12.92)
        block[mask] = high


def pq_encode_inplace(img, peak_nits=1000.0, mastering_nits=10000.0):
    """Encode linear BT.2020 values with ST 2084 using bounded scratch.

    ``colour.cctf_encoding`` returns a complete float64 frame for this path.
    The explicit formula keeps the float32 pipeline buffer in place and uses
    two reusable chunks regardless of native image dimensions.
    """
    peak = float(peak_nits)
    mastering = float(mastering_nits)
    if peak < 0.0 or mastering <= 0.0:
        raise ValueError("invalid PQ luminance range")

    m1 = np.float32(2610.0 / 16384.0)
    m2 = np.float32(2523.0 / 32.0)
    c1 = np.float32(3424.0 / 4096.0)
    c2 = np.float32(2413.0 / 128.0)
    c3 = np.float32(2392.0 / 128.0)
    scale = np.float32(peak / mastering)

    flat = img.reshape(-1)
    chunk = min(flat.size, _PIXEL_CHUNK * 3)
    power = np.empty(chunk, dtype=np.float32)
    denominator = np.empty(chunk, dtype=np.float32)
    for start in range(0, flat.size, chunk):
        block = flat[start:start + chunk]
        size = block.size
        np.maximum(block, np.float32(0.0), out=block)
        block *= scale
        np.minimum(block, np.float32(1.0), out=block)
        np.power(block, m1, out=power[:size])
        np.multiply(power[:size], c3, out=denominator[:size])
        denominator[:size] += np.float32(1.0)
        np.multiply(power[:size], c2, out=block)
        block += c1
        block /= denominator[:size]
        np.power(block, m2, out=block)
        np.clip(block, np.float32(0.0), np.float32(1.0), out=block)


# =========================================================
# Perspective Warp
# =========================================================

def perspective_warp_kernel(src, dst, M_inv):
    """Perspective warp using inverse mapping + bilinear interpolation (cv2).

    ``M_inv`` maps destination coordinates to source coordinates, matching
    the retired Taichi kernel; edge behaviour is replicate (the old kernel
    clamped sample coordinates to the image bounds).
    """
    M = np.ascontiguousarray(M_inv.astype(np.float64))
    h, w = dst.shape[:2]
    cv2.warpPerspective(
        src,
        M,
        (w, h),
        dst=dst,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


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


def _sep_gaussian_blur(src, kernel_1d, dst=None):
    """Separable convolution with REFLECT_101 borders (matches the old kernel)."""
    return cv2.sepFilter2D(
        src, -1, kernel_1d, kernel_1d, dst=dst, borderType=cv2.BORDER_REFLECT_101
    )


def richardson_lucy_channel(
    channel: np.ndarray,
    kernel_1d: np.ndarray,
    iterations: int = 10,
    clip: bool = True
) -> np.ndarray:
    """Apply Richardson-Lucy deconvolution to a single channel using separable Gaussian."""
    channel = np.ascontiguousarray(channel, dtype=np.float32)
    kernel_1d = np.ascontiguousarray(kernel_1d, dtype=np.float32)
    estimate = channel.copy()

    for _ in range(iterations):
        blurred = _sep_gaussian_blur(estimate, kernel_1d)
        np.maximum(blurred, np.float32(1e-8), out=blurred)
        ratio = channel / blurred
        correction = _sep_gaussian_blur(ratio, kernel_1d)
        estimate *= correction
        np.clip(estimate, 0.0, 2.0, out=estimate)

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


def sharpen_gpu(gpu_image, strength: float = 0.5, sigma: float = 1.0):
    """
    RL sharpening on a GpuImage, in-place (numpy/cv2 separable Gaussian).

    Iteration structure matches the retired Taichi implementation:
    per-channel 2D work buffers (estimate / blurred-correction / ratio /
    blend scratch) come from the shared buffer pool so they are recycled
    across ops/images and released with the pool (T7.2 semantics).
    """
    if strength <= 0 or not gpu_image.valid:
        return

    # Local import: gpu_buffer imports config only, no cycle back here.
    from raw_alchemy.gpu_buffer import acquire_ndarray, release_ndarray

    iterations = max(1, int(strength * 10))
    h, w = gpu_image.height, gpu_image.width
    shape2d = (h, w)

    kernel_1d = _gaussian_kernel_1d(sigma)

    # Acquire three 2D work buffers shared across channels. ``buf_b`` is the
    # iteration ratio and is reused as the final blend scratch; retaining a
    # fourth full-resolution plane cost ~244 MiB on a 61MP frame.
    estimate = acquire_ndarray(np.float32, shape2d)
    buf_a = acquire_ndarray(np.float32, shape2d)  # blurred, then correction
    buf_b = acquire_ndarray(np.float32, shape2d)  # ratio

    try:
        arr = gpu_image.arr
        for ch in range(3):
            channel = arr[:, :, ch]  # observed data (strided view)
            np.copyto(estimate, channel)

            for _ in range(iterations):
                # Forward: blurred = estimate * PSF (separable)
                _sep_gaussian_blur(estimate, kernel_1d, dst=buf_a)
                np.maximum(buf_a, np.float32(1e-8), out=buf_a)
                # Ratio: observed / blurred
                np.divide(channel, buf_a, out=buf_b)
                # Backward: correction = ratio * PSF_mirror (symmetric Gaussian)
                _sep_gaussian_blur(buf_b, kernel_1d, dst=buf_a)
                # Update estimate (clamped to [0, 2])
                estimate *= buf_a
                np.clip(estimate, 0.0, 2.0, out=estimate)

            # Blend with original if strength < 1, then insert back
            if strength < 1.0:
                estimate *= np.float32(strength)
                np.multiply(channel, np.float32(1.0 - strength), out=buf_b)
                estimate += buf_b
            arr[:, :, ch] = estimate
    finally:
        release_ndarray(estimate, np.float32, shape2d)
        release_ndarray(buf_a, np.float32, shape2d)
        release_ndarray(buf_b, np.float32, shape2d)

    logger.debug(f"RL sharpen: {iterations} iters, sigma={sigma}, strength={strength}")


# =========================================================
# Output conversion (float -> uint8, optional resize)
# =========================================================

def float_to_uint8_gpu(src, dst):
    """float32 [0,1] -> uint8 [0,255] with bounded temporary memory."""
    src_flat = np.asarray(src).reshape(-1)
    dst_flat = np.asarray(dst).reshape(-1)
    if src_flat.size != dst_flat.size:
        raise ValueError(f"source/destination size mismatch: {src.shape} -> {dst.shape}")
    chunk = min(src_flat.size, _PIXEL_CHUNK * 3)
    scratch = np.empty(chunk, dtype=np.float32)
    for start in range(0, src_flat.size, chunk):
        stop = min(start + chunk, src_flat.size)
        work = scratch[: stop - start]
        np.clip(src_flat[start:stop], np.float32(0.0), np.float32(1.0), out=work)
        work *= np.float32(255.0)
        work += np.float32(0.5)
        np.copyto(dst_flat[start:stop], work, casting="unsafe")


def resize_float_to_uint8_gpu(src, dst):
    """Bilinear resize + float32 [0,1] -> uint8 [0,255] conversion into dst."""
    dst_h, dst_w = dst.shape[:2]
    resized = cv2.resize(
        np.ascontiguousarray(src), (dst_w, dst_h), interpolation=cv2.INTER_LINEAR
    )
    float_to_uint8_gpu(resized.reshape(dst.shape[0], dst.shape[1], -1), dst)


def clip_inplace(img):
    """Clip image values to [0, 1] in-place."""
    np.clip(img, 0.0, 1.0, out=img)


def max_inplace(img, min_val):
    """np.maximum equivalent, in-place (clamp values to >= min_val)."""
    np.maximum(img, np.float32(min_val), out=img)


# =========================================================
# Highlight reconstruction reference average (numpy/cv2)
# =========================================================

def compute_hl_refavg(raw_data, color_map, wb_gains, raw_clips):
    """Opposing-channel cube-root reference average for highlight reconstruction.

    Pixel-exact port of the retired Taichi kernel: 3x3 clamped-neighbourhood
    per-color means == replicate-border box sums of (value*mask, mask).
    Returns ``(refavg float32, clipped bool)``.
    """
    color_map = np.asarray(color_map)
    h, w = raw_data.shape
    # Only two full float planes are retained: sum of the three cube-root
    # neighbourhood means and the value belonging to this pixel's CFA colour.
    # The previous 3-plane cbrt + 3-plane opposing stack exceeded 1GB at 61MP.
    cbrt_sum = np.zeros((h, w), dtype=np.float32)
    own_cbrt = np.empty((h, w), dtype=np.float32)
    stripe_rows = max(64, min(1024, _PIXEL_CHUNK // max(w, 1)))

    for c in range(3):
        gain = np.float32(wb_gains[c])
        for y in range(0, h, stripe_rows):
            y2 = min(y + stripe_rows, h)
            halo0 = max(0, y - 1)
            halo1 = min(h, y2 + 1)
            raw_strip = raw_data[halo0:halo1]
            mask = color_map[halo0:halo1] == c
            values = np.maximum(raw_strip, np.float32(0.0))
            values *= mask
            counts = mask.astype(np.float32)
            sums = cv2.boxFilter(
                values, -1, (3, 3), normalize=False,
                borderType=cv2.BORDER_REPLICATE,
            )
            cnts = cv2.boxFilter(
                counts, -1, (3, 3), normalize=False,
                borderType=cv2.BORDER_REPLICATE,
            )
            np.divide(sums, cnts, out=sums, where=cnts > 0)
            sums[cnts <= 0] = 0.0
            sums *= gain
            np.cbrt(sums, out=sums)

            src0 = y - halo0
            src1 = src0 + (y2 - y)
            core = sums[src0:src1]
            cbrt_sum[y:y2] += core
            core_mask = color_map[y:y2] == c
            own = own_cbrt[y:y2]
            own[core_mask] = core[core_mask]

    cbrt_sum -= own_cbrt
    del own_cbrt
    cbrt_sum *= np.float32(0.5)
    np.power(cbrt_sum, np.float32(3.0), out=cbrt_sum)
    for c in range(3):
        gain = float(wb_gains[c])
        if gain > 1e-6:
            cbrt_sum[color_map == c] /= np.float32(gain)

    clipped = np.empty((h, w), dtype=bool)
    for c in range(3):
        mask = color_map == c
        clipped[mask] = raw_data[mask] >= np.float32(raw_clips[c])
    return cbrt_sum, clipped


# =========================================================
# Geometry (Rotate / Flip) and Crop
# =========================================================

def geometry_gpu_shape(src_h, src_w, rotation):
    """Compute output shape after rotation."""
    rotation = rotation % 360
    if rotation == 90 or rotation == 270:
        return (src_w, src_h, 3)
    return (src_h, src_w, 3)


def apply_geometry_gpu(src_buf, dst_buf, rotation=0, flip_h=False, flip_v=False):
    """
    Geometry transform (rotate + flip). src_buf/dst_buf are GpuImage.
    dst_buf will be re-allocated if needed.
    rotation: 0, 90, 180, 270 (clockwise degrees)
    """
    if rotation == 0 and not flip_h and not flip_v:
        dst_buf.copy_from(src_buf)
        return

    rotation = rotation % 360
    arr = src_buf.arr
    if rotation == 90:      # clockwise
        res = np.rot90(arr, k=-1)
    elif rotation == 180:
        res = np.rot90(arr, k=2)
    elif rotation == 270:   # 90 CCW
        res = np.rot90(arr, k=1)
    else:
        res = arr

    if flip_h:
        res = res[:, ::-1]
    if flip_v:
        res = res[::-1, :]

    out_shape = geometry_gpu_shape(src_buf.height, src_buf.width, rotation)
    dst_buf._allocate(out_shape[0], out_shape[1], 3)
    # np.copyto handles the rotated/flipped strided view directly. Building
    # an intermediate contiguous frame first doubled peak memory traffic for
    # every 90-degree transform.
    np.copyto(dst_buf.arr, res)


def apply_crop_gpu(src_buf, dst_buf, crop_rect):
    """
    Crop. crop_rect: (x, y, w, h) normalized 0-1.
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
    np.copyto(dst_buf.arr, src_buf.arr[cy:cy + ch, cx:cx + cw])


def apply_crop_pixels_gpu(src_buf, dst_buf, x: int, y: int, width: int, height: int):
    """Crop using integer source pixel coordinates."""
    if width <= 0 or height <= 0:
        dst_buf.copy_from(src_buf)
        return

    src_h, src_w = src_buf.height, src_buf.width
    cx = max(0, min(int(x), src_w - 1))
    cy = max(0, min(int(y), src_h - 1))
    cw = max(1, min(int(width), src_w - cx))
    ch = max(1, min(int(height), src_h - cy))

    dst_buf._allocate(ch, cw, 3)
    np.copyto(dst_buf.arr, src_buf.arr[cy:cy + ch, cx:cx + cw])


# =========================================================
# 1D LUT / Log Encoding
# =========================================================

def apply_1d_lut_inplace(img, lut, domain_min, domain_max):
    """Apply a precomputed 1D LUT to an image in-place (linear interpolation)."""
    lut = np.ascontiguousarray(lut, dtype=np.float32)
    n = lut.shape[0]
    scale = np.float32((n - 1) / (float(domain_max) - float(domain_min)))

    flat = img.reshape(-1)
    chunk = _PIXEL_CHUNK * 3
    for start in range(0, flat.size, chunk):
        block = flat[start:start + chunk]
        pos = (block - np.float32(domain_min)) * scale
        np.clip(pos, 0.0, np.float32(n - 1), out=pos)
        i0 = np.minimum(pos.astype(np.int32), n - 2)
        pos -= i0  # reuse pos as the interpolation fraction
        block[...] = lut[i0] * (np.float32(1.0) - pos) + lut[i0 + 1] * pos


def log_encode_gpu(img, log_curve_name):
    """
    Apply log encoding via a pipeline-provided 1D LUT.

    This compatibility wrapper keeps older callers working while log curve
    generation lives in the pipeline layer.
    """
    from raw_alchemy.pipeline.log_encoding import (
        LOG_LUT_DOMAIN_MAX,
        LOG_LUT_DOMAIN_MIN,
        get_log_lut,
    )

    lut = get_log_lut(log_curve_name)
    if lut.size == 0:
        return False
    apply_1d_lut_inplace(img, lut, LOG_LUT_DOMAIN_MIN, LOG_LUT_DOMAIN_MAX)
    return True
