"""numpy/cv2 math_ops port — semantics guards.

Each active pipeline op is checked against an independent reference
(einsum, colour-science tetrahedral interpolation, np.rot90, direct cv2
calls, closed-form conversions) so the Taichi-to-numpy port stays honest.
"""
import colour
import cv2
import numpy as np
import pytest

from raw_alchemy import math_ops as mo
from raw_alchemy.gpu_buffer import GpuImage, NdarrayPool, gpu_pool


@pytest.fixture
def img():
    rng = np.random.default_rng(123)
    return rng.uniform(0.0, 1.2, size=(23, 31, 3)).astype(np.float32)


# ---------------------------------------------------------------- matrix

def test_apply_matrix_inplace_matches_einsum(img):
    m = np.array([[0.9, 0.2, -0.1], [0.05, 1.1, -0.15], [-0.02, 0.08, 0.94]])
    expected = np.einsum("ij,hwj->hwi", m, img.astype(np.float64))

    actual = img.copy()
    mo.apply_matrix_inplace(actual, m)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.dtype == np.float32


def test_apply_matrix_inplace_mutates_in_place(img):
    a = img.copy()
    view = a  # same object must observe the change
    mo.apply_matrix_inplace(a, np.eye(3) * 2.0)
    np.testing.assert_allclose(view, img * 2.0, rtol=1e-6)


# ---------------------------------------------------------------- gain / WB

def test_apply_gain_inplace(img):
    a = img.copy()
    mo.apply_gain_inplace(a, 1.37)
    np.testing.assert_allclose(a, img * np.float32(1.37), rtol=0, atol=0)


def test_apply_white_balance_inplace(img):
    a = img.copy()
    mo.apply_white_balance_inplace(a, 1.2, 0.95, 0.8)
    expected = img * np.array([1.2, 0.95, 0.8], dtype=np.float32)
    np.testing.assert_allclose(a, expected, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------- sat / contrast

def test_saturation_contrast_reference(img):
    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    sat, con, pivot = 1.35, 1.2, 0.18

    lum = (img.astype(np.float64) @ luma.astype(np.float64))[..., None]
    expected = lum + (img - lum) * sat
    expected = (expected - pivot) * con + pivot
    expected = np.maximum(expected, 0.0)

    actual = img.copy()
    mo.apply_saturation_contrast_inplace(actual, sat, con, pivot, luma)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- highlight / shadow

def test_highlight_shadow_reference(img):
    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    highlight, shadow = -0.4, 0.55

    f = img.astype(np.float64)
    lum = f @ luma.astype(np.float64)
    expected = f.copy()
    mask = 1.0 - lum
    factor = np.where(mask > 0.0, 1.0 + shadow * mask**3, 1.0)
    expected *= factor[..., None]
    t = 1.0 - np.clip(lum, 0.0, 1.0)
    factor = np.maximum(1.0 + highlight * (1.0 - t**3), 0.0)
    expected *= factor[..., None]
    expected = np.maximum(expected, 0.0)

    actual = img.copy()
    mo.apply_highlight_shadow_inplace(actual, highlight, shadow, luma)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_highlight_shadow_zero_is_identity(img):
    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    actual = img.copy()
    mo.apply_highlight_shadow_inplace(actual, 0.0, 0.0, luma)
    np.testing.assert_array_equal(actual, np.maximum(img, 0.0))


# ---------------------------------------------------------------- sRGB

def test_srgb_roundtrip(img):
    a = np.clip(img, 0.0, 1.0).copy()
    mo.linear_to_srgb_inplace(a)
    # against the closed-form IEC 61966-2-1 OETF
    lin = np.clip(img, 0.0, 1.0).astype(np.float64)
    expected = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    np.testing.assert_allclose(a, expected, rtol=1e-5, atol=1e-6)

    mo.srgb_to_linear_inplace(a)
    np.testing.assert_allclose(a, np.clip(img, 0.0, 1.0), rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------- 3D LUT

def test_lut3d_tetrahedral_matches_colour_reference(img):
    from colour.algebra import table_interpolation_tetrahedral

    rng = np.random.default_rng(7)
    table = rng.uniform(0.0, 1.0, size=(9, 9, 9, 3)).astype(np.float32)
    src = np.clip(img, 0.0, 1.0)

    expected = table_interpolation_tetrahedral(src.astype(np.float64), table.astype(np.float64))

    actual = src.copy()
    mo.apply_lut_inplace(actual, table, np.zeros(3), np.ones(3))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_lut3d_identity_lut_is_noop(img):
    lut = colour.LUT3D(size=5)  # identity
    src = np.clip(img, 0.0, 1.0)
    actual = src.copy()
    mo.apply_lut_inplace(actual, lut.table.astype(np.float32), lut.domain[0], lut.domain[1])
    np.testing.assert_allclose(actual, src, rtol=1e-5, atol=1e-6)


def test_lut1d_matches_interp(img):
    lut = np.sqrt(np.linspace(0.0, 1.0, 257)).astype(np.float32)
    actual = img.copy()
    mo.apply_1d_lut_inplace(actual, lut, 0.0, 1.0)
    expected = np.interp(np.clip(img, 0.0, 1.0), np.linspace(0.0, 1.0, 257), lut)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- geometry / crop

@pytest.mark.parametrize("rotation,k", [(90, -1), (180, 2), (270, 1)])
def test_geometry_rotation_matches_rot90(img, rotation, k):
    src = GpuImage()
    src.upload(np.clip(img, 0, 1))
    dst = GpuImage()
    mo.apply_geometry_gpu(src, dst, rotation=rotation)
    np.testing.assert_array_equal(dst.to_numpy(), np.rot90(np.clip(img, 0, 1), k=k))


@pytest.mark.parametrize("flip_h,flip_v", [(True, False), (False, True), (True, True)])
def test_geometry_flips(img, flip_h, flip_v):
    frame = np.clip(img, 0, 1)
    src = GpuImage()
    src.upload(frame)
    dst = GpuImage()
    mo.apply_geometry_gpu(src, dst, rotation=0, flip_h=flip_h, flip_v=flip_v)
    expected = frame
    if flip_h:
        expected = expected[:, ::-1]
    if flip_v:
        expected = expected[::-1, :]
    np.testing.assert_array_equal(dst.to_numpy(), expected)


def test_geometry_identity_copies(img):
    src = GpuImage()
    src.upload(np.clip(img, 0, 1))
    dst = GpuImage()
    mo.apply_geometry_gpu(src, dst, rotation=0, flip_h=False, flip_v=False)
    np.testing.assert_array_equal(dst.to_numpy(), np.clip(img, 0, 1))
    assert dst.arr is not src.arr  # a copy, not an alias


def test_crop_matches_slicing(img):
    frame = np.clip(img, 0, 1)
    h, w = frame.shape[:2]
    src = GpuImage()
    src.upload(frame)

    dst = GpuImage()
    rect = (0.12, 0.21, 0.55, 0.62)
    mo.apply_crop_gpu(src, dst, rect)
    cx, cy = int(rect[0] * w), int(rect[1] * h)
    cw, ch = int(rect[2] * w), int(rect[3] * h)
    np.testing.assert_array_equal(dst.to_numpy(), frame[cy:cy + ch, cx:cx + cw])

    dst_px = GpuImage()
    mo.apply_crop_pixels_gpu(src, dst_px, 5, 7, 12, 9)
    np.testing.assert_array_equal(dst_px.to_numpy(), frame[7:16, 5:17])


# ---------------------------------------------------------------- perspective

def test_perspective_matches_cv2_direct(img):
    frame = np.ascontiguousarray(np.clip(img, 0, 1))
    h, w = frame.shape[:2]
    corners = ((0.05, 0.02), (0.97, 0.06), (0.93, 0.96), (0.03, 0.9))
    _, m_inv = mo.compute_perspective_matrix(corners, w, h)

    dst = np.zeros_like(frame)
    mo.perspective_warp_kernel(frame, dst, m_inv)

    expected = cv2.warpPerspective(
        frame, np.ascontiguousarray(m_inv), (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    np.testing.assert_array_equal(dst, expected)


def test_perspective_identity_matrix_is_noop(img):
    # Note: "identity" *corners* are deliberately not a no-op (the op maps the
    # unit square onto [0, w-1]x[0, h-1], same as the retired kernel), so the
    # no-op check uses an identity matrix instead.
    frame = np.ascontiguousarray(np.clip(img, 0, 1))
    dst = np.zeros_like(frame)
    mo.perspective_warp_kernel(frame, dst, np.eye(3))
    np.testing.assert_allclose(dst, frame, rtol=0, atol=1e-6)


# ---------------------------------------------------------------- uint8 output

def test_float_to_uint8_rounding_and_clamping():
    src = np.array(
        [[[-0.5, 0.0, 0.004], [0.5, 0.998, 1.5]]], dtype=np.float32
    )
    dst = np.zeros((1, 2, 3), dtype=np.uint8)
    mo.float_to_uint8_gpu(src, dst)
    # round-half-up of clip(x)*255 (0.004*255+0.5 = 1.52 -> 1)
    np.testing.assert_array_equal(dst, [[[0, 0, 1], [128, 254, 255]]])


def test_resize_float_to_uint8_shape_dtype(img):
    src = np.ascontiguousarray(np.clip(img, 0, 1))
    dst = np.zeros((10, 17, 3), dtype=np.uint8)
    mo.resize_float_to_uint8_gpu(src, dst)
    assert dst.shape == (10, 17, 3)
    assert dst.dtype == np.uint8
    expected = cv2.resize(src, (17, 10), interpolation=cv2.INTER_LINEAR)
    expected8 = (np.clip(expected, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    np.testing.assert_array_equal(dst, expected8)


def test_clip_and_max_inplace(img):
    a = (img * 2 - 0.5).copy()
    mo.clip_inplace(a)
    assert a.min() >= 0.0 and a.max() <= 1.0
    b = (img * 2 - 0.5).copy()
    mo.max_inplace(b, 1e-6)
    assert b.min() >= 1e-6


# ---------------------------------------------------------------- RL sharpen

def test_rl_sharpen_reduces_blur_on_synthetic_edge():
    sharp = np.zeros((48, 64, 3), dtype=np.float32)
    sharp[:, 32:, :] = 0.8
    sharp[:, :32, :] = 0.2
    blurred = cv2.GaussianBlur(sharp, (0, 0), 1.2)

    image = GpuImage()
    image.upload(blurred)
    mo.sharpen_gpu(image, strength=1.0, sigma=1.2)
    result = image.to_numpy()

    # closer to the sharp target than the blurred input, edge visibly
    # steepened, and artifact-free
    err_before = float(np.abs(blurred - sharp).mean())
    err_after = float(np.abs(result - sharp).mean())
    assert err_after < err_before
    grad_before = float(np.abs(np.diff(blurred[24, :, 0])).max())
    grad_after = float(np.abs(np.diff(result[24, :, 0])).max())
    assert grad_after > 1.3 * grad_before
    assert np.isfinite(result).all()
    assert result.min() >= 0.0 and result.max() <= 2.0
    # far field untouched (no ringing spreading to flat regions)
    np.testing.assert_allclose(result[:, :8], blurred[:, :8], atol=1e-4)
    np.testing.assert_allclose(result[:, -8:], blurred[:, -8:], atol=1e-4)


def test_rl_sharpen_zero_strength_is_noop(img):
    image = GpuImage()
    image.upload(np.clip(img, 0, 1))
    before = image.to_numpy()
    mo.sharpen_gpu(image, strength=0.0)
    np.testing.assert_array_equal(image.to_numpy(), before)


def test_richardson_lucy_channel_matches_reference_math():
    rng = np.random.default_rng(3)
    channel = rng.uniform(0.1, 0.9, size=(20, 24)).astype(np.float32)
    k1d = mo._gaussian_kernel_1d(1.0)

    actual = mo.richardson_lucy_channel(channel, k1d, iterations=3, clip=False)

    # direct reference: separable REFLECT_101 convolution, f64
    def blur(x):
        pad = len(k1d) // 2
        p = np.pad(x, pad, mode="reflect")
        t = np.apply_along_axis(lambda r: np.convolve(r, k1d[::-1], "valid"), 1, p)
        t = np.apply_along_axis(lambda c: np.convolve(c, k1d[::-1], "valid"), 0, t)
        return t

    estimate = channel.astype(np.float64)
    for _ in range(3):
        b = np.maximum(blur(estimate), 1e-8)
        estimate = np.clip(estimate * blur(channel / b), 0.0, 2.0)

    np.testing.assert_allclose(actual, estimate, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------- buffer pool

def test_pool_numpy_dtype_acquire_release_reuse_and_trim():
    pool = NdarrayPool(max_bytes=1 << 20, max_per_key=4)

    f32 = pool.acquire(np.float32, (6, 5, 3))
    assert f32.dtype == np.float32 and f32.shape == (6, 5, 3)
    assert pool.release(f32, np.float32, (6, 5, 3)) is True
    assert pool.free_bytes() == 6 * 5 * 3 * 4

    # exact (dtype, shape) reuse returns the same object
    again = pool.acquire(np.float32, (6, 5, 3))
    assert again is f32
    assert pool.free_bytes() == 0

    # numpy and taichi-style dtype spellings share one pool key
    assert NdarrayPool._key(np.float32, (2, 2)) == NdarrayPool._key("f32", (2, 2))
    assert NdarrayPool._key(np.uint8, (2, 2)) == NdarrayPool._key("u8", (2, 2))

    u8 = pool.acquire(np.uint8, (6, 5, 3))
    assert u8.dtype == np.uint8
    pool.release(u8, np.uint8, (6, 5, 3))
    pool.release(again, np.float32, (6, 5, 3))
    assert pool.free_bytes() == 6 * 5 * 3 * 5  # 4 bytes f32 + 1 byte u8

    freed = pool.trim(0)
    assert freed == 6 * 5 * 3 * 5
    assert pool.free_bytes() == 0


def test_gpu_image_to_numpy_is_independent_copy(img):
    gpu_pool().clear()
    frame = np.clip(img, 0, 1)
    image = GpuImage()
    image.upload(frame)
    out = image.to_numpy()
    image.arr[...] = -1.0
    np.testing.assert_array_equal(out, frame)  # caller copy unaffected
    image.clear()
    gpu_pool().clear()
