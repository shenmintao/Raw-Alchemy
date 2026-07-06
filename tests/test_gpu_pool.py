"""T7.2 — GPU buffer pool: shape-keyed recycling, byte budget, VRAM lifecycle."""

import numpy as np

from raw_alchemy.gpu_buffer import GpuImage, NdarrayPool, gpu_pool


def _f32_bytes(shape):
    total = 4
    for dim in shape:
        total *= dim
    return total


def test_pool_recycles_exact_shape_and_dtype():
    pool = NdarrayPool(max_bytes=1 << 20, max_per_key=4)

    first = pool.acquire(np.float32, (4, 6, 3))
    assert pool.release(first, np.float32, (4, 6, 3)) is True
    assert pool.free_bytes() == _f32_bytes((4, 6, 3))

    # Same (dtype, shape): the very same ndarray object comes back.
    again = pool.acquire(np.float32, (4, 6, 3))
    assert again is first
    assert pool.free_bytes() == 0

    # Different shape or dtype never aliases a pooled buffer.
    pool.release(again, np.float32, (4, 6, 3))
    other_shape = pool.acquire(np.float32, (6, 4, 3))
    other_dtype = pool.acquire(np.uint8, (4, 6, 3))
    assert other_shape is not first
    assert other_dtype is not first
    assert pool.acquire(np.float32, (4, 6, 3)) is first


def test_pool_enforces_byte_budget_and_per_key_cap():
    entry = _f32_bytes((4, 4, 3))
    pool = NdarrayPool(max_bytes=2 * entry, max_per_key=2)

    buffers = [pool.acquire(np.float32, (4, 4, 3)) for _ in range(4)]
    retained = [pool.release(buf, np.float32, (4, 4, 3)) for buf in buffers]

    # Budget fits two entries and the per-key cap is two: the rest is dropped.
    assert retained.count(True) == 2
    assert pool.free_bytes() <= 2 * entry

    # A buffer larger than the whole budget is never pooled.
    big = pool.acquire(np.float32, (64, 64, 3))
    assert pool.release(big, np.float32, (64, 64, 3)) is False

    freed = pool.trim(0)
    assert freed == 2 * entry
    assert pool.free_bytes() == 0


def test_gpu_image_clear_returns_buffer_to_shared_pool():
    gpu_pool().clear()

    image = GpuImage(4, 6)
    arr = image.arr
    nbytes = image.nbytes
    image.clear()

    assert not image.valid
    assert gpu_pool().free_bytes() >= nbytes

    # The next same-shape GpuImage reuses the identical device buffer.
    reused = GpuImage(4, 6)
    assert reused.arr is arr
    reused.clear()
    gpu_pool().clear()


def test_gpu_image_upload_roundtrip_via_pooled_buffer():
    gpu_pool().clear()
    rng = np.random.default_rng(3)
    a = rng.uniform(0.0, 1.0, size=(5, 7, 3)).astype(np.float32)
    b = rng.uniform(0.0, 1.0, size=(5, 7, 3)).astype(np.float32)

    img = GpuImage()
    img.upload(a)
    img.clear()

    # Re-acquired from the pool: contents must be fully overwritten by upload.
    img2 = GpuImage()
    img2.upload(b)
    np.testing.assert_array_equal(img2.to_numpy(), b)
    img2.clear()
    gpu_pool().clear()


def test_sharpen_scratch_buffers_are_pooled_not_module_resident():
    from raw_alchemy import math_ops
    from raw_alchemy.math_ops import sharpen_gpu

    # T7.2 removed the permanently-resident module-global buffer cache.
    assert not hasattr(math_ops, "_sharpen_gpu_bufs")

    gpu_pool().clear()
    rng = np.random.default_rng(11)
    src = rng.uniform(0.1, 0.9, size=(12, 10, 3)).astype(np.float32)

    image = GpuImage()
    image.upload(src)
    sharpen_gpu(image, strength=0.5, sigma=1.0)

    # The four 2D scratch buffers were released back to the pool.
    scratch = 4 * _f32_bytes((12, 10))
    assert gpu_pool().free_bytes() >= scratch

    # A second run reuses the pooled scratch instead of growing the pool.
    before = gpu_pool().free_bytes()
    sharpen_gpu(image, strength=0.5, sigma=1.0)
    assert gpu_pool().free_bytes() == before

    result = image.to_numpy()
    assert result.shape == src.shape
    assert np.isfinite(result).all()
    assert not np.array_equal(result, src)  # it actually sharpened
    image.clear()
    gpu_pool().clear()
