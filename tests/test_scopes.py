import numpy as np
import pytest

from raw_alchemy.utils import compute_histogram_fast, compute_waveform_fast


def _histogram_reference(img, bins=100, sample_rate=4):
    sample = img[::sample_rate, ::sample_rate, :].copy()
    if sample.dtype == np.uint8:
        sample = sample.astype(np.float32) * (1.0 / 255.0)
    elif sample.dtype != np.float32:
        sample = sample.astype(np.float32)
    sample = np.clip(sample, 0.0, 1.0)
    return [
        np.histogram(sample[:, :, channel].ravel().copy(), bins=bins, range=(0.0, 1.0))[0]
        .astype(np.float64)
        for channel in range(3)
    ]


def _waveform_reference(img, bins=100, sample_rate=4):
    _h, w, _c = img.shape
    sampled_width = max(1, w // sample_rate)
    v_step = max(1, sample_rate // 2)
    img_sub = img[::v_step, :, :]
    if img_sub.dtype == np.uint8:
        img_f = img_sub.astype(np.float32) * (1.0 / 255.0)
    elif img_sub.dtype != np.float32:
        img_f = img_sub.astype(np.float32)
    else:
        img_f = img_sub.copy()
    img_f = np.clip(img_f, 0.0, 1.0)
    luma = (
        img_f[:, :, 0] * 0.2126
        + img_f[:, :, 1] * 0.7152
        + img_f[:, :, 2] * 0.0722
    ).astype(np.float32)
    waveform = np.zeros((sampled_width, bins), dtype=np.float32)
    for col_idx in range(sampled_width):
        col_data = luma[:, col_idx * sample_rate]
        bin_indices = np.clip((col_data * bins).astype(np.int32), 0, bins - 1)
        np.add.at(waveform[col_idx], bin_indices, 1.0)
    max_val = np.max(waveform)
    if max_val > 0:
        waveform /= max_val
    return waveform


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_histogram_fast_matches_reference(dtype):
    rng = np.random.default_rng(41)
    if dtype == np.uint8:
        img = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    else:
        img = rng.uniform(-0.2, 1.2, size=(37, 53, 3)).astype(np.float32)
    original = img.copy()

    actual = compute_histogram_fast(img, bins=37, sample_rate=4)
    expected = _histogram_reference(img, bins=37, sample_rate=4)

    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(img, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_waveform_fast_matches_reference(dtype):
    rng = np.random.default_rng(42)
    if dtype == np.uint8:
        img = rng.integers(0, 256, size=(39, 55, 3), dtype=np.uint8)
    else:
        img = rng.uniform(-0.2, 1.2, size=(39, 55, 3)).astype(np.float32)
    original = img.copy()

    actual = compute_waveform_fast(img, bins=41, sample_rate=4)
    expected = _waveform_reference(img, bins=41, sample_rate=4)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(img, original)
