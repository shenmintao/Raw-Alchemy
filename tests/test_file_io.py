from pathlib import Path

import numpy as np
import pillow_heif

from raw_alchemy.file_io import _quantize_image, save_image


class DummyLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def test_quantize_image_rounds_to_nearest_without_mutating_source():
    img = np.linspace(0.0, 1.0, 17 * 13 * 3, dtype=np.float32).reshape(17, 13, 3)
    original = img.copy()

    for dtype, scale in ((np.uint8, 255), (np.uint16, 65535)):
        expected = np.floor(img * scale + 0.5).astype(dtype)
        actual = _quantize_image(img, dtype, scale)
        np.testing.assert_array_equal(actual, expected)

    np.testing.assert_array_equal(img, original)


def test_save_hdr_heif_writes_readable_10_bit_pq_file():
    img = np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    output = Path("tests") / "_tmp_sample.hdr.heif"

    try:
        assert save_image(img.copy(), str(output), DummyLogger(), hdr_output=True)

        heif = pillow_heif.open_heif(output, convert_hdr_to_8bit=False)
        assert heif.size == (8, 8)
        assert heif.mode == "RGB;16"
        assert heif.info["bit_depth"] == 10
        assert heif.info["chroma"] == 420
        assert heif.info["nclx_profile"]["color_primaries"] == 9
        assert heif.info["nclx_profile"]["transfer_characteristics"] == 16
        assert heif.info["nclx_profile"]["matrix_coefficients"] == 9
        assert heif.info["nclx_profile"]["full_range_flag"] == 1
    finally:
        output.unlink(missing_ok=True)
