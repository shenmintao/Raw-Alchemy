"""Lens passthrough caches must keep linear pixels across edits (issue #38)."""

import numpy as np
import pytest

from raw_alchemy import config, lensfun_wrapper
from raw_alchemy.gpu_buffer import GpuImage
from raw_alchemy.pipeline.cache_manager import CachedImage
from raw_alchemy.pipeline.request import ProcessRequest
from raw_alchemy.workers.image_processor import ImageProcessor


def _processor(use_proxy):
    source = np.linspace(0.01, 0.3, 24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3)
    proxy = source[::2, ::2].copy() if use_proxy else None
    processor = ImageProcessor(warmup_sessions=False, background_denoise=False)
    processor.current_path = "lens-cache.raw"
    processor.cpu_linear = source
    processor.cpu_proxy_linear = proxy
    processor.cache_manager.put(
        processor.current_path,
        CachedImage(processor.current_path, source, None, None, proxy_linear=proxy),
    )
    return processor


@pytest.mark.parametrize("use_proxy", [False, True], ids=["full", "proxy"])
@pytest.mark.parametrize("lens_result", ["no-exif", "no-profile", "error", "tiled-none", "tiled-view"])
def test_skipped_lens_cache_does_not_borrow_mutable_working_buffer(
    monkeypatch, use_proxy, lens_result,
):
    processor = _processor(use_proxy)
    processor._executor_path = processor.current_path
    processor._executor_params = {"lens_correct": True}
    processor._executor_using_proxy = use_proxy
    processor.exif_data = None if lens_result == "no-exif" else {"camera_maker": "Xiaomi"}
    monkeypatch.setattr(config, "DISTORTION_MAP_CACHE_LIMIT_MB", 0 if "tiled" in lens_result else 256)

    def unavailable(image, **kwargs):
        if lens_result == "error":
            raise RuntimeError("unavailable lens profile")
        return image.view() if lens_result == "tiled-view" else None

    monkeypatch.setattr(lensfun_wrapper, "compute_lens_distortion_map", unavailable)
    monkeypatch.setattr(lensfun_wrapper, "apply_lens_correction_tiled", unavailable)
    source = processor.cpu_proxy_linear if use_proxy else processor.cpu_linear
    working = GpuImage()
    working.upload(source)
    corrected = processor._executor_lens_correct(working.arr)
    np.testing.assert_array_equal(corrected, source)

    # Subsequent colour ops and pool reuse may overwrite all of this storage.
    working.arr.fill(0.9)
    np.testing.assert_array_equal(corrected, source)
    entry = processor.cache_manager.get(processor.current_path)
    cached = entry.proxy_corrected_data if use_proxy else entry.corrected_data
    np.testing.assert_array_equal(cached, source)
    np.testing.assert_array_equal(processor._executor_corrected_source, source)
    working.clear()


@pytest.mark.parametrize("use_proxy", [False, True], ids=["full", "proxy"])
@pytest.mark.parametrize("fusion", ["0", "1"], ids=["per-op", "fused"])
def test_edits_and_reset_match_render_with_unavailable_lens_disabled(monkeypatch, use_proxy, fusion):
    monkeypatch.setenv("RAWALCHEMY_GRADE_GPU", fusion)
    monkeypatch.setenv("RAW_ALCHEMY_CPU_ONLY", "1")
    processor = _processor(use_proxy)
    reference = _processor(use_proxy)
    frames, reference_frames, errors = [], [], []
    processor.result_ready.connect(lambda image, *_: frames.append(image.copy()))
    reference.result_ready.connect(lambda image, *_: reference_frames.append(image.copy()))
    processor.error_occurred.connect(errors.append)
    reference.error_occurred.connect(errors.append)
    params = {
        "exposure_mode": "Auto", "metering_mode": "matrix",
        "saturation": 1.25, "contrast": 1.1, "denoise_enabled": False,
        "viewport_size": (32, 24),
    }
    source = processor.cpu_proxy_linear if use_proxy else processor.cpu_linear
    edits = ({}, {"wb_temp": 2.0}, {"saturation": 1.1}, {"wb_tint": -2.0}, {})
    for request_id, edit in enumerate(edits, 1):
        processor.current_request_id = reference.current_request_id = request_id
        processor._do_process(ProcessRequest(
            processor.current_path, dict(params, lens_correct=True, **edit), request_id,
        ))
        reference._do_process(ProcessRequest(
            reference.current_path, dict(params, lens_correct=False, **edit), request_id,
        ))
        assert not errors
        assert len(frames) == len(reference_frames) == request_id
        np.testing.assert_array_equal(frames[-1], reference_frames[-1])
        assert processor.last_applied_ev == pytest.approx(reference.last_applied_ev)
        corrected = processor.cpu_proxy_corrected if use_proxy else processor.cpu_corrected
        np.testing.assert_array_equal(corrected, source)
        # Force the reset through the executor, including after output eviction.
        processor.cache_manager.get(processor.current_path).drop_stage("output")
        reference.cache_manager.get(reference.current_path).drop_stage("output")

    np.testing.assert_array_equal(frames[0], frames[-1])
