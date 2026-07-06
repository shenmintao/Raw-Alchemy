"""Denoise + interactive preview: proxy path and GPU session release.

pre5 field report: with denoise on, (a) VRAM stayed allocated after the
denoise pass, (b) zooming was laggy. Root causes pinned here:
- the proxy preview was hard-disabled whenever denoise was enabled, so every
  interactive render ran the full-resolution pipeline;
- `_prepare_executor_source_state` seeded the *proxy* corrected cache with the
  full-resolution denoised array;
- the ONNX session was only released on the success path (and without a
  gc.collect(), which pybind session objects need).
"""
import numpy as np

from raw_alchemy.workers import image_processor as processor_module
from raw_alchemy.workers.image_processor import ImageProcessor


# _make_proxy only engages above PROXY_MIN_SOURCE_PIXELS (4MP), so the test
# frames must look like real camera files.
FULL = (1800, 2700, 3)


def _processor(path="img.raw"):
    p = ImageProcessor.__new__(ImageProcessor)  # no QThread init needed here
    p.cpu_linear = np.full(FULL, 0.2, np.float32)
    p.cpu_proxy_linear = ImageProcessor._make_proxy(p.cpu_linear)
    p.current_path = path
    p.cached_denoise_original = None
    p.cached_denoise_full = None
    p.cached_denoise_proxy = None
    p.last_denoise_key = None
    p._executor_using_proxy = False
    p._executor_path = path
    p._executor_corrected_source = None
    p.cpu_corrected = None
    p.cpu_proxy_corrected = None
    p.cached_lens_key = None
    p.cached_proxy_lens_key = None
    p.last_metering_key = None
    return p


def _preview_params(**over):
    params = {
        "denoise_enabled": True,
        "viewport_size": (800, 600),
        "preview_zoom": 1.0,
        "lens_correct": False,
    }
    params.update(over)
    return params


def test_proxy_preview_blocked_until_denoise_cached():
    p = _processor()
    assert p._should_use_proxy_preview(_preview_params()) is False


def test_proxy_preview_allowed_once_denoise_cached():
    p = _processor()
    p.cached_denoise_full = np.full(FULL, 0.1, np.float32)
    p.last_denoise_key = (p.current_path, "denoise")
    assert p._should_use_proxy_preview(_preview_params()) is True
    # ...but not for a different file's cache
    p.last_denoise_key = ("other.raw", "denoise")
    assert p._should_use_proxy_preview(_preview_params()) is False


def test_executor_denoise_serves_proxy_scale_in_proxy_mode():
    p = _processor()
    p.cached_denoise_full = np.full(FULL, 0.1, np.float32)
    p.last_denoise_key = (p.current_path, "denoise")
    p._executor_using_proxy = True
    out = p._executor_denoise(p.cpu_proxy_linear)
    # a downscale of the denoised full image — proxy-sized, denoised values
    assert out.shape[0] < FULL[0] and out.shape[1] < FULL[1]
    assert abs(float(out.mean()) - 0.1) < 1e-3
    # and it is cached for the next interaction
    assert p.cached_denoise_proxy is not None


def test_prepare_source_state_keeps_proxy_resolution():
    p = _processor()
    p.cached_denoise_full = np.full(FULL, 0.1, np.float32)
    p.last_denoise_key = (p.current_path, "denoise")
    p._executor_using_proxy = True
    p._prepare_executor_source_state(_preview_params())
    assert p.cpu_proxy_corrected is not None
    assert p.cpu_proxy_corrected.shape[0] < FULL[0]  # never the 40MP array
    # full mode still seeds the full-res corrected cache
    p2 = _processor()
    p2.cached_denoise_full = np.full(FULL, 0.1, np.float32)
    p2.last_denoise_key = (p2.current_path, "denoise")
    p2._executor_using_proxy = False
    p2._prepare_executor_source_state(_preview_params())
    assert p2.cpu_corrected.shape == FULL


def test_session_cleared_even_when_denoise_fails(monkeypatch):
    p = _processor()
    cleared = []
    monkeypatch.setattr(processor_module, "denoise_clear_session", lambda: cleared.append(1))

    def boom(src, progress_callback=None):
        raise RuntimeError("DML exploded")

    monkeypatch.setattr(processor_module, "denoise_rgb_linear", boom)

    class _Sig:  # signal stub
        def emit(self, *a):
            pass

    p.denoise_started = _Sig()
    p.denoise_progress = _Sig()
    p.denoise_finished = _Sig()
    out = p._executor_denoise(p.cpu_linear)
    np.testing.assert_array_equal(out, p.cpu_linear)  # graceful fallback
    assert cleared, "session must be released on the failure path too"
