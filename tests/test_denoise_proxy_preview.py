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
    p._executor_params = {}
    p.pending_request = None
    import threading
    p.lock = threading.RLock()
    p._should_stop = False
    p._decode_variant = 'test-canonical-decode'
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


def test_proxy_preview_always_allowed_with_denoise():
    """分层降噪:代理路径不再等待全图降噪(代理级 ~3s 自己出图)。"""
    p = _processor()
    assert p._should_use_proxy_preview(_preview_params()) is True
    p.cached_denoise_full = np.full(FULL, 0.1, np.float32)
    p.last_denoise_key = (p.current_path, "denoise", 0.25)
    assert p._should_use_proxy_preview(_preview_params()) is True


def test_denoise_runs_once_at_native_then_all_views_derive(monkeypatch, tmp_path):
    """解码链语义:无论首个视图是代理还是缩放 ROI,降噪都对原生全分辨率
    源只算一次;之后任何视图(代理/全图/缩放)零重算。"""
    monkeypatch.setenv("RAWALCHEMY_DENOISE_CACHE_DIR", str(tmp_path / "dc"))
    p = _processor()
    calls = []

    def fake_denoise(src, strength=0.25, progress_callback=None):
        calls.append(src.shape)
        return (src * 0.5).astype(np.float32)

    monkeypatch.setattr(processor_module, "denoise_rgb_linear", fake_denoise)
    monkeypatch.setattr(processor_module, "denoise_clear_session", lambda: None)

    class _Sig:
        def emit(self, *a):
            pass

    p.denoise_started = p.denoise_progress = p.denoise_finished = _Sig()
    p.pending_request = None

    # 首个请求是代理视图:仍然对 FULL 原生源计算,返回其降采样
    p._executor_using_proxy = True
    out = p._executor_denoise(p.cpu_proxy_linear)
    assert calls == [FULL]                      # 原生分辨率,而非代理分辨率
    assert out.shape[0] < FULL[0]               # 视图拿到的是派生降采样
    # 全图/缩放视图:零重算
    p._executor_using_proxy = False
    out_full = p._executor_denoise(p.cpu_linear)
    assert calls == [FULL]
    assert out_full.shape == FULL
    # 再回代理:仍零重算
    p._executor_using_proxy = True
    p._executor_denoise(p.cpu_proxy_linear)
    assert calls == [FULL]


def test_denoise_survives_restart_via_disk_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("RAWALCHEMY_DENOISE_CACHE_DIR", str(tmp_path / "dc"))
    raw = tmp_path / "img.raw"
    raw.write_bytes(b"raw bytes")
    calls = []

    def fake_denoise(src, strength=0.25, progress_callback=None):
        calls.append(1)
        return (src * 0.5).astype(np.float32)

    monkeypatch.setattr(processor_module, "denoise_rgb_linear", fake_denoise)
    monkeypatch.setattr(processor_module, "denoise_clear_session", lambda: None)

    class _Sig:
        def emit(self, *a):
            pass

    p1 = _processor(path=str(raw))
    p1.denoise_started = p1.denoise_progress = p1.denoise_finished = _Sig()
    p1.pending_request = None
    p1._executor_denoise(p1.cpu_linear)
    assert calls == [1]
    # 模拟重启:全新 worker,内存缓存为空 → 磁盘命中,零重算
    p2 = _processor(path=str(raw))
    p2.denoise_started = p2.denoise_progress = p2.denoise_finished = _Sig()
    p2.pending_request = None
    out = p2._executor_denoise(p2.cpu_linear)
    assert calls == [1]
    np.testing.assert_allclose(out, p2.cpu_linear * 0.5, atol=1e-3)


def test_executor_denoise_serves_proxy_scale_in_proxy_mode():
    p = _processor()
    p.cached_denoise_full = np.full(FULL, 0.1, np.float32)
    p.last_denoise_key = (p.current_path, "denoise", 0.25)
    p._executor_using_proxy = True
    out = p._executor_denoise(p.cpu_proxy_linear)
    # a downscale of the denoised full image — proxy-sized, denoised values
    assert out.shape[0] < FULL[0] and out.shape[1] < FULL[1]
    assert abs(float(out.mean()) - 0.1) < 1e-3
    # and it is cached for the next interaction
    assert p.cached_denoise_proxy is not None


def test_first_proxy_denoise_publishes_current_metering_source(monkeypatch, tmp_path):
    """The first denoise render must meter the denoised proxy, not raw pixels."""
    monkeypatch.setenv("RAWALCHEMY_DENOISE_CACHE_DIR", str(tmp_path / "dc"))
    p = _processor()
    params = _preview_params(denoise_strength=0.5, custom_db_path=None)
    p._executor_params = params
    p._executor_using_proxy = True

    monkeypatch.setattr(
        processor_module,
        "denoise_rgb_linear",
        lambda src, strength=0.25, progress_callback=None: np.ascontiguousarray(src * 0.5),
    )
    monkeypatch.setattr(processor_module, "denoise_clear_session", lambda: None)

    class _Sig:
        def emit(self, *_args):
            pass

    p.denoise_started = p.denoise_progress = p.denoise_finished = _Sig()
    p._prepare_executor_source_state(params)
    p.last_metering_key = ("stale",)

    out = p._executor_denoise(p.cpu_proxy_linear)

    assert p._executor_corrected_source is out
    assert p.cpu_proxy_corrected is out
    assert p.cached_proxy_lens_key == (False, None, 0.5)
    assert p.last_metering_key is None


def test_lens_cache_key_includes_denoise_strength():
    """Changing strength must not reuse lens-corrected pixels from the old denoise."""
    p = _processor()
    p.exif_data = None  # lens callback is a passthrough, making identity observable

    class _EmptyCache:
        @staticmethod
        def get(_path):
            return None

    p.cache_manager = _EmptyCache()
    p._executor_using_proxy = False
    first = np.full((8, 10, 3), 0.25, np.float32)
    second = np.full((8, 10, 3), 0.5, np.float32)

    p._executor_params = _preview_params(
        lens_correct=True, denoise_strength=0.25, custom_db_path=None
    )
    out_first = p._executor_lens_correct(first)
    assert out_first is first
    assert p.cached_lens_key == (True, None, 0.25)

    p._executor_params = _preview_params(
        lens_correct=True, denoise_strength=0.5, custom_db_path=None
    )
    out_second = p._executor_lens_correct(second)

    assert out_second is second
    assert p.cached_lens_key == (True, None, 0.5)


def test_prepare_source_state_keeps_proxy_resolution():
    p = _processor()
    p.cached_denoise_full = np.full(FULL, 0.1, np.float32)
    p.last_denoise_key = (p.current_path, "denoise", 0.25)
    p._executor_using_proxy = True
    p._prepare_executor_source_state(_preview_params())
    assert p.cpu_proxy_corrected is not None
    assert p.cpu_proxy_corrected.shape[0] < FULL[0]  # never the 40MP array
    # full mode still seeds the full-res corrected cache
    p2 = _processor()
    p2.cached_denoise_full = np.full(FULL, 0.1, np.float32)
    p2.last_denoise_key = (p2.current_path, "denoise", 0.25)
    p2._executor_using_proxy = False
    p2._prepare_executor_source_state(_preview_params())
    assert p2.cpu_corrected.shape == FULL


def test_session_cleared_even_when_denoise_fails(monkeypatch):
    p = _processor()
    cleared = []
    monkeypatch.setattr(processor_module, "denoise_clear_session", lambda: cleared.append(1))

    def boom(src, progress_callback=None, **kwargs):
        raise RuntimeError("DML exploded")

    monkeypatch.setattr(processor_module, "denoise_rgb_linear", boom)

    class _Sig:  # signal stub
        def emit(self, *a):
            pass

    p.denoise_started = _Sig()
    p.denoise_progress = _Sig()
    p.denoise_finished = _Sig()
    import pytest
    with pytest.raises(RuntimeError, match="DML exploded"):
        p._executor_denoise(p.cpu_linear)
    assert cleared, "session must be released on the failure path too"


def test_denoise_disk_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RAWALCHEMY_DENOISE_CACHE_DIR", str(tmp_path / "dc"))
    from raw_alchemy.pipeline import denoise_disk_cache as DC
    raw = tmp_path / "x.dng"
    raw.write_bytes(b"fake raw content")
    img = np.random.default_rng(0).random((64, 96, 3)).astype(np.float32)
    assert DC.load(str(raw), "m1") is None
    DC.save(str(raw), "m1", img)
    back = DC.load(str(raw), "m1")
    np.testing.assert_array_equal(back, img)  # lossless float32 export artifact
    assert DC.load(str(raw), "m2") is None            # 模型版本失效
    raw.write_bytes(b"changed content!!")             # 文件变更失效
    assert DC.load(str(raw), "m1") is None
