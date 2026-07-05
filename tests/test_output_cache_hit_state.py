"""Output-cache hit path must keep worker/export state in sync (M4/M5/M6).

The T7.4 multi-slot uint8 output cache made pure hits routine: parameter
round trips (lens/denoise ON->OFF->ON) and image switches with unchanged
view state re-emit a cached output and return before the executor runs.
Everything a hit skips therefore has to be either synced on the hit path
(when cheap) or key-validated by its consumer (when a resync would cost a
lens/denoise recompute). These tests pin that contract:

- M4: a hit re-publishes the emitted EV into ``processor.last_applied_ev``
  — the Auto-exposure export contract (``resolve_export_exposure``) reads
  it, so a stale value silently bakes another image's EV into the export.
- M5: ``get_cached_for_export`` never serves a ``corrected`` array whose
  lens/denoise state does not match the current UI params; after a lens
  round trip the hit path cheaply restores the entry's corrected
  write-back so the fast export path stays alive.
- M6: a neighbor preload aborted mid-decode (T7.3) caches
  ``proxy_linear=None``; the next real load backfills the proxy so the
  interactive proxy path is not silently disabled for that image.
"""
import numpy as np
import pytest

from raw_alchemy.exporting import resolve_export_exposure
from raw_alchemy.pipeline.cache_manager import CachedImage
from raw_alchemy.pipeline.executor import _BaseExecutor
from raw_alchemy.pipeline.request import ProcessRequest
from raw_alchemy.workers import image_processor as processor_module
from raw_alchemy.workers.image_processor import ImageProcessor


def _base_params():
    return {
        "lens_correct": False,
        "exposure_mode": "Manual",
        "exposure": 0.0,
        "metering_mode": "matrix",
        "wb_temp": 0.0,
        "wb_tint": 0.0,
        "highlight": 0.0,
        "shadow": 0.0,
        "saturation": 1.0,
        "contrast": 1.0,
        "log_space": "None",
        "lut_path": None,
        "rotation": 0,
        "flip_horizontal": False,
        "flip_vertical": False,
        "perspective_corners": None,
        "crop": (0.0, 0.0, 1.0, 1.0),
        "sharpen_strength": 0.0,
        "denoise_enabled": False,
    }


def _case(overrides):
    params = _base_params()
    params.update(overrides)
    return params


def _make_processor(path, source, proxy=None):
    processor = ImageProcessor()
    processor.cpu_linear = source
    processor.cpu_proxy_linear = proxy
    processor.current_path = path
    processor.exif_data = None
    processor.cache_manager.put(
        path, CachedImage(path, source, None, None, proxy_linear=proxy)
    )
    return processor


def _install_op_counter(monkeypatch):
    """Count every op application across all executors."""
    op_calls = []
    original_apply = _BaseExecutor._apply_op

    def counting_apply(self, buf, op):
        op_calls.append(op.name)
        return original_apply(self, buf, op)

    monkeypatch.setattr(_BaseExecutor, "_apply_op", counting_apply)
    return op_calls


def _export_snapshot(processor, params):
    """Fetch the single-image export fast-path data for the given UI params.

    Falls back to the legacy no-validation signature so that, run against
    the pre-fix code, this test fails on the actual defect (a stale
    ``corrected`` array) instead of on a TypeError.
    """
    try:
        return processor.get_cached_for_export(params)
    except TypeError:
        return processor.get_cached_for_export()


# =====================================================================
# M4 — hit path must publish the emitted EV to last_applied_ev
# =====================================================================


def test_output_cache_hit_syncs_last_applied_ev_for_auto_export(monkeypatch):
    path_a, path_b = "auto_a.raw", "auto_b.raw"
    bright = np.linspace(0.2, 0.5, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    dark = np.ascontiguousarray(bright * 0.05)

    processor = ImageProcessor()
    processor.cache_manager.put(path_a, CachedImage(path_a, bright, None, None))
    processor.cache_manager.put(path_b, CachedImage(path_b, dark, None, None))

    emitted = []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append((img_path, ev))
    )
    op_calls = _install_op_counter(monkeypatch)

    params = _case({"exposure_mode": "Auto"})

    processor._do_process(ProcessRequest(path_a, dict(params), 1))
    ev_a = processor.last_applied_ev
    processor._do_process(ProcessRequest(path_b, dict(params), 2))
    ev_b = processor.last_applied_ev
    # Distinct auto EVs are the precondition for observing the desync.
    assert abs(ev_a - ev_b) > 0.1

    # Switching back to A is a pure output-cache hit (T7.4: the uint8
    # slots live on the entry and survive image switches): no op re-runs,
    # and the preview correctly re-shows A's EV...
    calls_before = len(op_calls)
    processor._do_process(ProcessRequest(path_a, dict(params), 3))
    assert len(op_calls) == calls_before
    assert emitted[-1][0] == path_a
    assert emitted[-1][1] == pytest.approx(ev_a)

    # ...and the Auto-exposure export contract must see that same EV, or
    # exporting A silently bakes B's exposure (M4).
    assert processor.last_applied_ev == pytest.approx(ev_a)
    fast_path_ev, _ = resolve_export_exposure(params, processor.last_applied_ev)
    assert fast_path_ev == pytest.approx(emitted[-1][1])


# =====================================================================
# M5 — export must never get a corrected array that mismatches the params
# =====================================================================


def test_hit_after_denoise_roundtrip_never_serves_undenoised_export(monkeypatch):
    path = "denoise_roundtrip.raw"
    source = np.linspace(0.05, 0.6, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    denoised = np.ascontiguousarray(source * 0.5)

    denoise_calls = []

    def fake_denoise_raw(p, exposure_ratio=1.0, progress_callback=None):
        denoise_calls.append(p)
        return denoised.copy()

    monkeypatch.setattr(processor_module, "denoise_raw", fake_denoise_raw)
    monkeypatch.setattr(processor_module, "denoise_clear_session", lambda: None)

    processor = _make_processor(path, source)
    params_on = _case({"denoise_enabled": True})
    params_off = _case({"denoise_enabled": False})

    emitted = []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append(img.copy())
    )

    # Denoise ON: the compute path denoises and publishes the denoised
    # array as the export-facing pre-color source.
    processor._do_process(ProcessRequest(path, dict(params_on), 1))
    assert len(denoise_calls) == 1
    np.testing.assert_array_equal(processor.cpu_corrected, denoised)
    exported = _export_snapshot(processor, params_on)
    assert exported is not None
    np.testing.assert_array_equal(exported["corrected"], denoised)

    # Denoise OFF: re-render; the denoise caches are dropped and the
    # corrected source falls back to the linear data.
    processor._do_process(ProcessRequest(path, dict(params_off), 2))
    assert processor.cached_denoise_full is None
    np.testing.assert_array_equal(processor.cpu_corrected, source)

    # Denoise ON again: with the T7.4 multi-slot cache this is a pure
    # output-cache hit — the denoiser must NOT run again, and the preview
    # re-shows the denoised image...
    processor._do_process(ProcessRequest(path, dict(params_on), 3))
    assert len(denoise_calls) == 1
    np.testing.assert_array_equal(emitted[2], emitted[0])

    # ...so the export fast path must either resync to a denoised source
    # or refuse (None -> the controller takes the full re-processing
    # path). Serving the un-denoised linear source here bakes a
    # no-denoise export while the preview shows denoise (M5).
    exported = _export_snapshot(processor, params_on)
    if exported is not None:
        np.testing.assert_array_equal(exported["corrected"], denoised)


def test_hit_after_lens_roundtrip_restores_key_consistent_export_state(monkeypatch):
    path = "lens_roundtrip.raw"
    source = np.linspace(0.05, 0.6, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)

    processor = _make_processor(path, source)
    params_on = _case({"lens_correct": True})
    params_off = _case({"lens_correct": False})

    op_calls = _install_op_counter(monkeypatch)

    # Lens ON: the lens callback runs (passthrough without EXIF, but the
    # state flow is the real one) and writes the corrected array + lens
    # key back to the cache entry.
    processor._do_process(ProcessRequest(path, dict(params_on), 1))
    entry = processor.cache_manager.get(path)
    corrected_on = entry.corrected_data
    assert corrected_on is not None
    assert entry.lens_key == (True, None, False)
    exported = _export_snapshot(processor, params_on)
    assert exported is not None and exported["corrected"] is corrected_on

    # Lens OFF: re-render; the export-facing source becomes the plain
    # linear data.
    processor._do_process(ProcessRequest(path, dict(params_off), 2))
    assert processor.cpu_corrected is source

    # Lens ON again: pure output-cache hit — nothing recomputes...
    calls_before = len(op_calls)
    processor._do_process(ProcessRequest(path, dict(params_on), 3))
    assert len(op_calls) == calls_before

    # ...and the export fast path must be key-consistent with the lens-ON
    # preview again: either restored from the entry's corrected write-back
    # (cheap, no recompute) or refused. Pre-fix it serves the lens-OFF
    # linear source while the preview shows the lens-corrected render (M5).
    exported = _export_snapshot(processor, params_on)
    assert exported is None or exported["corrected"] is corrected_on
    # The cheap restore is expected to keep the fast path alive here.
    assert exported is not None


# =====================================================================
# M6 — aborted preload must not permanently disable the proxy path
# =====================================================================


def test_proxy_only_preload_backfills_full_res_on_load(monkeypatch):
    monkeypatch.setattr(processor_module, "PROXY_MIN_SOURCE_PIXELS", 200)
    monkeypatch.setattr(processor_module, "PROXY_TARGET_PIXELS", 100)
    path = "preloaded.raw"
    reduced = np.linspace(0.05, 0.6, 20 * 20 * 3, dtype=np.float32).reshape(20, 20, 3)
    full = np.linspace(0.0, 1.0, 40 * 40 * 3, dtype=np.float32).reshape(40, 40, 3)

    processor = ImageProcessor()
    # Approach B: neighbor preload decodes GPU-free (half_size) and stores
    # proxy-only; the full-res GPU demosaic is deferred to first view.
    monkeypatch.setattr(processor, "_rawpy_proxy_decode", lambda p: (reduced, None, None))
    monkeypatch.setattr(processor, "_rawpy_to_prophoto", lambda p: (full, None, None))

    processor._do_preload(ProcessRequest(path, {"_preload": True}, -1))
    entry = processor.cache_manager.get(path)
    assert entry is not None
    assert entry.linear_data is None
    assert entry.proxy_linear is not None

    # Opening the image runs the exact full-res render once and publishes it
    # back, so subsequent views (and zoom/denoise/export) reuse it.
    processor._do_load(ProcessRequest(path, {"_load": True}, 1))

    assert processor.cpu_linear is full
    assert entry.linear_data is full
    assert processor.cpu_proxy_linear is not None
    fit_params = _case({"viewport_size": (10, 10), "preview_zoom": 1.0})
    assert processor._should_use_proxy_preview(fit_params) is True
