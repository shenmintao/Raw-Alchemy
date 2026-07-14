"""T7.4 — proxy/full cache separation + zoom decoupled from the color pipeline.

Covers:
- ImageProcessor keeps one PreviewExecutor per source mode (proxy/full):
  zoom crossing 1.0 and idle full refines no longer evict each other's
  prefix/final caches, so mode switches cost only a resample.
- The output key is split into (pipeline key, view key): zoom/viewport/DPR
  changes never invalidate the pipeline side.
- CachedImage keeps a small multi-slot uint8 output cache (primary + LRU
  slots), so a fit<->zoom round trip re-emits without any recomputation.
- The executor cache byte budget is shared across the proxy/full pair with
  proxy priority.
"""
import numpy as np
import pytest

from raw_alchemy.pipeline.cache_manager import (
    OUTPUT_SLOT_LIMIT,
    CachedImage,
)
from raw_alchemy.pipeline.executor import _BaseExecutor
from raw_alchemy.pipeline.ops import build_op_list
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


def _proxy_processor(monkeypatch, path="synthetic.raw"):
    monkeypatch.setattr(processor_module, "PROXY_MIN_SOURCE_PIXELS", 200)
    monkeypatch.setattr(processor_module, "PROXY_TARGET_PIXELS", 100)
    source = np.linspace(0.05, 0.6, 20 * 20 * 3, dtype=np.float32).reshape(20, 20, 3)
    proxy = ImageProcessor._make_proxy(source)
    assert proxy is not None and proxy.shape == (10, 10, 3)
    return _make_processor(path, source, proxy=proxy), source, proxy


# =====================================================================
# Output key split: pipeline key vs view key
# =====================================================================


def test_output_key_separates_pipeline_and_view_state():
    params = _case(
        {
            "viewport_size": (100, 100),
            "preview_zoom": 0.5,
            "device_pixel_ratio": 1.0,
            "_preview_source": "proxy",
        }
    )
    base_key = ImageProcessor._make_output_key(params)
    assert isinstance(hash(base_key), int)

    # Zoom-only change: pipeline part identical, view part differs.
    zoom_key = ImageProcessor._make_output_key(dict(params, preview_zoom=0.8))
    assert zoom_key[0] == base_key[0]
    assert zoom_key[1] != base_key[1]
    assert zoom_key != base_key

    # Viewport-only change likewise stays on the view side.
    vp_key = ImageProcessor._make_output_key(dict(params, viewport_size=(80, 60)))
    assert vp_key[0] == base_key[0]
    assert vp_key[1] != base_key[1]

    # A color-pipeline change flips only the pipeline part.
    exp_key = ImageProcessor._make_output_key(dict(params, exposure=1.0))
    assert exp_key[0] != base_key[0]
    assert exp_key[1] == base_key[1]

    # Proxy/full source switch is pipeline-side (different result data).
    full_key = ImageProcessor._make_output_key(dict(params, _preview_source="full"))
    assert full_key[0] != base_key[0]

    # Mutable param containers (sidecar-loaded lists) normalize hashably.
    list_key = ImageProcessor._make_output_key(
        dict(params, viewport_size=[100, 100], crop=[0.0, 0.0, 1.0, 1.0])
    )
    assert list_key == base_key


# =====================================================================
# CachedImage multi-slot uint8 output cache
# =====================================================================


def _slot_img(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_output_slots_store_and_retrieve_multiple_keys():
    item = CachedImage("p", np.zeros((8, 8, 3), np.float32), None, None)
    assert item.get_output(("pipe", "v1")) is None

    item.store_output(("pipe", "v1"), _slot_img(1), 0.25, (10, 10))
    item.store_output(("pipe", "v2"), _slot_img(2), 0.5, (20, 20))

    out1 = item.get_output(("pipe", "v1"))
    out2 = item.get_output(("pipe", "v2"))
    assert out1 is not None and out1[0][0, 0, 0] == 1
    assert out1[1] == 0.25 and out1[2] == (10, 10)
    assert out2 is not None and out2[0][0, 0, 0] == 2
    assert out2[1] == 0.5 and out2[2] == (20, 20)
    assert item.get_output(("pipe", "v3")) is None
    # The most recent store is the primary slot.
    assert item.output_key == ("pipe", "v2")

    # Re-storing an existing key replaces it without leaving duplicates.
    item.store_output(("pipe", "v1"), _slot_img(3), 0.75, (10, 10))
    assert item.get_output(("pipe", "v1"))[0][0, 0, 0] == 3
    assert item.get_output(("pipe", "v2"))[0][0, 0, 0] == 2
    assert 1 + len(item._output_slots) == 2


def test_output_slots_capacity_evicts_lru():
    item = CachedImage("p", np.zeros((8, 8, 3), np.float32), None, None)
    for i in range(OUTPUT_SLOT_LIMIT + 1):
        item.store_output(("pipe", i), _slot_img(i), 0.0, (8, 8))

    # Oldest slot evicted, the rest retained (primary + secondary slots).
    assert item.get_output(("pipe", 0)) is None
    for i in range(1, OUTPUT_SLOT_LIMIT + 1):
        assert item.get_output(("pipe", i)) is not None
    assert 1 + len(item._output_slots) == OUTPUT_SLOT_LIMIT


def test_output_slots_also_obey_byte_cap(monkeypatch):
    from raw_alchemy.pipeline import cache_manager as cache_module

    monkeypatch.setattr(cache_module, "OUTPUT_CACHE_LIMIT_BYTES", 200)
    item = CachedImage("p", np.zeros((8, 8, 3), np.float32), None, None)
    for i in range(3):
        item.store_output(
            ("pipe", i), np.full((5, 5, 3), i, np.uint8), 0.0, (5, 5)
        )

    assert item.get_output(("pipe", 0)) is None
    assert item.get_output(("pipe", 1)) is not None
    assert item.get_output(("pipe", 2)) is not None


def test_output_slots_size_accounting_and_drop_stage():
    item = CachedImage("p", np.zeros((8, 8, 3), np.float32), None, None)
    item.store_output(("k", 1), np.zeros((16, 16, 3), np.uint8), 0.0, (8, 8))
    item.store_output(("k", 2), np.zeros((16, 16, 3), np.uint8), 0.0, (8, 8))
    item.update_size()

    expected_mb = 2 * (16 * 16 * 3) / (1024 * 1024)
    assert item.field_sizes_mb["output"] == pytest.approx(expected_mb)

    # T7.6 field-level eviction clears every slot in one stage.
    freed = item.drop_stage("output")
    assert freed == pytest.approx(expected_mb)
    assert item.output_uint8 is None
    assert item.output_key is None
    assert item.get_output(("k", 1)) is None
    assert item.get_output(("k", 2)) is None
    assert item.field_sizes_mb["output"] == 0.0
    assert item.drop_stage("output") == 0.0


# =====================================================================
# Dual executors: zoom across 1.0 / idle refine never cross-evict
# =====================================================================


def test_zoom_across_one_keeps_both_executor_caches(monkeypatch):
    processor, source, proxy = _proxy_processor(monkeypatch)
    path = processor.current_path

    emitted = []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append((img.copy(), size))
    )

    op_calls = _install_op_counter(monkeypatch)

    base = _case(
        {
            "exposure": 0.5,
            "wb_temp": 5.0,
            "saturation": 1.1,
            "viewport_size": (40, 40),
            "device_pixel_ratio": 1.0,
        }
    )
    ops_len = len(build_op_list(base))
    assert ops_len >= 3

    # First proxy render (zoom <= 1.0) and first full render (zoom > 1.0)
    # each run the op chain once, on their own executor.
    processor._do_process(ProcessRequest(path, dict(base, preview_zoom=0.5), 1))
    assert len(op_calls) == ops_len
    assert processor.last_preview_source == "proxy"
    processor._do_process(ProcessRequest(path, dict(base, preview_zoom=2.0), 2))
    assert len(op_calls) == 2 * ops_len
    assert processor.last_preview_source == "full"

    assert set(processor._preview_executors) == {"proxy", "full"}
    assert (
        processor._preview_executors["proxy"]
        is not processor._preview_executors["full"]
    )

    # Zoom ping-pong across 1.0, including fresh zoom values: from here on
    # no op kernel may re-run — repeats hit the uint8 output slots, new zoom
    # values hit the executor's GPU-resident final result (resample only).
    request_id = 3
    for zoom in (0.5, 2.0, 0.7, 3.0, 0.5, 2.0, 0.9, 4.0, 0.5, 2.0):
        processor._do_process(ProcessRequest(path, dict(base, preview_zoom=zoom), request_id))
        request_id += 1
    assert len(op_calls) == 2 * ops_len

    # Both executors kept their full cache generation through the switches.
    proxy_exec = processor._preview_executors["proxy"]
    full_exec = processor._preview_executors["full"]
    assert len(proxy_exec._prefix_cache) == ops_len
    assert len(full_exec._prefix_cache) == ops_len
    assert len(proxy_exec._final_cache) == 1
    assert len(full_exec._final_cache) == 1

    # Every request emitted exactly once, with mode-correct source sizes,
    # and repeats re-emitted identical cached images.
    assert len(emitted) == 12
    proxy_size = (proxy.shape[1], proxy.shape[0])
    full_size = (source.shape[1], source.shape[0])
    assert emitted[0][1] == proxy_size and emitted[2][1] == proxy_size
    assert emitted[1][1] == full_size and emitted[3][1] == full_size
    np.testing.assert_array_equal(emitted[2][0], emitted[0][0])  # zoom 0.5 repeat
    np.testing.assert_array_equal(emitted[3][0], emitted[1][0])  # zoom 2.0 repeat


def test_full_refine_preserves_proxy_caches_for_incremental_edit(monkeypatch):
    processor, _source, _proxy = _proxy_processor(monkeypatch)
    path = processor.current_path

    op_calls = _install_op_counter(monkeypatch)

    params_a = _case(
        {
            "exposure": 0.5,
            "wb_temp": 5.0,
            "saturation": 1.1,
            "viewport_size": (200, 200),
            "preview_zoom": 1.0,
        }
    )
    ops_a = build_op_list(params_a)

    # Interactive proxy render schedules the idle full refine.
    processor._do_process(ProcessRequest(path, dict(params_a), 1))
    assert op_calls == [op.name for op in ops_a]
    refine = processor._full_refine_request
    assert refine is not None

    # The refine runs on the *full* executor; the proxy executor's caches
    # must survive it untouched (no proxy/full cross-eviction — T7.4).
    proxy_exec = processor._preview_executors["proxy"]
    prefix_len_before = len(proxy_exec._prefix_cache)
    finals_before = dict(proxy_exec._final_cache)
    with processor.lock:
        processor._full_refine_request = None
    processor._do_process(refine)
    assert processor.last_preview_source == "full"
    assert processor._preview_executors["full"] is not proxy_exec
    assert len(op_calls) == 2 * len(ops_a)
    assert len(proxy_exec._prefix_cache) == prefix_len_before
    assert proxy_exec._final_cache == finals_before

    # The next slider move resumes incrementally on the proxy prefixes:
    # only the changed tail (sat_contrast onward) re-runs.
    params_b = dict(params_a, saturation=1.3)
    ops_b = build_op_list(params_b)
    shared = 0
    while shared < len(ops_a) and ops_a[shared] == ops_b[shared]:
        shared += 1
    assert 0 < shared < len(ops_b)

    del op_calls[:]
    processor._do_process(ProcessRequest(path, dict(params_b), 3))
    assert processor.last_preview_source == "proxy"
    assert op_calls == [op.name for op in ops_b[shared:]]


# =====================================================================
# Shared executor cache budget (proxy priority)
# =====================================================================


def test_trim_budget_is_shared_with_proxy_priority(monkeypatch):
    processor, source, proxy = _proxy_processor(monkeypatch)
    path = processor.current_path

    base = _case(
        {
            "exposure": 0.5,
            "wb_temp": 5.0,
            "saturation": 1.1,
            "viewport_size": (40, 40),
            "device_pixel_ratio": 1.0,
        }
    )
    processor._do_process(ProcessRequest(path, dict(base, preview_zoom=0.5), 1))
    processor._do_process(ProcessRequest(path, dict(base, preview_zoom=2.0), 2))

    proxy_exec = processor._preview_executors["proxy"]
    full_exec = processor._preview_executors["full"]
    proxy_bytes = proxy_exec.cache_bytes()
    full_bytes = full_exec.cache_bytes()
    assert proxy_bytes > 0 and full_bytes > 0
    full_entry = source.nbytes  # every full-executor stage has source shape

    # Budget covers the proxy caches plus one full-side entry: the proxy
    # executor (interactivity) is kept whole, the full side yields.
    budget = proxy_bytes + full_entry
    processor._trim_executor_caches(budget)
    assert proxy_exec.cache_bytes() == proxy_bytes
    assert full_exec.cache_bytes() <= full_entry
    assert proxy_exec.cache_bytes() + full_exec.cache_bytes() <= budget

    # A zero budget empties both sides.
    processor._trim_executor_caches(0)
    assert proxy_exec.cache_bytes() == 0
    assert full_exec.cache_bytes() == 0
