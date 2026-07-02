"""T7.3 — cooperative in-flight cancellation + preload slimming + refine gating.

Covers:
- PreviewExecutor.run_result(should_abort=...): abort at op boundaries,
  completed prefixes stay cached, superseding runs resume from them.
- ImageProcessor: a pending user request aborts the in-flight run silently
  (no result_ready, no error_occurred), and the next request starts on the
  cached prefix.
- _do_preload: no full-size lens-correction precompute; stage boundaries
  yield to pending user requests while keeping completed work cached.
- Full-refine trigger tightening: a due refine is postponed while queued
  work exists or a user interaction happened within the idle window.
"""
import sys
import time
import types

import numpy as np
import pytest

from raw_alchemy.pipeline.cache_manager import CachedImage
from raw_alchemy.pipeline.executor import (
    ExportExecutor,
    PipelineAborted,
    PreviewExecutor,
    _BaseExecutor,
)
from raw_alchemy.pipeline.ops import build_op_list
from raw_alchemy.pipeline.request import ProcessRequest
from raw_alchemy.workers import image_processor as processor_module
from raw_alchemy.workers.image_processor import (
    FULL_REFINE_IDLE_SECONDS,
    ImageProcessor,
)


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


# =====================================================================
# Executor-level cooperative abort
# =====================================================================


def test_preview_executor_abort_at_op_boundary_keeps_completed_prefixes(monkeypatch):
    rng = np.random.default_rng(71)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.1}))
    assert len(ops) >= 3

    op_calls = []
    original_apply = preview._apply_op

    def counting_apply(buf, op):
        op_calls.append(op.name)
        return original_apply(buf, op)

    monkeypatch.setattr(preview, "_apply_op", counting_apply)

    # Abort as soon as two ops have completed: the third op must never run.
    with pytest.raises(PipelineAborted):
        preview.run_result(ops, source=src, should_abort=lambda: len(op_calls) >= 2)

    assert op_calls == [op.name for op in ops[:2]]
    # Completed stages stay cached; no (stale) final result is kept.
    assert len(preview._prefix_cache) == 2
    assert set(preview._prefix_lengths.values()) == {1, 2}
    assert preview._final_cache == {}

    # The superseding run resumes from the kept prefixes: only the remaining
    # ops execute, and the output matches a from-scratch reference run.
    result = preview.run_result(ops, source=src)
    assert op_calls == [op.name for op in ops[:2]] + [op.name for op in ops[2:]]
    expected = ExportExecutor().run(ops, src.copy())
    np.testing.assert_allclose(result.image, expected, rtol=1e-5, atol=1e-5)


def test_preview_executor_abort_before_start_leaves_caches_untouched():
    rng = np.random.default_rng(73)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    ops = build_op_list(_case({"exposure": 0.5, "saturation": 1.1}))
    first = preview.run_result(ops, source=src)
    bytes_before = preview.cache_bytes()

    # A run that is already superseded on entry aborts before touching any
    # cache (the previous final result survives for future full hits).
    changed_ops = build_op_list(_case({"exposure": 0.5, "saturation": 1.3}))
    with pytest.raises(PipelineAborted):
        preview.run_result(changed_ops, source=src, should_abort=lambda: True)
    assert preview.cache_bytes() == bytes_before

    # Full hits are free — they are served even when an abort is requested.
    rerun = preview.run_result(ops, source=src, should_abort=lambda: True)
    np.testing.assert_array_equal(rerun.image, first.image)


# =====================================================================
# Worker-level: pending user request preempts the in-flight run
# =====================================================================


def test_worker_inflight_run_aborts_and_next_request_resumes_from_prefix(monkeypatch):
    source = np.full((32, 32, 3), 0.18, dtype=np.float32)
    path = "synthetic.raw"
    processor = _make_processor(path, source)

    emitted, errors = [], []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append(request_id)
    )
    processor.error_occurred.connect(lambda msg: errors.append(msg))

    slow_params = _case({"exposure": 1.0, "wb_temp": 5.0, "saturation": 1.1})
    new_params = _case({"exposure": 1.0, "wb_temp": 5.0, "saturation": 1.3})
    ops = build_op_list(slow_params)
    new_ops = build_op_list(new_params)
    assert ops[:2] == new_ops[:2]  # shared [exposure, white_balance] prefix

    op_calls = []
    original_apply = _BaseExecutor._apply_op

    def interrupting_apply(self, buf, op):
        op_calls.append(op.name)
        if len(op_calls) == 1:
            # A new user request arrives while op 1 of the slow run executes.
            with processor.lock:
                processor.pending_request = ProcessRequest(path, dict(new_params), 2)
        return original_apply(self, buf, op)

    monkeypatch.setattr(_BaseExecutor, "_apply_op", interrupting_apply)

    processor._do_process(ProcessRequest(path, dict(slow_params), 1))

    # Aborted at the first op boundary: exactly one op ran, nothing was
    # emitted (result_ready contract intact) and no error was raised.
    assert op_calls == [ops[0].name]
    assert emitted == []
    assert errors == []
    # The completed prefix survived the abort.
    assert len(processor.preview_executor._prefix_cache) == 1

    # The superseding request starts immediately and resumes from the cached
    # prefix: only the ops after the shared prefix execute.
    with processor.lock:
        follow_up = processor.pending_request
        processor.pending_request = None
    processor._do_process(follow_up)

    assert emitted == [2]
    assert errors == []
    assert op_calls == [ops[0].name] + [op.name for op in new_ops[1:]]


def test_worker_full_refine_run_is_preempted_silently(monkeypatch):
    monkeypatch.setattr(processor_module, "PROXY_MIN_SOURCE_PIXELS", 200)
    monkeypatch.setattr(processor_module, "PROXY_TARGET_PIXELS", 100)
    source = np.full((20, 20, 3), 0.18, dtype=np.float32)
    proxy = ImageProcessor._make_proxy(source)
    path = "synthetic.raw"
    processor = _make_processor(path, source, proxy=proxy)

    emitted, errors = [], []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append(request_id)
    )
    processor.error_occurred.connect(lambda msg: errors.append(msg))

    # Proxy render schedules the idle full refine.
    params = _case(
        {"viewport_size": (200, 200), "preview_zoom": 1.0, "exposure": 0.5}
    )
    processor._do_process(ProcessRequest(path, params, 1))
    assert emitted == [1]
    refine = processor._full_refine_request
    assert refine is not None

    op_calls = []
    original_apply = _BaseExecutor._apply_op

    def interrupting_apply(self, buf, op):
        op_calls.append(op.name)
        if len(op_calls) == 1:
            with processor.lock:
                processor.pending_request = ProcessRequest(path, dict(params), 3)
        return original_apply(self, buf, op)

    monkeypatch.setattr(_BaseExecutor, "_apply_op", interrupting_apply)

    # The refine goes in flight, then a new interaction arrives: the refine
    # must abort at the op boundary without emitting anything.
    with processor.lock:
        processor._full_refine_request = None
    processor._do_process(refine)

    assert emitted == [1]  # no refine emission
    assert errors == []
    assert len(op_calls) == 1  # stopped at the first op boundary


# =====================================================================
# Preload slimming + staged abort
# =====================================================================


def test_preload_no_longer_precomputes_full_lens_correction(monkeypatch):
    monkeypatch.setattr(processor_module, "PROXY_MIN_SOURCE_PIXELS", 200)
    monkeypatch.setattr(processor_module, "PROXY_TARGET_PIXELS", 100)

    processor = ImageProcessor()
    src = np.linspace(0.0, 1.0, 20 * 20 * 3, dtype=np.float32).reshape(20, 20, 3)
    exif = {
        "camera_maker": "Maker",
        "camera_model": "Model",
        "lens_maker": "Lens",
        "lens_model": "50mm",
        "focal_length": 50,
        "aperture": 2.8,
        "crop_factor": 1.0,
    }
    monkeypatch.setattr(
        processor, "_rawpy_to_prophoto", lambda path: (src, exif, {"meta": 1})
    )

    # If preload ever calls into lens correction again, this stub records it
    # (and no real lensfun/vendored library is touched by the test).
    lens_calls = []
    fake_lensfun = types.SimpleNamespace(
        compute_lens_distortion_map=lambda *a, **k: lens_calls.append(1)
    )
    monkeypatch.setitem(sys.modules, "raw_alchemy.lensfun_wrapper", fake_lensfun)

    processor._do_preload(ProcessRequest("neighbor.raw", {"_preload": True}, -1))

    cached = processor.cache_manager.get("neighbor.raw")
    assert cached is not None
    assert cached.linear_data is src
    assert cached.proxy_linear is not None
    assert cached.proxy_linear.shape == (10, 10, 3)
    # Neighbor preloads never precompute the full-size corrected image.
    assert lens_calls == []
    assert cached.corrected_data is None
    assert cached.proxy_corrected_data is None
    assert cached.lens_key is None


def test_preload_abandons_remaining_stages_when_user_request_arrives(monkeypatch):
    processor = ImageProcessor()
    src = np.full((20, 20, 3), 0.25, dtype=np.float32)

    def decode(path):
        # A user click lands while the RAW decode stage is running.
        with processor.lock:
            processor.pending_request = ProcessRequest("clicked.raw", {"_load": True}, 9)
        return src, None, None

    monkeypatch.setattr(processor, "_rawpy_to_prophoto", decode)

    proxy_calls = []
    monkeypatch.setattr(
        ImageProcessor, "_make_proxy", staticmethod(lambda img: proxy_calls.append(1))
    )

    processor._do_preload(ProcessRequest("neighbor.raw", {"_preload": True}, -1))

    # The completed decode stage still lands in the cache; the proxy stage
    # (and everything after it) is abandoned so the click is served sooner.
    cached = processor.cache_manager.get("neighbor.raw")
    assert cached is not None
    assert cached.linear_data is src
    assert cached.proxy_linear is None
    assert proxy_calls == []
    with processor.lock:
        assert processor.pending_request is not None  # untouched, served next


# =====================================================================
# Full-refine trigger tightening
# =====================================================================


def test_take_next_request_prefers_user_request_and_clears_preloads():
    processor = ImageProcessor()
    user = ProcessRequest("a.raw", {"exposure": 1.0}, 3)
    with processor.lock:
        processor.pending_request = user
        processor._preload_queue.append(ProcessRequest("b.raw", {"_preload": True}, -1))

    assert processor._take_next_request() is user
    with processor.lock:
        assert processor.pending_request is None
        assert processor._preload_queue == []


def test_due_full_refine_is_postponed_while_queue_has_work():
    processor = ImageProcessor()
    refine = ProcessRequest("a.raw", {"_force_full_preview": True}, 5)
    preload = ProcessRequest("b.raw", {"_preload": True}, -1)
    now = time.perf_counter()
    with processor.lock:
        processor._full_refine_request = refine
        processor._full_refine_due = now - 1.0  # overdue
        processor._last_interaction = now - 10.0  # no recent interaction
        processor._preload_queue.append(preload)

    # Queued work wins; the due refine is postponed (not dropped) by a fresh
    # idle window, so it will not fire right on the heels of the preload.
    assert processor._take_next_request() is preload
    assert processor._full_refine_request is refine
    assert processor._full_refine_due >= now + FULL_REFINE_IDLE_SECONDS

    # Still inside the restarted idle window: nothing to run.
    assert processor._take_next_request() is None

    # Once the window elapses with an empty queue, the refine finally runs.
    with processor.lock:
        processor._full_refine_due = time.perf_counter() - 0.001
    assert processor._take_next_request() is refine
    assert processor._full_refine_request is None


def test_due_full_refine_is_postponed_by_recent_interaction():
    processor = ImageProcessor()
    refine = ProcessRequest("a.raw", {"_force_full_preview": True}, 5)
    now = time.perf_counter()
    with processor.lock:
        processor._full_refine_request = refine
        processor._full_refine_due = now - 0.5  # overdue
        processor._last_interaction = now  # interaction just happened

    # Interaction within the idle window postpones the refine.
    assert processor._take_next_request() is None
    assert processor._full_refine_request is refine
    assert (
        processor._full_refine_due
        >= processor._last_interaction + FULL_REFINE_IDLE_SECONDS
    )

    # After a full quiet window the refine is picked up.
    with processor.lock:
        processor._last_interaction = (
            time.perf_counter() - FULL_REFINE_IDLE_SECONDS - 0.01
        )
        processor._full_refine_due = time.perf_counter() - 0.001
    assert processor._take_next_request() is refine


# =====================================================================
# m7 — shutdown participates in the cooperative abort predicate
# =====================================================================


def test_abort_predicate_includes_shutdown_stop():
    """stop_and_cleanup() sets _should_stop; the T7.3 abort predicate must
    report it so an in-flight run is cancelled at the next op boundary
    instead of running to completion past the shutdown wait timeout."""
    processor = ImageProcessor()
    assert processor._interactive_abort_requested() is False
    processor._should_stop = True
    assert processor._interactive_abort_requested() is True


def test_shutdown_aborts_inflight_run_silently(monkeypatch):
    source = np.full((32, 32, 3), 0.18, dtype=np.float32)
    path = "synthetic.raw"
    processor = _make_processor(path, source)

    emitted, errors = [], []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append(request_id)
    )
    processor.error_occurred.connect(lambda msg: errors.append(msg))

    params = _case({"exposure": 1.0, "wb_temp": 5.0, "saturation": 1.1})
    ops = build_op_list(params)
    assert len(ops) >= 2

    op_calls = []
    original_apply = _BaseExecutor._apply_op

    def stopping_apply(self, buf, op):
        op_calls.append(op.name)
        if len(op_calls) == 1:
            # stop_and_cleanup() runs on the GUI thread mid-render.
            processor._should_stop = True
        return original_apply(self, buf, op)

    monkeypatch.setattr(_BaseExecutor, "_apply_op", stopping_apply)

    processor._do_process(ProcessRequest(path, dict(params), 1))

    # Aborted at the first op boundary: the shutdown request cancels the
    # in-flight run (silent drop — no result, no error) so the worker loop
    # can observe _should_stop instead of finishing a long render first.
    assert op_calls == [ops[0].name]
    assert emitted == []
    assert errors == []
