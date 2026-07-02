from pathlib import Path

import colour
import numpy as np

from raw_alchemy import config
from raw_alchemy.math_ops import white_balance_matrix
from raw_alchemy.pipeline.executor import ExportExecutor, PreviewExecutor
from raw_alchemy.pipeline.ops import Op, build_op_list


IDENTITY_CUBE = Path(__file__).with_name("golden") / "identity.cube"


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


def test_build_op_list_is_ordered_and_hashable():
    params = _case(
        {
            "rotation": 90,
            "flip_horizontal": True,
            "perspective_corners": ((0.03, 0.02), (0.98, 0.05), (0.92, 0.97), (0.06, 0.93)),
            "crop": (0.1, 0.1, 0.8, 0.7),
            "exposure": 0.4,
            "wb_temp": 12.0,
            "saturation": 1.15,
            "log_space": "F-Log2",
            "sharpen_strength": 0.2,
        }
    )

    ops = build_op_list(params)

    assert [op.name for op in ops] == [
        "geometry",
        "perspective",
        "crop",
        "exposure",
        "white_balance",
        "sat_contrast",
        "log_transform",
        "sharpen",
    ]
    assert isinstance(hash(tuple(ops)), int)


def test_build_op_list_preserves_zero_adjustment_values():
    ops = build_op_list(_case({"saturation": 0.0, "contrast": 0.0}))

    assert ("sat_contrast", (0.0, 0.0, config.WORKING_SPACE)) in [
        (op.name, op.params) for op in ops
    ]


def test_build_op_list_omits_zero_white_balance():
    ops = build_op_list(_base_params())

    assert "white_balance" not in [op.name for op in ops]


def test_build_op_list_uses_pq_output_for_hdr_export():
    ops = build_op_list(_case({"hdr_output": True}))

    assert ops[-1].name == "pq_out"
    assert ops[-1].params == (
        config.WORKING_SPACE,
        config.HDR_OUTPUT_COLOURSPACE,
        config.HDR_PQ_TRANSFER_FUNCTION,
        config.HDR_PEAK_NITS,
        config.HDR_PQ_MASTERING_NITS,
    )


def test_build_op_list_hdr_export_ignores_log_and_lut_for_pq_output():
    ops = build_op_list(
        _case(
            {
                "hdr_output": True,
                "log_space": "F-Log",
                "lut_path": str(IDENTITY_CUBE),
            }
        )
    )

    op_names = [op.name for op in ops]
    assert "log_transform" not in op_names
    assert "lut" not in op_names
    assert op_names[-1] == "pq_out"


def test_export_executor_white_balance_op_uses_cat_matrix():
    src = np.array(
        [
            [[0.18, 0.20, 0.22], [0.35, 0.30, 0.25]],
            [[0.08, 0.12, 0.16], [0.60, 0.45, 0.32]],
        ],
        dtype=np.float32,
    )
    matrix = white_balance_matrix(18.0, -10.0, config.WORKING_SPACE)
    expected = np.clip(src @ matrix.T, 0.0, 1.0).astype(np.float32)

    actual = ExportExecutor(src).run(
        [Op("white_balance", (18.0, -10.0, config.WORKING_SPACE))]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_export_executor_pq_out_matches_colour_reference():
    src = np.array(
        [
            [[0.05, 0.10, 0.15], [0.35, 0.30, 0.25]],
            [[0.80, 0.65, 0.40], [1.00, 0.90, 0.70]],
        ],
        dtype=np.float32,
    )
    op = Op(
        "pq_out",
        (
            config.WORKING_SPACE,
            config.HDR_OUTPUT_COLOURSPACE,
            config.HDR_PQ_TRANSFER_FUNCTION,
            config.HDR_PEAK_NITS,
            config.HDR_PQ_MASTERING_NITS,
        ),
    )

    actual = ExportExecutor(src).run([op])

    matrix = colour.matrix_RGB_to_RGB(
        colour.RGB_COLOURSPACES[config.WORKING_SPACE],
        colour.RGB_COLOURSPACES[config.HDR_OUTPUT_COLOURSPACE],
    )
    expected_linear = np.clip(src @ matrix.T, 0.0, None)
    expected = colour.cctf_encoding(
        expected_linear * (config.HDR_PEAK_NITS / config.HDR_PQ_MASTERING_NITS),
        function=config.HDR_PQ_TRANSFER_FUNCTION,
    )
    expected = np.clip(expected, 0.0, 1.0).astype(np.float32)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_preview_and_export_executors_match_for_randomized_params():
    rng = np.random.default_rng(1234)
    src = rng.uniform(0.02, 0.65, size=(64, 64, 3)).astype(np.float32)

    cases = [
        _case({}),
        _case({"exposure": 0.5, "saturation": 1.2, "contrast": 1.1}),
        _case({"exposure": -0.75, "wb_temp": 18.0, "wb_tint": -10.0}),
        _case({"highlight": -25.0, "shadow": 30.0, "contrast": 0.9}),
        _case({"rotation": 90, "flip_vertical": True, "crop": (0.05, 0.08, 0.7, 0.65)}),
        _case(
            {
                "perspective_corners": ((0.04, 0.01), (0.96, 0.08), (0.91, 0.96), (0.08, 0.9)),
                "crop": (0.1, 0.12, 0.72, 0.7),
                "saturation": 0.85,
            }
        ),
        _case({"log_space": "F-Log2", "lut_path": str(IDENTITY_CUBE), "exposure": 0.25}),
        _case(
            {
                "rotation": 270,
                "flip_horizontal": True,
                "perspective_corners": ((0.0, 0.05), (0.92, 0.0), (1.0, 0.92), (0.08, 1.0)),
                "crop": (0.05, 0.05, 0.82, 0.82),
                "log_space": "S-Log3",
                "lut_path": str(IDENTITY_CUBE),
                "wb_temp": -8.0,
                "wb_tint": 6.0,
                "highlight": 12.0,
                "shadow": -18.0,
                "saturation": 1.05,
                "contrast": 1.08,
            }
        ),
    ]

    preview = PreviewExecutor(src)
    export = ExportExecutor(src)

    for params in cases:
        ops = build_op_list(params)
        actual = preview.run(ops)
        expected = export.run(ops)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_image_processor_preview_entry_uses_unified_op_list(monkeypatch):
    from raw_alchemy.pipeline.cache_manager import CachedImage
    from raw_alchemy.pipeline.request import ProcessRequest
    from raw_alchemy.workers import image_processor as processor_module
    from raw_alchemy.workers.image_processor import ImageProcessor

    calls = []
    original_build_op_list = processor_module.build_op_list

    def spy_build_op_list(params):
        calls.append(params.copy())
        return original_build_op_list(params)

    monkeypatch.setattr(processor_module, "build_op_list", spy_build_op_list)

    params = _case({"exposure": 1.0})
    source = np.full((8, 8, 3), 0.18, dtype=np.float32)
    path = "synthetic.raw"
    processor = ImageProcessor()
    processor.cpu_linear = source
    processor.current_path = path
    processor.exif_data = None
    processor.cache_manager.put(path, CachedImage(path, source, None, None))

    emitted = []
    processor.result_ready.connect(
        lambda img, img_path, request_id, applied_ev, source_size: emitted.append(
            (img.copy(), img_path, request_id, applied_ev, source_size)
        )
    )

    processor._do_process(ProcessRequest(path, params, 42))

    assert calls and calls[0]["exposure"] == 1.0
    assert processor.last_applied_ev == 1.0
    assert len(emitted) == 1
    img, img_path, request_id, applied_ev, source_size = emitted[0]
    assert img.shape == (8, 8, 3)
    assert img_path == path
    assert request_id == 42
    assert applied_ev == 1.0
    assert source_size == (8, 8)


def test_image_processor_builds_interactive_proxy(monkeypatch):
    from raw_alchemy.workers import image_processor as processor_module
    from raw_alchemy.workers.image_processor import ImageProcessor

    monkeypatch.setattr(processor_module, "PROXY_MIN_SOURCE_PIXELS", 200)
    monkeypatch.setattr(processor_module, "PROXY_TARGET_PIXELS", 100)
    src = np.linspace(0.0, 1.0, 20 * 20 * 3, dtype=np.float32).reshape(20, 20, 3)

    proxy = ImageProcessor._make_proxy(src)

    assert proxy is not None
    assert proxy.dtype == np.float32
    assert proxy.flags["C_CONTIGUOUS"]
    assert proxy.shape == (10, 10, 3)


def test_image_processor_proxy_preview_schedules_full_refinement(monkeypatch):
    from raw_alchemy.pipeline.cache_manager import CachedImage
    from raw_alchemy.pipeline.request import ProcessRequest
    from raw_alchemy.workers import image_processor as processor_module
    from raw_alchemy.workers.image_processor import ImageProcessor

    monkeypatch.setattr(processor_module, "PROXY_MIN_SOURCE_PIXELS", 200)
    monkeypatch.setattr(processor_module, "PROXY_TARGET_PIXELS", 100)
    source = np.full((20, 20, 3), 0.18, dtype=np.float32)
    proxy = ImageProcessor._make_proxy(source)
    path = "synthetic.raw"

    processor = ImageProcessor()
    processor.cpu_linear = source
    processor.cpu_proxy_linear = proxy
    processor.current_path = path
    processor.exif_data = None
    processor.cache_manager.put(
        path,
        CachedImage(path, source, None, None, proxy_linear=proxy),
    )

    emitted = []
    processor.result_ready.connect(
        lambda img, img_path, request_id, applied_ev, source_size: emitted.append(
            (img.copy(), img_path, request_id, applied_ev, source_size)
        )
    )

    params = _case({"viewport_size": (200, 200), "preview_zoom": 1.0})
    processor._do_process(ProcessRequest(path, params, 7))

    assert processor.last_preview_source == "proxy"
    assert processor._full_refine_request is not None
    assert processor._full_refine_request.request_id == 7
    assert processor._full_refine_request.params["_force_full_preview"] is True
    assert emitted[-1][4] == (proxy.shape[1], proxy.shape[0])


def test_image_processor_high_frequency_preview_updates_are_latest_wins(monkeypatch):
    from raw_alchemy.pipeline.request import ProcessRequest
    from raw_alchemy.workers.image_processor import ImageProcessor

    processor = ImageProcessor()
    monkeypatch.setattr(processor, "isRunning", lambda: True)

    processor._full_refine_request = ProcessRequest(
        "synthetic.raw",
        {"_force_full_preview": True, "exposure": 0.0},
        1,
    )

    for idx in range(40):
        request_id = processor.update_preview(
            "synthetic.raw",
            _case(
                {
                    "viewport_size": (1600, 1000),
                    "preview_zoom": 1.0,
                    "exposure": idx / 10.0,
                    "saturation": 1.0 + idx / 100.0,
                }
            ),
        )

    with processor.lock:
        pending = processor.pending_request
        stale_refine = processor._full_refine_request

    assert request_id == 40
    assert stale_refine is None
    assert pending is not None
    assert pending.request_id == 40
    assert pending.path == "synthetic.raw"
    assert pending.params["exposure"] == 3.9
    assert np.isclose(pending.params["saturation"], 1.39)


def test_image_processor_denoise_toggle_has_no_residual_state(monkeypatch):
    from raw_alchemy.pipeline.cache_manager import CachedImage
    from raw_alchemy.pipeline.request import ProcessRequest
    from raw_alchemy.workers import image_processor as processor_module
    from raw_alchemy.workers.image_processor import ImageProcessor

    source = np.full((8, 8, 3), 0.18, dtype=np.float32)
    denoised = np.full((8, 8, 3), 0.36, dtype=np.float32)
    path = "synthetic.raw"

    monkeypatch.setattr(processor_module, "denoise_raw", lambda *args, **kwargs: denoised.copy())
    monkeypatch.setattr(processor_module, "denoise_clear_session", lambda: None)

    processor = ImageProcessor()
    processor.cpu_linear = source
    processor.current_path = path
    processor.exif_data = None
    processor.cache_manager.put(path, CachedImage(path, source, None, None))

    emitted = []
    processor.result_ready.connect(
        lambda img, *_args: emitted.append(img.copy())
    )

    on_outputs = []
    off_outputs = []
    for idx in range(10):
        for enabled, bucket in ((True, on_outputs), (False, off_outputs)):
            cached = processor.cache_manager.get(path)
            cached.output_key = None
            cached.output_uint8 = None
            params = _case({"denoise_enabled": enabled})
            processor._do_process(ProcessRequest(path, params, idx * 2 + int(enabled)))
            bucket.append(emitted[-1])

    for img in on_outputs[1:]:
        np.testing.assert_array_equal(img, on_outputs[0])
    for img in off_outputs[1:]:
        np.testing.assert_array_equal(img, off_outputs[0])
    assert not np.array_equal(on_outputs[0], off_outputs[0])
    assert processor.last_denoise_key is None
    np.testing.assert_allclose(processor.cpu_corrected, source)


def test_image_processor_cached_export_request_uses_worker_handler(monkeypatch):
    from raw_alchemy import core
    from raw_alchemy.workers.image_processor import ImageProcessor

    source = np.full((8, 8, 3), 0.18, dtype=np.float32)
    captured = {}
    processor = ImageProcessor()
    emitted = []
    processor.export_finished.connect(lambda success, msg: emitted.append((success, msg)))
    monkeypatch.setattr(processor, "isRunning", lambda: True)
    monkeypatch.setattr(core, "save_image", lambda img, *args, **kwargs: captured.setdefault("img", img.copy()))

    payload = {
        "cached_img": source,
        "output_path": "out.tif",
        "exif_data": {},
        "exif_metadata": None,
        "log_space": "None",
        "lut_path": None,
        "exposure": 0.0,
        "metering_mode": "matrix",
        "wb_temp": 0.0,
        "wb_tint": 0.0,
        "saturation": 1.0,
        "contrast": 1.0,
        "highlight": 0.0,
        "shadow": 0.0,
        "rotation": 0,
        "flip_horizontal": False,
        "flip_vertical": False,
        "perspective_corners": None,
        "crop": (0.0, 0.0, 1.0, 1.0),
        "sharpen_strength": 0.0,
    }

    request_id = processor.export_from_cache("synthetic.raw", payload)

    with processor.lock:
        request = processor._export_queue.pop(0)

    assert request.request_id == request_id
    assert request.params["_export"] is True

    processor._do_export(request)

    assert emitted == [(True, "")]
    expected = ExportExecutor().run(build_op_list(_base_params()), source.copy())
    np.testing.assert_allclose(captured["img"], expected, rtol=1e-6, atol=1e-6)


def test_image_processor_full_export_request_uses_worker_handler(monkeypatch):
    from raw_alchemy import orchestrator
    from raw_alchemy.workers.image_processor import ImageProcessor

    payload = {
        "input_path": "synthetic.raw",
        "output_path": "out.tif",
        "log_space": "None",
        "lut_path": None,
        "exposure": 0.0,
        "lens_correct": False,
        "custom_db_path": None,
        "metering_mode": "matrix",
        "jobs": 1,
        "logger_func": lambda _msg: None,
        "output_format": "tif",
    }
    captured = {}
    processor = ImageProcessor()
    emitted = []
    processor.export_finished.connect(lambda success, msg: emitted.append((success, msg)))
    monkeypatch.setattr(processor, "isRunning", lambda: True)
    monkeypatch.setattr(
        orchestrator,
        "process_path",
        lambda **kwargs: captured.setdefault("payload", kwargs.copy()),
    )

    request_id = processor.export_path("synthetic.raw", payload)

    with processor.lock:
        request = processor._export_queue.pop(0)

    assert request.request_id == request_id
    assert request.params["_export"] is True
    assert request.params["export_kind"] == "path"

    processor._do_export(request)

    assert emitted == [(True, "")]
    assert captured["payload"]["input_path"] == "synthetic.raw"
    assert captured["payload"]["output_path"] == "out.tif"


def _install_executor_spies(monkeypatch, preview):
    """Count op applications on `preview` and GPU clip calls (T7.1 guards)."""
    from raw_alchemy.pipeline import executor as executor_module

    op_calls = []
    original_apply = preview._apply_op

    def counting_apply(buf, op):
        op_calls.append(op.name)
        return original_apply(buf, op)

    monkeypatch.setattr(preview, "_apply_op", counting_apply)

    clip_calls = []
    original_clip = executor_module.clip_inplace

    def counting_clip(arr):
        clip_calls.append(1)
        return original_clip(arr)

    monkeypatch.setattr(executor_module, "clip_inplace", counting_clip)
    return op_calls, clip_calls


def test_preview_executor_same_source_rerun_hits_cache_without_gpu(monkeypatch):
    rng = np.random.default_rng(7)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.1}))
    assert len(ops) >= 3

    op_calls, clip_calls = _install_executor_spies(monkeypatch, preview)

    first = preview.run_result(ops, source=src)
    assert len(op_calls) == len(ops)
    assert len(clip_calls) == 1

    # Same source object + identical op list (e.g. only zoom/viewport changed):
    # no op kernel may re-run and no GPU clip round-trip may happen.
    second = preview.run_result(ops, source=src)
    assert len(op_calls) == len(ops)
    assert len(clip_calls) == 1
    assert second.applied_ev == first.applied_ev
    np.testing.assert_array_equal(second.image, first.image)


def test_preview_executor_prefix_cache_survives_tail_param_change(monkeypatch):
    rng = np.random.default_rng(11)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.1}))
    preview.run_result(ops, source=src)

    op_calls, _ = _install_executor_spies(monkeypatch, preview)

    # Change only the sat_contrast op: the shared [exposure, white_balance]
    # prefix must come from cache, so only the changed suffix re-runs.
    changed_ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.3}))
    assert changed_ops[:2] == ops[:2]
    preview.run_result(changed_ops, source=src)
    assert op_calls == [op.name for op in changed_ops[2:]]


def test_preview_executor_source_swap_clears_cache(monkeypatch):
    rng = np.random.default_rng(23)
    src_a = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    src_b = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src_a)
    ops = build_op_list(_case({"exposure": 0.5, "saturation": 1.1}))

    op_calls, clip_calls = _install_executor_spies(monkeypatch, preview)

    first = preview.run_result(ops, source=src_a)
    assert len(op_calls) == len(ops)

    # New source object: caches must be dropped and the full chain re-run.
    second = preview.run_result(ops, source=src_b)
    assert len(op_calls) == 2 * len(ops)
    assert len(clip_calls) == 2
    assert not np.array_equal(second.image, first.image)

    # And the caches are now keyed to the new source.
    third = preview.run_result(ops, source=src_b)
    assert len(op_calls) == 2 * len(ops)
    np.testing.assert_array_equal(third.image, second.image)


def test_set_source_reports_change_only_for_new_objects():
    rng = np.random.default_rng(31)
    src64 = rng.uniform(0.02, 0.65, size=(4, 4, 3))  # float64: forces a conversion copy
    preview = PreviewExecutor()

    assert preview.set_source(src64) is True
    # The internal float32 copy must not make the same input count as a change.
    assert preview.set_source(src64) is False

    src32 = src64.astype(np.float32)
    assert preview.set_source(src32) is True
    assert preview.set_source(src32) is False


def test_image_processor_zoom_only_change_hits_executor_cache(monkeypatch):
    from raw_alchemy.pipeline.cache_manager import CachedImage
    from raw_alchemy.pipeline.executor import _BaseExecutor
    from raw_alchemy.pipeline.request import ProcessRequest
    from raw_alchemy.workers.image_processor import ImageProcessor

    op_calls = []
    original_apply = _BaseExecutor._apply_op

    def counting_apply(self, buf, op):
        op_calls.append(op.name)
        return original_apply(self, buf, op)

    monkeypatch.setattr(_BaseExecutor, "_apply_op", counting_apply)

    params = _case(
        {
            "exposure": 1.0,
            "wb_temp": 5.0,
            "saturation": 1.1,
            "viewport_size": (32, 32),
            "preview_zoom": 2.0,
            "device_pixel_ratio": 1.0,
        }
    )
    source = np.full((64, 64, 3), 0.18, dtype=np.float32)
    path = "synthetic.raw"
    processor = ImageProcessor()
    processor.cpu_linear = source
    processor.current_path = path
    processor.exif_data = None
    processor.cache_manager.put(path, CachedImage(path, source, None, None))

    processor._do_process(ProcessRequest(path, dict(params), 1))
    first_run_ops = len(op_calls)
    assert first_run_ops > 0

    # Zoom-only changes keep the op list identical: from the *second* request
    # onward the executor cache must fully absorb the pipeline (T7.1).
    processor._do_process(ProcessRequest(path, dict(params, preview_zoom=3.0), 2))
    assert len(op_calls) == first_run_ops
    processor._do_process(ProcessRequest(path, dict(params, preview_zoom=4.0), 3))
    assert len(op_calls) == first_run_ops

    # A tail slider change re-runs only the changed suffix, not the full chain.
    processor._do_process(ProcessRequest(path, dict(params, saturation=1.3), 4))
    suffix_ops = len(op_calls) - first_run_ops
    assert 0 < suffix_ops < first_run_ops


def test_preview_executor_trim_respects_budget_and_keeps_short_prefixes(monkeypatch):
    rng = np.random.default_rng(41)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.1}))
    assert len(ops) >= 3
    result = preview.run_result(ops, source=src)

    entry_bytes = result.image.nbytes  # all cached results share this shape
    assert preview.cache_bytes() == (len(ops) + 1) * entry_bytes  # prefixes + final

    # Budget for the final result plus one prefix: the longest prefixes must
    # be dropped first, keeping the (expensive-to-recompute) shortest one.
    budget = 2 * entry_bytes
    freed = preview.trim(budget)
    assert freed == (len(ops) - 1) * entry_bytes
    assert preview.cache_bytes() <= budget
    assert set(preview._prefix_lengths.values()) == {1}

    # The surviving final result still powers the full-hit path.
    rerun = preview.run_result(ops, source=src)
    np.testing.assert_array_equal(rerun.image, result.image)

    # A tail change after trim resumes from the surviving length-1 prefix.
    op_calls, _ = _install_executor_spies(monkeypatch, preview)
    changed_ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.3}))
    preview.run_result(changed_ops, source=src)
    assert op_calls == [op.name for op in changed_ops[1:]]


def test_preview_executor_trim_to_zero_then_recompute():
    rng = np.random.default_rng(43)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    ops = build_op_list(_case({"exposure": 0.25, "saturation": 1.2}))
    first = preview.run_result(ops, source=src)

    freed = preview.trim(0)
    assert freed > 0
    assert preview.cache_bytes() == 0
    assert preview._prefix_cache == {}
    assert preview._prefix_lengths == {}
    assert preview._final_cache == {}

    again = preview.run_result(ops, source=src)
    np.testing.assert_array_equal(again.image, first.image)


def test_image_processor_applies_executor_cache_budget(monkeypatch):
    from raw_alchemy.pipeline.cache_manager import CachedImage
    from raw_alchemy.pipeline.executor import PreviewExecutor as ExecutorClass
    from raw_alchemy.pipeline.request import ProcessRequest
    from raw_alchemy.workers.image_processor import ImageProcessor

    trim_calls = []
    original_trim = ExecutorClass.trim

    def spy_trim(self, budget_bytes):
        trim_calls.append(budget_bytes)
        return original_trim(self, budget_bytes)

    monkeypatch.setattr(ExecutorClass, "trim", spy_trim)

    source = np.full((8, 8, 3), 0.18, dtype=np.float32)
    path = "synthetic.raw"
    processor = ImageProcessor()
    processor.cpu_linear = source
    processor.current_path = path
    processor.exif_data = None
    processor.cache_manager.put(path, CachedImage(path, source, None, None))

    processor._do_process(ProcessRequest(path, _case({"exposure": 1.0}), 1))

    assert trim_calls == [int(config.EXECUTOR_CACHE_LIMIT_MB) * 1024 * 1024]


# =====================================================================
# T7.2 — GPU residency: prefix cache holds GPU buffers, zero host
# round-trips between ops, only the final uint8 is read back.
# =====================================================================


def _install_transfer_spies(monkeypatch):
    """Count every host<->device transfer made through GpuImage."""
    from raw_alchemy.gpu_buffer import GpuImage

    uploads = []
    downloads = []
    original_upload = GpuImage.upload
    original_to_numpy = GpuImage.to_numpy

    def spy_upload(self, np_array):
        uploads.append(np_array.nbytes)
        return original_upload(self, np_array)

    def spy_to_numpy(self):
        downloads.append(self.nbytes)
        return original_to_numpy(self)

    monkeypatch.setattr(GpuImage, "upload", spy_upload)
    monkeypatch.setattr(GpuImage, "to_numpy", spy_to_numpy)
    return uploads, downloads


def test_preview_executor_run_stays_gpu_resident(monkeypatch):
    rng = np.random.default_rng(51)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.1}))
    assert len(ops) >= 3

    uploads, downloads = _install_transfer_spies(monkeypatch)

    result = preview.run_result(ops, source=src)
    # Exactly one host->device transfer (source ingestion), zero device->host
    # transfers: prefix snapshots are GPU-to-GPU copies (T7.2).
    assert len(uploads) == 1
    assert downloads == []
    assert result.gpu is not None and result.gpu.valid

    # The host image is downloaded lazily, exactly once, on demand.
    image = result.image
    assert len(downloads) == 1
    assert image.shape == src.shape

    # A tail-only change resumes from the GPU-resident prefix: no upload of
    # the source and still no readback of intermediate stages.
    changed_ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.3}))
    assert changed_ops[:2] == ops[:2]
    preview.run_result(changed_ops, source=src)
    assert len(uploads) == 1
    assert len(downloads) == 1


def test_pipeline_result_survives_cache_eviction_and_pool_reuse():
    rng = np.random.default_rng(53)
    src_a = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    src_b = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.1}))
    expected = ExportExecutor().run(ops, src_a.copy())

    preview = PreviewExecutor(src_a)
    held = preview.run_result(ops, source=src_a)

    # Swapping the source clears the executor caches and recycles buffers
    # through the pool; the handed-out result must keep its own GPU buffer
    # alive and untouched (no use-after-release).
    preview.run_result(ops, source=src_b)
    preview.run_result(build_op_list(_case({"exposure": 1.5})), source=src_b)

    np.testing.assert_allclose(held.image, expected, rtol=1e-5, atol=1e-5)


def test_preview_executor_drops_stale_prefix_generations():
    rng = np.random.default_rng(59)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)

    ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.1}))
    preview.run_result(ops, source=src)
    assert len(preview._prefix_cache) == len(ops)

    # Only the shared [exposure, white_balance] prefix survives a tail change;
    # stale generations are recycled instead of accumulating VRAM.
    changed_ops = build_op_list(_case({"exposure": 0.5, "wb_temp": 5.0, "saturation": 1.3}))
    preview.run_result(changed_ops, source=src)
    assert len(preview._prefix_cache) == len(changed_ops)
    assert set(preview._prefix_lengths.values()) == set(range(1, len(changed_ops) + 1))

    entry_bytes = src.nbytes
    assert preview.cache_bytes() == (len(changed_ops) + 1) * entry_bytes


def test_image_processor_reads_back_only_final_uint8(monkeypatch):
    from raw_alchemy.pipeline.cache_manager import CachedImage
    from raw_alchemy.pipeline.request import ProcessRequest
    from raw_alchemy.workers.image_processor import ImageProcessor

    params = _case(
        {
            "exposure": 1.0,
            "wb_temp": 5.0,
            "saturation": 1.1,
            # Viewport >= source: target size == source size, so the output
            # is a direct float->uint8 conversion (bit-exact comparison below).
            "viewport_size": (64, 64),
            "preview_zoom": 1.0,
            "device_pixel_ratio": 1.0,
        }
    )
    source = np.random.default_rng(61).uniform(0.05, 0.5, size=(32, 32, 3)).astype(np.float32)
    path = "synthetic.raw"
    processor = ImageProcessor()
    processor.cpu_linear = source
    processor.current_path = path
    processor.exif_data = None
    processor.cache_manager.put(path, CachedImage(path, source, None, None))

    emitted = []
    processor.result_ready.connect(
        lambda img, *_args: emitted.append(img.copy())
    )

    uploads, downloads = _install_transfer_spies(monkeypatch)

    processor._do_process(ProcessRequest(path, dict(params), 1))

    # One upload (pipeline source), zero float32 readbacks: the preview is
    # converted to uint8 on-device and only the small uint8 image crosses
    # back to the host (T7.2).
    assert len(uploads) == 1
    assert downloads == []
    assert len(emitted) == 1
    assert emitted[0].dtype == np.uint8

    # Full-hit rerun (zoom-only style): still zero float32 traffic.
    processor._do_process(ProcessRequest(path, dict(params), 2))
    assert len(uploads) == 1
    assert downloads == []

    # The emitted uint8 matches the executor's post-clip output converted on
    # the CPU: round(clip(x) * 255).
    expected_float = ExportExecutor().run(build_op_list(params), source.copy())
    expected_uint8 = np.floor(
        np.clip(expected_float, 0.0, 1.0).astype(np.float32) * 255.0 + 0.5
    ).astype(np.uint8)
    assert emitted[0].shape == expected_uint8.shape
    np.testing.assert_array_equal(emitted[0], expected_uint8)


def test_repeated_slider_interactions_keep_cache_and_pool_bounded():
    """T7.2 acceptance guard: continuous tail-op changes must not grow VRAM.

    Both the executor caches (latest prefix generation + single final) and
    the buffer pool (recycled scratch) stay bounded across many interactions.
    """
    from raw_alchemy.gpu_buffer import gpu_pool

    gpu_pool().clear()
    rng = np.random.default_rng(67)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    entry_bytes = src.nbytes

    ops_len = None
    for i in range(50):
        ops = build_op_list(
            _case({"exposure": 0.5, "saturation": 1.05 + (i % 7) * 0.05})
        )
        if ops_len is None:
            ops_len = len(ops)
        assert len(ops) == ops_len
        preview.run_result(ops, source=src)

        assert preview.cache_bytes() <= (ops_len + 1) * entry_bytes
        assert len(preview._prefix_cache) <= ops_len

    # Steady state: exactly one generation of prefixes plus one final, and a
    # small constant number of recycled buffers in the pool.
    assert preview.cache_bytes() == (ops_len + 1) * entry_bytes
    assert gpu_pool().free_bytes() <= 8 * entry_bytes
    gpu_pool().clear()
