"""T7.5 — zoom>fit visible-region (ROI) rendering.

Covers:
- ROI resolution from the viewport's visible rect: margin expansion,
  pixel-grid snapping, and the gates (fit view, missing viewport info,
  forced full refine, near-full coverage fallback).
- The ROI op is injected between the shape ops and the color ops and only
  narrows the processed region: a ROI render is pixel-identical to the same
  crop of a full-frame render, output at native 1:1 resolution.
- result_ready reports the *full* frame as source_size (viewport geometry
  stays stable) with the ROI pixel rect attached; the multi-slot output
  cache round-trips both.
- Pans inside the ROI margin re-emit from the output cache with zero op
  re-runs; pans beyond the margin re-run only the ROI-side chain; the
  color-pipeline key never changes with the ROI (T7.4 key separation).
- Viewport-side geometry: visible-rect inversion, ROI overlay placement and
  the pan-out-of-coverage predicate that drives incremental re-renders.
"""
import numpy as np
import pytest

from raw_alchemy.pipeline.cache_manager import CachedImage
from raw_alchemy.pipeline.executor import _BaseExecutor
from raw_alchemy.pipeline.ops import build_op_list
from raw_alchemy.pipeline.request import ProcessRequest
from raw_alchemy.workers.image_processor import ImageProcessor, RoiSourceSize


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


def _make_processor(path="synthetic.raw", size=256):
    source = np.linspace(0.05, 0.85, size * size * 3, dtype=np.float32).reshape(
        size, size, 3
    )
    processor = ImageProcessor()
    processor.cpu_linear = source
    processor.cpu_proxy_linear = None  # full path only: zoom>1 previews
    processor.current_path = path
    processor.exif_data = None
    processor.cache_manager.put(path, CachedImage(path, source, None, None))
    return processor, source


def _install_op_counter(monkeypatch):
    op_calls = []
    original_apply = _BaseExecutor._apply_op

    def counting_apply(self, buf, op):
        op_calls.append(op.name)
        return original_apply(self, buf, op)

    monkeypatch.setattr(_BaseExecutor, "_apply_op", counting_apply)
    return op_calls


def _view_params(visible=None, zoom=4.0):
    params = _case(
        {
            "exposure": 0.7,
            "viewport_size": (64, 64),
            "preview_zoom": zoom,
            "device_pixel_ratio": 1.0,
        }
    )
    if visible is not None:
        params["preview_visible_rect"] = visible
    return params


# =====================================================================
# ROI resolution: dims mirror, gates, margin + snapping
# =====================================================================


def test_pipeline_output_dims_mirrors_shape_ops():
    dims = ImageProcessor._pipeline_output_dims
    assert dims(400, 300, _base_params()) == (400, 300)
    assert dims(400, 300, _case({"rotation": 90})) == (300, 400)
    assert dims(400, 300, _case({"rotation": 180})) == (400, 300)
    # Crop snapping replicated from apply_crop_gpu: int() truncation.
    assert dims(400, 300, _case({"crop": (0.25, 0.25, 0.5, 0.5)})) == (200, 150)
    # Rotation applies before the crop (build_op_list order).
    assert dims(400, 300, _case({"rotation": 90, "crop": (0.0, 0.0, 0.5, 0.5)})) == (
        150,
        200,
    )


def test_compute_preview_roi_margin_and_grid_snapping():
    processor, _source = _make_processor()

    roi_info = processor._compute_preview_roi(
        _view_params(visible=(0.375, 0.375, 0.625, 0.625), zoom=4.0)
    )
    assert roi_info is not None
    roi_rect, full_size = roi_info
    assert full_size == (256, 256)
    # Visible span 0.25 + 25% margin per side => [0.3125, 0.6875] => pixels
    # [80, 176] snapped outward to the 32px grid => [64, 192].
    assert roi_rect == (64, 64, 128, 128)

    # The ROI must cover the visible region (plus margins) and stay in-frame.
    x, y, w, h = roi_rect
    assert 0 <= x and x + w <= 256 and 0 <= y and y + h <= 256
    assert x <= int(0.375 * 256) and x + w >= int(0.625 * 256)

    # Sub-grid pan jitter maps to the same snapped rect (stable cache keys).
    jitter = processor._compute_preview_roi(
        _view_params(visible=(0.372, 0.375, 0.622, 0.625), zoom=4.0)
    )
    assert jitter is not None and jitter[0] == roi_rect


def test_compute_preview_roi_gates():
    processor, _source = _make_processor()
    visible = (0.375, 0.375, 0.625, 0.625)

    # Fit view: never a ROI.
    assert processor._compute_preview_roi(_view_params(visible, zoom=1.0)) is None
    # No visible rect / no viewport info: legacy full-frame path.
    assert processor._compute_preview_roi(_view_params(None, zoom=4.0)) is None
    no_vp = _view_params(visible, zoom=4.0)
    no_vp.pop("viewport_size")
    assert processor._compute_preview_roi(no_vp) is None
    # Idle full refine must stay full-frame.
    forced = _view_params(visible, zoom=4.0)
    forced["_force_full_preview"] = True
    assert processor._compute_preview_roi(forced) is None
    # Near-full coverage: cropping saves nothing, render the whole frame.
    assert (
        processor._compute_preview_roi(_view_params((0.0, 0.0, 1.0, 1.0), zoom=1.5))
        is None
    )


def test_roi_op_injected_before_color_ops():
    params = _case({"rotation": 90, "crop": (0.1, 0.1, 0.8, 0.8), "wb_temp": 5.0})
    ops = build_op_list(params)
    roi_ops = ImageProcessor._insert_roi_op(ops, (32, 0, 64, 96))

    names = [op.name for op in roi_ops]
    assert names.index("roi") == names.index("exposure") - 1
    assert names.index("roi") > names.index("crop")
    assert roi_ops[names.index("roi")].params == (32, 0, 64, 96)
    # The original list is not mutated (prefix keys of other runs intact).
    assert [op.name for op in ops].count("roi") == 0


def test_view_key_carries_roi_pipeline_key_does_not():
    params = _view_params((0.375, 0.375, 0.625, 0.625))
    with_roi = dict(params, _preview_roi=(64, 64, 128, 128))
    other_roi = dict(params, _preview_roi=(96, 96, 128, 128))

    assert ImageProcessor._make_pipeline_key(with_roi) == ImageProcessor._make_pipeline_key(
        other_roi
    )
    assert ImageProcessor._make_view_key(with_roi) != ImageProcessor._make_view_key(
        other_roi
    )
    assert ImageProcessor._make_view_key(params) != ImageProcessor._make_view_key(
        with_roi
    )


# =====================================================================
# End-to-end ROI render through _do_process
# =====================================================================


def test_roi_render_matches_crop_of_full_render_at_native_resolution():
    processor, _source = _make_processor()
    path = processor.current_path

    emitted = []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append((img.copy(), size))
    )

    # Legacy full-frame render (no visible rect): zoom 4 at fit 0.25 => 1:1.
    processor._do_process(ProcessRequest(path, _view_params(None, zoom=4.0), 1))
    assert len(emitted) == 1
    full_img, full_size = emitted[0]
    assert full_img.shape == (256, 256, 3)
    assert full_size == (256, 256)
    assert getattr(full_size, "roi_rect", None) is None

    # ROI render of the center: native resolution, full-frame source_size,
    # ROI rect attached, and pixel-identical to the full render's crop.
    processor._do_process(
        ProcessRequest(path, _view_params((0.375, 0.375, 0.625, 0.625), zoom=4.0), 2)
    )
    assert len(emitted) == 2
    roi_img, roi_size = emitted[1]
    assert roi_img.shape == (128, 128, 3)
    assert roi_size == (256, 256)  # tuple-compatible full-frame size
    assert isinstance(roi_size, RoiSourceSize)
    assert roi_size.roi_rect == (64, 64, 128, 128)
    np.testing.assert_array_equal(roi_img, full_img[64:192, 64:192])

    assert processor.last_preview_source == "full"
    # ROI previews schedule the idle native-base refine (T7.10) with the ROI
    # state stripped, so the base map converges to the frame at pipeline
    # resolution and later zoom/pan need no re-render.
    refine = processor._full_refine_request
    assert refine is not None
    assert refine.params.get('_force_full_preview') is True
    assert '_preview_roi' not in refine.params
    assert '_roi_full_size' not in refine.params


def test_roi_pan_within_margin_hits_cache_and_beyond_reruns_tail(monkeypatch):
    processor, _source = _make_processor()
    path = processor.current_path

    emitted = []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append((img.copy(), size))
    )
    op_calls = _install_op_counter(monkeypatch)

    center = _view_params((0.375, 0.375, 0.625, 0.625), zoom=4.0)
    ops_len = len(build_op_list(center)) + 1  # + injected roi op

    processor._do_process(ProcessRequest(path, dict(center), 1))
    assert len(op_calls) == ops_len

    # Pan inside the margin: same snapped ROI => pure output-cache hit.
    small_pan = _view_params((0.385, 0.375, 0.635, 0.625), zoom=4.0)
    assert (
        processor._compute_preview_roi(small_pan)[0]
        == processor._compute_preview_roi(center)[0]
    )
    processor._do_process(ProcessRequest(path, dict(small_pan), 2))
    assert len(op_calls) == ops_len  # zero op re-runs
    assert len(emitted) == 2
    np.testing.assert_array_equal(emitted[1][0], emitted[0][0])
    assert emitted[1][1] == (256, 256)
    assert getattr(emitted[1][1], "roi_rect", None) == (64, 64, 128, 128)

    # Pan beyond the margin: a new ROI re-runs only the ROI-side chain
    # (roi + color ops) — the color parameters and executor stay the same.
    big_pan = _view_params((0.7, 0.7, 0.95, 0.95), zoom=4.0)
    new_roi = processor._compute_preview_roi(big_pan)[0]
    assert new_roi != (64, 64, 128, 128)
    executor_before = processor._preview_executors.get("full")
    processor._do_process(ProcessRequest(path, dict(big_pan), 3))
    assert len(op_calls) == 2 * ops_len
    assert processor._preview_executors.get("full") is executor_before
    assert len(emitted) == 3
    assert getattr(emitted[2][1], "roi_rect", None) == new_roi
    x, y, w, h = new_roi
    assert emitted[2][0].shape == (h, w, 3)


# =====================================================================
# RoiSourceSize compatibility contract
# =====================================================================


def test_roi_source_size_behaves_like_plain_tuple():
    size = RoiSourceSize((256, 128), (64, 32, 96, 64))
    assert size == (256, 128)
    assert size[0] == 256 and size[1] == 128
    w, h = size
    assert (w, h) == (256, 128)
    assert size.roi_rect == (64, 32, 96, 64)


# =====================================================================
# Viewport-side geometry (offscreen; no GL context needed)
# =====================================================================


def _make_viewport(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication([])

    from raw_alchemy.ui.viewport_gl import ImageViewportGL

    viewport = ImageViewportGL()
    # Square viewport (above the widget's 320x240 minimum size) so the
    # display scale equals the zoom on both axes.
    viewport.resize(400, 400)
    return viewport


def test_viewport_visible_rect_inversion_and_roi_coverage(monkeypatch):
    viewport = _make_viewport(monkeypatch)
    assert viewport.visible_source_rect() is None  # nothing displayed yet

    # Square image in a square viewport: display scale == zoom.
    viewport.set_image(np.zeros((100, 100, 3), np.uint8), source_size=(400, 400))
    viewport._zoom = 2.0

    rect = viewport.visible_source_rect()
    assert rect == pytest.approx((0.25, 0.25, 0.75, 0.75))

    # Pan right by a quarter of the viewport: offset in NDC units.
    viewport._offset_x = -0.5
    rect = viewport.visible_source_rect()
    assert rect == pytest.approx((0.375, 0.25, 0.875, 0.75))
    viewport._offset_x = 0.0

    # ROI overlay: full-frame geometry retained, overlay state tracked.
    roi_img = np.zeros((200, 200, 3), np.uint8)
    viewport.set_roi_image(roi_img, RoiSourceSize((400, 400), (64, 64, 200, 200)), (64, 64, 200, 200))
    assert viewport.has_roi_image()
    assert viewport._display_dimensions() == (400, 400)
    assert viewport._roi_rect_norm() == pytest.approx((0.16, 0.16, 0.66, 0.66))

    # Visible [0.25,0.75] vs ROI [0.16,0.66]: not fully covered.
    assert not viewport._visible_within_roi()
    # Centered ROI covering the visible region: covered.
    viewport.set_roi_image(roi_img, (400, 400), (80, 80, 240, 240))
    assert viewport._visible_within_roi()

    # A full-frame image supersedes (and drops) the ROI overlay.
    viewport.set_image(np.zeros((100, 100, 3), np.uint8), source_size=(400, 400))
    assert not viewport.has_roi_image()
    assert viewport._visible_within_roi()  # vacuously covered again

    viewport.deleteLater()


def test_viewport_roi_uniform_placement_math(monkeypatch):
    viewport = _make_viewport(monkeypatch)
    viewport.set_image(np.zeros((100, 100, 3), np.uint8), source_size=(400, 400))
    viewport._zoom = 2.0
    viewport.set_roi_image(
        np.zeros((200, 200, 3), np.uint8), (400, 400), (100, 100, 200, 200)
    )

    # Replicate paintGL's overlay placement: the centered ROI (quarter to
    # three-quarters of the frame) maps to a quad centered at the image
    # center with half the image span.
    scale_x, scale_y = viewport._display_scale()
    x0, y0, x1, y1 = viewport._roi_rect_norm()
    roi_scale_x = (x1 - x0) * scale_x
    roi_scale_y = (y1 - y0) * scale_y
    roi_offset_x = (x0 + x1 - 1.0) * scale_x + viewport._offset_x
    roi_offset_y = (1.0 - y0 - y1) * scale_y + viewport._offset_y

    assert (scale_x, scale_y) == pytest.approx((2.0, 2.0))
    assert (roi_scale_x, roi_scale_y) == pytest.approx((1.0, 1.0))
    assert (roi_offset_x, roi_offset_y) == pytest.approx((0.0, 0.0))

    # Off-center ROI: top-left quarter sits in the top-left NDC quadrant.
    viewport.set_roi_image(
        np.zeros((100, 100, 3), np.uint8), (400, 400), (0, 0, 100, 100)
    )
    x0, y0, x1, y1 = viewport._roi_rect_norm()
    roi_offset_x = (x0 + x1 - 1.0) * scale_x + viewport._offset_x
    roi_offset_y = (1.0 - y0 - y1) * scale_y + viewport._offset_y
    assert roi_offset_x == pytest.approx(-1.5)  # left of center at zoom 2
    assert roi_offset_y == pytest.approx(1.5)  # above center at zoom 2

    viewport.deleteLater()


# =====================================================================
# m4 — resident source: ROI changes must not re-upload the source
# =====================================================================


def _install_upload_spy(monkeypatch):
    from raw_alchemy.gpu_buffer import GpuImage

    uploads = []
    original_upload = GpuImage.upload

    def spy_upload(self, np_array):
        uploads.append(np_array.nbytes)
        return original_upload(self, np_array)

    monkeypatch.setattr(GpuImage, "upload", spy_upload)
    return uploads


def test_roi_change_reuses_resident_source_without_reupload(monkeypatch):
    """In the default unedited case the roi op sits at position 0, so a
    cross-margin pan / zoom-tier change invalidates every nonzero prefix.
    The executor must resume from the device-resident length-0 source
    entry instead of re-uploading the full float32 frame over PCIe (m4)."""
    from raw_alchemy.pipeline.executor import ExportExecutor

    processor, source = _make_processor()
    path = processor.current_path

    emitted = []
    processor.result_ready.connect(
        lambda img, img_path, request_id, ev, size: emitted.append((img.copy(), size))
    )
    uploads = _install_upload_spy(monkeypatch)

    center = _view_params((0.375, 0.375, 0.625, 0.625), zoom=4.0)
    processor._do_process(ProcessRequest(path, dict(center), 1))
    assert uploads == [source.nbytes]  # single source ingestion

    # Cross-margin pan: new ROI rect, all nonzero prefixes stale — but the
    # source comes from the resident copy, no second host->device transfer.
    big_pan = _view_params((0.7, 0.7, 0.95, 0.95), zoom=4.0)
    new_roi = processor._compute_preview_roi(big_pan)[0]
    assert new_roi != processor._compute_preview_roi(center)[0]
    processor._do_process(ProcessRequest(path, dict(big_pan), 2))
    assert uploads == [source.nbytes]

    # Another cross-margin move: still zero further uploads.
    third_pan = _view_params((0.05, 0.05, 0.3, 0.3), zoom=4.0)
    third_roi = processor._compute_preview_roi(third_pan)[0]
    assert third_roi not in (new_roi, processor._compute_preview_roi(center)[0])
    processor._do_process(ProcessRequest(path, dict(third_pan), 3))
    assert uploads == [source.nbytes]

    # The renders resumed from the resident source are pixel-correct: each
    # equals the same crop of a from-scratch full-frame reference (the
    # worker's preview executors round the exposure gain; match that).
    expected_float = ExportExecutor(round_exposure_gain=True).run(
        build_op_list(center), source.copy()
    )
    expected_uint8 = np.floor(
        np.clip(expected_float, 0.0, 1.0).astype(np.float32) * 255.0 + 0.5
    ).astype(np.uint8)
    assert len(emitted) == 3
    for (img, size), rect in zip(emitted[1:], (new_roi, third_roi)):
        x, y, w, h = rect
        assert getattr(size, "roi_rect", None) == rect
        np.testing.assert_array_equal(img, expected_uint8[y:y + h, x:x + w])


def test_source_residency_accounting_trim_and_reingest():
    """The length-0 source entry participates in cache_bytes()/trim() like
    any prefix: it survives ROI-rect generations, is evicted only after
    all longer (cheaper-to-recompute) prefixes, and a zero budget clears
    it; the next run then re-ingests and re-caches the source."""
    from raw_alchemy.pipeline.executor import (
        ExportExecutor,
        PreviewExecutor,
        _SOURCE_PREFIX_KEY,
    )

    rng = np.random.default_rng(77)
    src = rng.uniform(0.02, 0.65, size=(16, 16, 3)).astype(np.float32)
    ops = build_op_list(_case({"exposure": 0.5, "saturation": 1.1}))
    roi_ops_a = ImageProcessor._insert_roi_op(ops, (0, 0, 8, 8))
    roi_ops_b = ImageProcessor._insert_roi_op(ops, (8, 8, 8, 8))
    n = len(roi_ops_b)
    roi_bytes = 8 * 8 * 3 * 4  # float32 ROI-sized stage entries

    preview = PreviewExecutor(src)
    preview.run_result(roi_ops_a, source=src)
    assert preview._prefix_lengths.get(_SOURCE_PREFIX_KEY) == 0
    assert preview.cache_bytes() == src.nbytes + (n + 1) * roi_bytes

    # A different rect drops the stale roi-tagged generation but keeps the
    # source entry (its validity depends only on the source array).
    result_b = preview.run_result(roi_ops_b, source=src)
    assert preview._prefix_lengths.get(_SOURCE_PREFIX_KEY) == 0
    assert preview.cache_bytes() == src.nbytes + (n + 1) * roi_bytes
    expected_b = ExportExecutor().run(roi_ops_b, src.copy())
    np.testing.assert_allclose(result_b.image, expected_b, rtol=1e-5, atol=1e-5)

    # Trim evicts longest prefixes first; the source entry outlives them
    # (dropping it would force a PCIe re-upload, the cost m4 removes).
    freed = preview.trim(src.nbytes + 2 * roi_bytes)
    assert freed == (n - 1) * roi_bytes
    assert set(preview._prefix_lengths.values()) == {0, 1}

    # A zero budget clears the source entry too (T7.6 semantics intact).
    preview.trim(0)
    assert preview.cache_bytes() == 0
    assert _SOURCE_PREFIX_KEY not in preview._prefix_cache

    # The next run re-ingests the source and re-caches the residency.
    again = preview.run_result(roi_ops_b, source=src)
    np.testing.assert_allclose(again.image, expected_b, rtol=1e-5, atol=1e-5)
    assert preview._prefix_lengths.get(_SOURCE_PREFIX_KEY) == 0


def test_non_roi_runs_do_not_create_source_residency():
    """Full-frame (non-ROI) runs keep their T7.2 behavior: the nonzero
    prefixes already cover incremental resume, so no length-0 entry is
    added (and the T7.2/T7.6 exact accounting stays unchanged)."""
    from raw_alchemy.pipeline.executor import PreviewExecutor, _SOURCE_PREFIX_KEY

    rng = np.random.default_rng(79)
    src = rng.uniform(0.02, 0.65, size=(6, 8, 3)).astype(np.float32)
    preview = PreviewExecutor(src)
    ops = build_op_list(_case({"exposure": 0.5, "saturation": 1.1}))
    preview.run_result(ops, source=src)

    assert _SOURCE_PREFIX_KEY not in preview._prefix_cache
    assert len(preview._prefix_cache) == len(ops)
    assert preview.cache_bytes() == (len(ops) + 1) * src.nbytes


# =====================================================================
# m3 — ROI results must not feed the scopes (histogram/waveform)
# =====================================================================


class _ScopeHarness:
    """Bare-attribute stand-in for MainWindow.on_process_result's `self`."""

    def __init__(self, path):
        self.denoise_progress_dialog = None
        self.current_raw_path = path
        self.current_request_id = 7
        self.gallery_items_by_path = {}
        self.file_params_cache = {}
        self.scope_updates = []
        self.roi_overlays = []
        self.full_frames = []
        harness = self

        class _State:
            full = None

            @staticmethod
            def update_full(*args, **kwargs):
                pass

        self.current = _State()
        self.original = _State()

        class _Radio:
            @staticmethod
            def isChecked():
                return False

        class _Panel:
            auto_exp_radio = _Radio()

            @staticmethod
            def get_params():
                return {}

        self.right_panel = _Panel()

        class _Viewport:
            @staticmethod
            def set_roi_image(img, source_size, roi_rect):
                harness.roi_overlays.append((img, roi_rect))

            @staticmethod
            def set_image(img, source_size=None):
                harness.full_frames.append(img)

        self.viewport = _Viewport()

        class _Label:
            @staticmethod
            def setText(text):
                pass

        self.preview_lbl = _Label()

    def _update_histogram_async(self, img):
        self.scope_updates.append(img)


def _pump_until(app, predicate, timeout=2.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_roi_results_do_not_feed_scopes_but_full_frames_do(monkeypatch):
    """Scopes are whole-frame statistics: feeding them the zoom>fit ROI crop
    silently turned the histogram/waveform into visible-area statistics
    that jump on every pan (m3). ROI results must update only the viewport
    overlay; the last full-frame scope data stays."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from raw_alchemy.ui.main_window import MainWindow

    harness = _ScopeHarness("photo.raw")
    harness._scope_source_for_result = MainWindow._scope_source_for_result

    # Control: a full-frame result refreshes the scopes (deferred update).
    img_full = np.full((64, 64, 3), 40, np.uint8)
    MainWindow.on_process_result(harness, img_full, "photo.raw", 7, 0.5, (64, 64))
    assert _pump_until(app, lambda: harness.scope_updates)
    assert len(harness.scope_updates) == 1
    assert harness.scope_updates[0] is img_full
    assert len(harness.full_frames) == 1

    # ROI result: viewport overlay updates, the scopes must not.
    roi_img = np.full((16, 16, 3), 250, np.uint8)
    roi_size = RoiSourceSize((64, 64), (8, 8, 16, 16))
    MainWindow.on_process_result(harness, roi_img, "photo.raw", 7, 0.5, roi_size)
    assert not _pump_until(app, lambda: len(harness.scope_updates) > 1, timeout=0.3)
    assert harness.roi_overlays and harness.roi_overlays[-1][1] == (8, 8, 16, 16)
    assert len(harness.scope_updates) == 1  # last full-frame data retained


def test_roi_dpi_tier_downscales_before_color_ops():
    """T7.5+: zoom between fit and 1:1 must not oversample the colour chain.

    The 6-tuple roi op crops then downscales to the screen-density target;
    the colour ops downstream see the small buffer.
    """
    from raw_alchemy.pipeline.executor import PreviewExecutor
    from raw_alchemy.pipeline.ops import Op

    yy, xx = np.mgrid[0:400, 0:600].astype(np.float32)
    src = np.stack([xx / 600, yy / 400, (xx + yy) / 1000], -1).astype(np.float32)
    src *= 0.9  # 平滑渐变——重采样与 OETF 的交换误差在平滑数据上应当很小
    ex = PreviewExecutor(src)
    ops = [Op("roi", (64, 64, 320, 256, 160, 128)), Op("exposure", ("Manual", 0.0, "matrix", None)), Op("srgb_out", (None,))]
    out = ex.run(ops)
    assert out.shape == (128, 160, 3)
    # 参考:全分辨率渲染后再缩(降采样与非线性 OETF 不严格交换,
    # 这是预览级近似——停手后的精化渲染回到全精度;容差放宽)
    import cv2
    ref_ops = [Op("roi", (64, 64, 320, 256)), ops[1], ops[2]]
    ref_full = PreviewExecutor(src.copy()).run(ref_ops)
    ref = cv2.resize(ref_full, (160, 128), interpolation=cv2.INTER_AREA)
    assert float(np.abs(out - ref).mean()) < 0.01
    np.testing.assert_allclose(out, ref, atol=0.08)


def test_make_roi_target_size_tiers_by_dpi_and_zoom():
    from raw_alchemy.workers.image_processor import ImageProcessor

    # 8000 宽全幅,2K 视口(DPR1),zoom=2(仍低于 1:1)→ 目标按屏幕密度缩
    params = {"viewport_size": (2560, 1440), "device_pixel_ratio": 1.0, "preview_zoom": 2.0}
    tw, th = ImageProcessor._make_roi_target_size(4000, 3000, 8000, 5320, params)
    assert tw < 4000 and th < 3000
    # DPR 2 → 目标翻倍(高 DPI 屏挡位更高)
    params2 = dict(params, device_pixel_ratio=2.0)
    tw2, th2 = ImageProcessor._make_roi_target_size(4000, 3000, 8000, 5320, params2)
    assert tw2 > tw
    # 1:1 及以上不再放大(封顶原生)
    params3 = dict(params, preview_zoom=10.0)
    tw3, th3 = ImageProcessor._make_roi_target_size(4000, 3000, 8000, 5320, params3)
    assert (tw3, th3) == (4000, 3000)


def test_native_base_refine_outputs_pipeline_resolution():
    """T7.10 — 原生分辨率底图。

    fit 视图的交互渲染输出视口尺寸,并排队 idle refine;refine 结果必须是
    管线原生尺寸(缩放纯采样的前提),且不得被交互结果的输出缓存槽短路
    (view key 以 _force_full_preview 区分)。
    """
    processor, _source = _make_processor()
    path = processor.current_path

    emitted = []
    processor.result_ready.connect(
        lambda img, p, rid, ev, size: emitted.append((img.copy(), size))
    )

    # fit 视图(zoom=1):64x64 视口 → 输出 64x64,低于原生 → 排 refine
    params = _view_params(zoom=1.0)
    processor._do_process(ProcessRequest(path, dict(params), 1))
    assert len(emitted) == 1
    fit_img, fit_size = emitted[0]
    assert fit_img.shape == (64, 64, 3)
    refine = processor._full_refine_request
    assert refine is not None and refine.params.get("_force_full_preview") is True

    # 执行 refine:输出必须为 256x256 原生(短路吞掉会回吐 64x64)
    processor._full_refine_request = None
    processor._do_process(refine)
    assert len(emitted) == 2
    native_img, native_size = emitted[1]
    assert native_img.shape == (256, 256, 3)
    assert tuple(native_size)[:2] == (256, 256)
    assert getattr(native_size, "roi_rect", None) is None
    # refine 不自我续期
    assert processor._full_refine_request is None

    # 再发一次同参交互请求:命中输出缓存,但因内容低于原生仍须再排 refine
    processor._do_process(ProcessRequest(path, dict(params), 2))
    assert len(emitted) == 3
    assert emitted[2][0].shape == (64, 64, 3)
    assert processor._full_refine_request is not None


def test_native_preview_target_size_caps():
    from raw_alchemy.workers.image_processor import (
        ImageProcessor,
        NATIVE_PREVIEW_MAX_PIXELS,
        NATIVE_PREVIEW_MAX_SIDE,
    )

    force = {"_force_full_preview": True, "viewport_size": (2560, 1440)}
    # 常规全幅:原样输出
    assert ImageProcessor._make_preview_target_size(7968, 5344, dict(force)) == (7968, 5344)
    # 超长边:按 GL 纹理上限缩
    tw, th = ImageProcessor._make_preview_target_size(20000, 2000, dict(force))
    assert tw <= NATIVE_PREVIEW_MAX_SIDE and abs(tw / th - 10.0) < 0.1
    # 超像素数:按host内存上限缩
    tw, th = ImageProcessor._make_preview_target_size(12000, 9000, dict(force))
    assert tw * th <= NATIVE_PREVIEW_MAX_PIXELS
