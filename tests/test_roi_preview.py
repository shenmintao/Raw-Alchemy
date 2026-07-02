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
    # ROI previews never schedule the idle full refine (they *are* full-source).
    assert processor._full_refine_request is None


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
