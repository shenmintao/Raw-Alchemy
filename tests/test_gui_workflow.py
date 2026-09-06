import shutil
from pathlib import Path

import numpy as np


def test_image_state_keeps_numpy_frame_without_full_pixmap_copy():
    from raw_alchemy.ui.image_state import ImageState

    state = ImageState()
    frame = np.zeros((32, 48, 3), np.uint8)
    state.update_full(object(), None, frame, source_size=(48, 32))

    assert state.full is None
    assert state.uint8_data is frame
    assert state.source_size == (48, 32)


class _FakeSignal:
    def __init__(self):
        self.handlers = []

    def connect(self, handler):
        self.handlers.append(handler)

    def disconnect(self, handler):
        self.handlers.remove(handler)

    def emit(self, *args):
        for handler in list(self.handlers):
            handler(*args)


class _FakeButton:
    def __init__(self):
        self.enabled = True
        self.checked = False

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setChecked(self, checked):
        self.checked = bool(checked)

    def blockSignals(self, _blocked):
        pass


class _FakeInfoBarInstance:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeInfoBar:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def info(cls, *args, **kwargs):
        cls.calls.append(("info", args, kwargs))
        return _FakeInfoBarInstance()

    @classmethod
    def success(cls, *args, **kwargs):
        cls.calls.append(("success", args, kwargs))
        return _FakeInfoBarInstance()

    @classmethod
    def warning(cls, *args, **kwargs):
        cls.calls.append(("warning", args, kwargs))
        return _FakeInfoBarInstance()

    @classmethod
    def error(cls, *args, **kwargs):
        cls.calls.append(("error", args, kwargs))
        return _FakeInfoBarInstance()


class _FakeProcessor:
    def __init__(self):
        self.last_applied_ev = 0.75
        self.export_finished = _FakeSignal()
        self.export_completed = _FakeSignal()
        self.cache_exports = []
        self.path_exports = []
        self.preview_requests = []

    def export_from_cache(self, path, payload):
        self.cache_exports.append((path, payload.copy()))
        return len(self.cache_exports)

    def export_path(self, path, payload):
        self.path_exports.append((path, payload.copy()))
        return len(self.path_exports)

    def update_preview(self, path, params):
        self.preview_requests.append((path, params.copy()))
        return len(self.preview_requests)


def _scratch_dir(name):
    root = Path.cwd() / ".test-output" / "gui-workflow-tests" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _ensure_qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeRightPanel:
    def __init__(self, params):
        self.params = params.copy()
        self.crop_updates = []
        self.set_updates = []

    def get_params(self):
        return self.params.copy()

    def set_params(self, params):
        self.params = params.copy()
        self.set_updates.append(params.copy())

    def update_crop_params(self, rotation, flip_h, flip_v, crop_rect):
        self.crop_updates.append((rotation, flip_h, flip_v, crop_rect))
        self.params.update(
            {
                "rotation": rotation,
                "flip_horizontal": flip_h,
                "flip_vertical": flip_v,
                "crop": crop_rect,
            }
        )


class _FakeProgress:
    def __init__(self):
        self.range = None
        self.values = []
        self.visible = False

    def setRange(self, start, end):
        self.range = (start, end)

    def setValue(self, value):
        self.values.append(value)

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False


class _ExportHarness:
    def __init__(self, params):
        self.current_raw_path = "current.raf"
        self.last_export_path = ""
        self.last_folder_path = ""
        self.processor = _FakeProcessor()
        self.right_panel = _FakeRightPanel(params)
        self.btn_export_curr = _FakeButton()
        self.btn_export_all = _FakeButton()
        self.export_progress = _FakeProgress()
        self.file_params_cache = {}
        self.marked_files = []
        self.loaded_sidecars = []
        self.export_calls = []

    def _load_sidecar_for_path(self, path):
        self.loaded_sidecars.append(path)
        return None


def _base_export_params(**overrides):
    params = {
        "exposure_mode": "Auto",
        "metering_mode": "matrix",
        "exposure": 0.0,
        "log_space": "F-Log",
        "lut_path": "look.cube",
        "lens_correct": True,
        "custom_db_path": None,
        "wb_temp": 100.0,
        "wb_tint": -5.0,
        "saturation": 1.1,
        "contrast": 0.9,
        "highlight": -0.2,
        "shadow": 0.3,
        "rotation": 90,
        "flip_horizontal": True,
        "flip_vertical": False,
        "perspective_corners": ((0.0, 0.0), (1.0, 0.1), (0.9, 1.0), (0.1, 0.9)),
        "crop": (0.1, 0.2, 0.7, 0.6),
        "denoise_enabled": False,
        "sharpen_strength": 0.25,
    }
    params.update(overrides)
    return params


def test_gui_single_export_builds_cached_and_full_worker_payloads(monkeypatch):
    from raw_alchemy.ui import export_controller

    monkeypatch.setattr(export_controller, "InfoBar", _FakeInfoBar)
    _FakeInfoBar.reset()

    cached_window = _ExportHarness(_base_export_params())
    source = np.full((4, 4, 3), 0.2, dtype=np.float32)

    export_controller.ExportControllerMixin.run_export(
        cached_window,
        "current.raf",
        "out.hdr.heif",
        is_single_export=True,
        cached_img=source,
        cached_exif_data={"camera": "synthetic"},
        cached_exif_metadata=None,
    )

    assert len(cached_window.processor.cache_exports) == 1
    cached_path, cached_payload = cached_window.processor.cache_exports[0]
    assert cached_path == "current.raf"
    assert cached_payload["cached_img"] is source
    assert cached_payload["output_path"] == "out.hdr.heif"
    assert cached_payload["exposure"] == 0.75
    assert cached_payload["perspective_corners"] == (
        (0.0, 0.0),
        (1.0, 0.1),
        (0.9, 1.0),
        (0.1, 0.9),
    )
    assert cached_payload["crop"] == (0.1, 0.2, 0.7, 0.6)
    assert cached_payload["hdr_output"] is True

    cached_window.btn_export_curr.setEnabled(False)
    cached_window.saving_infobar = _FakeInfoBarInstance()
    cached_window.processor.export_completed.emit(1, True, "")
    assert cached_window.btn_export_curr.enabled is True
    assert any(call[0] == "success" for call in _FakeInfoBar.calls)

    full_window = _ExportHarness(
        _base_export_params(exposure_mode="Manual", exposure=-0.5, denoise_enabled=True)
    )
    export_controller.ExportControllerMixin.run_export(
        full_window,
        "current.raf",
        "out.tif",
        is_single_export=True,
    )

    assert len(full_window.processor.path_exports) == 1
    full_path, full_payload = full_window.processor.path_exports[0]
    assert full_path == "current.raf"
    assert full_payload["input_path"] == "current.raf"
    assert full_payload["output_path"] == "out.tif"
    assert full_payload["exposure"] == -0.5
    assert full_payload["output_format"] == "tif"
    assert full_payload["jobs"] == 1
    assert full_payload["denoise_enabled"] is True
    assert full_payload["crop"] == (0.1, 0.2, 0.7, 0.6)


class _FakeBatchDialog:
    def __init__(self, _parent):
        pass

    def exec(self):
        return True

    def get_settings(self):
        return {"format": "JPEG", "ignore_lut": True}


def test_gui_batch_export_uses_per_image_params_and_lut_override(monkeypatch):
    from raw_alchemy.ui import export_controller

    scratch = _scratch_dir("batch-export")
    monkeypatch.setattr(export_controller, "InfoBar", _FakeInfoBar)
    monkeypatch.setattr(export_controller, "BatchExportDialog", _FakeBatchDialog)
    monkeypatch.setattr(
        export_controller.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(scratch),
    )
    _FakeInfoBar.reset()

    window = _ExportHarness(_base_export_params(lut_path="current.cube"))
    window.marked_files = ["current.raf", "other.arw"]
    window.file_params_cache["other.arw"] = _base_export_params(
        exposure_mode="Manual",
        exposure=1.25,
        lut_path="other.cube",
    )

    def fake_run_export(input_path, output_path, params=None, callback=None, **_kwargs):
        window.export_calls.append((input_path, output_path, None if params is None else params.copy()))
        if callback:
            callback()

    window.run_export = fake_run_export
    window.batch_export_next = export_controller.ExportControllerMixin.batch_export_next.__get__(
        window, _ExportHarness
    )

    export_controller.ExportControllerMixin.export_all(window)

    assert window.export_progress.range == (0, 2)
    assert window.export_progress.visible is False
    assert window.btn_export_all.enabled is True
    assert [call[0] for call in window.export_calls] == ["current.raf", "other.arw"]
    assert [Path(call[1]).name for call in window.export_calls] == [
        "current.jpg",
        "other.jpg",
    ]
    assert window.export_calls[0][2]["lut_path"] is None
    assert window.export_calls[1][2]["lut_path"] is None
    assert window.export_calls[1][2]["exposure"] == 1.25


class _FakeTimer:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1


class _ParamTimer(_FakeTimer):
    def __init__(self):
        super().__init__()
        self.started = 0

    def start(self):
        self.started += 1


def test_slider_interaction_has_leading_preview_and_exact_release(monkeypatch):
    from raw_alchemy.pipeline.ops import _as_hashable
    from raw_alchemy.ui import main_window

    class Harness:
        def __init__(self):
            self.current_raw_path = "photo.raw"
            self.file_params_cache = {}
            self.update_timer = _ParamTimer()
            self._param_interaction_active = False
            self._last_param_leading_time = 0.0
            self._last_param_submit_key = None
            self.current_params = None
            self.triggered = []
            self.sidecar_writes = 0

        def _schedule_current_sidecar_write(self):
            self.sidecar_writes += 1

        def trigger_processing(self):
            self._last_param_submit_key = _as_hashable(self.current_params)
            self.triggered.append(self.current_params.copy())

    harness = Harness()
    times = iter((1.0, 1.02))
    monkeypatch.setattr(main_window.time, "monotonic", lambda: next(times))

    main_window.MainWindow._on_param_interaction_started(harness)
    first = {"exposure": 0.1}
    harness.current_params = first
    main_window.MainWindow.on_param_changed(harness, first)
    assert harness.triggered == [first]

    second = {"exposure": 0.2}
    harness.current_params = second
    main_window.MainWindow.on_param_changed(harness, second)
    assert harness.update_timer.started == 1
    assert harness.triggered == [first]

    main_window.MainWindow._on_param_interaction_finished(harness, second)
    assert harness.triggered == [first, second]
    assert harness._param_interaction_active is False


class _FakeStack:
    def __init__(self):
        self.current = object()
        self.history = []

    def currentWidget(self):
        return self.current

    def setCurrentWidget(self, widget):
        self.current = widget
        self.history.append(widget)


class _FakeViewer:
    display_pixmap = None


class _EditHarness:
    def __init__(self, params):
        self.current_raw_path = "current.raf"
        self.file_params_cache = {"current.raf": params.copy()}
        self.processor = _FakeProcessor()
        self.right_panel = _FakeRightPanel(params)
        self.update_timer = _FakeTimer()
        self.center_stack = _FakeStack()
        self.crop_viewer = _FakeViewer()
        self.perspective_viewer = _FakeViewer()
        self.page_preview = object()
        self.processor_connection_mode = "normal"
        self.statuses = []
        self.scheduled_sidecar_writes = 0
        self.triggered = 0

    def update_status(self, message):
        self.statuses.append(message)

    def _schedule_current_sidecar_write(self):
        self.scheduled_sidecar_writes += 1

    def trigger_processing(self):
        self.triggered += 1


def test_gui_crop_and_perspective_modes_reset_preview_then_persist_params():
    from raw_alchemy.ui.edit_modes import EditModesMixin

    params = _base_export_params(rotation=180, flip_vertical=True)
    window = _EditHarness(params)
    window.exit_crop_mode = EditModesMixin.exit_crop_mode.__get__(window, _EditHarness)
    window.exit_perspective_mode = EditModesMixin.exit_perspective_mode.__get__(
        window, _EditHarness
    )

    EditModesMixin.enter_crop_mode(window)

    assert window.processor_connection_mode == "crop"
    crop_preview = window.processor.preview_requests[-1][1]
    assert crop_preview["rotation"] == 0
    assert crop_preview["flip_horizontal"] is False
    assert crop_preview["flip_vertical"] is False
    assert crop_preview["crop"] == (0.0, 0.0, 1.0, 1.0)

    EditModesMixin.on_crop_applied(window, 90, True, False, (0.2, 0.2, 0.5, 0.5))

    assert window.processor_connection_mode == "normal"
    assert window.file_params_cache["current.raf"]["rotation"] == 90
    assert window.file_params_cache["current.raf"]["flip_horizontal"] is True
    assert window.file_params_cache["current.raf"]["crop"] == (0.2, 0.2, 0.5, 0.5)
    assert window.scheduled_sidecar_writes == 1
    assert window.triggered == 1

    EditModesMixin.enter_perspective_mode(window)

    assert window.processor_connection_mode == "perspective"
    perspective_preview = window.processor.preview_requests[-1][1]
    assert perspective_preview["rotation"] == 0
    assert perspective_preview["flip_horizontal"] is False
    assert perspective_preview["flip_vertical"] is False
    assert perspective_preview["perspective_corners"] is None
    assert perspective_preview["crop"] == (0.0, 0.0, 1.0, 1.0)

    corners = ((0.0, 0.0), (1.0, 0.05), (0.95, 1.0), (0.1, 0.9))
    EditModesMixin.on_perspective_applied(window, corners)

    assert window.processor_connection_mode == "normal"
    assert window.file_params_cache["current.raf"]["perspective_corners"] == corners
    assert window.right_panel.set_updates[-1]["perspective_corners"] == corners
    assert window.scheduled_sidecar_writes == 2
    assert window.triggered == 2


def test_inspector_panel_enables_cans_denoise_ui(monkeypatch):
    _ensure_qapp(monkeypatch)

    from raw_alchemy.ui.widgets import inspector_panel

    # v14 raw-main is integrated: the AI-denoise UI ships enabled. The switch
    # itself follows model availability (onnxruntime + bundled ONNX present).
    assert inspector_panel.DENOISE_UI_ENABLED is True
    panel = inspector_panel.InspectorPanel()
    try:
        available = panel._denoise_available()
        assert panel.denoise_switch.isEnabled() is available
        panel.set_params({"denoise_enabled": True})
        # Checked/effective state only when the model is actually available.
        assert panel.denoise_switch.isChecked() is available
        assert panel.get_params()["denoise_enabled"] is available
    finally:
        panel.shutdown_scope_workers()
        panel.deleteLater()
