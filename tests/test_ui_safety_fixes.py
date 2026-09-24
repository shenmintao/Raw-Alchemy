"""Regression tests for the UI data-safety fixes (delete, shortcuts, sticky
geometry, LUT fallback, per-image baseline, crash handler, GPU detection)."""

import shutil
import sys
import threading
import types
from pathlib import Path

import pytest


def _scratch_dir(name):
    root = Path.cwd() / ".test-output" / "ui-safety-tests" / name
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


class _FakeInfoBar:
    calls = []

    @classmethod
    def _record(cls, kind, *args, **kwargs):
        cls.calls.append((kind, args, kwargs))

    @classmethod
    def success(cls, *args, **kwargs):
        cls._record("success", *args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        cls._record("error", *args, **kwargs)

    @classmethod
    def info(cls, *args, **kwargs):
        cls._record("info", *args, **kwargs)


def test_mark_prefix_is_the_green_circle_not_mojibake():
    from raw_alchemy.ui.library_controller import MARK_PREFIX

    assert MARK_PREFIX == "\U0001F7E2"


# --- delete -----------------------------------------------------------------


class _FakeGallery:
    def __init__(self, harness, paths):
        self.harness = harness
        self.items = list(paths)
        self.current = 0

    def currentRow(self):
        return self.current

    def row(self, item):
        return self.items.index(item) if item in self.items else -1

    def takeItem(self, row):
        self.items.pop(row)
        self.harness.on_selection_moved()

    def count(self):
        return len(self.items)

    def setCurrentRow(self, row):
        self.current = row
        self.harness.on_selection_moved()


class _FakeLabel:
    def __init__(self):
        self.text = None

    def setText(self, text):
        self.text = text


class _DeleteHarness:
    def __init__(self, path, others=()):
        self.current_raw_path = path
        paths = [path, *others]
        self.gallery_list = _FakeGallery(self, paths)
        self.gallery_items_by_path = {p: p for p in paths}
        self.marked_files = {path}
        self.file_params_cache = {path: {"exposure": 1.0}}
        self.file_baseline_params_cache = {path: {"exposure": 0.0}}
        self.preview_lbl = _FakeLabel()
        self.paths_current_during_moves = []

    def on_selection_moved(self):
        # MainWindow persists the sidecar of current_raw_path here.
        self.paths_current_during_moves.append(self.current_raw_path)

    def update_window_title(self):
        pass


def _patch_delete(monkeypatch, send2trash_impl):
    from PySide6.QtWidgets import QMessageBox

    from raw_alchemy.ui import library_controller

    _FakeInfoBar.calls = []
    monkeypatch.setattr(library_controller, "InfoBar", _FakeInfoBar)
    monkeypatch.setattr(
        library_controller.QMessageBox,
        "question",
        lambda *_a, **_k: QMessageBox.StandardButton.Yes,
    )
    module = types.ModuleType("send2trash")
    module.send2trash = send2trash_impl
    monkeypatch.setitem(sys.modules, "send2trash", module)
    return library_controller


def test_failed_recycle_bin_move_never_deletes_permanently(monkeypatch):
    scratch = _scratch_dir("delete-fail")
    photo = scratch / "photo.raf"
    photo.write_bytes(b"raw")

    def refuse(_path):
        raise OSError("recycle bin unavailable")

    removed = []
    library_controller = _patch_delete(monkeypatch, refuse)
    monkeypatch.setattr(library_controller.os, "remove", lambda p: removed.append(p))

    window = _DeleteHarness(str(photo), others=["other.raf"])
    library_controller.LibraryControllerMixin.delete_image(window)

    assert photo.exists()
    assert removed == []
    assert window.current_raw_path == str(photo)
    assert window.gallery_list.count() == 2
    assert [call[0] for call in _FakeInfoBar.calls] == ["error"]


def test_successful_delete_does_not_touch_the_deleted_path_again(monkeypatch):
    trashed = []
    library_controller = _patch_delete(monkeypatch, trashed.append)

    window = _DeleteHarness("a.raf", others=["b.raf"])
    window._forget_deleted_image = (
        library_controller.LibraryControllerMixin._forget_deleted_image.__get__(window)
    )
    library_controller.LibraryControllerMixin.delete_image(window)

    assert len(trashed) == 1
    # Selection moves while the row is removed; the deleted path must no
    # longer be current then, or its sidecar is written back to disk.
    assert window.paths_current_during_moves
    assert "a.raf" not in window.paths_current_during_moves
    assert window.gallery_list.items == ["b.raf"]
    assert "a.raf" not in window.marked_files
    assert "a.raf" not in window.file_params_cache
    assert "a.raf" not in window.file_baseline_params_cache


# --- switching images ---------------------------------------------------------


class _Clearable:
    def clear(self):
        pass


class _Unchecked:
    def isChecked(self):
        return False


class _SwitchPanel:
    def __init__(self, params):
        self.params = dict(params)
        self.auto_exp_radio = _Unchecked()
        self.baselines = []

    def get_params(self):
        return dict(self.params)

    def set_params(self, params):
        self.params.update(params)

    def set_saved_baseline(self, params):
        self.baselines.append(params)


class _SwitchGallery:
    def row(self, _item):
        return 0


class _SwitchItem:
    def __init__(self, path):
        self.path = path

    def data(self, _role):
        return self.path


class _SwitchHarness:
    def __init__(self, panel_params):
        self.current_raw_path = None
        self.right_panel = _SwitchPanel(panel_params)
        self.file_params_cache = {}
        self.file_baseline_params_cache = {}
        self.original = _Clearable()
        self.current = _Clearable()
        self.baseline = _Clearable()
        self.gallery_list = _SwitchGallery()
        self.loaded = []

    def _persist_current_sidecar_now(self):
        pass

    def update_window_title(self):
        pass

    def _load_sidecar_for_path(self, _path):
        return None

    def update_mark_button_state(self):
        pass

    def load_image(self, path):
        self.loaded.append(path)

    def _preload_neighbors(self, _row):
        pass

    def regenerate_baseline_for_current_image(self):
        pass


def test_new_image_keeps_the_look_but_not_the_previous_geometry():
    from raw_alchemy.ui.library_controller import LibraryControllerMixin

    corners = ((0.0, 0.0), (1.0, 0.1), (0.9, 1.0), (0.1, 0.9))
    window = _SwitchHarness(
        {
            "saturation": 1.3,
            "lut_path": "look.cube",
            "rotation": 90,
            "flip_horizontal": True,
            "crop": (0.1, 0.1, 0.5, 0.5),
            "perspective_corners": corners,
        }
    )
    LibraryControllerMixin.on_gallery_item_clicked(window, _SwitchItem("new.raf"))

    params = window.right_panel.params
    assert params["saturation"] == 1.3
    assert params["lut_path"] == "look.cube"
    assert params["rotation"] == 0
    assert params["flip_horizontal"] is False
    assert params["crop"] == (0.0, 0.0, 1.0, 1.0)
    assert params["perspective_corners"] is None


def test_reset_to_baseline_uses_the_shown_images_baseline(monkeypatch):
    _ensure_qapp(monkeypatch)  # the baseline re-render is scheduled on a QTimer
    from raw_alchemy.ui.library_controller import LibraryControllerMixin

    window = _SwitchHarness({"saturation": 1.0})
    window.file_baseline_params_cache["b.raf"] = {"saturation": 0.5}

    LibraryControllerMixin.on_gallery_item_clicked(window, _SwitchItem("a.raf"))
    LibraryControllerMixin.on_gallery_item_clicked(window, _SwitchItem("b.raf"))

    assert window.right_panel.baselines == [None, {"saturation": 0.5}]


# --- keyboard shortcuts -------------------------------------------------------


class _Visible:
    def __init__(self, visible=True):
        self.visible = visible

    def isVisible(self):
        return self.visible


class _Stack:
    def __init__(self, current):
        self.current = current

    def currentWidget(self):
        return self.current


def test_library_shortcuts_leave_editor_and_slider_keys_alone(monkeypatch):
    _ensure_qapp(monkeypatch)
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDoubleSpinBox, QLineEdit, QListWidget, QPushButton, QSlider, QTreeView,
    )

    from raw_alchemy.ui.main_window import MainWindow

    page_preview = object()
    window = types.SimpleNamespace(
        _KEY_OWNING_EDITORS=MainWindow._KEY_OWNING_EDITORS,
        main_widget=_Visible(True),
        center_stack=_Stack(page_preview),
        page_preview=page_preview,
        processor_connection_mode="normal",
        gallery_list=QListWidget(),
    )
    active = lambda obj, key: MainWindow._library_shortcuts_active(window, obj, key)  # noqa: E731
    Key = Qt.Key

    line_edit, spin, slider = QLineEdit(), QDoubleSpinBox(), QSlider()
    tree, button = QTreeView(), QPushButton()
    for editor in (line_edit, spin):
        for key in (Key.Key_Delete, Key.Key_T, Key.Key_Space, Key.Key_Left):
            assert not active(editor, key)
    assert not active(slider, Key.Key_Left)
    assert not active(tree, Key.Key_Right)
    assert active(slider, Key.Key_Space)
    assert active(button, Key.Key_Delete)
    assert active(window.gallery_list, Key.Key_Right)

    window.processor_connection_mode = "crop"  # entering crop mode
    assert not active(button, Key.Key_Right)
    window.processor_connection_mode = "normal"
    window.center_stack.current = object()  # crop/perspective viewer shown
    assert not active(button, Key.Key_Right)
    window.center_stack.current = page_preview
    window.main_widget.visible = False  # settings page shown
    assert not active(button, Key.Key_Delete)


# --- LUT restore --------------------------------------------------------------


def test_recorded_lut_is_never_replaced_by_the_previous_images_lut(monkeypatch):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.ui.widgets import inspector_panel

    scratch = _scratch_dir("lut-restore")
    folder = scratch / "luts"
    other = scratch / "elsewhere"
    folder.mkdir()
    other.mkdir()
    (folder / "a.cube").write_text("")
    (other / "b.cube").write_text("")

    panel = inspector_panel.InspectorPanel()
    try:
        panel.lut_folder = str(folder)
        panel.refresh_lut_list()
        base_count = panel.lut_combo.count()

        panel.set_params({"lut_path": str(folder / "a.cube")})
        assert panel.get_params()["lut_path"] == str(folder / "a.cube")

        # Recorded LUT outside the LUT folder: shown and kept, not "a.cube".
        panel.set_params({"lut_path": str(other / "b.cube")})
        assert panel.lut_combo.currentText() == "b.cube"
        assert panel.get_params()["lut_path"] == str(other / "b.cube")

        # Missing LUT: "none", never the previous image's LUT.
        panel.set_params({"lut_path": str(other / "gone.cube")})
        assert panel.get_params()["lut_path"] is None
        assert panel.lut_combo.count() == base_count

        panel.set_params({"lut_path": str(folder / "a.cube")})
        assert panel.get_params()["lut_path"] == str(folder / "a.cube")
    finally:
        panel.shutdown_scope_workers()
        panel.deleteLater()


def test_saved_baseline_setter_drives_reset_button(monkeypatch):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.ui.widgets import inspector_panel

    panel = inspector_panel.InspectorPanel()
    try:
        panel.set_saved_baseline({"saturation": 0.5})
        assert panel.saved_baseline_params == {"saturation": 0.5}
        assert panel.reset_baseline_btn.isEnabled()
        panel.set_saved_baseline(None)
        assert panel.saved_baseline_params is None
        assert not panel.reset_baseline_btn.isEnabled()
    finally:
        panel.shutdown_scope_workers()
        panel.deleteLater()


# --- crash handler --------------------------------------------------------------


def test_crash_handler_keeps_faults_out_of_the_rotating_log_and_dialogs_once(monkeypatch):
    scratch = _scratch_dir("crash-handler")
    monkeypatch.setenv("RAW_ALCHEMY_LOG_DIR", str(scratch))
    from raw_alchemy import crash_handler
    from raw_alchemy.ui import crash_dialog

    handler = crash_handler.CrashHandler()
    assert Path(handler.fault_log_path).parent == Path(handler.log_path).parent
    assert handler.fault_log_path != handler.log_path

    shown = []
    monkeypatch.setattr(crash_dialog, "show_crash_dialog", lambda *a: shown.append(a))

    worker = threading.Thread(target=handler._show_crash_dialog, args=("E", "off-thread"))
    worker.start()
    worker.join()
    assert shown == []  # no widgets off the GUI thread

    handler._show_crash_dialog("E", "first")
    handler._show_crash_dialog("E", "second")
    assert len(shown) == 1


# --- GPU detection --------------------------------------------------------------


@pytest.mark.parametrize(
    "names, vendor, cuda",
    [
        (["Intel(R) UHD Graphics", "NVIDIA GeForce RTX 4070 Laptop GPU"], "nvidia", True),
        (["Microsoft Remote Display Adapter", "AMD Radeon RX 9070 XT"], "amd", False),
        (["Intel(R) Arc(TM) Graphics"], "intel", False),
        (["Microsoft Basic Display Adapter"], "unknown", False),
        ([], "unknown", False),
    ],
)
def test_gpu_names_are_classified_without_wmic(names, vendor, cuda):
    from raw_alchemy.onnx import gpu_runtime

    result = {"vendor": "unknown", "name": "", "cuda_compatible": False}
    gpu_runtime._classify_gpu_names(names, result)
    assert result["vendor"] == vendor
    assert result["cuda_compatible"] is cuda
