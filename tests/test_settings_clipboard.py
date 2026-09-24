"""Copy / paste of develop settings between images."""

import shutil
from pathlib import Path

import pytest


def _scratch_dir(name):
    root = Path.cwd() / ".test-output" / "settings-clipboard" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _ensure_qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


CORNERS = ((0.0, 0.0), (1.0, 0.1), (0.9, 1.0), (0.1, 0.9))


def _params(**overrides):
    params = {
        "exposure_mode": "Manual", "metering_mode": "matrix", "exposure": 0.7,
        "log_space": "F-Log", "lut_path": "look.cube",
        "lens_correct": True, "custom_db_path": None,
        "rotation": 90, "flip_horizontal": True, "flip_vertical": False,
        "crop": (0.1, 0.1, 0.5, 0.5), "perspective_corners": CORNERS,
        "wb_temp": 12.0, "wb_tint": -3.0, "saturation": 1.4, "contrast": 1.2,
        "highlight": -20.0, "shadow": 15.0,
        "denoise_enabled": True, "denoise_strength": 0.3, "sharpen_strength": 0.2,
    }
    params.update(overrides)
    return params


def _defaults():
    return _params(
        exposure_mode="Auto", exposure=0.0, log_space="None", lut_path=None,
        rotation=0, flip_horizontal=False, crop=(0.0, 0.0, 1.0, 1.0),
        perspective_corners=None, wb_temp=0.0, wb_tint=0.0, saturation=1.25,
        contrast=1.1, highlight=0.0, shadow=0.0, denoise_enabled=False,
        denoise_strength=0.25, sharpen_strength=0.0,
    )


def test_geometry_is_not_copied_by_default():
    from raw_alchemy.ui.settings_clipboard import default_group_selection, pick_settings

    picked = pick_settings(_params(), default_group_selection())
    assert picked["log_space"] == "F-Log"
    assert picked["wb_temp"] == 12.0
    assert picked["denoise_strength"] == 0.3
    for key in ("rotation", "flip_horizontal", "crop", "perspective_corners"):
        assert key not in picked
    assert pick_settings(_params(), {"geometry"})["perspective_corners"] == CORNERS


class _Panel:
    def __init__(self, params):
        self.params = dict(params)

    def get_params(self):
        return dict(self.params)

    def set_params(self, params):
        self.params.update(params)

    def default_params(self):
        return _defaults()


class _Visible:
    def isVisible(self):
        return True


class _Stack:
    def __init__(self, page):
        self.page = page

    def currentWidget(self):
        return self.page


class _Button:
    def setEnabled(self, _enabled):
        pass


class _Harness:
    def __init__(self, current, panel_params):
        self.current_raw_path = current
        self.right_panel = _Panel(panel_params)
        self.file_params_cache = {}
        self.marked_files = set()
        self.main_widget = _Visible()
        self.page_preview = object()
        self.center_stack = _Stack(self.page_preview)
        self.processor_connection_mode = "normal"
        self.btn_paste_settings = _Button()
        self.sidecars = {}
        self.pastes = []

    def on_param_changed(self, params):
        self.file_params_cache[self.current_raw_path] = dict(params)

    def _write_sidecar_for_path(self, path, params):
        self.sidecars[path] = dict(params)

    def _notify_paste(self, count):
        self.pastes.append(count)


def _bind(harness):
    from raw_alchemy.ui.settings_clipboard import SettingsClipboardMixin

    for name in ("_settings_editable", "_apply_params_to_current",
                 "paste_settings_to_current", "paste_settings_to_marked",
                 "undo_last_paste"):
        setattr(harness, name, getattr(SettingsClipboardMixin, name).__get__(harness))
    return harness


def _copy(harness, source_params, groups):
    from raw_alchemy.ui.settings_clipboard import pick_settings

    harness._settings_clipboard = {
        "params": pick_settings(source_params, groups), "groups": groups, "source": "src.raf",
    }
    harness._paste_undo = None


@pytest.fixture
def quiet_ui(monkeypatch):
    from raw_alchemy.ui import settings_clipboard

    class Bar:
        @staticmethod
        def info(*_a, **_k):
            pass

        warning = info

    class Box:
        def __init__(self, *_a):
            pass

        def exec(self):
            return True

    monkeypatch.setattr(settings_clipboard, "InfoBar", Bar)
    monkeypatch.setattr(settings_clipboard, "MessageBox", Box)


def test_paste_to_current_keeps_its_geometry_and_can_be_undone(quiet_ui):
    from raw_alchemy.ui.settings_clipboard import default_group_selection

    own = _params(log_space="None", lut_path=None, saturation=1.0,
                  rotation=180, crop=(0.2, 0.2, 0.4, 0.4), perspective_corners=None)
    window = _bind(_Harness("a.raf", own))
    _copy(window, _params(), default_group_selection())

    window.paste_settings_to_current()
    pasted = window.file_params_cache["a.raf"]
    assert pasted["log_space"] == "F-Log" and pasted["saturation"] == 1.4
    assert pasted["rotation"] == 180 and pasted["crop"] == (0.2, 0.2, 0.4, 0.4)
    assert pasted["perspective_corners"] is None
    assert window.pastes == [1]

    window.undo_last_paste()
    assert window.file_params_cache["a.raf"] == own


def test_paste_to_marked_covers_unedited_images_and_undo_restores_them(quiet_ui):
    from raw_alchemy.ui.settings_clipboard import default_group_selection

    scratch = _scratch_dir("marked")
    current, edited, fresh = (str(scratch / n) for n in ("cur.raf", "edited.raf", "fresh.raf"))
    for path in (current, edited, fresh):
        Path(path).write_bytes(b"raw")
    missing = str(scratch / "gone.raf")

    window = _bind(_Harness(current, _params(saturation=1.0)))
    edited_before = _params(saturation=0.8, rotation=270)
    window.file_params_cache[edited] = dict(edited_before)
    window.marked_files = {current, edited, fresh, missing}
    _copy(window, _params(), {"tone_color", "log_lut"})

    window.paste_settings_to_marked()

    assert window.pastes == [3]  # the missing file is skipped
    assert window.file_params_cache[current]["saturation"] == 1.4
    assert window.file_params_cache[edited]["saturation"] == 1.4
    assert window.file_params_cache[edited]["rotation"] == 270  # geometry untouched
    fresh_params = window.file_params_cache[fresh]
    assert fresh_params["log_space"] == "F-Log"
    assert fresh_params["wb_temp"] == 0.0  # not copied -> default, not the current image's 12
    assert fresh_params["rotation"] == 0
    assert window.sidecars[edited]["saturation"] == 1.4

    window.undo_last_paste()
    assert window.file_params_cache[edited] == edited_before
    assert window.sidecars[edited] == edited_before
    assert fresh not in window.file_params_cache
    assert window.sidecars[fresh] == {}
    assert window.file_params_cache[current]["saturation"] == 1.0


def test_paste_without_a_copy_changes_nothing(quiet_ui):
    window = _bind(_Harness("a.raf", _params()))
    window._settings_clipboard = None
    window.paste_settings_to_current()
    window.paste_settings_to_marked()
    assert window.file_params_cache == {} and window.pastes == []


def test_copy_dialog_leaves_geometry_unchecked(monkeypatch):
    _ensure_qapp(monkeypatch)
    from PySide6.QtWidgets import QWidget

    from raw_alchemy.ui.settings_clipboard import CopySettingsDialog, default_group_selection

    parent = QWidget()
    parent.resize(800, 600)
    dialog = CopySettingsDialog(default_group_selection(), parent)
    try:
        assert not dialog.checks["geometry"].isChecked()
        assert dialog.checks["white_balance"].isChecked()
        dialog.checks["geometry"].setChecked(True)
        assert "geometry" in dialog.selected_groups()
    finally:
        dialog.deleteLater()
        parent.deleteLater()


def test_default_params_round_trip_through_the_panel(monkeypatch):
    _ensure_qapp(monkeypatch)
    from raw_alchemy.ui.widgets import inspector_panel

    panel = inspector_panel.InspectorPanel()
    try:
        defaults = panel.default_params()
        panel.set_params(defaults)
        assert set(panel.get_params()) == set(defaults)
        round_trip = panel.get_params()
        for key, value in defaults.items():
            if key == "denoise_enabled":
                continue  # gated on the model being available
            expected = pytest.approx(value) if isinstance(value, float) else value
            assert round_trip[key] == expected, key
    finally:
        panel.shutdown_scope_workers()
        panel.deleteLater()
