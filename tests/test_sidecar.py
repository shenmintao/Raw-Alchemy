import json
import shutil
from pathlib import Path

from raw_alchemy.sidecar import (
    SIDECAR_VERSION,
    jsonable_params,
    load_folder_sidecars,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)


def _scratch_dir(name):
    root = Path.cwd() / ".test-output" / "sidecar-tests" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def test_sidecar_round_trip_json_native_params():
    scratch = _scratch_dir("round-trip")
    raw_path = scratch / "image.dng"
    raw_path.write_bytes(b"raw")
    params = {
        "exposure_mode": "Manual",
        "exposure": 1.25,
        "crop": (0.1, 0.2, 0.7, 0.6),
        "perspective_corners": (
            (0.0, 0.0),
            (1.0, 0.1),
            (0.9, 1.0),
            (0.1, 0.9),
        ),
        "lut_path": None,
        "lens_correct": True,
    }

    written = write_sidecar(raw_path, params, marked=True)

    assert written == scratch / "image.dng.rwa.json"
    with written.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload == {
        "version": SIDECAR_VERSION,
        "params": jsonable_params(params),
        "marked": True,
    }

    loaded = read_sidecar(raw_path)
    assert loaded is not None
    assert loaded.version == SIDECAR_VERSION
    assert loaded.params == jsonable_params(params)
    assert loaded.marked is True


def test_load_folder_sidecars_uses_supported_raw_files():
    scratch = _scratch_dir("folder-scan")
    raw_path = scratch / "keep.cr3"
    raw_path.write_bytes(b"raw")
    ignored_path = scratch / "ignore.jpg"
    ignored_path.write_bytes(b"jpg")

    write_sidecar(raw_path, {"contrast": 1.2}, marked=False)
    write_sidecar(ignored_path, {"contrast": 2.0}, marked=True)

    loaded = load_folder_sidecars(scratch)

    assert set(loaded) == {str(raw_path)}
    assert loaded[str(raw_path)].params == {"contrast": 1.2}
    assert loaded[str(raw_path)].marked is False


def test_corrupt_sidecar_falls_back_to_missing():
    scratch = _scratch_dir("corrupt")
    raw_path = scratch / "broken.nef"
    raw_path.write_bytes(b"raw")
    sidecar_path(raw_path).write_text("{not json", encoding="utf-8")

    assert read_sidecar(raw_path) is None
    assert load_folder_sidecars(scratch) == {}


class _DummyState:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class _DummyRadio:
    def isChecked(self):
        return False


class _DummyRightPanel:
    def __init__(self):
        self.auto_exp_radio = _DummyRadio()
        self.applied_params = None

    def get_params(self):
        return {
            "exposure_mode": "Auto",
            "exposure": 0.0,
            "rotation": 0,
            "flip_horizontal": False,
            "flip_vertical": False,
            "crop": (0.0, 0.0, 1.0, 1.0),
        }

    def set_params(self, params):
        self.applied_params = params.copy()


class _DummyGalleryList:
    def row(self, _item):
        return 0


class _DummyGalleryItem:
    def __init__(self, path):
        self.path = path

    def data(self, _role):
        return self.path


class _SidecarWindowHarness:
    def __init__(self):
        self.write_sidecar_enabled = True
        self.current_raw_path = None
        self.file_params_cache = {}
        self.file_baseline_params_cache = {}
        self.marked_files = set()
        self.original = _DummyState()
        self.current = _DummyState()
        self.baseline = _DummyState()
        self.right_panel = _DummyRightPanel()
        self.gallery_list = _DummyGalleryList()
        self.mark_button_states = []
        self.loaded_path = None
        self.preload_args = None
        self.persisted = False
        self.title_updated = False

    def _persist_current_sidecar_now(self):
        self.persisted = True

    def update_window_title(self):
        self.title_updated = True

    def update_mark_button_state(self):
        self.mark_button_states.append(self.current_raw_path in self.marked_files)

    def load_image(self, path):
        self.loaded_path = path

    def _preload_neighbors(self, current_index, count=2):
        self.preload_args = (current_index, count)


def test_main_window_folder_scan_rehydrates_sidecar_state_after_restart():
    from raw_alchemy.ui.main_window import MainWindow

    scratch = _scratch_dir("ui-folder-restore")
    first = scratch / "first.cr3"
    second = scratch / "second.nef"
    first.write_bytes(b"raw")
    second.write_bytes(b"raw")
    first_params = {"exposure_mode": "Manual", "exposure": 1.1, "contrast": 1.25}
    second_params = {"exposure_mode": "Auto", "metering_mode": "center", "saturation": 0.9}
    write_sidecar(first, first_params, marked=True)
    write_sidecar(second, second_params, marked=False)

    window = _SidecarWindowHarness()

    MainWindow._load_sidecars_for_folder(window, str(scratch))

    assert window.file_params_cache[str(first)] == first_params
    assert window.file_params_cache[str(second)] == second_params
    assert str(first) in window.marked_files
    assert str(second) not in window.marked_files


def test_gallery_selection_restores_sidecar_params_and_mark_after_restart():
    from raw_alchemy.ui.library_controller import LibraryControllerMixin
    from raw_alchemy.ui.main_window import MainWindow

    scratch = _scratch_dir("ui-gallery-restore")
    raw_path = scratch / "selected.dng"
    raw_path.write_bytes(b"raw")
    params = {
        "exposure_mode": "Manual",
        "exposure": 0.7,
        "metering_mode": "spot",
        "rotation": 90,
        "flip_horizontal": True,
        "flip_vertical": False,
        "crop": [0.1, 0.2, 0.8, 0.9],
        "perspective_corners": [[0.0, 0.0], [1.0, 0.1], [0.9, 1.0], [0.1, 0.9]],
        "denoise_enabled": True,
    }
    expected = jsonable_params(params)
    write_sidecar(raw_path, params, marked=True)

    window = _SidecarWindowHarness()
    window._load_sidecar_for_path = MainWindow._load_sidecar_for_path.__get__(
        window,
        _SidecarWindowHarness,
    )

    LibraryControllerMixin.on_gallery_item_clicked(
        window,
        _DummyGalleryItem(str(raw_path)),
    )

    assert window.current_raw_path == str(raw_path)
    assert window.file_params_cache[str(raw_path)] == expected
    assert window.right_panel.applied_params == expected
    assert str(raw_path) in window.marked_files
    assert window.mark_button_states == [True]
    assert window.loaded_path == str(raw_path)
    assert window.preload_args == (0, 2)
    assert window.original.cleared is True
    assert window.current.cleared is True
    assert window.baseline.cleared is True
