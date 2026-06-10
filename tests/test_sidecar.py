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
