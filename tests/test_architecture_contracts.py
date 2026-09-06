"""Request isolation, artifact publication and honest export completion."""
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from raw_alchemy.pipeline.request import ProcessRequest
from raw_alchemy.pipeline import denoise_disk_cache as disk
from raw_alchemy.pipeline.stage_identity import source_identity
from raw_alchemy.pipeline.cancellation import cancellation_scope
from raw_alchemy.pipeline.executor import ExportExecutor, PipelineAborted
from raw_alchemy.pipeline.ops import Op


def test_request_nested_metadata_cannot_change_after_submission():
    image = np.ones((4, 6, 3), np.float32)
    params = {"export": {"cached_img": image, "crop": [0, 0, 1, 1]}, "corners": [[0, 0]]}
    request = ProcessRequest("a.raw", params, 1)
    params["export"]["crop"][0] = 0.5
    params["corners"][0][0] = 0.5
    assert request.params["export"]["crop"] == (0, 0, 1, 1)
    assert request.params["corners"] == ((0, 0),)
    assert request.params["export"]["cached_img"] is image
    with pytest.raises(TypeError):
        request.params["export"]["crop"] = ()
    with pytest.raises(FrozenInstanceError):
        request.path = "b.raw"


def test_replaced_source_cannot_receive_old_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("RAWALCHEMY_DENOISE_CACHE_DIR", str(tmp_path / "cache"))
    raw = tmp_path / "a.raw"
    raw.write_bytes(b"old source")
    token = source_identity(raw)
    raw.write_bytes(b"new source")
    disk.save(str(raw), "model", np.ones((4, 6, 3), np.float32), source_token=token)
    assert disk.load(str(raw), "model") is None
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_cancelled_export_stops_before_operation(monkeypatch):
    executor = ExportExecutor()
    monkeypatch.setattr(executor, "_apply_op", lambda *a: pytest.fail("cancelled work executed"))
    with cancellation_scope(lambda: False):
        with pytest.raises(PipelineAborted):
            with cancellation_scope(lambda: True):
                executor.run([Op("srgb_out", ())], np.ones((4, 6, 3), np.float32))


def test_failed_encode_preserves_existing_destination(tmp_path, monkeypatch):
    from raw_alchemy import file_io
    output = tmp_path / "image.jpg"
    output.write_bytes(b"previous good export")
    def fail(img, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        return False
    monkeypatch.setattr(file_io, "save_image", fail)
    with pytest.raises(OSError):
        file_io.save_image_atomic(np.zeros((2, 3, 3), np.float32), output)
    assert output.read_bytes() == b"previous good export"
    assert list(tmp_path.iterdir()) == [output]


def test_denoise_failure_is_not_export_success(tmp_path, monkeypatch):
    from raw_alchemy import core, exif
    from raw_alchemy.pipeline import source_artifacts
    monkeypatch.setattr(exif, "extract_lens_exif", lambda *a: ({}, {}))
    def fail(*a, **kw):
        raise RuntimeError("CPU denoise failed")
    monkeypatch.setattr(source_artifacts, "resolve_denoised_source", fail)
    monkeypatch.setattr(core, "save_image", lambda *a, **kw: pytest.fail("invalid export saved"))
    with pytest.raises(RuntimeError, match="CPU denoise failed"):
        core.process_image("a.raw", str(tmp_path / "a.jpg"), None, None, denoise_enabled=True)


def test_export_ids_do_not_change_preview_generation(monkeypatch):
    from raw_alchemy.workers.image_processor import ImageProcessor
    monkeypatch.setattr(ImageProcessor, "start", lambda self: None)
    worker = ImageProcessor()
    preview_id = worker.update_preview("a.raw", {})
    assert worker.export_path("b.raw", {}) == 1
    assert worker.export_path("c.raw", {}) == 2
    assert worker.current_request_id == preview_id
