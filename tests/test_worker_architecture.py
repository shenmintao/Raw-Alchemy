"""Concurrency, queue bounds and demand-only startup regressions."""
import threading

import pytest

from raw_alchemy.workers.image_processor import ImageProcessor, MAX_PENDING_EXPORTS


def test_same_image_new_parameters_cancel_inflight(monkeypatch):
    monkeypatch.setattr(ImageProcessor, "start", lambda self: None)
    worker = ImageProcessor()
    worker.current_path = "same.raw"
    assert not worker._interactive_abort_requested()
    worker.update_preview("same.raw", {"exposure": 1.0})
    assert worker._interactive_abort_requested()
    assert worker.current_request_id == 1
    worker.update_preview("same.raw", {"exposure": 2.0})
    assert worker.current_request_id == 2
    assert worker.pending_request.params["exposure"] == 2.0


def test_idle_export_snapshot_rejected_while_worker_is_busy():
    worker = ImageProcessor()
    assert worker._busy is False
    worker._busy = True
    assert worker.get_cached_for_export() is None


def test_worker_startup_does_not_warm_bayer(monkeypatch):
    from raw_alchemy.onnx import rcd_demosaic
    def fail(*args, **kwargs):
        pytest.fail("unconditional Bayer warmup")
    monkeypatch.setattr(rcd_demosaic, "rcd_demosaic", fail)
    worker = ImageProcessor()
    worker._warm_onnx_sessions()
    assert isinstance(worker.lock, type(threading.RLock()))


def test_export_fairness_and_queue_bound(monkeypatch):
    monkeypatch.setattr(ImageProcessor, "start", lambda self: None)
    worker = ImageProcessor()
    for _ in range(MAX_PENDING_EXPORTS):
        worker.export_path("export.raw", {})
    with pytest.raises(RuntimeError, match="queue is full"):
        worker.export_path("excess.raw", {})
    worker.update_preview("preview.raw", {"exposure": 1.0})
    assert worker._take_next_request().path == "preview.raw"
    worker.update_preview("preview.raw", {"exposure": 2.0})
    assert worker._take_next_request().path == "export.raw"
    assert worker._take_next_request().path == "preview.raw"
    assert worker._take_next_request().path == "export.raw"