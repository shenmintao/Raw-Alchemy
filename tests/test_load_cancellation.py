"""Loading must preserve cancellation and report failed cache backfills honestly."""
import numpy as np
import pytest

from raw_alchemy.pipeline.cache_manager import CachedImage
from raw_alchemy.pipeline.executor import PipelineAborted
from raw_alchemy.pipeline.request import ProcessRequest
from raw_alchemy.workers import image_processor as module


@pytest.mark.parametrize("operation", ["load", "backfill", "preload"])
@pytest.mark.parametrize("error_type", [PipelineAborted, MemoryError])
def test_decode_control_errors_reach_worker_boundary(monkeypatch, operation, error_type):
    worker = module.ImageProcessor(warmup_sessions=False)
    errors, completed = [], []
    worker.error_occurred.connect(errors.append)
    worker.load_complete.connect(lambda *args: completed.append(args))
    monkeypatch.setattr(module, "source_identity", lambda path: "source")
    error = error_type("cancelled" if error_type is PipelineAborted else "budget exceeded")

    def fail(*args):
        raise error

    monkeypatch.setattr(worker, "_decode_for_view", fail)
    monkeypatch.setattr(worker, "_cpu_decode_to_prophoto", fail)
    if operation == "backfill":
        worker.cache_manager.put("frame.raw", CachedImage(
            "frame.raw", None, {}, None, source_token="source",
        ))
    request = ProcessRequest("frame.raw", {"_load": True}, 1)
    action = worker._do_preload if operation == "preload" else worker._do_load
    with pytest.raises(error_type) as caught:
        action(request)
    assert caught.value is error
    assert errors == []
    assert completed == []


def test_failed_backfill_does_not_report_load_complete(monkeypatch):
    worker = module.ImageProcessor(warmup_sessions=False)
    errors, completed = [], []
    worker.error_occurred.connect(errors.append)
    worker.load_complete.connect(lambda *args: completed.append(args))
    monkeypatch.setattr(module, "source_identity", lambda path: "source")
    worker.cache_manager.put("frame.raw", CachedImage(
        "frame.raw", None, {}, None, source_token="source",
    ))

    def fail(path):
        raise RuntimeError("decoder rejected RAW")

    monkeypatch.setattr(worker, "_decode_for_view", fail)
    worker._do_load(ProcessRequest("frame.raw", {"_load": True}, 1))
    assert len(errors) == 1
    assert "decoder rejected RAW" in errors[0]
    assert completed == []
    assert worker.current_path is None


def test_proxy_memory_error_does_not_fall_back_to_full_frame(monkeypatch):
    import cv2
    monkeypatch.setattr(module, "PROXY_MIN_SOURCE_PIXELS", 1)
    monkeypatch.setattr(module, "PROXY_TARGET_PIXELS", 1)

    def fail(*args, **kwargs):
        raise MemoryError("proxy allocation failed")

    monkeypatch.setattr(cv2, "resize", fail)
    with pytest.raises(MemoryError, match="proxy allocation"):
        module.ImageProcessor._make_proxy(np.zeros((2, 3, 3), np.float32))