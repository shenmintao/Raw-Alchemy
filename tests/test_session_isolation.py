"""Real spawned workers: tensor IPC, timeout, crash and cancellation cleanup."""
import multiprocessing as mp
import os
import threading
import time

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from raw_alchemy.onnx.isolated_session import IsolatedSession
from raw_alchemy.pipeline.cancellation import cancellation_scope
from raw_alchemy.pipeline.executor import PipelineAborted


def _stuck(connection, *args):
    time.sleep(30)


def _stuck_inference(connection, *args):
    connection.send(("ready", ["CoreMLExecutionProvider"], []))
    connection.recv()
    time.sleep(30)


def _crash(connection, *args):
    os._exit(7)


def test_real_spawned_runtime_roundtrip_and_profile(tmp_path):
    info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3])
    out = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3])
    model = onnx.helper.make_model(onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "x"], ["y"])], "double", [info], [out]),
        opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8)
    path = tmp_path / "model.onnx"
    path.write_bytes(model.SerializeToString())
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "profile")
    session = IsolatedSession(path, options, ["CPUExecutionProvider"], variant="test")
    pid = session._process.pid
    try:
        assert session.get_inputs()[0].name == "x"
        x = np.arange(6, dtype=np.float32).reshape(3, 2).T  # noncontiguous
        for _ in range(3):
            np.testing.assert_array_equal(session.run(None, {"x": x})[0], x * 2)
        assert session.end_profiling().endswith(".json")
    finally:
        session.close()
    assert pid not in [child.pid for child in mp.active_children()]


@pytest.mark.parametrize("worker", [_stuck, _crash])
def test_native_startup_failure_is_bounded(worker, monkeypatch):
    monkeypatch.setenv("RAWALCHEMY_COREML_COMPILE_TIMEOUT", "0.4")
    before = {child.pid for child in mp.active_children()}
    start = time.monotonic()
    with pytest.raises((TimeoutError, RuntimeError, EOFError)):
        IsolatedSession("unused", ort.SessionOptions(), [], variant="test", worker=worker)
    assert time.monotonic() - start < 5
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize("cancel", [False, True])
def test_native_run_stall_cleanup_and_cancel_does_not_become_timeout(monkeypatch, cancel):
    monkeypatch.setenv("RAWALCHEMY_COREML_RUN_TIMEOUT", "0.3" if not cancel else "20")
    session = IsolatedSession("unused", ort.SessionOptions(), [], variant="test", worker=_stuck_inference)
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    if cancel:
        timer.start()
    try:
        with cancellation_scope(cancelled.is_set), pytest.raises(PipelineAborted if cancel else TimeoutError):
            session.run(None, {"image": np.zeros((8, 8), np.float32)})
        assert session._process is None
    finally:
        timer.cancel()
        session.close()
