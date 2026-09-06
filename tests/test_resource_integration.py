"""Real worker lanes and owned processes honour resource/cancellation boundaries."""
import multiprocessing as mp
import threading
import time

import numpy as np
import onnxruntime as ort
import pytest

from raw_alchemy.pipeline.cancellation import cancellation_scope
from raw_alchemy.pipeline.executor import PipelineAborted
from raw_alchemy.pipeline.resources import ResourceGovernor, MiB
from raw_alchemy.onnx.isolated_session import IsolatedSession, allocate_shared_memory


def _blocked_run(connection, *args):
    connection.send(('ready', ['CPUExecutionProvider'], []))
    connection.recv()
    time.sleep(30)


def _decoded(connection, *args):
    from multiprocessing import shared_memory
    connection.send(('result', (2, 3, 3), ({'camera': 'fixture'}, None)))
    block = shared_memory.SharedMemory(name=connection.recv())
    try:
        np.ndarray((2, 3, 3), np.float32, buffer=block.buf)[:] = 0.25
        connection.send(('done',))
    finally:
        block.close()


@pytest.mark.parametrize('mode', ['canonical', 'preload'])
def test_decode_roundtrip_preserves_pixels_and_metadata(mode):
    from raw_alchemy.native_decode import decode_raw
    before = {p.pid for p in mp.active_children()}
    result = decode_raw('synthetic.raw', mode, worker=_decoded)
    if mode == 'preload':
        result, exif, metadata = result
        assert exif == {'camera': 'fixture'} and metadata is None
    np.testing.assert_array_equal(result, np.full((2, 3, 3), 0.25, np.float32))
    assert {p.pid for p in mp.active_children()} == before


def test_memory_pressure_terminates_native_wait_without_cpu_retry():
    rss = [128 * MiB]
    gate = ResourceGovernor(1024 * MiB, sample=lambda: (rss[0], 8192 * MiB, 8192 * MiB))
    before = {p.pid for p in mp.active_children()}
    with gate.job(128 * MiB):
        session = IsolatedSession('unused', ort.SessionOptions(), [], variant='test', worker=_blocked_run)
        try:
            rss[0] = 2048 * MiB
            with pytest.raises(MemoryError):
                session.run(None, {'x': np.zeros((2, 3), np.float32)})
            assert session._process is None
        finally:
            session.close()
    assert {p.pid for p in mp.active_children()} == before
    assert gate.snapshot()['jobs'] == 0


def test_real_export_lane_yields_to_preview_and_closes(monkeypatch):
    from raw_alchemy.workers import image_processor as module, export_dispatcher
    gate = ResourceGovernor(8192 * MiB, sample=lambda: (128 * MiB, 16384 * MiB, 16384 * MiB))
    monkeypatch.setattr(module, 'governor', gate)
    monkeypatch.setattr(export_dispatcher, 'governor', gate)
    worker = module.ImageProcessor(warmup_sessions=False)
    started, preview, finished = threading.Event(), threading.Event(), threading.Event()
    order = []
    errors = []
    def export(request):
        try:
            order.append('export-start')
            started.set()
            until = time.monotonic() + 5
            while not preview.is_set() and time.monotonic() < until:
                gate.checkpoint()
                time.sleep(0.001)
            order.append('export-end')
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()
    def process(request):
        order.append('preview')
        preview.set()
    monkeypatch.setattr(worker, '_do_export', export)
    monkeypatch.setattr(worker, '_dispatch_request', process)
    try:
        worker.export_path('export.raw', {})
        assert started.wait(5)
        worker.update_preview('preview.raw', {})
        assert preview.wait(5)
        assert finished.wait(5)
    finally:
        worker.request_stop()
        assert worker.wait(5000)
    assert not errors
    assert not worker._export_dispatcher.is_alive()
    assert order == ['export-start', 'preview', 'export-end']
    assert gate.snapshot()['jobs'] == 0


def test_cancelled_cache_write_preserves_existing_artifact(tmp_path, monkeypatch):
    from raw_alchemy.pipeline import denoise_disk_cache as disk
    monkeypatch.setenv('RAWALCHEMY_DENOISE_CACHE_DIR', str(tmp_path / 'cache'))
    source = tmp_path / 'test.raw'
    source.write_bytes(b'fixture')
    original = np.ones((8, 9, 3), np.float32)
    disk.save(str(source), 'policy', original)
    cancel = threading.Event()
    with cancellation_scope(cancel.is_set):
        cancel.set()
        with pytest.raises(PipelineAborted):
            disk.save(str(source), 'policy', original * 2)
    np.testing.assert_array_equal(disk.load(str(source), 'policy'), original)
    assert not list((tmp_path / 'cache').glob('*.tmp'))


def test_linux_shared_memory_shortage_rejected_before_mapping(monkeypatch):
    import shutil
    import sys
    from types import SimpleNamespace
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024))
    with pytest.raises(MemoryError, match='/dev/shm'):
        allocate_shared_memory(4096)