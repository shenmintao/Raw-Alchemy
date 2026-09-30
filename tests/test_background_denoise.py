"""Real lanes keep colour responsive and reject superseded noise artifacts."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from raw_alchemy.pipeline.cache_manager import CachedImage
from raw_alchemy.pipeline import source_artifacts, stage_identity
from raw_alchemy.pipeline.resources import ResourceGovernor, MiB, checkpoint
from raw_alchemy.workers import image_processor, denoise_task
from raw_alchemy.pipeline.request import ProcessRequest


@pytest.fixture
def lane(tmp_path, monkeypatch):
    monkeypatch.setenv('RAW_ALCHEMY_CPU_ONLY', '1')
    monkeypatch.setenv('RAWALCHEMY_DENOISE_CACHE_DIR', str(tmp_path / 'cache'))
    gate = ResourceGovernor(8192 * MiB, sample=lambda: (128 * MiB, 16384 * MiB, 16384 * MiB))
    monkeypatch.setattr(image_processor, 'governor', gate)
    monkeypatch.setattr(denoise_task, 'governor', gate)
    monkeypatch.setattr(image_processor, 'denoise_tag', lambda *a, **kw: 'test-policy')
    monkeypatch.setattr(source_artifacts, 'denoise_tag', lambda s, **kw: f'test-{s}')
    raw = tmp_path / 'image.raw'
    raw.write_bytes(b'fixture raw')
    path = str(raw)
    source = np.full((12, 16, 3), 0.2, np.float32)
    worker = image_processor.ImageProcessor(warmup_sessions=False)
    worker.cache_manager.put(path, CachedImage(
        path, source, None, None, decode_variant='test-decode',
        source_token=stage_identity.source_identity(path),
    ))
    errors = []
    worker.error_occurred.connect(errors.append, Qt.ConnectionType.DirectConnection)
    params = dict(exposure_mode='Manual', exposure=0.0, denoise_enabled=True,
                  denoise_strength=0.25, lens_correct=False, log_space=None,
                  crop=(0, 0, 1, 1), viewport_size=(16, 12))
    yield worker, path, params, gate, errors
    worker.request_stop()
    assert worker.wait(5000)
    assert not worker.background_denoise_pending()
    assert gate.snapshot()['jobs'] == 0


def wait_until(predicate, timeout=5):
    until = time.monotonic() + timeout
    while not predicate() and time.monotonic() < until:
        time.sleep(0.005)
    assert predicate()


def test_colour_runs_between_tiles_then_latest_ev_gets_final_noise(lane, monkeypatch):
    worker, path, params, gate, errors = lane
    running, release, colour, final = (threading.Event() for _ in range(4))
    calls = []

    def model(source, *, strength, progress_callback):
        calls.append(strength)
        running.set()
        while not release.is_set():
            checkpoint()
            progress_callback(1, 2)
            time.sleep(0.001)
        return source * np.float32(0.5)

    monkeypatch.setattr(image_processor, 'denoise_rgb_linear', model)
    expected_id = [None]

    def result(image, path, request_id, ev, size):
        if ev == 1.0:
            if worker.preview_denoise_strength(path) == 0.25:
                assert request_id == expected_id[0]
                final.set()
            else:
                colour.set()

    worker.result_ready.connect(result, Qt.ConnectionType.DirectConnection)
    worker.update_preview(path, params)
    assert running.wait(5)
    expected_id[0] = worker.update_preview(path, dict(params, exposure=1.0))
    assert colour.wait(5), 'exposure waited for full denoise'
    assert worker.background_denoise_pending() and calls == [0.25]
    assert worker.get_cached_for_export(params) is None
    release.set()
    assert final.wait(5)
    snapshot = worker.get_cached_for_export(params)
    np.testing.assert_array_equal(snapshot['denoise_full'], np.full((12, 16, 3), 0.1, np.float32))
    assert calls == [0.25] and not errors


def test_new_strength_cancels_old_job_and_waits_for_settle(lane, monkeypatch):
    worker, path, params, _, errors = lane
    running, old_stopped = threading.Event(), threading.Event()
    calls = []

    def model(source, *, strength, progress_callback):
        calls.append(strength)
        if strength == 0.25:
            running.set()
            try:
                while True:
                    checkpoint()
                    progress_callback(1, 2)
                    time.sleep(0.001)
            finally:
                old_stopped.set()
        return source * np.float32(0.75)

    monkeypatch.setattr(image_processor, 'denoise_rgb_linear', model)
    worker.update_preview(path, params)
    assert running.wait(5)
    deferred = dict(params, denoise_enabled=False, _denoise_target=0.3, _denoise_deferred=True)
    worker.update_preview(path, deferred)
    assert old_stopped.wait(5)
    wait_until(lambda: not worker.background_denoise_pending())
    assert calls == [0.25] and worker.preview_denoise_strength(path) is None
    worker.update_preview(path, dict(params, denoise_strength=0.3))
    wait_until(lambda: worker.preview_denoise_strength(path) == 0.3)
    assert calls == [0.25, 0.3] and not errors


@pytest.mark.parametrize('action', ['disable', 'shutdown'])
def test_cancelled_noise_leaves_no_thread_or_artifact(lane, monkeypatch, action):
    worker, path, params, _, errors = lane
    running = threading.Event()

    def model(source, *, strength, progress_callback):
        running.set()
        while True:
            checkpoint()
            progress_callback(1, 2)
            time.sleep(0.001)

    monkeypatch.setattr(image_processor, 'denoise_rgb_linear', model)
    worker.update_preview(path, params)
    assert running.wait(5)
    if action == 'disable':
        worker.update_preview(path, dict(params, denoise_enabled=False))
        wait_until(lambda: not worker.background_denoise_pending())
        assert worker.preview_denoise_strength(path) is None
    else:
        worker.request_stop()
        assert worker.wait(5000)
    assert not errors


def test_failed_background_job_reports_once_without_retry_loop(lane, monkeypatch):
    worker, path, params, _, errors = lane
    calls = []

    def model(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('fixture inference failure')

    monkeypatch.setattr(image_processor, 'denoise_rgb_linear', model)
    worker.update_preview(path, params)
    wait_until(lambda: len(errors) == 1)
    for ev in (1.0, 2.0, 3.0):
        worker.update_preview(path, dict(params, exposure=ev))
        wait_until(lambda: not worker.has_interactive_request())
    assert calls == [1] and len(errors) == 1


def test_switching_images_never_publishes_old_noise(lane, monkeypatch, tmp_path):
    worker, path, params, _, errors = lane
    running = threading.Event()

    def model(source, **kwargs):
        running.set()
        while True:
            checkpoint()
            time.sleep(0.001)

    monkeypatch.setattr(image_processor, 'denoise_rgb_linear', model)
    worker.update_preview(path, params)
    assert running.wait(5)
    other = tmp_path / 'other.raw'
    other.write_bytes(b'another image')
    new_path = str(other)
    new_source = np.full((12, 16, 3), 0.4, np.float32)
    worker.cache_manager.put(new_path, CachedImage(
        new_path, new_source, None, None, decode_variant='test-decode',
        source_token=stage_identity.source_identity(new_path),
    ))
    worker.load_image(new_path)
    wait_until(lambda: worker.current_path == new_path and worker._denoise_task is None)
    assert worker.preview_denoise_strength(new_path) is None
    assert worker.cpu_linear is new_source and not errors


def test_validated_noise_is_displayed_before_disk_compression(lane, monkeypatch):
    from raw_alchemy.pipeline import denoise_disk_cache

    worker, path, params, _, errors = lane
    compressing, release, displayed = (threading.Event() for _ in range(3))
    monkeypatch.setattr(image_processor, 'denoise_rgb_linear', lambda source, **kw: source * 0.5)
    original_save = denoise_disk_cache.save

    def save(*args, **kwargs):
        compressing.set()
        while not release.is_set():
            checkpoint()
            time.sleep(0.001)
        original_save(*args, **kwargs)

    monkeypatch.setattr(denoise_disk_cache, 'save', save)
    worker.result_ready.connect(
        lambda *a: displayed.set() if worker.preview_denoise_strength(path) == 0.25 else None,
        Qt.ConnectionType.DirectConnection,
    )
    worker.update_preview(path, params)
    try:
        assert compressing.wait(5)
        assert displayed.wait(5), 'cache compression delayed final preview'
        assert worker._denoise_task.is_alive()
    finally:
        release.set()
    wait_until(lambda: worker._denoise_task is None)
    assert denoise_disk_cache.load(path, 'test-0.25') is not None
    assert not errors


def test_idle_full_refine_waits_for_native_noise_source():
    worker = image_processor.ImageProcessor(warmup_sessions=False)
    request = ProcessRequest('image.raw', {'_force_full_preview': True}, 1)
    worker._full_refine_request = request
    worker._denoise_task = SimpleNamespace(ready=threading.Event(), cancelled=threading.Event())
    try:
        assert worker._take_next_request() is None
        assert worker._full_refine_request is request
        worker._denoise_task.ready.set()
        assert worker._take_next_request() is request
    finally:
        worker._denoise_task = None
