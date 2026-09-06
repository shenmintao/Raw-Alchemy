"""Real threads validate admission, cancellation and foreground interleaving."""
import threading
import time

import pytest

from raw_alchemy.pipeline.resources import ResourceGovernor, MiB
from raw_alchemy.pipeline.cancellation import cancellation_scope
from raw_alchemy.pipeline.executor import PipelineAborted


def controller(limit=4096 * MiB):
    return ResourceGovernor(limit, sample=lambda: (128 * MiB, 8192 * MiB, 8192 * MiB), wait_seconds=0.2)


def test_foreground_executes_between_export_stages_without_overlap():
    gate = controller()
    running, preview_done = threading.Event(), threading.Event()
    order = []
    errors = []
    def export():
        try:
            with gate.job(512 * MiB, priority=1):
                order.append('export-start')
                running.set()
                end = time.monotonic() + 3
                while not preview_done.is_set() and time.monotonic() < end:
                    gate.checkpoint()
                    time.sleep(0.001)
                order.append('export-end')
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=export)
    thread.start()
    assert running.wait(2)
    with gate.job(512 * MiB, priority=0):
        order.append('preview')
        preview_done.set()
    thread.join(3)
    assert not thread.is_alive()
    assert not errors
    assert order == ['export-start', 'preview', 'export-end']
    assert gate.snapshot()['reserved_bytes'] == 0


def test_over_budget_job_cannot_start_and_reservations_are_released():
    gate = controller(256 * MiB)
    with pytest.raises(MemoryError):
        with gate.job(512 * MiB):
            pytest.fail('over-budget computation started')
    assert gate.snapshot()['jobs'] == 0


def test_waiting_cancel_does_not_leak_admission_or_compute_slot():
    gate = controller()
    cancel = threading.Event()
    done = threading.Event()
    errors = []
    def waiting():
        try:
            with cancellation_scope(cancel.is_set), gate.job(128 * MiB):
                pytest.fail('cancelled waiter ran')
        except PipelineAborted:
            done.set()
        except BaseException as exc:
            errors.append(exc)
    with gate.job(128 * MiB):
        thread = threading.Thread(target=waiting)
        thread.start()
        cancel.set()
        assert done.wait(2)
    thread.join(2)
    assert not errors
    assert gate.snapshot()['jobs'] == 0
    with gate.job(128 * MiB):
        pass


def test_gui_export_snapshot_performs_no_file_identity_reads(monkeypatch):
    import numpy as np
    from raw_alchemy.workers import image_processor as module
    worker = module.ImageProcessor()
    worker.current_path = 'photo.raw'
    worker.cpu_corrected = np.zeros((2, 2, 3), np.float32)
    worker._loaded_source_token = 'captured'
    worker._denoise_policy_token = 'policy'
    monkeypatch.setattr(module, 'source_identity', lambda *a: pytest.fail('GUI hashed RAW'))
    monkeypatch.setattr(module, 'denoise_tag', lambda *a: pytest.fail('GUI hashed model'))
    snapshot = worker.get_cached_for_export()
    assert snapshot['source_token'] == 'captured'
    assert snapshot['policy_token'] == 'policy'
