"""Decode subprocess stalls and cancellation cannot hold the GUI worker forever."""
import multiprocessing as mp
import threading
import time

import pytest

from raw_alchemy.native_decode import decode_raw
from raw_alchemy.pipeline.cancellation import cancellation_scope
from raw_alchemy.pipeline.executor import PipelineAborted


def _stalled(connection, *args):
    time.sleep(30)


@pytest.mark.parametrize('cancel', [False, True])
def test_decode_stall_is_terminated_and_reaped(monkeypatch, cancel):
    monkeypatch.setenv('RAWALCHEMY_DECODE_TIMEOUT', '10' if cancel else '0.3')
    before = {child.pid for child in mp.active_children()}
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    if cancel:
        timer.start()
    try:
        with cancellation_scope(stop.is_set), pytest.raises(PipelineAborted if cancel else TimeoutError):
            decode_raw('unused.raw', worker=_stalled)
    finally:
        timer.cancel()
    assert {child.pid for child in mp.active_children()} == before
