"""Decode subprocess stalls and cancellation cannot hold the GUI worker forever."""
import multiprocessing as mp
import os
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from raw_alchemy import native_decode
from raw_alchemy.native_decode import decode_raw
from raw_alchemy.pipeline.cancellation import cancellation_scope
from raw_alchemy.pipeline.executor import PipelineAborted


def _stalled(connection, *args):
    time.sleep(30)


def _pid_server(connection):
    """Persistent child for tests: answers each request with its own pid."""
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message[0] != 'decode':
            return
        if message[1] == 'stall.raw':
            time.sleep(30)
        connection.send(('result', (1, 1, 3), ()))
        block = shared_memory.SharedMemory(name=connection.recv())
        np.ndarray((1, 1, 3), np.float32, buffer=block.buf)[:] = os.getpid()
        block.close()
        connection.send(('done',))


def test_decode_process_is_reused_between_images():
    server = native_decode._DecodeServer(target=_pid_server)
    try:
        first = server.decode('a.raw')
        second = server.decode('b.raw')
        assert first[0, 0, 0] == second[0, 0, 0] == server.pid
    finally:
        server.stop()
    assert server.pid is None


def test_cancelled_decode_kills_the_process_and_the_next_one_respawns():
    before = {child.pid for child in mp.active_children()}
    server = native_decode._DecodeServer(target=_pid_server)
    try:
        first_pid = server.decode('a.raw')[0, 0, 0]
        stop = threading.Event()
        timer = threading.Timer(0.2, stop.set)
        timer.start()
        with cancellation_scope(stop.is_set), pytest.raises(PipelineAborted):
            server.decode('stall.raw')
        timer.cancel()
        assert server.pid is None  # the busy child was killed
        assert server.decode('c.raw')[0, 0, 0] != first_pid
    finally:
        server.stop()
    assert {child.pid for child in mp.active_children()} == before


def test_release_idle_decoders_stops_idle_processes(monkeypatch):
    server = native_decode._DecodeServer(target=_pid_server)
    monkeypatch.setattr(native_decode, "_servers", {threading.get_ident(): server})
    server.decode('a.raw')
    assert server.pid is not None
    native_decode.release_idle_decoders()
    assert server.pid is None


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
