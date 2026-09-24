"""Killable RAW decode boundary on Windows, Linux and macOS.

Only path/mode and small metadata use the pipe; the parent owns pixel storage.
Decoding in the child includes its ONNX lifetime, so it does not spawn again.

Each calling thread keeps one decode process alive between images, so imports
and the demosaic sessions stay warm (a fresh spawn cost ~1-1.7 s per open).
A cancelled, timed-out or failed decode kills that process -- the only way to
stop native work mid-flight -- and the next request starts a new one.
:func:`release_idle_decoders` shuts idle processes down to free their memory.
"""
import atexit
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import threading
import time

import numpy as np
from loguru import logger

from raw_alchemy.onnx.isolated_session import _seconds, allocate_shared_memory
from raw_alchemy.pipeline.resources import check_native_memory
from raw_alchemy.pipeline.cancellation import check_cancelled


def _isolate_child_sessions():
    os.environ['RAWALCHEMY_NATIVE_ISOLATION'] = '0'
    os.environ['RAWALCHEMY_COREML_ISOLATION'] = '0'


def _decode_one(connection, path, mode):
    """Decode one file in the child and hand it over; keeps the pipe open."""
    block = None
    try:
        if mode == 'canonical':
            from raw_alchemy.core import _rawpy_decode_to_prophoto
            result = _rawpy_decode_to_prophoto(path)
            metadata = ()
        elif mode == 'preload':
            result, *metadata = decode_cpu(path)
        else:
            raise ValueError('Unknown RAW decoder')
        result = np.ascontiguousarray(result, dtype=np.float32)
        connection.send(('result', result.shape, metadata))
        name = connection.recv()
        block = shared_memory.SharedMemory(name=name)
        np.copyto(np.ndarray(result.shape, dtype=np.float32, buffer=block.buf), result)
        connection.send(('done',))
    except BaseException as exc:
        try:
            connection.send(('error', type(exc).__name__, str(exc)[:2000]))
        except (OSError, EOFError):
            pass
    finally:
        if block is not None:
            block.close()


def _serve_decode(connection, path, mode):
    """One-shot child: decode a single file, then exit."""
    _isolate_child_sessions()
    try:
        _decode_one(connection, path, mode)
    finally:
        connection.close()


def _serve_forever(connection):
    """Persistent child: decode requests until told to stop or orphaned."""
    _isolate_child_sessions()
    try:
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                break
            if not message or message[0] != 'decode':
                break
            _decode_one(connection, message[1], message[2])
    finally:
        connection.close()


def _exchange(connection, process, mode, deadline):
    """Receive one decode from a running child into parent-owned memory."""
    def receive():
        while True:
            check_native_memory()
            if time.monotonic() >= deadline:
                raise TimeoutError('RAW decoding exceeded its time budget')
            if connection.poll(0.05):
                message = connection.recv()
                if message[0] == 'error':
                    error = MemoryError if message[1] == 'MemoryError' else RuntimeError
                    raise error(f'RAW {message[1]}: {message[2]}')
                return message
            if not process.is_alive():
                raise RuntimeError(f'RAW decoder exited ({process.exitcode})')

    block = None
    try:
        message = receive()
        if message[0] != 'result':
            raise RuntimeError('Invalid RAW result')
        shape, metadata = message[1:]
        if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0 or shape[0] * shape[1] > 200_000_000:
            raise ValueError('Invalid RAW output dimensions')
        block = allocate_shared_memory(int(np.prod(shape)) * 4)
        connection.send(block.name)
        if receive()[0] != 'done':
            raise RuntimeError('Incomplete RAW output')
        result = np.ndarray(shape, dtype=np.float32, buffer=block.buf).copy()
        return (result, *metadata) if mode == 'preload' else result
    finally:
        if block is not None:
            block.close()
            block.unlink()


def _reap(process, grace=0.0):
    if process.pid is None:
        return
    if grace > 0:
        process.join(grace)
    if process.is_alive():
        process.terminate()
    process.join(0.5)
    if process.is_alive():
        process.kill()
        process.join(0.5)
    if not process.is_alive():
        process.close()


def _decode_in_fresh_process(path, mode, worker):
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child, str(path), mode),
                              name='RawAlchemy-Decode', daemon=True)
    deadline = time.monotonic() + _seconds('RAWALCHEMY_DECODE_TIMEOUT', 120)
    try:
        check_cancelled()
        process.start()
        child.close()
        return _exchange(parent, process, mode, deadline)
    finally:
        child.close()
        _reap(process)
        parent.close()


class _DecodeServer:
    """One persistent decode process, used by one thread at a time."""

    def __init__(self, target=None):
        self._target = target or _serve_forever
        self._process = None
        self._connection = None
        self.lock = threading.Lock()  # held for a request or a shutdown

    @property
    def pid(self):
        process = self._process
        return process.pid if process is not None and process.is_alive() else None

    def decode(self, path, mode='canonical'):
        deadline = time.monotonic() + _seconds('RAWALCHEMY_DECODE_TIMEOUT', 120)
        check_cancelled()
        completed = False
        try:
            if self._process is None or not self._process.is_alive():
                self._start()
            self._connection.send(('decode', str(path), mode))
            result = _exchange(self._connection, self._process, mode, deadline)
            completed = True
            return result
        finally:
            if not completed:
                self.stop()  # kill mid-flight work; the next request respawns

    def _start(self):
        self.stop()
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=self._target, args=(child,),
                                  name='RawAlchemy-Decode', daemon=True)
        process.start()
        child.close()
        self._process, self._connection = process, parent

    def stop(self):
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if connection is not None:
            try:
                connection.send(('stop',))  # lets an idle child exit cleanly
            except (OSError, EOFError, ValueError):
                pass
        if process is not None:
            _reap(process, grace=0.5)
        if connection is not None:
            connection.close()


_servers = {}  # thread ident -> _DecodeServer
_servers_lock = threading.Lock()


def _server_for_current_thread():
    ident = threading.get_ident()
    with _servers_lock:
        live = {thread.ident for thread in threading.enumerate()}
        for stale in [key for key in _servers if key not in live]:
            _servers.pop(stale).stop()  # its thread is gone, so it is idle
        server = _servers.get(ident)
        if server is None:
            server = _servers[ident] = _DecodeServer()
        return server


def decode_raw(path, mode='canonical', *, worker=None):
    """Decode ``path`` in a child process (``worker`` = one-shot child)."""
    if worker is not None:
        return _decode_in_fresh_process(path, mode, worker)
    server = _server_for_current_thread()
    with server.lock:
        reused = server.pid is not None
        t0 = time.perf_counter()
        result = server.decode(path, mode)
        logger.debug(
            f"[Decode] {mode} {os.path.basename(str(path))} on "
            f"{threading.current_thread().name} "
            f"({'warm' if reused else 'new'} process) {time.perf_counter() - t0:.2f}s"
        )
        return result


def release_idle_decoders():
    """Stop decode processes that are not decoding, freeing RAM and VRAM."""
    with _servers_lock:
        servers = list(_servers.values())
    for server in servers:
        if server.lock.acquire(blocking=False):
            try:
                server.stop()
            finally:
                server.lock.release()


atexit.register(release_idle_decoders)


def decode_cpu(path: str):
    """LibRaw neighbour preview; its provenance differs from canonical decode.

    White balance and the working-space transform are shared, but demosaic
    algorithms differ. These pixels must not seed the canonical export cache.
    """
    from raw_alchemy.math_ops import apply_matrix_inplace
    import rawpy
    from raw_alchemy.colorspace_matrices import cam_to_working_space_matrix
    from raw_alchemy.onnx.denoiser import _apply_flip
    from raw_alchemy.exif import extract_lens_exif

    with rawpy.imread(path) as raw:
        wb = np.array(raw.camera_whitebalance, dtype=np.float32)
        flip = raw.sizes.flip
        xyz = np.array(raw.rgb_xyz_matrix, dtype=np.float64)
        # Camera-native demosaic only: unit WB, no auto-bright, linear,
        # unflipped — this app owns white balance, colour and orientation.
        cam = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True,
            user_wb=[1.0, 1.0, 1.0, 1.0], output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            user_flip=0, half_size=False, highlight_mode=2,
        )
    rgb = cam.astype(np.float32) / 65535.0
    del cam
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    elif rgb.shape[2] > 3:
        rgb = np.ascontiguousarray(rgb[:, :, :3])

    g = wb[1] if wb[1] > 0 else 1.0
    rgb[:, :, 0] *= wb[0] / g
    rgb[:, :, 2] *= wb[2] / g
    # cam->working matrix, per-pixel (M @ [r,g,b]); numpy keeps it GPU-free.
    m = cam_to_working_space_matrix(xyz).astype(np.float32)
    apply_matrix_inplace(rgb, m)
    rgb = np.ascontiguousarray(_apply_flip(rgb, flip))
    np.clip(rgb, 0.0, 1.0, out=rgb)
    exif_data, exif_metadata = extract_lens_exif(path, None)
    return rgb, exif_data, exif_metadata
