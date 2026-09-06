"""Owned native sessions with bounded IPC and a killable inference lifetime.

Only small descriptors travel through the pipe. Tensor storage is allocated,
bounded and unlinked by the parent, even after a native crash or timeout.
Spawn avoids inheriting Qt, accelerator state or locks from another thread.
"""
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import re
import threading
import time
from types import SimpleNamespace

import numpy as np

from raw_alchemy.pipeline.cancellation import check_cancelled
from raw_alchemy.pipeline.resources import check_native_memory

_MAX_BYTES = 512 * 1024 * 1024
_FIELDS = (
    "enable_mem_pattern", "enable_cpu_mem_arena", "intra_op_num_threads",
    "inter_op_num_threads", "log_severity_level", "enable_profiling",
    "profile_file_prefix", "logid", "optimized_model_filepath",
)


def allocate_shared_memory(size):
    """Reject unavailable shared storage before touching a POSIX mapping."""
    import shutil
    import sys
    check_native_memory()
    size = max(1, int(size))
    if sys.platform.startswith('linux'):
        free = shutil.disk_usage('/dev/shm').free
        if size + 1024 * 1024 > free:
            raise MemoryError('Insufficient /dev/shm space for native tensors')
    return shared_memory.SharedMemory(create=True, size=size)


def _seconds(name, default):
    try:
        value = float(os.environ.get(name, default))
        return min(600.0, max(0.1, value)) if math.isfinite(value) else default
    except ValueError:
        return default


def _options_payload(options, variant):
    return {
        "fields": {key: getattr(options, key) for key in _FIELDS},
        "execution_mode": int(options.execution_mode),
        "optimization": int(options.graph_optimization_level),
        "dimensions": dict((key, int(value)) for key, value in
                           re.findall(r"(?:^|[:,])(h|w)=(\d+)", variant)),
    }


def _array(spec, handles):
    name, shape, dtype = spec
    block = shared_memory.SharedMemory(name=name)
    handles.append(block)
    return np.ndarray(shape, dtype=np.dtype(dtype), buffer=block.buf)


def _serve(connection, model, options, providers):
    """Child owns ORT until exit; no Qt or application state is passed in."""
    try:
        from .migraphx_precision import prepare_child
        prepare_child(model, options, providers)
        # CUDA DLL preloading is process-local and must happen in this child.
        from raw_alchemy.onnx.denoiser import _setup_cuda_paths
        _setup_cuda_paths()
        import onnxruntime as ort
        so = ort.SessionOptions()
        for key, value in options["fields"].items():
            setattr(so, key, value)
        so.execution_mode = ort.ExecutionMode(options["execution_mode"])
        so.graph_optimization_level = ort.GraphOptimizationLevel(options["optimization"])
        for key, value in options["dimensions"].items():
            so.add_free_dimension_override_by_name(key, value)
        session = ort.InferenceSession(model, so, providers=providers)
        connection.send(("ready", session.get_providers(), [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_inputs()
        ]))
        while True:
            command = connection.recv()
            if command[0] == "close":
                return
            if command[0] == "profile":
                connection.send(("profile", session.end_profiling()))
                continue
            if command[0] != "run":
                raise ValueError("invalid session command")
            handles = []
            try:
                feeds = {key: _array(spec, handles) for key, spec in command[2].items()}
                outputs = session.run(command[1], feeds)
                connection.send(("outputs", [(arr.shape, arr.dtype.str) for arr in outputs]))
                destination = connection.recv()
                if destination[0] != "buffers":
                    raise ValueError("invalid output buffers")
                for output, spec in zip(outputs, destination[1], strict=True):
                    np.copyto(_array(spec, handles), output)
                connection.send(("done",))
            finally:
                for handle in handles:
                    handle.close()
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send(("error", type(exc).__name__, str(exc)[:2000]))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class IsolatedSession:
    def __init__(self, model, options, providers, *, variant, worker=_serve):
        from .migraphx_precision import applies
        self._compile_default = 180.0 if applies(model, providers) else 60.0
        self._arguments = (str(model), _options_payload(options, variant), list(providers))
        self._worker = worker
        self._lock = threading.Lock()
        self._process = self._connection = None
        self._providers = []
        self._inputs = []
        self._start()

    def _receive(self, deadline):
        while True:
            check_native_memory()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("native session exceeded its time budget")
            if self._connection.poll(min(0.05, remaining)):
                result = self._connection.recv()
                if result[0] == "error":
                    error = MemoryError if result[1] == "MemoryError" else RuntimeError
                    raise error(f"native {result[1]}: {result[2]}")
                return result
            if not self._process.is_alive():
                raise RuntimeError(f"native session process exited ({self._process.exitcode})")

    def _start(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=self._worker, args=(child, *self._arguments), daemon=True,
            name="RawAlchemy-ONNX",
        )
        try:
            check_cancelled()
            self._process.start()
            child.close()
            result = self._receive(time.monotonic() + _seconds("RAWALCHEMY_NATIVE_COMPILE_TIMEOUT", _seconds("RAWALCHEMY_COREML_COMPILE_TIMEOUT", self._compile_default)))
            if result[0] != "ready":
                raise RuntimeError("invalid native session initialization")
            self._providers, inputs = result[1:]
            self._inputs = [SimpleNamespace(**item) for item in inputs]
        except BaseException:
            child.close()
            self.close()
            raise

    def get_providers(self):
        return list(self._providers)

    def get_inputs(self):
        return list(self._inputs)

    def run(self, output_names, input_feed):
        with self._lock:
            # A cancelled request may have killed this session. The next
            # request starts a fresh lifetime, without poisoning CPU fallback.
            if self._process is None:
                self._start()
            blocks, arrays = [], []
            total = 0

            def allocate(shape, dtype):
                nonlocal total
                dtype = np.dtype(dtype)
                if dtype.hasobject or any(int(n) < 0 for n in shape):
                    raise ValueError("unsupported IPC tensor")
                size = math.prod(shape) * dtype.itemsize
                total += size
                if total > _MAX_BYTES:
                    raise MemoryError("native IPC tensors exceed 512 MiB budget")
                block = allocate_shared_memory(size)
                blocks.append(block)
                array = np.ndarray(shape, dtype=dtype, buffer=block.buf)
                arrays.append(array)
                return (block.name, tuple(shape), dtype.str), array

            try:
                check_cancelled()
                descriptors = {}
                for key, value in input_feed.items():
                    value = np.asarray(value)
                    spec, array = allocate(value.shape, value.dtype)
                    np.copyto(array, value)
                    descriptors[key] = spec
                deadline = time.monotonic() + _seconds("RAWALCHEMY_NATIVE_RUN_TIMEOUT", _seconds("RAWALCHEMY_COREML_RUN_TIMEOUT", 60.0))
                self._connection.send(("run", output_names, descriptors))
                result = self._receive(deadline)
                if result[0] != "outputs":
                    raise RuntimeError("invalid native output descriptor")
                outputs = [allocate(shape, dtype) for shape, dtype in result[1]]
                self._connection.send(("buffers", [spec for spec, arr in outputs]))
                if self._receive(deadline)[0] != "done":
                    raise RuntimeError("incomplete native inference")
                return [array.copy() for spec, array in outputs]
            except BaseException:
                self.close()
                raise
            finally:
                arrays.clear()
                for block in blocks:
                    block.close()
                    block.unlink()

    def end_profiling(self):
        with self._lock:
            try:
                self._connection.send(("profile",))
                result = self._receive(time.monotonic() + 10.0)
                if result[0] != "profile":
                    raise RuntimeError("invalid profiling response")
                return result[1]
            except BaseException:
                self.close()
                raise

    def close(self):
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(0.5)
            if process.is_alive():
                process.kill()
                process.join(0.5)
            if not process.is_alive():
                process.close()
        if connection is not None:
            connection.close()

    def __del__(self):
        self.close()