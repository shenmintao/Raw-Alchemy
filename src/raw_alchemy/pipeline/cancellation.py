"""Thread-local cancellation shared by decode, inference and export stages."""
from contextlib import contextmanager
import threading

_state = threading.local()


def check_cancelled():
    callback = getattr(_state, "callback", None)
    if callback is not None and callback():
        # Lazy import avoids importing the image pipeline in native workers.
        from .executor import PipelineAborted
        raise PipelineAborted("processing cancelled")


@contextmanager
def cancellation_scope(callback):
    previous = getattr(_state, "callback", None)
    _state.callback = callback
    try:
        check_cancelled()
        yield
    finally:
        _state.callback = previous
