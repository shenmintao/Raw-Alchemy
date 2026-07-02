from typing import Any, Protocol

import taichi as ti


class GpuBackend(Protocol):
    """Minimal GPU backend interface used outside kernel modules."""

    @property
    def runtime(self) -> Any:
        """Return the backend runtime module for kernel decorators and dtypes."""

    def ndarray(self, *, dtype: Any, shape: tuple[int, ...] | list[int] | Any) -> Any:
        """Allocate a backend-resident ndarray."""


class TaichiBackend:
    @property
    def runtime(self) -> Any:
        return ti

    def ndarray(self, *, dtype: Any, shape: tuple[int, ...] | list[int] | Any) -> Any:
        return ti.ndarray(dtype=dtype, shape=shape)


_BACKENDS: dict[str, GpuBackend] = {"taichi": TaichiBackend()}
_ACTIVE_BACKEND = "taichi"


def get_backend(name: str | None = None) -> GpuBackend:
    backend_name = name or _ACTIVE_BACKEND
    try:
        return _BACKENDS[backend_name]
    except KeyError as exc:
        raise ValueError(f"Unknown GPU backend: {backend_name}") from exc


def ndarray(*, dtype: Any, shape: tuple[int, ...] | list[int] | Any) -> Any:
    return get_backend().ndarray(dtype=dtype, shape=shape)
