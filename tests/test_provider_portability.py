"""Provider selection must respect CPU opt-out and the installed runtime build."""
from unittest.mock import Mock

import onnxruntime as ort
import pytest

from raw_alchemy.onnx import denoiser, gpu_runtime


@pytest.mark.parametrize("providers", [
    ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ["ROCMExecutionProvider", "CPUExecutionProvider"],
    ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
    ["DmlExecutionProvider", "CPUExecutionProvider"],
    ["CoreMLExecutionProvider", "CPUExecutionProvider"],
])
def test_cpu_only_never_initializes_accelerator(monkeypatch, providers):
    monkeypatch.setenv("RAW_ALCHEMY_CPU_ONLY", "1")
    monkeypatch.setattr(ort, "get_available_providers", lambda: providers)
    setup = Mock(side_effect=AssertionError("CPU mode loaded CUDA"))
    monkeypatch.setattr(gpu_runtime, "setup_cuda_dll_paths", setup)
    denoiser._setup_cuda_paths()
    assert denoiser._get_providers() == ["CPUExecutionProvider"]
    setup.assert_not_called()


@pytest.mark.parametrize("providers,expected", [
    (["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"),
    (["ROCMExecutionProvider", "CPUExecutionProvider"], "ROCMExecutionProvider"),
    (["MIGraphXExecutionProvider", "CPUExecutionProvider"], "MIGraphXExecutionProvider"),
    (["DmlExecutionProvider", "CPUExecutionProvider"], "DmlExecutionProvider"),
    (["CPUExecutionProvider"], "CPUExecutionProvider"),
])
def test_native_provider_preference_is_preserved(monkeypatch, providers, expected):
    monkeypatch.delenv("RAW_ALCHEMY_CPU_ONLY", raising=False)
    monkeypatch.setattr(ort, "get_available_providers", lambda: providers)
    assert denoiser._get_providers()[0] == expected


@pytest.mark.parametrize("providers,expected_calls", [
    (["DmlExecutionProvider", "CPUExecutionProvider"], 0),
    (["ROCMExecutionProvider", "CPUExecutionProvider"], 0),
    (["MIGraphXExecutionProvider", "CPUExecutionProvider"], 0),
    (["CPUExecutionProvider"], 0),
    (["CUDAExecutionProvider", "CPUExecutionProvider"], 1),
])
def test_cuda_preload_only_for_cuda_runtime_build(monkeypatch, providers, expected_calls):
    monkeypatch.delenv("RAW_ALCHEMY_CPU_ONLY", raising=False)
    monkeypatch.delenv("RAW_ALCHEMY_DISABLE_CUDA_PRELOAD", raising=False)
    monkeypatch.setattr(denoiser.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ort, "get_available_providers", lambda: providers)
    setup = Mock(return_value=True)
    monkeypatch.setattr(gpu_runtime, "setup_cuda_dll_paths", setup)
    denoiser._setup_cuda_paths()
    assert setup.call_count == expected_calls


def test_explicit_preload_opt_out(monkeypatch):
    monkeypatch.setenv("RAW_ALCHEMY_DISABLE_CUDA_PRELOAD", "1")
    monkeypatch.setattr(denoiser.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    setup = Mock(side_effect=AssertionError("preload disabled"))
    monkeypatch.setattr(gpu_runtime, "setup_cuda_dll_paths", setup)
    denoiser._setup_cuda_paths()
    setup.assert_not_called()

def test_linux_preloads_pip_cuda_libraries(monkeypatch):
    monkeypatch.delenv("RAW_ALCHEMY_CPU_ONLY", raising=False)
    monkeypatch.delenv("RAW_ALCHEMY_DISABLE_CUDA_PRELOAD", raising=False)
    monkeypatch.setattr(denoiser.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(gpu_runtime, "setup_cuda_dll_paths", lambda: False)
    preload = Mock()
    monkeypatch.setattr(ort, "preload_dlls", preload, raising=False)
    denoiser._setup_cuda_paths()
    preload.assert_called_once_with(cuda=True, cudnn=True, msvc=False, directory="")
