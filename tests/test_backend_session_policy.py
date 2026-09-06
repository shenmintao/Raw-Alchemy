"""Portable backend-policy regressions; no Apple/GPU hardware is simulated as real."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import onnxruntime as ort
import pytest

from raw_alchemy.onnx import coreml_cache, denoiser, session_policy as policy

CPU = policy.CPU
COREML = policy.COREML


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    monkeypatch.setattr(policy, "_failed", set())
    for name in ("RAWALCHEMY_COREML_DENOISE", "RAWALCHEMY_COREML_GRADE",
                 "RAWALCHEMY_COREML_DEMOSAIC"):
        monkeypatch.delenv(name, raising=False)


def apple(monkeypatch, *, os_version="27.0", ort_version="1.29.0", machine="arm64"):
    monkeypatch.setattr(policy.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(policy.platform, "machine", lambda: machine)
    monkeypatch.setattr(policy.platform, "mac_ver", lambda: (os_version, (), ""))
    monkeypatch.setattr(ort, "__version__", ort_version)


def test_fastdenoise_uses_mlprogram_only_for_rgb_stage(monkeypatch):
    apple(monkeypatch)
    result = policy.stage_providers([COREML, CPU], "rgb-denoiser")
    assert result == [(COREML, {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}), CPU]
    for stage in ("raw:bayer", "raw:xtrans", "rcd:h=120,w=120", "xtrans:h=120,w=120", ""):
        assert policy.stage_providers([COREML, CPU], stage) == [COREML, CPU]


@pytest.mark.parametrize("kwargs", [
    {"os_version": "11.7"}, {"ort_version": "1.19.2"}, {"machine": "x86_64"},
])
def test_ineligible_apple_rgb_stays_cpu(monkeypatch, kwargs):
    apple(monkeypatch, **kwargs)
    assert policy.stage_providers([COREML, CPU], "rgb-denoiser") == [CPU]


@pytest.mark.parametrize("value", ["cpu", "nonsense"])
def test_denoise_cpu_and_invalid_override(monkeypatch, value):
    apple(monkeypatch)
    monkeypatch.setenv("RAWALCHEMY_COREML_DENOISE", value)
    assert policy.stage_providers([COREML, CPU], "rgb-denoiser") == [CPU]


def test_grade_cpu_is_scoped_and_overridable(monkeypatch):
    apple(monkeypatch)
    assert policy.stage_providers([COREML, CPU], "grade") == [CPU]
    monkeypatch.setenv("RAWALCHEMY_COREML_GRADE", "coreml")
    assert policy.stage_providers([COREML, CPU], "grade") == [COREML, CPU]
    monkeypatch.delenv("RAWALCHEMY_COREML_GRADE")
    apple(monkeypatch, os_version="15.5")
    assert policy.stage_providers([COREML, CPU], "grade") == [COREML, CPU]
    apple(monkeypatch, ort_version="1.30.0")
    assert policy.stage_providers([COREML, CPU], "grade") == [COREML, CPU]


@pytest.mark.parametrize("system,providers", [
    ("Linux", ["CUDAExecutionProvider", "ROCMExecutionProvider", CPU]),
    ("Windows", ["DmlExecutionProvider", CPU]),
    ("Linux", [CPU]),
])
@pytest.mark.parametrize("stage", ["rgb-denoiser", "grade", "rcd:h=120,w=120", "raw:bayer"])
def test_non_apple_policy_unchanged(monkeypatch, system, providers, stage):
    monkeypatch.setattr(policy.platform, "system", lambda: system)
    monkeypatch.setenv("RAWALCHEMY_COREML_DENOISE", "cpu")
    monkeypatch.setenv("RAWALCHEMY_COREML_GRADE", "cpu")
    assert policy.stage_providers(providers, stage) == providers
    configured = denoiser._configure_providers(providers, variant=stage)
    names = [p[0] if isinstance(p, tuple) else p for p in configured]
    assert names == providers
    if system == "Windows":
        assert configured[0] == ("DmlExecutionProvider", {"device_id": 0})
    if "CUDAExecutionProvider" in providers:
        from raw_alchemy import config
        assert configured[0][1] == {
            "device_id": 0,
            "gpu_mem_limit": max(1, int(config.ONNX_GPU_MEMORY_LIMIT_MB)) * 1024 * 1024,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": False,
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }


def test_options_are_not_mutated_and_survive_unavailable_cache(monkeypatch):
    apple(monkeypatch)
    monkeypatch.setattr(coreml_cache, "coreml_cache_dir", lambda *a, **k: None)
    options = {"EnableOnSubgraphs": "1"}
    result = denoiser._configure_providers([(COREML, options), CPU], "fastdenoise_v4_512_fp16.onnx",
                                           variant="rgb-denoiser")
    assert options == {"EnableOnSubgraphs": "1"}
    assert result[0][1]["ModelFormat"] == "MLProgram"
    assert result[0][1]["EnableOnSubgraphs"] == "1"
    assert "ModelCacheDirectory" not in result[0][1]


def test_cache_identity_changes_with_provider_options(tmp_path, monkeypatch):
    import onnx
    apple(monkeypatch)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    model = tmp_path / "model.onnx"
    model.write_bytes(onnx.helper.make_model(onnx.helper.make_graph([], "test", [], [])).SerializeToString())
    def cached(options):
        configured = denoiser._configure_providers([(COREML, options), CPU], model, variant="test")
        return configured[0][1]["ModelCacheDirectory"]
    assert cached({"ModelFormat": "MLProgram"}) != cached({"ModelFormat": "NeuralNetwork"})
    assert cached({"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}) == cached(
        {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"})


def test_configuration_identity_tracks_preferences_and_available_providers(monkeypatch):
    apple(monkeypatch)
    monkeypatch.setattr(ort, "get_available_providers", lambda: [COREML, CPU])
    initial = policy.configuration_token("rgb-denoiser")
    monkeypatch.setenv("RAWALCHEMY_COREML_DENOISE", "cpu")
    assert policy.configuration_token("rgb-denoiser") != initial
    monkeypatch.delenv("RAWALCHEMY_COREML_DENOISE")
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CPU])
    assert policy.configuration_token("rgb-denoiser") != initial


def session(providers, *, error=None):
    result = Mock()
    result.get_providers.return_value = providers
    result.run = Mock(side_effect=error, return_value=["result"])
    return result


def runtime(factory):
    return SimpleNamespace(InferenceSession=Mock(side_effect=factory))


def create(runtime_, providers=None, path="test.onnx", variant="rgb-denoiser"):
    return policy.create_session(runtime_, path, lambda: object(), providers or [COREML, CPU],
                                 variant=variant)


@pytest.mark.parametrize("provider", [COREML, "CUDAExecutionProvider", "DmlExecutionProvider"])
def test_constructor_failure_retries_cpu_with_fresh_options_and_breaker(provider):
    cpu = session([CPU])
    runtime_ = runtime([RuntimeError("compile failed"), cpu, cpu])
    assert create(runtime_, [provider, CPU]) is cpu
    assert create(runtime_, [provider, CPU]) is cpu
    calls = runtime_.InferenceSession.call_args_list
    assert [c.kwargs["providers"] for c in calls] == [[provider, CPU], [CPU], [CPU]]
    assert calls[0].args[1] is not calls[1].args[1]


def test_runtime_failure_retries_same_feed_once_and_preserves_session_api():
    gpu = session([COREML, CPU], error=RuntimeError("device lost"))
    cpu = session([CPU])
    runtime_ = runtime([gpu, cpu])
    wrapper = create(runtime_)
    assert wrapper.get_inputs() is gpu.get_inputs.return_value
    feeds = {"rgb": object()}
    assert wrapper.run(None, feeds) == ["result"]
    assert wrapper.run(None, feeds) == ["result"]
    gpu.run.assert_called_once_with(None, feeds)
    assert cpu.run.call_count == 2
    assert cpu.run.call_args.args[1] is feeds
    assert wrapper.get_providers() == [CPU]


def test_concurrent_runtime_failure_constructs_cpu_once():
    gpu = session([COREML, CPU], error=RuntimeError("device lost"))
    cpu = session([CPU])
    runtime_ = runtime([gpu, cpu])
    wrapper = create(runtime_)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: wrapper.run(None, {}), range(8))) == [["result"]] * 8
    assert runtime_.InferenceSession.call_count == 2
    gpu.run.assert_called_once()


def test_cpu_constructor_and_execution_errors_are_not_hidden():
    runtime_ = runtime([RuntimeError("CPU initialization error")])
    with pytest.raises(RuntimeError, match="CPU initialization error"):
        create(runtime_, [CPU])
    assert runtime_.InferenceSession.call_count == 1
    gpu = session([COREML, CPU], error=RuntimeError("GPU error"))
    cpu = session([CPU], error=ValueError("invalid feed"))
    runtime_ = runtime([gpu, cpu])
    wrapper = create(runtime_)
    with pytest.raises(ValueError, match="invalid feed"):
        wrapper.run(None, {})
    with pytest.raises(ValueError, match="invalid feed"):
        wrapper.run(None, {})
    assert runtime_.InferenceSession.call_count == 2


def test_silent_ort_constructor_fallback_opens_circuit_breaker():
    cpu = session([CPU])
    runtime_ = runtime([cpu, cpu])
    create(runtime_)
    create(runtime_)
    assert runtime_.InferenceSession.call_args.kwargs["providers"] == [CPU]


def test_provider_options_and_model_replacement_reopen_circuit(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"original")
    cpu = session([CPU])
    runtime_ = runtime([cpu, cpu, cpu])
    create(runtime_, path=model)
    create(runtime_, [(COREML, {"ModelFormat": "MLProgram"}), CPU], path=model)
    assert runtime_.InferenceSession.call_args.kwargs["providers"][0][1]["ModelFormat"] == "MLProgram"
    model.write_bytes(b"new model contents")
    create(runtime_, path=model)
    assert runtime_.InferenceSession.call_args.kwargs["providers"] == [COREML, CPU]


def test_logging_distinguishes_requested_attempted_and_registered(monkeypatch):
    log = Mock()
    monkeypatch.setattr(policy.logger, "info", log)
    cpu = session([CPU])
    runtime_ = runtime([cpu, cpu])
    create(runtime_)
    create(runtime_)
    line = log.call_args.args[0]
    assert "requested=" in line and COREML in line
    assert f"attempted=((\'{CPU}\', ()),)" in line
    assert f"registered=['{CPU}']" in line
    assert "do not prove node placement" in line


@pytest.mark.parametrize("module_name,args,stage", [
    ("rgb_denoiser", (), "rgb-denoiser"),
    ("denoiser", ("bayer",), "raw:bayer"),
    ("denoiser", ("xtrans",), "raw:xtrans"),
    ("grade", ("grade_dyn.onnx",), "grade"),
])
def test_stage_factories_recover_and_do_not_retry_after_clear(monkeypatch, module_name, args, stage):
    import importlib
    module = importlib.import_module(f"raw_alchemy.onnx.{module_name}")
    for name, value in (("_session", None), ("_session_token", None), ("_sessions", {}),
                        ("_session_tokens", {}), ("_session_bayer", None), ("_session_xtrans", None)):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "_find_model", lambda _: "mock-model.onnx")
    monkeypatch.setattr(module, "_get_providers", lambda: [COREML, CPU])
    monkeypatch.setattr(module, "_configure_providers", lambda *a, **k: [COREML, CPU])
    cpu = session([CPU])
    constructor = Mock(side_effect=[RuntimeError("compile error"), cpu, cpu])
    monkeypatch.setattr(ort, "InferenceSession", constructor)
    assert module._get_session(*args) is cpu
    module.clear_session()
    assert module._get_session(*args) is cpu
    assert constructor.call_args.kwargs["providers"] == [CPU]


@pytest.mark.parametrize('phase', ['initialize', 'run'])
def test_memory_budget_failure_does_not_retry_or_blacklist_accelerator(phase):
    gpu = session([COREML, CPU], error=MemoryError('budget'))
    rt = runtime([MemoryError('budget')] if phase == 'initialize' else [gpu])
    with pytest.raises(MemoryError, match='budget'):
        result = create(rt)
        result.run(None, {})
    assert rt.InferenceSession.call_count == 1
    assert not policy._failed