"""Demosaic EP recovery contracts; all ORT construction/inference is mocked.

These tests must never compile a native CoreML graph. A fake onnxruntime
module is installed for each test, and all process-global wrapper state is
restored by monkeypatch (including the intentionally persistent CPU flag).
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"


class CompilationFailure(Exception):
    """An ORT-style error not covered by ValueError/RuntimeError handlers."""


class RecursiveLockAcquisition(BaseException):
    """Fail promptly rather than hanging on a recursive non-reentrant lock."""


class GuardedLock:
    def __init__(self):
        self.held = False

    def __enter__(self):
        if self.held:
            raise RecursiveLockAcquisition("recursive session lock acquisition")
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False


class FakeOptions:
    def __init__(self):
        self.dimensions = {}
        self.runtime_settings = {"execution_mode": "sequential"}

    def add_free_dimension_override_by_name(self, name, value):
        self.dimensions[name] = value


def provider_names(providers):
    return [entry[0] if isinstance(entry, tuple) else entry for entry in providers]


def fake_session(providers, error=None):
    output = np.zeros((12, 12, 3), dtype=np.float32)
    return SimpleNamespace(
        get_providers=Mock(return_value=providers),
        run=Mock(side_effect=error, return_value=[output]),
        output=output,
    )


@pytest.fixture(params=["rcd_demosaic", "xtrans_demosaic"], ids=["rcd", "xtrans"])
def harness(request, monkeypatch):
    module = importlib.import_module(f"raw_alchemy.onnx.{request.param}")
    monkeypatch.setattr(module, "_sessions", {})
    monkeypatch.setattr(module, "_session_token", None)
    monkeypatch.setattr(module, "_session_lock", GuardedLock())
    monkeypatch.setattr(module, "_cpu_fallback", False)
    monkeypatch.setattr(module, "_session_provider", None)
    monkeypatch.setattr(module, "TILE", 12)
    if hasattr(module, "_masks"):
        monkeypatch.setattr(module, "_masks", None)

    constructor = Mock(name="InferenceSession")
    fake_ort = SimpleNamespace(InferenceSession=constructor)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model_path = f"/mock-models/{module.MODEL_FILE}"
    find_model = Mock(return_value=model_path)
    get_providers = Mock(return_value=[COREML, CPU])

    def configure(providers, *args, **kwargs):
        return [
            (provider, {"ModelFormat": "MLProgram"}) if provider == COREML else provider
            for provider in providers
        ]

    configure_providers = Mock(side_effect=configure)
    make_options = Mock(side_effect=lambda ort: FakeOptions())
    monkeypatch.setattr(module, "_find_model", find_model)
    monkeypatch.setattr(module, "_get_providers", get_providers)
    monkeypatch.setattr(module, "_configure_providers", configure_providers)
    # Dedicated tests cover graph-format adaptation, independently of recovery.
    monkeypatch.setattr(module, "demosaic_providers", lambda providers: providers)
    monkeypatch.setattr(module, "_make_session_options", make_options)
    return SimpleNamespace(
        module=module,
        constructor=constructor,
        model_path=model_path,
        find_model=find_model,
        get_providers=get_providers,
        configure=configure_providers,
        make_options=make_options,
    )


def run_tile(module, session):
    """Exercise the real recovery wrapper without invoking a native session."""
    patch = np.zeros((12, 12), dtype=np.float32)
    if module.__name__.endswith("rcd_demosaic"):
        return module._run_tile(
            session, patch, np.zeros((3, 2, 2), dtype=np.float32),
            np.ones(3, dtype=np.float32), np.eye(3, dtype=np.float32),
        )
    return module._run_graph(session, {"raw": patch, "masks": module._masks})


def requested_providers(harness):
    return [
        provider_names(call.kwargs["providers"])
        for call in harness.constructor.call_args_list
    ]


def test_accelerated_initialization_failure_retries_cpu_with_fresh_options(harness):
    h, m = harness, harness.module
    cpu = fake_session([CPU])
    h.constructor.side_effect = [CompilationFailure("CoreML compiler rejected slice"), cpu]

    assert m._get_session() is cpu
    assert m._get_session() is cpu
    assert requested_providers(h) == [[COREML, CPU], [CPU]]
    assert m._sessions == {12: cpu}
    assert m._session_provider == CPU
    assert m._cpu_fallback is True
    assert not m._session_lock.held
    calls = h.constructor.call_args_list
    assert [call.args[0] for call in calls] == [h.model_path, h.model_path]
    first_options, cpu_options = [call.args[1] for call in calls]
    assert first_options is not cpu_options
    assert first_options.dimensions == cpu_options.dimensions == {"h": 12, "w": 12}
    assert first_options.runtime_settings == cpu_options.runtime_settings
    assert h.make_options.call_count == 2
    if hasattr(m, "_masks"):
        assert m._masks.shape == (15, 6, 6)


def test_fallback_is_committed_only_after_cpu_constructor_succeeds(harness):
    h, m = harness, harness.module
    cpu = fake_session([CPU])
    compile_error = CompilationFailure("accelerator initialization failed")

    def construct(*args, **kwargs):
        assert m._cpu_fallback is False
        assert m._sessions == {}
        if provider_names(kwargs["providers"]) != [CPU]:
            raise compile_error
        return cpu

    h.constructor.side_effect = construct
    assert m._get_session() is cpu
    assert m._cpu_fallback is True
    assert h.constructor.call_count == 2


def test_initialization_fallback_invalidates_other_tiles_and_survives_clear(harness):
    h, m = harness, harness.module
    stale = fake_session([COREML, CPU])
    m._sessions[24] = stale
    cpus = [fake_session([CPU]) for _ in range(3)]
    h.constructor.side_effect = [CompilationFailure("compile"), *cpus]

    assert m._get_session() is cpus[0]
    assert m._sessions == {12: cpus[0]}
    m.TILE = 24  # patched fixture restores the original value at teardown
    assert m._get_session() is cpus[1]
    assert m._get_session() is not stale
    m.clear_session()
    assert m._sessions == {}
    assert m._session_provider is None
    assert m._cpu_fallback is True
    assert m._get_session() is cpus[2]
    assert requested_providers(h) == [[COREML, CPU], [CPU], [CPU], [CPU]]
    assert h.get_providers.call_count == 1


@pytest.mark.parametrize("providers", [[CPU], [(CPU, {})]], ids=["name", "tuple"])
def test_cpu_only_initialization_error_is_not_retried(harness, providers):
    h, m = harness, harness.module
    h.get_providers.return_value = providers
    error = CompilationFailure("CPU model is invalid")
    h.constructor.side_effect = error

    with pytest.raises(CompilationFailure) as raised:
        m._get_session()
    assert raised.value is error
    assert h.constructor.call_count == 1
    assert m._sessions == {}
    assert m._session_provider is None
    assert m._cpu_fallback is False
    assert not m._session_lock.held


def test_cpu_retry_initialization_failure_is_not_masked_or_published(harness):
    h, m = harness, harness.module
    accelerator_error = CompilationFailure("CoreML compilation failed")
    cpu_error = CompilationFailure("CPU initialization failed too")
    h.constructor.side_effect = [accelerator_error, cpu_error]

    with pytest.raises(CompilationFailure) as raised:
        m._get_session()
    assert raised.value is cpu_error
    assert raised.value.__context__ is accelerator_error
    assert h.constructor.call_count == 2
    assert m._sessions == {}
    assert m._session_provider is None
    assert m._cpu_fallback is False
    assert not m._session_lock.held


def test_accelerated_initialization_success_does_not_enable_fallback(harness):
    h, m = harness, harness.module
    accelerated = fake_session([COREML, CPU])
    h.constructor.return_value = accelerated

    assert m._get_session() is accelerated
    assert m._get_session() is accelerated
    assert h.constructor.call_count == 1
    assert m._session_provider == COREML
    assert m._cpu_fallback is False


@pytest.mark.parametrize("failing_hook", ["find_model", "configure"])
def test_setup_errors_are_not_mislabeled_as_session_compilation_failures(harness, failing_hook):
    h, m = harness, harness.module
    error = OSError("model or provider setup failed")
    getattr(h, failing_hook).side_effect = error

    with pytest.raises(OSError) as raised:
        m._get_session()
    assert raised.value is error
    h.constructor.assert_not_called()
    assert m._cpu_fallback is False
    assert m._sessions == {}


def test_accelerated_runtime_failure_recovers_and_later_tiles_adopt_cpu(harness):
    h, m = harness, harness.module
    accelerated = fake_session([COREML, CPU], error=CompilationFailure("GPU run failed"))
    cpu = fake_session([CPU])
    h.constructor.side_effect = [accelerated, cpu]
    assert m._get_session() is accelerated

    assert run_tile(m, accelerated) is cpu.output
    assert run_tile(m, accelerated) is cpu.output  # deliberately pass stale GPU object
    assert accelerated.run.call_count == 1
    assert cpu.run.call_count == 2
    assert requested_providers(h) == [[COREML, CPU], [CPU]]
    assert m._sessions == {12: cpu}
    assert m._cpu_fallback is True
    # CPU retry must receive the original feeds without rewriting input values.
    assert accelerated.run.call_args.args[1] is cpu.run.call_args_list[0].args[1]


@pytest.mark.parametrize("fallback_flag", [False, True])
def test_actual_cpu_runtime_error_rethrows_even_with_stale_provider_global(harness, fallback_flag):
    h, m = harness, harness.module
    error = CompilationFailure("CPU inference failed")
    cpu = fake_session([CPU], error=error)
    m._sessions[12] = cpu
    m._session_provider = COREML  # actual session, not process-global label, wins
    m._cpu_fallback = fallback_flag

    with pytest.raises(CompilationFailure) as raised:
        run_tile(m, cpu)
    assert raised.value is error
    cpu.run.assert_called_once()
    h.constructor.assert_not_called()
    assert m._sessions == {12: cpu}
    assert m._cpu_fallback is fallback_flag


def test_actual_accelerated_runtime_failure_recovers_despite_stale_cpu_label(harness):
    h, m = harness, harness.module
    accelerated = fake_session([COREML, CPU], error=CompilationFailure("GPU run failed"))
    cpu = fake_session([CPU])
    m._sessions[12] = accelerated
    m._session_provider = CPU
    m._cpu_fallback = True
    h.constructor.return_value = cpu

    assert run_tile(m, accelerated) is cpu.output
    assert requested_providers(h) == [[CPU]]
    accelerated.run.assert_called_once()
    cpu.run.assert_called_once()


def test_cpu_runtime_retry_error_propagates_without_third_attempt(harness):
    h, m = harness, harness.module
    gpu_error = CompilationFailure("GPU run failed")
    cpu_error = CompilationFailure("CPU run failed too")
    accelerated = fake_session([COREML, CPU], error=gpu_error)
    cpu = fake_session([CPU], error=cpu_error)
    h.constructor.side_effect = [accelerated, cpu]
    assert m._get_session() is accelerated

    with pytest.raises(CompilationFailure) as raised:
        run_tile(m, accelerated)
    assert raised.value is cpu_error
    assert raised.value.__context__ is gpu_error
    assert h.constructor.call_count == 2
    accelerated.run.assert_called_once()
    cpu.run.assert_called_once()
