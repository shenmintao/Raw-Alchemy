"""Portable policy tests plus opt-in native CoreML compilation/inference."""

import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest

from raw_alchemy.onnx.demosaic_coreml import demosaic_providers


@pytest.mark.parametrize("provider", [
    "CPUExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
    "DmlExecutionProvider", ("CUDAExecutionProvider", {"device_id": 1}),
])
def test_non_coreml_unchanged(provider):
    original = [provider, "CPUExecutionProvider"]
    result = demosaic_providers(original)
    assert result == original
    assert result is not original


def test_empty_provider_sequence_remains_empty():
    assert demosaic_providers([]) == []


def test_forces_verified_compute_configuration(monkeypatch):
    monkeypatch.setenv("RAWALCHEMY_COREML_DEMOSAIC", "mlprogram")
    options = {"MLComputeUnits": "CPUAndGPU", "ModelFormat": "NeuralNetwork"}
    result = demosaic_providers([("CoreMLExecutionProvider", options)])
    assert result[0][1]["MLComputeUnits"] == "ALL"
    assert result[0][1]["ModelFormat"] == "MLProgram"
    assert options["ModelFormat"] == "NeuralNetwork"


def test_repairs_legacy_options_without_mutating_caller(monkeypatch):
    monkeypatch.setenv("RAWALCHEMY_COREML_DEMOSAIC", "mlprogram")
    options = {"ModelFormat": "NeuralNetwork", "ModelCacheDirectory": "/cache"}
    original = [("CoreMLExecutionProvider", options), "CPUExecutionProvider"]
    result = demosaic_providers(original)
    assert result[0][1] == {
        "ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1", "ModelCacheDirectory": "/cache",
        "AllowLowPrecisionAccumulationOnGPU": "0",
    }
    assert options["ModelFormat"] == "NeuralNetwork"
    assert result[1] == "CPUExecutionProvider"


@pytest.mark.parametrize("sensor", ["rcd", "xtrans"])
def test_native_coreml_compiles_and_matches_cpu(sensor, tmp_path, monkeypatch):
    if platform.system() != "Darwin" or os.environ.get("RAWALCHEMY_TEST_COREML") != "1":
        pytest.skip("opt-in real CoreML test: RAWALCHEMY_TEST_COREML=1 on macOS")
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setenv("RAWALCHEMY_COREML_DEMOSAIC", "mlprogram")
    assert "CoreMLExecutionProvider" in ort.get_available_providers()
    from raw_alchemy.onnx import rcd_demosaic as rcd, xtrans_demosaic as xt
    from raw_alchemy.onnx.denoiser import _find_model, _make_session_options

    module = rcd if sensor == "rcd" else xt
    size = 96  # covers both the 4-pixel and 9-pixel failing slices
    raw = np.random.default_rng(12).uniform(0, 1, (size, size)).astype(np.float32)
    raw[:24, :24] = 0
    if sensor == "rcd":
        masks = rcd._phase_masks(np.array([[0, 1], [3, 2]]))
        feeds = {"bayer": raw, "mr2": masks[0], "mg2": masks[1], "mb2": masks[2],
                 "wb3": np.ones(3, np.float32), "cam_mat": np.eye(3, dtype=np.float32)}
    else:
        feeds = {"raw": raw, "masks": xt._build_masks(xt.CANONICAL_PATTERN)}

    def options(profile=False):
        so = _make_session_options(ort)
        so.add_free_dimension_override_by_name("h", size)
        so.add_free_dimension_override_by_name("w", size)
        so.enable_profiling = profile
        so.profile_file_prefix = str(tmp_path / sensor)
        return so

    path = _find_model(module.MODEL_FILE)
    cpu = ort.InferenceSession(path, options(), providers=["CPUExecutionProvider"])
    expected = cpu.run(None, feeds)[0]
    del cpu
    providers = demosaic_providers(["CoreMLExecutionProvider", "CPUExecutionProvider"])
    gpu_path = _find_model(xt.model_file_for_providers(providers)) if sensor == "xtrans" else path
    session = ort.InferenceSession(gpu_path, options(True), providers=providers)
    actual = session.run(None, feeds)[0]
    events = json.loads(Path(session.end_profiling()).read_text())
    assert any(e.get("args", {}).get("provider") == "CoreMLExecutionProvider" for e in events)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("mode,affected,cpu", [
    ("auto", True, True), ("auto", False, True),
    ("cpu", True, True), ("cpu", False, True),
    ("mlprogram", True, False), ("mlprogram", False, False),
    ("invalid", True, True), ("invalid", False, True),
])
def test_measured_regression_policy(monkeypatch, mode, affected, cpu):
    from raw_alchemy.onnx import demosaic_coreml

    monkeypatch.setenv("RAWALCHEMY_COREML_DEMOSAIC", mode)
    monkeypatch.setattr(demosaic_coreml, "_measured_slow_runtime", lambda: affected)
    selected = demosaic_coreml.demosaic_providers([
        "CoreMLExecutionProvider", "CPUExecutionProvider"
    ])
    if cpu:
        assert selected == ["CPUExecutionProvider"]
    else:
        assert selected[0][1]["ModelFormat"] == "MLProgram"


@pytest.mark.parametrize("providers", [
    [], ["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ["DmlExecutionProvider", "CPUExecutionProvider"],
])
def test_policy_does_not_touch_other_backends(monkeypatch, providers):
    from raw_alchemy.onnx import demosaic_coreml

    monkeypatch.setenv("RAWALCHEMY_COREML_DEMOSAIC", "cpu")

    def unexpected_runtime_probe():
        pytest.fail("Non-CoreML backends must not probe Apple runtime")

    monkeypatch.setattr(demosaic_coreml, "_measured_slow_runtime", unexpected_runtime_probe)
    assert demosaic_coreml.demosaic_providers(providers) == providers


@pytest.mark.parametrize("system,machine,mac,version,affected", [
    ("Darwin", "arm64", "27.0", "1.29.0", True),
    ("Darwin", "arm64", "26.0", "1.29.0", False),
    ("Darwin", "arm64", "27.0", "1.28.0", False),
    ("Darwin", "x86_64", "27.0", "1.29.0", False),
    ("Windows", "AMD64", "", "1.29.0", False),
    ("Linux", "aarch64", "", "1.29.0", False),
])
def test_runtime_guard_uses_shared_measured_matrix(
    monkeypatch, system, machine, mac, version, affected
):
    import onnxruntime as ort
    from raw_alchemy.onnx import demosaic_coreml, session_policy

    monkeypatch.setattr(session_policy.platform, "system", lambda: system)
    monkeypatch.setattr(session_policy.platform, "machine", lambda: machine)
    monkeypatch.setattr(session_policy.platform, "mac_ver", lambda: (mac, (), ""))
    monkeypatch.setattr(ort, "__version__", version)
    assert demosaic_coreml._measured_slow_runtime() is affected


@pytest.mark.parametrize("mode,measured,coreml", [
    ("auto", True, True), ("auto", False, False),
    ("cpu", True, False), ("mlprogram", False, True), ("invalid", True, False),
])
def test_xtrans_precision_policy(monkeypatch, mode, measured, coreml):
    from raw_alchemy.onnx import demosaic_coreml

    monkeypatch.setenv("RAWALCHEMY_COREML_DEMOSAIC", mode)
    monkeypatch.setattr(demosaic_coreml, "_measured_slow_runtime", lambda: measured)
    providers = demosaic_coreml.xtrans_providers([
        "CoreMLExecutionProvider", "CPUExecutionProvider",
    ])
    if coreml:
        assert providers[0][1]["ModelFormat"] == "MLProgram"
        assert providers[0][1]["MLComputeUnits"] == "ALL"
        assert providers[0][1]["AllowLowPrecisionAccumulationOnGPU"] == "0"
    else:
        assert providers == ["CPUExecutionProvider"]


@pytest.mark.parametrize("mode,expected", [
    ("auto", ["CPUExecutionProvider"]), ("cpu", ["CPUExecutionProvider"]),
    ("invalid", ["CPUExecutionProvider"]),
    ("gpu", ["MIGraphXExecutionProvider", "CPUExecutionProvider"]),
])
@pytest.mark.parametrize("variant", ["rcd", "xtrans"])
def test_unaccepted_migraphx_demosaic_requires_explicit_diagnostics(monkeypatch, mode, expected, variant):
    from raw_alchemy.onnx import migraphx_precision
    monkeypatch.setattr(migraphx_precision, "validated_runtime", lambda: False)
    monkeypatch.setenv("RAWALCHEMY_MIGRAPHX_DEMOSAIC", mode)
    assert demosaic_providers([
        "MIGraphXExecutionProvider", "CPUExecutionProvider",
    ], variant=variant) == expected


def test_migraphx_override_invalidates_loaded_session(monkeypatch):
    from raw_alchemy.onnx.session_policy import configuration_token

    monkeypatch.setenv("RAWALCHEMY_MIGRAPHX_DEMOSAIC", "auto")
    before = configuration_token("xtrans")
    monkeypatch.setenv("RAWALCHEMY_MIGRAPHX_DEMOSAIC", "gpu")
    assert configuration_token("xtrans") != before
