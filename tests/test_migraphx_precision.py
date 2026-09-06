"""Portable checks for the precision settings proven on the AMD GPU."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from raw_alchemy.onnx import migraphx_precision as policy


@pytest.fixture
def controlled_runtime(monkeypatch):
    monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
    calls = []
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RLIMIT_STACK=3, RLIM_INFINITY=-1,
        getrlimit=lambda _: (8 * 1024 * 1024, -1),
        setrlimit=lambda limit, values: calls.append((limit, values)),
    ))
    for name in list(os.environ):
        if name.startswith(("MIGRAPHX_", "ORT_MIGRAPHX_")):
            monkeypatch.delenv(name)
    initial = {key: value for key, value in os.environ.items()
               if key.startswith(("MIGRAPHX_", "ORT_MIGRAPHX_"))}
    try:
        yield calls
    finally:
        for key in list(os.environ):
            if key.startswith(("MIGRAPHX_", "ORT_MIGRAPHX_")):
                os.environ.pop(key)
        os.environ.update(initial)


@pytest.mark.parametrize("model_name", [policy.MODEL_FILE, policy.RCD_MODEL_FILE])
def test_precision_cache_tracks_model_and_flags(tmp_path, monkeypatch, controlled_runtime, model_name):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("MIGRAPHX_DISABLE_PASSES", "rewrite_gelu")
    monkeypatch.setenv("MIGRAPHX_GPU_HIP_FLAGS", "-ffast-math")
    monkeypatch.setenv("ORT_MIGRAPHX_FP16_ENABLE", "1")
    model = tmp_path / model_name
    model.write_bytes(b"first")
    options = {"dimensions": {"h": 1560, "w": 1560}}
    providers = [policy.EP, "CPUExecutionProvider"]
    policy.prepare_child(model, options, providers)
    first = Path(os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"])
    assert first.is_dir()
    assert os.environ["MIGRAPHX_DISABLE_PASSES"] == "rewrite_gelu,simplify_algebra"
    assert os.environ["MIGRAPHX_GPU_HIP_FLAGS"].endswith("-fno-fast-math -ffp-contract=off")
    assert os.environ["ORT_MIGRAPHX_FP16_ENABLE"] == "0"
    assert controlled_runtime == [(3, (64 * 1024 * 1024, -1))]

    # A fresh child with otherwise identical effective controls must reuse it.
    monkeypatch.delenv("ORT_MIGRAPHX_MODEL_CACHE_PATH")
    monkeypatch.setenv("MIGRAPHX_GPU_HIP_FLAGS", "-ffast-math")
    policy.prepare_child(model, options, providers)
    assert Path(os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"]) == first
    monkeypatch.delenv("ORT_MIGRAPHX_MODEL_CACHE_PATH")
    monkeypatch.setenv("MIGRAPHX_GPU_HIP_FLAGS", "-ffast-math")
    model.write_bytes(b"other")  # same size still invalidates
    policy.prepare_child(model, options, providers)
    assert Path(os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"]) != first


@pytest.mark.parametrize("model,provider", [
    ("fastdenoise_v4_512_fp16.onnx", policy.EP),
    (policy.MODEL_FILE, "CUDAExecutionProvider"),
    (policy.MODEL_FILE, "CPUExecutionProvider"),
    (policy.MODEL_FILE, "CoreMLExecutionProvider"),
])
def test_other_models_and_backends_do_not_change_environment(
        model, provider, monkeypatch, controlled_runtime):
    before = dict(os.environ)
    policy.prepare_child(model, {}, [provider])
    assert dict(os.environ) == before
    assert controlled_runtime == []


def test_unwritable_cache_keeps_strict_precision(tmp_path, monkeypatch, controlled_runtime):
    model = tmp_path / policy.MODEL_FILE
    model.write_bytes(b"model")
    forbidden = tmp_path / "file"
    forbidden.write_text("not a directory")
    monkeypatch.setenv("ORT_MIGRAPHX_MODEL_CACHE_PATH", str(forbidden))
    policy.prepare_child(model, {}, [policy.EP])
    assert "ORT_MIGRAPHX_MODEL_CACHE_PATH" not in os.environ
    assert "simplify_algebra" in os.environ["MIGRAPHX_DISABLE_PASSES"]


def test_auto_demosaic_requires_isolation(monkeypatch):
    from raw_alchemy.onnx.demosaic_coreml import demosaic_providers

    monkeypatch.setenv("RAWALCHEMY_MIGRAPHX_DEMOSAIC", "auto")
    monkeypatch.setenv("RAWALCHEMY_NATIVE_ISOLATION", "1")
    monkeypatch.setattr(policy, "validated_runtime", lambda: True)
    providers = [policy.EP, "CPUExecutionProvider"]
    assert demosaic_providers(providers, variant="xtrans") == providers
    assert demosaic_providers(providers, variant="rcd") == providers
    monkeypatch.setenv("RAWALCHEMY_NATIVE_ISOLATION", "0")
    assert demosaic_providers(providers, variant="xtrans") == ["CPUExecutionProvider"]

@pytest.mark.parametrize("version,expected", [("7.2.0", True), ("6.4.0", False), ("", False)])
def test_auto_requires_measured_rocm_runtime(monkeypatch, version, expected):
    import onnxruntime as ort

    monkeypatch.setattr(policy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(policy.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(policy, "_rocm_version", lambda: version)
    monkeypatch.setattr(ort, "__version__", "1.23.2")
    monkeypatch.setenv("RAWALCHEMY_NATIVE_ISOLATION", "1")
    assert policy.validated_runtime() is expected
