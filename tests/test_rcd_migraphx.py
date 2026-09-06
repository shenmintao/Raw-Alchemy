"""Asset and fallback contracts for fixed-tile AMD Bayer demosaic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

from raw_alchemy.onnx import rcd_demosaic as rcd


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider",
                                     "DmlExecutionProvider", "CoreMLExecutionProvider"])
def test_other_backends_keep_original_model(provider):
    assert rcd.model_file_for_providers([provider]) == rcd.MODEL_FILE


@pytest.mark.parametrize("tile", [96, 1536])
def test_amd_asset_requires_its_fixed_tile(monkeypatch, tile):
    monkeypatch.setattr(rcd, "TILE", tile)
    expected = rcd.MIGRAPHX_MODEL_FILE if tile == 1536 else rcd.MODEL_FILE
    assert rcd.model_file_for_providers([("MIGraphXExecutionProvider", {})]) == expected


@pytest.mark.parametrize("tile", [96, 1536])
def test_compilation_failure_and_other_sizes_use_original_cpu(monkeypatch, tile):
    monkeypatch.setattr(rcd, "TILE", tile)
    monkeypatch.setattr(rcd, "_sessions", {})
    monkeypatch.setattr(rcd, "_cpu_fallback", False)
    monkeypatch.setattr(rcd, "_session_token", None)
    monkeypatch.setattr(rcd, "_session_provider", None)
    monkeypatch.setattr(rcd, "configuration_token", lambda _: "test")
    monkeypatch.setattr(rcd, "_get_providers", lambda: ["MIGraphXExecutionProvider"])
    monkeypatch.setattr(rcd, "demosaic_providers", lambda providers: providers)
    monkeypatch.setattr(rcd, "_find_model", lambda name: name)
    monkeypatch.setattr(rcd, "_configure_providers", lambda providers, *a, **kw: providers)
    calls = []

    def construct(runtime, model, options, providers, **kwargs):
        calls.append((model, providers))
        if providers == ["MIGraphXExecutionProvider"]:
            raise RuntimeError("compiler failure")
        return SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    monkeypatch.setattr(rcd, "construct_session", construct)
    assert rcd._get_session().get_providers() == ["CPUExecutionProvider"]
    assert calls[-1] == (rcd.MODEL_FILE, ["CPUExecutionProvider"])
    if tile == 1536:
        assert calls[0][0] == rcd.MIGRAPHX_MODEL_FILE
        assert rcd._cpu_fallback
    else:
        assert len(calls) == 1


def test_generator_reproduces_shipped_asset_and_rejects_changed_source(tmp_path):
    pytest.importorskip("onnx")
    path = Path(__file__).parents[1] / "tools/build_migraphx_rcd.py"
    spec = importlib.util.spec_from_file_location("build_migraphx_rcd", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    destination = tmp_path / "rcd.onnx"
    assert builder.build(Path(rcd._find_model(rcd.MODEL_FILE)), destination) == 3
    assert destination.read_bytes() == Path(rcd._find_model(rcd.MIGRAPHX_MODEL_FILE)).read_bytes()
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"unexpected source")
    with pytest.raises(ValueError, match="source changed"):
        builder.build(bad, destination)


@pytest.fixture(scope="module")
def reference_and_candidate():
    sessions = []
    for model in (rcd.MODEL_FILE, rcd.MIGRAPHX_MODEL_FILE):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.add_free_dimension_override_by_name("h", 1536)
        options.add_free_dimension_override_by_name("w", 1536)
        sessions.append(ort.InferenceSession(rcd._find_model(model), options,
                                            providers=["CPUExecutionProvider"]))
    return sessions


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_gather_asset_preserves_cfa_wb_and_matrix(reference_and_candidate, phase):
    raw = np.random.default_rng(904).uniform(0, 1, (1536, 1536)).astype(np.float32)
    pattern = np.roll(np.roll([[0, 1], [3, 2]], phase // 2, axis=0), phase % 2, axis=1)
    masks = rcd._phase_masks(pattern)
    feeds = {"bayer": raw, "mr2": masks[0], "mg2": masks[1], "mb2": masks[2],
             "wb3": np.array([2.1, 1.0, 1.6], np.float32),
             "cam_mat": np.array([[1.1, -.08, -.02], [-.03, 1.08, -.05],
                                   [-.02, -.1, 1.12]], np.float32)}
    expected, actual = [session.run(None, feeds)[0] for session in reference_and_candidate]
    np.testing.assert_array_equal(actual, expected)
