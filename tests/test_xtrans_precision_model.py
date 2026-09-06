"""The CoreML precision adapter must not affect CPU, CUDA, or DirectML."""
import numpy as np
import onnxruntime as ort
import pytest

from raw_alchemy.onnx import xtrans_demosaic as xt
from raw_alchemy.onnx.denoiser import _find_model


@pytest.mark.parametrize("providers,expected", [
    (["CPUExecutionProvider"], xt.MODEL_FILE),
    (["CUDAExecutionProvider", "CPUExecutionProvider"], xt.MODEL_FILE),
    (["MIGraphXExecutionProvider", "CPUExecutionProvider"], xt.MIGRAPHX_MODEL_FILE),
    (["DmlExecutionProvider", "CPUExecutionProvider"], xt.MODEL_FILE),
    (["CoreMLExecutionProvider", "CPUExecutionProvider"], xt.COREML_MODEL_FILE),
    ([("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"})], xt.COREML_MODEL_FILE),
])
def test_model_selection_is_specific_to_coreml(providers, expected):
    assert xt.model_file_for_providers(providers) == expected


@pytest.mark.parametrize("model", [xt.COREML_MODEL_FILE, xt.MIGRAPHX_MODEL_FILE])
@pytest.mark.parametrize("case", ["flat", "near_flat", "noise", "edges"])
def test_precision_graph_preserves_original_cpu_reference(case, model):
    size = 96
    rng = np.random.default_rng(2041)
    if case == "flat":
        raw = np.full((size, size), 0.2, np.float32)
    elif case == "near_flat":
        raw = np.full((size, size), 0.2, np.float32)
        raw += rng.uniform(-1e-7, 1e-7, raw.shape).astype(np.float32)
    elif case == "noise":
        raw = rng.uniform(0, 1, (size, size)).astype(np.float32)
    else:
        raw = np.zeros((size, size), np.float32)
        raw[:, size // 2:] = 1
        raw[size // 3:size // 3 + 1] = 0.37
    feeds = {"raw": raw, "masks": xt._build_masks(xt.CANONICAL_PATTERN)}
    outputs = []
    for name in (xt.MODEL_FILE, model):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        outputs.append(ort.InferenceSession(
            _find_model(name), options, providers=["CPUExecutionProvider"],
        ).run(None, feeds)[0])
    np.testing.assert_allclose(outputs[1], outputs[0], atol=3e-6, rtol=3e-6)