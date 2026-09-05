"""CoreML cache safety without requiring macOS or a CoreML-capable ORT build."""

import builtins
import os
from pathlib import Path
from unittest.mock import Mock

import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import pytest

from raw_alchemy.onnx import coreml_cache as cache
from raw_alchemy.onnx import denoiser


COREML = "CoreMLExecutionProvider"
CPU = "CPUExecutionProvider"


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cache.platform, "system", lambda: "Darwin")
    path = tmp_path / "model.onnx"
    model = helper.make_model(helper.make_graph([], "cache-test", [], []))
    path.write_bytes(model.SerializeToString())
    return path


def configured(path=None, variant=""):
    return denoiser._configure_providers([COREML, CPU], path, variant=variant)


def cache_path(path, variant=""):
    providers = configured(path, variant)
    assert providers[0][0] == COREML
    assert providers[1] == CPU
    result = providers[0][1]["ModelCacheDirectory"]
    assert Path(result).is_dir()
    return result


def test_no_model_path_skips_cache_and_onnx_import(model_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "onnx"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert configured() == [COREML, CPU]
    assert not (model_path.parent / "Library").exists()


def test_same_contents_reuse_and_same_path_model_change(model_path):
    first = cache_path(model_path)
    assert cache_path(model_path) == first
    copy = model_path.with_name("copy.onnx")
    copy.write_bytes(model_path.read_bytes())
    assert cache_path(copy) == first
    model = onnx.load_model(model_path)
    model.doc_string = "changed bytes, unchanged path"
    model_path.write_bytes(model.SerializeToString())
    assert cache_path(model_path) != first


def external_tensor(name, location):
    tensor = TensorProto(name=name, data_type=TensorProto.FLOAT, dims=[1])
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value=location)
    return tensor


@pytest.mark.parametrize("placement", [
    "initializer", "tensor", "tensors", "graph", "graphs", "sparse", "function",
])
def test_external_weights_invalidate_including_nested_tensors(model_path, placement):
    weights = model_path.parent / "weights.bin"
    weights.write_bytes(b"abcd")
    tensor = external_tensor("weight", weights.name)
    model = onnx.load_model(model_path)
    graph = model.graph
    if placement == "initializer":
        graph.initializer.append(tensor)
    elif placement == "sparse":
        sparse = graph.sparse_initializer.add()
        sparse.values.CopyFrom(tensor)
        sparse.indices.CopyFrom(helper.make_tensor("indices", TensorProto.INT64, [1], [0]))
        sparse.dims.append(1)
    else:
        node = helper.make_node("CacheTest", [], ["out"], domain="test")
        if placement in ("graph", "graphs"):
            # Graph attribute -> node -> tensor attribute (two nested levels).
            inner_node = helper.make_node("Constant", [], ["inner"], value=tensor)
            inner = helper.make_graph([inner_node], "inner", [], [])
            node.attribute.append(helper.make_attribute(
                "body", [inner] if placement == "graphs" else inner
            ))
        else:
            node.attribute.append(helper.make_attribute(
                "value", [tensor] if placement == "tensors" else tensor
            ))
        if placement == "function":
            function = model.functions.add(name="nested", domain="test")
            function.node.append(node)
        else:
            graph.node.append(node)
    model_path.write_bytes(model.SerializeToString())
    first = cache_path(model_path)
    assert cache_path(model_path) == first
    # Same byte count and deliberately preserved timestamps defeat stat-only keys.
    stat = weights.stat()
    weights.write_bytes(b"wxyz")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cache_path(model_path) != first
    weights.unlink()
    assert configured(model_path) == [COREML, CPU]


def test_all_referenced_external_files_are_hashed(model_path):
    model = onnx.load_model(model_path)
    paths = [model_path.parent / name for name in ("first.bin", "second.bin")]
    for index, path in enumerate(paths):
        path.write_bytes(b"abcd")
        model.graph.initializer.append(external_tensor(f"w{index}", path.name))
    model_path.write_bytes(model.SerializeToString())
    namespace = cache_path(model_path)
    for path in paths:
        path.write_bytes(b"changed")
        changed = cache_path(model_path)
        assert changed != namespace
        namespace = changed


@pytest.mark.parametrize("failure", ["mkdir", "existing_probe", "probe_write", "hash", "parse"])
def test_cache_failure_warns_and_preserves_coreml(model_path, monkeypatch, failure):
    warning = Mock()
    monkeypatch.setattr(cache.logger, "warning", warning)
    if failure == "mkdir":
        monkeypatch.setattr(Path, "mkdir", Mock(side_effect=PermissionError("denied")))
    elif failure in ("existing_probe", "probe_write"):
        existing = cache_path(model_path)
        assert Path(existing).is_dir()
        # Deterministic even on privileged CI; simulate EACCES on an existing dir.
        if failure == "existing_probe":
            probe = Mock(side_effect=PermissionError("read-only cache"))
        else:
            handle = Mock()
            handle.write.side_effect = OSError("disk full")
            probe = Mock()
            probe.return_value.__enter__ = Mock(return_value=handle)
            probe.return_value.__exit__ = Mock(return_value=False)
        monkeypatch.setattr(cache.tempfile, "TemporaryFile", probe)
    elif failure == "hash":
        monkeypatch.setattr(cache.hashlib, "sha256", Mock(side_effect=OSError("hash failed")))
    else:
        model_path.write_bytes(b"not an ONNX protobuf")
    assert configured(model_path) == [COREML, CPU]
    warning.assert_called_once()
    assert "continuing without cache" in warning.call_args.args[0]


def test_missing_onnx_only_disables_cache(model_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("optional parser unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert configured(model_path) == [COREML, CPU]


def test_session_variant_runtime_and_system_isolation(model_path, monkeypatch):
    first = cache_path(model_path, "rcd:h=1536,w=1536")
    assert cache_path(model_path, "rcd:h=768,w=768") != first
    assert cache_path(model_path, "xtrans:h=1536,w=1536") != first
    with monkeypatch.context() as patch:
        patch.setattr(ort, "__version__", "different-runtime")
        assert cache_path(model_path, "rcd:h=1536,w=1536") != first
    with monkeypatch.context() as patch:
        patch.setattr(cache.platform, "version", lambda: "different-system")
        assert cache_path(model_path, "rcd:h=1536,w=1536") != first


@pytest.mark.parametrize("system", ["Linux", "Windows", "Darwin"])
def test_other_platform_providers_unchanged(model_path, monkeypatch, system):
    monkeypatch.setattr(cache.platform, "system", lambda: system)
    monkeypatch.setattr(cache, "coreml_cache_dir", Mock(side_effect=AssertionError("unexpected cache")))
    providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider", CPU]
    result = denoiser._configure_providers(providers, model_path)
    assert result == denoiser._configure_providers(providers)
    assert result[0][1]["cudnn_conv_use_max_workspace"] is False
    assert result[1:3] == [(name, {"device_id": 0}) for name in providers[1:3]]
    assert result[3] == CPU
    if system != "Darwin":
        assert configured(model_path) == [COREML, CPU]


@pytest.mark.parametrize("module_name,args,variant", [
    ("denoiser", ("bayer",), "raw:bayer"),
    ("rgb_denoiser", (), "rgb-denoiser"),
    ("grade", ("grade.onnx",), "grade"),
    ("rcd_demosaic", (), "rcd:h=120,w=120"),
    ("xtrans_demosaic", (), "xtrans:h=120,w=120"),
])
def test_all_session_sites_forward_model_and_variant(model_path, monkeypatch,
                                                     module_name, args, variant):
    import importlib
    module = importlib.import_module(f"raw_alchemy.onnx.{module_name}")
    for name, value in [("_sessions", {}), ("_session", None), ("_session_bayer", None),
                        ("_session_provider", None), ("_session_bayer_provider", None),
                        ("TILE", 120), ("_masks", None), ("_w_full", None)]:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "_find_model", lambda _: str(model_path))
    monkeypatch.setattr(module, "_get_providers", lambda: [CPU])
    configure = Mock(return_value=[CPU])
    monkeypatch.setattr(module, "_configure_providers", configure)
    session = Mock()
    session.get_providers.return_value = [CPU]
    monkeypatch.setattr(ort, "InferenceSession", Mock(return_value=session))
    assert module._get_session(*args) is session
    configure.assert_called_once_with([CPU], str(model_path), variant=variant)
