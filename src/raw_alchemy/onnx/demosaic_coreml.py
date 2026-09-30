"""Repair CoreML graph lowering without changing demosaic algorithms.

Legacy NeuralNetwork lowering fails on rank-two Slice in the original graphs
(Espresso begin 4/9 against input_shape 1). MLProgram with fixed h/w session
overrides compiles and executes both graphs. Mixed CoreML/CPU runs are not
necessarily faster. X-Trans uses a precision-preserving graph variant that
keeps sensitive division on CPU. The measured ALL schedule accelerates the
repaired X-Trans graph. Both sensors use smaller CoreML tiles to bound the
mixed CPU/GPU working set.
"""

import os


def _eligible_runtime() -> bool:
    from raw_alchemy.onnx.session_policy import eligible_apple

    return eligible_apple()


def demosaic_providers(providers: list, *, variant="rcd") -> list:
    """Copy providers, requesting the working CoreML lowering path.

    Apply BEFORE cache configuration, so cache keys include these options.
    The caller must freeze h/w through ORT SessionOptions. The X-Trans caller selects its prebuilt precision graph; this policy needs
    no Apple-only import. Non-CoreML providers are unchanged.
    RAWALCHEMY_COREML_DEMOSAIC=auto|cpu|mlprogram is startup configuration.
    auto enables repaired MLProgram graphs on capable Apple runtimes;
    cpu provides an explicit override. Selection is not tied to an ORT release.
    """
    if not providers:
        return []
    first = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    if first == "MIGraphXExecutionProvider":
        from .migraphx_precision import validated_runtime
        mode = os.environ.get("RAWALCHEMY_MIGRAPHX_DEMOSAIC", "auto").strip().lower()
        # Both AMD variants have dedicated, validated graph lowering.
        isolated = os.environ.get("RAWALCHEMY_NATIVE_ISOLATION", "1") != "0"
        allowed = mode == "gpu" or (mode == "auto" and variant in {"rcd", "xtrans"} and validated_runtime())
        return list(providers) if allowed and isolated else ["CPUExecutionProvider"]
    if first != "CoreMLExecutionProvider":
        return list(providers)
    mode = os.environ.get("RAWALCHEMY_COREML_DEMOSAIC", "auto").strip().lower()
    # ALL was measured faster after the X-Trans division repair. Capability
    # checks protect the provider API; constructor/inference recovery handles
    # an unavailable accelerator without a release-specific allowlist.
    if mode not in {"auto", "cpu", "mlprogram"}:
        mode = "cpu"
    if mode == "cpu" or (mode == "auto" and not _eligible_runtime()):
        return ["CPUExecutionProvider"]
    result = []
    for entry in providers:
        name = entry[0] if isinstance(entry, tuple) else entry
        if name != "CoreMLExecutionProvider":
            result.append(entry)
            continue
        options = dict(entry[1]) if isinstance(entry, tuple) else {}
        options.update({
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "0",
        })
        result.append((name, options))
    return result


def xtrans_providers(providers: list) -> list:
    """Apply the validated X-Trans policy while preserving the one-argument API."""
    return demosaic_providers(providers, variant="xtrans")
