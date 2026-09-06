"""Opt-in real-frame quality gate, distinct from tiny compiler smoke tests."""
import os
import platform

import numpy as np
import pytest


def test_native_xtrans_real_frame_strict_precision(monkeypatch):
    """Known failing RAF gate. Never loosen tolerance or xfail to bless CoreML."""
    raw_path = os.environ.get("RAWALCHEMY_TEST_XTRANS_RAW")
    if (
        platform.system() != "Darwin"
        or os.environ.get("RAWALCHEMY_TEST_COREML") != "1"
        or not raw_path
    ):
        pytest.skip("opt-in real RAF quality gate: RAWALCHEMY_TEST_XTRANS_RAW")
    rawpy = pytest.importorskip("rawpy")
    ort = pytest.importorskip("onnxruntime")
    assert "CoreMLExecutionProvider" in ort.get_available_providers()
    from raw_alchemy.core import (
        fix_hot_pixels,
        highlight_inpaint_opposed,
        subtract_black_level,
    )
    from raw_alchemy.onnx import xtrans_demosaic as xt

    monkeypatch.setenv("RAWALCHEMY_COREML_DEMOSAIC", "mlprogram")
    units = os.environ.get("RAWALCHEMY_TEST_COREML_COMPUTE_UNITS")
    if units:
        assert units in {"ALL", "CPUAndGPU", "CPUOnly"}
        original_providers = xt.demosaic_providers

        def precision_providers(providers):
            configured = original_providers(providers)
            return [
                (name, {**options, "MLComputeUnits": units,
                        "AllowLowPrecisionAccumulationOnGPU": "0"})
                if name == "CoreMLExecutionProvider" else (name, options)
                for name, options in (
                    entry if isinstance(entry, tuple) else (entry, {})
                    for entry in configured
                )
            ]

        monkeypatch.setattr(xt, "demosaic_providers", precision_providers)
    with rawpy.imread(raw_path) as frame:
        pattern = frame.raw_pattern.copy()
        assert pattern.shape == (6, 6), "quality gate requires an X-Trans RAW"
        raw = subtract_black_level(
            frame.raw_image_visible.astype(np.float32),
            np.array(frame.black_level_per_channel, np.float32),
            float(frame.white_level), pattern,
        )
        fix_hot_pixels(raw, pattern)
        highlight_inpaint_opposed(
            raw, pattern, np.array(frame.camera_whitebalance, np.float32),
        )
    try:
        xt.clear_session()
        monkeypatch.setattr(xt, "_get_providers", lambda: ["CPUExecutionProvider"])
        reference = xt.xtrans_markesteijn_demosaic(raw, pattern)
        xt.clear_session()
        monkeypatch.setattr(xt, "_get_providers", lambda: [
            "CoreMLExecutionProvider", "CPUExecutionProvider",
        ])
        result = xt.xtrans_markesteijn_demosaic(raw, pattern)
        assert not xt._cpu_fallback
    finally:
        # Dispose native sessions before a failing assertion holds traceback frames.
        xt.clear_session()
    np.testing.assert_allclose(result, reference, atol=3e-6, rtol=3e-6)