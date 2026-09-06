"""Opt-in real-provider check. Portable mocks are NOT native GPU validation.

Run RAWALCHEMY_TEST_BACKEND_NATIVE=1 python -m pytest -s -q
    tests/test_backend_native.py
Requires the bundled FastDenoise model. Prints hardware/runtime, actual profiled
placement, timings and numerical error. No hard speed threshold (CI contention).
"""
import json
import os
import platform
import time
from collections import Counter

import numpy as np
import onnxruntime as ort
import pytest

from raw_alchemy.onnx import denoiser, rgb_denoiser, session_policy

pytestmark = pytest.mark.skipif(
    os.environ.get("RAWALCHEMY_TEST_BACKEND_NATIVE") != "1",
    reason="opt-in native backend/model validation",
)


def test_native_fastdenoise_matches_cpu_and_profiles_selected_ep(tmp_path):
    path = rgb_denoiser._find_model(rgb_denoiser.MODEL_FILE)
    providers = denoiser._configure_providers(
        denoiser._get_providers(), path, variant="rgb-denoiser"
    )
    selected_names = session_policy.provider_names(providers)
    assert selected_names[0] != "CPUExecutionProvider", "No accelerator selected"
    required = os.environ.get("RAWALCHEMY_TEST_REQUIRED_EP")
    if required:
        assert selected_names[0] == required, selected_names

    def options():
        result = denoiser._make_session_options(ort)
        result.enable_profiling = True
        result.profile_file_prefix = str(tmp_path / "fastdenoise")
        return result

    started = time.perf_counter()
    candidate = session_policy.create_session(
        ort, path, options, providers, variant="rgb-denoiser"
    )
    initialization = time.perf_counter() - started
    reference = ort.InferenceSession(path, denoiser._make_session_options(ort),
                                    providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(2026)
    # The filename describes weight precision, not the model's public inputs.
    # FastDenoise v4 accepts float32 RGB and a spatial float32 sigma map,
    # exactly as denoise_rgb_linear supplies in production.
    feeds = {"rgb": rng.uniform(0.02, 0.5, (1, 3, 512, 512)).astype(np.float32),
             "sigma": np.full((1, 1, 512, 512), 0.25, dtype=np.float32)}
    t0 = time.perf_counter()
    expected = reference.run(None, feeds)[0]
    cpu_seconds = time.perf_counter() - t0
    outputs, seconds = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        outputs.append(candidate.run(None, feeds)[0])
        seconds.append(time.perf_counter() - t0)
    profile_path = candidate.end_profiling()
    with open(profile_path) as stream:
        events = json.load(stream)
    placements = Counter(e.get("args", {}).get("provider") for e in events
                         if e.get("args", {}).get("provider"))
    error = np.abs(outputs[-1].astype(np.float32) - expected.astype(np.float32))
    report = {
        "platform": platform.platform(), "machine": platform.machine(),
        "ort": ort.__version__, "available": ort.get_available_providers(),
        "selected": selected_names, "registered": candidate.get_providers(),
        "profiled_node_events": dict(placements), "initialization_seconds": initialization,
        "cpu_seconds": cpu_seconds, "candidate_seconds": seconds,
        "max_absolute_error": float(error.max()), "mean_absolute_error": float(error.mean()),
    }
    print(json.dumps(report, indent=2))
    assert all(np.isfinite(out).all() for out in outputs)
    assert error.max() < 0.02
    assert error.mean() < 0.001
    # A silent CPU fallback is correctness-safe but must not pass an accelerator
    # acceptance test. Profiling, rather than registration alone, proves use.
    assert placements[selected_names[0]] > 0, report
    assert selected_names[0] in candidate.get_providers(), report
