"""Opt-in, full-frame RAW acceptance for the actual production demosaic path.

Set RAWALCHEMY_TEST_DEMOSAIC_NATIVE=1, RAWALCHEMY_TEST_REQUIRED_EP and both
RAWALCHEMY_TEST_XTRANS_RAW / RAWALCHEMY_TEST_BAYER_RAW. A registered provider,
a CPU fallback, or a small average hiding sparse errors must not pass.
"""
import gc
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RAWALCHEMY_TEST_DEMOSAIC_NATIVE") != "1",
    reason="opt-in native full-frame demosaic acceptance",
)


@pytest.mark.parametrize("sensor", ["bayer", "xtrans"])
def test_real_frame_matches_original_cpu_and_executes_gpu(sensor, monkeypatch, tmp_path):
    import onnxruntime as ort
    import rawpy

    from raw_alchemy.core import (
        fix_hot_pixels, highlight_inpaint_opposed, subtract_black_level,
    )
    from raw_alchemy.onnx import rcd_demosaic as rcd, xtrans_demosaic as xt

    required = os.environ.get("RAWALCHEMY_TEST_REQUIRED_EP")
    assert required and required != "CPUExecutionProvider", "Require an explicit accelerator"
    assert required in ort.get_available_providers()
    raw_path = os.environ.get(f"RAWALCHEMY_TEST_{sensor.upper()}_RAW")
    assert raw_path and Path(raw_path).is_file(), f"Missing {sensor} RAW fixture"
    module = xt if sensor == "xtrans" else rcd
    with rawpy.imread(raw_path) as frame:
        pattern = frame.raw_pattern.copy()
        assert pattern.shape == ((6, 6) if sensor == "xtrans" else (2, 2))
        raw = subtract_black_level(
            frame.raw_image_visible.astype(np.float32),
            np.array(frame.black_level_per_channel, np.float32),
            float(frame.white_level), pattern,
        )
        fix_hot_pixels(raw, pattern)
        highlight_inpaint_opposed(
            raw, pattern, np.array(frame.camera_whitebalance, np.float32),
        )

    original_options = module._make_session_options

    def options(runtime):
        result = original_options(runtime)
        result.enable_profiling = True
        result.profile_file_prefix = str(tmp_path / sensor)
        return result

    monkeypatch.setattr(module, "_make_session_options", options)
    run = xt.xtrans_markesteijn_demosaic if sensor == "xtrans" else rcd.rcd_demosaic
    rows = []
    reference = None
    try:
        for cpu in (True, False):
            # Select through the same provider and stage policies as the product.
            monkeypatch.setenv("RAW_ALCHEMY_CPU_ONLY", "1" if cpu else "0")
            module.clear_session()
            started = time.perf_counter()
            initial_session = module._get_session()
            initialization = time.perf_counter() - started
            current_session = initial_session
            try:
                started = time.perf_counter()
                actual = run(raw, pattern)
                elapsed = time.perf_counter() - started
                # Inference may replace the original child after a timeout.
                current_session = module._get_session()
                profile = current_session.end_profiling()
                events = json.loads(Path(profile).read_text())
                placements = Counter(
                    event.get("args", {}).get("provider") for event in events
                    if event.get("args", {}).get("provider")
                )
                row = {
                    "cpu": cpu, "initialization_seconds": initialization,
                    "inference_seconds": elapsed,
                    "registered": current_session.get_providers(),
                    "profiled_node_events": dict(placements),
                    "fallback": module._cpu_fallback,
                    "finite": bool(np.isfinite(actual).all()),
                }
                if cpu:
                    reference = actual
                else:
                    error = np.abs(actual - reference)
                    row.update(
                        max_absolute_error=float(error.max()),
                        mean_absolute_error=float(error.mean()),
                        bad_channels=int(np.count_nonzero(
                            error > 3e-6 + 3e-6 * np.abs(reference)
                        )),
                    )
                rows.append(row)
            finally:
                module.clear_session()
                for session in {id(initial_session): initial_session,
                                id(current_session): current_session}.values():
                    close = getattr(session, "close", None)
                    if close:
                        close()
                del initial_session, current_session
                gc.collect()
    finally:
        module.clear_session()

    report = {
        "sensor": sensor, "required": required,
        "platform": platform.platform(), "ort": ort.__version__,
        "shape": list(reference.shape), "runs": rows,
    }
    (tmp_path / f"{sensor}-acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    candidate = rows[-1]
    assert candidate["finite"] and not candidate["fallback"], report
    assert candidate["registered"][0] == required, report
    assert candidate["profiled_node_events"].get(required, 0) > 0, report
    assert candidate["bad_channels"] == 0, report