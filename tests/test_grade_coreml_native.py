"""Opt-in fixed-tile GPU grade precision and actual node placement."""
import gc
import json
import os
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("RAWALCHEMY_TEST_GRADE_COREML") != "1",
                    reason="requires native Apple CoreML")
@pytest.mark.parametrize("tail", ["plain", "lut", "log"])
def test_native_coreml_grade_matches_cpu_across_tiles(monkeypatch, tmp_path, tail):
    import onnxruntime as ort
    from raw_alchemy.onnx import grade
    from raw_alchemy.pipeline.log_encoding import get_log_lut, LOG_LUT_DOMAIN_MIN, LOG_LUT_DOMAIN_MAX

    assert "CoreMLExecutionProvider" in ort.get_available_providers()
    img = np.random.default_rng(30).uniform(-.03, .7, (1536, 2304, 3)).astype(np.float32)
    img[:8, :8] = 0
    core = dict(gain=2.8, mat_a=np.array([[1.1, -.08, -.02], [-.03, 1.08, -.05], [-.02, -.1, 1.12]], np.float32),
                highlight=-.2, shadow=.15, saturation=1.25, contrast=1.1,
                pivot=.18, luma=np.array([.288, .7119, .0001], np.float32))
    grid = np.linspace(0, 1, 17, dtype=np.float32)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    table = np.stack((.95 * r**1.1 + .05 * g, .97 * g**.95 + .03 * b, .96 * b**1.03 + .04 * r), -1).reshape(-1, 3)
    lut = dict(lut3d_flat=table, lut3d_size=17, d3_min=np.zeros(3, np.float32), d3_max=np.ones(3, np.float32))
    if tail == "plain":
        model = grade.MODEL_FILE
        run = lambda: grade.apply_grade(img, **core, mat_b=np.eye(3, dtype=np.float32), srgb_encode=True)
    elif tail == "lut":
        model = grade.MODEL_FILE_LUT
        run = lambda: grade.apply_grade_lut(img, **core, **lut, mat_b=np.eye(3, dtype=np.float32))
    else:
        model = grade.MODEL_FILE_LOG
        run = lambda: grade.apply_grade_log(img, **core, **lut, mat_log=np.eye(3, dtype=np.float32),
            lut1d=get_log_lut("F-Log2"), d1_min=LOG_LUT_DOMAIN_MIN, d1_max=LOG_LUT_DOMAIN_MAX)
    original_options = grade._make_session_options

    def options(runtime):
        so = original_options(runtime)
        so.enable_profiling = True
        so.profile_file_prefix = str(tmp_path / tail)
        return so

    monkeypatch.setattr(grade, "_make_session_options", options)
    report = []
    expected = None
    for cpu in (True, False):
        monkeypatch.setenv("RAW_ALCHEMY_CPU_ONLY", "1" if cpu else "0")
        monkeypatch.setenv("RAWALCHEMY_COREML_GRADE", "auto")
        grade.clear_session()
        session = grade._get_session(model)
        try:
            times = []
            for _ in range(2):
                start = time.perf_counter()
                actual = run()
                times.append((time.perf_counter() - start) * 1000)
            if expected is None:
                expected = actual
            delta = np.abs(actual - expected)
            events = json.loads(Path(session.end_profiling()).read_text())
            report.append(dict(cpu=cpu, times_ms=times, providers=session.get_providers(),
                max_delta=float(delta.max()), bad_channels=int(np.count_nonzero(delta > 3e-6 + 3e-6 * np.abs(expected))),
                placement=dict(Counter(e.get("args", {}).get("provider") for e in events if e.get("cat") == "Node"))))
        finally:
            grade.clear_session()
            close = getattr(session, "close", None)
            if close:
                close()
            del session
            gc.collect()
    print(json.dumps(report), flush=True)
    (tmp_path / "acceptance.json").write_text(json.dumps(report, indent=2))
    candidate = report[-1]
    assert candidate["providers"][0] == "CoreMLExecutionProvider"
    assert candidate["placement"].get("CoreMLExecutionProvider", 0) > 0
    assert np.isfinite(actual).all() and candidate["bad_channels"] == 0, report
