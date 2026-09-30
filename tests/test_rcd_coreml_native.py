"""Opt-in CoreML RCD acceptance covering CFA phases, seams and a real NEF."""
import gc
import json
import os
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("RAWALCHEMY_TEST_RCD_COREML") != "1",
                    reason="requires CoreML hardware and a real Bayer RAW")
def test_native_coreml_bayer_phases_seams_and_real_camera(monkeypatch, tmp_path):
    import onnxruntime as ort
    import rawpy
    from raw_alchemy.core import subtract_black_level
    from raw_alchemy.colorspace_matrices import cam_to_working_space_matrix
    from raw_alchemy.onnx import rcd_demosaic as rcd

    assert "CoreMLExecutionProvider" in ort.get_available_providers()
    with rawpy.imread(os.environ["RAWALCHEMY_TEST_BAYER_RAW"]) as frame:
        pattern = frame.raw_pattern.copy()
        raw = subtract_black_level(frame.raw_image_visible.astype(np.float32),
            np.array(frame.black_level_per_channel, np.float32), float(frame.white_level), pattern)
        wb = frame.camera_whitebalance
        wb3 = np.array([wb[0] / wb[1], 1, wb[2] / wb[1]], np.float32)
        matrix = cam_to_working_space_matrix(np.array(frame.rgb_xyz_matrix, np.float64)).astype(np.float32)
    rng = np.random.default_rng(918)
    test_wb = np.array([2.1, 1, 1.6], np.float32)
    test_matrix = np.array([[1.1, -.08, -.02], [-.03, 1.08, -.05], [-.02, -.1, 1.12]], np.float32)
    cases = []
    for phase in range(4):
        pat = np.roll(np.roll([[0, 1], [3, 2]], phase // 2, axis=0), phase % 2, axis=1)
        cases.append((f"cfa-{phase}", rng.uniform(0, 1.2, (126, 134)).astype(np.float32), pat, test_wb, test_matrix))
    seams = np.zeros((1574, 1598), np.float32)
    seams[:, 710:] = 1.2
    seams[710:738] = .37
    cases.extend([
        ("black", np.zeros((10, 12), np.float32), pattern, test_wb, test_matrix),
        ("near-flat", np.full((94, 102), .2, np.float32) + rng.uniform(-1e-7, 1e-7, (94, 102)).astype(np.float32),
         pattern, test_wb, test_matrix),
        ("tile-seams", seams, pattern, test_wb, test_matrix),
        ("real-camera", raw, pattern, wb3, matrix),
    ])
    monkeypatch.setenv("RAWALCHEMY_COREML_DEMOSAIC", "auto")
    original_options = rcd._make_session_options

    def options(runtime):
        so = original_options(runtime)
        so.enable_profiling = True
        so.profile_file_prefix = str(tmp_path / "rcd")
        return so

    monkeypatch.setattr(rcd, "_make_session_options", options)
    report = []
    for cpu in (True, False):
        monkeypatch.setenv("RAW_ALCHEMY_CPU_ONLY", "1" if cpu else "0")
        rcd.clear_session()
        start = time.perf_counter()
        session = rcd._get_session()
        row = dict(cpu=cpu, initialization_s=time.perf_counter() - start, cases=[])
        try:
            for name, mosaic, pat, wb_case, matrix_case in cases:
                start = time.perf_counter()
                actual = rcd.rcd_demosaic(mosaic, pat, wb_case, matrix_case)
                case = dict(name=name, seconds=time.perf_counter() - start, finite=bool(np.isfinite(actual).all()))
                if cpu:
                    np.save(tmp_path / f"{name}.npy", actual)
                else:
                    reference = np.load(tmp_path / f"{name}.npy")
                    delta = np.abs(actual - reference)
                    case.update(max_delta=float(delta.max()), bad_channels=int(np.count_nonzero(delta > 3e-6 + 3e-6 * np.abs(reference))))
                row["cases"].append(case)
            events = json.loads(Path(session.end_profiling()).read_text())
            row.update(provider=rcd._get_session().get_providers(), fallback=rcd._cpu_fallback,
                       placement=dict(Counter(e.get("args", {}).get("provider") for e in events if e.get("cat") == "Node")))
            report.append(row)
            print(json.dumps(row), flush=True)
        finally:
            rcd.clear_session()
            session.close()
            del session
            gc.collect()
        (tmp_path / "acceptance.json").write_text(json.dumps(report, indent=2))
    candidate = report[-1]
    assert candidate["provider"][0] == "CoreMLExecutionProvider" and not candidate["fallback"]
    assert candidate["placement"].get("CoreMLExecutionProvider", 0) > 0
    assert all(c["finite"] and c["bad_channels"] == 0 for c in candidate["cases"]), report
