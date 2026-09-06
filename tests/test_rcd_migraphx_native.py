"""Opt-in AMD production-path acceptance: phases, seams, real WB and cache."""
import gc
import json
import os
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pytest


def test_native_amd_bayer_cases_and_compiled_cache(monkeypatch, tmp_path):
    if os.environ.get("RAWALCHEMY_TEST_RCD_MIGRAPHX") != "1":
        pytest.skip("opt-in AMD Bayer hardware acceptance")
    import onnxruntime as ort
    import rawpy
    from raw_alchemy.colorspace_matrices import cam_to_working_space_matrix
    from raw_alchemy.core import (
        fix_hot_pixels, highlight_inpaint_opposed,
        subtract_black_level,
    )
    from raw_alchemy.onnx import rcd_demosaic as rcd

    assert "MIGraphXExecutionProvider" in ort.get_available_providers()
    path = Path(os.environ["RAWALCHEMY_TEST_BAYER_RAW"])
    with rawpy.imread(str(path)) as frame:
        pattern = frame.raw_pattern.copy()
        assert pattern.shape == (2, 2)
        wb = np.array(frame.camera_whitebalance, np.float32)
        raw = subtract_black_level(
            frame.raw_image_visible.astype(np.float32),
            np.array(frame.black_level_per_channel, np.float32),
            float(frame.white_level), pattern,
        )
        fix_hot_pixels(raw, pattern)
        highlight_inpaint_opposed(raw, pattern, wb)
        wb3 = np.array([wb[0] / wb[1], 1, wb[2] / wb[1]], np.float32)
        matrix = cam_to_working_space_matrix(np.array(frame.rgb_xyz_matrix, np.float64)).astype(np.float32)
    rng = np.random.default_rng(918)
    test_wb = np.array([2.1, 1, 1.6], np.float32)
    test_matrix = np.array([[1.1, -.08, -.02], [-.03, 1.08, -.05], [-.02, -.1, 1.12]], np.float32)
    cases = []
    for phase in range(4):
        pat = np.roll(np.roll([[0, 1], [3, 2]], phase // 2, axis=0), phase % 2, axis=1)
        mosaic = rng.uniform(0, 1.2, (126, 134)).astype(np.float32)
        cases.append((f"cfa-{phase}", mosaic, pat, test_wb, test_matrix))
    cases += [
        ("black", np.zeros((10, 12), np.float32), pattern, test_wb, test_matrix),
        ("near-flat", np.full((94, 102), .2, np.float32) + rng.uniform(-1e-7, 1e-7, (94, 102)).astype(np.float32),
         pattern, test_wb, test_matrix),
    ]
    edges = np.zeros((1574, 1598), np.float32)
    edges[:, 1490:] = 1.2
    edges[1490:1510, :] = .37
    edges[24, 24] = 1
    cases += [("tile-seams-highlights", edges, pattern, test_wb, test_matrix),
              ("real-camera-wb-matrix", raw, pattern, wb3, matrix)]
    monkeypatch.setenv("RAWALCHEMY_MIGRAPHX_DEMOSAIC", "auto")
    monkeypatch.setenv("RAWALCHEMY_NATIVE_ISOLATION", "1")
    monkeypatch.setenv("ORT_MIGRAPHX_MODEL_CACHE_PATH", str(tmp_path / "compiled"))
    original = rcd._make_session_options

    def options(runtime):
        result = original(runtime)
        result.enable_profiling = True
        result.profile_file_prefix = str(tmp_path / "rcd")
        return result

    monkeypatch.setattr(rcd, "_make_session_options", options)
    report = {"ort": ort.__version__, "runs": []}
    try:
        for mode in ("cpu", "gpu-cold", "gpu-warm"):
            monkeypatch.setenv("RAW_ALCHEMY_CPU_ONLY", "1" if mode == "cpu" else "0")
            rcd.clear_session()
            start = time.perf_counter()
            session = rcd._get_session()
            row = {"mode": mode, "initialization_seconds": time.perf_counter() - start, "cases": []}
            report["runs"].append(row)
            try:
                for name, mosaic, pat, wb_case, matrix_case in (cases[-1:] if mode == "gpu-warm" else cases):
                    start = time.perf_counter()
                    actual = rcd.rcd_demosaic(mosaic, pat, wb3=wb_case, cam_mat=matrix_case)
                    case = {"name": name, "seconds": time.perf_counter() - start,
                            "shape": list(actual.shape), "finite": bool(np.isfinite(actual).all())}
                    row["cases"].append(case)
                    if mode == "cpu":
                        np.save(tmp_path / f"{name}.npy", actual)
                    else:
                        reference = np.load(tmp_path / f"{name}.npy")
                        error = np.abs(actual - reference)
                        case.update(max_error=float(error.max()), bad_channels=int(np.count_nonzero(
                            error > 3e-6 + 3e-6 * np.abs(reference))))
                    print(json.dumps({"mode": mode, **case}), flush=True)
                profile = json.loads(Path(session.end_profiling()).read_text())
                row["placements"] = dict(Counter(e["args"]["provider"] for e in profile
                                                 if e.get("args", {}).get("provider")))
                row["fallback"] = rcd._cpu_fallback
                row["registered"] = rcd._get_session().get_providers()
            finally:
                rcd.clear_session()
                session.close()
                del session
                gc.collect()
            files = {str(p.relative_to(tmp_path)): (p.stat().st_size, p.stat().st_mtime_ns)
                     for p in (tmp_path / "compiled").rglob("*") if p.is_file()}
            row["cache_files"] = files
    finally:
        rcd.clear_session()
        (tmp_path / "acceptance.json").write_text(json.dumps(report, indent=2))
    cold, warm = report["runs"][1:]
    assert cold["cache_files"] and cold["cache_files"] == warm["cache_files"], report
    for row in (cold, warm):
        assert not row["fallback"] and row["registered"][0] == "MIGraphXExecutionProvider", report
        assert row["placements"].get("MIGraphXExecutionProvider", 0) > 0, report
        assert all(c["finite"] and c["bad_channels"] == 0 for c in row["cases"]), report
