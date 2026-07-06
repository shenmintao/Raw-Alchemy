"""Fused GPU colour-grade path (execution-time fusion in the executors).

The test-suite baseline runs the classic per-op path (conftest sets
RAWALCHEMY_GRADE_GPU=0); these tests flip fusion on and pin its contracts:
numeric equivalence with the classic path, EV publishing, cache behavior,
and the log/LUT/HDR chains staying unfused.
"""
import numpy as np
import pytest

from raw_alchemy.pipeline.executor import PreviewExecutor, ExportExecutor, _BaseExecutor
from raw_alchemy.pipeline.ops import Op, build_op_list


def _params(**over):
    p = {
        "lens_correct": False, "exposure_mode": "Manual", "exposure": 0.7,
        "metering_mode": "matrix", "wb_temp": 8.0, "wb_tint": -4.0,
        "highlight": -20.0, "shadow": 15.0, "saturation": 1.2, "contrast": 1.1,
        "log_space": "None", "lut_path": None, "rotation": 0,
        "flip_horizontal": False, "flip_vertical": False,
        "perspective_corners": None, "crop": (0.0, 0.0, 1.0, 1.0),
        "sharpen_strength": 0.0, "denoise_enabled": False,
    }
    p.update(over)
    return p


@pytest.fixture
def fusion_on(monkeypatch):
    monkeypatch.setenv("RAWALCHEMY_GRADE_GPU", "1")


def _src(seed=0, shape=(120, 160, 3)):
    return np.clip(np.random.default_rng(seed).random(shape), 0, 1).astype(np.float32)


def _spy_ops(monkeypatch):
    calls = []
    orig = _BaseExecutor._apply_op

    def spy(self, buf, op):
        calls.append(op.name)
        return orig(self, buf, op)

    monkeypatch.setattr(_BaseExecutor, "_apply_op", spy)
    return calls


def test_fused_matches_classic_path(fusion_on, monkeypatch):
    src = _src()
    ops = build_op_list(_params())
    fused = PreviewExecutor(src.copy()).run(ops)
    monkeypatch.setenv("RAWALCHEMY_GRADE_GPU", "0")
    classic = PreviewExecutor(src.copy()).run(ops)
    np.testing.assert_allclose(fused, classic, atol=2e-5)


def test_fused_runs_as_single_op(fusion_on, monkeypatch):
    calls = _spy_ops(monkeypatch)
    ops = build_op_list(_params(rotation=90))
    PreviewExecutor(_src()).run(ops)
    assert calls == ["geometry", "grade_fused"]


def test_fused_publishes_manual_and_auto_ev(fusion_on):
    ex = PreviewExecutor(_src())
    ex.run(build_op_list(_params(exposure=1.5)))
    assert ex.last_applied_ev == pytest.approx(1.5)
    ex2 = PreviewExecutor(_src())
    ex2.run(build_op_list(_params(exposure_mode="Auto")))
    assert np.isfinite(ex2.last_applied_ev)


def test_colour_param_change_reuses_geometry_prefix(fusion_on, monkeypatch):
    calls = _spy_ops(monkeypatch)
    ex = PreviewExecutor(_src())
    ex.run(build_op_list(_params(rotation=90, contrast=1.1)))
    calls.clear()
    ex.run(build_op_list(_params(rotation=90, contrast=1.3)))
    # geometry prefix hit: only the fused tail re-runs
    assert calls == ["grade_fused"]


def test_full_hit_runs_nothing(fusion_on, monkeypatch):
    calls = _spy_ops(monkeypatch)
    ex = PreviewExecutor(_src())
    ops = build_op_list(_params())
    ex.run(ops)
    calls.clear()
    ex.run(ops)
    assert calls == []


def test_export_path_fuses_and_matches(fusion_on, monkeypatch):
    src = _src(3)
    ops = build_op_list(_params())
    fused = ExportExecutor(src.copy()).run(ops)
    monkeypatch.setenv("RAWALCHEMY_GRADE_GPU", "0")
    classic = ExportExecutor(src.copy()).run(ops)
    np.testing.assert_allclose(fused, classic, atol=2e-5)


def test_minimal_tail_exposure_plus_srgb_only(fusion_on, monkeypatch):
    calls = _spy_ops(monkeypatch)
    src = _src(4)
    ops = build_op_list(_params(wb_temp=0.0, wb_tint=0.0, highlight=0.0,
                                shadow=0.0, saturation=1.0, contrast=1.0))
    assert [op.name for op in ops] == ["exposure", "srgb_out"]
    fused = PreviewExecutor(src.copy()).run(ops)
    assert calls == ["grade_fused"]
    monkeypatch.setenv("RAWALCHEMY_GRADE_GPU", "0")
    classic = PreviewExecutor(src.copy()).run(ops)
    np.testing.assert_allclose(fused, classic, atol=2e-5)


def _write_cube(path, size=8):
    g = np.linspace(0, 1, size, dtype=np.float32)
    lines = [f"LUT_3D_SIZE {size}"]
    # .cube 顺序:R 最快;colour 读入 table[r,g,b]
    for b in g:
        for gg in g:
            for r in g:
                lines.append(f"{r**1.1:.6f} {gg:.6f} {b**0.9:.6f}")
    path.write_text("\n".join(lines))
    return str(path)


def test_lut3d_chain_fused_matches_classic(fusion_on, monkeypatch, tmp_path):
    cube = _write_cube(tmp_path / "t.cube")
    src = _src(5)
    ops = build_op_list(_params(lut_path=cube))
    assert [op.name for op in ops][-2:] == ["lut", "srgb_out"]
    calls = _spy_ops(monkeypatch)
    fused = PreviewExecutor(src.copy()).run(ops)
    assert calls == ["grade_fused"]
    monkeypatch.setenv("RAWALCHEMY_GRADE_GPU", "0")
    classic = PreviewExecutor(src.copy()).run(ops)
    np.testing.assert_allclose(fused, classic, atol=5e-5)


def test_log_chain_fused_matches_classic(fusion_on, monkeypatch):
    from raw_alchemy import config as cfg
    log_space = next(iter(cfg.LOG_ENCODING_MAP)) if cfg.LOG_ENCODING_MAP else None
    if not log_space:
        pytest.skip("no log spaces configured")
    src = _src(6)
    ops = build_op_list(_params(log_space=log_space))
    if ops[-1].name != "log_transform":
        pytest.skip(f"log chain shape unexpected: {[o.name for o in ops]}")
    calls = _spy_ops(monkeypatch)
    fused = PreviewExecutor(src.copy()).run(ops)
    if "grade_fused" not in calls:
        pytest.skip("log curve LUT unavailable; window not fused")
    monkeypatch.setenv("RAWALCHEMY_GRADE_GPU", "0")
    classic = PreviewExecutor(src.copy()).run(ops)
    np.testing.assert_allclose(fused, classic, atol=5e-5)


def test_log_plus_lut_chain_fused_matches_classic(fusion_on, monkeypatch, tmp_path):
    from raw_alchemy import config as cfg
    log_space = next(iter(cfg.LOG_ENCODING_MAP)) if cfg.LOG_ENCODING_MAP else None
    if not log_space:
        pytest.skip("no log spaces configured")
    cube = _write_cube(tmp_path / "t2.cube")
    src = _src(7)
    ops = build_op_list(_params(log_space=log_space, lut_path=cube))
    calls = _spy_ops(monkeypatch)
    fused = PreviewExecutor(src.copy()).run(ops)
    if "grade_fused" not in calls:
        pytest.skip("log curve LUT unavailable; window not fused")
    monkeypatch.setenv("RAWALCHEMY_GRADE_GPU", "0")
    classic = PreviewExecutor(src.copy()).run(ops)
    np.testing.assert_allclose(fused, classic, atol=5e-5)
