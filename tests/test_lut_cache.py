import os

import numpy as np

from raw_alchemy import utils
from raw_alchemy.pipeline.ops import build_op_list


def _cube(value: float) -> str:
    lines = ["LUT_3D_SIZE 2"]
    lines.extend([f"{value:.1f} {value:.1f} {value:.1f}"] * 8)
    return "\n".join(lines)


def test_lut_cache_reloads_an_edited_file(tmp_path):
    path = tmp_path / "live.cube"
    path.write_text(_cube(0.0), encoding="utf-8")
    params = {
        "exposure_mode": "Manual",
        "exposure": 0.0,
        "metering_mode": "matrix",
        "log_space": "None",
        "lut_path": str(path),
    }
    first_op = next(op for op in build_op_list(params) if op.name == "lut")
    first = utils.load_lut_cached(path)
    first_mtime = path.stat().st_mtime_ns

    path.write_text(_cube(1.0), encoding="utf-8")
    os.utime(path, ns=(first_mtime + 1_000_000, first_mtime + 1_000_000))
    second_op = next(op for op in build_op_list(params) if op.name == "lut")
    second = utils.load_lut_cached(path)

    assert second is not first
    assert second_op != first_op
    assert first._raw_alchemy_table32.dtype == np.float32
    assert second._raw_alchemy_table32.dtype == np.float32
    np.testing.assert_array_equal(first.table, np.zeros_like(first.table))
    np.testing.assert_array_equal(second.table, np.ones_like(second.table))
