from functools import lru_cache

import colour
import numpy as np


LOG_LUT_SIZE = 16_384
LOG_LUT_DOMAIN_MIN = 0.0
LOG_LUT_DOMAIN_MAX = 16.0


@lru_cache(maxsize=32)
def get_log_lut(log_curve_name: str) -> np.ndarray:
    x = np.linspace(
        LOG_LUT_DOMAIN_MIN,
        LOG_LUT_DOMAIN_MAX,
        LOG_LUT_SIZE,
        dtype=np.float64,
    )
    try:
        with np.errstate(all="ignore"):
            y = colour.cctf_encoding(x, function=log_curve_name)
    except Exception:
        return np.empty((0,), dtype=np.float32)

    y = np.asarray(y, dtype=np.float32)
    if not np.isfinite(y).all():
        return np.empty((0,), dtype=np.float32)
    return np.ascontiguousarray(y)
