"""RCD Bayer demosaic on ONNX Runtime (GPU via DirectML/CUDA, CPU fallback).

Replaces the retired Taichi port. The graph is a vectorized translation of
the same algorithm (validated pixel-exact against the Taichi reference,
max |delta| ~2e-7 on real mosaics) exported with dynamic H/W.

Inference is TILED at a fixed 1536px tile (overlap 24 >> the algorithm's
~8px neighbourhood + 4px border logic):
- DirectML compiles the graph ONCE for the single tile shape — no per-sensor
  4-6s dimension-freeze, no pathological dynamic-shape partitions;
- VRAM is bounded by the tile working set (~0.4GB) instead of a full-frame
  arena (a frozen 42MP graph pinned multiple GB per sensor size).

Runs fp32 — the algorithm's EPSSQ=1e-10 discriminators underflow in fp16.

Input:  (H, W) float32 Bayer mosaic, black-subtracted, normalized [0, 1]
Output: (H, W, 3) float32 RGB
"""

from raw_alchemy.pipeline.resources import checkpoint
import threading
import time

import numpy as np
from loguru import logger

from .demosaic_coreml import demosaic_providers
from .session_policy import configuration_token, construct_session
from raw_alchemy.pipeline.cancellation import check_cancelled
from raw_alchemy.pipeline.executor import PipelineAborted
from .denoiser import (
    _configure_providers,
    _find_model,
    _get_providers,
    _make_session_options,
)

from .migraphx_precision import RCD_MODEL_FILE as MIGRAPHX_MODEL_FILE, RCD_TILE

MODEL_FILE = "rcd_demosaic_dyn2.onnx"
TILE = 1536   # even (CFA phase), single compiled shape
OVERLAP = 24  # even; > border(4) + neighbourhood influence (~8px)

_sessions: dict = {}
_session_lock = threading.Lock()
_session_provider = None
_session_token = None


def model_file_for_providers(providers):
    first = providers[0] if providers else None
    first = first[0] if isinstance(first, tuple) else first
    return MIGRAPHX_MODEL_FILE if first == "MIGraphXExecutionProvider" and TILE == RCD_TILE else MODEL_FILE


def _get_session():
    global _session_provider, _cpu_fallback, _session_token
    check_cancelled()
    token = configuration_token("rcd")
    if _session_token is not None and _session_token != token:
        with _session_lock:
            _sessions.clear()
            _cpu_fallback = False
    _session_token = token
    sess = _sessions.get(TILE)
    if sess is not None:
        return sess
    with _session_lock:
        sess = _sessions.get(TILE)
        if sess is not None:
            return sess
        import onnxruntime as ort

        # Freeze the dynamic h/w symbols through ORT itself. This avoids
        # importing/rewriting the model with the optional ``onnx`` package and
        # still gives CUDA/DirectML one fixed compiled tile graph.
        def session_options():
            so = _make_session_options(ort)
            so.add_free_dimension_override_by_name("h", TILE)
            so.add_free_dimension_override_by_name("w", TILE)
            return so

        if _cpu_fallback:
            providers = ["CPUExecutionProvider"]
        else:
            providers = demosaic_providers(_get_providers())
        # The AMD asset has fixed Gather indices; never run it at another tile size.
        first = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if first == "MIGraphXExecutionProvider" and TILE != RCD_TILE:
            providers = ["CPUExecutionProvider"]
        model_path = _find_model(model_file_for_providers(providers))
        configured = _configure_providers(
            providers, model_path, variant=f"rcd:h={TILE},w={TILE}"
        )
        so = session_options()
        try:
            sess = construct_session(ort, model_path, so, configured, variant=f"rcd:h={TILE},w={TILE}")
        except (PipelineAborted, MemoryError):
            raise
        except Exception as exc:
            if not any(
                (p[0] if isinstance(p, tuple) else p) != "CPUExecutionProvider"
                for p in providers
            ):
                raise
            logger.warning(
                f"RCD session initialization failed on {providers} "
                f"({type(exc).__name__}: {str(exc)[:200]}); retrying on CPU EP"
            )
            # CoreML can fail while compiling, before session.run(). CPU in
            # the provider list does not guarantee constructor-time recovery.
            # Do not recurse/clear_session(): we already hold the session lock.
            sess = construct_session(
                ort, _find_model(MODEL_FILE), session_options(), ["CPUExecutionProvider"],
                variant=f"rcd:h={TILE},w={TILE}",
            )
            _cpu_fallback = True
            _sessions.clear()  # Drop accelerated sessions for other tile sizes.
        _sessions[TILE] = sess
        _session_provider = sess.get_providers()[0]
        logger.info(f"RCD demosaic session (tile {TILE}): preferred EP {_session_provider} (not placement)")
    return sess


def clear_session() -> None:
    global _session_provider
    with _session_lock:
        _sessions.clear()
        _session_provider = None
    import gc
    gc.collect()


def _phase_masks(cfa_pattern: np.ndarray) -> np.ndarray:
    """(3, 2, 2) R/G/B position masks from a 2x2 CFA pattern (G2 -> G)."""
    m2 = np.zeros((3, 2, 2), np.float32)
    for r in range(2):
        for c in range(2):
            color = int(cfa_pattern[r, c])
            if color == 3:
                color = 1
            m2[color, r, c] = 1.0
    return m2


_cpu_fallback = False


def _run_tile(session, patch: np.ndarray, m2: np.ndarray,
              wb3: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    global _cpu_fallback
    # If an earlier tile fell back from GPU to CPU, do not keep invoking the
    # stale failed session for every later tile.
    checkpoint()
    session = _get_session()
    feeds = {
        "bayer": np.ascontiguousarray(patch, dtype=np.float32),
        "mr2": m2[0], "mg2": m2[1], "mb2": m2[2],
        "wb3": wb3, "cam_mat": cam_mat,
    }
    try:
        return session.run(None, feeds)[0]
    except (PipelineAborted, MemoryError):
        raise
    except Exception as e:
        if not any(p != "CPUExecutionProvider" for p in session.get_providers()):
            raise
        # DML 运行时故障(常见:显存耗尽;Windows 本地化错误文本还会被
        # pybind 以 utf-8 解码炸成 UnicodeDecodeError,掩盖真实原因)。
        # 重建 CPU-EP 会话兜底:慢,但绝不让解码整体失败。
        logger.warning(
            f"RCD tile failed on {_session_provider} "
            f"({type(e).__name__}: {str(e)[:80]}); rebuilding on CPU EP")
        _cpu_fallback = True
        clear_session()
        return _get_session().run(None, feeds)[0]


def rcd_demosaic(bayer: np.ndarray, cfa_pattern: np.ndarray,
                 wb3=None, cam_mat=None) -> np.ndarray:
    """Demosaic a Bayer mosaic with RCD on the ONNX runtime (tiled).

    Args:
        bayer: (H, W) float32, black-level subtracted, normalized to [0, 1]
        cfa_pattern: (2, 2) ints, 0=R 1=G 2=B 3=G2

    Returns:
        (H, W, 3) float32 RGB in [0, 1]
    """
    if bayer.ndim != 2:
        raise ValueError(f"expected (H, W) mosaic, got {bayer.shape}")
    H, W = bayer.shape
    if H < 10 or W < 10 or H % 2 or W % 2:
        raise ValueError(f"unsupported mosaic size {W}x{H}")

    t0 = time.time()
    checkpoint()
    session = _get_session()
    m2 = _phase_masks(np.asarray(cfa_pattern))
    # 输出端在图内折叠 WB 增益 + 相机矩阵 + clip(省 42MP 的 CPU einsum);
    # 缺省恒等 = 纯去马赛克(数值同旧图 + clip[0,1])
    wb3 = (np.ones(3, np.float32) if wb3 is None
           else np.ascontiguousarray(wb3, np.float32))
    cam_mat = (np.eye(3, dtype=np.float32) if cam_mat is None
               else np.ascontiguousarray(cam_mat, np.float32))

    if H <= TILE and W <= TILE:
        ph, pw = TILE - H, TILE - W
        patch = (np.pad(bayer, ((0, ph), (0, pw)), mode="reflect")
                 if (ph or pw) else bayer)
        rgb = _run_tile(session, patch, m2, wb3, cam_mat)[:H, :W]
    else:
        rgb = np.zeros((H, W, 3), np.float32)
        step = TILE - 2 * OVERLAP
        for y in range(0, H, step):
            for x in range(0, W, step):
                y0 = max(0, min(y - OVERLAP, H - TILE))
                x0 = max(0, min(x - OVERLAP, W - TILE))
                y0 -= y0 % 2  # keep CFA phase
                x0 -= x0 % 2
                y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
                patch = bayer[y0:y1, x0:x1]
                th, tw = patch.shape
                if th < TILE or tw < TILE:
                    patch = np.pad(patch, ((0, TILE - th), (0, TILE - tw)),
                                   mode="reflect")
                out = _run_tile(session, patch, m2, wb3, cam_mat)
                iy0, ix0 = y, x
                iy1, ix1 = min(H, y + step), min(W, x + step)
                rgb[iy0:iy1, ix0:ix1] = out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]

    logger.info(
        f"RCD demosaic: {W}x{H} in {(time.time() - t0) * 1000:.0f}ms "
        f"({_session_provider})"
    )
    return rgb
