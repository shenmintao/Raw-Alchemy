"""Versioned identities for lossless, full-resolution decoded-stage artifacts.

Only parameters *upstream* of this artifact belong in the identity. Exposure,
LUT, lens correction, crop and output format are downstream and intentionally
excluded. CPU neighbour preloads use a different decode algorithm and must not
seed the canonical full-export cache.
"""
import hashlib
import json
import os
import platform

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

STAGE_VERSION = 5
DECODE_CANONICAL = "rawspeed-or-rawpy/cfa-onnx/v3"
DECODE_PRELOAD = "libraw-neighbour-preload/v1"
DECODE_FALLBACK = "libraw-preview-fallback/v1"


def file_digest(path):
    """Hash current contents; equal stat times are not a content generation.

    Fast same-size Windows rewrites can preserve both reported timestamps.
    RAW/model identities are checked on worker lanes, never on the GUI thread.
    Cancellation is checked between blocks without yielding a held cache lock.
    """
    from .cancellation import check_cancelled

    h = hashlib.sha256()
    check_cancelled()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check_cancelled()
            h.update(block)
    check_cancelled()
    return h.hexdigest()


def source_identity(path):
    """Fail closed for persistent reuse; missing synthetic inputs have no ID."""
    try:
        return file_digest(path)
    except OSError:
        return None


def _identity_files():
    root = Path(__file__).resolve().parents[1]
    files = [
        root / name for name in (
            "core.py", "onnx/rgb_denoiser.py", "onnx/denoiser.py",
            "onnx/rcd_demosaic.py", "onnx/xtrans_demosaic.py",
            "onnx/demosaic_coreml.py", "onnx/session_policy.py", "onnx/migraphx_precision.py",
            "onnx/gpu_runtime.py", "onnx/coreml_cache.py", "math_ops.py",
            "rawspeed.py", "onnx/isolated_session.py", "native_decode.py",
            "colorspace_matrices.py", "demosaic_helpers.py",
            "pipeline/stage_identity.py", "pipeline/source_artifacts.py",
        )
    ]
    # Follow the same locator/constants as inference (including frozen builds).
    from raw_alchemy.onnx.denoiser import _find_model
    from raw_alchemy.onnx.rgb_denoiser import MODEL_FILE as RGB_MODEL
    from raw_alchemy.onnx.rcd_demosaic import (
        MODEL_FILE as RCD_MODEL, MIGRAPHX_MODEL_FILE as RCD_MIGRAPHX_MODEL,
    )
    from raw_alchemy.onnx.xtrans_demosaic import MODEL_FILE as XTRANS_MODEL
    from raw_alchemy.onnx.xtrans_demosaic import COREML_MODEL_FILE, MIGRAPHX_MODEL_FILE
    files += [Path(_find_model(name)) for name in (
        RGB_MODEL, RCD_MODEL, RCD_MIGRAPHX_MODEL, XTRANS_MODEL, COREML_MODEL_FILE, MIGRAPHX_MODEL_FILE,
    )]
    for model in list(files):
        if model.suffix == ".onnx":
            files += sorted(model.parent.glob(model.name + ".data*"))
    return files


def _runtime_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def denoise_tag(strength=0.25, *, decode_variant=DECODE_CANONICAL):
    """Complete upstream policy identity, or None if assets cannot be identified.

    Runtime/provider options are included because mixed precision/backends may
    change numeric results. Automatic CPU recovery remains part of that policy;
    this cache promises float32 storage, not bit-identical CPU/GPU inference.
    """
    from raw_alchemy.config import WORKING_SPACE
    from raw_alchemy.onnx.session_policy import configuration_token
    try:
        assets = [(str(p.name), file_digest(p)) for p in _identity_files()]
    except OSError:
        return None
    descriptor = {
        "stage": "native-denoise-before-lens-grade-geometry",
        "version": STAGE_VERSION,
        "decode": decode_variant,
        "colorspace": WORKING_SPACE,
        "transfer": "linear",
        "precision": "float32",
        "resolution": "native-oriented",
        "strength": float(strength),
        "algorithm_options": {name: os.environ.get(name) for name in (
            "CANS_HIGHLIGHT_GUARD", "CANS_DARK_LF_GUARD",
            "RAWALCHEMY_COREML_DENOISE", "RAWALCHEMY_COREML_GRADE",
        )},
        "assets": assets,
        "runtime": {p: _runtime_version(p) for p in
                    ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-migraphx",
                     "rawpy", "rawspeedpy", "numpy")},
        "platform": (platform.system(), platform.machine()),
        "session_policy": [configuration_token(v) for v in
                           ("rgb-denoiser", "rcd", "xtrans")],
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
