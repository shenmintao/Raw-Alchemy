"""Precision and compilation controls for the isolated MIGraphX demosaic child."""
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

from loguru import logger

MODEL_FILE = "xtrans_markesteijn_migraphx.onnx"
RCD_MODEL_FILE = "rcd_demosaic_migraphx_1536.onnx"
RCD_TILE = 1536
EP = "MIGraphXExecutionProvider"


def _rocm_version():
    try:
        return (Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / ".info/version").read_text().strip()
    except OSError:
        return ""


def validated_runtime():
    import onnxruntime as ort

    return (platform.system() == "Linux" and platform.machine().lower() in {"x86_64", "amd64"}
            and ort.__version__ == "1.23.2" and _rocm_version() == "7.2.0"
            and os.environ.get("RAWALCHEMY_NATIVE_ISOLATION", "1") != "0")


def applies(model, providers):
    first = providers[0] if providers else None
    first = first[0] if isinstance(first, tuple) else first
    return (platform.system() == "Linux" and first == EP
            and Path(model).name in {MODEL_FILE, RCD_MODEL_FILE})


def prepare_child(model, options, providers):
    """Call only in the owned child, before constructing the native provider.

    FastDenoise and other providers keep their own settings. Never mutate the
    GUI process environment or import POSIX resource controls on Windows.
    """
    if not applies(model, providers):
        return
    import resource
    import onnxruntime as ort

    disabled = {value.strip() for value in os.environ.get("MIGRAPHX_DISABLE_PASSES", "").split(",")
                if value.strip()}
    os.environ["MIGRAPHX_DISABLE_PASSES"] = ",".join(sorted(disabled | {"simplify_algebra"}))
    previous = os.environ.get("MIGRAPHX_GPU_HIP_FLAGS", "")
    os.environ["MIGRAPHX_GPU_HIP_FLAGS"] = (previous + " -fno-fast-math -ffp-contract=off").strip()
    # Provider environment variables take precedence over provider options.
    for precision in ("FP16", "BF16", "FP8", "INT8"):
        os.environ[f"ORT_MIGRAPHX_{precision}_ENABLE"] = "0"

    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    wanted = 64 * 1024 * 1024
    if soft != resource.RLIM_INFINITY and soft < wanted:
        resource.setrlimit(resource.RLIMIT_STACK, (
            min(wanted, hard) if hard != resource.RLIM_INFINITY else wanted, hard,
        ))

    # The EP adds its MIGraphX version, GPU architecture and graph/shape hash to
    # MXR filenames. Our namespace also binds source bytes, runtime and every
    # compiler override, which its default cache identity does not include.
    inherited_cache = os.environ.pop("ORT_MIGRAPHX_MODEL_CACHE_PATH", None)
    try:
        compiler_env = {key: value for key, value in os.environ.items()
                        if key.startswith(("MIGRAPHX_", "ORT_MIGRAPHX_"))}
        identity = {
            "schema": 1, "model": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
            "ort": ort.__version__, "platform": platform.platform(),
            "dimensions": options.get("dimensions", {}), "providers": providers,
            "compiler": compiler_env,
        }
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        base = Path(inherited_cache) if inherited_cache else Path(
            os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        ) / "RawAlchemy" / "migraphx"
        cache = base / namespace
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=cache) as probe:
            probe.write(b"cache write probe")
            probe.flush()
        os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = str(cache)
    except OSError as exc:
        logger.warning("MIGraphX cache unavailable; compiling without disk reuse: {}", exc)