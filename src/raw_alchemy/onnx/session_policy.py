"""Stage-specific ORT policy and recoverable accelerator sessions.

Provider registration is not node placement. Only ORT profiling can prove that
an accelerated EP actually executed a graph. Native session construction is
synchronous: a Python thread timeout cannot stop CoreML compilation.
"""

import os
import platform
import re
import threading
import time

from loguru import logger

from raw_alchemy.pipeline.cancellation import check_cancelled

CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"
POLICY_VERSION = 7


def provider_names(providers):
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def provider_identity(providers):
    """Stable, hashable identity including EP order and every provider option."""
    return tuple(
        (p[0], tuple(sorted((str(k), str(v)) for k, v in p[1].items())))
        if isinstance(p, tuple) else (p, ())
        for p in providers
    )


def _version(value):
    match = re.match(r"^(\d+)\.(\d+)", value or "")
    return tuple(map(int, match.groups())) if match else (0, 0)


def eligible_apple():
    """Conservative capability floor for the modern CoreML provider API.

    MLProgram exists on macOS 12+, but the string provider options used here
    require ORT 1.20+. Intel/Rosetta are not covered by the measured policy.
    """
    import onnxruntime as ort

    return (
        platform.system() == "Darwin"
        and platform.machine().lower() in ("arm64", "aarch64")
        and _version(platform.mac_ver()[0]) >= (12, 0)
        and _version(getattr(ort, "__version__", "")) >= (1, 20)
    )


def affected_apple_grade():
    """Limit automatic CPU grade to the macOS/ORT generation measured locally.

    Other generations keep their existing selection until measured, with an
    explicit override available for comparison. This is not a hardware speed
    guarantee for every Apple Silicon chip on this software generation.
    """
    import onnxruntime as ort

    return (eligible_apple() and _version(platform.mac_ver()[0])[0] == 27
            and _version(ort.__version__) == (1, 29))


def stage_providers(providers, variant):
    """Apply Apple-only stage policy; never replace CUDA/ROCm/DirectML."""
    names = provider_names(providers)
    if not names or names[0] != COREML or platform.system() != "Darwin":
        return list(providers)
    stage = variant.split(":", 1)[0]
    # Demosaic owns its repaired lowering configuration. Do not mask graph
    # compilation defects by permanently disabling its accelerator here.
    if stage == "grade":
        mode = os.environ.get("RAWALCHEMY_COREML_GRADE", "auto").lower().strip()
        if mode not in ("auto", "cpu", "coreml"):
            logger.warning(f"Invalid RAWALCHEMY_COREML_GRADE={mode!r}; using CPU")
            return [CPU]
        if mode == "cpu" or (mode == "auto" and affected_apple_grade()):
            return [CPU]
    if stage == "rgb-denoiser":
        mode = os.environ.get("RAWALCHEMY_COREML_DENOISE", "auto").lower().strip()
        if mode not in ("auto", "cpu", "mlprogram"):
            logger.warning(f"Invalid RAWALCHEMY_COREML_DENOISE={mode!r}; using CPU")
            return [CPU]
        if mode == "cpu" or not eligible_apple():
            return [CPU]
        first = providers[0]
        options = dict(first[1]) if isinstance(first, tuple) else {}
        options.update(ModelFormat="MLProgram", MLComputeUnits="ALL")
        return [(COREML, options), *providers[1:]]
    return list(providers)


def configuration_token(variant):
    """Cheap session invalidation token; no model parsing/hashing per tile.

    Includes all live controls that change generated provider/session options.
    Model content is hashed for the disk compilation identity at construction;
    replacing a loaded model still requires clear_session().
    """
    import onnxruntime as ort
    from raw_alchemy import config

    return (
        POLICY_VERSION, variant, platform.system(), platform.machine(),
        platform.mac_ver()[0], getattr(ort, "__version__", ""),
        provider_identity(stage_providers([COREML, CPU], variant)),
        tuple(getattr(ort, "get_available_providers", lambda: [])()),
        os.environ.get("RAWALCHEMY_COREML_DENOISE", "auto"),
        os.environ.get("RAWALCHEMY_COREML_GRADE", "auto"),
        os.environ.get("RAWALCHEMY_COREML_DEMOSAIC", "auto"),
        os.environ.get("RAWALCHEMY_MIGRAPHX_DEMOSAIC", "auto"),
        os.environ.get("RAW_ALCHEMY_CPU_ONLY", "0"),
        os.environ.get("RAWALCHEMY_COREML_ISOLATION", "1"),
        os.environ.get("RAWALCHEMY_NATIVE_ISOLATION", "1"),
        config.DEFAULT_CPU_THREADS, config.ONNX_GPU_MEMORY_LIMIT_MB,
    )


# An accelerator that raised is not retried on every clear_session()/new image.
# A different model/options key can retry; restart explicitly clears the breaker.
_failed = set()
_failed_lock = threading.Lock()


def _mark_failed(identity):
    with _failed_lock:
        _failed.add(identity)


def registered_provider_names(session):
    """Registration diagnostics, also supporting lightweight session adapters."""
    getter = getattr(session, "get_providers", None)
    return getter() if getter else ["unknown (session adapter)"]


def _log_registration(stage, requested, selected, session, elapsed):
    logger.info(
        f"ORT {stage}: requested={provider_identity(requested)}; "
        f"attempted={provider_identity(selected)}; "
        f"registered={session.get_providers()}; initialization={elapsed:.3f}s "
        "(registered EPs do not prove node placement)"
    )


class RecoverableSession:
    """Keep callers on the replacement CPU session after one failed EP run."""

    def __init__(self, session, cpu_factory, identity, stage):
        self._session = session
        self._cpu_factory = cpu_factory
        self.identity = identity
        self._stage = stage
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._session, name)

    def run(self, output_names, input_feed, *args, **kwargs):
        # Serializes one session's recovery, not unrelated stage sessions.
        with self._lock:
            try:
                return self._session.run(output_names, input_feed, *args, **kwargs)
            except Exception as exc:
                from raw_alchemy.pipeline.executor import PipelineAborted
                if isinstance(exc, (PipelineAborted, MemoryError)):
                    raise
                if not any(p != CPU for p in self._session.get_providers()):
                    raise
                logger.warning(
                    f"ORT {self._stage}: accelerator inference failed "
                    f"({type(exc).__name__}: {str(exc)[:200]}); retrying once on CPU"
                )
                _mark_failed(self.identity)
                replacement = self._cpu_factory()
                self._session = replacement
                return replacement.run(output_names, input_feed, *args, **kwargs)


def create_session(ort, model_path, options_factory, providers, *, variant):
    """Initialize with fresh CPU options after an accelerator constructor error.

    CPU errors propagate unchanged. This handles exceptions, not native hangs.
    The factory preserves tile dimension overrides and thread limits on retry.
    """
    # Compilation-cache options already carry a complete model-content digest.
    # A stat generation also permits replacement/retry when disk cache is off.
    try:
        stat = os.stat(model_path)
        generation = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        generation = None
    identity = (str(model_path), generation, variant, provider_identity(providers))
    requested = list(providers)
    with _failed_lock:
        if identity in _failed:
            providers = [CPU]
    accelerated = any(p != CPU for p in provider_names(providers))

    def construct(selected):
        start = time.perf_counter()
        result = construct_session(ort, model_path, options_factory(), selected, variant=variant)
        _log_registration(variant, requested, selected, result, time.perf_counter() - start)
        return result

    def cpu_factory():
        return construct([CPU])

    try:
        session = construct(providers)
    except Exception as exc:
        from raw_alchemy.pipeline.executor import PipelineAborted
        if isinstance(exc, (PipelineAborted, MemoryError)):
            raise
        if not accelerated:
            raise
        logger.warning(
            f"ORT {variant}: requested={provider_identity(requested)} initialization "
            f"failed ({type(exc).__name__}: {str(exc)[:200]}); retrying once on CPU"
        )
        _mark_failed(identity)
        session = cpu_factory()
    if any(p != CPU for p in session.get_providers()):
        return RecoverableSession(session, cpu_factory, identity, variant)
    if accelerated:
        # ORT may silently fall back inside its constructor. Avoid recompiling
        # the same rejected request each time a stage releases its session.
        _mark_failed(identity)
    return session


def construct_session(ort, model_path, options, providers, *, variant):
    """Production CoreML owns a spawned process for its full session lifetime."""
    check_cancelled()
    isolation = os.environ.get("RAWALCHEMY_NATIVE_ISOLATION", "1")
    if COREML in provider_names(providers):
        isolation = os.environ.get("RAWALCHEMY_COREML_ISOLATION", isolation)
    if (isolation != "0"
            and getattr(ort.InferenceSession, "__module__", "").startswith("onnxruntime.")):
        from .isolated_session import IsolatedSession
        return IsolatedSession(model_path, options, providers, variant=variant)
    return ort.InferenceSession(model_path, options, providers=providers)
