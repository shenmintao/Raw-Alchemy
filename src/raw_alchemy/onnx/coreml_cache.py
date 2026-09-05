"""Best-effort, content-addressed CoreML compilation cache.

Only the macOS CoreML provider calls this module. Hash every referenced external
file in full (not just its timestamp or a tensor slice), without loading weights
into memory. Protobuf's public reflection API covers tensors in nested graphs,
attributes, sparse tensors, functions and training graphs as well as initializers.
"""

import hashlib
import json
from pathlib import Path
import platform
import tempfile

from loguru import logger


def _external_locations(message, tensor_type):
    if isinstance(message, tensor_type):
        if message.data_location == tensor_type.EXTERNAL:
            locations = [item.value for item in message.external_data
                         if item.key == "location"]
            if len(locations) != 1 or not locations[0]:
                raise ValueError("External ONNX tensor has no unique location")
            yield locations[0]
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            repeated = getattr(field, "is_repeated", None)
            if repeated is None:  # Older protobuf releases supported by ONNX.
                repeated = field.label == field.LABEL_REPEATED
            if repeated:
                for child in value:
                    yield from _external_locations(child, tensor_type)
            else:
                yield from _external_locations(value, tensor_type)


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coreml_cache_dir(model_path=None, *, variant="") -> str | None:
    """Return a writable namespace, or None without preventing CoreML use.

    Never memoize by path/mtime: replacement weights at the same path must
    invalidate compiled models. The versioned namespace also isolates ORT,
    OS/architecture and session shape overrides. Old namespaces may be deleted
    while the app is closed; no model metadata or source files are modified.
    """
    if model_path is None:
        return None
    try:
        # Keep ONNX parsing entirely off the CPU/CUDA/DirectML startup path.
        import onnx
        import onnxruntime as ort

        path = Path(model_path)
        model_bytes = path.read_bytes()
        model = onnx.load_model_from_string(model_bytes)
        external = [
            (location, _file_digest(path.parent / location))
            for location in sorted(set(_external_locations(model, onnx.TensorProto)))
        ]
        identity = {
            "schema": 1,
            "model": hashlib.sha256(model_bytes).hexdigest(),
            "external": external,
            "ort": ort.__version__,
            "system": [platform.system(), platform.release(), platform.version(),
                       platform.mac_ver()[0], platform.machine()],
            "variant": variant,
        }
        namespace = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache = Path.home() / "Library" / "Caches" / "RawAlchemy" / "coreml" / namespace
        cache.mkdir(parents=True, exist_ok=True)
        # mkdir(exist_ok=True) and os.access alone miss existing read-only dirs.
        with tempfile.TemporaryFile(dir=cache) as probe:
            probe.write(b"cache write probe")
            probe.flush()
        return str(cache)
    except Exception as exc:
        # Parsing/import/hash/probe failures must cost only the cache speedup.
        logger.warning("CoreML cache unavailable; continuing without cache: {}", exc)
        return None
