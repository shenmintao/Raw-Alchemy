"""ONNX stages share a runtime with telemetry disabled from first import."""

# ORT starts its telemetry service on import. Disabling only at session
# construction can leave an import-time HTTP upload in flight when a short
# lived frozen decode process exits (ORT 1.30/macOS recursive-mutex crash).
try:
    import onnxruntime as _runtime
except ImportError:
    _runtime = None
else:
    _disable_telemetry = getattr(_runtime, "disable_telemetry_events", None)
    if _disable_telemetry is not None:
        _disable_telemetry()
