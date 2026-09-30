"""Measure real preview adjustments, detail views, denoise and export.

Run with the target checkout's Python; writes timings and the selected ONNX
providers to JSON. Example: python tools/bench_pipeline.py image.RAF --out run.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ort-dir", type=Path)
    parser.add_argument("--denoise", action="store_true", help="include expensive first denoise")
    parser.add_argument("--profile", action="store_true", help="profile the preview worker into OUT.prof")
    parser.add_argument("--large-roi", action="store_true", help="also render a large native-resolution viewport")
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "src"))
    if args.ort_dir:
        sys.path.insert(0, str(args.ort_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("RAW_ALCHEMY_LOG_DIR", str(args.out.parent / "logs"))

    import cv2
    import numpy as np
    import onnxruntime as ort
    from PySide6.QtCore import QCoreApplication, Qt
    from raw_alchemy.workers.image_processor import ImageProcessor
    from raw_alchemy.onnx import grade, rgb_denoiser

    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        process_preview = ImageProcessor._do_process

        def profiled_preview(self, request):
            profiler.enable()
            try:
                return process_preview(self, request)
            finally:
                profiler.disable()

        ImageProcessor._do_process = profiled_preview

    app = QCoreApplication([])
    # Throughput measurements wait for the final denoise result. GUI latency
    # probes separately exercise the default background scheduling.
    worker = ImageProcessor(warmup_sessions=False, background_denoise=False)
    done = threading.Event()
    state = {}
    report = dict(raw=args.raw, ort=ort.__version__, providers=ort.get_available_providers(), cases={})

    def result(image, path, request_id, ev, source_size):
        state.update(shape=list(image.shape), source_size=list(source_size), ev=ev)
        done.set()

    def error(message):
        state["error"] = message
        done.set()

    worker.result_ready.connect(result, Qt.ConnectionType.DirectConnection)
    worker.load_complete.connect(lambda *_: done.set(), Qt.ConnectionType.DirectConnection)
    worker.error_occurred.connect(error, Qt.ConnectionType.DirectConnection)
    params = dict(exposure_mode="Manual", exposure=1.6, wb_temp=0.0,
                  wb_tint=0.0, highlight=0.0, shadow=0.0, saturation=1.25,
                  contrast=1.1, log_space=None, lut_path=None, lens_correct=True,
                  sharpen_strength=0.0, denoise_enabled=False, denoise_strength=0.25,
                  rotation=0, crop=(0, 0, 1, 1), viewport_size=(1610, 1075),
                  preview_zoom=1.0, device_pixel_ratio=2.0, max_preview_pixels=4_000_000)

    def wait_result(start, timeout=300):
        if not done.wait(timeout):
            raise RuntimeError("processing timed out")
        if "error" in state:
            raise RuntimeError(state["error"])
        return (time.perf_counter() - start) * 1000

    def render(p):
        done.clear()
        state.clear()
        start = time.perf_counter()
        worker.update_preview(args.raw, p)
        return wait_result(start)

    def measure(name, variants):
        timings = [render(p) for p in variants]
        report["cases"][name] = dict(ms=timings, median_ms=float(np.median(timings)), **state)
        print(name, json.dumps(report["cases"][name]), flush=True)
        save()

    def save():
        report["grade_provider"] = grade._session_provider
        report["denoise_provider"] = rgb_denoiser._session_provider
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        done.clear()
        start = time.perf_counter()
        worker.load_image(args.raw)
        report["load_ms"] = wait_result(start)
        report["source_shape"] = list(worker.cpu_linear.shape)
        report["warmup_ms"] = render(params)
        for key, values in {
            "exposure": [1.7, 1.8, 1.9], "wb_temp": [5, 10, 15],
            "wb_tint": [5, 10, 15], "highlight": [-10, -20, -30],
            "shadow": [10, 20, 30], "saturation": [1.15, 1.2, 1.3],
            "contrast": [1.05, 1.15, 1.2],
        }.items():
            measure(key, [dict(params, **{key: value}) for value in values])
        measure("crop", [dict(params, crop=(0.05, 0.05, scale, scale)) for scale in [0.85, 0.8, 0.75]])
        measure("rotation", [dict(params, rotation=value) for value in [90, 180, 270]])
        measure("detail_pan", [dict(params, preview_zoom=3.0,
                 preview_visible_rect=(x, 0.3, x + 0.2, 0.5)) for x in [0.3, 0.32, 0.34]])
        if args.large_roi:
            measure("large_roi", [dict(params, preview_zoom=1.05,
                preview_visible_rect=(.08, .08, .92, .92), exposure=value,
                viewport_size=(4096, 2730), max_preview_pixels=26_000_000)
                for value in [1.72, 1.82, 1.92]])
        measure("sharpen", [dict(params, sharpen_strength=value) for value in [0.2, 0.4, 0.6]])
        lut = args.root / "tests/golden/identity.cube"
        measure("lut", [dict(params, lut_path=str(lut), exposure=value) for value in [1.55, 1.65, 1.75]])
        measure("log_lut", [dict(params, log_space="F-Log2", lut_path=str(lut), exposure=value)
                            for value in [1.55, 1.65, 1.75]])
        measure("lens_toggle", [dict(params, lens_correct=value, exposure=1.63)
                                for value in [False, True, False]])
        if args.denoise:
            # Covers first model inference and subsequent slider reuse.
            measure("denoise_first", [dict(params, denoise_enabled=True)])
            measure("denoise_reuse", [dict(params, denoise_enabled=True, exposure=value)
                                      for value in [1.73, 1.83, 1.93]])

        # Export math on the original decoded image, then lossless file I/O.
        # Denoise/lens first-use timings are reported separately above.
        from raw_alchemy.pipeline.executor import ExportExecutor
        from raw_alchemy.pipeline.ops import build_op_list
        full = worker.cpu_linear
        start = time.perf_counter()
        exported = ExportExecutor().run(build_op_list(dict(params, lens_correct=False)), full)
        report["export_grade_ms"] = (time.perf_counter() - start) * 1000
        import tifffile
        start = time.perf_counter()
        tifffile.imwrite(args.out.with_suffix(".tif"),
                         (np.clip(exported, 0, 1) * 65535 + 0.5).astype(np.uint16),
                         photometric="rgb")
        report["export_tiff_ms"] = (time.perf_counter() - start) * 1000
        report["export_finite"] = bool(np.isfinite(exported).all())
        save()
        print("BENCH_RESULT", json.dumps(report), flush=True)
    finally:
        worker.stop_and_cleanup()
        if args.profile:
            profiler.dump_stats(str(args.out.with_suffix(".prof")))


if __name__ == "__main__":
    main()
