# Raw Alchemy Studio v0.6.0

Raw Alchemy Studio 0.6 replaces the processing engine behind the editor. RawSpeed/LibRaw decode the sensor data; ONNX Runtime handles demosaicing, denoising and fused colour grading on Windows, macOS and Linux. Slider edits continue to update the preview while background denoising runs, and fixed-shape CoreML models substantially reduce editing latency on the tested Apple Silicon machine. These notes cover changes since studio-v0.5.0, including the nine 0.6.0 prereleases and the final decoding and responsiveness fixes.

## ✨ New RAW engine

- **Decoding.**
  - RawSpeed does the decoding, with LibRaw as the fallback.
  - Black level is subtracted per channel, and noise below black is kept rather than clipped. Clipping it turned dark night skies magenta.
  - Sensor samples are preserved in the standard path. The previous automatic hot-pixel heuristic erased real X-Trans detail and introduced false colour; defect correction is independent of RCD demosaicing.
  - Highlights use a simplified segmentation-based reconstruction adapted from darktable.
  - Camera-to-ProPhoto matrices use darktable's analytic method.
- **GPU demosaicing on ONNX Runtime.**
  - RCD for Bayer sensors and Markesteijn for Fujifilm X-Trans.
  - The X-Trans colour-filter layout is read per camera model from RawSpeed's camera database.
  - DirectML on Windows, CUDA on NVIDIA, CoreML on macOS, and verified AMD MIGraphX models on Linux.
  - Falls back to the CPU automatically.
- **Taichi retired.** GPU demosaicing, denoising and grading use ONNX Runtime.
- **Robust decoding.** Decoding runs in isolated processes that can be cancelled and stay warm between images. Work is cancelled cooperatively, and memory is admitted conservatively, so a failed or oversized image cannot take the editor down.
- **Compatibility scope.** The complete decoding pipeline is not pixel-equivalent to darktable. White-balance ordering, negative values, highlight reconstruction, crop/profile handling and output clipping differ; GPU parity checks compare against this project's CPU reference.

## 🧹 Denoise

- **FastDenoise v4 replaces the v0.5 UtNet model.** It is an RGB network that works on the linear ProPhoto image, with a strength slider.
- **No brightness or colour shift.** The denoised image's local brightness is re-anchored to the original for each luminance band. Mid-tones do not brighten and shadows do not turn green, with no halos around bright objects.
- **Denoise once, zoom freely.** Denoising runs once at native resolution; zooming and panning reuse the result. Results are cached on disk (20 GB by default).
- **Responsive background work.** One bounded background job yields between tiles so exposure, white balance and other preview edits can proceed. Changing the image or final strength, disabling denoise or closing the app cancels obsolete work. Validated results redraw the latest settings before lossless disk-cache compression finishes.
- **Strength settles before inference.** During strength dragging, the preview reuses existing denoise or displays the original source. The final strength is computed after release and 200 ms of quiet input. Saved settings and export retain the requested final strength.

## 🖼️ Preview and editing

- **GPU colour grading.** Exposure, colour, log curves and 3D LUTs run as one fused graph on the GPU.
- **Continuous slider feedback.** All Inspector sliders share event-driven request scheduling. A normal preview completes before the latest slider settings are submitted, avoiding repeated cancellation and GPU-session rebuilding. Image switches and viewport changes remain cancellable.
- **Apple Silicon acceleration.** CoreML uses reusable fixed pixel tiles for grading and smaller demosaic tiles. Selection checks architecture, runtime and provider capabilities rather than one exact OS/runtime version. Failed GPU configurations fall back to CPU without a retry loop; CPU-only grading avoids process-copy overhead.
- **Instant zoom.**
  - Above 100%, only the visible region is rendered, at a resolution matched to the screen's pixel density.
  - When GPU and memory limits allow, a native-resolution base image makes 1:1 and wheel zoom instant.
  - Zoom follows the mouse.
  - 100% accounts for display scaling.
- **Faster browsing.**
  - Layered caches, warm decode processes and streamlined preprocessing reduce repeated work.
  - Thumbnails have a disk cache and are generated for visible items first.
- **Settings copy and paste.** Copy chosen setting groups and paste them onto the current image or every marked image (Ctrl+Shift+C / Ctrl+Shift+V), with confirmation and one-step undo.
- **Library.** The panel is resizable and remembers its width, and the thumbnail grid and folder tree layout were fixed.
- **Lens correction.** Lensfun distortion maps are cached, lens-correction data is bundled, and remapping is faster.

## 📤 Export

- **HDR HEIF.** Export 10-bit BT.2020/PQ HDR HEIF, in addition to JPG, SDR HEIF, TIFF and DNG.
- **Scheduling.** Exports go through the same processing worker as the preview and are bounded.
- **Metadata.** More EXIF tags are written, and a TIFF export broken by incompatible EXIF data was fixed.

## 🐛 Notable fixes

- **X-Trans colour.** Fixed a magenta cast with stripes on X-Trans cameras whose colour-filter layout differs from the X-Trans IV layout (for example the X-T10).
- **X-Trans detail.** Removed automatic hot-pixel filtering that caused severe false colour on petals and branches in the tested X-S10 RAF, with no LUT or log conversion enabled.
- **Night-sky colour.** Fixed magenta or purple night skies caused by clipping noise below the black level.
- **Thumbnail orientation.** Fixed a double-rotated thumbnail for portrait RAF files and DNG thumbnail orientation.
- **Colour conversion.** Fixed the ProPhoto → sRGB conversion (thanks @y-g-jiang).
- **Memory.**
  - Fixed out-of-memory failures.
  - Background denoise has one admitted working set and checks source identity before publishing. Warm sessions and bounded caches remain reusable.
  - Fixed a buffer overrun in the GPU memory query that could crash on exit.
- **Stability.**
  - Deleting an image no longer falls back to a raw file removal.
  - The editor writes its log file.
  - Crashes in worker threads are reported.
  - Disabled ONNX Runtime telemetry before native-session creation to avoid the observed ORT 1.30 macOS shutdown crash.

## Performance measurements

Measured on the same 26 MP X-S10 RAF, with approximately 3 MP warmed previews and three parameter variants per ordinary edit. These are processing latencies, not display FPS or guarantees for other hardware.

| Request | Apple M4, macOS 27, CoreML / ORT 1.30 | Windows 11, DirectML / ORT 1.24.4 |
| --- | --- | --- |
| Exposure | 304.7 → 132.1 ms | 255.3 → 196.3 ms |
| White balance | 307.9 → 135.0 ms | 252.3 → 200.0 ms |
| LUT | 1488.3 → 372.6 ms | 266.6 → 210.8 ms |
| Log + LUT | 2511.2 → 440.8 ms | 266.8 → 216.3 ms |
| 26 MP export grading | 5904.5 → 784.4 ms | 836.7 → 840.4 ms |

Other colour controls, crop, rotation, detail panning, sharpening and lens correction were also measured and improved. During active denoise, a controlled exposure drag returned 9 intermediate results on Mac and 5 on Windows, versus 0 with synchronous denoise. After release, final exposure updated in 53 ms on Mac and 275 ms on Windows; complete denoise arrived about 0.22 / 0.47 s later because preview work took priority. Windows ordinary dragging gained intermediate feedback but its final frame did not become faster in that test.

First-load/model compilation and disk I/O remain separate costs. Windows first denoise and export grading were essentially unchanged. See [the detailed audit and benchmark report](https://github.com/shenmintao/Raw-Alchemy/blob/studio-v0.6.0/docs/decoding-performance-2026-09-30.md) for hardware, methods, accuracy and limitations.

## 📦 Downloads

- **Windows x86-64:** portable ZIP. Extract it anywhere and run `RawAlchemy.exe`.
- **macOS Apple Silicon:** DMG. Copy `RawAlchemy.app` to Applications. macOS 15 or later is required.
- **Linux x86-64:** portable tarball. Extract it and run `RawAlchemy/RawAlchemy`. It was built on Ubuntu 24.04 and needs a glibc at least that new; the desktop OpenGL/XCB libraries listed in the README are also required.
- `SHA256SUMS.txt` lists the archive checksums.
- Portable packages start faster than the single-file executables of 0.5 and do not auto-update.
- ONNX Runtime and the platform runtime provider are included. CUDA driver/runtime requirements and optional MIGraphX setup still depend on the host system.
- **Unsigned binaries.**
  - On Windows, SmartScreen may show "Windows protected your PC": choose "More info", then "Run anyway".
  - On macOS, if Gatekeeper refuses the app, run `xattr -dr com.apple.quarantine /Applications/RawAlchemy.app`.

## ✅ Validation

- **Hosted CI.** The full test suite passes on Windows 2022, macOS 15 and Ubuntu, including platform-wheel runtime checks.
- **Windows (RX 9070 XT, DirectML).**
  - The editor was exercised end to end on Sony DNG and Fujifilm RAF files: opening, zoom, denoise and colour checks.
  - The packaged app decoded both formats, reused its decode process, and left no processes behind after closing.
- **Local regression suites.** Windows and Mac each passed 601 tests, with 11 hardware/fixture-dependent skips. Coverage includes background denoise, cancellation, image switching, failure handling, latest-parameter replay and export-cache identity.
- **Apple M4 native acceptance.** Six GPU tests checked actual CoreML execution, finite pixels and CPU-reference accuracy for Bayer, X-Trans, RCD seams/phases, plain grading, LUT and log+LUT. All passed the configured tolerance; maximum absolute error was below 8e-7. Synchronous and background full-resolution denoise results were identical in the tested RAF.
- **Windows native execution.** Production-session profiling recorded DirectML nodes for grading and denoise with no CPU node events in those runs.
- **macOS packages.** A production-spec frozen GUI test decoded the real RAF, updated exposure during background denoise, published the final source and exited normally. A separate build without the test hook also started and exited normally; release builds use the unmodified production spec.

## Known limitations

- **Denoise.**
  - Background scheduling improves editing response, not model throughput. First compilation, a running native tile and memory admission can still delay a preview; strength settling is not a completion-time guarantee.
  - The denoise colour correction adds about 0.8 s per 42 MP image.
  - At very high ISO, the true shadow colour brings back some colour noise; raise the denoise strength if needed.
- **CANS / AI demosaic-denoise.** Still disabled pending quality acceptance.
- **AMD on Linux.** MIGraphX acceleration needs a separately configured ROCm/MIGraphX environment; the default Linux bundle contains the CUDA/CPU runtime.
- **Windows.**
  - DirectML Bayer demosaicing differs very slightly from the CPU reference (max abs error 0.00014 on the tested NEF). Use the CPU backend when strict reference consistency is required.
  - A missing WMIC can affect GPU-name detection.
- **Memory limits.** Memory admission is conservative rather than an OS-enforced hard cap.
- **GPU scope.** Some geometry, sharpening and lens-map operations still use NumPy/OpenCV. CoreML can retain CPU partitions for precision-sensitive operations.
- **Coverage.** GPU validation covers the measured hardware and runtime combinations, not every camera, driver or OS version.
