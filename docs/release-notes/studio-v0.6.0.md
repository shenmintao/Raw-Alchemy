# Raw Alchemy Studio v0.6.0

Raw Alchemy Studio 0.6 is a new engine behind the same editor. RAW decoding, demosaicing, denoising and colour grading now run on ONNX Runtime on Windows, macOS and Linux. The preview was rebuilt for fast opening and instant zoom, and the denoiser is faster and keeps colours accurate. These notes cover changes since studio-v0.5.0; the nine 0.6.0 prereleases carry the details.

## ✨ New RAW engine

- **Decoding.**
  - RawSpeed does the decoding, with LibRaw as the fallback.
  - Black level is subtracted per channel, and noise below black is kept rather than clipped. Clipping it turned dark night skies magenta.
  - Hot pixels are removed.
  - Highlights are reconstructed with a segmentation-based method (darktable "segments" semantics).
  - Camera-to-ProPhoto matrices use darktable's analytic method.
- **GPU demosaicing on ONNX Runtime.**
  - RCD for Bayer sensors and Markesteijn for Fujifilm X-Trans.
  - The X-Trans colour-filter layout is read per camera model from RawSpeed's camera database.
  - DirectML on Windows, CUDA on NVIDIA, CoreML on macOS, and verified AMD MIGraphX models on Linux.
  - Falls back to the CPU automatically.
- **Taichi retired.** All GPU compute now runs on one runtime.
- **Robust decoding.** Decoding runs in isolated processes that can be cancelled and stay warm between images. Work is cancelled cooperatively, and memory is admitted conservatively, so a failed or oversized image cannot take the editor down.

## 🧹 Denoise

- **FastDenoise v4 replaces the v0.5 UtNet model.** It is an RGB network that works on the linear ProPhoto image, with a strength slider.
- **No brightness or colour shift.** The denoised image's local brightness is re-anchored to the original for each luminance band. Mid-tones do not brighten and shadows do not turn green, with no halos around bright objects.
- **Denoise once, zoom freely.** Denoising runs once at native resolution; zooming and panning reuse the result. Results are cached on disk (20 GB by default).

## 🖼️ Preview and editing

- **GPU colour grading.** Exposure, colour, log curves and 3D LUTs run as one fused graph on the GPU.
- **Instant zoom.**
  - Above 100%, only the visible region is rendered, at a resolution matched to the screen's pixel density.
  - When GPU and memory limits allow, a native-resolution base image makes 1:1 and wheel zoom instant.
  - Zoom follows the mouse.
  - 100% accounts for display scaling.
- **Faster browsing.**
  - Opening images is faster thanks to layered caches, warm decode processes, and quicker hot-pixel removal and highlight reconstruction.
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
- **Night-sky colour.** Fixed magenta or purple night skies caused by clipping noise below the black level.
- **Thumbnail orientation.** Fixed a double-rotated thumbnail for portrait RAF files and DNG thumbnail orientation.
- **Colour conversion.** Fixed the ProPhoto → sRGB conversion (thanks @y-g-jiang).
- **Memory.**
  - Fixed out-of-memory failures.
  - GPU memory is now released after denoising.
  - Fixed a buffer overrun in the GPU memory query that could crash on exit.
- **Stability.**
  - Deleting an image no longer falls back to a raw file removal.
  - The editor writes its log file.
  - Crashes in worker threads are reported.

## 📦 Downloads

- **Windows x86-64:** portable ZIP. Extract it anywhere and run `RawAlchemy.exe`.
- **macOS Apple Silicon:** DMG. Copy `RawAlchemy.app` to Applications. macOS 15 or later is required.
- **Linux x86-64:** portable tarball. Extract it and run `RawAlchemy/RawAlchemy`. It was built on Ubuntu 24.04 and needs a glibc at least that new; the desktop OpenGL/XCB libraries listed in the README are also required.
- `SHA256SUMS.txt` lists the archive checksums.
- Portable packages start faster than the single-file executables of 0.5 and do not auto-update.
- **Unsigned binaries.**
  - On Windows, SmartScreen may show "Windows protected your PC": choose "More info", then "Run anyway".
  - On macOS, if Gatekeeper refuses the app, run `xattr -dr com.apple.quarantine /Applications/RawAlchemy.app`.

## ✅ Validation

- **Hosted CI.** The full test suite passes on Windows 2022, macOS 15 and Ubuntu, including platform-wheel runtime checks.
- **Windows (RX 9070 XT, DirectML).**
  - The editor was exercised end to end on Sony DNG and Fujifilm RAF files: opening, zoom, denoise and colour checks.
  - The packaged app decoded both formats, reused its decode process, and left no processes behind after closing.
- VALIDATION_PENDING

## Known limitations

- **Denoise.**
  - The denoise colour correction adds about 0.8 s per 42 MP image.
  - At very high ISO, the true shadow colour brings back some colour noise; raise the denoise strength if needed.
- **CANS / AI demosaic-denoise.** Still disabled pending quality acceptance.
- **AMD on Linux.** MIGraphX acceleration needs a separately configured ROCm/MIGraphX environment; the default Linux bundle contains the CUDA/CPU runtime.
- **Windows.**
  - DirectML Bayer demosaicing differs very slightly from the CPU reference (max abs error 0.00014 on the tested NEF). Use the CPU backend when strict reference consistency is required.
  - A missing WMIC can affect GPU-name detection.
- **Memory limits.** Memory admission is conservative rather than an OS-enforced hard cap.
- **Coverage.** GPU validation covers the measured hardware and runtime combinations, not every camera, driver or OS version.
