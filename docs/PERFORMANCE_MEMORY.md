# Performance and Memory Architecture

This document records the post-`studio-v0.6.0-pre7` resource-governance pass:
layered preview rendering, latest-frame presentation and explicit GPU scratch
reclamation.

## Preview tiers

- Interactive fit preview: 3MP proxy where available.
- Quality base: full-frame render capped at a 4096px long edge / 16MP. The
  downsample is inserted before exposure, white balance and the colour tail.
- Detail: native-resolution ROI with a margin around the visible viewport.
- Export: full resolution and independent from presentation limits.

After the first full lens correction, the corrected frame becomes the direct
source for later slider and ROI requests. With the common no-geometry path,
the executor crops/downsamples from that immutable source before the colour
tail; it no longer copies the native source out and uploads the corrected frame
again on every interaction.

The quality base replaces the previous native full-frame base texture. A 61MP
frame therefore no longer remains simultaneously as a host `uint8` image, PBO,
base texture and mip chain. Zoomed detail is still rendered at 1:1 through the
ROI tier.

## Host memory ownership

- Decoded-image cache: 2048MB absolute default and 35% available-memory cap.
- Numpy executor prefix/final cache: 768MB shared by proxy/full executors and
  enforced while a run is building prefixes, not only after completion. This
  is host RAM; ONNX/OpenGL VRAM is governed separately.
- Recyclable ndarray pool: 256MB; oversized native-frame scratch is freed
  instead of becoming a persistent allocator high-water mark.
- Per-image output slots: both a 10-slot LRU limit and a 256MB byte limit.
- GUI image state retains shared immutable `uint8` arrays only. It no longer
  stores full-resolution `QPixmap` copies, and the cache/state/viewport share
  the same result object where safe.
- Baseline rendering is one-shot and releases its executor buffers after the
  result is emitted.
- Scope workers coalesce pending frames, release their source reference as soon
  as computation finishes, and are reset on image switches.
- Lensfun coordinate maps use a 256MB byte LRU. Oversized native maps are never
  built: a two-pass striped remap uses bounded coordinate scratch instead.
- Lensfun database objects use a four-entry LRU and are cleared at shutdown.

CPU colour matrices, tone operations, transfer functions and 1D/3D LUTs use
bounded chunks. RGB denoising uses one output accumulator plus one weight map;
it no longer materialises full-frame clipped, gained, encoded, saturation-mask
and decoded arrays at the same time.

Export quantization writes directly from float32 into the uint8/uint16 output
buffer. HEIF receives a memory view rather than another whole-frame `bytes`
copy. The denoise disk cache streams zstd data to/from float16 storage instead
of materialising compressed and decompressed Python byte strings. ST 2084/PQ
encoding is float32, chunked and in-place.

## GPU / presentation lifetime

- CUDA sessions use same-as-requested arenas, a configurable 2048MB per-session
  cap and cuDNN heuristic search without maximum workspace reservation.
- Demosaic sessions use fixed tile dimensions through ONNX Runtime free-
  dimension overrides and are released after 15 seconds without decoding.
- All sessions are released after 120 seconds of application inactivity.
- OpenGL keeps mipmaps only for the base texture. ROI textures use linear
  filtering without the extra 33% mip-chain allocation.
- PBO storage is reused for small frames; allocations above 64MB are released
  immediately after presentation. Cleared ROI/base textures are shrunk with a
  current GL context instead of retaining their peak dimensions.

## Concurrency

Thumbnail extraction is capped at four low-priority workers. This prevents a
high-core-count machine from launching dozens of simultaneous rawpy fallback
decodes, while the visible-first, bounded-in-flight scheduling remains intact.
CLI batch processing is capped to one process when a GPU ONNX provider is
active; CPU-only concurrency is limited by available RAM using a conservative
3GB-per-process working-set estimate.

Slider interaction uses an immediate leading request, approximately 80ms
throttling while dragging, and an exact final request on release. The worker's
latest-wins queue and op-boundary cancellation prevent this tighter cadence
from building a backlog.

## Verification snapshot (2026-07-14)

- `pytest tests -q -p no:cacheprovider`: 258 passed.
- `python -m compileall -q src tests`: passed.
- Synthetic 24MP quality-base render (`RAWALCHEMY_GRADE_GPU=0`): 0.69s,
  approximately 599MB RSS increase above the already-decoded/cache baseline;
  all executor prefixes were `4096x2731` and the free pool retained 0MB.
- Synthetic 12MP identity denoise: 1.02s, approximately 225MB peak RSS increase,
  maximum round-trip error `1e-8`.
- RTX A1000 6GB, real CUDA RCD session: 835MB reported while resident and 77MB
  after `clear_session()` (the remaining allocation is the CUDA process
  context/runtime baseline).
- Lensfun striped-vs-full comparison on a real 800x600 calibration case: mean
  absolute error about `4.1e-7`, maximum about `5.5e-4`, without allocating a
  native full-frame coordinate map.

Hardware GUI acceptance should still check long browsing sessions and viewport
feel with representative 45/61MP RAW files, because driver texture accounting
and camera-specific decode peaks cannot be fully modelled by unit tests.
