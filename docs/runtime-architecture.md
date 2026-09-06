# Runtime ownership and platform contract

The foreground ImageProcessor and one bounded ExportDispatcher share a process-wide
ResourceGovernor. Expensive work has one compute owner; export yields at pipeline
operations, denoise tiles, grade strips and 4 MiB cache-I/O chunks. Foreground requests
have priority, with an admitted export allowed after three foreground grants.
Insufficient memory can defer admission, so preview latency is not guaranteed while a
large export retains its arrays. The export queue holds at most two pending requests.
GUI cache snapshots capture references and identity tokens without file reads; the
export lane validates identity before consuming them.

`RAWALCHEMY_MEMORY_LIMIT_MB` sets an application-plus-child RSS target; the default
is 70% of physical RAM. Admission reserves estimated future working sets and retains
256 MiB of available system memory. Stage checkpoints and native wait loops monitor
RSS. Shared mappings may be counted twice. This is conservative admission and sampled
failure handling, not an operating-system allocation limit. Queued image references,
Qt, ONNX arenas and pools are reflected in RSS; CLI ProcessPool jobs still use the
existing batch concurrency estimate rather than sharing the GUI governor. Native
libraries can allocate between samples, and driver-only GPU memory is not fully
represented by RSS.

RAW decode and production ONNX sessions use owned spawned processes on Windows,
Linux and macOS. The decode child owns its internal demosaic session directly, so it
does not recursively spawn. The parent can terminate/reap a stuck decode or inference;
CPU constructor fallback uses the same session boundary. Cancellation and memory-limit
failures propagate instead of starting extra CPU work. Encoding, metadata I/O and other
non-ONNX native operations remain cooperative between stages. These changes do not
promise a universal deadline for application shutdown.

Controls:

- `RAWALCHEMY_NATIVE_ISOLATION=0`: opt into in-process ONNX diagnostics.
- `RAWALCHEMY_COREML_ISOLATION`: optional CoreML-specific override.
- `RAWALCHEMY_NATIVE_COMPILE_TIMEOUT` / `RAWALCHEMY_NATIVE_RUN_TIMEOUT`: default
  60 seconds each; legacy CoreML timeout controls remain fallback defaults.
- `RAWALCHEMY_DECODE_TIMEOUT`: default 120 seconds for one complete RAW decode.
- Timeouts accept finite values clamped to 0.1–600 seconds.
- `RAW_ALCHEMY_CPU_ONLY=1`: select CPU across platforms; isolation still applies.

IPC tensor allocations are capped at 512 MiB per ONNX invocation. RAW output dimensions
are validated separately. Linux additionally checks available `/dev/shm` before mapping
buffers; containers need sufficient shared memory for their image sizes. A shortage is
reported rather than allowing an allocation to crash on first access.

Denoise artifacts retain float32 pixels and include source content, decoder provenance,
model/source versions, working colour space and backend-policy identity. Interrupted
cache writes remove their temporary file and preserve previously published data. Native
provider registration is still not proof of actual GPU node execution; acceptance tests
must profile the selected runtime and compare actual pixels.

## Reproducible native builds

`native-dependencies.json` pins Lensfun release, archive SHA-256, complete extracted
file-tree SHA-256 and verified architecture. Windows/macOS use verified binaries.
Linux compiles the pinned source commit with CMake against the build host ABI, then
checks that the library loads before publishing it into the generated build tree.
The former Linux binary required GLIBC 2.38 and failed on Debian 12 (glibc 2.36).
Build Linux release wheels on the oldest supported distribution; no manylinux
compatibility is claimed. Linux requires a C++17 compiler, CMake, pkg-config and
GLib development headers. Source builds are rebuilt with the current toolchain;
only matching Windows/macOS binary trees are reused. No build downloads into the
source checkout. `RAWALCHEMY_LENSFUN_ARCHIVE` supplies an offline archive,
which undergoes the same verification.

Both PyInstaller specs use the same verified Lensfun tree under their generated
work directory. Pre-existing Lensfun files in a developer checkout are excluded
from frozen bundles. For editable development, prepare a verified tree with
`build_support.ensure_lensfun(path)` and set `RAWALCHEMY_LENSFUN_DIR` to that
directory; the runtime library and camera database resolve from it together.
The application itself never downloads native code during startup.

Currently supported native build combinations are Windows x86-64, Linux x86-64
and macOS arm64. Other architectures fail with an explicit unsupported-binary error rather than
silently receiving a different architecture. Adding Intel macOS or ARM Windows/Linux
requires corresponding pinned binary assets and native validation. Python starts at
3.11; optional newer tar safety filters are used where available, after verifying the
exact archive digest on every supported interpreter.

The real X-Trans CoreML GPU precision failure remains unresolved. Automatic CoreML
macOS demosaic therefore uses CPU; explicit MLProgram remains diagnostic. Portable
unit tests do not certify Windows DirectML, Linux CUDA/ROCm, or macOS GPU quality.

## Content identity and frozen-library verification

RAW and model hashes are recomputed from the current file contents at validation
boundaries. A same-size rewrite can retain identical file timestamps on Windows;
timestamps therefore cannot authorize reuse of a previous content hash. These
checks run on worker lanes, including export snapshot validation. Hashing checks
cancellation between 1 MiB blocks and does not yield the compute slot while a
caller might own a cache lock. This trades additional file reads for correct
invalidation; no new whole-application performance guarantee is implied.

Both frozen-build specs preserve vendor library layout without stripping.
The Linux validation reproduced ELF LOAD misalignment in NumPy/SciPy OpenBLAS
after the build tool's strip command. Merely completing PyInstaller is
insufficient: the resulting executable must load its libraries, show its
OpenGL viewport, and close normally.
