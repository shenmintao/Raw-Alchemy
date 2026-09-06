# Stage-specific ONNX Runtime backend policy

## Defaults and scope

A registered provider is not evidence that it ran the expensive graph. The
September 2026 MacMini diagnosis found the bundled FastDenoise graph stayed on
CPU with the old CoreML NeuralNetwork format. The measured MLProgram format
actually executed CoreML partitions: approximately 16x full-image denoise speedup
on the two tested Bayer/X-Trans inputs. This is not a whole-application speedup,
first-compile timing, or a guarantee for other Apple devices.

| Stage | Apple default | Other backends |
| --- | --- | --- |
| Bundled `fastdenoise_v4_512_fp16.onnx` RGB denoise | CoreML `ModelFormat=MLProgram`, `MLComputeUnits=ALL` on native arm64 macOS 12+ / ORT 1.20+; CPU on ineligible Apple configurations | Existing CUDA/ROCm/DirectML/CPU ordering and provider options |
| Fused grade | CPU ORT on the diagnosed software generation: Apple Silicon, macOS 27, ORT 1.29; existing selection elsewhere | Existing selection |
| CANS packed RAW denoise | Existing selection; not covered by the RGB model benchmark | Existing selection |
| RCD demosaic | CPU; explicit MLProgram diagnostics remain available | CUDA/DirectML unchanged; MIGraphX uses its Gather-mask variant at tile 1536 on validated Linux x86-64 ORT 1.23.2 / ROCm 7.2.0; CPU elsewhere |
| X-Trans demosaic | Repaired MLProgram + ALL on measured Apple Silicon macOS 27 / ORT 1.29; CPU elsewhere unless explicitly selected | CUDA/DirectML unchanged; MIGraphX uses its precision variant on validated Linux x86-64 ORT 1.23.2 / ROCm 7.2.0; CPU elsewhere |

The grade CPU decision keeps the fused ONNX graph: it does **not** disable grade
or switch to the older per-operation NumPy path. Only one Apple machine was
measured; the version gate deliberately avoids asserting a speed regression on
unmeasured macOS/ORT generations. Capability eligibility does not guarantee a
speedup across all Apple Silicon hardware. DirectML options and CUDA memory limits/workspace controls are unchanged.
Linux now recognizes the modern MIGraphX EP after CUDA and before legacy ROCM.
Bundled FastDenoise passed native MIGraphX acceptance; demosaic is guarded separately.

The RGB MLProgram auto-selection is constrained to the bundled model filename.
The separately selectable experimental RGB model is not silently given the same
policy, because its graph has not received the same performance validation.

## Explicit controls

- `RAWALCHEMY_COREML_DENOISE=auto|cpu|mlprogram` (default `auto`). `mlprogram`
  requests the same measured options as auto, but does not override minimum
  capability checks or force CoreML on non-Apple systems. This applies to bundled
  RGB FastDenoise, not the packed RAW CANS denoiser.
- `RAWALCHEMY_COREML_GRADE=auto|cpu|coreml` (default `auto`). `coreml` permits
  explicit comparison on the affected Apple configuration. It does not force
  CoreML in place of a selected CUDA/DirectML/ROCm provider.
- Invalid Apple override values fail closed to CPU and emit a warning.
- `RAWALCHEMY_COREML_DEMOSAIC=auto|cpu|mlprogram` (default `auto`). Auto and
  CPU select by stage: auto enables repaired X-Trans on the measured runtime;
  cpu always selects CPU. RCD auto stays CPU. MLProgram uses static shapes,
  ALL compute units and full-precision GPU accumulation. X-Trans uses the
  prebuilt precision model, leaving the original model for CPU/CUDA/DirectML.
- `RAWALCHEMY_MIGRAPHX_DEMOSAIC=auto|cpu|gpu` (default `auto`). Auto and cpu
  select repaired X-Trans and fixed-tile RCD on the validated Linux x86-64 ORT 1.23.2 / ROCm 7.2.0 runtime;
  cpu always selects CPU. RCD uses the original CPU model at other tile sizes. gpu permits explicit
  diagnostics elsewhere. All MIGraphX demosaic requires child isolation.
  It does not disable MIGraphX FastDenoise.
- `RAW_ALCHEMY_CPU_ONLY=1` selects CPU for every stage on all three platforms.
- `RAW_ALCHEMY_DISABLE_CUDA_PRELOAD=1` disables optional CUDA library preloading.
  Linux CUDA wheels use ORT preload_dlls(cuda=True, cudnn=True, msvc=False,
  directory="") in the inference child. DirectML, MIGraphX, legacy ROCm and
  CPU-only ORT builds do not preload CUDA automatically.

Changing these live controls invalidates affected in-memory session identities.
To replace the contents of a model that is already loaded, call its existing
`clear_session()` API before the next use. A process restart explicitly retries a
previously failed accelerator configuration. Model or provider-option generation
changes also allow a retry; merely clearing a session does not trigger the same
known-failing accelerator repeatedly.

## Session safety, identity and diagnostics

`onnx/session_policy.py` supplies:

- `stage_providers(providers, variant)` — stage-specific selection without
  mutating caller-supplied provider options;
- `configuration_token(variant)` — cheap in-memory generation including policy,
  platform, ORT, provider availability, live controls and thread/memory settings;
- `provider_identity(providers)` — stable ordered provider identity including all
  options;
- `create_session(ort, model_path, options_factory, providers, *, variant)` —
  constructor and inference exception recovery to a freshly configured CPU
  session. Dimension overrides and thread limits are retained by the factory.

Compilation cache names include the model plus external-weight contents,
platform/ORT/policy identity and provider options. Different formats, compute
units or shape variants cannot reuse one compiled model accidentally. An
unavailable/unwritable cache disables disk caching, not MLProgram selection.

On accelerator constructor failure, session creation retries once on CPU. On
accelerator inference failure, the wrapper installs a CPU session and reruns the
same feed once; subsequent calls use that replacement. CPU failures propagate
rather than being hidden by an infinite retry loop. A process-local failure
circuit prevents repeat initialization of the same rejected configuration,
including cases where ORT silently registered CPU only. Wrapper recovery is
serialized per session, not globally across unrelated stages.

Logs distinguish requested providers, the attempted provider list (which can be
CPU because the failure circuit is open), actually registered providers, and
session initialization time. Post-denoise logs query the current session so a
runtime CPU fallback is not still advertised as the original accelerator.
`get_provider_info()` retains legacy fields but explicitly labels placement as
unknown. ORT profiling is required to attribute nodes/partitions to an EP.

## Native session isolation

Production ONNX sessions (CoreML, CUDA, DirectML, ROCm and CPU) now use spawned
processes by default; RAW decode has its own bounded lifetime. CPU retry retains
this boundary. Cancellation and memory-budget failures do not trigger fallback.
See [runtime ownership](runtime-architecture.md) for controls, cooperative scheduling,
memory accounting, IPC constraints, and the operations outside this boundary.

## Regression and native verification

Portable unit checks (mocked hardware is not native GPU validation):

```sh
.venv/bin/python -m pytest -q tests/test_backend_session_policy.py \
  tests/test_coreml_cache.py tests/test_rgb_denoiser.py tests/test_grade_fused.py \
  tests/test_demosaic_fallback.py tests/test_demosaic_coreml.py
```

Opt-in **real-runtime** FastDenoise profiling with the installed model/provider:

```sh
RAWALCHEMY_TEST_BACKEND_NATIVE=1 .venv/bin/python -m pytest -s -q \
  tests/test_backend_native.py
```

For Windows PowerShell, set `$env:RAWALCHEMY_TEST_BACKEND_NATIVE = "1"` and use
`.venv\Scripts\python.exe -m pytest -s -q tests/test_backend_native.py`.

The native check prints OS/architecture, ORT, available/selected/registered EPs,
actual profiled node-event counts, CPU reference timing, repeated candidate
inference timings and numerical differences. It fails if the selected EP never
executes a profiled event, even if registration claims it is available. It does
not enforce a speed threshold under uncontrolled load or replace visual quality
assessment. Linux CPU runs validate native Linux CPU only; actual Linux GPU and
Windows DirectML claims require corresponding hardware/runtime runs.
## September 6 GPU precision and acceptance follow-up

See [the measured report](gpu-resolution-2026-09-06.md). The X-Trans CoreML
variant evaluates 16 branch-sensitive divisions by three in float64 on CPU,
then returns to float32. It does not loosen the 3e-6 absolute/relative gate or
change coefficients/comparisons. The generator verifies the original graph SHA.
ALL scheduling passed real-frame precision and GPU compute-plan diagnostics;
restricting to CPUAndGPU was much slower on this Mac.

Full-frame production-wrapper acceptance (requires actual RAW files and GPU):

```sh
RAWALCHEMY_TEST_DEMOSAIC_NATIVE=1 \
RAWALCHEMY_TEST_REQUIRED_EP=CUDAExecutionProvider \
RAWALCHEMY_TEST_BAYER_RAW=/path/to/sample.NEF \
RAWALCHEMY_TEST_XTRANS_RAW=/path/to/sample.RAF \
PYTHONPATH=src python -m pytest -s -q tests/test_demosaic_backend_native.py
```

For CoreML use CoreMLExecutionProvider and select -k xtrans. For FastDenoise,
RAWALCHEMY_TEST_REQUIRED_EP also prevents accepting the wrong accelerator.
A failed or silently replaced child cannot be counted as a GPU pass. Both
Linux GPU machines tested here run Ubuntu 24.04 under WSL2 with physical GPUs;
this is not bare-metal Linux or whole-GUI GPU acceptance.

The existing Linux dependency selects the CUDA runtime. AMD validation uses
AMD's version-matched onnxruntime-migraphx wheel in a separate environment;
installing several packages that own the onnxruntime namespace together is not
a validated setup. CUDA and ROCm user-space libraries were isolated to the test
environments; no host graphics drivers or firewall settings were changed.

The MIGraphX X-Trans variant evaluates 61 float divisions via double and replaces
28 powers of two with explicit multiplication. Its owned child disables algebra
reordering and uses -fno-fast-math -ffp-contract=off, a bounded 64 MiB compiler
stack, and a 180-second default compile deadline (the existing timeout override
still takes precedence). These are child settings, not global GUI/compiler changes.

Compiled MXR files live under XDG_CACHE_HOME/RawAlchemy/migraphx, or the usual
~/.cache fallback. The namespace hashes model bytes, ORT/platform, dimensions,
providers and compiler environment; the EP adds its MIGraphX/GPU architecture
identity. Unwritable cache directories disable disk reuse. Changing model bytes
invalidates compiled and image-stage cache identities. All three platforms ship
all three small backend model assets, while CPU/CUDA/DirectML use the original graph.


## AMD Bayer/RCD integration

`rcd_demosaic_migraphx_1536.onnx` replaces three Tile mask expansions with
two-axis Gather indexing. It preserves the original arithmetic and is built
reproducibly by `tools/build_migraphx_rcd.py`, which verifies the source SHA-256.
Its input shape and selection guard both require the production tile size 1536.
CPU, CUDA and DirectML retain the original model; failed AMD compilation also
constructs the original CPU model. RCD shares the isolated child precision,
180-second compile deadline, cancellation and MXR cache controls described above.
Its asset participates in image-stage content identity. No graph conversion runs
on application startup and no additional runtime package is required.

Additional native acceptance covers all four CFA phases, nonidentity white
balance/matrix, padding, black and near-flat pixels, tile seams/highlights,
a real NEF using its camera WB/matrix, and a newly spawned child reusing the
compiled cache. It requires actual AMD hardware and is skipped in CPU CI:

```sh
RAWALCHEMY_TEST_RCD_MIGRAPHX=1 \
RAWALCHEMY_TEST_BAYER_RAW=/path/to/sample.NEF \
PYTHONPATH=src python -m pytest -s -q tests/test_rcd_migraphx_native.py
```

See [AMD Bayer integration acceptance](amd-bayer-integration-2026-09-06.md)
for measurements, platform checks, artifacts and remaining validation limits.
