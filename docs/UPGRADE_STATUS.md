# Upgrade Status for `studio-v0.6.0-pre2`

This prerelease is the active validation build for `UPGRADE_PLAN.md`, not final
acceptance for the whole plan. The main implementation work is present, but
several acceptance items still require target hardware or manual GUI
verification.

## Automated Verification

- `python -m compileall src tests`
- `python -m pytest tests\ -x -q -p no:cacheprovider`: 61 passed
- `uv run ruff check`: passed
- `python -m raw_alchemy.cli --help`: passed
- `python -m raw_alchemy.cli --help` includes `hdr-heif`: passed
- CLI supports optional `--log-space None`, `--no-lens-correct`,
  legacy `--lens-correct false`, and `--format dng`: passed
- CLI import smoke with `PySide6` blocked: passed
- PySide boundary: imports are limited to `ui/`, `workers/`, and `main.py`
- Taichi boundary: direct imports are limited to `backend.py` and `math_ops.py`
- T1.3 CLI entrypoint and batch orchestrator submission tests: passed
- T1.3 real RAW CLI single-file smoke: `DSCF0023.RAF` to JPG, passed
- T1.3 real RAW CLI batch-directory smoke: `DSCF0023.RAF` and `_DSC7822.ARW`
  to JPG, passed
- T1.3 GUI cached/full single-image export requests route through the
  `ImageProcessor` worker queue: passed
- T1.2/T5.3 GUI controller workflow smoke covers cached/full single export,
  batch export with per-image params and LUT override, plus crop/perspective
  parameter persistence: passed
- Real RAW CLI format smoke from `C:\Users\shenmintao\Downloads\Photo`: DNG,
  CR3, RW2, and X3F to JPG, passed
- T2.1 sidecar restart restore is covered by UI glue tests for folder scan,
  gallery selection, per-image params, and marked state: passed
- T3.1 proxy/full timing on real `DSC03687.ARW` (9568x6376): proxy result
  775 ms, full compute estimate 6051 ms, ratio 0.128 (< 0.25)
- T3.1 high-frequency preview update scheduling keeps latest-wins semantics
  and cancels stale full-refine requests during slider-style bursts: passed
- T6.1 real RAW `hdr-heif` smoke with `DSCF0023.RAF`: readback is 10-bit
  BT.2020/PQ HEIF (`nclx` primaries 9, transfer 16, matrix 9)
- T6.1 `hdr_output` ignores Log/LUT ops and always routes to `pq_out`: passed
- T6.1 real RAW CLI `hdr-heif` smoke without `--log-space`: `DSCF0023.RAF`
  to 10-bit BT.2020/PQ HEIF, passed
- T6.2 real CANS RAW V2 denoise smoke with Bayer `_DSC7822.ARW`: passed
- T6.2 real CANS RAW V2 denoise smoke with X-Trans `DSCF0023.RAF`: passed
- GUI offscreen constructor smoke: passed
- PyInstaller onedir build: passed
- Packaged `RawAlchemy.exe` launch smoke: stayed alive for 12 seconds

## Release Asset

- Manual Windows package: `RawAlchemy-0.6.0-pre2-windows-x64.zip`
- Built from tag `studio-v0.6.0-pre2`; includes runtime fixes through commit
  `b28ea9f`.
- Automated release workflow also publishes platform artifacts from the tag.
- The manual archive contains `RawAlchemy.exe` plus the `_internal` runtime directory.
- SHA256: `EEE868D66609976C8E3CDC8DE9C788AA417B47CDC00355A6BE21A22C8A8AC9E5`

## HDR Scope

- PQ HEIF is implemented as `hdr-heif`.
- Gain-map JPEG research is documented in `docs/HDR_GAINMAP_RESEARCH.md`.
- Ordinary `jpg` export remains SDR-only.

## Incomplete Or Unproven Acceptance Items

- T1.2/T5.3: GUI controller flow is automated, but live real GUI interaction and visual inspection are not manually accepted yet: image switching, sliders, crop, perspective, single export, and batch export through the running app.
- T3.1: slider-drag smoothness still needs manual GUI acceptance.
- T6.1: PQ HEIF local metadata readback passed, but HDR recognition in Windows Photos/Chrome is not manually verified.
- T6.1: ISO 21496-1 gain-map JPEG output is researched but not implemented.
