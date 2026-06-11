# Upgrade Status for `studio-v0.6.0-pre2`

This prerelease is a validation build for `UPGRADE_PLAN.md`, not final
acceptance for the whole plan. The main implementation work is present, but
several acceptance items still require real samples, target hardware, or manual
GUI verification.

## Automated Verification

- `python -m compileall src tests`
- `python -m pytest tests\ -x -q -p no:cacheprovider`: 53 passed
- `uv run ruff check`: passed
- `python -m raw_alchemy.cli --help`: passed
- `python -m raw_alchemy.cli --help` includes `hdr-heif`: passed
- CLI import smoke with `PySide6` blocked: passed
- PySide boundary: imports are limited to `ui/`, `workers/`, and `main.py`
- Taichi boundary: direct imports are limited to `backend.py` and `math_ops.py`
- T1.3 CLI entrypoint and batch orchestrator submission tests: passed
- T1.3 real RAW CLI single-file smoke: `DSCF0023.RAF` to JPG, passed
- T1.3 real RAW CLI batch-directory smoke: `DSCF0023.RAF` and `_DSC7822.ARW`
  to JPG, passed
- T1.3 GUI cached/full single-image export requests route through the
  `ImageProcessor` worker queue: passed
- T2.1 sidecar restart restore is covered by UI glue tests for folder scan,
  gallery selection, per-image params, and marked state: passed
- T6.2 real CANS RAW V2 denoise smoke with Bayer `_DSC7822.ARW`: passed
- T6.2 real CANS RAW V2 denoise smoke with X-Trans `DSCF0023.RAF`: passed
- GUI offscreen constructor smoke: passed
- PyInstaller onedir build: passed
- Packaged `RawAlchemy.exe` launch smoke: stayed alive for 12 seconds

## Release Asset

- Manual Windows package: `RawAlchemy-0.6.0-pre2-windows-x64.zip`
- Automated release workflow also publishes platform artifacts from the tag.
- The manual archive contains `RawAlchemy.exe` plus the `_internal` runtime directory.
- SHA256: `C253A53940401B1DB28F60EF98725D073557734EA346BB33CC047FFEDE2A01C8`

## HDR Scope

- PQ HEIF is implemented as `hdr-heif`.
- Gain-map JPEG research is documented in `docs/HDR_GAINMAP_RESEARCH.md`.
- Ordinary `jpg` export remains SDR-only.

## Incomplete Or Unproven Acceptance Items

- T1.2/T5.3: full real GUI workflow is not manually accepted yet: image switching, sliders, crop, perspective, single export, and batch export.
- T3.1: 45MP proxy/full preview timing and the `< 1/4` proxy-path target are not measured on target hardware.
- T6.1: PQ HEIF is implemented and unit-tested, but HDR recognition in Windows Photos/Chrome is not manually verified.
- T6.1: ISO 21496-1 gain-map JPEG output is researched but not implemented.
