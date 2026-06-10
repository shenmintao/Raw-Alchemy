# Upgrade Status for `studio-v0.6.0-pre2`

This prerelease is a validation build for `UPGRADE_PLAN.md`, not final
acceptance for the whole plan. The main implementation work is present, but
several acceptance items still require real samples, target hardware, or manual
GUI verification.

## Automated Verification

- `python -m compileall src tests`
- `python -m pytest tests\ -x -q -p no:cacheprovider`: 47 passed
- `uv run ruff check`: passed
- `python -m raw_alchemy.cli --help`: passed
- CLI import smoke with `PySide6` blocked: passed
- PySide boundary: imports are limited to `ui/`, `workers/`, and `main.py`
- Taichi boundary: direct imports are limited to `backend.py` and `math_ops.py`
- GUI offscreen constructor smoke: passed
- PyInstaller onedir build: passed
- Packaged `RawAlchemy.exe` launch smoke: stayed alive for 12 seconds

## Release Asset

- Manual Windows package: `RawAlchemy-0.6.0-pre2-windows-x64.zip`
- Automated release workflow also publishes platform artifacts from the tag.
- The manual archive contains `RawAlchemy.exe` plus the `_internal` runtime directory.
- SHA256: `4CDA7E347A5E51C1BE91C572C20E3D33DCD3D2161C8CEE9CF35D5F62187B997A`

## Incomplete Or Unproven Acceptance Items

- T1.2/T5.3: full real GUI workflow is not manually accepted yet: image switching, sliders, crop, perspective, single export, and batch export.
- T1.3: CLI single-file and batch-directory smoke with real RAW files is not proven in this environment.
- T2.1: sidecar restore after app restart still needs manual verification.
- T3.1: 45MP proxy/full preview timing and the `< 1/4` proxy-path target are not measured on target hardware.
- T6.1: PQ HEIF is implemented and unit-tested, but HDR recognition in Windows Photos/Chrome is not manually verified.
- T6.1: ISO 21496-1 gain-map JPEG output is not implemented.
- T6.2: denoise toggle regression is covered synthetically, but real Bayer and X-Trans model/sample runs are not proven.
