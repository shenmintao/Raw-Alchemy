# Upgrade Status

This prerelease build implements the `UPGRADE_PLAN.md` work through Phase 6.

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

- Windows package: `RawAlchemy-0.6.0-pre.1-windows-x64.zip`
- The archive contains `RawAlchemy.exe` plus the `_internal` runtime directory.
- SHA256: `4CDA7E347A5E51C1BE91C572C20E3D33DCD3D2161C8CEE9CF35D5F62187B997A`

## Manual Acceptance Still Needed

- Real RAW folder GUI workflow: image switching, sliders, crop, perspective, single export, batch export.
- 45MP proxy/full preview performance comparison on target hardware.
- HDR HEIF recognition in Windows Photos and Chrome.
- AI denoise with real Bayer and X-Trans model/sample combinations.
