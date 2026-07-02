# Upgrade Status

## `studio-v0.6.0-pre3` — Phase 7 交互性能与内存治理

pre3 在 pre2 之上实施了完整的 Phase 7（任务全文见 `docs/PHASE7_PLAN.md`）：

- T7.1 前缀缓存跨请求生效修复（此前 GUI 路径命中率恒 0，每次交互整链重跑）
- T7.2 GPU 驻留重做：前缀缓存持 GPU 缓冲、op 间零回读、NdarrayPool 池化、
  sharpen/demosaic 缓冲纳管（单次交互 host↔device 传输 GB 级 → <100MB 量级）
- T7.3 在飞任务 op 粒度协作式取消 + 邻居 preload 减负（不再预计算全尺寸镜头校正）
- T7.4 proxy/full 双执行器缓存分离 + zoom 与色彩管线解耦 + uint8 多槽输出缓存
- T7.5 zoom>100% ROI 渲染（突破 12MP 整图 cap：100% 视图逐像素清晰，处理规模
  从整图降到 2-4MP 可见区）
- T7.6 主机内存治理：缓存配额绝对上限（默认 6GB，设置页可调）+ 字段级驱逐 + 可观测日志
- T7.7 缩略图缩放解码（24MP 内嵌 JPEG 实测 87.5ms→7.6ms）+ 可视区优先 + 切文件夹不阻塞 UI
- T7.8 缩略图磁盘缓存（mtime 键控、LRU 容量治理、设置开关）

随后的三视角对抗复查确认 13 项实施缺陷并全部修复（3 个 fix commit，每项带
"修复前确定性失败"的复现测试）：CachedImage 跨线程锁与记账、输出缓存命中
路径的 last_applied_ev/导出状态同步（Auto 曝光导出烘焙错 EV）、preload 中止
后 proxy 永久失效回填、关闭路径线程治理、缩略图 prune 记账、示波器 ROI 统计
语义守卫、ROI 源 GPU 驻留免重传。6 项 major 逐条独立复核确认真修复。

### pre3 Automated Verification

- `pytest tests/ -q`（offscreen, taichi CPU）：**158 passed**（pre2 基线 62 + 新增 96）
- `python -m compileall src tests`: passed

### pre3 Pending Manual Acceptance (Windows target hardware)

- 缩放/滑杆体感：zoom 0.5↔2.0 往返除首次外无整链重跑（日志核实）、滑杆停 1s
  触发 refine 后再动滑杆响应 <300ms（此前最长 ~6s 冻结）
- 45MP 样张 100% 查看：首次 <1s、调参 <300ms、pan 边距内不掉帧、逐像素清晰
- 200 张目录：可视区首屏缩略图 <3s、切文件夹 UI 冻结 <50ms、二次打开 <1s 出齐
- 连续浏览 30 张 45MP RAW：进程 RSS 稳定在缓存上限+工作集内（此前 12-16GB）、
  VRAM 无增长趋势
- 61MP full refine 端到端 <2s（此前 ~6s）
- 设置页新增项生效：缓存上限 SpinBox、缩略图磁盘缓存开关/清空

---

# Upgrade Status for `studio-v0.6.0-pre2`

This prerelease is the active validation build for `UPGRADE_PLAN.md`, not final
acceptance for the whole plan. The main implementation work is present, but
several acceptance items still require target hardware or manual GUI
verification.

## Automated Verification

- `python -m compileall src tests`
- `python -m pytest tests\ -x -q -p no:cacheprovider`: 62 passed
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
- CANS / AI demosaic-denoise remains disabled in the GUI even if a sidecar
  requests it: passed
- GUI offscreen constructor smoke: passed
- PyInstaller onedir build: passed
- Packaged `RawAlchemy.exe` launch smoke: stayed alive for 12 seconds

## Release Assets

- Built by GitHub Actions from tag `studio-v0.6.0-pre2`; includes runtime fixes
  through the current pre2 tag commit.
- Published assets:
  - `RawAlchemyStudio-studio-v0.6.0-pre2-windows-x64-portable.zip`
  - `RawAlchemyStudio-studio-v0.6.0-pre2-linux-x64-portable.tar.gz`
  - `RawAlchemyStudio-studio-v0.6.0-pre2-macos.dmg`
- Windows, Linux, and macOS builds use PyInstaller onedir packaging to avoid
  onefile self-extraction during startup. The Windows artifact is a portable
  zip, Linux is a portable tarball, and macOS remains a DMG containing
  `RawAlchemy.app`.
- macOS Gatekeeper note for unsigned builds: after dragging the app to
  `/Applications`, run
  `xattr -dr com.apple.quarantine /Applications/RawAlchemy.app` if needed.

## HDR Scope

- PQ HEIF is implemented as `hdr-heif`.
- Gain-map JPEG research is documented in `docs/HDR_GAINMAP_RESEARCH.md`.
- Ordinary `jpg` export remains SDR-only.

## Incomplete Or Unproven Acceptance Items

- T1.2/T5.3: GUI controller flow is automated, but live real GUI interaction and visual inspection are not manually accepted yet: image switching, sliders, crop, perspective, single export, and batch export through the running app.
- T3.1: slider-drag smoothness still needs manual GUI acceptance.
- T6.1: PQ HEIF local metadata readback passed, but HDR recognition in Windows Photos/Chrome is not manually verified.
- T6.1: ISO 21496-1 gain-map JPEG output is researched but not implemented.
