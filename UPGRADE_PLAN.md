# Raw Alchemy 升级计划（Agent 执行版）

> 本文档是给 AI Agent 的施工图。按 Phase 顺序执行；同一 Phase 内的任务可并行，
> 跨 Phase 有依赖（后面标注）。每个任务都有「验收标准」，不满足不得标记完成。

---

## 执行须知（每个 Agent 开工前必读）

- **环境**：Windows 11，Python 用 `.venv/Scripts/python`（不要用系统 Python）。
  运行 GUI：`.venv/Scripts/python -m raw_alchemy.main`；CLI：`.venv/Scripts/python -m raw_alchemy.cli`。
- **分支**：从 `studio` 切出 `feat/<task-id>` 分支干活，完成后自行验证再合回 `studio`。
- **不可破坏的契约**：
  1. CLI 参数接口（README 中列出的选项）保持兼容；
  2. 视觉输出不允许悄悄改变——凡是会改变像素结果的任务都已显式标注「允许改变输出」，
     其余任务必须通过回归测试证明输出不变；
  3. `pipeline/processor.py` 的 Taichi 初始化必须留在 worker 线程内（CUDA context 是
     thread-local，见 `math_ops.py:35-37` 注释）。
- **测试**：`.venv/Scripts/python -m pytest tests/ -x -q`。Phase 0 之后所有任务完成前必须全绿。
- **忽略目录**：`.venv/`、`dist/`、`build/`、`__pycache__/` 一律不读不改。

---

## Phase 0 — 安全网 + 速修（无依赖，先做）

### T0.1 建立测试基础设施与色彩科学 golden 测试

**目标**：在重构前锁住当前数学行为。
**新建**：`tests/`、`tests/conftest.py`、`tests/test_color_math.py`、`tests/test_pipeline_ops.py`
**要点**：
- pytest + 合成数据，**不依赖真实 RAW 文件**（解码阶段用合成 Bayer/线性图代替）：
  - 对 `config.LOG_TO_WORKING_SPACE` 中每个 Log 空间：用 colour-science 的 CPU 参考实现
    （`colour.cctf_encoding`）对比 `math_ops.log_encode_gpu` 的 GPU 结果，容差 `atol=1e-4`；
  - `colorspace_matrices.cam_to_prophoto_matrix`：对 2-3 个已知 xyz_to_cam 矩阵存 golden 值；
  - `core.subtract_black_level`、`core.fix_hot_pixels`、`metering` 各策略：合成输入 + golden 输出；
  - `math_ops` 的 `apply_matrix_inplace / apply_saturation_contrast_inplace /
    apply_highlight_shadow_inplace / apply_lut_inplace`：合成 64x64 float32 图 + golden npy
    （存 `tests/golden/`，用 `np.savez_compressed`）。
- Taichi 测试用 `init_taichi(arch=ti.cpu)` 跑 CPU 后端，保证 CI 无 GPU 也能跑。
  注意：`init_taichi` 是进程级一次性的，测试统一在 conftest 里初始化为 CPU。
**验收**：`pytest tests/ -q` 全绿；测试不依赖网络与真实 RAW；golden 文件 < 5MB。

### T0.2 修复：透视校正导出丢失（用户可见 bug）

**目标**：导出结果包含透视校正，与预览一致。
**涉及**：`src/raw_alchemy/core.py`（`process_image`、`export_from_cache`）、
`src/raw_alchemy/utils.py`、`src/raw_alchemy/ui/main_window.py`（`run_export`）、
`src/raw_alchemy/orchestrator.py`
**要点**：
- 给 `process_image` 和 `export_from_cache` 增加 `perspective_corners` 参数，
  在几何变换之后、裁切之前应用（与 `pipeline/processor.py:694-712` 的顺序一致）；
- CPU 实现可走 `cv2.warpPerspective`，矩阵复用 `math_ops.compute_perspective_matrix`；
- `run_export` 把 `p.get('perspective_corners')` 传下去；`orchestrator.process_path` 同步透传。
**验收**：新增测试——合成图设置非默认 corners，断言导出路径输出与
`perspective_warp_kernel` 预览路径结果一致（容差内）；默认 corners 时输出与修改前
逐像素一致（回归）。

### T0.3 修复：自动测光导出依赖滑块回写

**目标**：导出时曝光增益与预览实际生效值严格一致，不再依赖 UI 滑块同步。
**涉及**：`src/raw_alchemy/ui/main_window.py`（`run_export`、`export_current`、`batch_export_next`）
**要点**：
- 单图导出（走 `export_from_cache` 的 fast path）：直接传
  `processor.last_applied_ev`（预览实际应用的 EV），不再传滑块值；
- 批量导出（走 `process_path` 全量路径）：保持 auto 语义——`exposure_mode` 为 Auto 时
  传 `exposure=None`，让 `apply_auto_exposure` 真正执行；为 Manual 时传滑块值。
**验收**：手动模式导出值不变（回归）；代码中不再出现「auto 模式下把滑块值当手动曝光传入」的路径。

### T0.4 依赖治理

**目标**：构建可复现，运行时依赖瘦身。
**涉及**：`pyproject.toml`
**要点**：
- `pyinstaller` 移到 `[project.optional-dependencies] build`；
- 检查 `matplotlib` 实际用途（grep src/），若仅直方图/波形图已被自绘 widget 取代则移除；
- 生成锁文件：`uv pip compile pyproject.toml -o requirements.lock`（或 pip-tools），
  锁定三个 git 依赖的 commit hash（`rawpy`、`rawspeedpy`、`colour-science` 用
  `@<commit-sha>` 固定）；
- CLI 默认 `saturation=1.25 / contrast=1.1`（`orchestrator.py:22-23`）改为 `1.0 / 1.0`，
  README 同步说明。【允许改变输出：仅 CLI 默认值，GUI 不受影响】
**验收**：`.venv/Scripts/python -m raw_alchemy.cli --help` 正常；锁文件入库；
干净 venv 下 `pip install -r requirements.lock` 可装。

---

## Phase 1 — 统一管线（核心重构；依赖 Phase 0 全部完成）

> 这是整个计划的地基。目标：预览与导出执行**同一份操作序列定义**，
> 消灭 `core.process_image` / `export_from_cache` / `pipeline/processor.py` 三份重复实现。

### T1.1 定义操作（Op）抽象与统一执行器

**新建**：`src/raw_alchemy/pipeline/ops.py`、`src/raw_alchemy/pipeline/executor.py`
**设计**（实施 Agent 可在此基础上细化，但不得偏离以下原则）：
```python
@dataclass(frozen=True)
class Op:
    name: str                 # 'lens_correct', 'geometry', 'perspective', 'crop',
                              # 'exposure', 'white_balance', 'highlight_shadow',
                              # 'sat_contrast', 'log_transform', 'lut', 'srgb_out', 'sharpen'
    params: tuple             # 可哈希的参数元组（缓存键由此自动派生）

def build_op_list(params: ProcessorParams) -> list[Op]:
    """唯一的管线定义来源。预览、导出、CLI 都调它。
    顺序以当前 GUI 预览路径为准（processor.py 的顺序）：
    denoise -> lens -> geometry -> perspective -> crop ->
    exposure -> wb -> highlight_shadow -> sat_contrast ->
    log_transform/lut 或 srgb_out -> sharpen
    """
```
- `executor.py` 提供两个执行器，吃同一份 op list：
  - `PreviewExecutor`：持有 GPU 缓冲与**逐级缓存**——缓存键 = 前缀 op 序列的哈希
    （`hash(tuple(ops[:i]))`），替代现在手写的 `cached_lens_key / last_geo_crop_key /
    last_grading_key / _make_output_key` 等 15 个字段；
  - `ExportExecutor`：无缓存直通，全分辨率，复用同一批 kernel 函数。
- kernel 本体不动（继续用 `math_ops.py`），本任务只重组调用结构。
**验收**：`tests/test_unified_pipeline.py`——同一合成线性图 + 同一参数集，
`PreviewExecutor`（不降采样）与 `ExportExecutor` 输出逐像素一致（atol=1e-5）；
对至少 8 组随机参数组合成立（含 log+lut、perspective、crop 非默认）。

### T1.2 预览路径迁移到统一执行器

**涉及**：`src/raw_alchemy/pipeline/processor.py`
**要点**：
- `_do_process` 改为：`ops = build_op_list(params)` → `PreviewExecutor.run(ops)`；
- 保留现有行为：latest-wins 请求合并、preload 队列、CPU LRU 缓存（`cache_manager.py` 不动）、
  denoise/sharpen 的跨图缓存语义、`result_ready` 信号签名不变（UI 不用改）；
- 删除被取代的手写缓存字段。
**验收**：Phase 0 的全部回归测试通过；手动验收——GUI 打开样张文件夹，
切换图片/拖动滑块/裁切/透视全部正常，日志中 pipeline 耗时与重构前同量级（±30%）。

### T1.3 导出与 CLI 迁移到统一执行器

**涉及**：`src/raw_alchemy/core.py`、`src/raw_alchemy/orchestrator.py`、
`src/raw_alchemy/ui/main_window.py`（`run_export`）
**要点**：
- `process_image` / `export_from_cache` 重写为薄壳：解码（或取缓存）→
  `build_op_list` → `ExportExecutor` → `save_image`。两函数签名保持兼容；
- 多进程批处理（`ProcessPoolExecutor`）路径保留——子进程内各自 `init_taichi`；
- 解决导出线程 Taichi 隐患：GUI 单图导出不再于独立 QThread 直接调 Taichi kernel，
  改为把导出任务投递给 processor worker 线程执行（在其 run loop 中增加 export 请求类型），
  或为 ExportExecutor 显式走 CPU 后端——二选一，实施时以改动小者为准。
**验收**：T1.1 的一致性测试改为跑真实入口函数后仍全绿；CLI 单文件 + 批量目录冒烟通过；
GUI 单图导出（fast path）与批量导出冒烟通过。

### T1.4 清理与文档

**要点**：删除 dead code（`export_from_cache` 中重复的 LUT/log 代码块等）；
在 `docs/PIPELINE.md` 写出 op 顺序、每个 op 的数学定义与所在色彩域（这是「管线即产品」
理念的第一步）。
**验收**：`grep -rn "apply_matrix_RGB_to_RGB\|cctf_encoding" src/` 的调用点只剩 ops 层；
文档与代码中 op 顺序一致。

---

## Phase 2 — 编辑持久化（依赖 Phase 1；T2.1 可与 Phase 1 并行）

### T2.1 Sidecar 读写

**新建**：`src/raw_alchemy/sidecar.py`；**涉及**：`src/raw_alchemy/ui/main_window.py`
**要点**：
- 每张图旁写 `<原文件名>.rwa.json`：`{"version": 1, "params": {...}, "marked": bool}`；
  params 即 `get_params()` 的 dict（全部为 JSON 原生类型，corners/crop 转 list）；
- 写入时机：参数变更后 debounce 2s + 切图时 + 退出时；读取时机：打开文件夹扫描、切图加载；
- `file_params_cache`（`main_window.py:998`）改为以 sidecar 为后备存储；
- schema 带 `version` 字段，预留迁移钩子；
- 设置页加开关「写入 sidecar」（默认开），关闭则维持纯内存行为。
**验收**：测试覆盖 round-trip（写→读→params 相等）与坏文件容错（损坏 JSON 不崩溃、回退默认）；
手动验收——调参后重启应用，每图参数与标记恢复。

---

## Phase 3 — 交互性能：代理预览（依赖 Phase 1）

### T3.1 代理分辨率交互 + 空闲全清晰

**涉及**：`pipeline/executor.py`、`pipeline/processor.py`、`ui/main_window.py`
**要点**：
- 解码完成后生成约 2-4MP 的代理图（`cv2.resize INTER_AREA`），与全尺寸一并缓存；
- 滑块拖动期间（参数高频变化）`PreviewExecutor` 在代理上跑；参数稳定 300ms 后
  自动用全尺寸重跑一次并刷新（progressive refinement）；
- 放大到 >100% 查看时直接走全尺寸（保持现在 `preview_zoom` 语义）；
- 直方图/波形图数据源标注清楚来自代理还是全尺寸。
**验收**：日志记录两种路径耗时；代理路径在 45MP 样张上 grading 阶段耗时
应 < 全尺寸的 1/4；滑块连续拖动无明显卡顿（手动验收）；静止 1s 后图像为全尺寸结果。

---

## Phase 4 — 色彩科学修正（依赖 Phase 0 测试网；与 Phase 2/3 可并行）

> 本 Phase 所有任务【允许改变输出】，但必须：旧行为保留可选项或在 sidecar
> version 中可追溯，且 golden 测试同步更新并在 commit message 里说明数值变化原因。

### T4.1 白平衡改为 CCT + 色适应变换

**涉及**：`math_ops.py`、`pipeline/ops.py`、`ui/widgets/inspector_panel.py`
**要点**：
- 现状是裸通道增益（`1 ± temp*0.005`，`processor.py:790-793`）；
  改为：temp/tint 映射到目标白点（CCT + Duv），用 colour-science 算 Bradford/CAT16
  适应矩阵（CPU 端算 3x3），GPU 端仍走 `apply_matrix_inplace`——kernel 无需新增；
- 滑块范围与中点语义保持（0 = 不变）；矩阵计算加 lru_cache。
**验收**：temp=tint=0 时输出与改前逐像素一致；新增测试——已知光源对
（D65→A）的适应矩阵与 colour 参考一致。

### T4.2 工作空间单点化 + 白点锚定测试（不切换默认空间）

> 决策记录：**保留 ProPhoto 为默认工作空间，ACEScg 不在本阶段启用。**
> 原因：(a) CANS RAW V2 降噪模型直出 ProPhoto Linear（ProPhoto 烤进训练，
> 见 `onnx/denoiser.py:2-6`），切换空间需补固定矩阵且 ProPhoto 蓝原色在 AP1 外，
> 高饱和色有负值/裁切代价；(b) 非 Bayer/非 X-Trans 传感器（Foveon、GMCY 等）
> 走 LibRaw `postprocess(output_color=ProPhoto)` 兜底（`processor.py:298-311`、
> `core.py:196-210`），LibRaw 的 ProPhoto 输出与解析矩阵路径同用 dcraw 行归一化
> 白点锚定，二者色彩一致；LibRaw 的 ACES 输出是 AP0+ACES 白点（≈D60），
> 直接混用会白点漂移（用户已实测踩坑）；(c) 计划中的 raw-to-raw 降噪模型
> （T6.2 方向）落地后降噪与工作空间天然解耦，届时再评估 ACEScg 成本更低。
> 白点漂移本身不构成主路径障碍：`colorspace_matrices.py:121-125` 的行归一化已把
> 「WB 后中性色 → 工作空间 (1,1,1)」锚死，目标空间只是 `cam_to_working_matrix`
> 的一个参数；但兜底路径若换空间需在 LibRaw ProPhoto 输出后追加固定转换矩阵。

**涉及**：`pipeline/ops.py`、`config.py`、`colorspace_matrices.py`、`tests/`
**要点**：
- 内部工作空间从散落各处的硬编码 `'ProPhoto RGB'` 收敛为 `config.WORKING_SPACE`
  单点定义（含对应的 `WORKING_TO_XYZ_D65` 矩阵常量），op 层全部引用它；
- 新增**白点锚定单元测试**：对 3 个不同相机的 xyz_to_cam 矩阵，构造 WB 后的
  中性输入（R=G=B），经 `cam_to_working_matrix` 变换后断言输出 R=G=B（atol=1e-6）。
  该测试是未来任何工作空间切换的硬性门禁；
- 在 `colorspace_matrices.py` 预置 `ACESCG_TO_XYZ_D65` 常量（AP1→XYZ_D60 + CAT），
  并让上述锚定测试同时覆盖它——证明机制可用，但**不接入 UI、不改默认**。
**验收**：全部回归测试不变（默认行为零变化）；锚定测试对 ProPhoto 与 ACEScg
两个常量均通过；`grep -rn "'ProPhoto RGB'" src/` 仅剩 `config.py` 一处定义。

---

## Phase 5 — 工程化与解耦（持续进行，无硬依赖）

### T5.1 引擎与 GUI 解耦
`pipeline/`、`core.py`、`math_ops.py`、`metering.py` 等处理模块中移除一切
`PySide6` import（`processor.py` 的 QThread/Signal 拆为：纯引擎类 + 薄 Qt 适配层）。
**验收**：`grep -rn "PySide6" src/raw_alchemy/ --include="*.py"` 仅出现在 `ui/`、
`main.py`、`workers/`；CLI 在未安装 PySide6 的环境可运行（用临时 venv 验证）。

### T5.2 GPU 后端接口隔离
**新建** `src/raw_alchemy/backend.py`：把 `math_ops` 的对外函数收口为一个显式接口
（protocol/registry），调用方不再直接 import taichi。为将来评估 wgpu/PyTorch 迁移
留出唯一切口。不做实际迁移。
**验收**：`import taichi` 仅存在于 `math_ops.py` 与 `backend.py`。

### T5.3 拆分 main_window.py（1547 行）
导出编排逻辑（`run_export`/`export_all`/`batch_export_next`）抽到
`src/raw_alchemy/ui/export_controller.py`；裁切/透视模式切换逻辑抽到
`ui/edit_modes.py`。纯移动不改行为。
**验收**：`main_window.py` < 900 行；GUI 冒烟全功能正常。

### T5.4 CI
GitHub Actions：windows + ubuntu，跑 `pytest`（Taichi CPU 后端）+ `ruff check`。
**验收**：workflow 文件入库且本地 `pytest` 全绿。

---

## Phase 6 — 趋势功能（依赖 Phase 1/2；做完前面才有资格做这些）

### T6.1 HDR 输出（优先级最高的新功能）
**目标**：输出 PQ HEIF（10bit BT.2020/PQ）与 gain map JPEG（ISO 21496-1 兼容）。
**涉及**：`file_io.py`、`pipeline/ops.py`（新增 `pq_out` op）、导出 UI
**要点**：管线本就持有线性高动态数据，在 `srgb_out` 的同位置增加 PQ 编码分支；
gain map 方案先调研 `pillow-heif`/`libultrahdr` 的现成支持再实施，调研结论写进 PR。
**验收**：输出文件在 Windows 照片查看器/Chrome 中被识别为 HDR；SDR 回退渲染正确。

### T6.2 AI 降噪（CANS RAW V2）回归
**目标**：重新启用 `onnx/denoiser.py` 路径作为统一管线的一等 op（toggle 已在
`aea7b29` 被禁用）。把「denoise 替换 demosaic」语义在 op 图中显式建模
（denoise op 存在时跳过 demosaic 分支）；显存管理保持 `clear_session` 即用即放。
**注**：若后续模型升级为 raw-to-raw（RAW 域降噪，去马赛克与色彩矩阵移回下游），
降噪将与工作空间彻底解耦——届时 T4.2 中搁置的 ACEScg 切换才值得重新评估。
**验收**：开关切换无残留状态 bug（连续切换 10 次，输出与首次一致）；
Bayer 与 X-Trans 样张均可跑通。

### T6.3 AI 辅助选片（调研 spike，1 个 Agent 会话）
评估本地小模型做清晰度/闭眼/重复组检测的可行性（onnxruntime 已在依赖中），
产出 `docs/CULLING_RESEARCH.md`：候选模型、推理耗时实测、UI 集成草案。**不写产品代码。**

---

## 给调度者（人类）的建议执行序

```
第 1 批（并行）：T0.1  T0.4
第 2 批（并行）：T0.2  T0.3        ← 依赖 T0.1 的测试网
第 3 批（串行）：T1.1 → T1.2 → T1.3 → T1.4
第 4 批（并行）：T2.1  T3.1  T4.1  T5.3
第 5 批（并行）：T4.2  T5.1  T5.2  T5.4
第 6 批（并行）：T6.1  T6.2  T6.3
```

每个任务设计为单个 Agent 会话可完成的粒度。T1.x 是整个计划的关键路径，
建议人工 review 其 PR；其余任务可信任验收标准自动推进。
