# Phase 7 提案 — 交互性能与显存/内存治理（2026-07-02 诊断产出）

> 由 25-agent 对抗验证诊断生成（4 路深潜 + 逐条复核，0 条被驳倒）。
> 审阅通过后可直接并入 UPGRADE_PLAN.md。分析基线：feat/phase0-upgrade-plan @ 9c0830d。

## 诊断结论

结论先行：不需要"系统性重构"。UPGRADE_PLAN 的架构方向（统一 Op 管线 + 逐级缓存 + proxy 预览）是对的，但有一处实现走样和两大计划盲区，三个症状可全部归因到少数根因链上，追加一个 Phase 7 即可治理。

【症状2：放大缩小显著卡顿】根因链（最深 → 最表）：
① 前缀缓存在 GUI 路径从未生效——PreviewExecutor.set_source（executor.py:266-268）在基类因"同一源对象"早退后仍无条件 clear 缓存，而 _do_process 每次都传 source（image_processor.py:924），于是每次缩放/调参都从零重跑整条 op 链（proxy ~775ms，zoom>1 全幅 ~6051ms）。
② 执行器丢失 GPU 驻留——与 T1.1"持有 GPU 缓冲与逐级缓存"的计划文本直接矛盾，实现落成了 CPU numpy 缓存：每个 op 全幅 GPU→CPU 回读+拷贝，输出再 CPU 中转+整幅重上传，61MP 下一次 N-op 执行 ≈ (2N+3)×700MB 总线流量。
③ 每次 proxy 预览后 1 秒空闲必发一次 ~6s 的全幅 refine，且 worker 内在飞任务无任何取消点（latest-wins 只覆盖未开始的排队请求），新交互最长冻结 6 秒。
④ zoom>1 一律整图进管线、输出 cap 12MP 整图纹理，处理规模比可见区域大 5-20 倍，且 100% 视图实际是模糊的——需求侧无 ROI 渲染。

【症状1：批量加载大量图片卡顿】根因链：
① 缩略图提取对内嵌 JPEG（现代机身为全分辨率 24-60MP）做全尺寸解码后才缩到 300px，8 线程一次性提交全部文件，无可视区优先、无取消、无磁盘缓存——解码风暴打满 CPU/内存带宽，与 UI 线程和用户点击后的正片解码三方抢资源；切文件夹时 UI 线程还 wait() 阻塞 0.3-3s。
② 唯一的 processor 线程串行执行不可中断的邻居 preload（全尺寸解码+全分辨率 INTER_CUBIC 镜头校正，单张 3-8s），快速翻图时点击请求排在其后。
③ CPU 图缓存配额 =70%×可用内存（32GB 机器自稳定在 11-13GB），且 preload/浏览路径为每张图急切预计算全尺寸 corrected（~515MB/张），主存被吃掉大半、文件系统缓存被挤掉，后续解码越来越慢。

【症状3：怀疑 GPU 显存分配问题】诊断结论：主要不是 VRAM，是主机 RAM——上述缓存膨胀 + 前缀缓存无淘汰 + 每请求新建/销毁 700MB 级 ti.ndarray、demosaic 2.3GB 瞬时峰值、sharpen 全局缓冲 931MB 永驻造成的 RAM 换页与 PCIe 搬运，被体感误判为"显存问题"。加绝对上限 + GPU 驻留化 + 缓冲池化即可根治，无需重写显存分配。

【建议执行序】第 1 批并行（速修，立竿见影）：T7.1、T7.7、T7.6 的配额速修部分；第 2 批：T7.2、T7.3、T7.8；第 3 批：T7.4；第 4 批：T7.5。预期第 1 批完成后症状 1/2 即有数量级改善，全部完成后缩放亚 300ms、200 张文件夹二次打开亚秒。

---

### T7.1 【速修】修复前缀缓存从未跨请求生效 + 全命中路径冗余 GPU 往返

**目标**：让 T1.1 的前缀缓存在 GUI 生产路径上真正命中（当前每次 run 开头即被清空，命中率恒为 0）。
**涉及**：`src/raw_alchemy/pipeline/executor.py`（`PreviewExecutor.set_source` :266-268、`run_result` :273-300、`_BaseExecutor.set_source` :58-65）、`src/raw_alchemy/workers/image_processor.py`（`_invalidate_executor_prefix_if_needed` :860-883 加注释说明分工）
**要点**：
- 根因修复：`_BaseExecutor.set_source` 对同一源对象早退（:59-60），但子类覆写仍无条件 `_prefix_cache.clear()`，而 `_do_process`（image_processor.py:924）每次都传 `source=`。改法：基类 `set_source` 返回「是否实际更换了源」的 bool，子类仅在 True 时 clear（或子类先自查 `self._source is source`）；
- 跨源/镜头/降噪失效统一走 `_invalidate_executor_prefix_if_needed`，executor 内不再隐式清；
- 全 ops 命中免往返：以 `hash(tuple(ops))` 额外缓存 post-clip 最终 `PipelineResult`，完全命中（如仅 zoom/viewport 变化）直接返回 CPU 数组，跳过当前 upload→clip_inplace→to_numpy 的 ~2.1GB 往返（:289-300）；
- 删除 op 循环内 `buf.to_numpy().copy()` 的冗余 `.copy()`（to_numpy 已返回新 host 数组），每 op 省一次全幅 memcpy；
- 新增守护单测：同对象连续两次 `run_result`，用计数 stub op 断言第二次不重跑 kernel；换源对象则断言缓存清空（防止回归）。

**验收**：62 项既有测试全绿 + 新增守护单测通过；GUI 日志验收——zoom>1 下连续缩放（ops 不变），自第二次请求起日志显示前缀/最终级命中、无 op kernel 重跑，管线段耗时较修复前下降 ≥80%（full 源 ~6s → 亚秒级）。

**依赖**：无（第一优先速修）

**预期收益**：症状2：主治——zoom 不跨 1.0 时从每步整链重跑变为缓存命中；症状1：间接——full refine 命中缓存后 worker 占用大幅缩短；症状3：部分——重复计算消除，PCIe/RAM 流量下降。

---

### T7.2 【结构】按 T1.1 原意重做 GPU 驻留：前缀缓存持有 GPU 缓冲 + 缓冲池化 + VRAM 生命周期治理

**目标**：纠偏 T1.1 实现偏离——计划明文「PreviewExecutor 持有 GPU 缓冲与逐级缓存」，实现落成了 CPU numpy 缓存，导致每 op 全幅 GPU→CPU 回读、输出 CPU 中转再整幅重上传（61MP N-op ≈ (2N+3)×700MB 总线流量，是 6051ms 的主要构成）。
**涉及**：`pipeline/executor.py`（:289-300）、`src/raw_alchemy/gpu_buffer.py`（:35-46 重分配路径）、`workers/image_processor.py`（:927-957 processed 回读 + gpu_graded 重上传 + `_gpu_uint8` 逐 zoom 重分配）、`math_ops.py`（:996-1027 `_sharpen_gpu_bufs`）、`demosaic.py`（:483-491）
**要点**：
- `_prefix_cache` 值改为 GPU 驻留（GpuImage + applied_ev），op 之间零 `to_numpy`，仅最终输出下载一次；
- 驻留级数设 VRAM 预算（建议 ≤4GB）：只保留当前 op 序列的最新一代前缀（新 run 后删除不在本次前缀集合内的旧键），full 源限 1-2 级；executor 暴露 `cache_bytes()/trim(budget)` 供 worker 调用（与 T7.6 协同）；
- `run_result` 返回 GPU 驻留结果；`_do_process` 删除 `result.image`→CPU→`gpu_graded.upload` 往返，直接对 executor 输出缓冲跑 `resize_float_to_uint8_gpu`，只回读最终 uint8 小图（≤12MP≈36MB）；
- GpuImage 按 shape 池化复用，消除每请求 700MB 级 ti.ndarray 的 vkAllocate/Free；`_gpu_uint8` 按 viewport 上限预分配复用；
- VRAM 生命周期：`_sharpen_gpu_bufs` 全局 931MB 永驻 → 纳入池管理、切图/空闲释放；demosaic 2.3GB 瞬时中间缓冲复用同池；
- clip 语义保持（缓存 pre-clip、输出前 clip），`ExportExecutor` 与导出一致性测试不受影响。

**验收**：T1.1 preview/export 一致性测试全绿（atol=1e-5）；日志统计单次交互（前缀命中 + 仅末级 op 变化）的 host↔device 传输 <100MB（当前 GB 级）；61MP full refine 端到端 <2s（当前 6051ms）；连续拖滑杆 30s，RSS 与 VRAM 均无增长趋势。

**依赖**：T7.1

**预期收益**：症状2：主治——每步 PCIe 搬运从 GB 级降至 ~36MB；症状3：主治之一——消除每请求 700MB ndarray churn、sharpen 931MB 永驻与 CPU 中间副本堆积；症状1：间接——preview/refine 更快释放 worker。

---

### T7.3 【结构】worker 在飞任务协作式取消 + 邻居 preload 减负

**目标**：latest-wins 从「队列粒度」升级到「op 粒度」——当前 full refine（~6s）与 _do_preload（3-8s/张）一旦开跑不可中断，新交互最长排队 6s+。
**涉及**：`workers/image_processor.py`（run loop :172-217、`_do_preload` :379-437、`_schedule_full_refinement` :650-660、FULL_REFINE_IDLE_SECONDS :39）、`pipeline/executor.py`（op 循环 :292-297）、`ui/library_controller.py`（:121、:186、:196-214）
**要点**：
- executor 接受 `should_abort: Callable[[], bool]`，op 循环每步之间检查；worker 注入「pending_request 非空即 True」；中止时保留已完成前缀缓存、静默丢弃结果（不 emit，不破坏 `result_ready` 契约）；
- `_do_preload` 拆为阶段（RAW 解码 / GPU demosaic / proxy 生成 / 镜头 map / remap），阶段间检查 pending_request，命中即弃置剩余阶段（已完成部分照常入 cache）；
- preload 减负【允许改变行为】：邻居预载只做 解码+demosaic+双 proxy，不再预计算全尺寸镜头校正 corrected（`compute_lens_distortion_map`+3×cv2.remap INTER_CUBIC 是单张数秒的大头，正片路径本就按需算）——同时消除每张 ~515MB corrected 驻留（配合 T7.6）；
- full refine 触发收紧：到期时队列非空或 1s 内有交互则顺延；开跑后同样受 should_abort 抢占；
- 注意与 T7.2 改同一批文件，建议排其后合并，避免冲突。

**验收**：新增单测：投递慢请求（stub 长 op 链）后立即投递新请求，断言慢请求在 op 边界中止、新请求在 1 个 op 耗时内开跑；GUI 验收——拖滑杆停 1s 触发 refine 后立即再动滑杆，预览响应 <300ms（当前最长 ~6s）；连续快速翻图 10 张，每次点击到出图 <1.5s，无在飞 preload 阻塞。

**依赖**：无硬依赖（可与 T7.1 并行开发；合并顺序建议在 T7.2 之后）

**预期收益**：症状1：主治之一——翻图不再被在飞 preload 卡数秒；症状2：主治之一——refine 在飞时新交互立即抢占，消除最长 6s 冻结；症状3：间接——preload 不再驻留全尺寸 corrected（每张 -515MB）。

---

### T7.4 【结构】proxy/full 缓存分离 + zoom 与色彩管线解耦

**目标**：消除「progressive refinement 每循环一次就互踢一次缓存」与「zoom 每个取值必 miss」两个结构性浪费。
**涉及**：`workers/image_processor.py`（`_should_use_proxy_preview` :633-643、`_invalidate_executor_prefix_if_needed` :860-883、`_make_output_key` :580-600、`_make_preview_target_size` :602-631）、`pipeline/executor.py`
**要点**：
- proxy 与 full 各持独立 PreviewExecutor 实例（或前缀缓存按源身份分桶），zoom 跨 1.0 / idle refine 的源切换不再互相清空；删除 `_last_executor_source_mode` 触发的整体 clear（T7.1 后它已是冗余保险）；
- zoom/viewport 从管线键剥离：`_make_output_key` 拆为管线键（ops 哈希）+ 输出键（round(zoom,3)/viewport_size）；ops 未变仅 zoom 变时，直接取 T7.2 的 GPU 驻留最终级跑 `resize_float_to_uint8_gpu` 出图，完全不进 op 链；
- 输出 uint8 缓存从单槽改为按输出键 2-4 槽（fit 与当前 zoom 各留一份），fit↔100% 往返零重算；
- refine 完成后 proxy 缓存保留，下一次滑杆立即在 proxy 前缀上增量执行；
- 依赖全图统计的 op（auto exposure/metering）的键与统计源保持全图/proxy 语义不变，防止数值回归。

**验收**：GUI 日志验收——zoom 在 0.5↔2.0 间往返 10 次，除首次外无 op kernel 重跑，每步出图 proxy 侧 <100ms、full 侧 <300ms（仅 resize+readback）；拖滑杆→停 1s refine→再拖滑杆，proxy 路径耗时与 refine 前一致（缓存未被互踢）；回归测试全绿。

**依赖**：T7.1, T7.2

**预期收益**：症状2：收尾——缩放本身退化为纯输出重采样（亚 100ms）；症状3：间接——两套缓存各自稳定复用缓冲，消除 gpu_graded 在 34MB↔698MB 间反复重分配。

---

### T7.5 【结构·大】zoom>100% 可见区域（ROI）渲染

**目标**：修正 T3.1「放大 >100% 直接走全尺寸」这一计划自身的设计决定——当前 100% 查看 45MP 图时管线在整图上全量跑、输出 cap 12MP 整图纹理：处理规模比可见区（~2.3MP）大 5-20 倍，且 12MP cap 使 100% 实际显示为模糊纹理。
**涉及**：`workers/image_processor.py`、`ui/viewport_gl.py`（:221-259 整图纹理、:29-44 uniform 变换）、`ui/main_window.py`（zoom/pan 信号链 :114-117、:648-653）、`pipeline/ops.py`
**要点**：
- zoom>fit 时由 viewport 反算源坐标 ROI + 边距（约 1.5×viewport），对 full 源 crop 后进管线（~2-4MP，与 proxy 同量级），输出 ROI 纹理；
- viewport_gl 增加 ROI 纹理定位（offset/scale uniform），pan 在边距内零重跑，超边距增量重跑；ROI 外用 proxy 整图纹理垫底，避免 pan 出界闪黑；
- ROI 只影响源裁剪，不失效色彩管线参数缓存（依赖 T7.4 的键分离）；依赖全图统计的 op（auto exposure/metering）继续用全图或 proxy 统计，不随 ROI 变化；
- 仅改预览路径，导出/CLI 不动；
- 退化方案（若实施成本超预期）：full 源两级金字塔（全图 12MP + 当前 ROI 原生分辨率）。

**验收**：45MP 样张 100% 查看：首次进入 <1s，后续调参每步 <300ms（在 2-4MP ROI 上跑）；pan 在边距内不触发管线、帧率不掉；100% 视图逐像素清晰（摆脱 12MP cap 模糊）；导出一致性测试与全部回归不受影响。

**依赖**：T7.4

**预期收益**：症状2：终局——放大后处理规模与显示需求对齐，6s/步 → 亚 300ms/步，且画质从模糊变逐像素清晰；症状1/3：无直接影响。

---

### T7.6 【速修+结构】主机内存治理：缓存绝对上限 + 字段级驱逐 + 内存可观测

**目标**：根治被用户误判为「GPU 显存问题」的主机 RAM 膨胀——当前配额 =(available+current)×0.7、到 quota×0.8 才驱逐，32GB 机器自稳定在 11-13GB；每张 45MP 条目 ~1.1-1.5GB（含急切预计算的全尺寸 corrected）。
**涉及**：`pipeline/cache_manager.py`（:74-82 配额公式、:106-111 驱逐阈值）、`workers/image_processor.py`（preload corrected 写入 :399-432、交互浏览路径 corrected 回写 :783-793、output 挂载 :970-979）
**要点**：
- 速修：配额改为 `min((available+current)*0.5, 固定上限)`（默认 6GB，设置页可调）；驱逐阈值从 quota×0.8 改为达 quota 即驱逐；
- 字段级驱逐：CachedImage 按 corrected → output_uint8 → denoise → 整条目（linear+proxy 最后）顺序降级淘汰，而非整条目一刀切；corrected 与 linear 为同对象时不重复计费（尺寸统计修正）；
- 交互浏览路径（:783-793）不再为每张浏览过的图写回全尺寸 corrected（与 T7.3 preload 减负同一原则；镜头校正开启时 corrected 只保当前图）；
- 前缀缓存字节预算入口：worker 每次 run 后调用 executor 的 `trim(budget)`（T7.2 落地前为 RAM 预算，落地后为 VRAM 预算）；
- 可观测：每次驱逐/超限记录各缓存分类占用日志，便于 Windows 现场核实「不是显存、是主存」。

**验收**：单测：模拟 20 张 GB 级条目 put，断言总占用 ≤ 上限、corrected 先于 linear 被逐；GUI 验收——连续浏览 30 张 45MP RAW，进程 RSS 稳定在 上限+工作集 以内（当前可达 12-16GB），提交内存不超物理内存、无换页迹象。

**依赖**：无（配额速修可立即做；corrected 语义部分与 T7.3 协调）

**预期收益**：症状3：主治——「显存问题」实为 RAM 膨胀→换页，加绝对上限后消除；症状1：主治之一——文件系统缓存不再被挤掉，连续翻图的 RAW 解码不再逐渐劣化。

---

### T7.7 【速修】缩略图管线：缩放解码 + 可视区优先 + 切文件夹不阻塞 UI

**目标**：消除批量加载的「解码风暴」——当前对内嵌 JPEG（现代机身为全分辨率 24-60MP）全尺寸解码后才缩到 300px，8 线程一次性提交全部文件，切文件夹时 UI 线程还 wait() 冻结 0.3-3s。UPGRADE_PLAN Phase0-6 完全没有缩略图任务，属计划盲区。
**涉及**：`workers/thumbnail_worker.py`（:41 max_workers、:55-135 extract_thumbnail、:165-171 提交与 stop 检查、:187-188 stop）、`ui/library_controller.py`（:94-107 _open_folder/start_thumbnail_scan、:121）、`ui/main_window.py`（:846-849 closeEvent）
**要点**：
- 用 `QImageReader` + `setScaledSize` 替代 `QImage.loadFromData` 全尺寸解码（libjpeg DCT 分级缩放解码，速度 ~8x、峰值内存 ~1/60）；旋转移到缩放之后（在 300px 上 transformed，而非 60MP 上再造一份 72MB 副本）；
- 回退路径（:100-119）：half_size 全图 float32+布尔掩码 np.power 改为先 cv2.resize 到 ~600px 再做 sRGB（临时数组从数百 MB → <2MB）；
- 并发从固定 8 改为 `max(2, cpu_count-2)` 且线程降优先级；提交顺序按 gallery 可视区优先，滚出视口且未开始的任务取消；
- 切文件夹/退出不再 UI 线程 `wait()`：旧 worker stop() 后经 finished→deleteLater 异步回收，立即启动新 worker；`extract_thumbnail` 在解码前检查 stop 标志，使实际收尾 <100ms。

**验收**：200 张目录：可视区首屏缩略图 <3s（当前数十秒），全量提取期间 UI 事件循环无 >50ms 卡顿（加延迟探针验证）；连续切换文件夹 UI 冻结 <50ms（当前 0.3-3s）；竖拍图方向正确（旋转后置回归测试）。

**依赖**：无（第一优先速修，与 T7.1 并行）

**预期收益**：症状1：主治——解码风暴消除，CPU 占用与内存 churn 降一个数量级，首屏数十秒→秒级，点击选片不再与缩略图线程池抢 CPU/磁盘；症状2/3：无直接影响。

---

### T7.8 【结构】缩略图磁盘缓存（mtime 键控）

**目标**：对齐 Lightroom/Bridge/FastRawViewer 的基本行为——当前无任何缩略图持久化，同一会话内重进同一文件夹也全量重解码。
**新建**：`src/raw_alchemy/thumb_cache.py`；**涉及**：`workers/thumbnail_worker.py`、`ui/library_controller.py`（:94-107）
**要点**：
- 以 (绝对路径, mtime, size) 的 hash 为键，300px 高 JPEG(q85)/WebP 写入 `QStandardPaths.CacheLocation`/`raw_alchemy/thumbs/`；命中直接读文件（~5ms/张），未命中走 T7.7 提取后异步写入；
- 键中带格式版本号；读失败/损坏即回退重提取并覆盖，不崩溃；
- 容量治理：缓存目录超 500MB 或 2 万张时按 atime LRU 后台清理；
- 会话内存层：`_open_folder` 不再无条件清空 gallery 数据，同文件夹重进直接复用已解码 QPixmap；
- 设置页加「缩略图缓存」开关与清空按钮（默认开）。

**验收**：同一 200 张文件夹第二次打开，全部缩略图 <1s 出齐（当前与首次同量级）；修改某 RAW 的 mtime 后仅该缩略图重新生成；容量清理单测通过；关闭开关后行为回退为现状。

**依赖**：T7.7（复用其缩放解码产物作为缓存写入源）

**预期收益**：症状1：选片工作流反复进出文件夹的场景从数十秒 → 亚秒级；症状2/3：无直接影响。

---
