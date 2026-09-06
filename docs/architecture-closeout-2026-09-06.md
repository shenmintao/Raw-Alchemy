# Raw Alchemy 架构收尾与三平台验收

日期：2026-09-06（Asia/Singapore）

主工作区：Mac mini 的 `/Users/shenmintao/Raw-Alchemy-pr31`。
分支 `fix/pr31-coreml-cache`，HEAD `e73be8c`。当前包含此前持续开发的未提交改动。
本轮增量已同步到 Windows 和 Linux 测试副本；没有提交、推送或合并。
此前 `Raw-Alchemy-pr31-verification.md` 的“工作区干净、321 项通过”描述仅属于历史状态。

## 已有架构与本轮修复

已有实现包括共享资源管理器、独立有界导出线程、阶段/分块让出计算、
三平台 RAW/ONNX 子进程隔离、共享 float32 降噪缓存、不可变请求参数、
过时 UI 结果拦截、原子导出及固定版本的 Lensfun 构建。这些不是本轮重新实现的内容。

本轮发现并修复了三类实际问题：

1. 加载、预加载及缓存回填吞掉取消和内存异常。现在这些异常交给统一任务边界处理；
   正常取消不再显示加载失败。缓存回填真正失败时显示错误，不发送加载完成。
   代理图分配内存失败时停止任务，不再退回内存需求更大的全分辨率处理。
2. Windows 快速等长改写会产生相同 stat 时间戳，旧哈希因此被错误复用。
   1000 次诊断改写中记录了 772 次 stat 碰撞；原有源文件替换测试因此实际失败。
   现在按文件当前内容重新计算身份，读取期间响应取消。
   预加载的身份读取移出发布锁，避免 GUI 等锁时承担文件读取。
3. Linux PyInstaller 的 strip 步骤破坏 NumPy/SciPy OpenBLAS 的 ELF LOAD 对齐。
   原库可加载，使用实际构建工具 strip 的临时副本复现相同失败。
   两种打包 spec 均禁用 strip；最终 Linux onedir 程序已重新构建并实际启动验收。

新增 10 个回归用例：8 个取消/回填/内存用例、2 个相同元数据但内容改变的身份用例。
修复前分别复现 8 项和 2 项失败；修复后全部通过。

## 最终验证

三平台的 171 个快照文件逐一 SHA-256 比对一致。
`final-manifest.json` 保存具体文件身份；本轮可审阅增量见 `closeout.patch`。

| 环境 | 全量 CPU 测试 | 实际 GUI RAW 流程 |
| --- | --- | --- |
| macOS arm64 / Mac mini | 463 passed，4 skipped，44.47s | Bayer、X-Trans 均通过，默认后端 |
| Windows x86-64 / MinQ-PC | 463 passed，4 skipped，29.67s | Bayer、X-Trans 均通过，默认 DirectML 路径 |
| Linux x86-64 / MinQ-NAS | 463 passed，4 skipped，23.33s | Bayer、X-Trans 均通过，CPU + Xvfb/Mesa 软件渲染 |

实际 GUI 验收使用公开 NEF/RAF 样片及隔离配置/缓存目录，禁用 sidecar 写入。
覆盖打开、有效预览、100%/适应窗口缩放、曝光调整、降噪、当前图像缓存导出、
完整处理入口导出、JPEG 像素/尺寸验证及正常关闭。各流程进程 exit 0。
Bayer 输出 4284×2844，X-Trans 输出 4934×3296。
这些是自动化功能验收，不是人工色彩评价或受控性能基准。

Linux 最终 onedir 安装包额外通过：无需源码路径覆盖即可加载 Lensfun/NumPy，
显示可见主窗口、初始化 OpenGL，并以 exit 0 正常关闭。
验收脚本首次误选 Qt 隐藏窗口导致关闭检查超时；更正为可见且支持关闭协议的
主窗口后通过。该次超时不作为产品退出缺陷。
Windows/macOS 的最终 wheel 与 onedir 程序已在后续收尾中重新构建并验证，见下方记录。

Ruff、`git diff --check` 通过。
4 个跳过项为显式开启的真实后端性能分析、RCD/X-Trans CoreML 原生测试和
完整 RAF CoreML 精度测试。CPU 全量通过不能代替这些硬件/画质验证。

## 证据位置

- Mac：主工作区 `.test-output/architecture-closeout-20260906/`
  - `final.log/xml`、`bayer-final/result.json`、`xtrans-final/result.json`
  - `closeout.patch`、`final-manifest.json`
- Windows：`C:/Users/shenm/raw-alchemy-validation-20260905/continuation-final-20260905/source/.test-output/architecture-closeout-20260906/`
  - `final.log/xml`、`bayer-final/result.json`、`xtrans-final/result.json`
  - `probe_identity.py` 为时间戳碰撞诊断
- Linux：`/home/shenmintao/cross-platform-final/continuation-final-20260905/`
  - `source/.test-output/architecture-closeout-20260906/final.log/xml`
  - `architecture-closeout-20260906/linux-bayer-final/result.json`
  - `architecture-closeout-20260906/linux-xtrans-final/result.json`
  - `architecture-closeout-20260906/linux-frozen-result.json`、`linux-pyinstaller.log`
  - 最终程序：`architecture-closeout-20260906/dist/RawAlchemy/RawAlchemy`

## 剩余边界

- X-Trans 的 CoreML GPU 严格精度失败仍未解决；自动去马赛克继续选 CPU，
  显式 MLProgram 仅供诊断。
- Linux CUDA/ROCm 未完成真实 GPU 验收；本轮 Linux 结论限定为 CPU 与软件 OpenGL。
- 资源预算是保守估算与 RSS 采样，不是操作系统强制内存上限；编码、元数据和
  部分其他原生调用仍只支持阶段间取消，不能承诺任意情况下的关闭时限。
- 内容身份校验增加了文件读取。保留正确失效语义，尚未进行整个应用的吞吐/延迟
  性能验收，不把局部降噪提速当成全软件提速。
- 托管 CI、正式发行归档/签名、代码提交/推送与合并尚未进行。

## 最终打包补验（2026-09-06 08:55，Asia/Singapore）

Mac mini 与 Windows PC 均从最终 171 文件清单验证一致的源码重新构建。
未修改产品代码，没有提交、推送、触发托管 CI 或发布。

| 平台 | 最终 wheel | 最终 onedir 程序 |
| --- | --- | --- |
| macOS arm64 | 独立安装后 Lensfun、数据库、模型身份、隔离 CPU 推理和 Qt 通过 | 可见主窗口、包内 Lensfun 数据库、OpenGL 初始化、正常退出 0 |
| Windows x86-64 | 同上，全部通过 | 同上，正常退出 0 |

wheel 检查借用构建环境的依赖目录，应用自身从独立环境加载；不是全新操作系统验收。
冻结程序从独立工作目录启动，移除 PYTHONPATH 与 Lensfun 路径覆盖，使用隔离用户目录。
macOS 使用按 PID 定位的 NSRunningApplication 正常退出请求；Windows 向该进程的
可见主窗口发送 WM_CLOSE。均无需强制终止。窗口测试初次只扫描文件日志，
实际日志在 stdout；随后独立断言 stdout 中 Lensfun/OpenGL 初始化成功。
本次冻结程序只验证启动和关闭，RAW 编辑/导出流程仍引用上轮源码 GUI 验收。

证据和程序在两平台源码目录的 `.test-output/release-refresh-20260906/`：
- `wheel-result.json`、`installed-wheel.log`、`wheels/`
- `frozen-result.json`、`frozen-stdout.log`、`check_frozen.py`、`verify_logs.py`
- macOS 应用：`dist/RawAlchemy.app`
- Windows 程序：`dist/RawAlchemy/RawAlchemy.exe`

wheel SHA-256：
- macOS：`0fca454927c6e8645ced85e1061a32b839286b8874fb12dd1cf9a3db6d8d8b80`
- Windows：`803fee563584d1b0b8baa9a2359c3138cbf5295c527c4fbedd41f63671ce5cb3`

硬件确认：NAS 的 lspci 仅列 Intel Alder Lake-P 核显，存在 /dev/dri，
未发现 NVIDIA/AMD 计算设备，无法用这台机器完成 CUDA/ROCm 验收。

新增待办：Windows 冻结程序日志显示 WMIC 不存在，
`onnx/gpu_runtime.py:detect_gpu_vendor` 仍用 WMIC，失败后仅探测 nvidia-smi，
因此没有后者的 AMD/Intel 设备可能被识别为 unknown。当前程序仍启动成功；
这个缺口尚未修复，不能把本次启动通过等同于 GPU 自动识别正确。