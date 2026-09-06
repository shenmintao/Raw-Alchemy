# AMD Bayer/RCD 产品接入与验收

日期：2026-09-06（Asia/Singapore）。
主工作区：Mac mini 的 `/Users/shenmintao/Raw-Alchemy-pr31`，
分支 `fix/pr31-coreml-cache`。本轮未提交、推送或发布。

## 已完成的产品改动

- 新增可复现生成器 `tools/build_migraphx_rcd.py` 与
  `rcd_demosaic_migraphx_1536.onnx`，原图 SHA 校验通过才生成。
  三处 CFA 掩码 Tile 替换为两级 Gather，保留算法、系数、白平衡、相机矩阵及边界逻辑。
  图接口固定 1536 尺寸，调用方也检查分块尺寸，不在启动时转换模型。
- MIGraphX 在已测 Linux x86-64 / ORT 1.23.2 / ROCm 7.2.0、
  启用原生子进程隔离时自动选择新 RCD 图。其他分块尺寸使用原始 CPU 模型。
  显式 cpu 设置有效；CUDA、DirectML、CoreML 的既有模型选择保持原策略。
- 新 RCD 图复用隔离子进程的严格浮点、禁用代数重排、64 MiB 编译栈、
  默认 180 秒编译预算及内容寻址 MXR 缓存。超时和取消仍可结束子进程。
- 编译失败重建原始 CPU 模型。新资产加入图像阶段身份，策略版本递增，
  防止会话或图像缓存跨策略变更误用。
- 新增 14 项普通回归用例（含扩展现有测试参数），以及独立开启的 AMD 真机验收。

## AMD 原生验收

物理 GPU：RX 9070 XT；Ubuntu 24.04 WSL2、ROCm 7.2、ORT MIGraphX 1.23.2。
测试调用当前产品的 provider 选择、固定分块、进程隔离和模型定位路径。
未猴子补丁绕过 AMD 自动策略，未放宽 `atol=3e-6, rtol=3e-6`。

| 场景 | 最大绝对误差 | 超差通道 |
| --- | ---: | ---: |
| 四种 CFA 相位，随机高亮输入、非默认白平衡/矩阵 | 4.18e-7 以下 | 全部 0 |
| 最小尺寸黑场，含反射填充 | 0 | 0 |
| 近似平坦区域 | 0 | 0 |
| 跨分块边缘、高光、单点输入，1598×1574 | 0 | 0 |
| 真实 NEF，使用相机自身白平衡/颜色矩阵，4284×2844 | 1.49e-7 | 0 |

所有输出有限、无 CPU 回退。首次 GPU 会话记录 16 次 MIGraphX 分区执行；
新建进程加载缓存后的真实 NEF 记录 6 次 MIGraphX 分区执行。
独立 CPU 资产对照另验证四种 CFA 相位在改图前后逐位一致。

首次初始化/编译 **27.47 秒**；新建推理进程加载同一缓存 **1.64 秒**。
生成一个 MXR 文件，再次运行未重写。真实 NEF 去马赛克 CPU 5.10 秒，
GPU 两次 0.269 / 0.359 秒。时间为功能诊断，未经受控性能基准设计，
仅含去马赛克及分块/通信，不能当成整软件导入或导出的总耗时。

实际编译期间取消：5.06 秒结束，PipelineAborted 正常传播、未置 CPU 回退标记；
实际编译预算设为 5 秒：5.35 秒后成功回退 CPU。
两次结束后都无遗留 multiprocessing 子进程。
对应异常传播、IPC、超时和恢复回归也在全量测试中通过。

AMD 真机测试初次因验收脚本从错误模块导入颜色矩阵函数而失败；
更正为 `colorspace_matrices` 后完整测试通过，产品无对应导入改动。

## 三平台回归和打包

| 平台 | 全量回归 | wheel / 独立程序 |
| --- | --- | --- |
| macOS arm64 | 519 passed，7 skipped，72.41 s | 重建并验证通过 |
| Windows x86-64 | 519 passed，7 skipped，59.10 s | 重建并验证通过 |
| Linux x86-64（NAS） | 519 passed，7 skipped，43.73 s | 重建并验证通过 |

7 个跳过项为显式开启的真实 GPU/RAW 验收；本轮 AMD 扩大验收另行通过。
Windows 首次测试漏设 PYTHONPATH 导致收集失败；补齐源码路径后的全量结果如表。
Ruff、git diff --check 通过。本轮 11 个修改/新增源码、文档、测试及模型文件，
在 Mac、Windows、NAS 的 SHA-256 清单一致；AMD WSL 副本也已同步。

三平台 wheel 均在独立应用安装目录验证 Lensfun、数据库、模型身份、
隔离 CPU 推理及 Qt。依赖目录借用构建环境，不是空白操作系统验收。
wheel 与 onedir 中的新 RCD 资产逐字节匹配源码。
三平台最终 onedir 程序均显示可见窗口、加载包内 Lensfun 数据库、
初始化 OpenGL，并正常退出 0。Linux 窗口验证使用 Xvfb/Mesa 软件渲染。
本轮没有重跑整套 GUI RAW 编辑/导出流程，也没有进行颜色视觉评价。

Linux 默认发行依赖仍使用 CUDA/CPU 运行时；AMD 加速需要匹配的
ROCm/MIGraphX 环境。本轮 AMD GPU 结论来自现有 WSL 验证环境中的产品源码路径，
未制作 AMD 专用冻结程序，不将 NAS 打包成功等同于 AMD 冻结程序验收。
未更改宿主驱动、系统安装或防火墙。

## 尚存的其他问题

- Windows DirectML 原始 Bayer 图此前独立严格对照仍有 2 个超差通道；
  WMIC 缺失也仍影响 GPU 名称识别。这两项不属于本轮 MIGraphX 修复。
- 一张真实 NEF 和合成边界案例不能代表所有相机、驱动及 ROCm 版本；
  自动策略仍限定已测运行时组合。
- 正式发行归档、签名、托管 CI、提交及合并未进行。

## 文件、程序与校验值

本地审阅目录：`/data/amd-bayer-integration-20260906/`：
`source/`、`before/`、`changes.json`、`integration.patch`、`integration.zip`、
`manifest.json`、`native-acceptance.json`、`native-cancellation.json`、
`macos-result.json`、`windows-result.json`、`linux-result.json`。

各平台源码目录的 `.test-output/amd-bayer-integration-20260906/`：
`full.log/xml`（Windows 为 `full-v2.log/xml`）、`package-result.json`、
`installed-wheel.log`、`dist/`、`wheels/` 与冻结程序结果。
Mac 应用为 `dist/RawAlchemy.app`；
Windows 为 `dist/RawAlchemy/RawAlchemy.exe`；
NAS Linux 为 `dist/RawAlchemy/RawAlchemy`。

AMD 原生证据位于
`/home/shenmintao/raw-alchemy-gpu-20260906/source/.test-output/amd-bayer-integration-20260906/`：
`native-v2.log`、`native-v2/test_native_amd_bayer_cases_an0/acceptance.json`、
同目录 ORT profiles / MXR 文件、`cancel.log`、`native-cancellation.json`。

新 RCD 模型 SHA-256：
`f881c81c532020c0ee58f3c323b9ef06b2364f1bdc7f48f55914bc8cc92dcde7`。