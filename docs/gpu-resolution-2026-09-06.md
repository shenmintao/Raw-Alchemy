# Raw-Alchemy GPU 精度与加速验收

日期：2026-09-06（Asia/Singapore）。主工作区为 Mac mini 的
`/Users/shenmintao/Raw-Alchemy-pr31`，分支 `fix/pr31-coreml-cache`。
改动已同步到 Windows、原生 Linux 和两台 WSL2 GPU 验证副本；未提交、推送或发布。

## 已验证结果

下表是同一张 Fuji X-T1 RAF（4934×3296）的去马赛克阶段，
不是整个软件的导入、降噪、调色、编码总耗时。精度门槛始终为
`abs(error) <= 3e-6 + 3e-6 * abs(reference)`，没有放宽阈值或忽略坏像素。

| 环境 | CPU 推理 | GPU 推理 | 精度与执行证据 |
| --- | ---: | ---: | --- |
| Mac mini M4 / macOS 27 / ORT 1.29 / CoreML ALL | 30.48 s | 默认路径 18.13 s；连续三次 17.61 / 15.39 / 14.87 s | 超差通道 0，最大误差 5.96e-8；实际 CoreML 分区执行，计算计划含 1957 条 GPU 分配记录 |
| NVIDIA RTX A1000 Laptop / Ubuntu 24.04 WSL2 / ORT CUDA 1.29 | 19.21 s | 5.04 s | 输出与 CPU 完全一致；CUDA 21948 条执行事件；无 CPU 回退 |
| AMD RX 9070 XT / Ubuntu 24.04 WSL2 / ROCm 7.2 / ORT MIGraphX 1.23.2 | 35.04 s | 缓存复用后 0.792 s | 输出与 CPU 完全一致；MIGraphX 12 条分区执行事件；无 CPU 回退 |

时间包含生产分块及进程通信，初始化单列，不是受控跨机器跑分。
AMD 首次初始化/编译实测 116.24 s；独立进程再次加载编译缓存为 5.50 s。
不能把 0.79 s 当成首次打开整张照片的总耗时。
Mac 默认路径初始化为 1.96 s，CUDA X-Trans 为 2.43 s。

CUDA Bayer 的真实 NEF（4284×2844）也通过：CPU 3.49 s，GPU 1.45 s，
最大误差 5.96e-8，超差通道 0，CUDA 3696 条执行事件。
CUDA 与 MIGraphX 的 FastDenoise 原生测试均通过原有 max < 0.02、
mean < 0.001 门槛，最大差异约 0.000732；不是只检查 provider 是否注册。

## 修复内容

1. **CoreML X-Trans**：原图会产生 153 个超差通道。新的预生成图仅把 16 处
   除以三的浮点运算通过 float64 计算后还原 float32，保留系数、比较和边界语义。
   ALL 调度通过整帧严格检查，并明显快于 CPU；CPUAndGPU 限制调度曾测得
   145.87 s，因此采用实际验证过的 ALL。自动启用限定为已测 Apple Silicon
   macOS 27 / ORT 1.29；其他组合默认 CPU，可显式诊断。
2. **Linux CUDA**：修复子进程内 cuDNN 等 pip CUDA 依赖预加载。
   通过 ORT 的 preload_dlls 接口加载，CPU-only、禁用预加载开关和其他后端仍有效。
   原生测试新增“必须是预期 GPU”的断言，CPU 静默回退无法通过验收。
3. **Linux ROCm/MIGraphX X-Trans**：识别现代 MIGraphX EP。默认融合曾有
   844 个超差通道；严格浮点及关闭代数重排后缩小到 2 个，最后用明确的 x*x
   替换 28 个二次幂运算后归零。产品使用独立 AMD 变体（61 处精确除法、
   28 处平方），仅在该推理子进程中设置严格浮点、禁止代数重排和 64 MiB 编译栈。
   自动选择限定为 Linux x86-64、ORT 1.23.2、ROCm 7.2.0，并要求子进程隔离。
4. **AMD 编译缓存**：独立命名空间包含模型字节、运行时、形状、provider 选项和
   编译环境；EP 再加入 MIGraphX/GPU 架构身份。跨进程复用已验证。
   缓存不可写时保持精度并关闭磁盘复用。X-Trans 默认编译预算为 180 秒，
   仍可取消、超时结束及回退 CPU。
5. **资产与测试**：两个精度模型及其生成器纳入源码；图更新参与阶段缓存身份，
   CPU/CUDA/DirectML 保持原图。新增整张 RAW 的生产路径验收，同时检查
   有限输出、逐通道误差、无回退、实际 GPU 执行。

## 验证范围

三平台最终全量测试：**各 505 passed，6 skipped**。
Mac 63.30 s、Windows 55.52 s、原生 Linux 33.83 s。
跳过的是需要显式开启及真实硬件/RAW 的原生验收，它们的对应硬件结果单独记录。
Mac 原生 CoreML 图与策略测试 42 项通过；后续全量覆盖新增 AMD 配置与版本门槛。
CUDA 环境另有 120 项兼容性检查通过，最终版本门槛增量也已同步。

Mac wheel 已重建，并检查两个精度模型与源码字节一致；
独立安装后的 Lensfun、数据库、模型身份、隔离 CPU 推理和 Qt 检查通过。
本轮没有重新制作三平台冻结 GUI 安装程序，也没有重复整套 GUI RAW 编辑/导出流程。
此前 GUI 与冻结包通过的记录属于旧源码快照，不冒充本轮 GPU 或发行验收。

## 尚未解决

**AMD Bayer/RCD 的 GPU 编译仍未通过。** 默认编译、加大编译栈/预算、
拆分逐点融合均遇到长时间编译或编译器栈耗尽。产品默认让这个阶段使用 CPU，
保留已经验收通过的 AMD FastDenoise 和 X-Trans 加速。
不把 RCD 的 CPU 回退结果记作 ROCm GPU 验收成功。

Linux GPU 结果来自 WSL2 上的物理 NVIDIA/AMD GPU，不是裸机 Linux 全套驱动/GUI验收。
两个 GPU 测试环境没有打包的 Linux Lensfun，测试限于 RAW 预处理、去马赛克、
FastDenoise，不覆盖镜头校正或整套导出。原生 Linux 软件回归在 NAS 独立完成。
没有修改宿主显卡驱动、防火墙，也没有上传私有照片；样片是既有公开测试样片。

## 文件与复查

- 主工作区证据：`.test-output/gpu-resolution-20260906/`
- 本地报告、源码改动、校验清单及日志：`/data/gpu-resolution-20260906/`
- CUDA 日志：Laptop WSL 的 `/home/shenmintao/raw-alchemy-gpu-20260906/`
- AMD 日志：PC WSL 的同名路径
- 实际硬件验收：`tests/test_demosaic_backend_native.py`、`tests/test_backend_native.py`
- 配置说明：`docs/backend-session-policy.md`

关键模型 SHA-256：
- CoreML：`83bd7385c6368c3dc1edf88e741aa084c51ea9140777afb212d50d88a8ffac40`
- MIGraphX：`1f2aabef5ba3a15afaf289bb89fbfa63b269e592f72af7c46f9de22572d0a98b`

实现参考：[CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)、
[CUDA 依赖预加载](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、
[MIGraphX 编译设置](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/docs-7.2.0/reference/MIGraphX-dev-env-vars.html)。
以上性能与通过结论均来自本地实测日志。