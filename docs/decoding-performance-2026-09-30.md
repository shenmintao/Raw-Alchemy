# RAW 解码审计与 Windows/Mac 性能验证（2026-09-30）

本次修复去掉了误伤真实细节的默认坏点处理，改进全部 Inspector 滑块的请求调度，并启用固定分块的 CoreML 调色、Bayer RCD 和 X-Trans 去马赛克。降噪强度改为在交互稳定后计算最终值。

## DSCF2367.RAF 的彩色伪影

输入为 `E:\Photos\2023-03-18 22.07.37\DSCF2367.RAF`，FUJIFILM X-S10 的 X-Trans RAW。无 LUT、Log 空间为“无”。该照片使用 Markesteijn 1-pass，使用 RCD 的是 Bayer 照片。

问题来自标准解码强制调用的 `fix_hot_pixels()`：它对每个 6×6 CFA 相位的子平面做 3×3 中值判断，实际邻居相距 6 个传感器像素，把花瓣、细枝等真实结构当作坏点。在全帧改写了 454,638 个样本，900×900 样本裁切改写 17,986 个样本，最大归一化 RAW 改变量约 0.417。该照片的高光重建改写 0 个样本。

CPU 和 Windows DirectML 都能复现伪影；绕开坏点处理后都恢复正常。完整解码后的裁切已目视对照，严重彩边消失、细枝恢复。标准路径保留实测 CFA 数据；函数仍保留，供兼容或显式诊断使用。Bayer 和 X-Trans 的真实细节保真回归覆盖了这个入口。

原始诊断和图像保留在忽略目录 `.test-output/raf-color-overflow/`，其中 `current_center.png` 为修前，`fixed_center.png` 为修后。`diagnose.py` 复用修前 `current.npy`，避免新代码覆盖旧证据。

坏点处理适合确有异常感光点的照片，应当作为独立、可选、保守的校正步骤。它不是官方 RCD 的必需步骤；darktable 的 `hotpixels.c` 是独立模块，RCD 实现在另一个文件中。

## 与 darktable 是否等价

**整条解码链不等价。** 现有 ONNX parity 主要证明与本项目退休的 Taichi 移植结果接近，不能用来证明与当前 darktable 全管线逐像素相等。

审计固定上游 commit：[`426d8adc4cd1406ef979cdb5dce29f270e3b600a`](https://github.com/darktable-org/darktable/tree/426d8adc4cd1406ef979cdb5dce29f270e3b600a)，提交时间 2026-09-29 20:54:14 UTC。下载的源文件及 SHA-256 保存在 `.test-output/raf-color-overflow/darktable-pinned/sources.json`，包含 `segbased.c` 和 `segmentation.c`。

| 环节 | 本项目 | darktable 审计证据与差异 |
| --- | --- | --- |
| RAW 解包与画幅 | RawSpeed 优先、rawpy 回退 | 样本 RawSpeed 为 4160×6240，rawpy 为 4170×6246；首先需要对齐有效画幅和 CFA 相位 |
| 坏点 | 标准路径已移除启发式中值修正 | [hotpixels.c](https://github.com/darktable-org/darktable/blob/426d8adc4cd1406ef979cdb5dce29f270e3b600a/src/iop/hotpixels.c) 独立于 RCD |
| 白平衡顺序 | 去马赛克后乘 WB，再转工作空间 | [temperature.c](https://github.com/darktable-org/darktable/blob/426d8adc4cd1406ef979cdb5dce29f270e3b600a/src/iop/temperature.c#L550) 对 CFA 感光点乘 WB；非线性去马赛克不能假设与 WB 交换顺序仍相等 |
| 负值及归一化 | 保留黑电平以下噪声，做 affine lift/scale 后去马赛克，再撤销 lift | [rcd.c 的 `_safe_in`](https://github.com/darktable-org/darktable/blob/426d8adc4cd1406ef979cdb5dce29f270e3b600a/src/iop/demosaicing/rcd.c#L80) 使用 `max(0, a) * scale`；其 scale 与 processed maximum 相关 |
| X-Trans | ONNX Markesteijn 1-pass | [xtrans.c](https://github.com/darktable-org/darktable/blob/426d8adc4cd1406ef979cdb5dce29f270e3b600a/src/iop/demosaicing/xtrans.c) 有不同 pass 数和边界处理，不能直接对比默认外观 |
| 高光重建 | 固定阈值、OpenCV 分割、对向通道参考和分段色度修正的简化实现 | [highlights.c](https://github.com/darktable-org/darktable/blob/426d8adc4cd1406ef979cdb5dce29f270e3b600a/src/iop/highlights.c#L1000) 及 [segbased.c](https://github.com/darktable-org/darktable/blob/426d8adc4cd1406ef979cdb5dce29f270e3b600a/src/iop/hlreconstruct/segbased.c) 包含候选选择、可调 combine/recovery 等；仅有相同名称不能说明相等 |
| 工作空间与高光范围 | Camera→ProPhoto/所选工作空间；解码输出上限裁到 1 | darktable 输入 profile、工作空间、模块配置影响结果；[scene-referred 高光模式](https://github.com/darktable-org/darktable/blob/426d8adc4cd1406ef979cdb5dce29f270e3b600a/src/iop/highlights.c#L1040) 可保留超过 1 的值 |

本次没有重写白平衡顺序或高光动态范围。这些是后续追求 darktable 兼容时需要单独定义、校准和验证的算法变化。

## GPU 选择和性能规避策略

Mac 安装包已经包含 ONNX Runtime，原 `/Applications/RawAlchemy.app` 中为 1.30.0。存在 ORT 只是推理能力的前提，是否有 CoreML/CUDA/DirectML 等实际 provider 仍需检查。

Apple auto 取消“仅某个精确 macOS/ORT 版本可用”的门槛，改为原生 arm64、macOS 12+、ORT 1.20+ 和 provider 能力检查。CoreML 配置失败或推理失败会一次回退 CPU，并阻止对同一失败配置不断重试。

| 阶段 | 现在的 Apple 策略 |
| --- | --- |
| 普通调色 | MLProgram + CPUAndGPU，固定 1024×3072 像素块 |
| LUT/Log 调色 | 同上，固定 512×1536 像素块，限制 Gather 中间张量 |
| RCD | MLProgram + ALL，768px 固定块，24px overlap |
| X-Trans | 修复精度的 MLProgram + ALL 图，780px 固定块，24px overlap |
| FastDenoise | 现有 MLProgram + ALL 策略，512px 模型块 |
| 仅 CPU 的融合调色 | 在现有处理线程运行，减少 IPC 拷贝；其他原生/加速会话保留子进程隔离 |

调色模型没有空间算子，因此可以展平像素并补齐最后一块，丢弃补齐输出，避免每种裁切/ROI 尺寸重新编译。失败回退按会话的固定形状继续分块。锐化、镜头映射、部分几何/resize 仍用 NumPy/OpenCV；窗口显示使用 OpenGL。CoreML 也会保留 CPU 分区，尤其是精度敏感运算，不能称全部节点都在 GPU 上。

详细控制参数与回退条件见 [backend-session-policy.md](backend-session-policy.md)，调度和内存边界见 [runtime-architecture.md](runtime-architecture.md)。

## 同机端到端性能

测试为 Apple M4、16GB、macOS 27.0、ORT 1.30.0、同一 RAF。旧版是本地 HEAD `5fff0bc` 的隔离 baseline；新版为本次修改。同一脚本通过真实 ImageProcessor 请求测量，常规项目每项三个不同参数的中位数，预览源为约 3MP。下表主要反映会话和相关前缀缓存已经准备好的请求；首次编译和首次缓存构建另列。

| 请求 | 修前 ms | 修后 ms | 倍数 |
| --- | ---: | ---: | ---: |
| 曝光 | 304.7 | 132.1 | 2.31× |
| 色温 | 307.9 | 135.0 | 2.28× |
| 色调 | 301.2 | 130.2 | 2.31× |
| 高光 | 300.9 | 136.3 | 2.21× |
| 阴影 | 298.1 | 142.2 | 2.10× |
| 饱和度 | 299.1 | 138.5 | 2.16× |
| 对比度 | 299.2 | 140.7 | 2.13× |
| 裁切 | 256.7 | 141.1 | 1.82× |
| 旋转 | 353.0 | 161.4 | 2.19× |
| 细节区域平移 | 270.0 | 125.1 | 2.16× |
| 锐化 | 199.6 | 140.7 | 1.42× |
| LUT | 1488.3 | 372.6 | 3.99× |
| Log + LUT | 2511.2 | 440.8 | 5.70× |
| 镜头开关 | 619.6 | 179.7 | 3.45× |
| 首次降噪（单次） | 6359.3 | 4507.2 | 1.41× |
| 降噪后普通调色 | 326.3 | 132.7 | 2.46× |

26MP 全帧导出中的融合调色为 5904.5→784.4ms；实际 TIFF 写盘为 126.0→179.5ms，没有 I/O 加速结论。首次 RAW 打开为 9407.9→8688.1ms，去马赛克并未获得调色那样的大倍数提升。

LUT 首次约 745ms，Log+LUT 首次约 1.94s。第一次准备细节平移的完整镜头/源缓存仍约 1.28s。单独带 profiling 的 26MP 大 ROI 请求约 907ms；该值没有严格同条件的 baseline。以上数字都不是所有设备、图像尺寸或首次使用场景的保证。

数据为 `.test-output/mac-exposure/suite-before.json` 和 `suite-final.json`。中途的 `suite-after.json` 是内存失败实验，不用于最终结论。

可复测：

```sh
PYTHONPATH=src .venv/bin/python tools/bench_pipeline.py /path/to/DSCF2367.RAF \
  --denoise --out .test-output/pipeline.json
```

## 连续拖动与降噪延后

旧调度每 80ms 提交新请求，会让仍在推理的 GPU 会话被取消。真实 Mac GUI 的曝光拖动测试发送 100 次、间隔约 16ms 的滑块事件：只替换调度、保持同一优化后的 GPU 管线，旧调度中间帧为 0、会话重启 15 次；新调度中间帧为 8、重启 0 次，释放后最终值分别为约 340/233ms。

全部 Inspector 连续滑块共用修复：当前普通预览完成后提交最新参数，相同参数不重复提交，worker 用事件立即唤醒。换图和 viewport 几何请求仍可取消旧工作；邻图 preload、空闲精细预览仍可被用户打断。

| 另测的连续滑块 | 拖动中帧数 | 释放后最终值 ms | 拖动期间会话重建 |
| --- | ---: | ---: | ---: |
| 色温 | 9 | 146.6 | 0 |
| 色调 | 8 | 199.2 | 0 |
| 高光 | 8 | 235.9 | 0 |
| 阴影 | 8 | 190.1 | 0 |
| 饱和度 | 9 | 269.3 | 0 |
| 对比度 | 8 | 96.3 | 0 |
| 锐化 | 13 | 99.8 | 0 |

降噪强度是模型的 sigma 输入，不能把它当作输出混合比例来廉价插值而声称保持同一算法。现在拖动期间保持已有降噪；没有结果时先显示未降噪预览。松手且输入稳定 200ms 后，才计算最终强度。期间调色继续使用当前图像的已有降噪缓存，UI/sidecar/导出始终保留用户选择的最终参数。

实际 GUI 验证：首次开启并连续调强度，拖动中没有降噪事件，释放后只有一次最终 0.27 的降噪；已完成 0.25 降噪后，混合改变强度与曝光，拖动中显示 8 帧调色、没有新降噪事件，释放后约 268ms 发出最终降噪开始信号，最终完整结果约 5.1s。首次场景约 5.75s，包含模型准备、全帧推理和约 277MB 磁盘缓存写入。

200ms 是调度等待，不是降噪完成时间。以上为第一次暂缓调度的测量；当时已经开始的全帧推理仍可能占用前台计算通道。后续修复见下一节。导出通过最终参数和源缓存身份校验，不会把暂缓预览的旧强度误当最终结果。

## 后续：降噪运行期间继续调色

稳定后的原生全帧降噪现由一个后台 `DenoiseTask` 执行，与预览和导出共用现有 `ResourceGovernor`。昂贵计算仍只有一个所有者，后台任务在 tile 边界让出计算通道给预览；曝光、白平衡等变化不取消降噪。换图、改变最终强度、关闭降噪、退出会取消旧任务；旧任务退出后才能启动下一个，不积压多个全帧工作集。

后台结果由预览线程核对源内容、解码来源、策略、照片和最终强度后发布，再用同一个请求 id 重绘最新参数。没有结果时先显示未降噪图像，有结果时继续复用已有降噪。验证后的新结果可以先显示，277MB 的无损磁盘压缩和写入由同一个有界任务继续完成。中断写入仍会清理临时文件，导出不会把临时预览当作最终源。首次模型编译、正在运行的 native tile 和内存准入仍可能带来等待。

空闲的完整质量预览等待新降噪源，避免为临时未降噪/旧强度图像重复建立全帧镜头和显示缓存。吞吐量 benchmark 显式使用 `background_denoise=False`，基线/一次性处理器继续同步处理。

真实 Mac GUI 对照使用同一优化后的图像管线，开启原生降噪，第一块开始后发送 100 次、间隔约 16ms 的曝光变化。同步和后台各测一次：

| 降噪已启动后的交互 | 同步方式 | 后台方式 |
| --- | ---: | ---: |
| 拖动期间图像帧 | 0 | 9 |
| 松手后最终曝光更新 | 3381.5ms | 53.1ms |
| 松手后完整降噪结果 | 3381.5ms | 3602.8ms |
| 拖动期间 GPU 会话重启 | 0 | 0 |

后台方式优先保持调色响应，完整降噪约多等 0.22s。这是一次受控场景的实测，不是所有设备和首次编译场景的延迟保证。数据为 `.test-output/mac-exposure/active-denoise-sync.json` 和 `active-denoise-background.json`。启用镜头校正的 fit 预览只有代理级 corrected 缓存时，导出快路径仍拒绝该缓存并回退完整处理；测试没有把它误记成原生导出缓存命中。

两个完整 4170×6246×3 float32 降噪磁盘结果直接比较，最大绝对差异 **0.0**、超精度门槛通道 **0**，见 `background-denoise-parity.json`。另测已有 0.25 结果时混合改变强度与曝光：拖动中 9 帧复用 0.25，没有新降噪事件或会话重建；释放后约 263ms 启动唯一一次 0.27 推理，最终结果约 4.53s，见 `drag-denoise_strength-background-warm.json`。

## Windows 同机补测

Windows 11 26200、Ryzen 9 9950X3D、约 96GB 内存，机器安装 RX 9070 XT 和 AMD 集显。使用同一 ORT 1.24.4 DirectML、同一 RAF、相同镜头数据库和 DLL、约 3MP 预览；旧版为 `5fff0bc` 的隔离源码，新版为本次代码。常规项目仍取三个不同参数的中位数。普通调色和降噪另做生产会话 profiling，分别记录 45 个和 1 个 DirectML 节点事件，没有 CPU 节点事件，输出均为有限值；这是运行分区证据，不是仅检查 provider 注册。

| 请求 | 修前 ms | 修后 ms | 耗时减少 |
| --- | ---: | ---: | ---: |
| 曝光 | 255.3 | 196.3 | 23.1% |
| 色温 | 252.3 | 200.0 | 20.7% |
| 色调 | 259.4 | 211.3 | 18.5% |
| 高光 | 251.5 | 202.6 | 19.4% |
| 阴影 | 246.9 | 200.6 | 18.7% |
| 饱和度 | 254.1 | 197.8 | 22.2% |
| 对比度 | 247.0 | 203.6 | 17.6% |
| 裁切 | 231.3 | 178.2 | 22.9% |
| 旋转 | 295.3 | 247.4 | 16.2% |
| 细节区域平移 | 262.2 | 235.6 | 10.1% |
| 锐化 | 363.0 | 312.6 | 13.9% |
| LUT | 266.6 | 210.8 | 20.9% |
| Log + LUT | 266.8 | 216.3 | 18.9% |
| 镜头开关 | 252.3 | 201.8 | 20.0% |
| 首次降噪（单次） | 8323.7 | 8113.4 | 2.5% |
| 降噪后普通调色 | 248.0 | 194.5 | 21.6% |

Windows 主要收益来自共同调度改动，原来已启用 DirectML，并没有重复 Mac 的 CoreML 改造收益。26MP 导出融合调色为 836.7→840.4ms，基本持平；TIFF 写盘 404.5→548.9ms，没有 I/O 加速结论。首次 RAW 打开 3996.4→3897.9ms，单次差异不能说明显著加速。首次完整镜头/细节缓存仍约 3.88/3.84s，首次模型准备也未消除。

Windows Qt GUI 以最小化窗口运行、保留有效 OpenGL 上下文，记录实际处理器返回的当前请求结果，未测显示器呈现 FPS。100 次、约 16ms 间隔的曝光变化，在同一新版管线上只对比调度：旧调度中间结果 0、新调度 6，GPU 会话重启均为 0；松手后最终值约 192/315ms，新调度能持续反馈，但这个场景的最后一帧并未变快。

全帧降噪第一块开始后再拖曝光：同步方式中间结果 0、后台方式 5，松手后最终曝光 4431.7→275.2ms；完整降噪结果 4431.7→4906.4ms，约多等 0.47s，会话重启均为 0。两次均完成，无处理错误。这说明 Windows 也改善了降噪中的调节响应，不表示模型计算本身大幅加速。

证据位于 `.test-output/windows-perf/`：`suite-before-lens.json`、`suite-after.json`、`drag-exposure-old.json`、`drag-exposure-current.json`、`active-denoise-sync.json`、`active-denoise-background.json`、`provider-placement.json`。最初的 `suite-before.json` 没有加载同一 Lensfun 库，已作废，未用于上表。以上为同机一次基准中的参数中位数和单次拖动，不是所有 Windows 设备的保证。

## 精度、测试与冻结包

Windows 和 Mac 最新全量测试分别为 **601 passed, 11 skipped**；跳过项主要需要实际特定 GPU/RAW。Windows 64.79s、Mac 27.66s，正常退出；包括后台降噪、提前显示结果、取消/换图/失败、导出缓存身份和空闲 refine 回归。Ruff 检查通过。前一阶段 Mac 原生 GPU 验收另外 **6 passed**，同时检查实际 CoreML profiling、无静默 CPU 替换、有限像素和 `3e-6 + 3e-6 × abs(reference)` 精度门槛；后续调度改动没有更换推理图。

| 原生验收 | 最大绝对误差 | 超门槛通道 |
| --- | ---: | ---: |
| 完整 Bayer NEF | 7.45e-9 | 0 |
| 完整 X-Trans RAF | 1.19e-7 | 0 |
| RCD 四 CFA 相位/黑/近灰/块边缘/相机矩阵 | 1.79e-7 | 0 |
| 普通融合调色 | 7.71e-7 | 0 |
| LUT 调色 | 4.02e-7 | 0 |
| Log+LUT 调色 | 5.07e-7 | 0 |

这些是本项目 GPU 与本项目 CPU 的验收，不是 darktable 逐像素等价测试，也不是本次重新验收所有 CUDA/AMD 硬件。

Mac 退出曾出现 ORT 1.30 的遥测 HTTP 线程递归互斥锁异常。现在从 ONNX 包导入时就关闭遥测，并在每个原生会话构造前再次配置；图像推理和节点 profiling 不受影响。全量测试正常退出。独立的 PyInstaller GUI 测试包已在真实 RAF 上显示 OpenGL 预览，将曝光从 1.6 改到 2.1，并正常退出；其测试 hook 只用于自动操作与记录。

最终可测试 app 使用原始 `RawAlchemy-onedir.spec` 构建，位置在 Mac 的 `/Users/shenmintao/Raw-Alchemy-perf-20260930/.test-output/dist-perf/RawAlchemy.app`。无测试 hook 的正式构建也已确认正常启动，通过针对该可执行文件的 `NSRunningApplication.terminate()` 请求正常退出，退出码为 0；证据为 `.test-output/mac-exposure/plain-app.log` 和 `plain-app.exit`。

另已复制到 Mac 桌面：`/Users/shenmintao/Desktop/RawAlchemy-性能测试-20260930.app`，便于直接试用。原 `/Applications/RawAlchemy.app` 未替换，用户配置和照片旁参数文件未被测试修改。依赖 Lensfun 的下载曾超时，使用同一清单 SHA-256 校验通过的离线压缩包重建。

后台降噪新版使用同一原始 spec 构建，位于 `/Users/shenmintao/Raw-Alchemy-perf-20260930/.test-output/dist-background/RawAlchemy.app`，另复制到 Mac 桌面的 `RawAlchemy-性能测试-20260930-后台降噪.app`。专用冻结 GUI 测试包确认真实 RAF 的 1147×1718 OpenGL 预览、后台降噪期间曝光 1.6→2.1 的更新（约 242ms）、最终 0.25 降噪源和同 id 的最终重绘，无处理错误且退出码为 0。证据为 `.test-output/mac-exposure/frozen-background.json`、截图和退出码文件。

无 hook 的后台降噪正式构建也已正常启动，经精确匹配可执行文件路径请求正常退出，退出码为 0；日志及退出码为 `plain-background.log` 和 `plain-background.exit`。

本次没有配置端口转发：Mac 的 SSH 使用局域网 `192.168.10.53:22` 直连，图像/GPU 处理在本机进行。
