# Raw Alchemy 全链路审计与重构记录（2026-07-14）

## 结论

项目的主架构方向是正确的：RAW 解码、统一 `Op` 管线、proxy/quality-base/ROI
三级预览、分层缓存、ONNX 推理和独立导出链已经形成完整闭环。本轮不需要推倒重写。
收益最大的工作是继续收紧“状态键一致性”和“数组所有权”，消除 CPU 化后遗留的
历史 GPU 下载/上传语义，以及修复 UI scopes 的无效计算。

本轮基线为工作区现有 Phase 7 / bounded-working-set 改动；不回滚已有修改。

## 当前链路

```text
文件夹扫描
  -> 缩略图提取/磁盘缓存
  -> RAW 解码（RawSpeed/rawpy + ONNX demosaic）
  -> linear ProPhoto 缓存 + proxy
  -> 可选 RGB denoise
  -> 可选 Lensfun 校正
  -> geometry / perspective / crop / ROI
  -> exposure / WB / highlight-shadow / saturation-contrast
  -> log + LUT，或 sRGB/PQ 输出
  -> uint8 预览 / 8-16bit 导出
  -> OpenGL 显示 + histogram/waveform
```

预览和导出共享 `build_op_list`，这是保证质量一致性的核心；worker 负责源选择、
状态缓存、取消和渐进细化，executor 负责纯操作序列与前缀缓存。

## 发现并修复的问题

### 1. 降噪强度未进入镜头校正状态键（质量，严重）

旧状态键只记录 `denoise_enabled`。从强度 0.25 切到 0.50 时，输出缓存虽然 miss，
但 Lensfun 校正仍可能直接返回 0.25 对应的旧 corrected frame；自动测光也可能复用
旧强度的 EV。首次 proxy 降噪预览还可能对未降噪源测光，随后缓存命中时发生亮度跳变。

修复：统一 denoise strength/signature，并将其纳入 full/proxy lens key、测光 key、
导出状态验证和恢复逻辑；denoise callback 在 exposure 前发布正确 corrected source。

### 2. Host-backed pipeline 仍执行整帧“下载/上传”（速度/内存）

`GpuImage` 已经是 numpy host buffer，但 exposure metering、denoise/lens callback、
log/LUT fallback 和 ONNX fused grade 仍调用 `to_numpy()`，之后又 `upload()`。
12MP 单次 download+upload 微基准约 90ms；融合调色输入复制和输出复制会叠加。

修复：只读 callback/测光直接读取工作 buffer；新增显式 `GpuImage.adopt()`，让
ONNX/colour 新生成且独占的输出转移所有权，不再复制。cache-owned 数组仍使用 copy，
防止后续 in-place op 污染缓存。

### 3. ROI resize、透视和 90° 几何存在完整中间帧（速度/峰值内存）

- quality-base/ROI：`cv2.resize -> temporary -> upload copy`
- perspective：`cv2.warpPerspective -> temporary -> np.copyto`
- rotate/flip：strided view -> `ascontiguousarray` -> `np.copyto`

修复：OpenCV 直接写入池化 destination；`np.copyto` 直接消费旋转/翻转视图。

### 4. Waveform 计算了最终不会使用的列（UI 响应）

旧实现只显示每第 N 列，却先把所有列转换为 float/luma，再逐列执行 `np.add.at`。
修复后先做横纵采样，再用一次 `bincount` 构建二维 waveform。Histogram 的 uint8
常用路径也直接统计原始字节，不再构造 sampled float RGB 和三个 channel copy。

### 5. LUT 文件缓存无法感知文件被修改（质量/工作流）

旧缓存只以路径为键。同路径覆盖 `.cube` 后，应用仍使用旧 LUT，直到进程重启。
修复：缓存键加入 `mtime_ns + size`，并在解析时一次性准备 executor/ONNX 所需的
float32 contiguous table/domain。

### 6. 批量目录按扩展名重复 `listdir`（启动速度/确定性）

修复：单次 `os.scandir` 完成过滤，忽略同名目录，并按名称稳定排序。

### 7. 导出量化使用截断（输出质量）

8/16bit 导出由向下截断改为分块 round-to-nearest，与预览量化规则一致，消除
约 0.5 LSB 的系统性负偏差。24MP uint16 量化中位耗时约增加 15-20ms，但不再产生
完整 float 临时帧，质量收益发生在所有 JPEG/TIFF/HEIF/DNG 输出。

## 同机合成微基准

Windows / Python 3.12，同一进程、同一数据规模；表中为 3-5 次运行中位数。

| 热点 | 数据规模 | 改前 | 改后 | 变化 |
|---|---:|---:|---:|---:|
| Histogram | 24MP uint8 | 99.8ms | 52.7ms | -47% |
| Waveform | 24MP uint8 | 927.7ms | 82.3ms | -91% |
| Rotate 90° + flip | 12MP float32 | 302.4ms | 165.6ms | -45% |
| Perspective identity | 12MP float32 | 72.6ms | 21.0ms | -71% |
| Quality-base resize handoff | 12MP -> 3.1MP | 18.6ms | 10.5ms | -44% |
| Fused-grade buffer handoff模拟 | 12MP float32 | 163.1ms | 78.5ms | -52% |

最后一项只测 host buffer 交接，不包含 ONNX provider 本身的推理时间；实际收益取决于
CUDA/DirectML/CPU provider，但固定减少两次整帧内存复制。

## 验证

- `pytest -q`: **268 passed**（原基线 258，新增 10 项回归测试）
- `uv run --no-sync ruff check src tests`: passed
- `python -m compileall -q src tests`: passed
- 数值回归覆盖：统一 pipeline、ROI、fused grade、色彩数学、DNG/HEIF、缓存与取消。

pytest 结束时 Windows 对 `%TEMP%/pytest-current` 的清理会报告一次 `WinError 5`；
测试进程退出码仍为 0，属于本机临时目录权限问题，不是项目失败。

## 后续优先级

1. 用 45MP/61MP 真机 RAW 做 GUI 长会话验收：连续浏览、滑杆、100% ROI、导出。
2. 记录真实 provider 下 decode / denoise / lens / grade / output 的分段 P50/P95，避免
   后续优化只依赖合成微基准。
3. `ImageProcessor._do_process`、Lensfun wrapper 和 thumbnail extractor 仍是高复杂度
   函数；后续拆分应围绕状态对象和纯函数边界进行，不改变已验证的缓存所有权。
4. `GpuImage` 名称已经与 host-backed 实现不符。可在下一次破坏性版本中引入
   `ImageBuffer` 新名并保留兼容别名，降低维护误判。
5. DNG 的 ColorMatrix/LinearRaw 元数据语义需要单独做 Adobe/Resolve/ExifTool
   互操作验证；这属于格式兼容性项目，不应与性能重构混在一起。
