# Raw Alchemy

[English](README.md)

---

一个基于 Python 的命令行工具，用于实现高级 RAW 图像处理流程。它旨在将 RAW 文件转换到广色域线性空间 (ProPhoto RGB)，应用相机特定的 Log 曲线，并集成创意 LUT，以实现一个完整且色彩管理精确的工作流。

### 核心理念

许多摄影师和摄像师都依赖创意 LUT (色彩查找表) 来实现特定的视觉风格。然而，一个普遍的痛点是：**将在视频工作流中表现完美的 LUT 应用于 RAW 格式的照片时，色彩往往会出错。**

这个问题源于色彩空间的不匹配。大多数创意 LUT 被设计用于特定的 Log 色彩空间 (例如索尼的 S-Log3/S-Gamut3.Cine 或富士的 F-Log2/F-Gamut)。当您在 Photoshop 或 Lightroom 中打开一张 RAW 照片并直接应用这些 LUT 时，软件默认的 RAW 解码色彩空间与 LUT 期望的输入空间不符，导致色彩和影调的严重偏差。

**Raw Alchemy** 正是为解决这一问题而生。它通过构建一个严谨、自动化的色彩管线，确保任何 LUT 都能被精确地应用于任何 RAW 文件：

1.  **标准化解码**: 项目首先将任何来源的 RAW 文件解码到一个标准化的、广色域中间空间——ProPhoto RGB (线性)。这消除了不同品牌相机自带的色彩科学差异，为所有操作提供了一个统一的起点。
2.  **精确准备 Log 信号**: 接着，它将线性空间的图像数据，精确地转换为目标 LUT 所期望的 Log 格式，例如 `F-Log2` 曲线和 `F-Gamut` 色域。这一步是确保色彩一致性的关键，它完美模拟了相机内部生成 Log 视频信号的过程。
3.  **正确应用 LUT**: 在这个被精确“伪装”好的 Log 图像上应用您的创意 LUT，其色彩和影调表现将与在专业视频软件 (如达芬奇) 中完全一致。
4.  **高位深输出**: 最后，将处理后的图像（保持 Log 编码或已应用 LUT 效果）保存为 16 位 TIFF 文件，最大程度保留动态范围和色彩信息，便于后续在 Photoshop 或 DaVinci Resolve 中进行专业级调色。

通过这个流程，`Raw Alchemy` 打破了 RAW 摄影与专业视频调色之间的壁垒，让摄影师也能享受到电影级别的色彩管理精度。

### 处理流程

本工具遵循以下精确的色彩转换步骤：

`RAW (相机原生)` -> `ProPhoto RGB (线性)` -> `目标 Log 色域 (线性)` -> `目标 Log 曲线 (例如 F-Log2)` -> `(可选) 创意 LUT` -> `16-bit TIFF`

### 特性

-   **RAW 转 Linear**: 将 RAW 文件直接解码到 ProPhoto RGB (线性) 工作色彩空间。
-   **Log 转换**: 支持多种相机特定的 Log 格式（F-Log2, S-Log3, LogC4 等）。
-   **LUT 应用**: 支持在转换过程中直接应用 `.cube` 创意 LUT 文件。
-   **曝光控制**: 提供灵活的曝光逻辑：手动曝光覆盖、或智能自动测光（混合、平均、中央重点、高光保护/ETTR）。
-   **高质量输出**: 将最终图像以 16 位 TIFF 文件保存。
-   **技术栈**: 使用 `rawpy` 进行 RAW 解码，并利用 `colour-science` 进行高精度色彩转换。

### 效果示例

| RAW (线性预览) | Log 空间 (V-Log) | 最终效果 (FujiFilm Class-Neg) |
| :---: | :---: | :---: |
| ![RAW](Samples/RAW.jpeg) | ![V-Log](Samples/V-Log.jpeg) | ![Class-Neg](Samples/FujiFilm%20Class-Neg.jpeg) |

#### 精度验证

与松下 Lumix 实时 LUT (Real-time LUT) 的对比。

| 机内直出 (Real-time LUT) | Raw Alchemy 处理结果 |
| :---: | :---: |
| ![机内直出](Samples/P1013122.jpg) | ![Raw Alchemy](Samples/Converted.jpg) |

### 安装

安装 Raw Alchemy：

```bash
# 克隆本仓库
git clone https://github.com/shenmintao/raw-alchemy.git
cd raw-alchemy

# 安装工具
pip install .
```

*注意：本项目依赖特定版本的 `rawpy` 和 `colour-science`。*

### 使用方法

通过 `raw-alchemy` 命令来使用该工具。

#### 基本语法

```bash
raw-alchemy [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_TIFF_PATH>
```

#### 示例 1: 基本 Log 转换

此示例将一个 RAW 文件转换为线性空间，然后应用 F-Log2 曲线，并将结果保存为 TIFF 文件（保持 F-Log2/F-Gamut 空间，适合后续调色）。

```bash
raw-alchemy "path/to/your/image.CR3" "path/to/output/image.tiff" --log-space "F-Log2"
```

#### 示例 2: 使用创意 LUT 进行转换

此示例转换 RAW 文件，应用 S-Log3 曲线，然后应用一个创意 LUT (`my_look.cube`)，并保存最终结果。

```bash
raw-alchemy "input.ARW" "output.tiff" --log-space "S-Log3" --lut "looks/my_look.cube"
```

#### 示例 3: 手动曝光调整

此示例手动应用 +1.5 档的曝光补偿，它将覆盖任何自动曝光逻辑。

```bash
raw-alchemy "input.CR3" "output_bright.tiff" --log-space "S-Log3" --exposure 1.5
```

### 命令行选项

-   `<INPUT_RAW_PATH>`: (必需) 输入的 RAW 文件路径 (例如 .CR3, .ARW, .NEF)。
-   `<OUTPUT_TIFF_PATH>`: (必需) 输出的 16 位 TIFF 文件的保存路径。

-   `--log-space TEXT`: (必需) 目标 Log 色彩空间。
-   `--exposure FLOAT`: (可选) 手动曝光调整，单位为档 (stops)，例如 -0.5, 1.0。此选项会覆盖所有自动曝光逻辑。
-   `--lut TEXT`: (可选) 在 Log 转换后应用的 `.cube` LUT 文件路径。
-   `--lens-correct / --no-lens-correct`: (可选, 默认: True) 启用或禁用镜头畸变校正。
-   `--metering TEXT`: (可选, 默认: `hybrid`) 自动曝光测光模式: `average` (平均), `center-weighted` (中央重点), `highlight-safe` (高光保护), 或 `hybrid` (混合)。

### 支持的 Log 空间

`--log-space` 选项支持以下值:
-   `F-Log`
-   `F-Log2`
-   `F-Log2C`
-   `V-Log`
-   `N-Log`
-   `Canon Log 2`
-   `Canon Log 3`
-   `S-Log3`
-   `S-Log3.Cine`
-   `LogC3`
-   `LogC4`
-   `Log3G10`