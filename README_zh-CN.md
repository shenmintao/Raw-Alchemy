# 🧪 Raw Alchemy Studio

[English](README.md) | [简体中文](README_zh-CN.md)

> **一个快速选片/调色工具，以数学精度将电影级 LUT 应用于 RAW 照片。**

---

## 🔗 相关项目

### V-Log Alchemy
**[V-Log Alchemy](https://gitee.com/MinQ/V-Log-Alchemy)** - 一系列专门为 V-Log 色彩空间设计的 `.cube` LUT 文件集合。

这些专业级 LUT（包括富士胶片模拟、徕卡风格、ARRI 色彩科学等）可以直接在 Raw Alchemy Studio 中应用，为您的工作流程实现各种创意风格。非常适合希望将电影级调色带入 RAW 图像的摄影师。

---

## 📖 核心理念

许多摄影师和摄像师都依赖创意 LUT (色彩查找表) 来实现特定的视觉风格。然而，一个普遍的痛点是：**将在视频工作流中表现完美的 LUT 应用于 RAW 格式的照片时，色彩往往会出错。**

这个问题源于色彩空间的不匹配。大多数创意 LUT 被设计用于特定的 Log 色彩空间 (例如索尼的 S-Log3/S-Gamut3.Cine 或富士的 F-Log2/F-Gamut)。当您在 Photoshop 或 Lightroom 中打开一张 RAW 照片并直接应用这些 LUT 时，软件默认的 RAW 解码色彩空间与 LUT 期望的输入空间不符，导致色彩和影调的严重偏差。

**Raw Alchemy Studio** 正是为解决这一问题而生。它通过构建一个严谨、自动化的色彩管线，确保任何 LUT 都能被精确地应用于任何 RAW 文件：

1.  **标准化解码**: 项目首先将任何来源的 RAW 文件解码到一个标准化的、广色域中间空间——ProPhoto RGB (线性)。这消除了不同品牌相机自带的色彩科学差异，为所有操作提供了一个统一的起点。
2.  **精确准备 Log 信号**: 接着，它将线性空间的图像数据，精确地转换为目标 LUT 所期望的 Log 格式，例如 `F-Log2` 曲线和 `F-Gamut` 色域。这一步是确保色彩一致性的关键，它完美模拟了相机内部生成 Log 视频信号的过程。
3.  **正确应用 LUT**: 在这个被精确“伪装”好的 Log 图像上应用您的创意 LUT，其色彩和影调表现将与在专业视频软件 (如达芬奇) 中完全一致。
4.  **高位深输出**: 最后，将处理后的图像（保持 Log 编码或已应用 LUT 效果）保存为 16 位 TIFF、HEIF 或 JPEG 文件，最大程度保留动态范围和色彩信息，便于后续专业级调色。

通过这个流程，`Raw Alchemy Studio` 打破了 RAW 摄影与专业视频调色之间的壁垒，让摄影师也能享受到电影级别的色彩管理精度。

## 🔄 处理流程

本工具遵循以下精确的色彩转换步骤：

`RAW (相机原生)` -> `ProPhoto RGB (线性)` -> `镜头校正` -> `曝光与白平衡` -> `目标 Log 色域 (线性)` -> `目标 Log 曲线 (例如 F-Log2)` -> `(可选) 创意 LUT` -> `输出 (TIFF/HEIF/JPG)`

## ✨ 特性

-   **RAW 转 Linear**: 将 RAW 文件直接解码到 ProPhoto RGB (线性) 工作色彩空间。
-   **Log 转换**: 支持多种相机特定的 Log 格式（F-Log2, S-Log3, LogC4 等）。
-   **LUT 应用**: 支持在转换过程中直接应用 `.cube` 创意 LUT 文件。
-   **高级曝光控制**:
    -   **自动测光**: 矩阵测光、混合测光、平均测光、中央重点测光、高光保护 (ETTR)。
    -   **手动覆盖**: 精确的 EV 曝光值调整。
-   **基础调整**: 白平衡 (色温/色调)、饱和度、对比度、高光和阴影微调。
-   **镜头校正**: 使用 Lensfun 自动进行镜头畸变校正，支持加载自定义 LCP 转换数据库。
-   **批量处理**: 高效处理整个文件夹的 RAW 图像。
-   **多格式输出**: 支持保存为 16 位 TIFF、HEIF (10位) 或 JPEG。

## 📸 效果示例

| RAW (线性预览) | Log 空间 (V-Log) | 最终效果 (FujiFilm Class-Neg) |
| :---: | :---: | :---: |
| ![RAW](Samples/RAW.jpeg) | ![V-Log](Samples/V-Log.jpeg) | ![Class-Neg](Samples/FujiFilm%20Class-Neg.jpeg) |

#### ✅ 精度验证

与松下 Lumix 实时 LUT (Real-time LUT) 的对比。

| 机内直出 (Real-time LUT) | Raw Alchemy Studio 处理结果 |
| :---: | :---: |
| ![机内直出](Samples/P1013122.jpg) | ![Raw Alchemy](Samples/Converted.jpg) |

## 🚀 快速入门 (推荐)

对于大多数用户，使用 Raw Alchemy Studio 最简单的方式是下载为您操作系统预编译的可执行文件。这无需安装 Python 或任何依赖。

1.  前往 [**Releases**](https://gitee.com/MinQ/Raw-Alchemy/releases) 页面。
2.  下载适用于您系统的最新可执行文件 (例如 `RawAlchemy-vX.Y.Z-windows.exe` 或 `RawAlchemy-vX.Y.Z-linux`)。
3.  运行工具。详情请参阅 [使用方法](#使用方法) 部分。

## 💻 从源码安装 (开发者选项)

如果您希望从源码安装本项目，可以按照以下步骤操作：

```bash
# 克隆本仓库
git clone https://github.com/shenmintao/raw-alchemy.git
cd raw-alchemy

# 安装工具及其依赖
pip install .
```

*注意：本项目依赖特定版本的 `rawpy` 和 `colour-science`。*

## 🛠️ 使用方法

可执行文件同时提供了图形用户界面 (GUI) 和命令行界面 (CLI)。

*   **启动 GUI**: 直接运行可执行文件，不带任何参数。
*   **使用 CLI**: 带命令行参数运行可执行文件。

## 🖥️ GUI 教程

图形界面提供了一种直观的方式来处理您的图像。

![GUI 界面截图](Samples/gui_screenshot.png)

#### 1. 库与预览
*   **导入**: 点击 **Open Folder** (打开文件夹) 加载包含 RAW 图像的目录。
*   **导航**: 使用左侧的图库或键盘方向键进行切换。
*   **对比**: 长按图像（或按住空格键）可即时对比处理结果与原始 RAW 图像。
*   **标记**: 使用 "Tag" (标记) 按钮标记图像以便批量导出。

#### 2. 检查器面板 (右侧)

*   **直方图**: 实时 RGB 直方图，监控曝光和色彩分布。
*   **曝光**:
    *   **Auto** (自动): 选择智能测光模式，如 `Matrix` (矩阵/均衡)、`Highlight-Safe` (高光保护) 或 `Hybrid` (混合)。
    *   **Manual** (手动): 使用滑块精确调整 EV (曝光值)。
*   **色彩管理**:
    *   **Log Space**: 选择目标 Log 空间 (例如 `F-Log2`, `S-Log3`) 以匹配您的 LUT。
    *   **LUT**: 加载 `.cube` 文件以应用创意风格。
*   **镜头校正**: 开启/关闭畸变校正，或加载自定义 Lensfun 数据库 XML。
*   **调整**: 微调白平衡 (色温/色调)、饱和度、对比度、高光和阴影。

#### 3. 导出
*   **Export Current** (导出当前): 保存当前选中的图像。
*   **Export All Marked** (导出所有标记): 批量处理并保存所有已标记的图像。
*   **格式**: 支持选择 JPEG, HEIF 或 TIFF。

## 🔧 高级用法：导入 Adobe 镜头配置文件 (LCP)

Raw Alchemy Studio 包含一个脚本，用于转换和导入 Adobe LCP 镜头配置文件。

安装后，转换脚本 `lensfun-convert-lcp` 可以在 `src/raw_alchemy/vendor/lensfun` 目录或源码中找到。

**步骤：**

1.  **找到您的 LCP 文件** (通常位于 Adobe 的 ProgramData 或 Library 文件夹中)。
2.  **运行转换脚本**:
    ```bash
    python src/raw_alchemy/vendor/lensfun/win-x86_64/lensfun-convert-lcp "path/to/lcp/folder" --output "my_lenses.xml"
    ```
3.  **加载 XML**: 在 GUI 中，使用 "Custom Lensfun Database" (自定义 Lensfun 数据库) 的浏览按钮加载您生成的 XML 文件。

## ⌨️ CLI 用法

**注意**: 在 Linux 上，您可能需要先为文件授予执行权限 (例如 `chmod +x ./RawAlchemy-v0.1.0-linux`)。

#### CLI 基本语法

```bash
# 使用可执行文件
RawAlchemy-v0.1.0-windows.exe [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_PATH>
```

#### 示例

**1. 基本 Log 转换**
```bash
./RawAlchemy-linux "image.CR3" "output.tiff" --log-space "F-Log2"
```

**2. 应用创意 LUT 并输出 HEIF**
```bash
./RawAlchemy-linux "input.ARW" "output.heif" --log-space "S-Log3" --lut "looks/my_look.cube" --format heif
```

**3. 批量处理**
```bash
./RawAlchemy-linux "C:/Photos/2024" "C:/Photos/Processed" --log-space "LogC4" --jobs 8
```

## ⚙️ 命令行选项

-   `<INPUT_RAW_PATH>`: (必需) 输入 RAW 文件或文件夹路径。
-   `<OUTPUT_PATH>`: (必需) 输出文件或文件夹路径。
-   `--log-space TEXT`: (必需) 目标 Log 色彩空间。
-   `--lut TEXT`: (可选) `.cube` LUT 文件路径。
-   `--exposure FLOAT`: (可选) 手动曝光调整 (档)。
-   `--metering TEXT`: (可选) 自动曝光模式: `matrix`, `hybrid`, `average`, `center-weighted`, `highlight-safe`。
-   `--format TEXT`: (可选) 输出格式: `tif` (默认), `heif`, `jpg`。
-   `--jobs INTEGER`: (可选) 批量处理的并发任务数。
-   `--lens-correct / --no-lens-correct`: (可选) 启用/禁用镜头校正。
-   `--custom-lensfun-db TEXT`: (可选) 自定义 Lensfun XML 路径。

## 📋 支持的 Log 空间

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
-   `Arri LogC3`
-   `Arri LogC4`
-   `Log3G10`
-   `D-Log`