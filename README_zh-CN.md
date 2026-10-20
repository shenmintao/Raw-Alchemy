# 🧪 Raw Alchemy

[English](README.md) | [简体中文](README_zh-CN.md)

> **以数学精度将电影级 LUT 应用于 RAW 照片。**

---

## 🔗 相关项目

### V-Log Alchemy
**[V-Log Alchemy](https://gitee.com/MinQ/V-Log-Alchemy)** - 一系列专门为 V-Log 色彩空间设计的 `.cube` LUT 文件集合。

这些专业级 LUT（包括富士胶片模拟、徕卡风格、ARRI 色彩科学等）可以直接在 Raw Alchemy 中应用，为您的工作流程实现各种创意风格。非常适合希望将电影级调色带入 RAW 图像的摄影师。

---

## 📖 核心理念

许多摄影师和摄像师都依赖创意 LUT (色彩查找表) 来实现特定的视觉风格。然而，一个普遍的痛点是：**将在视频工作流中表现完美的 LUT 应用于 RAW 格式的照片时，色彩往往会出错。**

这个问题源于色彩空间的不匹配。大多数创意 LUT 被设计用于特定的 Log 色彩空间 (例如索尼的 S-Log3/S-Gamut3.Cine 或富士的 F-Log2/F-Gamut)。当您在 Photoshop 或 Lightroom 中打开一张 RAW 照片并直接应用这些 LUT 时，软件默认的 RAW 解码色彩空间与 LUT 期望的输入空间不符，导致色彩和影调的严重偏差。

**Raw Alchemy** 正是为解决这一问题而生。它通过构建一个严谨、自动化的色彩管线，确保任何 LUT 都能被精确地应用于任何 RAW 文件：

1.  **标准化解码**: 项目首先将任何来源的 RAW 文件解码到一个标准化的、广色域中间空间——ProPhoto RGB (线性)。这消除了不同品牌相机自带的色彩科学差异，为所有操作提供了一个统一的起点。
2.  **精确准备 Log 信号**: 接着，它将线性空间的图像数据，精确地转换为目标 LUT 所期望的 Log 格式，例如 `F-Log2` 曲线和 `F-Gamut` 色域。这一步是确保色彩一致性的关键，它完美模拟了相机内部生成 Log 视频信号的过程。
3.  **正确应用 LUT**: 在这个被精确“伪装”好的 Log 图像上应用您的创意 LUT，其色彩和影调表现将与在专业视频软件 (如达芬奇) 中完全一致。
4.  **高位深输出**: 最后，将处理后的图像（保持 Log 编码或已应用 LUT 效果）保存为 16 位 TIFF 文件，最大程度保留动态范围和色彩信息，便于后续在 Photoshop 或 DaVinci Resolve 中进行专业级调色。

通过这个流程，`Raw Alchemy` 打破了 RAW 摄影与专业视频调色之间的壁垒，让摄影师也能享受到电影级别的色彩管理精度。

## 🔄 处理流程

本工具遵循以下精确的色彩转换步骤：

`RAW (相机原生)` -> `ProPhoto RGB (线性)` -> `目标 Log 色域 (线性)` -> `目标 Log 曲线 (例如 F-Log2)` -> `(可选) 创意 LUT` -> `16-bit TIFF`

## ✨ 特性

-   **RAW 转 Linear**: 将 RAW 文件直接解码到 ProPhoto RGB (线性) 工作色彩空间。
-   **Log 转换**: 支持多种相机特定的 Log 格式（F-Log2, S-Log3, LogC4 等）。
-   **LUT 应用**: 支持在转换过程中直接应用 `.cube` 创意 LUT 文件。
-   **曝光控制**: 提供灵活的曝光逻辑：手动曝光覆盖、或智能自动测光（混合、平均、中央重点、高光保护/ETTR）。
-   **高质量输出**: 将最终图像以 16 位 TIFF 文件保存。
-   **技术栈**: 使用 `rawpy` 进行 RAW 解码，并利用 `colour-science` 进行高精度色彩转换。

## 📸 效果示例

| RAW (线性预览) | Log 空间 (V-Log) | 最终效果 (FujiFilm Class-Neg) |
| :---: | :---: | :---: |
| ![RAW](Samples/RAW.jpeg) | ![V-Log](Samples/V-Log.jpeg) | ![Class-Neg](Samples/FujiFilm%20Class-Neg.jpeg) |

#### ✅ 精度验证

与松下 Lumix 实时 LUT (Real-time LUT) 的对比。

| 机内直出 (Real-time LUT) | Raw Alchemy 处理结果 |
| :---: | :---: |
| ![机内直出](Samples/P1013122.jpg) | ![Raw Alchemy](Samples/Converted.jpg) |

## 🚀 快速入门 (推荐)

对于大多数用户，使用 Raw Alchemy 最简单的方式是下载为您操作系统预编译的可执行文件。这无需安装 Python 或任何依赖。

1.  前往 [**Releases**](https://gitee.com/MinQ/Raw-Alchemy/releases) 页面。
2.  下载适用于您系统的最新可执行文件 (例如 `RawAlchemy-vX.Y.Z-windows.exe` 或 `RawAlchemy-vX.Y.Z-linux`)。
3.  运行工具。详情请参阅 [使用方法](#使用方法) 部分。

### macOS：通过 Homebrew 安装

需要已安装 Homebrew，且使用 **Apple Silicon 和 macOS 26 或更新版本**。
独立的 [Homebrew tap 仓库](https://github.com/shenmintao/homebrew-raw-alchemy)
提供用于安装预编译图形应用的 Cask。添加 tap 时只下载安装配置，
安装应用时 Homebrew 再从 Releases 下载应用包。

```bash
brew tap shenmintao/raw-alchemy
brew install --cask shenmintao/raw-alchemy/raw-alchemy-studio
open -a RawAlchemy
```

Homebrew 会将自带 Python 的 `RawAlchemy.app` 安装到“应用程序”。
照片预览、调色和导出均在本机完成，不上传照片，处理时可以断网使用。
安装、升级以及应用内手动“检查更新”需要联网。

当前 macOS 安装包尚未通过 Apple 公证。如果首次启动被 macOS 拦截，
先点“完成”关闭提示，再进入 **系统设置 → 隐私与安全性**，
找到 RawAlchemy 并点 **“仍要打开”**，随后确认 **“打开”**。
详情参阅 [Apple 关于审核并打开应用的说明](https://support.apple.com/zh-cn/102445)。

升级或卸载：

```bash
brew update
brew upgrade --cask shenmintao/raw-alchemy/raw-alchemy-studio
brew uninstall --cask shenmintao/raw-alchemy/raw-alchemy-studio
```

如果之前用显式的 `https://github.com/shenmintao/Raw-Alchemy.git` 地址添加过 tap，
请执行一次以下迁移，移除较大的源码仓库副本并改用独立 tap：

```bash
# 如果尚未安装应用，跳过 uninstall 这一行。
brew uninstall --cask shenmintao/raw-alchemy/raw-alchemy-studio
brew untap shenmintao/raw-alchemy
brew tap shenmintao/raw-alchemy
brew install --cask shenmintao/raw-alchemy/raw-alchemy-studio
```

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

*   **启动 GUI**: 直接运行可执行文件，不带任何参数。详情请参阅下面的教程。
*   **使用 CLI**: 带命令行参数运行可执行文件。

## 🖥️ GUI 教程

图形界面提供了一种直观的方式来处理您的图像。

![GUI 界面截图](Samples/gui_screenshot.png)

#### 1. 选择输入和输出

*   **输入路径**:
    *   点击 **Select File...** (选择文件) 来处理单个 RAW 文件。
    *   点击 **Select Folder...** (选择文件夹) 来处理一个目录下的所有 RAW 文件 (批处理模式)。
*   **输出路径**:
    *   如果处理单个文件，您可以通过点击 **Save As...** (另存为) 来指定确切的输出文件路径。
    *   如果处理文件夹，您必须通过点击 **Select Folder...** (选择文件夹) 来选择一个输出目录。所有处理完的文件将以 `.tif` 扩展名保存在那里。

#### 2. 配置处理设置

*   **Log Space** (Log 空间): 从下拉菜单中选择目标 Log 色彩空间 (例如 `F-Log2`, `S-Log3`)。这是一个必填项。
*   **LUT File (.cube)** (LUT 文件): (可选) 如果您想应用一个创意风格，点击 **Browse...** (浏览) 并选择一个 `.cube` 格式的 LUT 文件。
*   **Custom Lensfun DB** (自定义镜头数据库): (可选) 要使用自定义的镜头数据库 (例如从 LCP 文件生成的)，点击 **Browse...** (浏览) 并选择对应的 `.xml` 文件。

#### 3. 调整曝光

您可以在两种模式之间选择：

*   **Auto** (自动): 这是默认模式。您可以从 **Metering** (测光) 下拉菜单中选择一种测光方式 (例如 `hybrid`, `average` 等)，让程序自动确定最佳曝光。
*   **Manual** (手动): 选择此模式以覆盖自动曝光。然后您可以在 **EV Stops** (曝光档位) 输入框中输入一个特定的 EV 值，或使用滑块来手动调整曝光补偿。

**Metering** (测光) 下拉菜单（在 `Auto` 模式下可用）允许您选择自动曝光的策略：

*   **`matrix` (矩阵测光，默认)**: 高级评价测光模式。它将图像划分为 7x7 网格，并根据每个区域的亮度和位置进行智能加权。它会主动抑制高光、提升阴影，为复杂场景提供最均衡、最可靠的曝光。
*   **`hybrid` (混合测光)**: 一个更简单、更快速的智能模式。它以平衡的平均曝光为目标，但如果检测到高光有过曝风险，会自动降低亮度以保护细节。
*   **`average` (平均测光)**: 计算整个场景的平均亮度并将其调整到中性灰。最适合光线均匀的场景。
*   **`center-weighted` (中央重点测光)**: 优先考虑画面中心的亮度。非常适合人像或主体在中心的拍摄。
*   **`highlight-safe` (高光保护测光, ETTR)**: 在不裁剪高光的前提下，尽可能地提高画面曝光。这种方法能捕捉到最丰富的暗部细节，但可能需要您在后期处理中降低曝光。

#### 4. 开始处理

*   点击 **Start Processing** (开始处理) 按钮。
*   底部的 **Log** (日志) 窗口将实时显示转换的进度和状态。
*   处理完成后，日志中会出现 "processing complete" (处理完成) 的消息。

## 🔧 高级用法：导入 Adobe 镜头配置文件 (LCP)

Raw Alchemy 现在包含一个强大的脚本，用于转换和导入 Adobe LCP 格式的镜头配置文件。LCP 格式被 Adobe Camera Raw 和 DNG Converter 使用，这意味着您可以访问一个更庞大、更及时的镜头数据库。

用于转换的脚本 lensfun-convert-lcp-new 可在 [**Lensfun**](https://gitee.com/MinQ/lensfun/tree/master/apps) 中找到。

**步骤：**

1.  **找到您的 LCP 文件。**
    您可以通过安装免费的 [Adobe DNG Converter](https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html) 来获取它们。这些配置文件通常位于：
    *   **Windows**: `C:\ProgramData\Adobe\CameraRaw\LensProfiles\1.0\`
    *   **macOS**: `/Library/Application Support/Adobe/CameraRaw/LensProfiles/1.0/`

2.  **运行转换脚本。**
    该脚本位于 lensfun 项目的 apps/ 目录下，您需要安装 Python 才能运行它。

3.  脚本将创建一个 `.xml` 文件 (例如 `_lcps.xml`)。您现在可以按照下面章节的说明，在图形界面或命令行中加载此文件。

    转换脚本会保存到默认位置，但您也可以使用其 `--output` 参数将 `.xml` 文件保存到任何您喜欢的地方。更多详情，请使用 `--help` 参数运行该脚本。


## ⌨️ CLI 用法

**注意**: 在 Linux 上，您可能需要先为文件授予执行权限 (例如 `chmod +x ./RawAlchemy-v0.1.0-linux`)。

#### CLI 基本语法

无论您是使用可执行文件还是从源码安装，命令结构都是相同的。

```bash
# 在 Linux 上使用可执行文件 (请替换为您的实际文件名)
./RawAlchemy-v0.1.0-linux [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_TIFF_PATH>

# 在 Windows 上使用可执行文件 (请替换为您的实际文件名)
RawAlchemy-v0.1.0-windows.exe [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_TIFF_PATH>

# 如果从源码安装
raw-alchemy [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_TIFF_PATH>
```

#### 示例 1: 基本 Log 转换

此示例将一个 RAW 文件转换为线性空间，然后应用 F-Log2 曲线，并将结果保存为 TIFF 文件（保持 F-Log2/F-Gamut 空间，适合后续调色）。

```bash
# 将 './RawAlchemy-linux' 替换为您的可执行文件名，或在源码安装时使用 'raw-alchemy'
./RawAlchemy-linux "path/to/your/image.CR3" "path/to/output/image.tiff" --log-space "F-Log2"
```

#### 示例 2: 使用创意 LUT 进行转换

此示例转换 RAW 文件，应用 S-Log3 曲线，然后应用一个创意 LUT (`my_look.cube`)，并保存最终结果。

```bash
# 将 './RawAlchemy-linux' 替换为您的可执行文件名或 'raw-alchemy'
./RawAlchemy-linux "input.ARW" "output.tiff" --log-space "S-Log3" --lut "looks/my_look.cube"
```

#### 示例 3: 手动曝光调整

此示例手动应用 +1.5 档的曝光补偿，它将覆盖任何自动曝光逻辑。

```bash
# 将 './RawAlchemy-linux' 替换为您的可执行文件名或 'raw-alchemy'
./RawAlchemy-linux "input.CR3" "output_bright.tiff" --log-space "S-Log3" --exposure 1.5
```

#### 示例 4: 使用自定义镜头数据库

此示例使用一个自定义的镜头数据库文件，以获得更精确的镜头校正。

```bash
# 将 './RawAlchemy-linux' 替换为您的可执行文件名或 'raw-alchemy'
./RawAlchemy-linux "input.ARW" "output.tiff" --log-space "S-Log3" --custom-lensfun-db "path/to/your/_lcps.xml"
```

## ⚙️ 命令行选项

-   `<INPUT_RAW_PATH>`: (必需) 输入的 RAW 文件路径 (例如 .CR3, .ARW, .NEF)。
-   `<OUTPUT_TIFF_PATH>`: (必需) 输出的 16 位 TIFF 文件的保存路径。

-   `--log-space TEXT`: (必需) 目标 Log 色彩空间。
-   `--exposure FLOAT`: (可选) 手动曝光调整，单位为档 (stops)，例如 -0.5, 1.0。此选项会覆盖所有自动曝光逻辑。
-   `--lut TEXT`: (可选) 在 Log 转换后应用的 `.cube` LUT 文件路径。
-   `--lens-correct / --no-lens-correct`: (可选, 默认: True) 启用或禁用镜头畸变校正。
-   `--custom-lensfun-db TEXT`: (可选) 自定义 Lensfun 数据库 XML 文件的路径 (例如从 LCP 文件生成的)。
-   `--metering TEXT`: (可选, 默认: `hybrid`) 自动曝光测光模式: `average` (平均), `center-weighted` (中央重点), `highlight-safe` (高光保护), 或 `hybrid` (混合)。

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

---

## ☕ 请我喝咖啡

如果 **Raw Alchemy** 节省了你的工作时间，欢迎请作者喝杯咖啡提提神。☕

<details>
<summary><strong>👉 点击展开二维码 (微信/支付宝)</strong></summary>

<br>
<div align="center">
  <img src="Samples/sponsor.png" width="300px">
  <p><sub>感谢支持，随缘打赏</sub></p>
</div>

</details>