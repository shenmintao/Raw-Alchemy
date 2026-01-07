# 🧪 Raw Alchemy Studio

[English](README.md) | [简体中文](README_zh-CN.md)

> **A fast culling/grading tool that applies cinematic LUTs to RAW photos with mathematical precision.**

---

## 🔗 Related Projects

### V-Log Alchemy
**[V-Log Alchemy](https://github.com/shenmintao/V-Log-Alchemy)** - A curated collection of `.cube` LUT files specifically designed for V-Log color space.

These professional-grade LUTs (including Fujifilm film simulations, Leica looks, ARRI color science, and more) can be directly applied in Raw Alchemy Studio to achieve various creative looks for your workflow. Perfect for photographers who want to bring cinematic color grading to their RAW images.

---

## 📖 Core Philosophy

Many photographers and videographers rely on creative LUTs (Look Up Tables) to achieve specific visual styles. However, a common pain point is: **When applying LUTs that perform perfectly in video workflows to RAW photos, colors often go wrong.**

This issue stems from color space mismatches. Most creative LUTs are designed for specific Log color spaces (e.g., Sony S-Log3/S-Gamut3.Cine or Fujifilm F-Log2/F-Gamut). When you open a RAW photo in Photoshop or Lightroom and directly apply these LUTs, the software's default RAW decoding color space does not match the LUT's expected input space, leading to severe color and tonal deviations.

**Raw Alchemy Studio** was born to solve this problem. It builds a rigorous, automated color pipeline to ensure any LUT can be accurately applied to any RAW file:

1.  **Standardized Decoding**: The project first decodes RAW files from any source into a standardized, wide-gamut intermediate space — ProPhoto RGB (Linear). This eliminates color science differences inherent in different camera brands, providing a unified starting point for all operations.
2.  **Precise Log Signal Preparation**: Next, it precisely converts the linear image data into the Log format expected by the target LUT, such as `F-Log2` curve and `F-Gamut` color space. This step is critical for ensuring color consistency, perfectly simulating the process of generating Log video signals inside a camera.
3.  **Correct LUT Application**: Applying your creative LUT on this accurately "disguised" Log image results in color and tonal performance identical to that in professional video software (like DaVinci Resolve).
4.  **High-Bit Depth Output**: Finally, the processed image (keeping Log encoding or with LUT effects applied) is saved as a 16-bit TIFF, HEIF, or JPEG file, maximizing dynamic range and color information retention for professional grading.

Through this process, `Raw Alchemy Studio` breaks down the barrier between RAW photography and professional video color grading, allowing photographers to enjoy cinema-level color management precision.

## 🔄 Process Flow

This tool follows these precise color conversion steps:

`RAW (Camera Native)` -> `ProPhoto RGB (Linear)` -> `Lens Correction` -> `Exposure & WB` -> `Target Log Gamut (Linear)` -> `Target Log Curve (e.g. F-Log2)` -> `(Optional) Creative LUT` -> `Output (TIFF/HEIF/JPG)`

## ✨ Features

-   **RAW to Linear**: Decodes RAW files directly into ProPhoto RGB (Linear) working color space.
-   **Log Conversion**: Supports various camera-specific Log formats (F-Log2, S-Log3, LogC4, etc.).
-   **LUT Application**: Supports applying `.cube` creative LUT files directly during conversion.
-   **Advanced Exposure Control**:
    -   **Auto Metering**: Matrix, Hybrid, Average, Center-Weighted, Highlight-Safe (ETTR).
    -   **Manual Override**: Precise EV adjustment.
-   **Basic Adjustments**: White Balance (Temp/Tint), Saturation, Contrast, Highlights, and Shadows.
-   **Lens Correction**: Automatic lens distortion correction using Lensfun, with support for custom LCP-converted databases.
-   **Batch Processing**: Efficiently process entire folders of RAW images.
-   **Multi-Format Output**: Save as 16-bit TIFF, HEIF (10-bit), or JPEG.

## 📸 Samples

| RAW (Linear Preview) | Log Space (V-Log) | Final Look (FujiFilm Class-Neg) |
| :---: | :---: | :---: |
| ![RAW](Samples/RAW.jpeg) | ![V-Log](Samples/V-Log.jpeg) | ![Class-Neg](Samples/FujiFilm%20Class-Neg.jpeg) |

#### ✅ Accuracy Verification

Comparison with Panasonic Lumix Real-time LUT.

| In-Camera Real-time LUT | Raw Alchemy Studio Processing |
| :---: | :---: |
| ![In-Camera](Samples/P1013122.jpg) | ![Raw Alchemy](Samples/Converted.jpg) |

## 🚀 Getting Started (Recommended)

For most users, the easiest way to use Raw Alchemy Studio is to download the pre-compiled executable. This does not require installing Python or any dependencies.

1.  Go to the [**Releases**](https://github.com/shenmintao/raw-alchemy/releases) page.
2.  Download the latest executable for your system (e.g., `RawAlchemy-vX.Y.Z-windows.exe` or `RawAlchemy-vX.Y.Z-linux`).
3.  Run the tool. See the [Usage](#usage) section for details.

## 💻 Installation from Source (For Developers)

If you want to install the project from source, you can follow these steps:

```bash
# Clone the repository
git clone https://github.com/shenmintao/raw-alchemy.git
cd raw-alchemy

# Install the tool and its dependencies
pip install .
```

*Note: This project depends on specific versions of `rawpy` and `colour-science`.*

## 🛠️ Usage

The executable provides both a Graphical User Interface (GUI) and a Command-Line Interface (CLI).

*   **To launch the GUI**: Simply run the executable without any arguments.
*   **To use the CLI**: Run the executable with command-line arguments.

## 🖥️ GUI Tutorial

The graphical interface provides an intuitive way to process your images.

![Image of GUI](Samples/gui_screenshot.png)

#### 1. Library & Preview
*   **Import**: Click **Open Folder** to load a directory of RAW images.
*   **Navigation**: Use the gallery on the left or arrow keys to navigate.
*   **Compare**: Click and hold the image (or press Spacebar) to instantly compare the processed result with the original RAW.
*   **Marking**: Use the "Tag" button to mark images for batch export.

#### 2. Inspector Panel (Right Side)

*   **Histogram**: Real-time RGB histogram to monitor exposure and color distribution.
*   **Exposure**:
    *   **Auto**: Choose from intelligent metering modes like `Matrix` (balanced), `Highlight-Safe` (protects highlights), or `Hybrid`.
    *   **Manual**: Use the slider to adjust Exposure Value (EV).
*   **Color Management**:
    *   **Log Space**: Select the target Log space (e.g., `F-Log2`, `S-Log3`) to match your LUTs.
    *   **LUT**: Load a `.cube` file to apply a creative look.
*   **Lens Correction**: Toggle distortion correction or load a custom Lensfun database XML.
*   **Adjustments**: Fine-tune White Balance (Temp/Tint), Saturation, Contrast, Highlights, and Shadows.

#### 3. Export
*   **Export Current**: Save the currently selected image.
*   **Export All Marked**: Batch process and save all marked images.
*   **Formats**: Choose between JPEG, HEIF, or TIFF.

## 🔧 Advanced Usage: Importing Adobe Lens Profiles (LCP)

Raw Alchemy Studio includes a script to convert and import Adobe LCP lens profiles.

The conversion script, `lensfun-convert-lcp`, can be found in the `src/raw_alchemy/vendor/lensfun` directory after installation or in the source.

**Steps:**

1.  **Locate your LCP files** (typically in Adobe's ProgramData or Library folders).
2.  **Run the conversion script**:
    ```bash
    python src/raw_alchemy/vendor/lensfun/win-x86_64/lensfun-convert-lcp "path/to/lcp/folder" --output "my_lenses.xml"
    ```
3.  **Load the XML**: In the GUI, use the "Custom Lensfun Database" browse button to load your generated XML file.

## ⌨️ CLI Usage

**Note**: On Linux, you may need to make the file executable first (e.g., `chmod +x ./RawAlchemy-v0.1.0-linux`).

#### CLI Basic Syntax

```bash
# Using the executable
RawAlchemy-v0.1.0-windows.exe [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_PATH>
```

#### Examples

**1. Basic Log Conversion**
```bash
./RawAlchemy-linux "image.CR3" "output.tiff" --log-space "F-Log2"
```

**2. With Creative LUT and HEIF Output**
```bash
./RawAlchemy-linux "input.ARW" "output.heif" --log-space "S-Log3" --lut "looks/my_look.cube" --format heif
```

**3. Batch Processing**
```bash
./RawAlchemy-linux "C:/Photos/2024" "C:/Photos/Processed" --log-space "LogC4" --jobs 8
```

## ⚙️ Command Line Options

-   `<INPUT_RAW_PATH>`: (Required) Input RAW file or folder.
-   `<OUTPUT_PATH>`: (Required) Output file or folder.
-   `--log-space TEXT`: (Required) Target Log color space.
-   `--lut TEXT`: (Optional) Path to a `.cube` LUT file.
-   `--exposure FLOAT`: (Optional) Manual exposure adjustment in stops.
-   `--metering TEXT`: (Optional) Auto exposure mode: `matrix`, `hybrid`, `average`, `center-weighted`, `highlight-safe`.
-   `--format TEXT`: (Optional) Output format: `tif` (default), `heif`, `jpg`.
-   `--jobs INTEGER`: (Optional) Number of concurrent jobs for batch processing.
-   `--lens-correct / --no-lens-correct`: (Optional) Enable/disable lens correction.
-   `--custom-lensfun-db TEXT`: (Optional) Path to custom Lensfun XML.

## 📋 Supported Log Spaces

`--log-space` supports the following values:
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