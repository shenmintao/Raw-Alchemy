# 🧪 Raw Alchemy

[English](README.md) | [简体中文](README_zh-CN.md)

> **Apply cinematic LUTs to RAW photos with mathematical precision.**

---

## 🔗 Related Projects

### V-Log Alchemy
**[V-Log Alchemy](https://github.com/shenmintao/V-Log-Alchemy)** - A curated collection of `.cube` LUT files specifically designed for V-Log color space.

These professional-grade LUTs (including Fujifilm film simulations, Leica looks, ARRI color science, and more) can be directly applied in Raw Alchemy to achieve various creative looks for your workflow. Perfect for photographers who want to bring cinematic color grading to their RAW images.

---

## 📖 Core Philosophy

Many photographers and videographers rely on creative LUTs (Look Up Tables) to achieve specific visual styles. However, a common pain point is: **When applying LUTs that perform perfectly in video workflows to RAW photos, colors often go wrong.**

This issue stems from color space mismatches. Most creative LUTs are designed for specific Log color spaces (e.g., Sony S-Log3/S-Gamut3.Cine or Fujifilm F-Log2/F-Gamut). When you open a RAW photo in Photoshop or Lightroom and directly apply these LUTs, the software's default RAW decoding color space does not match the LUT's expected input space, leading to severe color and tonal deviations.

**Raw Alchemy** was born to solve this problem. It builds a rigorous, automated color pipeline to ensure any LUT can be accurately applied to any RAW file:

1.  **Standardized Decoding**: The project first decodes RAW files from any source into a standardized, wide-gamut intermediate space — ProPhoto RGB (Linear). This eliminates color science differences inherent in different camera brands, providing a unified starting point for all operations.
2.  **Precise Log Signal Preparation**: Next, it precisely converts the linear image data into the Log format expected by the target LUT, such as `F-Log2` curve and `F-Gamut` color space. This step is critical for ensuring color consistency, perfectly simulating the process of generating Log video signals inside a camera.
3.  **Correct LUT Application**: Applying your creative LUT on this accurately "disguised" Log image results in color and tonal performance identical to that in professional video software (like DaVinci Resolve).
4.  **High-Bit Depth Output**: Finally, the processed image (keeping Log encoding or with LUT effects applied) is saved as a 16-bit TIFF file, maximizing dynamic range and color information retention for professional grading in Photoshop or DaVinci Resolve.

Through this process, `Raw Alchemy` breaks down the barrier between RAW photography and professional video color grading, allowing photographers to enjoy cinema-level color management precision.

## 🔄 Process Flow

This tool follows these precise color conversion steps:

`RAW (Camera Native)` -> `ProPhoto RGB (Linear)` -> `Target Log Gamut (Linear)` -> `Target Log Curve (e.g. F-Log2)` -> `(Optional) Creative LUT` -> `16-bit TIFF`

## ✨ Features

-   **RAW to Linear**: Decodes RAW files directly into ProPhoto RGB (Linear) working color space.
-   **Log Conversion**: Supports various camera-specific Log formats (F-Log2, S-Log3, LogC4, etc.).
-   **LUT Application**: Supports applying `.cube` creative LUT files directly during conversion.
-   **Exposure Control**: Provides flexible exposure logic: Manual exposure override, or smart auto-metering (Hybrid, Average, Center-Weighted, Highlight-Safe/ETTR).
-   **High Quality Output**: Saves the final image as a 16-bit TIFF file.
-   **Tech Stack**: Uses `rawpy` for RAW decoding and utilizes `colour-science` for high-precision color transformations.

## 📸 Samples

| RAW (Linear Preview) | Log Space (V-Log) | Final Look (FujiFilm Class-Neg) |
| :---: | :---: | :---: |
| ![RAW](Samples/RAW.jpeg) | ![V-Log](Samples/V-Log.jpeg) | ![Class-Neg](Samples/FujiFilm%20Class-Neg.jpeg) |

#### ✅ Accuracy Verification

Comparison with Panasonic Lumix Real-time LUT.

| In-Camera Real-time LUT | Raw Alchemy Processing |
| :---: | :---: |
| ![In-Camera](Samples/P1013122.jpg) | ![Raw Alchemy](Samples/Converted.jpg) |

## 🚀 Getting Started (Recommended)

For most users, the easiest way to use Raw Alchemy is to download the pre-compiled executable. This does not require installing Python or any dependencies.

1.  Go to the [**Releases**](https://github.com/shenmintao/raw-alchemy/releases) page.
2.  Download the latest executable for your system (e.g., `RawAlchemy-vX.Y.Z-windows.exe` or `RawAlchemy-vX.Y.Z-linux`).
3.  Run the tool. See the [Usage](#usage) section for details.

### macOS: install with Homebrew

Requires Homebrew, **Apple Silicon and macOS 26 or later**. The dedicated
[Homebrew tap](https://github.com/shenmintao/homebrew-raw-alchemy) provides a
Cask for the pre-built graphical application. Tapping downloads only the
installation metadata; Homebrew downloads the app from Releases when installing.

```bash
brew tap shenmintao/raw-alchemy
brew install --cask shenmintao/raw-alchemy/raw-alchemy-studio
open -a RawAlchemy
```

Homebrew installs `RawAlchemy.app` in Applications with Python bundled.
Photo previews, grading and exports run locally; photos are not uploaded.
Processing works offline. Installation, upgrades and the app's manual
"Check for updates" action need an internet connection.

The current macOS release is not notarized. If macOS blocks the first launch,
close the alert with **Done**, open **System Settings > Privacy & Security**,
click **Open Anyway** for RawAlchemy, then confirm **Open**.
See [Apple's instructions for reviewing and opening an app](https://support.apple.com/102445).

To upgrade or uninstall:

```bash
brew update
brew upgrade --cask shenmintao/raw-alchemy/raw-alchemy-studio
brew uninstall --cask shenmintao/raw-alchemy/raw-alchemy-studio
```

If you previously tapped the application source repository with an explicit
`https://github.com/shenmintao/Raw-Alchemy.git` URL, migrate once to remove that
large checkout and use the dedicated tap:

```bash
# Skip the uninstall line if the app is not installed.
brew uninstall --cask shenmintao/raw-alchemy/raw-alchemy-studio
brew untap shenmintao/raw-alchemy
brew tap shenmintao/raw-alchemy
brew install --cask shenmintao/raw-alchemy/raw-alchemy-studio
```

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

*   **To launch the GUI**: Simply run the executable without any arguments. See the tutorial below.
*   **To use the CLI**: Run the executable with command-line arguments.

## 🖥️ GUI Tutorial

The graphical interface provides an intuitive way to process your images.

![Image of GUI](Samples/gui_screenshot.png)

#### 1. Select Input and Output

*   **Input Path**:
    *   Click **Select File...** to process a single RAW file.
    *   Click **Select Folder...** to process all RAW files within a directory (batch mode).
*   **Output Path**:
    *   If processing a single file, you can specify the exact output file path by clicking **Save As...**.
    *   If processing a folder, you must select an output directory by clicking **Select Folder...**. All processed files will be saved there with a `.tif` extension.

#### 2. Configure Processing Settings

*   **Log Space**: Choose the target Log color space from the dropdown menu (e.g., `F-Log2`, `S-Log3`). This is a required setting.
*   **LUT File (.cube)**: (Optional) If you want to apply a creative look, click **Browse...** and select a `.cube` LUT file.
*   **Custom Lensfun DB**: (Optional) To use a custom lens database (e.g., one generated from LCP files), click **Browse...** and select the `.xml` file.

#### 3. Adjust Exposure

You can choose between two modes:

*   **Auto**: This is the default mode. You can select a **Metering** method from the dropdown (`hybrid`, `average`, etc.) to let the application determine the best exposure automatically.
*   **Manual**: Select this mode to override auto-exposure. You can then enter a specific EV value in the **EV Stops** box or use the slider to adjust the exposure compensation manually.

The **Metering** dropdown (available in `Auto` mode) lets you choose a strategy for automatic exposure adjustment:

*   **`matrix` (Default)**: An advanced evaluative metering mode. It divides the image into a 7x7 grid, intelligently weighting each zone based on brightness and position. It actively suppresses highlights and boosts shadows, providing the most balanced and reliable exposure for complex scenes.
*   **`hybrid`**: A simpler, faster intelligent mode. It aims for a balanced average exposure but will automatically reduce brightness to prevent highlights from blowing out.
*   **`average`**: Calculates the average brightness of the entire scene and adjusts it to middle gray. Best for evenly lit scenes.
*   **`center-weighted`**: Prioritizes the brightness of the center of the frame. Ideal for portraits or centered subjects.
*   **`highlight-safe` (ETTR)**: Exposes the image as brightly as possible without clipping highlights. This captures maximum shadow detail but may require lowering exposure in post.

#### 4. Start Processing

*   Click the **Start Processing** button.
*   The **Log** window at the bottom will display the real-time progress and status of the conversion.
*   Once finished, a "processing complete" message will appear in the log.

## 🔧 Advanced Usage: Importing Adobe Lens Profiles (LCP)

Raw Alchemy now includes a powerful script to convert and import lens profiles from Adobe's LCP format, which is used by Adobe Camera Raw and DNG Converter. This gives you access to a much larger and more up-to-date lens database.

The conversion script, lensfun-convert-lcp-new, can be found at [**Lensfun**](https://github.com/shenmintao/lensfun/tree/master/apps).

**Steps:**

1.  **Locate your LCP files.**
    You can get them by installing the free [Adobe DNG Converter](https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html). The profiles are typically located in:
    *   **Windows**: `C:\ProgramData\Adobe\CameraRaw\LensProfiles\1.0\`
    *   **macOS**: `/Library/Application Support/Adobe/CameraRaw/LensProfiles/1.0/`

2.  **Run the conversion script.**
    The script is located in the apps/ directory of the lensfun project. You will need Python installed to run it.

    Open your terminal, navigate to the Raw Alchemy project directory, and run the appropriate script for your OS:

    ```bash
    # On Windows
    python src/raw_alchemy/vendor/lensfun/win-x86_64/lensfun-convert-lcp "C:\ProgramData\Adobe\CameraRaw\LensProfiles\1.0"

    # On Linux
    python3 src/raw_alchemy/vendor/lensfun/linux-x86_64/lensfun-convert-lcp /path/to/your/lcp/files
    ```

    **Recommendation for De-duplication:** The script can check against an existing database to avoid creating duplicate entries. It is highly recommended to use the `--db-path` argument to point to the bundled database. This ensures that only new, unconverted lenses are added to your custom file.

    ```bash
    # Example with de-duplication on Windows
    python src/raw_alchemy/vendor/lensfun/win-x86_64/lensfun-convert-lcp "C:\ProgramData\Adobe\CameraRaw\LensProfiles\1.0" --db-path "src/raw_alchemy/vendor/lensfun/db"
    ```

3.  The script will create an `.xml` file (e.g., `_lcps.xml`). You can now load this file into Raw Alchemy using the GUI or CLI, as explained in the sections below.

    The conversion script saves to a default location, but you can use its `--output` argument to save the `.xml` file anywhere you like. For more details, run the script with `--help`.

## ⌨️ CLI Usage

**Note**: On Linux, you may need to make the file executable first (e.g., `chmod +x ./RawAlchemy-v0.1.0-linux`).

#### CLI Basic Syntax

The command structure is the same whether you are using the executable or installed from source.

```bash
# Using the executable on Linux (replace with your actual file name)
./RawAlchemy-v0.1.0-linux [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_TIFF_PATH>

# Using the executable on Windows (replace with your actual file name)
RawAlchemy-v0.1.0-windows.exe [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_TIFF_PATH>

# If installed from source
raw-alchemy [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_TIFF_PATH>
```

#### Example 1: Basic Log Conversion

This example converts a RAW file to linear space, then applies the F-Log2 curve, and saves the result as a TIFF file (keeping F-Log2/F-Gamut space, suitable for subsequent grading).

```bash
# Replace './RawAlchemy-linux' with your executable name or 'raw-alchemy' if installed from source
./RawAlchemy-linux "path/to/your/image.CR3" "path/to/output/image.tiff" --log-space "F-Log2"
```

#### Example 2: Conversion with Creative LUT

This example converts a RAW file, applies the S-Log3 curve, then applies a creative LUT (`my_look.cube`), and saves the final result.

```bash
# Replace './RawAlchemy-linux' with your executable name or 'raw-alchemy'
./RawAlchemy-linux "input.ARW" "output.tiff" --log-space "S-Log3" --lut "looks/my_look.cube"
```

#### Example 3: Manual Exposure Adjustment

This example manually applies a +1.5 stop exposure compensation, overriding any auto-exposure logic.

```bash
# Replace './RawAlchemy-linux' with your executable name or 'raw-alchemy'
./RawAlchemy-linux "input.CR3" "output_bright.tiff" --log-space "S-Log3" --exposure 1.5
```

#### Example 4: Using a Custom Lens Database

This example uses a custom lens database file for more accurate lens corrections.

```bash
# Replace './RawAlchemy-linux' with your executable name or 'raw-alchemy'
./RawAlchemy-linux "input.ARW" "output.tiff" --log-space "S-Log3" --custom-lensfun-db "path/to/your/_lcps.xml"
```

## ⚙️ Command Line Options

-   `<INPUT_RAW_PATH>`: (Required) Input RAW file path (e.g., .CR3, .ARW, .NEF).
-   `<OUTPUT_TIFF_PATH>`: (Required) Output 16-bit TIFF file save path.

-   `--log-space TEXT`: (Required) Target Log color space.
-   `--exposure FLOAT`: (Optional) Manual exposure adjustment in stops (e.g., -0.5, 1.0). Overrides all auto exposure logic.
-   `--lut TEXT`: (Optional) Path to a `.cube` LUT file to apply after Log conversion.
-   `--lens-correct / --no-lens-correct`: (Optional, Default: True) Enable or disable lens distortion correction.
-   `--custom-lensfun-db TEXT`: (Optional) Path to a custom Lensfun database XML file (e.g., one generated from LCP files).
-   `--metering TEXT`: (Optional, Default: `hybrid`) Auto exposure metering mode: `average` (geometric mean), `center-weighted`, `highlight-safe` (ETTR), or `hybrid` (default).

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

---

## ☕ Buy me a coffee

If **Raw Alchemy** saved your time, consider buying me a coffee to keep the code flowing. ☕

<details>
<summary><strong>👉 Click to expand (WeChat/Alipay)</strong></summary>

<br>
<div align="center">
  <img src="Samples/sponsor.png" width="300px">
  <p><sub>Thank you for your support!</sub></p>
</div>

</details>