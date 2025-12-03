# Raw Alchemy

[English](README.md) | [简体中文](README_zh-CN.md)

---

A Python-based command-line tool for advanced RAW image processing pipelines. It is designed to convert RAW files into a wide-gamut linear space (ProPhoto RGB), apply camera-specific Log curves, and integrate creative LUTs, achieving a complete and color-managed workflow.

### Core Philosophy

Many photographers and videographers rely on creative LUTs (Look Up Tables) to achieve specific visual styles. However, a common pain point is: **When applying LUTs that perform perfectly in video workflows to RAW photos, colors often go wrong.**

This issue stems from color space mismatches. Most creative LUTs are designed for specific Log color spaces (e.g., Sony S-Log3/S-Gamut3.Cine or Fujifilm F-Log2/F-Gamut). When you open a RAW photo in Photoshop or Lightroom and directly apply these LUTs, the software's default RAW decoding color space does not match the LUT's expected input space, leading to severe color and tonal deviations.

**Raw Alchemy** was born to solve this problem. It builds a rigorous, automated color pipeline to ensure any LUT can be accurately applied to any RAW file:

1.  **Standardized Decoding**: The project first decodes RAW files from any source into a standardized, wide-gamut intermediate space — ProPhoto RGB (Linear). This eliminates color science differences inherent in different camera brands, providing a unified starting point for all operations.
2.  **Precise Log Signal Preparation**: Next, it precisely converts the linear image data into the Log format expected by the target LUT, such as `F-Log2` curve and `F-Gamut` color space. This step is critical for ensuring color consistency, perfectly simulating the process of generating Log video signals inside a camera.
3.  **Correct LUT Application**: Applying your creative LUT on this accurately "disguised" Log image results in color and tonal performance identical to that in professional video software (like DaVinci Resolve).
4.  **High-Bit Depth Output**: Finally, the processed image (keeping Log encoding or with LUT effects applied) is saved as a 16-bit TIFF file, maximizing dynamic range and color information retention for professional grading in Photoshop or DaVinci Resolve.

Through this process, `Raw Alchemy` breaks down the barrier between RAW photography and professional video color grading, allowing photographers to enjoy cinema-level color management precision.

### Process Flow

This tool follows these precise color conversion steps:

`RAW (Camera Native)` -> `ProPhoto RGB (Linear)` -> `Target Log Gamut (Linear)` -> `Target Log Curve (e.g. F-Log2)` -> `(Optional) Creative LUT` -> `16-bit TIFF`

### Features

-   **RAW to Linear**: Decodes RAW files directly into ProPhoto RGB (Linear) working color space.
-   **Log Conversion**: Supports various camera-specific Log formats (F-Log2, S-Log3, LogC4, etc.).
-   **LUT Application**: Supports applying `.cube` creative LUT files directly during conversion.
-   **Exposure Control**: Provides flexible exposure logic: Manual exposure override, or smart auto-metering (Hybrid, Average, Center-Weighted, Highlight-Safe/ETTR).
-   **High Quality Output**: Saves the final image as a 16-bit TIFF file.
-   **Tech Stack**: Uses `rawpy` for RAW decoding and utilizes `colour-science` for high-precision color transformations.

### Samples

| RAW (Linear Preview) | Log Space (V-Log) | Final Look (FujiFilm Class-Neg) |
| :---: | :---: | :---: |
| ![RAW](Samples/RAW.jpeg) | ![V-Log](Samples/V-Log.jpeg) | ![Class-Neg](Samples/FujiFilm%20Class-Neg.jpeg) |

#### Accuracy Verification

Comparison with Panasonic Lumix Real-time LUT.

| In-Camera Real-time LUT | Raw Alchemy Processing |
| :---: | :---: |
| ![In-Camera](Samples/P1013122.jpg) | ![Raw Alchemy](Samples/Converted.jpg) |

### Installation

Install Raw Alchemy:

```bash
# Clone the repository
git clone https://github.com/shenmintao/raw-alchemy.git
cd raw-alchemy

# Install the tool
pip install .
```

*Note: This project depends on specific versions of `rawpy` and `colour-science`.*

### Usage

Use via the `raw-alchemy` command.

#### Basic Syntax

```bash
raw-alchemy [OPTIONS] <INPUT_RAW_PATH> <OUTPUT_TIFF_PATH>
```

#### Example 1: Basic Log Conversion

This example converts a RAW file to linear space, then applies the F-Log2 curve, and saves the result as a TIFF file (keeping F-Log2/F-Gamut space, suitable for subsequent grading).

```bash
raw-alchemy "path/to/your/image.CR3" "path/to/output/image.tiff" --log-space "F-Log2"
```

#### Example 2: Conversion with Creative LUT

This example converts a RAW file, applies the S-Log3 curve, then applies a creative LUT (`my_look.cube`), and saves the final result.

```bash
raw-alchemy "input.ARW" "output.tiff" --log-space "S-Log3" --lut "looks/my_look.cube"
```

#### Example 3: Manual Exposure Adjustment

This example manually applies a +1.5 stop exposure compensation, overriding any auto-exposure logic.

```bash
raw-alchemy "input.CR3" "output_bright.tiff" --log-space "S-Log3" --exposure 1.5
```

### Command Line Options

-   `<INPUT_RAW_PATH>`: (Required) Input RAW file path (e.g., .CR3, .ARW, .NEF).
-   `<OUTPUT_TIFF_PATH>`: (Required) Output 16-bit TIFF file save path.

-   `--log-space TEXT`: (Required) Target Log color space.
-   `--exposure FLOAT`: (Optional) Manual exposure adjustment in stops (e.g., -0.5, 1.0). Overrides all auto exposure logic.
-   `--lut TEXT`: (Optional) Path to a `.cube` LUT file to apply after Log conversion.
-   `--lens-correct / --no-lens-correct`: (Optional, Default: True) Enable or disable lens distortion correction.
-   `--metering TEXT`: (Optional, Default: `hybrid`) Auto exposure metering mode: `average` (geometric mean), `center-weighted`, `highlight-safe` (ETTR), or `hybrid` (default).

### Supported Log Spaces

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
-   `LogC3`
-   `LogC4`
-   `Log3G10`
