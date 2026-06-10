# Raw Alchemy Pipeline

This document describes the unified operation list built by
`raw_alchemy.pipeline.ops.build_op_list` and executed by
`PreviewExecutor` and `ExportExecutor`.

## Operation Order

The current order is:

```text
denoise -> lens_correct -> geometry -> perspective -> crop ->
exposure -> white_balance -> highlight_shadow -> sat_contrast ->
log_transform -> lut

or, when no log output is selected:

denoise -> lens_correct -> geometry -> perspective -> crop ->
exposure -> white_balance -> highlight_shadow -> sat_contrast ->
srgb_out

sharpen runs last when enabled.
```

For HDR HEIF export, `pq_out` replaces `srgb_out` in the same position.

`build_op_list` omits no-op stages. For example, default crop and default
perspective corners are not represented as ops. The resulting `Op` objects are
hashable, so preview caching can key each prefix with `hash(tuple(ops[:i]))`.

## Domains

`denoise`

Input is the RAW file path and packed sensor data. Output is linear ProPhoto RGB.
When enabled, CANS RAW V2 replaces normal demosaic output.

`lens_correct`

Input and output are linear ProPhoto RGB. Lens correction is a geometric/remap
operation using EXIF lens metadata and optional custom Lensfun data.

`geometry`

Input and output are linear ProPhoto RGB. The op applies rotation and horizontal
or vertical flips.

`perspective`

Input and output are linear ProPhoto RGB. The op applies the normalized corner
warp from the UI using the same perspective matrix and warp kernel as preview.

`crop`

Input and output are linear ProPhoto RGB. The crop rectangle is normalized as
`(x, y, w, h)`.

`exposure`

Input and output are linear ProPhoto RGB. Manual mode applies `2 ** EV`. Auto
mode asks the active executor for the metering gain; preview and export provide
their own source selection while sharing the same op.

`white_balance`

Input and output are linear ProPhoto RGB by default. Temperature and tint map
to a target white point (CCT + Duv), then a cached Bradford chromatic
adaptation matrix is applied in the configured working RGB space. A zero
temperature/tint adjustment returns the identity matrix.

`highlight_shadow`

Input and output are linear ProPhoto RGB. Luma coefficients come from ProPhoto
RGB and the operation applies the existing highlight/shadow kernel.

`sat_contrast`

Input and output are linear ProPhoto RGB. Saturation is applied around luma, and
contrast is applied around a `0.18` pivot.

`log_transform`

Input is linear ProPhoto RGB. The op converts to the selected log gamut, clamps
to a small positive floor, then applies the selected log transfer function.

`lut`

Input and output are in the current graded domain. For log output this is after
`log_transform`; for standard output it is omitted unless the user supplies a
LUT.

`srgb_out`

Input is linear ProPhoto RGB. The op converts to linear sRGB and applies the
sRGB transfer function.

`pq_out`

Input is linear ProPhoto RGB. The op converts to BT.2020 RGB and applies
ST 2084/PQ. Raw Alchemy maps linear value `1.0` to 1000 nits before encoding,
then the HEIF writer tags the file with BT.2020 primaries and PQ transfer
metadata.

`sharpen`

Input and output are the current display/export domain. It runs last and uses
the Richardson-Lucy sharpening path.

## Executors

`PreviewExecutor` and `ExportExecutor` execute the same op list.

`PreviewExecutor` stores prefix results keyed by `hash(tuple(ops[:i]))`, so
parameter changes can reuse prior stages. It reports `applied_ev` for the UI and
for export consistency.

`ExportExecutor` executes the op list directly without prefix caching. Full RAW
exports decode or denoise first, then run the same exported op list before
saving. Cached GUI exports are submitted back to the processor worker thread so
Taichi kernels run on the worker-owned runtime context.
