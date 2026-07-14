import gc
import numpy as np
import colour
import os
from typing import Optional

from raw_alchemy import utils
from raw_alchemy.logger import create_logger
from raw_alchemy.file_io import save_image
from raw_alchemy import config, metering
from raw_alchemy.onnx.rgb_denoiser import denoise_rgb_linear
from raw_alchemy.math_ops import apply_matrix_inplace, compute_hl_refavg, init_taichi
from raw_alchemy.pipeline.executor import ExportExecutor
from raw_alchemy.pipeline.ops import build_op_list


# ==========================================
#          RAW 棰勫鐞?
# ==========================================

def subtract_black_level(sensor_raw, bl, wl, cfa_pattern):
    """Per-channel black subtraction, normalising the fresh decode in-place."""
    sensor_raw = np.asarray(sensor_raw, dtype=np.float32)
    pat_size = cfa_pattern.shape[0]
    bl = np.asarray(bl, np.float32)
    # 常见情况:各通道黑电平相同 → 单次全图向量运算(42MP 省 ~0.5s)
    if np.all(bl == bl.flat[0]):
        b = float(bl.flat[0])
        sensor_raw -= np.float32(b)
        np.maximum(sensor_raw, np.float32(0.0), out=sensor_raw)
        sensor_raw /= np.float32(wl - b)
        return sensor_raw
    # 通用:黑电平图按 CFA 铺开,单次运算
    H, W = sensor_raw.shape
    blk = np.empty((pat_size, pat_size), np.float32)
    for r in range(pat_size):
        for c in range(pat_size):
            blk[r, c] = bl[min(int(cfa_pattern[r, c]), len(bl) - 1)]
    bl_map = np.tile(blk, ((H + pat_size - 1) // pat_size,
                           (W + pat_size - 1) // pat_size))[:H, :W]
    sensor_raw -= bl_map
    np.maximum(sensor_raw, np.float32(0.0), out=sensor_raw)
    np.subtract(np.float32(wl), bl_map, out=bl_map)
    np.divide(sensor_raw, bl_map, out=sensor_raw)
    return sensor_raw


def fix_hot_pixels(raw_norm, cfa_pattern, threshold=4.0):
    """Detect and replace hot/dead pixels using per-channel median comparison."""
    import cv2
    pat_size = cfa_pattern.shape[0]
    for r in range(pat_size):
        for c in range(pat_size):
            plane = raw_norm[r::pat_size, c::pat_size]
            # uint16 量化后中值(cv2 直方图法,较 fp32 快 ~5x);
            # 检测在量化域,替换值用 fp32 中值邻域近似(量化步长 1/65535,
            # 远小于热像素阈值,无观感差异)
            q = np.clip(plane * 65535.0, 0, 65535).astype(np.uint16)
            med_q = cv2.medianBlur(np.ascontiguousarray(q), 3)
            diff = np.abs(q.astype(np.int32) - med_q.astype(np.int32)).astype(np.float32)
            std = max(float(diff.std()), 1e-3)
            hot = diff > threshold * std
            if hot.any():
                plane[hot] = med_q[hot].astype(np.float32) / 65535.0


# ==========================================
#     楂樺厜閲嶅缓 (Segmentation Based, GPU)
# ==========================================

def highlight_inpaint_opposed(raw_data, cfa_pattern, wb):
    """Segmentation-based highlight reconstruction.

    Same semantics as the darktable ``DT_IOP_HIGHLIGHTS_SEGMENTS`` mode:
    per-pixel opposing-channel reference average + per-segment chroma
    correction.

    Implementation: numpy/cv2 throughout — ``math_ops.compute_hl_refavg`` for
    the per-pixel opposing-channel reference (SIMD box filters), morphology /
    CCL / max-filter via OpenCV.
    """
    import cv2

    H, W = raw_data.shape
    pat_size = cfa_pattern.shape[0]
    g = max(float(wb[1]), 1e-6)

    color_map = np.tile(cfa_pattern,
                        ((H + pat_size - 1) // pat_size,
                         (W + pat_size - 1) // pat_size))[:H, :W]
    color_map = np.where(color_map >= 3, 1, color_map).astype(np.uint8)

    wb_gains = np.array([wb[0] / g, 1.0, wb[2] / g], dtype=np.float32)
    CLIP = 0.987
    raw_clips = np.array([CLIP / max(wg, 1e-6) for wg in wb_gains],
                         dtype=np.float32)

    # 快速门控:无任何接近饱和的像素时整段跳过(夜景/正常曝光常见)
    if float(raw_data.max()) < float(raw_clips.min()):
        return
    refavg, clipped = compute_hl_refavg(raw_data, color_map, wb_gains, raw_clips)
    if not np.any(clipped):
        return

    diff = raw_data - refavg
    del refavg
    # 7x7 square SE 鈥?closes gaps up to 6 px wide.
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for c in range(3):
        clipped_c = clipped & (color_map == c)
        if not np.any(clipped_c):
            continue

        # Morphological closing (SIMD-accelerated).
        cc_u8 = clipped_c.astype(np.uint8)
        closed = cv2.morphologyEx(cc_u8, cv2.MORPH_CLOSE, closing_kernel)

        # Connected components, 8-connectivity. Matches darktable's
        # segmentation choice; closing has already merged any 4-connected
        # clusters so the connectivity choice is near-equivalent here.
        num_seg_plus_one, labels = cv2.connectedComponents(closed, connectivity=8)
        num_seg = num_seg_plus_one - 1
        if num_seg == 0:
            continue

        # 7x7 max-filter on labels via cv2.dilate(float32) 鈥?used to
        # identify the segment-border zone for chroma estimation.
        expanded = cv2.dilate(labels.astype(np.float32), closing_kernel).astype(np.int32)

        lo = raw_clips[c] * 0.2
        unclipped_valid = (color_map == c) & ~clipped & (raw_data > lo)

        border = (expanded > 0) & (labels == 0) & unclipped_valid
        border_labels = expanded[border]
        border_diffs = diff[border]

        seg_sum = np.bincount(border_labels, weights=border_diffs,
                              minlength=num_seg + 1)
        seg_cnt = np.bincount(border_labels, minlength=num_seg + 1)

        global_chroma = 0.0
        total_cnt = seg_cnt[1:].sum()
        if total_cnt > 100:
            global_chroma = seg_sum[1:].sum() / total_cnt

        seg_chroma = np.where(seg_cnt > 10,
                              seg_sum / np.maximum(seg_cnt, 1),
                              global_chroma).astype(np.float32)

        target = clipped_c & (labels > 0)
        target_labels = labels[target]
        raw_data[target] = np.maximum(
            raw_data[target],
            raw_data[target] - diff[target] + seg_chroma[target_labels]
        )


# ==========================================
#              鏍稿績澶勭悊鍑芥暟
# ==========================================

_ORIENT_TO_FLIP = {1: 0, 3: 3, 6: 6, 8: 5}  # EXIF orientation -> libraw flip


def _read_orientation_flip(raw_path: str) -> int:
    """零依赖读取 TIFF/EXIF Orientation(tag 0x0112)→ libraw flip 语义。

    rawspeed C API 不暴露方向;此前为拿 flip 专门做一次 rawpy 解包
    (整包白付 ~1-2s)。非 TIFF 容器(RAF 等)返回 -1,调用方回退。
    """
    import struct
    try:
        with open(raw_path, "rb") as f:
            data = f.read(65536)
        if len(data) < 8 or data[:2] not in (b"II", b"MM"):
            return -1
        bo = "<" if data[:2] == b"II" else ">"
        u16 = lambda o: struct.unpack_from(bo + "H", data, o)[0]
        u32 = lambda o: struct.unpack_from(bo + "I", data, o)[0]
        off = u32(4)
        if off + 2 > len(data):
            return -1
        n = u16(off)
        if n > 4096:
            return -1
        for i in range(n):
            e = off + 2 + i * 12
            if e + 12 > len(data):
                return -1
            if u16(e) == 0x0112:
                return _ORIENT_TO_FLIP.get(u16(e + 8), 0)
        return 0  # 无 Orientation 标签 = 不旋转
    except Exception:
        return -1


def _rawpy_decode_to_prophoto(raw_path: str) -> np.ndarray:
    """Decode RAW to working-space linear float32 (H, W, 3).

    Bayer: RawSpeed (or rawpy) raw decode + RCD demosaic on the ONNX runtime
    (GPU via DirectML/CUDA) — the Taichi port is retired; the ONNX graph is
    pixel-exact against it.
    X-Trans / other sensors: libraw demosaic (unit WB, linear), then this
    app's white balance + cam->working matrix (colour-matched, ~0.16% delta).
    """
    import rawpy
    from raw_alchemy.onnx.denoiser import _apply_flip
    from raw_alchemy.colorspace_matrices import cam_to_working_space_matrix

    # ---- CFA fast path: rawspeed/rawpy decode + ONNX demosaic (GPU) ----
    cfa_data = None
    rs = None
    XTRANS_PATTERN = None
    try:
        from rawspeedpy import try_decode, XTRANS_PATTERN
        rs = try_decode(raw_path)
    except Exception:
        # rawspeedpy 在损坏 makernote(如 Sony 转制 DNG 的 Sony2 目录)上
        # 可能抛 UnicodeDecodeError 等——回退 rawpy 解码,仍走 GPU 去马赛克
        rs = None
    try:
        rs_ok = bool(rs and (rs.is_bayer or rs.is_xtrans) and rs.color_matrix is not None)
    except Exception:
        rs_ok = False
    if rs_ok:
        from raw_alchemy.demosaic_helpers import get_cfa_pattern_from_filters
        flip = _read_orientation_flip(raw_path)
        if flip < 0:
            try:
                with rawpy.imread(raw_path) as _r:
                    flip = _r.sizes.flip
            except Exception:
                flip = 0
        pattern = (get_cfa_pattern_from_filters(rs.filters) if rs.is_bayer
                   else np.asarray(XTRANS_PATTERN))
        cfa_data = (
            rs.bayer.astype(np.float32),
            np.array(rs.black_levels, dtype=np.float32),
            float(rs.white_level),
            np.array(rs.wb_coeffs, dtype=np.float32),
            rs.color_matrix.astype(np.float64),
            pattern,
            flip,
        )
    else:
        with rawpy.imread(raw_path) as raw:
            cfa = raw.raw_pattern
            if cfa is not None and cfa.shape in ((2, 2), (6, 6)):
                cfa_data = (
                    raw.raw_image_visible.astype(np.float32),
                    np.array(raw.black_level_per_channel, dtype=np.float32),
                    float(raw.white_level),
                    np.array(raw.camera_whitebalance, dtype=np.float32),
                    np.array(raw.rgb_xyz_matrix, dtype=np.float64),
                    cfa.copy(),
                    raw.sizes.flip,
                )

    if cfa_data is not None:
        sensor_raw, bl, wl, wb, xyz_to_cam, cfa_pattern, flip = cfa_data
        raw_norm = subtract_black_level(sensor_raw, bl, wl, cfa_pattern)
        fix_hot_pixels(raw_norm, cfa_pattern)
        highlight_inpaint_opposed(raw_norm, cfa_pattern, wb)
        g = wb[1] if wb[1] > 0 else 1.0
        wb3 = np.array([wb[0] / g, 1.0, wb[2] / g], np.float32)
        m = cam_to_working_space_matrix(xyz_to_cam).astype(np.float32)
        if cfa_pattern.shape == (2, 2):
            # WB+矩阵+clip 折叠在 RCD 图输出端(GPU),省 42MP CPU einsum
            from raw_alchemy.onnx.rcd_demosaic import rcd_demosaic as onnx_rcd
            rgb = onnx_rcd(raw_norm, cfa_pattern, wb3=wb3, cam_mat=m)
        else:
            from raw_alchemy.onnx.xtrans_demosaic import xtrans_markesteijn_demosaic
            rgb = xtrans_markesteijn_demosaic(raw_norm, cfa_pattern)
            rgb *= wb3
            from raw_alchemy.math_ops import apply_matrix_inplace
            apply_matrix_inplace(rgb, m)
            np.clip(rgb, 0.0, 1.0, out=rgb)
        return np.ascontiguousarray(_apply_flip(rgb, flip))

    # ---- 罕见传感器(Foveon 等):libraw 兜底 ----
    with rawpy.imread(raw_path) as raw:
        cfa = raw.raw_pattern
        has_cfa = cfa is not None and cfa.shape in ((2, 2), (6, 6))
        if not has_cfa:
            # Foveon & friends: let libraw handle colour end-to-end.
            rgb16 = raw.postprocess(
                gamma=(1, 1), no_auto_bright=True, use_camera_wb=True,
                use_auto_wb=False, output_bps=16,
                output_color=rawpy.ColorSpace.ProPhoto,
                bright=1.0, highlight_mode=2,
            )
            rgb = (rgb16 / 65535.0).astype(np.float32)
            del rgb16
            if rgb.ndim == 3 and rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            np.maximum(rgb, 0.0, out=rgb)
            return rgb

        wb = np.array(raw.camera_whitebalance, dtype=np.float32)
        flip = raw.sizes.flip
        xyz_to_cam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)
        # Camera-native demosaic only: unit WB, no auto-bright, linear,
        # unflipped — this app owns white balance, colour and orientation.
        cam = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True,
            user_wb=[1.0, 1.0, 1.0, 1.0], output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            user_flip=0, half_size=False, highlight_mode=2,
        )

    rgb = cam.astype(np.float32) / 65535.0
    del cam
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    elif rgb.shape[2] > 3:
        rgb = np.ascontiguousarray(rgb[:, :, :3])

    g = wb[1] if wb[1] > 0 else 1.0
    rgb[:, :, 0] *= wb[0] / g
    rgb[:, :, 2] *= wb[2] / g

    cam_to_working = cam_to_working_space_matrix(xyz_to_cam).astype(np.float32)
    from raw_alchemy.math_ops import apply_matrix_inplace
    apply_matrix_inplace(rgb, cam_to_working)
    rgb = np.ascontiguousarray(_apply_flip(rgb, flip))
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb


def _build_export_params(
    *,
    log_space: str,
    lut_path: Optional[str],
    exposure: Optional[float],
    metering_mode: str,
    wb_temp: float,
    wb_tint: float,
    saturation: float,
    contrast: float,
    highlight: float,
    shadow: float,
    rotation: int,
    flip_horizontal: bool,
    flip_vertical: bool,
    perspective_corners: Optional[tuple],
    crop: Optional[tuple],
    lens_correct: bool = False,
    custom_db_path: Optional[str] = None,
    denoise_enabled: bool = False,
    sharpen_strength: float = 0.0,
    hdr_output: bool = False,
):
    return {
        "denoise_enabled": denoise_enabled,
        "lens_correct": lens_correct,
        "custom_db_path": custom_db_path,
        "rotation": rotation,
        "flip_horizontal": flip_horizontal,
        "flip_vertical": flip_vertical,
        "perspective_corners": perspective_corners,
        "crop": crop or (0.0, 0.0, 1.0, 1.0),
        "exposure_mode": "Auto" if exposure is None else "Manual",
        "exposure": 0.0 if exposure is None else exposure,
        "metering_mode": metering_mode,
        "wb_temp": wb_temp,
        "wb_tint": wb_tint,
        "highlight": highlight,
        "shadow": shadow,
        "saturation": saturation,
        "contrast": contrast,
        "log_space": log_space,
        "lut_path": lut_path,
        "hdr_output": hdr_output,
        "sharpen_strength": sharpen_strength,
    }


def _linearize_for_dng(img: np.ndarray, output_path: str, log_space: str, lut_path: Optional[str], logger):
    color_matrix = None
    if not output_path.lower().endswith('.dng'):
        return img, color_matrix

    logger.info("  [Pre-Save] Converting to Linear for DNG export...")
    try:
        utils.srgb_to_linear_inplace(img)
        color_matrix = colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
    except Exception as e:
        logger.warning(f"  Linearization failed: {e}. Saving as is.")
    return img, color_matrix


def _run_export_executor(
    source_img: np.ndarray,
    params: dict,
    metering_source: np.ndarray | None = None,
    lens_corrector=None,
) -> np.ndarray:
    def auto_gain_resolver(_img: np.ndarray, mode: str) -> float:
        source_cs = colour.RGB_COLOURSPACES[config.WORKING_SPACE]
        if callable(metering_source):
            source = metering_source()
        else:
            source = metering_source
        if source is None:
            source = source_img
        return metering.get_metering_strategy(mode).calculate_gain(source, source_cs)

    executor = ExportExecutor(
        auto_gain_resolver=auto_gain_resolver,
        lens_corrector=lens_corrector,
    )
    ops = build_op_list(params)
    return executor.run(ops, source_img)


def process_image(
    raw_path: str,
    output_path: str,
    log_space: str,
    lut_path: Optional[str],
    exposure: Optional[float] = None,
    lens_correct: bool = True,
    metering_mode: str = 'hybrid',
    custom_db_path: Optional[str] = None,
    log_queue: Optional[object] = None,
    # New params
    wb_temp: float = 0.0,
    wb_tint: float = 0.0,
    saturation: float = 1.25,
    contrast: float = 1.1,
    highlight: float = 0.0,
    shadow: float = 0.0,
    # Geometry
    rotation: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    perspective_corners: Optional[tuple] = None,
    crop: Optional[tuple] = None,
    # Denoising
    denoise_enabled: bool = False,
    denoise_strength: float = 0.25,
    # Sharpening (Richardson-Lucy)
    sharpen_strength: float = 0.0,
    # HDR output
    hdr_output: bool = False,
):
    filename = os.path.basename(raw_path)
    
    # 鍒涘缓缁熶竴鐨勬棩蹇楀鐞嗗櫒
    logger = create_logger(log_queue, filename)
    
    logger.info(f"馃И [Raw Alchemy] Processing: {raw_path}")
    init_taichi()

    from raw_alchemy.exif import extract_lens_exif

    exif_data, exif_metadata = extract_lens_exif(raw_path, None)

    logger.info("  [Step 1] Decoding RAW (rawpy + RCD)...")
    img = _rawpy_decode_to_prophoto(raw_path)
    if denoise_enabled:
        logger.info(f"  [Step 1b] FastDenoise v4 (s={denoise_strength:.2f})...")
        try:
            img = denoise_rgb_linear(img, strength=denoise_strength)
        except Exception as e:
            logger.error(f"  FastDenoise failed, continuing without denoise: {e}")

    lens_state = {"corrected": img}

    def lens_corrector(src: np.ndarray) -> np.ndarray:
        logger.info("  [Step 2] Lens Correction...")
        corrected = utils.apply_lens_correction(
            src,
            exif_data=exif_data,
            custom_db_path=custom_db_path,
        )
        lens_state["corrected"] = corrected
        return corrected

    params = _build_export_params(
        log_space=log_space,
        lut_path=lut_path,
        exposure=exposure,
        metering_mode=metering_mode,
        wb_temp=wb_temp,
        wb_tint=wb_tint,
        saturation=saturation,
        contrast=contrast,
        highlight=highlight,
        shadow=shadow,
        rotation=rotation,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        perspective_corners=perspective_corners,
        crop=crop,
        lens_correct=lens_correct,
        custom_db_path=custom_db_path,
        denoise_enabled=False,
        sharpen_strength=sharpen_strength,
        hdr_output=hdr_output,
    )

    img = _run_export_executor(
        img,
        params,
        metering_source=lambda: lens_state["corrected"],
        lens_corrector=lens_corrector if lens_correct else None,
    )
    img, color_matrix = _linearize_for_dng(img, output_path, log_space, lut_path, logger)

    logger.info(f"  Saving to {os.path.basename(output_path)}...")
    save_image(
        img,
        output_path,
        logger,
        exif_metadata=exif_metadata,
        exif_dict=exif_data,
        color_matrix=color_matrix,
        hdr_output=hdr_output,
    )

    del img
    gc.collect()
    return


def export_from_cache(
    cached_img: np.ndarray,
    output_path: str,
    exif_data: dict,
    exif_metadata: Optional[dict],
    log_space: str,
    lut_path: Optional[str],
    exposure: Optional[float] = None,
    metering_mode: str = 'hybrid',
    log_queue: Optional[object] = None,
    wb_temp: float = 0.0,
    wb_tint: float = 0.0,
    saturation: float = 1.25,
    contrast: float = 1.1,
    highlight: float = 0.0,
    shadow: float = 0.0,
    rotation: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    perspective_corners: Optional[tuple] = None,
    crop: Optional[tuple] = None,
    sharpen_strength: float = 0.0,
    hdr_output: bool = False,
):
    """Export using cached ProPhoto Linear data (after demosaic + lens correction + denoise).

    Skips RAW decode, demosaicing, lens correction and denoising 鈥?reuses the
    preview pipeline's cached result.  Everything from geometry onward is
    identical to process_image().
    """
    filename = os.path.basename(output_path)
    logger = create_logger(log_queue, filename)
    logger.info(f"馃И [Raw Alchemy] Export from cache -> {output_path}")
    init_taichi()

    params = _build_export_params(
        log_space=log_space,
        lut_path=lut_path,
        exposure=exposure,
        metering_mode=metering_mode,
        wb_temp=wb_temp,
        wb_tint=wb_tint,
        saturation=saturation,
        contrast=contrast,
        highlight=highlight,
        shadow=shadow,
        rotation=rotation,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        perspective_corners=perspective_corners,
        crop=crop,
        lens_correct=False,
        denoise_enabled=False,
        sharpen_strength=sharpen_strength,
        hdr_output=hdr_output,
    )

    # ExportExecutor ingests the source into its own working buffer before any
    # in-place operation. Passing the immutable cache array directly avoids a
    # redundant full-resolution copy (about 732MB for 61MP float32 RGB).
    img = _run_export_executor(cached_img, params, metering_source=cached_img)
    img, color_matrix = _linearize_for_dng(img, output_path, log_space, lut_path, logger)

    logger.info(f"  Saving to {filename}...")
    save_image(
        img,
        output_path,
        logger,
        exif_metadata=exif_metadata,
        exif_dict=exif_data,
        color_matrix=color_matrix,
        hdr_output=hdr_output,
    )

    del img
    gc.collect()
    return
