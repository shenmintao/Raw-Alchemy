from collections.abc import Callable, Sequence
from dataclasses import dataclass

import colour
import numpy as np

from raw_alchemy import metering, utils
from raw_alchemy.gpu_buffer import GpuImage
from raw_alchemy.math_ops import (
    apply_crop_gpu,
    apply_gain_inplace,
    apply_geometry_gpu,
    apply_highlight_shadow_inplace,
    apply_lut_inplace,
    apply_matrix_inplace,
    apply_saturation_contrast_inplace,
    clip_inplace,
    compute_perspective_matrix,
    linear_to_srgb_inplace,
    log_encode_gpu,
    max_inplace,
    perspective_warp_kernel,
    sharpen_gpu,
    white_balance_matrix,
)
from raw_alchemy.pipeline.ops import Op


ImageCallback = Callable[[np.ndarray], np.ndarray]
AutoGainResolver = Callable[[np.ndarray, str], float]


@dataclass(frozen=True)
class PipelineResult:
    image: np.ndarray
    applied_ev: float


class _BaseExecutor:
    def __init__(
        self,
        source: np.ndarray | None = None,
        *,
        denoiser: ImageCallback | None = None,
        lens_corrector: ImageCallback | None = None,
        auto_gain_resolver: AutoGainResolver | None = None,
        round_exposure_gain: bool = False,
    ):
        self._source: np.ndarray | None = None
        self.denoiser = denoiser
        self.lens_corrector = lens_corrector
        self.auto_gain_resolver = auto_gain_resolver
        self.round_exposure_gain = round_exposure_gain
        self.last_applied_ev = 0.0
        if source is not None:
            self.set_source(source)

    def set_source(self, source: np.ndarray):
        if self._source is source:
            return
        if source.dtype != np.float32:
            source = source.astype(np.float32)
        if not source.flags["C_CONTIGUOUS"]:
            source = np.ascontiguousarray(source)
        self._source = source

    def _resolve_source(self, source: np.ndarray | None) -> np.ndarray:
        if source is not None:
            self.set_source(source)
        if self._source is None:
            raise ValueError("No source image was provided to the executor")
        return self._source

    def _run_direct_result(self, ops: Sequence[Op], source: np.ndarray | None = None) -> PipelineResult:
        src = self._resolve_source(source)
        self.last_applied_ev = 0.0
        buf = GpuImage()
        buf.upload(src)
        for op in ops:
            buf = self._apply_op(buf, op)
        clip_inplace(buf.arr)
        return PipelineResult(buf.to_numpy(), self.last_applied_ev)

    def _apply_op(self, buf: GpuImage, op: Op) -> GpuImage:
        if op.name == "denoise":
            if self.denoiser is None:
                raise RuntimeError("denoise op requires an executor denoiser callback")
            buf.upload(self.denoiser(buf.to_numpy()))
            return buf

        if op.name == "lens_correct":
            if self.lens_corrector is None:
                raise RuntimeError("lens_correct op requires an executor lens_corrector callback")
            buf.upload(self.lens_corrector(buf.to_numpy()))
            return buf

        if op.name == "geometry":
            rotation, flip_h, flip_v = op.params
            dst = GpuImage()
            apply_geometry_gpu(buf, dst, rotation=rotation, flip_h=flip_h, flip_v=flip_v)
            buf.clear()
            return dst

        if op.name == "perspective":
            corners = op.params[0]
            h, w = buf.height, buf.width
            _, m_inv = compute_perspective_matrix(corners, w, h)
            dst = GpuImage()
            dst._allocate(h, w, 3)
            perspective_warp_kernel(buf.arr, dst.arr, m_inv)
            buf.clear()
            return dst

        if op.name == "crop":
            dst = GpuImage()
            apply_crop_gpu(buf, dst, op.params[0])
            buf.clear()
            return dst

        if op.name == "exposure":
            exposure_mode, exposure, metering_mode, *rest = op.params
            working_space = rest[0] if rest else None
            if exposure_mode == "Manual":
                gain = 2.0 ** float(exposure)
                self.last_applied_ev = float(exposure)
            else:
                metering_img = buf.to_numpy()
                if self.auto_gain_resolver is not None:
                    gain = self.auto_gain_resolver(metering_img, metering_mode)
                else:
                    source_cs = (
                        colour.RGB_COLOURSPACES[working_space]
                        if working_space
                        else utils.get_working_colourspace()
                    )
                    gain = metering.get_metering_strategy(metering_mode).calculate_gain(
                        metering_img,
                        source_cs,
                    )
                self.last_applied_ev = float(np.log2(gain))
            gain_to_apply = round(float(gain), 4) if self.round_exposure_gain else float(gain)
            apply_gain_inplace(buf.arr, gain_to_apply)
            return buf

        if op.name == "white_balance":
            wb_temp, wb_tint, *rest = op.params
            working_space = rest[0] if rest else None
            matrix = white_balance_matrix(wb_temp, wb_tint, working_space)
            apply_matrix_inplace(buf.arr, matrix)
            return buf

        if op.name == "highlight_shadow":
            highlight, shadow, *rest = op.params
            working_space = rest[0] if rest else None
            source_cs = (
                colour.RGB_COLOURSPACES[working_space]
                if working_space
                else utils.get_working_colourspace()
            )
            luma = utils.get_luminance_coeffs(source_cs).astype(np.float32)
            apply_highlight_shadow_inplace(
                buf.arr,
                float(highlight) / 100.0,
                float(shadow) / 100.0,
                luma,
            )
            return buf

        if op.name == "sat_contrast":
            saturation, contrast, *rest = op.params
            working_space = rest[0] if rest else None
            source_cs = (
                colour.RGB_COLOURSPACES[working_space]
                if working_space
                else utils.get_working_colourspace()
            )
            luma = utils.get_luminance_coeffs(source_cs).astype(np.float32)
            apply_saturation_contrast_inplace(
                buf.arr,
                float(saturation),
                float(contrast),
                0.18,
                luma,
            )
            return buf

        if op.name == "log_transform":
            if len(op.params) == 4:
                _, working_space, log_color_space_name, log_curve_name = op.params
            else:
                _, log_color_space_name, log_curve_name = op.params
                working_space = None
            if not log_color_space_name:
                raise ValueError(f"Unknown Log Space: {op.params[0]}")
            matrix = colour.matrix_RGB_to_RGB(
                colour.RGB_COLOURSPACES[working_space]
                if working_space
                else utils.get_working_colourspace(),
                colour.RGB_COLOURSPACES[log_color_space_name],
            )
            apply_matrix_inplace(buf.arr, matrix)
            max_inplace(buf.arr, 1e-6)
            if not log_encode_gpu(buf.arr, log_curve_name):
                encoded = colour.cctf_encoding(buf.to_numpy(), function=log_curve_name)
                buf.upload(encoded.astype(np.float32))
            return buf

        if op.name == "lut":
            lut = utils.load_lut_cached(op.params[0])
            if isinstance(lut, colour.LUT3D):
                table = np.ascontiguousarray(lut.table.astype(np.float32))
                domain_min = np.ascontiguousarray(lut.domain[0].astype(np.float64))
                domain_max = np.ascontiguousarray(lut.domain[1].astype(np.float64))
                apply_lut_inplace(buf.arr, table, domain_min, domain_max)
            else:
                buf.upload(lut.apply(buf.to_numpy()).astype(np.float32))
            return buf

        if op.name == "srgb_out":
            working_space = op.params[0] if op.params else None
            matrix = colour.matrix_RGB_to_RGB(
                colour.RGB_COLOURSPACES[working_space]
                if working_space
                else utils.get_working_colourspace(),
                colour.RGB_COLOURSPACES["sRGB"],
            )
            apply_matrix_inplace(buf.arr, matrix)
            linear_to_srgb_inplace(buf.arr)
            return buf

        if op.name == "pq_out":
            working_space, output_space, transfer_function, peak_nits, mastering_nits = op.params
            matrix = colour.matrix_RGB_to_RGB(
                colour.RGB_COLOURSPACES[working_space],
                colour.RGB_COLOURSPACES[output_space],
            )
            apply_matrix_inplace(buf.arr, matrix)
            max_inplace(buf.arr, 0.0)
            encoded = colour.cctf_encoding(
                np.clip(buf.to_numpy(), 0.0, None) * (float(peak_nits) / float(mastering_nits)),
                function=transfer_function,
            )
            buf.upload(np.clip(encoded, 0.0, 1.0).astype(np.float32))
            return buf

        if op.name == "sharpen":
            sharpen_gpu(buf, strength=float(op.params[0]), sigma=1.0)
            return buf

        raise ValueError(f"Unknown pipeline op: {op.name}")


class ExportExecutor(_BaseExecutor):
    def run(self, ops: Sequence[Op], source: np.ndarray | None = None) -> np.ndarray:
        return self.run_result(ops, source).image

    def run_result(self, ops: Sequence[Op], source: np.ndarray | None = None) -> PipelineResult:
        return self._run_direct_result(ops, source)


class PreviewExecutor(_BaseExecutor):
    def __init__(self, source: np.ndarray | None = None, **kwargs):
        self._prefix_cache: dict[int, PipelineResult] = {}
        super().__init__(source, **kwargs)

    def set_source(self, source: np.ndarray):
        super().set_source(source)
        self._prefix_cache.clear()

    def run(self, ops: Sequence[Op], source: np.ndarray | None = None) -> np.ndarray:
        return self.run_result(ops, source).image

    def run_result(self, ops: Sequence[Op], source: np.ndarray | None = None) -> PipelineResult:
        if source is not None:
            self.set_source(source)
        src = self._resolve_source(None)

        start_index = 0
        cached_result = None
        for index in range(len(ops), 0, -1):
            prefix_hash = hash(tuple(ops[:index]))
            if prefix_hash in self._prefix_cache:
                cached_result = self._prefix_cache[prefix_hash]
                start_index = index
                break

        self.last_applied_ev = cached_result.applied_ev if cached_result is not None else 0.0

        buf = GpuImage()
        buf.upload(cached_result.image if cached_result is not None else src)

        for index, op in enumerate(ops[start_index:], start=start_index + 1):
            buf = self._apply_op(buf, op)
            self._prefix_cache[hash(tuple(ops[:index]))] = PipelineResult(
                buf.to_numpy().copy(),
                self.last_applied_ev,
            )

        clip_inplace(buf.arr)
        return PipelineResult(buf.to_numpy(), self.last_applied_ev)
