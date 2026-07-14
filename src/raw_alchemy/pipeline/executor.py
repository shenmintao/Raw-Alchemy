import sys
from collections.abc import Callable, Sequence
from functools import lru_cache

import colour
import numpy as np
from loguru import logger

from raw_alchemy import metering, utils
from raw_alchemy.gpu_buffer import GpuImage
from raw_alchemy.math_ops import (
    apply_crop_gpu,
    apply_crop_pixels_gpu,
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
    pq_encode_inplace,
    sharpen_gpu,
    white_balance_matrix,
)
from raw_alchemy.pipeline.ops import Op


ImageCallback = Callable[[np.ndarray], np.ndarray]
AutoGainResolver = Callable[[np.ndarray, str], float]
AbortCallback = Callable[[], bool]


@lru_cache(maxsize=32)
def _colourspace(name: str | None):
    return (
        colour.RGB_COLOURSPACES[name]
        if name
        else utils.get_working_colourspace()
    )


@lru_cache(maxsize=64)
def _rgb_matrix(source_name: str | None, target_name: str) -> np.ndarray:
    matrix = np.ascontiguousarray(
        colour.matrix_RGB_to_RGB(_colourspace(source_name), _colourspace(target_name)),
        dtype=np.float32,
    )
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=32)
def _luminance_coeffs(working_space: str | None) -> np.ndarray:
    coeffs = np.ascontiguousarray(
        utils.get_luminance_coeffs(_colourspace(working_space)), dtype=np.float32
    )
    coeffs.setflags(write=False)
    return coeffs


def _lut3d_arrays(lut: colour.LUT3D) -> tuple[np.ndarray, np.ndarray]:
    table = getattr(lut, "_raw_alchemy_table32", None)
    domain = getattr(lut, "_raw_alchemy_domain32", None)
    if table is None:
        table = np.ascontiguousarray(lut.table, dtype=np.float32)
    if domain is None:
        domain = np.ascontiguousarray(lut.domain, dtype=np.float32)
    return table, domain


class PipelineAborted(Exception):
    """A preview run was cooperatively cancelled between ops (T7.3).

    Raised by PreviewExecutor.run_result when its ``should_abort`` callback
    reports that a newer request superseded the in-flight one. Prefix-cache
    entries built before the abort are kept, so the superseding run resumes
    from the completed stages. Callers must discard the run silently (no
    result emission)."""


class PipelineResult:
    """Result of a pipeline run.

    ``gpu`` is the compatibility name for the pooled numpy working buffer;
    ``image`` materializes an independent stable copy only when requested.
    The result owns a strong reference to its GpuImage wrapper so eviction
    cannot recycle a buffer while a caller still reads it.
    """

    __slots__ = ("_image", "applied_ev", "gpu")

    def __init__(
        self,
        image: np.ndarray | None = None,
        applied_ev: float = 0.0,
        gpu: GpuImage | None = None,
    ):
        self._image = image
        self.applied_ev = applied_ev
        self.gpu = gpu

    @property
    def image(self) -> np.ndarray:
        """Host copy of the result (downloads from the GPU on first access)."""
        if self._image is None and self.gpu is not None:
            self._image = self.gpu.to_numpy()
        return self._image

    @property
    def nbytes(self) -> int:
        """Device-side (preferred) or host-side byte size of the result."""
        if self.gpu is not None and self.gpu.valid:
            return self.gpu.nbytes
        if self._image is not None:
            return int(self._image.nbytes)
        return 0


class _CachedStage:
    """Pooled host prefix-cache entry: op output buffer + applied EV."""

    __slots__ = ("gpu", "applied_ev")

    def __init__(self, gpu: GpuImage, applied_ev: float):
        self.gpu = gpu
        self.applied_ev = applied_ev


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
        self._source_input: np.ndarray | None = None
        self.denoiser = denoiser
        self.lens_corrector = lens_corrector
        self.auto_gain_resolver = auto_gain_resolver
        self.round_exposure_gain = round_exposure_gain
        self.last_applied_ev = 0.0
        if source is not None:
            self.set_source(source)

    def set_source(self, source: np.ndarray) -> bool:
        """Set the pipeline source array.

        Returns True only when the source actually changed. Identity is
        checked against both the internally stored array and the original
        caller-provided object, so repeatedly passing the same array never
        counts as a change even when a dtype/contiguity conversion forced an
        internal copy on first ingestion.
        """
        if source is self._source or source is self._source_input:
            return False
        self._source_input = source
        if source.dtype != np.float32:
            source = source.astype(np.float32)
        if not source.flags["C_CONTIGUOUS"]:
            source = np.ascontiguousarray(source)
        self._source = source
        return True

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
        i = 0
        while i < len(ops):
            j = self._grade_fuse_end(ops, i)
            if j is not None:
                buf = self._apply_op(buf, Op("grade_fused", tuple(ops[i:j + 1])))
                i = j + 1
            else:
                buf = self._apply_op(buf, ops[i])
                i += 1
        clip_inplace(buf.arr)
        image = buf.to_numpy()
        buf.clear()  # recycle the working buffer (export path materializes to host)
        return PipelineResult(image, self.last_applied_ev)

    def _slice_crop(self, src: GpuImage, op: Op) -> GpuImage:
        """crop 的只读切片实现:不触碰 src(T7.9 恢复点免整帧复制的前提)。"""
        dst = GpuImage()
        apply_crop_gpu(src, dst, op.params[0])
        return dst

    def _slice_roi(self, src: GpuImage, op: Op) -> GpuImage:
        """roi 的只读切片实现(T7.5 语义不变,见原注释):
        preview-only 可见区裁剪;6-tuple 形式尾随 DPI 挡位下采样。
        不触碰 src——run_result 的恢复点借用缓存前缀时直接调用。"""
        x, y, w, h = op.params[:4]
        dst = GpuImage()
        if len(op.params) == 6:
            import cv2

            out_w, out_h = int(op.params[4]), int(op.params[5])
            if 0 < out_w < int(w) or 0 < out_h < int(h):
                # Crop+resize directly from the source view. The previous
                # path first copied the native ROI, copied it again via
                # to_numpy(), then resized — a full-frame quality-base crop
                # could transiently allocate two extra 45/61MP buffers.
                cx = max(0, min(int(x), src.width - 1))
                cy = max(0, min(int(y), src.height - 1))
                cw = max(1, min(int(w), src.width - cx))
                ch = max(1, min(int(h), src.height - cy))
                dst._allocate(out_h, out_w, src._channels)
                cv2.resize(
                    src.arr[cy:cy + ch, cx:cx + cw], (out_w, out_h),
                    dst=dst.arr,
                    interpolation=cv2.INTER_AREA,
                )
                return dst
        apply_crop_pixels_gpu(src, dst, int(x), int(y), int(w), int(h))
        return dst

    def _apply_op(self, buf: GpuImage, op: Op) -> GpuImage:
        if op.name == "denoise":
            if self.denoiser is None:
                raise RuntimeError("denoise op requires an executor denoiser callback")
            denoised = self.denoiser(buf.arr)
            if denoised is not buf.arr:
                buf.upload(denoised)
            return buf

        if op.name == "lens_correct":
            if self.lens_corrector is None:
                raise RuntimeError("lens_correct op requires an executor lens_corrector callback")
            corrected = self.lens_corrector(buf.arr)
            if corrected is not buf.arr:
                buf.upload(corrected)
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
            dst = self._slice_crop(buf, op)
            buf.clear()
            return dst

        if op.name == "roi":
            dst = self._slice_roi(buf, op)
            buf.clear()
            return dst

        if op.name == "grade_fused":
            # Synthetic execution-time op: params is the fused sub-sequence
            # [exposure, (wb|hs|sc)*, srgb_out]. Never part of op-list cache
            # keys — the loops build it on the fly via _grade_fuse_end.
            return self._apply_grade_fused(buf, op.params)

        if op.name == "exposure":
            apply_gain_inplace(buf.arr, self._resolve_exposure_gain(buf, op.params))
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
            apply_highlight_shadow_inplace(
                buf.arr,
                float(highlight) / 100.0,
                float(shadow) / 100.0,
                _luminance_coeffs(working_space),
            )
            return buf

        if op.name == "sat_contrast":
            saturation, contrast, *rest = op.params
            working_space = rest[0] if rest else None
            apply_saturation_contrast_inplace(
                buf.arr,
                float(saturation),
                float(contrast),
                0.18,
                _luminance_coeffs(working_space),
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
            apply_matrix_inplace(
                buf.arr, _rgb_matrix(working_space, log_color_space_name)
            )
            max_inplace(buf.arr, 1e-6)
            if not log_encode_gpu(buf.arr, log_curve_name):
                encoded = colour.cctf_encoding(buf.arr, function=log_curve_name)
                replacement = GpuImage()
                replacement.adopt(encoded.astype(np.float32, copy=False))
                buf.clear()
                buf = replacement
            return buf

        if op.name == "lut":
            lut = utils.load_lut_cached(op.params[0])
            if isinstance(lut, colour.LUT3D):
                table, domain = _lut3d_arrays(lut)
                apply_lut_inplace(buf.arr, table, domain[0], domain[1])
            else:
                replacement = GpuImage()
                replacement.adopt(lut.apply(buf.arr).astype(np.float32, copy=False))
                buf.clear()
                buf = replacement
            return buf

        if op.name == "srgb_out":
            working_space = op.params[0] if op.params else None
            apply_matrix_inplace(buf.arr, _rgb_matrix(working_space, "sRGB"))
            linear_to_srgb_inplace(buf.arr)
            return buf

        if op.name == "pq_out":
            working_space, output_space, transfer_function, peak_nits, mastering_nits = op.params
            apply_matrix_inplace(buf.arr, _rgb_matrix(working_space, output_space))
            if transfer_function == "ST 2084":
                pq_encode_inplace(buf.arr, peak_nits, mastering_nits)
            else:
                # Non-default transfer functions retain the generic fallback.
                host = buf.arr
                np.maximum(host, 0.0, out=host)
                # colour's ST 2084-style transfer functions accept absolute
                # display luminance (cd/m²), not a pre-normalized fraction.
                host *= float(peak_nits)
                encoded = colour.cctf_encoding(host, function=transfer_function)
                np.clip(encoded, 0.0, 1.0, out=encoded)
                replacement = GpuImage()
                replacement.adopt(encoded.astype(np.float32, copy=False))
                buf.clear()
                buf = replacement
            return buf

        if op.name == "sharpen":
            sharpen_gpu(buf, strength=float(op.params[0]), sigma=1.0)
            return buf

        raise ValueError(f"Unknown pipeline op: {op.name}")

    def _resolve_exposure_gain(self, buf: GpuImage, exposure_params) -> float:
        """Shared Manual/Auto gain resolution (exposure op + fused grade)."""
        exposure_mode, exposure, metering_mode, *rest = exposure_params
        working_space = rest[0] if rest else None
        if exposure_mode == "Manual":
            gain = 2.0 ** float(exposure)
            self.last_applied_ev = float(exposure)
        else:
            metering_img = buf.arr
            if self.auto_gain_resolver is not None:
                gain = self.auto_gain_resolver(metering_img, metering_mode)
            else:
                gain = metering.get_metering_strategy(metering_mode).calculate_gain(
                    metering_img,
                    _colourspace(working_space),
                )
            self.last_applied_ev = float(np.log2(gain))
        return round(float(gain), 4) if self.round_exposure_gain else float(gain)

    _GRADE_MID_OPS = ("white_balance", "highlight_shadow", "sat_contrast")

    @staticmethod
    def _lut3d_cached(path):
        """The loaded LUT when it is a 3D LUT (fusable), else None."""
        try:
            lut = utils.load_lut_cached(path)
        except Exception:
            return None
        return lut if isinstance(lut, colour.LUT3D) else None

    def _grade_fuse_end(self, ops: Sequence[Op], i: int):
        """End index of a fusable colour tail starting at ops[i] (exposure).

        Execution-time fusion only: the op list (and every cache key derived
        from it) is untouched — the executor just runs the window as one GPU
        call when a grade graph covers it. Fusable windows:
          A: exposure, (wb|hs|sc)*, srgb_out
          B: exposure, (wb|hs|sc)*, log_transform[, lut(3D)]
          C: exposure, (wb|hs|sc)*, lut(3D), srgb_out
        Returns None when not fusable (HDR/1D-file-LUT/unknown chains run the
        classic per-op path).
        """
        if ops[i].name != "exposure":
            return None
        from raw_alchemy.onnx import grade

        if not grade.is_enabled():
            return None
        j = i + 1
        while j < len(ops) and ops[j].name in self._GRADE_MID_OPS:
            j += 1
        if j >= len(ops):
            return None
        if ops[j].name == "srgb_out":
            return j
        if ops[j].name == "lut":
            if self._lut3d_cached(ops[j].params[0]) is None:
                return None
            if j + 1 < len(ops) and ops[j + 1].name == "srgb_out":
                return j + 1
            return None
        if ops[j].name == "log_transform":
            from raw_alchemy.pipeline.log_encoding import get_log_lut

            log_curve = ops[j].params[-1]
            if get_log_lut(log_curve).size == 0:
                return None
            if j + 1 < len(ops) and ops[j + 1].name == "lut":
                if self._lut3d_cached(ops[j + 1].params[0]) is None:
                    return None
                return j + 1
            return j
        return None

    def _apply_grade_fused(self, buf: GpuImage, seq: Sequence[Op]) -> GpuImage:
        """The colour tail as one GPU call. Window shapes (see
        _grade_fuse_end): A ends in srgb_out, B ends in log_transform[,lut],
        C is lut+srgb_out. On any GPU failure the window is replayed through
        the classic per-op handlers (identical math)."""
        exposure_params = seq[0].params
        mid = {op.name: op.params for op in seq[1:] if op.name in self._GRADE_MID_OPS}
        tail = [op for op in seq[1:] if op.name not in self._GRADE_MID_OPS]
        tail_names = [op.name for op in tail]

        gain = self._resolve_exposure_gain(buf, exposure_params)
        exposure_rest = exposure_params[3:]
        working_space = exposure_rest[0] if exposure_rest else None
        luma = _luminance_coeffs(working_space)

        wb_params = mid.get("white_balance")
        hs_params = mid.get("highlight_shadow")
        sc_params = mid.get("sat_contrast")
        if wb_params is not None:
            wb_temp, wb_tint, *wb_rest = wb_params
            mat_a = white_balance_matrix(
                wb_temp, wb_tint, wb_rest[0] if wb_rest else None
            ).astype(np.float32)
        else:
            mat_a = np.eye(3, dtype=np.float32)
        highlight = float(hs_params[0]) / 100.0 if hs_params is not None else 0.0
        shadow = float(hs_params[1]) / 100.0 if hs_params is not None else 0.0
        saturation = float(sc_params[0]) if sc_params is not None else 1.0
        contrast = float(sc_params[1]) if sc_params is not None else 1.0

        core = dict(
            gain=gain, mat_a=mat_a, highlight=highlight, shadow=shadow,
            saturation=saturation, contrast=contrast, pivot=0.18, luma=luma,
        )

        from raw_alchemy.onnx import grade

        try:
            if tail_names == ["srgb_out"]:
                mat_b = _rgb_matrix(working_space, "sRGB")
                out = grade.apply_grade(
                    buf.arr, mat_b=mat_b, srgb_encode=True, **core
                )
            elif tail_names == ["lut", "srgb_out"]:
                lut = self._lut3d_cached(tail[0].params[0])
                lut_table, lut_domain = _lut3d_arrays(lut)
                mat_b = _rgb_matrix(working_space, "sRGB")
                out = grade.apply_grade_lut(
                    buf.arr,
                    lut3d_flat=lut_table.reshape(-1, 3),
                    lut3d_size=lut_table.shape[0],
                    d3_min=lut_domain[0], d3_max=lut_domain[1],
                    mat_b=mat_b, **core,
                )
            elif tail_names[0] == "log_transform":
                from raw_alchemy.pipeline.log_encoding import (
                    LOG_LUT_DOMAIN_MAX, LOG_LUT_DOMAIN_MIN, get_log_lut,
                )

                lp = tail[0].params
                log_cs_name = lp[2] if len(lp) == 4 else lp[1]
                log_curve = lp[-1]
                mat_log = _rgb_matrix(working_space, log_cs_name)
                lut3 = (
                    self._lut3d_cached(tail[1].params[0])
                    if len(tail) > 1 else None
                )
                lut3_table, lut3_domain = (
                    _lut3d_arrays(lut3) if lut3 is not None else (None, None)
                )
                out = grade.apply_grade_log(
                    buf.arr,
                    mat_log=mat_log, lut1d=get_log_lut(log_curve),
                    d1_min=LOG_LUT_DOMAIN_MIN, d1_max=LOG_LUT_DOMAIN_MAX,
                    lut3d_flat=(lut3_table.reshape(-1, 3) if lut3 is not None else None),
                    lut3d_size=(lut3_table.shape[0] if lut3 is not None else 0),
                    d3_min=(lut3_domain[0] if lut3 is not None else None),
                    d3_max=(lut3_domain[1] if lut3 is not None else None),
                    **core,
                )
            else:
                raise ValueError(f"unfusable window {tail_names}")
            dst = GpuImage()
            dst.adopt(out)
            buf.clear()
            return dst
        except Exception as e:
            logger.warning(f"grade GPU path failed ({type(e).__name__}: {e}); per-op replay")

        # Fallback: replay the window through the classic per-op handlers.
        # _resolve_exposure_gain already ran (Auto metering must not run
        # twice), so apply the resolved gain directly, then the rest.
        apply_gain_inplace(buf.arr, gain)
        for sub in seq[1:]:
            buf = self._apply_op(buf, sub)
        return buf


class ExportExecutor(_BaseExecutor):
    def run(self, ops: Sequence[Op], source: np.ndarray | None = None) -> np.ndarray:
        return self.run_result(ops, source).image

    def run_result(self, ops: Sequence[Op], source: np.ndarray | None = None) -> PipelineResult:
        return self._run_direct_result(ops, source)


# Historical key kept for import/test compatibility. The interactive buffers
# are now host-backed numpy arrays, so retaining a length-0 "GPU source"
# duplicates an entire RAW frame without avoiding any transfer: upload() and
# copy_from() are both host memcpy operations. New runs therefore ingest the
# cache-owned source directly into their one working buffer.
_SOURCE_PREFIX_KEY = hash(())


class PreviewExecutor(_BaseExecutor):
    def __init__(
        self,
        source: np.ndarray | None = None,
        *,
        cache_budget_bytes: int | None = None,
        **kwargs,
    ):
        # Pooled numpy, pre-clip op outputs keyed by hash(tuple(ops[:i])).
        self._prefix_cache: dict[int, _CachedStage] = {}
        # Prefix length (op count) per prefix-cache key, used by trim() to
        # drop the cheapest-to-recompute (longest) prefixes first.
        self._prefix_lengths: dict[int, int] = {}
        # Post-clip final result of the most recent run, keyed by
        # hash(tuple(ops)). Kept to a single entry; trim() enforces the
        # overall byte budget (T7.6). The stored PipelineResult keeps its
        # pooled working buffer; the worker converts it to uint8 in chunks.
        self._final_cache: dict[int, PipelineResult] = {}
        self._cache_budget_bytes = (
            None if cache_budget_bytes is None else max(0, int(cache_budget_bytes))
        )
        super().__init__(source, **kwargs)

    def set_cache_budget(self, budget_bytes: int | None):
        """Set the live cache cap and immediately shed any excess.

        Enforcing the budget while a run is building prefixes avoids the old
        high-water behaviour where a full pipeline first allocated every
        stage and was trimmed only after the frame completed.
        """
        self._cache_budget_bytes = (
            None if budget_bytes is None else max(0, int(budget_bytes))
        )
        if self._cache_budget_bytes is not None:
            self.trim(self._cache_budget_bytes)

    def _reserve_cache_space(self, incoming_bytes: int) -> bool:
        """Make room for one cache entry; False when it cannot ever fit."""
        if self._cache_budget_bytes is None:
            return True
        incoming = max(0, int(incoming_bytes))
        if incoming > self._cache_budget_bytes:
            return False
        self.trim(self._cache_budget_bytes - incoming)
        return True

    def set_source(self, source: np.ndarray) -> bool:
        # Only invalidate the caches when the source array actually changed.
        # The GUI worker passes `source=` on every request, so clearing
        # unconditionally here would reduce the prefix-cache hit rate to zero.
        # Semantic invalidation (proxy/full switch, lens/denoise state) is the
        # worker's job: see ImageProcessor._invalidate_executor_prefix_if_needed.
        changed = super().set_source(source)
        if changed:
            self.clear_cache()
        return changed

    def clear_cache(self):
        """Drop all cached intermediate and final pipeline results.

        Prefix buffers are executor-internal (sole owner) and go straight
        back to the buffer pool. Final results may still be referenced by
        callers, so they are recycled only when this cache holds the last
        reference; otherwise the buffer stays valid for the holder and is
        freed by taichi when the result is eventually collected.
        """
        self._drop_prefix_entries(self._prefix_cache.keys())
        self._drop_final_entries()

    def _drop_prefix_entries(self, keys) -> int:
        """Pop prefix entries, recycle their host buffers; return bytes freed."""
        freed = 0
        for key in list(keys):
            stage = self._prefix_cache.pop(key, None)
            self._prefix_lengths.pop(key, None)
            if stage is not None:
                freed += stage.gpu.nbytes
                stage.gpu.clear()
        return freed

    def _drop_final_entries(self) -> int:
        """Empty the final-result cache; return bytes dropped from it.

        A final's working buffer is returned to the pool only when the cache is
        the sole owner of the PipelineResult (refcount check). Handed-out
        results keep their buffer alive — pooling must never steal a buffer
        a caller could still read (and must not rely on __del__: see
        GpuImage.clear for why finalizer-time pooling is unsafe).
        """
        freed = 0
        while self._final_cache:
            _, result = self._final_cache.popitem()
            freed += result.nbytes
            # 2 == this local `result` + getrefcount's own argument.
            if result.gpu is not None and sys.getrefcount(result) == 2:
                result.gpu.clear()
                result.gpu = None
        return freed

    def cache_bytes(self) -> int:
        """Total host bytes held by the prefix/final result caches."""
        total = sum(stage.gpu.nbytes for stage in self._prefix_cache.values())
        total += sum(result.nbytes for result in self._final_cache.values())
        return int(total)

    def trim(self, budget_bytes: int) -> int:
        """Evict cached pipeline results until ``cache_bytes() <= budget``.

        Longest prefixes are dropped first: tail (color) ops are cheap to
        re-run, while short prefixes hold the expensive early stages
        (denoise, lens correction, geometry). The final post-clip result is
        dropped last because it powers the zoom-only full-hit path (T7.1).
        Returns the number of bytes freed.
        """
        budget = max(0, int(budget_bytes))
        total = self.cache_bytes()
        freed = 0
        while total - freed > budget and self._prefix_lengths:
            key = max(self._prefix_lengths, key=self._prefix_lengths.get)
            freed += self._drop_prefix_entries([key])
        if total - freed > budget and self._final_cache:
            freed += self._drop_final_entries()
        return freed

    def run(
        self,
        ops: Sequence[Op],
        source: np.ndarray | None = None,
        *,
        should_abort: AbortCallback | None = None,
    ) -> np.ndarray:
        return self.run_result(ops, source, should_abort=should_abort).image

    def run_result(
        self,
        ops: Sequence[Op],
        source: np.ndarray | None = None,
        *,
        should_abort: AbortCallback | None = None,
    ) -> PipelineResult:
        """Run ``ops``, resuming from cached prefixes when possible.

        ``should_abort`` (T7.3) is polled at op boundaries; when it returns
        True the run raises PipelineAborted. Already-snapshotted prefix
        stages stay cached so the superseding run picks up from them.
        """
        if source is not None:
            self.set_source(source)
        src = self._resolve_source(None)

        # Full hit (same source, identical op list — e.g. only zoom/viewport
        # changed): return the cached post-clip result directly, no GPU work.
        ops_key = hash(tuple(ops))
        final_result = self._final_cache.get(ops_key)
        if final_result is not None:
            self.last_applied_ev = final_result.applied_ev
            return final_result

        # Abort before any cache invalidation or GPU work: a request that is
        # already superseded on entry leaves every cache untouched.
        if should_abort is not None and should_abort():
            raise PipelineAborted("preview run superseded before start")

        prefix_keys = [hash(tuple(ops[:index])) for index in range(1, len(ops) + 1)]

        start_index = 0
        cached_stage = None
        for index in range(len(ops), 0, -1):
            stage = self._prefix_cache.get(prefix_keys[index - 1])
            if stage is not None:
                cached_stage = stage
                start_index = index
                break

        self.last_applied_ev = cached_stage.applied_ev if cached_stage is not None else 0.0

        # The previous final result is stale for this run; drop it up front so
        # its buffer can be recycled by this very run (unless a caller still
        # holds the PipelineResult, in which case it keeps its buffer alive).
        self._drop_final_entries()
        # Keep only the latest generation of prefixes: entries outside the
        # current op sequence can never hit again (their tail params changed),
        # so recycle their VRAM before allocating this run's buffers. The
        # length-0 source entry is never stale for op changes — its validity
        # depends only on the source, and set_source clears it on change.
        current_keys = set(prefix_keys)
        self._drop_prefix_entries(
            [key for key in self._prefix_cache if key not in current_keys]
        )

        buf = GpuImage()
        if cached_stage is not None:
            resume_op = ops[start_index] if start_index < len(ops) else None
            if resume_op is not None and resume_op.name in ("roi", "crop"):
                # T7.9:恢复点是非破坏性裁剪(平移/缩放只改 roi 及其后)时,
                # 直接在缓存前缀上只读切片,跳过整帧 copy_from——
                # 45MP≈540MB 的主机 memcpy 是 zoom>1 交互的最大稳态成本。
                if should_abort is not None and should_abort():
                    raise PipelineAborted("preview run superseded before resume slice")
                buf = (self._slice_roi if resume_op.name == "roi"
                       else self._slice_crop)(cached_stage.gpu, resume_op)
                prefix_key = prefix_keys[start_index]
                old_stage = self._prefix_cache.get(prefix_key)
                if old_stage is not None:
                    old_stage.gpu.clear()
                    self._prefix_cache.pop(prefix_key, None)
                    self._prefix_lengths.pop(prefix_key, None)
                if self._reserve_cache_space(buf.nbytes):
                    snapshot = GpuImage()
                    snapshot.copy_from(buf)
                    self._prefix_cache[prefix_key] = _CachedStage(
                        snapshot, self.last_applied_ev
                    )
                    self._prefix_lengths[prefix_key] = start_index + 1
                start_index += 1
            else:
                # Resume from the resident prefix (T7.2): whole-frame copy —
                # the following op mutates the working buffer in place.
                buf.copy_from(cached_stage.gpu)
        else:
            # A leading ROI/crop is read-only, so slice the cache-owned source
            # directly instead of first copying the entire native frame into
            # a working buffer. This is the common pan/zoom and quality-base
            # path when no shape op precedes the colour tail.
            if ops and ops[0].name in ("roi", "crop"):
                borrowed = GpuImage()
                borrowed._arr = src
                borrowed._height, borrowed._width = src.shape[:2]
                borrowed._channels = src.shape[2] if src.ndim == 3 else 1
                buf = (self._slice_roi if ops[0].name == "roi"
                       else self._slice_crop)(borrowed, ops[0])
                prefix_key = prefix_keys[0]
                if self._reserve_cache_space(buf.nbytes):
                    snapshot = GpuImage()
                    snapshot.copy_from(buf)
                    self._prefix_cache[prefix_key] = _CachedStage(snapshot, 0.0)
                    self._prefix_lengths[prefix_key] = 1
                start_index = 1
            else:
                # Host-backed source ingestion. Keeping a second resident
                # source frame would consume hundreds of MiB and still
                # perform the same memcpy on every run.
                buf.upload(src)

        pos = start_index
        while pos < len(ops):
            op = ops[pos]
            # Cooperative cancellation (T7.3): checked at every op boundary.
            # Prefixes snapshotted so far stay cached; only the working
            # buffer (already snapshotted last stage) is recycled.
            if should_abort is not None and should_abort():
                buf.clear()
                raise PipelineAborted(
                    f"preview run superseded before op {pos + 1}/{len(ops)} ({op.name})"
                )
            # Colour-tail fusion: run [exposure..srgb_out] as one GPU call.
            # Cache keys stay per-op-list; only the post-window stage is
            # snapshotted (mid-window prefixes never materialize, which just
            # means a colour-param change recomputes the fused tail — the
            # tail costs 16-99ms fused vs 0.5-1.2s on the per-op numpy path).
            fuse_end = self._grade_fuse_end(ops, pos)
            if fuse_end is not None:
                buf = self._apply_op(buf, Op("grade_fused", tuple(ops[pos:fuse_end + 1])))
                index = fuse_end + 1
            else:
                buf = self._apply_op(buf, op)
                index = pos + 1
            # Snapshot the (pre-clip) op output into a pooled host buffer.
            prefix_key = prefix_keys[index - 1]
            old_stage = self._prefix_cache.get(prefix_key)
            if old_stage is not None:
                old_stage.gpu.clear()
                self._prefix_cache.pop(prefix_key, None)
                self._prefix_lengths.pop(prefix_key, None)
            if self._reserve_cache_space(buf.nbytes):
                snapshot = GpuImage()
                snapshot.copy_from(buf)
                self._prefix_cache[prefix_key] = _CachedStage(snapshot, self.last_applied_ev)
                self._prefix_lengths[prefix_key] = index
            pos = index

        clip_inplace(buf.arr)
        result = PipelineResult(applied_ev=self.last_applied_ev, gpu=buf)
        if self._reserve_cache_space(result.nbytes):
            self._final_cache[ops_key] = result
        return result
