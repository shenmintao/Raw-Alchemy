"""Shared native-resolution, pre-lens float32 denoise artifacts.

The caller must supply the actual decode provenance. Only canonical decodes
may seed the canonical export cache; proxies and LibRaw preloads cannot.
Cancellation is cooperative between stages/tiles, not a compile timeout.
"""
import time

from loguru import logger

from . import denoise_disk_cache
from .cancellation import check_cancelled
from .resources import checkpoint
from .executor import PipelineAborted
from .stage_identity import DECODE_CANONICAL, denoise_tag, source_identity


def resolve_denoised_source(raw_path, strength, *, decode=None, denoise=None,
                            source=None, decode_variant=DECODE_CANONICAL,
                            should_abort=None, progress_callback=None):
    def check():
        check_cancelled()
        if should_abort is not None and should_abort():
            raise PipelineAborted("denoise source superseded or stopping")

    def progress(current, total):
        check()
        if progress_callback is not None:
            progress_callback(current, total)

    check()
    t0 = time.perf_counter()
    generation = source_identity(raw_path)
    tag = denoise_tag(strength, decode_variant=decode_variant)

    def check_identity():
        check()
        if generation != source_identity(raw_path):
            raise PipelineAborted("source changed during denoising")
        if tag != denoise_tag(strength, decode_variant=decode_variant):
            raise PipelineAborted("model/policy changed during denoising")

    cached = denoise_disk_cache.load(raw_path, tag, source_token=generation) if tag is not None else None
    check_identity()
    if cached is not None:
        logger.info(f"[StageTiming] denoise artifact hit {time.perf_counter() - t0:.3f}s")
        return cached
    if source is None:
        if decode is None:
            from raw_alchemy.native_decode import decode_raw
            decode = decode_raw
        source = decode(raw_path)
    checkpoint()
    check()
    t_decode = time.perf_counter()
    if denoise is None:
        from raw_alchemy.onnx.rgb_denoiser import denoise_rgb_linear
        denoise = denoise_rgb_linear
    kwargs = {"strength": strength, "progress_callback": progress}
    result = denoise(source, **kwargs)
    check_identity()
    checkpoint()
    t_denoise = time.perf_counter()
    # Never publish data computed across a source-file replacement.
    if tag is not None and generation is not None:
        denoise_disk_cache.save(raw_path, tag, result, source_token=generation)
    check_identity()
    logger.info(
        f"[StageTiming] source/cache/decode={t_decode - t0:.3f}s "
        f"denoise={t_denoise - t_decode:.3f}s "
        f"cache-write={time.perf_counter() - t_denoise:.3f}s"
    )
    return result
