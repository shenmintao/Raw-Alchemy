from dataclasses import dataclass
from typing import Any

from raw_alchemy import config
from raw_alchemy.pipeline.request import ProcessorParams


@dataclass(frozen=True)
class Op:
    name: str
    params: tuple[Any, ...] = ()


def _as_hashable(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _as_hashable(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_as_hashable(item) for item in value)
    return value


def _corners_are_default(corners) -> bool:
    if corners is None:
        return True
    return _as_hashable(corners) == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )


def _crop_is_default(crop) -> bool:
    return not crop or tuple(crop) == (0.0, 0.0, 1.0, 1.0)


def _float_param(params, name: str, default: float) -> float:
    value = params.get(name, default)
    if value is None:
        return default
    return float(value)


def build_op_list(params: ProcessorParams) -> list[Op]:
    """Build the single ordered operation list used by preview/export paths."""
    ops: list[Op] = []
    working_space = config.WORKING_SPACE

    if params.get("denoise_enabled", False):
        ops.append(Op("denoise", ()))

    if params.get("lens_correct", False):
        ops.append(Op("lens_correct", (_as_hashable(params.get("custom_db_path")),)))

    rotation = int(params.get("rotation", 0) or 0) % 360
    flip_h = bool(params.get("flip_horizontal", False))
    flip_v = bool(params.get("flip_vertical", False))
    if rotation != 0 or flip_h or flip_v:
        ops.append(Op("geometry", (rotation, flip_h, flip_v)))

    corners = params.get("perspective_corners")
    if not _corners_are_default(corners):
        ops.append(Op("perspective", (_as_hashable(corners),)))

    crop = params.get("crop", (0.0, 0.0, 1.0, 1.0))
    if not _crop_is_default(crop):
        ops.append(Op("crop", (_as_hashable(crop),)))

    exposure_mode = params.get("exposure_mode", "Auto")
    ops.append(
        Op(
            "exposure",
            (
                exposure_mode,
                _float_param(params, "exposure", 0.0),
                params.get("metering_mode", "matrix"),
                working_space,
            ),
        )
    )

    wb_temp = _float_param(params, "wb_temp", 0.0)
    wb_tint = _float_param(params, "wb_tint", 0.0)
    if wb_temp != 0.0 or wb_tint != 0.0:
        ops.append(Op("white_balance", (wb_temp, wb_tint, working_space)))

    highlight = _float_param(params, "highlight", 0.0)
    shadow = _float_param(params, "shadow", 0.0)
    if highlight != 0.0 or shadow != 0.0:
        ops.append(Op("highlight_shadow", (highlight, shadow, working_space)))

    saturation = _float_param(params, "saturation", 1.0)
    contrast = _float_param(params, "contrast", 1.0)
    if saturation != 1.0 or contrast != 1.0:
        ops.append(Op("sat_contrast", (saturation, contrast, working_space)))

    log_space = params.get("log_space")
    if log_space and log_space != "None":
        ops.append(
            Op(
                "log_transform",
                (
                    log_space,
                    working_space,
                    config.LOG_TO_WORKING_SPACE.get(log_space),
                    config.LOG_ENCODING_MAP.get(log_space, log_space),
                ),
            )
        )

    lut_path = params.get("lut_path")
    if lut_path:
        ops.append(Op("lut", (str(lut_path),)))

    if not log_space or log_space == "None":
        if params.get("hdr_output", False):
            ops.append(
                Op(
                    "pq_out",
                    (
                        working_space,
                        config.HDR_OUTPUT_COLOURSPACE,
                        config.HDR_PQ_TRANSFER_FUNCTION,
                        config.HDR_PEAK_NITS,
                        config.HDR_PQ_MASTERING_NITS,
                    ),
                )
            )
        else:
            ops.append(Op("srgb_out", (working_space,)))

    sharpen_strength = _float_param(params, "sharpen_strength", 0.0)
    if sharpen_strength > 0.0:
        ops.append(Op("sharpen", (sharpen_strength,)))

    return ops
