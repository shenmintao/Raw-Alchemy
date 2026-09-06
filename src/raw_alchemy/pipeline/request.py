from dataclasses import dataclass
from typing import TypedDict, Optional, Tuple, Literal

class ProcessorParams(TypedDict, total=False):
    """Type definition for image processing parameters"""
    # Exposure & WB
    exposure_value: float
    metering_mode: str
    wb_temp: float
    wb_tint: float
    
    # Tone & Color
    contrast: float
    saturation: float
    highlight: float
    shadow: float
    
    # Color Management
    log_space: str
    lut_path: Optional[str]
    
    # Corrections
    lens_correct: bool
    custom_db_path: Optional[str]
    
    # Geometry
    rotation: int
    flip_horizontal: bool
    flip_vertical: bool
    crop: Tuple[float, float, float, float]  # left, top, right, bottom (normalized)
    
    # Perspective
    keystone_h: float
    keystone_v: float
    
    # Internal system flags
    _load: bool
    _preload: bool
    
    # New Features
    denoise_enabled: bool
    sharpen_strength: float
    viewport_size: Tuple[int, int]
    preview_zoom: float
    device_pixel_ratio: float
    max_preview_pixels: int
    tile_preview_pixels: int
    tile_preview_threshold: int
    tile_overlap_pixels: int
    _detail_preview: bool
    perspective_corners: Optional[list]

@dataclass(frozen=True)
class ProcessRequest:
    """Immutable processing request. Eliminates race conditions."""
    path: str
    params: ProcessorParams
    request_id: int
    
    def __post_init__(self):
        object.__setattr__(self, 'params', _snapshot(self.params))


class FrozenParams(dict):
    """Read-only metadata. copy() returns a private mutable working dict."""
    def _readonly(self, *args, **kwargs):
        raise TypeError("request metadata is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly


def _snapshot(value):
    if isinstance(value, dict):
        return FrozenParams({key: _snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, set):
        return frozenset(_snapshot(item) for item in value)
    # Full-resolution arrays are borrowed, pinned references: workers must never
    # mutate published arrays. Deep-copying these would defeat memory admission.
    return value
