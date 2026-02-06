from dataclasses import dataclass, field
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
    denoise_strength: float
    sharpen_strength: float
    viewport_size: Tuple[int, int]
    perspective_corners: Optional[list]

@dataclass
class ProcessRequest:
    """Immutable processing request. Eliminates race conditions."""
    path: str
    params: ProcessorParams
    request_id: int
    
    def __post_init__(self):
        # Defensive copy of mutable dict
        self.params = self.params.copy()

