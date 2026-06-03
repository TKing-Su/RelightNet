from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PortraitLight:
    name: str
    direction: Tuple[float, float, float]
    color: Tuple[float, float, float]
    intensity: float
    size: float
    diffuse_scale: float
    specular_scale: float
    rim_scale: float


@dataclass
class LightingInfo:
    ambient_color: Tuple[float, float, float]
    ambient_intensity: float
    key_color: Tuple[float, float, float]
    key_intensity: float
    lights: List[Dict[str, object]]
    global_mean_color: Tuple[float, float, float]
    palette_points: List[Dict[str, object]]
    palette_diversity: float
    hue_entropy: float
    dominant_hue_share: float
    adaptive_light_count: int
    background_mode: str = 'balanced'
    neon_strength: str = 'off'
    gradient_field: Optional[Dict[str, object]] = None


@dataclass
class LookPolicy:
    """Single look-safe routing object.

    The core route reads the continuous atmosphere budget through this policy
    instead of re-classifying the background by filename or discrete
    warm/cyber/natural branches.
    """
    route: str
    creative_profile: str
    v32_style: str
    filename_style_hints_enabled: bool
    extractor_style: str
    descriptor: Dict[str, object]
    style_expression: Dict[str, float]
    exposure: Dict[str, float]
    chroma: Dict[str, float]
    direction: Dict[str, float]
    region: Dict[str, float]
    render_weight: Dict[str, float]
    display: Dict[str, float]
    budget: Dict[str, object]
