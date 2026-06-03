from __future__ import annotations

from rendering.environment.background_gradient import RendererBackgroundGradientMixin
from rendering.environment.hdri_arc import RendererHDRIArcMixin
from rendering.environment.hdri_bodylight import RendererHDRIBodyLightMixin
from rendering.environment.pbr_env import RendererPBREnvironmentMixin
from rendering.environment.reflective_finish import RendererReflectiveFinishMixin


class RendererEnvironmentLightingMixin(
    RendererBackgroundGradientMixin,
    RendererReflectiveFinishMixin,
    RendererHDRIBodyLightMixin,
    RendererHDRIArcMixin,
    RendererPBREnvironmentMixin,
):
    pass
