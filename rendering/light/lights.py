from __future__ import annotations

from rendering.environment.environment_lighting import RendererEnvironmentLightingMixin
from rendering.light.lighting_effects import RendererLightingEffectsMixin
from rendering.light.virtual_lights import RendererVirtualLightsMixin


class RendererLightsMixin(
    RendererVirtualLightsMixin,
    RendererLightingEffectsMixin,
    RendererEnvironmentLightingMixin,
):
    pass
