from __future__ import annotations

from typing import Optional
from lighting.models import LightingInfo


class RendererV32PolicyMixin:
    def _v32_style_key(self, lighting_info: Optional[LightingInfo]) -> str:
        """Current V32 route is continuous and background-budget driven.

        Discrete warm/cyber/natural style routers were legacy compatibility code.
        The active route keeps background expression in the continuous LookPolicy.
        """
        return "continuous"
