from __future__ import annotations

from rendering.look.colored_light import RendererColoredLightMixin
from rendering.look.compact_context import RendererCompactContextMixin
from rendering.look.compact_policy_layer import RendererCompactPolicyLayerMixin
from rendering.look.directional_field import RendererDirectionalFieldMixin
from rendering.look.look_safe_atmosphere import RendererLookSafeAtmosphereMixin


class RendererCompactMixin(
    RendererCompactContextMixin,
    RendererColoredLightMixin,
    RendererDirectionalFieldMixin,
    RendererLookSafeAtmosphereMixin,
    RendererCompactPolicyLayerMixin,
):
    pass
