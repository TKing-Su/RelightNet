from __future__ import annotations

from rendering.display.display_finish import RendererDisplayFinishMixin
from rendering.display.portrait_recipe import RendererPortraitRecipeMixin
from rendering.display.shadows import RendererShadowMixin


class RendererShadowDisplayMixin(
    RendererPortraitRecipeMixin,
    RendererShadowMixin,
    RendererDisplayFinishMixin,
):
    pass
