from __future__ import annotations

from rendering.reconcile.v32_policy import RendererV32PolicyMixin
from rendering.reconcile.v32_regions import RendererV32RegionsMixin


class RendererV32Mixin(
    RendererV32PolicyMixin,
    RendererV32RegionsMixin,
):
    pass
