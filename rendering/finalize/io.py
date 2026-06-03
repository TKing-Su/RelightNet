from __future__ import annotations

from rendering.finalize.batch import RendererBatchMixin
from rendering.finalize.composite import RendererCompositeMixin
from rendering.finalize.debug_outputs import RendererDebugOutputMixin
from rendering.finalize.quality import RendererQualityReportMixin


class RendererIOMixin(
    RendererCompositeMixin,
    RendererQualityReportMixin,
    RendererDebugOutputMixin,
    RendererBatchMixin,
):
    pass
