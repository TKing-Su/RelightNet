from __future__ import annotations

from rendering.face.face_detail import RendererFaceDetailMixin
from rendering.face.metric_balancing import RendererMetricBalancingMixin
from rendering.face.subject_regions import RendererSubjectRegionsMixin


class RendererFaceBalanceMixin(
    RendererFaceDetailMixin,
    RendererMetricBalancingMixin,
    RendererSubjectRegionsMixin,
):
    pass
