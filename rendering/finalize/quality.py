from __future__ import annotations

import os
import json
from typing import Dict, Optional
import numpy as np
from config.constants import *
from lighting.models import *
from lighting.presets import *
from config.paths import *
from lighting.style_mode import *
from tools.color import *
from tools.filters import *
from tools.geometry import *
from tools.image_io import *
from lighting.background_analyzer import *
from lighting.light_scene import *

class RendererQualityReportMixin:
    def _image_luma_stats(self, img_linear: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        lum = rgb_luminance(np.clip(img_linear, 0.0, 1.0))
        if mask is not None:
            m = np.asarray(mask, dtype=np.float32) > 0.50
            if np.any(m):
                lum = lum[m]
        return {
            'mean': float(np.mean(lum)) if lum.size else 0.0,
            'p10': float(np.percentile(lum, 10.0)) if lum.size else 0.0,
            'p50': float(np.percentile(lum, 50.0)) if lum.size else 0.0,
            'p90': float(np.percentile(lum, 90.0)) if lum.size else 0.0,
        }


    def _write_quality_report(
        self,
        report_file: str,
        source_linear: np.ndarray,
        relit_linear: np.ndarray,
        composite_linear: np.ndarray,
        background_linear: Optional[np.ndarray],
        mask: np.ndarray,
        alpha: np.ndarray,
        depth_map: np.ndarray,
        lighting_info: LightingInfo,
    ) -> None:
        if not self.save_quality_report or not report_file:
            return
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        edge = np.clip(4.0 * alpha * (1.0 - alpha), 0.0, 1.0)
        warnings = []
        if float(edge.mean()) > 0.10:
            warnings.append('wide_or_soft_alpha_edge')
        d_valid = depth_map[np.isfinite(depth_map)]
        if d_valid.size and float(np.percentile(d_valid, 99.0) - np.percentile(d_valid, 1.0)) < 0.08:
            warnings.append('depth_range_too_flat')
        if background_linear is not None:
            bg_stats = self._image_luma_stats(background_linear, None)
        else:
            bg_stats = None
        report = {
            'style_mode': self.style_mode,
            'lighting_pattern_requested': self.lighting_pattern,
            'lighting_pattern_selected': getattr(self, '_last_selected_lighting_pattern', 'natural'),
            'light_source_mode': getattr(self, 'light_source_mode', 'hybrid'),
            'direct_light_strength': float(getattr(self, 'direct_light_strength', 1.55)),
            'specular_boost_strength': float(getattr(self, 'specular_boost_strength', 0.16)),
            'environment_light_scale': float(getattr(self, 'environment_light_scale', 0.55)),
            'key_side': getattr(self, 'key_side', 'auto'),
            'background_mode': getattr(lighting_info, 'background_mode', None),
            'neon_strength': getattr(lighting_info, 'neon_strength', None),
            'subject_luma_source': self._image_luma_stats(source_linear, mask),
            'subject_luma_relit': self._image_luma_stats(relit_linear, mask),
            'composite_luma': self._image_luma_stats(composite_linear, None),
            'background_luma': bg_stats,
            'alpha_edge_mean': float(np.mean(edge)),
            'alpha_coverage': float(np.mean(alpha > 0.50)),
            'depth_min': float(np.min(d_valid)) if d_valid.size else None,
            'depth_max': float(np.max(d_valid)) if d_valid.size else None,
            'ground_shadow_enabled_after_auto': bool(getattr(self, '_last_ground_shadow_auto_enabled', False)),
            'contact_shadow_enabled': bool(getattr(self, 'contact_shadow_enabled', True)),
            'edge_cleanup_strength': float(getattr(self, 'edge_cleanup_strength', 0.0)),
            'warnings': warnings,
        }
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
