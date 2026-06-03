from __future__ import annotations

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


class RendererCompactContextMixin:
    def _compact_policy_context(
        self,
        look_policy: LookPolicy,
        lighting_info: Optional[LightingInfo],
    ) -> Dict[str, object]:
        direction = look_policy.direction if isinstance(look_policy.direction, dict) else {}
        chroma = look_policy.chroma if isinstance(look_policy.chroma, dict) else {}
        exposure = look_policy.exposure if isinstance(look_policy.exposure, dict) else {}
        render_weight = look_policy.render_weight if isinstance(look_policy.render_weight, dict) else {}
        display = look_policy.display if isinstance(look_policy.display, dict) else {}
        region = look_policy.region if isinstance(look_policy.region, dict) else {}
        descriptor = look_policy.descriptor if isinstance(look_policy.descriptor, dict) else {}
        budget = look_policy.budget if isinstance(look_policy.budget, dict) else {}

        lowkey_chroma_gate = float(np.clip(budget.get('lowkey_chroma_direction_gate', 0.0), 0.0, 1.0))
        air_skin_guard = float(np.clip(budget.get('low_chroma_air_skin_guard', 0.0), 0.0, 1.0))
        bg_luma = float(np.clip(descriptor.get('global_luma', 0.25), 0.0, 1.0))
        bg_sat = float(np.clip(descriptor.get('average_saturation', descriptor.get('colorfulness', 0.25)), 0.0, 1.0))
        bg_colorfulness = float(np.clip(descriptor.get('colorfulness', bg_sat), 0.0, 1.0))
        bg_diversity = float(np.clip(descriptor.get('palette_diversity', 0.35), 0.0, 1.0))
        bg_flatness = float(np.clip(descriptor.get('flatness_score', 0.0), 0.0, 1.0))
        bg_haze = float(np.clip(descriptor.get('haze_score', 0.0), 0.0, 1.0))
        bg_lowkey = float(np.clip(descriptor.get('lowkey_score', 0.0), 0.0, 1.0))
        bg_highkey = float(np.clip(descriptor.get('highkey_score', 0.0), 0.0, 1.0))
        chroma_pressure = float(np.clip(0.55 * bg_colorfulness + 0.30 * bg_sat + 0.15 * bg_diversity, 0.0, 1.0))

        field = getattr(lighting_info, 'gradient_field', {}) if lighting_info is not None else {}
        field = field if isinstance(field, dict) else {}
        color_key_uv = direction.get('key_uv', field.get('key_uv', descriptor.get('brightest_region_uv', [0.5, 0.32])))
        try:
            color_key_uv = [float(np.clip(color_key_uv[0], 0.0, 1.0)), float(np.clip(color_key_uv[1], 0.0, 1.0))]
        except Exception:
            color_key_uv = [0.5, 0.32]

        hb = float(descriptor.get('horizontal_bias', descriptor.get('left_right_luma_diff', field.get('horizontal_bias', 0.0))))
        vb = float(descriptor.get('vertical_bias', descriptor.get('top_bottom_luma_diff', field.get('vertical_bias', 0.0))))
        bg_bias_sum = abs(hb) + abs(vb)
        bg_bias_strength = float(np.clip(bg_bias_sum / 0.12, 0.0, 1.0))
        dark_chroma_env = float(np.clip(
            np.clip((0.24 - bg_luma) / 0.18, 0.0, 1.0)
            * np.clip((chroma_pressure - 0.24) / 0.46, 0.0, 1.0)
            * (0.55 + 0.45 * bg_diversity),
            0.0,
            1.0,
        ))
        warm_direction_env = float(np.clip(
            float(descriptor.get('warm_presence', descriptor.get('warm_ratio', 0.0)))
            * bg_sat
            * np.clip((bg_bias_strength + float(descriptor.get('local_light_confidence', 0.0))) * 0.5, 0.0, 1.0)
            * np.clip((0.52 - bg_luma) / 0.34, 0.0, 1.0),
            0.0,
            1.0,
        ))
        if bg_bias_sum > 1e-5:
            bg_key_uv = [
                float(np.clip(0.5 + 0.46 * hb / bg_bias_sum, 0.04, 0.96)),
                float(np.clip(0.5 + 0.46 * vb / bg_bias_sum, 0.04, 0.96)),
            ]
            key_uv = [
                float(np.clip(0.90 * bg_key_uv[0] + 0.10 * color_key_uv[0], 0.0, 1.0)),
                float(np.clip(0.90 * bg_key_uv[1] + 0.10 * color_key_uv[1], 0.0, 1.0)),
            ]
        else:
            key_uv = list(color_key_uv)
        key_dir = safe_norm(np.array(policy_direction_from_uv(key_uv), dtype=np.float32))

        return {
            'direction': direction,
            'chroma': chroma,
            'exposure': exposure,
            'render_weight': render_weight,
            'display': display,
            'region': region,
            'descriptor': descriptor,
            'budget': budget,
            'field': field,
            'lowkey_chroma_gate': lowkey_chroma_gate,
            'air_skin_guard': air_skin_guard,
            'bg_luma': bg_luma,
            'bg_sat': bg_sat,
            'bg_colorfulness': bg_colorfulness,
            'bg_diversity': bg_diversity,
            'bg_flatness': bg_flatness,
            'bg_haze': bg_haze,
            'bg_lowkey': bg_lowkey,
            'bg_highkey': bg_highkey,
            'chroma_pressure': chroma_pressure,
            'hb': hb,
            'vb': vb,
            'bg_bias_sum': bg_bias_sum,
            'bg_bias_strength': bg_bias_strength,
            'dark_chroma_env': dark_chroma_env,
            'warm_direction_env': warm_direction_env,
            'color_key_uv': color_key_uv,
            'key_uv': key_uv,
            'key_dir': key_dir,
        }
