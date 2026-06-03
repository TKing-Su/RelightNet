from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Optional
import numpy as np
from PIL import Image
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

class RendererDebugOutputMixin:
    def _save_debug_intermediates(
        self,
        base_name: str,
        source_linear: np.ndarray,
        mask: np.ndarray,
        albedo_linear: np.ndarray,
        normal_map: np.ndarray,
        depth_map: np.ndarray,
        background_linear: Optional[np.ndarray],
        relit_display: np.ndarray,
    ) -> None:
        debug_dir = os.path.join(self.output_base_path, 'Debug')
        os.makedirs(debug_dir, exist_ok=True)

        def save_rgb(name: str, img: np.ndarray) -> None:
            out = np.clip(linear_to_srgb(np.clip(img, 0.0, 1.0).astype(np.float32)), 0.0, 1.0)
            Image.fromarray((out * 255).astype(np.uint8)).save(os.path.join(debug_dir, f'{base_name}_{name}.png'))

        def save_gray(name: str, gray: np.ndarray) -> None:
            out = np.clip(gray, 0.0, 1.0)
            Image.fromarray((out * 255).astype(np.uint8)).save(os.path.join(debug_dir, f'{base_name}_{name}.png'))

        def save_srgb(name: str, img: np.ndarray) -> None:
            out = np.clip(img, 0.0, 1.0)
            Image.fromarray((out * 255).astype(np.uint8)).save(os.path.join(debug_dir, f'{base_name}_{name}.png'))

        def save_rgb_exact(name: str, img: np.ndarray) -> None:
            out = np.clip(linear_to_srgb(np.clip(img, 0.0, 1.0).astype(np.float32)), 0.0, 1.0)
            Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(os.path.join(debug_dir, name))

        def save_gray_exact(name: str, gray: np.ndarray) -> None:
            out = np.clip(gray.astype(np.float32), 0.0, 1.0)
            Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(os.path.join(debug_dir, name))

        def save_srgb_exact(name: str, img: np.ndarray) -> None:
            out = np.clip(img.astype(np.float32), 0.0, 1.0)
            Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(os.path.join(debug_dir, name))

        save_rgb('01_source', source_linear)
        save_gray('02_alpha', mask)
        save_rgb('03_basecolor', albedo_linear)
        save_srgb('04_normal', normal_map)
        d = depth_map
        d_min, d_max = float(d.min()), float(d.max())
        if d_max - d_min > 1e-6:
            save_gray('05_depth', (d - d_min) / (d_max - d_min))
        else:
            save_gray('05_depth', d)
        if background_linear is not None:
            save_rgb('06_background', background_linear)

        di = self._debug_intermediates
        if 'ambient_fill' in di:
            save_rgb('07_ambient_fill', di['ambient_fill'])
        if 'diffuse' in di:
            save_rgb('08_diffuse', di['diffuse'])
        if 'specular' in di:
            save_rgb('09_specular', di['specular'] * 4.0)
        if 'rim' in di:
            save_rgb('10_rim', di['rim'] * 6.0)
        if 'key_shadow' in di and 'contact_shadow' in di:
            save_gray('11_shadow', di['key_shadow'] * di['contact_shadow'])
        elif 'key_shadow' in di:
            save_gray('11_shadow', di['key_shadow'])
        if 'pre_tonemap' in di:
            save_rgb('12_pre_tonemap', di['pre_tonemap'])
        if 'post_tonemap' in di:
            save_srgb('13_post_tonemap', di['post_tonemap'])
        save_srgb('14_display_finish', relit_display)
        if self.look_safe and hasattr(self, '_compact_direction_field'):
            _df = self._compact_direction_field
            _df_norm = np.clip((_df - float(np.min(_df))) / max(float(np.max(_df) - np.min(_df)), 1e-5), 0.0, 1.0)
            _fallback_linear = srgb_to_linear(np.clip(relit_display, 0.0, 1.0).astype(np.float32))
            save_gray_exact('15_direction_field.png', _df_norm)
            save_srgb_exact('16_lit_shadow_rim_masks.png', getattr(self, '_compact_masks_rgb', np.zeros((*_df.shape, 3), dtype=np.float32)))
            save_rgb_exact('17_before_compact_policy.png', getattr(self, '_compact_before', _fallback_linear))
            save_rgb_exact('18_after_compact_policy.png', getattr(self, '_compact_after', _fallback_linear))
            save_gray_exact('19_compact_policy_delta.png', getattr(self, '_compact_delta', np.zeros_like(_df)))
        if self.look_safe and self._atmosphere_descriptor and self._atmosphere_budget:
            bd = background_descriptor_debug_view(self._atmosphere_descriptor)
            background_summary = {
                'global_luma': bd.get('global_luma', 0.0),
                'contrast': bd.get('dynamic_range', 0.0),
                'saturation': bd.get('average_saturation', 0.0),
                'warm_presence': bd.get('warm_presence', 0.0),
                'cool_presence': bd.get('cool_presence', 0.0),
                'palette_diversity': bd.get('palette_diversity', 0.0),
                'horizontal_bias': bd.get('horizontal_bias', 0.0),
                'vertical_bias': bd.get('vertical_bias', 0.0),
                'local_light_confidence': bd.get('local_light_confidence', 0.0),
                'flatness_score': bd.get('flatness_score', 0.0),
            }
            budget_data = {
                'background_summary': background_summary,
                'compact_policy': getattr(self, '_compact_policy_runtime', {}),
                'runtime_check': getattr(self, '_compact_runtime_check', {}),
            }
            with open(os.path.join(debug_dir, 'look_policy.json'), 'w', encoding='utf-8') as jf:
                json.dump(asdict(self._look_policy) if self._look_policy is not None else {}, jf, indent=2, ensure_ascii=False)
            with open(os.path.join(debug_dir, 'look_safe_budget.json'), 'w', encoding='utf-8') as jf:
                json.dump(budget_data, jf, indent=2, ensure_ascii=False)
        self._debug_intermediates = {}
