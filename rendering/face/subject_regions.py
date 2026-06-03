from __future__ import annotations

from typing import Dict
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

class RendererSubjectRegionsMixin:
    def _estimate_portrait_light_recipe(self, lighting_info: LightingInfo) -> Dict[str, object]:
        """Infer a portrait-lighting recipe from the background statistics.

        The key is not only hue, but also whether the background behaves like a
        wrapped ambience, a directional warm key, a dual-color neon setup, or a
        neutral studio source.
        """
        field = getattr(lighting_info, 'gradient_field', {}) or {}
        palette = getattr(lighting_info, 'palette_points', []) or []
        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower()
        colorfulness = float(np.clip(field.get('colorfulness', getattr(lighting_info, 'palette_diversity', 0.25)), 0.0, 1.0))
        local_contrast = float(np.clip(field.get('local_contrast', 0.35), 0.0, 2.0))
        p50 = float(field.get('p50_luma', 0.25))
        p95 = float(field.get('p95_luma', 0.55))

        left_color = np.array(field.get('left_color', getattr(lighting_info, 'ambient_color', (1.0, 1.0, 1.0))), dtype=np.float32)
        right_color = np.array(field.get('right_color', getattr(lighting_info, 'ambient_color', (1.0, 1.0, 1.0))), dtype=np.float32)
        top_color = np.array(field.get('top_color', getattr(lighting_info, 'ambient_color', (1.0, 1.0, 1.0))), dtype=np.float32)
        bottom_color = np.array(field.get('bottom_color', getattr(lighting_info, 'ambient_color', (1.0, 1.0, 1.0))), dtype=np.float32)
        center_color = np.array(field.get('center_color', getattr(lighting_info, 'ambient_color', (1.0, 1.0, 1.0))), dtype=np.float32)

        warm_sum = 0.0
        cool_sum = 0.0
        warm_cols = []
        cool_cols = []
        for p in palette:
            try:
                c = np.array(p.get('color', p.get('rgb', (1.0, 1.0, 1.0))), dtype=np.float32)
                s = float(p.get('score', p.get('power', 1.0))) * (0.30 + 0.70 * float(p.get('sat', 0.5)))
            except Exception:
                continue
            role = self._classify_light_hue(c)
            if role == 'warm':
                warm_sum += s; warm_cols.append(c * max(s, 1e-4))
            elif role == 'cool':
                cool_sum += s; cool_cols.append(c * max(s, 1e-4))

        key_color = np.array(getattr(lighting_info, 'key_color', getattr(lighting_info, 'ambient_color', (1.0, 1.0, 1.0))), dtype=np.float32)
        amb_color = np.array(getattr(lighting_info, 'ambient_color', key_color), dtype=np.float32)
        warm_color = (np.sum(np.stack(warm_cols, axis=0), axis=0) / max(warm_sum, 1e-6)) if warm_cols else np.maximum(key_color, center_color)
        cool_color = (np.sum(np.stack(cool_cols, axis=0), axis=0) / max(cool_sum, 1e-6)) if cool_cols else np.maximum(amb_color, center_color)
        warm_color = brighten_preserve_hue(np.clip(warm_color, 0.0, 4.0), 0.52)
        cool_color = brighten_preserve_hue(np.clip(cool_color, 0.0, 4.0), 0.40)
        neutral_color = np.array([1.0, 0.965, 0.92], dtype=np.float32)

        total = max(warm_sum + cool_sum, 1e-6)
        warm_ratio = warm_sum / total
        cool_ratio = cool_sum / total
        two_tone = min(warm_sum, cool_sum) / max(max(warm_sum, cool_sum), 1e-6)

        def hue_vec(c):
            c = np.array(c, dtype=np.float32)
            l = max(float(np.dot(c, LUMA)), 1e-5)
            return np.clip(c / l, 0.0, 4.0)
        left_h = hue_vec(left_color); right_h = hue_vec(right_color)
        top_h = hue_vec(top_color); bottom_h = hue_vec(bottom_color)
        lr_hue_gap = float(np.mean(np.abs(left_h - right_h)))
        tb_hue_gap = float(np.mean(np.abs(top_h - bottom_h)))
        lr_luma_gap = abs(float(np.dot(left_color, LUMA)) - float(np.dot(right_color, LUMA)))
        tb_luma_gap = abs(float(np.dot(top_color, LUMA)) - float(np.dot(bottom_color, LUMA)))
        left_warm = self._classify_light_hue(left_color) == 'warm'
        right_warm = self._classify_light_hue(right_color) == 'warm'
        left_cool = self._classify_light_hue(left_color) == 'cool'
        right_cool = self._classify_light_hue(right_color) == 'cool'

        # Robust recipe selection.
        # In continuous look-safe mode the recipe name is no longer a style router.
        # The actual color/direction values below are still extracted from the background,
        # but the downstream modules must consume the unified atmosphere budget.
        if self._using_continuous_policy():
            recipe = 'balanced_soft'
        elif neon != 'off' or (bg_mode == 'rich' and colorfulness > 0.28 and ((left_warm and right_cool) or (left_cool and right_warm) or two_tone > 0.15 or lr_hue_gap > 0.30)):
            recipe = 'night_mixed'
        elif cool_ratio > 0.56 and p50 < 0.24 and colorfulness > 0.18 and tb_hue_gap < 0.24:
            recipe = 'cool_env'
        elif warm_ratio > 0.50 and (lr_luma_gap > 0.03 or tb_luma_gap > 0.04 or p95 > 0.34 or colorfulness > 0.16):
            recipe = 'warm_side'
        elif colorfulness < 0.16 and local_contrast < 0.42:
            recipe = 'neutral_soft'
        else:
            recipe = 'balanced_soft'

        # Choose side from peak direction or warm-dominant side.
        side_sign = 1.0
        if (left_warm and not right_warm) or (left_cool and not right_cool):
            side_sign = -1.0
        elif (right_warm and not left_warm) or (right_cool and not left_cool):
            side_sign = 1.0
        else:
            try:
                if getattr(lighting_info, 'lights', None):
                    side_sign = float(np.array(lighting_info.lights[0]['direction'], dtype=np.float32)[0])
                    if abs(side_sign) < 0.08:
                        side_sign = 1.0
            except Exception:
                side_sign = 1.0

        return {
            'recipe': recipe,
            'warm_color': warm_color,
            'cool_color': cool_color,
            'neutral_color': neutral_color,
            'warm_ratio': float(warm_ratio),
            'cool_ratio': float(cool_ratio),
            'two_tone': float(two_tone),
            'colorfulness': float(colorfulness),
            'local_contrast': float(local_contrast),
            'p50': float(p50),
            'p95': float(p95),
            'lr_hue_gap': float(lr_hue_gap),
            'side_sign': float(1.0 if side_sign >= 0.0 else -1.0),
        }


    def _estimate_skin_proxy(self, base_subject: np.ndarray, subject_mask: np.ndarray, face_core: np.ndarray, hair_region: np.ndarray, edge_band: np.ndarray) -> np.ndarray:
        rgb = np.clip(base_subject, 0.0, 4.0).astype(np.float32)
        luma = rgb_luminance(rgb)
        chroma = rgb / np.maximum(np.sum(rgb, axis=-1, keepdims=True), 1e-5)
        r, g, b = chroma[..., 0], chroma[..., 1], chroma[..., 2]
        # Soft skin-likelihood that works under warm/cool grading.  It mainly
        # rejects dark clothing and highly saturated blue/green regions.
        rg = np.clip((r - g * 0.82) / 0.18, 0.0, 1.0)
        gb = np.clip((g - b * 0.72) / 0.18, 0.0, 1.0)
        not_dark = np.clip((luma - 0.075) / 0.22, 0.0, 1.0)
        not_cloth = np.clip((luma - 0.11) / 0.20, 0.0, 1.0)
        color_gate = np.clip(0.55 * rg + 0.45 * gb, 0.0, 1.0) * not_dark
        skin = subject_mask * np.clip(0.72 * color_gate + 0.36 * face_core * not_cloth, 0.0, 1.0)
        skin *= np.clip(1.0 - 0.40 * edge_band - 0.16 * hair_region, 0.0, 1.0)
        skin = box_blur_gray(skin.astype(np.float32), passes=1)
        return np.clip(skin, 0.0, 1.0).astype(np.float32)


    def _estimate_clothing_mask(self, base_subject: np.ndarray, subject_mask: np.ndarray, face_core: np.ndarray, hair_region: np.ndarray, edge_band: np.ndarray) -> np.ndarray:
        skin = self._estimate_skin_proxy(base_subject, subject_mask, face_core, hair_region, edge_band)
        clothing = np.clip(subject_mask - skin - hair_region * 0.80 - edge_band * 0.50 - face_core * 0.90, 0.0, 1.0).astype(np.float32)
        clothing = box_blur_gray(clothing, passes=1)
        return np.clip(clothing, 0.0, 1.0).astype(np.float32)
