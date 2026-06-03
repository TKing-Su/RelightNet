from __future__ import annotations

from dataclasses import asdict
from typing import Tuple
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

class RendererPortraitRecipeMixin:
    def _compute_portrait_recipe_finish(
        self,
        relit: np.ndarray,
        base_subject: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        N: np.ndarray,
        V: np.ndarray,
        P: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        specular_map: np.ndarray,
        roughness_map: np.ndarray,
        lighting_info: LightingInfo,
    ) -> np.ndarray:
        if not np.any(subject_mask > 0.08):
            return np.zeros_like(relit, dtype=np.float32)
        h, w = subject_mask.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ys, xs = np.where(subject_mask > 0.08)
        y0, y1 = float(ys.min()), float(ys.max())
        x0, x1 = float(xs.min()), float(xs.max())
        x_rel = np.clip((xx - x0) / max(x1 - x0, 1.0), 0.0, 1.0).astype(np.float32)
        y_rel = np.clip((yy - y0) / max(y1 - y0, 1.0), 0.0, 1.0).astype(np.float32)

        recipe = self._estimate_portrait_light_recipe(lighting_info)
        name = str(recipe['recipe'])
        warm = np.asarray(recipe['warm_color'], dtype=np.float32)
        cool = np.asarray(recipe['cool_color'], dtype=np.float32)
        neutral = np.asarray(recipe['neutral_color'], dtype=np.float32)
        colorfulness = float(recipe['colorfulness'])
        local_contrast = float(recipe['local_contrast'])
        side_sign = float(recipe['side_sign'])
        p50 = float(recipe.get('p50', 0.22))

        def dir_color(c, lo=0.72, hi=1.42):
            l = max(float(np.dot(c, LUMA)), 1e-5)
            return np.clip(0.76 + 0.24 * np.clip(np.array(c, dtype=np.float32) / l, 0.56, 1.90), lo, hi)
        warm_dir = dir_color(warm)
        cool_dir = dir_color(cool)
        neutral_dir = np.clip(neutral / max(float(np.dot(neutral, LUMA)), 1e-5), 0.86, 1.16)

        skin = self._estimate_skin_proxy(base_subject, subject_mask, face_core, hair_region, edge_band)
        facing = np.clip(np.sum(N * V, axis=-1), 0.0, 1.0).astype(np.float32)
        gloss = np.clip(1.0 - roughness_map, 0.0, 1.0).astype(np.float32)
        material = np.clip(0.18 + 0.42 * specular_map, 0.0, 0.58) * np.power(gloss, 0.50)
        mask_gate = np.power(np.clip(subject_mask, 0.0, 1.0), 0.86).astype(np.float32)
        side_gate = self._compute_signed_side_mask(P, side_sign, subject_mask, power=0.80)
        opp_side = self._compute_signed_side_mask(P, -side_sign, subject_mask, power=0.80)
        center_gate = np.exp(-0.5 * (((x_rel - 0.50) / 0.32) ** 2)).astype(np.float32)
        lower_fill = np.exp(-0.5 * (((y_rel - 0.71) / 0.20) ** 2)).astype(np.float32)
        blur_skin = self._gaussianish_blur_rgb(base_subject, radius=3)

        relit_luma = rgb_luminance(np.clip(relit, 0.0, 4.0))
        shadow_gate = np.clip((0.52 - relit_luma) / 0.52, 0.0, 1.0) * skin
        mid_gate = np.clip(1.0 - np.abs(relit_luma - 0.40) / 0.40, 0.0, 1.0) * skin
        translucency_gate = np.clip(0.58 * shadow_gate + 0.42 * mid_gate, 0.0, 1.0)

        # curved facial masks
        nose_bridge = np.exp(-0.5 * (((x_rel - 0.50) / 0.070) ** 2 + ((y_rel - 0.40) / 0.22) ** 2)).astype(np.float32)
        cheek_l = np.exp(-0.5 * (((x_rel - 0.36) / 0.14) ** 2 + ((y_rel - 0.45) / 0.18) ** 2)).astype(np.float32)
        cheek_r = np.exp(-0.5 * (((x_rel - 0.64) / 0.14) ** 2 + ((y_rel - 0.45) / 0.18) ** 2)).astype(np.float32)
        chin = np.exp(-0.5 * (((x_rel - 0.50) / 0.25) ** 2 + ((y_rel - 0.61) / 0.20) ** 2)).astype(np.float32)
        beauty_region = np.clip((0.40 * nose_bridge + 0.24 * cheek_l + 0.24 * cheek_r + 0.12 * chin) * skin * mask_gate, 0.0, 1.0)

        def light_terms(Lvec: np.ndarray, wrap: float = 0.18, expn: float = 0.95):
            Lf = np.ones_like(N) * safe_norm(Lvec.reshape(1, 1, 3))
            H = safe_norm(Lf + V)
            ndotl_raw = np.sum(N * Lf, axis=-1).astype(np.float32)
            ndotl_wrap = np.clip((ndotl_raw + wrap) / (1.0 + wrap), 0.0, 1.0).astype(np.float32)
            ndoth = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0).astype(np.float32)
            gloss_exp = 9.0 + 38.0 * np.power(gloss, 0.74)
            spec = np.power(ndoth, gloss_exp).astype(np.float32)
            return np.power(ndotl_wrap, expn).astype(np.float32), spec

        L_front = np.array([0.0, -0.10, 1.0], dtype=np.float32)
        L_side = np.array([0.48 * side_sign, -0.18, 1.0], dtype=np.float32)
        L_fill = np.array([-0.24 * side_sign, 0.02, 1.0], dtype=np.float32)
        L_top = np.array([0.0, -0.28, 1.0], dtype=np.float32)
        L_low = np.array([0.0, 0.16, 1.0], dtype=np.float32)
        diff_front, spec_front = light_terms(L_front, wrap=0.28, expn=0.88)
        diff_side, spec_side = light_terms(L_side, wrap=0.16, expn=0.96)
        diff_fill, spec_fill = light_terms(L_fill, wrap=0.36, expn=0.86)
        diff_top, spec_top = light_terms(L_top, wrap=0.20, expn=0.92)
        diff_low, spec_low = light_terms(L_low, wrap=0.34, expn=0.86)

        acc = np.zeros_like(relit, dtype=np.float32)
        hair_edge = np.clip(0.42 * hair_region + 0.58 * edge_band, 0.0, 1.0) * mask_gate

        if name == 'cool_env':
            cool_mix = np.clip(0.56 * neutral_dir + 0.44 * cool_dir, 0.72, 1.26)
            cool_wrap = skin * np.clip(0.46 * diff_front + 0.28 * diff_low + 0.20 * center_gate + 0.10 * lower_fill, 0.0, 1.0)
            acc += blur_skin * cool_mix.reshape(1,1,3) * translucency_gate[...,None] * 0.070
            acc += cool_mix.reshape(1,1,3) * cool_wrap[...,None] * 0.055
            cool_beauty = beauty_region * material * np.clip(0.44 * spec_front + 0.16 * spec_low, 0.0, 1.0)
            acc += cool_dir.reshape(1,1,3) * cool_beauty[...,None] * 0.026
            acc += cool_dir.reshape(1,1,3) * hair_edge[...,None] * 0.020

        elif name == 'warm_side':
            warm_mix = np.clip(0.54 * neutral_dir + 0.46 * warm_dir, 0.74, 1.34)
            key_region = skin * np.clip(0.64 * diff_side + 0.18 * diff_top + 0.14 * side_gate, 0.0, 1.0)
            bounce_region = skin * np.clip(0.44 * diff_fill + 0.26 * lower_fill + 0.12 * opp_side, 0.0, 1.0)
            acc += blur_skin * warm_mix.reshape(1,1,3) * key_region[...,None] * 0.080
            acc += blur_skin * neutral_dir.reshape(1,1,3) * bounce_region[...,None] * 0.034
            warm_beauty = beauty_region * material * np.clip(0.72 * spec_side + 0.24 * spec_top + 0.10 * diff_side, 0.0, 1.0)
            acc += warm_dir.reshape(1,1,3) * warm_beauty[...,None] * 0.072
            gold_sep = np.clip(0.42 * side_gate + 0.40 * hair_edge + 0.08 * face_core, 0.0, 1.0) * mask_gate
            acc += warm_dir.reshape(1,1,3) * gold_sep[...,None] * material[...,None] * 0.038

        elif name == 'neutral_soft':
            soft_region = skin * np.clip(0.62 * diff_front + 0.22 * diff_low + 0.12 * center_gate, 0.0, 1.0)
            under_fill = skin * np.clip(0.34 * lower_fill + 0.26 * center_gate, 0.0, 1.0)
            acc += blur_skin * neutral_dir.reshape(1,1,3) * soft_region[...,None] * 0.068
            acc += blur_skin * neutral_dir.reshape(1,1,3) * under_fill[...,None] * 0.028
            neutral_beauty = beauty_region * material * np.clip(0.56 * spec_front + 0.20 * spec_low, 0.0, 1.0)
            acc += neutral_dir.reshape(1,1,3) * neutral_beauty[...,None] * 0.038
            acc += neutral_dir.reshape(1,1,3) * hair_edge[...,None] * 0.010

        elif name == 'night_mixed':
            frontal_mix = np.clip(0.78 * neutral_dir + 0.14 * warm_dir + 0.08 * cool_dir, 0.74, 1.28)
            frontal = skin * np.clip(0.54 * diff_front + 0.18 * center_gate + 0.10 * diff_low, 0.0, 1.0)
            acc += blur_skin * frontal_mix.reshape(1,1,3) * frontal[...,None] * 0.056
            front_beauty = beauty_region * material * np.clip(0.62 * spec_front + 0.14 * spec_top, 0.0, 1.0)
            acc += neutral_dir.reshape(1,1,3) * front_beauty[...,None] * 0.042
            warm_rim = np.clip(0.48 * side_gate + 0.44 * hair_edge, 0.0, 1.0) * mask_gate
            cool_rim = np.clip(0.48 * opp_side + 0.44 * hair_edge, 0.0, 1.0) * mask_gate
            acc += warm_dir.reshape(1,1,3) * warm_rim[...,None] * material[...,None] * (0.040 + 0.014 * colorfulness)
            acc += cool_dir.reshape(1,1,3) * cool_rim[...,None] * material[...,None] * (0.040 + 0.014 * colorfulness)
            cheek_dual = skin * np.clip(0.18 * side_gate + 0.18 * opp_side, 0.0, 0.58)
            acc += (0.55 * warm_dir.reshape(1,1,3) + 0.45 * cool_dir.reshape(1,1,3)) * cheek_dual[...,None] * 0.018

        else:  # balanced_soft
            bal_mix = np.clip(0.76 * neutral_dir + 0.16 * warm_dir + 0.08 * cool_dir, 0.74, 1.26)
            soft_key = skin * np.clip(0.56 * diff_front + 0.14 * diff_side + 0.14 * center_gate, 0.0, 1.0)
            acc += blur_skin * bal_mix.reshape(1,1,3) * soft_key[...,None] * 0.060
            bal_beauty = beauty_region * material * np.clip(0.54 * spec_front + 0.14 * spec_side, 0.0, 1.0)
            acc += bal_mix.reshape(1,1,3) * bal_beauty[...,None] * 0.038
            sep = np.clip(0.18 * side_gate + 0.28 * hair_edge, 0.0, 0.78) * mask_gate
            acc += warm_dir.reshape(1,1,3) * sep[...,None] * 0.016

        return np.clip(acc, 0.0, 0.64).astype(np.float32)


    def _estimate_palette_temperature(self, lighting_info: LightingInfo) -> Tuple[float, float]:
        warm = 0.0
        cool = 0.0
        total = 0.0
        for item in lighting_info.palette_points:
            color = np.array(item.get('color', [0.0, 0.0, 0.0]), dtype=np.float32)
            score = float(item.get('score', 1.0))
            hue, sat, _ = rgb_to_hsv_approx(color)
            total += score
            if self.extractor._is_warm_hue(hue, sat):
                warm += score
            elif self.extractor._is_cool_hue(hue, sat):
                cool += score
        if total <= 1e-6:
            return 0.0, 0.0
        return warm / total, cool / total


    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return float(a * (1.0 - t) + b * t)


    def _blend_param_to_default(self, current: float, default_value: float, gate: float) -> float:
        return self._lerp(default_value, current, gate)


    def _adapt_preset_to_lighting_info(self, lighting_info: LightingInfo) -> None:
        """Background-aware moderation so the quality modules stay useful without over-stylizing."""
        if self._using_continuous_policy():
            # Continuous look-safe mode has a single policy outlet: the atmosphere budget.
            # Do not re-moderate the preset based on legacy background_mode/neon_strength.
            self._apply_preset(RelightPreset(**asdict(self.base_preset)))
            return
        preset = RelightPreset(**asdict(self.base_preset))
        default = QUALITY_PROFILE
        respect = float(np.clip(getattr(preset, 'background_respect', 0.90), 0.0, 1.0))
        mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
        neon_strength = str(getattr(lighting_info, 'neon_strength', 'off')).lower()
        palette_diversity = float(np.clip(getattr(lighting_info, 'palette_diversity', 0.0), 0.0, 1.0))
        warm_ratio, cool_ratio = self._estimate_palette_temperature(lighting_info)

        if self.style_mode == 'neon':
            gate = 1.0 if neon_strength == 'strong' else (0.72 if neon_strength == 'soft' else 0.40)
            gate = self._lerp(1.0 - 0.65 * respect, 1.0, gate)
            for name in ('multi_ambient_strength', 'rim_strength', 'edge_local_spill_strength', 'post_saturation', 'neon_dual_tint_strength', 'neon_side_separation', 'post_bloom_strength'):
                setattr(preset, name, self._blend_param_to_default(getattr(preset, name), getattr(default, name), gate))
        elif self.style_mode == 'cinematic':
            darkness_gate = 0.85 if mode in ('balanced', 'monotone') else 0.70
            gate = self._lerp(1.0 - 0.55 * respect, 1.0, darkness_gate)
            for name in ('shadow_sculpt_strength', 'key_shadow_strength', 'rim_strength', 'post_contrast', 'post_vignette_strength', 'post_local_contrast_strength'):
                setattr(preset, name, self._blend_param_to_default(getattr(preset, name), getattr(default, name), gate))
        else:  # quality
            if mode == 'monotone' or palette_diversity < 0.22:
                preset.post_saturation *= self._lerp(1.0, 0.94, respect)
                preset.multi_ambient_strength *= self._lerp(1.0, 0.88, respect)
            if cool_ratio > warm_ratio * 1.4:
                preset.split_shadow_cool = max(preset.split_shadow_cool, 0.03)
            if warm_ratio > cool_ratio * 1.4:
                preset.split_highlight_warm = max(preset.split_highlight_warm, 0.04)

        if mode == 'monotone' or palette_diversity < 0.20:
            damp = self._lerp(1.0, 0.86, respect)
            preset.edge_local_spill_strength *= damp
            preset.post_bloom_strength *= damp
            preset.post_haze_strength *= damp
            preset.neon_dual_tint_strength *= damp

        # Stage38 safety rails.  Keep readability, but do not force every dark
        # background into a bright neutral portrait.  Colored light needs room.
        if self.style_mode == 'neon':
            preset.target_subject_p70 = max(float(preset.target_subject_p70), 0.365)
            preset.post_exposure = max(float(preset.post_exposure), 1.055)
            preset.ambient_strength = max(float(preset.ambient_strength), 0.102)
            preset.fill_strength = max(float(preset.fill_strength), 0.058)
            preset.multi_ambient_strength = max(float(preset.multi_ambient_strength), 0.46)
            preset.post_saturation = min(max(float(preset.post_saturation), 1.10), 1.16)
            preset.neon_dual_tint_strength = min(max(float(preset.neon_dual_tint_strength), 0.34), 0.44)
            preset.edge_local_spill_strength = min(max(float(preset.edge_local_spill_strength), 0.016), 0.020)
            preset.edge_mix_strength = min(float(preset.edge_mix_strength), 0.010)
            preset.edge_cleanup_strength = min(float(preset.edge_cleanup_strength), 0.085)
            preset.skin_protect_strength = max(float(preset.skin_protect_strength), 0.085)
        else:
            preset.target_subject_p70 = max(float(preset.target_subject_p70), 0.382)
            preset.post_exposure = max(float(preset.post_exposure), 1.085)
            preset.ambient_strength = max(float(preset.ambient_strength), 0.086)
            preset.fill_strength = max(float(preset.fill_strength), 0.048)
            preset.multi_ambient_strength = max(float(preset.multi_ambient_strength), 0.24)
            preset.skin_protect_strength = min(float(preset.skin_protect_strength), 0.08)
            preset.detail_strength = min(float(preset.detail_strength), 0.021)
            preset.detail_limit = min(float(preset.detail_limit), 0.021)
        preset.shadow_sculpt_strength = min(float(preset.shadow_sculpt_strength), 0.18)
        preset.key_shadow_strength = min(float(preset.key_shadow_strength), 0.12)
        preset.post_local_contrast_strength = min(float(preset.post_local_contrast_strength), 0.028)

        self._apply_preset(preset)
