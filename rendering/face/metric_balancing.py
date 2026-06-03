from __future__ import annotations

from typing import Optional
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

class RendererMetricBalancingMixin:
    def _apply_metric_guided_cyber_balance(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo] = None,
    ) -> np.ndarray:
        """Metric-guided correction for cyber/neon backgrounds.

        Triggered only for rich/neon scenes. It fixes the pattern shown by the
        general evaluator: low exposure, excessive face/body unsupported cast,
        zero atmosphere alignment, while keeping cyber edge/rim mood.
        """
        if lighting_info is None or relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower()
        style = normalize_style_mode(getattr(self, 'style_mode', 'quality'), fallback='quality', look_safe=self.look_safe)
        if not (bg_mode == 'rich' or neon != 'off' or style == 'neon'):
            return relit

        out = np.clip(relit.astype(np.float32, copy=True), 0.0, 8.0)
        src = np.clip(source_linear.astype(np.float32), 0.0, 8.0)
        subj = np.clip(subject_mask.astype(np.float32), 0.0, 1.0)
        face = np.clip(face_core.astype(np.float32) * subj * (1.0 - 0.35 * np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 1.0)
        body = np.clip(subj * (1.0 - 0.90 * face_core) * (1.0 - 0.38 * hair_region) * (1.0 - 0.45 * edge_band), 0.0, 1.0)
        if not np.any(face > 0.05) and not np.any(body > 0.05):
            return out

        h, w = subj.shape[:2]
        yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h, dtype=np.float32), np.linspace(-1.0, 1.0, w, dtype=np.float32), indexing='ij')
        gf = getattr(lighting_info, 'gradient_field', {}) if hasattr(lighting_info, 'gradient_field') else {}
        if not isinstance(gf, dict):
            gf = {}
        bg_p50 = float(gf.get('p50_luma', 0.08))
        bg_p95 = float(gf.get('p95_luma', 0.38))
        hb = float(gf.get('horizontal_bias', 0.0))
        side_sign = 1.0 if hb >= 0.0 else -1.0
        if abs(hb) < 0.020:
            left_c = np.array(gf.get('left_color', getattr(lighting_info, 'ambient_color', (0.4, 0.4, 0.7))), dtype=np.float32)
            right_c = np.array(gf.get('right_color', getattr(lighting_info, 'ambient_color', (0.4, 0.4, 0.7))), dtype=np.float32)
            side_sign = 1.0 if float(np.dot(right_c - left_c, LUMA)) >= 0.0 else -1.0
        lit_side = np.clip(0.5 + 0.5 * side_sign * xx, 0.0, 1.0).astype(np.float32)
        shadow_side = np.clip(1.0 - lit_side, 0.0, 1.0).astype(np.float32)
        upper_body = np.clip(body * np.clip((yy + 0.05) / 0.88, 0.0, 1.0) * np.clip((1.00 - yy) / 1.12, 0.0, 1.0), 0.0, 1.0)
        lit_face = np.clip(face * np.power(lit_side, 1.30), 0.0, 1.0)
        lit_body = np.clip(upper_body * np.power(lit_side, 1.15), 0.0, 1.0)
        shadow_face = np.clip(face * np.power(shadow_side, 1.15), 0.0, 1.0)
        shadow_body = np.clip(upper_body * np.power(shadow_side, 1.08), 0.0, 1.0)

        src_luma = np.maximum(rgb_luminance(src), 1e-5).astype(np.float32)
        out_luma = np.maximum(rgb_luminance(out), 1e-5).astype(np.float32)
        src_dir = np.clip(src / src_luma[..., None], 0.58, 1.78).astype(np.float32)
        out_dir = np.clip(out / out_luma[..., None], 0.30, 3.60).astype(np.float32)
        dir_delta = np.linalg.norm(out_dir - src_dir, axis=-1).astype(np.float32)

        # 1) Face/core: keep cyber illumination luminance but pull unsupported hue
        # back toward the source skin direction. This directly targets face_cast=0.
        face_target_luma = float(np.clip(0.355 + 0.090 * (bg_p95 - bg_p50), 0.335, 0.435))
        face_mask_bool = face > 0.12
        if np.count_nonzero(face_mask_bool) > 64:
            cur_face = float(np.percentile(out_luma[face_mask_bool], 70.0))
            lift_gap = float(np.clip(face_target_luma - cur_face, 0.0, 0.18))
        else:
            lift_gap = 0.0
        target_luma = out_luma + lift_gap * np.clip(0.42 * face + 0.25 * lit_face, 0.0, 1.0)
        natural_face = np.clip(src_dir * target_luma[..., None], 0.0, 8.0).astype(np.float32)
        face_restore = np.clip(face * (0.36 + 0.58 * np.clip(dir_delta / 0.46, 0.0, 1.0)), 0.0, 0.78).astype(np.float32)
        # Stronger in the center, weaker near the lit side so a small neon cue remains.
        face_restore *= np.clip(1.0 - 0.16 * lit_side - 0.50 * edge_band, 0.22, 1.0)
        out = out * (1.0 - face_restore[..., None]) + natural_face * face_restore[..., None]

        # Re-add safe luma-only facial detail to recover detail without RGB color noise.
        out_luma = np.maximum(rgb_luminance(out), 1e-5).astype(np.float32)
        src_hi = np.clip(src_luma - box_blur_gray(src_luma, passes=2), -0.030, 0.030).astype(np.float32)
        detail_gate = np.clip(face * 0.38 + lit_face * 0.20, 0.0, 0.46).astype(np.float32)
        out *= np.clip((out_luma + src_hi * detail_gate) / out_luma, 0.92, 1.10)[..., None]

        # 2) Body/shoulder: lift and partially de-cast skin, but leave rim/edge neon.
        out_luma = np.maximum(rgb_luminance(out), 1e-5).astype(np.float32)
        body_target_luma = float(np.clip(0.335 + 0.075 * (bg_p95 - bg_p50), 0.315, 0.405))
        body_mask_bool = upper_body > 0.12
        if np.count_nonzero(body_mask_bool) > 64:
            cur_body = float(np.percentile(out_luma[body_mask_bool], 70.0))
            body_gap = float(np.clip(body_target_luma - cur_body, 0.0, 0.16))
        else:
            body_gap = 0.0
        target_body_luma = out_luma + body_gap * np.clip(0.55 * upper_body + 0.32 * lit_body, 0.0, 1.0)
        natural_body = np.clip(src_dir * target_body_luma[..., None], 0.0, 8.0).astype(np.float32)
        out_dir = np.clip(out / out_luma[..., None], 0.30, 3.60).astype(np.float32)
        body_delta = np.linalg.norm(out_dir - src_dir, axis=-1).astype(np.float32)
        body_restore = np.clip(body * (0.20 + 0.46 * np.clip(body_delta / 0.46, 0.0, 1.0)) * (1.0 - 0.62 * edge_band), 0.0, 0.58).astype(np.float32)
        out = out * (1.0 - body_restore[..., None]) + natural_body * body_restore[..., None]

        # 3) Cyber atmosphere belongs on hair/rim/edge/shadow side, not face core.
        ambient = np.array(getattr(lighting_info, 'ambient_color', (0.25, 0.45, 1.0)), dtype=np.float32)
        global_bg = np.array(getattr(lighting_info, 'global_mean_color', ambient), dtype=np.float32)
        cool = np.array([0.42, 0.70, 1.25], dtype=np.float32)
        mag = np.array([1.10, 0.35, 1.15], dtype=np.float32)
        bg_mix = np.clip(0.35 * ambient + 0.35 * global_bg + 0.30 * cool, 1e-5, 8.0)
        bg_mix = bg_mix / max(float(np.dot(bg_mix, LUMA)), 1e-5)
        mag = mag / max(float(np.dot(mag, LUMA)), 1e-5)
        rim_gate = np.clip(0.62 * edge_band + 0.48 * hair_region + 0.32 * upper_body * lit_side - 0.42 * face_core, 0.0, 0.78).astype(np.float32)
        opp_gate = np.clip(0.48 * edge_band * shadow_side + 0.32 * hair_region * shadow_side + 0.12 * upper_body * shadow_side - 0.34 * face_core, 0.0, 0.58).astype(np.float32)
        out += bg_mix.reshape(1, 1, 3) * rim_gate[..., None] * 0.092
        out += mag.reshape(1, 1, 3) * opp_gate[..., None] * 0.052

        # 4) Preserve side separation after de-cast.
        out_luma = np.maximum(rgb_luminance(out), 1e-5).astype(np.float32)
        lit_region = (lit_face + 0.40 * lit_body) > 0.10
        sh_region = (shadow_face + 0.35 * shadow_body) > 0.10
        if np.count_nonzero(lit_region) > 48 and np.count_nonzero(sh_region) > 48:
            lit_p = float(np.percentile(out_luma[lit_region], 62.0))
            sh_p = float(np.percentile(out_luma[sh_region], 62.0))
            need = float(np.clip(0.046 - (lit_p - sh_p), 0.0, 0.070))
            if need > 1e-5:
                out += bg_mix.reshape(1, 1, 3) * np.clip(lit_face + 0.45 * lit_body, 0.0, 1.0)[..., None] * need
                out *= (1.0 - np.clip((0.35 * shadow_face + 0.22 * shadow_body) * need, 0.0, 0.030)[..., None])

        return np.clip(out, 0.0, 8.0).astype(np.float32)



    def _apply_metric_guided_warm_balance(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo] = None,
    ) -> np.ndarray:
        """Metric-guided correction for warm/red/sunset backgrounds.

        The general evaluator showed: detail/exposure/edge are already good,
        while warm/red suffers from face/body cast and weak directionality.
        This pass therefore avoids global exposure changes; it restores face/body
        chroma toward the source at the same luminance, then adds a luma-dominant
        warm side key and warm rim only on edge/hair.
        """
        if lighting_info is None or relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        try:
            recipe = self._estimate_portrait_light_recipe(lighting_info)
            recipe_name = str(recipe.get('recipe', 'balanced_soft')).lower()
            warm_ratio = float(recipe.get('warm_ratio', 0.0))
            cool_ratio = float(recipe.get('cool_ratio', 0.0))
        except Exception:
            recipe_name = 'balanced_soft'; warm_ratio = 0.0; cool_ratio = 0.0
        # Trigger only for clearly warm scenes, not for cyber's magenta accents.
        style = normalize_style_mode(getattr(self, 'style_mode', 'quality'), fallback='quality', look_safe=self.look_safe)
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower()
        if neon != 'off' or style == 'neon':
            return relit
        if not (recipe_name == 'warm_side' or warm_ratio > max(0.38, cool_ratio * 1.30)):
            return relit

        out = np.clip(relit.astype(np.float32, copy=True), 0.0, 8.0)
        src = np.clip(source_linear.astype(np.float32), 0.0, 8.0)
        subj = np.clip(subject_mask.astype(np.float32), 0.0, 1.0)
        face = np.clip(face_core.astype(np.float32) * subj * (1.0 - 0.32 * np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 1.0)
        body = np.clip(subj * (1.0 - 0.88 * face_core) * (1.0 - 0.30 * hair_region) * (1.0 - 0.48 * edge_band), 0.0, 1.0)
        if not np.any(face > 0.05) and not np.any(body > 0.05):
            return out

        h, w = subj.shape[:2]
        yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h, dtype=np.float32), np.linspace(-1.0, 1.0, w, dtype=np.float32), indexing='ij')
        gf = getattr(lighting_info, 'gradient_field', {}) if hasattr(lighting_info, 'gradient_field') else {}
        if not isinstance(gf, dict):
            gf = {}
        hb = float(gf.get('horizontal_bias', 0.0))
        side_sign = 1.0 if hb >= 0.0 else -1.0
        if abs(hb) < 0.025:
            left_c = np.array(gf.get('left_color', getattr(lighting_info, 'ambient_color', (1.0, 0.55, 0.35))), dtype=np.float32)
            right_c = np.array(gf.get('right_color', getattr(lighting_info, 'ambient_color', (1.0, 0.55, 0.35))), dtype=np.float32)
            # Use the warmer/brighter side as the key side.
            left_warmness = float(left_c[0] - 0.35 * left_c[2] + 0.20 * np.dot(left_c, LUMA))
            right_warmness = float(right_c[0] - 0.35 * right_c[2] + 0.20 * np.dot(right_c, LUMA))
            side_sign = 1.0 if right_warmness >= left_warmness else -1.0
        lit_side = np.clip(0.5 + 0.5 * side_sign * xx, 0.0, 1.0).astype(np.float32)
        shadow_side = np.clip(1.0 - lit_side, 0.0, 1.0).astype(np.float32)
        upper_body = np.clip(body * np.clip((yy + 0.05) / 0.86, 0.0, 1.0) * np.clip((1.00 - yy) / 1.12, 0.0, 1.0), 0.0, 1.0)
        lit_face = np.clip(face * np.power(lit_side, 1.28), 0.0, 1.0)
        lit_body = np.clip(upper_body * np.power(lit_side, 1.12), 0.0, 1.0)
        shadow_face = np.clip(face * np.power(shadow_side, 1.10), 0.0, 1.0)
        shadow_body = np.clip(upper_body * np.power(shadow_side, 1.05), 0.0, 1.0)

        src_luma = np.maximum(rgb_luminance(src), 1e-5).astype(np.float32)
        out_luma = np.maximum(rgb_luminance(out), 1e-5).astype(np.float32)
        src_dir = np.clip(src / src_luma[..., None], 0.64, 1.60).astype(np.float32)
        out_dir = np.clip(out / out_luma[..., None], 0.30, 3.60).astype(np.float32)
        dir_delta = np.linalg.norm(out_dir - src_dir, axis=-1).astype(np.float32)

        # 1) De-cast face and body at current luminance. This targets red/pink raw
        # and unsupported cast while keeping the already-good exposure.
        natural_same_luma = np.clip(src_dir * out_luma[..., None], 0.0, 8.0).astype(np.float32)
        face_restore = np.clip(face * (0.28 + 0.48 * np.clip(dir_delta / 0.42, 0.0, 1.0)) * (1.0 - 0.55 * edge_band), 0.0, 0.62).astype(np.float32)
        body_restore = np.clip(body * (0.18 + 0.40 * np.clip(dir_delta / 0.46, 0.0, 1.0)) * (1.0 - 0.60 * edge_band), 0.0, 0.50).astype(np.float32)
        restore = np.maximum(face_restore, body_restore)
        out = out * (1.0 - restore[..., None]) + natural_same_luma * restore[..., None]

        # 2) Restore directionality as luminance-dominant side light.
        out_luma = np.maximum(rgb_luminance(out), 1e-5).astype(np.float32)
        warm_c = np.array(recipe.get('warm_color', getattr(lighting_info, 'ambient_color', (1.0, 0.56, 0.32))), dtype=np.float32)
        warm_c = np.clip(warm_c, 1e-5, 8.0)
        warm_dir = np.clip(warm_c / max(float(np.dot(warm_c, LUMA)), 1e-5), 0.72, 1.45).astype(np.float32)
        # Face/body key uses mostly source chroma, edge/rim uses warm chroma.
        key_luma = np.clip(0.040 * lit_face + 0.046 * lit_body, 0.0, 0.062).astype(np.float32)
        out += src_dir * key_luma[..., None]
        rim_gate = np.clip(0.52 * edge_band + 0.38 * hair_region + 0.18 * lit_body - 0.36 * face_core, 0.0, 0.58).astype(np.float32)
        out += warm_dir.reshape(1, 1, 3) * rim_gate[..., None] * 0.044
        # Lightly hold the opposite side down to create visible side contrast.
        shadow_gate = np.clip(0.040 * shadow_face + 0.030 * shadow_body, 0.0, 0.052).astype(np.float32)
        out *= (1.0 - shadow_gate[..., None])

        # 3) Ensure a minimum left-right separation without changing global exposure.
        out_luma = np.maximum(rgb_luminance(out), 1e-5).astype(np.float32)
        lit_region = (lit_face + 0.36 * lit_body) > 0.10
        sh_region = (shadow_face + 0.32 * shadow_body) > 0.10
        if np.count_nonzero(lit_region) > 48 and np.count_nonzero(sh_region) > 48:
            lit_p = float(np.percentile(out_luma[lit_region], 62.0))
            sh_p = float(np.percentile(out_luma[sh_region], 62.0))
            need = float(np.clip(0.052 - (lit_p - sh_p), 0.0, 0.065))
            if need > 1e-5:
                out += src_dir * np.clip(lit_face + 0.45 * lit_body, 0.0, 1.0)[..., None] * need
                out *= (1.0 - np.clip(shadow_face + 0.35 * shadow_body, 0.0, 1.0)[..., None] * need * 0.45)

        return np.clip(out, 0.0, 8.0).astype(np.float32)


    def _apply_metric_guided_natural_balance(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo] = None,
    ) -> np.ndarray:
        """Metric-guided correction for misty/natural/balanced backgrounds.

        Triggered when the scene is not neon-rich. It addresses the typical metric
        pattern from misty backgrounds: face high-frequency excess, body too dark,
        and moderate direction score. It keeps face/skin/edge cast stable.
        """
        if lighting_info is None or relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower()
        if bg_mode == 'rich' or neon != 'off' or normalize_style_mode(getattr(self, 'style_mode', 'quality'), fallback='quality', look_safe=self.look_safe) == 'neon':
            return relit

        out = relit.astype(np.float32, copy=True)
        subj = np.clip(subject_mask.astype(np.float32), 0.0, 1.0)
        face = np.clip(face_core.astype(np.float32) * subj * (1.0 - 0.25 * np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 1.0)
        body = np.clip(subj * (1.0 - 0.88 * face_core) * (1.0 - 0.25 * hair_region), 0.0, 1.0)
        if not np.any(face > 0.05) and not np.any(body > 0.05):
            return out

        h, w = subj.shape[:2]
        yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h, dtype=np.float32), np.linspace(-1.0, 1.0, w, dtype=np.float32), indexing='ij')
        gf = getattr(lighting_info, 'gradient_field', {}) if hasattr(lighting_info, 'gradient_field') else {}
        if not isinstance(gf, dict):
            gf = {}
        bg_p50 = float(gf.get('p50_luma', 0.36))
        bg_p95 = float(gf.get('p95_luma', 0.70))
        hb = float(gf.get('horizontal_bias', 0.0))
        side_sign = 1.0 if hb >= 0.0 else -1.0
        if abs(hb) < 0.025:
            left_c = np.array(gf.get('left_color', getattr(lighting_info, 'ambient_color', (1, 1, 1))), dtype=np.float32)
            right_c = np.array(gf.get('right_color', getattr(lighting_info, 'ambient_color', (1, 1, 1))), dtype=np.float32)
            side_sign = 1.0 if float(np.dot(right_c - left_c, LUMA)) >= 0.0 else -1.0
        lit_side = np.clip(0.5 + 0.5 * side_sign * xx, 0.0, 1.0).astype(np.float32)
        shadow_side = np.clip(1.0 - lit_side, 0.0, 1.0).astype(np.float32)
        upper_body = np.clip(body * np.clip((yy + 0.05) / 0.85, 0.0, 1.0) * np.clip((0.98 - yy) / 1.12, 0.0, 1.0), 0.0, 1.0)
        lit_face = np.clip(face * np.power(lit_side, 1.45), 0.0, 1.0)
        lit_body = np.clip(upper_body * np.power(lit_side, 1.20), 0.0, 1.0)
        sh_face = np.clip(face * np.power(shadow_side, 1.25), 0.0, 1.0)
        sh_body = np.clip(upper_body * np.power(shadow_side, 1.15), 0.0, 1.0)

        # 1) Reduce face-only high-frequency excess while preserving low-frequency light and hue.
        lum = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5).astype(np.float32)
        low = box_blur_gray(lum, passes=2).astype(np.float32)
        face_detail_gate = np.clip(face * (0.22 + 0.12 * np.clip((bg_p50 - 0.24) / 0.32, 0.0, 1.0)), 0.0, 0.32)
        damp_lum = low + (lum - low) * (1.0 - 0.55 * face_detail_gate)
        out *= np.clip(damp_lum / lum, 0.84, 1.10)[..., None]

        # 2) Lift body/shoulder for bright misty/natural backgrounds.
        lum = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5).astype(np.float32)
        body_mask = upper_body > 0.12
        if np.count_nonzero(body_mask) > 64:
            cur_body = float(np.percentile(lum[body_mask], 70.0))
            target_body = float(np.clip(0.315 + 0.14 * (bg_p50 - 0.28) + 0.035 * (bg_p95 - 0.55), 0.315, 0.405))
            gap = float(np.clip(target_body - cur_body, 0.0, 0.115))
        else:
            gap = 0.0
        ambient = np.array(getattr(lighting_info, 'ambient_color', (1.0, 1.0, 1.0)), dtype=np.float32)
        global_bg = np.array(getattr(lighting_info, 'global_mean_color', ambient), dtype=np.float32)
        light_color = np.clip(0.55 * ambient + 0.45 * global_bg, 1e-5, 8.0)
        light_color = desaturate_color(light_color, 0.66)
        light_color = brighten_preserve_hue(light_color, 0.42)
        light_dir = np.clip(light_color / max(float(np.dot(light_color, LUMA)), 1e-5), 0.75, 1.35).astype(np.float32)
        if gap > 1e-5:
            body_gate = np.clip(0.72 * upper_body + 0.45 * lit_body + 0.22 * edge_band, 0.0, 1.0)
            out += light_dir.reshape(1, 1, 3) * body_gate[..., None] * gap

        # 3) Add a mild directional cue without hurting skin/edge scores.
        dir_strength = float(np.clip(0.018 + 0.045 * (0.45 - abs(hb)), 0.014, 0.040))
        out += light_dir.reshape(1, 1, 3) * (lit_face * 0.45 + lit_body * 0.85 + edge_band * 0.20)[..., None] * dir_strength
        darken_gate = np.clip(0.018 * sh_face + 0.014 * sh_body, 0.0, 0.026)
        out *= (1.0 - darken_gate[..., None])

        # 4) Keep atmosphere on edge/hair/body, not face core.
        atmos_gate = np.clip(0.10 * hair_region + 0.08 * edge_band + 0.06 * upper_body - 0.035 * face_core, 0.0, 0.12)
        out = out * (1.0 - atmos_gate[..., None]) + out * light_dir.reshape(1, 1, 3) * atmos_gate[..., None]
        return np.clip(out, 0.0, 8.0).astype(np.float32)
