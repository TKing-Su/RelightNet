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

class RendererDirectionalFieldMixin:
    def _apply_directional_light_field(
        self,
        relit: np.ndarray,
        base_subject: np.ndarray,
        N: np.ndarray,
        P: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        face_side: np.ndarray,
        body_region: np.ndarray,
        cloth_region: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        shoulder_region: np.ndarray,
        lighting_info: Optional['LightingInfo'],
        look_policy: Optional[LookPolicy],
    ) -> np.ndarray:
        """Unified continuous direction field consumed by the final pixels.

        It combines normal response, reconstructed portrait position, background
        gradient and local-light UV.  The field is deliberately continuous: no
        filename, style string, recipe name or background label can change routing.
        """
        if relit.size == 0 or lighting_info is None or look_policy is None:
            return relit
        subj = np.clip(subject_mask.astype(np.float32), 0.0, 1.0)
        if not np.any(subj > 0.08):
            return relit

        direction = look_policy.direction if isinstance(look_policy.direction, dict) else {}
        chroma = look_policy.chroma if isinstance(look_policy.chroma, dict) else {}
        region = look_policy.region if isinstance(look_policy.region, dict) else {}
        render_weight = look_policy.render_weight if isinstance(look_policy.render_weight, dict) else {}
        exposure = look_policy.exposure if isinstance(look_policy.exposure, dict) else {}

        key_uv = direction.get('key_uv', [0.5, 0.28])
        try:
            key_uv = [float(np.clip(key_uv[0], 0.0, 1.0)), float(np.clip(key_uv[1], 0.0, 1.0))]
        except Exception:
            key_uv = [0.5, 0.28]
        key_dir_raw = direction.get('key_dir', None)
        try:
            key_dir = safe_norm(np.array(key_dir_raw, dtype=np.float32).reshape(3))
        except Exception:
            key_dir = safe_norm(np.array(policy_direction_from_uv(key_uv), dtype=np.float32))
        if not np.all(np.isfinite(key_dir)):
            key_dir = safe_norm(np.array([0.28, -0.20, 0.94], dtype=np.float32))

        h, w = subj.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)

        NdotL_raw = np.sum(N * key_dir.reshape(1, 1, 3), axis=-1).astype(np.float32)
        diffusion = float(np.clip(direction.get('diffusion_spread', 0.34), 0.08, 0.88))
        NdotL = np.clip((NdotL_raw + diffusion) / (1.0 + diffusion), 0.0, 1.0).astype(np.float32)

        x = P[..., 0].astype(np.float32)
        y = P[..., 1].astype(np.float32)
        subj_bool = subj > 0.08
        if np.any(subj_bool):
            xr = float(np.percentile(np.abs(x[subj_bool]), 90.0))
            yr = float(np.percentile(np.abs(y[subj_bool]), 90.0))
        else:
            xr, yr = 1.0, 1.0
        x_norm = np.clip(x / max(xr, 1e-5), -1.0, 1.0)
        y_norm = np.clip(y / max(yr, 1e-5), -1.0, 1.0)
        pos_field = np.clip(0.5 + 0.34 * key_dir[0] * x_norm - 0.22 * key_dir[1] * y_norm, 0.0, 1.0).astype(np.float32)

        field = getattr(lighting_info, 'gradient_field', {}) if hasattr(lighting_info, 'gradient_field') else {}
        if not isinstance(field, dict):
            field = {}
        hb = float(field.get('horizontal_bias', self._atmosphere_descriptor.get('horizontal_bias', 0.0) if self._atmosphere_descriptor else 0.0))
        vb = float(field.get('vertical_bias', self._atmosphere_descriptor.get('vertical_bias', 0.0) if self._atmosphere_descriptor else 0.0))
        gradient_strength = float(np.clip(direction.get('gradient_light_strength', self._atmosphere_descriptor.get('gradient_strength', 0.0) if self._atmosphere_descriptor else 0.0), 0.0, 1.0))
        bg_axis = hb * (u - 0.5) + vb * (v - 0.5)
        bg_axis_norm = np.clip(0.5 + bg_axis / max(abs(hb) + abs(vb), 1e-4), 0.0, 1.0).astype(np.float32)

        uv_vec_x = (key_uv[0] - 0.5)
        uv_vec_y = (key_uv[1] - 0.5)
        uv_len = max(float(np.sqrt(uv_vec_x * uv_vec_x + uv_vec_y * uv_vec_y)), 1e-4)
        uv_field = np.clip(0.5 + ((u - 0.5) * uv_vec_x + (v - 0.5) * uv_vec_y) / uv_len, 0.0, 1.0).astype(np.float32)

        ndotl_std = float(np.std(NdotL[subj_bool])) if np.any(subj_bool) else 0.0
        frontal_fallback = float(np.clip((0.075 - ndotl_std) / 0.075, 0.0, 1.0))
        normal_w = 0.56 * (1.0 - frontal_fallback) + 0.18 * frontal_fallback
        pos_w = 0.26 + 0.28 * frontal_fallback
        bg_w = 0.12 + 0.28 * gradient_strength
        uv_w = 0.06 + 0.18 * float(np.clip(direction.get('local_light_confidence', 0.0), 0.0, 1.0))
        denom = max(normal_w + pos_w + bg_w + uv_w, 1e-5)
        directional_field = (
            normal_w * NdotL
            + pos_w * pos_field
            + bg_w * bg_axis_norm
            + uv_w * uv_field
        ) / denom
        directional_field = box_blur_gray(np.clip(directional_field, 0.0, 1.0).astype(np.float32), passes=max(1, int(round(1 + diffusion * 2))))

        side_sep = float(np.clip(direction.get('side_separation', 0.24), 0.04, 0.90))
        spread = float(np.clip(0.18 + 0.30 * diffusion - 0.10 * side_sep, 0.10, 0.44))
        lit_side = smoothstep(0.50 - spread, 0.50 + spread, directional_field) * subj
        shadow_side = (1.0 - smoothstep(0.42 - spread * 0.5, 0.58 + spread * 0.5, directional_field)) * subj
        edge = np.clip(edge_band * subj, 0.0, 1.0).astype(np.float32)
        hair = np.clip(hair_region * subj, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj, 0.0, 1.0).astype(np.float32)
        face_side = np.clip(face_side * subj, 0.0, 1.0).astype(np.float32)
        body = np.clip(body_region * subj, 0.0, 1.0).astype(np.float32)
        cloth = np.clip(cloth_region * subj, 0.0, 1.0).astype(np.float32)
        shoulder = np.clip(shoulder_region * subj, 0.0, 1.0).astype(np.float32)
        rim_side = np.clip((edge * 0.78 + hair * 0.42 + shoulder * 0.28) * (0.42 + 0.58 * shadow_side), 0.0, 1.0).astype(np.float32)

        direct_w = float(np.clip(render_weight.get('direct_weight', render_weight.get('direct_light', 1.0)), 0.0, 1.65))
        shadow_w = float(np.clip(render_weight.get('shadow_weight', 0.72), 0.0, 1.45))
        rim_w = float(np.clip(render_weight.get('rim_weight', render_weight.get('rim', 1.0)), 0.0, 1.80))
        gradient_w = float(np.clip(render_weight.get('gradient_weight', render_weight.get('gradient_field', 1.0)), 0.0, 1.45))
        multi_w = float(np.clip(render_weight.get('multicolor_weight', render_weight.get('multicolor', 1.0)), 0.0, 1.45))
        fill_w = float(np.clip(render_weight.get('fill_weight', render_weight.get('fill', 1.0)), 0.0, 1.35))
        ambient_w = float(np.clip(render_weight.get('ambient_weight', render_weight.get('ambient', 1.0)), 0.0, 1.35))
        spec_w = float(np.clip(render_weight.get('specular_weight', render_weight.get('specular', 1.0)), 0.0, 1.35))

        direct_strength = float(np.clip(direction.get('directional_light_strength', 0.36), 0.0, 1.0))
        shadow_strength = float(np.clip(direction.get('shadow_strength', direction.get('directional_shadow_strength', 0.18)), 0.0, 0.85))
        rim_strength = float(np.clip(direction.get('rim_strength', 0.24), 0.0, 1.0))
        face_readability = float(np.clip(exposure.get('face_readability', 0.48), 0.0, 1.0))
        shadow_floor = float(np.clip(exposure.get('shadow_floor', 0.08), 0.02, 0.24))

        face_core_w = float(np.clip(region.get('face_core_weight', 0.25), 0.05, 0.55))
        face_side_w = float(np.clip(region.get('face_side_weight', 0.45), 0.12, 0.95))
        body_w = float(np.clip(region.get('body_weight', 0.48), 0.12, 1.0))
        cloth_w = float(np.clip(region.get('cloth_weight', region.get('clothing_weight', 0.60)), 0.12, 1.0))
        hair_w = float(np.clip(region.get('hair_weight', 0.72), 0.12, 1.2))
        edge_w = float(np.clip(region.get('edge_weight', 0.76), 0.12, 1.2))
        shoulder_w = float(np.clip(region.get('shoulder_weight', 0.56), 0.12, 1.0))

        lit_weight = np.clip(
            face * face_core_w * 0.42
            + face_side * face_side_w
            + body * body_w
            + cloth * cloth_w
            + hair * hair_w
            + edge * edge_w
            + shoulder * shoulder_w,
            0.0,
            1.5,
        ).astype(np.float32)
        shadow_weight_map = np.clip(
            face * face_core_w * (0.30 + 0.28 * face_readability)
            + face_side * face_side_w * 0.82
            + body * body_w
            + cloth * cloth_w
            + hair * hair_w * 0.85
            + shoulder * shoulder_w,
            0.0,
            1.35,
        ).astype(np.float32)

        rel_l = np.maximum(rgb_luminance(np.clip(relit, 0.0, None)), 1e-5).astype(np.float32)
        lift = lit_side * lit_weight * direct_strength * direct_w * (0.030 + 0.055 * side_sep)
        lift += face * face_readability * fill_w * 0.010
        darken = shadow_side * shadow_weight_map * shadow_strength * shadow_w * (0.030 + 0.052 * side_sep)
        darken *= np.clip(1.0 - face * face_readability * 0.42, 0.44, 1.0)
        new_l = np.clip(rel_l + lift - darken, shadow_floor * subj + rel_l * (1.0 - subj), 1.0).astype(np.float32)
        out = relit * np.clip(new_l / rel_l, 0.76, 1.28)[..., None]

        try:
            key_color = np.array(getattr(lighting_info, 'key_color', getattr(lighting_info, 'ambient_color', (0.6, 0.6, 0.6))), dtype=np.float32)
            ambient_color = np.array(getattr(lighting_info, 'ambient_color', key_color), dtype=np.float32)
            global_color = np.array(getattr(lighting_info, 'global_mean_color', ambient_color), dtype=np.float32)
        except Exception:
            key_color = ambient_color = global_color = np.array([0.6, 0.6, 0.6], dtype=np.float32)

        if isinstance(field, dict):
            side_color = np.array(field.get('right_color' if key_uv[0] >= 0.5 else 'left_color', key_color), dtype=np.float32)
            rim_color = np.array(field.get('left_color' if key_uv[0] >= 0.5 else 'right_color', ambient_color), dtype=np.float32)
        else:
            side_color = key_color
            rim_color = ambient_color
        light_color = np.clip(0.50 * key_color + 0.30 * side_color + 0.20 * ambient_color, 1e-5, 6.0)
        rim_color = np.clip(0.55 * rim_color + 0.25 * global_color + 0.20 * ambient_color, 1e-5, 6.0)

        def color_dir(c: np.ndarray, lo: float = 0.45, hi: float = 2.40) -> np.ndarray:
            c = np.clip(c.astype(np.float32), 1e-5, 6.0)
            return np.clip(c / max(float(np.dot(c, LUMA)), 1e-5), lo, hi).astype(np.float32)

        light_dir_color = color_dir(light_color)
        rim_dir_color = color_dir(rim_color)
        skin_limit = float(np.clip(chroma.get('skin_tint_limit', 0.080), 0.015, 0.20))
        body_tint = float(np.clip(chroma.get('body_tint_strength', chroma.get('body', 0.06)), 0.0, 0.30))
        cloth_tint = float(np.clip(chroma.get('cloth_tint_strength', chroma.get('clothing', 0.10)), 0.0, 0.50))
        hair_tint = float(np.clip(chroma.get('hair_tint_strength', chroma.get('hair', 0.18)), 0.0, 0.75))
        edge_spill = float(np.clip(chroma.get('edge_color_spill', chroma.get('edge', 0.20)), 0.0, 0.85))
        palette_sep = float(np.clip(chroma.get('palette_separation', 0.22), 0.0, 1.0))

        face_core_spill = np.clip(face * skin_limit * 0.20 * lit_side, 0.0, skin_limit * 0.28)
        face_side_spill = np.clip(face_side * min(skin_limit * 1.65, 0.13) * lit_side * (0.55 + 0.45 * side_sep), 0.0, 0.16)
        body_spill = np.clip(body * body_tint * lit_side * (0.38 + 0.34 * gradient_w), 0.0, 0.26)
        cloth_spill = np.clip(cloth * cloth_tint * lit_side * (0.46 + 0.34 * multi_w), 0.0, 0.42)
        hair_spill = np.clip(hair * hair_tint * (0.35 + 0.65 * np.maximum(lit_side, rim_side)), 0.0, 0.58)
        edge_spill_m = np.clip(edge * edge_spill * (0.40 + 0.60 * rim_side), 0.0, 0.70)
        shoulder_spill = np.clip(shoulder * max(body_tint, cloth_tint) * (0.35 + 0.55 * lit_side), 0.0, 0.36)
        color_spill_mask = np.clip(
            face_core_spill + face_side_spill + body_spill + cloth_spill + hair_spill + edge_spill_m + shoulder_spill,
            0.0,
            0.72,
        ).astype(np.float32)
        color_spill_mask *= float(np.clip(0.38 + 0.32 * gradient_w + 0.30 * multi_w, 0.22, 1.0))

        out_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5).astype(np.float32)
        out_dir = np.clip(out / out_l[..., None], 0.20, 3.0).astype(np.float32)
        target_dir = np.clip(
            light_dir_color.reshape(1, 1, 3) * (1.0 - rim_side[..., None] * palette_sep)
            + rim_dir_color.reshape(1, 1, 3) * (rim_side[..., None] * palette_sep),
            0.20,
            3.0,
        ).astype(np.float32)
        out_dir = out_dir * (1.0 - color_spill_mask[..., None]) + target_dir * color_spill_mask[..., None]
        out = np.clip(out_l[..., None] * out_dir, 0.0, 8.0)

        rim_lift = rim_side * rim_strength * rim_w * (0.030 + 0.060 * edge_spill) * np.clip(edge_w * edge + hair_w * hair + shoulder_w * shoulder, 0.0, 1.4)
        spec_hint = rim_side * spec_w * (0.006 + 0.018 * rim_strength) * np.clip(hair + edge + cloth * 0.35, 0.0, 1.0)
        out += rim_dir_color.reshape(1, 1, 3) * (rim_lift + spec_hint)[..., None]
        out = out * subj[..., None] + relit * (1.0 - subj[..., None])
        out = np.clip(out, 0.0, 8.0).astype(np.float32)

        skin_protect_mask = np.clip(face * (1.0 - skin_limit / 0.20) * color_spill_mask * 1.8, 0.0, 1.0).astype(np.float32)
        self._unified_directional_field = directional_field.astype(np.float32)
        self._unified_lit_side = lit_side.astype(np.float32)
        self._unified_shadow_side = shadow_side.astype(np.float32)
        self._unified_rim_side = rim_side.astype(np.float32)
        self._unified_region_weights = np.dstack([
            np.clip(face * face_core_w + face_side * face_side_w, 0.0, 1.0),
            np.clip(body * body_w + cloth * cloth_w + shoulder * shoulder_w, 0.0, 1.0),
            np.clip(hair * hair_w + edge * edge_w, 0.0, 1.0),
        ]).astype(np.float32)
        self._unified_before_directional = np.clip(relit, 0.0, 8.0).astype(np.float32)
        self._unified_after_directional = out
        self._unified_directional_delta = np.clip(np.abs(out - relit).mean(axis=-1) * 8.0, 0.0, 1.0).astype(np.float32)
        self._unified_color_spill_mask = color_spill_mask
        self._unified_skin_protect_mask = skin_protect_mask
        self._direction_runtime = {
            'key_dir': [float(x) for x in key_dir],
            'key_uv': [float(x) for x in key_uv],
            'directional_light_strength_used': direct_strength,
            'shadow_strength_used': shadow_strength,
            'rim_strength_used': rim_strength,
            'side_separation_used': side_sep,
            'diffusion_spread_used': diffusion,
            'direct_weight_used': direct_w,
            'ambient_weight_used': ambient_w,
            'fill_weight_used': fill_w,
            'gradient_weight_used': gradient_w,
            'multicolor_weight_used': multi_w,
            'rim_weight_used': rim_w,
            'specular_weight_used': spec_w,
            'shadow_weight_used': shadow_w,
            'lit_mean': float(np.mean(lit_side[subj_bool])) if np.any(subj_bool) else 0.0,
            'shadow_mean': float(np.mean(shadow_side[subj_bool])) if np.any(subj_bool) else 0.0,
            'rim_mean': float(np.mean(rim_side[subj_bool])) if np.any(subj_bool) else 0.0,
            'normal_field_std': ndotl_std,
            'frontal_fallback': frontal_fallback,
        }
        print(
            "[DirectionField] "
            f"key_dir=({key_dir[0]:.2f},{key_dir[1]:.2f},{key_dir[2]:.2f}) "
            f"lit_mean={self._direction_runtime['lit_mean']:.2f} "
            f"shadow_mean={self._direction_runtime['shadow_mean']:.2f} "
            f"rim_mean={self._direction_runtime['rim_mean']:.2f} "
            f"direct={direct_strength * direct_w:.2f} "
            f"shadow={shadow_strength * shadow_w:.2f} "
            f"rim={rim_strength * rim_w:.2f}"
        )
        return out
