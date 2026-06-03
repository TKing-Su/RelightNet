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

class RendererBackgroundGradientMixin:
    def _compute_background_gradient_light(
        self,
        base_subject: np.ndarray,
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
        source_shape: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project the background as a continuous optical light field.

        This version does not collapse the background to one average color.  It uses:
        1) a compact low-frequency 2D color grid sampled by screen position;
        2) a normal-projected environment sample, so face sides/hair receive different colors;
        3) local source peaks, so sunsets/neon/signs create localized rim/specular lift;
        4) a luminance light-map separate from chroma, preventing the old transparent paint look.
        """
        field = getattr(lighting_info, 'gradient_field', None)
        if not isinstance(field, dict) or not bool(field.get('enabled', False)):
            return np.zeros_like(base_subject, dtype=np.float32)

        h, w = subject_mask.shape
        if h <= 1 or w <= 1 or not np.any(subject_mask > 0.08):
            return np.zeros_like(base_subject, dtype=np.float32)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u_screen = xx / max(w - 1, 1)
        v_screen = yy / max(h - 1, 1)

        ys, xs = np.where(subject_mask > 0.08)
        y0, y1 = float(ys.min()), float(ys.max())
        x0, x1 = float(xs.min()), float(xs.max())
        x_rel = np.clip((xx - x0) / max(x1 - x0, 1.0), 0.0, 1.0).astype(np.float32)
        y_rel = np.clip((yy - y0) / max(y1 - y0, 1.0), 0.0, 1.0).astype(np.float32)

        a = float(field.get('a', 0.0))
        b = float(field.get('b', 0.0))
        c = float(field.get('c', 0.0))
        p05 = float(field.get('p05_luma', 0.0))
        p95 = float(field.get('p95_luma', max(p05 + 1e-3, 0.1)))
        p50 = float(field.get('p50_luma', 0.12))
        conf = float(np.clip(field.get('confidence', 0.35), 0.0, 1.0))
        plane_strength = float(np.clip(field.get('plane_strength', 0.0), 0.0, 2.0))
        vertical_strength = float(np.clip(field.get('vertical_strength', 0.0), 0.0, 2.0))
        horizontal_strength = float(np.clip(field.get('horizontal_strength', 0.0), 0.0, 2.0))
        local_contrast = float(np.clip(field.get('local_contrast', 0.5), 0.0, 2.0))
        colorfulness = float(np.clip(field.get('colorfulness', 0.3), 0.0, 1.0))
        background_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
        if self._using_continuous_policy():
            background_mode = 'balanced'

        top_color = np.array(field.get("top_color", lighting_info.ambient_color), dtype=np.float32)
        bottom_color = np.array(field.get("bottom_color", lighting_info.ambient_color), dtype=np.float32)
        left_color = np.array(field.get("left_color", lighting_info.ambient_color), dtype=np.float32)
        right_color = np.array(field.get("right_color", lighting_info.ambient_color), dtype=np.float32)
        center_color = np.array(field.get("center_color", lighting_info.ambient_color), dtype=np.float32)

        min_light_luma = 0.12 + 0.07 * conf + (0.035 if p50 < 0.11 else 0.0)
        top_color = brighten_preserve_hue(top_color, max(float(np.dot(top_color, LUMA)), min_light_luma * 0.80))
        bottom_color = brighten_preserve_hue(bottom_color, max(float(np.dot(bottom_color, LUMA)), min_light_luma))
        left_color = brighten_preserve_hue(left_color, max(float(np.dot(left_color, LUMA)), min_light_luma * 0.88))
        right_color = brighten_preserve_hue(right_color, max(float(np.dot(right_color, LUMA)), min_light_luma * 0.88))
        center_color = brighten_preserve_hue(center_color, max(float(np.dot(center_color, LUMA)), min_light_luma * 0.84))

        # Screen-space low-frequency field: preserves complex backgrounds.
        grid_colors = field.get('grid_colors', None)
        grid_luma = field.get('grid_luma', None)
        if grid_colors is not None:
            try:
                gcol = np.asarray(grid_colors, dtype=np.float32)
                screen_color = self._bilinear_sample_field_grid(gcol, u_screen, v_screen)
                # Environment sample from normal direction: sides/hair receive different colors.
                n_u = np.clip(0.5 + 0.46 * N[..., 0], 0.0, 1.0)
                n_v = np.clip(0.50 - 0.38 * N[..., 1] + 0.10 * (1.0 - np.clip(N[..., 2], 0.0, 1.0)), 0.0, 1.0)
                env_color = self._bilinear_sample_field_grid(gcol, n_u, n_v)
                if grid_luma is not None:
                    gl = np.asarray(grid_luma, dtype=np.float32)
                    screen_luma_grid = self._bilinear_sample_field_grid(gl, u_screen, v_screen)[..., 0]
                else:
                    screen_luma_grid = rgb_luminance(screen_color)
            except Exception:
                screen_color = None
                env_color = None
                screen_luma_grid = None
        else:
            screen_color = None
            env_color = None
            screen_luma_grid = None

        vertical_color = top_color.reshape(1, 1, 3) * (1.0 - y_rel[..., None]) + bottom_color.reshape(1, 1, 3) * y_rel[..., None]
        horizontal_color = left_color.reshape(1, 1, 3) * (1.0 - x_rel[..., None]) + right_color.reshape(1, 1, 3) * x_rel[..., None]
        axis_sum = max(vertical_strength + horizontal_strength, 1e-6)
        v_mix = float(np.clip(0.50 + 0.40 * vertical_strength / axis_sum, 0.50, 0.88))
        ramp_color = vertical_color * v_mix + horizontal_color * (1.0 - v_mix)
        ramp_color = ramp_color * 0.84 + center_color.reshape(1, 1, 3) * 0.16

        if screen_color is not None and env_color is not None:
            # For complex/rich backgrounds, screen color carries the actual palette;
            # env color restores geometry, especially on hair/cheeks/edges.
            if background_mode == 'rich':
                field_color = 0.26 * ramp_color + 0.42 * screen_color + 0.32 * env_color
            elif background_mode == 'monotone':
                field_color = 0.52 * ramp_color + 0.26 * screen_color + 0.22 * env_color
            else:
                field_color = 0.38 * ramp_color + 0.36 * screen_color + 0.26 * env_color
        else:
            field_color = ramp_color

        refl_color = None
        if grid_colors is not None:
            try:
                gcol = np.asarray(grid_colors, dtype=np.float32)
                R = safe_norm(2.0 * N * np.sum(N * V, axis=-1, keepdims=True) - V)
                r_u = np.clip(0.5 + 0.48 * R[..., 0], 0.0, 1.0)
                r_v = np.clip(0.50 - 0.42 * R[..., 1] + 0.08 * (1.0 - np.clip(R[..., 2], 0.0, 1.0)), 0.0, 1.0)
                refl_color = self._bilinear_sample_field_grid(gcol, r_u, r_v)
            except Exception:
                refl_color = None

        plane = a * (u_screen - 0.5) + b * (v_screen - 0.5) + c
        screen_profile = np.clip((plane - p05) / max(p95 - p05, 1e-6), 0.0, 1.0).astype(np.float32)
        if screen_luma_grid is not None:
            grid_profile = np.clip((screen_luma_grid - p05) / max(p95 - p05, 1e-6), 0.0, 1.0).astype(np.float32)
            screen_profile = np.clip(0.55 * screen_profile + 0.45 * grid_profile, 0.0, 1.0)

        vertical_bias = float(field.get('vertical_bias', 0.0))
        horizontal_bias = float(field.get('horizontal_bias', 0.0))
        bottom_to_top = vertical_bias >= 0.0
        right_to_left = horizontal_bias >= 0.0
        y_axis_profile = y_rel if bottom_to_top else (1.0 - y_rel)
        x_axis_profile = x_rel if right_to_left else (1.0 - x_rel)
        axis_profile = (vertical_strength * y_axis_profile + horizontal_strength * x_axis_profile) / axis_sum
        axis_profile = np.clip(axis_profile, 0.0, 1.0).astype(np.float32)
        profile = np.clip(0.38 * screen_profile + 0.62 * axis_profile, 0.0, 1.0)
        profile = (profile * profile * (3.0 - 2.0 * profile)).astype(np.float32)

        facing = np.clip(N[..., 2], 0.0, 1.0).astype(np.float32)
        lower_region = np.clip((y_rel - 0.14) / 0.76, 0.0, 1.0)
        upper_region = np.clip((0.86 - y_rel) / 0.76, 0.0, 1.0)
        vertical_region = lower_region if bottom_to_top else upper_region
        side_region = x_axis_profile
        geom_region = (vertical_strength * vertical_region + horizontal_strength * side_region) / axis_sum
        geom_region = np.clip(geom_region, 0.0, 1.0).astype(np.float32)

        # Use normals, but tolerate imperfect normal maps.  Hair/edges need more
        # background color than central skin to achieve the reference style.
        normal_gate = np.clip(0.42 + 0.48 * facing + 0.20 * geom_region, 0.0, 1.20).astype(np.float32)
        edge_hair_boost = np.clip(1.0 + 0.34 * hair_region + 0.26 * edge_band, 1.0, 1.55).astype(np.float32)
        face_chroma_gate = np.clip(1.02 - 0.015 * face_core + 0.48 * geom_region + 0.16 * colorfulness, 0.88, 1.42).astype(np.float32)
        if bottom_to_top and vertical_strength >= horizontal_strength * 0.65:
            face_chroma_gate = np.maximum(face_chroma_gate, np.clip(0.86 + 0.55 * lower_region, 0.86, 1.34))

        source_shape_gate = 1.0
        if source_shape is not None:
            source_shape_gate = np.clip(0.70 + 0.30 * source_shape, 0.62, 1.18).astype(np.float32)

        scene_dark_bonus = 0.10 if p50 < 0.13 else 0.0
        gradient_contrast = float(np.clip((p95 - p05) / max(p50 + 0.05, 1e-6), 0.0, 1.8))
        strength = float(np.clip(0.13 + 0.28 * conf + 0.16 * min(plane_strength, 1.0) + 0.10 * gradient_contrast + 0.10 * local_contrast + scene_dark_bonus, 0.12, 0.78))
        if background_mode == 'rich':
            strength *= 1.04
        elif background_mode == 'monotone':
            strength *= 1.02
        strength = float(np.clip(strength, 0.11, 0.82))

        field_luma = rgb_luminance(field_color)
        # Preserve hue but do not normalize everything to white.  The ratio is
        # deliberately allowed to be stronger on non-face regions.
        color_dir = field_color / np.maximum(field_luma[..., None], 1e-5)
        color_dir = np.clip(color_dir, 0.48, 2.15).astype(np.float32)

        mask_gate = np.power(np.clip(subject_mask, 0.0, 1.0), 0.84).astype(np.float32)
        profile_gate = np.clip(0.12 + 0.88 * profile, 0.0, 1.0) * mask_gate * normal_gate * edge_hair_boost * source_shape_gate

        # Separate luminance lift from chroma.  Too much neutral lift caused the
        # pale/white result; too much chroma caused transparent paint.  The updated
        # version keeps diffuse/environment energy and glossy energy separate.
        diffuse_gate = np.clip(0.16 + 0.54 * profile + 0.18 * geom_region + 0.12 * facing, 0.0, 1.0).astype(np.float32)
        neutral_lift = base_subject * (strength * 0.065) * profile_gate[..., None] * diffuse_gate[..., None]
        chroma_lift = base_subject * color_dir * (strength * (0.78 + 0.62 * colorfulness)) * profile_gate[..., None] * face_chroma_gate[..., None] * (0.74 + 0.32 * diffuse_gate[..., None])

        # Environment sheen: this is the missing "skin gloss / hair sheen" term.
        # It uses the reflected field sample and roughness/specular maps, so portraits
        # no longer look dry under gradient or sunset backgrounds.
        env_spec_acc = np.zeros_like(base_subject, dtype=np.float32)
        micro = np.clip(1.0 - roughness_map, 0.0, 1.0).astype(np.float32)
        fresnel = np.power(np.clip(1.0 - facing, 0.0, 1.0), 5.0).astype(np.float32)
        fresnel = 0.06 + 0.94 * fresnel
        if refl_color is not None:
            refl_luma = rgb_luminance(refl_color)
            refl_dir = np.clip(refl_color / np.maximum(refl_luma[..., None], 1e-5), 0.42, 2.60).astype(np.float32)
            sheen_region = np.clip(0.22 + 0.58 * face_core + 0.44 * hair_region + 0.26 * edge_band + 0.10 * geom_region, 0.0, 1.55).astype(np.float32)
            sheen_strength = float(np.clip(0.035 + 0.115 * conf + 0.045 * local_contrast + 0.040 * colorfulness, 0.025, 0.24))
            env_spec_acc = refl_dir * specular_map[..., None] * micro[..., None] * fresnel[..., None] * sheen_region[..., None] * mask_gate[..., None] * sheen_strength
            env_spec_acc *= (0.82 + 0.18 * source_shape_gate[..., None])

        # Local source peaks from sunsets/neon/signage.  They now produce not only
        # local color spill but also directional micro-specular and soft diffuse boost.
        peak_acc = np.zeros_like(base_subject, dtype=np.float32)
        peak_spec_acc = np.zeros_like(base_subject, dtype=np.float32)
        peaks = field.get('source_peaks', []) or []
        if isinstance(peaks, list) and peaks:
            peak_scores = np.array([max(float(p.get('score', p.get('power', 0.0))), 1e-6) for p in peaks], dtype=np.float32)
            peak_norm = float(np.percentile(peak_scores, 90.0)) if peak_scores.size else 1.0
            for p in peaks[:8]:
                try:
                    pu = float(p.get('u', 0.5)); pv = float(p.get('v', 0.5))
                    pc = np.array(p.get('color', center_color), dtype=np.float32)
                    ps = float(p.get('score', p.get('power', 1.0))) / max(peak_norm, 1e-6)
                    ps = float(np.clip(ps, 0.0, 1.50))
                except Exception:
                    continue
                pc = brighten_preserve_hue(pc, max(float(np.dot(pc, LUMA)), min_light_luma))
                pc_l = max(float(np.dot(pc, LUMA)), 1e-5)
                pc_dir = np.clip(pc / pc_l, 0.45, 2.50)
                # side relation: source at left lights left outline/face side; same for right.
                side_sign = -1.0 if pu < 0.5 else 1.0
                side_gate = self._compute_signed_side_mask(P, side_sign, subject_mask, power=0.82)
                vert_gate = np.exp(-0.5 * ((y_rel - np.clip(pv, 0.0, 1.0)) / 0.34) ** 2).astype(np.float32)
                edge_gate = np.clip(0.18 + 0.52 * side_gate + 0.22 * hair_region + 0.30 * edge_band, 0.0, 1.35)
                face_gate = np.clip(0.24 + 0.76 * side_gate, 0.0, 1.0) * np.clip(1.0 - 0.36 * face_core + 0.24 * geom_region, 0.34, 1.0)
                local_gate = mask_gate * vert_gate * np.maximum(edge_gate, face_gate)
                peak_acc += base_subject * pc_dir.reshape(1, 1, 3) * local_gate[..., None] * (0.060 * ps * (0.70 + 0.72 * colorfulness))

                Lp = self._uv_to_direction(pu, pv, (h, w))
                Lpf = np.ones_like(N) * Lp.reshape(1, 1, 3)
                H = safe_norm(Lpf + V)
                NdotL = np.clip(np.sum(N * Lpf, axis=-1), 0.0, 1.0).astype(np.float32)
                NdotH = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0).astype(np.float32)
                peak_diff = np.power(np.clip((NdotL + 0.16) / 1.16, 0.0, 1.0), 0.95)
                gloss_exp = 12.0 + 44.0 * micro
                peak_gloss = np.power(np.clip(NdotH, 0.0, 1.0), gloss_exp).astype(np.float32)
                peak_region = local_gate * np.clip(0.36 + 0.64 * side_gate + 0.16 * geom_region, 0.0, 1.25)
                peak_acc += base_subject * pc_dir.reshape(1, 1, 3) * peak_diff[..., None] * peak_region[..., None] * (0.045 * ps)
                peak_spec_acc += pc_dir.reshape(1, 1, 3) * specular_map[..., None] * micro[..., None] * fresnel[..., None] * peak_gloss[..., None] * peak_region[..., None] * (0.055 * ps)

        readable_shadow = np.clip((0.26 - rgb_luminance(base_subject)) / 0.26, 0.0, 1.0) * mask_gate
        shadow_lift = base_subject * color_dir * readable_shadow[..., None] * (0.030 + 0.060 * scene_dark_bonus) * (0.35 + 0.65 * profile[..., None])

        total = neutral_lift + chroma_lift + env_spec_acc + peak_acc + peak_spec_acc + shadow_lift
        return np.clip(total, 0.0, 4.0).astype(np.float32)
