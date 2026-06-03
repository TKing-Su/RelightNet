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

class RendererReflectiveFinishMixin:
    def _compute_background_reflective_finish(
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
        intrinsic_gloss: Optional[np.ndarray] = None,
        camera_params: Optional[CameraParams] = None,
    ) -> np.ndarray:
        """Add a visible background-driven reflective layer.

        The main relight pass is physically conservative, which is safe but often
        looks dry on portraits.  This pass is still background-driven: it samples the
        low-frequency background light field by reflection direction and converts
        local bright/color peaks into glossy lobes.  It is added after AO/contact
        shadows so highlights remain visible, similar to real skin/hair sheen.
        """
        field = getattr(lighting_info, 'gradient_field', None)
        if not isinstance(field, dict) or not bool(field.get('enabled', False)):
            return np.zeros_like(base_subject, dtype=np.float32)
        h, w = subject_mask.shape
        if h <= 2 or w <= 2 or not np.any(subject_mask > 0.08):
            return np.zeros_like(base_subject, dtype=np.float32)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u_screen = xx / max(w - 1, 1)
        v_screen = yy / max(h - 1, 1)
        ys, xs = np.where(subject_mask > 0.08)
        y0, y1 = float(ys.min()), float(ys.max())
        x0, x1 = float(xs.min()), float(xs.max())
        x_rel = np.clip((xx - x0) / max(x1 - x0, 1.0), 0.0, 1.0).astype(np.float32)
        y_rel = np.clip((yy - y0) / max(y1 - y0, 1.0), 0.0, 1.0).astype(np.float32)

        conf = float(np.clip(field.get('confidence', 0.35), 0.0, 1.0))
        local_contrast = float(np.clip(field.get('local_contrast', 0.45), 0.0, 2.0))
        colorfulness = float(np.clip(field.get('colorfulness', 0.25), 0.0, 1.0))
        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
        bg_p50 = float(field.get('p50_luma', 0.20))
        bg_p95 = float(field.get('p95_luma', 0.42))
        dark_scene = 1.0 if bg_p50 < 0.14 else 0.0

        grid_colors = field.get('grid_colors', None)
        key_color = np.array(getattr(lighting_info, 'key_color', getattr(lighting_info, 'ambient_color', (0.7, 0.7, 0.7))), dtype=np.float32)
        center_color = np.array(field.get('center_color', key_color), dtype=np.float32)
        top_color = np.array(field.get('top_color', center_color), dtype=np.float32)
        bottom_color = np.array(field.get('bottom_color', center_color), dtype=np.float32)
        left_color = np.array(field.get('left_color', center_color), dtype=np.float32)
        right_color = np.array(field.get('right_color', center_color), dtype=np.float32)

        R = safe_norm(2.0 * N * np.sum(N * V, axis=-1, keepdims=True) - V)
        r_u = np.clip(0.5 + 0.48 * R[..., 0], 0.0, 1.0)
        r_v = np.clip(0.50 - 0.42 * R[..., 1] + 0.10 * (1.0 - np.clip(R[..., 2], 0.0, 1.0)), 0.0, 1.0)
        if grid_colors is not None:
            try:
                gcol = np.asarray(grid_colors, dtype=np.float32)
                refl_color = self._bilinear_sample_field_grid(gcol, r_u, r_v)
                screen_color = self._bilinear_sample_field_grid(gcol, u_screen, v_screen)
            except Exception:
                refl_color = None
                screen_color = None
        else:
            refl_color = None
            screen_color = None
        if refl_color is None:
            vcol = top_color.reshape(1, 1, 3) * (1.0 - y_rel[..., None]) + bottom_color.reshape(1, 1, 3) * y_rel[..., None]
            hcol = left_color.reshape(1, 1, 3) * (1.0 - x_rel[..., None]) + right_color.reshape(1, 1, 3) * x_rel[..., None]
            refl_color = 0.52 * vcol + 0.36 * hcol + 0.12 * center_color.reshape(1, 1, 3)
            screen_color = refl_color

        refl_luma = rgb_luminance(refl_color)
        refl_dir_raw = np.clip(refl_color / np.maximum(refl_luma[..., None], 1e-5), 0.62, 1.85).astype(np.float32)
        refl_dir = np.clip(0.68 + 0.32 * refl_dir_raw, 0.72, 1.42).astype(np.float32)
        screen_luma = rgb_luminance(screen_color)
        screen_dir_raw = np.clip(screen_color / np.maximum(screen_luma[..., None], 1e-5), 0.62, 1.80).astype(np.float32)
        screen_dir = np.clip(0.66 + 0.34 * screen_dir_raw, 0.72, 1.45).astype(np.float32)

        mask_gate = np.power(np.clip(subject_mask, 0.0, 1.0), 0.82).astype(np.float32)
        facing = np.clip(np.sum(N * V, axis=-1), 0.0, 1.0).astype(np.float32)
        fresnel = 0.05 + 0.95 * np.power(np.clip(1.0 - facing, 0.0, 1.0), 4.0)
        gloss = np.clip(1.0 - roughness_map, 0.0, 1.0).astype(np.float32)
        gloss_soft = np.power(gloss, 0.58).astype(np.float32)
        intrinsic_term = np.ones_like(gloss_soft, dtype=np.float32) if intrinsic_gloss is None else np.clip(intrinsic_gloss, 0.0, 1.12).astype(np.float32)
        material = np.clip(0.12 + 0.34 * specular_map, 0.0, 0.44) * gloss_soft * (0.80 + 0.20 * intrinsic_term)

        # Portrait-region masks.  These are intentionally soft and only act where
        # the alpha/normal maps say the subject faces camera, avoiding a sticker look.
        upper_window = np.exp(-0.5 * (((x_rel - 0.50) / 0.42) ** 2 + ((y_rel - 0.36) / 0.34) ** 2)).astype(np.float32)
        nose_bridge = np.exp(-0.5 * (((x_rel - 0.50) / 0.075) ** 2 + ((y_rel - 0.40) / 0.24) ** 2)).astype(np.float32)
        cheek_l = np.exp(-0.5 * (((x_rel - 0.36) / 0.13) ** 2 + ((y_rel - 0.43) / 0.18) ** 2)).astype(np.float32)
        cheek_r = np.exp(-0.5 * (((x_rel - 0.64) / 0.13) ** 2 + ((y_rel - 0.43) / 0.18) ** 2)).astype(np.float32)
        chin_neck = np.exp(-0.5 * (((x_rel - 0.50) / 0.28) ** 2 + ((y_rel - 0.60) / 0.22) ** 2)).astype(np.float32)
        face_sheen_region = np.clip((0.40 * nose_bridge + 0.20 * (cheek_l + cheek_r) + 0.10 * chin_neck + 0.06 * upper_window) * face_core, 0.0, 0.85)
        hair_edge_region = np.clip(0.40 * hair_region + 0.58 * edge_band, 0.0, 0.95)
        shape_gate = np.ones_like(mask_gate, dtype=np.float32)
        if source_shape is not None:
            shape_gate = np.clip(0.56 + 0.44 * source_shape, 0.45, 1.22).astype(np.float32)

        # Global reflected environment sheen: this is what gives the first purple
        # gradient background a moist/glossy skin response instead of dry diffuse color.
        base_strength = float(np.clip(0.022 + 0.052 * conf + 0.020 * local_contrast + 0.014 * colorfulness + 0.012 * dark_scene, 0.016, 0.095))
        if bg_mode == 'rich':
            base_strength *= 1.04
        elif bg_mode == 'monotone':
            base_strength *= 1.02
        global_spec_mask = mask_gate * material * shape_gate * np.clip(0.54 * face_sheen_region + 0.40 * hair_edge_region + 0.045 * fresnel, 0.0, 0.95)
        acc = refl_dir * global_spec_mask[..., None] * base_strength

        # Soft beauty highlight from background dominant color. It is not white; it
        # inherits the background hue, and roughness controls how visible it is.
        key_l = max(float(np.dot(key_color, LUMA)), 1e-5)
        key_dir_color = np.clip(key_color / key_l, 0.70, 1.62)
        beauty = np.clip((0.46 * nose_bridge + 0.16 * cheek_l + 0.16 * cheek_r + 0.08 * chin_neck) * face_core * material * shape_gate * mask_gate, 0.0, 0.78)
        beauty_strength = float(np.clip(0.018 + 0.026 * conf + 0.010 * max(bg_p95 - bg_p50, 0.0), 0.012, 0.052))
        acc += key_dir_color.reshape(1, 1, 3) * beauty[..., None] * beauty_strength

        # Colorful low-frequency screen sheen. This helps rich/explosive backgrounds
        # affect the correct side of the portrait instead of becoming a flat orange wash.
        sidefulness = np.clip(np.abs(x_rel - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
        chroma_side = mask_gate * material * np.clip(0.08 + 0.44 * sidefulness + 0.22 * hair_region + 0.26 * edge_band, 0.0, 0.85)
        acc += screen_dir * chroma_side[..., None] * float(np.clip(0.008 + 0.026 * colorfulness + 0.014 * local_contrast, 0.004, 0.052))

        # Local bright/color peaks become actual glossy lobes and rim/sheens.
        peaks = field.get('source_peaks', []) or []
        if isinstance(peaks, list) and peaks:
            scores = np.array([max(float(p.get('score', p.get('power', 1.0))), 1e-6) for p in peaks], dtype=np.float32)
            norm = float(np.percentile(scores, 90.0)) if scores.size else 1.0
            for p in peaks[:10]:
                try:
                    pu = float(p.get('u', 0.5)); pv = float(p.get('v', 0.5))
                    pc = np.array(p.get('color', center_color), dtype=np.float32)
                    ps = float(p.get('score', p.get('power', 1.0))) / max(norm, 1e-6)
                    ps = float(np.clip(ps, 0.0, 1.65))
                except Exception:
                    continue
                pc = brighten_preserve_hue(pc, max(float(np.dot(pc, LUMA)), 0.13))
                pc_l = max(float(np.dot(pc, LUMA)), 1e-5)
                pc_dir = np.clip(pc / pc_l, 0.66, 1.78)
                Lp = self._uv_to_direction(pu, pv, (h, w), camera_params=camera_params)
                Lpf = np.ones_like(N) * Lp.reshape(1, 1, 3)
                H = safe_norm(Lpf + V)
                ndotl_raw = np.sum(N * Lpf, axis=-1).astype(np.float32)
                ndotl_wrap = np.clip((ndotl_raw + 0.24) / 1.24, 0.0, 1.0).astype(np.float32)
                ndoth = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0).astype(np.float32)
                # Broad glossy lobe, not a pin-point sparkle.  This is more suitable
                # for one-image relighting where normals/roughness are approximate.
                exp_map = 9.0 + 52.0 * np.power(gloss, 0.80)
                glossy_lobe = np.power(ndoth, exp_map).astype(np.float32)
                side_sign = -1.0 if pu < 0.5 else 1.0
                side_gate = self._compute_signed_side_mask(P, side_sign, subject_mask, power=0.78)
                vert_gate = np.exp(-0.5 * ((y_rel - np.clip(pv, 0.0, 1.0)) / 0.36) ** 2).astype(np.float32)
                # Explosive/splatter images have strong off-center peaks; let them
                # color rim/hair strongly and face side moderately.
                local_region = mask_gate * vert_gate * np.clip(0.08 + 0.36 * side_gate + 0.25 * hair_region + 0.30 * edge_band + 0.10 * face_sheen_region, 0.0, 0.95)
                peak_gloss = glossy_lobe * material * local_region * shape_gate * (0.78 + 0.12 * intrinsic_term)
                peak_soft = ndotl_wrap * material * local_region * np.clip(0.22 + 0.50 * side_gate, 0.0, 0.78)
                acc += pc_dir.reshape(1, 1, 3) * peak_gloss[..., None] * (0.038 * ps)
                acc += pc_dir.reshape(1, 1, 3) * peak_soft[..., None] * (0.014 * ps * (0.62 + 0.46 * colorfulness))

        return np.clip(acc, 0.0, 0.72).astype(np.float32)
