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

class RendererCompactPolicyLayerMixin:
    def _apply_compact_lookpolicy_layer(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        background_linear: Optional[np.ndarray],
        N: np.ndarray,
        P: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        clothing_mask: Optional[np.ndarray],
        lighting_info: Optional['LightingInfo'],
        look_policy: Optional[LookPolicy],
    ) -> np.ndarray:
        """Single look-safe final style layer.

        The base renderer and auto-gain establish a usable portrait.  This layer,
        called after that normalization, is the only look-safe place that adds
        directional luma, shadow side, rim light, compact chroma spill and a light
        display finish.
        """
        if relit.size == 0 or look_policy is None or not np.any(subject_mask > 0.08):
            return relit

        before = np.clip(relit.astype(np.float32), 0.0, 8.0)
        out = before.copy()
        subj = np.clip(subject_mask.astype(np.float32), 0.0, 1.0)
        h, w = subj.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)

        ctx = self._compact_policy_context(look_policy, lighting_info)
        direction = ctx['direction']
        chroma = ctx['chroma']
        exposure = ctx['exposure']
        render_weight = ctx['render_weight']
        display = ctx['display']
        region = ctx['region']
        descriptor = ctx['descriptor']
        budget = ctx['budget']
        field = ctx['field']
        lowkey_chroma_gate = ctx['lowkey_chroma_gate']
        air_skin_guard = ctx['air_skin_guard']
        bg_luma = ctx['bg_luma']
        bg_sat = ctx['bg_sat']
        bg_colorfulness = ctx['bg_colorfulness']
        bg_diversity = ctx['bg_diversity']
        bg_flatness = ctx['bg_flatness']
        bg_haze = ctx['bg_haze']
        bg_lowkey = ctx['bg_lowkey']
        bg_highkey = ctx['bg_highkey']
        chroma_pressure = ctx['chroma_pressure']
        hb = ctx['hb']
        vb = ctx['vb']
        bg_bias_sum = ctx['bg_bias_sum']
        bg_bias_strength = ctx['bg_bias_strength']
        dark_chroma_env = ctx['dark_chroma_env']
        warm_direction_env = ctx['warm_direction_env']
        color_key_uv = ctx['color_key_uv']
        key_uv = ctx['key_uv']
        key_dir = ctx['key_dir']

        subj_bool = subj > 0.08
        x = P[..., 0].astype(np.float32)
        y = P[..., 1].astype(np.float32)
        xr = float(np.percentile(np.abs(x[subj_bool]), 90.0)) if np.any(subj_bool) else 1.0
        yr = float(np.percentile(np.abs(y[subj_bool]), 90.0)) if np.any(subj_bool) else 1.0
        x_norm = np.clip(x / max(xr, 1e-5), -1.0, 1.0)
        y_norm = np.clip(y / max(yr, 1e-5), -1.0, 1.0)

        ndotl_raw = np.sum(N * key_dir.reshape(1, 1, 3), axis=-1).astype(np.float32)
        normal_term = np.clip((ndotl_raw + 0.22) / 1.22, 0.0, 1.0)
        normal_std = float(np.std(normal_term[subj_bool])) if np.any(subj_bool) else 0.0
        flat_normal = float(np.clip((0.075 - normal_std) / 0.075, 0.0, 1.0))

        p_ramp = np.clip(0.5 + 0.36 * key_dir[0] * x_norm - 0.22 * key_dir[1] * y_norm, 0.0, 1.0)
        if abs(hb) + abs(vb) > 1e-5:
            bg_axis = hb * (u - 0.5) + vb * (v - 0.5)
            bg_term = np.clip(0.5 + bg_axis / max(abs(hb) + abs(vb), 1e-5), 0.0, 1.0)
        else:
            bg_term = np.full_like(subj, 0.5, dtype=np.float32)
        uv_dx = key_uv[0] - 0.5
        uv_dy = key_uv[1] - 0.5
        uv_len = max(float(np.sqrt(uv_dx * uv_dx + uv_dy * uv_dy)), 1e-4)
        uv_term = np.clip(0.5 + ((u - 0.5) * uv_dx + (v - 0.5) * uv_dy) / uv_len, 0.0, 1.0)
        screen_ramp_term = np.clip(
            (0.16 - 0.10 * bg_bias_strength) * p_ramp
            + (0.68 + 0.22 * bg_bias_strength) * bg_term
            + (0.16 - 0.08 * bg_bias_strength) * uv_term,
            0.0,
            1.0,
        )

        # All compact ownership masks are deliberately low-frequency.  The normal
        # pass and alpha edge can contain hard semantic cuts; using them directly
        # creates the visible face/neck shadow plates seen in earlier outputs.
        subj_soft = feather_mask(subj, passes=2)
        face = feather_mask(np.clip(face_core * subj, 0.0, 1.0), passes=5)
        hair = feather_mask(np.clip(hair_region * subj, 0.0, 1.0), passes=4)
        edge = feather_mask(np.clip(edge_band * subj, 0.0, 1.0), passes=3)
        face = np.clip(face * subj_soft, 0.0, 1.0).astype(np.float32)
        hair = np.clip(hair * subj_soft, 0.0, 1.0).astype(np.float32)
        edge = np.clip(edge * subj_soft, 0.0, 1.0).astype(np.float32)
        body_base = np.clip(subj_soft * (1.0 - 0.78 * face) * (1.0 - 0.24 * hair), 0.0, 1.0)
        body = feather_mask(body_base, passes=3)
        cloth = feather_mask(np.clip(clothing_mask * subj_soft, 0.0, 1.0), passes=3) if clothing_mask is not None else np.clip(body * 0.55, 0.0, 1.0)
        skin_proxy_compact = self._estimate_skin_proxy(
            np.clip(source_linear, 0.0, 4.0),
            subj,
            np.clip(face_core, 0.0, 1.0),
            np.clip(hair_region, 0.0, 1.0),
            np.clip(edge_band, 0.0, 1.0),
        )
        if np.any(subj_bool):
            ys, xs = np.where(subj_bool)
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            x_rel = np.clip((xx - x0) / max(x1 - x0, 1.0), 0.0, 1.0)
            y_rel = np.clip((yy - y0) / max(y1 - y0, 1.0), 0.0, 1.0)
        else:
            x_rel = u
            y_rel = v
        face_center = np.exp(-0.5 * (((x_rel - 0.50) / 0.30) ** 2 + ((y_rel - 0.42) / 0.34) ** 2)).astype(np.float32)
        face_core_clean = feather_mask(np.clip(face * face_center, 0.0, 1.0), passes=4)
        face_side = feather_mask(np.clip(face * (1.0 - 0.48 * face_center), 0.0, 1.0), passes=4)
        shoulder = feather_mask(np.clip(body * np.clip((y_rel - 0.36) / 0.34, 0.0, 1.0) * np.clip(1.0 - np.abs(x_rel - 0.5) / 0.62, 0.0, 1.0), 0.0, 1.0), passes=3)
        neck_skin = feather_mask(
            np.clip(
                subj_soft
                * skin_proxy_compact
                * np.clip((y_rel - 0.38) / 0.34, 0.0, 1.0)
                * np.clip(1.0 - np.abs(x_rel - 0.5) / 0.48, 0.0, 1.0)
                * (1.0 - 0.70 * hair)
                * (1.0 - 0.42 * edge),
                0.0,
                1.0,
            ),
            passes=3,
        ).astype(np.float32)
        body_skin_region = feather_mask(
            np.clip((body + 0.78 * shoulder + 0.62 * neck_skin) * skin_proxy_compact * (1.0 - 0.62 * cloth), 0.0, 1.0),
            passes=3,
        ).astype(np.float32)
        non_skin_body_region = np.clip(body * (1.0 - 0.72 * body_skin_region), 0.0, 1.0).astype(np.float32)
        lower_body_prior = np.clip((y_rel - 0.48) / 0.42, 0.0, 1.0).astype(np.float32)
        side_body_prior = np.clip((np.abs(x_rel - 0.50) - 0.16) / 0.34, 0.0, 1.0).astype(np.float32)
        hand_proxy = feather_mask(
            np.clip(
                body
                * skin_proxy_compact
                * (1.0 - 0.70 * hair)
                * (1.0 - 0.64 * face_core_clean)
                * (0.30 + 0.45 * lower_body_prior + 0.42 * side_body_prior + 0.20 * edge),
                0.0,
                1.0,
            ).astype(np.float32),
            passes=3,
        )
        upper_body_light_region = feather_mask(
            np.clip(
                0.62 * body
                + 0.78 * shoulder
                + 0.62 * neck_skin
                + 0.60 * body_skin_region
                + 0.58 * hand_proxy
                + 0.48 * cloth
                - 0.50 * face_core_clean
                - 0.18 * hair,
                0.0,
                1.0,
            ).astype(np.float32),
            passes=2,
        )
        skin_edge_region = feather_mask(
            np.clip(edge * skin_proxy_compact * (0.72 + 0.28 * shoulder + 0.18 * neck_skin), 0.0, 1.0),
            passes=2,
        ).astype(np.float32)
        air_skin_region = feather_mask(
            np.clip(
                1.20 * neck_skin
                + 0.92 * body_skin_region
                + 0.64 * shoulder * (1.0 - 0.78 * cloth) * (1.0 - 0.40 * hair)
                + 0.50 * skin_edge_region
                + 0.22 * face_side * (1.0 - 0.72 * face_core_clean),
                0.0,
                1.0,
            ).astype(np.float32),
            passes=2,
        )
        face_read_mask = face_core_clean > 0.12
        if not np.any(face_read_mask):
            face_read_mask = face > 0.12
        face_read_gate = face_read_mask.astype(np.float32)

        edge_rim_term = np.clip(0.82 * edge + 0.34 * hair + 0.18 * shoulder, 0.0, 1.0)
        nw = (0.34 * (1.0 - flat_normal) + 0.12 * flat_normal) * (1.0 - 0.42 * bg_bias_strength)
        sw = 0.46 + 0.34 * flat_normal + 0.22 * bg_bias_strength
        ew = 0.14
        direction_field = (nw * normal_term + sw * screen_ramp_term + ew * edge_rim_term) / max(nw + sw + ew, 1e-5)
        direction_softness = float(np.clip(
            direction.get('direction_softness', 0.32 + 0.28 * bg_haze + 0.22 * bg_flatness + 0.18 * chroma_pressure)
            + 0.16 * flat_normal
            + 0.10 * chroma_pressure
            + 0.08 * bg_haze,
            0.28,
            0.88,
        ))
        direction_softness = float(np.clip(direction_softness - 0.15 * lowkey_chroma_gate, 0.20, 0.88))
        direction_field = box_blur_gray(np.clip(direction_field, 0.0, 1.0).astype(np.float32), passes=3 + int(round(direction_softness * 3.0)))

        lit_side = smoothstep(0.42 - 0.18 * direction_softness, 0.66 + 0.18 * direction_softness, direction_field) * subj
        shadow_side = (1.0 - smoothstep(0.28 - 0.12 * direction_softness, 0.66 + 0.24 * direction_softness, direction_field)) * subj
        lit_side = box_blur_gray(lit_side.astype(np.float32), passes=1)
        shadow_side = box_blur_gray(shadow_side.astype(np.float32), passes=1)
        rim_side = np.clip(edge_rim_term * (0.42 + 0.58 * shadow_side), 0.0, 1.0).astype(np.float32)
        hair_rim = feather_mask(
            np.clip((0.70 * edge + 0.58 * hair * rim_side + 0.18 * hair * lit_side) * subj_soft, 0.0, 1.0),
            passes=2,
        ).astype(np.float32)
        hair_core = feather_mask(
            np.clip(hair * (1.0 - 0.58 * edge) * (1.0 - 0.46 * rim_side), 0.0, 1.0),
            passes=1,
        ).astype(np.float32)

        direct_w = float(np.clip(render_weight.get('direct_weight', 1.0), 0.0, 1.45))
        shadow_w = float(np.clip(render_weight.get('shadow_weight', 0.75), 0.0, 1.35))
        rim_w = float(np.clip(render_weight.get('rim_weight', 0.90), 0.0, 1.55))
        ambient_w = float(np.clip(render_weight.get('ambient_weight', 0.85), 0.0, 1.20))
        fill_w = float(np.clip(render_weight.get('fill_weight', render_weight.get('body_fill_weight', 0.78)), 0.0, 1.25))
        body_fill_w = float(np.clip(render_weight.get('body_fill_weight', fill_w), 0.0, 1.25))
        spill_w = float(np.clip(render_weight.get('color_spill_weight', 0.55), 0.0, 1.25))
        display_w = float(np.clip(render_weight.get('display_weight', 0.70), 0.0, 1.0))

        direct_strength = float(np.clip(direction.get('direction_strength', direction.get('directional_light_strength', direction.get('key_strength', 0.38))), 0.10, 0.90))
        shadow_strength = float(np.clip(direction.get('shadow_strength', 0.22), 0.04, 0.70))
        rim_strength = float(np.clip(direction.get('rim_strength', 0.28), 0.06, 0.90))
        side_sep = float(np.clip(direction.get('side_separation', 0.30), 0.08, 0.85))
        direction_magnitude = float(np.clip(direction.get('direction_magnitude', direct_strength), 0.10, 0.95))
        direction_contrast = float(np.clip(direction.get('direction_contrast', shadow_strength), 0.06, 0.85))
        shadow_floor = float(np.clip(exposure.get('shadow_luma_floor', 0.045 + 0.035 * ambient_w), 0.035, 0.18))
        face_readability = float(np.clip(exposure.get('face_readability', 0.48), 0.25, 0.85))

        # Pose-aware strength tuning without discrete style routing:
        # profile view can accept stronger side/rim structure; frontal view keeps
        # softer shadow while preserving stable left/right volume.
        if np.any(face > 0.10):
            yaw_proxy = float(np.mean(np.abs(N[..., 0][face > 0.10])))
        else:
            yaw_proxy = float(np.mean(np.abs(N[..., 0][subj_bool]))) if np.any(subj_bool) else 0.22
        profile_factor = float(np.clip((yaw_proxy - 0.16) / 0.32, 0.0, 1.0))
        frontal_factor = 1.0 - profile_factor
        direct_strength *= float(np.clip(0.82 + 0.30 * direction_magnitude + 0.16 * profile_factor + 0.06 * (1.0 - flat_normal) + 0.10 * dark_chroma_env + 0.06 * warm_direction_env + 0.16 * lowkey_chroma_gate, 0.86, 1.46))
        shadow_strength *= float(np.clip(0.42 + 0.46 * direction_contrast + 0.12 * profile_factor - 0.20 * frontal_factor * face_readability - 0.22 * direction_softness - 0.18 * lowkey_chroma_gate, 0.24, 0.96))
        rim_strength *= float(np.clip(0.84 + 0.28 * direction_magnitude + 0.22 * profile_factor + 0.18 * lowkey_chroma_gate, 0.86, 1.50))
        side_sep = float(np.clip(side_sep + 0.08 * direction_magnitude + 0.05 * bg_bias_strength + 0.12 * lowkey_chroma_gate, 0.08, 0.98))

        face_core_luma_weight = float(np.clip(0.14 - 0.03 * warm_direction_env - 0.030 * lowkey_chroma_gate, 0.08, 0.18))
        face_side_luma_weight = float(np.clip(region.get('face_side_weight', 0.52) + 0.05 * dark_chroma_env + 0.04 * warm_direction_env + 0.10 * lowkey_chroma_gate, 0.48, 0.82))
        body_luma_region_w = float(np.clip(region.get('body_side_weight', region.get('body_weight', 0.70)), 0.30, 1.05))
        shoulder_region_w = float(np.clip(region.get('shoulder_weight', 0.72), 0.30, 1.10))
        neck_region_w = float(np.clip(region.get('jaw_neck_weight', 0.72), 0.34, 1.15))
        hand_region_w = float(np.clip(region.get('hand_weight', body_luma_region_w), 0.34, 1.16))
        cloth_region_w = float(np.clip(region.get('cloth_weight', region.get('clothing_weight', 0.76)), 0.28, 1.10))
        hair_region_w = float(np.clip(region.get('hair_weight', 0.74), 0.34, 1.15))
        edge_region_w = float(np.clip(region.get('edge_weight', 0.84), 0.36, 1.20))
        edge_chroma_weight = 0.70
        luma_weight = np.clip(
            face_core_clean * face_core_luma_weight
            + face_side * face_side_luma_weight
            + body * (0.56 + 0.54 * body_luma_region_w + 0.18 * direction_magnitude + 0.16 * dark_chroma_env + 0.12 * warm_direction_env + 0.18 * lowkey_chroma_gate)
            + neck_skin * (0.54 + 0.54 * neck_region_w + 0.18 * direction_magnitude + 0.12 * ambient_w + 0.16 * lowkey_chroma_gate)
            + hand_proxy * (0.58 + 0.56 * hand_region_w + 0.18 * direction_magnitude + 0.16 * ambient_w + 0.16 * lowkey_chroma_gate)
            + shoulder * (0.58 + 0.60 * shoulder_region_w + 0.20 * direction_magnitude + 0.18 * dark_chroma_env + 0.16 * warm_direction_env + 0.22 * lowkey_chroma_gate)
            + cloth * (0.50 + 0.54 * cloth_region_w + 0.18 * direction_magnitude + 0.12 * dark_chroma_env + 0.08 * warm_direction_env + 0.16 * lowkey_chroma_gate)
            + hair_core * (0.28 + 0.34 * hair_region_w + 0.10 * direction_magnitude + 0.06 * dark_chroma_env)
            + hair_rim * (0.46 + 0.52 * hair_region_w + 0.16 * direction_magnitude + 0.12 * dark_chroma_env + 0.16 * lowkey_chroma_gate)
            + edge * (0.40 + 0.56 * edge_region_w + 0.18 * direction_magnitude + 0.10 * dark_chroma_env + 0.20 * lowkey_chroma_gate),
            0.0,
            1.35,
        ).astype(np.float32)
        base_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)

        # Readable range anchor: dark scenes can stay lower, bright/hazy scenes
        # cannot lift the face into a flat white plate.
        face_low = float(np.clip(exposure.get('face_target_luma_min', 0.30), 0.220, 0.460))
        face_high = float(np.clip(exposure.get('face_target_luma_max', 0.42), max(face_low + 0.055, 0.320), 0.580))
        target_face_luma = float(np.clip(0.5 * (face_low + face_high), 0.270, 0.520))
        if np.any(face_read_mask):
            cur_face_luma = float(np.percentile(base_l[face_read_mask], 60.0))
        else:
            cur_face_luma = float(np.percentile(base_l[subj_bool], 65.0)) if np.any(subj_bool) else target_face_luma
        if cur_face_luma < face_low:
            exposure_gain = float(np.clip(face_low / max(cur_face_luma, 1e-4), 1.0, 1.20))
        elif cur_face_luma > face_high:
            exposure_gain = float(np.clip(face_high / max(cur_face_luma, 1e-4), 0.90, 1.0))
        else:
            exposure_gain = 1.0
        if exposure_gain > 1.0:
            gain_map = np.clip(
                face_core_clean * (0.24 + 0.14 * face_readability)
                + face_side * 0.24
                + upper_body_light_region * (0.36 + 0.10 * body_fill_w)
                + neck_skin * 0.24
                + hand_proxy * 0.30
                + shoulder * 0.24,
                0.0,
                1.0,
            ).astype(np.float32)
            out *= (1.0 + (exposure_gain - 1.0) * gain_map)[..., None]
            base_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        elif exposure_gain < 1.0:
            reduce_map = np.clip(face_core_clean * 0.32 + face_side * 0.18 + body * 0.08, 0.0, 1.0).astype(np.float32)
            out *= (1.0 - (1.0 - exposure_gain) * reduce_map)[..., None]
            base_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)

        body_sync = np.clip(0.62 * body + 0.84 * shoulder + 0.58 * neck_skin + 0.62 * hand_proxy + 0.58 * cloth + 0.12 * hair_core + 0.25 * hair_rim + 0.20 * edge, 0.0, 1.0)
        lift = lit_side * luma_weight * direct_w * direct_strength * (0.030 + 0.056 * side_sep + 0.036 * bg_bias_strength + 0.026 * direction_magnitude + 0.026 * lowkey_chroma_gate)
        lift += lit_side * body_sync * direct_w * direct_strength * (0.020 + 0.026 * bg_bias_strength + 0.022 * direction_magnitude + 0.014 * dark_chroma_env + 0.008 * warm_direction_env + 0.030 * lowkey_chroma_gate)
        fill_coverage = np.clip(0.42 + 0.30 * shadow_side + 0.18 * (1.0 - lit_side) + 0.10 * bg_haze, 0.0, 1.0)
        lift += upper_body_light_region * fill_coverage * fill_w * body_fill_w * (0.014 + 0.018 * ambient_w + 0.012 * dark_chroma_env + 0.016 * lowkey_chroma_gate)
        lift += (neck_skin * 0.72 + hand_proxy * 0.82 + shoulder * 0.68) * fill_coverage * fill_w * (0.010 + 0.012 * ambient_w + 0.010 * lowkey_chroma_gate)
        volume_lift = lit_side * direct_w * direct_strength * direction_magnitude * np.clip(
            face_side * (0.030 + 0.014 * lowkey_chroma_gate)
            + body * (0.064 + 0.026 * lowkey_chroma_gate)
            + neck_skin * (0.060 + 0.026 * lowkey_chroma_gate)
            + hand_proxy * (0.066 + 0.028 * lowkey_chroma_gate)
            + shoulder * (0.090 + 0.036 * lowkey_chroma_gate)
            + cloth * (0.074 + 0.030 * lowkey_chroma_gate)
            + hair_rim * (0.048 + 0.024 * lowkey_chroma_gate)
            + edge * (0.058 + 0.030 * lowkey_chroma_gate)
            + hair_core * 0.014
            - face_core_clean * (0.040 + 0.020 * lowkey_chroma_gate),
            0.0,
            0.125,
        )
        lift += volume_lift
        dark = shadow_side * luma_weight * shadow_w * shadow_strength * (0.008 + 0.014 * side_sep + 0.012 * bg_bias_strength + 0.010 * direction_contrast)
        dark += shadow_side * body_sync * shadow_w * shadow_strength * (0.004 + 0.006 * bg_bias_strength)
        dark *= np.clip(1.0 - face_core_clean * 0.90 - face_side * (0.52 + 0.18 * lowkey_chroma_gate) - hair * 0.46 - upper_body_light_region * (0.22 + 0.10 * ambient_w) - cloth * (0.30 + 0.12 * lowkey_chroma_gate) - shoulder * (0.32 + 0.16 * lowkey_chroma_gate) - body * (0.12 + 0.08 * lowkey_chroma_gate), 0.10, 1.0)
        rim_luma = rim_side * rim_w * rim_strength * (0.040 + 0.032 * spill_w)
        rim_luma += rim_side * rim_w * rim_strength * (
            hair_rim * (0.020 + 0.012 * chroma_pressure + 0.024 * lowkey_chroma_gate)
            + edge * (0.026 + 0.018 * chroma_pressure + 0.030 * lowkey_chroma_gate)
            + shoulder * (0.016 + 0.014 * warm_direction_env + 0.020 * lowkey_chroma_gate)
            + neck_skin * (0.010 + 0.010 * warm_direction_env + 0.014 * lowkey_chroma_gate)
            + hand_proxy * (0.012 + 0.010 * warm_direction_env + 0.014 * lowkey_chroma_gate)
            + cloth * (0.010 + 0.010 * warm_direction_env + 0.016 * lowkey_chroma_gate)
        )
        new_l = np.clip(base_l + lift - dark + rim_luma, shadow_floor * subj + base_l * (1.0 - subj), 1.0)
        out *= np.clip(new_l / base_l, 0.84, 1.24)[..., None]
        # Keep face readable after directional shadow: protect luma only, not hue.
        out_l_read = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        face_core_floor = float(np.clip(face_low * (0.90 + 0.04 * face_readability), 0.200, 0.440))
        face_side_floor = float(np.clip(face_core_floor * 0.86, 0.180, 0.390))
        core_floor_l = np.maximum(out_l_read, face_core_floor)
        side_floor_l = np.maximum(out_l_read, face_side_floor)
        core_gate_floor = np.clip(face_core_clean * 0.42, 0.0, 0.46)
        floor_l = out_l_read * (1.0 - core_gate_floor) + core_floor_l * core_gate_floor
        side_gate_floor = np.clip(face_side * (1.0 - 0.55 * face_core_clean) * 0.32, 0.0, 0.34)
        floor_l = floor_l * (1.0 - side_gate_floor) + np.maximum(floor_l, side_floor_l) * side_gate_floor
        floor_ratio_cap = np.clip(1.06 + 0.12 * core_gate_floor + 0.06 * face_side, 1.06, 1.22)
        out *= np.clip(floor_l / out_l_read, 1.0, floor_ratio_cap)[..., None]
        hair_l_now = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        src_l_for_hair = rgb_luminance(np.clip(source_linear, 0.0, None)).astype(np.float32)
        hair_floor = np.maximum(
            shadow_floor * (0.88 + 0.20 * ambient_w),
            src_l_for_hair * (0.46 + 0.12 * bg_haze),
        ).astype(np.float32)
        hair_floor_gate = np.clip((0.30 * hair_core + 0.54 * hair_rim + 0.34 * edge + 0.18 * shoulder) * subj, 0.0, 1.0)
        hair_floor_l = hair_l_now * (1.0 - hair_floor_gate) + np.maximum(hair_l_now, hair_floor) * hair_floor_gate
        out *= np.clip(hair_floor_l / hair_l_now, 1.0, 1.18)[..., None]
        material_l_now = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        src_l_material = rgb_luminance(np.clip(source_linear, 0.0, None)).astype(np.float32)
        body_face_gap = float(np.clip(exposure.get('body_face_luma_gap_target', 0.085), 0.045, 0.135))
        body_target_luma = float(np.clip(max(exposure.get('body_target_luma', max(face_low - 0.030, 0.28)), face_low - body_face_gap * 0.72), 0.27, 0.50))
        body_material_region = np.clip(
            body * 0.44
            + shoulder * 0.72
            + neck_skin * 0.54
            + hand_proxy * 0.66
            + body_skin_region * 0.42
            + cloth * 0.86
            - face_core_clean * 0.44,
            0.0,
            1.0,
        ).astype(np.float32)
        hair_material_region = np.clip(0.34 * hair_core + 0.46 * hair_rim + 0.36 * edge + shoulder * 0.10 - face_core_clean * 0.28, 0.0, 1.0).astype(np.float32)
        material_side_gate = np.clip(0.44 + 0.38 * lit_side + 0.24 * rim_side + 0.16 * fill_coverage + 0.12 * bg_bias_strength, 0.0, 1.0)
        body_material_target = np.maximum(
            src_l_material * (0.66 + 0.10 * warm_direction_env + 0.06 * dark_chroma_env + 0.08 * bg_haze),
            body_target_luma * (0.76 + 0.16 * lit_side + 0.10 * rim_side + 0.10 * fill_coverage + 0.08 * warm_direction_env),
        ).astype(np.float32)
        hair_material_target = np.maximum(
            src_l_material * (0.58 + 0.08 * warm_direction_env + 0.08 * dark_chroma_env + 0.05 * bg_haze),
            shadow_floor * (1.30 + 0.34 * dark_chroma_env + 0.20 * warm_direction_env) + body_target_luma * (0.12 + 0.10 * rim_side),
        ).astype(np.float32)
        body_material_lift = np.clip((body_material_target - material_l_now) * body_material_region * material_side_gate, 0.0, 0.115)
        hair_material_lift = np.clip((hair_material_target - material_l_now) * hair_material_region * np.clip(0.28 + 0.22 * lit_side + 0.40 * rim_side, 0.0, 1.0), 0.0, 0.055)
        material_lift = np.maximum(body_material_lift, hair_material_lift)
        out *= np.clip((material_l_now + material_lift) / material_l_now, 1.0, 1.20)[..., None]
        sync_l_now = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        subject_sync = float(np.clip(exposure.get('whole_subject_sync_strength', 0.42), 0.20, 0.70))
        body_sync_region = feather_mask(
            np.clip(
                body_material_region * 0.78
                + shoulder * 0.64
                + cloth * 0.50
                + neck_skin * 0.58
                + hand_proxy * 0.66
                + skin_edge_region * 0.20
                - face_core_clean * 0.56
                - hair_core * 0.18,
                0.0,
                0.94,
            ).astype(np.float32),
            passes=3,
        )
        body_sync_mask = body_sync_region > 0.10
        if np.any(face_read_mask) and np.count_nonzero(body_sync_mask) > 80:
            compact_face_p70 = float(np.percentile(sync_l_now[face_read_mask], 70.0))
            compact_body_p70 = float(np.percentile(sync_l_now[body_sync_mask], 70.0))
            compact_target_body = float(np.clip(compact_face_p70 - body_face_gap, body_target_luma * 0.86, max(body_target_luma * 1.12, compact_face_p70 - 0.065)))
            compact_lift = float(np.clip((compact_target_body - compact_body_p70) * subject_sync, 0.0, 0.055))
            if compact_lift > 1e-5:
                compact_sync_gate = np.clip(
                    body_sync_region
                    * (0.42 + 0.30 * lit_side + 0.14 * fill_coverage + 0.18 * rim_side + 0.08 * lowkey_chroma_gate)
                    * (1.0 - 0.62 * face_core_clean),
                    0.0,
                    0.72,
                ).astype(np.float32)
                synced_l = np.clip(sync_l_now + compact_lift * compact_sync_gate, 0.0, 1.0)
                out *= np.clip(synced_l / sync_l_now, 1.0, 1.10)[..., None]
        out_l_cap = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        highlight_ceiling = float(np.clip(exposure.get('highlight_luma_ceiling', 0.82), 0.58, 0.90))
        cap_gate = np.clip(face_core_clean * 0.72 + face_side * 0.22, 0.0, 1.0)
        capped_l = np.minimum(out_l_cap, highlight_ceiling)
        cap_ratio = capped_l / out_l_cap
        out *= (1.0 - cap_gate[..., None]) + np.clip(cap_ratio, 0.82, 1.0)[..., None] * cap_gate[..., None]

        try:
            key_color = np.array(getattr(lighting_info, 'key_color', (0.6, 0.6, 0.6)), dtype=np.float32)
            ambient_color = np.array(getattr(lighting_info, 'ambient_color', key_color), dtype=np.float32)
            global_color = np.array(getattr(lighting_info, 'global_mean_color', ambient_color), dtype=np.float32)
        except Exception:
            key_color = ambient_color = global_color = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        if field:
            side_color = np.array(field.get('right_color' if color_key_uv[0] >= 0.5 else 'left_color', key_color), dtype=np.float32)
            rim_color = np.array(field.get('left_color' if color_key_uv[0] >= 0.5 else 'right_color', ambient_color), dtype=np.float32)
        else:
            side_color = key_color
            rim_color = ambient_color

        def _dir_color(c: np.ndarray) -> np.ndarray:
            c = np.clip(c.astype(np.float32), 1e-5, 6.0)
            return np.clip(c / max(float(np.dot(c, LUMA)), 1e-5), 0.35, 2.65).astype(np.float32)

        soft_bg_carrier = float(np.clip(
            0.55 * bg_haze
            + 0.45 * np.clip((0.34 - bg_colorfulness) / 0.34, 0.0, 1.0),
            0.0,
            1.0,
        ))
        lit_mix_color = (0.55 * key_color + 0.30 * side_color + 0.15 * ambient_color) * (1.0 - 0.12 * soft_bg_carrier) + global_color * (0.12 * soft_bg_carrier)
        rim_mix_color = (
            rim_color * (0.55 - 0.30 * soft_bg_carrier)
            + global_color * (0.25 + 0.40 * soft_bg_carrier)
            + ambient_color * (0.20 - 0.10 * soft_bg_carrier)
        )
        lit_dir = _dir_color(lit_mix_color)
        rim_dir = _dir_color(rim_mix_color)
        shadow_mix_color = 0.50 * ambient_color + 0.34 * global_color + 0.16 * rim_color
        shadow_dir = _dir_color(shadow_mix_color)
        skin_limit = float(np.clip(chroma.get('skin_tint_limit', 0.08), 0.025, 0.16))
        palette_sep = float(np.clip(chroma.get('palette_separation', 0.25), 0.0, 0.9))
        face_core_chroma_weight = float(np.clip(chroma.get('face_core_chroma_limit', min(0.006, skin_limit * 0.08)) * (1.0 - 0.55 * lowkey_chroma_gate), 0.0, 0.006))
        face_side_chroma_weight = float(np.clip(chroma.get('face_side_chroma_allowance', min(0.028, skin_limit * (0.22 + 0.08 * (1.0 - bg_haze)))) * (1.0 + 0.18 * lowkey_chroma_gate + 0.10 * palette_sep), 0.004, 0.052))
        body_chroma_weight = float(np.clip(chroma.get('body_chroma_allowance', 0.08 + 0.06 * chroma_pressure) + 0.018 * lowkey_chroma_gate, 0.04, 0.27))
        cloth_chroma_weight = float(np.clip(chroma.get('clothing_chroma_budget', chroma.get('clothing_chroma_allowance', 0.18 + 0.16 * chroma_pressure)) + 0.040 * lowkey_chroma_gate, 0.08, 0.62))
        hair_chroma_weight = float(np.clip(chroma.get('hair_chroma_allowance', 0.42 + 0.24 * chroma_pressure) + 0.028 * lowkey_chroma_gate, 0.25, 0.82))
        edge_chroma_weight = float(np.clip(chroma.get('edge_chroma_allowance', 0.54 + 0.26 * chroma_pressure) + 0.038 * lowkey_chroma_gate, 0.35, 0.92))
        body_skin_chroma_weight = float(np.clip(body_chroma_weight * (1.0 - 1.18 * air_skin_guard), 0.010, body_chroma_weight))
        face_side_chroma_weight *= float(np.clip(1.0 - 0.62 * air_skin_guard, 0.38, 1.0))
        skin_edge_chroma_weight = float(np.clip(edge_chroma_weight * (1.0 - 0.92 * air_skin_guard), 0.05, edge_chroma_weight))
        hair_core_chroma_factor = float(np.clip(0.24 - 0.10 * lowkey_chroma_gate, 0.12, 0.24))
        chroma_weight = np.clip(
            face_core_clean * face_core_chroma_weight * np.clip(1.0 - 0.80 * chroma_pressure, 0.12, 0.55)
            + face_side * face_side_chroma_weight * np.clip(1.0 - 0.36 * face_core_clean, 0.34, 1.0)
            + body_skin_region * body_skin_chroma_weight
            + non_skin_body_region * body_chroma_weight
            + shoulder * min(body_skin_chroma_weight * 1.04, 0.30)
            + cloth * cloth_chroma_weight
            + hair_core * hair_chroma_weight * hair_core_chroma_factor
            + hair_rim * hair_chroma_weight
            + np.clip(edge - skin_edge_region, 0.0, 1.0) * edge_chroma_weight
            + skin_edge_region * skin_edge_chroma_weight,
            0.0,
            0.85,
        ).astype(np.float32)
        style_side = np.clip(np.maximum(lit_side, rim_side) + 0.42 * shadow_side * (0.35 + 0.65 * palette_sep), 0.0, 1.0)
        spill_mask = np.clip(chroma_weight * spill_w * (0.34 + 0.66 * style_side + 0.20 * lowkey_chroma_gate * np.maximum(rim_side, shadow_side)), 0.0, 0.70)
        spill_mask = np.clip(
            spill_mask
            + soft_bg_carrier * spill_w * np.clip(0.04 * hair_core + 0.18 * hair_rim + 0.20 * edge + 0.08 * shoulder + 0.06 * cloth, 0.0, 0.24)
            + lowkey_chroma_gate * spill_w * np.clip(0.04 * hair_core + 0.16 * hair_rim + 0.18 * edge + 0.08 * shoulder + 0.07 * cloth + 0.04 * face_side, 0.0, 0.26),
            0.0,
            0.76,
        )
        carrier_spill = spill_w * np.clip(
            (0.045 + 0.125 * chroma_pressure + 0.060 * warm_direction_env + 0.060 * dark_chroma_env + 0.070 * lowkey_chroma_gate)
            * (
                0.16 * body_skin_region * (1.0 - 0.76 * air_skin_guard)
                + 0.34 * non_skin_body_region
                + 0.54 * shoulder * (1.0 - 0.55 * air_skin_guard)
                + 0.96 * cloth
                + 0.16 * hair_core
                + 1.08 * hair_rim
                + 1.22 * np.clip(edge - skin_edge_region, 0.0, 1.0)
                + 0.34 * skin_edge_region * (1.0 - 0.70 * air_skin_guard)
                + 0.16 * face_side * (1.0 - 0.65 * face_core_clean)
            )
            * (0.38 + 0.46 * np.maximum(lit_side, rim_side) + 0.28 * shadow_side),
            0.0,
            0.40,
        ).astype(np.float32)
        spill_mask = np.clip(spill_mask + carrier_spill * np.clip(1.0 - 0.92 * face_core_clean - 0.48 * face_side, 0.06, 1.0), 0.0, 0.78)
        spill_mask *= np.clip(1.0 - (0.98 + 0.02 * lowkey_chroma_gate) * face_core_clean - (0.52 + 0.06 * lowkey_chroma_gate) * face_side, 0.035, 1.0)
        out_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        out_dir = np.clip(out / out_l[..., None], 0.25, 3.0)
        rim_mix = rim_side * palette_sep
        shadow_mix = np.clip(shadow_side * (0.16 + 0.18 * palette_sep + 0.12 * lowkey_chroma_gate), 0.0, 0.34)
        lit_mix = np.clip(1.0 - rim_mix - shadow_mix, 0.0, 1.0)
        target_dir = np.clip(
            lit_dir.reshape(1, 1, 3) * lit_mix[..., None]
            + rim_dir.reshape(1, 1, 3) * rim_mix[..., None]
            + shadow_dir.reshape(1, 1, 3) * shadow_mix[..., None],
            0.25,
            3.0,
        )
        out_dir = out_dir * (1.0 - spill_mask[..., None]) + target_dir * spill_mask[..., None]
        out = np.clip(out_l[..., None] * out_dir, 0.0, 8.0)

        protected_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        src_l = np.maximum(rgb_luminance(np.clip(source_linear, 0.0, None)), 1e-5)
        src_dir = np.clip(source_linear / src_l[..., None], 0.55, 1.85)
        warm_support = float(np.clip(
            float(descriptor.get('warm_presence', descriptor.get('warm_ratio', 0.0)))
            * max(bg_sat, bg_colorfulness)
            * 2.4,
            0.0,
            1.0,
        ))
        cool_support = float(np.clip(
            float(descriptor.get('cool_presence', descriptor.get('cool_ratio', 0.0)))
            * max(bg_sat, bg_colorfulness)
            * 2.4,
            0.0,
            1.0,
        ))
        ambient_warmth = float(np.clip(descriptor.get('ambient_warmth', 0.0), -1.0, 1.0))
        air_color = desaturate_color(
            0.42 * global_color + 0.36 * ambient_color + 0.22 * key_color,
            0.48 + 0.30 * air_skin_guard,
        )
        air_color = np.clip(
            air_color
            + np.array([
                0.030 * warm_support - 0.034 * cool_support + 0.024 * max(ambient_warmth, 0.0),
                0.004 + 0.008 * cool_support,
                0.034 * cool_support - 0.020 * warm_support + 0.026 * max(-ambient_warmth, 0.0),
            ], dtype=np.float32),
            1e-5,
            None,
        )
        air_bg_dir = _dir_color(air_color).reshape(1, 1, 3)
        src_chroma_keep = float(np.clip(1.0 - air_skin_guard * (0.86 + 0.30 * soft_bg_carrier), 0.18, 1.0))
        bg_air_mix = float(np.clip(air_skin_guard * (0.36 + 0.30 * soft_bg_carrier + 0.16 * cool_support - 0.08 * warm_support), 0.0, 0.55))
        air_skin_dir = 1.0 + (src_dir - 1.0) * src_chroma_keep
        air_skin_dir = air_skin_dir * (1.0 - bg_air_mix) + air_bg_dir * bg_air_mix
        if air_skin_guard > 1e-5:
            red_ref = 0.58 * air_skin_dir[..., 1] + 0.42 * air_skin_dir[..., 2]
            red_limit = 1.015 + 0.26 * warm_support + 0.06 * (1.0 - air_skin_guard)
            air_skin_dir[..., 0] = np.minimum(air_skin_dir[..., 0], red_ref * red_limit)
            air_l = np.maximum(np.sum(air_skin_dir * LUMA.reshape(1, 1, 3), axis=-1), 1e-5)
            air_skin_dir = np.clip(air_skin_dir / air_l[..., None], 0.48, 2.05).astype(np.float32)
        if air_skin_guard > 1e-5:
            skin_chroma_guard = feather_mask(
                np.clip(
                    air_skin_guard
                    * (
                        1.42 * neck_skin
                        + 1.18 * body_skin_region
                        + 0.86 * shoulder * (1.0 - 0.78 * cloth)
                        + 0.74 * skin_edge_region
                        + 0.24 * face_side * (1.0 - 0.70 * face_core_clean)
                    )
                    * (0.65 + 0.35 * np.maximum(lit_side, shadow_side)),
                    0.0,
                    0.82,
                ).astype(np.float32),
                passes=2,
            )
            out = out * (1.0 - skin_chroma_guard[..., None]) + (air_skin_dir * protected_l[..., None]) * skin_chroma_guard[..., None]
            protected_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        hair_chroma_guard = feather_mask(
            np.clip(hair_core * (0.10 + 0.16 * chroma_pressure) * (1.0 - 0.54 * rim_side) * (1.0 - 0.30 * shadow_side), 0.0, 0.24).astype(np.float32),
            passes=1,
        )
        out = out * (1.0 - hair_chroma_guard[..., None]) + (src_dir * protected_l[..., None]) * hair_chroma_guard[..., None]
        protected_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        core_protect = face_core_clean * (
            0.18
            + spill_mask
            + 0.62 * chroma_pressure
            + 0.08 * bg_lowkey
            + 0.04 * bg_highkey
            + 0.03 * bg_haze
            + 0.08 * lowkey_chroma_gate
        )
        side_protect = face_side * (spill_mask + 0.08 * chroma_pressure) * (0.56 - 0.18 * lowkey_chroma_gate + 0.12 * air_skin_guard) * np.clip(1.0 - 0.35 * shadow_side - 0.25 * rim_side, 0.42, 1.0)
        body_skin_protect = air_skin_guard * (0.64 * neck_skin + 0.46 * body_skin_region + 0.34 * skin_edge_region + 0.30 * shoulder * (1.0 - 0.78 * cloth))
        skin_protect = np.clip((core_protect + side_protect) * (0.58 + 1.05 * (0.16 - skin_limit) / 0.135), 0.0, 0.92)
        skin_protect = feather_mask(np.clip(skin_protect, 0.0, 0.92).astype(np.float32), passes=3)
        out = out * (1.0 - skin_protect[..., None]) + (src_dir * protected_l[..., None]) * skin_protect[..., None]
        if air_skin_guard > 1e-5:
            protected_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
            air_body_protect = feather_mask(np.clip(body_skin_protect, 0.0, 0.86).astype(np.float32), passes=3)
            out = out * (1.0 - air_body_protect[..., None]) + (air_skin_dir * protected_l[..., None]) * air_body_protect[..., None]
        final_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        sat_strength = float(np.clip((display.get('display_saturation', display.get('saturation', 1.0)) - 1.0) * 0.16 * display_w - 0.035 * bg_haze - 0.018 * chroma_pressure, -0.09, 0.050))
        out = final_l[..., None] + (out - final_l[..., None]) * (1.0 + sat_strength)
        src_l_mid = rgb_luminance(np.clip(source_linear, 0.0, None)).astype(np.float32)
        final_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        hair_structure_mask = feather_mask(
            np.clip(hair_core * (0.18 + 0.18 * chroma_pressure) * (1.0 - 0.32 * rim_side), 0.0, 0.32).astype(np.float32),
            passes=1,
        )
        hair_struct_bool = hair_structure_mask > 0.06
        if np.any(hair_struct_bool):
            cur_hair_ref = float(np.percentile(final_l[hair_struct_bool], 70.0))
            src_hair_ref = float(np.percentile(src_l_mid[hair_struct_bool], 70.0))
            hair_struct_scale = float(np.clip(cur_hair_ref / max(src_hair_ref, 1e-5), 0.62, 1.42))
            source_hair_l = np.clip(src_l_mid * hair_struct_scale, final_l * 0.70, final_l * 1.34).astype(np.float32)
            restored_hair_l = final_l * (1.0 - hair_structure_mask) + source_hair_l * hair_structure_mask
            out *= np.clip(restored_hair_l / final_l, 0.88, 1.12)[..., None]
            final_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        local_contrast = float(np.clip(display.get('display_contrast', display.get('local_contrast', 1.0)), 0.65, 1.25))
        detail_l = final_l - box_blur_gray(final_l, passes=3)
        lc_gate = np.clip(subj * (0.34 + 0.24 * face_core_clean + 0.42 * face_side + 0.32 * body + 0.48 * cloth + 0.58 * shoulder + 0.38 * hair_core + 0.34 * hair_rim + 0.62 * edge), 0.0, 1.0)
        texture_strength = float(np.clip(display.get('texture_preserve_strength', 0.10 + 0.14 * (local_contrast - 1.0) + 0.10 * bg_haze) * display_w, 0.05, 0.20))
        out *= np.clip((final_l + detail_l * texture_strength * lc_gate) / final_l, 0.94, 1.07)[..., None]
        final_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        src_mid = np.clip(box_blur_gray(src_l_mid, passes=1) - box_blur_gray(src_l_mid, passes=5), -0.030, 0.032)
        mid_gate = np.clip((0.26 * face_core_clean + 0.20 * face_side + 0.16 * body + 0.22 * cloth + 0.20 * shoulder + 0.46 * hair_core + 0.30 * hair_rim + 0.30 * edge) * (0.55 + 0.45 * face_readability), 0.0, 0.46)
        out *= np.clip((final_l + src_mid * texture_strength * mid_gate) / final_l, 0.94, 1.07)[..., None]
        final_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        src_fine_hair = np.clip(src_l_mid - box_blur_gray(src_l_mid, passes=1), -0.018, 0.018)
        hair_detail_gate = np.clip((0.56 * hair_core + 0.30 * hair_rim + 0.20 * edge + 0.12 * cloth + 0.08 * shoulder) * (0.34 + 0.42 * shadow_side + 0.18 * lit_side), 0.0, 0.46)
        out *= np.clip((final_l + src_fine_hair * hair_detail_gate) / final_l, 0.95, 1.06)[..., None]
        bloom_strength = float(np.clip(display.get('bloom', 0.0), 0.0, 1.4)) * 0.020 * display_w
        if bloom_strength > 1e-5:
            rim_glow = box_blur_rgb(out * rim_side[..., None], passes=2)
            out += rim_glow * bloom_strength
        out = np.clip(out * subj[..., None] + before * (1.0 - subj[..., None]), 0.0, 8.0).astype(np.float32)

        out_l_final = rgb_luminance(np.clip(out, 0.0, None))
        final_face_luma = float(np.percentile(out_l_final[face_read_mask], 60.0)) if np.any(face_read_mask) else 0.0
        if np.any(face_read_mask) and final_face_luma < face_low:
            read_gain = float(np.clip(face_low / max(final_face_luma, 1e-5), 1.0, 2.35))
            read_gate = feather_mask(
                np.clip(
                    subj_soft * 0.50
                    + face_core_clean * (0.18 + 0.16 * face_readability)
                    + face_side * 0.16
                    + body * 0.12
                    - edge * 0.16,
                    0.0,
                    0.86,
                ).astype(np.float32),
                passes=4,
            )
            out *= (1.0 + (read_gain - 1.0) * read_gate)[..., None]
            out = np.clip(out, 0.0, 8.0).astype(np.float32)
            out_l_final = rgb_luminance(np.clip(out, 0.0, None))
            final_face_luma = float(np.percentile(out_l_final[face_read_mask], 60.0))
        if abs(float(descriptor.get('horizontal_bias', hb))) + abs(float(descriptor.get('vertical_bias', vb))) > 1e-5:
            metric_region = np.clip(0.55 * subj_soft + 0.28 * body + 0.20 * face_side - 0.20 * edge, 0.0, 1.0).astype(np.float32)

            def _wm_luma(values: np.ndarray, weights: np.ndarray, default: float) -> float:
                ww = np.clip(weights.astype(np.float32), 0.0, None)
                ss = float(ww.sum())
                if ss <= 1e-6:
                    return float(default)
                return float((values * ww).sum() / ss)

            region_mean = _wm_luma(out_l_final, metric_region, float(np.mean(out_l_final[subj_bool])) if np.any(subj_bool) else 0.0)
            cur_lr = _wm_luma(out_l_final, metric_region * (u >= 0.5), region_mean) - _wm_luma(out_l_final, metric_region * (u < 0.5), region_mean)
            cur_tb = _wm_luma(out_l_final, metric_region * (v >= 0.5), region_mean) - _wm_luma(out_l_final, metric_region * (v < 0.5), region_mean)
            key_lr_bias = float(np.clip(float(key_uv[0]) - 0.5, -0.5, 0.5))
            key_tb_bias = float(np.clip(float(key_uv[1]) - 0.5, -0.5, 0.5))
            target_lr = float(np.clip(
                float(descriptor.get('horizontal_bias', hb)) * (0.48 + 0.16 * lowkey_chroma_gate)
                + key_lr_bias * 0.18 * float(np.clip(direction.get('local_light_confidence', 0.0), 0.0, 1.0)),
                -0.130,
                0.130,
            ))
            target_tb = float(np.clip(
                float(descriptor.get('vertical_bias', vb)) * (0.48 + 0.16 * lowkey_chroma_gate)
                + key_tb_bias * 0.18 * float(np.clip(direction.get('local_light_confidence', 0.0), 0.0, 1.0)),
                -0.130,
                0.130,
            ))
            delta_lr = float(np.clip(target_lr - cur_lr, -0.16, 0.16))
            delta_tb = float(np.clip(target_tb - cur_tb, -0.16, 0.16))
            align_region = feather_mask(
                np.clip(
                    subj_soft * 0.26
                    + face_side * 0.34
                    + body * 0.44
                    + cloth * 0.52
                    + hair * 0.42
                    + edge * 0.40
                    - face_core_clean * 0.30,
                    0.0,
                    0.82,
                ).astype(np.float32),
                passes=4,
            )
            align_delta = (2.0 * delta_lr * (u - 0.5) + 2.0 * delta_tb * (v - 0.5)) * align_region
            align_delta = np.clip(align_delta, -0.052, 0.090).astype(np.float32)
            aligned_l = np.clip(out_l_final + align_delta, shadow_floor * subj + out_l_final * (1.0 - subj), 1.0)
            out *= np.clip(aligned_l / np.maximum(out_l_final, 1e-5), 0.90, 1.24)[..., None]
            out = np.clip(out, 0.0, 8.0).astype(np.float32)
            out_l_final = rgb_luminance(np.clip(out, 0.0, None))
            final_face_luma = float(np.percentile(out_l_final[face_read_mask], 60.0)) if np.any(face_read_mask) else final_face_luma
        lit_m = lit_side > 0.45
        shadow_m = shadow_side > 0.45
        lit_mean_luma = float(np.mean(out_l_final[lit_m])) if np.any(lit_m) else 0.0
        shadow_mean_luma = float(np.mean(out_l_final[shadow_m])) if np.any(shadow_m) else 0.0
        target_delta = float(np.clip(
            0.018
            + 0.026 * direct_strength * direct_w
            + 0.018 * direction_magnitude
            + 0.010 * bg_bias_strength
            + 0.020 * lowkey_chroma_gate
            - 0.010 * direction_softness,
            0.018,
            0.086,
        ))
        if np.any(lit_m) and np.any(shadow_m) and (lit_mean_luma - shadow_mean_luma) < target_delta:
            gap = target_delta - (lit_mean_luma - shadow_mean_luma)
            rel_l_sep = np.maximum(out_l_final, 1e-5)
            sep_region = np.clip(
                face_side * 0.40
                + body * 0.62
                + shoulder * 0.82
                + cloth * 0.74
                + hair_rim * 0.58
                + edge * 0.64
                + hair_core * 0.18
                - face_core_clean * 0.30,
                0.0,
                1.0,
            ).astype(np.float32)
            sep_lit = lit_side * sep_region * gap * 1.10
            sep_shadow = shadow_side * sep_region * gap * 0.14
            sep = sep_lit - sep_shadow + rim_side * gap * 0.30
            new_sep_l = np.clip(rel_l_sep + sep, shadow_floor * subj + rel_l_sep * (1.0 - subj), 1.0)
            out *= np.clip(new_sep_l / rel_l_sep, 0.98, 1.16)[..., None]
            out = np.clip(out, 0.0, 8.0).astype(np.float32)
            out_l_final = rgb_luminance(np.clip(out, 0.0, None))
            lit_mean_luma = float(np.mean(out_l_final[lit_m])) if np.any(lit_m) else 0.0
            shadow_mean_luma = float(np.mean(out_l_final[shadow_m])) if np.any(shadow_m) else 0.0
        final_luma_for_chroma = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5)
        final_dir_for_chroma = np.clip(out / final_luma_for_chroma[..., None], 0.25, 3.0)
        source_luma_for_chroma = np.maximum(rgb_luminance(np.clip(source_linear, 0.0, None)), 1e-5)
        source_dir_for_chroma = np.clip(source_linear / source_luma_for_chroma[..., None], 0.45, 2.05)
        face_chroma_delta = np.linalg.norm(final_dir_for_chroma - source_dir_for_chroma, axis=-1).astype(np.float32)
        face_chroma_guard_region = np.clip(face_core_clean + 0.34 * face + 0.18 * face_side, 0.0, 1.0).astype(np.float32)
        adaptive_core_guard = feather_mask(
            np.clip(
                face_chroma_guard_region
                * np.clip((face_chroma_delta - (0.055 + 0.025 * face_readability)) / 0.22, 0.0, 1.0)
                * (0.50 + 0.58 * chroma_pressure + 0.24 * bg_lowkey),
                0.0,
                0.76,
            ).astype(np.float32),
            passes=2,
        )
        adaptive_side_guard = feather_mask(
            np.clip(
                face_side
                * np.clip((face_chroma_delta - 0.11) / 0.28, 0.0, 1.0)
                * (0.08 + 0.12 * chroma_pressure)
                * np.clip(1.0 - 0.40 * shadow_side - 0.28 * rim_side, 0.45, 1.0),
                0.0,
                0.12,
            ).astype(np.float32),
            passes=2,
        )
        adaptive_chroma_guard = np.clip(adaptive_core_guard + adaptive_side_guard, 0.0, 0.80).astype(np.float32)
        if np.any(adaptive_chroma_guard > 1e-4):
            repaired_dir = final_dir_for_chroma * (1.0 - adaptive_chroma_guard[..., None]) + source_dir_for_chroma * adaptive_chroma_guard[..., None]
            repaired_dir = np.clip(repaired_dir, 0.25, 3.0)
            repaired_dir_l = np.maximum(rgb_luminance(repaired_dir), 1e-5)
            repaired_dir = np.clip(repaired_dir / repaired_dir_l[..., None], 0.25, 3.0)
            out = np.clip(final_luma_for_chroma[..., None] * repaired_dir, 0.0, 8.0).astype(np.float32)
            out_l_final = rgb_luminance(np.clip(out, 0.0, None))
            lit_mean_luma = float(np.mean(out_l_final[lit_m])) if np.any(lit_m) else 0.0
            shadow_mean_luma = float(np.mean(out_l_final[shadow_m])) if np.any(shadow_m) else 0.0
        before_dir = np.clip(before / np.maximum(rgb_luminance(np.clip(before, 0.0, None)), 1e-5)[..., None], 0.25, 3.0)
        after_dir = np.clip(out / np.maximum(out_l_final, 1e-5)[..., None], 0.25, 3.0)
        fc_m = face_core_clean > 0.18
        fs_m = face_side > 0.12
        fc_shift = float(np.mean(np.linalg.norm(after_dir[fc_m] - before_dir[fc_m], axis=-1))) if np.any(fc_m) else 0.0
        fs_shift = float(np.mean(np.linalg.norm(after_dir[fs_m] - before_dir[fs_m], axis=-1))) if np.any(fs_m) else 0.0

        self._compact_direction_field = direction_field.astype(np.float32)
        self._compact_masks_rgb = np.dstack([lit_side, shadow_side, rim_side]).astype(np.float32)
        self._compact_before = before
        self._compact_after = out
        self._compact_delta = np.clip(np.abs(out - before).mean(axis=-1) * 8.0, 0.0, 1.0).astype(np.float32)
        self._compact_policy_runtime = {
            'key_uv': [float(key_uv[0]), float(key_uv[1])],
            'key_dir': [float(x) for x in key_dir],
            'direct_weight': direct_w,
            'shadow_weight': shadow_w,
            'rim_weight': rim_w,
            'ambient_weight': ambient_w,
            'fill_weight': fill_w,
            'body_fill_weight': body_fill_w,
            'color_spill_weight': spill_w,
            'display_weight': display_w,
            'face_core_luma_weight': face_core_luma_weight,
            'face_side_luma_weight': face_side_luma_weight,
            'face_core_chroma_weight': face_core_chroma_weight,
            'face_side_chroma_weight': face_side_chroma_weight,
            'edge_chroma_weight': edge_chroma_weight,
            'direction_softness': direction_softness,
            'direction_magnitude': direction_magnitude,
            'direction_contrast': direction_contrast,
            'target_face_luma': target_face_luma,
            'face_exposure_low': face_low,
            'face_exposure_high': face_high,
            'body_target_luma': body_target_luma,
            'body_face_luma_gap_target': body_face_gap,
        }
        self._compact_runtime_check = {
            'lit_mean_luma': lit_mean_luma,
            'shadow_mean_luma': shadow_mean_luma,
            'lit_shadow_delta': lit_mean_luma - shadow_mean_luma,
            'rim_mean': float(np.mean(rim_side[subj_bool])) if np.any(subj_bool) else 0.0,
            'face_core_chroma_shift': fc_shift,
            'face_side_chroma_shift': fs_shift,
            'face_luma_before_anchor': cur_face_luma,
            'face_luma_after_compact': final_face_luma,
            'face_luma_target_low': face_low,
            'face_luma_target_high': face_high,
            'auto_gain_after_direction': True,
        }
        print(
            f"[CompactLookPolicy] key_uv=({key_uv[0]:.2f},{key_uv[1]:.2f}) "
            f"direct={direct_w:.2f}, shadow={shadow_w:.2f}, rim={rim_w:.2f}, spill={spill_w:.2f}"
        )
        print(
            f"[DirectionField] lit={float(np.mean(lit_side[subj_bool])) if np.any(subj_bool) else 0.0:.2f}, "
            f"shadow={float(np.mean(shadow_side[subj_bool])) if np.any(subj_bool) else 0.0:.2f}, "
            f"delta={lit_mean_luma - shadow_mean_luma:.3f}, "
            f"rim={self._compact_runtime_check['rim_mean']:.2f}"
        )
        print(
            f"[RuntimeCheck] face_core_shift={fc_shift:.3f}, "
            f"face_side_shift={fs_shift:.3f}, "
            f"final_lit_shadow_delta={lit_mean_luma - shadow_mean_luma:.3f}"
        )
        return out
