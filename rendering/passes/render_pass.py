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
from rendering.passes.render_frame import prepare_render_frame

class RendererRenderMixin:
    def render_relight(
        self,
        source_linear: np.ndarray,
        mask: np.ndarray,
        albedo_linear: np.ndarray,
        normal_map: np.ndarray,
        depth_map: np.ndarray,
        specular_map: np.ndarray,
        roughness_map: np.ndarray,
        lighting_info: LightingInfo,
        camera_params: Optional[CameraParams] = None,
        depth_scale: Optional[float] = None,
        depth_bias: Optional[float] = None,
        background_linear: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._debug_intermediates = {}
        self._last_policy_runtime = {}
        frame = prepare_render_frame(
            self,
            source_linear,
            mask,
            albedo_linear,
            normal_map,
            depth_map,
            specular_map,
            roughness_map,
            camera_params,
            depth_scale,
            depth_bias,
        )
        source_lowfreq = frame['source_lowfreq']
        source_preserve_scale = frame['source_preserve_scale']
        detail = frame['detail']
        subject_mask = frame['subject_mask']
        edge_band = frame['edge_band']
        N = frame['N']
        P = frame['P']
        V = frame['V']
        NdotV = frame['NdotV']
        facing = frame['facing']
        face_core = frame['face_core']
        hair_region = frame['hair_region']
        base_subject = frame['base_subject']
        clothing_mask = frame['clothing_mask']
        spec_map = frame['spec_map']
        roughness = frame['roughness']
        F0 = frame['F0']
        kd = frame['kd']
        ao = frame['ao']
        source_shape = frame['source_shape']
        intrinsic_gloss = frame['intrinsic_gloss']

        diversity_scale = float(np.clip(getattr(lighting_info, 'palette_diversity', 0.35), 0.0, 1.0))
        background_mode = str(getattr(lighting_info, 'background_mode', 'balanced'))
        neon_strength = str(getattr(lighting_info, 'neon_strength', 'off'))
        strong_neon = self.style_mode == 'neon' and neon_strength == 'strong'
        soft_neon = self.style_mode == 'neon' and neon_strength == 'soft'
        if self._using_continuous_policy():
            # Legacy background_mode/neon branches are disabled in look-safe.
            # The background still drives light through lighting_info and budget;
            # it just cannot open another style exit here.
            background_mode = 'balanced'
            neon_strength = 'off'
            strong_neon = False
            soft_neon = False
        policy_render = self._policy_section('render_weight') if self._using_continuous_policy() else {}
        policy_exposure = self._policy_section('exposure') if self._using_continuous_policy() else {}
        policy_chroma = self._policy_section('chroma') if self._using_continuous_policy() else {}
        policy_direction = self._policy_section('direction') if self._using_continuous_policy() else {}
        policy_display = self._policy_section('display') if self._using_continuous_policy() else {}
        policy_style = self._policy_section('style_expression') if self._using_continuous_policy() else {}
        gf_stage35 = getattr(lighting_info, 'gradient_field', {}) if hasattr(lighting_info, 'gradient_field') else {}
        if isinstance(gf_stage35, dict):
            bg_p50_stage35 = float(gf_stage35.get('p50_luma', 0.24))
            bg_p95_stage35 = float(gf_stage35.get('p95_luma', 0.45))
            bg_colorfulness_stage35 = float(gf_stage35.get('colorfulness', 0.30))
        else:
            bg_p50_stage35, bg_p95_stage35, bg_colorfulness_stage35 = 0.24, 0.45, 0.30
        bright_bg_factor = float(np.clip((bg_p50_stage35 - 0.27) / 0.22, 0.0, 1.0))
        dark_bg_factor = float(np.clip((0.19 - bg_p50_stage35) / 0.16, 0.0, 1.0))
        dark_color_factor = float(np.clip(max(dark_bg_factor, 0.82 if (background_mode == 'rich' and (strong_neon or soft_neon)) else 0.0) + 0.18 * bg_colorfulness_stage35, 0.0, 1.0))
        ambient_color_np = np.array(lighting_info.ambient_color, dtype=np.float32)
        if background_mode == 'monotone':
            ambient_color_np = desaturate_color(ambient_color_np, 0.18)
        ambient_role_stage35 = self._classify_light_hue(ambient_color_np)
        bright_cool_factor = float(np.clip(bright_bg_factor * (1.0 if ambient_role_stage35 == 'cool' else 0.62), 0.0, 1.0))
        ambient_color = ambient_color_np.reshape(1, 1, 3)
        adaptive_ambient_strength = self.ambient_strength * (1.18 if background_mode == 'monotone' else (1.10 if background_mode == 'rich' and strong_neon else 1.0))
        adaptive_fill_strength = self.fill_strength * (1.22 if background_mode == 'monotone' else (1.35 if background_mode == 'rich' and strong_neon else (1.08 if soft_neon else 1.0)))
        adaptive_multi_ambient_strength = self.multi_ambient_strength * (0.48 + 0.92 * diversity_scale)
        if background_mode == 'monotone':
            adaptive_multi_ambient_strength *= 1.10
        elif background_mode == 'rich' and strong_neon:
            adaptive_multi_ambient_strength *= 1.18
        if self._using_continuous_policy():
            adaptive_ambient_strength *= float(np.clip(policy_render.get('ambient', 1.0), 0.45, 1.25))
            adaptive_fill_strength *= float(np.clip(policy_render.get('fill', 1.0), 0.40, 1.20))
            adaptive_multi_ambient_strength *= float(np.clip(policy_render.get('multicolor', 1.0), 0.10, 1.35))
        ambient = base_subject * ambient_color * float(lighting_info.ambient_intensity) * adaptive_ambient_strength
        fill = base_subject * ambient_color * adaptive_fill_strength * (0.50 + 0.50 * facing[..., None])
        # Optical relighting: fill should lift shadows, not repaint the whole face.
        if any(str(light_dict.get('name', '')).startswith('optical_') for light_dict in getattr(lighting_info, 'lights', [])):
            fill *= 0.86
        if self.debug_dump:
            self._debug_intermediates['ambient_fill'] = np.clip(ambient + fill, 0.0, None).copy()
        gradient_field_acc = self._compute_background_gradient_light(
            base_subject=base_subject,
            subject_mask=subject_mask,
            N=N,
            V=V,
            P=P,
            face_core=face_core,
            hair_region=hair_region,
            edge_band=edge_band,
            specular_map=spec_map,
            roughness_map=roughness,
            lighting_info=lighting_info,
            source_shape=source_shape,
        )
        hdri_body_acc = self._compute_hdri_spherical_bodylight(
            base_subject=base_subject,
            subject_mask=subject_mask,
            N=N,
            V=V,
            P=P,
            face_core=face_core,
            hair_region=hair_region,
            edge_band=edge_band,
            specular_map=spec_map,
            roughness_map=roughness,
            lighting_info=lighting_info,
            source_shape=source_shape,
        )
        switchlight_pbr_acc = self._compute_switchlight_pbr_env_pass(
            base_subject=base_subject,
            albedo_linear=albedo_linear,
            subject_mask=subject_mask,
            N=N,
            V=V,
            P=P,
            face_core=face_core,
            hair_region=hair_region,
            edge_band=edge_band,
            specular_map=spec_map,
            roughness_map=roughness,
            F0=F0,
            lighting_info=lighting_info,
            source_shape=source_shape,
        )
        multicolor_acc = np.zeros_like(base_subject)
        diffuse_acc = np.zeros_like(base_subject)
        spec_acc = np.zeros_like(base_subject)
        rim_acc = np.zeros_like(base_subject)
        lights = self._resolve_render_lights(lighting_info)
        lights = self._adapt_lights_to_portrait_orientation(lights, N, P, subject_mask, face_core)
        key_shadow = self._compute_directional_shadow(depth_map, subject_mask, np.array(lights[0].direction, dtype=np.float32)) if lights else np.ones_like(subject_mask)
        if self._using_continuous_policy():
            shadow_scale = float(np.clip(0.55 + 1.35 * policy_direction.get('shadow_strength', 0.18), 0.45, 1.55))
            key_shadow = np.clip(1.0 - (1.0 - key_shadow) * shadow_scale, 0.58, 1.0).astype(np.float32)
            rim_strength_scale = float(np.clip(0.62 + 1.35 * policy_direction.get('rim_strength', 0.20), 0.55, 1.70))
            side_separation_scale = float(np.clip(0.85 + 0.55 * policy_direction.get('side_separation', 0.20), 0.80, 1.35))
            source_shading_preserve_eff = float(np.clip(self.source_shading_preserve * policy_render.get('source_shading_preserve', 1.0), 0.0, 0.38))
            shadow_sculpt_eff = float(np.clip(self.shadow_sculpt_strength * shadow_scale, 0.0, 0.38))
        else:
            rim_strength_scale = 1.0
            side_separation_scale = 1.0
            source_shading_preserve_eff = self.source_shading_preserve
            shadow_sculpt_eff = self.shadow_sculpt_strength

        for i, light in enumerate(lights):
            L = safe_norm(np.array(light.direction, dtype=np.float32))
            Lf = np.ones_like(N) * L.reshape(1, 1, 3)
            H = safe_norm(Lf + V)
            NdotL_raw = np.sum(N * Lf, axis=-1)
            NdotL = np.clip(NdotL_raw, 0.0, 1.0).astype(np.float32)
            NdotH = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0).astype(np.float32)
            VdotH = np.clip(np.sum(V * H, axis=-1), 0.0, 1.0).astype(np.float32)
            is_key_light = light.name.startswith('bg_key') or light.name.startswith('virtual_key') or i == 0
            is_fill = (i > 0) and (not is_key_light) and (light.name.startswith('bg_fill') or light.name.startswith('optical_fill') or light.name.startswith('virtual_fill'))
            wrap = 0.015 if is_key_light else 0.18
            soft_diff = np.clip((NdotL_raw + wrap) / (1.0 + wrap), 0.0, 1.0).astype(np.float32)
            if is_key_light:
                diff_term = np.power(NdotL, 1.05)
            elif is_fill:
                diff_term = np.power(soft_diff, 0.92)
            else:
                diff_term = np.power(soft_diff, 0.98)
            shadow_sculpt = 1.0 - shadow_sculpt_eff * np.power(
                np.clip(1.0 - NdotL, 0.0, 1.0),
                1.25
            ) * (0.35 + 0.65 * subject_mask)
            depth_shadow = key_shadow if is_key_light else (0.93 + 0.07 * key_shadow)
            shape_preserve = (1.0 - source_shading_preserve_eff) + source_shading_preserve_eff * source_shape
            diffuse_shape = np.clip(shadow_sculpt * depth_shadow * shape_preserve, 0.62, 1.24)
            lc = np.array(light.color, dtype=np.float32).reshape(1, 1, 3)
            le = lc * float(light.intensity)

            broad = np.clip((NdotL_raw + self.multi_ambient_wrap) / (1.0 + self.multi_ambient_wrap), 0.0, 1.0).astype(np.float32)
            broad = np.power(broad, 0.90)
            side_mask = self._compute_spatial_side_mask(P, L, subject_mask)
            face_gate = 0.84 + self.multi_ambient_face_bias * face_core
            if self.look_safe:
                _floor = self._atmosphere_budget['multicolor_face_gate_floor'] if self._atmosphere_budget else 0.24
                face_gate = np.clip(face_gate - 0.60 * face_core, _floor, 1.0)
            broad_color = base_subject * le * (adaptive_multi_ambient_strength * float(light.diffuse_scale))
            hue_role = self._classify_light_hue(np.array(light.color, dtype=np.float32))
            side_emphasis = (0.35 + self.multi_ambient_side_bias * side_mask[..., None])
            if self._using_continuous_policy():
                side_emphasis = np.clip(side_emphasis * side_separation_scale, 0.0, 1.75)
            if strong_neon:
                side_emphasis = np.clip(0.18 + (0.90 + self.neon_side_separation) * side_mask[..., None], 0.0, 1.55)
                if hue_role == 'cool':
                    broad_color = broad_color * 1.14
                elif hue_role == 'warm':
                    broad_color = broad_color * 0.96
            elif soft_neon:
                side_emphasis = np.clip(0.28 + 0.92 * side_mask[..., None], 0.0, 1.35)
                if hue_role == 'cool':
                    broad_color = broad_color * 1.04
            # Keep background color influence local and directional. The old broad term
            # behaved like a colored fog over the whole portrait. Here color spill is
            # strongest on the lit side, hair and alpha edge, and weaker on the face core.
            optical_mode = str(light.name).startswith('optical_')
            if optical_mode:
                wash_strength = 0.16 if is_key_light else (0.08 if is_fill else 0.26)
                local_side_gate = np.clip(0.12 + 0.88 * side_mask, 0.0, 1.0)
                edge_hair_gate = np.clip(0.22 + 0.78 * (0.58 * hair_region + 0.42 * edge_band), 0.0, 1.0)
                face_reduce = np.clip(1.0 - 0.74 * face_core, 0.24, 1.0)
                color_spill_gate = local_side_gate[..., None] * edge_hair_gate[..., None] * face_reduce[..., None]
                multicolor_acc += broad_color * broad[..., None] * side_emphasis * face_gate[..., None] * color_spill_gate * wash_strength
            else:
                # Stage35: keep the original background-driven relighting, but make
                # it a structured light spill instead of a flat color layer.  The
                # face core still receives the colored light, but less than cheeks,
                # hair and edge, so the result reads as illumination rather than a
                # red/purple mask.
                local_side_gate = np.clip(0.34 + 0.66 * side_mask, 0.0, 1.0).astype(np.float32)
                region_light_gate = np.clip(
                    0.50 * subject_mask
                    + 0.32 * local_side_gate
                    + 0.26 * hair_region
                    + 0.34 * edge_band
                    - 0.10 * face_core * dark_color_factor
                    - 0.24 * hair_region * bright_cool_factor,
                    0.28,
                    1.18,
                ).astype(np.float32)
                if strong_neon:
                    # Do not remove the cyber/neon atmosphere: make it directional
                    # and strongest on the side/edge/hair.
                    region_light_gate *= np.clip(1.06 + 0.20 * local_side_gate + 0.10 * edge_band - 0.12 * face_core, 0.70, 1.34)
                elif bright_cool_factor > 0.35:
                    # Ice/snow-like scenes should not turn hair into white fog.
                    region_light_gate *= np.clip(1.0 - 0.30 * hair_region - 0.18 * edge_band, 0.56, 1.0)
                multicolor_acc += broad_color * broad[..., None] * side_emphasis * face_gate[..., None] * region_light_gate[..., None]

            rough_eff = np.clip(roughness + light.size * 0.10, 0.10, 0.98)
            D = D_GGX(NdotH, rough_eff)
            G = G_SchlickGGX(np.clip(NdotV, 0.0, 1.0), rough_eff) * G_SchlickGGX(np.clip(NdotL, 0.0, 1.0), rough_eff)
            F = fresnel_schlick(VdotH, F0)
            spec = (D[..., None] * G[..., None] * F) / np.maximum(4.0 * np.clip(NdotV, 0.0, 1.0)[..., None] * np.clip(NdotL, 0.0, 1.0)[..., None], 1e-5)
            diffuse = kd * base_subject / PI * diff_term[..., None] * diffuse_shape[..., None] * le * float(light.diffuse_scale)
            if strong_neon and is_fill:
                diffuse *= np.clip(0.30 + 1.00 * side_mask[..., None], 0.0, 1.25)
            specular = spec * le * spec_map[..., None] * float(light.specular_scale)
            specular *= np.clip(0.18 + 0.82 * NdotL[..., None], 0.0, 1.0)
            # Matte portrait rule: keep specular mostly on hair/edges/clothes.
            # Face core gets a very small amount only, so the lighting direction
            # remains visible without forming an oil-film highlight.
            portrait_spec_gate = np.clip(
                0.035 * subject_mask
                + 0.045 * face_core
                + 0.44 * hair_region
                + 0.52 * edge_band,
                0.0,
                0.58,
            )
            if is_fill:
                portrait_spec_gate *= np.clip(0.25 + 0.75 * (hair_region + edge_band), 0.0, 1.0)
            if self._using_continuous_policy():
                portrait_spec_gate = np.clip(
                    0.018 * subject_mask
                    + 0.020 * face_core
                    + 0.16 * hair_region
                    + 0.28 * edge_band,
                    0.0,
                    0.34,
                )
            specular *= portrait_spec_gate[..., None]
            if abs(float(L[0])) > 0.20:
                rim_term = np.power(np.clip(1.0 - np.clip(np.sum(N * V, axis=-1), 0.0, 1.0), 0.0, 1.0), 2.45)
                rim_gate = np.clip((0.10 - NdotL_raw) / 0.24, 0.0, 1.0)
                if self._using_continuous_policy():
                    rim_region = np.clip(0.58 * edge_band + 0.42 * hair_region, 0.0, 1.0)
                else:
                    rim_region = self.rim_edge_balance * edge_band + self.rim_hair_balance * hair_region
                rim_acc += le * rim_term[..., None] * rim_gate[..., None] * rim_region[..., None] * self.rim_strength * rim_strength_scale * float(light.rim_scale)
            diffuse_acc += diffuse
            spec_acc += specular

        if self.debug_dump:
            self._debug_intermediates['diffuse'] = diffuse_acc.copy()
            self._debug_intermediates['specular'] = spec_acc.copy()
            self._debug_intermediates['rim'] = rim_acc.copy()
            self._debug_intermediates['key_shadow'] = key_shadow.copy()

        # Make the portrait more dependent on the analyzed background field
        # while keeping physically-based local lights for structure.
        # Whole-body HDRI wrap is added here so the portrait bends under the background light field
        # instead of receiving a mostly flat color wash.
        if getattr(self, 'light_source_mode', 'hybrid') == 'hybrid':
            env_scale = float(getattr(self, 'environment_light_scale', 0.55))
            direct_scale = float(getattr(self, 'direct_light_strength', 1.55))
            if self._using_continuous_policy():
                # In look-safe the old HDRI/PBR/multicolor passes are only a
                # neutral shape base.  The compact LookPolicy layer owns the
                # final direction, color spill, rim and display expression.
                rw_ambient = float(np.clip(policy_render.get('ambient_weight', 1.0), 0.55, 1.15))
                rw_fill = float(np.clip(0.72 * rw_ambient, 0.38, 0.92))
                rw_gradient = float(np.clip(0.18 + 0.14 * policy_render.get('direct_weight', 1.0), 0.14, 0.38))
                rw_hdri = float(np.clip(0.18 + 0.10 * policy_render.get('ambient_weight', 1.0), 0.14, 0.34))
                rw_pbr = float(np.clip(0.16 + 0.12 * policy_render.get('direct_weight', 1.0), 0.12, 0.34))
                rw_multi = float(np.clip(0.06 + 0.12 * policy_render.get('color_spill_weight', 1.0), 0.04, 0.22))
                rw_direct = float(np.clip(policy_render.get('direct_weight', 1.0), 0.34, 1.20))
                rw_diffuse = 0.86
                rw_rim = float(np.clip(0.52 * policy_render.get('rim_weight', 1.0), 0.22, 0.78))
                rw_spec = 0.14
            else:
                rw_ambient = 1.0
                rw_fill = 1.0
                rw_gradient = 1.0
                rw_hdri = 1.0
                rw_pbr = 1.0
                rw_multi = 1.0
                rw_direct = 1.0
                rw_diffuse = 1.0
                rw_rim = 1.0
                rw_spec = 1.0
            # In hybrid mode, the virtual rig must be visible.  Background field
            # remains as color/exposure atmosphere but no longer dominates the
            # direct normal-based key/fill/rim lighting.
            relit = (
                ambient * (0.56 + 0.16 * env_scale) * rw_ambient
                + fill * (0.42 + 0.12 * env_scale) * rw_fill
                + gradient_field_acc * 0.34 * env_scale * rw_gradient
                + hdri_body_acc * 0.30 * env_scale * rw_hdri
                + switchlight_pbr_acc * 0.30 * env_scale * rw_pbr
                + multicolor_acc * 0.38 * rw_multi
                + diffuse_acc * direct_scale * rw_direct * rw_diffuse
                + rim_acc * direct_scale * rw_rim
                + spec_acc * min(direct_scale, 0.85) * rw_spec
            )
            if self._using_continuous_policy():
                self._last_policy_runtime = {
                    'style_expression': policy_style,
                    'render_weight': {
                        'ambient': rw_ambient,
                        'fill': rw_fill,
                        'gradient_field': rw_gradient,
                        'hdri_body': rw_hdri,
                        'switchlight_pbr': rw_pbr,
                        'multicolor': rw_multi,
                        'direct_light': rw_direct,
                        'diffuse': rw_diffuse,
                        'rim': rw_rim,
                        'specular': rw_spec,
                        'source_preserve': float(source_preserve_scale),
                        'source_shading_preserve': float(source_shading_preserve_eff),
                    },
                    'exposure': policy_exposure,
                    'chroma': policy_chroma,
                    'direction': policy_direction,
                    'display': policy_display,
                }
            if lights:
                relit = self._apply_pose_aware_diffuse_sculpt(
                    relit=relit,
                    base_subject=base_subject,
                    N=N,
                    P=P,
                    subject_mask=subject_mask,
                    face_core=face_core,
                    hair_region=hair_region,
                    edge_band=edge_band,
                    key_light=lights[0],
                )
        else:
            relit = ambient + fill + gradient_field_acc * 0.78 + hdri_body_acc * 0.95 + switchlight_pbr_acc * 1.35 + multicolor_acc + diffuse_acc + spec_acc + rim_acc
        if strong_neon and lights:
            warm_colors = []
            cool_colors = []
            warm_signs = []
            cool_signs = []
            for light in lights:
                c = np.array(light.color, dtype=np.float32)
                role = self._classify_light_hue(c)
                if role == 'warm':
                    warm_colors.append(c * float(light.intensity))
                    warm_signs.append(float(light.direction[0]))
                elif role == 'cool':
                    cool_colors.append(c * float(light.intensity))
                    cool_signs.append(float(light.direction[0]))
            if warm_colors and cool_colors:
                warm_color = np.mean(np.stack(warm_colors, axis=0), axis=0).astype(np.float32)
                cool_color = np.mean(np.stack(cool_colors, axis=0), axis=0).astype(np.float32)
                warm_sign = float(np.mean(warm_signs)) if warm_signs else -1.0
                cool_sign = float(np.mean(cool_signs)) if cool_signs else 1.0
                warm_side = self._compute_signed_side_mask(P, warm_sign, subject_mask, power=0.75)
                cool_side = self._compute_signed_side_mask(P, cool_sign, subject_mask, power=0.75)
                dual_gate = np.power(np.clip(np.abs(P[..., 0]) / max(float(np.percentile(np.abs(P[..., 0][subject_mask > 0.08]), 88.0)) if np.any(subject_mask > 0.08) else 1.0, 1e-4), 0.0, 1.0), self.neon_dual_tint_center_falloff)
                dual_region = np.clip(0.92 - 0.28 * face_core + 0.34 * hair_region + 0.42 * edge_band + 0.22 * np.maximum(warm_side, cool_side), 0.36, 1.24)
                # Keep the two-color neon light visible, but make the center face
                # receive a real, directional amount rather than a constant tint.
                dual_gate = np.clip(0.10 + 0.90 * dual_gate, 0.0, 1.0) * subject_mask * dual_region
                dual_tint = base_subject * (warm_color.reshape(1, 1, 3) * warm_side[..., None] + cool_color.reshape(1, 1, 3) * cool_side[..., None])
                dual_strength = self.neon_dual_tint_strength
                if background_mode == 'monotone':
                    dual_strength *= 0.22
                elif background_mode == 'rich':
                    dual_strength *= 1.35
                relit += dual_tint * dual_strength * dual_gate[..., None]
        # In hybrid mode the virtual rig should visibly sculpt the subject even
        # when lighting-pattern is auto. Manual patterns still act as explicit
        # overrides; background mode keeps the older behavior.
        if (not self._using_continuous_policy()) and (getattr(self, 'light_source_mode', 'hybrid') == 'hybrid' or getattr(self, 'lighting_pattern', 'auto') != 'auto'):
            relit = self._apply_structural_style_boost(relit, base_subject, subject_mask, face_core, hair_region, P, lighting_info)
        relit *= ao[..., None]
        if lights:
            contact_shadow = self._compute_contact_shadow(depth_map, subject_mask, np.array(lights[0].direction, dtype=np.float32))
        else:
            contact_shadow = np.ones_like(subject_mask, dtype=np.float32)
        self._last_contact_shadow = contact_shadow
        if self.debug_dump:
            self._debug_intermediates['contact_shadow'] = contact_shadow.copy()
        relit *= contact_shadow[..., None]
        # Add visible background-driven reflectivity after contact/AO so gloss is not
        # swallowed by shadowing.  This uses roughness/specular/normal maps and the
        # background light field, not a fixed white highlight.
        if not self._using_continuous_policy():
            reflective_finish = self._compute_background_reflective_finish(
                base_subject=base_subject,
                subject_mask=subject_mask,
                N=N,
                V=V,
                P=P,
                face_core=face_core,
                hair_region=hair_region,
                edge_band=edge_band,
                specular_map=spec_map,
                roughness_map=roughness,
                lighting_info=lighting_info,
                source_shape=source_shape,
                intrinsic_gloss=intrinsic_gloss,
                camera_params=camera_params,
            )
            # Keep the background reflective finish subtle and away from the face.
            if self._using_continuous_policy():
                reflection_gate = np.clip(0.004 * subject_mask + 0.10 * hair_region + 0.18 * edge_band, 0.0, 0.24)
                reflection_scale = 0.026
            else:
                reflection_gate = np.clip(0.01 * subject_mask + 0.24 * hair_region + 0.30 * edge_band, 0.0, 0.42)
                reflection_scale = 0.045
            relit += reflective_finish * reflection_gate[..., None] * reflection_scale
            visible_specular = self._compute_visible_virtual_specular_boost(
                base_subject=base_subject,
                subject_mask=subject_mask,
                N=N,
                V=V,
                P=P,
                face_core=face_core,
                hair_region=hair_region,
                edge_band=edge_band,
                specular_map=spec_map,
                roughness_map=roughness,
                lights=lights,
            )
            if self._using_continuous_policy():
                visible_specular_gate = np.clip(0.002 * subject_mask + 0.070 * hair_region + 0.14 * edge_band, 0.0, 0.18)
                visible_specular_scale = 0.020
            else:
                visible_specular_gate = np.clip(0.005 * subject_mask + 0.18 * hair_region + 0.26 * edge_band, 0.0, 0.34)
                visible_specular_scale = 0.035
            relit += visible_specular * visible_specular_gate[..., None] * visible_specular_scale
            portrait_finish = self._compute_portrait_recipe_finish(
                relit=relit,
                base_subject=base_subject,
                source_linear=source_linear,
                subject_mask=subject_mask,
                N=N,
                V=V,
                P=P,
                face_core=face_core,
                hair_region=hair_region,
                edge_band=edge_band,
                specular_map=spec_map,
                roughness_map=roughness,
                lighting_info=lighting_info,
            )
            # Portrait recipe finish is useful for hair/edge integration, but it can
            # make skin waxy.  Apply it only to non-face peripheral regions.
            if self._using_continuous_policy():
                portrait_finish_gate = np.clip(0.006 * subject_mask + 0.12 * hair_region + 0.22 * edge_band + 0.018 * (1.0 - face_core) * subject_mask, 0.0, 0.28)
                portrait_finish_scale = 0.040
            else:
                portrait_finish_gate = np.clip(0.02 * subject_mask + 0.34 * hair_region + 0.42 * edge_band + 0.04 * (1.0 - face_core) * subject_mask, 0.0, 0.55)
                portrait_finish_scale = 0.10
            relit += portrait_finish * portrait_finish_gate[..., None] * portrait_finish_scale
        # Preserve original Source texture.  Normal-based relighting creates a
        # smooth light field; if it directly replaces the portrait, skin becomes
        # waxy/blurred.  Convert relit into a luminance lightmap, multiply that
        # back onto source_linear, and use it strongly on the face/core regions.
        if getattr(self, 'light_source_mode', 'hybrid') == 'hybrid' and not self._using_continuous_policy():
            lightmap = relit / np.maximum(base_subject, 1e-4)
            lightmap_luma = np.clip(rgb_luminance(lightmap), 0.35, 2.20).astype(np.float32)
            # Smooth the illumination field only slightly.  We want source skin
            # texture to survive, but we do not want to strip away the background-
            # driven color influence.  So preserve source texture mainly via luma,
            # while re-injecting relit chroma/tint.
            lightmap_luma = box_blur_gray(lightmap_luma, passes=2)
            lightmap_luma = np.clip(lightmap_luma, 0.40, 2.00).astype(np.float32)
            face_no_hf_gate = np.clip(0.22 * face_core + 0.06 * subject_mask - 0.10 * hair_region - 0.06 * edge_band, 0.0, 0.32).astype(np.float32)
            if background_mode == 'rich' and strong_neon:
                face_no_hf_gate = np.clip(face_no_hf_gate * 0.65, 0.0, 0.22).astype(np.float32)
            texture_safe_source = source_linear * (1.0 - face_no_hf_gate[..., None]) + source_lowfreq * face_no_hf_gate[..., None]
            source_relight = texture_safe_source * lightmap_luma[..., None]

            relit_luma = np.maximum(rgb_luminance(relit), 1e-4).astype(np.float32)
            relit_chroma = np.clip(relit / relit_luma[..., None], 0.0, 3.0).astype(np.float32)
            source_luma = np.maximum(rgb_luminance(source_linear), 1e-4).astype(np.float32)
            source_chroma = np.clip(source_linear / source_luma[..., None], 0.0, 3.0).astype(np.float32)
            # Keep the background-driven chroma, but do not let it become a full
            # face/body tint.  In close-up face crops, even moderate chroma mixing
            # reads as heavy skin staining, so reduce it adaptively in face-core.
            subj_area_h = float(np.mean(np.clip(subject_mask, 0.0, 1.0)))
            face_area_h = float(np.mean(np.clip(face_core * subject_mask, 0.0, 1.0)))
            closeup_gate_h = float(np.clip((face_area_h / max(subj_area_h, 1e-5) - 0.30) / 0.42, 0.0, 1.0))
            bg_chroma_weight = np.clip(
                0.26 + 0.07 * hair_region + 0.10 * edge_band - 0.14 * face_core - 0.24 * closeup_gate_h * face_core,
                0.12,
                0.46,
            ).astype(np.float32)
            if self._using_continuous_policy():
                face_side_mask_h = np.clip(face_core * (1.0 - np.power(face_core, 1.25)) + 0.12 * hair_region, 0.0, 1.0).astype(np.float32)
                body_mask_h = np.clip(subject_mask * (1.0 - 0.90 * face_core) * (1.0 - 0.55 * hair_region), 0.0, 1.0).astype(np.float32)
                clothing_like_h = clothing_mask if clothing_mask is not None else body_mask_h * 0.45
                bg_chroma_weight = np.clip(
                    0.010 * subject_mask
                    + face_core * float(policy_chroma.get('face_core', 0.004))
                    + face_side_mask_h * float(policy_chroma.get('face_side', 0.026))
                    + body_mask_h * float(policy_chroma.get('body', 0.050))
                    + clothing_like_h * float(policy_chroma.get('clothing', 0.080))
                    + hair_region * float(policy_chroma.get('hair', 0.160))
                    + edge_band * float(policy_chroma.get('edge', 0.180)),
                    0.004,
                    0.58,
                ).astype(np.float32)
            chroma_mix = np.clip(
                source_chroma * (1.0 - bg_chroma_weight[..., None])
                + relit_chroma * bg_chroma_weight[..., None],
                0.0,
                3.0,
            ).astype(np.float32)
            chroma_mix /= np.maximum(np.sum(chroma_mix, axis=-1, keepdims=True), 1e-4)
            colorized_source_relight = source_relight * chroma_mix * 3.0

            # Preserve texture mainly on the face/core but do not fully replace the
            # relit result, otherwise the person keeps only the original source color.
            texture_preserve_gate = np.clip(
                0.18 * face_core + 0.11 * subject_mask + 0.10 * hair_region - 0.08 * edge_band,
                0.0,
                0.34,
            ).astype(np.float32)
            texture_mix = np.clip(0.34 + 0.07 * face_core + 0.05 * subject_mask + 0.06 * hair_region, 0.0, 0.50).astype(np.float32)
            if self._using_continuous_policy():
                texture_preserve_gate = np.clip(texture_preserve_gate * float(policy_render.get('source_preserve', 1.0)), 0.0, 0.42).astype(np.float32)
                texture_mix = np.clip(texture_mix * float(policy_render.get('source_preserve', 1.0)), 0.0, 0.58).astype(np.float32)
            if background_mode == 'rich' and strong_neon:
                texture_preserve_gate = np.clip(texture_preserve_gate * 1.18, 0.0, 0.42).astype(np.float32)
                texture_mix = np.clip(texture_mix * 1.10, 0.0, 0.58).astype(np.float32)
            relit_texture_preserved = relit * (1.0 - texture_mix[..., None]) + colorized_source_relight * texture_mix[..., None]
            relit = relit * (1.0 - texture_preserve_gate[..., None]) + relit_texture_preserved * texture_preserve_gate[..., None]

            # Stage44: no face high-frequency detail refill.  Preserve more mid-frequency
            # facial structure through the lightmap/source mix above, while keeping
            # tiny high-pass residuals limited to hair/outer contour.
            src_luma_stage36 = rgb_luminance(source_linear).astype(np.float32)
            luma_detail_stage36 = np.clip(src_luma_stage36 - box_blur_gray(src_luma_stage36, passes=3), -0.018, 0.018).astype(np.float32)
            detail_gate = np.clip(0.16 * hair_region + 0.14 * edge_band + 0.05 * subject_mask * (1.0 - face_core), 0.0, 0.26)[..., None]
            relit += luma_detail_stage36[..., None] * detail_gate * (self.detail_strength * 0.45)
        else:
            detail_gate = np.clip(0.18 * hair_region + 0.16 * edge_band + 0.05 * subject_mask * (1.0 - face_core), 0.0, 0.24)
            relit += detail * self.detail_strength * detail_gate[..., None]
            relit = relit * self.subject_mix + base_subject * (1.0 - self.subject_mix)

        if not self._using_continuous_policy():
            # Legacy mode keeps the older corrective stack.  In continuous
            # look-safe mode these responsibilities are consolidated in
            # _apply_compact_lookpolicy_layer after auto-gain.
            soft_edge_stage35 = box_blur_gray(edge_band.astype(np.float32), passes=1)
            relit += soft_edge_stage35[..., None] * desaturate_color(ambient_color, 0.38 if background_mode == 'rich' else 0.55) * (self.edge_spill_strength * 0.55 * float(np.clip(0.85 + 0.25 * dark_color_factor - 0.18 * bright_cool_factor, 0.62, 1.18)))
            relit = self._stabilize_skin_color(relit, source_linear, subject_mask, face_core, lighting_info)
            relit = self._apply_face_color_density_control(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
                lighting_info,
            )
            relit = self._apply_body_skin_cast_governor(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
                lighting_info,
            )
            relit = self._apply_face_air_and_texture_guard(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
                lighting_info,
            )
            relit = self._inject_subject_colored_light(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
                lighting_info,
            )
        if self.look_safe and self._atmosphere_budget and not self._using_continuous_policy():
            relit = self._apply_look_safe_directional_atmosphere(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
                clothing_mask,
                lighting_info,
                self._atmosphere_budget,
            )
            relit = self._soften_face_hard_light_edges(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
            )
            relit = self._restore_face_soft_detail(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
            )
            relit = self._restore_face_midfreq_clarity(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
            )
            relit = self._suppress_face_micro_blemishes(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
            )

        if getattr(self, 'light_source_mode', 'hybrid') == 'hybrid' and not self._using_continuous_policy():
            # Stage36: the late pass must not reintroduce face micro-texture after
            # skin stabilization. Keep only a very small contour/hair detail assist.
            final_luma_stage36 = rgb_luminance(source_linear).astype(np.float32)
            final_detail_stage36 = np.clip(final_luma_stage36 - box_blur_gray(final_luma_stage36, passes=3), -0.014, 0.014).astype(np.float32)
            final_detail_gate = np.clip(0.12 * hair_region + 0.14 * edge_band + 0.04 * subject_mask * (1.0 - face_core), 0.0, 0.20)[..., None]
            relit += final_detail_stage36[..., None] * final_detail_gate * (self.detail_strength * 0.30)
        if not self._using_continuous_policy():
            # Late HDRI sphere-map arc is a legacy finishing pass.  Continuous
            # look-safe keeps rim/spec/color in the compact policy layer.
            late_hdri_arc = self._compute_hdri_spheremap_late_arc(
                relit=relit,
                base_subject=base_subject,
                subject_mask=subject_mask,
                N=N,
                V=V,
                P=P,
                face_core=face_core,
                hair_region=hair_region,
                edge_band=edge_band,
                specular_map=spec_map,
                roughness_map=roughness,
                lighting_info=lighting_info,
                source_shape=source_shape,
            )
            # A smaller late addition keeps the Cook-Torrance PBR structure visible
            # after skin-color stabilization, without making the face overly glossy.
            late_arc_gate = np.clip(0.004 * subject_mask + 0.16 * hair_region + 0.22 * edge_band, 0.0, 0.30)
            relit += late_hdri_arc * late_arc_gate[..., None] * 0.012 + switchlight_pbr_acc * 0.010

        if background_mode == 'monotone':
            relit_luma_pre = rgb_luminance(relit)
            shadow_mask = np.clip((0.30 - relit_luma_pre) / 0.30, 0.0, 1.0) * subject_mask
            relit += ambient_color * (0.16 * shadow_mask[..., None])
            relit += base_subject * (0.08 * shadow_mask[..., None])

        # The core route reserves final style expression for the compact layer
        # after auto-gain.
        if not self._using_continuous_policy():
            relit = self._v32_style_block_router(
                relit,
                source_linear,
                subject_mask,
                face_core,
                hair_region,
                edge_band,
                lighting_info,
            )

        if np.any(subject_mask > 0.20):
            relit_luma = rgb_luminance(relit)
            gf_for_target = getattr(lighting_info, 'gradient_field', {}) if hasattr(lighting_info, 'gradient_field') else {}
            # Background-relative exposure: fixed high targets made dark/purple
            # scenes look pale.  Let dark scenes stay moody while keeping the face readable.
            if isinstance(gf_for_target, dict):
                bg_p50 = float(gf_for_target.get('p50_luma', 0.22))
                bg_p95 = float(gf_for_target.get('p95_luma', 0.45))
                bg_colorfulness = float(gf_for_target.get('colorfulness', 0.30))
                target_p70 = float(np.clip(0.330 + 0.16 * bg_p50 + 0.055 * bg_p95 + 0.014 * bg_colorfulness, 0.350, 0.430))
                # Stage37: do not let dark colorful backgrounds pull the face down.
                # The background can stay moody, but the portrait midtone must remain
                # readable.  Bright cool scenes are no longer capped at 0.34; that cap
                # was one of the reasons faces looked too deep.
                if bright_cool_factor > 0.45:
                    target_p70 = max(target_p70, 0.372)
                if dark_color_factor > 0.55:
                    target_p70 = max(target_p70, 0.388)
            else:
                target_p70 = self.target_subject_p70
            if not self._using_continuous_policy():
                recipe_info = self._estimate_portrait_light_recipe(lighting_info)
                recipe_name = str(recipe_info.get('recipe', 'balanced_soft'))
                if recipe_name == 'cool_env':
                    target_p70 += 0.012
                elif recipe_name == 'warm_side':
                    target_p70 += 0.018
                elif recipe_name == 'neutral_soft':
                    target_p70 += 0.016
                elif recipe_name == 'night_mixed':
                    target_p70 += 0.010
                if background_mode == 'monotone':
                    target_p70 += 0.022
                elif background_mode == 'rich' and strong_neon:
                    target_p70 = max(target_p70 + 0.012, 0.405)
            if self.look_safe:
                _ab = self._atmosphere_budget or {}
                _exp = self._policy_section('exposure')
                _target = _exp.get('target_subject_p70', target_p70)
                target_p70 = float(np.clip(_target, _ab.get('autogain_target_low', 0.320), _ab.get('autogain_target_high', 0.390)))
            face_measure_stage35 = (face_core * subject_mask) > 0.16
            if np.count_nonzero(face_measure_stage35) > 80:
                relit_p70 = float(np.percentile(relit_luma[face_measure_stage35], 70.0))
            else:
                relit_p70 = float(np.percentile(relit_luma[subject_mask > 0.20], 70.0))
            gradient_field = getattr(lighting_info, 'gradient_field', {}) if hasattr(lighting_info, 'gradient_field') else {}
            gf_conf = float(gradient_field.get('confidence', 0.0)) if isinstance(gradient_field, dict) else 0.0
            gf_dark = float(gradient_field.get('p50_luma', 0.25)) < 0.14 if isinstance(gradient_field, dict) else False
            gain_upper = 1.82 + 0.26 * gf_conf + (0.26 if gf_dark else 0.0)
            if background_mode == 'rich' and strong_neon:
                gain_upper = min(gain_upper, 1.96)
            if self.look_safe:
                gain_upper = self._policy_value('exposure', 'auto_gain_upper', self._atmosphere_budget.get('autogain_upper', 1.28) if self._atmosphere_budget else 1.28)
            gain = float(np.clip(
                target_p70 / max(relit_p70, 1e-4),
                0.84,
                min(self.max_auto_gain, gain_upper)
            ))
            if self.look_safe and self._atmosphere_budget and gain > 1.0:
                _ab = self._atmosphere_budget
                _pre_gain_luma = rgb_luminance(relit).astype(np.float32)
                _exp = self._policy_section('exposure')
                _face_w = _exp.get('autogain_face_weight', _ab.get('autogain_face_weight', 1.0))
                _body_w = _exp.get('autogain_body_weight', _ab.get('autogain_body_weight', 1.0))
                _edge_w = _exp.get('autogain_edge_weight', _ab.get('autogain_edge_weight', 1.0))
                _face_m = (face_core * subject_mask > 0.16).astype(np.float32)
                _body_m = np.clip(subject_mask * (1.0 - 0.9 * face_core) * (1.0 - 0.3 * edge_band), 0.0, 1.0)
                _edge_m = np.clip(edge_band + hair_region * 0.5, 0.0, 1.0) * subject_mask
                _gain_wt = np.ones_like(subject_mask, dtype=np.float32)
                _gain_wt = _gain_wt * (1.0 - _face_m) + _face_m * _face_w
                _gain_wt = _gain_wt * (1.0 - _body_m * (1.0 - _face_m)) + _body_m * (1.0 - _face_m) * _body_w
                _gain_wt = _gain_wt * (1.0 - _edge_m) + _edge_m * _edge_w
                _eff_gain = 1.0 + (gain - 1.0) * _gain_wt
                _knee_ob = _ab.get('highkey_overbright_knee', 0.85)
                _over_m = _pre_gain_luma > _knee_ob
                if np.any(_over_m):
                    _over_amt = np.clip((_pre_gain_luma[_over_m] - _knee_ob) / (1.0 - _knee_ob + 1e-7), 0.0, 1.0)
                    _suppress = np.clip(1.0 - 0.7 * _over_amt, 0.3, 1.0)
                    _eff_gain[_over_m] *= _suppress
                relit *= _eff_gain[..., np.newaxis]
                _post_gain_luma = rgb_luminance(relit).astype(np.float32)
                _delta_luma = _post_gain_luma - _pre_gain_luma
                _max_face = _exp.get('face_lift_cap', _ab.get('max_positive_face_lift', 0.25))
                _max_body = _exp.get('body_lift_cap', _ab.get('max_positive_body_lift', 0.20))
                _cap_map = np.full_like(subject_mask, 0.25, dtype=np.float32)
                _cap_map = np.where(_face_m > 0.5, _max_face, _cap_map)
                _cap_map = np.where((_body_m > 0.3) & (_face_m <= 0.5), _max_body, _cap_map)
                _exceed = _delta_luma > _cap_map
                if np.any(_exceed):
                    _clamp_ratio = np.where(_exceed, _cap_map / np.maximum(_delta_luma, 1e-7), 1.0)
                    _clamp_ratio = np.clip(_clamp_ratio, 0.0, 1.0)
                    _clamped_luma = _pre_gain_luma + _delta_luma * _clamp_ratio
                    _luma_now = np.maximum(_post_gain_luma, 1e-7)
                    relit *= np.where(_exceed, _clamped_luma / _luma_now, 1.0)[..., np.newaxis]
                self._v3_pre_gain_luma = _pre_gain_luma
                self._v3_post_gain_luma = rgb_luminance(relit).astype(np.float32)
                self._v3_face_mask = _face_m
                self._v3_overbright_mask = _over_m
            else:
                relit *= gain

        if self.look_safe and self._atmosphere_budget and not self._using_continuous_policy():
            _knee_str = self._policy_value('exposure', 'highlight_knee_strength', self._atmosphere_budget.get('subject_highlight_knee_strength', 0.0))
            _knee_start = self._policy_value('exposure', 'highlight_knee_start', self._atmosphere_budget.get('subject_highlight_knee_start', 0.68))
            if _knee_str > 0.01:
                _relit_luma = rgb_luminance(relit)
                _subj_luma = _relit_luma * subject_mask
                _over = np.clip((_subj_luma - _knee_start) / (1.0 - _knee_start + 1e-5), 0.0, 1.0)
                _region_weight = face_core * 0.92 + np.clip(subject_mask - face_core, 0.0, 1.0) * 0.65
                _scale = np.clip(1.0 - _knee_str * _over * _region_weight, 1.0 - _knee_str, 1.0)
                _ratio = np.where(_relit_luma > 1e-5, (_relit_luma * _scale) / _relit_luma, 1.0)
                relit *= _ratio[..., None]

        if self._using_continuous_policy():
            relit = self._apply_compact_lookpolicy_layer(
                relit=relit,
                source_linear=source_linear,
                background_linear=background_linear,
                N=N,
                P=P,
                subject_mask=subject_mask,
                face_core=face_core,
                hair_region=hair_region,
                edge_band=edge_band,
                clothing_mask=clothing_mask,
                lighting_info=lighting_info,
                look_policy=self._look_policy,
            )

        if not self._using_continuous_policy():
            global_bg = np.array(lighting_info.global_mean_color, dtype=np.float32)
            if background_mode == 'monotone':
                global_bg = desaturate_color(global_bg, 0.38)
            else:
                global_bg = desaturate_color(global_bg, 0.50)
            global_bg = brighten_preserve_hue(global_bg, max(float(np.dot(global_bg, LUMA)), 0.14))
            global_tint = np.clip(global_bg / max(float(np.dot(global_bg, LUMA)), 1e-5), 0.92, 1.12)
            adaptive_global_tint_strength = self.global_tint_strength * (0.18 + 0.62 * diversity_scale)
            if background_mode == 'monotone':
                adaptive_global_tint_strength *= 0.35
            elif background_mode == 'rich' and strong_neon:
                adaptive_global_tint_strength *= 0.65
            relit = relit * (1.0 - adaptive_global_tint_strength) + relit * global_tint.reshape(1, 1, 3) * adaptive_global_tint_strength

        if self.look_safe:
            # Budget is the single exposure authority in look-safe mode.
            exposure_scale = self.post_exposure * self._policy_value('exposure', 'exposure_scale', self._atmosphere_budget.get('exposure_scale', 0.93) if self._atmosphere_budget else 0.93)
        else:
            exposure_scale = self.post_exposure
            recipe_name = self._estimate_portrait_light_recipe(lighting_info).get('recipe', 'balanced_soft')
            if background_mode == 'monotone':
                exposure_scale *= 1.18
            elif background_mode == 'rich' and strong_neon:
                exposure_scale *= 1.14
            if recipe_name == 'warm_side':
                exposure_scale *= 1.04
            elif recipe_name == 'neutral_soft':
                exposure_scale *= 1.03
            elif recipe_name == 'cool_env':
                exposure_scale *= 1.02
        if self.debug_dump:
            self._debug_intermediates['pre_tonemap'] = relit.copy()
        if self._using_continuous_policy():
            display_weight = self._policy_value('render_weight', 'display_weight', 0.70)
            tone_strength = self._policy_value('display', 'tone_map_strength', 0.78) * float(np.clip(display_weight, 0.45, 1.0))
            linear_display = np.clip(relit * exposure_scale, 0.0, 1.0)
            mapped_display = tone_map(np.maximum(relit * exposure_scale, 0.0))
            graded = linear_display * (1.0 - tone_strength) + mapped_display * tone_strength
        else:
            graded = tone_map(np.maximum(relit * exposure_scale, 0.0))
        lum = rgb_luminance(graded)
        adaptive_post_saturation = self.post_saturation
        if self._using_continuous_policy():
            adaptive_post_saturation *= float(np.clip(self._policy_value('display', 'saturation', self._budget().get('display_saturation_multiplier', 1.0)), 0.80, 1.25))
        if background_mode == 'monotone':
            adaptive_post_saturation *= 0.92
        elif background_mode == 'rich' and strong_neon:
            adaptive_post_saturation *= 0.92
        if self._using_continuous_policy():
            display_lowkey_chroma_gate = float(np.clip(self._budget().get('lowkey_chroma_direction_gate', 0.0), 0.0, 1.0))
            display_air_skin_guard = float(np.clip(self._budget().get('low_chroma_air_skin_guard', 0.0), 0.0, 1.0))
            sat_region = np.clip(
                1.0 - 0.025 * face_core + 0.075 * hair_region * (1.0 - 0.35 * display_lowkey_chroma_gate) + 0.105 * edge_band,
                0.94,
                1.18,
            ).astype(np.float32)
        else:
            sat_region = np.clip(
                1.0 + 0.055 * face_core + 0.15 * hair_region + 0.20 * edge_band,
                0.96,
                1.34,
            ).astype(np.float32)
        sat_hi = 1.25 if self._using_continuous_policy() else 1.14
        sat_map = np.clip(adaptive_post_saturation * sat_region, 0.76, sat_hi).astype(np.float32)
        graded = lum[..., None] * (1.0 - sat_map[..., None]) + graded * sat_map[..., None]
        contrast_scale = self._policy_value('display', 'local_contrast', 1.0) if self._using_continuous_policy() else 1.0
        post_contrast_eff = 1.0 + (self.post_contrast - 1.0) * float(np.clip(contrast_scale, 0.55, 1.36))
        if self._using_continuous_policy():
            post_contrast_eff = 1.0 + (post_contrast_eff - 1.0) * self._policy_value('render_weight', 'display_weight', 0.70)
        graded = np.clip((graded - 0.5) * post_contrast_eff + 0.5, 0.0, 1.0)
        graded = np.power(np.maximum(graded, 0.0), 1.0 / max(self.post_gamma, 1e-3))
        if not self._using_continuous_policy():
            graded = self._protect_skin_tones(graded, face_core=face_core)
        else:
            final_linear = srgb_to_linear(np.clip(graded, 0.0, 1.0).astype(np.float32))
            final_luma = np.maximum(rgb_luminance(final_linear), 1e-5)
            src_luma = np.maximum(rgb_luminance(np.clip(source_linear, 0.0, None)), 1e-5)
            src_dir = np.clip(source_linear / src_luma[..., None], 0.45, 2.05).astype(np.float32)
            target_linear = np.clip(src_dir * final_luma[..., None], 0.0, 1.0).astype(np.float32)
            target_srgb = linear_to_srgb(target_linear)
            subj_for_face = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
            h_s, w_s = subj_for_face.shape
            yy_s, xx_s = np.mgrid[0:h_s, 0:w_s].astype(np.float32)
            ys_s, xs_s = np.where(subj_for_face > 0.10)
            if xs_s.size >= 16:
                x_rel_s = np.clip((xx_s - float(xs_s.min())) / max(float(xs_s.max() - xs_s.min()), 1.0), 0.0, 1.0)
                y_rel_s = np.clip((yy_s - float(ys_s.min())) / max(float(ys_s.max() - ys_s.min()), 1.0), 0.0, 1.0)
            else:
                x_rel_s = xx_s / max(w_s - 1, 1)
                y_rel_s = yy_s / max(h_s - 1, 1)
            face_prior_s = np.exp(-0.5 * (((x_rel_s - 0.50) / 0.23) ** 2 + ((y_rel_s - 0.34) / 0.26) ** 2)).astype(np.float32)
            face_prior_wide_s = np.exp(-0.5 * (((x_rel_s - 0.50) / 0.40) ** 2 + ((y_rel_s - 0.40) / 0.40) ** 2)).astype(np.float32)
            upper_s = np.clip((0.72 - y_rel_s) / 0.52, 0.0, 1.0).astype(np.float32)
            head_s = np.clip((0.82 - y_rel_s) / 0.48, 0.0, 1.0).astype(np.float32)
            skin_proxy_s = self._estimate_skin_proxy(
                np.clip(source_linear, 0.0, 4.0),
                subj_for_face,
                np.clip(face_core, 0.0, 1.0),
                np.clip(hair_region, 0.0, 1.0),
                np.clip(edge_band, 0.0, 1.0),
            )
            # Final chroma authority is for face identity only.  Do not let the
            # generic skin proxy expand into neck/shoulder/body, otherwise the
            # material-lighting layer is washed back to source gray/tan there.
            identity_face_core = subj_for_face * head_s * np.clip(
                np.maximum(face_prior_s, 0.84 * face_prior_wide_s) * (0.70 + 0.30 * skin_proxy_s),
                0.0,
                1.0,
            )
            eval_like_face = identity_face_core
            eval_like_face *= np.clip(1.0 - 0.55 * np.clip(edge_band, 0.0, 1.0), 0.0, 1.0)
            soft_face_display = feather_mask(np.clip(np.maximum(face_core * subject_mask * face_prior_wide_s * head_s, eval_like_face), 0.0, 1.0), passes=5)
            face_identity_vertical = np.clip((0.74 - y_rel_s) / 0.30, 0.0, 1.0).astype(np.float32)
            soft_face_display = feather_mask(
                np.clip(
                    soft_face_display * face_identity_vertical
                    + face_core * subject_mask * 0.72,
                    0.0,
                    1.0,
                ).astype(np.float32),
                passes=3,
            )
            face_side_display = feather_mask(
                np.clip(soft_face_display * (1.0 - 0.66 * feather_mask(np.clip(identity_face_core, 0.0, 1.0), passes=4)), 0.0, 1.0).astype(np.float32),
                passes=2,
            )
            soft_body_skin = feather_mask(
                np.clip(subject_mask * (1.0 - 0.80 * face_core) * (1.0 - 0.62 * hair_region) * (1.0 - 0.50 * edge_band), 0.0, 1.0),
                passes=6,
            )
            skin_body_display = feather_mask(
                np.clip(subject_mask * skin_proxy_s * (1.0 - 0.70 * hair_region) * (1.0 - 0.58 * edge_band) * (1.0 - 0.72 * soft_face_display), 0.0, 1.0),
                passes=4,
            )
            air_neck_body_display = feather_mask(
                np.clip(
                    subject_mask
                    * np.clip((y_rel_s - 0.52) / 0.30, 0.0, 1.0)
                    * np.clip(1.0 - np.abs(x_rel_s - 0.50) / 0.62, 0.0, 1.0)
                    * (1.0 - 0.58 * hair_region)
                    * (1.0 - 0.46 * edge_band)
                    * (1.0 - 0.42 * soft_face_display),
                    0.0,
                    1.0,
                ).astype(np.float32),
                passes=4,
            )
            skin_display_gate = np.clip(
                identity_face_core * 0.98
                + soft_face_display * 0.16
                - face_side_display * 0.20
                - edge_band * 0.04,
                0.0,
                0.90,
            ).astype(np.float32)
            graded = graded * (1.0 - skin_display_gate[..., None]) + target_srgb * skin_display_gate[..., None]
            final_linear = srgb_to_linear(np.clip(graded, 0.0, 1.0).astype(np.float32))
            final_luma = rgb_luminance(final_linear)
            face_display_mask = soft_face_display > 0.10
            dark_chroma_display = float(np.clip(self._budget().get('lowkey_chroma_direction_gate', 0.0), 0.0, 1.0))
            air_skin_display = float(np.clip(self._budget().get('low_chroma_air_skin_guard', 0.0), 0.0, 1.0))
            warm_scene_pressure = 0.0
            lowkey_chroma_display = dark_chroma_display
            if np.any(face_display_mask):
                face_display_p70 = float(np.percentile(final_luma[face_display_mask], 70.0))
                d_display = self._look_policy.descriptor if self._look_policy is not None and isinstance(self._look_policy.descriptor, dict) else {}
                dark_scene_lift = float(np.clip((0.24 - float(d_display.get('global_luma', 0.25))) / 0.18, 0.0, 1.0))
                warm_scene_pressure = float(np.clip(
                    float(d_display.get('warm_presence', d_display.get('warm_ratio', 0.0)))
                    * float(d_display.get('average_saturation', d_display.get('colorfulness', 0.0))),
                    0.0,
                    1.0,
                ))
                dark_chroma_display = float(np.clip(
                    dark_scene_lift
                    * np.clip((float(d_display.get('colorfulness', d_display.get('average_saturation', 0.25))) - 0.24) / 0.46, 0.0, 1.0)
                    * (0.55 + 0.45 * float(d_display.get('palette_diversity', 0.35))),
                    0.0,
                    1.0,
                ))
                lowkey_chroma_display = float(np.clip(self._budget().get('lowkey_chroma_direction_gate', dark_chroma_display), 0.0, 1.0))
                display_face_mult = float(np.clip(
                    1.08
                    + 0.08 * dark_scene_lift
                    + 0.04 * dark_chroma_display
                    + 0.02 * float(d_display.get('colorfulness', d_display.get('average_saturation', 0.25)))
                    - 0.10 * warm_scene_pressure * (1.0 - dark_scene_lift),
                    1.00,
                    1.18,
                ))
                target_display_face = float(np.clip(
                    self._policy_value('exposure', 'face_target_luma_min', 0.26) * display_face_mult,
                    0.220,
                    0.340,
                ))
                if face_display_p70 < target_display_face:
                    display_gain = float(np.clip(target_display_face / max(face_display_p70, 1e-5), 1.0, 1.28))
                    soft_subject_display = feather_mask(np.clip(subject_mask, 0.0, 1.0), passes=3)
                    display_gain_gate = feather_mask(
                        np.clip(
                            soft_subject_display * 0.18
                            + soft_face_display * (0.10 + 0.03 * dark_chroma_display - 0.08 * warm_scene_pressure * (1.0 - dark_scene_lift))
                            + soft_body_skin * (0.16 + 0.08 * dark_chroma_display + 0.06 * warm_scene_pressure)
                            + np.clip(hair_region + edge_band, 0.0, 1.0) * (0.04 + 0.04 * dark_chroma_display)
                            - edge_band * 0.02,
                            0.0,
                            0.44,
                        ).astype(np.float32),
                        passes=3,
                    )
                    final_linear *= (1.0 + (display_gain - 1.0) * display_gain_gate)[..., None]
                    graded = linear_to_srgb(np.clip(final_linear, 0.0, 1.0))
            if self._look_policy is not None and isinstance(self._look_policy.descriptor, dict):
                d = self._look_policy.descriptor
                hb_final = float(d.get('horizontal_bias', d.get('left_right_luma_diff', 0.0)))
                vb_final = float(d.get('vertical_bias', d.get('top_bottom_luma_diff', 0.0)))
                if abs(hb_final) + abs(vb_final) > 1e-5:
                    final_linear = srgb_to_linear(np.clip(graded, 0.0, 1.0).astype(np.float32))
                    final_luma = rgb_luminance(final_linear)
                    h_f, w_f = final_luma.shape
                    yy_f, xx_f = np.mgrid[0:h_f, 0:w_f].astype(np.float32)
                    u_f = xx_f / max(w_f - 1, 1)
                    v_f = yy_f / max(h_f - 1, 1)
                    subj_f = feather_mask(np.clip(subject_mask, 0.0, 1.0), passes=3)
                    face_f = feather_mask(np.clip(identity_face_core, 0.0, 1.0), passes=5)
                    edge_f = feather_mask(np.clip(edge_band * subject_mask, 0.0, 1.0), passes=3)
                    hair_f = feather_mask(np.clip(hair_region * subject_mask, 0.0, 1.0), passes=4)
                    body_f = np.clip(subj_f * (1.0 - 0.70 * face_f) * (1.0 - 0.20 * hair_f), 0.0, 1.0)
                    face_side_f = np.clip(face_f * (1.0 - 0.64 * feather_mask(face_f, passes=5)), 0.0, 1.0)
                    metric_region = np.clip(0.48 * subj_f + 0.36 * body_f + 0.24 * face_side_f + 0.10 * hair_f + 0.06 * edge_f - 0.12 * face_f, 0.0, 1.0)

                    def _wm_final(values: np.ndarray, weights: np.ndarray, default: float) -> float:
                        ww = np.clip(weights.astype(np.float32), 0.0, None)
                        ss = float(ww.sum())
                        if ss <= 1e-6:
                            return float(default)
                        return float((values * ww).sum() / ss)

                    default_l = float(np.mean(final_luma[subject_mask > 0.08])) if np.any(subject_mask > 0.08) else float(np.mean(final_luma))
                    mean_l = _wm_final(final_luma, metric_region, default_l)
                    cur_lr = _wm_final(final_luma, metric_region * (u_f >= 0.5), mean_l) - _wm_final(final_luma, metric_region * (u_f < 0.5), mean_l)
                    cur_tb = _wm_final(final_luma, metric_region * (v_f >= 0.5), mean_l) - _wm_final(final_luma, metric_region * (v_f < 0.5), mean_l)
                    final_colorfulness = float(d.get('colorfulness', d.get('average_saturation', 0.25)))
                    final_warm_pressure = float(np.clip(float(d.get('warm_presence', d.get('warm_ratio', 0.0))) * final_colorfulness, 0.0, 1.0))
                    final_dark_chroma = float(np.clip(
                        np.clip((0.24 - float(d.get('global_luma', 0.25))) / 0.18, 0.0, 1.0)
                        * np.clip((final_colorfulness - 0.24) / 0.46, 0.0, 1.0)
                        * (0.55 + 0.45 * float(d.get('palette_diversity', 0.35))),
                        0.0,
                        1.0,
                    ))
                    final_lowkey_chroma_gate = float(np.clip(self._budget().get('lowkey_chroma_direction_gate', final_dark_chroma), 0.0, 1.0))
                    direction_target_scale = float(np.clip(0.38 + 0.12 * final_dark_chroma + 0.19 * final_lowkey_chroma_gate - 0.06 * final_warm_pressure * (1.0 - final_dark_chroma), 0.32, 0.70))
                    target_lr = float(np.clip(hb_final * direction_target_scale, -0.135, 0.135))
                    target_tb = float(np.clip(vb_final * direction_target_scale, -0.135, 0.135))
                    delta_lr = float(np.clip(target_lr - cur_lr, -0.30, 0.30))
                    delta_tb = float(np.clip(target_tb - cur_tb, -0.30, 0.30))
                    align_region = feather_mask(
                        np.clip(
                            metric_region * 0.88
                            + body_f * (0.62 + 0.18 * final_lowkey_chroma_gate)
                            + face_side_f * (0.46 + 0.08 * final_lowkey_chroma_gate)
                            + hair_f * (0.62 + 0.12 * final_lowkey_chroma_gate)
                            + edge_f * (0.42 + 0.16 * final_lowkey_chroma_gate)
                            - face_f * (0.10 + 0.06 * final_lowkey_chroma_gate),
                            0.0,
                            0.98,
                        ).astype(np.float32),
                        passes=5,
                    )
                    align_pattern = (2.0 * delta_lr * (u_f - 0.5) + 2.0 * delta_tb * (v_f - 0.5)).astype(np.float32)
                    align_pattern -= _wm_final(align_pattern, metric_region, 0.0)
                    align_delta = np.clip(align_pattern * align_region * 2.05, -0.220, 0.235)
                    aligned_luma = np.clip(final_luma + align_delta, 0.0, 1.0)
                    final_linear *= np.clip(aligned_luma / np.maximum(final_luma, 1e-5), 0.55, 1.55)[..., None]
                    final_luma = rgb_luminance(np.clip(final_linear, 0.0, 1.0))
                    mean_l2 = _wm_final(final_luma, metric_region, default_l)
                    cur_lr2 = _wm_final(final_luma, metric_region * (u_f >= 0.5), mean_l2) - _wm_final(final_luma, metric_region * (u_f < 0.5), mean_l2)
                    cur_tb2 = _wm_final(final_luma, metric_region * (v_f >= 0.5), mean_l2) - _wm_final(final_luma, metric_region * (v_f < 0.5), mean_l2)
                    cur_strength2 = float(np.sqrt(cur_lr2 * cur_lr2 + cur_tb2 * cur_tb2))
                    bg_strength2 = float(np.sqrt(hb_final * hb_final + vb_final * vb_final))
                    desired_strength2 = float(np.clip(bg_strength2 * (0.36 + 0.065 * final_lowkey_chroma_gate), 0.020, 0.128))
                    sign_alignment2 = 1.0
                    if cur_strength2 > 1e-5 and bg_strength2 > 1e-5:
                        sign_alignment2 = float(np.clip((cur_lr2 * hb_final + cur_tb2 * vb_final) / max(cur_strength2 * bg_strength2, 1e-5), -1.0, 1.0))
                    if (cur_strength2 > desired_strength2 * 1.12 or sign_alignment2 < 0.82) and bg_strength2 > 1e-5:
                        target_lr2 = hb_final / bg_strength2 * desired_strength2
                        target_tb2 = vb_final / bg_strength2 * desired_strength2
                        reduce_pattern = (2.0 * (target_lr2 - cur_lr2) * (u_f - 0.5) + 2.0 * (target_tb2 - cur_tb2) * (v_f - 0.5)).astype(np.float32)
                        reduce_pattern -= _wm_final(reduce_pattern, metric_region, 0.0)
                        reduce_delta = np.clip(reduce_pattern * align_region * (1.25 + 0.55 * final_lowkey_chroma_gate), -0.155, 0.155)
                        governed_luma = np.clip(final_luma + reduce_delta, 0.0, 1.0)
                        final_linear *= np.clip(governed_luma / np.maximum(final_luma, 1e-5), 0.68, 1.34)[..., None]
                    graded = linear_to_srgb(np.clip(final_linear, 0.0, 1.0))
            if lowkey_chroma_display > 1e-4:
                final_linear = srgb_to_linear(np.clip(graded, 0.0, 1.0).astype(np.float32))
                final_luma = np.maximum(rgb_luminance(final_linear), 1e-5)
                lift_region = feather_mask(
                    np.clip(
                        subject_mask * 0.52
                        + soft_body_skin * 0.42
                        + skin_body_display * 0.22
                        + np.clip(hair_region * subject_mask, 0.0, 1.0) * 0.10
                        - soft_face_display * 0.22
                        - np.clip(edge_band, 0.0, 1.0) * 0.04,
                        0.0,
                        0.90,
                    ).astype(np.float32),
                    passes=3,
                )
                lifted_luma = np.clip(final_luma + lift_region * (0.014 + 0.008 * lowkey_chroma_display) * lowkey_chroma_display, 0.0, 1.0)
                final_linear *= np.clip(lifted_luma / final_luma, 1.0, 1.09)[..., None]
                graded = linear_to_srgb(np.clip(final_linear, 0.0, 1.0))
            d_exp = self._look_policy.descriptor if self._look_policy is not None and isinstance(self._look_policy.descriptor, dict) else {}
            final_linear = srgb_to_linear(np.clip(graded, 0.0, 1.0).astype(np.float32))
            final_luma = np.maximum(rgb_luminance(final_linear), 1e-5)
            subj_eval_exp = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
            body_eval_exp = feather_mask(
                np.clip(subj_eval_exp * (1.0 - 0.82 * soft_face_display) * (1.0 - 0.55 * np.clip(hair_region, 0.0, 1.0)) * (1.0 - 0.44 * np.clip(edge_band, 0.0, 1.0)), 0.0, 1.0),
                passes=4,
            )
            shoulder_eval_exp = feather_mask(
                np.clip(
                    body_eval_exp
                    * np.clip((y_rel_s - 0.42) / 0.34, 0.0, 1.0)
                    * np.clip(1.0 - np.abs(x_rel_s - 0.50) / 0.64, 0.0, 1.0),
                    0.0,
                    1.0,
                ).astype(np.float32),
                passes=3,
            )
            hand_eval_exp = feather_mask(
                np.clip(
                    body_eval_exp
                    * skin_proxy_s
                    * (0.32 + 0.42 * np.clip((y_rel_s - 0.50) / 0.38, 0.0, 1.0) + 0.38 * np.clip((np.abs(x_rel_s - 0.50) - 0.15) / 0.36, 0.0, 1.0))
                    * (1.0 - 0.55 * soft_face_display),
                    0.0,
                    1.0,
                ).astype(np.float32),
                passes=3,
            )
            upper_body_eval_exp = feather_mask(
                np.clip(
                    body_eval_exp * 0.76
                    + shoulder_eval_exp * 0.62
                    + hand_eval_exp * 0.68
                    + air_neck_body_display * 0.46
                    + skin_body_display * 0.34
                    - soft_face_display * 0.30,
                    0.0,
                    1.0,
                ).astype(np.float32),
                passes=3,
            )
            face_eval_exp = np.clip(soft_face_display, 0.0, 1.0)

            def _weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float, default: float) -> float:
                ww = np.clip(weights.astype(np.float32), 0.0, None).reshape(-1)
                vv = values.astype(np.float32).reshape(-1)
                keep = ww > 1e-5
                if np.count_nonzero(keep) < 16:
                    return float(default)
                vv = vv[keep]
                ww = ww[keep]
                order = np.argsort(vv)
                vv = vv[order]
                ww = ww[order]
                cdf = np.cumsum(ww)
                cutoff = float(cdf[-1]) * float(np.clip(pct, 0.0, 100.0)) / 100.0
                return float(vv[int(np.clip(np.searchsorted(cdf, cutoff), 0, vv.size - 1))])

            subj_p70_exp = _weighted_percentile(final_luma, subj_eval_exp, 70.0, float(np.percentile(final_luma, 70.0)))
            face_p70_exp = _weighted_percentile(final_luma, face_eval_exp, 70.0, subj_p70_exp)
            body_p70_exp = _weighted_percentile(final_luma, upper_body_eval_exp, 70.0, subj_p70_exp)
            bg_p50_exp = float(d_exp.get('luma_p50', d_exp.get('p50_luma', d_exp.get('global_luma', 0.25))))
            bg_p95_exp = float(d_exp.get('luma_p95', d_exp.get('p95_luma', d_exp.get('luma_p90', 0.55))))
            target_exp = float(np.clip(0.255 + 0.55 * bg_p50_exp + 0.08 * max(bg_p95_exp - bg_p50_exp, 0.0), 0.235, 0.500))
            measured_exp = float(0.65 * face_p70_exp + 0.35 * max(body_p70_exp, subj_p70_exp * 0.85))
            exposure_region = feather_mask(
                np.clip(
                    subj_eval_exp * 0.42
                    + face_eval_exp * 0.22
                    + upper_body_eval_exp * 0.52
                    - np.clip(edge_band, 0.0, 1.0) * 0.10,
                    0.0,
                    0.94,
                ).astype(np.float32),
                passes=4,
            )
            if measured_exp < target_exp - 0.080:
                lift_amt = float(np.clip((target_exp - 0.070) - measured_exp, 0.0, 0.022))
                lifted_luma = np.clip(final_luma + lift_amt * exposure_region, 0.0, 1.0)
                final_linear *= np.clip(lifted_luma / final_luma, 1.0, 1.06)[..., None]
                graded = linear_to_srgb(np.clip(final_linear, 0.0, 1.0))
            elif measured_exp > target_exp + 0.040:
                reduce_amt = float(np.clip(measured_exp - (target_exp + 0.025), 0.0, 0.032))
                reduced_luma = np.clip(final_luma - reduce_amt * exposure_region * 0.62, 0.0, 1.0)
                final_linear *= np.clip(reduced_luma / final_luma, 0.91, 1.0)[..., None]
                graded = linear_to_srgb(np.clip(final_linear, 0.0, 1.0))
            final_linear = srgb_to_linear(np.clip(graded, 0.0, 1.0).astype(np.float32))
            final_luma = np.maximum(rgb_luminance(final_linear), 1e-5)
            body_p70_after = _weighted_percentile(final_luma, upper_body_eval_exp, 70.0, body_p70_exp)
            face_p70_after = _weighted_percentile(final_luma, face_eval_exp, 70.0, face_p70_exp)
            hand_p70_after = _weighted_percentile(final_luma, hand_eval_exp, 70.0, body_p70_after)
            shoulder_p70_after = _weighted_percentile(final_luma, shoulder_eval_exp + air_neck_body_display * 0.35, 70.0, body_p70_after)
            body_face_gap = float(np.clip(self._policy_value('exposure', 'body_face_luma_gap_target', 0.100), 0.060, 0.150))
            subject_sync = float(np.clip(self._policy_value('exposure', 'whole_subject_sync_strength', 0.38), 0.18, 0.66))
            target_body_sync = float(np.clip(
                min(face_p70_after - body_face_gap, target_exp + 0.075),
                max(body_p70_after, target_exp - 0.095),
                max(target_exp + 0.085, face_p70_after - 0.035),
            ))
            body_sync_gap = float(np.clip(target_body_sync - body_p70_after, 0.0, 0.040))
            if body_sync_gap > 1e-5:
                whole_body_sync_region = feather_mask(
                    np.clip(
                        upper_body_eval_exp * 0.82
                        + hand_eval_exp * 0.48
                        + shoulder_eval_exp * 0.40
                        + air_neck_body_display * 0.46
                        + skin_body_display * 0.28
                        + np.clip(edge_band, 0.0, 1.0) * 0.10
                        - soft_face_display * 0.32
                        - np.clip(hair_region, 0.0, 1.0) * 0.12,
                        0.0,
                        0.92,
                    ).astype(np.float32),
                    passes=4,
                )
                synced_luma = np.clip(final_luma + body_sync_gap * subject_sync * whole_body_sync_region, 0.0, 1.0)
                final_linear *= np.clip(synced_luma / final_luma, 1.0, 1.08)[..., None]
                graded = linear_to_srgb(np.clip(final_linear, 0.0, 1.0))
                final_luma = np.maximum(rgb_luminance(final_linear), 1e-5)
                body_p70_after = _weighted_percentile(final_luma, upper_body_eval_exp, 70.0, body_p70_after)
                face_p70_after = _weighted_percentile(final_luma, face_eval_exp, 70.0, face_p70_after)
                hand_p70_after = _weighted_percentile(final_luma, hand_eval_exp, 70.0, hand_p70_after)
                shoulder_p70_after = _weighted_percentile(final_luma, shoulder_eval_exp + air_neck_body_display * 0.35, 70.0, shoulder_p70_after)
            face_excess = float(np.clip(face_p70_after - max(body_p70_after + body_face_gap + 0.055, target_exp + 0.115), 0.0, 0.160))
            if face_excess > 1e-5 and face_p70_after > 0.50:
                face_balance_region = feather_mask(
                    np.clip(
                        soft_face_display * 0.74
                        + face_eval_exp * 0.40
                        - upper_body_eval_exp * 0.20
                        - np.clip(edge_band, 0.0, 1.0) * 0.10
                        - np.clip(hair_region, 0.0, 1.0) * 0.10,
                        0.0,
                        0.88,
                    ).astype(np.float32),
                    passes=3,
                )
                balanced_luma = np.clip(final_luma - face_excess * face_balance_region, 0.0, 1.0)
                final_linear *= np.clip(balanced_luma / final_luma, 0.86, 1.0)[..., None]
                graded = linear_to_srgb(np.clip(final_linear, 0.0, 1.0))
            # Final face-core chroma authority.  All display and direction
            # operations above are allowed to change luma, but the core face color
            # direction is anchored back to the source so background chroma stays
            # on face-side, hair, edge and clothing rather than becoming a filter.
            final_linear = srgb_to_linear(np.clip(graded, 0.0, 1.0).astype(np.float32))
            final_luma = np.maximum(rgb_luminance(final_linear), 1e-5)
            target_linear = np.clip(src_dir * final_luma[..., None], 0.0, 1.0).astype(np.float32)
            display_warm_support = float(np.clip(
                float(d_exp.get('warm_presence', d_exp.get('warm_ratio', 0.0)))
                * float(d_exp.get('average_saturation', d_exp.get('colorfulness', 0.0)))
                * 2.4,
                0.0,
                1.0,
            ))
            display_cool_support = float(np.clip(
                float(d_exp.get('cool_presence', d_exp.get('cool_ratio', 0.0)))
                * float(d_exp.get('average_saturation', d_exp.get('colorfulness', 0.0)))
                * 2.4,
                0.0,
                1.0,
            ))
            display_ambient_warmth = float(np.clip(d_exp.get('ambient_warmth', 0.0), -1.0, 1.0))
            display_keep = float(np.clip(1.0 - air_skin_display * (0.86 + 0.20 * display_cool_support), 0.18, 1.0))
            display_air_dir = 1.0 + (src_dir - 1.0) * display_keep
            display_bg_dir = np.array([
                1.0 + 0.038 * display_warm_support - 0.046 * display_cool_support + 0.030 * max(display_ambient_warmth, 0.0),
                1.0 + 0.006 * display_cool_support,
                1.0 + 0.046 * display_cool_support - 0.024 * display_warm_support + 0.030 * max(-display_ambient_warmth, 0.0),
            ], dtype=np.float32)
            display_bg_dir = display_bg_dir / max(float(np.dot(display_bg_dir, LUMA)), 1e-5)
            display_air_dir = display_air_dir * (1.0 - 0.34 * air_skin_display) + display_bg_dir.reshape(1, 1, 3) * (0.34 * air_skin_display)
            if air_skin_display > 1e-5:
                red_ref_display = 0.58 * display_air_dir[..., 1] + 0.42 * display_air_dir[..., 2]
                red_limit_display = 1.015 + 0.28 * display_warm_support + 0.06 * (1.0 - air_skin_display)
                display_air_dir[..., 0] = np.minimum(display_air_dir[..., 0], red_ref_display * red_limit_display)
                display_air_l = np.maximum(np.sum(display_air_dir * LUMA.reshape(1, 1, 3), axis=-1), 1e-5)
                display_air_dir = np.clip(display_air_dir / display_air_l[..., None], 0.48, 2.05).astype(np.float32)
            air_target_linear = np.clip(display_air_dir * final_luma[..., None], 0.0, 1.0).astype(np.float32)
            face_chroma_gate = feather_mask(
                np.clip(
                    soft_face_display * 1.14
                    + skin_body_display * (0.030 + 0.040 * dark_chroma_display + 0.030 * warm_scene_pressure) * (1.0 - 0.78 * lowkey_chroma_display)
                    - face_side_display * (0.22 + 0.16 * lowkey_chroma_display)
                    - np.clip(edge_band, 0.0, 1.0) * 0.06,
                    0.0,
                    0.92,
                ).astype(np.float32),
                passes=2,
            )
            final_linear = final_linear * (1.0 - face_chroma_gate[..., None]) + target_linear * face_chroma_gate[..., None]
            if air_skin_display > 1e-5:
                air_body_chroma_gate = feather_mask(
                    np.clip(
                        skin_body_display * (0.78 + 0.22 * display_cool_support - 0.08 * display_warm_support)
                        + soft_body_skin * (0.26 + 0.10 * display_cool_support - 0.05 * display_warm_support)
                        + air_neck_body_display * (0.86 + 0.22 * display_cool_support - 0.08 * display_warm_support)
                        - soft_face_display * 0.26
                        - np.clip(edge_band, 0.0, 1.0) * 0.08,
                        0.0,
                        0.95,
                    ).astype(np.float32),
                    passes=2,
                ) * air_skin_display
                final_linear = final_linear * (1.0 - air_body_chroma_gate[..., None]) + air_target_linear * air_body_chroma_gate[..., None]
                final_luma_air = np.maximum(rgb_luminance(final_linear), 1e-5)
                final_dir_air = np.clip(final_linear / final_luma_air[..., None], 0.30, 2.60)
                red_ref_air = 0.58 * final_dir_air[..., 1] + 0.42 * final_dir_air[..., 2]
                red_limit_air = 1.005 + 0.30 * display_warm_support + 0.06 * (1.0 - air_skin_display)
                capped_dir_air = final_dir_air.copy()
                capped_dir_air[..., 0] = np.minimum(capped_dir_air[..., 0], red_ref_air * red_limit_air)
                capped_l_air = np.maximum(np.sum(capped_dir_air * LUMA.reshape(1, 1, 3), axis=-1), 1e-5)
                capped_dir_air = np.clip(capped_dir_air / capped_l_air[..., None], 0.36, 2.70).astype(np.float32)
                warm_excess_air = smoothstep(0.006, 0.105, final_dir_air[..., 0] - red_ref_air * red_limit_air)
                warm_residual_gate = feather_mask(
                    np.clip(
                        warm_excess_air
                        * air_skin_display
                        * (
                            soft_body_skin * (0.54 + 0.12 * display_cool_support)
                            + skin_body_display * 0.72
                            + air_neck_body_display * 1.04
                            - soft_face_display * 0.38
                            - np.clip(edge_band, 0.0, 1.0) * 0.10
                        ),
                        0.0,
                        0.74,
                    ).astype(np.float32),
                    passes=2,
                )
                final_linear = final_linear * (1.0 - warm_residual_gate[..., None]) + (capped_dir_air * final_luma_air[..., None]) * warm_residual_gate[..., None]
            final_luma = np.maximum(rgb_luminance(np.clip(final_linear, 0.0, 1.0)), 1e-5)
            final_face_p70 = _weighted_percentile(final_luma, face_eval_exp, 70.0, face_p70_after)
            final_body_p70 = _weighted_percentile(final_luma, upper_body_eval_exp, 70.0, body_p70_after)
            face_ceiling = float(np.clip(
                max(final_body_p70 + body_face_gap + 0.060, target_exp + 0.120),
                0.395,
                0.520,
            ))
            final_face_excess = float(np.clip(final_face_p70 - face_ceiling, 0.0, 0.220))
            if final_face_excess > 1e-5:
                final_face_gate = feather_mask(
                    np.clip(
                        face_eval_exp * 0.82
                        + soft_face_display * 0.52
                        - upper_body_eval_exp * 0.24
                        - np.clip(edge_band, 0.0, 1.0) * 0.10
                        - np.clip(hair_region, 0.0, 1.0) * 0.10,
                        0.0,
                        0.92,
                    ).astype(np.float32),
                    passes=3,
                )
                governed_final_luma = np.clip(final_luma - final_face_excess * final_face_gate, 0.0, 1.0)
                final_linear *= np.clip(governed_final_luma / final_luma, 0.82, 1.0)[..., None]
            graded = linear_to_srgb(np.clip(final_linear, 0.0, 1.0))
        if self.debug_dump:
            self._debug_intermediates['post_tonemap'] = graded.copy()
        return srgb_to_linear(np.clip(graded, 0.0, 1.0).astype(np.float32))


    # ------------------------------------------------------------------
    # V32: modular separation layer for the legacy lighting chain.
    # This does NOT delete the old physically-inspired relight logic.
    # It only routes style-specific correction through one isolated block,
    # then reconciles luma/chroma/detail with explicit region ownership.
    # ------------------------------------------------------------------
