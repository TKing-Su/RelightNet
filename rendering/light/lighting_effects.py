from __future__ import annotations

from typing import List
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

class RendererLightingEffectsMixin:
    def _apply_pose_aware_diffuse_sculpt(
        self,
        relit: np.ndarray,
        base_subject: np.ndarray,
        N: np.ndarray,
        P: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        key_light: PortraitLight,
    ) -> np.ndarray:
        """Add visible pose-aware diffuse sculpting without adding specular gloss."""
        if key_light is None or not np.any(subject_mask > 0.08):
            return relit

        L = safe_norm(np.array(key_light.direction, dtype=np.float32))
        NdotL_raw = np.sum(N * L.reshape(1, 1, 3), axis=-1).astype(np.float32)
        lit = np.clip((NdotL_raw + 0.04) / 1.02, 0.0, 1.0)
        lit = np.power(lit, 0.94).astype(np.float32)
        shade = np.clip((0.16 - NdotL_raw) / 0.84, 0.0, 1.0)
        shade = np.power(shade, 0.88).astype(np.float32)

        # Keep the effect mostly on face/neck/shoulder surfaces; hair and edges get
        # only a little contribution from this diffuse sculpt layer.
        sculpt_gate = np.clip(
            subject_mask * (0.36 + 0.70 * face_core + 0.24 * (1.0 - face_core))
            + 0.18 * hair_region
            + 0.12 * edge_band,
            0.0,
            1.0,
        ).astype(np.float32)

        key_color = brighten_preserve_hue(np.array(key_light.color, dtype=np.float32), 0.32)
        key_color = np.clip(key_color, 0.0, 3.0).reshape(1, 1, 3)

        # Directional darkening is what makes the light direction readable.  The
        # addition is diffuse and color-preserving, not a mirror highlight.
        shadow_strength = 0.055
        light_strength = 0.085
        out = relit * (1.0 - shadow_strength * shade[..., None] * sculpt_gate[..., None])
        out += base_subject * key_color * (light_strength * lit[..., None] * sculpt_gate[..., None])
        return np.clip(out, 0.0, 8.0).astype(np.float32)


    def _compute_visible_virtual_specular_boost(
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
        lights: List[PortraitLight],
    ) -> np.ndarray:
        """Art-directed but normal-based specular layer for the virtual rig.

        The physically based GGX layer is still kept in the main loop, but for
        portraits it is often too subtle because skin F0 is intentionally low.
        This layer makes the virtual key/rim lights visibly reflect on hair,
        cheekbone, nose bridge and alpha edges while still depending on N, V,
        roughness, specular_map and light direction.
        """
        strength = float(np.clip(getattr(self, 'specular_boost_strength', 0.0), 0.0, 0.80))
        if strength <= 1e-6 or not lights or not np.any(subject_mask > 0.08):
            return np.zeros_like(base_subject, dtype=np.float32)

        out = np.zeros_like(base_subject, dtype=np.float32)
        NdotV = np.clip(np.sum(N * V, axis=-1), 0.0, 1.0).astype(np.float32)
        facing = np.clip(N[..., 2], 0.0, 1.0).astype(np.float32)
        rough = np.clip(roughness_map.astype(np.float32), 0.10, 0.90)
        spec = np.clip(specular_map.astype(np.float32), 0.0, 1.0)

        # Face gets a controlled glossy sheen; hair and edges get stronger reflection.
        region_gate = np.clip(
            0.20 * subject_mask
            + 0.22 * face_core
            + 0.95 * hair_region
            + 1.15 * edge_band,
            0.0,
            1.45,
        ).astype(np.float32)
        # Fresnel-like grazing term makes rim/edge reflection readable.
        grazing = np.power(np.clip(1.0 - NdotV, 0.0, 1.0), 1.35).astype(np.float32)
        spec_level = np.clip(0.28 + 1.05 * spec + 0.55 * grazing, 0.0, 1.65).astype(np.float32)

        for i, light in enumerate(lights):
            lname = str(light.name)
            if not lname.startswith('virtual_'):
                continue
            L = safe_norm(np.array(light.direction, dtype=np.float32))
            Lf = np.ones_like(N, dtype=np.float32) * L.reshape(1, 1, 3)
            H = safe_norm(Lf + V)
            NdotL = np.clip(np.sum(N * Lf, axis=-1), 0.0, 1.0).astype(np.float32)
            NdotH = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0).astype(np.float32)
            VdotH = np.clip(np.sum(V * H, axis=-1), 0.0, 1.0).astype(np.float32)

            # Variable exponent: low roughness gives tight highlight, high roughness
            # gives a wider sheen. np.exp/log is stable for per-pixel exponent.
            tight_exp = np.clip(18.0 + 92.0 * (1.0 - rough), 18.0, 110.0)
            broad_exp = np.clip(5.0 + 24.0 * (1.0 - rough), 5.0, 34.0)
            tight = np.exp(np.log(np.maximum(NdotH, 1e-5)) * tight_exp).astype(np.float32)
            broad = np.exp(np.log(np.maximum(NdotH, 1e-5)) * broad_exp).astype(np.float32)

            light_color = np.array(light.color, dtype=np.float32)
            light_color = brighten_preserve_hue(light_color, max(float(np.dot(light_color, LUMA)), 0.24))
            le = light_color.reshape(1, 1, 3) * float(light.intensity)
            light_scale = float(light.specular_scale)

            if lname.startswith('virtual_key'):
                role_scale = 0.82
                role_gate = np.clip(0.35 + 0.65 * facing, 0.0, 1.0)
            elif lname.startswith('virtual_rim') or lname.startswith('virtual_kicker'):
                role_scale = 1.18
                role_gate = np.clip(0.25 + 0.75 * (hair_region + edge_band + grazing), 0.0, 1.45)
            else:
                role_scale = 0.28
                role_gate = np.clip(0.18 + 0.82 * hair_region, 0.0, 1.0)

            fres = np.clip(0.20 + 0.80 * np.power(1.0 - VdotH, 3.0), 0.0, 1.0).astype(np.float32)
            lobe = (0.72 * tight + 0.20 * broad) * np.power(np.clip(NdotL, 0.0, 1.0), 0.42)
            lobe *= role_gate * region_gate * spec_level * (0.35 + 0.65 * fres)
            # Avoid turning flat dark clothing into chrome; keep highlight connected
            # to existing source shape and subject-facing geometry.
            lobe *= np.clip(0.42 + 0.58 * facing, 0.0, 1.0)
            out += le * lobe[..., None] * strength * light_scale * role_scale

        return np.clip(out, 0.0, 3.0).astype(np.float32)


    def _apply_structural_style_boost(
        self,
        relit: np.ndarray,
        base_subject: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        P: np.ndarray,
        lighting_info: LightingInfo,
    ) -> np.ndarray:
        """Visible geometric lighting patterns.

        Previous version was too conservative: it added only tiny color terms, then
        later exposure/color matching mostly normalized the difference away.
        This version changes the *light map* itself: one side/top is explicitly
        brightened while the opposite/lower region is darkened, so side/top/movie
        lighting remains visible after tone mapping.
        """
        if not np.any(subject_mask > 0.08):
            return relit

        pattern = self._select_lighting_pattern(lighting_info)
        self._last_selected_lighting_pattern = pattern
        lights = getattr(lighting_info, 'lights', None) or []
        if lights:
            try:
                raw_key_dir = safe_norm(np.array(lights[0]['direction'], dtype=np.float32))
                key_color = np.array(lights[0]['color'], dtype=np.float32) * float(lights[0].get('intensity', 1.0))
            except Exception:
                raw_key_dir = np.array([-0.55, -0.25, 0.80], dtype=np.float32)
                key_color = np.array(lighting_info.global_mean_color, dtype=np.float32)
        else:
            raw_key_dir = np.array([-0.55, -0.25, 0.80], dtype=np.float32)
            key_color = np.array(lighting_info.global_mean_color, dtype=np.float32)

        key_dir = self._forced_pattern_direction(pattern, raw_key_dir)
        side_sign = self._resolve_key_side_sign(key_dir)

        key_color = brighten_preserve_hue(
            np.clip(key_color.astype(np.float32), 0.0, None),
            max(float(np.dot(key_color, LUMA)), 0.24),
        )
        bg_color = np.array(lighting_info.global_mean_color, dtype=np.float32)
        bg_color = brighten_preserve_hue(np.clip(bg_color.astype(np.float32), 0.0, None), max(float(np.dot(bg_color, LUMA)), 0.18))

        masks = self._build_subject_region_masks(P, subject_mask)
        xn = masks['xn']
        center = masks['center']
        upper = masks['upper']
        lower = masks['lower']
        edge = masks['edge']

        lit_side = self._compute_signed_side_mask(P, side_sign, subject_mask, power=0.85)
        shadow_side = self._compute_signed_side_mask(P, -side_sign, subject_mask, power=0.95)
        rim_side = np.clip(edge * (0.30 + 0.70 * shadow_side), 0.0, 1.0)
        upper_soft = np.clip(0.25 * center + 0.75 * upper, 0.0, 1.0)
        lower_soft = np.clip(0.25 * center + 0.75 * lower, 0.0, 1.0)

        # Keep face readable but do not flatten it: face can still receive structure.
        readable_face = 0.72 + 0.28 * face_core
        nonface_shadow_gate = 0.72 + 0.28 * (1.0 - face_core)

        # These numbers are intentionally much stronger than the previous version.
        # They affect luminance contrast first, then add a smaller colored-light term.
        if pattern == 'side':
            bright = np.clip(0.75 * lit_side + 0.22 * upper_soft + 0.20 * rim_side, 0.0, 1.0) * subject_mask
            shade = np.clip(0.85 * shadow_side + 0.18 * lower_soft, 0.0, 1.0) * subject_mask
            relit *= (1.0 + 0.34 * bright[..., None] * readable_face[..., None] - 0.30 * shade[..., None] * nonface_shadow_gate[..., None])
            relit += base_subject * key_color.reshape(1, 1, 3) * bright[..., None] * 0.17
            relit += base_subject * bg_color.reshape(1, 1, 3) * rim_side[..., None] * 0.055

        elif pattern == 'top':
            bright = np.clip(0.88 * upper_soft + 0.24 * center + 0.20 * hair_region, 0.0, 1.0) * subject_mask
            shade = np.clip(0.85 * lower_soft + 0.28 * shadow_side, 0.0, 1.0) * subject_mask
            relit *= (1.0 + 0.38 * bright[..., None] - 0.32 * shade[..., None] * (0.75 + 0.25 * face_core[..., None]))
            relit += base_subject * key_color.reshape(1, 1, 3) * bright[..., None] * 0.15
            relit += base_subject * bg_color.reshape(1, 1, 3) * rim_side[..., None] * 0.035

        elif pattern == 'cinematic':
            bright = np.clip(0.62 * lit_side + 0.38 * upper_soft + 0.26 * rim_side, 0.0, 1.0) * subject_mask
            shade = np.clip(0.75 * shadow_side + 0.20 * lower_soft, 0.0, 1.0) * subject_mask
            relit *= (1.0 + 0.36 * bright[..., None] * readable_face[..., None] - 0.34 * shade[..., None] * nonface_shadow_gate[..., None])
            relit += base_subject * key_color.reshape(1, 1, 3) * bright[..., None] * 0.16
            relit += base_subject * bg_color.reshape(1, 1, 3) * rim_side[..., None] * (0.065 + 0.035 * hair_region[..., None])

        elif pattern == 'rembrandt':
            bright = np.clip(0.74 * lit_side + 0.26 * upper_soft, 0.0, 1.0) * subject_mask
            shade = np.clip(0.82 * shadow_side + 0.18 * lower_soft, 0.0, 1.0) * subject_mask
            triangle = np.clip(shadow_side * upper_soft * center * face_core, 0.0, 1.0)
            relit *= (1.0 + 0.32 * bright[..., None] - 0.34 * shade[..., None] * nonface_shadow_gate[..., None])
            relit += base_subject * key_color.reshape(1, 1, 3) * (bright[..., None] * 0.14 + triangle[..., None] * 0.18)
            relit += base_subject * bg_color.reshape(1, 1, 3) * rim_side[..., None] * 0.050

        elif pattern == 'split':
            left = np.clip(-xn, 0.0, 1.0) * subject_mask
            right = np.clip(xn, 0.0, 1.0) * subject_mask
            cool = np.array([0.42, 0.78, 1.18], dtype=np.float32).reshape(1, 1, 3)
            warm = np.array([1.20, 0.42, 0.88], dtype=np.float32).reshape(1, 1, 3)
            side_light = np.clip(left + right, 0.0, 1.0)
            relit *= (1.0 + 0.22 * side_light[..., None] - 0.12 * center[..., None] * (1.0 - 0.55 * face_core[..., None]))
            relit += base_subject * (cool * left[..., None] + warm * right[..., None]) * 0.16
            relit += base_subject * bg_color.reshape(1, 1, 3) * rim_side[..., None] * 0.040

        else:  # natural: still add a mild but visible directional cue
            bright = np.clip(0.52 * lit_side + 0.18 * upper_soft + 0.12 * rim_side, 0.0, 1.0) * subject_mask
            shade = np.clip(0.50 * shadow_side + 0.12 * lower_soft, 0.0, 1.0) * subject_mask
            relit *= (1.0 + 0.18 * bright[..., None] - 0.14 * shade[..., None] * nonface_shadow_gate[..., None])
            relit += base_subject * key_color.reshape(1, 1, 3) * bright[..., None] * 0.08

        return np.clip(relit, 0.0, 8.0).astype(np.float32)



    def _classify_light_hue(self, color: np.ndarray) -> str:
        hue, sat, _ = rgb_to_hsv_approx(color)
        if self.extractor._is_cool_hue(hue, sat):
            return 'cool'
        if self.extractor._is_warm_hue(hue, sat):
            return 'warm'
        return 'neutral'


    def _compute_signed_side_mask(self, P: np.ndarray, side_sign: float, subject_mask: np.ndarray, power: float = 1.0) -> np.ndarray:
        x = P[..., 0].astype(np.float32)
        if not np.any(subject_mask > 0.08):
            return np.ones_like(subject_mask, dtype=np.float32)
        scale = float(np.percentile(np.abs(x[subject_mask > 0.08]), 88.0))
        scale = max(scale, 1e-4)
        xn = np.clip(x / scale, -1.0, 1.0)
        mask = 0.5 + 0.5 * float(np.sign(side_sign if abs(side_sign) > 1e-6 else 1.0)) * xn
        mask = np.clip(mask, 0.0, 1.0)
        return np.power(mask, max(power, 1e-4)).astype(np.float32)



    @staticmethod
    def _bilinear_sample_field_grid(grid: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Sample a compact [H,W,C] background light-field grid at normalized UVs."""
        g = np.asarray(grid, dtype=np.float32)
        if g.ndim == 2:
            g = g[..., None]
        gh, gw = g.shape[:2]
        if gh < 1 or gw < 1:
            out_shape = u.shape + ((g.shape[2] if g.ndim == 3 else 1),)
            return np.zeros(out_shape, dtype=np.float32)
        uu = np.clip(u.astype(np.float32), 0.0, 1.0) * max(gw - 1, 0)
        vv = np.clip(v.astype(np.float32), 0.0, 1.0) * max(gh - 1, 0)
        x0 = np.floor(uu).astype(np.int32)
        y0 = np.floor(vv).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, gw - 1)
        y1 = np.clip(y0 + 1, 0, gh - 1)
        tx = (uu - x0.astype(np.float32))[..., None]
        ty = (vv - y0.astype(np.float32))[..., None]
        c00 = g[y0, x0]
        c10 = g[y0, x1]
        c01 = g[y1, x0]
        c11 = g[y1, x1]
        c0 = c00 * (1.0 - tx) + c10 * tx
        c1 = c01 * (1.0 - tx) + c11 * tx
        out = c0 * (1.0 - ty) + c1 * ty
        return out.astype(np.float32)
