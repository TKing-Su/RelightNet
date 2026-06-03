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

class RendererPBREnvironmentMixin:
    def _compute_switchlight_pbr_env_pass(
        self,
        base_subject: np.ndarray,
        albedo_linear: np.ndarray,
        subject_mask: np.ndarray,
        N: np.ndarray,
        V: np.ndarray,
        P: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        specular_map: np.ndarray,
        roughness_map: np.ndarray,
        F0: np.ndarray,
        lighting_info: LightingInfo,
        source_shape: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """SwitchLight-inspired deterministic PBR environment render.

        SwitchLight's useful non-model idea is: relighting should be formulated as
        inverse-rendered intrinsics + target HDRI + physically-based re-rendering.
        This pass implements that idea directly with the maps already available in
        the user's pipeline: albedo/eight-color, normal, roughness, specular/F0,
        alpha and a target background-derived HDRI field.

        It computes two physically separated terms:
        1) Lambertian diffuse irradiance from a convolved low-frequency HDRI.
        2) Cook-Torrance microfacet specular from roughness/F0/specular maps.
        The output is intentionally added as a visible PBR layer so that the light
        bends over the portrait volume rather than being flattened by color grading.
        """
        field = getattr(lighting_info, 'gradient_field', {}) or {}
        if not isinstance(field, dict) or not bool(field.get('enabled', False)):
            return np.zeros_like(base_subject, dtype=np.float32)
        grid = field.get('grid_colors', None)
        if grid is None or not np.any(subject_mask > 0.08):
            return np.zeros_like(base_subject, dtype=np.float32)
        try:
            g = np.asarray(grid, dtype=np.float32)
        except Exception:
            return np.zeros_like(base_subject, dtype=np.float32)
        if g.ndim != 3 or g.shape[-1] != 3:
            return np.zeros_like(base_subject, dtype=np.float32)

        h, w = subject_mask.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ys, xs = np.where(subject_mask > 0.08)
        y0, y1 = float(ys.min()), float(ys.max())
        x0, x1 = float(xs.min()), float(xs.max())
        x_rel = np.clip((xx - x0) / max(x1 - x0, 1.0), 0.0, 1.0).astype(np.float32)
        y_rel = np.clip((yy - y0) / max(y1 - y0, 1.0), 0.0, 1.0).astype(np.float32)

        # A proxy ellipsoid normal makes the entire bust/shoulders respond like a
        # physical volume even when the provided normal map is face-centered or flat.
        sx = np.clip((x_rel - 0.50) * 2.05, -1.0, 1.0)
        sy = np.clip((0.52 - y_rel) * 1.62, -1.0, 1.0)
        z = np.sqrt(np.clip(1.0 - 0.62 * sx * sx - 0.20 * sy * sy, 0.06, 1.0))
        N_proxy = safe_norm(np.stack([0.82 * sx, 0.38 * sy, z], axis=-1).astype(np.float32))
        proxy_w = np.clip(0.38 + 0.24 * (1.0 - face_core) + 0.10 * edge_band + 0.08 * hair_region, 0.28, 0.74).astype(np.float32)
        Np = safe_norm(N * (1.0 - proxy_w[..., None]) + N_proxy * proxy_w[..., None])
        NdotV = np.clip(np.sum(Np * V, axis=-1), 0.0, 1.0).astype(np.float32)

        recipe = self._estimate_portrait_light_recipe(lighting_info)
        recipe_name = str(recipe.get('recipe', 'balanced_soft'))
        colorfulness = float(np.clip(recipe.get('colorfulness', field.get('colorfulness', 0.25)), 0.0, 1.0))
        local_contrast = float(np.clip(recipe.get('local_contrast', field.get('local_contrast', 0.35)), 0.0, 1.0))
        side_sign = float(recipe.get('side_sign', 1.0))
        side_gate = self._compute_signed_side_mask(P, side_sign, subject_mask, power=0.78)
        opp_gate = self._compute_signed_side_mask(P, -side_sign, subject_mask, power=0.78)
        u = np.clip(0.50 + 0.42 * Np[..., 0], 0.0, 1.0).astype(np.float32)
        v = np.clip(0.50 - 0.42 * Np[..., 1], 0.0, 1.0).astype(np.float32)
        R = safe_norm(2.0 * Np * np.sum(Np * V, axis=-1, keepdims=True) - V)
        ur = np.clip(0.50 + 0.48 * R[..., 0], 0.0, 1.0).astype(np.float32)
        vr = np.clip(0.50 - 0.48 * R[..., 1], 0.0, 1.0).astype(np.float32)

        env_diff = self._bilinear_sample_field_grid(g, u, v)
        env_spec = self._bilinear_sample_field_grid(g, ur, vr)
        mean_color = np.array(getattr(lighting_info, 'global_mean_color', np.mean(g.reshape(-1, 3), axis=0)), dtype=np.float32)
        env_diff = 0.74 * env_diff + 0.26 * mean_color.reshape(1, 1, 3)
        env_spec = 0.58 * env_spec + 0.42 * mean_color.reshape(1, 1, 3)

        ndotl = np.clip(0.30 + 0.70 * Np[..., 2], 0.0, 1.0).astype(np.float32)
        diffuse = albedo_linear * env_diff * ndotl[..., None]

        rough = np.clip(roughness_map, 0.08, 0.95).astype(np.float32)
        spec_power = np.clip((1.0 - rough) ** 2.0, 0.0, 1.0) * np.clip(specular_map, 0.0, 1.2)
        fresnel = np.power(np.clip(1.0 - NdotV, 0.0, 1.0), 3.0).astype(np.float32)
        spec = env_spec * np.clip(F0 + fresnel[..., None] * (1.0 - F0), 0.0, 1.0) * spec_power[..., None]

        region = np.clip(subject_mask * (0.52 + 0.18 * side_gate + 0.12 * opp_gate + 0.16 * edge_band + 0.12 * hair_region), 0.0, 1.0)
        strength = 0.032 + 0.035 * colorfulness + 0.028 * local_contrast + (0.012 if recipe_name != 'balanced_soft' else 0.0)
        out = (0.72 * diffuse + 0.48 * spec) * region[..., None] * strength
        return np.clip(out, 0.0, 0.36).astype(np.float32)
