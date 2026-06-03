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

class RendererHDRIArcMixin:
    def _compute_hdri_spheremap_late_arc(
        self,
        relit: np.ndarray,
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
        """Late-stage HDRI sphere-map pass.

        This is deliberately stronger than the earlier environment contribution.
        It samples the background low-frequency light field by *surface normal* and
        by *reflection vector*, exactly like the diagnostic lighting spheres in the
        user's reference.  Therefore the lighting forms curved arcs across the
        face/body instead of a uniform flat tint.
        """
        field = getattr(lighting_info, 'gradient_field', {}) or {}
        if not isinstance(field, dict) or not bool(field.get('enabled', False)):
            return np.zeros_like(relit, dtype=np.float32)
        grid = field.get('grid_colors', None)
        if grid is None or not np.any(subject_mask > 0.08):
            return np.zeros_like(relit, dtype=np.float32)
        try:
            g = np.asarray(grid, dtype=np.float32)
        except Exception:
            return np.zeros_like(relit, dtype=np.float32)
        if g.ndim < 3 or g.shape[0] < 1 or g.shape[1] < 1:
            return np.zeros_like(relit, dtype=np.float32)

        h, w = subject_mask.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ys, xs = np.where(subject_mask > 0.08)
        y0, y1 = float(ys.min()), float(ys.max())
        x0, x1 = float(xs.min()), float(xs.max())
        x_rel = np.clip((xx - x0) / max(x1 - x0, 1.0), 0.0, 1.0).astype(np.float32)
        y_rel = np.clip((yy - y0) / max(y1 - y0, 1.0), 0.0, 1.0).astype(np.float32)

        # Pseudo portrait volume normals.  Real normals are often too smooth or
        # face-only; blending them with an ellipsoid/cylinder proxy makes the light
        # visibly roll over the whole person.
        sx = np.clip((x_rel - 0.50) * 2.0, -1.0, 1.0)
        sy = np.clip((0.52 - y_rel) * 1.55, -1.0, 1.0)
        body_z = np.sqrt(np.clip(1.0 - 0.58 * sx * sx - 0.16 * sy * sy, 0.08, 1.0))
        N_proxy = safe_norm(np.stack([0.78 * sx, 0.34 * sy, body_z], axis=-1).astype(np.float32))
        # Face keeps more of the estimated normal; body/shoulder gets more proxy curvature.
        proxy_weight = np.clip(0.46 + 0.22 * (1.0 - face_core) + 0.10 * edge_band, 0.30, 0.70).astype(np.float32)
        N_arc = safe_norm(N * (1.0 - proxy_weight[..., None]) + N_proxy * proxy_weight[..., None])
        R_arc = safe_norm(2.0 * N_arc * np.sum(N_arc * V, axis=-1, keepdims=True) - V)

        # Sphere-map UVs.  Center background maps to front; sides/top/bottom map
        # to normal direction.  Reflection UV makes specular/chroma arcs.
        un = np.clip(0.50 + 0.43 * N_arc[..., 0], 0.0, 1.0).astype(np.float32)
        vn = np.clip(0.50 - 0.43 * N_arc[..., 1], 0.0, 1.0).astype(np.float32)
        ur = np.clip(0.50 + 0.47 * R_arc[..., 0], 0.0, 1.0).astype(np.float32)
        vr = np.clip(0.50 - 0.47 * R_arc[..., 1], 0.0, 1.0).astype(np.float32)
        env_n = self._bilinear_sample_field_grid(g, un, vn)
        env_r = self._bilinear_sample_field_grid(g, ur, vr)
        # Add a very blurred average to prevent noisy patch colors from becoming paint.
        global_mean = np.array(getattr(lighting_info, 'global_mean_color', np.mean(g.reshape(-1, 3), axis=0)), dtype=np.float32)
        env_n = 0.78 * env_n + 0.22 * global_mean.reshape(1,1,3)
        env_r = 0.64 * env_r + 0.36 * global_mean.reshape(1,1,3)
        recipe = self._estimate_portrait_light_recipe(lighting_info)
        colorfulness = float(np.clip(recipe.get('colorfulness', field.get('colorfulness', 0.25)), 0.0, 1.0))
        local_contrast = float(np.clip(recipe.get('local_contrast', field.get('local_contrast', 0.35)), 0.0, 1.0))
        side_sign = float(recipe.get('side_sign', 1.0))
        side_gate = self._compute_signed_side_mask(P, side_sign, subject_mask, power=0.72)

        fresnel = np.power(np.clip(1.0 - np.sum(N_arc * V, axis=-1), 0.0, 1.0), 2.0).astype(np.float32)
        rough_gate = np.clip(1.0 - roughness_map * 0.55, 0.25, 1.0).astype(np.float32)
        spec_gate = np.clip(0.35 + 0.65 * specular_map, 0.20, 1.25).astype(np.float32)
        region = np.clip(subject_mask * (0.32 + 0.34 * edge_band + 0.26 * hair_region + 0.16 * side_gate + 0.10 * (1.0 - face_core)), 0.0, 1.0)
        arc = (0.48 * env_n + 0.52 * env_r) * region[..., None]
        arc *= (0.035 + 0.050 * colorfulness + 0.030 * local_contrast)
        arc *= (0.62 + 0.38 * fresnel * rough_gate * spec_gate)[..., None]
        return np.clip(arc, 0.0, 0.34).astype(np.float32)
