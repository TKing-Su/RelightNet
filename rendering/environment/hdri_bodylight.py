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

class RendererHDRIBodyLightMixin:
    def _compute_hdri_spherical_bodylight(
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
        """Approximate HDRI lighting over the whole portrait volume.

        The crucial difference from a flat color wash is that we treat the
        background as an environment with several directional emitters. Each emitter
        is evaluated on the subject normal field, so the illumination bends over the
        face, neck, shoulders, and torso instead of being uniformly painted.
        """
        field = getattr(lighting_info, 'gradient_field', None)
        if not isinstance(field, dict) or not bool(field.get('enabled', False)):
            return np.zeros_like(base_subject, dtype=np.float32)
        if not np.any(subject_mask > 0.08):
            return np.zeros_like(base_subject, dtype=np.float32)

        recipe = self._estimate_portrait_light_recipe(lighting_info)
        recipe_name = str(recipe.get('recipe', 'balanced_soft'))
        colorfulness = float(recipe.get('colorfulness', 0.25))
        local_contrast = float(recipe.get('local_contrast', 0.35))
        p50 = float(recipe.get('p50', 0.22))
        side_sign = float(recipe.get('side_sign', 1.0))

        h, w = subject_mask.shape
        ys, xs = np.where(subject_mask > 0.08)
        y0, y1 = float(ys.min()), float(ys.max())
        x0, x1 = float(xs.min()), float(xs.max())
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        x_rel = np.clip((xx - x0) / max(x1 - x0, 1.0), 0.0, 1.0).astype(np.float32)
        y_rel = np.clip((yy - y0) / max(y1 - y0, 1.0), 0.0, 1.0).astype(np.float32)

        grid_colors = field.get('grid_colors', None)
        if grid_colors is None:
            return np.zeros_like(base_subject, dtype=np.float32)
        try:
            gcol = np.asarray(grid_colors, dtype=np.float32)
        except Exception:
            return np.zeros_like(base_subject, dtype=np.float32)
        if gcol.ndim != 3 or gcol.shape[-1] != 3:
            return np.zeros_like(base_subject, dtype=np.float32)

        sx = np.clip((x_rel - 0.50) * 2.0, -1.0, 1.0)
        sy = np.clip((0.54 - y_rel) * 1.52, -1.0, 1.0)
        z = np.sqrt(np.clip(1.0 - 0.58 * sx * sx - 0.18 * sy * sy, 0.08, 1.0))
        proxy_n = safe_norm(np.stack([0.78 * sx, 0.34 * sy, z], axis=-1).astype(np.float32))
        proxy_w = np.clip(0.36 + 0.22 * (1.0 - face_core) + 0.10 * edge_band + 0.08 * hair_region, 0.25, 0.72).astype(np.float32)
        n_env = safe_norm(N * (1.0 - proxy_w[..., None]) + proxy_n * proxy_w[..., None])

        u = np.clip(0.50 + 0.42 * n_env[..., 0], 0.0, 1.0).astype(np.float32)
        v = np.clip(0.50 - 0.42 * n_env[..., 1], 0.0, 1.0).astype(np.float32)
        env = self._bilinear_sample_field_grid(gcol, u, v)
        mean_color = np.array(getattr(lighting_info, 'global_mean_color', np.mean(gcol.reshape(-1, 3), axis=0)), dtype=np.float32)
        env = 0.72 * env + 0.28 * mean_color.reshape(1, 1, 3)

        ndotl = np.clip(0.34 + 0.66 * n_env[..., 2], 0.0, 1.0).astype(np.float32)
        side_gate = self._compute_signed_side_mask(P, side_sign, subject_mask, power=0.75)
        edge_gain = np.clip(0.55 + 0.25 * edge_band + 0.18 * hair_region + 0.14 * side_gate, 0.35, 1.25)
        body_gate = np.clip(subject_mask * (0.72 + 0.20 * (1.0 - face_core)), 0.0, 1.0)
        style_gain = 0.055 + 0.050 * colorfulness + 0.030 * local_contrast + (0.015 if recipe_name != 'balanced_soft' else 0.0)
        if p50 < 0.12:
            style_gain *= 0.78
        acc = env * ndotl[..., None] * edge_gain[..., None] * body_gate[..., None] * style_gain
        return np.clip(acc, 0.0, 0.42).astype(np.float32)
