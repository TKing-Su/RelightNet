from __future__ import annotations

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

class RendererMasksMixin:
    def _smoothstep01(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


    def _prepare_subject_mask(self, mask: np.ndarray) -> np.ndarray:
        matte = np.clip(mask.astype(np.float32), 0.0, 1.0)
        matte = feather_mask(matte, passes=1)
        matte = np.clip((matte - self.alpha_tighten) / max(self.alpha_edge_softness - self.alpha_tighten, 1e-6), 0.0, 1.0)
        matte = self._smoothstep01(matte)
        if abs(self.subject_mask_expand) > 1e-6:
            matte = np.clip(matte + self.subject_mask_expand, 0.0, 1.0)
        return matte.astype(np.float32)


    def _prepare_composite_alpha(self, mask: np.ndarray) -> np.ndarray:
        alpha = self._prepare_subject_mask(mask)
        alpha = feather_mask(alpha, passes=max(0, int(self.alpha_blur)))
        alpha = np.clip((alpha - 0.01) / 0.99, 0.0, 1.0)
        return alpha.astype(np.float32)


    def _compute_occlusion(self, N: np.ndarray, depth_map: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
        nl = np.roll(N, 1, axis=1); nr = np.roll(N, -1, axis=1); nu = np.roll(N, 1, axis=0); nd = np.roll(N, -1, axis=0)
        normal_var = (1.0 - np.sum(N * nl, axis=-1) + 1.0 - np.sum(N * nr, axis=-1) + 1.0 - np.sum(N * nu, axis=-1) + 1.0 - np.sum(N * nd, axis=-1)) * 0.25
        dl = np.roll(depth_map, 1, axis=1); dr = np.roll(depth_map, -1, axis=1); du = np.roll(depth_map, 1, axis=0); dd = np.roll(depth_map, -1, axis=0)
        depth_var = (np.abs(dl - depth_map) + np.abs(dr - depth_map) + np.abs(du - depth_map) + np.abs(dd - depth_map)) * 0.25
        cavity = np.clip((normal_var - 0.01) / (0.13 - 0.01), 0.0, 1.0)
        depth_occ = np.clip((depth_var - 0.002) / (0.028 - 0.002), 0.0, 1.0)
        ao = 1.0 - (0.10 * cavity + 0.08 * depth_occ) * subject_mask
        return np.clip(ao, 0.72, 1.0).astype(np.float32)


    def _compute_source_shading(self, source_linear: np.ndarray, albedo_linear: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
        src_l = rgb_luminance(source_linear)
        alb_l = rgb_luminance(np.clip(albedo_linear, 1e-4, None))
        shading = box_blur_gray(np.clip(src_l / np.maximum(alb_l, 1e-3), 0.0, 4.0), passes=2)
        med = float(np.median(shading[subject_mask > 0.2])) if np.any(subject_mask > 0.2) else float(np.median(shading))
        shading = shading / max(med, 1e-4)
        return np.clip(shading, 0.62, 1.32).astype(np.float32)


    def _compute_normal_curvature(self, N: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
        nx, ny, nz = N[..., 0], N[..., 1], N[..., 2]
        gxx, gxy = np.gradient(nx)
        gyx, gyy = np.gradient(ny)
        gzx, gzy = np.gradient(nz)
        curv = np.sqrt(gxx * gxx + gxy * gxy + gyx * gyx + gyy * gyy + gzx * gzx + gzy * gzy).astype(np.float32)
        curv = box_blur_gray(curv, passes=1)
        ref = float(np.percentile(curv[subject_mask > 0.08], 95.0)) if np.any(subject_mask > 0.08) else float(np.percentile(curv, 95.0))
        curv = np.clip(curv / max(ref, 1e-6), 0.0, 1.0)
        return curv.astype(np.float32)


    def _compute_depth_edge(self, depth_map: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
        dy, dx = np.gradient(depth_map.astype(np.float32))
        edge = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        edge = box_blur_gray(edge, passes=1)
        ref = float(np.percentile(edge[subject_mask > 0.08], 95.0)) if np.any(subject_mask > 0.08) else float(np.percentile(edge, 95.0))
        edge = np.clip(edge / max(ref, 1e-6), 0.0, 1.0)
        return edge.astype(np.float32)


    def _compute_intrinsic_gloss_control(
        self,
        source_linear: np.ndarray,
        albedo_linear: np.ndarray,
        N: np.ndarray,
        depth_map: np.ndarray,
        specular_map: np.ndarray,
        roughness_map: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
    ) -> np.ndarray:
        source_shape = self._compute_source_shading(source_linear, albedo_linear, subject_mask)
        src_hi = np.clip((source_shape - 0.98) / 0.22, 0.0, 1.0)
        curv = self._compute_normal_curvature(N, subject_mask)
        dep_edge = self._compute_depth_edge(depth_map, subject_mask)
        micro = np.clip(1.0 - roughness_map, 0.0, 1.0).astype(np.float32)
        spec = np.clip(specular_map, 0.0, 1.0).astype(np.float32)
        structure = np.clip(0.52 * curv + 0.28 * dep_edge + 0.20 * src_hi, 0.0, 1.0)
        portrait_prior = np.clip(0.42 * face_core + 0.34 * hair_region + 0.24 * edge_band, 0.0, 1.0)
        gloss = 0.24 + 0.42 * spec + 0.30 * micro + 0.34 * structure + 0.18 * portrait_prior
        gloss *= np.clip(subject_mask, 0.0, 1.0)
        gloss = box_blur_gray(gloss.astype(np.float32), passes=1)
        return np.clip(gloss, 0.0, 1.20).astype(np.float32)
