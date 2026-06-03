from __future__ import annotations

import os
from typing import Optional
import numpy as np
from PIL import Image
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

class RendererShadowMixin:
    def _compute_directional_shadow(self, depth_map: np.ndarray, subject_mask: np.ndarray, light_dir: np.ndarray) -> np.ndarray:
        h, w = depth_map.shape
        sx = int(np.clip(np.round(-float(light_dir[0]) * max(2.0, w * 0.012)), -8, 8))
        sy = int(np.clip(np.round(float(light_dir[1]) * max(2.0, h * 0.012)), -8, 8))
        if sx == 0 and sy == 0:
            return np.ones_like(depth_map, dtype=np.float32)
        shifted = np.roll(depth_map, shift=(sy, sx), axis=(0, 1))
        occ = np.clip((shifted - depth_map - 0.006) / 0.030, 0.0, 1.0)
        shadow = 1.0 - self.key_shadow_strength * occ * subject_mask
        return np.clip(shadow, 0.72, 1.0).astype(np.float32)


    def _compute_contact_shadow(self, depth_map: np.ndarray, subject_mask: np.ndarray, light_dir: np.ndarray) -> np.ndarray:
        """Multi-sample screen-space contact shadow on the subject body.

        This upgrades the old one-shift depth shadow with several small samples
        along the opposite of the light direction. It is intentionally conservative
        because depth PNGs from different pipelines may have slightly different
        conventions. If the depth direction is wrong, keep this enabled but switch
        --depth-invert / camera.json first.
        """
        if not bool(getattr(self, 'contact_shadow_enabled', True)):
            return np.ones_like(depth_map, dtype=np.float32)
        h, w = depth_map.shape
        strength = float(np.clip(getattr(self, 'contact_shadow_strength', 0.18), 0.0, 1.0))
        if strength <= 1e-6 or not np.any(subject_mask > 0.05):
            return np.ones_like(depth_map, dtype=np.float32)

        radius = int(max(1, getattr(self, 'contact_shadow_radius_px', 10)))
        steps = int(max(1, getattr(self, 'contact_shadow_steps', 6)))
        bias = float(getattr(self, 'contact_shadow_depth_bias', 0.004))
        depth_range = max(float(getattr(self, 'contact_shadow_depth_range', 0.035)), 1e-6)

        dx = -float(light_dir[0])
        dy = float(light_dir[1])
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        dx /= norm
        dy /= norm

        occ = np.zeros_like(depth_map, dtype=np.float32)
        weight_sum = 0.0
        for i in range(1, steps + 1):
            dist = radius * i / float(steps)
            sx = int(np.clip(round(dx * dist), -radius, radius))
            sy = int(np.clip(round(dy * dist), -radius, radius))
            if sx == 0 and sy == 0:
                continue
            shifted_depth = np.roll(depth_map, shift=(sy, sx), axis=(0, 1))
            shifted_mask = np.roll(subject_mask, shift=(sy, sx), axis=(0, 1))
            local_occ = np.clip((shifted_depth - depth_map - bias) / depth_range, 0.0, 1.0)
            local_occ *= np.clip(shifted_mask, 0.0, 1.0)
            weight = 1.0 - (i - 1) / float(steps)
            occ += local_occ * weight
            weight_sum += weight

        occ = occ / max(weight_sum, 1e-6)
        occ = box_blur_gray(occ * subject_mask, passes=max(0, int(getattr(self, 'contact_shadow_blur_passes', 1))))
        min_factor = float(np.clip(getattr(self, 'contact_shadow_min_factor', 0.72), 0.0, 1.0))
        shadow = 1.0 - strength * np.clip(occ, 0.0, 1.0) * subject_mask
        return np.clip(shadow, min_factor, 1.0).astype(np.float32)


    def _compute_ground_shadow(self, alpha: np.ndarray, light_dir: Optional[np.ndarray] = None) -> np.ndarray:
        """Simple screen-space ellipse shadow under the cutout.

        It does not require a real 3D ground plane. It only uses the lower part
        of the alpha matte and the horizontal light direction, so it is robust
        for compositing sequences where only RGB/matte/depth/normal passes exist.
        """
        if not bool(getattr(self, 'ground_shadow_enabled', True)):
            return np.ones_like(alpha, dtype=np.float32)
        h, w = alpha.shape
        strength = float(np.clip(getattr(self, 'ground_shadow_strength', 0.26), 0.0, 1.0))
        if strength <= 1e-6:
            return np.ones_like(alpha, dtype=np.float32)
        ys, xs = np.where(alpha > 0.25)
        self._last_ground_shadow_auto_enabled = False
        if xs.size < 16:
            return np.ones_like(alpha, dtype=np.float32)

        bottom_y = float(np.percentile(ys, 99.4))
        top_y = float(np.percentile(ys, 0.6))
        bottom_ratio = bottom_y / max(float(h - 1), 1.0)
        subject_height_ratio = (bottom_y - top_y) / max(float(h - 1), 1.0)
        if bool(getattr(self, 'ground_shadow_auto_disable', True)):
            min_bottom = float(getattr(self, 'ground_shadow_min_bottom_ratio', 0.72))
            min_height = float(getattr(self, 'ground_shadow_min_subject_height_ratio', 0.42))
            if bottom_ratio < min_bottom or subject_height_ratio < min_height:
                return np.ones_like(alpha, dtype=np.float32)
        self._last_ground_shadow_auto_enabled = True

        center_x = float(np.mean(xs))
        person_width = float(np.percentile(xs, 90.0) - np.percentile(xs, 10.0))
        person_width = max(person_width, w * 0.12)
        ldx = float(light_dir[0]) if light_dir is not None else 0.0

        cx = center_x - ldx * person_width * float(getattr(self, 'ground_shadow_light_x_offset_scale', 0.22))
        cy = bottom_y + person_width * float(getattr(self, 'ground_shadow_y_offset_scale', 0.10))
        rx = max(person_width * float(getattr(self, 'ground_shadow_width_scale', 0.46)), 2.0)
        ry = max(person_width * float(getattr(self, 'ground_shadow_height_scale', 0.14)), 2.0)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        softness = max(float(getattr(self, 'ground_shadow_softness', 0.58)), 1e-3)
        shadow_blob = np.exp(-ellipse * (1.35 / softness)).astype(np.float32)
        below = yy > (bottom_y - person_width * 0.06)
        shadow_blob *= below.astype(np.float32)
        shadow_blob *= (1.0 - np.clip(alpha, 0.0, 1.0))
        shadow_blob = box_blur_gray(shadow_blob, passes=max(0, int(getattr(self, 'ground_shadow_blur_passes', 4))))
        min_factor = float(np.clip(getattr(self, 'ground_shadow_min_factor', 0.62), 0.0, 1.0))
        return np.clip(1.0 - strength * shadow_blob, min_factor, 1.0).astype(np.float32)


    def _save_shadow_debug(self, prefix: Optional[str], contact_shadow: Optional[np.ndarray] = None, ground_shadow: Optional[np.ndarray] = None) -> None:
        if not prefix or not self.debug_shadows:
            return
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        if contact_shadow is not None:
            contact_occ = 1.0 - np.clip(contact_shadow, 0.0, 1.0)
            Image.fromarray((contact_occ * 255.0 + 0.5).astype(np.uint8)).save(prefix + '_contact_shadow.png')
        if ground_shadow is not None:
            ground_occ = 1.0 - np.clip(ground_shadow, 0.0, 1.0)
            Image.fromarray((ground_occ * 255.0 + 0.5).astype(np.uint8)).save(prefix + '_ground_shadow.png')


    def _compute_spatial_side_mask(self, P: np.ndarray, light_dir: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
        x = P[..., 0].astype(np.float32)
        if not np.any(subject_mask > 0.08):
            return np.ones_like(subject_mask, dtype=np.float32)
        scale = float(np.percentile(np.abs(x[subject_mask > 0.08]), 88.0))
        scale = max(scale, 1e-4)
        xn = np.clip(x / scale, -1.0, 1.0)
        side = 0.5 + 0.5 * float(np.sign(light_dir[0])) * xn
        front = 1.0 - min(abs(float(light_dir[0])) * 1.2, 0.85)
        return np.clip(front + (1.0 - front) * side, 0.0, 1.0).astype(np.float32)
