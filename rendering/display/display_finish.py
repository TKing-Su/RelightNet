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

class RendererDisplayFinishMixin:
    def _gaussianish_blur_rgb(self, rgb: np.ndarray, radius: int = 2) -> np.ndarray:
        out = rgb.astype(np.float32)
        for _ in range(max(1, int(radius))):
            out = box_blur_rgb(out, passes=1)
        return out.astype(np.float32)


    def _apply_bloom(self, rgb: np.ndarray) -> np.ndarray:
        strength = float(np.clip(getattr(self, 'post_bloom_strength', 0.0), 0.0, 1.0))
        if self.look_safe and self._atmosphere_budget and not self._using_continuous_policy():
            strength *= self._policy_value('display', 'bloom', self._atmosphere_budget.get('bloom_multiplier', 1.0))
        if strength <= 1e-6:
            return rgb
        thr = float(np.clip(getattr(self, 'post_bloom_threshold', 0.74), 0.0, 1.0))
        rad = int(max(1, getattr(self, 'post_bloom_radius', 2)))
        lum = rgb_luminance(rgb)
        gate = np.clip((lum - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
        highlight = rgb * gate[..., None]
        blurred = self._gaussianish_blur_rgb(highlight, radius=rad)
        return np.clip(rgb + blurred * strength, 0.0, 1.0).astype(np.float32)


    def _apply_vignette(self, rgb: np.ndarray) -> np.ndarray:
        strength = float(np.clip(getattr(self, 'post_vignette_strength', 0.0), 0.0, 1.0))
        if self._using_continuous_policy():
            strength *= self._policy_value('display', 'vignette', 1.0)
        if strength <= 1e-6:
            return rgb
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(w - 1, 1) - 0.5) * 2.0
        ny = (yy / max(h - 1, 1) - 0.5) * 2.0
        rr = np.sqrt(nx * nx + ny * ny)
        vignette = 1.0 - strength * np.clip((rr - 0.20) / 0.95, 0.0, 1.0) ** 1.6
        return np.clip(rgb * vignette[..., None], 0.0, 1.0).astype(np.float32)


    def _apply_local_contrast(self, rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
        strength = float(np.clip(getattr(self, 'post_local_contrast_strength', 0.0), 0.0, 1.0))
        if self._using_continuous_policy():
            strength *= float(np.clip(self._policy_value('display', 'local_contrast', self._budget().get('lowkey_local_contrast_boost', 1.0)), 0.55, 1.36))
        if strength <= 1e-6:
            return rgb
        blur = self._gaussianish_blur_rgb(rgb, radius=2)
        detail = np.clip(rgb - blur, -0.18, 0.18)
        if alpha is not None:
            # Stage36: local contrast is a display-grade finishing pass. Applying it
            # uniformly after relighting re-sharpens pores/noise on the face and
            # cancels the no-HF-refill policy. Keep it mostly for background/edge.
            a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
            if a.shape[:2] == rgb.shape[:2]:
                a = box_blur_gray(a, passes=1)
                contrast_gate = np.clip(1.0 - 0.84 * a, 0.14, 1.0).astype(np.float32)
                return np.clip(rgb + detail * strength * contrast_gate[..., None], 0.0, 1.0).astype(np.float32)
        return np.clip(rgb + detail * strength, 0.0, 1.0).astype(np.float32)


    def _apply_split_toning(self, rgb: np.ndarray) -> np.ndarray:
        sc = float(np.clip(getattr(self, 'split_shadow_cool', 0.0), 0.0, 1.0))
        hw = float(np.clip(getattr(self, 'split_highlight_warm', 0.0), 0.0, 1.0))
        if self._using_continuous_policy():
            _ab = self._budget()
            sc *= float(np.clip(_ab.get('shadow_tint_budget', 0.0), 0.0, 1.0))
            hw *= float(np.clip(_ab.get('highlight_tint_budget', 0.0), 0.0, 1.0))
        if sc <= 1e-6 and hw <= 1e-6:
            return rgb
        lum = rgb_luminance(rgb)
        shadow_gate = np.clip((0.45 - lum) / 0.45, 0.0, 1.0)
        high_gate = np.clip((lum - 0.55) / 0.45, 0.0, 1.0)
        cool = np.array([0.93, 0.98, 1.05], dtype=np.float32).reshape(1, 1, 3)
        warm = np.array([1.05, 1.01, 0.95], dtype=np.float32).reshape(1, 1, 3)
        out = rgb * (1.0 - sc * shadow_gate[..., None]) + rgb * cool * (sc * shadow_gate[..., None])
        out = out * (1.0 - hw * high_gate[..., None]) + out * warm * (hw * high_gate[..., None])
        return np.clip(out, 0.0, 1.0).astype(np.float32)


    def _apply_haze(self, rgb: np.ndarray, background_linear: Optional[np.ndarray] = None) -> np.ndarray:
        strength = float(np.clip(getattr(self, 'post_haze_strength', 0.0), 0.0, 1.0))
        if self.look_safe and self._atmosphere_budget and not self._using_continuous_policy():
            strength *= self._policy_value('display', 'haze', self._atmosphere_budget.get('haze_multiplier', 1.0))
        if strength <= 1e-6:
            return rgb
        lum = rgb_luminance(rgb)
        source = background_linear if background_linear is not None else rgb
        if source.shape[:2] != rgb.shape[:2]:
            source = resize_linear_image(source, rgb.shape[:2])
        haze_color = np.mean(source.reshape(-1, 3), axis=0).astype(np.float32)
        haze_color = brighten_preserve_hue(haze_color, max(float(np.dot(haze_color, LUMA)), 0.18))
        haze_mask = box_blur_gray(np.clip(lum, 0.0, 1.0), passes=2)
        haze_mask = np.clip((haze_mask - 0.25) / 0.55, 0.0, 1.0)
        return np.clip(
            rgb * (1.0 - strength * haze_mask[..., None])
            + haze_color.reshape(1, 1, 3) * (strength * 0.55 * haze_mask[..., None]),
            0.0,
            1.0,
        ).astype(np.float32)


    def _protect_skin_tones(self, rgb: np.ndarray, face_core: Optional[np.ndarray] = None) -> np.ndarray:
        strength = float(np.clip(getattr(self, 'skin_protect_strength', 0.0), 0.0, 1.0))
        if strength <= 1e-6 or face_core is None:
            return rgb
        # Stage38: protection must be conditional.  Stage37 pulled the whole face
        # toward neutral gray, so neon/purple/cyan light disappeared and skin became
        # white.  Only compress extreme over-saturation in the face center; normal
        # background color cast is preserved.
        lum = rgb_luminance(rgb)
        mx = np.max(np.clip(rgb, 0.0, 1.0), axis=-1)
        mn = np.min(np.clip(rgb, 0.0, 1.0), axis=-1)
        sat = np.where(mx > 1e-5, (mx - mn) / np.maximum(mx, 1e-5), 0.0).astype(np.float32)
        excess = np.clip((sat - 0.62) / 0.20, 0.0, 1.0).astype(np.float32)
        gate = np.clip(face_core, 0.0, 1.0) * excess
        if self._using_continuous_policy():
            prev = getattr(self, '_unified_skin_protect_mask', np.zeros_like(gate, dtype=np.float32))
            self._unified_skin_protect_mask = np.maximum(prev, np.clip(gate * strength, 0.0, 1.0)).astype(np.float32)
        gate = gate[..., None]
        protected = lum[..., None] * 0.055 + rgb * 0.945
        return np.clip(rgb * (1.0 - strength * gate) + protected * (strength * gate), 0.0, 1.0).astype(np.float32)


    def _apply_display_finish(self, rgb: np.ndarray, alpha: Optional[np.ndarray] = None, background_linear: Optional[np.ndarray] = None) -> np.ndarray:
        out = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        out = self._apply_local_contrast(out, alpha=alpha)
        out = self._apply_bloom(out)
        out = self._apply_split_toning(out)
        out = self._apply_haze(out, background_linear=background_linear)
        out = self._apply_vignette(out)
        if self._using_continuous_policy() and alpha is not None and np.any(alpha > 0.08):
            subj = np.clip(alpha.astype(np.float32), 0.0, 1.0)
            luma = np.maximum(rgb_luminance(np.clip(out, 0.0, 1.0)), 1e-5)
            local_ref = box_blur_gray(luma, passes=4)
            local_peak = np.clip(luma - np.maximum(local_ref, 0.18), 0.0, 1.0)
            peak_gate = feather_mask(
                np.clip(subj * np.clip((local_peak - 0.070) / 0.18, 0.0, 1.0), 0.0, 0.80).astype(np.float32),
                passes=2,
            )
            if np.any(peak_gate > 1e-4):
                peak_ceiling = np.maximum(local_ref + 0.105, luma * 0.84)
                governed_luma = luma * (1.0 - peak_gate) + np.minimum(luma, peak_ceiling) * peak_gate
                out *= np.clip(governed_luma / luma, 0.78, 1.0)[..., None]
        return np.clip(out, 0.0, 1.0).astype(np.float32)
