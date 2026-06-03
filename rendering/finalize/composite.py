from __future__ import annotations

from typing import Optional, Tuple
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

class RendererCompositeMixin:
    def global_foreground_background_match(self, relit_linear: np.ndarray, mask: np.ndarray, background_linear: np.ndarray) -> np.ndarray:
        alpha = np.clip(mask, 0.0, 1.0)
        fg_pixels = relit_linear[alpha > 0.30]
        bg_pixels = background_linear.reshape(-1, 3)
        if fg_pixels.shape[0] < 16 or bg_pixels.shape[0] < 16:
            return relit_linear
        fg_mean = fg_pixels.mean(axis=0).astype(np.float32)
        bg_mean = bg_pixels.mean(axis=0).astype(np.float32)
        fg_l = float(np.dot(fg_mean, LUMA)); bg_l = float(np.dot(bg_mean, LUMA))
        if fg_l < 1e-5 or bg_l < 1e-5:
            return relit_linear
        fg_chroma = fg_mean / max(fg_l, 1e-5)
        bg_chroma = bg_mean / max(bg_l, 1e-5)
        color_gain = np.clip(bg_chroma / np.maximum(fg_chroma, 1e-4), 0.992, 1.010)
        fg_low = box_blur_rgb(relit_linear, passes=1)
        matched_low = np.clip(fg_low * color_gain.reshape(1, 1, 3), 0.0, None)
        low_delta = matched_low - fg_low
        edge = np.clip(4.0 * alpha * (1.0 - alpha), 0.0, 1.0)
        blend = 0.45 * self.core_color_match_strength * alpha[..., None] + 0.85 * self.edge_color_match_strength * edge[..., None]
        return np.clip(relit_linear + low_delta * blend, 0.0, 8.0).astype(np.float32)


    def _cleanup_foreground_edge(self, relit_linear: np.ndarray, bg_linear: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Reduce halo/grey edges by gently harmonizing only the matte boundary.

        This is intentionally conservative: it does not blur the subject core.
        It only mixes a small amount of local background color into the foreground
        where alpha is semi-transparent or rapidly changing.
        """
        strength = float(np.clip(getattr(self, 'edge_cleanup_strength', 0.10), 0.0, 0.5))
        if strength <= 1e-6:
            return relit_linear
        a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        soft = box_blur_gray(a, passes=max(1, int(getattr(self, 'edge_cleanup_blur_passes', 2))))
        # matte transition + geometric boundary band
        transition = np.clip(4.0 * a * (1.0 - a), 0.0, 1.0)
        boundary = np.clip(np.abs(soft - a) * 3.0, 0.0, 1.0)
        edge_gate = np.clip(np.maximum(transition, boundary), 0.0, 1.0)
        bg_local = box_blur_rgb(bg_linear, passes=max(1, int(getattr(self, 'edge_cleanup_blur_passes', 2))))
        fg_luma = np.maximum(rgb_luminance(relit_linear), 1e-4)
        bg_luma = np.maximum(rgb_luminance(bg_local), 1e-4)
        bg_local_toned = np.clip(bg_local * np.clip((fg_luma / bg_luma), 0.75, 1.25)[..., None], 0.0, 8.0)
        mix = strength * 0.62 * edge_gate[..., None]
        return np.clip(relit_linear * (1.0 - mix) + bg_local_toned * mix, 0.0, 8.0).astype(np.float32)


    def composite_with_background(
        self,
        relit_linear: np.ndarray,
        mask: np.ndarray,
        background_linear: Optional[np.ndarray],
        lighting_info: Optional[LightingInfo] = None,
        debug_prefix: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = self._prepare_composite_alpha(mask)
        if background_linear is None:
            self._save_shadow_debug(debug_prefix, contact_shadow=self._last_contact_shadow, ground_shadow=None)
            return relit_linear, alpha
        bg = background_linear.astype(np.float32)

        light_dir = None
        if lighting_info is not None and getattr(lighting_info, 'lights', None):
            try:
                light_dir = safe_norm(np.array(lighting_info.lights[0]['direction'], dtype=np.float32))
            except Exception:
                light_dir = None

        ground_shadow = self._compute_ground_shadow(alpha, light_dir=light_dir)
        self._last_ground_shadow = ground_shadow
        bg_shadowed = bg * ground_shadow[..., None]

        relit_matched = self._v32_shell_only_background_match(relit_linear, alpha, bg_shadowed, lighting_info)
        edge = np.clip(4.0 * alpha * (1.0 - alpha), 0.0, 1.0)
        bg_blur = box_blur_rgb(bg_shadowed, passes=max(1, int(self.edge_blur_passes)))
        fg_luma = np.maximum(rgb_luminance(relit_matched), 1e-4)
        bg_luma = np.maximum(rgb_luminance(bg_blur), 1e-4)
        bg_spill_toned = np.clip(desaturate_color(bg_blur, 0.30) * np.clip((fg_luma / bg_luma), 0.78, 1.18)[..., None], 0.0, 8.0)
        local_spill = bg_spill_toned * (self.edge_local_spill_strength * 0.65 * edge[..., None])
        relit_matched = relit_matched * (1.0 - self.edge_mix_strength * 0.65 * edge[..., None]) + local_spill
        relit_matched = self._cleanup_foreground_edge(relit_matched, bg_shadowed, alpha)
        comp = relit_matched * alpha[..., None] + bg_shadowed * (1.0 - alpha[..., None])
        comp_display = linear_to_srgb(np.clip(comp, 0.0, 1.0).astype(np.float32))
        comp_display = self._apply_display_finish(comp_display, alpha=alpha, background_linear=background_linear)
        comp = srgb_to_linear(np.clip(comp_display, 0.0, 1.0).astype(np.float32))
        self._save_shadow_debug(debug_prefix, contact_shadow=self._last_contact_shadow, ground_shadow=ground_shadow)
        return np.clip(comp, 0.0, 8.0), alpha
