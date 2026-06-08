from __future__ import annotations

from typing import Optional
import numpy as np
from lighting.models import LightingInfo
from tools.color import rgb_luminance
from tools.filters import box_blur_gray, box_blur_rgb


class RendererV32RegionsMixin:
    def _v32_shell_only_background_match(
        self,
        relit_linear: np.ndarray,
        mask: np.ndarray,
        background_linear: Optional[np.ndarray],
        lighting_info: Optional[LightingInfo],
    ) -> np.ndarray:
        """Match only the subject shell/edge to the background chroma."""
        if background_linear is None:
            return relit_linear
        alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)
        if not np.any(alpha > 0.1):
            return relit_linear

        edge = np.clip(4.0 * alpha * (1.0 - alpha), 0.0, 1.0).astype(np.float32)
        shell = np.clip(box_blur_gray(edge, passes=2) * alpha, 0.0, 1.0).astype(np.float32)
        if not np.any(shell > 1e-4):
            return relit_linear

        bg_low = box_blur_rgb(background_linear, passes=2)
        fg_y = np.maximum(rgb_luminance(relit_linear), 1e-5).astype(np.float32)
        bg_y = np.maximum(rgb_luminance(bg_low), 1e-5).astype(np.float32)
        bg_chroma = np.clip(bg_low / bg_y[..., None], 0.0, 3.0).astype(np.float32)
        matched = np.clip(fg_y[..., None] * bg_chroma, 0.0, 8.0).astype(np.float32)

        budget = self._budget()
        strength = float(np.clip(0.018 + 0.12 * budget.get("shell_atmosphere_budget", 0.10), 0.004, 0.085))
        cap = float(np.clip(0.030 + 0.20 * budget.get("rim_chroma_budget", 0.05), 0.035, 0.12))
        mix = np.clip(shell * strength, 0.0, cap)[..., None]
        return np.clip(relit_linear * (1.0 - mix) + matched * mix, 0.0, 8.0).astype(np.float32)
