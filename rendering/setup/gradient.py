from __future__ import annotations

from typing import Dict, Optional, Tuple
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

class RendererGradientMixin:
    def _sample_gradient_color_grid(self, field: Dict[str, object], size_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        """Bilinearly resize the background low-frequency color grid to image size."""
        try:
            grid = np.array(field.get('grid_colors'), dtype=np.float32)
        except Exception:
            return None
        if grid.ndim != 3 or grid.shape[-1] != 3 or grid.shape[0] < 2 or grid.shape[1] < 2:
            return None
        h, w = size_hw
        gh, gw = grid.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx = xx / max(w - 1, 1) * float(gw - 1)
        gy = yy / max(h - 1, 1) * float(gh - 1)
        x0 = np.floor(gx).astype(np.int32)
        y0 = np.floor(gy).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, gw - 1)
        y1 = np.clip(y0 + 1, 0, gh - 1)
        wx = (gx - x0.astype(np.float32))[..., None]
        wy = (gy - y0.astype(np.float32))[..., None]
        c00 = grid[y0, x0]
        c10 = grid[y0, x1]
        c01 = grid[y1, x0]
        c11 = grid[y1, x1]
        c0 = c00 * (1.0 - wx) + c10 * wx
        c1 = c01 * (1.0 - wx) + c11 * wx
        return np.clip(c0 * (1.0 - wy) + c1 * wy, 0.0, 6.0).astype(np.float32)
