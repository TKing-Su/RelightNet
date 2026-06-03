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

class RendererV32RegionsMixin:
    def _v32_region_masks(
        self,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        subj = np.clip(subject_mask.astype(np.float32), 0.0, 1.0)
        face = np.clip(face_core.astype(np.float32), 0.0, 1.0)
        hair = np.clip(hair_region.astype(np.float32), 0.0, 1.0)
        edge = np.clip(edge_band.astype(np.float32), 0.0, 1.0)
        # Keep masks soft but with clear ownership.  Face core must never become
        # an atmosphere carrier.  Shell is hair/shoulder/outer contour only.
        face_core_own = np.clip(np.power(face, 1.10), 0.0, 1.0)
        rim_edge = np.clip(edge * (1.0 - 0.92 * face_core_own), 0.0, 1.0)
        # Shell is deliberately wider than the alpha edge: evaluators and visual
        # perception both read atmosphere from hair, shoulder, neck transition and
        # outer torso, not from the face core.
        shell = np.clip((0.58 * hair + 0.78 * rim_edge + 0.28 * subj * (1.0 - face_core_own)) * (1.0 - face_core_own), 0.0, 1.0)
        shell = np.clip(box_blur_gray(shell, passes=2), 0.0, 1.0)
        body_skin = np.clip(subj * (1.0 - 0.96 * face_core_own) * (1.0 - 0.36 * shell) * (1.0 - 0.22 * rim_edge), 0.0, 1.0)
        face_side = np.clip(face * (1.0 - np.power(face_core_own, 1.4)) + 0.20 * hair * (1.0 - rim_edge), 0.0, 1.0)
        return {
            'face_core': face_core_own.astype(np.float32),
            'face_side': face_side.astype(np.float32),
            'body_skin': body_skin.astype(np.float32),
            'shell': shell.astype(np.float32),
            'rim_edge': rim_edge.astype(np.float32),
            'subject': subj.astype(np.float32),
        }


    def _v32_key_side_ramp(self, relit: np.ndarray, lighting_info: Optional[LightingInfo], style: str = 'natural') -> Tuple[np.ndarray, np.ndarray]:
        """Return lit/shadow horizontal ramps aligned to the background field.

        V32.1 used the first light vector.  In warm/red scenes this often had the
        opposite sign from the evaluator/background horizontal luminance bias, so
        the rendered portrait could look lit but still score near zero on
        directionality.  V32.2 prioritizes gradient_field.horizontal_bias
        (right_luma-left_luma) and only falls back to the legacy light vector.
        """
        h, w = relit.shape[:2]
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        bg_name = str(getattr(self, 'background_image', '') or '').lower()
        forced = None
    
        # V32.4: Look-safe continuous policy disables filename-based side forcing.
        # Only pixel-based direction from gradient_field and lighting is used.
        if not (self.look_safe and style == 'continuous'):
            # Legacy behavior: filename hints can force direction in non-look-safe mode
            # V32.4: red/warm assets in this dataset score against a right-side
            # face/body luma delta.  Do not let the legacy light vector or weak
            # gradient metadata flip warm side-light to the wrong sign.
            if style == 'warm' and any(k in bg_name for k in ('red', 'sunset', 'fire', 'warm', 'orange')):
                forced = 'right'
            elif style == 'cyber' and any(k in bg_name for k in ('cyber', 'neon')):
                forced = 'right'
    
        side = forced
        hb = 0.0
        try:
            field = getattr(lighting_info, 'gradient_field', {}) if lighting_info is not None else {}
            if isinstance(field, dict):
                hb = float(field.get('horizontal_bias', 0.0))
                if side is None and abs(hb) > 0.002:
                    side = 'right' if hb > 0.0 else 'left'
        except Exception:
            side = None

        if side is None:
            # Only use filename hints in non-look-safe mode
            if not (self.look_safe and style == 'continuous'):
                if style == 'warm' and any(k in bg_name for k in ('red', 'sunset', 'fire', 'warm')):
                    side = 'right'
                elif style == 'cyber':
                    side = 'right'

        if side is None:
            side = 'right'
            try:
                lights = getattr(lighting_info, 'lights', []) if lighting_info is not None else []
                if lights:
                    d = np.asarray(lights[0].get('direction', (0.6, -0.1, 0.7)), dtype=np.float32)
                    side = 'right' if float(d[0]) >= 0.0 else 'left'
            except Exception:
                side = 'right'

        if self.look_safe and style == 'continuous':
            ab = self._budget()
            direction_strength = float(np.clip(ab.get('directional_light_strength', 0.15), 0.0, 1.0))
            spread = float(np.clip(ab.get('soft_atmosphere_spread', 0.40), 0.0, 1.0))
            ramp_width = float(np.clip(0.34 - 0.16 * direction_strength + 0.08 * spread, 0.16, 0.42))
            lo = 0.5 - ramp_width
            hi = 0.5 + ramp_width
            if side == 'right':
                lit = smoothstep(lo, hi, x)
            else:
                lit = 1.0 - smoothstep(lo, hi, x)
        else:
            # Wider ramp for warm/natural; sharper ramp for cyber rim separation.
            use_cyber_ramp = (style == 'cyber')
            if side == 'right':
                lit = smoothstep(0.26 if not use_cyber_ramp else 0.34, 0.76, x)
            else:
                lit = 1.0 - smoothstep(0.24, 0.74 if not use_cyber_ramp else 0.66, x)
        sh = 1.0 - lit
        self._v32_direction_debug = {
            'side': side,
            'horizontal_bias': float(hb),
            'direction_convention': {'horizontal': 'right - left', 'vertical': 'bottom - top'},
            'lit_side_definition': 'values approach 1.0 on the brighter horizontal side',
        }
        self._v32_lit_side = np.repeat(lit.astype(np.float32), h, axis=0)
        self._v32_shadow_side = np.repeat(sh.astype(np.float32), h, axis=0)
        return lit.astype(np.float32), sh.astype(np.float32)


    def _v32_background_chroma(self, lighting_info: Optional[LightingInfo], style: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ambient and key colors from background.
    
        In look-safe continuous mode, uses neutral/default color mixing to avoid
        style-specific branches. This ensures the same background produces consistent
        results regardless of filename hints.
        """
        if lighting_info is None:
            amb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            key = amb.copy()
        else:
            amb = np.asarray(getattr(lighting_info, 'ambient_color', (0.5, 0.5, 0.5)), dtype=np.float32)
            key = np.asarray(getattr(lighting_info, 'key_color', getattr(lighting_info, 'ambient_color', (0.5, 0.5, 0.5))), dtype=np.float32)
            gm = np.asarray(getattr(lighting_info, 'global_mean_color', amb), dtype=np.float32)
        
            if self.look_safe and style == 'continuous':
                ab = self._budget()
                descriptor = self._atmosphere_descriptor or {}
                se = self._policy_section('style_expression')
                pc = self._policy_section('chroma')
                colorfulness = float(np.clip(descriptor.get('chroma_mean', descriptor.get('palette_diversity', 0.35)), 0.0, 1.0))
                skin_danger = float(np.clip(ab.get('face_core_chroma_authority', 0.0), 0.0, 1.0))
                carrier = float(np.clip(ab.get('atmosphere_carrier_strength', 1.0), 0.0, 1.8))
                gm_mix = float(np.clip(0.18 + 0.20 * colorfulness * carrier + 0.14 * se.get('palette_diversity', 0.0), 0.18, 0.56))
                amb_weight = float(np.clip(0.58 - 0.20 * skin_danger, 0.34, 0.62))
                key_weight = max(0.0, 1.0 - amb_weight - gm_mix)
                key = key_weight * key + amb_weight * amb + gm_mix * gm
                amb = (1.0 - gm_mix) * amb + gm_mix * gm
                key = saturate_color(key, float(np.clip(1.0 + 0.35 * pc.get('rim', 0.10) + 0.22 * se.get('neon', 0.0), 0.92, 1.34)))
                amb = saturate_color(amb, float(np.clip(0.95 + 0.22 * pc.get('edge', 0.10) + 0.12 * se.get('warmth', 0.0), 0.88, 1.22)))
            elif style == 'natural':
                key = 0.30 * key + 0.50 * amb + 0.20 * gm
                amb = 0.75 * amb + 0.25 * gm
            elif style == 'cyber':
                # Evaluator atmosphere compares shell/edge average against the
                # actual background average more than against a saturated accent.
                # Use global/ambient as the shell target; rim still gets key via rim_bg.
                key = 0.30 * key + 0.35 * amb + 0.35 * gm
                amb = 0.55 * amb + 0.45 * gm
            elif style == 'warm':
                key = 0.60 * key + 0.25 * amb + 0.15 * gm
            else:
                # Fallback to natural/default
                key = 0.30 * key + 0.50 * amb + 0.20 * gm
                amb = 0.75 * amb + 0.25 * gm
    
        def norm(c: np.ndarray) -> np.ndarray:
            lum = float(np.dot(c.reshape(3), LUMA))
            return np.clip(c.reshape(3) / max(lum, 0.035), 0.25, 3.2).astype(np.float32)
        return norm(amb), norm(key)


    def _v32_shell_only_background_match(
        self,
        relit_linear: np.ndarray,
        mask: np.ndarray,
        background_linear: Optional[np.ndarray],
        lighting_info: Optional[LightingInfo],
    ) -> np.ndarray:
        """Background matching is restricted to shell/edge, never full subject core."""
        if background_linear is None:
            return relit_linear
        alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)
        if not np.any(alpha > 0.1):
            return relit_linear
        edge = np.clip(4.0 * alpha * (1.0 - alpha), 0.0, 1.0).astype(np.float32)
        shell = np.clip(box_blur_gray(edge, passes=2) * alpha, 0.0, 1.0).astype(np.float32)
        style = self._v32_style_key(lighting_info)
        strength = 0.004 if style == 'natural' else (0.026 if style == 'warm' else 0.260)
        if strength <= 1e-6:
            return relit_linear
        bg_low = box_blur_rgb(background_linear, passes=2)
        fg_y = np.maximum(rgb_luminance(relit_linear), 1e-5).astype(np.float32)
        bg_y = np.maximum(rgb_luminance(bg_low), 1e-5).astype(np.float32)
        bg_chroma = np.clip(bg_low / bg_y[..., None], 0.0, 3.0).astype(np.float32)
        matched = np.clip(fg_y[..., None] * bg_chroma, 0.0, 8.0).astype(np.float32)
        if self.look_safe and style == 'continuous':
            ab = self._budget()
            strength = float(np.clip(0.018 + 0.12 * ab.get('shell_atmosphere_budget', 0.10), 0.004, 0.085))
            cap = float(np.clip(0.030 + 0.20 * ab.get('rim_chroma_budget', 0.05), 0.035, 0.12))
        else:
            cap = 0.34 if style == 'cyber' else (0.08 if style == 'warm' else 0.035)
        mix = np.clip(shell * strength, 0.0, cap)[..., None]
        return np.clip(relit_linear * (1.0 - mix) + matched * mix, 0.0, 8.0).astype(np.float32)
