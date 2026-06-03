from __future__ import annotations

from typing import Dict, Optional
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

class RendererV32PolicyMixin:
    def _v32_style_key(self, lighting_info: Optional[LightingInfo]) -> str:
        # Look-safe mode must be background-pixel driven, not filename/profile driven.
        # Route all V32 ownership correction through a single continuous budget;
        # the legacy discrete warm/cyber/natural router is kept only for default mode.
        if bool(getattr(self, 'look_safe', False)):
            return 'continuous'

        # V32.1: style block routing must not rely only on style_mode because
        # the CLI exposes quality/cinematic/neon, while red/warm/misty are
        # encoded in the background file name.  Filename hints are used only for
        # router selection; the legacy lighting chain is still preserved.
        bg_name = str(getattr(self, 'background_image', '') or '').lower()
        if any(k in bg_name for k in ('cyber', 'neon', 'futuristic', 'city_night', 'rain_neon')):
            return 'cyber'
        if any(k in bg_name for k in ('sunset', 'fire', 'warm', 'autumn', 'red', 'orange')):
            return 'warm'
        if any(k in bg_name for k in ('mist', 'fog', 'haze', 'cloud', 'forest', 'morning', 'natural')):
            return 'natural'

        mode = str(getattr(self, 'style_mode', 'quality')).lower()
        if mode in ('neon', 'cyber', 'cyber_neon'):
            return 'cyber'
        if mode in ('cinematic', 'lowkey'):
            return 'cinematic'
        if lighting_info is None:
            return 'natural'
        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower()
        g = np.asarray(getattr(lighting_info, 'global_mean_color', getattr(lighting_info, 'ambient_color', (0.5, 0.5, 0.5))), dtype=np.float32)
        a = np.asarray(getattr(lighting_info, 'ambient_color', (0.5, 0.5, 0.5)), dtype=np.float32)
        k = np.asarray(getattr(lighting_info, 'key_color', a), dtype=np.float32)
        color = 0.45 * g + 0.35 * a + 0.20 * k
        r, gg, b = [float(x) for x in color.reshape(3)]
        if neon != 'off' or (bg_mode == 'rich' and (b + gg) > 1.15 * r):
            return 'cyber'
        if r > b * 1.18 and r > gg * 0.88:
            return 'warm'
        if b > r * 1.12 and bg_mode == 'rich':
            return 'cyber'
        return 'natural'


    def _v32_style_budgets(self, style: str) -> Dict[str, float]:
        if style == 'continuous':
            # Continuous look-safe policy: V32 no longer owns style classification.
            # It consumes the continuous atmosphere budget and only enforces region
            # ownership. This removes the old warm/cyber/natural hard switch from
            # the look-safe path while preserving default behavior.
            ab = self._atmosphere_budget or {}
            pc = ab.get('chroma', {}) if isinstance(ab.get('chroma', {}), dict) else {}
            pd = ab.get('direction', {}) if isinstance(ab.get('direction', {}), dict) else {}
            face_target = float(0.5 * (ab.get('autogain_target_low', 0.320) + ab.get('autogain_target_high', 0.380)))
            body_target = float(max(0.240, face_target - 0.025))

            edge_auth = float(ab.get('edge_shell_chroma_authority', 1.0))
            carrier = float(ab.get('atmosphere_carrier_strength', 1.0))
            rim_budget = float(pc.get('rim', ab.get('rim_chroma_budget', 0.12)))
            dir_strength = float(pd.get('directional_light_strength', ab.get('directional_light_strength', 0.15)))
            shadow_strength = float(pd.get('directional_shadow_strength', ab.get('directional_shadow_strength', 0.08)))
            luma_authority = float(ab.get('luma_authority_strength', 0.0))
            pbr_preserve = float(ab.get('pbr_preserve_strength', 0.0))

            body_guard = float(ab.get('body_skin_chroma_authority', 0.0))
            clothing_budget = float(ab.get('clothing_chroma_authority', 0.0))
            body_bg = float(np.clip(pc.get('body', 0.010 * (1.0 - body_guard) + 0.018 * clothing_budget), 0.000, 0.145))
            shell_bg = float(np.clip(max(pc.get('hair', 0.035), pc.get('edge', 0.035)) + 0.16 * rim_budget * carrier, 0.035, 0.72))
            rim_bg = float(np.clip(0.060 + 0.92 * rim_budget * max(carrier, 0.75), 0.060, 0.82))

            return {
                'face_bg': float(np.clip(pc.get('face_core', ab.get('face_core_bg_chroma_budget', ab.get('v32_face_bg_budget', 0.000))), 0.0, 0.016)),
                'face_side_bg': float(np.clip(pc.get('face_side', ab.get('v32_face_side_budget', 0.004)), 0.0, 0.095)),
                'body_bg': body_bg,
                'shell_bg': shell_bg,
                'rim_bg': rim_bg,
                'face_target': face_target,
                'body_target': body_target,
                'dir': float(np.clip(0.035 + 0.34 * dir_strength, 0.035, 0.255)),
                'shadow': float(np.clip(0.012 + 0.15 * shadow_strength, 0.012, 0.085)),
                'detail': float(np.clip(0.42 + 0.32 * (1.0 - luma_authority) + 0.10 * pbr_preserve, 0.30, 0.78)),
                'reconcile': float(np.clip(0.46 + 0.24 * luma_authority - 0.12 * pbr_preserve, 0.36, 0.72)),
            }

        if style == 'cyber':
            return {
                # V32.4: face/body protected; atmosphere budget pushed hard into shell/rim only.
                'face_bg': 0.010, 'face_side_bg': 0.026, 'body_bg': 0.050, 'shell_bg': 0.68, 'rim_bg': 0.82,
                'face_target': 0.335, 'body_target': 0.310, 'dir': 0.090, 'shadow': 0.030,
                'detail': 0.48, 'reconcile': 0.72,
            }
        if style == 'warm':
            return {
                # V32.4: keep the successful right-side luma direction, but stop
                # red/warm chroma from leaking into face/body.  Let shell/rim carry
                # warmth; face/body are almost source chroma with stronger luma detail.
                'face_bg': 0.000, 'face_side_bg': 0.003, 'body_bg': 0.006, 'shell_bg': 0.075, 'rim_bg': 0.160,
                'face_target': 0.358, 'body_target': 0.323, 'dir': 0.255, 'shadow': 0.045,
                'detail': 0.78, 'reconcile': 0.92,
            }
        if style == 'cinematic':
            return {
                'face_bg': 0.010, 'face_side_bg': 0.030, 'body_bg': 0.050, 'shell_bg': 0.14, 'rim_bg': 0.28,
                'face_target': 0.300, 'body_target': 0.275, 'dir': 0.110, 'shadow': 0.065,
                'detail': 0.50, 'reconcile': 0.50,
            }
        return {
            'face_bg': 0.001, 'face_side_bg': 0.004, 'body_bg': 0.010, 'shell_bg': 0.040, 'rim_bg': 0.060,
            'face_target': 0.385, 'body_target': 0.335, 'dir': 0.045, 'shadow': 0.015,
            'detail': 0.36, 'reconcile': 0.42,
        }


    def _v32_style_block_router(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo],
    ) -> np.ndarray:
        """Route to exactly one style correction block, then enforce modular ownership."""
        style = self._v32_style_key(lighting_info)
        # V32.4: the legacy PBR/light chain is kept, but the old metric-guided
        # style correction blocks are no longer executed here.  They were the
        # remaining overlap source: warm/cyber/natural cleanup could pre-tint or
        # flatten the image before modular ownership was enforced.  The single
        # modular reconcile layer below is now the only style post-chain.
        routed = relit
        return self._v32_reconcile_modular_blocks(routed, source_linear, subject_mask, face_core, hair_region, edge_band, lighting_info)
