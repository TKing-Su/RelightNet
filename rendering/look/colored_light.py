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

class RendererColoredLightMixin:
    def _inject_subject_colored_light(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo] = None,
    ) -> np.ndarray:
        """Inject background chroma as soft illumination, not as a flat tint.

        Stage41: the previous versions either removed the cyber color from the
        face, or added color in a way that still felt weak.  This version samples
        both the low-frequency background grid and the left/right palette colors,
        then blends chroma while preserving current luminance.  The center of the
        face is protected; color is stronger on cheek sides, neck, shoulder, hair
        and alpha edges where real environment spill should appear.
        """
        if lighting_info is None or relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        field = getattr(lighting_info, 'gradient_field', {})
        if not isinstance(field, dict) or not field.get('enabled', False):
            return relit
        h, w = relit.shape[:2]
        field_color = self._sample_gradient_color_grid(field, (h, w))
        if field_color is None:
            return relit

        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj, 0.0, 1.0).astype(np.float32)
        hair = np.clip(hair_region * subj, 0.0, 1.0).astype(np.float32)
        edge = np.clip(edge_band * subj, 0.0, 1.0).astype(np.float32)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)
        face_center_protect = np.exp(-0.5 * (((u - 0.50) / 0.22) ** 2 + ((v - 0.42) / 0.28) ** 2)).astype(np.float32)
        face_side = np.clip(1.0 - face_center_protect, 0.0, 1.0) * face

        # Build a smooth palette ramp from the estimated background sides.  This
        # prevents the subject center from sampling only the dark central building
        # area and losing the neon side colors.
        try:
            left_col = np.array(field.get('left_color'), dtype=np.float32)
            right_col = np.array(field.get('right_color'), dtype=np.float32)
            top_col = np.array(field.get('top_color'), dtype=np.float32)
            bottom_col = np.array(field.get('bottom_color'), dtype=np.float32)
            if left_col.shape != (3,) or right_col.shape != (3,):
                raise ValueError
            side_ramp = left_col.reshape(1, 1, 3) * (1.0 - u[..., None]) + right_col.reshape(1, 1, 3) * u[..., None]
            vertical_ramp = top_col.reshape(1, 1, 3) * (1.0 - v[..., None]) + bottom_col.reshape(1, 1, 3) * v[..., None]
            palette_field = np.clip(0.58 * field_color + 0.30 * side_ramp + 0.12 * vertical_ramp, 0.0, 6.0).astype(np.float32)
        except Exception:
            palette_field = field_color

        colorfulness = float(np.clip(field.get('colorfulness', getattr(lighting_info, 'palette_diversity', 0.35)), 0.0, 1.0))
        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower()
        if self._using_continuous_policy():
            _ab = self._budget()
            cyber_gate = float(np.clip(0.34 + 0.18 * _ab.get('atmosphere_carrier_strength', 1.0), 0.32, 0.62))
        else:
            cyber_gate = 0.84 if (self.style_mode == 'neon' or neon == 'strong') else (0.56 if (bg_mode == 'rich' or neon != 'off') else 0.40)
        subj_area = float(np.mean(subj))
        face_area = float(np.mean(face))
        face_ratio = face_area / max(subj_area, 1e-5)
        closeup_gate = float(np.clip((face_ratio - 0.30) / 0.42, 0.0, 1.0))
        face_chroma_scale = float(np.clip(1.0 - 0.82 * closeup_gate, 0.18, 1.0))
        side_chroma_scale = float(np.clip(1.0 - 0.58 * closeup_gate, 0.40, 1.0))
        body_chroma_scale = float(np.clip(1.0 - 0.70 * closeup_gate, 0.24, 1.0))

        fld_l = np.maximum(rgb_luminance(palette_field), 1e-5).astype(np.float32)
        p50 = float(field.get('p50_luma', np.percentile(fld_l, 50.0)))
        p95 = float(field.get('p95_luma', np.percentile(fld_l, 95.0)))
        field_profile = np.clip((fld_l - p50) / max(p95 - p50, 1e-5), 0.0, 1.0).astype(np.float32)
        field_profile = box_blur_gray(field_profile, passes=2)

        rel_l = np.maximum(rgb_luminance(np.clip(relit, 0.0, None)), 1e-5).astype(np.float32)
        fld_dir = np.clip(palette_field / fld_l[..., None], 0.22, 3.60).astype(np.float32)

        # Chroma is strongest where environment spill naturally belongs.  Avoid
        # a global body tint: keep the face center light, emphasize side cheek,
        # neck, hair edge and silhouette, and use only a small amount on torso.
        body = np.clip(subj * (1.0 - face), 0.0, 1.0).astype(np.float32)
        chroma_gate = np.clip(
            body * body_chroma_scale * (0.004 + 0.012 * field_profile)
            + face * face_chroma_scale * (0.008 + 0.014 * field_profile) * (1.0 - 0.88 * face_center_protect)
            + face_side * side_chroma_scale * (0.030 + 0.046 * field_profile)
            + hair * (0.13 + 0.075 * field_profile)
            + edge * (0.16 + 0.095 * field_profile),
            0.0,
            0.18,
        ).astype(np.float32)
        if self.look_safe:
            ab = self._atmosphere_budget or {}
            _face_core_auth = ab.get('face_core_chroma_authority', 0.0)
            _face_side_auth = ab.get('face_side_chroma_authority', 0.0)
            _body_skin_auth = ab.get('body_skin_chroma_authority', 0.0)
            _clothing_auth = ab.get('clothing_chroma_authority', 0.0)
            _hair_auth = ab.get('hair_chroma_authority', 1.0)
            _edge_shell_auth = ab.get('edge_shell_chroma_authority', 1.0)
            _carrier_str = ab.get('atmosphere_carrier_strength', 1.0)
            skin_proxy = self._estimate_skin_proxy(
                np.clip(relit, 0.0, 4.0), np.clip(subject_mask, 0.0, 1.0),
                face_core, hair_region, edge_band)
            body_skin = np.clip(body * skin_proxy, 0.0, 1.0).astype(np.float32)
            clothing_region = np.clip(body * (1.0 - skin_proxy) * (1.0 - hair * 0.5), 0.0, 1.0).astype(np.float32)
            authority_map = (
                face * face_center_protect * (1.0 - _face_core_auth)
                + face_side * (1.0 - _face_side_auth)
                + body_skin * (1.0 - _body_skin_auth)
                + clothing_region * (1.0 - _clothing_auth)
                + hair * _hair_auth * _carrier_str
                + edge * _edge_shell_auth * _carrier_str
            ).astype(np.float32)
            authority_map = np.clip(authority_map, 0.0, 2.0).astype(np.float32)
            chroma_gate *= authority_map
        chroma_gate *= float(np.clip((0.52 + 0.30 * colorfulness) * cyber_gate, 0.28, 0.78))
        chroma_gate = np.clip(chroma_gate, 0.0, 0.18).astype(np.float32)
        if self.look_safe:
            ab = self._budget()
            pc = self._policy_section('chroma')
            core_max = float(np.clip(pc.get('face_core', ab.get('face_core_bg_chroma_budget', 0.001)), 0.001, 0.016))
            side_max = float(np.clip(pc.get('face_side', ab.get('face_side_chroma_inject', 0.020)), core_max, 0.095))
            core_mask = np.clip(face * face_center_protect, 0.0, 1.0).astype(np.float32)
            side_mask = np.clip(face_side, 0.0, 1.0).astype(np.float32)
            non_face_allow = np.full_like(
                chroma_gate,
                float(np.clip(max(pc.get('body', 0.10), pc.get('clothing', 0.12), pc.get('hair', 0.18), pc.get('edge', 0.18)), 0.08, 0.58)),
                dtype=np.float32,
            )
            face_allow = np.clip(core_mask * core_max + side_mask * side_max, core_max, side_max)
            max_allow = np.where(face > 0.01, face_allow, non_face_allow).astype(np.float32)
            chroma_gate = np.minimum(chroma_gate, max_allow).astype(np.float32)
            if not hasattr(self, '_face_protection_debug'):
                self._face_protection_debug = {}
            self._face_protection_debug.update({
                'face_core_bg_chroma_budget': core_max,
                'face_side_chroma_inject_cap': side_max,
                'face_core_protection_weight': float(np.clip(ab.get('face_core_protection_weight', 0.90), 0.0, 1.0)),
            })
        if not np.any(chroma_gate > 1e-4):
            return relit

        target = rel_l[..., None] * fld_dir
        target_l = np.maximum(rgb_luminance(np.clip(target, 0.0, None)), 1e-5)
        target *= (rel_l / target_l)[..., None]
        out = relit * (1.0 - chroma_gate[..., None]) + target * chroma_gate[..., None]
        return np.clip(out, 0.0, 8.0).astype(np.float32)
