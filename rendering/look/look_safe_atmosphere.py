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

class RendererLookSafeAtmosphereMixin:
    def _apply_look_safe_directional_atmosphere(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        clothing_mask: Optional[np.ndarray],
        lighting_info: Optional['LightingInfo'],
        budget: dict,
    ) -> np.ndarray:
        if lighting_info is None or relit.size == 0:
            return relit
        field = getattr(lighting_info, 'gradient_field', {})
        if not isinstance(field, dict) or not field.get('enabled', False):
            return relit
        h, w = relit.shape[:2]
        hb = float(field.get('horizontal_bias', 0.0))
        vb = float(field.get('vertical_bias', 0.0))
        conf = float(field.get('confidence', 0.0))
        if conf < 0.05 and abs(hb) < 0.01 and abs(vb) < 0.01:
            return relit

        x_arr = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
        y_arr = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
        dir_field = (hb * (x_arr - 0.5) + vb * (y_arr - 0.5)).astype(np.float32)
        dir_field *= conf

        spread = budget.get('soft_atmosphere_spread', 0.40)
        ramp_w = float(np.clip(0.15 + 0.30 * spread, 0.15, 0.45))
        lit_side = np.clip((dir_field + ramp_w) / (2.0 * ramp_w + 1e-7), 0.0, 1.0).astype(np.float32)
        shadow_side = (1.0 - lit_side).astype(np.float32)

        locality = budget.get('atmosphere_locality', 0.50)
        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj, 0.0, 1.0).astype(np.float32)
        face_side = np.clip(subj * np.clip(face_core * 1.3 - 0.30, 0.0, 1.0) * (1.0 - face * 0.7), 0.0, 1.0).astype(np.float32)
        hair = np.clip(hair_region * subj, 0.0, 1.0).astype(np.float32)
        edge = np.clip(edge_band * subj, 0.0, 1.0).astype(np.float32)
        shell = np.clip(subj * (1.0 - face * 0.9 - hair * 0.4), 0.0, 1.0).astype(np.float32)
        body_skin = np.clip(subj * (1.0 - face - hair * 0.6 - edge * 0.5), 0.0, 1.0).astype(np.float32)
        cloth = np.clip(clothing_mask * subj, 0.0, 1.0).astype(np.float32) if clothing_mask is not None else np.clip(body_skin * 0.3, 0.0, 1.0).astype(np.float32)

        rim_field = np.clip(shadow_side * (edge + 0.6 * hair) * locality, 0.0, 1.0).astype(np.float32)

        self._v5_dir_field = dir_field
        self._v5_lit_side = lit_side * subj
        self._v5_shadow_side = shadow_side * subj
        self._v5_rim_field = rim_field

        out = relit.copy()
        rel_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5).astype(np.float32)

        side_term = (lit_side - 0.5).astype(np.float32)
        _fc_dir = budget.get('face_core_directional_budget', 0.03)
        _fs_dir = budget.get('face_side_directional_budget', 0.10)
        _bd_dir = budget.get('body_directional_budget', 0.15)
        _cl_dir = budget.get('clothing_directional_budget', 0.20)
        _dir_str = budget.get('directional_light_strength', 0.25)
        _shd_str = budget.get('directional_shadow_strength', 0.15)
        _ctr_str = budget.get('directional_contrast_strength', 0.20)
        _fc_prot = budget.get('face_core_protection_weight', 0.90)

        luma_sculpt = (
            face * _fc_dir * (1.0 - _fc_prot) * side_term
            + face_side * _fs_dir * side_term
            + body_skin * _bd_dir * side_term
            + cloth * _cl_dir * side_term
        ).astype(np.float32)
        contrast_boost = _ctr_str * side_term * np.clip(1.0 - face * _fc_prot, 0.0, 1.0)
        luma_sculpt += contrast_boost * np.clip(hair + edge + shell * 0.5, 0.0, 1.0).astype(np.float32) * 0.3
        luma_sculpt *= _dir_str

        shadow_darken = _shd_str * shadow_side * np.clip(hair * 0.4 + edge * 0.3 + body_skin * 0.2 + cloth * 0.25, 0.0, 1.0).astype(np.float32)
        luma_sculpt -= shadow_darken * 0.5

        luma_sculpt = np.clip(luma_sculpt, -0.12, 0.12).astype(np.float32)
        new_l = np.clip(rel_l + luma_sculpt, 0.0, 1.0).astype(np.float32)
        out *= np.clip(new_l / rel_l, 0.85, 1.18)[..., None]

        try:
            left_col = np.array(field.get('left_color', [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3)
            right_col = np.array(field.get('right_color', [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3)
            top_col = np.array(field.get('top_color', [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3)
            bottom_col = np.array(field.get('bottom_color', [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3)
        except Exception:
            left_col = right_col = top_col = bottom_col = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        def _to_dir(c):
            cl = float(np.dot(c, LUMA))
            return np.clip(c / max(cl, 0.04), 0.30, 3.0).astype(np.float32)

        if hb >= 0:
            lit_color = _to_dir(right_col)
            rim_color = _to_dir(left_col)
        else:
            lit_color = _to_dir(left_col)
            rim_color = _to_dir(right_col)

        _hair_rim_b = budget.get('hair_rim_budget', 0.20)
        _edge_rim_b = budget.get('edge_rim_budget', 0.25)
        _shell_atm_b = budget.get('shell_atmosphere_budget', 0.15)
        _dir_chroma_b = budget.get('directional_chroma_budget', 0.10)
        _rim_chroma_b = budget.get('rim_chroma_budget', 0.15)
        _shadow_tint_b = budget.get('shadow_tint_budget', 0.05)
        _highlight_tint_b = budget.get('highlight_tint_budget', 0.04)

        out_l = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5).astype(np.float32)
        out_dir = np.clip(out / out_l[..., None], 0.0, 3.0).astype(np.float32)

        rim_inject = rim_field * _rim_chroma_b
        hair_rim_gate = np.clip(hair * _hair_rim_b * shadow_side, 0.0, 1.0).astype(np.float32)
        edge_rim_gate = np.clip(edge * _edge_rim_b * shadow_side, 0.0, 1.0).astype(np.float32)
        shell_atm_gate = np.clip(shell * _shell_atm_b * 0.5, 0.0, 1.0).astype(np.float32)

        chroma_gate = np.clip(
            rim_inject
            + hair_rim_gate
            + edge_rim_gate
            + shell_atm_gate
            + cloth * _dir_chroma_b * lit_side * 0.4,
            0.0, 0.30
        ).astype(np.float32)

        chroma_gate *= np.clip(1.0 - face * _fc_prot, 0.0, 1.0)
        chroma_gate *= np.clip(1.0 - body_skin * 0.7, 0.0, 1.0)

        lit_chroma_dir = lit_color.reshape(1, 1, 3)
        rim_chroma_dir = rim_color.reshape(1, 1, 3)
        chroma_target = (
            lit_side[..., None] * lit_chroma_dir
            + shadow_side[..., None] * rim_chroma_dir
        ).astype(np.float32)
        chroma_target_l = np.maximum(rgb_luminance(chroma_target), 1e-5)
        chroma_dir_norm = np.clip(chroma_target / chroma_target_l[..., None], 0.30, 3.0).astype(np.float32)

        out_dir = out_dir * (1.0 - chroma_gate[..., None]) + chroma_dir_norm * chroma_gate[..., None]
        out_dir = np.clip(out_dir, 0.0, 3.0).astype(np.float32)

        shadow_tint_gate = np.clip(shadow_side * _shadow_tint_b * np.clip(face_side + body_skin * 0.5, 0.0, 1.0), 0.0, 0.08).astype(np.float32)
        shadow_tint_gate *= np.clip(1.0 - face * _fc_prot, 0.0, 1.0)
        highlight_tint_gate = np.clip(lit_side * _highlight_tint_b * np.clip(face_side * 0.5 + body_skin * 0.3, 0.0, 1.0), 0.0, 0.06).astype(np.float32)
        highlight_tint_gate *= np.clip(1.0 - face * _fc_prot * 0.8, 0.0, 1.0)

        cool_dir = _to_dir(np.mean([left_col, right_col, top_col, bottom_col], axis=0).astype(np.float32))
        out_dir = out_dir * (1.0 - shadow_tint_gate[..., None]) + cool_dir.reshape(1, 1, 3) * shadow_tint_gate[..., None]
        out_dir = out_dir * (1.0 - highlight_tint_gate[..., None]) + lit_color.reshape(1, 1, 3) * highlight_tint_gate[..., None]

        result = np.clip(out_l[..., None] * out_dir, 0.0, 8.0).astype(np.float32)
        result = result * subj[..., None] + relit * (1.0 - subj[..., None])

        self._v5_after_directional = result
        return result
