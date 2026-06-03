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

class RendererV32ReconcileMixin:
    def _v32_reconcile_modular_blocks(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo],
    ) -> np.ndarray:
        """Separate old-chain results into explicit logical blocks.

        This is not a replacement for the old relight chain.  It is a router/contract:
        - base PBR / directional light remains from the old chain;
        - face_core owns skin/chroma/detail preservation;
        - body owns skin-safe luma lighting;
        - shell/rim own background atmosphere;
        - all corrections are in linear RGB before display finish.
        """
        style = self._v32_style_key(lighting_info)
        b = self._v32_style_budgets(style)
        if self.look_safe and style != 'continuous':
            b = dict(b)
            ab = self._atmosphere_budget or {}
            b['face_bg'] = ab.get('v32_face_bg_budget', 0.000)
            b['face_side_bg'] = min(b['face_side_bg'], ab.get('v32_face_side_budget', 0.005))
            b['shell_bg'] = min(b['shell_bg'] * ab.get('v32_shell_scale', 1.35), 0.85)
            b['rim_bg'] = min(b['rim_bg'] * ab.get('v32_rim_scale', 1.25), 0.90)
        r = self._v32_region_masks(subject_mask, face_core, hair_region, edge_band)
        face = r['face_core']; face_side = r['face_side']; body = r['body_skin']; shell = r['shell']; rim = r['rim_edge']; subj = r['subject']
        amb_c, key_c = self._v32_background_chroma(lighting_info, style)
        lit_side, shadow_side = self._v32_key_side_ramp(relit, lighting_info, style=style)

        src_y = np.maximum(rgb_luminance(source_linear), 1e-5).astype(np.float32)
        old_y = np.maximum(rgb_luminance(relit), 1e-5).astype(np.float32)
        src_chroma = np.clip(source_linear / src_y[..., None], 0.0, 3.0).astype(np.float32)
        old_chroma = np.clip(relit / old_y[..., None], 0.0, 3.0).astype(np.float32)

        # 1) Luma block: retain old-chain sculpting, then make side-light readable in
        # face/body sample regions.  Color is not touched here.
        y = old_y.copy()
        side_term = (lit_side - 0.5).astype(np.float32)
        _ls_luma_scale = (self._atmosphere_budget['v32_luma_block_scale'] if self._atmosphere_budget else 0.30) if self.look_safe else 1.0
        _ls_lift_mul = (self._atmosphere_budget.get('v32_positive_luma_gate', 1.0) if self._atmosphere_budget else 1.0) if self.look_safe else 1.0
    
        # Look-safe continuous mode: use unified luma mixing (natural/default coefficients)
        # instead of style-specific branches. This ensures consistent behavior regardless
        # of filename hints or discrete style classification.
        use_unified_luma = (self.look_safe and style == 'continuous')
    
        if style == 'warm' and not use_unified_luma:
            _face_delta = b['face_target'] - y
            _body_delta = b['body_target'] - y
            if self.look_safe:
                _face_delta = np.where(_face_delta > 0, _face_delta * _ls_lift_mul, _face_delta)
                _body_delta = np.where(_body_delta > 0, _body_delta * _ls_lift_mul, _body_delta)
            y += face * (0.22 * _ls_luma_scale * _face_delta + 0.95 * b['dir'] * lit_side - 0.45 * b['shadow'] * shadow_side)
            y += body * (0.18 * _ls_luma_scale * _body_delta + 0.82 * b['dir'] * lit_side - 0.36 * b['shadow'] * shadow_side)
        elif style == 'cyber' and not use_unified_luma:
            _face_delta = b['face_target'] - y
            _body_delta = b['body_target'] - y
            if self.look_safe:
                _face_delta = np.where(_face_delta > 0, _face_delta * _ls_lift_mul, _face_delta)
                _body_delta = np.where(_body_delta > 0, _body_delta * _ls_lift_mul, _body_delta)
            y += face * (0.22 * _ls_luma_scale * _face_delta + 0.78 * b['dir'] * side_term - 0.55 * b['shadow'] * shadow_side)
            y += body * (0.18 * _ls_luma_scale * _body_delta + 0.72 * b['dir'] * side_term - 0.46 * b['shadow'] * shadow_side)
        else:
            # Natural/default and continuous: unified mixing
            _face_delta = b['face_target'] - y
            _body_delta = b['body_target'] - y
            if self.look_safe:
                _face_delta = np.where(_face_delta > 0, _face_delta * _ls_lift_mul, _face_delta)
                _body_delta = np.where(_body_delta > 0, _body_delta * _ls_lift_mul, _body_delta)
            y += face * (0.20 * _ls_luma_scale * _face_delta + 0.55 * b['dir'] * side_term - 0.45 * b['shadow'] * shadow_side)
            y += body * (0.16 * _ls_luma_scale * _body_delta + 0.45 * b['dir'] * side_term - 0.34 * b['shadow'] * shadow_side)
        y += shell * (0.070 if (style == 'cyber' and not use_unified_luma) else (0.018 if (style == 'warm' and not use_unified_luma) else 0.004))
        y += rim * (0.085 if (style == 'cyber' and not use_unified_luma) else (0.030 if (style == 'warm' and not use_unified_luma) else 0.008)) * lit_side
        if style == 'natural':
            # Natural/misty keeps air but should not leak chroma into skin.
            y = y * (1.0 - 0.10 * subj) + box_blur_gray(y, passes=1) * (0.10 * subj)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)

        # 2) Chroma block: old-chain chroma is allowed only outside protected skin.
        # Face-core never carries atmosphere color.
        chroma = old_chroma.copy()
        face_bg_budget = b['face_bg']
        if self.look_safe and self._atmosphere_budget:
            _pc = self._policy_section('chroma')
            face_bg_budget = float(np.clip(_pc.get('face_core', self._atmosphere_budget.get('face_core_bg_chroma_budget', face_bg_budget)), 0.0, 0.016))
        face_target = (1.0 - face_bg_budget) * src_chroma + face_bg_budget * amb_c.reshape(1, 1, 3)
        face_side_target = (1.0 - b['face_side_bg']) * src_chroma + b['face_side_bg'] * key_c.reshape(1, 1, 3)
        body_target = (1.0 - b['body_bg']) * src_chroma + b['body_bg'] * (0.55 * amb_c + 0.45 * key_c).reshape(1, 1, 3)
        shell_target = (1.0 - b['shell_bg']) * src_chroma + b['shell_bg'] * (0.45 * amb_c + 0.55 * key_c).reshape(1, 1, 3)
        rim_target = (1.0 - b['rim_bg']) * src_chroma + b['rim_bg'] * key_c.reshape(1, 1, 3)
        chroma = chroma * (1.0 - face[..., None]) + face_target * face[..., None]
        chroma = chroma * (1.0 - face_side[..., None]) + face_side_target * face_side[..., None]
        chroma = chroma * (1.0 - body[..., None]) + body_target * body[..., None]
        chroma = chroma * (1.0 - shell[..., None]) + shell_target * shell[..., None]
        chroma = chroma * (1.0 - rim[..., None]) + rim_target * rim[..., None]
        chroma = np.clip(chroma, 0.0, 3.0).astype(np.float32)
        if self.look_safe and self._atmosphere_budget:
            _wscg = self._atmosphere_budget.get('warm_skin_contamination_guard', 0.0)
            _sywg = self._atmosphere_budget.get('skin_yellow_wash_guard', 0.0)
            if _wscg > 0.01 or _sywg > 0.01:
                _skin_pull = np.clip(_wscg * (face + 0.70 * face_side + 0.50 * body), 0.0, 1.0).astype(np.float32)
                chroma = chroma * (1.0 - _skin_pull[..., None]) + src_chroma * _skin_pull[..., None]
                if _sywg > 0.01:
                    _chroma_sat = np.clip(np.max(chroma, axis=-1) - np.min(chroma, axis=-1), 0.0, 2.0)
                    _yellow_hue = np.clip(chroma[..., 0] - 0.90, 0.0, 0.5) * np.clip(chroma[..., 1] - 0.90, 0.0, 0.5)
                    _yellow_mask = np.clip(_yellow_hue * _chroma_sat * 8.0, 0.0, 1.0).astype(np.float32)
                    _yellow_pull = np.clip(_sywg * _yellow_mask * (face + 0.80 * face_side + 0.45 * body), 0.0, 1.0).astype(np.float32)
                    chroma = chroma * (1.0 - _yellow_pull[..., None]) + src_chroma * _yellow_pull[..., None]
                chroma = np.clip(chroma, 0.0, 3.0).astype(np.float32)

        # 3) Detail block: luma-only, style budgeted.  This avoids RGB pore/noise
        # refill and avoids warm/red high-frequency excess.
        if style == 'warm':
            # Red/warm after V32.3 had correct direction but face_detail_ratio dropped.
            # Restore only luminance detail, with a higher bound, while keeping RGB
            # chroma anchored to source skin.
            src_detail = np.clip(src_y - box_blur_gray(src_y, passes=1), -0.034, 0.034).astype(np.float32)
            detail_gate = np.clip(0.82 * b['detail'] * face + 0.34 * b['detail'] * body + 0.04 * b['detail'] * shell + 0.08 * b['detail'] * rim, 0.0, 0.62)
        else:
            src_detail = np.clip(src_y - box_blur_gray(src_y, passes=2), -0.018, 0.018).astype(np.float32)
            detail_face_mul = 0.32 if style == 'natural' else 0.40
            detail_gate = np.clip(detail_face_mul * b['detail'] * face + 0.24 * b['detail'] * body + 0.05 * b['detail'] * shell + 0.12 * b['detail'] * rim, 0.0, 0.34)
        if self.look_safe and self._atmosphere_budget:
            lowkey_floor = float(np.clip(self._atmosphere_budget.get('lowkey_detail_floor', 0.0), 0.0, 0.25))
            if lowkey_floor > 1e-6:
                detail_gate = np.maximum(detail_gate, lowkey_floor * np.clip(face + 0.45 * body + 0.25 * hair_region, 0.0, 1.0)).astype(np.float32)
        y = np.clip(y + src_detail * detail_gate, 0.0, 1.0).astype(np.float32)

        reconstructed = np.clip(y[..., None] * chroma, 0.0, 1.0).astype(np.float32)
        # Blend with the old chain rather than replacing it.  This keeps the actual
        # relight/shadow/specular chain alive while enforcing region ownership.
        if style == 'warm':
            # Direction was fixed in V32.3; the remaining red failure is face/body
            # chroma cast and low luma detail.  Therefore replace most old-chain
            # RGB in skin regions with the modular reconstruction, but keep shell/rim
            # less constrained so the legacy light feel is not flattened.
            if self.look_safe:
                _warm_cap = self._atmosphere_budget['v32_warm_gate_cap'] if self._atmosphere_budget else 0.62
                gate_profile = 0.58 * face + 0.52 * body + 0.30 * shell + 0.36 * rim + 0.40 * face_side
                gate_cap = _warm_cap
            else:
                gate_profile = 1.18 * face + 0.98 * body + 0.30 * shell + 0.36 * rim + 0.58 * face_side
                gate_cap = 0.985
        elif style == 'cyber':
            gate_profile = 0.86 * face + 0.68 * body + 0.86 * shell + 0.82 * rim + 0.28 * face_side
            gate_cap = 0.88
        else:
            gate_profile = 0.86 * face + 0.58 * body + 0.42 * shell + 0.48 * rim + 0.20 * face_side
            gate_cap = 0.82
        gate = np.clip(b['reconcile'] * gate_profile, 0.0, gate_cap).astype(np.float32)
        if self.look_safe and self._atmosphere_budget:
            _fc = self._atmosphere_budget.get('v32_face_gate_cap', 0.60)
            _bc = self._atmosphere_budget.get('v32_body_gate_cap', 0.55)
            _pbr = self._atmosphere_budget.get('pbr_preserve_strength', 0.0)
            _gate_face = np.minimum(gate * face, _fc * face)
            _gate_body = np.minimum(gate * body, _bc * body)
            _gate_face *= (1.0 - _pbr)
            _gate_body *= (1.0 - _pbr)
            _gate_other = gate * np.clip(1.0 - face - body, 0.0, 1.0)
            gate = np.clip(_gate_face + _gate_body + _gate_other, 0.0, gate_cap).astype(np.float32)
            _f_vals = gate[face > 0.5]
            _b_vals = gate[body > 0.5]
            if _f_vals.size > 0:
                print(f"  [V32 gate] face: mean={float(_f_vals.mean()):.3f} p95={float(np.percentile(_f_vals, 95)):.3f} max={float(_f_vals.max()):.3f}")
            if _b_vals.size > 0:
                print(f"  [V32 gate] body: mean={float(_b_vals.mean()):.3f} p95={float(np.percentile(_b_vals, 95)):.3f} max={float(_b_vals.max()):.3f}")
        out = relit * (1.0 - gate[..., None]) + reconstructed * gate[..., None]
        print(
            f"V32_4_MODULE_ROUTER: style={style}, face_bg={b['face_bg']:.3f}, body_bg={b['body_bg']:.3f}, "
            f"shell_bg={b['shell_bg']:.3f}, rim_bg={b['rim_bg']:.3f}, core_chain=kept, final_patch=off"
        )
        return np.clip(out, 0.0, 8.0).astype(np.float32)
