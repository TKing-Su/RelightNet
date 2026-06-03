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

class RendererSkinControlMixin:
    def _stabilize_skin_color(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        lighting_info: Optional[LightingInfo] = None,
    ) -> np.ndarray:
        """Limit unnatural face hue/color wash while preserving background tint.

        This is not a full face parser.  It uses the front-facing subject mask as a
        soft skin/face proxy and only limits chroma direction, not luminance shape.
        It prevents sunset/neon backgrounds from turning the face into an unnatural
        solid-color surface.
        """
        if relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        gate = np.clip(face_core * subject_mask, 0.0, 1.0).astype(np.float32)
        subj_area = float(np.mean(np.clip(subject_mask, 0.0, 1.0)))
        face_area = float(np.mean(np.clip(face_core * subject_mask, 0.0, 1.0)))
        face_ratio = face_area / max(subj_area, 1e-5)
        closeup_gate = float(np.clip((face_ratio - 0.30) / 0.42, 0.0, 1.0))

        if self._using_continuous_policy():
            skin_limit = self._policy_value('chroma', 'skin_tint_limit', 0.08)
            face_readability = self._policy_value('exposure', 'face_readability', 0.48)
            strength = float(np.clip(0.055 + 0.36 * max(0.0, 0.10 - skin_limit) + 0.045 * face_readability, 0.055, 0.18))
        elif lighting_info is not None:
            mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower()
            neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower()
            if neon != 'off' or mode == 'rich':
                strength = 0.095
            elif mode == 'monotone':
                strength = 0.085
            else:
                strength = 0.080
        else:
            strength = 0.18
        # When the crop is mostly face, colored spill is visually amplified.
        # Pull only the face-core chroma direction slightly toward the source skin
        # direction while preserving the relit luminance.
        strength = float(np.clip(strength + 0.175 * closeup_gate, 0.0, 0.34))
        src_l = rgb_luminance(np.clip(source_linear, 0.0, None))
        out_l = rgb_luminance(np.clip(relit, 0.0, None))
        src_dir = source_linear / np.maximum(src_l[..., None], 1e-5)
        src_dir = np.clip(src_dir, 0.62, 1.72)
        natural = src_dir * out_l[..., None]
        # Only pull back the face core.  Hair/edge/background-driven rim stay colorful.
        g = (strength * np.power(gate, 0.78))[..., None]
        return np.clip(relit * (1.0 - g) + natural * g, 0.0, 8.0).astype(np.float32)


    def _apply_face_color_density_control(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo] = None,
    ) -> np.ndarray:
        """Compress excessive face color density without removing background lighting.

        This is deliberately not a source-color replacement.  The relit image keeps
        its background-driven hue, rim color and shadow color; only over-saturated
        or over-dark face-core chroma is softened.  Hair, shoulders and alpha edges
        remain strongly affected by the background.
        """
        if relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        out = relit.astype(np.float32, copy=True)
        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj, 0.0, 1.0).astype(np.float32)
        if not np.any(face > 0.05):
            return out

        field = getattr(lighting_info, 'gradient_field', {}) if lighting_info is not None else {}
        if not isinstance(field, dict):
            field = {}
        colorfulness = float(np.clip(field.get('colorfulness', getattr(lighting_info, 'palette_diversity', 0.30) if lighting_info is not None else 0.30), 0.0, 1.0))
        if self._using_continuous_policy():
            colorfulness = float(np.clip(max(colorfulness, self._policy_value('chroma', 'palette_separation', 0.25) * 0.60), 0.0, 1.0))
        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower() if lighting_info is not None else 'balanced'
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower() if lighting_info is not None else 'off'
        subj_area = float(np.mean(subj))
        face_area = float(np.mean(face))
        face_ratio = face_area / max(subj_area, 1e-5)
        closeup_gate = float(np.clip((face_ratio - 0.30) / 0.42, 0.0, 1.0))

        l_relit = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5).astype(np.float32)
        l_src = np.maximum(rgb_luminance(np.clip(source_linear, 0.0, None)), 1e-5).astype(np.float32)
        gray_relit = l_relit[..., None]
        gray_src = l_src[..., None]

        relit_chroma_amount = (np.mean(np.abs(out - gray_relit), axis=-1) / l_relit).astype(np.float32)
        src_chroma_amount = (np.mean(np.abs(source_linear - gray_src), axis=-1) / l_src).astype(np.float32)

        # Allow a real background color cast, especially for neon/rich scenes, but
        # do not allow the face core to become a solid red/purple/orange plate.
        if self._using_continuous_policy():
            skin_limit = self._policy_value('chroma', 'skin_tint_limit', 0.08)
            allowed_hi = float(np.clip(0.18 + 1.35 * skin_limit + 0.05 * colorfulness, 0.18, 0.34))
        else:
            allowed_hi = 0.340 if (bg_mode == 'rich' or neon != 'off') else 0.290
        allowed = np.clip(
            src_chroma_amount * (1.10 + 0.08 * colorfulness) + 0.045 + 0.060 * colorfulness,
            0.085,
            allowed_hi * (1.0 - 0.38 * closeup_gate),
        ).astype(np.float32)
        oversat = np.clip((relit_chroma_amount - allowed) / np.maximum(allowed, 1e-4), 0.0, 1.0).astype(np.float32)
        # Stronger in face close-ups, weaker near hair and edges so rim lighting is kept.
        compression_gate = np.clip(
            face * (0.090 + 0.18 * closeup_gate + 0.34 * oversat) * (1.0 - 0.38 * np.clip(hair_region + edge_band, 0.0, 1.0)),
            0.0,
            0.46,
        ).astype(np.float32)
        compressed = gray_relit + (out - gray_relit) * (1.0 - (0.54 + 0.18 * closeup_gate) * compression_gate[..., None])
        out = out * (1.0 - compression_gate[..., None]) + compressed * compression_gate[..., None]

        # If a dark neon scene makes the face both very saturated and under-exposed,
        # lift luminance by scaling the current relit color.  This keeps the hue from
        # background lighting; it does not pull the face back to source color.
        l_after = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5).astype(np.float32)
        min_face_luma = np.clip(l_src * (0.56 + 0.06 * colorfulness), 0.082, 0.360).astype(np.float32)
        dark_gate = np.clip((min_face_luma - l_after) / np.maximum(min_face_luma, 1e-4), 0.0, 1.0)
        dark_gate *= np.clip(face * (0.38 + 0.42 * oversat), 0.0, 0.74)
        l_target = l_after * (1.0 - dark_gate) + min_face_luma * dark_gate
        out *= np.clip(l_target / l_after, 0.85, 1.38)[..., None]

        return np.clip(out, 0.0, 8.0).astype(np.float32)


    def _apply_body_skin_cast_governor(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo] = None,
    ) -> np.ndarray:
        """Reduce unsupported body/shoulder skin tint while preserving cyber rim color.

        Metrics showed that after detail recovery the remaining issue is largely
        skin/body cast rather than face structure.  This function only targets the
        non-face body region and avoids killing colorful hair/edge light.
        """
        if relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        body = np.clip(subj * (1.0 - 0.92 * np.clip(face_core, 0.0, 1.0)) * (1.0 - 0.34 * np.clip(hair_region, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
        rim_guard = np.clip(1.0 - 0.72 * np.clip(edge_band, 0.0, 1.0), 0.18, 1.0).astype(np.float32)
        body *= rim_guard
        if not np.any(body > 0.05):
            return relit

        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower() if lighting_info is not None else 'balanced'
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower() if lighting_info is not None else 'off'
        field = getattr(lighting_info, 'gradient_field', {}) if lighting_info is not None else {}
        if not isinstance(field, dict):
            field = {}
        colorfulness = float(np.clip(field.get('colorfulness', getattr(lighting_info, 'palette_diversity', 0.30) if lighting_info is not None else 0.30), 0.0, 1.0))
        if self._using_continuous_policy():
            colorfulness = float(np.clip(max(colorfulness, self._policy_value('chroma', 'body_tint_strength', 0.08) * 2.0), 0.0, 1.0))

        l_relit = np.maximum(rgb_luminance(np.clip(relit, 0.0, None)), 1e-5).astype(np.float32)
        l_src = np.maximum(rgb_luminance(np.clip(source_linear, 0.0, None)), 1e-5).astype(np.float32)
        gray_relit = l_relit[..., None]
        gray_src = l_src[..., None]
        relit_chroma_amount = (np.mean(np.abs(relit - gray_relit), axis=-1) / l_relit).astype(np.float32)
        src_chroma_amount = (np.mean(np.abs(source_linear - gray_src), axis=-1) / l_src).astype(np.float32)
        if self._using_continuous_policy():
            allowed_hi = float(np.clip(0.20 + 0.70 * self._policy_value('chroma', 'body_tint_strength', 0.08), 0.20, 0.34))
        else:
            allowed_hi = 0.300 if (bg_mode == 'rich' or neon != 'off') else 0.250
        allowed = np.clip(
            src_chroma_amount * (1.14 + 0.06 * colorfulness) + 0.040 + 0.040 * colorfulness,
            0.075,
            allowed_hi,
        ).astype(np.float32)
        oversat = np.clip((relit_chroma_amount - allowed) / np.maximum(allowed, 1e-4), 0.0, 1.0).astype(np.float32)
        src_dir = np.clip(source_linear / l_src[..., None], 0.64, 1.64).astype(np.float32)
        natural = np.clip(src_dir * l_relit[..., None], 0.0, 8.0).astype(np.float32)
        mix = np.clip(body * (0.05 + 0.26 * oversat) * (1.0 - 0.38 * np.clip(edge_band, 0.0, 1.0)), 0.0, 0.22).astype(np.float32)
        out = relit * (1.0 - mix[..., None]) + natural * mix[..., None]
        return np.clip(out, 0.0, 8.0).astype(np.float32)


    def _apply_face_air_and_texture_guard(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
        lighting_info: Optional[LightingInfo] = None,
    ) -> np.ndarray:
        """Add back face readability and fine texture without killing background color.

        This is intentionally gentle: it does not force the face back to source
        skin tone.  It only prevents dark/color-dense mask areas and restores a
        luminance-detail residual, while keeping the relit hue and rim colors.
        """
        if relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        out = relit.astype(np.float32, copy=True)
        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj * (1.0 - 0.38 * np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
        if not np.any(face > 0.05):
            return out

        bg_mode = str(getattr(lighting_info, 'background_mode', 'balanced')).lower() if lighting_info is not None else 'balanced'
        neon = str(getattr(lighting_info, 'neon_strength', 'off')).lower() if lighting_info is not None else 'off'
        if self._using_continuous_policy():
            face_readability = self._policy_value('exposure', 'face_readability', 0.48)
            color_carrier = self._policy_value('chroma', 'palette_separation', 0.22)
            rich_scene = color_carrier > 0.42
        else:
            face_readability = 0.48
            rich_scene = (bg_mode == 'rich' or neon != 'off')

        l_out = np.maximum(rgb_luminance(np.clip(out, 0.0, None)), 1e-5).astype(np.float32)
        l_src = np.maximum(rgb_luminance(np.clip(source_linear, 0.0, None)), 1e-5).astype(np.float32)
        target = np.clip(l_src * ((0.58 + 0.06 * face_readability) if rich_scene else (0.56 + 0.04 * face_readability)), 0.090, 0.390).astype(np.float32)
        lift_gate = np.clip((target - l_out) / np.maximum(target, 1e-4), 0.0, 1.0)
        lift_gate = np.clip(lift_gate * face * (0.36 if rich_scene else 0.28), 0.0, 0.44)
        if self.look_safe:
            lift_gate *= self._atmosphere_budget['face_luma_lift_gate'] if self._atmosphere_budget else 0.28
            if self._atmosphere_budget:
                _air_cap = self._atmosphere_budget.get('air_guard_luma_lift_cap', 0.12)
                lift_gate = np.minimum(lift_gate, _air_cap)
        l_new = l_out * (1.0 - lift_gate) + target * lift_gate
        out *= np.clip(l_new / l_out, 0.90, 1.28)[..., None]

        # Stage36: do not add source high-frequency residual back to face.
        # The previous luminance-detail addition still restored pores, color noise
        # and small blemishes after every dark relight. Instead, use a very small
        # face-only high-frequency damping pass so the relit skin surface stays even,
        # while all 3D lighting/shadow structure remains in the low-frequency field.
        face_soft_gate = np.zeros_like(face, dtype=np.float32)
        # Stage44: do not run a global face smoothing pass here.  Hard light
        # boundaries are handled by _soften_face_hard_light_edges(), while this
        # guard should not blur eyes, nose, mouth or cheek structure.
        return np.clip(out, 0.0, 8.0).astype(np.float32)
