from __future__ import annotations

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

class RendererFaceDetailMixin:
    def _soften_face_hard_light_edges(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
    ) -> np.ndarray:
        """Soften artificial hard lighting transitions while preserving real facial features.

        It compares the relit luminance band-pass with the source band-pass.  If an
        edge exists only after relighting, it is likely a hard illumination boundary;
        if the edge already exists in the source, it is likely an eye/lip/nose/skin
        structure and should be preserved.  This solves the 'hard face' problem
        without going back to a blurred/airbrushed result.
        """
        if relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj * (1.0 - 0.35 * np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
        if not np.any(face > 0.05):
            return relit

        l_rel = np.maximum(rgb_luminance(np.clip(relit, 0.0, None)), 1e-5).astype(np.float32)
        l_src = np.maximum(rgb_luminance(np.clip(source_linear, 0.0, None)), 1e-5).astype(np.float32)
        rel_band = np.abs(box_blur_gray(l_rel, passes=2) - box_blur_gray(l_rel, passes=6))
        src_band = np.abs(box_blur_gray(l_src, passes=2) - box_blur_gray(l_src, passes=6))
        artificial_edge = np.clip((rel_band - src_band * 0.78 - 0.003) / 0.024, 0.0, 1.0).astype(np.float32)
        source_feature_keep = np.clip((src_band - 0.004) / 0.026, 0.0, 1.0).astype(np.float32)

        # Stage44: only smooth artificial illumination boundaries.  Source features
        # such as eyes, lips, nostril/bridge edges and cheek folds are protected,
        # otherwise the face loses mid-frequency structure again.
        smooth_l = box_blur_gray(l_rel, passes=4)
        soft_gate = np.clip(
            face * artificial_edge * (1.0 - 0.70 * source_feature_keep) * 0.36,
            0.0,
            0.24,
        ).astype(np.float32)
        if not np.any(soft_gate > 1e-4):
            return relit
        l_new = l_rel * (1.0 - soft_gate) + smooth_l * soft_gate
        out = relit * np.clip(l_new / l_rel, 0.88, 1.10)[..., None]
        return np.clip(out, 0.0, 8.0).astype(np.float32)


    def _restore_face_soft_detail(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
    ) -> np.ndarray:
        """Restore soft facial structure without high-frequency pore refill.

        Stage41 separates structure from micro-texture.  It restores a controlled
        blur1-blur4 RGB band-pass and uses a source-luminance feature gate so eyes,
        mouth, nose and cheek transitions recover, while tiny pores/noise/blemishes
        are still excluded.
        """
        if relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj * (1.0 - 0.25 * np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
        if not np.any(face > 0.05):
            return relit

        src_l = rgb_luminance(np.clip(source_linear, 0.0, None)).astype(np.float32)
        src_mid_l = box_blur_gray(src_l, passes=1) - box_blur_gray(src_l, passes=4)
        feature_gate = np.clip((np.abs(src_mid_l) - 0.0060) / 0.027, 0.0, 1.0).astype(np.float32)
        src_bp = box_blur_rgb(source_linear, passes=1) - box_blur_rgb(source_linear, passes=4)
        src_bp = np.clip(src_bp, -0.030, 0.030).astype(np.float32)
        # Recover only a tiny amount of fine structural detail.  A higher feature
        # threshold avoids bringing back isolated pores, spots and small blemishes.
        src_fine = np.clip(source_linear - box_blur_rgb(source_linear, passes=1), -0.008, 0.008).astype(np.float32)
        subj_area = float(np.mean(subj))
        face_area = float(np.mean(face))
        closeup_gate = float(np.clip((face_area / max(subj_area, 1e-5) - 0.30) / 0.42, 0.0, 1.0))

        detail_damp = 1.0 if normalize_style_mode(getattr(self, 'style_mode', 'quality'), fallback='quality', look_safe=self.look_safe) == 'neon' else 0.58
        gate = np.clip(face * (0.070 + 0.080 * feature_gate) * detail_damp + subj * 0.010 * (1.0 - np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 0.16).astype(np.float32)
        fine_gate = np.clip(face * (0.003 + 0.008 * feature_gate) * (1.0 - 0.70 * closeup_gate) * detail_damp, 0.0, 0.010).astype(np.float32)
        out = relit + src_bp * gate[..., None] + src_fine * fine_gate[..., None]
        return np.clip(out, 0.0, 8.0).astype(np.float32)


    def _restore_face_midfreq_clarity(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
    ) -> np.ndarray:
        """Final luminance-only clarity pass for readable but soft facial features."""
        if relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj * (1.0 - 0.30 * np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
        if not np.any(face > 0.05):
            return relit
        src_l = rgb_luminance(np.clip(source_linear, 0.0, None)).astype(np.float32)
        mid = box_blur_gray(src_l, passes=1) - box_blur_gray(src_l, passes=5)
        mid = np.clip(mid, -0.032, 0.034).astype(np.float32)
        feature = np.clip((np.abs(mid) - 0.004) / 0.024, 0.0, 1.0).astype(np.float32)
        rel_l = np.maximum(rgb_luminance(np.clip(relit, 0.0, None)), 1e-5).astype(np.float32)
        rel_dir = np.clip(relit / rel_l[..., None], 0.45, 2.35).astype(np.float32)
        detail_damp = 1.0 if normalize_style_mode(getattr(self, 'style_mode', 'quality'), fallback='quality', look_safe=self.look_safe) == 'neon' else 0.56
        gate = np.clip(face * (0.088 + 0.066 * feature) * detail_damp + 0.030 * subj * (1.0 - np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 0.16).astype(np.float32)
        return np.clip(relit + mid[..., None] * rel_dir * gate[..., None], 0.0, 8.0).astype(np.float32)



    def _suppress_face_micro_blemishes(
        self,
        relit: np.ndarray,
        source_linear: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
        hair_region: np.ndarray,
        edge_band: np.ndarray,
    ) -> np.ndarray:
        """Reduce isolated skin micro-blemishes without blurring real facial features.

        This pass is intentionally local and weak.  It only acts on smooth face
        areas where the source has tiny isolated high-frequency luma/chroma spots,
        while preserving mid-frequency structures such as eyes, brows, nose, lips
        and cheek transitions.
        """
        if relit.size == 0 or not np.any(subject_mask > 0.10):
            return relit
        subj = np.clip(subject_mask, 0.0, 1.0).astype(np.float32)
        face = np.clip(face_core * subj * (1.0 - 0.32 * np.clip(hair_region + edge_band, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
        if not np.any(face > 0.05):
            return relit

        src_l = rgb_luminance(np.clip(source_linear, 0.0, None)).astype(np.float32)
        rel_l = rgb_luminance(np.clip(relit, 0.0, None)).astype(np.float32)

        src_mid = np.abs(box_blur_gray(src_l, passes=1) - box_blur_gray(src_l, passes=5))
        structure_keep = np.clip((src_mid - 0.0075) / 0.030, 0.0, 1.0).astype(np.float32)

        src_fine = src_l - box_blur_gray(src_l, passes=1)
        rel_fine = rel_l - box_blur_gray(rel_l, passes=1)
        fine_energy = np.maximum(np.abs(src_fine), np.abs(rel_fine))
        micro_spot = np.clip((fine_energy - 0.0045) / 0.016, 0.0, 1.0).astype(np.float32)

        # Small red/purple chroma spots are often amplified by cyber lighting.
        src_gray = src_l[..., None]
        src_chroma_fine = np.mean(np.abs((source_linear - src_gray) - box_blur_rgb(source_linear - src_gray, passes=1)), axis=-1)
        chroma_spot = np.clip((src_chroma_fine - 0.0035) / 0.018, 0.0, 1.0).astype(np.float32)

        subj_area = float(np.mean(subj))
        face_area = float(np.mean(face))
        closeup_gate = float(np.clip((face_area / max(subj_area, 1e-5) - 0.30) / 0.42, 0.0, 1.0))

        smooth_skin = face * (1.0 - 0.84 * structure_keep)
        blemish_gate = np.clip(
            smooth_skin * np.maximum(micro_spot, chroma_spot) * (0.20 + 0.18 * closeup_gate),
            0.0,
            0.24,
        ).astype(np.float32)
        if not np.any(blemish_gate > 1e-4):
            return relit

        # Keep this local and gentle: slightly stronger in close-ups, but still
        # only one blur pass so the face does not become waxy.
        smooth_rgb = box_blur_rgb(relit, passes=1)
        out = relit * (1.0 - blemish_gate[..., None]) + smooth_rgb * blemish_gate[..., None]
        return np.clip(out, 0.0, 8.0).astype(np.float32)
