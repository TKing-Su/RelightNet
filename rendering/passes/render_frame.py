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


def prepare_render_frame(
    renderer,
    source_linear: np.ndarray,
    mask: np.ndarray,
    albedo_linear: np.ndarray,
    normal_map: np.ndarray,
    depth_map: np.ndarray,
    specular_map: np.ndarray,
    roughness_map: np.ndarray,
    camera_params: Optional[CameraParams],
    depth_scale: Optional[float],
    depth_bias: Optional[float],
) -> Dict[str, object]:
    src_blur = box_blur_rgb(source_linear, passes=2)
    source_lowfreq = box_blur_rgb(source_linear, passes=3)
    detail = np.clip(source_linear - src_blur, -renderer.detail_limit, renderer.detail_limit)
    subject_mask = renderer._prepare_subject_mask(mask)
    edge_band = np.power(
        np.clip(4.0 * subject_mask * (1.0 - subject_mask), 0.0, 1.0),
        renderer.edge_band_power,
    ).astype(np.float32)
    N = decode_normal(normal_map)
    effective_depth_scale = float(renderer.depth_scale if depth_scale is None else depth_scale)
    effective_depth_bias = float(renderer.depth_bias if depth_bias is None else depth_bias)
    P = reconstruct_position(depth_map, renderer.focal_uv, effective_depth_scale, effective_depth_bias, camera_params=camera_params)
    V = safe_norm(-P)
    NdotV = np.clip(np.sum(N * V, axis=-1), 1e-4, 1.0).astype(np.float32)
    facing = np.clip(N[..., 2], 0.0, 1.0).astype(np.float32)
    face_core = (subject_mask * np.clip((facing - 0.28) / (0.92 - 0.28), 0.0, 1.0)).astype(np.float32)
    hair_region = np.clip(subject_mask - face_core * 0.70, 0.0, 1.0).astype(np.float32)

    source_preserve_scale = renderer._policy_value('render_weight', 'source_preserve', 1.0) if renderer._using_continuous_policy() else 1.0
    source_keep = renderer.source_preserve * source_preserve_scale * np.power(np.clip((subject_mask - 0.18) / 0.82, 0.0, 1.0), 1.15)
    source_keep = source_keep[..., None]
    base_subject = (albedo_linear * (1.0 - source_keep) + source_linear * source_keep).astype(np.float32)
    base_subject = base_subject * renderer.local_albedo_keep + source_linear * (1.0 - renderer.local_albedo_keep)
    if renderer._using_continuous_policy():
        intrinsic_face_pullback = np.clip(0.055 * face_core + 0.010 * hair_region, 0.0, 0.080)[..., None]
    else:
        intrinsic_face_pullback = np.clip(0.055 * face_core + 0.035 * hair_region, 0.0, 0.085)[..., None]
    base_subject = base_subject * (1.0 - intrinsic_face_pullback) + source_lowfreq * intrinsic_face_pullback

    clothing_mask = None
    if renderer.look_safe:
        clothing_mask = renderer._estimate_clothing_mask(base_subject, subject_mask, face_core, hair_region, edge_band)

    spec_map = np.clip(box_blur_gray(specular_map, passes=1), 0.0, 1.0).astype(np.float32)
    rough_map = np.clip(box_blur_gray(roughness_map, passes=1), 0.0, 1.0).astype(np.float32)
    roughness = np.clip(0.28 + 0.50 * rough_map - 0.040 * face_core - 0.020 * hair_region - 0.010 * edge_band, 0.16, 0.88).astype(np.float32)
    spec_map = np.clip(0.10 + 0.30 * spec_map + 0.080 * face_core + 0.050 * hair_region + 0.030 * edge_band, 0.02, 0.56).astype(np.float32)
    if renderer._using_continuous_policy():
        roughness = np.clip(roughness + 0.12 * hair_region - 0.02 * edge_band, 0.18, 0.94).astype(np.float32)
        spec_map = np.clip(spec_map - 0.050 * hair_region + 0.014 * edge_band, 0.018, 0.46).astype(np.float32)
    F0_scalar = np.clip(0.028 + 0.040 * spec_map, 0.028, 0.065).astype(np.float32)
    F0 = np.repeat(F0_scalar[..., None], 3, axis=-1).astype(np.float32)
    kd = np.clip(1.0 - spec_map[..., None] * 0.10, 0.76, 1.0).astype(np.float32)
    ao = renderer._compute_occlusion(N, depth_map, subject_mask)
    source_shape = renderer._compute_source_shading(source_linear, albedo_linear, subject_mask)
    intrinsic_gloss = renderer._compute_intrinsic_gloss_control(
        source_linear,
        albedo_linear,
        N,
        depth_map,
        spec_map,
        roughness,
        subject_mask,
        face_core,
        hair_region,
        edge_band,
    )

    return {
        'source_lowfreq': source_lowfreq,
        'source_preserve_scale': source_preserve_scale,
        'detail': detail,
        'subject_mask': subject_mask,
        'edge_band': edge_band,
        'N': N,
        'P': P,
        'V': V,
        'NdotV': NdotV,
        'facing': facing,
        'face_core': face_core,
        'hair_region': hair_region,
        'base_subject': base_subject,
        'clothing_mask': clothing_mask,
        'spec_map': spec_map,
        'roughness': roughness,
        'F0': F0,
        'kd': kd,
        'ao': ao,
        'source_shape': source_shape,
        'intrinsic_gloss': intrinsic_gloss,
    }
