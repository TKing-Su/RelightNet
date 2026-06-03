from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class RelightPreset:
    """
    Runtime-tunable relighting parameters.

    The preset is NOT meant to replace background-driven lighting extraction.
    It only defines the style boundary: how strong shadows/rims/spill/post-grade
    should be. The actual light colors, directions and count are still extracted
    from the background image by BackgroundStudioLightExtractor.
    """
    source_preserve: float = 0.02
    source_shading_preserve: float = 0.18
    subject_mix: float = 0.84
    ambient_strength: float = 0.10
    fill_strength: float = 0.04
    multi_ambient_strength: float = 0.26
    multi_ambient_wrap: float = 0.42
    multi_ambient_side_bias: float = 0.65
    multi_ambient_face_bias: float = 0.12
    shadow_sculpt_strength: float = 0.34
    key_shadow_strength: float = 0.28
    rim_strength: float = 0.035
    edge_spill_strength: float = 0.00
    # Stage36: disable visible high-frequency source detail refill on skin.
    # Detail is now reserved for hair/edge and is extracted through gated, low-amplitude paths.
    detail_strength: float = 0.030
    detail_limit: float = 0.008
    local_albedo_keep: float = 0.995
    alpha_blur: int = 1
    alpha_tighten: float = 0.035
    alpha_edge_softness: float = 0.985
    subject_mask_expand: float = 0.0
    edge_band_power: float = 1.40
    edge_mix_strength: float = 0.035
    edge_local_spill_strength: float = 0.02
    edge_blur_passes: int = 1
    edge_cleanup_strength: float = 0.10
    edge_cleanup_blur_passes: int = 2
    core_color_match_strength: float = 0.008
    edge_color_match_strength: float = 0.015
    fill_edge_spec_strength: float = 0.06
    fill_hair_spec_strength: float = 0.70
    rim_edge_balance: float = 0.15
    rim_hair_balance: float = 0.85
    global_tint_strength: float = 0.03
    post_exposure: float = 1.16
    post_contrast: float = 1.05
    post_saturation: float = 1.06
    post_gamma: float = 1.00
    target_subject_p70: float = 0.34
    max_auto_gain: float = 2.0
    background_subject_scale: float = 1.0
    neon_dual_tint_strength: float = 0.0
    neon_dual_tint_center_falloff: float = 1.20
    neon_side_separation: float = 0.0
    background_respect: float = 0.88

    # New: body/geometry grounding controls.
    contact_shadow_enabled: bool = True
    contact_shadow_strength: float = 0.18
    contact_shadow_radius_px: int = 10
    contact_shadow_steps: int = 6
    contact_shadow_depth_bias: float = 0.004
    contact_shadow_depth_range: float = 0.035
    contact_shadow_blur_passes: int = 1
    contact_shadow_min_factor: float = 0.72

    ground_shadow_enabled: bool = True
    ground_shadow_strength: float = 0.26
    ground_shadow_softness: float = 0.58
    ground_shadow_width_scale: float = 0.46
    ground_shadow_height_scale: float = 0.14
    ground_shadow_y_offset_scale: float = 0.10
    ground_shadow_light_x_offset_scale: float = 0.22
    ground_shadow_blur_passes: int = 4
    ground_shadow_min_factor: float = 0.62
    ground_shadow_auto_disable: bool = True
    ground_shadow_min_bottom_ratio: float = 0.72
    ground_shadow_min_subject_height_ratio: float = 0.42

    # Finishing / style-separation controls.
    post_bloom_strength: float = 0.0
    post_bloom_radius: int = 2
    post_bloom_threshold: float = 0.72
    post_haze_strength: float = 0.0
    post_vignette_strength: float = 0.0
    post_local_contrast_strength: float = 0.0
    split_shadow_cool: float = 0.0
    split_highlight_warm: float = 0.0
    skin_protect_strength: float = 0.10


QUALITY_PROFILE = RelightPreset(
    source_preserve=0.010,
    source_shading_preserve=0.08,
    subject_mix=0.92,

    # Stage37: brighter portrait baseline. The background still decides the
    # direction and palette, but the face is no longer forced to follow a dark
    # cyber/night background into muddy midtones.
    ambient_strength=0.095,
    fill_strength=0.050,
    multi_ambient_strength=0.22,
    multi_ambient_wrap=0.38,

    # Keep visible 3D structure, but stop using heavy shadow sculpting as the
    # main source of depth. Excess shadow was the direct cause of the deep,
    # purple-looking face in dark backgrounds.
    shadow_sculpt_strength=0.14,
    key_shadow_strength=0.075,
    rim_strength=0.090,
    detail_strength=0.020,
    detail_limit=0.020,

    edge_local_spill_strength=0.016,
    edge_mix_strength=0.014,

    post_exposure=1.10,
    post_contrast=1.045,
    post_saturation=1.055,
    target_subject_p70=0.385,
    background_respect=0.92,

    contact_shadow_strength=0.12,
    ground_shadow_strength=0.20,

    post_bloom_strength=0.025,
    post_bloom_radius=2,
    post_bloom_threshold=0.78,
    post_haze_strength=0.010,
    post_vignette_strength=0.030,
    post_local_contrast_strength=0.022,

    split_shadow_cool=0.015,
    split_highlight_warm=0.020,
    skin_protect_strength=0.08,
)

CINEMATIC_PROFILE = RelightPreset(
    source_preserve=0.010,
    source_shading_preserve=0.09,
    subject_mix=0.90,
    ambient_strength=0.115,
    fill_strength=0.060,
    multi_ambient_strength=0.29,
    multi_ambient_wrap=0.42,
    shadow_sculpt_strength=0.15,
    key_shadow_strength=0.080,
    rim_strength=0.075,
    detail_strength=0.020,
    detail_limit=0.020,
    edge_local_spill_strength=0.016,
    edge_mix_strength=0.012,
    post_exposure=1.12,
    post_contrast=1.040,
    post_saturation=1.050,
    target_subject_p70=0.392,
    background_respect=0.91,
    contact_shadow_strength=0.13,
    ground_shadow_strength=0.20,
    post_bloom_strength=0.020,
    post_bloom_radius=2,
    post_bloom_threshold=0.80,
    post_haze_strength=0.008,
    post_vignette_strength=0.045,
    post_local_contrast_strength=0.020,
    split_shadow_cool=0.025,
    split_highlight_warm=0.020,
    skin_protect_strength=0.09,
)

NEON_PROFILE = RelightPreset(
    source_preserve=0.008,
    source_shading_preserve=0.09,
    subject_mix=0.89,
    ambient_strength=0.108,
    fill_strength=0.064,
    multi_ambient_strength=0.50,
    multi_ambient_wrap=0.50,
    multi_ambient_side_bias=0.95,
    multi_ambient_face_bias=0.08,
    shadow_sculpt_strength=0.16,
    key_shadow_strength=0.085,
    rim_strength=0.102,
    edge_local_spill_strength=0.018,
    edge_mix_strength=0.008,
    post_exposure=1.075,
    post_contrast=1.038,
    post_saturation=1.135,
    target_subject_p70=0.372,
    neon_dual_tint_strength=0.44,
    neon_dual_tint_center_falloff=1.26,
    neon_side_separation=0.34,
    background_respect=0.90,
    contact_shadow_strength=0.12,
    contact_shadow_radius_px=8,
    ground_shadow_strength=0.20,
    ground_shadow_softness=0.66,
    post_bloom_strength=0.055,
    post_bloom_radius=2,
    post_bloom_threshold=0.74,
    post_haze_strength=0.008,
    post_vignette_strength=0.035,
    post_local_contrast_strength=0.020,
    split_shadow_cool=0.040,
    split_highlight_warm=0.012,
    skin_protect_strength=0.090,
)

PROFILE_LIBRARY = {
    'quality': QUALITY_PROFILE,
    'cinematic': CINEMATIC_PROFILE,
    'neon': NEON_PROFILE,
    # backward-compat aliases
    'default': QUALITY_PROFILE,
    'sunset': QUALITY_PROFILE,
    'moonlight': CINEMATIC_PROFILE,
    'lowkey': CINEMATIC_PROFILE,
    'highkey': QUALITY_PROFILE,
}


def load_quality_profile(style_mode: str = 'quality') -> RelightPreset:
    base = PROFILE_LIBRARY.get(str(style_mode or 'quality').lower(), QUALITY_PROFILE)
    return RelightPreset(**asdict(base))


def write_builtin_profile_files(output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, preset in PROFILE_LIBRARY.items():
        # only write canonical profiles, skip aliases
        if name not in ('quality', 'cinematic', 'neon'):
            continue
        with open(path / f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(asdict(preset), f, ensure_ascii=False, indent=2)
    print(f'Wrote built-in profile JSON files to: {path}')
