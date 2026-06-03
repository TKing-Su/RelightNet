from __future__ import annotations

from typing import List, Optional
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

def D_GGX(NdotH: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    a = np.maximum(roughness * roughness, 1e-3)
    a2 = a * a
    denom = NdotH * NdotH * (a2 - 1.0) + 1.0
    return (a2 / np.maximum(PI * denom * denom, 1e-6)).astype(np.float32)


def G_SchlickGGX(NdotX: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    r = roughness + 1.0
    k = (r * r) / 8.0
    return (NdotX / np.maximum(NdotX * (1.0 - k) + k, 1e-6)).astype(np.float32)


def fresnel_schlick(cos_theta: np.ndarray, F0: np.ndarray) -> np.ndarray:
    one_minus = np.power(np.clip(1.0 - cos_theta[..., None], 0.0, 1.0), 5.0).astype(np.float32)
    return (F0 + (1.0 - F0) * one_minus).astype(np.float32)


# ---------------------------------------------------------------------------
# Continuous atmosphere budget system (default look-safe flow)
# ---------------------------------------------------------------------------

def _smooth01(x: float, lo: float, hi: float) -> float:
    t = float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _lerp_f(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clip01_f(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def compute_style_expression(descriptor: dict) -> dict:
    """Convert background statistics into continuous look dimensions.

    This intentionally avoids filename hints and discrete style labels.  The
    values are wide-range controls used by LookPolicy sections below, not a
    classifier.
    """
    d = descriptor or {}
    sm = _smooth01
    luma_std = float(d.get('luma_std', 0.12))
    chroma_mean = float(d.get('chroma_mean', 0.10))
    chroma_peak = float(d.get('chroma_max_region', chroma_mean))
    warm_ratio = float(d.get('warm_ratio', 0.0))
    cool_ratio = float(d.get('cool_ratio', 0.0))
    ambient_warmth = float(d.get('ambient_warmth', 0.0))
    palette_diversity = float(d.get('palette_diversity', 0.0))
    hue_entropy = float(d.get('hue_entropy', 0.0))
    gradient_confidence = float(d.get('gradient_confidence', 0.0))
    gradient_magnitude = float(d.get('gradient_magnitude', 0.0))
    gradient_local_contrast = float(d.get('gradient_local_contrast', luma_std))
    gradient_colorfulness = float(d.get('gradient_colorfulness', chroma_mean))
    source_peak_count = float(d.get('source_peak_count', 0.0))
    source_peak_strength = float(d.get('source_peak_strength', 0.0))
    source_peak_chroma = float(d.get('source_peak_chroma', chroma_peak))
    lr = abs(float(d.get('left_right_luma_diff', 0.0)))
    tb = abs(float(d.get('top_bottom_luma_diff', 0.0)))

    high_chroma = sm(max(chroma_mean, chroma_peak, 0.65 * gradient_colorfulness), 0.08, 0.38)
    peak_chroma_gate = sm(source_peak_chroma, 0.18, 0.52)
    peak_activity = _clip01_f(
        0.46 * sm(source_peak_strength, 0.04, 0.45) * peak_chroma_gate
        + 0.36 * sm(source_peak_chroma, 0.08, 0.42)
        + 0.15 * sm(source_peak_count / 5.0, 0.0, 1.0)
    )
    directionality = _clip01_f(
        0.38 * sm(gradient_confidence, 0.10, 0.75)
        + 0.32 * sm(lr + tb, 0.025, 0.24)
        + 0.30 * sm(max(gradient_magnitude, gradient_local_contrast), 0.04, 0.32)
    )
    warmth = _clip01_f(
        0.58 * warm_ratio
        + 0.22 * max(ambient_warmth, 0.0)
        + 0.20 * high_chroma * sm(warm_ratio, 0.08, 0.42)
    )
    coolness = _clip01_f(
        0.60 * cool_ratio
        + 0.22 * max(-ambient_warmth, 0.0)
        + 0.18 * high_chroma * sm(cool_ratio, 0.08, 0.42)
    )
    color_split = _clip01_f(
        0.44 * min(warmth, coolness) * 2.0
        + 0.28 * hue_entropy
        + 0.20 * palette_diversity
        + 0.08 * sm(abs(warm_ratio - cool_ratio), 0.05, 0.55)
    )
    neon_raw = _clip01_f(
        0.34 * high_chroma
        + 0.24 * color_split
        + 0.20 * peak_activity
        + 0.14 * sm(palette_diversity, 0.28, 0.72)
        + 0.08 * sm(float(d.get('luma_p90', 0.45)), 0.35, 0.85)
    )
    neon_identity = _clip01_f(
        0.18
        + 0.45 * coolness
        + 0.26 * color_split
        + 0.22 * peak_chroma_gate
        - 0.55 * max(warmth - coolness, 0.0)
    )
    neon = _clip01_f(neon_raw * neon_identity)
    mist = _clip01_f(
        0.30 * float(d.get('misty_score', 0.0))
        + 0.24 * (1.0 - sm(source_peak_chroma, 0.16, 0.42))
        + 0.22 * (1.0 - sm(chroma_peak, 0.18, 0.46))
        + 0.16 * sm(float(d.get('luma_mean', 0.35)), 0.30, 0.58)
        + 0.16 * (1.0 - sm(gradient_magnitude, 0.06, 0.18))
        - 0.12 * sm(peak_activity, 0.45, 0.95)
    )
    lowkey = _clip01_f(
        0.70 * float(d.get('lowkey_score', 0.0))
        + 0.18 * sm(1.0 - float(d.get('luma_mean', 0.35)), 0.55, 0.86)
        + 0.12 * peak_activity
    )
    highkey = _clip01_f(float(d.get('highkey_score', 0.0)))
    locality = _clip01_f(
        0.46 * peak_activity
        + 0.30 * sm(source_peak_count / 6.0, 0.0, 1.0)
        + 0.24 * sm(gradient_local_contrast, 0.18, 0.90)
    )
    contrast = _clip01_f(
        0.62 * sm(luma_std, 0.055, 0.28)
        + 0.24 * sm(gradient_local_contrast, 0.18, 0.85)
        + 0.14 * sm(float(d.get('luma_p90', 0.45)) - float(d.get('luma_p10', 0.10)), 0.18, 0.70)
    )
    atmosphere = _clip01_f(
        0.20 * warmth
        + 0.18 * coolness
        + 0.22 * neon
        + 0.18 * mist
        + 0.14 * palette_diversity
        + 0.08 * directionality
    )
    return {
        'warmth': warmth,
        'coolness': coolness,
        'neon': neon,
        'mist': mist,
        'lowkey': lowkey,
        'highkey': highkey,
        'color_split': color_split,
        'directionality': directionality,
        'locality': locality,
        'atmosphere': atmosphere,
        'contrast': contrast,
        'palette_diversity': _clip01_f(palette_diversity),
    }


def compute_background_descriptor(
    bg_linear: np.ndarray,
    lighting_info: 'LightingInfo',
    source_linear: Optional[np.ndarray] = None,
) -> dict:
    bg = np.clip(bg_linear.astype(np.float32), 0.0, None)
    luma = np.dot(bg, LUMA)

    luma_mean = float(np.mean(luma))
    luma_std = float(np.std(luma))
    luma_p10 = float(np.percentile(luma, 10))
    luma_p50 = float(np.percentile(luma, 50))
    luma_p90 = float(np.percentile(luma, 90))
    luma_p98 = float(np.percentile(luma, 98))
    dynamic_range = float(np.clip(luma_p98 - luma_p10, 0.0, 1.0))
    highlight_area = float(np.mean(luma >= max(luma_p90, luma_p50 + 0.45 * dynamic_range)))
    shadow_area = float(np.mean(luma <= min(luma_p10 + 0.16 * dynamic_range, luma_p50 * 0.72)))

    r, g, b_ch = bg[..., 0], bg[..., 1], bg[..., 2]
    cb = b_ch - luma
    cr = r - luma
    chroma_mag = np.sqrt(cb * cb + cr * cr)
    chroma_mean = float(np.mean(chroma_mag))
    top_chroma_mask = chroma_mag >= np.percentile(chroma_mag, 90)
    chroma_max_region = float(np.mean(chroma_mag[top_chroma_mask])) if np.any(top_chroma_mask) else chroma_mean

    bg_srgb = linear_to_srgb(np.clip(bg, 0.0, 1.0))
    r_s, g_s, b_s = bg_srgb[..., 0], bg_srgb[..., 1], bg_srgb[..., 2]
    cmax = np.maximum(np.maximum(r_s, g_s), b_s)
    cmin = np.minimum(np.minimum(r_s, g_s), b_s)
    delta = cmax - cmin + 1e-7
    sat = np.where(cmax > 1e-5, delta / (cmax + 1e-7), 0.0)
    hue = np.zeros_like(cmax)
    mask_r = (cmax == r_s) & (delta > 1e-5)
    mask_g = (cmax == g_s) & (delta > 1e-5) & ~mask_r
    mask_b = ~mask_r & ~mask_g & (delta > 1e-5)
    hue[mask_r] = (60.0 * ((g_s[mask_r] - b_s[mask_r]) / delta[mask_r]) + 360.0) % 360.0
    hue[mask_g] = (60.0 * ((b_s[mask_g] - r_s[mask_g]) / delta[mask_g]) + 120.0) % 360.0
    hue[mask_b] = (60.0 * ((r_s[mask_b] - g_s[mask_b]) / delta[mask_b]) + 240.0) % 360.0

    sat_mask = sat > 0.08
    if np.count_nonzero(sat_mask) > 20:
        hue_vals = hue[sat_mask]
        hist, _ = np.histogram(hue_vals, bins=12, range=(0.0, 360.0))
        hist_f = hist.astype(np.float32) / (hist.sum() + 1e-9)
        hue_entropy = float(-np.sum(hist_f * np.log(hist_f + 1e-9)) / np.log(12.0))
        dominant_hue_strength = float(np.max(hist_f))
    else:
        hue_entropy = 0.0
        dominant_hue_strength = 1.0

    warm_mask = ((hue < 60.0) | (hue > 300.0)) & sat_mask
    cool_mask = ((hue > 150.0) & (hue < 270.0)) & sat_mask
    green_mask = ((hue >= 75.0) & (hue <= 155.0)) & sat_mask
    magenta_mask = ((hue >= 275.0) & (hue <= 330.0)) & sat_mask
    n_sat = max(float(np.count_nonzero(sat_mask)), 1.0)
    warm_ratio = float(np.count_nonzero(warm_mask)) / n_sat
    cool_ratio = float(np.count_nonzero(cool_mask)) / n_sat
    green_ratio = float(np.count_nonzero(green_mask)) / n_sat
    magenta_ratio = float(np.count_nonzero(magenta_mask)) / n_sat
    average_saturation = float(np.mean(sat))
    colorfulness = float(np.clip(0.58 * average_saturation + 0.42 * chroma_mean, 0.0, 1.0))
    mean_color = np.mean(bg.reshape(-1, 3), axis=0).astype(np.float32)
    if np.any(sat_mask):
        color_w = np.clip(sat, 0.0, 1.0) * np.clip(luma / max(luma_p98, 1e-6), 0.0, 1.5)
        dominant_color = ((bg * color_w[..., None]).sum(axis=(0, 1)) / max(float(color_w.sum()), 1e-6)).astype(np.float32)
    else:
        dominant_color = mean_color.astype(np.float32)

    gy, gx = np.gradient(luma)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    gradient_magnitude = float(np.clip(np.mean(grad_mag) / (luma_mean + 1e-5), 0.0, 1.0))
    mean_gx_abs = float(np.mean(np.abs(gx)))
    mean_gy_abs = float(np.mean(np.abs(gy)))
    gradient_anisotropy = mean_gx_abs / (mean_gx_abs + mean_gy_abs + 1e-9)

    h4, w4 = luma.shape[0] // 4, luma.shape[1] // 4
    if h4 > 0 and w4 > 0:
        blocks = luma[:h4 * 4, :w4 * 4].reshape(4, h4, 4, w4).mean(axis=(1, 3))
        spatial_luma_variance = float(np.clip(np.var(blocks), 0.0, 1.0))
    else:
        spatial_luma_variance = 0.0

    h3 = luma.shape[0] // 3
    if h3 > 0:
        # Direction convention: positive means bottom is brighter than top.
        top_luma = float(np.mean(luma[:h3]))
        bottom_luma = float(np.mean(luma[-h3:]))
        top_bottom_luma_diff = bottom_luma - top_luma
    else:
        top_luma = luma_mean
        bottom_luma = luma_mean
        top_bottom_luma_diff = 0.0

    w2 = luma.shape[1] // 2
    if w2 > 0:
        # Direction convention: positive means right is brighter than left.
        left_luma = float(np.mean(luma[:, :w2]))
        right_luma = float(np.mean(luma[:, w2:]))
        left_right_luma_diff = right_luma - left_luma
    else:
        left_luma = luma_mean
        right_luma = luma_mean
        left_right_luma_diff = 0.0

    palette_diversity = float(getattr(lighting_info, 'palette_diversity', 0.3))
    key_int = float(getattr(lighting_info, 'key_intensity', 1.0))
    amb_c = np.array(getattr(lighting_info, 'ambient_color', (0.5, 0.5, 0.5)), dtype=np.float32)
    fill_int = float(getattr(lighting_info, 'ambient_intensity', 0.5))
    key_fill_ratio = key_int / (fill_int + 1e-5)
    ambient_warmth = float(np.clip(amb_c[0] - amb_c[2], -1.0, 1.0))
    _gf = getattr(lighting_info, 'gradient_field', None)
    gradient_confidence = float((_gf or {}).get('confidence', 0.0)) if isinstance(_gf, dict) else 0.0
    gradient_local_contrast = float((_gf or {}).get('local_contrast', 0.0)) if isinstance(_gf, dict) else 0.0
    gradient_colorfulness = float((_gf or {}).get('colorfulness', 0.0)) if isinstance(_gf, dict) else 0.0
    brightest_region_uv = [0.5, 0.5]
    if isinstance(_gf, dict) and isinstance(_gf.get('key_uv'), (list, tuple)) and len(_gf.get('key_uv')) >= 2:
        brightest_region_uv = [float(_gf.get('key_uv')[0]), float(_gf.get('key_uv')[1])]
    else:
        h_l, w_l = luma.shape[:2]
        yy_l, xx_l = np.mgrid[0:h_l, 0:w_l].astype(np.float32)
        bright_w = np.clip((luma - luma_p90) / max(luma_p98 - luma_p90, 1e-6), 0.0, 1.0)
        if float(bright_w.sum()) > 1e-6:
            brightest_region_uv = [
                float((xx_l * bright_w).sum() / max(float(bright_w.sum()) * max(w_l - 1, 1), 1e-6)),
                float((yy_l * bright_w).sum() / max(float(bright_w.sum()) * max(h_l - 1, 1), 1e-6)),
            ]
    source_peak_count = 0.0
    source_peak_strength = 0.0
    source_peak_chroma = 0.0
    source_peak_luma = 0.0
    local_light_peaks = []
    if isinstance(_gf, dict):
        peaks = _gf.get('source_peaks', []) or []
        if isinstance(peaks, list) and peaks:
            source_peak_count = float(len(peaks))
            scores = []
            chromas = []
            lumas = []
            for p in peaks:
                if not isinstance(p, dict):
                    continue
                scores.append(float(p.get('score', p.get('power', 0.0))))
                chromas.append(float(p.get('sat', 0.0)))
                lumas.append(float(p.get('luma', 0.0)))
                if len(local_light_peaks) < 8:
                    local_light_peaks.append({
                        'u': float(np.clip(p.get('u', 0.5), 0.0, 1.0)),
                        'v': float(np.clip(p.get('v', 0.5), 0.0, 1.0)),
                        'score': float(np.clip(p.get('score', p.get('power', 0.0)), 0.0, 4.0)),
                        'sat': float(np.clip(p.get('sat', 0.0), 0.0, 1.0)),
                        'luma': float(np.clip(p.get('luma', 0.0), 0.0, 1.0)),
                        'color': [float(x) for x in np.clip(np.array(p.get('color', mean_color), dtype=np.float32), 0.0, 4.0).reshape(-1)[:3]],
                    })
            if scores:
                score_arr = np.asarray(scores, dtype=np.float32)
                chroma_arr = np.asarray(chromas, dtype=np.float32) if chromas else np.zeros_like(score_arr)
                luma_arr = np.asarray(lumas, dtype=np.float32) if lumas else np.zeros_like(score_arr)
                source_peak_strength = float(np.clip(np.percentile(score_arr, 75.0), 0.0, 1.0))
                source_peak_chroma = float(np.clip(np.mean(chroma_arr), 0.0, 1.0))
                source_peak_luma = float(np.clip(np.mean(luma_arr), 0.0, 1.0))

    if source_linear is not None:
        src_luma = np.dot(np.clip(source_linear.astype(np.float32), 0.0, None), LUMA)
        bg_subject_luma_gap = abs(luma_mean - float(np.median(src_luma)))
    else:
        bg_subject_luma_gap = 0.0

    _lm = float(np.clip(luma_mean, 0.0, 1.0))
    _ls = float(np.clip(luma_std, 0.0, 1.0))
    _cm = float(np.clip(chroma_mean, 0.0, 1.0))
    highkey_score = _smooth01(_lm, 0.35, 0.70) * (1.0 - _smooth01(_ls, 0.12, 0.30))
    lowkey_score = _smooth01(1.0 - _lm, 0.50, 0.80) * _smooth01(_cm, 0.10, 0.35)
    misty_score = _smooth01(_lm, 0.30, 0.55) * (1.0 - _smooth01(_cm, 0.08, 0.25)) * (1.0 - _smooth01(_ls, 0.10, 0.25))
    gradient_strength = float(np.clip(max(gradient_magnitude, gradient_confidence * max(abs(left_right_luma_diff), abs(top_bottom_luma_diff)) * 3.0), 0.0, 1.0))
    local_light_confidence = float(np.clip(0.42 * gradient_confidence + 0.34 * _smooth01(source_peak_strength, 0.03, 0.45) + 0.24 * _smooth01(source_peak_chroma, 0.10, 0.45), 0.0, 1.0))
    flatness_score = float(np.clip((1.0 - _smooth01(_ls, 0.045, 0.18)) * (1.0 - _smooth01(gradient_magnitude, 0.035, 0.16)), 0.0, 1.0))
    texture_strength = float(np.clip(_smooth01(gradient_magnitude + _ls, 0.06, 0.34), 0.0, 1.0))
    haze_score = float(np.clip(0.55 * misty_score + 0.30 * flatness_score + 0.15 * (1.0 - _smooth01(average_saturation, 0.10, 0.36)), 0.0, 1.0))
    hard_light_score = float(np.clip(0.42 * local_light_confidence + 0.32 * _smooth01(dynamic_range, 0.22, 0.70) + 0.26 * _smooth01(highlight_area, 0.015, 0.16), 0.0, 1.0))
    soft_light_score = float(np.clip(0.55 * haze_score + 0.28 * (1.0 - hard_light_score) + 0.17 * _smooth01(luma_p50, 0.18, 0.56), 0.0, 1.0))
    edge_light_potential = float(np.clip(0.34 * local_light_confidence + 0.26 * colorfulness + 0.22 * dynamic_range + 0.18 * palette_diversity, 0.0, 1.0))
    multicolor_complexity = float(np.clip(0.45 * palette_diversity + 0.35 * hue_entropy + 0.20 * colorfulness, 0.0, 1.0))
    subject_background_contrast = float(np.clip(bg_subject_luma_gap, 0.0, 1.0))
    expected_rim_need = float(np.clip(0.42 * lowkey_score + 0.28 * edge_light_potential + 0.18 * subject_background_contrast + 0.12 * colorfulness, 0.0, 1.0))
    expected_shadow_strength = float(np.clip(0.38 * lowkey_score + 0.34 * hard_light_score + 0.20 * dynamic_range - 0.22 * haze_score, 0.0, 1.0))
    expected_color_spill_strength = float(np.clip(0.48 * colorfulness + 0.30 * palette_diversity + 0.22 * local_light_confidence, 0.0, 1.0))
    face_protection_need = float(np.clip(0.40 * expected_color_spill_strength + 0.26 * dominant_hue_strength * colorfulness + 0.20 * highkey_score + 0.14 * subject_background_contrast, 0.0, 1.0))
    gradient_axis = 'horizontal' if abs(left_right_luma_diff) >= abs(top_bottom_luma_diff) else 'vertical'

    return dict(
        global_luma=_lm,
        luma_mean=_lm,
        luma_std=_ls,
        luma_p10=float(np.clip(luma_p10, 0.0, 1.0)),
        luma_p50=float(np.clip(luma_p50, 0.0, 1.0)),
        luma_p90=float(np.clip(luma_p90, 0.0, 1.0)),
        luma_p98=float(np.clip(luma_p98, 0.0, 1.0)),
        p10_luma=float(np.clip(luma_p10, 0.0, 1.0)),
        p50_luma=float(np.clip(luma_p50, 0.0, 1.0)),
        p90_luma=float(np.clip(luma_p90, 0.0, 1.0)),
        p98_luma=float(np.clip(luma_p98, 0.0, 1.0)),
        dynamic_range=dynamic_range,
        highlight_area=float(np.clip(highlight_area, 0.0, 1.0)),
        shadow_area=float(np.clip(shadow_area, 0.0, 1.0)),
        chroma_mean=_cm,
        chroma_max_region=float(np.clip(chroma_max_region, 0.0, 1.0)),
        mean_color=[float(x) for x in np.clip(mean_color, 0.0, 4.0)],
        ambient_color=[float(x) for x in np.clip(amb_c, 0.0, 4.0)],
        dominant_color=[float(x) for x in np.clip(dominant_color, 0.0, 4.0)],
        hue_entropy=float(np.clip(hue_entropy, 0.0, 1.0)),
        warm_ratio=float(np.clip(warm_ratio, 0.0, 1.0)),
        cool_ratio=float(np.clip(cool_ratio, 0.0, 1.0)),
        warm_presence=float(np.clip(warm_ratio, 0.0, 1.0)),
        cool_presence=float(np.clip(cool_ratio, 0.0, 1.0)),
        green_presence=float(np.clip(green_ratio, 0.0, 1.0)),
        magenta_presence=float(np.clip(magenta_ratio, 0.0, 1.0)),
        average_saturation=float(np.clip(average_saturation, 0.0, 1.0)),
        colorfulness=float(np.clip(colorfulness, 0.0, 1.0)),
        gradient_magnitude=float(np.clip(gradient_magnitude, 0.0, 1.0)),
        gradient_anisotropy=float(np.clip(gradient_anisotropy, 0.0, 1.0)),
        gradient_strength=gradient_strength,
        spatial_luma_variance=float(np.clip(spatial_luma_variance, 0.0, 1.0)),
        left_luma=float(np.clip(left_luma, 0.0, 1.0)),
        right_luma=float(np.clip(right_luma, 0.0, 1.0)),
        top_luma=float(np.clip(top_luma, 0.0, 1.0)),
        bottom_luma=float(np.clip(bottom_luma, 0.0, 1.0)),
        horizontal_bias=float(np.clip(left_right_luma_diff, -1.0, 1.0)),
        vertical_bias=float(np.clip(top_bottom_luma_diff, -1.0, 1.0)),
        top_bottom_luma_diff=float(np.clip(top_bottom_luma_diff, -1.0, 1.0)),
        left_right_luma_diff=float(np.clip(left_right_luma_diff, -1.0, 1.0)),
        gradient_axis=gradient_axis,
        brightest_region_uv=[float(np.clip(brightest_region_uv[0], 0.0, 1.0)), float(np.clip(brightest_region_uv[1], 0.0, 1.0))],
        local_light_peaks=local_light_peaks,
        local_light_confidence=local_light_confidence,
        direction_convention={
            'horizontal': 'right - left',
            'vertical': 'bottom - top',
        },
        dominant_hue_strength=float(np.clip(dominant_hue_strength, 0.0, 1.0)),
        palette_diversity=float(np.clip(palette_diversity, 0.0, 1.0)),
        key_fill_ratio=float(np.clip(key_fill_ratio, 0.0, 20.0)),
        ambient_warmth=float(np.clip(ambient_warmth, -1.0, 1.0)),
        bg_subject_luma_gap=float(np.clip(bg_subject_luma_gap, 0.0, 1.0)),
        gradient_confidence=float(np.clip(gradient_confidence, 0.0, 1.0)),
        gradient_local_contrast=float(np.clip(gradient_local_contrast, 0.0, 1.0)),
        gradient_colorfulness=float(np.clip(gradient_colorfulness, 0.0, 1.0)),
        flatness_score=flatness_score,
        texture_strength=texture_strength,
        haze_score=haze_score,
        hard_light_score=hard_light_score,
        soft_light_score=soft_light_score,
        edge_light_potential=edge_light_potential,
        multicolor_complexity=multicolor_complexity,
        subject_background_contrast=subject_background_contrast,
        expected_rim_need=expected_rim_need,
        expected_shadow_strength=expected_shadow_strength,
        expected_color_spill_strength=expected_color_spill_strength,
        face_protection_need=face_protection_need,
        source_peak_count=float(np.clip(source_peak_count, 0.0, 32.0)),
        source_peak_strength=float(np.clip(source_peak_strength, 0.0, 1.0)),
        source_peak_chroma=float(np.clip(source_peak_chroma, 0.0, 1.0)),
        source_peak_luma=float(np.clip(source_peak_luma, 0.0, 1.0)),
        highkey_score=float(np.clip(highkey_score, 0.0, 1.0)),
        lowkey_score=float(np.clip(lowkey_score, 0.0, 1.0)),
        misty_score=float(np.clip(misty_score, 0.0, 1.0)),
    )


def policy_direction_from_uv(key_uv: List[float]) -> List[float]:
    """Approximate portrait-space light direction from a background UV point."""
    try:
        u = float(key_uv[0])
        v = float(key_uv[1])
    except Exception:
        u, v = 0.5, 0.28
    x = (u - 0.5) * 1.85
    y = (0.50 - v) * 1.28
    z = 0.72 + 0.40 * (1.0 - min(abs(u - 0.5) * 1.7, 1.0))
    d = safe_norm(np.array([x, y, z], dtype=np.float32))
    return [float(d[0]), float(d[1]), float(d[2])]


def background_descriptor_debug_view(descriptor: dict) -> dict:
    """Compact stable descriptor view for Debug/look_safe_budget.json."""
    d = descriptor or {}
    keys = [
        'global_luma', 'dynamic_range', 'highlight_area', 'shadow_area',
        'average_saturation', 'colorfulness', 'hue_entropy',
        'palette_diversity', 'warm_presence', 'cool_presence',
        'green_presence', 'magenta_presence', 'horizontal_bias',
        'vertical_bias', 'gradient_strength', 'local_light_confidence',
        'flatness_score', 'haze_score', 'edge_light_potential',
        'hard_light_score', 'soft_light_score', 'multicolor_complexity',
        'subject_background_contrast', 'expected_rim_need',
        'expected_shadow_strength', 'expected_color_spill_strength',
        'face_protection_need', 'p50_luma', 'p95_luma', 'chroma_mean',
        'chroma_max_region', 'gradient_confidence', 'gradient_local_contrast',
        'source_peak_strength', 'source_peak_chroma',
    ]
    out = {k: float(d.get(k, 0.0)) for k in keys}
    out['brightest_region_uv'] = list(d.get('brightest_region_uv', [0.5, 0.5]))
    out['gradient_axis'] = str(d.get('gradient_axis', 'horizontal'))
    out['local_light_peaks'] = d.get('local_light_peaks', [])
    out['mean_color'] = d.get('mean_color', [0.0, 0.0, 0.0])
    out['ambient_color'] = d.get('ambient_color', [0.0, 0.0, 0.0])
    out['dominant_color'] = d.get('dominant_color', [0.0, 0.0, 0.0])
    return out


def compute_atmosphere_budget(descriptor: dict) -> dict:
    d = descriptor
    sm = _smooth01

    cast_danger = d['chroma_mean'] * d['dominant_hue_strength']
    warm_cast = d['warm_ratio'] * d['chroma_mean']
    hk = d.get('highkey_score', 0.0)
    lk = d.get('lowkey_score', 0.0)
    _misty = d.get('misty_score', 0.0)

    face_luma_lift_gate = _lerp_f(0.20, 1.0,
        sm(d['luma_std'], 0.05, 0.25) * (1.0 - sm(d['bg_subject_luma_gap'], 0.0, 0.3)))

    face_chroma_inject = _lerp_f(0.0, 1.0,
        1.0 - sm(cast_danger, 0.15, 0.5))
    face_chroma_inject *= (1.0 - 0.4 * hk)

    face_side_chroma_inject = float(np.clip(face_chroma_inject * 1.3, 0.0, 1.0))

    hair_chroma_boost = _lerp_f(1.0, 1.5,
        sm(d['hue_entropy'], 0.3, 0.8))

    edge_chroma_boost = _lerp_f(0.9, 1.4,
        sm(d['hue_entropy'] * d['gradient_magnitude'], 0.1, 0.4))

    multicolor_face_gate_floor = _lerp_f(0.24, 1.0,
        1.0 - sm(max(d['chroma_mean'], d['warm_ratio']), 0.2, 0.6))

    autogain_target_low_ceiling = _lerp_f(0.38, 0.30, hk)
    autogain_target_low = _lerp_f(0.320, autogain_target_low_ceiling,
        sm(d['luma_mean'], 0.2, 0.6))

    autogain_target_high = autogain_target_low + _lerp_f(0.06, 0.02, hk)

    autogain_upper_floor = _lerp_f(1.35, 1.08, hk)
    autogain_upper_floor = max(autogain_upper_floor, _lerp_f(1.08, 1.35, lk))
    autogain_upper = _lerp_f(autogain_upper_floor, 1.50,
        sm(d['luma_std'], 0.08, 0.20))

    global_tint_scale = _lerp_f(0.30, 1.0,
        1.0 - sm(cast_danger, 0.12, 0.45))
    global_tint_scale *= _lerp_f(1.0, 0.55, sm(d['chroma_mean'], 0.15, 0.40))

    exposure_scale = _lerp_f(0.86, 1.0, 1.0 - hk * 0.8)
    exposure_scale = max(exposure_scale, _lerp_f(0.94, 1.0, _misty))
    display_saturation_multiplier = _lerp_f(0.94, 1.06, sm(d['hue_entropy'], 0.20, 0.75))
    display_saturation_multiplier *= _lerp_f(1.0, 0.94, hk * sm(d['chroma_mean'], 0.15, 0.45))

    v32_face_bg_budget = _lerp_f(0.000, 0.015,
        1.0 - sm(max(d['chroma_mean'], d['warm_ratio']), 0.25, 0.55))
    v32_face_bg_budget *= (1.0 - 0.5 * hk)

    v32_face_side_budget = float(np.clip(v32_face_bg_budget * 1.5, 0.0, 0.015))

    v32_shell_scale = _lerp_f(1.35, 1.0,
        sm(d['hue_entropy'], 0.2, 0.7))

    v32_rim_scale = _lerp_f(1.25, 1.0,
        sm(d['key_fill_ratio'], 1.5, 4.0))

    v32_luma_block_scale = _lerp_f(0.30, 1.0,
        sm(d['bg_subject_luma_gap'], 0.0, 0.20))

    v32_warm_gate_cap = _lerp_f(0.55, 1.0,
        1.0 - sm(warm_cast, 0.10, 0.40))

    highlight_soft_knee = _lerp_f(0.0, 0.70,
        hk * (0.6 + 0.4 * warm_cast))

    bloom_multiplier = _lerp_f(1.0, 0.15,
        hk * (0.5 + 0.5 * d['warm_ratio']))

    haze_multiplier = _lerp_f(1.0, 0.12,
        hk * warm_cast)

    colorfulness = d['chroma_mean']
    gc = d.get('gradient_confidence', 0.0)
    hk_safe = hk * (1.0 - 0.7 * lk)

    v32_face_gate_cap = _lerp_f(0.60, 0.32, hk_safe)
    v32_body_gate_cap = _lerp_f(0.55, 0.28, hk_safe)
    v32_luma_lift_multiplier = _lerp_f(1.0, 0.18,
        hk * (0.5 + 0.5 * colorfulness) * (1.0 - 0.6 * _misty))
    pbr_preserve_strength = _lerp_f(0.0, 0.45, sm(gc, 0.2, 0.7))
    subject_highlight_knee_start = _lerp_f(0.68, 0.55, hk)
    subject_highlight_knee_strength = _lerp_f(0.0, 0.82,
        hk * (0.55 + 0.45 * warm_cast) * (1.0 - 0.5 * _misty))

    # V3 Single Luma Authority fields
    luma_authority_strength = float(np.clip(
        sm(hk, 0.3, 0.7) * 0.8 + sm(warm_cast, 0.25, 0.6) * 0.5, 0.0, 1.0))
    air_guard_luma_lift_cap = _lerp_f(0.12, 0.02, luma_authority_strength)
    v32_positive_luma_gate = _lerp_f(1.0, 0.05, luma_authority_strength)
    autogain_face_weight = _lerp_f(1.0, 0.30, luma_authority_strength)
    autogain_body_weight = _lerp_f(1.0, 0.50, luma_authority_strength)
    autogain_edge_weight = _lerp_f(1.0, 0.85, luma_authority_strength)
    max_positive_face_lift = _lerp_f(0.25, 0.06, luma_authority_strength)
    max_positive_body_lift = _lerp_f(0.20, 0.08, luma_authority_strength)
    highkey_overbright_knee = _lerp_f(0.85, 0.65, hk)

    # V4 Color / Atmosphere Authority fields
    skin_danger = sm(cast_danger, 0.12, 0.50) * sm(warm_cast, 0.08, 0.35)
    face_core_chroma_authority = _lerp_f(0.0, 0.95, skin_danger)
    face_side_chroma_authority = _lerp_f(0.0, 0.72, skin_danger)
    body_skin_chroma_authority = _lerp_f(0.0, 0.58, skin_danger)
    clothing_chroma_authority = _lerp_f(0.0, 0.30, skin_danger)
    hair_chroma_authority = _lerp_f(1.0, 1.45,
        sm(d['hue_entropy'], 0.3, 0.8) * (1.0 - 0.3 * skin_danger))
    edge_shell_chroma_authority = _lerp_f(1.0, 1.55,
        sm(d['hue_entropy'] * d['gradient_magnitude'], 0.1, 0.4))
    warm_skin_contamination_guard = _lerp_f(0.0, 0.90,
        sm(warm_cast, 0.10, 0.40) * sm(d['dominant_hue_strength'], 0.25, 0.60))
    skin_yellow_wash_guard = _lerp_f(0.0, 0.85,
        sm(warm_cast, 0.08, 0.30) * sm(d['chroma_mean'], 0.10, 0.35))
    subject_global_tint_guard = _lerp_f(0.0, 0.80,
        sm(cast_danger, 0.10, 0.40))
    atmosphere_carrier_strength = _lerp_f(1.0, 1.50,
        sm(d['hue_entropy'], 0.25, 0.75) * (1.0 - 0.4 * skin_danger))

    # V5 Directional Atmosphere Reinjection fields
    _gc = d.get('gradient_confidence', 0.0)
    _lr_diff = abs(d.get('left_right_luma_diff', 0.0))
    _slv = d.get('spatial_luma_variance', 0.0)
    _pd = d.get('palette_diversity', 0.3)
    _cmr = d.get('chroma_max_region', 0.0)
    _gm = d.get('gradient_magnitude', 0.0)

    directional_light_strength = _lerp_f(0.15, 0.65,
        sm(_gc, 0.15, 0.70) * sm(_lr_diff, 0.03, 0.20))
    directional_shadow_strength = _lerp_f(0.08, 0.45,
        sm(d['key_fill_ratio'], 1.5, 5.0) * sm(d['luma_std'], 0.06, 0.22))
    directional_contrast_strength = _lerp_f(0.10, 0.50,
        sm(_slv, 0.01, 0.08) * (1.0 - _misty))
    face_core_directional_budget = _lerp_f(0.02, 0.08,
        sm(_gc, 0.3, 0.8)) * (1.0 - 0.6 * hk)
    face_side_directional_budget = _lerp_f(0.06, 0.25, directional_light_strength)
    body_directional_budget = _lerp_f(0.08, 0.30,
        directional_light_strength * (1.0 - 0.4 * hk))
    clothing_directional_budget = _lerp_f(0.10, 0.40, directional_light_strength)
    hair_rim_budget = _lerp_f(0.12, 0.55,
        sm(d['chroma_mean'], 0.08, 0.35) * sm(_pd, 0.2, 0.6) + 0.3 * lk)
    edge_rim_budget = _lerp_f(0.15, 0.60,
        sm(d['chroma_mean'], 0.10, 0.40) + 0.4 * lk)
    shell_atmosphere_budget = _lerp_f(0.10, 0.50,
        sm(d['hue_entropy'], 0.2, 0.7) * sm(_pd, 0.2, 0.6))
    directional_chroma_budget = _lerp_f(0.0, 0.35,
        sm(d['chroma_mean'], 0.10, 0.40) * directional_light_strength)
    rim_chroma_budget = _lerp_f(0.05, 0.45,
        sm(_cmr, 0.12, 0.40) + 0.25 * lk)
    shadow_tint_budget = _lerp_f(0.0, 0.20,
        sm(d['chroma_mean'], 0.08, 0.30) * directional_shadow_strength)
    highlight_tint_budget = _lerp_f(0.0, 0.15,
        sm(d['chroma_mean'], 0.10, 0.35) * (1.0 - hk))
    soft_atmosphere_spread = _lerp_f(0.20, 0.75,
        _misty + 0.3 * hk * (1.0 - sm(d['luma_std'], 0.08, 0.20)))
    atmosphere_locality = _lerp_f(0.30, 0.85,
        sm(_gc, 0.2, 0.7) * (1.0 - _misty))
    face_core_protection_weight = _lerp_f(0.80, 0.98,
        sm(cast_danger, 0.10, 0.45))
    face_core_bg_chroma_budget = 0.001

    # Low-key scenes need detail/readability preservation without global exposure lift.
    lowkey_detail_floor = _lerp_f(0.0, 0.25, lk)
    lowkey_local_contrast_boost = _lerp_f(1.0, 1.35, lk)
    lowkey_face_readability_gain = _lerp_f(0.0, 0.15, lk)

    style_expression = compute_style_expression(d)
    warmth = style_expression['warmth']
    coolness = style_expression['coolness']
    neon = style_expression['neon']
    mist = style_expression['mist']
    lowkey = style_expression['lowkey']
    highkey = style_expression['highkey']
    color_split = style_expression['color_split']
    directionality = style_expression['directionality']
    locality = style_expression['locality']
    atmosphere = style_expression['atmosphere']
    contrast = style_expression['contrast']
    palette = style_expression['palette_diversity']
    chroma_mean = float(d.get('chroma_mean', colorfulness))

    target_subject_p70 = float(np.clip(
        0.355 + 0.038 * highkey + 0.018 * mist - 0.050 * lowkey + 0.012 * warmth,
        0.300,
        0.420,
    ))
    exposure = {
        'exposure_scale': float(np.clip(1.045 + 0.045 * highkey + 0.018 * mist + 0.045 * lowkey - 0.018 * chroma_mean, 0.98, 1.18)),
        'auto_gain_upper': float(np.clip(1.42 + 0.12 * highkey - 0.24 * lowkey - 0.08 * neon + 0.05 * mist, 1.10, 1.58)),
        'target_subject_p70': target_subject_p70,
        'target_subject_p70_low': float(np.clip(target_subject_p70 - (0.032 + 0.010 * lowkey), 0.285, 0.405)),
        'target_subject_p70_high': float(np.clip(target_subject_p70 + (0.030 - 0.010 * highkey), 0.315, 0.430)),
        'highlight_knee_start': float(np.clip(0.70 - 0.13 * highkey + 0.055 * lowkey, 0.54, 0.82)),
        'highlight_knee_strength': float(np.clip(0.08 + 0.40 * highkey + 0.18 * warmth + 0.20 * lowkey, 0.0, 0.82)),
        'lowkey_preserve': lowkey,
        'face_lift_cap': float(np.clip(0.22 - 0.09 * highkey - 0.055 * neon + 0.035 * mist + 0.025 * lowkey, 0.055, 0.26)),
        'body_lift_cap': float(np.clip(0.20 - 0.06 * highkey - 0.035 * neon + 0.030 * mist + 0.030 * lowkey, 0.070, 0.24)),
        'autogain_face_weight': float(np.clip(1.0 - 0.46 * highkey - 0.22 * neon - 0.20 * lowkey, 0.28, 1.0)),
        'autogain_body_weight': float(np.clip(1.0 - 0.26 * highkey - 0.12 * neon - 0.12 * lowkey, 0.48, 1.0)),
        'autogain_edge_weight': float(np.clip(0.86 + 0.08 * neon + 0.08 * lowkey, 0.72, 1.0)),
    }
    chroma = {
        'face_core': float(np.clip(0.003 + 0.010 * mist + 0.004 * color_split - 0.004 * neon, 0.001, 0.016)),
        'face_side': float(np.clip(0.016 + 0.034 * warmth + 0.038 * color_split + 0.032 * neon - 0.014 * mist, 0.006, 0.095)),
        'body': float(np.clip(0.018 + 0.052 * atmosphere + 0.038 * warmth + 0.044 * neon, 0.010, 0.145)),
        'clothing': float(np.clip(0.040 + 0.160 * atmosphere + 0.130 * palette + 0.060 * neon, 0.020, 0.340)),
        'hair': float(np.clip(0.100 + 0.300 * neon + 0.180 * warmth + 0.160 * palette + 0.090 * color_split, 0.075, 0.600)),
        'edge': float(np.clip(0.120 + 0.360 * neon + 0.220 * warmth + 0.210 * color_split + 0.110 * palette, 0.090, 0.700)),
        'rim': float(np.clip(0.110 + 0.420 * neon + 0.280 * warmth + 0.220 * color_split + 0.180 * lowkey, 0.080, 0.780)),
        'global_tint': float(np.clip(0.14 + 0.22 * atmosphere + 0.10 * warmth - 0.12 * neon - 0.10 * highkey, 0.035, 0.360)),
    }
    direction = {
        'key_strength': float(np.clip(0.26 + 0.50 * directionality + 0.18 * warmth + 0.10 * lowkey - 0.18 * mist, 0.16, 0.86)),
        'shadow_strength': float(np.clip(0.08 + 0.32 * lowkey + 0.20 * contrast + 0.12 * directionality - 0.18 * mist, 0.035, 0.58)),
        'rim_strength': float(np.clip(0.16 + 0.42 * neon + 0.28 * warmth + 0.18 * lowkey + 0.14 * color_split, 0.10, 0.74)),
        'diffusion_spread': float(np.clip(0.20 + 0.56 * mist + 0.12 * highkey - 0.14 * directionality, 0.12, 0.82)),
        'side_separation': float(np.clip(0.14 + 0.50 * color_split + 0.24 * directionality + 0.18 * neon, 0.07, 0.82)),
        'directional_light_strength': float(np.clip(0.18 + 0.62 * directionality + 0.10 * warmth - 0.16 * mist, 0.12, 0.88)),
        'directional_shadow_strength': float(np.clip(0.06 + 0.46 * directionality * contrast + 0.20 * lowkey - 0.14 * mist, 0.03, 0.62)),
    }
    region = {
        'face_core_protection': float(np.clip(0.82 + 0.12 * neon + 0.08 * warmth + 0.05 * highkey, 0.78, 0.985)),
        'face_side_weight': float(np.clip(0.30 + 0.34 * directionality + 0.18 * warmth + 0.16 * neon, 0.22, 0.84)),
        'body_weight': float(np.clip(0.34 + 0.26 * atmosphere + 0.18 * directionality, 0.25, 0.82)),
        'clothing_weight': float(np.clip(0.40 + 0.34 * palette + 0.22 * neon + 0.14 * warmth, 0.28, 0.96)),
        'hair_weight': float(np.clip(0.48 + 0.34 * neon + 0.20 * warmth + 0.12 * lowkey, 0.38, 1.0)),
        'edge_weight': float(np.clip(0.52 + 0.36 * neon + 0.24 * warmth + 0.14 * color_split, 0.40, 1.0)),
    }
    render_weight = {
        'ambient': float(np.clip(0.72 + 0.28 * mist + 0.14 * highkey - 0.12 * lowkey, 0.55, 1.16)),
        'fill': float(np.clip(0.66 + 0.28 * mist + 0.12 * highkey - 0.16 * lowkey, 0.44, 1.12)),
        'gradient_field': float(np.clip(0.34 + 0.42 * atmosphere + 0.22 * palette + 0.20 * directionality, 0.20, 1.20)),
        'hdri_body': float(np.clip(0.30 + 0.30 * atmosphere + 0.18 * mist + 0.18 * palette, 0.18, 1.05)),
        'switchlight_pbr': float(np.clip(0.24 + 0.28 * directionality + 0.18 * lowkey + 0.16 * contrast, 0.16, 0.92)),
        'multicolor': float(np.clip(0.16 + 0.46 * neon + 0.24 * color_split + 0.22 * palette + 0.12 * warmth, 0.10, 0.98)),
        'direct_light': float(np.clip(0.54 + 0.42 * directionality + 0.20 * warmth + 0.12 * lowkey - 0.18 * mist, 0.34, 1.26)),
        'diffuse': float(np.clip(0.78 + 0.22 * directionality + 0.14 * warmth - 0.10 * mist, 0.55, 1.20)),
        'rim': float(np.clip(0.52 + 0.62 * neon + 0.32 * warmth + 0.22 * lowkey + 0.24 * color_split, 0.34, 1.46)),
        'specular': float(np.clip(0.44 + 0.28 * neon + 0.20 * lowkey + 0.14 * locality, 0.24, 1.06)),
        'source_preserve': float(np.clip(1.02 - 0.22 * neon - 0.14 * warmth - 0.10 * directionality + 0.16 * mist, 0.62, 1.18)),
        'source_shading_preserve': float(np.clip(1.00 - 0.20 * directionality + 0.12 * mist + 0.08 * lowkey, 0.65, 1.14)),
    }
    display = {
        'saturation': float(np.clip(0.94 + 0.16 * warmth + 0.22 * neon + 0.10 * palette - 0.10 * mist - 0.08 * highkey, 0.82, 1.25)),
        'bloom': float(np.clip(0.60 + 0.72 * neon + 0.25 * highkey + 0.18 * warmth - 0.20 * lowkey, 0.30, 1.46)),
        'haze': float(np.clip(0.64 + 0.76 * mist + 0.18 * highkey - 0.20 * lowkey, 0.34, 1.50)),
        'vignette': float(np.clip(0.74 + 0.42 * lowkey + 0.16 * neon - 0.25 * highkey, 0.45, 1.36)),
        'local_contrast': float(np.clip(0.80 + 0.30 * lowkey + 0.25 * contrast + 0.15 * neon - 0.30 * mist, 0.55, 1.36)),
        'tone_preserve': float(np.clip(0.70 + 0.20 * lowkey + 0.15 * neon - 0.10 * highkey, 0.55, 1.0)),
    }
    key_uv = list(d.get('brightest_region_uv', [0.5, 0.28]))
    if len(key_uv) < 2:
        key_uv = [0.5, 0.28]
    key_uv = [float(np.clip(key_uv[0], 0.0, 1.0)), float(np.clip(key_uv[1], 0.0, 1.0))]
    key_dir = policy_direction_from_uv(key_uv)

    exposure.update({
        'global_exposure_scale': exposure['exposure_scale'],
        'face_target_luma': float(np.clip(0.5 * (exposure['target_subject_p70_low'] + exposure['target_subject_p70_high']), 0.28, 0.48)),
        'body_target_luma': float(np.clip(exposure['target_subject_p70'] - 0.020 + 0.030 * atmosphere + 0.020 * directionality, 0.260, 0.450)),
        'target_subject_luma': exposure['target_subject_p70'],
        'max_gain': exposure['auto_gain_upper'],
        'highlight_knee': exposure['highlight_knee_strength'],
        'shadow_floor': float(np.clip(0.055 + 0.090 * lowkey + 0.035 * mist, 0.035, 0.18)),
        'face_readability': float(np.clip(0.34 + 0.34 * lowkey + 0.18 * mist + 0.10 * highkey, 0.25, 0.82)),
        'background_respect': float(np.clip(0.92 + 0.06 * lowkey + 0.04 * atmosphere - 0.08 * highkey, 0.72, 1.0)),
    })
    chroma_pressure_policy = float(np.clip(
        0.44 * float(d.get('colorfulness', chroma_mean))
        + 0.24 * float(d.get('average_saturation', chroma_mean))
        + 0.18 * palette
        + 0.14 * float(d.get('dominant_hue_strength', 0.5)),
        0.0,
        1.0,
    ))
    bg_p50_policy = float(d.get('luma_p50', d.get('p50_luma', d.get('global_luma', 0.25))))
    bg_p95_policy = float(d.get('luma_p95', d.get('p95_luma', d.get('luma_p90', 0.45))))
    bias_strength_policy = float(np.clip(
        (
            abs(float(d.get('horizontal_bias', d.get('left_right_luma_diff', 0.0))))
            + abs(float(d.get('vertical_bias', d.get('top_bottom_luma_diff', 0.0))))
        ) / 0.12,
        0.0,
        1.0,
    ))
    peak_structure_policy = float(np.clip(
        0.42 * sm(float(d.get('gradient_confidence', 0.0)), 0.12, 0.74)
        + 0.22 * sm(float(d.get('source_peak_strength', 0.0)), 0.035, 0.45)
        + 0.14 * sm(float(d.get('source_peak_chroma', d.get('chroma_max_region', chroma_mean))), 0.12, 0.45)
        + 0.14 * bias_strength_policy
        + 0.08 * sm(float(d.get('gradient_local_contrast', d.get('local_contrast', 0.0))), 0.10, 0.55),
        0.0,
        1.0,
    ))
    lowkey_chroma_direction_gate = float(np.clip(
        sm(0.34 - bg_p50_policy, 0.02, 0.30)
        * sm(max(chroma_mean, 0.72 * float(d.get('chroma_max_region', chroma_mean)), 0.62 * float(d.get('gradient_colorfulness', chroma_mean))), 0.10, 0.40)
        * (0.34 + 0.66 * peak_structure_policy)
        * (1.0 - sm(bg_p50_policy, 0.28, 0.50))
        * (1.0 - 0.55 * sm(bg_p95_policy, 0.62, 0.90))
        * (1.0 - 0.68 * mist)
        * (1.0 - 0.50 * highkey),
        0.0,
        1.0,
    ))
    low_chroma_air_skin_guard = float(np.clip(
        3.10
        * sm(bg_p50_policy, 0.24, 0.48)
        * (1.0 - sm(max(chroma_mean, 0.52 * float(d.get('chroma_max_region', chroma_mean))), 0.13, 0.34))
        * (0.42 + 0.58 * sm(float(d.get('haze_score', mist)), 0.18, 0.72))
        * (0.50 + 0.50 * sm(float(d.get('flatness_score', 0.0)), 0.10, 0.72))
        * (0.78 + 0.22 * sm(abs(float(d.get('warm_presence', d.get('warm_ratio', 0.0))) - float(d.get('cool_presence', d.get('cool_ratio', 0.0)))) * float(d.get('average_saturation', chroma_mean)), 0.02, 0.18))
        * (1.0 - 0.80 * lowkey_chroma_direction_gate),
        0.0,
        1.0,
    ))
    local_direction_policy = float(np.clip(
        0.58 * directionality
        + 0.28 * bias_strength_policy
        + 0.14 * float(d.get('local_light_confidence', 0.0)),
        0.0,
        1.0,
    ))
    eval_style_target = float(np.clip(
        0.255
        + 0.55 * float(d.get('luma_p50', d.get('global_luma', 0.25)))
        + 0.08 * max(float(d.get('luma_p95', d.get('luma_p90', 0.45))) - float(d.get('luma_p50', 0.25)), 0.0),
        0.235,
        0.500,
    ))
    face_target_mid = float(np.clip(
        0.305
        + 0.34 * eval_style_target
        + 0.030 * lowkey
        - 0.020 * highkey
        - 0.012 * mist
        - 0.018 * chroma_pressure_policy,
        0.310,
        0.500,
    ))
    exposure.update({
        'face_target_luma_min': float(np.clip(
            face_target_mid
            - 0.080
            - 0.035 * lowkey
            - 0.030 * chroma_pressure_policy
            - 0.020 * lowkey * (1.0 - float(d.get('global_luma', 0.25)))
            - 0.020 * mist,
            0.235,
            0.420,
        )),
        'face_target_luma_max': float(np.clip(face_target_mid + 0.090 - 0.024 * mist, 0.330, 0.560)),
        'highlight_luma_ceiling': float(np.clip(0.78 - 0.08 * highkey - 0.05 * mist + 0.03 * lowkey, 0.58, 0.88)),
        'shadow_luma_floor': float(np.clip(0.070 + 0.060 * lowkey + 0.026 * chroma_pressure_policy + 0.014 * mist, 0.055, 0.165)),
    })
    exposure['face_target_luma_min'] = float(np.clip(exposure['face_target_luma_min'] + 0.010 * lowkey_chroma_direction_gate, 0.235, 0.430))
    exposure['body_target_luma'] = float(np.clip(exposure.get('body_target_luma', exposure['target_subject_p70'] - 0.020) + 0.008 * lowkey_chroma_direction_gate, 0.255, 0.455))
    exposure['face_lift_cap'] = float(np.clip(exposure['face_lift_cap'] + 0.010 * lowkey_chroma_direction_gate, 0.045, 0.230))
    exposure['body_lift_cap'] = float(np.clip(exposure['body_lift_cap'] + 0.012 * lowkey_chroma_direction_gate, 0.055, 0.210))
    exposure['body_face_luma_gap_target'] = float(np.clip(
        0.085
        + 0.026 * lowkey
        + 0.018 * chroma_pressure_policy
        + 0.010 * lowkey_chroma_direction_gate
        - 0.018 * mist
        - 0.012 * highkey,
        0.060,
        0.150,
    ))
    exposure['body_target_luma'] = float(np.clip(
        max(
            exposure['body_target_luma'],
            exposure['face_target_luma_min'] - exposure['body_face_luma_gap_target'],
            exposure['target_subject_p70'] - 0.030,
        ),
        0.255,
        0.455,
    ))
    exposure['whole_subject_sync_strength'] = float(np.clip(
        0.30
        + 0.16 * local_direction_policy
        + 0.10 * lowkey_chroma_direction_gate
        + 0.08 * chroma_pressure_policy
        + 0.06 * atmosphere,
        0.24,
        0.62,
    ))
    exposure['body_fill_floor_strength'] = float(np.clip(
        0.20
        + 0.14 * atmosphere
        + 0.12 * mist
        + 0.10 * lowkey
        + 0.08 * chroma_pressure_policy,
        0.12,
        0.48,
    ))
    chroma.update({
        'ambient_tint_strength': chroma['global_tint'],
        'skin_tint_limit': float(np.clip(0.070 + 0.070 * mist + 0.040 * highkey - 0.038 * neon - 0.028 * palette - 0.070 * low_chroma_air_skin_guard, 0.018, 0.160)),
        'body_tint_strength': chroma['body'],
        'cloth_tint_strength': chroma['clothing'],
        'hair_tint_strength': chroma['hair'],
        'edge_color_spill': chroma['edge'],
        'palette_separation': float(np.clip(0.12 + 0.52 * color_split + 0.28 * palette + 0.18 * neon, 0.08, 0.86)),
    })
    chroma.update({
        'face_core_chroma_limit': float(np.clip(0.018 - 0.010 * chroma_pressure_policy - 0.004 * float(d.get('dominant_hue_strength', 0.5)) + 0.004 * mist - 0.004 * lowkey_chroma_direction_gate - 0.004 * low_chroma_air_skin_guard, 0.002, 0.018)),
        'face_side_chroma_allowance': float(np.clip(0.010 + 0.024 * color_split + 0.008 * (1.0 - mist) + 0.010 * lowkey_chroma_direction_gate + 0.010 * palette - 0.014 * low_chroma_air_skin_guard, 0.004, 0.060)),
        'body_chroma_allowance': float(np.clip(0.060 + 0.120 * chroma_pressure_policy + 0.055 * palette + 0.030 * directionality + 0.040 * lowkey_chroma_direction_gate - 0.130 * low_chroma_air_skin_guard, 0.014, 0.300)),
        'clothing_chroma_budget': float(np.clip(0.18 + 0.26 * chroma_pressure_policy + 0.14 * palette + 0.06 * directionality + 0.095 * lowkey_chroma_direction_gate, 0.14, 0.66)),
        'hair_chroma_allowance': float(np.clip(0.38 + 0.40 * chroma_pressure_policy + 0.22 * palette + 0.06 * lowkey + 0.120 * lowkey_chroma_direction_gate, 0.32, 0.96)),
        'edge_chroma_allowance': float(np.clip(0.50 + 0.42 * chroma_pressure_policy + 0.22 * palette + 0.06 * lowkey + 0.150 * lowkey_chroma_direction_gate, 0.44, 1.04)),
    })
    chroma.update({
        'face_core_chroma_budget': chroma['face_core_chroma_limit'],
        'face_side_chroma_budget': chroma['face_side_chroma_allowance'],
        'body_skin_chroma_budget': chroma['body_chroma_allowance'],
        'hair_chroma_budget': chroma['hair_chroma_allowance'],
        'edge_chroma_budget': chroma['edge_chroma_allowance'],
        'global_tint_strength': float(np.clip(chroma['global_tint'] * (1.0 - 0.44 * local_direction_policy) * (1.0 - 0.30 * chroma_pressure_policy) * (1.0 - 0.42 * low_chroma_air_skin_guard), 0.010, 0.260)),
    })
    direction.update({
        'key_dir': key_dir,
        'key_uv': key_uv,
        'local_light_confidence': float(np.clip(d.get('local_light_confidence', directionality), 0.0, 1.0)),
        'gradient_light_strength': float(np.clip(d.get('gradient_strength', d.get('gradient_magnitude', 0.0)), 0.0, 1.0)),
        'direction_strength': direction['directional_light_strength'],
        'key_direction_strength': direction['key_strength'],
        'direction_magnitude': float(np.clip(0.22 + 0.46 * local_direction_policy + 0.18 * bias_strength_policy + 0.16 * contrast + 0.08 * color_split + 0.16 * lowkey_chroma_direction_gate, 0.16, 0.98)),
        'direction_contrast': float(np.clip(0.16 + 0.40 * local_direction_policy * (0.55 + 0.45 * contrast) + 0.18 * bias_strength_policy + 0.12 * lowkey + 0.12 * lowkey_chroma_direction_gate, 0.10, 0.88)),
        'direction_softness': float(np.clip(
            0.32
            + 0.28 * float(d.get('haze_score', mist))
            + 0.22 * float(d.get('flatness_score', 0.0))
            + 0.18 * chroma_pressure_policy,
            0.28,
            0.72,
        )),
        'direction_spread': direction['diffusion_spread'],
        'direction_locality': float(np.clip(0.32 + 0.34 * locality + 0.22 * local_direction_policy + 0.12 * bias_strength_policy, 0.25, 0.94)),
    })
    region.update({
        'face_core_weight': float(np.clip(0.22 + 0.20 * exposure['face_readability'] - 0.10 * chroma['palette_separation'] - 0.035 * lowkey_chroma_direction_gate, 0.10, 0.42)),
        'jaw_neck_weight': float(np.clip(0.40 + 0.26 * local_direction_policy + 0.12 * atmosphere + 0.08 * lowkey_chroma_direction_gate, 0.30, 0.98)),
        'body_side_weight': float(np.clip(region['body_weight'] + 0.16 * local_direction_policy + 0.08 * bias_strength_policy + 0.08 * atmosphere + 0.10 * lowkey_chroma_direction_gate, 0.36, 1.06)),
        'body_weight': float(np.clip(region['body_weight'] + 0.05 * atmosphere + 0.04 * lowkey_chroma_direction_gate, 0.30, 0.92)),
        'hand_weight': float(np.clip(0.44 + 0.16 * local_direction_policy + 0.10 * atmosphere + 0.08 * lowkey_chroma_direction_gate + 0.08 * bias_strength_policy, 0.34, 1.02)),
        'cloth_weight': float(np.clip(region['clothing_weight'] + 0.10 * local_direction_policy + 0.06 * palette + 0.08 * lowkey_chroma_direction_gate, 0.32, 1.02)),
        'hair_weight': float(np.clip(region['hair_weight'] + 0.12 * local_direction_policy + 0.06 * bias_strength_policy + 0.11 * lowkey_chroma_direction_gate, 0.38, 1.18)),
        'edge_weight': float(np.clip(region['edge_weight'] + 0.14 * local_direction_policy + 0.06 * bias_strength_policy + 0.14 * lowkey_chroma_direction_gate, 0.40, 1.20)),
        'shoulder_weight': float(np.clip(0.46 + 0.26 * local_direction_policy + 0.12 * lowkey + 0.10 * palette + 0.08 * bias_strength_policy + 0.12 * lowkey_chroma_direction_gate, 0.34, 1.06)),
    })
    render_weight.update({
        'direct_weight': render_weight['direct_light'],
        'ambient_weight': render_weight['ambient'],
        'fill_weight': render_weight['fill'],
        'gradient_weight': render_weight['gradient_field'],
        'multicolor_weight': render_weight['multicolor'],
        'rim_weight': render_weight['rim'],
        'specular_weight': render_weight['specular'],
        'shadow_weight': float(np.clip(0.46 + 0.46 * lowkey + 0.32 * contrast + 0.18 * directionality - 0.18 * mist - 0.08 * lowkey_chroma_direction_gate, 0.24, 1.18)),
        'direct_light_weight': float(np.clip(render_weight['direct_light'] + 0.12 * local_direction_policy + 0.12 * lowkey_chroma_direction_gate, 0.34, 1.42)),
        'gradient_light_weight': float(np.clip(render_weight['gradient_field'] + 0.16 * local_direction_policy + 0.08 * bias_strength_policy, 0.20, 1.30)),
        'hdri_weight': float(np.clip(render_weight['hdri_body'] + 0.12 * atmosphere + 0.10 * local_direction_policy, 0.18, 1.16)),
        'pbr_weight': float(np.clip(render_weight['switchlight_pbr'] + 0.12 * local_direction_policy + 0.08 * contrast, 0.16, 1.06)),
    })
    # Compact look-safe compositor reads only these six weights.  The legacy
    # aliases above remain for non-look-safe compatibility, but they no longer
    # split the final look-safe style path.
    render_weight['direct_weight'] = float(np.clip(render_weight['direct_light_weight'], 0.34, 1.36))
    render_weight['shadow_weight'] = float(np.clip(render_weight['shadow_weight'], 0.26, 1.18))
    render_weight['rim_weight'] = float(np.clip(render_weight['rim_weight'] + 0.055 * lowkey_chroma_direction_gate, 0.34, 1.54))
    render_weight['ambient_weight'] = float(np.clip(render_weight['ambient_weight'] * (1.0 - 0.18 * local_direction_policy), 0.42, 1.10))
    render_weight['fill_weight'] = float(np.clip(render_weight.get('fill_weight', render_weight['fill']) + 0.04 * exposure['body_fill_floor_strength'], 0.42, 1.08))
    render_weight['body_fill_weight'] = float(np.clip(0.38 + 0.18 * exposure['body_fill_floor_strength'] + 0.08 * (1.0 - local_direction_policy), 0.30, 0.70))
    render_weight['color_spill_weight'] = float(np.clip(
        0.34 + 0.48 * chroma['palette_separation'] + 0.22 * palette + 0.22 * color_split + 0.12 * atmosphere + 0.105 * lowkey_chroma_direction_gate,
        0.22,
        1.34,
    ))
    render_weight['display_weight'] = float(np.clip(0.62 + 0.18 * mist + 0.12 * lowkey - 0.08 * highkey, 0.50, 0.92))
    display.update({
        'contrast': float(np.clip(0.92 + 0.16 * contrast + 0.10 * lowkey - 0.14 * mist, 0.78, 1.18)),
        'tone_map_strength': float(np.clip(0.82 + 0.12 * highkey - 0.10 * lowkey, 0.68, 1.0)),
        'display_saturation': display['saturation'],
        'saturation_scale': display['saturation'],
        'bloom_strength': display['bloom'],
        'haze_strength': display['haze'],
        'vignette_strength': display['vignette'],
        'local_contrast_strength': display['local_contrast'],
        'display_contrast': float(np.clip(0.92 + 0.16 * contrast + 0.10 * lowkey - 0.14 * mist, 0.78, 1.18)),
        'texture_preserve_strength': float(np.clip(0.08 + 0.10 * mist + 0.08 * contrast + 0.06 * lowkey - 0.020 * lowkey_chroma_direction_gate, 0.045, 0.22)),
    })

    autogain_target_low = exposure['target_subject_p70_low']
    autogain_target_high = exposure['target_subject_p70_high']
    autogain_upper = exposure['auto_gain_upper']
    exposure_scale = exposure['exposure_scale']
    display_saturation_multiplier = display['saturation']
    bloom_multiplier = display['bloom']
    haze_multiplier = display['haze']
    subject_highlight_knee_start = exposure['highlight_knee_start']
    subject_highlight_knee_strength = exposure['highlight_knee_strength']
    autogain_face_weight = exposure['autogain_face_weight']
    autogain_body_weight = exposure['autogain_body_weight']
    autogain_edge_weight = exposure['autogain_edge_weight']
    max_positive_face_lift = exposure['face_lift_cap']
    max_positive_body_lift = exposure['body_lift_cap']
    global_tint_scale = chroma['global_tint']
    directional_light_strength = float(np.clip(direction['directional_light_strength'] + 0.10 * lowkey_chroma_direction_gate, 0.0, 0.96))
    directional_shadow_strength = float(np.clip(direction['directional_shadow_strength'] - 0.035 * lowkey_chroma_direction_gate, 0.03, 0.62))
    directional_contrast_strength = float(np.clip(0.08 + 0.42 * directionality * (1.0 - 0.55 * mist) + 0.13 * lowkey_chroma_direction_gate, 0.06, 0.62))
    face_core_directional_budget = float(np.clip(0.016 + 0.050 * directionality * (1.0 - highkey) + 0.012 * lowkey_chroma_direction_gate, 0.008, 0.082))
    face_side_directional_budget = float(np.clip(0.040 + 0.240 * direction['key_strength'] + 0.070 * lowkey_chroma_direction_gate, 0.035, 0.350))
    body_directional_budget = float(np.clip(0.060 + 0.260 * direction['key_strength'] + 0.090 * lowkey_chroma_direction_gate, 0.050, 0.410))
    clothing_directional_budget = float(np.clip(0.080 + 0.330 * direction['key_strength'] + 0.095 * lowkey_chroma_direction_gate, 0.060, 0.520))
    hair_rim_budget = chroma['hair']
    edge_rim_budget = chroma['edge']
    shell_atmosphere_budget = float(np.clip(0.5 * (chroma['hair'] + chroma['edge']), 0.08, 0.65))
    directional_chroma_budget = float(np.clip(chroma['face_side'] + 0.18 * color_split + 0.040 * lowkey_chroma_direction_gate, 0.0, 0.42))
    rim_chroma_budget = float(np.clip(chroma['rim'] + 0.080 * lowkey_chroma_direction_gate + 0.040 * palette, 0.08, 0.88))
    shadow_tint_budget = float(np.clip(0.025 + 0.28 * chroma_mean * direction['shadow_strength'] + 0.050 * palette, 0.0, 0.30))
    highlight_tint_budget = float(np.clip(0.02 + 0.18 * warmth + 0.10 * neon + 0.035 * palette, 0.0, 0.26))
    soft_atmosphere_spread = direction['diffusion_spread']
    atmosphere_locality = float(np.clip(0.25 + 0.65 * locality * (1.0 - 0.55 * mist), 0.20, 0.90))
    face_core_protection_weight = region['face_core_protection']
    face_core_chroma_authority = 1.0 - chroma['face_core']
    face_side_chroma_authority = float(np.clip(1.0 - 4.0 * chroma['face_side'], 0.45, 0.95))
    body_skin_chroma_authority = float(np.clip(1.0 - 2.6 * chroma['body'] + 0.34 * low_chroma_air_skin_guard, 0.45, 0.985))
    clothing_chroma_authority = chroma['clothing']
    hair_chroma_authority = float(np.clip(0.90 + 1.20 * chroma['hair'] + 0.08 * lowkey_chroma_direction_gate, 0.9, 1.65))
    edge_shell_chroma_authority = float(np.clip(0.90 + 1.20 * chroma['edge'] + 0.10 * lowkey_chroma_direction_gate, 0.9, 1.75))
    atmosphere_carrier_strength = float(np.clip(0.90 + 0.55 * atmosphere + 0.30 * neon + 0.16 * lowkey_chroma_direction_gate, 0.85, 1.70))
    v32_face_bg_budget = chroma['face_core']
    v32_face_side_budget = chroma['face_side']
    v32_shell_scale = float(np.clip(1.0 + 0.40 * atmosphere, 1.0, 1.55))
    v32_rim_scale = float(np.clip(0.90 + 0.80 * direction['rim_strength'], 0.90, 1.50))
    v32_luma_block_scale = float(np.clip(0.35 + 0.65 * (1.0 - highkey) + 0.25 * lowkey, 0.25, 1.15))
    v32_warm_gate_cap = float(np.clip(0.55 + 0.40 * (1.0 - warmth * chroma_mean), 0.45, 1.0))
    face_core_bg_chroma_budget = chroma['face_core']
    luma_authority_strength = float(np.clip(0.65 * highkey + 0.35 * warmth * chroma_mean, 0.0, 1.0))
    air_guard_luma_lift_cap = exposure['face_lift_cap']
    v32_positive_luma_gate = float(np.clip(1.0 - 0.82 * luma_authority_strength, 0.05, 1.0))
    highkey_overbright_knee = float(np.clip(0.84 - 0.20 * highkey + 0.08 * lowkey, 0.62, 0.88))
    pbr_preserve_strength = float(np.clip(0.10 + 0.42 * directionality + 0.20 * lowkey, 0.0, 0.65))
    lowkey_local_contrast_boost = display['local_contrast']
    lowkey_face_readability_gain = float(np.clip(0.04 + 0.16 * lowkey, 0.0, 0.20))
    direction['directional_light_strength'] = directional_light_strength
    direction['direction_strength'] = directional_light_strength
    direction['directional_shadow_strength'] = directional_shadow_strength
    direction['shadow_strength'] = directional_shadow_strength
    direction['direction_contrast'] = max(float(direction.get('direction_contrast', directional_contrast_strength)), directional_contrast_strength)
    light_scene = {
        'key_uv': key_uv,
        'key_dir': key_dir,
        'local_light_confidence': float(np.clip(d.get('local_light_confidence', directionality), 0.0, 1.0)),
        'direction_gate': lowkey_chroma_direction_gate,
        'air_skin_guard': low_chroma_air_skin_guard,
        'bg_p50_luma': bg_p50_policy,
        'bg_p95_luma': bg_p95_policy,
        'chroma_pressure': chroma_pressure_policy,
        'body_face_luma_gap_target': exposure['body_face_luma_gap_target'],
        'whole_subject_sync_strength': exposure['whole_subject_sync_strength'],
    }

    return dict(
        light_scene=light_scene,
        style_expression=style_expression,
        exposure=exposure,
        chroma=chroma,
        direction=direction,
        region=region,
        render_weight=render_weight,
        display=display,
        face_luma_lift_gate=face_luma_lift_gate,
        face_chroma_inject=face_chroma_inject,
        face_side_chroma_inject=face_side_chroma_inject,
        hair_chroma_boost=hair_chroma_boost,
        edge_chroma_boost=edge_chroma_boost,
        multicolor_face_gate_floor=multicolor_face_gate_floor,
        autogain_target_low=autogain_target_low,
        autogain_target_high=autogain_target_high,
        autogain_upper=autogain_upper,
        global_tint_scale=global_tint_scale,
        exposure_scale=exposure_scale,
        display_saturation_multiplier=float(np.clip(display_saturation_multiplier, 0.80, 1.18)),
        v32_face_bg_budget=v32_face_bg_budget,
        v32_face_side_budget=v32_face_side_budget,
        v32_shell_scale=v32_shell_scale,
        v32_rim_scale=v32_rim_scale,
        v32_luma_block_scale=v32_luma_block_scale,
        v32_warm_gate_cap=v32_warm_gate_cap,
        highlight_soft_knee=highlight_soft_knee,
        bloom_multiplier=bloom_multiplier,
        haze_multiplier=haze_multiplier,
        v32_face_gate_cap=v32_face_gate_cap,
        v32_body_gate_cap=v32_body_gate_cap,
        v32_luma_lift_multiplier=v32_luma_lift_multiplier,
        pbr_preserve_strength=pbr_preserve_strength,
        subject_highlight_knee_start=subject_highlight_knee_start,
        subject_highlight_knee_strength=subject_highlight_knee_strength,
        luma_authority_strength=luma_authority_strength,
        air_guard_luma_lift_cap=air_guard_luma_lift_cap,
        v32_positive_luma_gate=v32_positive_luma_gate,
        autogain_face_weight=autogain_face_weight,
        autogain_body_weight=autogain_body_weight,
        autogain_edge_weight=autogain_edge_weight,
        max_positive_face_lift=max_positive_face_lift,
        max_positive_body_lift=max_positive_body_lift,
        highkey_overbright_knee=highkey_overbright_knee,
        face_core_chroma_authority=face_core_chroma_authority,
        face_side_chroma_authority=face_side_chroma_authority,
        body_skin_chroma_authority=body_skin_chroma_authority,
        clothing_chroma_authority=clothing_chroma_authority,
        hair_chroma_authority=hair_chroma_authority,
        edge_shell_chroma_authority=edge_shell_chroma_authority,
        warm_skin_contamination_guard=warm_skin_contamination_guard,
        skin_yellow_wash_guard=skin_yellow_wash_guard,
        subject_global_tint_guard=subject_global_tint_guard,
        atmosphere_carrier_strength=atmosphere_carrier_strength,
        directional_light_strength=directional_light_strength,
        directional_shadow_strength=directional_shadow_strength,
        directional_contrast_strength=directional_contrast_strength,
        face_core_directional_budget=face_core_directional_budget,
        face_side_directional_budget=face_side_directional_budget,
        body_directional_budget=body_directional_budget,
        clothing_directional_budget=clothing_directional_budget,
        hair_rim_budget=hair_rim_budget,
        edge_rim_budget=edge_rim_budget,
        shell_atmosphere_budget=shell_atmosphere_budget,
        directional_chroma_budget=directional_chroma_budget,
        rim_chroma_budget=rim_chroma_budget,
        shadow_tint_budget=shadow_tint_budget,
        highlight_tint_budget=highlight_tint_budget,
        soft_atmosphere_spread=soft_atmosphere_spread,
        atmosphere_locality=atmosphere_locality,
        lowkey_chroma_direction_gate=lowkey_chroma_direction_gate,
        source_peak_direction_evidence=peak_structure_policy,
        low_chroma_air_skin_guard=low_chroma_air_skin_guard,
        face_core_protection_weight=face_core_protection_weight,
        face_core_bg_chroma_budget=float(np.clip(face_core_bg_chroma_budget, 0.0, 0.016)),
        lowkey_detail_floor=lowkey_detail_floor,
        lowkey_local_contrast_boost=lowkey_local_contrast_boost,
        lowkey_face_readability_gain=lowkey_face_readability_gain,
    )


