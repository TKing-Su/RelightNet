from __future__ import annotations
import os
import numpy as np
from PIL import Image
from tools.color import linear_to_srgb, rgb_luminance, srgb_to_linear
from tools.image_io import read_color_image_linear

def normalize_style_mode(style: str, fallback: str = 'quality', look_safe: bool = True) -> str:
    """Normalize legacy style aliases used by different refactor branches.

    Older code paths sometimes refer to `neon`, while other branches use
    `cyber`; V21 only exposes quality/cinematic/neon in argparse, so this
    small compatibility shim prevents runtime NameError and keeps branches
    style-isolated.

    In look_safe mode, always returns 'quality' to disable filename/profile-based
    style routing and rely entirely on continuous atmosphere budget.
    """
    # Look-safe mode: disable all discrete style routing
    if look_safe:
        return 'quality'
    
    s = str(style or fallback).strip().lower()
    aliases = {
        'auto': fallback,
        'cyber': 'neon',
        'cyber_neon': 'neon',
        'neon': 'neon',
        'quality': 'quality',
        'natural': 'quality',
        'misty': 'quality',
        'misty_natural': 'quality',
        'balanced': 'quality',
        'cinema': 'cinematic',
        'lowkey': 'cinematic',
        'cinematic': 'cinematic',
        'warm': 'quality',
        'cool': 'quality',
        'studio': 'quality',
    }
    return aliases.get(s, fallback)

def choose_style_mode_from_background_path(background_path: str, look_safe: bool = True) -> str:
    """Quality-first auto mode with filename hints.

    V21.1 fix: cyber.png was being classified as cinematic because it is
    dark/cool-dominant.  For experiments where background names are explicit,
    file-name hints should win over conservative pixel statistics.

    In look-safe mode, filename hints and discrete style routing are disabled.
    Background pixels still drive lighting through the continuous atmosphere budget.
    """
    # Look-safe mode: disable filename-based style routing
    if look_safe:
        print('[LookPolicy] filename_style_hints disabled in look-safe mode; selected=quality')
        return 'quality'
    else:
        # Only use filename hints in non-look-safe mode
        name = os.path.basename(str(background_path)).lower()
        if any(k in name for k in ('cyber', 'neon', 'futuristic', 'city_night', 'rain_neon')):
            print('Auto background name hint: selected=neon')
            return 'neon'
        if any(k in name for k in ('mist', 'fog', 'haze', 'cloud', 'forest', 'morning', 'natural')):
            print('Auto background name hint: selected=quality')
            return 'quality'
        if any(k in name for k in ('sunset', 'fire', 'warm', 'autumn', 'red', 'orange')):
            print('Auto background name hint: selected=quality (V32 warm/red router will be used)')
            return 'quality'
        if any(k in name for k in ('cinematic', 'lowkey', 'stage', 'spotlight', 'dramatic')):
            print('Auto background name hint: selected=cinematic')
            return 'cinematic'
    
    # Pixel-based classification (used in look-safe mode or as fallback)
    try:
        bg = read_color_image_linear(background_path)
    except Exception:
        return 'quality'

    bg_small = bg
    h, w = bg_small.shape[:2]
    scale = min(1.0, 320.0 / max(h, w))
    if scale < 1.0:
        nh = max(64, int(round(h * scale)))
        nw = max(64, int(round(w * scale)))
        img_u8 = np.clip(linear_to_srgb(bg_small) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        bg_small = np.asarray(
            Image.fromarray(img_u8).resize((nw, nh), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        bg_small = srgb_to_linear(bg_small)

    lum = rgb_luminance(bg_small)
    global_luma = float(np.mean(lum))
    p75_luma = float(np.percentile(lum, 75.0))
    p90_luma = float(np.percentile(lum, 90.0))

    rgb = np.clip(bg_small.astype(np.float32), 0.0, None)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    diff = mx - mn
    sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0).astype(np.float32)
    hue = np.zeros_like(mx, dtype=np.float32)

    valid_hue = diff > 1e-6
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    mask_r = valid_hue & (mx == r)
    mask_g = valid_hue & (mx == g)
    mask_b = valid_hue & (mx == b)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / np.maximum(diff[mask_r], 1e-6)) % 6.0
    hue[mask_g] = ((b[mask_g] - r[mask_g]) / np.maximum(diff[mask_g], 1e-6)) + 2.0
    hue[mask_b] = ((r[mask_b] - g[mask_b]) / np.maximum(diff[mask_b], 1e-6)) + 4.0
    hue = (hue / 6.0).astype(np.float32)

    luma_norm = np.clip(lum / max(np.percentile(lum, 98.0), 1e-6), 0.0, 1.5)
    color_weight = np.clip(luma_norm, 0.0, 1.5) * np.power(np.clip(sat, 0.0, 1.0), 1.25)
    weighted_sat = float((sat * color_weight).sum() / max(float(color_weight.sum()), 1e-6))
    high_sat_area = float(np.mean((sat > 0.32) & (lum >= p75_luma)))
    neon_like_area = float(np.mean((sat > 0.38) & (lum >= p90_luma)))

    total = float(color_weight.sum()) + 1e-6
    cool_mask = (hue >= 0.43) & (hue <= 0.74) & (sat > 0.18)
    warm_mask = ((hue <= 0.16) | (hue >= 0.92)) & (sat > 0.16)
    cool_presence = float(color_weight[cool_mask].sum() / total) if np.any(cool_mask) else 0.0
    warm_presence = float(color_weight[warm_mask].sum() / total) if np.any(warm_mask) else 0.0
    two_tone_balance = float(min(cool_presence, warm_presence) / max(max(cool_presence, warm_presence), 1e-6))

    bins = 12
    hist = np.zeros((bins,), dtype=np.float32)
    hue_idx = np.floor(np.clip(hue, 0.0, 0.9999) * bins).astype(np.int32)
    for i in range(bins):
        hist[i] = float(color_weight[hue_idx == i].sum())
    probs = hist / max(float(hist.sum()), 1e-6)
    valid = probs > 1e-8
    hue_entropy = float(-(probs[valid] * np.log(probs[valid])).sum() / np.log(bins)) if np.any(valid) else 0.0
    dominant_share = float(probs.max()) if probs.size else 1.0
    palette_diversity = float(np.clip(0.55 * hue_entropy + 0.25 * weighted_sat + 0.20 * (1.0 - dominant_share), 0.0, 1.0))

    # Dark, mostly blue/purple cyber scenes often look better in cinematic than neon.
    # Keep them out of neon unless there is clear warm/cool balance.
    if (
        global_luma <= 0.14
        and weighted_sat >= 0.38
        and high_sat_area >= 0.055
        and cool_presence >= 0.42
        and warm_presence <= 0.18
        and two_tone_balance < 0.30
    ):
        selected = 'neon'
    elif (
        cool_presence >= 0.14
        and warm_presence >= 0.12
        and two_tone_balance >= 0.42
        and palette_diversity >= 0.52
        and weighted_sat >= 0.30
        and high_sat_area >= 0.045
        and neon_like_area >= 0.010
    ):
        selected = 'neon'
    elif (
        global_luma <= 0.16
        and weighted_sat >= 0.30
        and (high_sat_area >= 0.040 or neon_like_area >= 0.008)
        and palette_diversity >= 0.32
    ):
        selected = 'cinematic'
    elif global_luma <= 0.10:
        selected = 'cinematic'
    else:
        selected = 'quality'

    print(
        "Auto background stats: "
        f"luma={global_luma:.3f}, weighted_sat={weighted_sat:.3f}, "
        f"high_sat_area={high_sat_area:.3f}, neon_area={neon_like_area:.3f}, "
        f"warm={warm_presence:.3f}, cool={cool_presence:.3f}, "
        f"balance={two_tone_balance:.3f}, diversity={palette_diversity:.3f}, selected={selected}"
    )
    return selected
