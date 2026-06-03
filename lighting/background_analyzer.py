from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from config.constants import *
from lighting.models import *
from lighting.presets import *
from config.paths import *
from lighting.style_mode import *
from tools.color import *
from tools.filters import *
from tools.geometry import *
from tools.image_io import *

class BackgroundStudioLightExtractor:
    def __init__(self, max_lights: int = 6, style_mode: str = 'default') -> None:
        self.max_lights = max(4, int(max_lights))
        self.max_candidates = max(self.max_lights * 4, 16)
        self.style_mode = str(style_mode or 'default').lower()
        self.cool_hue_min = 0.43
        self.cool_hue_max = 0.74
        self.warm_hue_lo = 0.18
        self.warm_hue_hi = 0.92
        self.min_sat_for_palette = 0.10 if self.style_mode == 'neon' else 0.12
        self.monochrome_diversity_threshold = 0.20
        self.rich_diversity_threshold = 0.42
        self.monochrome_dominant_share = 0.60

    @staticmethod
    def _weighted_region_mean(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
        w = np.clip(weight.astype(np.float32), 0.0, None)
        s = float(w.sum())
        if s < 1e-6:
            return np.mean(img.reshape(-1, 3), axis=0).astype(np.float32)
        return ((img * w[..., None]).sum(axis=(0, 1)) / s).astype(np.float32)

    @staticmethod
    def _sample_patch_mean(img: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
        h, w = img.shape[:2]
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            return np.mean(img.reshape(-1, 3), axis=0).astype(np.float32)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = xx - float(cx)
        dy = yy - float(cy)
        sigma = max(radius * 0.60, 2.0)
        wgt = np.exp(-(dx * dx + dy * dy) / max(2.0 * sigma * sigma, 1e-5)).astype(np.float32)
        return ((patch * wgt[..., None]).sum(axis=(0, 1)) / max(float(wgt.sum()), 1e-6)).astype(np.float32)

    @staticmethod
    def _rgb_to_hsv_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = np.clip(img.astype(np.float32), 0.0, None)
        mx = rgb.max(axis=-1)
        mn = rgb.min(axis=-1)
        diff = mx - mn
        sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0).astype(np.float32)
        hue = np.zeros_like(mx, dtype=np.float32)
        mask = diff > 1e-6
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        mask_r = mask & (mx == r)
        mask_g = mask & (mx == g)
        mask_b = mask & (mx == b)
        hue[mask_r] = ((g[mask_r] - b[mask_r]) / np.maximum(diff[mask_r], 1e-6)) % 6.0
        hue[mask_g] = ((b[mask_g] - r[mask_g]) / np.maximum(diff[mask_g], 1e-6)) + 2.0
        hue[mask_b] = ((r[mask_b] - g[mask_b]) / np.maximum(diff[mask_b], 1e-6)) + 4.0
        hue = (hue / 6.0).astype(np.float32)
        val = mx.astype(np.float32)
        return hue, sat, val

    @staticmethod
    def _candidate_from_region(name: str, color_ref: np.ndarray, weight: np.ndarray, u: np.ndarray, v: np.ndarray, score_map: np.ndarray) -> Optional[Dict[str, object]]:
        if float(weight.sum()) < 1e-5:
            return None
        w = np.clip(weight.astype(np.float32), 0.0, None)
        s = float(w.sum())
        cu = float((u * w).sum() / max(s, 1e-6))
        cv = float((v * w).sum() / max(s, 1e-6))
        h, ww = score_map.shape
        cx = int(np.clip(round(cu * max(ww - 1, 1)), 0, max(ww - 1, 0)))
        cy = int(np.clip(round(cv * max(h - 1, 1)), 0, max(h - 1, 0)))
        radius = max(4, int(round(min(h, ww) * 0.045)))
        color = BackgroundStudioLightExtractor._sample_patch_mean(color_ref, cx, cy, radius)
        hh, ss, vv = rgb_to_hsv_approx(color)
        return {
            'u': cu,
            'v': cv,
            'x': float(cx),
            'y': float(cy),
            'score': float((score_map * w).sum() / max(s, 1e-6)),
            'color': color.astype(np.float32),
            'chroma': float(np.linalg.norm(color - np.dot(color, LUMA))),
            'hue': float(hh),
            'sat': float(ss),
            'val': float(vv),
            'kind': name,
        }

    def _is_cool_hue(self, hue: float, sat: float) -> bool:
        return self.cool_hue_min <= float(hue) <= self.cool_hue_max and float(sat) > (0.14 if self.style_mode == 'neon' else 0.16)

    def _is_warm_hue(self, hue: float, sat: float) -> bool:
        return ((float(hue) <= self.warm_hue_lo) or (float(hue) >= self.warm_hue_hi)) and float(sat) > (0.10 if self.style_mode == 'neon' else 0.12)


    def _estimate_gradient_field(
        self,
        color_ref: np.ndarray,
        bg_smooth: np.ndarray,
        lum: np.ndarray,
        sat_map: np.ndarray,
    ) -> Dict[str, object]:
        """Fit a low-frequency 2D illumination field from the background.

        A large class of backgrounds (purple/blue gradient, sunset paper,
        smoky studio backdrop, LED wall, overcast sky) does not contain a clean
        point light.  Reducing such images to key/fill/rim loses the optical cue.
        This function estimates a continuous light ramp and stores enough data for
        the compositor to project that ramp onto the portrait.
        """
        h, w = lum.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)

        # Heavy blur isolates illumination from texture. Repeated small box blurs
        # are cheap and dependency-free.
        lf_lum = lum.astype(np.float32)
        lf_col = bg_smooth.astype(np.float32)
        for _ in range(7):
            lf_lum = box_blur_gray(lf_lum, passes=1)
            lf_col = box_blur_rgb(lf_col, passes=1)

        # Robust linear plane: L(u,v)=a*u+b*v+c.  b>0 means the lower image is brighter.
        X = np.stack([u.reshape(-1) - 0.5, v.reshape(-1) - 0.5, np.ones(h * w, dtype=np.float32)], axis=1)
        y = lf_lum.reshape(-1).astype(np.float32)
        try:
            coeff, *_ = np.linalg.lstsq(X.astype(np.float32), y, rcond=None)
        except Exception:
            coeff = np.array([0.0, 0.0, float(np.mean(y))], dtype=np.float32)
        a, b, c = [float(x) for x in coeff]

        p05 = float(np.percentile(lf_lum, 5.0))
        p50 = float(np.percentile(lf_lum, 50.0))
        p95 = float(np.percentile(lf_lum, 95.0))
        lum_span = max(p95 - p05, 1e-6)
        plane_strength = float(np.clip(np.sqrt(a * a + b * b) / max(p50 + 0.04, 1e-6), 0.0, 2.0))
        vertical_strength = float(np.clip(abs(b) / max(p50 + 0.04, 1e-6), 0.0, 2.0))
        horizontal_strength = float(np.clip(abs(a) / max(p50 + 0.04, 1e-6), 0.0, 2.0))

        def weighted_mean(mask_weight: np.ndarray) -> np.ndarray:
            mw = np.clip(mask_weight.astype(np.float32), 0.0, None)
            return self._weighted_region_mean(lf_col, mw)

        top_w = np.clip(1.0 - v / 0.42, 0.0, 1.0)
        bottom_w = np.clip((v - 0.58) / 0.42, 0.0, 1.0)
        left_w = np.clip(1.0 - u / 0.42, 0.0, 1.0)
        right_w = np.clip((u - 0.58) / 0.42, 0.0, 1.0)
        center_w = np.exp(-0.5 * (((u - 0.5) / 0.30) ** 2 + ((v - 0.52) / 0.34) ** 2)).astype(np.float32)

        # Weight colors slightly by luminance: a brighter colored part contributes
        # more as illumination, but saturation alone cannot dominate.
        lum_w = np.clip((lf_lum - p05) / lum_span, 0.0, 1.0)
        chroma_w = 0.72 + 0.28 * np.clip(sat_map, 0.0, 1.0)
        base_w = 0.42 + 0.88 * lum_w * chroma_w
        top_color = weighted_mean(top_w * base_w)
        bottom_color = weighted_mean(bottom_w * base_w)
        left_color = weighted_mean(left_w * base_w)
        right_color = weighted_mean(right_w * base_w)
        center_color = weighted_mean(center_w * base_w)

        top_luma = float(np.dot(top_color, LUMA))
        bottom_luma = float(np.dot(bottom_color, LUMA))
        left_luma = float(np.dot(left_color, LUMA))
        right_luma = float(np.dot(right_color, LUMA))
        center_luma = float(np.dot(center_color, LUMA))

        # Confidence is high for smooth directional ramps or meaningful low-frequency
        # contrast. It stays lower for flat backgrounds so the compositor falls back
        # to exposure lift rather than fake colored light.
        smooth_contrast = float(np.clip((p95 - p05) / max(p50 + 0.05, 1e-6), 0.0, 1.8))
        chroma_mean = float(np.mean(sat_map))
        confidence = float(np.clip(0.22 + 0.34 * plane_strength + 0.28 * smooth_contrast + 0.12 * chroma_mean, 0.18, 1.0))

        # The brightest side determines the dominant projected ramp.
        vertical_bias = bottom_luma - top_luma
        horizontal_bias = right_luma - left_luma
        if abs(vertical_bias) >= abs(horizontal_bias):
            dominant_axis = 'vertical'
            dominant_direction = 'bottom_to_top' if vertical_bias > 0 else 'top_to_bottom'
        else:
            dominant_axis = 'horizontal'
            dominant_direction = 'right_to_left' if horizontal_bias > 0 else 'left_to_right'

        # Store a compact low-frequency light field grid.  This is the key change
        # for complex backgrounds: the compositor can sample the whole spatial
        # color field instead of reducing the scene to only top/bottom or left/right.
        grid_h = 16
        grid_w = 16
        grid_colors = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
        grid_luma = np.zeros((grid_h, grid_w), dtype=np.float32)
        grid_sat = np.zeros((grid_h, grid_w), dtype=np.float32)
        for gy_i in range(grid_h):
            y0g = int(round(gy_i * h / grid_h))
            y1g = int(round((gy_i + 1) * h / grid_h))
            y1g = max(y1g, y0g + 1)
            for gx_i in range(grid_w):
                x0g = int(round(gx_i * w / grid_w))
                x1g = int(round((gx_i + 1) * w / grid_w))
                x1g = max(x1g, x0g + 1)
                patch_col = lf_col[y0g:y1g, x0g:x1g]
                patch_w = base_w[y0g:y1g, x0g:x1g]
                col = self._weighted_region_mean(patch_col, patch_w)
                grid_colors[gy_i, gx_i] = np.clip(col, 0.0, 4.0)
                grid_luma[gy_i, gx_i] = float(np.dot(col, LUMA))
                _, sat_c, _ = rgb_to_hsv_approx(col)
                grid_sat[gy_i, gx_i] = float(sat_c)

        local_contrast = float(np.clip((np.percentile(grid_luma, 95.0) - np.percentile(grid_luma, 5.0)) / max(float(np.percentile(grid_luma, 50.0)) + 0.05, 1e-6), 0.0, 2.0))
        colorfulness = float(np.clip(np.mean(grid_sat), 0.0, 1.0))

        return {
            'enabled': True,
            'a': a,
            'b': b,
            'c': c,
            'p05_luma': p05,
            'p50_luma': p50,
            'p95_luma': p95,
            'plane_strength': plane_strength,
            'vertical_strength': vertical_strength,
            'horizontal_strength': horizontal_strength,
            'confidence': confidence,
            'dominant_axis': dominant_axis,
            'dominant_direction': dominant_direction,
            'direction_convention': {
                'horizontal': 'right - left',
                'vertical': 'bottom - top',
            },
            'vertical_bias': float(vertical_bias),
            'horizontal_bias': float(horizontal_bias),
            'top_luma': top_luma,
            'bottom_luma': bottom_luma,
            'left_luma': left_luma,
            'right_luma': right_luma,
            'center_luma': center_luma,
            'top_color': [float(x) for x in np.clip(top_color, 0.0, 4.0)],
            'bottom_color': [float(x) for x in np.clip(bottom_color, 0.0, 4.0)],
            'left_color': [float(x) for x in np.clip(left_color, 0.0, 4.0)],
            'right_color': [float(x) for x in np.clip(right_color, 0.0, 4.0)],
            'center_color': [float(x) for x in np.clip(center_color, 0.0, 4.0)],
            'grid_w': int(grid_w),
            'grid_h': int(grid_h),
            'grid_colors': grid_colors.tolist(),
            'grid_luma': grid_luma.tolist(),
            'grid_sat': grid_sat.tolist(),
            'local_contrast': local_contrast,
            'colorfulness': colorfulness,
        }


    def _extract_background_driven_lighting(
        self,
        color_ref: np.ndarray,
        bg_smooth: np.ndarray,
        lum: np.ndarray,
        hue_map: np.ndarray,
        sat_map: np.ndarray,
        global_mean: np.ndarray,
        palette_diversity: float,
        hue_entropy: float,
        dominant_share: float,
        background_mode: str,
        neon_strength: str,
        camera_params: Optional[CameraParams] = None,
    ) -> LightingInfo:
        """Infer an optical key/fill/rim model from background structure.

        The older implementation used a weighted average of bright colorful regions.
        That tends to turn a red / blue / purple background into a global color wash
        on the subject. This version treats the background as an illumination field:
        it estimates *where* light comes from first, then assigns color and strength
        locally to a small number of directional lights.
        """
        h, w = lum.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)
        gradient_field = self._estimate_gradient_field(color_ref, bg_smooth, lum, sat_map)

        lum_f = box_blur_gray(lum.astype(np.float32), passes=3)
        p35 = float(np.percentile(lum_f, 35.0))
        p55 = float(np.percentile(lum_f, 55.0))
        p78 = float(np.percentile(lum_f, 78.0))
        p92 = float(np.percentile(lum_f, 92.0))
        p98 = float(np.percentile(lum_f, 98.0))
        lum_range = max(p98 - p55, 1e-6)

        # Structural cue: true illumination usually has a luminance ramp or highlight
        # boundary. Saturated but flat color should not become full-face lighting.
        gy, gx = np.gradient(lum_f)
        grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        grad_n = grad / max(float(np.percentile(grad, 96.0)), 1e-6)
        grad_n = np.clip(grad_n, 0.0, 1.0)

        high = np.clip((lum_f - p78) / max(p98 - p78, 1e-6), 0.0, 1.0)
        mid = np.clip((lum_f - p55) / max(p92 - p55, 1e-6), 0.0, 1.0)
        # Prefer spatially informative regions but do NOT hard-code top light.
        # Bottom gradients, floor LEDs, sunsets and stage uplights must remain valid.
        side_prior = 0.72 + 0.48 * np.clip(np.abs(u - 0.5) * 2.0, 0.0, 1.0)
        vertical_prior = 0.90 + 0.22 * np.clip(np.abs(v - 0.5) * 2.0, 0.0, 1.0)
        center_penalty = 1.0 - 0.18 * np.exp(-0.5 * (((u - 0.5) / 0.24) ** 2 + ((v - 0.50) / 0.32) ** 2)).astype(np.float32)
        chroma_gate = 0.86 + 0.28 * np.clip(sat_map, 0.0, 1.0)
        field_a = float(gradient_field.get('a', 0.0))
        field_b = float(gradient_field.get('b', 0.0))
        # Small prior toward the brightest end of the fitted light field.
        field_pred = field_a * (u - 0.5) + field_b * (v - 0.5)
        field_pred = np.clip((field_pred - float(field_pred.min())) / max(float(field_pred.max() - field_pred.min()), 1e-6), 0.0, 1.0)
        field_prior = 0.86 + 0.30 * field_pred
        source_score = (
            np.power(high, 2.00) * (0.78 + 0.45 * grad_n)
            + 0.26 * np.power(mid, 2.25) * grad_n
            + 0.16 * np.power(np.clip((lum_f - p55) / max(p98 - p55, 1e-6), 0.0, 1.0), 1.70) * field_pred
        ) * side_prior * vertical_prior * center_penalty * chroma_gate * field_prior

        # If the background has no real highlight, still infer a soft direction from
        # the luminance field, but lower confidence to avoid obvious fake lighting.
        fallback_used = False
        if float(source_score.sum()) < 1e-5:
            fallback_used = True
            source_score = np.power(np.clip(lum_f / max(p98, 1e-6), 0.0, 1.4), 2.2) * side_prior * vertical_prior * field_prior

        def region_weight(cu0: float, cv0: float, su: float, sv: float) -> np.ndarray:
            return np.exp(-0.5 * (((u - cu0) / max(su, 1e-3)) ** 2 + ((v - cv0) / max(sv, 1e-3)) ** 2)).astype(np.float32)

        # Sector candidates preserve direction: split backgrounds can produce warm
        # light from one side and cool rim/fill from the other instead of averaging.
        sectors = [
            ('left', 0.04, 0.52, 0.18, 0.42),
            ('right', 0.96, 0.52, 0.18, 0.42),
            ('top_left', 0.18, 0.14, 0.22, 0.20),
            ('top_right', 0.82, 0.14, 0.22, 0.20),
            ('bottom_left', 0.18, 0.86, 0.24, 0.20),
            ('bottom_right', 0.82, 0.86, 0.24, 0.20),
            ('upper', 0.50, 0.08, 0.36, 0.18),
            ('lower', 0.50, 0.92, 0.38, 0.18),
            ('center_left', 0.22, 0.48, 0.22, 0.28),
            ('center_right', 0.78, 0.48, 0.22, 0.28),
        ]
        candidates: List[Dict[str, object]] = []
        for name, cu0, cv0, su, sv in sectors:
            rw = region_weight(cu0, cv0, su, sv)
            weight = source_score * rw
            power = float(weight.sum())
            if power <= max(float(source_score.sum()) * 0.025, 1e-6):
                continue
            cand = self._candidate_from_region('optical_' + name, color_ref, weight, u, v, source_score)
            if cand is None:
                continue
            cand['power'] = power
            cand['score'] = power * (0.75 + 0.25 * float(cand.get('sat', 0.0)))
            candidates.append(cand)

        if not candidates:
            cand = self._candidate_from_region('optical_global', color_ref, source_score, u, v, source_score)
            if cand is not None:
                cand['power'] = float(source_score.sum())
                cand['score'] = float(source_score.sum())
                candidates.append(cand)

        candidates.sort(key=lambda c: float(c.get('score', 0.0)), reverse=True)
        key_cand = candidates[0] if candidates else {
            'u': 0.35, 'v': 0.28, 'color': global_mean, 'power': 1.0, 'score': 1.0, 'kind': 'fallback'
        }
        total_power = max(float(source_score.sum()), 1e-6)
        key_power = float(key_cand.get('power', key_cand.get('score', 1.0)))
        key_u = float(key_cand['u'])
        key_v = float(key_cand['v'])
        key_dir = self._uv_to_direction(key_u, key_v, lum.shape, camera_params=camera_params)

        # Direction confidence controls how hard the shadows are. Large colorful but
        # flat regions get softer, weaker diffuse and stronger preservation of source.
        high_area = float(np.mean(high > 0.45))
        contrast = float(np.clip((p98 - p35) / max(p98, 1e-6), 0.0, 1.0))
        dir_conf = float(np.clip(0.35 + 0.42 * contrast + 0.18 * min(key_power / total_power, 1.0) + 0.15 * (1.0 - min(high_area / 0.55, 1.0)), 0.28, 1.0))
        if fallback_used:
            dir_conf *= 0.62

        key_color_raw = np.array(key_cand.get('color', global_mean), dtype=np.float32)
        # Stage38: keep the cyber/neon hue in the portrait light.  Stage37 fixed
        # the dark face by over-neutralizing the key, but that made skin look white
        # and removed the background color.  Here the key is still lifted for
        # readability, while only a small neutral component is mixed in.  Saturated
        # background color remains visible on face-side, hair, rim and specular.
        key_desat = 0.16 if self.style_mode == 'neon' else 0.38
        neutral_mix = 0.025 if self.style_mode == 'neon' else 0.085
        key_floor = 0.285 if self.style_mode == 'neon' else 0.325
        key_color = desaturate_color(key_color_raw, key_desat)
        neutral_key = np.array([key_floor, key_floor, key_floor], dtype=np.float32)
        key_color = key_color * (1.0 - neutral_mix) + neutral_key * neutral_mix
        key_color = brighten_preserve_hue(key_color, max(float(np.dot(key_color, LUMA)), key_floor + 0.020 * dir_conf))
        key_intensity = float(np.clip(1.02 + 0.92 * dir_conf + 0.22 * contrast + 0.08 * high_area, 0.98, 1.88))

        # Ambient is a low-frequency, non-highlight estimate.  It is moderated,
        # not destroyed: the portrait should still inherit the scene palette.
        non_source = np.clip(1.0 - np.power(high, 0.75) * 0.88, 0.0, 1.0)
        non_source *= 0.72 + 0.28 * np.clip(1.0 - grad_n, 0.0, 1.0)
        ambient_raw = self._weighted_region_mean(bg_smooth, non_source)
        ambient_color = desaturate_color(ambient_raw, 0.34 if self.style_mode == 'neon' else 0.46)
        ambient_color = brighten_preserve_hue(ambient_color, max(float(np.dot(ambient_color, LUMA)), 0.095 + 0.045 * (1.0 - contrast)))
        ambient_intensity = float(np.clip(0.065 + 0.095 * p55 + 0.045 * (1.0 - dir_conf), 0.060, 0.155))

        fill_dir = safe_norm(np.array([-0.42 * float(key_dir[0]), 0.05, 0.91], dtype=np.float32))
        fill_color = brighten_preserve_hue(0.68 * ambient_color + 0.32 * desaturate_color(key_color, 0.18 if self.style_mode == 'neon' else 0.32), max(float(np.dot(ambient_color, LUMA)), 0.12))
        fill_intensity = float(np.clip(0.26 + 0.22 * (1.0 - dir_conf), 0.22, 0.46))

        lights: List[PortraitLight] = []
        lights.append(PortraitLight(
            name='optical_key_from_directional_field',
            direction=tuple(float(x) for x in key_dir),
            color=tuple(float(x) for x in np.clip(key_color, 0.0, 4.0)),
            intensity=key_intensity,
            size=float(0.18 + 0.22 * (1.0 - dir_conf)),
            diffuse_scale=float(1.02 + 0.32 * dir_conf),
            specular_scale=float(0.70 + 0.34 * dir_conf),
            rim_scale=0.12,
        ))
        lights.append(PortraitLight(
            name='optical_fill_from_low_frequency_bg',
            direction=tuple(float(x) for x in fill_dir),
            color=tuple(float(x) for x in np.clip(fill_color, 0.0, 4.0)),
            intensity=fill_intensity,
            size=0.55,
            diffuse_scale=0.16,
            specular_scale=0.08,
            rim_scale=0.02,
        ))

        # Add at most two local side/rim lights that are spatially separated from key.
        side_added = 0
        used_sides = {'left' if key_u < 0.5 else 'right'}
        for cand in candidates[1:]:
            cu2 = float(cand['u'])
            cv2 = float(cand['v'])
            spatial_sep = abs(cu2 - key_u) + 0.35 * abs(cv2 - key_v)
            if spatial_sep < 0.32:
                continue
            side_name = 'left' if cu2 < 0.5 else 'right'
            if side_name in used_sides and side_added > 0:
                continue
            power_ratio = float(cand.get('power', cand.get('score', 0.0))) / total_power
            if power_ratio < 0.055 and side_added > 0:
                continue
            rim_dir = self._uv_to_direction(cu2, cv2, lum.shape, camera_params=camera_params)
            rim_raw = np.array(cand.get('color', global_mean), dtype=np.float32)
            rim_color = saturate_color(rim_raw, 1.05 if self.style_mode == 'neon' else 0.96)
            rim_color = brighten_preserve_hue(rim_color, max(float(np.dot(rim_color, LUMA)), 0.16 + 0.06 * dir_conf))
            rim_strength = float(np.clip(0.42 + 1.80 * power_ratio + 0.18 * float(cand.get('sat', 0.0)), 0.32, 1.18))
            lights.append(PortraitLight(
                name=f'optical_local_{side_name}_rim_fill',
                direction=tuple(float(x) for x in rim_dir),
                color=tuple(float(x) for x in np.clip(rim_color, 0.0, 4.0)),
                intensity=rim_strength,
                size=0.22,
                diffuse_scale=0.085,
                specular_scale=0.34,
                rim_scale=1.55,
            ))
            used_sides.add(side_name)
            side_added += 1
            if side_added >= 2:
                break

        if side_added == 0:
            # Keep a subtle opposite rim so the subject is grounded in the scene, but
            # never strong enough to repaint the face.
            fallback_rim_dir = safe_norm(np.array([0.78 if float(key_dir[0]) < 0 else -0.78, 0.02, 0.62], dtype=np.float32))
            fallback_rim_color = brighten_preserve_hue(0.55 * key_color + 0.45 * ambient_color, max(float(np.dot(key_color, LUMA)), 0.15))
            lights.append(PortraitLight(
                name='optical_opposite_subtle_rim',
                direction=tuple(float(x) for x in fallback_rim_dir),
                color=tuple(float(x) for x in np.clip(fallback_rim_color, 0.0, 4.0)),
                intensity=0.34 + 0.22 * dir_conf,
                size=0.30,
                diffuse_scale=0.045,
                specular_scale=0.18,
                rim_scale=1.15,
            ))

        palette_points = []
        for idx, light in enumerate(lights):
            d = np.array(light.direction, dtype=np.float32)
            azimuth = float(np.degrees(np.arctan2(float(d[0]), float(d[2]))))
            elevation = float(np.degrees(np.arcsin(float(np.clip(d[1], -1.0, 1.0)))))
            palette_points.append({
                'name': light.name,
                'kind': 'optical_directional' if idx == 0 else 'optical_fill_or_rim',
                'u': key_u if idx == 0 else (0.5 + 0.5 * float(np.sign(d[0]))),
                'v': key_v if idx == 0 else 0.50,
                'score': float(light.intensity),
                'direction_azimuth_deg': azimuth,
                'direction_elevation_deg': elevation,
                'direction_confidence': float(dir_conf),
                'color': [float(x) for x in light.color],
            })

        # Save local bright/color sources for the compositor.  This prevents complex
        # scenes from collapsing into one or two average colors.  We combine regional
        # sector candidates with true local peaks from the source score map.
        source_peaks = []
        merged_peak_candidates = list(candidates[:6])
        try:
            merged_peak_candidates += self._find_peak_candidates(color_ref, np.clip(source_score * (0.70 + 0.30 * field_prior), 0.0, None), sat_map)[:6]
        except Exception:
            pass
        dedup_local = []
        for cand in merged_peak_candidates:
            cu = float(cand.get('u', 0.5)); cv = float(cand.get('v', 0.5))
            keep = True
            for prev in dedup_local:
                if abs(cu - float(prev.get('u', 0.5))) + abs(cv - float(prev.get('v', 0.5))) < 0.18:
                    keep = False
                    break
            if keep:
                dedup_local.append(cand)
            if len(dedup_local) >= 8:
                break
        for cand in dedup_local[:8]:
            col = np.array(cand.get('color', global_mean), dtype=np.float32)
            lum_c = float(np.dot(col, LUMA))
            source_peaks.append({
                'u': float(cand.get('u', 0.5)),
                'v': float(cand.get('v', 0.5)),
                'score': float(cand.get('score', cand.get('power', 1.0))),
                'power': float(cand.get('power', cand.get('score', 1.0))),
                'sat': float(cand.get('sat', 0.0)),
                'luma': lum_c,
                'kind': str(cand.get('kind', 'source')),
                'color': [float(x) for x in np.clip(col, 0.0, 4.0)],
            })
        try:
            gradient_field['source_peaks'] = source_peaks
            gradient_field['direction_confidence'] = float(dir_conf)
            gradient_field['key_uv'] = [float(key_u), float(key_v)]
            gradient_field['high_area'] = float(high_area)
            gradient_field['contrast'] = float(contrast)
        except Exception:
            pass

        # Keep original mode labels for downstream preset moderation, but expose the
        # optical analysis in palette_points and light names.
        return LightingInfo(
            ambient_color=tuple(float(x) for x in np.clip(ambient_color, 0.0, 4.0)),
            ambient_intensity=ambient_intensity,
            key_color=tuple(float(x) for x in np.clip(key_color, 0.0, 4.0)),
            key_intensity=key_intensity,
            lights=[asdict(light) for light in lights],
            global_mean_color=tuple(float(x) for x in np.clip(desaturate_color(global_mean, 0.60), 0.0, 4.0)),
            palette_points=palette_points,
            palette_diversity=float(palette_diversity),
            hue_entropy=float(hue_entropy),
            dominant_hue_share=float(dominant_share),
            adaptive_light_count=int(len(lights)),
            background_mode=str(background_mode),
            neon_strength=str(neon_strength),
            gradient_field=gradient_field,
        )

    @staticmethod
    def _uv_to_direction(u: float, v: float, size_hw: Tuple[int, int], camera_params: Optional[CameraParams] = None) -> np.ndarray:
        h, w = size_hw
        scaled = camera_params.scaled_intrinsics((h, w)) if camera_params is not None else None
        if scaled is not None:
            fx, fy, cx, cy = scaled
            px = u * max(w - 1, 1)
            py = v * max(h - 1, 1)
            x = (px - float(cx)) / max(float(fx), 1e-6)
            y = -(py - float(cy)) / max(float(fy), 1e-6)
            z = 1.0
            return safe_norm(np.array([x, y, z], dtype=np.float32))
        x = (u - 0.5) * 1.85
        y = (0.50 - v) * 1.28
        z = 0.72 + 0.40 * (1.0 - min(abs(u - 0.5) * 1.7, 1.0))
        return safe_norm(np.array([x, y, z], dtype=np.float32))

    def _find_peak_candidates(self, color_ref: np.ndarray, score_map: np.ndarray, sat_map: np.ndarray) -> List[Dict[str, object]]:
        h, w = score_map.shape
        work = score_map.copy().astype(np.float32)
        threshold = float(np.percentile(score_map, 70.0))
        suppress_radius = max(6, int(round(min(h, w) * 0.10)))
        patch_radius = max(4, int(round(min(h, w) * 0.045)))
        out: List[Dict[str, object]] = []
        for _ in range(self.max_candidates * 2):
            idx = int(np.argmax(work))
            peak_score = float(work.reshape(-1)[idx])
            if peak_score <= threshold:
                break
            cy, cx = divmod(idx, w)
            color = self._sample_patch_mean(color_ref, cx, cy, patch_radius)
            hue, sat, val = rgb_to_hsv_approx(color)
            out.append({
                'u': float(cx / max(w - 1, 1)),
                'v': float(cy / max(h - 1, 1)),
                'x': float(cx),
                'y': float(cy),
                'score': float(peak_score * (0.55 + 0.90 * sat + 0.35 * float(sat_map[cy, cx]))),
                'color': color.astype(np.float32),
                'chroma': float(np.linalg.norm(color - np.dot(color, LUMA))),
                'hue': float(hue),
                'sat': float(sat),
                'val': float(val),
                'kind': 'peak',
            })
            y0 = max(0, cy - suppress_radius)
            y1 = min(h, cy + suppress_radius + 1)
            x0 = max(0, cx - suppress_radius)
            x1 = min(w, cx + suppress_radius + 1)
            yy2, xx2 = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d2 = (xx2 - float(cx)) ** 2 + (yy2 - float(cy)) ** 2
            suppress = np.exp(-d2 / max(2.0 * (suppress_radius * 0.72) ** 2, 1e-5)).astype(np.float32)
            work[y0:y1, x0:x1] *= (1.0 - suppress)
        out.sort(key=lambda p: (float(p['score']) + 0.40 * float(p['sat'])), reverse=True)
        return out[: self.max_candidates]


    def _estimate_palette_statistics(self, hue_map: np.ndarray, sat_map: np.ndarray, score_map: np.ndarray) -> Dict[str, float]:
        weight = np.clip(score_map.astype(np.float32), 0.0, None) * (0.20 + 0.80 * np.clip(sat_map.astype(np.float32), 0.0, 1.0))
        total = float(weight.sum())
        if total <= 1e-6:
            return {
                'hue_entropy': 0.0,
                'dominant_share': 1.0,
                'mean_sat': 0.0,
                'palette_diversity': 0.0,
                'occupied_bins': 1.0,
            }
        bins = 16
        hist = np.zeros((bins,), dtype=np.float32)
        hue_idx = np.floor(np.clip(hue_map, 0.0, 0.9999) * bins).astype(np.int32)
        for i in range(bins):
            hist[i] = float(weight[hue_idx == i].sum())
        probs = hist / max(float(hist.sum()), 1e-6)
        valid = probs > 1e-8
        hue_entropy = float(-(probs[valid] * np.log(probs[valid])).sum() / np.log(bins)) if np.any(valid) else 0.0
        dominant_share = float(probs.max()) if probs.size else 1.0
        occupied_bins = float(np.count_nonzero(probs > 0.03)) / bins
        mean_sat = float((sat_map * weight).sum() / max(total, 1e-6))
        palette_diversity = float(np.clip(0.52 * hue_entropy + 0.18 * occupied_bins + 0.20 * mean_sat + 0.10 * (1.0 - dominant_share), 0.0, 1.0))
        return {
            'hue_entropy': hue_entropy,
            'dominant_share': dominant_share,
            'mean_sat': mean_sat,
            'palette_diversity': palette_diversity,
            'occupied_bins': occupied_bins,
        }

    def _classify_background_mode(self, palette_diversity: float, hue_entropy: float, dominant_share: float, cool_presence: float, warm_presence: float) -> str:
        if dominant_share >= self.monochrome_dominant_share or palette_diversity <= self.monochrome_diversity_threshold or hue_entropy <= 0.34:
            return 'monotone'
        if self.style_mode == 'neon' and palette_diversity >= self.rich_diversity_threshold and cool_presence >= 0.08 and warm_presence >= 0.08:
            return 'rich'
        if palette_diversity >= self.rich_diversity_threshold + 0.06 and (cool_presence >= 0.05 or warm_presence >= 0.14):
            return 'rich'
        return 'balanced'


    def extract(self, background_linear: np.ndarray, camera_params: Optional[CameraParams] = None) -> LightingInfo:
        bg = np.clip(background_linear.astype(np.float32), 0.0, None)
        h0, w0 = bg.shape[:2]
        scale = min(1.0, 448.0 / max(h0, w0))
        if scale < 1.0:
            nh = max(64, int(round(h0 * scale)))
            nw = max(64, int(round(w0 * scale)))
            bg_small = np.asarray(
                Image.fromarray(np.clip(linear_to_srgb(bg) * 255.0 + 0.5, 0, 255).astype(np.uint8)).resize((nw, nh), Image.Resampling.LANCZOS),
                dtype=np.float32,
            ) / 255.0
            bg_small = srgb_to_linear(bg_small)
        else:
            bg_small = bg

        color_ref = box_blur_rgb(bg_small, passes=1)
        bg_smooth = box_blur_rgb(bg_small, passes=3)
        lum = rgb_luminance(bg_smooth)
        hue_map, sat_map, val_map = self._rgb_to_hsv_image(np.clip(color_ref, 0.0, None))
        lum_n = lum / max(float(np.percentile(lum, 98.0)), 1e-6)
        val_n = val_map / max(float(np.percentile(val_map, 98.0)), 1e-6)
        chroma_score = np.power(np.clip(sat_map, 0.0, 1.0), 0.72 if self.style_mode == 'neon' else 0.75)
        score_map = (0.34 * np.sqrt(np.clip(lum_n, 0.0, 2.0)) + 0.66 * np.sqrt(np.clip(val_n, 0.0, 2.0))) * (0.28 + (1.85 if self.style_mode == 'neon' else 1.45) * chroma_score)
        global_mean = np.mean(color_ref.reshape(-1, 3), axis=0).astype(np.float32)

        cool_mask = (hue_map >= self.cool_hue_min) & (hue_map <= self.cool_hue_max) & (sat_map > 0.14)
        warm_mask = ((hue_map <= self.warm_hue_lo) | (hue_map >= self.warm_hue_hi)) & (sat_map > 0.10)
        cool_presence = float(score_map[cool_mask].sum() / max(float(score_map.sum()), 1e-6)) if np.any(cool_mask) else 0.0
        warm_presence = float(score_map[warm_mask].sum() / max(float(score_map.sum()), 1e-6)) if np.any(warm_mask) else 0.0
        palette_stats = self._estimate_palette_statistics(hue_map, sat_map, score_map)
        palette_diversity = float(palette_stats['palette_diversity'])
        hue_entropy = float(palette_stats['hue_entropy'])
        dominant_share = float(palette_stats['dominant_share'])
        background_mode = self._classify_background_mode(palette_diversity, hue_entropy, dominant_share, cool_presence, warm_presence)

        neon_strength = 'off'
        if self.style_mode == 'neon':
            if background_mode == 'rich' and palette_diversity >= self.rich_diversity_threshold and cool_presence >= 0.10 and warm_presence >= 0.10:
                neon_strength = 'strong'
            elif background_mode != 'monotone' and ((cool_presence >= 0.05 and warm_presence >= 0.04) or palette_diversity >= self.rich_diversity_threshold - 0.02):
                neon_strength = 'soft'

        # Background-driven light inversion: directly infer key/fill/rim from the background.
        return self._extract_background_driven_lighting(
            color_ref=color_ref,
            bg_smooth=bg_smooth,
            lum=lum,
            hue_map=hue_map,
            sat_map=sat_map,
            global_mean=global_mean,
            palette_diversity=palette_diversity,
            hue_entropy=hue_entropy,
            dominant_share=dominant_share,
            background_mode=background_mode,
            neon_strength=neon_strength,
            camera_params=camera_params,
        )


    @staticmethod
    def direction_to_latlong_uv(direction: np.ndarray) -> Tuple[float, float]:
        d = safe_norm(direction.astype(np.float32))
        phi = np.arctan2(float(d[0]), float(d[2]))
        theta = np.arccos(float(np.clip(d[1], -1.0, 1.0)))
        u = (phi / np.pi + 1.0) * 0.5
        v = theta / np.pi
        return float(u), float(v)

    def save_hdri_preview(self, lighting: LightingInfo, filename: str, width: int = 1024, height: int = 512) -> None:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        uu = xx / max(width - 1, 1)
        vv = yy / max(height - 1, 1)
        base_ambient_luma = max(float(np.dot(np.array(lighting.ambient_color, dtype=np.float32), LUMA)) * float(lighting.ambient_intensity), 0.015)
        pano = np.ones((height, width, 3), dtype=np.float32) * base_ambient_luma * 0.20
        for light_dict in lighting.lights:
            light = PortraitLight(**light_dict)
            cu, cv = self.direction_to_latlong_uv(np.array(light.direction, dtype=np.float32))
            du = np.minimum(np.abs(uu - cu), 1.0 - np.abs(uu - cu))
            dv = np.abs(vv - cv)
            sigma_u = max(light.size * 0.050, 0.017)
            sigma_v = max(light.size * 0.070, 0.020)
            blob = np.exp(-0.5 * ((du / sigma_u) ** 2 + (dv / sigma_v) ** 2)).astype(np.float32)
            color = np.array(light.color, dtype=np.float32) * float(light.intensity)
            pano += blob[..., None] * color[None, None, :] * (0.70 + 0.30 * light.specular_scale)
        pano = pano / max(np.percentile(pano, 99.5), 1e-6)
        pano = np.power(np.clip(pano, 0.0, 1.0), 0.92)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        Image.fromarray((pano * 255.0 + 0.5).astype(np.uint8)).save(filename)
