from __future__ import annotations

from typing import Dict, List
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

class RendererVirtualLightsMixin:
    def _select_lighting_pattern(self, lighting_info: LightingInfo) -> str:
        requested = str(getattr(self, 'lighting_pattern', 'auto') or 'auto').lower()
        if requested != 'auto':
            return requested
        lights = getattr(lighting_info, 'lights', None) or []
        if not lights:
            return 'natural'
        try:
            key_dir = safe_norm(np.array(lights[0]['direction'], dtype=np.float32))
        except Exception:
            return 'natural'
        side = abs(float(key_dir[0]))
        top = max(0.0, float(-key_dir[1]))
        if (not self._using_continuous_policy()) and self.style_mode == 'neon':
            return 'split'
        if side > 0.56 and top > 0.18:
            return 'cinematic'
        if top > 0.42 and side < 0.38:
            return 'top'
        if side > 0.60:
            return 'side'
        if side > 0.36 and top > 0.10:
            return 'rembrandt'
        return 'natural'


    def _build_subject_region_masks(self, P: np.ndarray, subject_mask: np.ndarray) -> Dict[str, np.ndarray]:
        x = P[..., 0].astype(np.float32)
        y = P[..., 1].astype(np.float32)
        if not np.any(subject_mask > 0.08):
            ones = np.ones_like(subject_mask, dtype=np.float32)
            zeros = np.zeros_like(subject_mask, dtype=np.float32)
            return {'xn': zeros, 'yn': zeros, 'center': ones, 'upper': zeros, 'lower': zeros, 'edge': zeros}
        scale_x = max(float(np.percentile(np.abs(x[subject_mask > 0.08]), 88.0)), 1e-4)
        scale_y = max(float(np.percentile(np.abs(y[subject_mask > 0.08]), 88.0)), 1e-4)
        xn = np.clip(x / scale_x, -1.0, 1.0)
        yn = np.clip(y / scale_y, -1.0, 1.0)
        center = np.clip(1.0 - np.abs(xn), 0.0, 1.0) * subject_mask
        upper = np.clip((0.10 - yn) / 1.05, 0.0, 1.0) * subject_mask
        lower = np.clip((yn + 0.06) / 1.02, 0.0, 1.0) * subject_mask
        edge = np.power(np.clip(np.abs(xn), 0.0, 1.0), 0.9) * subject_mask
        return {'xn': xn.astype(np.float32), 'yn': yn.astype(np.float32), 'center': center.astype(np.float32), 'upper': upper.astype(np.float32), 'lower': lower.astype(np.float32), 'edge': edge.astype(np.float32)}


    def _resolve_key_side_sign(self, key_dir: np.ndarray) -> float:
        side = str(getattr(self, 'key_side', 'auto') or 'auto').lower()
        if side in ('left', 'l'):
            return -1.0
        if side in ('right', 'r'):
            return 1.0
        x = float(key_dir[0]) if key_dir is not None and len(key_dir) >= 1 else 0.0
        if abs(x) < 0.08:
            return -1.0
        return 1.0 if x > 0.0 else -1.0


    def _forced_pattern_direction(self, pattern: str, key_dir: np.ndarray) -> np.ndarray:
        """Build a clearly directional virtual key light.

        Stage11 still kept too much Z/front component, so the normal map could not
        visibly sculpt the portrait.  This version uses more side/top energy while
        staying diffuse-oriented; specular is handled separately and remains low.
        """
        side = self._resolve_key_side_sign(key_dir)
        pattern = str(pattern or 'natural').lower()
        if pattern == 'side':
            d = np.array([side * 0.62, 0.12, 0.78], dtype=np.float32)
        elif pattern == 'top':
            d = np.array([side * 0.14, 0.76, 0.64], dtype=np.float32)
        elif pattern == 'cinematic':
            d = np.array([side * 0.50, 0.24, 0.83], dtype=np.float32)
        elif pattern == 'rembrandt':
            d = np.array([side * 0.46, 0.20, 0.86], dtype=np.float32)
        elif pattern == 'split':
            d = np.array([side * 0.56, 0.10, 0.82], dtype=np.float32)
        elif pattern == 'natural':
            # Natural is still directional enough to show cheek/nose/shoulder
            # normals.  It is no longer a mostly frontal beauty light.
            d = np.array([side * 0.42, 0.18, 0.89], dtype=np.float32)
        else:
            d = np.array(key_dir, dtype=np.float32)
        return safe_norm(d)


    def _pick_virtual_rim_color(self, lighting_info: LightingInfo, key_color: np.ndarray, ambient_color: np.ndarray) -> np.ndarray:
        candidates = []
        for light_dict in getattr(lighting_info, 'lights', []) or []:
            try:
                c = np.array(light_dict.get('color', key_color), dtype=np.float32)
                inten = float(light_dict.get('intensity', 1.0))
                sat = float(rgb_to_hsv_approx(c)[1])
                score = sat * 0.75 + inten * 0.25
                candidates.append((score, c))
            except Exception:
                continue
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            rim = candidates[0][1].astype(np.float32)
        else:
            rim = (0.60 * key_color + 0.40 * ambient_color).astype(np.float32)
        _creative_neon = (self.style_mode == 'neon') and (not self._using_continuous_policy())
        rim = saturate_color(rim, 1.02 if _creative_neon else 0.92)
        rim = brighten_preserve_hue(rim, max(float(np.dot(rim, LUMA)), 0.16))
        return np.clip(rim, 0.0, 4.0).astype(np.float32)


    def _build_virtual_background_referenced_lights(self, lighting_info: LightingInfo) -> List[PortraitLight]:
        """Use virtual portrait-light directions, but keep color/intensity grounded in the background.

        This is the hybrid mode requested by the user: key/fill/rim directions are
        stable, photographer-like virtual lights; background analysis contributes
        color palette and approximate energy only.
        """
        bg_lights = []
        for d in getattr(lighting_info, 'lights', []) or []:
            try:
                bg_lights.append(PortraitLight(**d))
            except Exception:
                continue

        if bg_lights:
            raw_key_dir = safe_norm(np.array(bg_lights[0].direction, dtype=np.float32))
        else:
            raw_key_dir = np.array([-0.52, -0.22, 0.82], dtype=np.float32)
        pattern = self._select_lighting_pattern(lighting_info)
        self._last_selected_lighting_pattern = pattern
        key_dir = self._forced_pattern_direction(pattern, raw_key_dir)
        side_sign = self._resolve_key_side_sign(key_dir)

        raw_virtual_key = np.array(lighting_info.key_color, dtype=np.float32)
        _creative_neon = (self.style_mode == 'neon') and (not self._using_continuous_policy())
        key_floor = 0.31 if _creative_neon else 0.35
        key_color = desaturate_color(raw_virtual_key, 0.38 if _creative_neon else 0.50)
        key_color = key_color * 0.86 + np.array([key_floor, key_floor, key_floor], dtype=np.float32) * 0.14
        key_color = brighten_preserve_hue(key_color, max(float(np.dot(key_color, LUMA)), key_floor))
        ambient_color = brighten_preserve_hue(desaturate_color(np.array(lighting_info.ambient_color, dtype=np.float32), 0.58), max(float(np.dot(np.array(lighting_info.ambient_color, dtype=np.float32), LUMA)), 0.12))
        global_color = brighten_preserve_hue(desaturate_color(np.array(lighting_info.global_mean_color, dtype=np.float32), 0.52), max(float(np.dot(np.array(lighting_info.global_mean_color, dtype=np.float32), LUMA)), 0.14))
        fill_color = np.clip(0.65 * ambient_color + 0.35 * desaturate_color(key_color, 0.35), 0.0, 4.0).astype(np.float32)
        rim_color = self._pick_virtual_rim_color(lighting_info, raw_virtual_key, ambient_color)

        diversity = float(np.clip(getattr(lighting_info, 'palette_diversity', 0.35), 0.0, 1.0))
        key_intensity = float(np.clip(0.70 + 0.45 * float(getattr(lighting_info, 'key_intensity', 1.0)), 0.84, 1.38))
        fill_intensity = float(np.clip(0.30 + 1.95 * float(getattr(lighting_info, 'ambient_intensity', 0.06)), 0.36, 0.72))
        rim_intensity = float(np.clip(0.42 + 0.28 * key_intensity + 0.34 * diversity, 0.44, 1.20))

        pattern = str(pattern or 'natural').lower()
        if pattern == 'top':
            fill_dir = safe_norm(np.array([-side_sign * 0.16, 0.18, 0.97], dtype=np.float32))
            rim_dir = safe_norm(np.array([-side_sign * 0.76, 0.22, 0.61], dtype=np.float32))
            key_size = 0.24
        elif pattern == 'side':
            fill_dir = safe_norm(np.array([-side_sign * 0.38, 0.04, 0.92], dtype=np.float32))
            rim_dir = safe_norm(np.array([-side_sign * 0.94, 0.10, 0.33], dtype=np.float32))
            key_size = 0.18
        elif pattern == 'cinematic':
            fill_dir = safe_norm(np.array([-side_sign * 0.26, 0.06, 0.96], dtype=np.float32))
            rim_dir = safe_norm(np.array([-side_sign * 0.90, 0.18, 0.39], dtype=np.float32))
            key_size = 0.18
        elif pattern == 'rembrandt':
            fill_dir = safe_norm(np.array([-side_sign * 0.22, 0.06, 0.97], dtype=np.float32))
            rim_dir = safe_norm(np.array([-side_sign * 0.86, 0.16, 0.48], dtype=np.float32))
            key_size = 0.25
        elif pattern == 'split':
            fill_dir = safe_norm(np.array([-side_sign * 0.70, 0.12, 0.70], dtype=np.float32))
            rim_dir = safe_norm(np.array([-side_sign * 0.96, 0.08, 0.28], dtype=np.float32))
            key_size = 0.23
        else:  # natural
            fill_dir = safe_norm(np.array([-side_sign * 0.30, 0.06, 0.95], dtype=np.float32))
            rim_dir = safe_norm(np.array([-side_sign * 0.82, 0.12, 0.56], dtype=np.float32))
            key_size = 0.26

        virtual_lights: List[PortraitLight] = [
            PortraitLight(
                name='virtual_key_from_background_palette',
                direction=tuple(float(x) for x in key_dir),
                color=tuple(float(x) for x in np.clip(key_color, 0.0, 4.0)),
                intensity=key_intensity,
                size=float(key_size),
                diffuse_scale=1.00 if pattern in ('cinematic', 'rembrandt', 'side', 'top') else 0.92,
                specular_scale=0.16 if pattern in ('cinematic', 'split', 'side') else 0.12,
                rim_scale=0.14,
            ),
            PortraitLight(
                name='virtual_fill_from_background_ambient',
                direction=tuple(float(x) for x in fill_dir),
                color=tuple(float(x) for x in np.clip(fill_color, 0.0, 4.0)),
                intensity=fill_intensity,
                size=0.52,
                diffuse_scale=0.20,
                specular_scale=0.07,
                rim_scale=0.02,
            ),
            PortraitLight(
                name='virtual_rim_from_background_accent',
                direction=tuple(float(x) for x in rim_dir),
                color=tuple(float(x) for x in np.clip(rim_color, 0.0, 4.0)),
                intensity=rim_intensity,
                size=0.24,
                diffuse_scale=0.045,
                specular_scale=0.18,
                rim_scale=1.85,
            ),
        ]

        # For split / neon, add a small opposite kicker so both sides read clearly.
        if pattern == 'split' or ((self.style_mode == 'neon') and (not self._using_continuous_policy())):
            alt_color = np.clip(0.55 * global_color + 0.45 * ambient_color, 0.0, 4.0).astype(np.float32)
            alt_dir = safe_norm(np.array([side_sign * 0.80, -0.04, 0.60], dtype=np.float32))
            virtual_lights.append(PortraitLight(
                name='virtual_kicker_from_background_palette',
                direction=tuple(float(x) for x in alt_dir),
                color=tuple(float(x) for x in alt_color),
                intensity=float(np.clip(0.22 + 0.18 * diversity, 0.18, 0.48)),
                size=0.28,
                diffuse_scale=0.05,
                specular_scale=0.08,
                rim_scale=0.58,
            ))

        return virtual_lights


    def _resolve_render_lights(self, lighting_info: LightingInfo) -> List[PortraitLight]:
        mode = str(getattr(self, 'light_source_mode', 'hybrid') or 'hybrid').lower()
        if mode == 'background':
            out = []
            for d in getattr(lighting_info, 'lights', []) or []:
                try:
                    out.append(PortraitLight(**d))
                except Exception:
                    continue
            return out
        return self._build_virtual_background_referenced_lights(lighting_info)


    def _estimate_portrait_orientation_from_normals(
        self,
        N: np.ndarray,
        P: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
    ) -> Dict[str, float]:
        """Estimate portrait yaw/pitch from the normal field and face-space asymmetry.

        Mean normal alone is often almost zero for a frontal face, so stage11 barely
        changed the light.  Here we also compare the left/right halves of the face:
        if one cheek is more front-facing, the subject has a subtle yaw.  The value
        is intentionally bounded; it only steers a virtual diffuse rig.
        """
        valid = (subject_mask > 0.12) & (face_core > 0.04)
        if not np.any(valid):
            valid = subject_mask > 0.12
        if not np.any(valid):
            return {'yaw': 0.0, 'pitch': 0.0, 'confidence': 0.0, 'relief': 0.0}

        weights = np.clip(subject_mask, 0.0, 1.0) * (0.35 + 0.65 * np.clip(face_core, 0.0, 1.0))
        weights = np.where(valid, weights, 0.0).astype(np.float32)
        denom = max(float(np.sum(weights)), 1e-6)
        mean_n = np.sum(N * weights[..., None], axis=(0, 1)) / denom
        mean_n = safe_norm(mean_n.astype(np.float32))

        x = P[..., 0].astype(np.float32)
        x_scale = float(np.percentile(np.abs(x[valid]), 85.0)) if np.any(valid) else 1.0
        x_scale = max(x_scale, 1e-4)
        left = valid & (x < -0.10 * x_scale)
        right = valid & (x > 0.10 * x_scale)

        asym_yaw = 0.0
        if np.any(left) and np.any(right):
            wl = weights[left]
            wr = weights[right]
            left_front = float(np.sum(np.clip(N[..., 2][left], 0.0, 1.0) * wl) / max(float(np.sum(wl)), 1e-6))
            right_front = float(np.sum(np.clip(N[..., 2][right], 0.0, 1.0) * wr) / max(float(np.sum(wr)), 1e-6))
            left_x = float(np.sum(N[..., 0][left] * wl) / max(float(np.sum(wl)), 1e-6))
            right_x = float(np.sum(N[..., 0][right] * wr) / max(float(np.sum(wr)), 1e-6))
            # Positive yaw means the visible face tends toward image-right.
            asym_yaw = 0.55 * (right_front - left_front) + 0.25 * (left_x + right_x)

        yaw = float(np.clip(0.70 * float(mean_n[0]) + 0.80 * asym_yaw, -0.62, 0.62))
        pitch = float(np.clip(float(mean_n[1]), -0.46, 0.46))
        nz = np.clip(N[..., 2], 0.0, 1.0)
        confidence = float(np.clip(np.mean(nz[valid]), 0.0, 1.0))
        relief = float(np.clip(np.std(N[..., 0][valid]) + np.std(N[..., 1][valid]), 0.0, 1.0))
        return {'yaw': yaw, 'pitch': pitch, 'confidence': confidence, 'relief': relief}


    def _adapt_lights_to_portrait_orientation(
        self,
        lights: List[PortraitLight],
        N: np.ndarray,
        P: np.ndarray,
        subject_mask: np.ndarray,
        face_core: np.ndarray,
    ) -> List[PortraitLight]:
        """Rotate virtual diffuse lights according to portrait orientation.

        This deliberately changes only light directions/intensity for diffuse/rim
        structure.  It does not add a new reflective layer.
        """
        if not lights or str(getattr(self, 'light_source_mode', 'hybrid')) != 'hybrid':
            return lights

        orient = self._estimate_portrait_orientation_from_normals(N, P, subject_mask, face_core)
        yaw = float(orient.get('yaw', 0.0))
        pitch = float(orient.get('pitch', 0.0))
        conf = float(orient.get('confidence', 0.0))
        relief = float(orient.get('relief', 0.0))
        if conf < 0.12:
            return lights

        adapted: List[PortraitLight] = []
        for idx, light in enumerate(lights):
            d = np.array(light.direction, dtype=np.float32)
            name = str(light.name)

            if idx == 0 or name.startswith('virtual_key'):
                # If the subject turns, swing the key around the visible cheek and
                # reduce the frontal Z component.  If the subject is almost frontal,
                # still keep a strong side key so facial normals are visible.
                if abs(yaw) >= 0.025:
                    d[0] += -1.05 * yaw
                else:
                    d[0] += 0.18 * np.sign(d[0] if abs(float(d[0])) > 1e-5 else 1.0)
                d[1] += 0.34 * pitch + 0.08
                d[2] = max(0.16, d[2] - 0.24 * abs(yaw) - 0.10 * relief)
                intensity = float(light.intensity) * float(np.clip(1.08 + 0.26 * relief, 1.08, 1.28))
                diffuse_scale = float(light.diffuse_scale) * 1.10
                specular_scale = min(float(light.specular_scale), 0.10)

            elif name.startswith('virtual_fill'):
                # Keep fill weak and frontal; otherwise it flattens the key shading.
                d[0] += 0.10 * yaw
                d[1] += 0.04 * pitch
                d[2] = max(0.72, d[2])
                intensity = float(light.intensity) * 0.70
                diffuse_scale = float(light.diffuse_scale) * 0.70
                specular_scale = min(float(light.specular_scale), 0.035)

            elif name.startswith('virtual_rim') or name.startswith('virtual_kicker'):
                # Rim goes to the opposite side of the visible turn and emphasizes
                # hair/shoulder silhouette without making face skin glossy.
                if abs(yaw) >= 0.025:
                    d[0] += 0.70 * yaw
                else:
                    d[0] += -0.16 * np.sign(d[0] if abs(float(d[0])) > 1e-5 else 1.0)
                d[1] += 0.16 * pitch
                d[2] = max(0.18, d[2] - 0.12 * abs(yaw))
                intensity = float(light.intensity) * 1.10
                diffuse_scale = float(light.diffuse_scale)
                specular_scale = min(float(light.specular_scale), 0.06)

            else:
                intensity = float(light.intensity)
                diffuse_scale = float(light.diffuse_scale)
                specular_scale = min(float(light.specular_scale), 0.06)

            d = safe_norm(d)
            adapted.append(PortraitLight(
                name=light.name,
                direction=tuple(float(x) for x in d),
                color=light.color,
                intensity=intensity,
                size=float(light.size),
                diffuse_scale=diffuse_scale,
                specular_scale=specular_scale,
                rim_scale=float(light.rim_scale),
            ))
        return adapted
