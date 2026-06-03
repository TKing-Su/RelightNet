from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Optional, Tuple
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
from rendering.look.compact import RendererCompactMixin
from rendering.face.face_balance import RendererFaceBalanceMixin
from rendering.display.shadow_display import RendererShadowDisplayMixin
from rendering.light.lights import RendererLightsMixin
from rendering.passes.render_pass import RendererRenderMixin
from rendering.reconcile.v32 import RendererV32Mixin
from rendering.finalize.io import RendererIOMixin
from rendering.setup.policy import RendererPolicyMixin
from rendering.setup.inputs import RendererInputMixin
from rendering.setup.masks import RendererMasksMixin
from rendering.setup.skin_controls import RendererSkinControlMixin
from rendering.setup.gradient import RendererGradientMixin


class BackgroundDrivenPortraitRelight(
    RendererPolicyMixin,
    RendererInputMixin,
    RendererMasksMixin,
    RendererSkinControlMixin,
    RendererGradientMixin,
    RendererCompactMixin,
    RendererFaceBalanceMixin,
    RendererShadowDisplayMixin,
    RendererLightsMixin,
    RendererRenderMixin,
    RendererV32Mixin,
    RendererIOMixin,
):
    @staticmethod
    def _uv_to_direction(u: float, v: float, size_hw: Tuple[int, int], camera_params: Optional[CameraParams] = None) -> np.ndarray:
        """Map a 2D background location to an approximate portrait-space light direction.

        This method mirrors BackgroundStudioLightExtractor._uv_to_direction.  It is
        intentionally duplicated here because the renderer also needs to convert
        local background peaks into directions for glossy/specular lighting.
        """
        h, w = size_hw
        scaled = camera_params.scaled_intrinsics((h, w)) if camera_params is not None else None
        if scaled is not None:
            fx, fy, cx, cy = scaled
            px = float(u) * max(w - 1, 1)
            py = float(v) * max(h - 1, 1)
            x = (px - float(cx)) / max(float(fx), 1e-6)
            y = -(py - float(cy)) / max(float(fy), 1e-6)
            z = 1.0
            return safe_norm(np.array([x, y, z], dtype=np.float32))
        x = (float(u) - 0.5) * 1.85
        y = (0.50 - float(v)) * 1.28
        z = 0.72 + 0.40 * (1.0 - min(abs(float(u) - 0.5) * 1.7, 1.0))
        return safe_norm(np.array([x, y, z], dtype=np.float32))

    def __init__(
        self,
        input_path: str,
        mask_path: str,
        albedo_path: str,
        normal_path: str,
        depth_path: str,
        output_base_path: str,
        background_dir: str,
        specular_path: Optional[str] = None,
        roughness_path: Optional[str] = None,
        background_image: Optional[str] = None,
        camera_json_path: Optional[str] = None,
        max_lights: int = 6,
        style_mode: str = 'default',
        contact_shadow: bool = True,
        ground_shadow: bool = True,
        debug_shadows: bool = False,
        debug_dump: bool = False,
        save_quality_report: bool = True,
        lighting_pattern: str = 'auto',
        key_side: str = 'auto',
        light_source_mode: str = 'hybrid',
        direct_light_strength: float = 2.10,
        specular_boost_strength: float = 0.00,
        environment_light_scale: float = 0.38,
        look_safe: bool = True,
    ) -> None:
        self.input_path = input_path
        self.mask_path = mask_path
        self.albedo_path = albedo_path
        self.normal_path = normal_path
        self.depth_path = depth_path
        self.specular_path = specular_path
        self.roughness_path = roughness_path
        self.output_base_path = output_base_path
        self.background_dir = background_dir
        self.background_image = background_image
        self.camera_json_path = camera_json_path
        self.style_mode = str(style_mode or 'default').lower()
        self.lighting_pattern = str(lighting_pattern or 'auto').lower()
        self.key_side = str(key_side or 'auto').lower()
        self.light_source_mode = str(light_source_mode or 'hybrid').lower()
        if self.light_source_mode not in ('background', 'hybrid'):
            self.light_source_mode = 'hybrid'
        self.direct_light_strength = float(np.clip(direct_light_strength, 0.20, 4.00))
        self.specular_boost_strength = float(np.clip(specular_boost_strength, 0.00, 0.80))
        self.environment_light_scale = float(np.clip(environment_light_scale, 0.10, 1.40))
        self._last_selected_lighting_pattern: str = 'natural'
        self.camera_data = None
        if self.camera_json_path and os.path.isfile(self.camera_json_path):
            try:
                with open(self.camera_json_path, 'r', encoding='utf-8') as f:
                    self.camera_data = json.load(f)
            except Exception as e:
                print(f'Warning: failed to load camera json {self.camera_json_path}: {e}')
                self.camera_data = None
        self.depth_scale = 2.2
        self.depth_bias = 0.05
        self.depth_invert = True
        self.focal_uv = (1.35, 1.35)

        # Parameter engineering: parameters are still automatic for users, but now
        # they are managed by RelightPreset and can be overridden from JSON.
        self.debug_shadows = bool(debug_shadows)
        self.debug_dump = bool(debug_dump)
        self.look_safe = bool(look_safe)
        self._atmosphere_budget: Optional[dict] = None
        self._atmosphere_descriptor: Optional[dict] = None
        self._look_policy: Optional[LookPolicy] = None
        self.save_quality_report = bool(save_quality_report)
        self._last_contact_shadow: Optional[np.ndarray] = None
        self._last_ground_shadow: Optional[np.ndarray] = None
        self._last_ground_shadow_auto_enabled: bool = False
        preset = load_quality_profile(self.style_mode)
        if not contact_shadow:
            preset.contact_shadow_enabled = False
        if not ground_shadow:
            preset.ground_shadow_enabled = False
        self.base_preset = RelightPreset(**asdict(preset))
        self._apply_preset(self.base_preset)
        # Hybrid pose lighting should keep the portrait crisp.  The older
        # cinematic haze/bloom makes cold backgrounds look like the face is
        # blurred, so damp only those display-softening terms here.
        if self.light_source_mode == 'hybrid':
            # Keep cold/bright backgrounds from softening the subject.  The old
            # code accidentally damped self.bloom_strength, but display finish
            # actually reads self.post_bloom_strength, so bloom was still active.
            self.post_haze_strength = float(getattr(self, 'post_haze_strength', 0.0)) * 0.08
            self.post_bloom_strength = float(getattr(self, 'post_bloom_strength', 0.0)) * 0.12

        # Look-safe mode treats the extractor as an objective light-field analyzer.
        # Creative neon/cinematic styling may still choose a base preset, but it must
        # not change the background analysis path.
        extractor_style = 'default' if self.look_safe else ('neon' if self.style_mode == 'neon' else 'default')
        self.extractor = BackgroundStudioLightExtractor(max_lights=max_lights, style_mode=extractor_style)
