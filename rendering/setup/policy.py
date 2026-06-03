from __future__ import annotations

from dataclasses import fields
from typing import Dict, Optional
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

class RendererPolicyMixin:
    def _apply_preset(self, preset: RelightPreset) -> None:
        self.preset = preset
        for f in fields(RelightPreset):
            setattr(self, f.name, getattr(preset, f.name))


    def _using_continuous_policy(self) -> bool:
        return bool(
            getattr(self, 'look_safe', False)
            and getattr(self, '_look_policy', None) is not None
            and getattr(self._look_policy, 'route', '') == 'continuous_budget'
        )


    def _budget(self) -> Dict[str, object]:
        if self._look_policy is not None and isinstance(self._look_policy.budget, dict):
            return self._look_policy.budget
        return self._atmosphere_budget or {}


    def _policy_section(self, name: str) -> Dict[str, float]:
        if self._look_policy is not None:
            section = getattr(self._look_policy, name, None)
            if isinstance(section, dict):
                return section
        budget = self._budget()
        section = budget.get(name, {}) if isinstance(budget, dict) else {}
        return section if isinstance(section, dict) else {}


    def _policy_value(self, section: str, key: str, default: float) -> float:
        values = self._policy_section(section)
        try:
            return float(values.get(key, default))
        except Exception:
            return float(default)


    def _set_look_policy(self, descriptor: dict, budget: dict) -> None:
        self._atmosphere_descriptor = descriptor
        self._atmosphere_budget = budget
        self._look_policy = self._build_look_policy(descriptor=descriptor, budget=budget)


    def _clear_look_policy(self) -> None:
        self._atmosphere_descriptor = None
        self._atmosphere_budget = None
        self._look_policy = None


    def _build_look_policy(self, descriptor: Optional[dict] = None, budget: Optional[dict] = None) -> LookPolicy:
        """Build the unified LookPolicy object for downstream routing."""
        descriptor = descriptor or (self._atmosphere_descriptor or {})
        budget = budget or (compute_atmosphere_budget(descriptor) if descriptor else (self._atmosphere_budget or {}))
        style_expression = budget.get('style_expression', compute_style_expression(descriptor) if descriptor else {})
        exposure = budget.get('exposure', {})
        chroma = budget.get('chroma', {})
        direction = budget.get('direction', {})
        region = budget.get('region', {})
        render_weight = budget.get('render_weight', {})
        display = budget.get('display', {})
        if not self.look_safe:
            return LookPolicy(
                route='legacy',
                creative_profile=str(getattr(self, 'style_mode', 'quality')),
                v32_style=self._v32_style_key(None),
                filename_style_hints_enabled=True,
                extractor_style='neon' if getattr(self, 'style_mode', 'quality') == 'neon' else 'default',
                descriptor=descriptor,
                style_expression=style_expression,
                exposure=exposure,
                chroma=chroma,
                direction=direction,
                region=region,
                render_weight=render_weight,
                display=display,
                budget=budget,
            )
        return LookPolicy(
            route='continuous_budget',
            creative_profile=str(getattr(self, 'style_mode', 'quality')),
            v32_style='continuous',
            filename_style_hints_enabled=False,
            extractor_style='default',
            descriptor=descriptor,
            style_expression=style_expression,
            exposure=exposure,
            chroma=chroma,
            direction=direction,
            region=region,
            render_weight=render_weight,
            display=display,
            budget=budget,
        )


    def _build_and_apply_look_policy(self, descriptor: dict) -> None:
        """Build and apply unified LookPolicy.
    
        This is the single exit point for all look_safe routing decisions.
        It ensures that filename hints, style branches, and creative profiles
        do not bypass the continuous atmosphere_budget.
    
        Must be called after background analysis is complete and descriptor is computed.
        """
        if not self.look_safe:
            # Non-look-safe mode: preserve legacy behavior with all style branches
            self._clear_look_policy()
            return
    
        # Look-safe mode: enforce continuous budget
        budget = compute_atmosphere_budget(descriptor) if descriptor else {}
        self._set_look_policy(descriptor, budget)
    
        # Compact look-safe logging: descriptor summary here, final layer runtime
        # logs after auto-gain inside _apply_compact_lookpolicy_layer().
        print("[LookPolicy] route=continuous_budget final_style_layer=compact filename_hint=disabled")
        rw = budget.get('render_weight', {}) if isinstance(budget, dict) else {}
        ch = budget.get('chroma', {}) if isinstance(budget, dict) else {}
        dv = background_descriptor_debug_view(descriptor)
        print(
            "[BackgroundDescriptor] "
            f"luma={dv.get('global_luma', 0.0):.3f} "
            f"contrast={dv.get('dynamic_range', 0.0):.3f} "
            f"sat={dv.get('average_saturation', 0.0):.3f} "
            f"diversity={dv.get('palette_diversity', 0.0):.3f} "
            f"gradient={dv.get('gradient_strength', 0.0):.3f} "
            f"local_light_conf={dv.get('local_light_confidence', 0.0):.3f}"
        )
        print(
            "[LookPolicy] "
            f"direct={rw.get('direct_weight', 0.0):.2f} "
            f"shadow={rw.get('shadow_weight', 0.0):.2f} "
            f"rim={rw.get('rim_weight', 0.0):.2f} "
            f"ambient={rw.get('ambient_weight', 0.0):.2f} "
            f"spill={rw.get('color_spill_weight', 0.0):.2f} "
            f"display={rw.get('display_weight', 0.0):.2f} "
            f"lowkey_chroma_dir={budget.get('lowkey_chroma_direction_gate', 0.0):.2f} "
            f"air_skin_guard={budget.get('low_chroma_air_skin_guard', 0.0):.2f} "
            f"skin_limit={ch.get('skin_tint_limit', 0.0):.3f}"
        )
