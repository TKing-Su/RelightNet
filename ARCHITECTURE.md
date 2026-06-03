# Core Route Architecture

The repository has one supported execution route:

```text
main.py
-> interface/cli.py
-> lighting/background_analyzer.py
-> lighting/light_scene.py
-> rendering/pipeline.py
-> rendering/passes/render_pass.py
-> rendering/finalize/batch.py
```

## Top-Level Packages

- `lighting/`: owns background feature extraction, lighting dataclasses,
  runtime presets, and conversion from background descriptors into continuous
  LookPolicy and light-scene budgets.
- `rendering/`: applies the light scene to the portrait.
- `config/constants.py`: constants and supported extensions.
- `config/paths.py`: path resolution and output naming.
- `lighting/style_mode.py`: style-name normalization and background fallback
  helpers.
- `tools/`: low-level color, image IO, filtering and geometry helpers.

## Rendering Subsystems

- `rendering/pipeline.py`: renderer composition and initialization.
- `rendering/setup/`: policy helpers, input pass lookup, masks, skin controls
  and gradient sampling.
- `rendering/look/`: look-safe context extraction, region allocation,
  directional field, atmosphere and compact final layer.
- `rendering/face/`: face detail restoration, metric balancing and subject
  region estimation.
- `rendering/light/`: virtual key/fill/rim lights and local light effects.
- `rendering/environment/`: background gradient, HDRI body light, HDRI arc, PBR
  environment and reflective finish.
- `rendering/display/`: portrait recipe, shadows and display finish.
- `rendering/finalize/`: compositing, quality report, debug image output and batch
  processing.
- `rendering/reconcile/`: V32 region/reconciliation modules used by the current
  route.
- `rendering/passes/`: main render pass and frame/material preparation.

## Where To Change Things

- Background feature extraction: `lighting/background_analyzer.py`
- Lighting dataclasses: `lighting/models.py`
- Runtime relight presets: `lighting/presets.py`
- Style-mode compatibility helpers: `lighting/style_mode.py`
- Path resolution/output naming: `config/paths.py`
- LookPolicy budgets: `lighting/light_scene.py`
- Subject light allocation: `rendering/look/`
- Face/body stability: `rendering/setup/skin_controls.py`, `rendering/face/`
- Virtual lights: `rendering/light/`
- Environment/PBR light: `rendering/environment/`
- Shadows/display finish: `rendering/display/`
- Final composite/debug/reporting: `rendering/finalize/`

## Rules

- Keep `main.py` and `interface/cli.py` thin.
- Keep one user-facing route. The CLI should not expose legacy/style router
  branches.
- Do not import from removed compatibility files.
- New helpers should live in the subsystem that owns the behavior.
