from __future__ import annotations

import os
import json
import argparse
from typing import Optional
from config.constants import *
from lighting.models import *
from lighting.presets import *
from config.paths import *
from lighting.style_mode import *
from tools.color import *
from tools.filters import *
from tools.backend import backend_message, configure_device
from tools.geometry import *
from tools.image_io import *
from lighting.background_analyzer import *
from lighting.light_scene import *
from rendering.pipeline import BackgroundDrivenPortraitRelight

def validate_inputs(base_path: str, background_file: str,
                     camera_json_path: Optional[str]) -> None:
    """Check that all required inputs exist before processing starts."""
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Input base directory does not exist: {base_path}")

    required_dirs = ['Source', 'Alpha', 'Normal', 'Depth']
    for name in required_dirs:
        d = os.path.join(base_path, name)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Required sub-directory missing: {d}")
        if not any(f.lower().endswith(IMAGE_EXTS) for f in os.listdir(d)):
            raise FileNotFoundError(f"No image files found in required directory: {d}")

    albedo_candidates = ['BaseColor', 'EightColor', 'Color']
    if not any(os.path.isdir(os.path.join(base_path, c)) for c in albedo_candidates):
        raise FileNotFoundError(
            f"No albedo directory found (need one of {albedo_candidates}): {base_path}"
        )

    cam = camera_json_path or os.path.join(base_path, 'Camera.json')
    if os.path.isfile(cam):
        try:
            with open(cam, 'r', encoding='utf-8') as f:
                json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Camera.json failed to parse: {cam}\nReason: {e}")

    if not os.path.isfile(background_file):
        raise FileNotFoundError(f"Background image does not exist: {background_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portrait relighting with virtual key/fill/rim directions and background-referenced color/intensity (no model dependency)"
    )
    parser.add_argument(
        "--base-path",
        "--input-dir",
        dest="base_path",
        default=None,
        help=(
            "Input pass root. It must contain Source, Alpha, Normal, Depth and "
            "one of BaseColor/EightColor/Color. Default: batch_test/0008_all_passes_uncompressed"
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output folder name created under the input root. Ignored when --output-dir is set.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output directory for generated Render/Relit/Cutout/HDRI/LightingInfo folders.",
    )
    parser.add_argument("--background-dir", default=None)
    parser.add_argument("--camera-json", default=None)
    parser.add_argument("--back", "--background-image", dest="background_image", default=None)
    parser.add_argument("--style-mode", default="auto", choices=["auto", "quality", "cinematic", "neon"], help="Quality-first mode. auto chooses quality/cinematic/neon from the background.")
    parser.add_argument("--lighting-pattern", default="auto", choices=["auto", "natural", "side", "top", "cinematic", "rembrandt", "split"], help="Lighting geometry preset. In hybrid mode, directions come from a virtual portrait-light rig; auto still chooses the pattern from the background analysis.")
    parser.add_argument("--key-side", default="auto", choices=["auto", "left", "right"], help="Choose whether the virtual key light hits the image-left or image-right side of the subject.")
    parser.add_argument("--light-source-mode", default="hybrid", choices=["hybrid", "background"], help="hybrid = virtual key/fill/rim directions with background-referenced color/intensity; background = use fully background-inferred light directions.")
    parser.add_argument("--direct-light-strength", type=float, default=2.10, help="Hybrid mode only: multiplier for direct virtual diffuse/spec/rim light. Higher = more visible lamp effect.")
    parser.add_argument("--specular-boost-strength", type=float, default=0.0, help="Hybrid mode only: visible normal-based specular boost. Try 0.08-0.30.")
    parser.add_argument("--environment-light-scale", type=float, default=0.38, help="Hybrid mode only: scale background field/HDRI environment terms. Lower = direct lights more visible.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Execution backend. cpu is default; cuda uses optional CuPy blur acceleration; auto uses CUDA when available.")
    parser.add_argument("--gpu", action="store_true", help="Shortcut for --device cuda.")
    parser.add_argument("--write-builtins", default=None, help="Write the built-in quality/cinematic/neon profile JSON files to this directory and exit. Recommended path: /autodl-fs/data/config/presets")
    parser.add_argument("--no-contact-shadow", action="store_true", help="Disable depth-based subject contact shadow.")
    parser.add_argument("--no-ground-shadow", action="store_true", help="Disable screen-space ground/contact shadow on the background.")
    parser.add_argument("--debug-shadows", action="store_true", help="Save DebugShadow/*_contact_shadow.png and *_ground_shadow.png.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate pipeline images to a Debug/ subdirectory for each output.")
    parser.add_argument("--no-quality-report", action="store_true", help="Disable per-image QualityReport JSON output.")
    args = parser.parse_args()
    look_safe = True
    device = "cuda" if args.gpu else args.device
    configure_device(device)
    print(f"Device: {backend_message()}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    if args.write_builtins:
        out_dir = resolve_user_path(args.write_builtins, project_root, prefer_script_dir=True)
        write_builtin_profile_files(out_dir)
        return

    background_dir = resolve_user_path(args.background_dir, project_root, prefer_script_dir=True) if args.background_dir else os.path.join(project_root, "config", "background")
    camera_json_path = resolve_user_path(args.camera_json, project_root, prefer_script_dir=False) if args.camera_json else None

    if args.base_path:
        base_path = os.path.abspath(args.base_path)
    else:
        base_path = os.path.abspath(os.path.join(project_root, "batch_test", "0008_all_passes_uncompressed"))

    background_file = resolve_background_file(background_dir, args.background_image)
    if not background_file:
        raise FileNotFoundError(f"No background found in {background_dir}.")

    validate_inputs(base_path, background_file, camera_json_path)

    style_mode = args.style_mode
    if style_mode == "auto":
        # Core route: filename style hints are disabled. The background pixels
        # still drive lighting through descriptor/budget.
        style_mode = "quality"
        print("Core auto style-mode: selected=quality; filename style hints disabled")

    output_name = args.output_name or make_output_dir_name(
        os.path.basename(background_file),
        style_mode=style_mode,
        lighting_pattern=args.lighting_pattern,
        key_side=args.key_side,
        include_style_suffix=(args.style_mode != "auto"),
    )
    if look_safe:
        output_name = output_name + '_looksafe'

    if args.output_dir:
        output_base_path = resolve_user_path(args.output_dir, project_root, prefer_script_dir=False)
    else:
        output_base_path = os.path.join(base_path, output_name)

    print(f"Input pass root: {base_path}")
    print(f"Output root: {output_base_path}")
    print(f"Final renders: {os.path.join(output_base_path, 'Render')}")
    print("Output groups: Render, Relit, Cutout, HDRI, LightingInfo, QualityReport")

    renderer = BackgroundDrivenPortraitRelight(
        input_path=os.path.join(base_path, "Source"),
        mask_path=os.path.join(base_path, "Alpha"),
        albedo_path=(os.path.join(base_path, "BaseColor") if os.path.isdir(os.path.join(base_path, "BaseColor")) else (os.path.join(base_path, "EightColor") if os.path.isdir(os.path.join(base_path, "EightColor")) else (os.path.join(base_path, "Color") if os.path.isdir(os.path.join(base_path, "Color")) else os.path.join(base_path, "BaseColor")))),
        normal_path=os.path.join(base_path, "Normal"),
        depth_path=os.path.join(base_path, "Depth"),
        specular_path=os.path.join(base_path, "Specular") if os.path.isdir(os.path.join(base_path, "Specular")) else None,
        roughness_path=os.path.join(base_path, "Roughness") if os.path.isdir(os.path.join(base_path, "Roughness")) else None,
        output_base_path=output_base_path,
        background_dir=os.path.abspath(background_dir),
        background_image=os.path.basename(background_file),
        camera_json_path=camera_json_path,
        style_mode=style_mode,
        contact_shadow=not args.no_contact_shadow,
        ground_shadow=not args.no_ground_shadow,
        debug_shadows=bool(args.debug_shadows),
        debug_dump=bool(args.debug),
        save_quality_report=not args.no_quality_report,
        lighting_pattern=args.lighting_pattern,
        key_side=args.key_side,
        light_source_mode=args.light_source_mode,
        direct_light_strength=args.direct_light_strength,
        specular_boost_strength=args.specular_boost_strength,
        environment_light_scale=args.environment_light_scale,
        look_safe=look_safe,
    )
    renderer.batch_process()

    render_dir = os.path.join(output_base_path, "Render")
    render_files = []
    if os.path.isdir(render_dir):
        render_files = sorted(
            os.path.join(render_dir, f)
            for f in os.listdir(render_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
    print(f"Result root: {output_base_path}")
    print(f"Final render folder: {render_dir}")
    if render_files:
        print(f"Generated renders: {len(render_files)}")
        print(f"First render: {render_files[0]}")
    else:
        print("Generated renders: 0")


# Look-safe validation checklist:
# Run the same source with red/cyber/misty/sunset backgrounds and --debug.
# In Debug/look_safe_budget.json compare style_expression,
# render_weight, exposure, chroma, direction and display. Red/sunset should show
# higher warmth, side/rim chroma and direct/rim weights; cyber should show higher
# neon, color_split, multicolor, edge/hair/rim chroma and bloom; misty should show
# higher mist, haze and diffusion_spread with lower shadow/rim chroma; lowkey
# scenes should show lower auto_gain_upper plus higher lowkey_preserve, shadow and
# local_contrast. Visual checks: 10_rim.png, 11_shadow.png, 12_pre_tonemap.png,
# 15_directional_field.png, 18_rim_field.png and the final Render should differ
# in direction, rim color, haze/contrast and edge/hair atmosphere, not only in
# scalar quality scores.
