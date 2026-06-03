from __future__ import annotations

import os
from typing import Optional
import numpy as np
from tqdm import tqdm
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

class RendererBatchMixin:
    def process_single_image(
        self,
        input_file: str,
        mask_file: str,
        albedo_file: str,
        normal_file: str,
        depth_file: str,
        specular_file: Optional[str],
        roughness_file: Optional[str],
        relit_output_file: str,
        composite_output_file: str,
        cutout_output_file: str,
        hdri_file: Optional[str],
        light_info_file: Optional[str],
        quality_report_file: Optional[str],
        background_file: Optional[str],
        lighting_info: Optional[LightingInfo],
    ) -> bool:
        try:
            source_linear = read_color_image_linear(input_file)
            mask = read_mask(mask_file)
            albedo_linear = read_color_image_linear(albedo_file)
            normal_map = read_normal(normal_file)
            h, w = source_linear.shape[:2]
            camera_params = self.get_camera_params_for_image(os.path.basename(input_file), size_hw=(h, w))
            depth_invert = self.depth_invert if camera_params is None or camera_params.depth_invert is None else bool(camera_params.depth_invert)
            depth_scale = self.depth_scale if camera_params is None or camera_params.depth_scale is None else float(camera_params.depth_scale)
            depth_bias = self.depth_bias if camera_params is None or camera_params.depth_bias is None else float(camera_params.depth_bias)
            depth_map = read_depth(depth_file, invert=depth_invert)
            specular_map = read_scalar_map(specular_file, (h, w), 0.25)
            roughness_map = read_scalar_map(roughness_file, (h, w), 0.56)
            background_linear = load_background_cover_linear(background_file, (h, w)) if background_file else None
            if lighting_info is None and background_linear is not None:
                lighting_info = self.extractor.extract(background_linear, camera_params=camera_params)
            if lighting_info is None:
                raise RuntimeError('No background lighting found.')
            if self.look_safe and background_linear is not None:
                desc = compute_background_descriptor(background_linear, lighting_info, source_linear=source_linear)
                self._build_and_apply_look_policy(desc)
                if self.debug_dump:
                    print(
                        "[look-safe route] filename_style_hint=disabled, "
                        "final_style_layer=compact_lookpolicy"
                    )
            else:
                self._clear_look_policy()
            self._adapt_preset_to_lighting_info(lighting_info)
            relit_linear = self.render_relight(
                source_linear,
                mask,
                albedo_linear,
                normal_map,
                depth_map,
                specular_map,
                roughness_map,
                lighting_info,
                camera_params=camera_params,
                depth_scale=depth_scale,
                depth_bias=depth_bias,
                background_linear=background_linear,
            )
            if background_linear is not None:
                # V32: background matching is a shell/edge module, not a full-face/global tint.
                relit_linear = self._v32_shell_only_background_match(relit_linear, mask, background_linear, lighting_info)
            relit_display = linear_to_srgb(np.clip(relit_linear, 0.0, 1.0).astype(np.float32))
            relit_display = self._apply_display_finish(relit_display, alpha=np.clip(mask,0.0,1.0), background_linear=background_linear)
            relit_linear = srgb_to_linear(np.clip(relit_display, 0.0, 1.0).astype(np.float32))
            save_linear_image(relit_output_file, relit_linear)
            debug_prefix = None
            if self.debug_shadows:
                base = os.path.splitext(os.path.basename(composite_output_file))[0]
                debug_prefix = os.path.join(self.output_base_path, 'DebugShadow', base)
            composite_linear, alpha = self.composite_with_background(
                relit_linear,
                mask,
                background_linear,
                lighting_info=lighting_info,
                debug_prefix=debug_prefix,
            )
            save_linear_image(composite_output_file, composite_linear)
            save_rgba_cutout(cutout_output_file, relit_linear, alpha)
            if hdri_file:
                self.extractor.save_hdri_preview(lighting_info, hdri_file)
            if light_info_file:
                self.save_lighting_info_json(lighting_info, light_info_file)
            if quality_report_file:
                self._write_quality_report(
                    quality_report_file,
                    source_linear=source_linear,
                    relit_linear=relit_linear,
                    composite_linear=composite_linear,
                    background_linear=background_linear,
                    mask=mask,
                    alpha=alpha,
                    depth_map=depth_map,
                    lighting_info=lighting_info,
                )
            if self.debug_dump:
                self._save_debug_intermediates(
                    os.path.splitext(os.path.basename(input_file))[0],
                    source_linear=source_linear,
                    mask=mask,
                    albedo_linear=albedo_linear,
                    normal_map=normal_map,
                    depth_map=depth_map,
                    background_linear=background_linear,
                    relit_display=relit_display,
                )
            return True
        except Exception as e:
            print(f'Error processing {os.path.basename(input_file)}: {e}')
            return False


    def batch_process(self) -> None:
        input_files = [f for f in os.listdir(self.input_path) if f.lower().endswith(IMAGE_EXTS)]
        if not input_files:
            print('No input images found.')
            return
        background_file = resolve_background_file(self.background_dir, self.background_image)
        if not background_file:
            raise FileNotFoundError(f'No background found in {self.background_dir}.')
        print(f'Using background: {background_file}')
        print(f'ACTIVE_PIPELINE={RELIGHT_VERSION}; route=core_look_safe; style_blocks=exclusive; final_patch=off')
        if self.look_safe:
            print(
                'CORE_ROUTE=ON: route=continuous_budget; filename_style_hints=disabled; '
                'v32_style=continuous; extractor_style=default; '
                'display_finish=budget_only'
            )
        if self.camera_json_path and os.path.isfile(self.camera_json_path):
            print(f'Using camera json: {self.camera_json_path}')
            preview_cam = self.get_camera_params_for_image(input_files[0]) if input_files else None
            if preview_cam is not None:
                print(
                    'Camera intrinsics | '
                    f'fx={preview_cam.fx_px:.3f} fy={preview_cam.fy_px:.3f} '
                    f'cx={preview_cam.cx_px:.3f} cy={preview_cam.cy_px:.3f} '
                    f'depth_bias={preview_cam.depth_bias:.6f} depth_scale={preview_cam.depth_scale:.6f} '
                    f'depth_invert={preview_cam.depth_invert}'
                )
        success_count = 0
        fail_count = 0
        lighting_preview = read_color_image_linear(background_file)
        for filename in tqdm(sorted(input_files), desc='Background-driven relight'):
            input_file = os.path.join(self.input_path, filename)
            base_name = os.path.splitext(filename)[0]
            relit_file = os.path.join(self.output_base_path, 'Relit', base_name.replace('Source', 'Relit') + '.png')
            render_file = os.path.join(self.output_base_path, 'Render', base_name.replace('Source', 'Render') + '.png')
            cutout_file = os.path.join(self.output_base_path, 'Cutout', base_name.replace('Source', 'Cutout') + '.png')
            hdri_file = os.path.join(self.output_base_path, 'HDRI', base_name.replace('Source', 'HDRI') + '.png')
            light_json_file = os.path.join(self.output_base_path, 'LightingInfo', base_name.replace('Source', 'LightingInfo') + '.json')
            quality_report_file = os.path.join(self.output_base_path, 'QualityReport', base_name.replace('Source', 'QualityReport') + '.json')
            mask_file = self.resolve_pass_file(self.mask_path, filename, 'Source', 'Alpha')
            albedo_folders = [self.albedo_path]
            base_root = os.path.dirname(self.albedo_path) if self.albedo_path else ''
            for alt in ('EightColor', 'BaseColor', 'Color', 'Albedo'):
                p_alt = os.path.join(base_root, alt) if base_root else ''
                if p_alt and p_alt not in albedo_folders:
                    albedo_folders.append(p_alt)
            albedo_file = self.resolve_pass_from_folders(albedo_folders, filename, 'Source', 'BaseColor')
            normal_file = self.resolve_pass_file(self.normal_path, filename, 'Source', 'Normal')
            depth_file = self.resolve_pass_file(self.depth_path, filename, 'Source', 'Depth')
            specular_file = self.resolve_pass_file(self.specular_path, filename, 'Source', 'Specular') if self.specular_path and os.path.exists(self.resolve_pass_file(self.specular_path, filename, 'Source', 'Specular')) else None
            roughness_file = self.resolve_pass_file(self.roughness_path, filename, 'Source', 'Roughness') if self.roughness_path and os.path.exists(self.resolve_pass_file(self.roughness_path, filename, 'Source', 'Roughness')) else None
            camera_params = self.get_camera_params_for_image(filename)
            lighting_info = self.extractor.extract(lighting_preview, camera_params=camera_params)
            ok = self.process_single_image(
                input_file,
                mask_file,
                albedo_file,
                normal_file,
                depth_file,
                specular_file,
                roughness_file,
                relit_file,
                render_file,
                cutout_file,
                hdri_file,
                light_json_file,
                quality_report_file,
                background_file,
                lighting_info,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
        print(f'Processing completed: {success_count} succeeded, {fail_count} failed.')
