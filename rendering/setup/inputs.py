from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import List, Optional, Tuple
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

class RendererInputMixin:
    @staticmethod
    def resolve_pass_file(folder: str, filename: str, src_token: str, dst_token: str) -> str:
        candidates = [filename.replace(src_token, dst_token), filename]
        for cand in candidates:
            p = os.path.join(folder, cand)
            if os.path.exists(p):
                return p
        return os.path.join(folder, candidates[0])


    @classmethod
    def resolve_pass_from_folders(cls, folders: List[str], filename: str, src_token: str, dst_token: str) -> str:
        for folder in folders:
            if folder and os.path.isdir(folder):
                p = cls.resolve_pass_file(folder, filename, src_token, dst_token)
                if os.path.exists(p):
                    return p
        folder0 = folders[0] if folders else ''
        return cls.resolve_pass_file(folder0, filename, src_token, dst_token)


    def save_lighting_info_json(self, info: LightingInfo, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(asdict(info), f, indent=2, ensure_ascii=False)


    def get_camera_params_for_image(self, filename: Optional[str], size_hw: Optional[Tuple[int, int]] = None) -> Optional[CameraParams]:
        if self.camera_data is None:
            return None
        structured = _parse_structured_camera_params(self.camera_data, filename=filename, fallback_size_hw=size_hw)
        if structured is not None:
            return structured
        camera_node = _camera_entry_for_filename(self.camera_data, filename) if filename else self.camera_data
        params = _parse_camera_params(camera_node, fallback_size_hw=size_hw)
        if params.fx_px is None and params.fy_px is None and params.depth_scale is None and params.depth_bias is None and params.depth_invert is None:
            return None
        return params
