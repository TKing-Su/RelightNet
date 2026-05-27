from __future__ import annotations
import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

PI = 3.14159265359
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp')
BACKGROUND_EXTS = IMAGE_EXTS


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def read_color_image_linear(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return srgb_to_linear(arr)


def read_mask(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def read_normal(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def read_depth(path: str, invert: bool) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    src_dtype = arr.dtype
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / (65535.0 if src_dtype == np.uint16 else 255.0)
    arr = np.clip(arr, 0.0, 1.0)
    if invert:
        arr = 1.0 - arr
    return arr.astype(np.float32)


def read_scalar_map(path: Optional[str], size_hw: Tuple[int, int], fallback: float) -> np.ndarray:
    h, w = size_hw
    if path and os.path.exists(path):
        arr = np.asarray(Image.open(path))
        src_dtype = arr.dtype
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / (65535.0 if src_dtype == np.uint16 else 255.0)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    return np.full((h, w), fallback, dtype=np.float32)


def save_linear_image(path: str, img_linear: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    srgb = linear_to_srgb(np.clip(img_linear, 0.0, 1.0))
    Image.fromarray((srgb * 255.0 + 0.5).astype(np.uint8), mode='RGB').save(path)


def save_rgba_cutout(path: str, img_linear: np.ndarray, alpha: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    rgb = linear_to_srgb(np.clip(img_linear, 0.0, 1.0))
    rgb = rgb * (0.18 + 0.82 * alpha[..., None])
    rgba = np.dstack([rgb, alpha])
    Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), mode='RGBA').save(path)


def box_blur_gray(gray: np.ndarray, passes: int = 1) -> np.ndarray:
    out = gray.astype(np.float32).copy()
    for _ in range(max(1, passes)):
        p = np.pad(out, ((1, 1), (1, 1)), mode='edge')
        out = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return out.astype(np.float32)


def box_blur_rgb(rgb: np.ndarray, passes: int = 1) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for _ in range(max(1, passes)):
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode='edge')
        out = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return out.astype(np.float32)


def feather_mask(mask: np.ndarray, passes: int = 2) -> np.ndarray:
    if passes <= 0:
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
    return np.clip(box_blur_gray(mask, passes=max(1, passes)), 0.0, 1.0).astype(np.float32)


def blur_source_rgb(src: np.ndarray) -> np.ndarray:
    return box_blur_rgb(src, passes=1)


def tone_map(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0).astype(np.float32)


def rgb_luminance(rgb_linear: np.ndarray) -> np.ndarray:
    return (rgb_linear[..., 0] * LUMA[0] + rgb_linear[..., 1] * LUMA[1] + rgb_linear[..., 2] * LUMA[2]).astype(np.float32)


def safe_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return (v / max(n, eps)).astype(np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, eps)).astype(np.float32)


def decode_normal(raw_n: np.ndarray) -> np.ndarray:
    n = raw_n.astype(np.float32) * 2.0 - 1.0
    n[..., 1] = -n[..., 1]
    n[..., 2] = np.abs(n[..., 2])
    return safe_norm(n)


def reconstruct_position(depth01: np.ndarray, focal_uv: Tuple[float, float], depth_scale: float, depth_bias: float, camera_params: Optional['CameraParams'] = None) -> np.ndarray:
    h, w = depth01.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth_bias + depth01 * depth_scale
    scaled = camera_params.scaled_intrinsics((h, w)) if camera_params is not None else None
    if scaled is not None:
        fx, fy, cx, cy = scaled
        x = (xx - float(cx)) * z / max(float(fx), 1e-6)
        y = -(yy - float(cy)) * z / max(float(fy), 1e-6)
    else:
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)
        x_ndc = u * 2.0 - 1.0
        y_ndc = 1.0 - v * 2.0
        x = x_ndc * z / float(focal_uv[0])
        y = y_ndc * z / float(focal_uv[1])
    return np.dstack([x, y, z]).astype(np.float32)


def saturate_color(color: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 2.0))
    lum = float(np.dot(color, LUMA))
    neutral = np.array([lum, lum, lum], dtype=np.float32)
    return np.clip(neutral + (color - neutral) * amount, 0.0, None).astype(np.float32)


def desaturate_color(color: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    lum = float(np.dot(color, LUMA))
    neutral = np.array([lum, lum, lum], dtype=np.float32)
    return (neutral * amount + color * (1.0 - amount)).astype(np.float32)


def brighten_preserve_hue(color: np.ndarray, target_luma: float) -> np.ndarray:
    cur = float(np.dot(color, LUMA))
    if cur < 1e-5:
        return np.array([target_luma, target_luma, target_luma], dtype=np.float32)
    return np.clip(color * (target_luma / cur), 0.0, None).astype(np.float32)


def rgb_to_hsv_approx(color: np.ndarray) -> Tuple[float, float, float]:
    color = np.clip(color.astype(np.float32), 0.0, None)
    mx = float(color.max())
    mn = float(color.min())
    diff = mx - mn
    if diff < 1e-6:
        return 0.0, 0.0, mx
    if mx == float(color[0]):
        h = ((float(color[1]) - float(color[2])) / diff) % 6.0
    elif mx == float(color[1]):
        h = (float(color[2]) - float(color[0])) / diff + 2.0
    else:
        h = (float(color[0]) - float(color[1])) / diff + 4.0
    h /= 6.0
    s = diff / max(mx, 1e-6)
    return h, s, mx


def hue_distance(a: np.ndarray, b: np.ndarray) -> float:
    ha, _, _ = rgb_to_hsv_approx(a)
    hb, _, _ = rgb_to_hsv_approx(b)
    d = abs(ha - hb)
    return float(min(d, 1.0 - d))


def list_background_files(background_dir: str) -> List[str]:
    if not os.path.isdir(background_dir):
        return []
    return sorted([n for n in os.listdir(background_dir) if n.lower().endswith(BACKGROUND_EXTS)])


def resolve_background_file(background_dir: str, background_name: Optional[str]) -> Optional[str]:
    files = list_background_files(background_dir)
    if not files:
        return None
    if background_name is None:
        return os.path.join(background_dir, files[0])
    if os.path.isfile(background_name):
        return os.path.abspath(background_name)
    direct = os.path.join(background_dir, background_name)
    if os.path.exists(direct):
        return direct
    stem = os.path.splitext(background_name)[0]
    for f in files:
        if os.path.splitext(f)[0] == stem:
            return os.path.join(background_dir, f)
    raise FileNotFoundError(f"Background '{background_name}' not found in {background_dir}")


def load_background_cover_linear(path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.open(path).convert('RGB')
    fit = ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS)
    arr = np.asarray(fit, dtype=np.float32) / 255.0
    return srgb_to_linear(arr)


def make_output_dir_name(background_name: Optional[str], style_mode: str = "default") -> str:
    if not background_name:
        stem = "default"
    else:
        stem = os.path.splitext(os.path.basename(background_name))[0]
        stem = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in stem).strip('_')
        stem = stem or "default"

    mode = (style_mode or "default").lower()
    if mode == "neon":
        return f"output_{stem}_neon"
    return f"output_{stem}"


@dataclass
class CameraParams:
    fx_px: Optional[float] = None
    fy_px: Optional[float] = None
    cx_px: Optional[float] = None
    cy_px: Optional[float] = None
    width_px: Optional[float] = None
    height_px: Optional[float] = None
    depth_scale: Optional[float] = None
    depth_bias: Optional[float] = None
    depth_invert: Optional[bool] = None
    frame_idx: Optional[int] = None
    focal_length_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None

    def scaled_intrinsics(self, size_hw: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
        h, w = size_hw
        if self.fx_px is None or self.fy_px is None:
            return None
        ref_w = float(self.width_px) if self.width_px and self.width_px > 1 else float(w)
        ref_h = float(self.height_px) if self.height_px and self.height_px > 1 else float(h)
        sx = float(w) / max(ref_w, 1.0)
        sy = float(h) / max(ref_h, 1.0)
        fx = float(self.fx_px) * sx
        fy = float(self.fy_px) * sy
        cx_default = ref_w * 0.5
        cy_default = ref_h * 0.5
        cx = float(self.cx_px if self.cx_px is not None else cx_default) * sx
        cy = float(self.cy_px if self.cy_px is not None else cy_default) * sy
        return fx, fy, cx, cy


def _extract_int_from_filename(filename: Optional[str]) -> Optional[int]:
    if not filename:
        return None
    stem = os.path.splitext(os.path.basename(filename))[0]
    numbers = re.findall(r'\d+', stem)
    if not numbers:
        return None
    try:
        return int(numbers[-1])
    except Exception:
        return None


def _parse_pair(value: object) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        a = _to_float(value[0])
        b = _to_float(value[1])
        if a is not None and b is not None:
            return float(a), float(b)
    return None


def _frame_from_structured_camera_json(camera_data: object, filename: Optional[str] = None) -> Optional[Dict[str, object]]:
    if not isinstance(camera_data, dict):
        return None
    frames = camera_data.get('frames')
    if not isinstance(frames, list) or not frames or not all(isinstance(item, dict) for item in frames):
        return None

    if filename:
        base = os.path.basename(filename).lower()
        stem = os.path.splitext(base)[0]
        for frame in frames:
            for name_key in ('file_path', 'filepath', 'file_name', 'filename', 'image_path', 'image', 'name', 'path', 'id'):
                name_value = frame.get(name_key)
                if not isinstance(name_value, str):
                    continue
                item_base = os.path.basename(name_value).lower()
                item_stem = os.path.splitext(item_base)[0]
                if item_base == base or item_stem == stem:
                    return frame

    frame_idx = _extract_int_from_filename(filename)
    if frame_idx is not None:
        for frame in frames:
            idx_value = frame.get('frame_idx')
            if isinstance(idx_value, (int, np.integer)) and int(idx_value) == frame_idx:
                return frame

    return frames[0]


def _parse_structured_camera_params(camera_data: object, filename: Optional[str] = None, fallback_size_hw: Optional[Tuple[int, int]] = None) -> Optional[CameraParams]:
    if not isinstance(camera_data, dict) or 'frames' not in camera_data:
        return None

    frame = _frame_from_structured_camera_json(camera_data, filename)
    if not isinstance(frame, dict):
        return None

    res_pair = _parse_pair(frame.get('resolution')) or _parse_pair(camera_data.get('resolution'))
    width = float(res_pair[0]) if res_pair is not None else (float(fallback_size_hw[1]) if fallback_size_hw is not None else None)
    height = float(res_pair[1]) if res_pair is not None else (float(fallback_size_hw[0]) if fallback_size_hw is not None else None)

    sensor_height = _to_float(frame.get('sensor_height'))
    if sensor_height is None:
        sensor_height = _to_float(camera_data.get('sensor_height'))
    sensor_width = _to_float(frame.get('sensor_width'))
    if sensor_width is None:
        sensor_width = _to_float(camera_data.get('sensor_width'))
    if sensor_width is None and sensor_height is not None and width is not None and height is not None and height > 1e-6:
        sensor_width = sensor_height * width / height

    focal_length = _to_float(frame.get('focal_length'))
    if focal_length is None:
        focal_length = _to_float(camera_data.get('focal_length'))

    fx = fy = None
    if focal_length is not None and width is not None and height is not None:
        if sensor_width is not None and sensor_width > 1e-6:
            fx = focal_length / sensor_width * width
        if sensor_height is not None and sensor_height > 1e-6:
            fy = focal_length / sensor_height * height
        if fx is None and fy is not None:
            fx = fy
        if fy is None and fx is not None:
            fy = fx

    pp = _parse_pair(frame.get('principal_point')) or _parse_pair(camera_data.get('principal_point'))
    cx = float(pp[0]) if pp is not None else None
    cy = float(pp[1]) if pp is not None else None

    lens_shift_x = _to_float(frame.get('lens_shift_x'))
    if lens_shift_x is None:
        lens_shift_x = _to_float(camera_data.get('lens_shift_x'))
    lens_shift_y = _to_float(frame.get('lens_shift_y'))
    if lens_shift_y is None:
        lens_shift_y = _to_float(camera_data.get('lens_shift_y'))
    if cx is None and width is not None:
        cx = width * 0.5 + float(lens_shift_x or 0.0) * width
    if cy is None and height is not None:
        cy = height * 0.5 + float(lens_shift_y or 0.0) * height

    depth_min = _to_float(frame.get('depth_min'))
    depth_max = _to_float(frame.get('depth_max'))
    depth_bias = None
    depth_scale = None
    depth_invert = None
    if depth_min is not None and depth_max is not None:
        if depth_max >= depth_min:
            depth_bias = depth_min
            depth_scale = depth_max - depth_min
            depth_invert = False
        else:
            depth_bias = depth_max
            depth_scale = depth_min - depth_max
            depth_invert = True

    frame_idx_value = frame.get('frame_idx')
    frame_idx = int(frame_idx_value) if isinstance(frame_idx_value, (int, np.integer)) else None

    if fx is None and fy is None and depth_scale is None and depth_bias is None:
        return None

    return CameraParams(
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        width_px=width,
        height_px=height,
        depth_scale=depth_scale,
        depth_bias=depth_bias,
        depth_invert=depth_invert,
        frame_idx=frame_idx,
        focal_length_mm=focal_length,
        sensor_height_mm=sensor_height,
        sensor_width_mm=sensor_width,
    )


def _iter_json_objects(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_json_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_json_objects(item)


def _to_float(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _first_numeric_value(node: object, keys: Tuple[str, ...]) -> Optional[float]:
    lowered = {k.lower() for k in keys}
    for obj in _iter_json_objects(node):
        for key, value in obj.items():
            if str(key).lower() in lowered:
                out = _to_float(value)
                if out is not None:
                    return out
    return None


def _first_bool_value(node: object, keys: Tuple[str, ...]) -> Optional[bool]:
    lowered = {k.lower() for k in keys}
    for obj in _iter_json_objects(node):
        for key, value in obj.items():
            if str(key).lower() in lowered:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    v = value.strip().lower()
                    if v in ('true', '1', 'yes', 'y', 'on'):
                        return True
                    if v in ('false', '0', 'no', 'n', 'off'):
                        return False
    return None


def _first_value(node: object, keys: Tuple[str, ...]) -> Optional[object]:
    lowered = {k.lower() for k in keys}
    for obj in _iter_json_objects(node):
        for key, value in obj.items():
            if str(key).lower() in lowered:
                return value
    return None


def _camera_entry_for_filename(camera_data: object, filename: str) -> object:
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    base = os.path.basename(filename).lower()
    if isinstance(camera_data, dict):
        for key in (filename, base, stem):
            if key in camera_data:
                return camera_data[key]
        for key, value in camera_data.items():
            if str(key).lower() in (base, stem):
                return value
        for list_key in ('frames', 'images', 'cameras', 'views', 'shots', 'items'):
            frames = camera_data.get(list_key)
            if isinstance(frames, list):
                for item in frames:
                    if not isinstance(item, dict):
                        continue
                    for name_key in ('file_path', 'filepath', 'file_name', 'filename', 'image_path', 'image', 'name', 'path', 'id'):
                        name_value = item.get(name_key)
                        if not isinstance(name_value, str):
                            continue
                        item_base = os.path.basename(name_value).lower()
                        item_stem = os.path.splitext(item_base)[0]
                        if item_base == base or item_stem == stem:
                            return item
    return camera_data


def _parse_camera_params(camera_node: object, fallback_size_hw: Optional[Tuple[int, int]] = None) -> CameraParams:
    width = _first_numeric_value(camera_node, ('image_width', 'width', 'render_width', 'resolution_x', 'w'))
    height = _first_numeric_value(camera_node, ('image_height', 'height', 'render_height', 'resolution_y', 'h'))

    intrinsics_matrix = _first_value(camera_node, ('K', 'k', 'intrinsics_matrix', 'camera_matrix'))
    fx = fy = cx = cy = None
    if isinstance(intrinsics_matrix, list):
        try:
            if len(intrinsics_matrix) == 3 and all(isinstance(row, list) and len(row) >= 3 for row in intrinsics_matrix):
                fx = _to_float(intrinsics_matrix[0][0])
                fy = _to_float(intrinsics_matrix[1][1])
                cx = _to_float(intrinsics_matrix[0][2])
                cy = _to_float(intrinsics_matrix[1][2])
            elif len(intrinsics_matrix) >= 9:
                fx = _to_float(intrinsics_matrix[0])
                fy = _to_float(intrinsics_matrix[4])
                cx = _to_float(intrinsics_matrix[2])
                cy = _to_float(intrinsics_matrix[5])
        except Exception:
            fx = fy = cx = cy = None

    if fx is None:
        fx = _first_numeric_value(camera_node, ('fx', 'fl_x', 'focal_x', 'focal_length_x'))
    if fy is None:
        fy = _first_numeric_value(camera_node, ('fy', 'fl_y', 'focal_y', 'focal_length_y'))
    if cx is None:
        cx = _first_numeric_value(camera_node, ('cx', 'ppx', 'principal_x', 'center_x'))
    if cy is None:
        cy = _first_numeric_value(camera_node, ('cy', 'ppy', 'principal_y', 'center_y'))

    if width is None and fallback_size_hw is not None:
        width = float(fallback_size_hw[1])
    if height is None and fallback_size_hw is not None:
        height = float(fallback_size_hw[0])

    if fx is None and width is not None:
        fov_x_deg = _first_numeric_value(camera_node, ('fov_x', 'xfov'))
        fov_x_rad = _first_numeric_value(camera_node, ('camera_angle_x',))
        if fov_x_deg is not None:
            fx = 0.5 * float(width) / np.tan(np.deg2rad(fov_x_deg) * 0.5)
        elif fov_x_rad is not None:
            fx = 0.5 * float(width) / np.tan(float(fov_x_rad) * 0.5)

    if fy is None and height is not None:
        fov_y_deg = _first_numeric_value(camera_node, ('fov_y', 'yfov'))
        fov_y_rad = _first_numeric_value(camera_node, ('camera_angle_y',))
        if fov_y_deg is not None:
            fy = 0.5 * float(height) / np.tan(np.deg2rad(fov_y_deg) * 0.5)
        elif fov_y_rad is not None:
            fy = 0.5 * float(height) / np.tan(float(fov_y_rad) * 0.5)

    common_fov_deg = _first_numeric_value(camera_node, ('fov', 'field_of_view'))
    if common_fov_deg is not None:
        if fx is None and width is not None:
            fx = 0.5 * float(width) / np.tan(np.deg2rad(common_fov_deg) * 0.5)
        if fy is None and height is not None:
            fy = 0.5 * float(height) / np.tan(np.deg2rad(common_fov_deg) * 0.5)

    if cx is None and width is not None:
        cx = float(width) * 0.5
    if cy is None and height is not None:
        cy = float(height) * 0.5

    depth_scale = _first_numeric_value(camera_node, ('depth_scale', 'z_scale'))
    depth_bias = _first_numeric_value(camera_node, ('depth_bias', 'z_bias'))
    depth_invert = _first_bool_value(camera_node, ('depth_invert', 'invert_depth'))

    linear_depth = _first_bool_value(camera_node, ('linear_depth', 'depth_linear'))
    near = _first_numeric_value(camera_node, ('znear', 'near', 'near_plane'))
    far = _first_numeric_value(camera_node, ('zfar', 'far', 'far_plane'))
    if linear_depth and near is not None and far is not None:
        if depth_bias is None:
            depth_bias = near
        if depth_scale is None:
            depth_scale = far - near

    return CameraParams(
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        width_px=width,
        height_px=height,
        depth_scale=depth_scale,
        depth_bias=depth_bias,
        depth_invert=depth_invert,
    )


def load_camera_params(camera_json_path: str, filename: Optional[str] = None, fallback_size_hw: Optional[Tuple[int, int]] = None) -> Optional[CameraParams]:
    if not camera_json_path or not os.path.isfile(camera_json_path):
        return None
    with open(camera_json_path, 'r', encoding='utf-8') as f:
        camera_data = json.load(f)

    structured = _parse_structured_camera_params(camera_data, filename=filename, fallback_size_hw=fallback_size_hw)
    if structured is not None:
        return structured

    camera_node = _camera_entry_for_filename(camera_data, filename) if filename else camera_data
    params = _parse_camera_params(camera_node, fallback_size_hw=fallback_size_hw)
    if params.fx_px is None and params.fy_px is None and params.depth_scale is None and params.depth_bias is None and params.depth_invert is None:
        return None
    return params


@dataclass
class PortraitLight:
    name: str
    direction: Tuple[float, float, float]
    color: Tuple[float, float, float]
    intensity: float
    size: float
    diffuse_scale: float
    specular_scale: float
    rim_scale: float


@dataclass
class LightingInfo:
    ambient_color: Tuple[float, float, float]
    ambient_intensity: float
    key_color: Tuple[float, float, float]
    key_intensity: float
    lights: List[Dict[str, object]]
    global_mean_color: Tuple[float, float, float]
    palette_points: List[Dict[str, object]]
    palette_diversity: float
    hue_entropy: float
    dominant_hue_share: float
    adaptive_light_count: int
    background_mode: str
    neon_strength: str = 'off'


class BackgroundStudioLightExtractor:
    def __init__(self, max_lights: int = 6, style_mode: str = 'default') -> None:
        self.max_lights = max(4, int(max_lights))
        self.max_candidates = max(self.max_lights * 4, 16)
        self.style_mode = str(style_mode or 'default').lower()
        self.hue_bins = 12 if self.style_mode == 'neon' else 10
        self.cool_hue_min = 0.43
        self.cool_hue_max = 0.74
        self.warm_hue_lo = 0.18
        self.warm_hue_hi = 0.92
        self.min_sat_for_palette = 0.10 if self.style_mode == 'neon' else 0.12
        self.neon_force_cool = False
        self.min_monochrome_lights = 3
        self.max_monochrome_lights = 4
        self.rich_neon_bonus_lights = 1
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

    def _neon_region_candidates(self, color_ref: np.ndarray, score_map: np.ndarray, hue_map: np.ndarray, sat_map: np.ndarray) -> List[Dict[str, object]]:
        h, w = score_map.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)
        left = np.clip(1.0 - u / 0.52, 0.0, 1.0)
        right = np.clip((u - 0.48) / 0.52, 0.0, 1.0)
        center = np.exp(-0.5 * (((u - 0.5) / 0.22) ** 2 + ((v - 0.52) / 0.30) ** 2)).astype(np.float32)
        cool_mask = ((hue_map >= self.cool_hue_min) & (hue_map <= self.cool_hue_max) & (sat_map > 0.14)).astype(np.float32)
        warm_mask = (((hue_map <= self.warm_hue_lo) | (hue_map >= self.warm_hue_hi)) & (sat_map > 0.10)).astype(np.float32)
        out: List[Dict[str, object]] = []
        region_specs = [
            ('neon_warm_left', warm_mask * left * (0.45 + 0.55 * center)),
            ('neon_warm_right', warm_mask * right * (0.45 + 0.55 * center)),
            ('neon_cool_left', cool_mask * left * (0.45 + 0.55 * center)),
            ('neon_cool_right', cool_mask * right * (0.45 + 0.55 * center)),
        ]
        for name, region_weight in region_specs:
            weight = region_weight * score_map * (0.55 + 1.10 * sat_map)
            cand = self._candidate_from_region(name, color_ref, weight, u, v, score_map)
            if cand is None:
                continue
            hh, ss, vv = rgb_to_hsv_approx(np.array(cand['color'], dtype=np.float32))
            cand['hue'] = float(hh)
            cand['sat'] = float(ss)
            cand['val'] = float(vv)
            if 'cool' in name:
                cand['score'] = float(cand['score']) * (1.18 + 0.65 * ss)
            else:
                cand['score'] = float(cand['score']) * (1.05 + 0.35 * ss)
            out.append(cand)
        out.sort(key=lambda p: (float(p['score']), float(p.get('sat', 0.0))), reverse=True)
        return out


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

    def _region_candidates(self, color_ref: np.ndarray, score_map: np.ndarray, sat_map: np.ndarray) -> List[Dict[str, object]]:
        h, w = score_map.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)
        regions = [
            ('left', 0.16, 0.52, 0.24, 0.34),
            ('right', 0.84, 0.52, 0.24, 0.34),
            ('top', 0.50, 0.20, 0.28, 0.22),
            ('bottom', 0.50, 0.82, 0.28, 0.22),
            ('top_left', 0.18, 0.22, 0.20, 0.20),
            ('top_right', 0.82, 0.22, 0.20, 0.20),
            ('bottom_left', 0.18, 0.80, 0.20, 0.20),
            ('bottom_right', 0.82, 0.80, 0.20, 0.20),
            ('center', 0.50, 0.50, 0.22, 0.22),
        ]
        out: List[Dict[str, object]] = []
        score_n = score_map / max(float(np.percentile(score_map, 98.0)), 1e-6)
        for name, cu, cv, su, sv in regions:
            weight = np.exp(-0.5 * (((u - cu) / max(su, 1e-3)) ** 2 + ((v - cv) / max(sv, 1e-3)) ** 2)).astype(np.float32)
            weight *= (0.35 + 0.95 * np.clip(score_n, 0.0, 1.8))
            weight *= (0.30 + 1.10 * np.clip(sat_map, 0.0, 1.0))
            cand = self._candidate_from_region(name, color_ref, weight, u, v, score_map)
            if cand is not None:
                out.append(cand)
        out.sort(key=lambda p: (float(p['sat']), float(p['score'])), reverse=True)
        return out

    def _hue_bucket_candidates(self, color_ref: np.ndarray, score_map: np.ndarray, hue_map: np.ndarray, sat_map: np.ndarray) -> List[Dict[str, object]]:
        h, w = score_map.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = yy / max(h - 1, 1)
        valid = sat_map > self.min_sat_for_palette
        out: List[Dict[str, object]] = []
        if not np.any(valid):
            return out
        hue_edges = np.linspace(0.0, 1.0, self.hue_bins + 1, dtype=np.float32)
        for i in range(self.hue_bins):
            h0 = float(hue_edges[i])
            h1 = float(hue_edges[i + 1])
            mask = valid & (hue_map >= h0) & (hue_map < h1)
            if np.count_nonzero(mask) < max(12, (h * w) // 400):
                continue
            weight = np.where(mask, score_map * (0.45 + 0.95 * sat_map), 0.0).astype(np.float32)
            if float(weight.sum()) <= 1e-6:
                continue
            cand = self._candidate_from_region(f'hue_bin_{i}', color_ref, weight, u, v, score_map)
            if cand is None:
                continue
            hue, sat, _ = rgb_to_hsv_approx(np.array(cand['color'], dtype=np.float32))
            cand['hue'] = float(hue)
            cand['sat'] = float(sat)
            cand['score'] = float(cand['score']) * (0.65 + 0.85 * float(cand['sat']))
            out.append(cand)
        out.sort(key=lambda p: (float(p['sat']), float(p['score'])), reverse=True)
        return out


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

    def _adaptive_target_lights(self, background_mode: str) -> int:
        if background_mode == 'monotone':
            return max(self.min_monochrome_lights, min(self.max_monochrome_lights, self.max_lights // 2))
        if background_mode == 'rich' and self.style_mode == 'neon':
            return min(self.max_lights + self.rich_neon_bonus_lights, max(self.max_lights, 7))
        return self.max_lights

    def _select_diverse_candidates(
        self,
        candidates: List[Dict[str, object]],
        fallback_color: np.ndarray,
        cool_presence: float,
        warm_presence: float,
        target_lights: int,
        neon_strength: str = 'off',
    ) -> List[Dict[str, object]]:
        if not candidates:
            hue0, sat0, val0 = rgb_to_hsv_approx(fallback_color)
            return [{
                'u': 0.25,
                'v': 0.45,
                'x': 0.0,
                'y': 0.0,
                'score': 1.0,
                'color': fallback_color.astype(np.float32),
                'chroma': float(np.linalg.norm(fallback_color - np.dot(fallback_color, LUMA))),
                'hue': float(hue0),
                'sat': float(sat0),
                'val': float(val0),
                'kind': 'fallback',
            }]
        deduped: List[Dict[str, object]] = []
        for cand in sorted(candidates, key=lambda p: (float(p['score']) + 0.55 * float(p.get('sat', 0.0))), reverse=True):
            color = np.array(cand['color'], dtype=np.float32)
            if float(np.dot(color, LUMA)) < 0.05:
                continue
            keep = True
            for prev in deduped:
                hd = hue_distance(color, np.array(prev['color'], dtype=np.float32))
                spatial = ((float(cand['u']) - float(prev['u'])) ** 2 + (float(cand['v']) - float(prev['v'])) ** 2) ** 0.5
                if hd < (0.040 if self.style_mode == 'neon' else 0.045) and spatial < (0.20 if self.style_mode == 'neon' else 0.16):
                    keep = False
                    break
            if keep:
                deduped.append(cand)
            if len(deduped) >= self.max_candidates:
                break
        if not deduped:
            deduped.append(candidates[0])

        selected: List[Dict[str, object]] = []
        remaining = deduped[:]

        if neon_strength == 'strong':
            cool_pool = [c for c in remaining if self._is_cool_hue(float(c.get('hue', 0.0)), float(c.get('sat', 0.0)))]
            warm_pool = [c for c in remaining if self._is_warm_hue(float(c.get('hue', 0.0)), float(c.get('sat', 0.0)))]
            cool_best = sorted(cool_pool, key=lambda p: (float(p['score']) + 0.80 * float(p.get('sat', 0.0))), reverse=True)[:2]
            warm_best = sorted(warm_pool, key=lambda p: (float(p['score']) + 0.55 * float(p.get('sat', 0.0))), reverse=True)[:2]
            if warm_best:
                selected.append(warm_best[0])
            if cool_best:
                choose = cool_best[0]
                if selected:
                    choose = sorted(cool_best, key=lambda p: abs(float(p['u']) - float(selected[0]['u'])), reverse=True)[0]
                selected.append(choose)
            used_ids = {id(x) for x in selected}
            remaining = [c for c in remaining if id(c) not in used_ids]

        while remaining and len(selected) < target_lights:
            best_idx = 0
            best_value = -1.0
            cool_selected = any(self._is_cool_hue(float(p.get('hue', 0.0)), float(p.get('sat', 0.0))) for p in selected)
            warm_selected = any(self._is_warm_hue(float(p.get('hue', 0.0)), float(p.get('sat', 0.0))) for p in selected)
            for idx, cand in enumerate(remaining):
                hue = float(cand.get('hue', 0.0))
                sat = float(cand.get('sat', 0.0))
                base = float(cand['score']) * (0.50 + 1.00 * sat)
                if not selected:
                    novelty = 1.0
                    spatial_gain = 1.0
                else:
                    hds = [hue_distance(np.array(cand['color'], dtype=np.float32), np.array(prev['color'], dtype=np.float32)) for prev in selected]
                    sps = [((float(cand['u']) - float(prev['u'])) ** 2 + (float(cand['v']) - float(prev['v'])) ** 2) ** 0.5 for prev in selected]
                    min_hd = min(hds)
                    min_sp = min(sps)
                    novelty = 0.55 + 1.10 * np.clip(min_hd / (0.18 if self.style_mode == 'neon' else 0.16), 0.0, 1.0)
                    spatial_gain = 0.80 + 0.45 * np.clip(min_sp / (0.34 if self.style_mode == 'neon' else 0.24), 0.0, 1.0)
                hue_bonus = 1.0
                if neon_strength == 'strong':
                    if self._is_cool_hue(hue, sat):
                        hue_bonus *= 1.45 if cool_presence > 0.08 else 1.18
                        if not cool_selected:
                            hue_bonus *= 1.20
                    elif self._is_warm_hue(hue, sat):
                        hue_bonus *= 1.08 if warm_presence > 0.10 else 1.0
                        if not warm_selected:
                            hue_bonus *= 1.08
                    else:
                        hue_bonus *= 0.78
                else:
                    if self._is_cool_hue(hue, sat) and cool_presence > 0.12:
                        hue_bonus *= 1.22 if not cool_selected else 1.06
                value = base * novelty * spatial_gain * hue_bonus
                if value > best_value:
                    best_value = value
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        cool_candidates = [cand for cand in deduped if self._is_cool_hue(float(cand.get('hue', 0.0)), float(cand.get('sat', 0.0)))]
        warm_candidates = [cand for cand in deduped if self._is_warm_hue(float(cand.get('hue', 0.0)), float(cand.get('sat', 0.0)))]
        has_cool = any(self._is_cool_hue(float(p.get('hue', 0.0)), float(p.get('sat', 0.0))) for p in selected)
        has_warm = any(self._is_warm_hue(float(p.get('hue', 0.0)), float(p.get('sat', 0.0))) for p in selected)
        if neon_strength == 'strong' and cool_candidates and not has_cool:
            selected[-1] = sorted(cool_candidates, key=lambda p: (float(p['score']), float(p.get('sat', 0.0))), reverse=True)[0]
        if neon_strength == 'strong' and warm_candidates and not has_warm and selected:
            selected[0] = sorted(warm_candidates, key=lambda p: (float(p['score']), float(p.get('sat', 0.0))), reverse=True)[0]
        return selected[: target_lights]

    def extract(self, background_linear: np.ndarray, camera_params: Optional[CameraParams] = None) -> LightingInfo:
        bg = np.clip(background_linear.astype(np.float32), 0.0, None)
        h0, w0 = bg.shape[:2]
        scale = min(1.0, 448.0 / max(h0, w0))
        if scale < 1.0:
            nh = max(64, int(round(h0 * scale)))
            nw = max(64, int(round(w0 * scale)))
            bg_small = np.asarray(
                Image.fromarray(np.clip(linear_to_srgb(bg) * 255.0 + 0.5, 0, 255).astype(np.uint8), mode='RGB').resize((nw, nh), Image.Resampling.LANCZOS),
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
        adaptive_light_count = self._adaptive_target_lights(background_mode)

        neon_strength = 'off'
        if self.style_mode == 'neon':
            if background_mode == 'rich' and palette_diversity >= self.rich_diversity_threshold and cool_presence >= 0.10 and warm_presence >= 0.10:
                neon_strength = 'strong'
            elif background_mode != 'monotone' and ((cool_presence >= 0.05 and warm_presence >= 0.04) or palette_diversity >= self.rich_diversity_threshold - 0.02):
                neon_strength = 'soft'

        peaks = self._find_peak_candidates(color_ref, score_map, sat_map)
        regions = self._region_candidates(color_ref, score_map, sat_map)
        hue_bins = self._hue_bucket_candidates(color_ref, score_map, hue_map, sat_map)
        neon_regions = self._neon_region_candidates(color_ref, score_map, hue_map, sat_map) if neon_strength == 'strong' else []
        selected = self._select_diverse_candidates(
            peaks + regions + hue_bins + neon_regions,
            fallback_color=desaturate_color(global_mean, 0.45),
            cool_presence=cool_presence,
            warm_presence=warm_presence,
            target_lights=adaptive_light_count,
            neon_strength=neon_strength,
        )

        palette_points: List[Dict[str, object]] = []
        lights: List[PortraitLight] = []
        peak_scores = np.array([max(float(p['score']), 1e-6) for p in selected], dtype=np.float32)
        peak_scores /= max(float(peak_scores.max()), 1e-6)

        strong_neon = neon_strength == 'strong'
        soft_neon = neon_strength == 'soft'

        palette_colors: List[np.ndarray] = []
        warm_palette: List[np.ndarray] = []
        cool_palette: List[np.ndarray] = []
        for i, p in enumerate(selected):
            color = np.array(p['color'], dtype=np.float32)
            hue, sat, _ = rgb_to_hsv_approx(color)
            is_cool = self._is_cool_hue(hue, sat)
            is_warm = self._is_warm_hue(hue, sat)
            sat_boost = 1.05 if i == 0 else 1.18
            if strong_neon:
                sat_boost += 0.10
                if is_cool:
                    sat_boost += 0.20
                elif is_warm:
                    sat_boost += 0.08
            elif soft_neon:
                sat_boost += 0.04
                if is_cool:
                    sat_boost += 0.08
                elif is_warm:
                    sat_boost += 0.04
            elif is_cool:
                sat_boost += 0.16
            if background_mode == 'monotone':
                sat_boost = 0.92 if i == 0 else 0.86
                if is_warm:
                    sat_boost *= 0.96
            elif background_mode == 'rich' and strong_neon:
                sat_boost += 0.12
                if is_cool:
                    sat_boost += 0.08
                elif is_warm:
                    sat_boost += 0.04
            color = saturate_color(color, sat_boost)
            if background_mode == 'monotone':
                color = desaturate_color(color, 0.12)
            color = brighten_preserve_hue(color, max(float(np.dot(color, LUMA)), 0.12 if i else 0.16))
            direction = self._uv_to_direction(float(p['u']), float(p['v']), bg_small.shape[:2], camera_params=camera_params)
            is_key = i == 0
            if strong_neon and is_warm and not is_cool:
                intensity = float(np.clip(1.02 + 0.32 * peak_scores[i] + sat * 0.08, 0.98, 1.42))
                diffuse_scale, specular_scale, rim_scale = 0.88, 0.56, 0.045
                size = float(0.30 + 0.10 * (1.0 - peak_scores[i]))
                name = 'neon_warm_key' if is_key else f'neon_warm_{i}'
            elif strong_neon and is_cool:
                intensity = float(np.clip(1.05 + 0.42 * peak_scores[i] + sat * 0.12, 1.00, 1.55))
                diffuse_scale, specular_scale, rim_scale = 1.10, 0.46, 0.055
                size = float(0.28 + 0.12 * (1.0 - peak_scores[i]))
                name = 'neon_cool_fill' if i <= 1 else f'neon_cool_{i}'
            elif soft_neon and is_cool:
                intensity = float(np.clip(0.96 + 0.30 * peak_scores[i] + sat * 0.08, 0.94, 1.34))
                diffuse_scale, specular_scale, rim_scale = 1.02, 0.48, 0.05
                size = float(0.30 + 0.10 * (1.0 - peak_scores[i]))
                name = 'soft_neon_cool' if i <= 1 else f'soft_neon_cool_{i}'
            elif is_key:
                intensity = float(np.clip(0.92 + 0.46 * peak_scores[i], 0.92, 1.42))
                size = float(0.28 + 0.10 * (1.0 - peak_scores[i]))
                diffuse_scale, specular_scale, rim_scale = 0.84, 0.70, 0.05
                name = 'dominant_key'
            else:
                intensity = float(np.clip(0.86 + 0.58 * peak_scores[i] + sat * 0.12 + (0.10 if is_cool else 0.0), 0.86, 1.42))
                size = float(0.30 + 0.12 * (1.0 - peak_scores[i]))
                diffuse_scale = 0.98 if is_cool else 0.90
                specular_scale = 0.50
                rim_scale = 0.07 if abs(float(direction[0])) > 0.22 else 0.04
                name = f'fill_{i}'
            if background_mode == 'monotone':
                intensity *= 0.96 if i == 0 else 0.90
                diffuse_scale *= 0.96
                specular_scale *= 0.94
                size = min(size + 0.03, 0.44)
            elif background_mode == 'rich' and strong_neon:
                if is_cool:
                    intensity *= 1.14
                    diffuse_scale *= 1.10
                elif is_warm:
                    intensity *= 1.04
                else:
                    intensity *= 1.06
                specular_scale *= 0.94
                size = max(size - 0.02, 0.20)
            lights.append(PortraitLight(
                name=name,
                direction=tuple(float(v) for v in direction),
                color=tuple(float(v) for v in np.clip(color, 0.0, 4.0)),
                intensity=intensity,
                size=size,
                diffuse_scale=diffuse_scale,
                specular_scale=specular_scale,
                rim_scale=rim_scale,
            ))
            color_clipped = np.clip(color, 0.0, 4.0).astype(np.float32)
            palette_colors.append(color_clipped)
            if is_cool:
                cool_palette.append(color_clipped)
            if is_warm:
                warm_palette.append(color_clipped)
            palette_points.append({
                'name': str(name),
                'kind': str(p.get('kind', 'unknown')),
                'u': float(p['u']),
                'v': float(p['v']),
                'score': float(p['score']),
                'chroma': float(p['chroma']),
                'hue': float(hue),
                'sat': float(sat),
                'color': [float(v) for v in color_clipped],
            })

        palette_mix = np.mean(np.stack(palette_colors, axis=0), axis=0).astype(np.float32) if palette_colors else global_mean
        ambient_gray = float(np.percentile(lum, 26.0 if self.style_mode == 'neon' else 28.0))
        if background_mode == 'monotone':
            palette_mix = 0.58 * palette_mix + 0.42 * desaturate_color(global_mean, 0.45)
            ambient_base = desaturate_color(0.62 * palette_mix + 0.38 * global_mean, 0.28)
            ambient_color = brighten_preserve_hue(ambient_base, max(ambient_gray, 0.08))
            ambient_intensity = float(np.clip(0.08 + ambient_gray * 0.22, 0.08, 0.16))
        elif strong_neon and background_mode == 'rich' and warm_palette and cool_palette:
            warm_mix = np.mean(np.stack(warm_palette, axis=0), axis=0).astype(np.float32)
            cool_mix = np.mean(np.stack(cool_palette, axis=0), axis=0).astype(np.float32)
            ambient_base = 0.34 * warm_mix + 0.46 * cool_mix + 0.20 * desaturate_color(global_mean, 0.82)
            ambient_color = brighten_preserve_hue(desaturate_color(ambient_base, 0.74), max(ambient_gray, 0.05))
            ambient_intensity = float(np.clip(0.06 + ambient_gray * 0.18, 0.05, 0.12))
        elif soft_neon and warm_palette and cool_palette:
            warm_mix = np.mean(np.stack(warm_palette, axis=0), axis=0).astype(np.float32)
            cool_mix = np.mean(np.stack(cool_palette, axis=0), axis=0).astype(np.float32)
            ambient_base = 0.34 * warm_mix + 0.36 * cool_mix + 0.30 * desaturate_color(global_mean, 0.74)
            ambient_color = brighten_preserve_hue(desaturate_color(ambient_base, 0.72), max(ambient_gray, 0.06))
            ambient_intensity = float(np.clip(0.06 + ambient_gray * 0.18, 0.05, 0.11))
        else:
            ambient_base = 0.55 * palette_mix + 0.45 * desaturate_color(global_mean, 0.65)
            ambient_color = brighten_preserve_hue(desaturate_color(ambient_base, 0.72), max(ambient_gray, 0.06))
            ambient_intensity = float(np.clip(0.07 + ambient_gray * 0.22, 0.05, 0.13))

        key_color = lights[0].color
        key_intensity = float(lights[0].intensity)
        return LightingInfo(
            ambient_color=tuple(float(v) for v in np.clip(ambient_color, 0.0, 4.0)),
            ambient_intensity=ambient_intensity,
            key_color=key_color,
            key_intensity=key_intensity,
            lights=[asdict(l) for l in lights],
            global_mean_color=tuple(float(v) for v in np.clip(palette_mix, 0.0, 4.0)),
            palette_points=palette_points,
            palette_diversity=float(palette_diversity),
            hue_entropy=float(hue_entropy),
            dominant_hue_share=float(dominant_share),
            adaptive_light_count=int(len(lights)),
            background_mode=str(background_mode),
            neon_strength=neon_strength,
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
        Image.fromarray((pano * 255.0 + 0.5).astype(np.uint8), mode='RGB').save(filename)

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


class BackgroundDrivenPortraitRelight:
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
        metallic_path: Optional[str] = None,
        background_image: Optional[str] = None,
        camera_json_path: Optional[str] = None,
        max_lights: int = 6,
        style_mode: str = 'default',
    ) -> None:
        self.input_path = input_path
        self.mask_path = mask_path
        self.albedo_path = albedo_path
        self.normal_path = normal_path
        self.depth_path = depth_path
        self.specular_path = specular_path
        self.roughness_path = roughness_path
        self.metallic_path = metallic_path
        self.output_base_path = output_base_path
        self.background_dir = background_dir
        self.background_image = background_image
        self.camera_json_path = camera_json_path
        self.style_mode = str(style_mode or 'default').lower()
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

        self.source_preserve = 0.02
        self.source_shading_preserve = 0.18
        self.subject_mix = 0.84
        self.ambient_strength = 0.10
        self.fill_strength = 0.04
        self.multi_ambient_strength = 0.26
        self.multi_ambient_wrap = 0.42
        self.multi_ambient_side_bias = 0.65
        self.multi_ambient_face_bias = 0.12
        self.shadow_sculpt_strength = 0.34
        self.key_shadow_strength = 0.28
        self.rim_strength = 0.035
        self.edge_spill_strength = 0.00
        self.detail_strength = 0.10
        self.detail_limit = 0.020
        self.local_albedo_keep = 0.985
        self.alpha_blur = 1
        self.alpha_tighten = 0.035
        self.alpha_edge_softness = 0.985
        self.subject_mask_expand = 0.0
        self.edge_band_power = 1.40
        self.edge_mix_strength = 0.035
        self.edge_local_spill_strength = 0.02
        self.edge_blur_passes = 1
        self.core_color_match_strength = 0.008
        self.edge_color_match_strength = 0.015
        self.fill_edge_spec_strength = 0.06
        self.fill_hair_spec_strength = 0.70
        self.rim_edge_balance = 0.15
        self.rim_hair_balance = 0.85
        self.global_tint_strength = 0.03
        self.post_exposure = 1.16
        self.post_contrast = 1.05
        self.post_saturation = 1.06
        self.post_gamma = 1.00
        self.target_subject_p70 = 0.34
        self.max_auto_gain = 2.0
        self.background_subject_scale = 1.0
        self.neon_dual_tint_strength = 0.0
        self.neon_dual_tint_center_falloff = 1.20
        self.neon_side_separation = 0.0
        self.extractor = BackgroundStudioLightExtractor(max_lights=max_lights, style_mode=self.style_mode)
        self._apply_style_mode()

    @staticmethod
    def resolve_pass_file(folder: str, filename: str, src_token: str, dst_token: str) -> str:
        candidates = [filename.replace(src_token, dst_token), filename]
        for cand in candidates:
            p = os.path.join(folder, cand)
            if os.path.exists(p):
                return p
        return os.path.join(folder, candidates[0])

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

    def _smoothstep01(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        return (x * x * (3.0 - 2.0 * x)).astype(np.float32)

    def _prepare_subject_mask(self, mask: np.ndarray) -> np.ndarray:
        matte = np.clip(mask.astype(np.float32), 0.0, 1.0)
        matte = feather_mask(matte, passes=1)
        matte = np.clip((matte - self.alpha_tighten) / max(self.alpha_edge_softness - self.alpha_tighten, 1e-6), 0.0, 1.0)
        matte = self._smoothstep01(matte)
        if abs(self.subject_mask_expand) > 1e-6:
            matte = np.clip(matte + self.subject_mask_expand, 0.0, 1.0)
        return matte.astype(np.float32)

    def _prepare_composite_alpha(self, mask: np.ndarray) -> np.ndarray:
        alpha = self._prepare_subject_mask(mask)
        alpha = feather_mask(alpha, passes=max(0, int(self.alpha_blur)))
        alpha = np.clip((alpha - 0.01) / 0.99, 0.0, 1.0)
        return alpha.astype(np.float32)

    def _compute_occlusion(self, N: np.ndarray, depth_map: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
        nl = np.roll(N, 1, axis=1); nr = np.roll(N, -1, axis=1); nu = np.roll(N, 1, axis=0); nd = np.roll(N, -1, axis=0)
        normal_var = (1.0 - np.sum(N * nl, axis=-1) + 1.0 - np.sum(N * nr, axis=-1) + 1.0 - np.sum(N * nu, axis=-1) + 1.0 - np.sum(N * nd, axis=-1)) * 0.25
        dl = np.roll(depth_map, 1, axis=1); dr = np.roll(depth_map, -1, axis=1); du = np.roll(depth_map, 1, axis=0); dd = np.roll(depth_map, -1, axis=0)
        depth_var = (np.abs(dl - depth_map) + np.abs(dr - depth_map) + np.abs(du - depth_map) + np.abs(dd - depth_map)) * 0.25
        cavity = np.clip((normal_var - 0.01) / (0.13 - 0.01), 0.0, 1.0)
        depth_occ = np.clip((depth_var - 0.002) / (0.028 - 0.002), 0.0, 1.0)
        ao = 1.0 - (0.10 * cavity + 0.08 * depth_occ) * subject_mask
        return np.clip(ao, 0.72, 1.0).astype(np.float32)

    def _compute_source_shading(self, source_linear: np.ndarray, albedo_linear: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
        src_l = rgb_luminance(source_linear)
        alb_l = rgb_luminance(np.clip(albedo_linear, 1e-4, None))
        shading = box_blur_gray(np.clip(src_l / np.maximum(alb_l, 1e-3), 0.0, 4.0), passes=2)
        med = float(np.median(shading[subject_mask > 0.2])) if np.any(subject_mask > 0.2) else float(np.median(shading))
        shading = shading / max(med, 1e-4)
        return np.clip(shading, 0.62, 1.32).astype(np.float32)

    def _compute_directional_shadow(self, depth_map: np.ndarray, subject_mask: np.ndarray, light_dir: np.ndarray) -> np.ndarray:
        h, w = depth_map.shape
        sx = int(np.clip(np.round(-float(light_dir[0]) * max(2.0, w * 0.012)), -8, 8))
        sy = int(np.clip(np.round(float(light_dir[1]) * max(2.0, h * 0.012)), -8, 8))
        if sx == 0 and sy == 0:
            return np.ones_like(depth_map, dtype=np.float32)
        shifted = np.roll(depth_map, shift=(sy, sx), axis=(0, 1))
        occ = np.clip((shifted - depth_map - 0.006) / 0.030, 0.0, 1.0)
        shadow = 1.0 - self.key_shadow_strength * occ * subject_mask
        return np.clip(shadow, 0.72, 1.0).astype(np.float32)

    def _compute_spatial_side_mask(self, P: np.ndarray, light_dir: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
        x = P[..., 0].astype(np.float32)
        if not np.any(subject_mask > 0.08):
            return np.ones_like(subject_mask, dtype=np.float32)
        scale = float(np.percentile(np.abs(x[subject_mask > 0.08]), 88.0))
        scale = max(scale, 1e-4)
        xn = np.clip(x / scale, -1.0, 1.0)
        side = 0.5 + 0.5 * float(np.sign(light_dir[0])) * xn
        front = 1.0 - min(abs(float(light_dir[0])) * 1.2, 0.85)
        return np.clip(front + (1.0 - front) * side, 0.0, 1.0).astype(np.float32)

    def _apply_style_mode(self) -> None:
        if self.style_mode != 'neon':
            return
        self.source_preserve = 0.01
        self.source_shading_preserve = 0.14
        self.subject_mix = 0.82
        self.ambient_strength = 0.06
        self.fill_strength = 0.02
        self.multi_ambient_strength = 0.38
        self.multi_ambient_wrap = 0.52
        self.multi_ambient_side_bias = 1.05
        self.multi_ambient_face_bias = 0.18
        self.shadow_sculpt_strength = 0.28
        self.key_shadow_strength = 0.24
        self.rim_strength = 0.03
        self.global_tint_strength = 0.0
        self.core_color_match_strength = 0.004
        self.edge_color_match_strength = 0.008
        self.edge_local_spill_strength = 0.015
        self.edge_mix_strength = 0.025
        self.fill_edge_spec_strength = 0.03
        self.fill_hair_spec_strength = 0.55
        self.post_exposure = 1.14
        self.post_contrast = 1.06
        self.post_saturation = 1.10
        self.target_subject_p70 = 0.32
        self.neon_dual_tint_strength = 0.34
        self.neon_dual_tint_center_falloff = 1.35
        self.neon_side_separation = 0.28

    def _classify_light_hue(self, color: np.ndarray) -> str:
        hue, sat, _ = rgb_to_hsv_approx(color)
        if self.extractor._is_cool_hue(hue, sat):
            return 'cool'
        if self.extractor._is_warm_hue(hue, sat):
            return 'warm'
        return 'neutral'

    def _compute_signed_side_mask(self, P: np.ndarray, side_sign: float, subject_mask: np.ndarray, power: float = 1.0) -> np.ndarray:
        x = P[..., 0].astype(np.float32)
        if not np.any(subject_mask > 0.08):
            return np.ones_like(subject_mask, dtype=np.float32)
        scale = float(np.percentile(np.abs(x[subject_mask > 0.08]), 88.0))
        scale = max(scale, 1e-4)
        xn = np.clip(x / scale, -1.0, 1.0)
        mask = 0.5 + 0.5 * float(np.sign(side_sign if abs(side_sign) > 1e-6 else 1.0)) * xn
        mask = np.clip(mask, 0.0, 1.0)
        return np.power(mask, max(power, 1e-4)).astype(np.float32)


    def render_relight(
        self,
        source_linear: np.ndarray,
        mask: np.ndarray,
        albedo_linear: np.ndarray,
        normal_map: np.ndarray,
        depth_map: np.ndarray,
        specular_map: np.ndarray,
        roughness_map: np.ndarray,
        metallic_map: np.ndarray,
        lighting_info: LightingInfo,
        camera_params: Optional[CameraParams] = None,
        depth_scale: Optional[float] = None,
        depth_bias: Optional[float] = None,
    ) -> np.ndarray:
        src_blur = blur_source_rgb(source_linear)
        detail = np.clip(source_linear - src_blur, -self.detail_limit, self.detail_limit)
        subject_mask = self._prepare_subject_mask(mask)
        edge_band = np.power(np.clip(4.0 * subject_mask * (1.0 - subject_mask), 0.0, 1.0), self.edge_band_power).astype(np.float32)
        N = decode_normal(normal_map)
        effective_depth_scale = float(self.depth_scale if depth_scale is None else depth_scale)
        effective_depth_bias = float(self.depth_bias if depth_bias is None else depth_bias)
        P = reconstruct_position(depth_map, self.focal_uv, effective_depth_scale, effective_depth_bias, camera_params=camera_params)
        V = safe_norm(-P)
        NdotV = np.clip(np.sum(N * V, axis=-1), 1e-4, 1.0).astype(np.float32)
        facing = np.clip(N[..., 2], 0.0, 1.0).astype(np.float32)
        face_core = (subject_mask * np.clip((facing - 0.28) / (0.92 - 0.28), 0.0, 1.0)).astype(np.float32)
        hair_region = np.clip(subject_mask - face_core * 0.70, 0.0, 1.0).astype(np.float32)

        source_keep = self.source_preserve * np.power(np.clip((subject_mask - 0.18) / 0.82, 0.0, 1.0), 1.15)
        source_keep = source_keep[..., None]
        base_subject = (albedo_linear * (1.0 - source_keep) + source_linear * source_keep).astype(np.float32)
        base_subject = base_subject * self.local_albedo_keep + source_linear * (1.0 - self.local_albedo_keep)

        spec_map = np.clip(box_blur_gray(specular_map, passes=1), 0.0, 1.0).astype(np.float32)
        rough_map = np.clip(box_blur_gray(roughness_map, passes=1), 0.0, 1.0).astype(np.float32)
        metallic = np.clip(box_blur_gray(metallic_map, passes=1), 0.0, 1.0).astype(np.float32)
        roughness = np.clip(0.22 + 0.56 * rough_map, 0.12, 0.92)
        spec_map = np.clip(0.16 + 0.46 * spec_map, 0.0, 0.68)
        metallic = np.clip(metallic * 0.36, 0.0, 0.36)
        F0 = (0.04 * (1.0 - metallic[..., None]) + albedo_linear * metallic[..., None]).astype(np.float32)
        kd = ((1.0 - metallic[..., None]) * (1.0 - spec_map[..., None] * 0.12)).astype(np.float32)
        ao = self._compute_occlusion(N, depth_map, subject_mask)
        source_shape = self._compute_source_shading(source_linear, albedo_linear, subject_mask)

        diversity_scale = float(np.clip(getattr(lighting_info, 'palette_diversity', 0.35), 0.0, 1.0))
        background_mode = str(getattr(lighting_info, 'background_mode', 'balanced'))
        neon_strength = str(getattr(lighting_info, 'neon_strength', 'off'))
        strong_neon = self.style_mode == 'neon' and neon_strength == 'strong'
        soft_neon = self.style_mode == 'neon' and neon_strength == 'soft'
        ambient_color_np = np.array(lighting_info.ambient_color, dtype=np.float32)
        if background_mode == 'monotone':
            ambient_color_np = desaturate_color(ambient_color_np, 0.18)
        ambient_color = ambient_color_np.reshape(1, 1, 3)
        adaptive_ambient_strength = self.ambient_strength * (1.18 if background_mode == 'monotone' else (1.10 if background_mode == 'rich' and strong_neon else 1.0))
        adaptive_fill_strength = self.fill_strength * (1.22 if background_mode == 'monotone' else (1.35 if background_mode == 'rich' and strong_neon else (1.08 if soft_neon else 1.0)))
        adaptive_multi_ambient_strength = self.multi_ambient_strength * (0.48 + 0.92 * diversity_scale)
        if background_mode == 'monotone':
            adaptive_multi_ambient_strength *= 1.10
        elif background_mode == 'rich' and strong_neon:
            adaptive_multi_ambient_strength *= 1.18
        ambient = base_subject * ambient_color * float(lighting_info.ambient_intensity) * adaptive_ambient_strength
        fill = base_subject * ambient_color * adaptive_fill_strength * (0.50 + 0.50 * facing[..., None])
        multicolor_acc = np.zeros_like(base_subject)
        diffuse_acc = np.zeros_like(base_subject)
        spec_acc = np.zeros_like(base_subject)
        rim_acc = np.zeros_like(base_subject)
        lights = [PortraitLight(**d) for d in lighting_info.lights]
        key_shadow = self._compute_directional_shadow(depth_map, subject_mask, np.array(lights[0].direction, dtype=np.float32)) if lights else np.ones_like(subject_mask)

        for i, light in enumerate(lights):
            L = safe_norm(np.array(light.direction, dtype=np.float32))
            Lf = np.ones_like(N) * L.reshape(1, 1, 3)
            H = safe_norm(Lf + V)
            NdotL_raw = np.sum(N * Lf, axis=-1)
            NdotL = np.clip(NdotL_raw, 0.0, 1.0).astype(np.float32)
            NdotH = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0).astype(np.float32)
            VdotH = np.clip(np.sum(V * H, axis=-1), 0.0, 1.0).astype(np.float32)
            is_fill = i > 0
            wrap = 0.05 if not is_fill else 0.14
            soft_diff = np.clip((NdotL_raw + wrap) / (1.0 + wrap), 0.0, 1.0).astype(np.float32)
            diff_term = np.power(NdotL if not is_fill else soft_diff, 1.15 if not is_fill else 0.96)
            shadow_sculpt = 1.0 - self.shadow_sculpt_strength * np.power(np.clip(1.0 - NdotL, 0.0, 1.0), 1.45) * (0.20 + 0.80 * face_core)
            depth_shadow = key_shadow if i == 0 else (0.93 + 0.07 * key_shadow)
            shape_preserve = (1.0 - self.source_shading_preserve) + self.source_shading_preserve * source_shape
            diffuse_shape = np.clip(shadow_sculpt * depth_shadow * shape_preserve, 0.48, 1.26)
            lc = np.array(light.color, dtype=np.float32).reshape(1, 1, 3)
            le = lc * float(light.intensity)

            broad = np.clip((NdotL_raw + self.multi_ambient_wrap) / (1.0 + self.multi_ambient_wrap), 0.0, 1.0).astype(np.float32)
            broad = np.power(broad, 0.90)
            side_mask = self._compute_spatial_side_mask(P, L, subject_mask)
            face_gate = 0.84 + self.multi_ambient_face_bias * face_core
            broad_color = base_subject * le * (adaptive_multi_ambient_strength * float(light.diffuse_scale))
            hue_role = self._classify_light_hue(np.array(light.color, dtype=np.float32))
            side_emphasis = (0.35 + self.multi_ambient_side_bias * side_mask[..., None])
            if strong_neon:
                side_emphasis = np.clip(0.18 + (0.90 + self.neon_side_separation) * side_mask[..., None], 0.0, 1.55)
                if hue_role == 'cool':
                    broad_color = broad_color * 1.14
                elif hue_role == 'warm':
                    broad_color = broad_color * 0.96
            elif soft_neon:
                side_emphasis = np.clip(0.28 + 0.92 * side_mask[..., None], 0.0, 1.35)
                if hue_role == 'cool':
                    broad_color = broad_color * 1.04
            multicolor_acc += broad_color * broad[..., None] * side_emphasis * face_gate[..., None]

            rough_eff = np.clip(roughness + light.size * 0.10, 0.10, 0.98)
            D = D_GGX(NdotH, rough_eff)
            G = G_SchlickGGX(np.clip(NdotV, 0.0, 1.0), rough_eff) * G_SchlickGGX(np.clip(NdotL, 0.0, 1.0), rough_eff)
            F = fresnel_schlick(VdotH, F0)
            spec = (D[..., None] * G[..., None] * F) / np.maximum(4.0 * np.clip(NdotV, 0.0, 1.0)[..., None] * np.clip(NdotL, 0.0, 1.0)[..., None], 1e-5)
            diffuse = kd * base_subject / PI * diff_term[..., None] * diffuse_shape[..., None] * le * float(light.diffuse_scale)
            if strong_neon and is_fill:
                diffuse *= np.clip(0.30 + 1.00 * side_mask[..., None], 0.0, 1.25)
            specular = spec * le * spec_map[..., None] * float(light.specular_scale)
            specular *= np.clip(0.18 + 0.82 * NdotL[..., None], 0.0, 1.0)
            specular *= (0.54 + 0.46 * face_core[..., None]) if not is_fill else (
                0.54 + 0.46 * (hair_region[..., None] * self.fill_hair_spec_strength + edge_band[..., None] * self.fill_edge_spec_strength)
            )
            if abs(float(L[0])) > 0.20:
                rim_term = np.power(np.clip(1.0 - np.clip(np.sum(N * V, axis=-1), 0.0, 1.0), 0.0, 1.0), 2.45)
                rim_gate = np.clip((0.10 - NdotL_raw) / 0.24, 0.0, 1.0)
                rim_region = self.rim_edge_balance * edge_band + self.rim_hair_balance * hair_region
                rim_acc += le * rim_term[..., None] * rim_gate[..., None] * rim_region[..., None] * self.rim_strength * float(light.rim_scale)
            diffuse_acc += diffuse
            spec_acc += specular

        relit = ambient + fill + multicolor_acc + diffuse_acc + spec_acc + rim_acc
        if strong_neon and lights:
            warm_colors = []
            cool_colors = []
            warm_signs = []
            cool_signs = []
            for light in lights:
                c = np.array(light.color, dtype=np.float32)
                role = self._classify_light_hue(c)
                if role == 'warm':
                    warm_colors.append(c * float(light.intensity))
                    warm_signs.append(float(light.direction[0]))
                elif role == 'cool':
                    cool_colors.append(c * float(light.intensity))
                    cool_signs.append(float(light.direction[0]))
            if warm_colors and cool_colors:
                warm_color = np.mean(np.stack(warm_colors, axis=0), axis=0).astype(np.float32)
                cool_color = np.mean(np.stack(cool_colors, axis=0), axis=0).astype(np.float32)
                warm_sign = float(np.mean(warm_signs)) if warm_signs else -1.0
                cool_sign = float(np.mean(cool_signs)) if cool_signs else 1.0
                warm_side = self._compute_signed_side_mask(P, warm_sign, subject_mask, power=0.75)
                cool_side = self._compute_signed_side_mask(P, cool_sign, subject_mask, power=0.75)
                dual_gate = np.power(np.clip(np.abs(P[..., 0]) / max(float(np.percentile(np.abs(P[..., 0][subject_mask > 0.08]), 88.0)) if np.any(subject_mask > 0.08) else 1.0, 1e-4), 0.0, 1.0), self.neon_dual_tint_center_falloff)
                dual_gate = np.clip(0.22 + 0.78 * dual_gate, 0.0, 1.0) * subject_mask
                dual_tint = base_subject * (warm_color.reshape(1, 1, 3) * warm_side[..., None] + cool_color.reshape(1, 1, 3) * cool_side[..., None])
                dual_strength = self.neon_dual_tint_strength
                if background_mode == 'monotone':
                    dual_strength *= 0.22
                elif background_mode == 'rich':
                    dual_strength *= 1.35
                relit += dual_tint * dual_strength * dual_gate[..., None]
        relit *= ao[..., None]
        relit += detail * self.detail_strength * (0.34 + 0.66 * subject_mask[..., None])
        relit += edge_band[..., None] * ambient_color * self.edge_spill_strength
        relit = relit * self.subject_mix + base_subject * (1.0 - self.subject_mix)

        if background_mode == 'monotone':
            relit_luma_pre = rgb_luminance(relit)
            shadow_mask = np.clip((0.30 - relit_luma_pre) / 0.30, 0.0, 1.0) * subject_mask
            relit += ambient_color * (0.16 * shadow_mask[..., None])
            relit += base_subject * (0.08 * shadow_mask[..., None])

        if np.any(subject_mask > 0.20):
            relit_luma = rgb_luminance(relit)
            target_p70 = self.target_subject_p70
            if background_mode == 'monotone':
                target_p70 += 0.05
            elif background_mode == 'rich' and strong_neon:
                target_p70 -= 0.01
            relit_p70 = float(np.percentile(relit_luma[subject_mask > 0.20], 70.0))
            gain = float(np.clip(target_p70 / max(relit_p70, 1e-4), 0.95 if background_mode == 'rich' else 0.98, self.max_auto_gain))
            relit *= gain

        global_bg = np.array(lighting_info.global_mean_color, dtype=np.float32)
        if background_mode == 'monotone':
            global_bg = desaturate_color(global_bg, 0.38)
        else:
            global_bg = desaturate_color(global_bg, 0.86)
        global_bg = brighten_preserve_hue(global_bg, max(float(np.dot(global_bg, LUMA)), 0.14))
        global_tint = np.clip(global_bg / max(float(np.dot(global_bg, LUMA)), 1e-5), 0.98, 1.02)
        adaptive_global_tint_strength = self.global_tint_strength * (0.24 + 0.92 * diversity_scale)
        if background_mode == 'monotone':
            adaptive_global_tint_strength *= 0.35
        elif background_mode == 'rich' and strong_neon:
            adaptive_global_tint_strength *= 0.65
        relit = relit * (1.0 - adaptive_global_tint_strength) + relit * global_tint.reshape(1, 1, 3) * adaptive_global_tint_strength

        exposure_scale = self.post_exposure
        if background_mode == 'monotone':
            exposure_scale *= 1.16
        elif background_mode == 'rich' and strong_neon:
            exposure_scale *= 0.98
        graded = tone_map(np.maximum(relit * exposure_scale, 0.0))
        lum = rgb_luminance(graded)
        adaptive_post_saturation = self.post_saturation
        if background_mode == 'monotone':
            adaptive_post_saturation *= 0.92
        elif background_mode == 'rich' and strong_neon:
            adaptive_post_saturation *= 1.08
        graded = lum[..., None] * (1.0 - adaptive_post_saturation) + graded * adaptive_post_saturation
        graded = np.clip((graded - 0.5) * self.post_contrast + 0.5, 0.0, 1.0)
        graded = np.power(np.maximum(graded, 0.0), 1.0 / max(self.post_gamma, 1e-3))
        return srgb_to_linear(graded.astype(np.float32))

    def global_foreground_background_match(self, relit_linear: np.ndarray, mask: np.ndarray, background_linear: np.ndarray) -> np.ndarray:
        alpha = np.clip(mask, 0.0, 1.0)
        fg_pixels = relit_linear[alpha > 0.30]
        bg_pixels = background_linear.reshape(-1, 3)
        if fg_pixels.shape[0] < 16 or bg_pixels.shape[0] < 16:
            return relit_linear
        fg_mean = fg_pixels.mean(axis=0).astype(np.float32)
        bg_mean = bg_pixels.mean(axis=0).astype(np.float32)
        fg_l = float(np.dot(fg_mean, LUMA)); bg_l = float(np.dot(bg_mean, LUMA))
        if fg_l < 1e-5 or bg_l < 1e-5:
            return relit_linear
        fg_chroma = fg_mean / max(fg_l, 1e-5)
        bg_chroma = bg_mean / max(bg_l, 1e-5)
        color_gain = np.clip(bg_chroma / np.maximum(fg_chroma, 1e-4), 0.985, 1.015)
        matched = np.clip(relit_linear * color_gain.reshape(1, 1, 3), 0.0, None)
        edge = np.clip(4.0 * alpha * (1.0 - alpha), 0.0, 1.0)
        blend = self.core_color_match_strength * alpha[..., None] + self.edge_color_match_strength * edge[..., None]
        return relit_linear * (1.0 - blend) + matched * blend

    def composite_with_background(self, relit_linear: np.ndarray, mask: np.ndarray, background_linear: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        alpha = self._prepare_composite_alpha(mask)
        if background_linear is None:
            return relit_linear, alpha
        bg = background_linear.astype(np.float32)
        relit_matched = self.global_foreground_background_match(relit_linear, alpha, bg)
        edge = np.clip(4.0 * alpha * (1.0 - alpha), 0.0, 1.0)
        bg_blur = box_blur_rgb(bg, passes=max(1, int(self.edge_blur_passes)))
        local_spill = bg_blur * (self.edge_local_spill_strength * edge[..., None])
        relit_matched = relit_matched * (1.0 - self.edge_mix_strength * edge[..., None]) + local_spill
        comp = relit_matched * alpha[..., None] + bg * (1.0 - alpha[..., None])
        return np.clip(comp, 0.0, 8.0), alpha

    def process_single_image(
        self,
        input_file: str,
        mask_file: str,
        albedo_file: str,
        normal_file: str,
        depth_file: str,
        specular_file: Optional[str],
        roughness_file: Optional[str],
        metallic_file: Optional[str],
        relit_output_file: str,
        composite_output_file: str,
        cutout_output_file: str,
        hdri_file: Optional[str],
        light_info_file: Optional[str],
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
            metallic_map = read_scalar_map(metallic_file, (h, w), 0.0)
            background_linear = load_background_cover_linear(background_file, (h, w)) if background_file else None
            if lighting_info is None and background_linear is not None:
                lighting_info = self.extractor.extract(background_linear, camera_params=camera_params)
            if lighting_info is None:
                raise RuntimeError('No background lighting found.')
            relit_linear = self.render_relight(
                source_linear,
                mask,
                albedo_linear,
                normal_map,
                depth_map,
                specular_map,
                roughness_map,
                metallic_map,
                lighting_info,
                camera_params=camera_params,
                depth_scale=depth_scale,
                depth_bias=depth_bias,
            )
            if background_linear is not None:
                relit_linear = self.global_foreground_background_match(relit_linear, mask, background_linear)
            save_linear_image(relit_output_file, relit_linear)
            composite_linear, alpha = self.composite_with_background(relit_linear, mask, background_linear)
            save_linear_image(composite_output_file, composite_linear)
            save_rgba_cutout(cutout_output_file, relit_linear, alpha)
            if hdri_file:
                self.extractor.save_hdri_preview(lighting_info, hdri_file)
            if light_info_file:
                self.save_lighting_info_json(lighting_info, light_info_file)
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
            mask_file = self.resolve_pass_file(self.mask_path, filename, 'Source', 'Alpha')
            albedo_file = self.resolve_pass_file(self.albedo_path, filename, 'Source', 'BaseColor')
            normal_file = self.resolve_pass_file(self.normal_path, filename, 'Source', 'Normal')
            depth_file = self.resolve_pass_file(self.depth_path, filename, 'Source', 'Depth')
            specular_file = self.resolve_pass_file(self.specular_path, filename, 'Source', 'Specular') if self.specular_path and os.path.exists(self.resolve_pass_file(self.specular_path, filename, 'Source', 'Specular')) else None
            roughness_file = self.resolve_pass_file(self.roughness_path, filename, 'Source', 'Roughness') if self.roughness_path and os.path.exists(self.resolve_pass_file(self.roughness_path, filename, 'Source', 'Roughness')) else None
            metallic_file = self.resolve_pass_file(self.metallic_path, filename, 'Source', 'Metallic') if self.metallic_path and os.path.exists(self.resolve_pass_file(self.metallic_path, filename, 'Source', 'Metallic')) else None
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
                metallic_file,
                relit_file,
                render_file,
                cutout_file,
                hdri_file,
                light_json_file,
                background_file,
                lighting_info,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
        print(f'Processing completed: {success_count} succeeded, {fail_count} failed.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Background-driven portrait relighting"
    )
    parser.add_argument("--base-path", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--background-dir", default=None)
    parser.add_argument("--camera-json", default=None)
    parser.add_argument("--back", "--background-image", dest="background_image", default=None)
    parser.add_argument("--style-mode", default="default", choices=["default", "neon"])
    parser.add_argument("--neon", action="store_true", help="Shortcut for --style-mode neon")
    args = parser.parse_args()

    script_dir = os.path.abspath(os.path.dirname(__file__))
    background_dir = args.background_dir or os.path.join(script_dir, "lighting_presets", "background")

    if args.base_path:
        base_path = os.path.abspath(args.base_path)
    else:
        base_path = os.path.abspath(os.path.join(script_dir, "0331_all_passes_uncompressed"))

    background_file = resolve_background_file(background_dir, args.background_image)
    if not background_file:
        raise FileNotFoundError(f"No background found in {background_dir}.")

    style_mode = "neon" if args.neon else args.style_mode

    camera_json_path = os.path.abspath(args.camera_json) if args.camera_json else os.path.join(base_path, "camera.json")
    output_name = args.output_name or make_output_dir_name(os.path.basename(background_file), style_mode=style_mode)

    renderer = BackgroundDrivenPortraitRelight(
        input_path=os.path.join(base_path, "Source"),
        mask_path=os.path.join(base_path, "Alpha"),
        albedo_path=os.path.join(base_path, "BaseColor"),
        normal_path=os.path.join(base_path, "Normal"),
        depth_path=os.path.join(base_path, "Depth"),
        specular_path=os.path.join(base_path, "Specular") if os.path.isdir(os.path.join(base_path, "Specular")) else None,
        roughness_path=os.path.join(base_path, "Roughness") if os.path.isdir(os.path.join(base_path, "Roughness")) else None,
        metallic_path=os.path.join(base_path, "Metallic") if os.path.isdir(os.path.join(base_path, "Metallic")) else None,
        output_base_path=os.path.join(base_path, output_name),
        background_dir=os.path.abspath(background_dir),
        background_image=os.path.basename(background_file),
        camera_json_path=camera_json_path,
        style_mode=style_mode,
    )
    renderer.batch_process()


if __name__ == '__main__':
    main()
