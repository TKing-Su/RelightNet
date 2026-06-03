from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
from tools.color import srgb_to_linear, linear_to_srgb

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
    Image.fromarray((srgb * 255.0 + 0.5).astype(np.uint8)).save(path)

def save_rgba_cutout(path: str, img_linear: np.ndarray, alpha: np.ndarray) -> None:
    """Save a straight-alpha cutout.

    Earlier versions darkened RGB by alpha for preview convenience. That can create
    grey/black fringes when the cutout is reused in another compositor. This
    version keeps the RGB clean and stores alpha separately.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    rgb = linear_to_srgb(np.clip(img_linear, 0.0, 1.0))
    rgba = np.dstack([rgb, alpha])
    Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8)).save(path)

def load_background_cover_linear(path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.open(path).convert('RGB')
    fit = ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS)
    arr = np.asarray(fit, dtype=np.float32) / 255.0
    return srgb_to_linear(arr)
