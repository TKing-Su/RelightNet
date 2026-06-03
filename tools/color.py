from __future__ import annotations
from typing import Tuple
import numpy as np
from config.constants import LUMA

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)

def rgb_luminance(rgb_linear: np.ndarray) -> np.ndarray:
    return (rgb_linear[..., 0] * LUMA[0] + rgb_linear[..., 1] * LUMA[1] + rgb_linear[..., 2] * LUMA[2]).astype(np.float32)

def _color_luma_any(color: np.ndarray) -> np.ndarray:
    c = np.asarray(color, dtype=np.float32)
    return np.sum(c * LUMA.reshape((1,) * (c.ndim - 1) + (3,)), axis=-1)

def saturate_color(color: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 2.0))
    c = np.asarray(color, dtype=np.float32)
    lum = _color_luma_any(c)
    neutral = np.repeat(lum[..., None], 3, axis=-1)
    return np.clip(neutral + (c - neutral) * amount, 0.0, None).astype(np.float32)

def desaturate_color(color: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    c = np.asarray(color, dtype=np.float32)
    lum = _color_luma_any(c)
    neutral = np.repeat(lum[..., None], 3, axis=-1)
    return (neutral * amount + c * (1.0 - amount)).astype(np.float32)

def brighten_preserve_hue(color: np.ndarray, target_luma: float) -> np.ndarray:
    c = np.asarray(color, dtype=np.float32)
    cur = np.maximum(_color_luma_any(c), 1e-5)
    if c.ndim == 1:
        return np.clip(c * (float(target_luma) / float(cur)), 0.0, None).astype(np.float32)
    return np.clip(c * (float(target_luma) / cur)[..., None], 0.0, None).astype(np.float32)

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
