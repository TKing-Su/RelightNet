from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

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
