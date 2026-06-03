from __future__ import annotations
import numpy as np
from tools.backend import cupy_module, gpu_enabled, gpu_min_pixels

def box_blur_gray(gray: np.ndarray, passes: int = 1) -> np.ndarray:
    if gpu_enabled() and gray.size >= gpu_min_pixels():
        cp = cupy_module()
        out_gpu = cp.asarray(gray, dtype=cp.float32)
        for _ in range(max(1, passes)):
            p = cp.pad(out_gpu, ((1, 1), (1, 1)), mode='edge')
            out_gpu = (
                p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
            ) / 9.0
        return cp.asnumpy(out_gpu).astype(np.float32, copy=False)

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
    if gpu_enabled() and rgb.size >= gpu_min_pixels():
        cp = cupy_module()
        out_gpu = cp.asarray(rgb, dtype=cp.float32)
        for _ in range(max(1, passes)):
            p = cp.pad(out_gpu, ((1, 1), (1, 1), (0, 0)), mode='edge')
            out_gpu = (
                p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
            ) / 9.0
        return cp.asnumpy(out_gpu).astype(np.float32, copy=False)

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

def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Vectorized smoothstep used by V32 modular ramp.

    Added in V32.1 because V32 called smoothstep in the router but the
    legacy file only exposed a class-local _smoothstep01.
    """
    x = np.asarray(x, dtype=np.float32)
    t = np.clip((x - float(edge0)) / max(float(edge1) - float(edge0), 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)

def blur_source_rgb(src: np.ndarray) -> np.ndarray:
    return box_blur_rgb(src, passes=1)

def tone_map(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0).astype(np.float32)
