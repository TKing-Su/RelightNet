from __future__ import annotations

import os
from typing import Any

_DEVICE = "cpu"
_CUPY: Any = None
_GPU_REASON = "CPU backend selected."


def configure_device(device: str = "cpu") -> str:
    """Configure optional array acceleration.

    The renderer remains NumPy/Pillow based. CUDA mode currently accelerates
    selected large blur operations through CuPy and returns NumPy arrays to the
    existing pipeline.
    """
    global _DEVICE, _CUPY, _GPU_REASON

    requested = str(device or "cpu").strip().lower()
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError(f"Unsupported device: {device}")

    if requested == "cpu":
        _DEVICE = "cpu"
        _CUPY = None
        _GPU_REASON = "CPU backend selected."
        return _DEVICE

    try:
        import cupy as cp  # type: ignore

        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            raise RuntimeError("No CUDA device found.")
        _DEVICE = "cuda"
        _CUPY = cp
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(int(dev.id))
        name = props.get("name", b"CUDA GPU")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        _GPU_REASON = f"CUDA backend selected: {name}"
        return _DEVICE
    except Exception as exc:
        _DEVICE = "cpu"
        _CUPY = None
        _GPU_REASON = f"CUDA backend unavailable; using CPU. Reason: {exc}"
        if requested == "cuda":
            raise RuntimeError(_GPU_REASON) from exc
        return _DEVICE


def active_device() -> str:
    return _DEVICE


def backend_message() -> str:
    return _GPU_REASON


def gpu_enabled() -> bool:
    return _DEVICE == "cuda" and _CUPY is not None


def cupy_module() -> Any:
    return _CUPY


def gpu_min_pixels() -> int:
    raw = os.environ.get("RENDER_GPU_MIN_PIXELS", "")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 512 * 512
