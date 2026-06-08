from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ====================== EXR -> PNG 默认配置 ======================
# 默认会在文件整理完成后，扫描 filter/Depth 下的 .exr/.EXR，并输出同名 16-bit PNG。
DEFAULT_DELETE_EXR_AFTER_CONVERT = False
DEFAULT_CONVERT_MODE = "robust_quantile"  # robust_quantile / fixed_range
DEFAULT_FIXED_MIN = 0.0
DEFAULT_FIXED_MAX = 10.0
DEFAULT_LOW_Q = 0.002
DEFAULT_HIGH_Q = 0.998
DEFAULT_ALLOW_RGB_FALLBACK = True
DEFAULT_SAVE_PREVIEW_8BIT = True
DEFAULT_OUTPUT_NEAR_BRIGHT = True
DEFAULT_EXR_CONVERT_CONFLICT = "skip"  # skip / overwrite
# ===============================================================

@dataclass
class ExrConvertResult:
    exr: str
    png: Optional[str]
    status: str  # success/failed/skipped
    reason: Optional[str] = None
    selected_channel: Optional[str] = None


# ====================== EXR -> PNG 转换模块 ======================
def _load_exr_dependencies():
    """延迟导入：只有真正转换 EXR 时才要求这些库存在。"""
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少 EXR 转换依赖。请先安装：pip install OpenEXR numpy pillow\n"
            f"原始错误：{e}"
        ) from e
    return OpenEXR, Imath, np, Image


def pick_depth_channel(header, allow_rgb_fallback: bool = False):
    channels = list(header.get("channels", {}).keys())

    exact_candidates = [
        "Z", "Depth", "DEPTH", "depth", "z",
        "Y", "y", "V", "v",
    ]
    for c in exact_candidates:
        if c in channels:
            return c

    token_candidates = [
        ".z", ".depth", ".y", ".v",
        "_z", "_depth", "_y", "_v",
        "depth.z", "depth", "distance", "dist",
    ]
    for c in channels:
        cl = c.lower()
        if any(tok in cl for tok in token_candidates):
            return c

    if len(channels) == 1:
        return channels[0]

    non_color = [c for c in channels if c.lower() not in {"r", "g", "b", "a"}]
    if len(non_color) == 1:
        return non_color[0]

    if allow_rgb_fallback:
        for c in ["R", "G", "B", "Y", "V"]:
            if c in channels:
                return c
        for c in channels:
            if c.lower() in {"r", "g", "b", "y", "v"}:
                return c

    return None


def read_exr_depth(exr_path: Path, allow_rgb_fallback: bool = False):
    OpenEXR, Imath, np, _Image = _load_exr_dependencies()

    exr = OpenEXR.InputFile(str(exr_path))
    header = exr.header()

    dw = header["dataWindow"]
    width = int(dw.max.x - dw.min.x + 1)
    height = int(dw.max.y - dw.min.y + 1)

    channels = list(header.get("channels", {}).keys())
    depth_channel = pick_depth_channel(header, allow_rgb_fallback=allow_rgb_fallback)
    if depth_channel is None:
        raise RuntimeError(f"未找到深度通道: {exr_path}\n可用通道: {channels}")

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channel(depth_channel, pt)
    depth = np.frombuffer(raw, dtype=np.float32).reshape((height, width))

    return depth, depth_channel, channels


def sanitize_depth(depth):
    _OpenEXR, _Imath, np, _Image = _load_exr_dependencies()

    depth = depth.astype(np.float32, copy=True)
    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        return np.zeros_like(depth, dtype=np.float32)

    finite_vals = depth[finite_mask]
    fill_value = float(np.percentile(finite_vals, 1.0))
    depth[~finite_mask] = fill_value
    return depth


def encode_depth_to_uint16(
    depth,
    mode: str,
    fixed_min: float,
    fixed_max: float,
    low_q: float,
    high_q: float,
    output_near_bright: bool,
):
    _OpenEXR, _Imath, np, _Image = _load_exr_dependencies()

    depth = sanitize_depth(depth)
    finite_mask = np.isfinite(depth)
    finite_vals = depth[finite_mask]

    if finite_vals.size == 0:
        depth_u16 = np.zeros_like(depth, dtype=np.uint16)
        meta = {"mode": mode, "note": "输入深度无有效值，输出全 0。"}
        return depth_u16, meta

    raw_min = float(np.min(finite_vals))
    raw_max = float(np.max(finite_vals))
    raw_mean = float(np.mean(finite_vals))

    if mode == "robust_quantile":
        if not (0.0 <= low_q < high_q <= 1.0):
            raise ValueError("LOW_Q / HIGH_Q 必须满足 0 <= LOW_Q < HIGH_Q <= 1")

        lo = float(np.quantile(finite_vals, low_q))
        hi = float(np.quantile(finite_vals, high_q))
        if hi <= lo + 1e-8:
            hi = lo + 1e-6

        d = np.clip(depth, lo, hi)
        norm01 = (d - lo) / (hi - lo)

    elif mode == "fixed_range":
        if fixed_max <= fixed_min:
            raise ValueError("fixed_range 模式下 FIXED_MAX 必须大于 FIXED_MIN")

        d = np.clip(depth, fixed_min, fixed_max)
        norm01 = (d - fixed_min) / (fixed_max - fixed_min)
        lo, hi = float(fixed_min), float(fixed_max)

    else:
        raise ValueError(f"不支持的 mode: {mode}")

    if output_near_bright:
        png01 = 1.0 - norm01
    else:
        png01 = norm01

    depth_u16 = np.round(np.clip(png01, 0.0, 1.0) * 65535.0).astype(np.uint16)

    meta = {
        "mode": mode,
        "source_range_used": [lo, hi],
        "quantiles": [low_q, high_q] if mode == "robust_quantile" else None,
        "png_range": [0, 65535],
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_mean": raw_mean,
        "output_near_bright": output_near_bright,
        "note": "16-bit PNG。默认输出近处亮、远处暗，可适配 relight1 中 depth_invert=True 的用法。",
    }
    return depth_u16, meta


def save_u16_png(depth_u16, png_path: Path):
    _OpenEXR, _Imath, _np, Image = _load_exr_dependencies()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(depth_u16)
    img.save(str(png_path), format="PNG")


def save_preview_png(depth_u16, preview_path: Path):
    _OpenEXR, _Imath, np, Image = _load_exr_dependencies()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_8 = np.round(depth_u16.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    Image.fromarray(preview_8).save(str(preview_path), format="PNG")


def save_meta_json(meta: dict, meta_path: Path):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def convert_single_exr_depth(
    exr_file: Path,
    png_file: Path,
    delete_exr: bool = False,
    mode: str = "robust_quantile",
    fixed_min: float = 0.0,
    fixed_max: float = 10.0,
    low_q: float = 0.002,
    high_q: float = 0.998,
    save_preview_8bit: bool = True,
    allow_rgb_fallback: bool = True,
    output_near_bright: bool = True,
    conflict: str = "skip",
    logger: Optional[logging.Logger] = None,
) -> ExrConvertResult:
    log = logger or logging.getLogger("vfx_filter")

    if exr_file.suffix.lower() != ".exr":
        return ExrConvertResult(str(exr_file), None, "skipped", "not_exr")

    if not exr_file.exists():
        return ExrConvertResult(str(exr_file), None, "failed", "source_not_found")

    if png_file.exists():
        if conflict == "skip":
            log.info("[EXR2PNG] 目标 PNG 已存在，跳过：%s", png_file)
            return ExrConvertResult(str(exr_file), str(png_file), "skipped", "png_exists")
        if conflict == "overwrite":
            try:
                png_file.unlink()
            except OSError as e:
                return ExrConvertResult(str(exr_file), str(png_file), "failed", f"cannot_overwrite_png:{e}")
        else:
            return ExrConvertResult(str(exr_file), str(png_file), "failed", f"unsupported_conflict:{conflict}")

    try:
        depth, depth_channel, channels = read_exr_depth(exr_file, allow_rgb_fallback=allow_rgb_fallback)

        depth_u16, meta = encode_depth_to_uint16(
            depth=depth,
            mode=mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            low_q=low_q,
            high_q=high_q,
            output_near_bright=output_near_bright,
        )

        meta["selected_channel"] = depth_channel
        meta["available_channels"] = channels
        meta["source_exr"] = str(exr_file)
        meta["output_png"] = str(png_file)

        save_u16_png(depth_u16, png_file)
        save_meta_json(meta, png_file.with_suffix(".json"))

        if save_preview_8bit:
            preview_path = png_file.with_name(png_file.stem + "_preview.png")
            save_preview_png(depth_u16, preview_path)

        if delete_exr:
            exr_file.unlink()
            log.info("[EXR2PNG] ✅ 转换成功并删除 EXR: %s -> %s | 通道: %s", exr_file, png_file, depth_channel)
        else:
            log.info("[EXR2PNG] ✅ 转换成功: %s -> %s | 通道: %s", exr_file, png_file, depth_channel)

        return ExrConvertResult(str(exr_file), str(png_file), "success", None, depth_channel)

    except Exception as e:
        log.warning("[EXR2PNG] ❌ 转换失败: %s | 错误: %s", exr_file, e)
        return ExrConvertResult(str(exr_file), str(png_file), "failed", str(e))


def debug_print_exr_channels(root_dir: Path, max_files: int, allow_rgb_fallback: bool, logger: logging.Logger):
    exr_files = sorted(list(root_dir.rglob("*.exr")) + list(root_dir.rglob("*.EXR")), key=lambda p: str(p).casefold())
    if not exr_files:
        logger.info("[EXR2PNG] 未找到 EXR 文件用于调试通道")
        return

    OpenEXR, _Imath, _np, _Image = _load_exr_dependencies()

    logger.info("===== EXR 通道调试 =====")
    for exr_file in exr_files[:max_files]:
        try:
            exr = OpenEXR.InputFile(str(exr_file))
            header = exr.header()
            channels = list(header.get("channels", {}).keys())
            picked = pick_depth_channel(header, allow_rgb_fallback=allow_rgb_fallback)
            logger.info("%s | channels=%s | picked=%s", exr_file.name, channels, picked)
        except Exception as e:
            logger.warning("%s: 读取失败 -> %s", exr_file.name, e)
    logger.info("========================")


def batch_convert_depth_exr(
    root_dir: Path,
    delete_exr: bool = False,
    mode: str = "robust_quantile",
    fixed_min: float = 0.0,
    fixed_max: float = 10.0,
    low_q: float = 0.002,
    high_q: float = 0.998,
    save_preview_8bit: bool = True,
    allow_rgb_fallback: bool = True,
    output_near_bright: bool = True,
    conflict: str = "skip",
    debug_print_channels: bool = False,
    debug_max_files: int = 5,
    logger: Optional[logging.Logger] = None,
) -> List[ExrConvertResult]:
    """
    扫描 root_dir 下所有名为 Depth 的目录，并把其中 EXR 转成同名 16-bit PNG。
    一般传入 filter_dir，这样会处理 filter_dir/Depth。
    """
    log = logger or logging.getLogger("vfx_filter")
    root = root_dir.expanduser().resolve()

    if not root.is_dir():
        log.warning("[EXR2PNG] 目录不存在，跳过转换：%s", root)
        return []

    if debug_print_channels:
        try:
            debug_print_exr_channels(root, debug_max_files, allow_rgb_fallback, log)
        except Exception as e:
            log.warning("[EXR2PNG] 通道调试失败：%s", e)

    depth_folders = [p for p in root.rglob("*") if p.is_dir() and p.name.casefold() == "depth"]
    if root.name.casefold() == "depth":
        depth_folders.insert(0, root)
    depth_folders = sorted(set(depth_folders), key=lambda p: str(p).casefold())

    if not depth_folders:
        log.info("[EXR2PNG] 未找到任何名为 Depth 的文件夹：%s", root)
        return []

    log.info("[EXR2PNG] 找到 %d 个 Depth 文件夹", len(depth_folders))
    log.info("[EXR2PNG] 转换模式=%s | conflict=%s | 输出近处更亮=%s", mode, conflict, output_near_bright)

    results: List[ExrConvertResult] = []
    for depth_folder in depth_folders:
        exr_files = sorted(
            list(depth_folder.glob("*.exr")) + list(depth_folder.glob("*.EXR")),
            key=lambda p: str(p).casefold(),
        )
        if not exr_files:
            log.info("[EXR2PNG] Depth 文件夹没有 EXR，跳过：%s", depth_folder)
            continue

        for exr_file in exr_files:
            png_file = exr_file.with_suffix(".png")
            result = convert_single_exr_depth(
                exr_file=exr_file,
                png_file=png_file,
                delete_exr=delete_exr,
                mode=mode,
                fixed_min=fixed_min,
                fixed_max=fixed_max,
                low_q=low_q,
                high_q=high_q,
                save_preview_8bit=save_preview_8bit,
                allow_rgb_fallback=allow_rgb_fallback,
                output_near_bright=output_near_bright,
                conflict=conflict,
                logger=log,
            )
            results.append(result)

    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    log.info("[EXR2PNG] 完成：success=%d | failed=%d | skipped=%d", success, failed, skipped)
    return results

