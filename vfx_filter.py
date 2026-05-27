from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from exr2png import (
    DEFAULT_CONVERT_MODE,
    DEFAULT_DELETE_EXR_AFTER_CONVERT,
    DEFAULT_EXR_CONVERT_CONFLICT,
    DEFAULT_FIXED_MAX,
    DEFAULT_FIXED_MIN,
    DEFAULT_HIGH_Q,
    DEFAULT_LOW_Q,
    ExrConvertResult,
    batch_convert_depth_exr,
)


CATEGORIES_ORDER = [
    "Source",
    "Alpha",
    "BaseColor",
    "Normal",
    "Depth",
    "Specular",
    "Roughness",
]

CATEGORY_MARKERS = {
    "Source": "source",
    "Alpha": "alpha",
    "BaseColor": "basecolor",
    "Normal": "normal",
    "Depth": "depth",
    "Specular": "specular",
    "Roughness": "roughness",
}



@dataclass
class FileTask:
    category: str
    src: Path
    group_id: str

    def __hash__(self):
        return hash((self.category, self.src, self.group_id))


@dataclass
class FileResult:
    group_id: str
    group_index: int
    category: str
    src: str
    dst: Optional[str]
    status: str  # success/failed/skipped
    reason: Optional[str] = None



@dataclass
class RunReport:
    start_time: str
    end_time: str
    elapsed_seconds: float
    root: str
    filter_dir: str
    mode: str
    conflict: str
    digits: int
    sort: str
    group_sort: str
    prefix_template: str
    id_regex: str

    png_dirs_found: int
    folders_traversed: int
    files_seen: int

    matched_total: int
    matched_by_category: Dict[str, int]
    ambiguous_multi_match: int

    groups_total: int
    processed_groups: int
    processed_by_category: Dict[str, int]
    success: int
    failed: int
    skipped: int

    exr_convert_enabled: bool
    exr_convert_root: str
    exr_convert_mode: str
    exr_convert_success: int
    exr_convert_failed: int
    exr_convert_skipped: int

    results: List[FileResult]
    exr_results: List[ExrConvertResult]


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("vfx_filter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ====================== 文件整理/分类部分 ======================
def find_input_dirs(root: Path) -> List[Path]:
    """
    查找可扫描的目录（不限层级）：
    1) .../output/PNG
    2) 目录名为 Source/Alpha/BaseColor/Normal/Depth/Specular/Roughness 的目录
    """
    if not root.exists():
        return []

    scan_dirs: List[Path] = []

    # 保留原有 output/PNG 目录识别
    for p in root.rglob("PNG"):
        if p.is_dir() and p.parent.is_dir() and p.parent.name.casefold() == "output":
            scan_dirs.append(p)

    # 识别目标类别目录，包含 Depth
    category_dir_names = {c.casefold() for c in CATEGORIES_ORDER}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.casefold() in category_dir_names:
            scan_dirs.append(p)

    return sorted(set(scan_dirs), key=lambda x: str(x).casefold())


def categorize_file(file_path: Path) -> Tuple[Optional[str], int]:
    """
    判断文件属于哪个类别（大小写不敏感），支持 EXR/PNG 等所有文件类型。
    若命中多个类别，按 CATEGORIES_ORDER 优先级取第一个，并返回 multi_match_count。
    """
    p = str(file_path).casefold()
    matched = []
    for cat in CATEGORIES_ORDER:
        if CATEGORY_MARKERS[cat] in p:
            matched.append(cat)

    if not matched:
        return None, 0
    if len(matched) == 1:
        return matched[0], 1
    return matched[0], len(matched)


def extract_group_id_from_path(src: Path, id_regex: re.Pattern) -> str:
    """
    默认策略：优先从路径中定位到 'output' 段，取其上一级目录名作为“原始 ID 容器”，
    并用 id_regex 提取 ID。例如 '00000 (2)' -> '00000'。

    若路径里没有 output 段，则退化为：从各级目录名里按顺序找第一个能匹配 id_regex 的片段。
    若仍失败，则用父目录名作为 group_id。
    """
    parts = src.parts
    output_idx = None
    for i, part in enumerate(parts):
        if part.casefold() == "output":
            output_idx = i

    if output_idx is not None and output_idx - 1 >= 0:
        candidate = parts[output_idx - 1]
        m = id_regex.search(candidate)
        if m:
            return m.group(1)
        return candidate

    for part in parts:
        m = id_regex.search(part)
        if m:
            return m.group(1)

    return src.parent.name


def collect_tasks(
    scan_dirs: List[Path],
    id_regex: re.Pattern,
) -> Tuple[List[FileTask], Dict[str, int], int, int]:
    """
    收集所有目标文件（支持 EXR/PNG 等），按类别生成任务，同时提取 group_id。
    返回：tasks, matched_by_category, ambiguous_count, seen_files
    """
    tasks: List[FileTask] = []
    matched_by_category = {c: 0 for c in CATEGORIES_ORDER}
    ambiguous = 0
    seen_files = 0

    for scan_dir in scan_dirs:
        for f in scan_dir.rglob("*"):
            if not f.is_file():
                continue
            seen_files += 1

            if f.name.startswith(".") or f.name.endswith("~"):
                continue

            cat, match_count = categorize_file(f)
            if cat is None:
                continue
            if match_count > 1:
                ambiguous += 1

            gid = extract_group_id_from_path(f, id_regex)
            tasks.append(FileTask(category=cat, src=f, group_id=gid))
            matched_by_category[cat] += 1

    return tasks, matched_by_category, ambiguous, seen_files


def sort_tasks_within_group(tasks: List[FileTask], sort_mode: str) -> List[FileTask]:
    if sort_mode == "name":
        return sorted(tasks, key=lambda t: (t.src.name.casefold(), str(t.src).casefold()))

    def key_mtime(t: FileTask) -> Tuple[float, str]:
        try:
            mt = t.src.stat().st_mtime
        except OSError:
            mt = 0.0
        return mt, t.src.name.casefold()

    return sorted(tasks, key=key_mtime)


def sort_groups(
    grouped: Dict[str, List[FileTask]],
    group_sort: str,
) -> List[Tuple[str, List[FileTask]]]:
    items = list(grouped.items())

    if group_sort == "id":
        def key_id(item: Tuple[str, List[FileTask]]) -> Tuple[int, str]:
            gid = item[0]
            try:
                return 0, f"{int(gid):020d}"
            except ValueError:
                return 1, gid.casefold()
        return sorted(items, key=key_id)

    def key_group_mtime(item: Tuple[str, List[FileTask]]) -> Tuple[float, str]:
        gid, ts = item
        mtime_list = []
        for t in ts:
            try:
                mtime_list.append(t.src.stat().st_mtime)
            except OSError:
                continue
        base = min(mtime_list) if mtime_list else 0.0
        return base, gid.casefold()

    return sorted(items, key=key_group_mtime)


def resolve_prefix(prefix_template: str, category: str) -> str:
    if "{category}" in prefix_template:
        return prefix_template.format(category=category)
    return prefix_template


def parse_existing_max_index_any_ext(dest_dir: Path, prefix: str) -> int:
    """
    扫描目标目录中已有的 {prefix}_0001.png / {prefix}_0001.exr / {prefix}_0001_1.exr 等文件，
    返回最大序号。这里不限定扩展名，避免 PNG 与 EXR 混合时编号起点不准。
    """
    if not dest_dir.exists():
        return 0

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:_\d+)?\.[^.]+$", re.IGNORECASE)
    max_idx = 0
    for f in dest_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return max_idx


def compute_global_start_index(filter_dir: Path, prefix_template: str) -> int:
    """
    多类别共用统一序号，因此用所有类别目录中已存在的最大序号作为全局起点。
    新组编号从 global_max + 1 开始。
    """
    global_max = 0
    for cat in CATEGORIES_ORDER:
        dest_dir = filter_dir / cat
        prefix = resolve_prefix(prefix_template, cat)
        global_max = max(global_max, parse_existing_max_index_any_ext(dest_dir, prefix))
    return global_max + 1


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))


def safe_move(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))


def build_group_destinations(
    group_tasks: List[FileTask],
    filter_dir: Path,
    prefix_template: str,
    digits: int,
    group_index: int,
) -> Dict[FileTask, Path]:
    """
    为该组里的每个任务生成目标路径。同一组使用同一个 group_index。
    保留原文件扩展名，支持 Depth 文件夹中的 EXR 文件。
    """
    by_cat: Dict[str, List[FileTask]] = {c: [] for c in CATEGORIES_ORDER}
    for t in group_tasks:
        by_cat[t.category].append(t)

    dest_map: Dict[FileTask, Path] = {}
    for cat, tasks in by_cat.items():
        if not tasks:
            continue
        prefix = resolve_prefix(prefix_template, cat)

        for i, t in enumerate(tasks):
            ext = t.src.suffix.lstrip(".") or "dat"
            if i == 0:
                base_name = f"{prefix}_{group_index:0{digits}d}.{ext}"
            else:
                base_name = f"{prefix}_{group_index:0{digits}d}_{i}.{ext}"

            dest_dir = filter_dir / cat
            dest_map[t] = dest_dir / base_name

    return dest_map


def group_has_conflict(destinations: Dict[FileTask, Path]) -> bool:
    return any(dst.exists() for dst in destinations.values())


def apply_group_conflict_policy(
    destinations: Dict[FileTask, Path],
    conflict: str,
) -> Tuple[Dict[FileTask, Path], Dict[FileTask, str]]:
    reasons: Dict[FileTask, str] = {}
    if conflict == "overwrite":
        for t, dst in destinations.items():
            if dst.exists():
                try:
                    dst.unlink()
                except OSError as e:
                    reasons[t] = f"cannot_overwrite:{e}"
    elif conflict == "skip":
        for t, dst in destinations.items():
            if dst.exists():
                reasons[t] = "dst_exists"
    return destinations, reasons


def process_groups(
    grouped: Dict[str, List[FileTask]],
    filter_dir: Path,
    mode: str,
    conflict: str,
    digits: int,
    sort_mode: str,
    group_sort: str,
    prefix_template: str,
    logger: logging.Logger,
) -> Tuple[int, Dict[str, int], List[FileResult]]:
    """
    按 group_id 分配统一序号，所有类别共用该序号。
    返回：processed_groups, processed_by_category, results
    """
    ensure_dir(filter_dir)
    for cat in CATEGORIES_ORDER:
        ensure_dir(filter_dir / cat)

    start_index = compute_global_start_index(filter_dir, prefix_template)
    logger.info("统一编号起点（global start index）：%d", start_index)

    ordered_groups = sort_groups(grouped, group_sort)

    processed_groups = 0
    processed_by_category = {c: 0 for c in CATEGORIES_ORDER}
    results: List[FileResult] = []

    group_index = start_index

    for gid, tasks in ordered_groups:
        tasks_sorted = sort_tasks_within_group(tasks, sort_mode)

        destinations = build_group_destinations(
            group_tasks=tasks_sorted,
            filter_dir=filter_dir,
            prefix_template=prefix_template,
            digits=digits,
            group_index=group_index,
        )

        if conflict == "rename":
            while group_has_conflict(destinations):
                group_index += 1
                destinations = build_group_destinations(
                    group_tasks=tasks_sorted,
                    filter_dir=filter_dir,
                    prefix_template=prefix_template,
                    digits=digits,
                    group_index=group_index,
                )

        destinations, pre_reasons = apply_group_conflict_policy(destinations, conflict)

        any_success_in_group = False
        for t in tasks_sorted:
            src = t.src
            dst = destinations.get(t)

            if not src.exists():
                results.append(FileResult(gid, group_index, t.category, str(src), None, "failed", "source_not_found"))
                logger.warning("[GID=%s IDX=%d %s] 源文件不存在：%s", gid, group_index, t.category, src)
                continue

            try:
                if src.stat().st_size == 0:
                    results.append(FileResult(gid, group_index, t.category, str(src), None, "skipped", "empty_file"))
                    logger.warning("[GID=%s IDX=%d %s] 跳过空文件：%s", gid, group_index, t.category, src)
                    continue
            except OSError as e:
                results.append(FileResult(gid, group_index, t.category, str(src), None, "failed", f"stat_failed:{e}"))
                logger.warning("[GID=%s IDX=%d %s] 读取文件信息失败：%s (%s)", gid, group_index, t.category, src, e)
                continue

            if dst is None:
                results.append(FileResult(gid, group_index, t.category, str(src), None, "failed", "dst_not_built"))
                continue

            if t in pre_reasons and pre_reasons[t] == "dst_exists":
                results.append(FileResult(gid, group_index, t.category, str(src), str(dst), "skipped", "dst_exists"))
                logger.info("[GID=%s IDX=%d %s] 目标已存在，跳过：%s", gid, group_index, t.category, dst.name)
                continue

            if t in pre_reasons and pre_reasons[t].startswith("cannot_overwrite"):
                results.append(FileResult(gid, group_index, t.category, str(src), str(dst), "failed", pre_reasons[t]))
                logger.warning("[GID=%s IDX=%d %s] 覆盖失败：%s", gid, group_index, t.category, pre_reasons[t])
                continue

            try:
                if mode == "copy":
                    safe_copy(src, dst)
                else:
                    safe_move(src, dst)

                results.append(FileResult(gid, group_index, t.category, str(src), str(dst), "success", None))
                processed_by_category[t.category] += 1
                any_success_in_group = True
                logger.info("[GID=%s IDX=%d %s] %s -> %s", gid, group_index, t.category, src.name, dst.name)

            except PermissionError as e:
                results.append(FileResult(gid, group_index, t.category, str(src), str(dst), "failed", f"permission:{e}"))
                logger.warning("[GID=%s IDX=%d %s] 权限不足：%s (%s)", gid, group_index, t.category, src, e)
            except OSError as e:
                results.append(FileResult(gid, group_index, t.category, str(src), str(dst), "failed", f"oserror:{e}"))
                logger.warning("[GID=%s IDX=%d %s] 处理失败：%s (%s)", gid, group_index, t.category, src, e)

        if any_success_in_group:
            processed_groups += 1
        group_index += 1

    return processed_groups, processed_by_category, results


# ====================== 报告部分 ======================
def write_reports(
    report: RunReport,
    out_dir: Path,
    logger: logging.Logger,
) -> Tuple[Path, Path, Path, Path]:
    ensure_dir(out_dir)

    json_path = out_dir / "report.json"
    csv_path = out_dir / "mapping.csv"
    exr_csv_path = out_dir / "exr2png_mapping.csv"
    txt_path = out_dir / "report.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "group_index", "category", "src", "dst", "status", "reason"])
        for r in report.results:
            writer.writerow([r.group_id, r.group_index, r.category, r.src, r.dst or "", r.status, r.reason or ""])

    with exr_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exr", "png", "status", "reason", "selected_channel"])
        for r in report.exr_results:
            writer.writerow([r.exr, r.png or "", r.status, r.reason or "", r.selected_channel or ""])

    lines = []
    lines.append(f"Start: {report.start_time}")
    lines.append(f"End:   {report.end_time}")
    lines.append(f"Elapsed(s): {report.elapsed_seconds:.3f}")
    lines.append("")
    lines.append(f"Root: {report.root}")
    lines.append(f"Filter: {report.filter_dir}")
    lines.append(f"Mode: {report.mode}")
    lines.append(f"Conflict: {report.conflict}")
    lines.append(f"Digits: {report.digits}")
    lines.append(f"Sort (within group): {report.sort}")
    lines.append(f"Group sort: {report.group_sort}")
    lines.append(f"Prefix template: {report.prefix_template}")
    lines.append(f"ID regex: {report.id_regex}")
    lines.append("")
    lines.append(f"scan dirs found: {report.png_dirs_found}")
    lines.append(f"folders traversed: {report.folders_traversed}")
    lines.append(f"files seen: {report.files_seen}")
    lines.append("")
    lines.append(f"matched_total: {report.matched_total}")
    lines.append(f"ambiguous_multi_match: {report.ambiguous_multi_match}")
    lines.append("matched_by_category:")
    for k, v in report.matched_by_category.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append(f"groups_total: {report.groups_total}")
    lines.append(f"processed_groups: {report.processed_groups}")
    lines.append("processed_by_category:")
    for k, v in report.processed_by_category.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append(f"success: {report.success}")
    lines.append(f"failed:  {report.failed}")
    lines.append(f"skipped: {report.skipped}")
    lines.append("")
    lines.append("EXR2PNG:")
    lines.append(f"  enabled: {report.exr_convert_enabled}")
    lines.append(f"  root: {report.exr_convert_root}")
    lines.append(f"  mode: {report.exr_convert_mode}")
    lines.append(f"  success: {report.exr_convert_success}")
    lines.append(f"  failed:  {report.exr_convert_failed}")
    lines.append(f"  skipped: {report.exr_convert_skipped}")
    lines.append("")
    lines.append("Details:")
    for r in report.results:
        lines.append(
            f"[GID={r.group_id} IDX={r.group_index}] {r.category} | {r.status} | "
            f"{r.src} -> {r.dst or '-'}"
            + (f" | reason={r.reason}" if r.reason else "")
        )
    lines.append("")
    lines.append("EXR2PNG Details:")
    for r in report.exr_results:
        lines.append(
            f"{r.status} | {r.exr} -> {r.png or '-'}"
            + (f" | channel={r.selected_channel}" if r.selected_channel else "")
            + (f" | reason={r.reason}" if r.reason else "")
        )

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("已写入报告：%s", json_path)
    logger.info("已写入映射：%s", csv_path)
    logger.info("已写入 EXR2PNG 映射：%s", exr_csv_path)
    logger.info("已写入文本：%s", txt_path)

    return json_path, csv_path, exr_csv_path, txt_path


# ====================== CLI / Main ======================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter and renumber VFX files, then automatically convert Depth EXR to 16-bit PNG."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="输入根目录路径（支持 .../output/PNG 或 .../Source|Alpha|BaseColor|Normal|Depth 结构）",
    )
    parser.add_argument(
        "--filter",
        required=True,
        help="目标 filter 文件夹路径（会自动创建 Source/Alpha/BaseColor/Normal/Depth 等子目录）",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="复制或移动（默认 copy）",
    )
    parser.add_argument(
        "--conflict",
        choices=["skip", "overwrite", "rename"],
        default="skip",
        help="整理文件时的冲突处理。推荐 rename，避免同组编号冲突。",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="编号位数（默认 4，例如 0001；可设为 6 -> 000001）",
    )
    parser.add_argument(
        "--sort",
        choices=["mtime", "name"],
        default="mtime",
        help="组内排序方式：mtime 或 name（默认 mtime）",
    )
    parser.add_argument(
        "--group-sort",
        choices=["mtime", "id"],
        default="mtime",
        help="组排序方式：mtime 或 id（默认 mtime）",
    )
    parser.add_argument(
        "--prefix-template",
        default="{category}",
        help='命名前缀模板（默认 "{category}"）。可写 "SHOT" 或 "SHOT_{category}" 等。',
    )
    parser.add_argument(
        "--id-regex",
        default=r"^(\d+)",
        help=r'用于从原始ID目录名中提取ID的正则，必须含一个捕获组。默认 "^(\d+)"。',
    )

    # EXR 转换参数：默认开启，满足“提取所有文件后自动调用 exr2png”
    parser.add_argument(
        "--no-convert-exr",
        action="store_true",
        help="关闭整理完成后的 Depth EXR -> PNG 自动转换。",
    )
    parser.add_argument(
        "--exr-convert-conflict",
        choices=["skip", "overwrite"],
        default=DEFAULT_EXR_CONVERT_CONFLICT,
        help="EXR 转 PNG 时，如果同名 PNG 已存在：skip=跳过，overwrite=覆盖。默认 skip。",
    )
    parser.add_argument(
        "--delete-exr-after-convert",
        action="store_true",
        default=DEFAULT_DELETE_EXR_AFTER_CONVERT,
        help="EXR 转 PNG 成功后删除原 EXR。默认不删除。",
    )
    parser.add_argument(
        "--exr-convert-mode",
        choices=["robust_quantile", "fixed_range"],
        default=DEFAULT_CONVERT_MODE,
        help="EXR 深度映射模式。默认 robust_quantile。",
    )
    parser.add_argument("--fixed-min", type=float, default=DEFAULT_FIXED_MIN, help="fixed_range 模式最小深度。")
    parser.add_argument("--fixed-max", type=float, default=DEFAULT_FIXED_MAX, help="fixed_range 模式最大深度。")
    parser.add_argument("--low-q", type=float, default=DEFAULT_LOW_Q, help="robust_quantile 低分位数，默认 0.002。")
    parser.add_argument("--high-q", type=float, default=DEFAULT_HIGH_Q, help="robust_quantile 高分位数，默认 0.998。")
    parser.add_argument(
        "--no-rgb-fallback",
        action="store_true",
        help="找不到标准深度通道时，不允许回退到 RGB/Y/V。默认允许。",
    )
    parser.add_argument(
        "--no-preview-8bit",
        action="store_true",
        help="不生成 8-bit 预览图。默认生成。",
    )
    parser.add_argument(
        "--far-bright",
        action="store_true",
        help="输出远处亮、近处暗。默认输出近处亮、远处暗。",
    )
    parser.add_argument(
        "--debug-exr-channels",
        action="store_true",
        help="转换前打印前几个 EXR 的通道信息。",
    )
    parser.add_argument("--debug-max-files", type=int, default=5, help="EXR 通道调试最多打印几个文件。")

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    filter_dir = Path(args.filter).expanduser().resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = filter_dir / "_reports" / ts
    ensure_dir(report_dir)
    logger = setup_logger(report_dir / "run.log")

    try:
        id_regex = re.compile(args.id_regex)
    except re.error as e:
        logger.error("id-regex 无效：%s (%s)", args.id_regex, e)
        return 2

    start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not root.exists() or not root.is_dir():
        logger.error("root 不存在或不是目录：%s", root)
        return 2
    ensure_dir(filter_dir)

    scan_dirs = find_input_dirs(root)
    logger.info("找到可扫描目录数量：%d", len(scan_dirs))

    tasks, matched_by_category, ambiguous, seen_files = collect_tasks(scan_dirs, id_regex)
    logger.info("扫描到文件总数：%d", seen_files)
    logger.info("匹配到目标类别文件总数：%d", len(tasks))
    logger.info("多类别命中（按优先级取第一个）的数量：%d", ambiguous)

    grouped: Dict[str, List[FileTask]] = {}
    for t in tasks:
        grouped.setdefault(t.group_id, []).append(t)

    logger.info("识别到 group_id 组数：%d", len(grouped))

    processed_groups, processed_by_category, results = process_groups(
        grouped=grouped,
        filter_dir=filter_dir,
        mode=args.mode,
        conflict=args.conflict,
        digits=args.digits,
        sort_mode=args.sort,
        group_sort=args.group_sort,
        prefix_template=args.prefix_template,
        logger=logger,
    )

    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")

    # 关键：整理/提取所有文件完成后，自动调用 exr2png。
    exr_enabled = not args.no_convert_exr
    exr_results: List[ExrConvertResult] = []
    if exr_enabled:
        logger.info("==== 开始自动执行 EXR -> PNG 转换：目标为 filter/Depth 下的 EXR ====")
        exr_results = batch_convert_depth_exr(
            root_dir=filter_dir,
            delete_exr=args.delete_exr_after_convert,
            mode=args.exr_convert_mode,
            fixed_min=args.fixed_min,
            fixed_max=args.fixed_max,
            low_q=args.low_q,
            high_q=args.high_q,
            save_preview_8bit=not args.no_preview_8bit,
            allow_rgb_fallback=not args.no_rgb_fallback,
            output_near_bright=not args.far_bright,
            conflict=args.exr_convert_conflict,
            debug_print_channels=args.debug_exr_channels,
            debug_max_files=args.debug_max_files,
            logger=logger,
        )
    else:
        logger.info("已关闭 EXR -> PNG 自动转换。")

    exr_success = sum(1 for r in exr_results if r.status == "success")
    exr_failed = sum(1 for r in exr_results if r.status == "failed")
    exr_skipped = sum(1 for r in exr_results if r.status == "skipped")

    end = time.time()
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = RunReport(
        start_time=start_time,
        end_time=end_time,
        elapsed_seconds=end - start,
        root=str(root),
        filter_dir=str(filter_dir),
        mode=args.mode,
        conflict=args.conflict,
        digits=args.digits,
        sort=args.sort,
        group_sort=args.group_sort,
        prefix_template=args.prefix_template,
        id_regex=args.id_regex,
        png_dirs_found=len(scan_dirs),
        folders_traversed=len(scan_dirs),
        files_seen=seen_files,
        matched_total=len(tasks),
        matched_by_category=matched_by_category,
        ambiguous_multi_match=ambiguous,
        groups_total=len(grouped),
        processed_groups=processed_groups,
        processed_by_category=processed_by_category,
        success=success,
        failed=failed,
        skipped=skipped,
        exr_convert_enabled=exr_enabled,
        exr_convert_root=str(filter_dir / "Depth"),
        exr_convert_mode=args.exr_convert_mode,
        exr_convert_success=exr_success,
        exr_convert_failed=exr_failed,
        exr_convert_skipped=exr_skipped,
        results=results,
        exr_results=exr_results,
    )

    write_reports(report, report_dir, logger)

    logger.info("==== 运行完成 ====")
    logger.info("Groups=%d | ProcessedGroups=%d", report.groups_total, report.processed_groups)
    logger.info("File Success=%d | Failed=%d | Skipped=%d", success, failed, skipped)
    logger.info("EXR2PNG Success=%d | Failed=%d | Skipped=%d", exr_success, exr_failed, exr_skipped)
    logger.info("Elapsed=%.3fs", report.elapsed_seconds)

    for cat in CATEGORIES_ORDER:
        logger.info("[%s] matched=%d, processed=%d", cat, matched_by_category[cat], processed_by_category[cat])

    # 只要整理文件失败或 EXR 转换失败，就返回 1；否则返回 0。
    return 0 if failed == 0 and exr_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
