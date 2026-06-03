from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
from config.constants import BACKGROUND_EXTS

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

def make_output_dir_name(
    background_name: Optional[str],
    style_mode: str = "default",
    lighting_pattern: str = "auto",
    key_side: str = "auto",
    include_style_suffix: bool = True,
) -> str:
    if not background_name:
        stem = "default"
    else:
        stem = os.path.splitext(os.path.basename(background_name))[0]
        stem = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in stem).strip('_')
        stem = stem or "default"

    mode = (style_mode or "default").lower()
    pattern = (lighting_pattern or "auto").lower()
    side = (key_side or "auto").lower()
    suffix = ""
    if include_style_suffix and mode not in ("default", "auto", "quality"):
        suffix += f"_{mode}"
    if pattern != "auto":
        suffix += f"_{pattern}"
        if side in ("left", "right") and pattern in ("side", "cinematic", "rembrandt"):
            suffix += f"_{side}"
    return f"output_{stem}{suffix}"

def resolve_user_path(path: Optional[str], script_dir: str, prefer_script_dir: bool = False) -> Optional[str]:
    """Resolve absolute/relative user paths robustly.

    Absolute paths are returned directly. Relative paths are looked up against both
    the current working directory and the directory containing this script.

    prefer_script_dir=True is useful for project-local defaults such as
    configs/presets, so the same command works even when launched from another cwd.
    """
    if not path:
        return None

    path = os.path.expanduser(os.path.expandvars(str(path)))
    if os.path.isabs(path):
        return os.path.abspath(path)

    cwd_candidate = os.path.abspath(path)
    script_candidate = os.path.abspath(os.path.join(script_dir, path))

    candidates = [script_candidate, cwd_candidate] if prefer_script_dir else [cwd_candidate, script_candidate]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # If it does not exist yet, return the first candidate by policy. This is useful
    # for --write-default-presets, where the directory may be created later.
    return candidates[0]
