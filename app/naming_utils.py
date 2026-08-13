"""
Readable 2-step pipeline filenames (FlashVSR + Toolbox).

Step 1 — Upscale:
  <original>_<1080p|2K|4K>_<9x16>_<Upscaled>.mp4|png
  Example:  myclip_4K_9x16_Upscaled.mp4

Step 2 — Interp + Export (same process step):
  <step1 stem>_<30fps>.mp4
  Example:  myclip_4K_9x16_Upscaled_30fps.mp4
  After step 2 succeeds, the step-1 file is moved to a sibling Bin\\ folder
  so you can delete intermediates when finished.

Legacy _1/_2/_3 tags are still stripped if present on old files.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui_config")

VALID_MODES = ("toolbox", "vsr", "both")  # kept for settings compat; step names ignore mode

# Legacy stage digits (old scheme)
STAGE_UPSCALE = 1
STAGE_INTERP = 2
STAGE_POSTED = 3
_STAGE_RE = re.compile(r"_([123])$")

# New readable tags
_UPSCALED_RE = re.compile(r"_Upscaled$", re.I)
_FPS_RE = re.compile(r"_(\d+)fps$", re.I)
_RES_RE = re.compile(r"_(8K|4K|2K-QHD|2K|1080p|720p|\d+p)$", re.I)
_AR_RE = re.compile(r"_(\d+x\d+)$", re.I)

# junk from old toolbox suffixes
_JUNK_SUFFIX_RE = re.compile(
    r"(_exported_\d+w_\d+q|_frames_[^_]+|_loop_[^_]+|_chunked|_\d{6})+$",
    re.I,
)


def _parse_config_value(value: str):
    text = str(value).strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("/") or text.startswith("\\\\"):
        return os.path.normpath(text)
    try:
        return int(text) if "." not in text else float(text)
    except ValueError:
        return text


def load_naming_mode() -> str:
    mode = "both"
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key.strip() == "naming_mode":
                        mode = str(_parse_config_value(value.strip())).lower()
                        break
        except OSError:
            pass
    return mode if mode in VALID_MODES else "both"


def res_tier_from_height(height: int) -> str:
    """Backward-compat wrapper — prefer res_tier_from_size for accuracy."""
    return res_tier_from_size(0, height)


def res_tier_from_size(width: int, height: int) -> str:
    """Estimate delivery tier from longest edge (works for portrait too)."""
    m = max(int(width or 0), int(height or 0))
    if m <= 0:
        return "4K"
    if m >= 3800:
        return "4K"
    if m >= 2500:
        return "2K"
    if m >= 1800:
        return "1080p"
    if m >= 1200:
        return "720p"
    return f"{m}p"


def aspect_ratio_label(width: int, height: int) -> str:
    """
    Human aspect like 16x9 / 9x16 / 1x1 (colon avoided for filenames).
    Snaps to common ratios when close.
    """
    w, h = int(width or 0), int(height or 0)
    if w <= 0 or h <= 0:
        return "16x9"
    g = math.gcd(w, h) or 1
    rw, rh = w // g, h // g
    # Snap to familiar ratios when within ~3%
    ratio = w / h
    known = [
        (16 / 9, "16x9"),
        (9 / 16, "9x16"),
        (4 / 3, "4x3"),
        (3 / 4, "3x4"),
        (1 / 1, "1x1"),
        (21 / 9, "21x9"),
        (2 / 1, "2x1"),
        (1 / 2, "1x2"),
    ]
    for target, label in known:
        if abs(ratio - target) / target < 0.04:
            return label
    # Cap large reduced fractions (e.g. 1920x1081 → still 16x9-ish above)
    if rw > 32 or rh > 32:
        # approximate with small ints
        for den in range(1, 17):
            num = round(ratio * den)
            if num > 0 and abs(ratio - num / den) / ratio < 0.03:
                return f"{num}x{den}"
        return f"{min(rw, 99)}x{min(rh, 99)}"
    return f"{rw}x{rh}"


def strip_stage(stem: str) -> Tuple[str, Optional[int]]:
    """Legacy: strip trailing _1/_2/_3."""
    text = str(stem or "")
    match = _STAGE_RE.search(text)
    if not match:
        return text, None
    return text[: match.start()], int(match.group(1))


def get_stage(stem: str) -> Optional[int]:
    _, stage = strip_stage(stem)
    return stage


def with_stage(stem: str, stage: int) -> str:
    """Legacy helper — maps 1→Upscaled tag style when possible."""
    stage = int(stage)
    bare, _ = strip_stage(stem)
    bare = bare.rstrip("_")
    if stage == STAGE_UPSCALE:
        if not _UPSCALED_RE.search(bare):
            return f"{bare}_Upscaled"
        return bare
    if stage in (STAGE_INTERP, STAGE_POSTED):
        # Prefer fps tag only when known; fallback keep Upscaled
        return bare if _UPSCALED_RE.search(bare) else f"{bare}_Upscaled"
    return f"{bare}_{stage}"


def promote_stage(stem: str, stage: int) -> str:
    return with_stage(stem, stage)


def apply_stage_to_filename(filename: str, stage: int, *, promote: bool = True) -> str:
    path = str(filename or "")
    if not path:
        return path
    root, ext = os.path.splitext(path)
    directory, base = os.path.split(root)
    new_stem = with_stage(base, stage)
    rebuilt = new_stem + ext
    return os.path.join(directory, rebuilt) if directory else rebuilt


def clean_original_stem(stem: str) -> str:
    """
    Strip pipeline tags so we can re-apply step1 cleanly.
    Keeps the human original name as much as possible.
    """
    text = str(stem or "").strip()
    if not text:
        return "clip"
    # Drop Windows-illegal chars
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    text = text.replace(" ", "_")
    # Remove fps / Upscaled / res / aspect / legacy stage from the right
    changed = True
    while changed:
        changed = False
        for rx in (_FPS_RE, _UPSCALED_RE, _AR_RE, _RES_RE, _STAGE_RE, _JUNK_SUFFIX_RE):
            m = rx.search(text)
            if m:
                text = text[: m.start()].rstrip("_")
                changed = True
    # Old VSR prefixes
    text = re.sub(r"^UpScale(8K|4K|2K-QHD|2K|1080p|720p|\d+p)_", "", text, flags=re.I)
    text = re.sub(r"_upscaled_x\d+", "", text, flags=re.I)
    text = re.sub(r"_S_I", "", text, flags=re.I)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "clip"


def step1_stem(original_stem: str, width: int, height: int, *, chunked: bool = False) -> str:
    """Step 1: original + resolution tier + aspect + Upscaled."""
    bare = clean_original_stem(original_stem)
    tier = res_tier_from_size(width, height)
    ar = aspect_ratio_label(width, height)
    base = f"{bare}_{tier}_{ar}_Upscaled"
    if chunked:
        base += "_chunked"
    return base


def step1_filename(
    original_stem: str,
    width: int,
    height: int,
    *,
    ext: str = ".mp4",
    chunked: bool = False,
) -> str:
    if not ext.startswith("."):
        ext = "." + ext
    return step1_stem(original_stem, width, height, chunked=chunked) + ext.lower()


def step2_stem(source_stem: str, fps: float | int | None) -> str:
    """Step 2: keep step1 readable tags, append approximate FPS."""
    text = str(source_stem or "")
    text, _ = strip_stage(text)
    text = _FPS_RE.sub("", text).rstrip("_")
    text = _JUNK_SUFFIX_RE.sub("", text).rstrip("_")
    # Ensure Upscaled marker remains for clarity
    if not _UPSCALED_RE.search(text):
        text = f"{clean_original_stem(text)}_Upscaled"
    try:
        fps_i = int(round(float(fps))) if fps is not None else 30
    except (TypeError, ValueError):
        fps_i = 30
    fps_i = max(1, min(fps_i, 480))
    return f"{text}_{fps_i}fps"


def step2_filename(source_stem: str, fps: float | int | None, *, ext: str = ".mp4") -> str:
    if not ext.startswith("."):
        ext = "." + ext
    return step2_stem(source_stem, fps) + ext.lower()


def upscale_video_filename(
    stem: str,
    scale: int,
    *,
    chunked: bool = False,
    output_height: int = 0,
    output_width: int = 0,
) -> str:
    """Step 1 video name (used by FlashVSR saves)."""
    h = int(output_height or 0)
    w = int(output_width or 0)
    if h and not w:
        # assume landscape 16:9 if only height known
        w = int(round(h * 16 / 9))
    if w and not h:
        h = int(round(w * 9 / 16))
    if not w and not h:
        # fallback estimate from scale
        h = int(1080 * max(1, int(scale or 4)))
        w = int(round(h * 16 / 9))
    return step1_filename(stem, w, h, ext=".mp4", chunked=chunked)


def upscale_image_filename(
    stem: str,
    scale: int,
    *,
    output_height: int = 0,
    output_width: int = 0,
) -> str:
    h = int(output_height or 0)
    w = int(output_width or 0)
    if h and not w:
        w = h
    if w and not h:
        h = w
    if not w and not h:
        h = int(1024 * max(1, int(scale or 4)))
        w = h
    return step1_filename(stem, w, h, ext=".png", chunked=False)


def upscale_combine_stem(stem: str, scale: int, output_height: int = 0, output_width: int = 0) -> str:
    h = int(output_height or 0)
    w = int(output_width or 0)
    if h and not w:
        w = int(round(h * 16 / 9))
    return step1_stem(stem, w, h, chunked=False)


def comparison_video_filename(stem: str) -> str:
    bare = clean_original_stem(stem)
    return f"{bare}_comparison.mp4"


def comparison_image_filename(stem: str) -> str:
    bare = clean_original_stem(stem)
    return f"{bare}_comparison.png"


def move_to_bin(source_path: str, *, bin_name: str = "Bin") -> Optional[str]:
    """
    Move a step-1 file into <same_dir>/Bin/ after step 2 finishes.
    Returns destination path or None.
    """
    try:
        src = Path(source_path)
        if not src.is_file():
            return None
        # Don't re-bin files already under Bin
        if src.parent.name.lower() == bin_name.lower():
            return str(src)
        bin_dir = src.parent / bin_name
        bin_dir.mkdir(parents=True, exist_ok=True)
        dest = bin_dir / src.name
        if dest.exists():
            dest = bin_dir / f"{src.stem}_{time.strftime('%Y%m%d_%H%M%S')}{src.suffix}"
        shutil.move(str(src), str(dest))
        return str(dest)
    except OSError as e:
        print(f"[naming] move_to_bin failed: {e}")
        return None


def rename_to_step2(path: str, *, source_stem: str, fps: float | int | None, ext: str | None = None) -> str:
    """
    Rename a finished step-2 file to the readable FPS name in the same folder.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return path
        use_ext = ext or p.suffix or ".mp4"
        new_name = step2_filename(source_stem, fps, ext=use_ext)
        dest = p.parent / new_name
        if dest.resolve() == p.resolve():
            return str(p)
        if dest.exists():
            dest = p.parent / f"{Path(new_name).stem}_{time.strftime('%H%M%S')}{use_ext}"
        p.rename(dest)
        return str(dest)
    except OSError as e:
        print(f"[naming] rename_to_step2 failed: {e}")
        return path
