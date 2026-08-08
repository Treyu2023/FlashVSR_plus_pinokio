"""Output filenames aligned with AI HTML Toolbox + VSR Pipeline conventions.

Pipeline stage tags (always the last token before the extension):
  _1  = FlashVSR upscale done
  _2  = interpolation / frame-adjust (RIFE) done
  _3  = exported / ready-to-post (social / CIV deliverable)

Examples:
  UpScale4K_clip_upscaled_x4_S_I_1.mp4
  UpScale4K_clip_upscaled_x4_S_I_2.mp4   (after RIFE)
  UpScale4K_clip_upscaled_x4_S_I_3.mp4   (after Export)
"""
from __future__ import annotations

import os
import re

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui_config")

VALID_MODES = ("toolbox", "vsr", "both")

# Pipeline stages — single digit at end of stem for at-a-glance status
STAGE_UPSCALE = 1
STAGE_INTERP = 2
STAGE_POSTED = 3  # export / posted-ready

# Matches trailing _1 / _2 / _3 (not longer numbers like _12 or timestamps)
_STAGE_RE = re.compile(r"_([123])$")


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
    h = max(0, int(height or 0))
    if h >= 4320:
        return "8K"
    if h >= 2160:
        return "4K"
    if h >= 1440:
        return "2K-QHD"
    if h >= 1080:
        return "1080p"
    if h >= 720:
        return "720p"
    return f"{h}p" if h else "4K"


def strip_stage(stem: str) -> tuple[str, int | None]:
    """Remove trailing _1/_2/_3 stage tag. Returns (bare_stem, stage_or_None)."""
    text = str(stem or "")
    match = _STAGE_RE.search(text)
    if not match:
        return text, None
    return text[: match.start()], int(match.group(1))


def get_stage(stem: str) -> int | None:
    _, stage = strip_stage(stem)
    return stage


def with_stage(stem: str, stage: int) -> str:
    """Set absolute stage tag (replaces any existing _1/_2/_3)."""
    stage = int(stage)
    if stage not in (STAGE_UPSCALE, STAGE_INTERP, STAGE_POSTED):
        raise ValueError(f"stage must be 1, 2, or 3 (got {stage})")
    bare, _ = strip_stage(stem)
    bare = bare.rstrip("_")
    return f"{bare}_{stage}"


def promote_stage(stem: str, stage: int) -> str:
    """Apply stage tag, never lowering an existing higher stage."""
    stage = int(stage)
    bare, current = strip_stage(stem)
    final = max(stage, current or 0)
    if final not in (STAGE_UPSCALE, STAGE_INTERP, STAGE_POSTED):
        return bare
    return with_stage(bare, final)


def apply_stage_to_filename(filename: str, stage: int, *, promote: bool = True) -> str:
    """Return filename with stage tag on the stem (keeps extension)."""
    path = str(filename or "")
    if not path:
        return path
    root, ext = os.path.splitext(path)
    # If path has directories, only rewrite the basename stem
    directory, base = os.path.split(root)
    new_stem = promote_stage(base, stage) if promote else with_stage(base, stage)
    rebuilt = new_stem + ext
    return os.path.join(directory, rebuilt) if directory else rebuilt


def _upscale_stem(stem: str, scale: int, *, chunked: bool = False, output_height: int = 0) -> str:
    mode = load_naming_mode()
    bare, _ = strip_stage(stem)
    tier = res_tier_from_height(output_height or scale * 1080)
    if mode == "toolbox":
        base = f"{bare}_upscaled_x{scale}"
    elif mode == "vsr":
        base = f"UpScale{tier}_{bare}_S_I"
    else:
        base = f"UpScale{tier}_{bare}_upscaled_x{scale}_S_I"
    if chunked:
        base += "_chunked"
    # Stage 1 = first upscale complete
    return with_stage(base, STAGE_UPSCALE)


def upscale_video_filename(stem: str, scale: int, *, chunked: bool = False, output_height: int = 0) -> str:
    return f"{_upscale_stem(stem, scale, chunked=chunked, output_height=output_height)}.mp4"


def upscale_image_filename(stem: str, scale: int, *, output_height: int = 0) -> str:
    mode = load_naming_mode()
    bare, _ = strip_stage(stem)
    tier = res_tier_from_height(output_height or scale * 1080)
    if mode == "toolbox":
        base = f"{bare}_upscaled_x{scale}"
    elif mode == "vsr":
        base = f"UpScale{tier}_{bare}_S_I"
    else:
        base = f"UpScale{tier}_{bare}_upscaled_x{scale}_S_I"
    return f"{with_stage(base, STAGE_UPSCALE)}.png"


def upscale_combine_stem(stem: str, scale: int, output_height: int = 0) -> str:
    return _upscale_stem(stem, scale, output_height=output_height)


def comparison_video_filename(stem: str) -> str:
    bare, _ = strip_stage(stem)
    return f"{bare}_comparison.mp4"


def comparison_image_filename(stem: str) -> str:
    bare, _ = strip_stage(stem)
    return f"{bare}_comparison.png"
