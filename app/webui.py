import os

# Must be set before torch is imported (reduces CUDA fragmentation OOMs on 24GB GPUs).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

import sys
import argparse
import gradio as gr
from gradio import ImageSlider
import re
import math
import uuid
import torch
import shutil
import imageio
import ffmpeg
import numpy as np
import torch.nn.functional as F
import random
import time
import subprocess
import psutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from huggingface_hub import snapshot_download
from gradio_videoslider import VideoSlider

from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from src.models import wan_video_dit
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj, Causal_LQ4x_Proj

from toolbox.system_monitor import SystemMonitor
from toolbox.toolbox import ToolboxProcessor
from toolbox.batch_queue import (
    BatchQueueManager,
    write_live_batch_progress,
    write_batch_inputs_list,
)
from flashvsr_work_queue import FlashVSRWorkQueue, ExclusiveQueueLock, VIDEO_EXTS, IMAGE_EXTS
import group_therapy as gt
from naming_utils import (
    upscale_video_filename,
    upscale_image_filename,
    move_to_bin,
    rename_to_step2,
    step2_filename,
    upscale_combine_stem,
    comparison_video_filename,
    comparison_image_filename,
    strip_stage,
    with_stage,
)

# Initialize toolbox_processor after load_config is defined
toolbox_processor = None

# Suppress annoyingly persistent Windows asyncio proactor errors
if os.name == 'nt':  # Windows only
    import asyncio
    from functools import wraps
    import socket # Required for the ConnectionResetError
    
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    def silence_connection_errors(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (ConnectionResetError, BrokenPipeError):
                pass
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise
        return wrapper
    
    from asyncio import proactor_events
    if hasattr(proactor_events, '_ProactorBasePipeTransport'):
        proactor_events._ProactorBasePipeTransport._call_connection_lost = silence_connection_errors(
            proactor_events._ProactorBasePipeTransport._call_connection_lost
        )

parser = argparse.ArgumentParser(description="FlashVSR+ WebUI")
parser.add_argument("--listen", action="store_true", help="Allow LAN access")
parser.add_argument("--port", type=int, default=7860, help="Service Port")
args = parser.parse_args()
        
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
TEMP_DIR = os.path.join(ROOT_DIR, "_temp")
CONFIG_FILE = os.path.join(ROOT_DIR, "webui_config")


# --- Machine profile (this PC) — used for UI defaults/tooltips ---
MACHINE = {
    "gpu": "NVIDIA GeForce RTX 4090",
    "vram_gb": 24,
    "cpu": "Intel Core i9-14900K",
    "ram_gb": 64,
    "models_drive": "O:\\MODELS",
    "profile_name": "RTX 4090 · 24GB VRAM · 64GB RAM",
}

def tip(text: str) -> str:
    """Prefix tooltips with machine context when useful."""
    return text

# Hover-tooltip catalog (machine-aware). Keys used in create_ui info= fields.
TIPS = {
    "mode": (
        "Pipeline Mode — your RTX 4090 (24GB):\n"
        "• tiny (default): best speed/VRAM balance for FlashVSR streaming; recommended for almost all video work.\n"
        "• full: higher fidelity but can push or exceed 24GB on long/high-res clips — only try with short clips, "
        "tiled DiT on, unload DiT on, and input width ≤512–768px."
    ),
    "model_version": (
        "Model Version — v1.1 is the current best for stability + detail (Nov 2025 weights on O:\\MODELS\\FlashVSR-v1.1). "
        "Keep v1.1 unless you intentionally compare against older v1.0 checkpoints."
    ),
    "seed": (
        "Seed — fixed seed = reproducible upscale. Leave 0 with Randomize off for a stable baseline; "
        "enable Randomize when you want variety across re-runs of the same clip."
    ),
    "randomize_seed": (
        "Randomize Seed — new seed every run. Off (default) keeps results consistent so you can judge setting changes on your 4090."
    ),
    "scale": (
        "Upscale Factor — model was trained primarily for 4×. On a 4090, 4× at ~1024px input width is the clarity sweet spot. "
        "Use 2× if source is already ≥1080p and you want less VRAM. Output ≈ input × scale (grid-aligned)."
    ),
    "tiled_dit": (
        "Tiled DiT — ON (default for this PC). Splits the diffusion transformer work into tiles to avoid VRAM spikes. "
        "Your 4090 previously OOM'd without careful tiling on longer clips — leave this on for video; "
        "turning off is only for short tiny-mode tests."
    ),
    "tile_size": (
        "Tile Size — default 320. Larger tiles = fewer seams / more detail per tile; uses more VRAM. "
        "If OOM on tall 4× outputs, drop to 256 or lower. Must keep overlap < half of tile size."
    ),
    "tile_overlap": (
        "Tile Overlap — default 32. Softens tile seams without eating too much of each tile. "
        "Must stay < half of tile size (e.g. tile 320 → overlap ≤160). Raise to 48 if you see soft grid seams."
    ),
    "enable_chunks": (
        "Process as Chunks — ON (default). Splits long videos into segments so 64GB system RAM and 24GB VRAM stay healthy. "
        "Keep on for anything longer than ~10–15s at 4×. Disable only for short tests."
    ),
    "chunk_duration": (
        "Max Chunk Duration — default 10s with 1024px inputs (was 12s @ 768). Longer = fewer seams but more peak VRAM. "
        "If free VRAM drops under ~4GB mid-run, lower to 8s. Raise to 12–15s only when VRAM headroom is large."
    ),
    "batch_resize": (
        "Pre-downscale before FlashVSR. Default 4K-safe (auto): math so (input × scale) never exceeds "
        "UHD 4K — 3840×2160 for 16:9 landscape, or 2160×3840 for 9:16 portrait (at 4× → max 960×540 or 540×960). "
        "Only shrinks, never upsizes. Fixed px presets also clamp so 4× stays inside that 4K box. "
        "No Resize = leave source as-is (may exceed 4K after upscale)."
    ),
    "autosave": (
        "Autosave Output — ON saves finished upscales automatically to the FlashVSR output folder. "
        "Batch jobs always save. Off means you must click Save Manually."
    ),
    "comparison": (
        "Create Comparison — side-by-side before/after. Handy for QA; uses extra disk + encode time. "
        "Not available for chunked or batch jobs. Always writes a comparison file when enabled."
    ),
    "clear_on_start": (
        "Clear Temp on Start — ON (default). Wipes FlashVSR temp files before a run so stale frames/caches "
        "do not eat disk space or confuse the pipeline. Safe to leave on with 64GB RAM."
    ),
    "sparse_ratio": (
        "Sparse Ratio — attention sparsity. Default 1.0 = densest attention / max detail (slower). "
        "Higher (1.2–2.0) = faster batch throughput, slightly softer. If output flickers, try 1.2–1.5."
    ),
    "local_range": (
        "Local Range — temporal attention window (odd values). Default 7 = sharper temporal response. "
        "Raise to 9–11 for smoother temporal stability on motion-heavy clips."
    ),
    "quality": (
        "Output Video Quality — encode quality slider (1–10). Default 9 keeps fine detail for CIV/export. "
        "8–10 = near-lossless but large files on 4K-class outputs. 5–7 = smaller previews / drafts."
    ),
    "kv_ratio": (
        "KV Cache Ratio — temporal consistency vs VRAM. 3 (default) is optimal on 24GB. "
        "4+ reduces flicker but costs VRAM; avoid 6–8 unless clips are short and tiled."
    ),
    "fps_override": (
        "Output FPS — only for image-sequence inputs. Ignored for normal video files (source FPS is used). "
        "30 is a sensible default if you feed frame folders."
    ),
    "device": (
        "Device — cuda:0 is your only RTX 4090. Leave as cuda:0. "
        "'cpu' is extremely slow (debug only). 'auto' usually picks the same GPU."
    ),
    "attention_mode": (
        "Attention Mode — sage (default) is recommended when the env supports it; best throughput on Ada GPUs. "
        "If sage fails to load, switch to block. Block is the compatibility fallback."
    ),
    "dtype": (
        "Data Type — bf16 (default) is preferred on RTX 4090 (native bfloat16): more stable than fp16 for diffusion. "
        "fp16 is slightly faster in some ops but can be less stable; use only if debugging speed."
    ),
    "color_fix": (
        "Color Fix — ON corrects color drift from the super-res model. Keep on for real/AI video; "
        "turn off only if you want the raw model look for comparison."
    ),
    "tiled_vae": (
        "Tiled VAE — ON reduces decode VRAM (important after DiT). Small speed cost; keep on for 4× video on 24GB."
    ),
    "unload_dit": (
        "Unload DiT Before Decoding — ON (default). Frees multi-GB VRAM before VAE decode. "
        "This was part of the fix path after hard OOMs on this 4090. Slightly slower; strongly recommended."
    ),
    "trim_start": "Trim Start — seconds from the beginning to keep. Use to drop logos/intros before upscale (saves VRAM/time).",
    "trim_end": "Trim End — end time in seconds (0 = through end of file). Keep clips short when testing full mode.",
    "resize_width": (
        "Target Width — pre-scale input width before FlashVSR (aspect preserved, then grid-aligned). "
        "On this PC, 768–1024px is the practical range for 4× clarity. Higher width ⇒ much more VRAM at 4×."
    ),
    "img_scale": (
        "Image Upscale Factor — model trained for 4×. Single images are lighter than video; 4× is fine on 4090. "
        "Pre-resize huge sources (e.g. 4K stills) down first if VRAM spikes."
    ),
    "tb_pipeline": (
        "Pipeline steps — check Frame Adjust / Video Loop / Export, then Start Pipeline. "
        "Export is for social-sized encodes. Batch requires at least one step checked. "
        "Toolbox final saves go to your Ready for CIV folder on D:."
    ),
    "rife": (
        "RIFE Interpolation — 2× frames (default) smooths motion after upscale. 4× runs RIFE twice (heavier on CPU/GPU). "
        "With i9-14900K + 4090, 2× is the practical default; 4× for hero clips."
    ),
    "speed_factor": (
        "Speed Factor — <1 slows (more frames feel), >1 speeds up. Audio follows. "
        "Ignored when Streaming (Low Memory) RIFE mode is enabled."
    ),
    "frames_quality": (
        "Frame Adjust Output Quality — 95 default (high). CRF-style mapping: 100≈CRF15, 85≈CRF18. "
        "Keep high before final export; lower only for draft previews."
    ),
    "rife_stream": (
        "RIFE Streaming — low-memory path for long videos (does not load all frames into 64GB RAM at once). "
        "Enable for multi-minute clips. Note: Speed Factor is ignored in this mode."
    ),
    "loop_type": "Loop Type — loop = restart; ping-pong = forward then reverse. Good for seamless social loops.",
    "num_loops": "Number of Loops — additional full plays after the first. 1 ⇒ two total plays of the segment.",
    "loop_quality": "Loop encode quality — 85 default. Same CRF-style scale as other toolbox quality sliders.",
    "export_format": (
        "Export Format — H.264 MP4 for max compatibility (Discord/web). H.265 smaller files, slower encode. "
        "WebM/VP9 for web; GIF only for short loops."
    ),
    "export_quality": "Export Quality — 92 default for near-final delivery. Lower to shrink Discord uploads.",
    "export_width": (
        "Max Width — 3840 default keeps full 4K-class FlashVSR outputs. Lower (1920/1280) only when posting size-capped platforms."
    ),
    "export_name": "Optional output filename stem (no extension). Empty = auto name from source + naming mode.",
    "theme": "UI theme — cosmetic only. Interstellar is saved as your preference; restart page after Apply.",
    "custom_theme": "Custom Gradio theme from Hugging Face Spaces (username/theme). Only used when Theme = Custom.",
    "naming_mode": (
        "Legacy setting (kept for compatibility). Real names are now 2-step:\n"
        "  Step 1 Upscale → name_4K_9x16_Upscaled.mp4\n"
        "  Step 2 Interp+Export → name_4K_9x16_Upscaled_30fps.mp4 (+ step-1 moved to Bin\\)"
    ),
    "output_dir": "Legacy alias — use Step 3 (After upscale / Ready for Toolbox). Same as batch_upscale_handoff_dir.",
    "toolbox_output_dir": "Step 6 — final Ready for CIV folder after Toolbox export.",
    "batch_watch_folder": "Step 1 — intake. New downloads drop here; queues scan on Start / Resume.",
    "batch_source_archive_dir": "Step 2 — originals archive after upscale (for pairing).",
    "batch_upscale_handoff_dir": "Step 3 — where upscaled VIDEOS are saved (Ready for Toolbox).",
    "img_upscale_handoff_dir": "Step 4 — where upscaled IMAGES are saved (skip Toolbox → Ready for CIV\\images).",
    "tb_inbox_folder": "Step 5 — Toolbox watches / picks from here (usually same as Step 3).",
    "gt_group_size": (
        "Group Therapy size — how many originals go through one full pipeline pass "
        "before the next group starts. 10 = 10 upscale → 10 RIFE → 10 RIFE → 10 export."
    ),
    "gt_before_dir": "Before folder (flat) — originals land here as name_PID_xxxxxxxx.mp4 (same id as After). Title metadata is also set to PID_xxxxxxxx; Media Center tags are not touched.",
    "gt_after_dir": "After folder (flat) — finals land here as name_PID_xxxxxxxx.mp4 (same id as Before). No per-file subfolders.",
    "folder_path": "Folder of videos/images for batch. Use absolute Windows paths (e.g. D:\\clips\\batch).",
    "img_quality": "Still image encode quality. Higher = larger files. 7 default matches video profile.",
    "img_fps": "Unused for still images (placeholder control shared with video advanced block).",
    "two_pass": "Two-Pass Encoding — better compression at same quality; slower. Hidden/experimental for long clips.",
}

os.environ['GRADIO_TEMP_DIR'] = TEMP_DIR

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ── Clear pipeline steps (readable + configurable) ─────────────────────────
# Away from the old "app\\outputs only" archetype: every production step lives
# under D:\OUTPUTS\__X_GROK\... with a plain-English name.
WORKFLOW_DEFAULTS = {
    "batch_watch_folder": r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS",
    "batch_source_archive_dir": r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos",
    "batch_upscale_handoff_dir": r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox",
    "img_upscale_handoff_dir": r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV\images",
    "tb_inbox_folder": r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox",
    "toolbox_output_dir": r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV",
}

# (step#, short title, config key, what lands here)
WORKFLOW_STEPS = (
    (1, "Intake / watch", "batch_watch_folder", "New downloads · queue picks up from here"),
    (2, "Originals archive", "batch_source_archive_dir", "Pre-upscale sources moved here for pairing"),
    (3, "After upscale (videos)", "batch_upscale_handoff_dir", "FIRST save · upscaled videos · Ready for Toolbox"),
    (4, "After upscale (images)", "img_upscale_handoff_dir", "Upscaled images (skip RIFE) · Ready for CIV\\images"),
    (5, "Toolbox inbox", "tb_inbox_folder", "Toolbox reads from here (usually = step 3)"),
    (6, "Final / Ready for CIV", "toolbox_output_dir", "After Toolbox export · done for CIV"),
)

STAGE_NAME_LEGEND = (
    "<b>Step 1 (Upscale):</b> <code>name_4K_9x16_Upscaled.mp4</code> &nbsp;·&nbsp; "
    "<b>Step 2 (Interp+Export):</b> <code>name_4K_9x16_Upscaled_30fps.mp4</code> "
    "(step-1 file moves to <code>Bin\\</code>)"
)


def _abs_path_or_default(raw: str, fallback: str) -> str:
    text = str(raw or "").strip()
    if not text:
        text = fallback
    text = os.path.normpath(text)
    if os.path.isabs(text):
        try:
            os.makedirs(text, exist_ok=True)
        except OSError:
            pass
        return text
    # Non-absolute → fall back to known good default
    fb = os.path.normpath(fallback)
    try:
        os.makedirs(fb, exist_ok=True)
    except OSError:
        pass
    return fb


def get_workflow_paths(config: Optional[dict] = None) -> dict:
    """Resolved absolute paths for every pipeline step (config + defaults)."""
    cfg = load_config() if config is None else dict(config)
    out = {}
    for key, default in WORKFLOW_DEFAULTS.items():
        raw = cfg.get(key, default)
        if raw is None or str(raw).strip() == "":
            raw = default
        out[key] = _abs_path_or_default(raw, default)
    # output_dir is an alias of step 3 (upscale videos) — never default to app\\outputs
    legacy = str(cfg.get("output_dir", "") or "").strip()
    if legacy and os.path.isabs(legacy) and os.path.normpath(legacy) != os.path.normpath(DEFAULT_OUTPUT_DIR):
        out["output_dir"] = _abs_path_or_default(legacy, out["batch_upscale_handoff_dir"])
    else:
        out["output_dir"] = out["batch_upscale_handoff_dir"]
    return out


def get_output_dir():
    """
    Step 3 — where upscaled VIDEOS are saved (Ready for Toolbox).
    No longer defaults to app\\outputs; that folder is only for queue logs/temp.
    """
    paths = get_workflow_paths()
    return paths["output_dir"] or paths["batch_upscale_handoff_dir"]


def get_image_output_dir():
    """Step 4 — where upscaled IMAGES are saved."""
    return get_workflow_paths()["img_upscale_handoff_dir"]


def get_toolbox_output_dir():
    """Step 6 — Toolbox final save (Ready for CIV)."""
    return get_workflow_paths()["toolbox_output_dir"]


def get_queue_log_dir():
    """Internal status/logs only (not production deliverables)."""
    log_root = os.path.join(DEFAULT_OUTPUT_DIR, "queue_logs")
    os.makedirs(log_root, exist_ok=True)
    return log_root


def workflow_paths_html(config: Optional[dict] = None) -> str:
    """Dark readable map of every step → folder (shown in UI + Settings)."""
    paths = get_workflow_paths(config)
    rows = []
    for num, title, key, blurb in WORKFLOW_STEPS:
        p = paths.get(key, "")
        rows.append(
            f"<tr>"
            f"<td style='padding:6px 10px;color:#7dd3fc;font-weight:700;white-space:nowrap;'>Step {num}</td>"
            f"<td style='padding:6px 10px;color:#e2e8f0;font-weight:600;'>{title}</td>"
            f"<td style='padding:6px 10px;color:#94a3b8;font-size:0.88em;'>{blurb}</td>"
            f"<td style='padding:6px 10px;'><code style='color:#cbd5e1;background:#0b1220;"
            f"padding:2px 6px;border-radius:4px;font-size:0.82em;'>{p}</code></td>"
            f"</tr>"
        )
    name_row = (
        f"<div style='margin-top:8px;font-size:0.85em;color:#94a3b8;'>"
        f"Name tags: {STAGE_NAME_LEGEND}</div>"
    )
    process_blurb = (
        "<div style='margin-top:10px;padding:8px 10px;background:#0b1220;border-radius:6px;"
        "border:1px solid #334155;font-size:0.88em;line-height:1.45;'>"
        "<b style='color:#7dd3fc;'>Process steps (only 2):</b><br>"
        "① <b>Upscale</b> → name like <code style='color:#86efac;'>clip_4K_9x16_Upscaled.mp4</code> "
        "into Ready for Toolbox<br>"
        "② <b>Interp + Export</b> → <code style='color:#fbbf24;'>clip_4K_9x16_Upscaled_30fps.mp4</code> "
        "into Ready for CIV · Step‑1 file moves to <code>Ready for Toolbox\\Bin\\</code><br>"
        "Group Therapy pairing: same files stay <b>flat</b> in Before/After with "
        "<code style='color:#86efac;'>_PID_xxxxxxxx</code> on the name (Title metadata too, not tags)."
        "</div>"
    )
    return (
        "<div class='fvsr-workflow-map' style='margin:0 0 12px 0;padding:12px 14px;"
        "border-radius:8px;border:1px solid #2d3748;background:linear-gradient(135deg,#0f1419,#1a2332);"
        "color:#e2e8f0;'>"
        "<div style='font-weight:700;color:#7dd3fc;margin-bottom:8px;'>"
        "📁 Folders + 2-step naming (what actually saves where)</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
        + "".join(rows)
        + "</table>"
        + process_blurb
        + name_row
        + "<div style='margin-top:6px;font-size:0.78em;color:#64748b;'>"
        "Edit any path in ⚙️ Settings → folders are selectable. "
        "Queue STATUS logs stay under app\\outputs\\work_queue_* (not deliverables).</div>"
        "</div>"
    )


def _apply_toolbox_output_dir():
    """Sync live toolbox processor with config."""
    if toolbox_processor is None:
        return
    toolbox_processor.output_dir = Path(get_toolbox_output_dir())

def unique_dest_path(dest_dir: str, filename: str) -> str:
    """Avoid overwriting when two outputs share a basename."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if not os.path.exists(dest):
        return dest
    stem, ext = os.path.splitext(filename)
    return os.path.join(dest_dir, f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}")


def _path_is_under(path: str, root: str) -> bool:
    try:
        p = os.path.abspath(path)
        r = os.path.abspath(root)
        return os.path.commonpath([p, r]) == r
    except (ValueError, OSError):
        return False


def finalize_output_once(src_path: str, dest_dir: str) -> str:
    """
    Ensure the deliverable exists once under dest_dir.

    Bug this fixes: toolbox autosave already writes into Ready for CIV, then the
    queue called unique_dest_path() which saw the file, invented a longer
    ``_YYYYMMDD_HHMMSS`` name, and copied again — two full-size copies per job.
    """
    if not src_path or not os.path.isfile(src_path):
        return src_path or ""
    os.makedirs(dest_dir, exist_ok=True)
    src_abs = os.path.abspath(src_path)
    dest_dir_abs = os.path.abspath(dest_dir)
    name = os.path.basename(src_abs)
    dest = os.path.join(dest_dir_abs, name)

    # Already the final path — nothing to do
    if os.path.normcase(src_abs) == os.path.normcase(dest):
        return dest

    # Already under the destination folder (autosave wrote it there) — do NOT re-copy
    if _path_is_under(src_abs, dest_dir_abs):
        return src_abs

    # Need to place into dest_dir from temp / other drive
    if not os.path.exists(dest):
        try:
            # Prefer move out of temp to avoid a second full file
            if _path_is_under(src_abs, TEMP_DIR) or _path_is_under(
                src_abs, os.path.join(ROOT_DIR, "_temp")
            ):
                shutil.move(src_abs, dest)
            else:
                shutil.copy2(src_abs, dest)
        except OSError:
            shutil.copy2(src_abs, dest)
        return dest

    # Name collision with a different file already in dest
    try:
        if os.path.getsize(src_abs) == os.path.getsize(dest):
            # Same size → treat as already delivered; drop temp source
            if _path_is_under(src_abs, TEMP_DIR) or _path_is_under(
                src_abs, os.path.join(ROOT_DIR, "_temp")
            ):
                try:
                    os.remove(src_abs)
                except OSError:
                    pass
            return dest
    except OSError:
        pass

    # True clash: only then invent a longer unique name (one copy, not two of the same)
    unique = unique_dest_path(dest_dir_abs, name)
    try:
        if _path_is_under(src_abs, TEMP_DIR) or _path_is_under(
            src_abs, os.path.join(ROOT_DIR, "_temp")
        ):
            shutil.move(src_abs, unique)
        else:
            shutil.copy2(src_abs, unique)
            # leave non-temp source alone
    except OSError:
        shutil.copy2(src_abs, unique)
    return unique


def get_gradio_allowed_paths():
    """Absolute paths Gradio may serve (required for outputs on other drives, e.g. D:)."""
    config = load_config()
    ui = get_ui_defaults(config)
    candidates = [
        ROOT_DIR,
        TEMP_DIR,
        os.path.join(TEMP_DIR, "toolbox"),
        DEFAULT_OUTPUT_DIR,
        get_output_dir(),
        get_toolbox_output_dir(),
        str(config.get("output_dir", "")).strip(),
        str(config.get("toolbox_output_dir", "")).strip(),
        ui.get("batch_watch_folder", ""),
        ui.get("batch_source_archive_dir", ""),
        ui.get("batch_upscale_handoff_dir", ""),
        ui.get("img_upscale_handoff_dir", ""),
        ui.get("tb_inbox_folder", ""),
        r"D:\OUTPUTS\__X_GROK",
        os.environ.get("TMPDIR", ""),
        os.environ.get("TEMP", ""),
        os.environ.get("TMP", ""),
    ]
    allowed = []
    seen = set()
    for path in candidates:
        if not path:
            continue
        path = os.path.normpath(os.path.abspath(path))
        if path in seen or not os.path.isabs(path):
            continue
        seen.add(path)
        allowed.append(path)
    return allowed

# For backward compatibility, OUTPUT_DIR is now a function call
# Use get_output_dir() throughout the code for dynamic resolution
OUTPUT_DIR = DEFAULT_OUTPUT_DIR  # Initial value, will be updated dynamically

def _parse_config_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text.lower() in ('true', 'false'):
        return text.lower() == 'true'
    if re.match(r'^[A-Za-z]:[\\/]', text) or text.startswith('/') or text.startswith('\\\\'):
        return os.path.normpath(text)
    try:
        return int(text) if '.' not in text else float(text)
    except ValueError:
        return text

def load_config():
    """Load user preferences from config file."""
    config = {"clear_temp_on_start": False, "autosave": True, "tb_autosave": True, "naming_mode": "both"}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    config[key.strip()] = _parse_config_value(value.strip())
        except:
            pass
    return config

def get_ui_defaults(config=None):
    """Load UI control defaults from webui_config."""
    if config is None:
        config = load_config()
    # Fallbacks when webui_config key is missing (user profile defaults).
    specs = {
        "chunk_duration": (10.0, float),
        "enable_chunks": (True, bool),
        "tiled_dit": (True, bool),
        "tiled_vae": (True, bool),
        "unload_dit": (True, bool),
        "tile_size": (320, int),
        "tile_overlap": (32, int),
        "attention_mode": ("sage", str),
        "sparse_ratio": (1.0, float),
        "local_range": (7, int),
        "kv_ratio": (3, int),
        "quality": (9, int),
        "randomize_seed": (False, bool),
        "scale": (4, int),
        "mode": ("tiny", str),
        "model_version": ("v1.1", str),
        "dtype": ("bf16", str),
        "color_fix": (True, bool),
        "fps_override": (30, int),
        "device": ("cuda:0", str),
        # 1024×4 ≈ 4096 wide — keeps more source detail than 768 (was crushing Grok clips)
        "batch_resize_preset": ("4K-safe (auto)", str),
        "batch_watch_folder": (r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS", str),
        "batch_source_archive_dir": (
            r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos",
            str,
        ),
        # Next step after FlashVSR video upscale (hand-sort + toolbox inbox)
        "batch_upscale_handoff_dir": (
            r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox",
            str,
        ),
        # Image upscales skip RIFE → go straight to Ready for CIV
        "img_upscale_handoff_dir": (
            r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV\images",
            str,
        ),
        "tb_inbox_folder": (
            r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox",
            str,
        ),
        "tb_fps_mode": ("4x Frames", str),
        "tb_max_out_fps": (120, int),
        "tb_high_fps_floor": (160, int),
        "tb_scale_back_fps": (60, int),
        "tb_pipeline_ops": ("Frame Adjust,Export", str),
        "tb_frames_quality": (98, int),
        "tb_export_quality": (96, int),
        "tb_export_max_width": (3840, int),
        # Quality-first toolbox export (slower encode, sharper finals)
        "tb_export_preset": ("slow", str),
        "tb_prefer_nvenc": (True, bool),
        "gt_group_size": (10, int),
        "gt_do_upscale": (True, bool),
        "gt_do_rife1": (True, bool),
        "gt_do_rife2": (True, bool),
        "gt_do_export": (True, bool),
        "gt_before_dir": (
            r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos",
            str,
        ),
        "gt_after_dir": (
            r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV",
            str,
        ),
    }
    defaults = {}
    for key, (fallback, typ) in specs.items():
        raw = config.get(key, fallback)
        if typ is bool:
            defaults[key] = raw if isinstance(raw, bool) else str(raw).lower() == "true"
        elif typ is int:
            defaults[key] = int(float(raw))
        elif typ is float:
            defaults[key] = float(raw)
        else:
            defaults[key] = str(raw)
    return defaults

def input_align_step(scale, output_multiple=128):
    """Input pixel step so (dimension * scale) lands on the model's output grid."""
    g = math.gcd(int(scale), output_multiple)
    return output_multiple // g

def codec_align_step(scale, macro_block=16):
    """Input pixel step so (dimension * scale) is divisible by macro_block (H.264/RIFE)."""
    g = math.gcd(int(scale), macro_block)
    return macro_block // g

def resize_align_step(scale, output_multiple=128):
    """Combined grid: model output alignment + codec macro-block alignment."""
    return math.lcm(input_align_step(scale, output_multiple), codec_align_step(scale))

def crop_to_scaled_dimensions(tensor, src_h, src_w, scale):
    """Center-crop upscaled output to exact src dimensions × scale (avoids tile-boundary trim)."""
    target_h = int(src_h) * int(scale)
    target_w = int(src_w) * int(scale)
    out_h, out_w = tensor.shape[1], tensor.shape[2]
    if out_h < target_h or out_w < target_w:
        log(
            f"Warning: upscaled {out_w}×{out_h} smaller than target {target_w}×{target_h}",
            message_type="warning",
        )
        return tensor
    crop_top = (out_h - target_h) // 2
    crop_left = (out_w - target_w) // 2
    cropped = tensor[:, crop_top:crop_top + target_h, crop_left:crop_left + target_w, :]
    aligned = (target_w % 16 == 0 and target_h % 16 == 0)
    log(
        f"Output dimensions: {target_w}×{target_h}"
        + (" (codec-safe, divisible by 16)" if aligned else ""),
        message_type="info",
    )
    return cropped

def is_no_resize_preset(preset) -> bool:
    s = str(preset or "").strip().lower()
    return s in ("", "no resize", "none", "off")


def is_4k_safe_preset(preset) -> bool:
    s = str(preset or "").strip().lower().replace("_", "-")
    return s in (
        "4k-safe (auto)",
        "4k-safe",
        "4k safe",
        "auto 4k",
        "auto-4k",
        "uhd-safe",
        "4k_safe",
    )


def parse_batch_resize_preset(preset, scale=None):
    """
    Returns (mode, max_width) where mode is 'none' | '4k_safe' | 'width'.
    Width presets still get a 4K post-scale clamp inside calculate_resize_dimensions.
    """
    if is_no_resize_preset(preset):
        return "none", None
    if is_4k_safe_preset(preset):
        return "4k_safe", None
    raw = str(preset).strip().lower().replace("px", "")
    try:
        return "width", int(float(raw))
    except (TypeError, ValueError):
        # Unknown label → safe default
        return "4k_safe", None


PIPELINE_MODES = ("tiny", "full", "tiny-long")


def normalize_pipeline_mode(mode, default="tiny"):
    """
    FlashVSR inference mode only. Never reuse this name for batch-resize
    presets (4k_safe / width) — those are a different 'mode' and will load
    the tiny-long writer with output_path=None (imageio URI error).
    """
    m = str(mode or "").strip().lower()
    if m in PIPELINE_MODES:
        return m
    return default


def apply_batch_resize_preset(video_path, batch_resize_preset, scale=None, progress=None):
    """Resize video so upscale×scale stays within UHD 4K (or legacy width cap + 4K clamp)."""
    if not video_path or is_no_resize_preset(batch_resize_preset):
        return video_path
    if scale is None:
        scale = get_ui_defaults()["scale"]
    mode, max_width = parse_batch_resize_preset(batch_resize_preset, scale=scale)
    if mode == "none":
        return video_path
    current_width, current_height = get_video_dimensions(video_path)
    new_width, new_height, will_resize = calculate_resize_dimensions(
        current_width,
        current_height,
        max_width=max_width,
        scale=scale,
        mode=mode,
    )
    if not will_resize:
        out_w, out_h = int(current_width) * int(scale), int(current_height) * int(scale)
        log(
            f"Video {current_width}×{current_height} already 4K-safe at {scale}× "
            f"(→ {out_w}×{out_h}) — no resize",
            message_type="info",
        )
        return video_path
    align = resize_align_step(scale)
    out_w, out_h = new_width * int(scale), new_height * int(scale)
    log(
        f"Resizing video {current_width}×{current_height} → {new_width}×{new_height} "
        f"(align {align}px, {scale}× → {out_w}×{out_h} within UHD 4K)",
        message_type="info",
    )
    if progress is None:
        class _NoProgress:
            def __call__(self, *args, **kwargs):
                pass
        progress = _NoProgress()
    return resize_input_video(
        video_path, max_width, scale=scale, progress=progress, mode=mode
    )

def save_config(config):
    """Save user preferences to config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}={value}\n")
    except Exception as e:
        log(f"Error saving config: {e}", message_type="error")

def log(message:str, message_type:str="normal"):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

def dummy_tqdm(iterable, *args, **kwargs):
    return iterable

def model_download(model_version="v1.0"):
    """Download FlashVSR models from HuggingFace. Supports v1.0 and v1.1."""
    if model_version == "v1.1":
        model_name = "JunhaoZhuang/FlashVSR-v1.1"
        model_dir = os.path.join(ROOT_DIR, "models", "FlashVSR-v1.1")
    else:  # v1.0
        model_name = "JunhaoZhuang/FlashVSR"
        model_dir = os.path.join(ROOT_DIR, "models", "FlashVSR")
    
    # Check if critical model files exist
    required_files = [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "Wan2.1_VAE.pth",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt"
    ]
    
    needs_download = not os.path.exists(model_dir)
    if not needs_download:
        # Check if all required files exist
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        needs_download = len(missing_files) > 0
        if needs_download:
            log(f"Incomplete {model_version} model files detected. Re-downloading...", message_type='warning')
    
    if needs_download:
        log(f"Downloading {model_version} model '{model_name}' from huggingface...", message_type='info')
        try:
            # snapshot_download will automatically resume interrupted downloads
            # and skip already downloaded files
            snapshot_download(
                repo_id=model_name, 
                local_dir=model_dir,
                local_dir_use_symlinks=False  # Keep for compatibility, warnings are harmless
            )
            log(f"{model_version} model download complete!", message_type='finish')
            print()
        except Exception as e:
            log(f"Error downloading models: {e}", message_type='error')
            log("Please check your internet connection and try again.", message_type='warning')
            raise

def check_model_status(model_version="v1.0"):
    """Check if models need to be downloaded and return appropriate status message."""
    if model_version == "v1.1":
        model_dir = os.path.join(ROOT_DIR, "models", "FlashVSR-v1.1")
    else:
        model_dir = os.path.join(ROOT_DIR, "models", "FlashVSR")
    
    # Check if directory exists AND contains the critical model files
    required_files = [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "Wan2.1_VAE.pth",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt"
    ]
    
    if not os.path.exists(model_dir):
        return f'<div style="padding: 8px; background: #3d2e0a; border: 1px solid #854d0e; border-radius: 6px; color: #fbbf24; font-size: 0.95em;">⏳ First-time setup: Downloading {model_version} models (~6-7GB) from HuggingFace. This may take several minutes depending on your connection. Please be patient and check the terminal for progress...</div>'
    
    # Check if all required files exist
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing_files:
        return f'<div style="padding: 8px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5; font-size: 0.95em;">⚠️ Incomplete model files detected: Re-downloading missing {model_version} model(s). Previous download may have been interrupted. Please be patient and check the terminal for progress...</div>'
    
    return gr.update()  # Return no update if models exist and are complete

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def clean_video_filename(filename, max_length=80):
    """
    Cleans video filenames to prevent path length issues while preserving operation chain.
    - KEEPS preprocessing suffixes (_resized_, _trim_, _preprocessed_) to show operation history
    - REMOVES timestamps from preprocessing steps to prevent length accumulation
    - PRESERVES pipeline stage tags _1 / _2 / _3 (stripped only so upscale renames cleanly)
    - Truncates to max_length characters while preserving readability
    """
    # Stage tags are re-applied by upscale/toolbox naming; strip while cleaning source stems
    bare, _stage = strip_stage(filename)
    filename = bare

    # Remove timestamps from preprocessing (format: _YYYYMMDD_HHMMSS or _HHMMSS)
    # These accumulate with each operation and cause length issues
    filename = re.sub(r'_\d{8}_\d{6}', '', filename)
    filename = re.sub(r'_\d{6}', '', filename)

    # Clean up multiple underscores that may result from timestamp removal
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_')

    # Truncate to max_length while preserving some readability
    if len(filename) > max_length:
        # Keep the first max_length characters
        filename = filename[:max_length]
        # Remove trailing underscore if present
        filename = filename.rstrip('_')

    return filename

def clean_image_filename(filename, max_length=80):
    """
    Cleans image filenames to prevent path length issues while preserving operation chain.
    - KEEPS preprocessing suffixes (_resized_, _preprocessed_) to show operation history
    - REMOVES timestamps from preprocessing steps to prevent length accumulation
    - Stage tags stripped so upscale naming can re-apply _1 cleanly
    - Truncates to max_length characters
    """
    bare, _stage = strip_stage(filename)
    filename = bare

    # Remove timestamps from preprocessing (format: _YYYYMMDD_HHMMSS)
    filename = re.sub(r'_\d{8}_\d{6}', '', filename)

    # Clean up multiple underscores that may result from timestamp removal
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_')

    # Truncate to max_length while preserving some readability
    if len(filename) > max_length:
        # Keep the first max_length characters
        filename = filename[:max_length]
        # Remove trailing underscore if present
        filename = filename.rstrip('_')

    return filename

def largest_8n1_leq(n):
    # Find largest value of form 8n+1 that is <= n
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def smallest_8n1_geq(n):
    # Find smallest value of form 8n+1 that is >= n (rounds up to preserve frames)
    if n < 1:
        return 1
    # If n is already 8k+1, return n
    if (n - 1) % 8 == 0:
        return n
    # Otherwise round up
    return ((n - 1)//8 + 1)*8 + 1

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def is_ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def save_video(frames, save_path, fps=30, quality=5, progress_desc="Saving video..."):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with imageio.get_writer(save_path, fps=fps, quality=quality, macro_block_size=1) as writer:
        for i in tqdm(range(frames.shape[0]), desc=f"[FlashVSR] {progress_desc}"):
            frame_np = (frames[i].cpu().float() * 255.0).clip(0, 255).numpy().astype(np.uint8)
            writer.append_data(frame_np)

def prepare_tensors(path: str, dtype=torch.bfloat16):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0: raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0: w0, h0 = _img0.size
        frames = [torch.from_numpy(np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0).to(dtype) for p in tqdm(paths0, desc="Loading images")]
        return torch.stack(frames, 0), 30
    if is_video(path):
        with imageio.get_reader(path) as rdr:
            meta = rdr.get_meta_data()
            fps = meta.get('fps', 30)
            # Explicitly convert to numpy array to avoid NumPy 2.0 deprecation warning
            frames = [torch.from_numpy(np.asarray(frame_data, dtype=np.float32) / 255.0).to(dtype) for frame_data in tqdm(rdr, desc="Loading video frames")]
        return torch.stack(frames, 0), fps
    raise ValueError(f"Unsupported input: {path}")

def get_input_params(image_tensor, scale):
    N0, h0, w0, _ = image_tensor.shape
    # Dimensions must be multiples of 128 for proper processing:
    # - VAE downsamples by 8x (latent space is height//8, width//8)
    # - DiT patch embedding has stride (1,2,2) -> height//16, width//16
    # - Window partition requires (height//16) % 8 == 0 and (width//16) % 8 == 0
    # - Therefore: height % 128 == 0 and width % 128 == 0
    multiple = 128
    # Calculate scaled dimensions
    scaled_w = w0 * scale
    scaled_h = h0 * scale
    
    # Round UP to nearest multiple of 128 to ensure we never have negative padding
    # This adds small black borders instead of distorting the image
    import math
    tW = math.ceil(scaled_w / multiple) * multiple
    tH = math.ceil(scaled_h / multiple) * multiple
    
    # Ensure minimum size
    tW = max(multiple, tW)
    tH = max(multiple, tH)
    
    # Log padding info if significant
    pad_w = tW - scaled_w
    pad_h = tH - scaled_h
    if pad_w > 0 or pad_h > 0:
        log(f"Adding padding to preserve aspect ratio: {int(scaled_w)}x{int(scaled_h)} → {tW}x{tH} (padding: {int(pad_w)}px width, {int(pad_h)}px height)", message_type='info')
    
    # Use smallest_8n1_geq to round UP and preserve all frames
    F = smallest_8n1_geq(N0 + 4)
    if F == 0: raise RuntimeError(f"Not enough frames. Got {N0 + 4}.")
    return tH, tW, F

def input_tensor_generator(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, Fs = get_input_params(image_tensor, scale)
    
    # Calculate padding needed to reach target dimensions
    scaled_h = h0 * scale
    scaled_w = w0 * scale
    pad_h = tH - scaled_h
    pad_w = tW - scaled_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    for i in range(Fs):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
        # Resize to exact scaled dimensions (preserves aspect ratio)
        upscaled_tensor = F.interpolate(tensor_bchw, size=(scaled_h, scaled_w), mode='bicubic', align_corners=False)
        # Pad to reach target dimensions (multiple of 128)
        if pad_h > 0 or pad_w > 0:
            upscaled_tensor = F.pad(upscaled_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        tensor_out = (upscaled_tensor.squeeze(0) * 2.0 - 1.0)
        yield tensor_out.to('cpu').to(dtype)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, Fs = get_input_params(image_tensor, scale)
    
    # Calculate padding needed to reach target dimensions
    scaled_h = h0 * scale
    scaled_w = w0 * scale
    pad_h = tH - scaled_h
    pad_w = tW - scaled_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    frames = []
    for i in range(Fs):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
        # Resize to exact scaled dimensions (preserves aspect ratio)
        upscaled_tensor = F.interpolate(tensor_bchw, size=(scaled_h, scaled_w), mode='bicubic', align_corners=False)
        # Pad to reach target dimensions (multiple of 128)
        if pad_h > 0 or pad_w > 0:
            upscaled_tensor = F.pad(upscaled_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        tensor_out = (upscaled_tensor.squeeze(0) * 2.0 - 1.0).to('cpu').to(dtype)
        frames.append(tensor_out)
    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    clean_vram()
    return vid_final, tH, tW, Fs

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    stride = tile_size - overlap
    num_rows, num_cols = math.ceil((height - overlap) / stride), math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + tile_size, height), min(x1 + tile_size, width)
            if y2 - y1 < tile_size: y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size: x1 = max(0, x2 - tile_size)
            coords.append((x1, y1, x2, y2))
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    return mask

def stitch_video_tiles(
    tile_paths,
    tile_coords,
    final_dims,
    scale,
    overlap,
    output_path,
    fps,
    quality,
    cleanup=True,
    chunk_size=40
):
    if not tile_paths:
        log("No tile videos found to stitch.", message_type='error')
        return

    final_W, final_H = final_dims

    readers = [imageio.get_reader(p) for p in tile_paths]

    try:
        num_frames = readers[0].count_frames()
        if num_frames is None or num_frames <= 0:
            num_frames = len([_ for _ in readers[0]])
            for r in readers: r.close()
            readers = [imageio.get_reader(p) for p in tile_paths]

        with imageio.get_writer(output_path, fps=fps, quality=quality, macro_block_size=1) as writer:
            for start_frame in tqdm(range(0, num_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):
                end_frame = min(start_frame + chunk_size, num_frames)
                current_chunk_size = end_frame - start_frame
                chunk_canvas = np.zeros((current_chunk_size, final_H, final_W, 3), dtype=np.float32)
                weight_canvas = np.zeros_like(chunk_canvas, dtype=np.float32)

                for i, reader in enumerate(readers):
                    try:
                        tile_chunk_frames = [
                            frame.astype(np.float32) / 255.0
                            for idx, frame in enumerate(reader.iter_data())
                            if start_frame <= idx < end_frame
                        ]
                        tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                    except Exception as e:
                        log(f"Warning: Could not read chunk from tile {i}. Error: {e}", message_type='warning')
                        continue

                    if tile_chunk_np.shape[0] != current_chunk_size:
                        log(f"Warning: Tile {i} chunk has incorrect frame count. Skipping.", message_type='warning')
                        continue

                    tile_H, tile_W, _ = tile_chunk_np.shape[1:]
                    ramp = np.linspace(0, 1, overlap * scale, dtype=np.float32)
                    mask = np.ones((tile_H, tile_W, 1), dtype=np.float32)
                    mask[:, :overlap*scale, :] *= ramp[np.newaxis, :, np.newaxis]
                    mask[:, -overlap*scale:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
                    mask[:overlap*scale, :, :] *= ramp[:, np.newaxis, np.newaxis]
                    mask[-overlap*scale:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
                    mask_4d = mask[np.newaxis, :, :, :]

                    x1_orig, y1_orig, _, _ = tile_coords[i]
                    out_y1, out_x1 = y1_orig * scale, x1_orig * scale
                    out_y2, out_x2 = out_y1 + tile_H, out_x1 + tile_W

                    chunk_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile_chunk_np * mask_4d
                    weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_4d

                weight_canvas[weight_canvas == 0] = 1.0
                stitched_chunk = chunk_canvas / weight_canvas

                for frame_idx_in_chunk in range(current_chunk_size):
                    frame_uint8 = (np.clip(stitched_chunk[frame_idx_in_chunk], 0, 1) * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)

    finally:
        log("Closing all tile reader instances...")
        for reader in readers:
            reader.close()

    if cleanup:
        log("Cleaning up temporary tile files...")
        for path in tile_paths:
            try:
                os.remove(path)
            except OSError as e:
                log(f"Could not remove temporary file '{path}': {e}", message_type='warning')


def create_side_by_side_comparison(input_path, output_path, comparison_output_path):
    """
    Creates a side-by-side comparison video with input on left and output on right.
    Uses FFmpeg's hstack filter for horizontal stacking.
    Scales both videos to match the output video's height.
    """
    if not is_ffmpeg_available():
        log("[FlashVSR] FFmpeg not found. Cannot create side-by-side comparison.", message_type='warning')
        return None
    
    try:
        log("[FlashVSR] Creating side-by-side comparison...", message_type='info')
        
        # Build FFmpeg command for side-by-side comparison
        # Use scale2ref to scale input to match output's height, then hstack
        # Force even dimensions for H.264 compatibility using -2 (auto-calculate to even number)
        # [0:v] is input (to be scaled), [1:v] is output (reference - the larger one)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-i', output_path,
            '-filter_complex',
            '[0:v][1:v]scale2ref=-2:ih[left][right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]',
            '-map', '1:a?',  # Use audio from output video if available
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            comparison_output_path
        ]
        
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        log(f"[FlashVSR] Side-by-side comparison created: {comparison_output_path}", message_type='finish')
        return comparison_output_path
        
    except subprocess.CalledProcessError as e:
        log(f"[FlashVSR] Error creating side-by-side comparison: {e}", message_type='error')
        if e.stderr:
            log(f"FFmpeg stderr: {e.stderr}", message_type='error')
        return None
    except Exception as e:
        log(f"[FlashVSR] Unexpected error creating comparison: {e}", message_type='error')
        return None

def merge_video_with_audio(video_only_path, audio_source_path, output_path):
    """
    Merges the video from video_only_path with audio from audio_source_path into output_path.
    Provides clean, concise logging and gracefully handles errors.
    """
    if not is_ffmpeg_available():
        shutil.move(video_only_path, output_path)
        log("[FlashVSR] FFmpeg not found. The video has been processed without audio.", message_type='warning')
        return

    try:
        # Check if the source video has an audio stream
        probe = ffmpeg.probe(audio_source_path)
        if not any(s['codec_type'] == 'audio' for s in probe.get('streams', [])):
            shutil.move(video_only_path, output_path)
            log("[FlashVSR] No audio stream found in the source. The video has been processed without audio.", message_type='info')
            return
    except ffmpeg.Error:
        # If probing fails, we can't get the audio.
        shutil.move(video_only_path, output_path)
        log("[FlashVSR] Could not probe source for audio. The video has been processed without audio.", message_type='warning')
        return

    try:
        # Perform the merge
        input_video = ffmpeg.input(video_only_path)
        input_audio = ffmpeg.input(audio_source_path)
        ffmpeg.output(
            input_video['v'],
            input_audio['a'],
            output_path,
            vcodec='copy',
            acodec='copy'
        ).run(overwrite_output=True, quiet=True)

        # Never leave an audio-only file as the "upscaled" deliverable
        try:
            probe_out = ffmpeg.probe(output_path)
            if not any(s.get("codec_type") == "video" for s in probe_out.get("streams", [])):
                log(
                    "[FlashVSR] Audio merge produced no video stream — keeping silent video instead.",
                    message_type="warning",
                )
                if os.path.isfile(output_path):
                    os.remove(output_path)
                shutil.move(video_only_path, output_path)
                return
        except Exception:
            pass

        log("[FlashVSR] Audio successfully merged.", message_type='finish')

    except ffmpeg.Error:
        # If the merge operation fails, save the silent video.
        shutil.move(video_only_path, output_path)
        log("[FlashVSR] Audio merge failed. The video has been processed without audio.", message_type='warning')

    finally:
        # Clean up the source video-only file if it still exists
        if os.path.exists(video_only_path):
            try:
                os.remove(video_only_path)
            except OSError as e:
                log(f"[FlashVSR] Could not remove temporary file '{video_only_path}': {e}", message_type='error')

def save_file_manually(temp_path):
    if not temp_path or not os.path.exists(temp_path):
        log("Error: No file to save.", message_type="error")
        return '<div style="padding: 1px; background-color: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 4px; color: #fca5a5;">❌ No file to save.</div>'
    
    filename = os.path.basename(temp_path)
    output_dir = get_output_dir()
    
    # Determine if it's an image or video based on extension
    ext = os.path.splitext(filename)[1].lower()
    is_image = ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
    
    # Save to appropriate subfolder
    if is_image:
        images_output_dir = get_image_output_dir()
        os.makedirs(images_output_dir, exist_ok=True)
        final_path = os.path.join(images_output_dir, filename)
    else:
        final_path = os.path.join(output_dir, filename)
    
    try:
        shutil.copy(temp_path, final_path)
        log(f"File saved to: {final_path}", message_type="finish")
        return f'<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ File saved to: {final_path}</div>'
    except Exception as e:
        log(f"Error saving file: {e}", message_type="error")
        return f'<div style="padding: 1px; background-color: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 4px; color: #fca5a5;">❌ Error saving file: {e}</div>'

def clear_temp_files():
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)
            log("Temp files cleared.", message_type="finish")
            return '<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ Temp files cleared.</div>'
        else:
            log("Temp directory doesn't exist.", message_type="info")
            return '<div style="padding: 1px; background-color: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc;">ℹ️ Temp directory doesn\'t exist.</div>'
    except Exception as e:
        log(f"Error clearing temp files: {e}", message_type="error")
        return f'<div style="padding: 1px; background-color: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 4px; color: #fca5a5;">❌ Error clearing temp files: {e}</div>'
    

def init_pipeline(mode, device, dtype, model_version="v1.0"):
    """Initialize FlashVSR pipeline with specified model version (v1.0 or v1.1)."""
    resolved = normalize_pipeline_mode(mode)
    if resolved != str(mode or "").strip().lower():
        log(
            f"[FlashVSR] Invalid pipeline mode {mode!r} — using {resolved!r} "
            f"(resize presets like 4k_safe are not pipeline modes)",
            message_type="warning",
        )
    mode = resolved
    model_download(model_version=model_version)
    
    # Select model path and projection class based on version
    if model_version == "v1.1":
        model_path = os.path.join(ROOT_DIR, "models", "FlashVSR-v1.1")
        proj_class = Causal_LQ4x_Proj  # v1.1 uses causal projection for improved stability
        log(f"Initializing FlashVSR v1.1 ({mode} mode) - Enhanced stability + fidelity", message_type='info')
    else:  # v1.0
        model_path = os.path.join(ROOT_DIR, "models", "FlashVSR")
        proj_class = Buffer_LQ4x_Proj  # v1.0 uses original buffer projection
        log(f"Initializing FlashVSR v1.0 ({mode} mode)", message_type='info')
    
    ckpt_path, vae_path, lq_path, tcd_path, prompt_path = [os.path.join(model_path, f) for f in ["diffusion_pytorch_model_streaming_dmd.safetensors", "Wan2.1_VAE.pth", "LQ_proj_in.ckpt", "TCDecoder.ckpt", "../posi_prompt.pth"]]
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path]); pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
    else:
        mm.load_models([ckpt_path]); pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device) if mode == "tiny" else FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device=device, dtype=dtype, new_latent_channels=16+768)
        pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device, weights_only=False), strict=False); pipe.TCDecoder.clean_mem()
    
    # Use version-specific projection class
    pipe.denoising_model().LQ_proj_in = proj_class(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    if os.path.exists(lq_path): pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu", weights_only=False), strict=True)
    pipe.to(device, dtype=dtype); pipe.enable_vram_management(); pipe.init_cross_kv(prompt_path=prompt_path); pipe.load_models_to_device(["dit", "vae"])
    return pipe

def is_cuda_oom(exc):
    msg = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in msg
        or "cudaerrormemoryallocation" in msg
    )

def oom_recovery_hint():
    return (
        "CUDA ran out of VRAM. Try: Tile Size 128, enable Unload DiT + Tiled VAE, "
        "enable chunk processing, use batch resize 512px, or Restart FlashVSR in Pinokio "
        "to reclaim stuck GPU memory after a previous OOM."
    )

def get_vram_free_mb():
    """Return free VRAM in MB, or None if CUDA status is unavailable/poisoned."""
    if not torch.cuda.is_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 2)
    except Exception:
        return None

def cuda_context_poisoned(min_free_mb=1500):
    """
    After a hard CUDA OOM the driver often leaves nearly all VRAM allocated and
    mem_get_info itself can fail. Further model reloads then fail immediately.
    """
    free_mb = get_vram_free_mb()
    if free_mb is None:
        return True
    return free_mb < min_free_mb

def log_vram_status(tag=""):
    """Log free/allocated CUDA memory for diagnostics."""
    if not torch.cuda.is_available():
        return
    try:
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        prefix = f"{tag} " if tag else ""
        log(
            f"[FlashVSR] {prefix}VRAM free={free / 1024**2:.0f}MB "
            f"alloc={allocated / 1024**2:.0f}MB reserved={reserved / 1024**2:.0f}MB "
            f"total={total / 1024**2:.0f}MB",
            message_type="info",
        )
    except Exception as e:
        log(f"[FlashVSR] VRAM status unavailable: {e}", message_type="warning")

def ensure_vram_headroom(min_free_mb=2000):
    """Clear caches and warn when free VRAM is too low to start a run."""
    clean_vram()
    if not torch.cuda.is_available():
        return True
    try:
        free, total = torch.cuda.mem_get_info()
        free_mb = free / (1024 ** 2)
        log_vram_status("pre-run")
        if free_mb < min_free_mb:
            log(
                f"[FlashVSR] Low free VRAM ({free_mb:.0f}MB of {total / 1024**2:.0f}MB). "
                "Close other GPU apps or Restart FlashVSR in Pinokio to reclaim memory "
                "left by a previous OOM, then retry.",
                message_type="warning",
            )
            return False
        return True
    except Exception:
        return True

def oom_fallback_profiles(tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit):
    """Progressive VRAM-saving settings to retry after OOM (user settings first)."""
    profiles = []
    seen = set()

    def add(tv, td, ts, to, ud, label):
        ts = int(ts)
        to = int(to)
        if td:
            to = min(to, max(8, ts // 4))
            if to > ts / 2:
                to = max(8, ts // 4)
        key = (bool(tv), bool(td), ts, to, bool(ud))
        if key in seen:
            return
        seen.add(key)
        profiles.append({
            "tiled_vae": bool(tv),
            "tiled_dit": bool(td),
            "tile_size": ts,
            "tile_overlap": to,
            "unload_dit": bool(ud),
            "label": label,
        })

    add(tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, "user")
    # Mid step keeps clarity longer than jumping straight to 128
    add(True, True, min(int(tile_size), 192), min(int(tile_overlap), 32), True, "safe")
    add(True, True, 128, 16, True, "max_save")
    return profiles

def release_pipeline(pipe):
    """Fully offload and drop a pipeline so the next chunk/tile starts with a clean GPU."""
    if pipe is None:
        clean_vram()
        return
    try:
        if hasattr(pipe, "offload_model"):
            pipe.offload_model(keep_vae=False)
    except Exception as e:
        log(f"[FlashVSR] Pipeline offload warning: {e}", message_type="warning")
    try:
        if hasattr(pipe, "dit") and hasattr(pipe.dit, "LQ_proj_in"):
            pipe.dit.LQ_proj_in.clear_cache()
    except Exception as e:
        log(f"[FlashVSR] Pipeline cache warning: {e}", message_type="warning")
    try:
        if hasattr(pipe, "TCDecoder") and hasattr(pipe.TCDecoder, "clean_mem"):
            pipe.TCDecoder.clean_mem()
    except Exception as e:
        log(f"[FlashVSR] TCDecoder cleanup warning: {e}", message_type="warning")
    # Prefer deleting GPU-resident modules over pipe.to("cpu") after OOM —
    # moving tensors can re-enter a poisoned CUDA context and leave VRAM stuck.
    for attr in ("dit", "vae", "TCDecoder", "text_encoder", "image_encoder", "scheduler"):
        try:
            mod = getattr(pipe, attr, None)
            if mod is None:
                continue
            if hasattr(mod, "cpu"):
                try:
                    mod.cpu()
                except Exception:
                    pass
            try:
                delattr(pipe, attr)
            except Exception:
                setattr(pipe, attr, None)
            del mod
        except Exception:
            pass
    try:
        if hasattr(pipe, "to"):
            pipe.to("cpu")
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    clean_vram()
    log_vram_status("after-release")

# --- Integrated Core Logic Function (Updated) ---
def run_flashvsr_single(
    input_path,
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    autosave,
    create_comparison=False,
    progress=gr.Progress(track_tqdm=True)
):
    if not input_path:
        log("No input video provided.", message_type='warning')
        return None, None, None
    mode = normalize_pipeline_mode(mode)

    # --- Parameter Preparation ---
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}; dtype = dtype_map.get(dtype_str, torch.bfloat16)
    devices = get_device_list(); _device = device
    if device == "auto": _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    if _device not in devices and _device != "cpu": raise gr.Error(f"Device '{_device}' is not available! Available devices: {devices}")
    if _device.startswith("cuda"): torch.cuda.set_device(_device)
    if tiled_dit and (tile_overlap > tile_size / 2): raise gr.Error("The overlap must be less than half of the tile size!")
    wan_video_dit.USE_BLOCK_ATTN = (attention_mode == "block")

    # --- Output Path ---
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    input_basename = clean_video_filename(input_basename)  # Clean filename to prevent length issues
    input_w, input_h = get_video_dimensions(input_path)
    output_filename = upscale_video_filename(
        input_basename,
        scale,
        output_height=int(input_h or 0) * scale,
        output_width=int(input_w or 0) * scale,
    )
    output_dir = get_output_dir()
    output_path = os.path.join(output_dir, output_filename)
    temp_video_path = os.path.join(TEMP_DIR, f"video_only_{output_filename}")
    final_output_location = os.path.join(output_dir, output_filename) if autosave else os.path.join(TEMP_DIR, output_filename)

    # Reclaim leftover GPU memory before loading frames (common after prior OOM).
    ensure_vram_headroom(min_free_mb=2000)

    # --- Core Logic ---
    progress(0, desc="Loading video frames...")
    log(f"Loading frames from {input_path}...", message_type='info')
    frames, original_fps = prepare_tensors(input_path, dtype=dtype)
    _fps = original_fps if is_video(input_path) else fps_override
    if frames.shape[0] < 21: raise gr.Error(f"Input must have at least 21 frames, but got {frames.shape[0]} frames.")
    log("Video frames loaded successfully.", message_type="finish")

    final_output_tensor = None
    profiles = oom_fallback_profiles(tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit)
    last_err = None
    success = False

    for attempt_idx, profile in enumerate(profiles):
        tiled_vae = profile["tiled_vae"]
        tiled_dit = profile["tiled_dit"]
        tile_size = profile["tile_size"]
        tile_overlap = profile["tile_overlap"]
        unload_dit = profile["unload_dit"]

        if attempt_idx > 0:
            log(
                f"[FlashVSR] OOM retry {attempt_idx}/{len(profiles) - 1}: profile '{profile['label']}' "
                f"(tiled_dit={tiled_dit}, tiled_vae={tiled_vae}, unload_dit={unload_dit}, "
                f"tile={tile_size}/{tile_overlap})",
                message_type="warning",
            )
            clean_vram()
            log_vram_status(f"retry-{attempt_idx}")
            # Hard CUDA OOMs often leave the context unusable; reloading the DiT
            # just fails instantly and burns time on every remaining batch item.
            if cuda_context_poisoned(min_free_mb=1500):
                log(
                    "[FlashVSR] GPU memory still exhausted after OOM cleanup "
                    f"({get_vram_free_mb() or 0:.0f}MB free). Aborting retries — "
                    "Restart FlashVSR in Pinokio to reclaim stuck VRAM.",
                    message_type="error",
                )
                raise gr.Error(oom_recovery_hint())

        log(
            f"[FlashVSR] VRAM settings: tiled_dit={tiled_dit}, tiled_vae={tiled_vae}, "
            f"unload_dit={unload_dit}, tile={tile_size}/{tile_overlap}",
            message_type="info",
        )

        # Build a common pipe parameter dictionary
        pipe_kwargs = {
            "prompt": "", "negative_prompt": "", "cfg_scale": 1.0, "num_inference_steps": 1,
            "seed": seed, "tiled": tiled_vae, "is_full_block": False, "if_buffer": True,
            "kv_ratio": kv_ratio, "local_range": local_range, "color_fix": color_fix,
            "unload_dit": unload_dit, "fps": _fps, "tiled_dit": tiled_dit,
        }

        pipe = None
        try:
            if tiled_dit:
                N, H, W, C = frames.shape
                tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
                num_tiles = len(tile_coords)
                progress(0.1, desc="Initializing model pipeline...")
                pipe = init_pipeline(mode, _device, dtype, model_version=model_version)

                if mode == "tiny-long":
                    local_temp_dir = os.path.join(TEMP_DIR, str(uuid.uuid4()))
                    os.makedirs(local_temp_dir, exist_ok=True)
                    temp_videos = []
                    for i in tqdm(range(num_tiles), desc="[FlashVSR] Processing tiles"):
                        tile_progress = 0.1 + (i / num_tiles) * 0.75
                        progress(tile_progress, desc=f"Processing tiles: {i+1}/{num_tiles}")

                        x1, y1, x2, y2 = tile_coords[i]
                        input_tile = frames[:, y1:y2, x1:x2, :]
                        temp_name = os.path.join(local_temp_dir, f"{i+1:05d}.mp4")
                        th, tw, F = get_input_params(input_tile, scale)
                        LQ_tile = input_tensor_generator(input_tile, _device, scale=scale, dtype=dtype)
                        pipe(
                            LQ_video=LQ_tile, num_frames=F, height=th, width=tw,
                            topk_ratio=sparse_ratio*768*1280/(th*tw),
                            quality=quality, output_path=temp_name, **pipe_kwargs
                        )
                        temp_videos.append(temp_name)
                        del LQ_tile, input_tile
                        clean_vram()

                    progress(0.85, desc="Stitching tiles...")
                    stitch_video_tiles(temp_videos, tile_coords, (W*scale, H*scale), scale, tile_overlap, temp_video_path, _fps, quality, True)
                    shutil.rmtree(local_temp_dir)
                else:
                    num_aligned_frames = N
                    expected_H = max(128, round(H * scale / 128) * 128) + 128
                    expected_W = max(128, round(W * scale / 128) * 128) + 128
                    final_output_canvas = torch.zeros((num_aligned_frames, expected_H, expected_W, C), dtype=torch.float32)
                    weight_sum_canvas = torch.zeros((num_aligned_frames, expected_H, expected_W, C), dtype=torch.float32)

                    for i in tqdm(range(num_tiles), desc="[FlashVSR] Processing tiles"):
                        tile_progress = 0.1 + (i / num_tiles) * 0.75
                        progress(tile_progress, desc=f"Processing tiles: {i+1}/{num_tiles}")

                        x1, y1, x2, y2 = tile_coords[i]
                        input_tile = frames[:, y1:y2, x1:x2, :]
                        tile_h_in, tile_w_in = y2 - y1, x2 - x1

                        LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
                        LQ_tile = LQ_tile.to(_device)
                        output_tile_gpu = pipe(
                            LQ_video=LQ_tile, num_frames=F, height=th, width=tw,
                            topk_ratio=sparse_ratio*768*1280/(th*tw), **pipe_kwargs
                        )
                        processed_tile_cpu = tensor2video(output_tile_gpu).cpu()
                        processed_tile_cpu = processed_tile_cpu[:num_aligned_frames]

                        tile_h_out, tile_w_out = processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]
                        x1_s = x1 * scale
                        y1_s = y1 * scale
                        expected_tile_w = tile_w_in * scale
                        expected_tile_h = tile_h_in * scale
                        offset_x = (tile_w_out - expected_tile_w) // 2
                        offset_y = (tile_h_out - expected_tile_h) // 2
                        x1_s = max(0, x1_s - offset_x)
                        y1_s = max(0, y1_s - offset_y)
                        x2_s = min(x1_s + tile_w_out, expected_W)
                        y2_s = min(y1_s + tile_h_out, expected_H)
                        tile_w_actual = x2_s - x1_s
                        tile_h_actual = y2_s - y1_s
                        processed_tile_cpu = processed_tile_cpu[:, :tile_h_actual, :tile_w_actual, :]
                        mask = create_feather_mask((tile_h_actual, tile_w_actual), tile_overlap * scale).cpu().permute(0, 2, 3, 1)
                        final_output_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += processed_tile_cpu * mask
                        weight_sum_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += mask
                        del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile, mask
                        clean_vram()

                    weight_sum_canvas[weight_sum_canvas == 0] = 1.0
                    final_output_tensor = final_output_canvas / weight_sum_canvas
                    final_output_tensor = crop_to_scaled_dimensions(final_output_tensor, H, W, scale)
                    del final_output_canvas, weight_sum_canvas
            else:  # Non-tiled mode
                progress(0.1, desc="Initializing model pipeline...")
                pipe = init_pipeline(mode, _device, dtype, model_version=model_version)
                log(f"Processing {frames.shape[0]} frames...", message_type='info')

                N, H, W, C = frames.shape
                th, tw, F = get_input_params(frames, scale)
                if mode == "tiny-long":
                    progress(0.2, desc="Processing video...")
                    LQ = input_tensor_generator(frames, _device, scale=scale, dtype=dtype)
                    pipe(
                        LQ_video=LQ, num_frames=F, height=th, width=tw,
                        topk_ratio=sparse_ratio*768*1280/(th*tw),
                        output_path=temp_video_path, quality=quality, **pipe_kwargs
                    )
                else:
                    progress(0.2, desc="Processing video...")
                    LQ, _, _, _ = prepare_input_tensor(frames, _device, scale=scale, dtype=dtype)
                    LQ = LQ.to(_device)
                    progress(0.3, desc="Running model inference...")
                    video = pipe(
                        LQ_video=LQ, num_frames=F, height=th, width=tw,
                        topk_ratio=sparse_ratio*768*1280/(th*tw), **pipe_kwargs
                    )
                    progress(0.8, desc="Converting output...")
                    final_output_tensor = tensor2video(video).cpu()
                    final_output_tensor = final_output_tensor[:frames.shape[0]]
                    final_output_tensor = crop_to_scaled_dimensions(final_output_tensor, H, W, scale)
                    del video
            success = True
            break
        except Exception as e:
            last_err = e
            if is_cuda_oom(e) and attempt_idx < len(profiles) - 1:
                log(
                    f"[FlashVSR] CUDA OOM on profile '{profile['label']}'. "
                    f"Will retry with safer VRAM settings.",
                    message_type="warning",
                )
                continue
            if is_cuda_oom(e):
                raise gr.Error(oom_recovery_hint()) from e
            raise
        finally:
            release_pipeline(pipe)
            clean_vram()
            if last_err is not None and is_cuda_oom(last_err) and not success:
                # Extra pass after dropping the pipeline; still-low free VRAM means
                # the CUDA context needs a full process restart.
                if cuda_context_poisoned(min_free_mb=1500):
                    log(
                        "[FlashVSR] VRAM still stuck after pipeline release. "
                        "Restart FlashVSR before the next video.",
                        message_type="warning",
                    )

    if not success:
        if last_err is not None and is_cuda_oom(last_err):
            raise gr.Error(oom_recovery_hint()) from last_err
        if last_err is not None:
            raise last_err
        raise gr.Error(oom_recovery_hint())

    if final_output_tensor is not None:
        progress(0.9, desc="Saving final video...")
        del frames
        clean_vram()
        save_video(final_output_tensor, temp_video_path, fps=_fps, quality=quality)
        del final_output_tensor
        clean_vram()

    # Always save to temp directory first (persists during session)
    temp_output_path = os.path.join(TEMP_DIR, output_filename)

    if is_video(input_path):
        progress(0.95, desc="Merging audio...")
        merge_video_with_audio(temp_video_path, input_path, temp_output_path)
    else:
        shutil.move(temp_video_path, temp_output_path)
    
    # Create side-by-side comparison if requested
    comparison_path = None
    if create_comparison and is_video(input_path):
        progress(0.97, desc="Creating side-by-side comparison...")
        comparison_filename = comparison_video_filename(input_basename)
        comparison_temp_path = os.path.join(TEMP_DIR, comparison_filename)
        comparison_path = create_side_by_side_comparison(input_path, temp_output_path, comparison_temp_path)
        
        # Always save comparison video when it's created (regardless of autosave state)
        if comparison_path:
            comparison_save_path = os.path.join(output_dir, comparison_filename)
            shutil.copy(comparison_path, comparison_save_path)
            log(f"Side-by-side comparison saved to: {comparison_save_path}", message_type="finish")
    
    # Autosave upscaled output to outputs folder if enabled
    if autosave:  
        final_save_path = os.path.join(output_dir, output_filename)
        shutil.copy(temp_output_path, final_save_path)
        log(f"Processing complete! Auto-saved to: {final_save_path}", message_type="finish")
        status_msg = f'<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ Processing complete! Auto-saved to: {final_save_path}</div>'
    else:
        log(f"Processing complete! Use 'Save Output' to save to outputs folder.", message_type="finish")
        status_msg = '<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ Processing complete! Use \'Save Output\' to save to outputs folder.</div>'
    
    progress(1, desc="Done!")
    
    # Always display the upscaled output video (not the comparison)
    # This makes the manual save button behavior consistent
    return (
        temp_output_path,  # Display the upscaled output
        temp_output_path,  # Path for manual save
        (input_path, temp_output_path),  # Video slider comparison
        status_msg  # Status message for UI
    )


def analyze_output_video(video_path):
    """Analyzes output video and returns compact HTML display with visibility update."""
    if not video_path:
        return gr.update(visible=False)
    
    try:
        resolved_path = str(Path(video_path).resolve())
        
        # Get file size
        file_size_display = "N/A"
        if os.path.exists(resolved_path):
            size_bytes = os.path.getsize(resolved_path)
            if size_bytes < 1024**2:
                file_size_display = f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                file_size_display = f"{size_bytes/1024**2:.1f} MB"
            else:
                file_size_display = f"{size_bytes/1024**3:.2f} GB"
        
        # Try imageio for quick analysis
        reader = imageio.get_reader(resolved_path)
        meta = reader.get_meta_data()
        
        # Extract info
        duration = meta.get('duration', 0)
        fps = meta.get('fps', 30)
        size = meta.get('size', (0, 0))
        width, height = int(size[0]), int(size[1]) if isinstance(size, tuple) else (0, 0)
        
        # Frame count
        nframes = meta.get('nframes')
        if nframes and nframes != float('inf'):
            frame_count = int(nframes)
        elif duration and fps:
            frame_count = int(duration * fps)
        else:
            frame_count = 0
        
        reader.close()
        
        # Build compact HTML display (same styling as input)
        html = f'''
        <div style="padding: 16px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border: 1px solid #667eea40; border-radius: 8px; font-family: 'Segoe UI', sans-serif;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #1a2838 0%, rgba(26, 40, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">RESOLUTION</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{width}×{height}</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e1a38 0%, rgba(30, 26, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FRAMES</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{frame_count}</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a2838 0%, rgba(26, 40, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">DURATION</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{duration:.2f}s @ {fps:.1f} FPS</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e1a38 0%, rgba(30, 26, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FILE SIZE</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{file_size_display}</div>
                </div>
            </div>
        </div>
        '''
        return gr.update(value=html, visible=True)
        
    except Exception as e:
        error_html = f'<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ Error analyzing output: {str(e)}</div>'
        return gr.update(value=error_html, visible=True)


def analyze_output_image(image_path):
    """Analyzes output image and returns compact HTML display with visibility update."""
    if not image_path:
        return gr.update(visible=False)
    
    try:
        resolved_path = str(Path(image_path).resolve())
        
        # Get file size
        file_size_display = "N/A"
        if os.path.exists(resolved_path):
            size_bytes = os.path.getsize(resolved_path)
            if size_bytes < 1024**2:
                file_size_display = f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                file_size_display = f"{size_bytes/1024**2:.1f} MB"
            else:
                file_size_display = f"{size_bytes/1024**3:.2f} GB"
        
        # Load image to get dimensions
        img = Image.open(resolved_path)
        width, height = img.size
        
        # Calculate megapixels
        megapixels = (width * height) / 1_000_000
        
        # Build compact HTML display (same styling as input)
        html = f'''
        <div style="padding: 16px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border: 1px solid #667eea40; border-radius: 8px; font-family: 'Segoe UI', sans-serif;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #1a2838 0%, rgba(26, 40, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">RESOLUTION</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{width}×{height}</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e1a38 0%, rgba(30, 26, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">MEGAPIXELS</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{megapixels:.2f} MP</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a2838 0%, rgba(26, 40, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FILE SIZE</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{file_size_display}</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e1a38 0%, rgba(30, 26, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FORMAT</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{img.format or 'Unknown'}</div>
                </div>
            </div>
        </div>
        '''
        return gr.update(value=html, visible=True)
        
    except Exception as e:
        error_html = f'<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ Error analyzing output: {str(e)}</div>'
        return gr.update(value=error_html, visible=True)


def analyze_input_image(image_path):
    """Analyzes image and returns compact HTML display for Image Upscaling tab."""
    if not image_path:
        return '<div style="padding: 12px; background: #3d2e0a; border: 1px solid #854d0e; border-radius: 6px; color: #fbbf24;">⚠️ No image provided</div>', 0, 0
    
    try:
        resolved_path = str(Path(image_path).resolve())
        
        # Get file size
        file_size_display = "N/A"
        if os.path.exists(resolved_path):
            size_bytes = os.path.getsize(resolved_path)
            if size_bytes < 1024**2:
                file_size_display = f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                file_size_display = f"{size_bytes/1024**2:.1f} MB"
            else:
                file_size_display = f"{size_bytes/1024**3:.2f} GB"
        
        # Load image to get dimensions
        img = Image.open(resolved_path)
        width, height = img.size
        
        # Calculate megapixels
        megapixels = (width * height) / 1_000_000
        
        # Build compact HTML display (2-column layout for images)
        html = f'''
        <div style="padding: 16px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border: 1px solid #667eea40; border-radius: 8px; font-family: 'Segoe UI', sans-serif;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #1a2838 0%, rgba(26, 40, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">RESOLUTION</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{width}×{height}</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e1a38 0%, rgba(30, 26, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">MEGAPIXELS</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{megapixels:.2f} MP</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a2838 0%, rgba(26, 40, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FILE SIZE</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{file_size_display}</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e1a38 0%, rgba(30, 26, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FORMAT</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{img.format or 'Unknown'}</div>
                </div>
            </div>
            <div style="font-size: 0.8em; color: #666; text-align: center; margin-top: 8px;">
                ℹ️ Model requires output frame dimensions in multiples of 128px. We pad input frames to maintain aspect ratio. Padding is removed during upscale processing.
            </div>
        </div>
        '''
        return html, width, height
        
    except Exception as e:
        return f'<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ Error analyzing image: {str(e)}</div>', 0, 0


def get_image_dimensions(image_path):
    """Get image dimensions quickly. Returns (width, height) or (0, 0) on error."""
    try:
        if not image_path or not os.path.exists(image_path):
            return 0, 0
        img = Image.open(image_path)
        return img.size
    except:
        return 0, 0


def preview_image_resize(image_path, max_width):
    """Generate preview text showing what resize will do for images."""
    if not image_path:
        return '<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">No image loaded</div>'
    
    current_width, current_height = get_image_dimensions(image_path)
    if current_width == 0:
        return '<div style="padding: 8px; background: #3d2e0a; border: 1px solid #854d0e; border-radius: 4px; color: #fbbf24; font-size: 0.9em; text-align: center;">⚠️ Could not read image dimensions</div>'
    
    # Use even dimensions (aspect ratio preserved, padding to 128 handled during upscaling)
    new_width, new_height, will_resize = calculate_resize_dimensions(current_width, current_height, max_width)
    
    # Check if image is small enough to not need tiled DiT
    pixels = current_width * current_height
    small_image_threshold = 512 * 512  # ~512p or smaller
    
    if will_resize:
        reduction = ((current_width * current_height - new_width * new_height) / (current_width * current_height)) * 100
        return f'<div style="padding: 8px; background: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac; font-size: 0.9em; text-align: center;">{current_width}×{current_height} → {new_width}×{new_height} ({reduction:.0f}% reduction) ✓</div>'
    else:
        if pixels <= small_image_threshold:
            return f'<div style="padding: 8px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.9em; text-align: center;">{current_width}×{current_height} (no resize needed) ✓<br><span style="color: #0c5460; font-size: 0.9em;">💡 Small resolution - consider disabling Tiled DiT for better speed and quality</span></div>'
        else:
            return f'<div style="padding: 8px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.9em; text-align: center;">{current_width}×{current_height} (no resize needed) ✓</div>'


def center_crop_cover_pil(img, target_w, target_h):
    """Scale image to cover target box, then center-crop (no aspect distortion)."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    cover_w = max(target_w, int(round(src_w * scale)))
    cover_h = max(target_h, int(round(src_h * scale)))
    resized = img.resize((cover_w, cover_h), Image.LANCZOS)
    left = (cover_w - target_w) // 2
    top = (cover_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def resize_input_image(image_path, max_width, scale=4, progress=gr.Progress(), mode=None):
    """
    Resizes image for FlashVSR preprocessing using PIL.
    Never upsizes - only downsizes if needed.
    mode="4k_safe" fits so (size × scale) stays within UHD 4K 16:9 / 9:16.
    Center-crops to the upscale grid (no stretch).
    Returns path to resized image (or original if no resize needed).
    """
    if not image_path or not os.path.exists(image_path):
        log("No image provided for resize", message_type="warning")
        return image_path
    
    current_width, current_height = get_image_dimensions(image_path)
    if mode is None and isinstance(max_width, str) and is_4k_safe_preset(max_width):
        mode = "4k_safe"
        max_width = None
    new_width, new_height, will_resize = calculate_resize_dimensions(
        current_width, current_height, max_width, scale=scale, mode=mode
    )
    
    if not will_resize:
        log(f"Image is already {current_width}×{current_height}, no resize needed", message_type="info")
        return image_path
    
    try:
        log(
            f"Resizing image {current_width}×{current_height} → {new_width}×{new_height} "
            f"(center crop, aspect preserved)...",
            message_type="info",
        )
        progress(0.3, desc="Resizing input image...")
        
        img = Image.open(image_path).convert("RGB")
        img_resized = center_crop_cover_pil(img, new_width, new_height)
        
        # Generate output path in temp directory
        input_basename = os.path.splitext(os.path.basename(image_path))[0]
        input_basename = clean_image_filename(input_basename)  # Clean filename to prevent length issues
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(image_path)[1] or '.png'
        output_filename = f"{input_basename}_resized_{new_width}x{new_height}_{timestamp}{ext}"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Save resized image
        img_resized.save(output_path, quality=95)
        
        progress(1.0, desc="Resize complete!")
        log(f"Image resized successfully: {output_path}", message_type="finish")
        return output_path
        
    except Exception as e:
        log(f"Error resizing image: {e}", message_type="error")
        import traceback
        log(traceback.format_exc(), message_type="error")
        return image_path


def run_flashvsr_batch_image(
    input_paths,
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    create_comparison,
    batch_resize_preset,
    progress=gr.Progress(track_tqdm=True)
):
    """Processes a batch of images through FlashVSR, saving all to a timestamped subfolder."""
    if not input_paths:
        log("No files provided for batch image processing.", message_type='warning')
        return None, "⚠️ No files provided for batch processing.", None
    
    total_images = len(input_paths)
    
    log(f"Starting batch processing for {total_images} images...", message_type='info')
    if batch_resize_preset != "No Resize":
        log(f"Batch resize preset: {batch_resize_preset}", message_type='info')
    
    # Create batch subfolder with timestamp in images folder
    batch_folder_name = f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    images_output_dir = get_image_output_dir()
    batch_output_dir = os.path.join(images_output_dir, batch_folder_name)
    os.makedirs(batch_output_dir, exist_ok=True)
    
    batch_messages = [f"🚀 Starting batch process for {total_images} images..."]
    last_output_path = None
    
    for i, image_path in enumerate(input_paths):
        try:
            # Update batch progress
            batch_progress = (i / total_images)
            progress(batch_progress, desc=f"Batch: Processing image {i+1}/{total_images}: {os.path.basename(image_path)}")
            log(f"\n--- Processing image {i+1}/{total_images}: {os.path.basename(image_path)} ---", message_type='info')
            batch_messages.append(f"\n--- Image {i+1}/{total_images}: {os.path.basename(image_path)} ---")
            
            # Apply batch resize if preset is selected (4K-safe or width cap).
            # Do NOT reuse the pipeline `mode` name — that used to pass "4k_safe"
            # into run_flashvsr_image and imageio died with "URI: None".
            processed_image_path = image_path
            resize_mode, max_width = parse_batch_resize_preset(batch_resize_preset, scale=get_ui_defaults().get("scale", 4))
            if resize_mode != "none":
                current_width, current_height = get_image_dimensions(image_path)
                sc = int(get_ui_defaults().get("scale", 4) or 4)
                nw, nh, will = calculate_resize_dimensions(
                    current_width, current_height, max_width=max_width, scale=sc, mode=resize_mode
                )
                if will:
                    log(
                        f"Resizing image {current_width}×{current_height} → {nw}×{nh} "
                        f"({sc}× → {nw * sc}×{nh * sc} within UHD 4K)...",
                        message_type="info",
                    )
                    batch_messages.append(
                        f"  Resizing: {current_width}×{current_height} → {nw}×{nh} "
                        f"(at {sc}× → {nw * sc}×{nh * sc})"
                    )

                    class DummyProgress:
                        def __call__(self, *args, **kwargs):
                            pass

                    processed_image_path = resize_input_image(
                        image_path, max_width, scale=sc, progress=DummyProgress(), mode=resize_mode
                    )
                else:
                    log(
                        f"Image {current_width}×{current_height} already 4K-safe at {sc}× — no resize",
                        message_type="info",
                    )
                    batch_messages.append(f"  No resize needed ({current_width}×{current_height})")
            
            image_path = processed_image_path
            
            # Create a dummy progress object that doesn't interfere with batch progress
            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass
                def tqdm(self, iterable, *args, **kwargs):
                    return iterable
            
            # Process the image using the single image function
            temp_output_path, _, _, _ = run_flashvsr_image(
                image_path=image_path,
                mode=mode,
                model_version=model_version,
                scale=scale,
                color_fix=color_fix,
                tiled_vae=tiled_vae,
                tiled_dit=tiled_dit,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                unload_dit=unload_dit,
                dtype_str=dtype_str,
                seed=seed,
                device=device,
                fps_override=fps_override,
                quality=quality,
                attention_mode=attention_mode,
                sparse_ratio=sparse_ratio,
                kv_ratio=kv_ratio,
                local_range=local_range,
                autosave=False,  # Don't autosave to main outputs folder
                create_comparison=create_comparison,
                progress=DummyProgress()  # Use dummy progress to avoid conflicts
            )
            
            # Copy the result to the batch subfolder
            if temp_output_path and os.path.exists(temp_output_path):
                filename = os.path.basename(temp_output_path)
                final_path = os.path.join(batch_output_dir, filename)
                shutil.copy(temp_output_path, final_path)
                last_output_path = final_path
                log(f"✅ Saved to batch folder: {final_path}", message_type='finish')
                batch_messages.append(f"✅ Saved to: {filename}")
            else:
                log(f"❌ Processing failed for {os.path.basename(image_path)}", message_type='error')
                batch_messages.append(f"❌ Processing failed")
                
        except Exception as e:
            log(f"❌ Error processing {os.path.basename(image_path)}: {e}", message_type='error')
            batch_messages.append(f"❌ Error: {str(e)}")
            continue
    
    progress(1.0, desc="Batch processing complete!")
    batch_messages.append(f"\n✅ Batch processing complete! All results saved to: {batch_output_dir}")
    log(f"Batch processing complete! Results saved to: {batch_output_dir}", message_type='finish')
    
    # Return the last processed image and status messages
    status_message = "\n".join(batch_messages)
    status_html = f'<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ Batch processing complete! All results saved to: {batch_output_dir}</div>'
    return last_output_path, status_message, status_html


def run_flashvsr_image(
    image_path,
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    autosave,
    create_comparison,
    progress=gr.Progress(track_tqdm=True)
):
    """Process a single image by duplicating it 21 times and extracting the middle frame from output."""
    if not image_path:
        log("No input image provided.", message_type='warning')
        return None, None, None, gr.update(visible=False)
    mode = normalize_pipeline_mode(mode)
    
    temp_frames_dir = None
    try:
        # Prepare image as frames
        progress(0.05, desc="Preparing image frames...")
        temp_frames_dir = prepare_image_as_frames(image_path)
        if not temp_frames_dir:
            return None, None, None
        
        # Process through the video pipeline
        video_output, save_path, slider_data, _ = run_flashvsr_single(
            input_path=temp_frames_dir,
            mode=mode,
            model_version=model_version,
            scale=scale,
            color_fix=color_fix,
            tiled_vae=tiled_vae,
            tiled_dit=tiled_dit,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            unload_dit=unload_dit,
            dtype_str=dtype_str,
            seed=seed,
            device=device,
            fps_override=fps_override,
            quality=quality,
            attention_mode=attention_mode,
            sparse_ratio=sparse_ratio,
            kv_ratio=kv_ratio,
            local_range=local_range,
            autosave=False,  # We'll handle saving separately
            create_comparison=False,
            progress=progress
        )
        
        if not video_output or not os.path.exists(video_output):
            log("Image processing failed", message_type="error")
            return None, None, None
        
        # Extract middle frame from the output video
        progress(0.95, desc="Extracting upscaled image...")
        log("Extracting middle frame from output...", message_type="info")
        
        with imageio.get_reader(video_output) as reader:
            num_frames = reader.count_frames()
            middle_frame_idx = num_frames // 2
            
            # Read the middle frame
            for idx, frame in enumerate(reader):
                if idx == middle_frame_idx:
                    middle_frame = frame
                    break
        
        # Get original image dimensions to crop padding
        input_img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = input_img.size
        target_w = orig_w * scale
        target_h = orig_h * scale
        
        # Convert frame to PIL and crop padding if present
        output_img = Image.fromarray(middle_frame)
        output_w, output_h = output_img.size
        
        if output_w > target_w or output_h > target_h:
            # Center crop to remove padding
            crop_left = (output_w - target_w) // 2
            crop_top = (output_h - target_h) // 2
            crop_right = crop_left + target_w
            crop_bottom = crop_top + target_h
            output_img = output_img.crop((crop_left, crop_top, crop_right, crop_bottom))
            log(f"Cropped padding from image: {output_w}x{output_h} → {target_w}x{target_h}", message_type='info')
        
        # Save the cropped image with cleaned filename
        input_basename = os.path.splitext(os.path.basename(image_path))[0]
        clean_basename = clean_image_filename(input_basename, max_length=20)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = upscale_image_filename(
            clean_basename,
            scale,
            output_height=orig_h * scale,
            output_width=orig_w * scale,
        )
        temp_image_path = os.path.join(TEMP_DIR, output_filename)
        
        output_img.save(temp_image_path)
        
        # Autosave if enabled (to images subfolder)
        output_dir = get_output_dir()
        if autosave:
            images_output_dir = get_image_output_dir()
            os.makedirs(images_output_dir, exist_ok=True)
            final_save_path = os.path.join(images_output_dir, output_filename)
            shutil.copy(temp_image_path, final_save_path)
            log(f"Image processing complete! Auto-saved to: {final_save_path}", message_type="finish")
            status_msg = f'<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ Image processing complete! Auto-saved to: {final_save_path}</div>'
        else:
            log("Image processing complete! Use 'Save Output' to save (Step 4 image folder).", message_type="finish")
            status_msg = (
                '<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; '
                'border-radius: 4px; color: #86efac;">✅ Image processing complete! '
                'Use \'Save Output\' to write to Step 4 (Ready for CIV\\images).</div>'
            )
        
        progress(1, desc="Done!")
        
        # Prepare images for ImageSlider (before/after tuple)
        try:
            # Upscale input to match output for proper comparison (no stretching)
            input_upscaled = input_img.resize((target_w, target_h), Image.LANCZOS)
            
            # Save upscaled input for ImageSlider with short filename
            input_upscaled_filename = f"{clean_basename}_input_{timestamp}.png"
            input_upscaled_path = os.path.join(TEMP_DIR, input_upscaled_filename)
            input_upscaled.save(input_upscaled_path)
            
            # ImageSlider expects tuple of (before, after) paths
            comparison_tuple = (input_upscaled_path, temp_image_path)
            
            # Create stitched side-by-side comparison if requested
            if create_comparison:
                log("Creating side-by-side comparison image...", message_type="info")
                comparison_width = input_upscaled.width + output_img.width
                comparison_height = max(input_upscaled.height, output_img.height)
                comparison_img = Image.new('RGB', (comparison_width, comparison_height))
                comparison_img.paste(input_upscaled, (0, 0))
                comparison_img.paste(output_img, (input_upscaled.width, 0))
                
                # Save stitched comparison into image step folder
                images_output_dir = get_image_output_dir()
                os.makedirs(images_output_dir, exist_ok=True)
                comparison_filename = comparison_image_filename(clean_basename)
                comparison_save_path = os.path.join(images_output_dir, comparison_filename)
                comparison_img.save(comparison_save_path, quality=95)
                log(f"Side-by-side comparison saved to: {comparison_save_path}", message_type="finish")
                
        except Exception as e:
            log(f"Could not create comparison: {e}", message_type="warning")
            comparison_tuple = None
        
        # Return: output_image, output_path_for_save, comparison_tuple_for_slider, status_message
        return temp_image_path, temp_image_path, comparison_tuple, status_msg
        
    finally:
        # Cleanup temp frames directory
        if temp_frames_dir and os.path.exists(temp_frames_dir):
            try:
                shutil.rmtree(temp_frames_dir)
                log(f"Cleaned up temp frames directory", message_type="info")
            except Exception as e:
                log(f"Warning: Could not clean up temp frames: {e}", message_type="warning")

def release_processing_vram():
    """Aggressive VRAM release between batch videos to avoid intermittent OOM."""
    clean_vram()


def run_flashvsr_batch(
    input_paths,
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    batch_resize_preset,
    enable_chunks,
    chunk_duration,
    progress=gr.Progress(track_tqdm=True)
):
    """Processes a batch of videos through FlashVSR, saving all to a timestamped subfolder."""
    if not input_paths:
        log("No files provided for batch processing.", message_type='warning')
        return None, "⚠️ No files provided for batch processing."
    
    total_videos = len(input_paths)
    
    log(f"Starting batch processing for {total_videos} videos...", message_type='info')
    if enable_chunks:
        log(f"Batch chunk mode enabled ({chunk_duration}s segments per video)", message_type='info')
    else:
        log("Batch chunk mode disabled — processing each video as a whole", message_type='info')
    if batch_resize_preset != "No Resize":
        log(f"Batch resize preset: {batch_resize_preset}", message_type='info')
    
    # Create batch subfolder with timestamp
    batch_folder_name = f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = get_output_dir()
    batch_output_dir = os.path.join(output_dir, batch_folder_name)
    os.makedirs(batch_output_dir, exist_ok=True)

    # Persist full input list so Toolbox → Batch Queue can import a crashed run.
    write_batch_inputs_list(batch_output_dir, input_paths)
    
    batch_messages = [f"🚀 Starting batch process for {total_videos} videos..."]
    batch_messages.append(f"📋 Progress tracking: {os.path.join(batch_output_dir, 'BATCH_PROGRESS.txt')}")
    batch_messages.append(f"📋 Remaining after crash: {os.path.join(batch_output_dir, 'REMAINING.txt')}")
    last_output_path = None
    fatal_oom = False

    for i, video_path in enumerate(input_paths):
        try:
            # Update batch progress
            batch_progress = (i / total_videos)
            progress(batch_progress, desc=f"Batch: Processing video {i+1}/{total_videos}: {os.path.basename(video_path)}")
            log(f"\n--- Processing video {i+1}/{total_videos}: {os.path.basename(video_path)} ---", message_type='info')
            batch_messages.append(f"\n--- Video {i+1}/{total_videos}: {os.path.basename(video_path)} ---")

            # Skip remaining items if a prior OOM left the CUDA context unusable.
            if fatal_oom or cuda_context_poisoned(min_free_mb=1500):
                fatal_oom = True
                msg = (
                    "⏭ Skipped — GPU VRAM still exhausted after a previous OOM. "
                    "Restart FlashVSR in Pinokio, then re-run the remaining videos."
                )
                log(msg, message_type="warning")
                batch_messages.append(msg)
                write_live_batch_progress(
                    batch_output_dir,
                    total=total_videos,
                    index=i,
                    source=video_path,
                    status="failed",
                    error="skipped after OOM — restart app and use REMAINING.txt / Batch Queue import",
                    all_sources=input_paths,
                )
                continue

            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass

            before_w, before_h = get_video_dimensions(video_path)
            resized_path = apply_batch_resize_preset(
                video_path, batch_resize_preset, scale=scale, progress=DummyProgress()
            )
            if resized_path != video_path:
                after_w, after_h = get_video_dimensions(resized_path)
                batch_messages.append(
                    f"  Resizing: {before_w}×{before_h} → {after_w}×{after_h} "
                    f"(at {scale}× → {after_w * int(scale)}×{after_h * int(scale)}, UHD-safe)"
                )
            elif not is_no_resize_preset(batch_resize_preset):
                batch_messages.append(
                    f"  No resize needed ({before_w}×{before_h}; "
                    f"{scale}× → {before_w * int(scale)}×{before_h * int(scale)} already 4K-safe)"
                )
            video_path = resized_path
            
            class BatchProgress(DummyProgress):
                def tqdm(self, iterable, *args, **kwargs):
                    return iterable

            batch_progress = BatchProgress()
            if enable_chunks:
                temp_output_path, _, _, _ = process_video_with_chunks(
                    input_path=video_path,
                    chunk_duration=chunk_duration,
                    mode=mode,
                    model_version=model_version,
                    scale=scale,
                    color_fix=color_fix,
                    tiled_vae=tiled_vae,
                    tiled_dit=tiled_dit,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    unload_dit=unload_dit,
                    dtype_str=dtype_str,
                    seed=seed,
                    device=device,
                    fps_override=fps_override,
                    quality=quality,
                    attention_mode=attention_mode,
                    sparse_ratio=sparse_ratio,
                    kv_ratio=kv_ratio,
                    local_range=local_range,
                    autosave=False,
                    progress=batch_progress,
                )
            else:
                temp_output_path, _, _, _ = run_flashvsr_single(
                    input_path=video_path,
                    mode=mode,
                    model_version=model_version,
                    scale=scale,
                    color_fix=color_fix,
                    tiled_vae=tiled_vae,
                    tiled_dit=tiled_dit,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    unload_dit=unload_dit,
                    dtype_str=dtype_str,
                    seed=seed,
                    device=device,
                    fps_override=fps_override,
                    quality=quality,
                    attention_mode=attention_mode,
                    sparse_ratio=sparse_ratio,
                    kv_ratio=kv_ratio,
                    local_range=local_range,
                    autosave=False,
                    progress=batch_progress,
                )
            
            # Copy the result to the batch subfolder
            if temp_output_path and os.path.exists(temp_output_path):
                filename = os.path.basename(temp_output_path)
                final_path = os.path.join(batch_output_dir, filename)
                shutil.copy(temp_output_path, final_path)
                last_output_path = final_path
                log(f"✅ Saved to batch folder: {final_path}", message_type='finish')
                batch_messages.append(f"✅ Saved to: {filename}")
                write_live_batch_progress(
                    batch_output_dir,
                    total=total_videos,
                    index=i,
                    source=video_path,
                    status="done",
                    output=final_path,
                    all_sources=input_paths,
                )
            else:
                log(f"❌ Processing failed for {os.path.basename(video_path)}", message_type='error')
                batch_messages.append(f"❌ Processing failed")
                write_live_batch_progress(
                    batch_output_dir,
                    total=total_videos,
                    index=i,
                    source=video_path,
                    status="failed",
                    error="processing returned no output",
                    all_sources=input_paths,
                )
                
        except Exception as e:
            log(f"❌ Error processing {os.path.basename(video_path)}: {e}", message_type='error')
            batch_messages.append(f"❌ Error: {str(e)}")
            write_live_batch_progress(
                batch_output_dir,
                total=total_videos,
                index=i,
                source=video_path,
                status="failed",
                error=str(e),
                all_sources=input_paths,
            )
            if is_cuda_oom(e) or cuda_context_poisoned(min_free_mb=1500):
                fatal_oom = True
                batch_messages.append(
                    "🛑 Unrecoverable GPU OOM — remaining batch items will be skipped. "
                    "Restart FlashVSR in Pinokio to free VRAM, then re-queue unfinished videos."
                )
                log(
                    "[FlashVSR] Aborting rest of batch: CUDA context still out of memory. "
                    "Restart the app to reclaim VRAM.",
                    message_type="error",
                )
        finally:
            release_processing_vram()
    
    progress(1.0, desc="Batch processing complete!")
    if fatal_oom:
        batch_messages.append(
            f"\n⚠️ Batch stopped early due to VRAM OOM. Partial results (if any) are in: {batch_output_dir}"
        )
        batch_messages.append(
            f"📋 Progress log: {os.path.join(batch_output_dir, 'BATCH_PROGRESS.txt')} "
            f"— open it to see exactly which files finished."
        )
        log(f"Batch stopped early (OOM). Partial results in: {batch_output_dir}", message_type="warning")
    else:
        batch_messages.append(f"\n✅ Batch processing complete! All results saved to: {batch_output_dir}")
        batch_messages.append(
            f"📋 Progress log: {os.path.join(batch_output_dir, 'BATCH_PROGRESS.txt')}"
        )
        log(f"Batch processing complete! Results saved to: {batch_output_dir}", message_type='finish')
    
    # Return the last processed video and a status message
    status_message = "\n".join(batch_messages)
    return last_output_path, status_message


def get_flashvsr_work_queue() -> FlashVSRWorkQueue:
    return FlashVSRWorkQueue(
        ROOT_DIR, name="video", extensions=VIDEO_EXTS, label="FlashVSR video queue"
    )


def get_flashvsr_image_queue() -> FlashVSRWorkQueue:
    return FlashVSRWorkQueue(
        ROOT_DIR, name="image", extensions=IMAGE_EXTS, label="FlashVSR image queue"
    )


def get_toolbox_work_queue() -> FlashVSRWorkQueue:
    return FlashVSRWorkQueue(
        ROOT_DIR, name="toolbox", extensions=VIDEO_EXTS, label="Toolbox post-process queue"
    )


def get_group_therapy_queue() -> FlashVSRWorkQueue:
    return gt.get_queue(ROOT_DIR)


def get_exclusive_queue_lock() -> ExclusiveQueueLock:
    return ExclusiveQueueLock(ROOT_DIR)


def _queue_busy_html(wq: FlashVSRWorkQueue, message: str) -> str:
    """Status panel when another queue owns the exclusive lock."""
    return wq.status_html(message) + get_exclusive_queue_lock().status_html_snippet()


class _DummyProgress:
    def __call__(self, *args, **kwargs):
        pass

    def tqdm(self, iterable, *args, **kwargs):
        return iterable


def _gt_upscale_one(video_path, *, mode, model_version, scale, color_fix, tiled_vae,
                    tiled_dit, tile_size, tile_overlap, unload_dit, dtype_str, seed,
                    device, fps_override, quality, attention_mode, sparse_ratio,
                    kv_ratio, local_range, batch_resize_preset, enable_chunks,
                    chunk_duration):
    """Run FlashVSR on one file. Returns (output_path, resized_path_or_none)."""
    resized_path = apply_batch_resize_preset(
        video_path, batch_resize_preset, scale=scale, progress=_DummyProgress()
    )
    process_path = resized_path or video_path
    extra_resized = resized_path if resized_path and os.path.normcase(
        os.path.abspath(resized_path)
    ) != os.path.normcase(os.path.abspath(video_path)) else None
    if enable_chunks:
        out, _, _, _ = process_video_with_chunks(
            input_path=process_path,
            chunk_duration=chunk_duration,
            mode=mode,
            model_version=model_version,
            scale=scale,
            color_fix=color_fix,
            tiled_vae=tiled_vae,
            tiled_dit=tiled_dit,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            unload_dit=unload_dit,
            dtype_str=dtype_str,
            seed=seed,
            device=device,
            fps_override=fps_override,
            quality=quality,
            attention_mode=attention_mode,
            sparse_ratio=sparse_ratio,
            kv_ratio=kv_ratio,
            local_range=local_range,
            autosave=False,
            progress=_DummyProgress(),
        )
    else:
        out, _, _, _ = run_flashvsr_single(
            input_path=process_path,
            mode=mode,
            model_version=model_version,
            scale=scale,
            color_fix=color_fix,
            tiled_vae=tiled_vae,
            tiled_dit=tiled_dit,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            unload_dit=unload_dit,
            dtype_str=dtype_str,
            seed=seed,
            device=device,
            fps_override=fps_override,
            quality=quality,
            attention_mode=attention_mode,
            sparse_ratio=sparse_ratio,
            kv_ratio=kv_ratio,
            local_range=local_range,
            autosave=False,
            progress=_DummyProgress(),
        )
    return out, extra_resized


def _gt_rife_one(src_path, *, frames_q, use_streaming):
    global toolbox_processor
    if toolbox_processor is None:
        toolbox_processor = ToolboxProcessor(True)
    return toolbox_processor.adjust_frames(
        src_path,
        fps_mode="2x Frames",
        speed_factor=1.0,
        use_streaming=use_streaming,
        output_quality=frames_q,
        progress=_DummyProgress(),
    )


def _gt_export_one(src_path, *, export_q, export_w, after_dir, export_preset, prefer_nvenc):
    global toolbox_processor
    if toolbox_processor is None:
        toolbox_processor = ToolboxProcessor(True)
    toolbox_processor.output_dir = Path(after_dir)
    toolbox_processor.autosave_enabled = True
    toolbox_processor.export_preset = export_preset
    toolbox_processor.prefer_nvenc = prefer_nvenc
    return toolbox_processor.export_video(
        src_path,
        export_format="MP4 (H.264)",
        quality=export_q,
        max_width=export_w,
        output_name="",
        two_pass=False,
        progress=_DummyProgress(),
    )


def _gt_copy_into(src_path, dest_dir):
    if not src_path or not os.path.isfile(src_path):
        return None
    dest = unique_dest_path(dest_dir, os.path.basename(src_path))
    if os.path.normcase(os.path.abspath(src_path)) != os.path.normcase(os.path.abspath(dest)):
        shutil.copy2(src_path, dest)
    return dest


def run_group_therapy(
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    batch_resize_preset,
    enable_chunks,
    chunk_duration,
    group_size,
    watch_folder,
    before_dir,
    after_dir,
    do_upscale,
    do_rife1,
    do_rife2,
    do_export,
    progress=gr.Progress(track_tqdm=True),
):
    wq = get_group_therapy_queue()
    lock = get_exclusive_queue_lock()
    ok, lock_msg = lock.try_acquire("group")
    if not ok:
        log(lock_msg, message_type="warning")
        return None, _queue_busy_html(wq, lock_msg)
    try:
        return _run_group_therapy_body(
            wq,
            mode=mode,
            model_version=model_version,
            scale=scale,
            color_fix=color_fix,
            tiled_vae=tiled_vae,
            tiled_dit=tiled_dit,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            unload_dit=unload_dit,
            dtype_str=dtype_str,
            seed=seed,
            device=device,
            fps_override=fps_override,
            quality=quality,
            attention_mode=attention_mode,
            sparse_ratio=sparse_ratio,
            kv_ratio=kv_ratio,
            local_range=local_range,
            batch_resize_preset=batch_resize_preset,
            enable_chunks=enable_chunks,
            chunk_duration=chunk_duration,
            group_size=group_size,
            watch_folder=watch_folder,
            before_dir=before_dir,
            after_dir=after_dir,
            do_upscale=do_upscale,
            do_rife1=do_rife1,
            do_rife2=do_rife2,
            do_export=do_export,
            progress=progress,
        )
    finally:
        lock.release("group")


def _run_group_therapy_body(
    wq,
    *,
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    batch_resize_preset,
    enable_chunks,
    chunk_duration,
    group_size,
    watch_folder,
    before_dir,
    after_dir,
    do_upscale,
    do_rife1,
    do_rife2,
    do_export,
    progress,
):
    global toolbox_processor
    wq.clear_stop()
    ui = get_ui_defaults()
    paths = ensure_workflow_dirs(ui)
    watch_folder = (watch_folder or ui.get("batch_watch_folder") or paths.get("watch") or "").strip()
    before_dir = (before_dir or ui.get("gt_before_dir") or ui.get("batch_source_archive_dir") or paths["pre_scaled"]).strip()
    after_dir = (after_dir or ui.get("gt_after_dir") or get_toolbox_output_dir()).strip()
    group_size = max(1, int(group_size or ui.get("gt_group_size") or 10))
    stages = gt.selected_stages(
        do_upscale=bool(do_upscale),
        do_rife1=bool(do_rife1),
        do_rife2=bool(do_rife2),
        do_export=bool(do_export),
    )
    last_stage = stages[-1]
    frames_q = int(ui.get("tb_frames_quality") or 98)
    export_q = int(ui.get("tb_export_quality") or 96)
    export_w = int(ui.get("tb_export_max_width") or 3840)
    use_streaming = True if ui.get("tb_use_streaming") is None else bool(ui.get("tb_use_streaming"))
    export_preset = (ui.get("tb_export_preset") or "slow").strip().lower()
    prefer_nvenc = True if ui.get("tb_prefer_nvenc") is None else bool(ui.get("tb_prefer_nvenc"))

    if toolbox_processor is None:
        toolbox_processor = ToolboxProcessor(True)
    toolbox_processor.output_dir = Path(after_dir)
    toolbox_processor.autosave_enabled = True
    toolbox_processor.export_preset = export_preset
    toolbox_processor.prefer_nvenc = prefer_nvenc

    cfg = load_config()
    if watch_folder:
        cfg["batch_watch_folder"] = watch_folder
    cfg["gt_group_size"] = group_size
    cfg["gt_before_dir"] = before_dir
    cfg["gt_after_dir"] = after_dir
    cfg["gt_do_upscale"] = bool(do_upscale)
    cfg["gt_do_rife1"] = bool(do_rife1)
    cfg["gt_do_rife2"] = bool(do_rife2)
    cfg["gt_do_export"] = bool(do_export)
    save_config(cfg)

    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    if watch_folder and os.path.isdir(watch_folder):
        _log_hygiene(
            watch_folder,
            "watch",
            hygiene_scan_folder(watch_folder, role="watch", scale=int(scale or 4)),
        )
        added, skipped = wq.add_folder(watch_folder)
        if added:
            log(f"Group Therapy: added {added} from {watch_folder}", message_type="info")
        elif not skipped:
            log(f"Group Therapy: no new videos in {watch_folder}", message_type="info")

    stuck = wq.reset_stuck_running()
    if stuck:
        log(f"Re-queued {stuck} stuck Group Therapy job(s)", message_type="info")
    wq.requeue_failed()
    gt.assign_groups(wq, group_size)
    already = gt.mark_already_paired(wq, after_dir)
    if already:
        log(f"Group Therapy: skipped {already} already-paired file(s)", message_type="info")
    wq.set_fixed_completed_dir(after_dir)
    wq.set_meta(
        gt_group_size=group_size,
        gt_before_dir=before_dir,
        gt_after_dir=after_dir,
        gt_stages=",".join(stages),
    )

    items = [it for it in wq.all_items() if it.get("status") != "done"]
    groups = gt.ordered_groups(items)
    if not groups:
        note = f"Group Therapy empty — drop videos in {watch_folder or 'the original folder'}."
        return None, wq.status_html(note)

    log(
        f"Group Therapy: {len(groups)} group(s) × {group_size}  "
        f"stages={' → '.join(gt.STAGE_LABELS[s] for s in stages)}  "
        f"before={before_dir}  after={after_dir}",
        message_type="info",
    )

    last_output = None
    finished = 0
    failed = 0

    def _settle_item(item):
        nonlocal last_output, finished
        path = item["path"]
        fresh = next((x for x in wq.all_items() if os.path.normcase(x.get("path", "")) == os.path.normcase(path)), item)
        stem = Path(path).stem
        extras = gt.collect_temp_globs(
            stem,
            TEMP_DIR,
            str(gt.work_root(ROOT_DIR)),
            str(gt.stage_dir(ROOT_DIR, "upscale")),
            str(gt.stage_dir(ROOT_DIR, "rife1")),
            str(gt.stage_dir(ROOT_DIR, "rife2")),
            os.path.join(TEMP_DIR, "toolbox"),
        )
        try:
            before_p, after_p, deleted = gt.settle_pair(
                fresh, before_dir=before_dir, after_dir=after_dir, extra_temps=extras
            )
        except Exception as e:
            log(f"Settle skipped for {os.path.basename(path)}: {e}", message_type="warning")
            return
        wq.update_item(
            path,
            gt_original=before_p,
            gt_export=after_p,
            gt_before=before_p,
            gt_after=after_p,
            gt_pair_id=fresh.get("gt_pair_id"),
            gt_pair_folder=fresh.get("gt_pair_folder"),
        )
        wq.set_item_status(path, "done", output=after_p)
        last_output = after_p
        finished += 1
        pid = fresh.get("gt_pair_id") or "?"
        folder = fresh.get("gt_pair_folder") or ""
        log(
            f"🧹 Pair {pid} — original + final only ({deleted} temps deleted)\n"
            f"   id: {folder}\n   before: {before_p}\n   after:  {after_p}",
            message_type="finish",
        )

    for g_i, gid in enumerate(groups):
        members = [m for m in gt.group_members(wq.all_items(), gid) if m.get("status") != "done"]
        if not members:
            continue
        log(
            f"\n══ Group {gid} ({g_i + 1}/{len(groups)}) — {len(members)} file(s) ══",
            message_type="info",
        )
        wq.set_meta(gt_current_group=gid)

        for stage in stages:
            wq.set_meta(gt_current_stage=stage)
            label = gt.STAGE_LABELS.get(stage, stage)
            log(f"— Group {gid}: {label} ({len(members)} files) —", message_type="info")
            if stage in ("rife1", "rife2", "export"):
                release_processing_vram()

            for f_i, item in enumerate(list(members)):
                path = item["path"]
                if not os.path.isfile(path):
                    relocated = find_relocated_source(path, watch_folder, before_dir)
                    if relocated:
                        wq.set_item_status(path, item.get("status") or "pending", new_path=relocated)
                        path = relocated
                        item["path"] = path
                    else:
                        # maybe already archived and later stages have intermediates
                        if not gt.input_for_stage(item, stage):
                            log(f"Missing source, skip: {os.path.basename(item.get('path') or '')}", message_type="warning")
                            continue

                # refresh item
                item = next(
                    (x for x in wq.all_items() if os.path.normcase(x.get("path", "")) == os.path.normcase(path)),
                    item,
                )
                if item.get("status") == "done":
                    continue
                if gt.stage_already_done(item, stage):
                    log(f"  skip {label} (already have output): {os.path.basename(path)}", message_type="info")
                    if stage == last_stage:
                        _settle_item(item)
                    continue

                src = gt.input_for_stage(item, stage)
                if not src:
                    log(f"❌ No input for {label}: {os.path.basename(path)}", message_type="error")
                    wq.set_item_status(path, "failed", error=f"no input for {stage}")
                    failed += 1
                    continue

                progress(
                    (g_i + (stages.index(stage) + (f_i / max(len(members), 1))) / max(len(stages), 1)) / max(len(groups), 1),
                    desc=f"Group {gid} · {label} · {f_i + 1}/{len(members)} · {os.path.basename(path)}",
                )
                wq.set_item_status(path, "running")
                pair_id = gt.ensure_pair_id(item)
                pair_folder = item.get("gt_pair_folder") or gt.pair_folder_name(pair_id, path)
                wq.update_item(
                    path,
                    gt_stage=stage,
                    gt_group=gid,
                    gt_pair_id=pair_id,
                    gt_pair_folder=pair_folder,
                )
                if not item.get("gt_original"):
                    wq.update_item(path, gt_original=path)

                try:
                    if stage == "upscale":
                        out, resized = _gt_upscale_one(
                            src,
                            mode=mode,
                            model_version=model_version,
                            scale=scale,
                            color_fix=color_fix,
                            tiled_vae=tiled_vae,
                            tiled_dit=tiled_dit,
                            tile_size=tile_size,
                            tile_overlap=tile_overlap,
                            unload_dit=unload_dit,
                            dtype_str=dtype_str,
                            seed=seed,
                            device=device,
                            fps_override=fps_override,
                            quality=quality,
                            attention_mode=attention_mode,
                            sparse_ratio=sparse_ratio,
                            kv_ratio=kv_ratio,
                            local_range=local_range,
                            batch_resize_preset=batch_resize_preset,
                            enable_chunks=enable_chunks,
                            chunk_duration=chunk_duration,
                        )
                        if not out or not os.path.isfile(out):
                            raise RuntimeError("upscale produced no file")
                        kept = _gt_copy_into(out, str(gt.stage_dir(ROOT_DIR, "upscale")))
                        wq.update_item(path, gt_upscale=kept or out, gt_resized=resized, gt_stage="upscale")
                        log(f"✅ Upscale → {kept or out}", message_type="finish")
                    elif stage in ("rife1", "rife2"):
                        cur_fps = probe_file_fps(src)
                        if cur_fps >= 160:
                            raise RuntimeError(
                                f"already {cur_fps:.0f} FPS — not interpolating (would be 200+)"
                            )
                        if stage == "rife2" and cur_fps * 2.0 > 121:
                            log(
                                f"Skip RIFE pass 2 — {os.path.basename(src)} is already "
                                f"{cur_fps:.0f} FPS (4× would be {cur_fps * 2:.0f})",
                                message_type="info",
                            )
                            wq.update_item(path, gt_rife2=src, gt_stage="rife2")
                            if last_stage == "rife2":
                                _settle_item(item)
                            continue
                        out = _gt_rife_one(src, frames_q=frames_q, use_streaming=use_streaming)
                        if not out or not os.path.isfile(out):
                            raise RuntimeError(f"{stage} produced no file")
                        kept = _gt_copy_into(out, str(gt.stage_dir(ROOT_DIR, stage)))
                        field = "gt_rife1" if stage == "rife1" else "gt_rife2"
                        wq.update_item(path, **{field: kept or out, "gt_stage": stage})
                        log(f"✅ {label} → {kept or out}", message_type="finish")
                    elif stage == "export":
                        export_into = after_dir
                        os.makedirs(export_into, exist_ok=True)
                        out = _gt_export_one(
                            src,
                            export_q=export_q,
                            export_w=export_w,
                            after_dir=export_into,
                            export_preset=export_preset,
                            prefer_nvenc=prefer_nvenc,
                        )
                        if not out or not os.path.isfile(out):
                            raise RuntimeError("export produced no file")
                        try:
                            fps_est = probe_file_fps(out)
                            tagged = re.search(r"_(\d+)fps$", Path(out).stem, re.I)
                            if tagged:
                                fps_est = float(tagged.group(1))
                            if not fps_est:
                                fps_est = probe_file_fps(src) or 30.0
                            out = rename_to_step2(
                                out,
                                source_stem=Path(src).stem,
                                fps=fps_est,
                                ext=Path(out).suffix or ".mp4",
                            )
                            pid = pair_id or gt.ensure_pair_id(item)
                            pid_name = gt.with_pid_name(out, pid)
                            pid_dest = os.path.join(os.path.dirname(out), os.path.basename(pid_name))
                            if os.path.normcase(out) != os.path.normcase(pid_dest):
                                if os.path.exists(pid_dest):
                                    pid_dest = gt.unique_pid_dest(os.path.dirname(out), os.path.basename(pid_name))
                                os.replace(out, pid_dest)
                                out = pid_dest
                            gt.stamp_title_pid(out, pid)
                        except Exception:
                            pass
                        wq.update_item(path, gt_export=out, gt_stage="export")
                        log(f"✅ Export → {out}", message_type="finish")

                    wq.set_item_status(path, "pending")
                    if stage == last_stage:
                        _settle_item(item)
                except Exception as e:
                    log(f"❌ Group {gid} {label} error: {e}", message_type="error")
                    wq.set_item_status(path, "failed", error=str(e))
                    failed += 1
                    if is_cuda_oom(e) or cuda_context_poisoned(min_free_mb=1500):
                        note = (
                            f"⚠️ Group Therapy paused on OOM at group {gid} ({label}). "
                            "Restart FlashVSR, then Start / Resume."
                        )
                        log(note, message_type="error")
                        return last_output, wq.status_html(note)
                finally:
                    if stage == "upscale":
                        release_processing_vram()
                    try:
                        if toolbox_processor and getattr(toolbox_processor, "rife_handler", None) and stage.startswith("rife"):
                            if f_i == len(members) - 1:
                                toolbox_processor.rife_handler.unload_model()
                    except Exception:
                        pass

                if wq.stop_requested():
                    wq.clear_stop()
                    note = (
                        f"⏹ Stopped after group {gid} · {label} · "
                        f"{os.path.basename(path)}. {finished} settled this run."
                    )
                    log(note, message_type="warning")
                    return last_output, wq.status_html(note)

            # refresh members after a stage
            members = [m for m in gt.group_members(wq.all_items(), gid) if m.get("status") != "done"]

        log(f"══ Group {gid} complete ══", message_type="finish")
        gt.cleanup_empty_work(ROOT_DIR)

    remaining = len(wq.pending_items())
    note = (
        f"✅ Group Therapy done — {finished} settled (original + final only) → {after_dir}"
        if remaining == 0
        else f"Pass done ({finished} settled, {failed} failed, {remaining} pending)."
    )
    progress(1.0, desc=note)
    wq.set_meta(gt_current_stage="idle")
    return last_output, wq.status_html(note)


def ensure_workflow_dirs(ui: Optional[dict] = None) -> dict:
    """Create standard D: workflow folders and return a short alias map."""
    resolved = get_workflow_paths()
    # Optional UI overrides (when Gradio fields differ mid-session)
    if ui:
        for ui_key, res_key in (
            ("batch_watch_folder", "batch_watch_folder"),
            ("batch_source_archive_dir", "batch_source_archive_dir"),
            ("batch_upscale_handoff_dir", "batch_upscale_handoff_dir"),
            ("img_upscale_handoff_dir", "img_upscale_handoff_dir"),
            ("tb_inbox_folder", "tb_inbox_folder"),
            ("toolbox_output_dir", "toolbox_output_dir"),
        ):
            val = str(ui.get(ui_key, "") or "").strip()
            if val and os.path.isabs(val):
                resolved[res_key] = _abs_path_or_default(val, resolved[res_key])
    paths = {
        "watch": resolved["batch_watch_folder"],
        "pre_scaled": resolved["batch_source_archive_dir"],
        "ready_toolbox": resolved["batch_upscale_handoff_dir"],
        "ready_civ": resolved["toolbox_output_dir"],
        "img_handoff": resolved["img_upscale_handoff_dir"],
        "tb_inbox": resolved["tb_inbox_folder"],
        "output_dir": resolved["output_dir"],
    }
    for p in paths.values():
        if p:
            try:
                os.makedirs(p, exist_ok=True)
            except OSError:
                pass
    return paths


def archive_original_source(source_path: str, archive_dir: str) -> Optional[str]:
    """
    Move the original (pre-upscale) video into the archive folder for later pairing.
    Upscaled output is left in its normal completed/output location.
    """
    if not source_path or not archive_dir:
        return None
    if not os.path.isfile(source_path):
        log(f"Original not found to archive: {source_path}", message_type="warning")
        return None
    try:
        os.makedirs(archive_dir, exist_ok=True)
        name = os.path.basename(source_path)
        dest = os.path.join(archive_dir, name)
        if os.path.abspath(source_path) == os.path.abspath(dest):
            return dest
        if os.path.exists(dest):
            stem, ext = os.path.splitext(name)
            dest = os.path.join(
                archive_dir, f"{stem}_src_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
            )
        shutil.move(source_path, dest)
        log(f"📦 Original moved for pairing → {dest}", message_type="info")
        return dest
    except Exception as e:
        log(f"Could not archive original {source_path}: {e}", message_type="warning")
        return None


def _queue_search_dirs(source_path: str, *extra_dirs: str) -> list:
    """Parent folder, parent/done, and any extra dirs (+ their done/)."""
    dirs: list = []
    parent = os.path.dirname(source_path) if source_path else ""
    if parent:
        dirs.append(parent)
        dirs.append(os.path.join(parent, "done"))
    for d in extra_dirs:
        d = (d or "").strip()
        if not d:
            continue
        if d not in dirs:
            dirs.append(d)
        done = os.path.join(d, "done")
        if done not in dirs:
            dirs.append(done)
    return dirs


def find_relocated_source(source_path: str, *extra_dirs: str) -> Optional[str]:
    """If the queued path is gone, find the same basename in archive/done folders."""
    if source_path and os.path.isfile(source_path):
        return source_path
    name = os.path.basename(source_path or "")
    if not name:
        return None
    for d in _queue_search_dirs(source_path, *extra_dirs):
        if not d or not os.path.isdir(d):
            continue
        cand = os.path.join(d, name)
        if os.path.isfile(cand):
            return cand
    return None


def find_matching_deliverable(
    source_path: str,
    *output_dirs: str,
    prefer_exported: bool = False,
) -> Optional[str]:
    """
    Find an existing handoff/export for a source whose inbox path is gone.

    Matching is intentionally strict so siblings like ``…019f8889…`` vs
    ``…019f8883…`` or chunk ``(3)`` vs ``(5)`` do not share a hit:
    - require the Grok/video UUID (full or truncated for export names)
    - prefer full stem / variant number when present
    """
    stem = Path(source_path or "").stem
    if not stem or len(stem) < 8:
        return None

    uuid_re = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8,12}",
        re.I,
    )
    uuids = uuid_re.findall(stem)
    variant = None
    vm = re.search(r"\((\d+)\)", stem)
    if vm:
        variant = vm.group(1)
    else:
        vm = re.search(r"(?:^|[\s_])(\d+)(?=_resized|_upscaled|$)", stem)
        if vm:
            variant = vm.group(1)

    media_ext = {
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
        ".png", ".jpg", ".jpeg", ".webp",
    }
    scored: list = []  # (score, mtime, path)

    for d in output_dirs:
        d = (d or "").strip()
        if not d or not os.path.isdir(d):
            continue
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for fn in names:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in media_ext:
                continue
            score = 0
            if stem in fn:
                score = 100000
            else:
                if uuids:
                    uuid_hit = False
                    for u in uuids:
                        for L in (len(u), 30, 26, 22):
                            if L >= 22 and L <= len(u) and u[:L] in fn:
                                score += 100 + L
                                uuid_hit = True
                                break
                        if uuid_hit:
                            break
                    if not uuid_hit:
                        continue
                else:
                    # No UUID — require a long unique stem prefix (min 40)
                    best = 0
                    for n in range(min(len(stem), 80), 39, -1):
                        if stem[:n] in fn:
                            best = n
                            break
                    if best < 40:
                        continue
                    score += best

                if variant:
                    # Prefer the same chunk index; reject clear other (N) variants
                    other = re.search(r"\((\d+)\)", fn)
                    if other and other.group(1) != variant:
                        continue
                    has_var = bool(
                        re.search(
                            rf"(?:\({re.escape(variant)}\)|"
                            rf"(?:^|[\s_]){re.escape(variant)}(?=_resized|_upscaled|_exported|[\s_\.]|$))",
                            fn,
                        )
                    )
                    if has_var:
                        score += 250
                    elif prefer_exported:
                        # Toolbox export names often drop the chunk index — UUID is enough
                        score += 10
                    else:
                        # Video/image handoff should keep chunk identity when present
                        continue

            if prefer_exported and "_exported" in fn.lower():
                score += 500

            full = os.path.join(d, fn)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                mtime = 0
            scored.append((score, mtime, full))

    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


def reconcile_work_queue(
    wq: FlashVSRWorkQueue,
    *,
    archive_dirs: Sequence[str],
    output_dirs: Sequence[str],
    prefer_exported: bool = False,
    label: str = "queue",
) -> Tuple[int, int]:
    """
    Auto-heal missing sources: mark done if output exists, or retarget path
    if the file was moved to archive/done. Call at the start of each queue run.
    """
    archive_dirs = [d for d in archive_dirs if d]
    output_dirs = [d for d in output_dirs if d]

    def _find_out(path: str) -> Optional[str]:
        return find_matching_deliverable(
            path, *output_dirs, prefer_exported=prefer_exported
        )

    def _find_reloc(path: str) -> Optional[str]:
        return find_relocated_source(path, *archive_dirs)

    done_n, fix_n = wq.reconcile_missing_sources(
        find_output=_find_out, find_relocated=_find_reloc
    )
    if done_n or fix_n:
        log(
            f"🔧 {label} reconcile: {done_n} marked done (output/archive found), "
            f"{fix_n} path(s) updated",
            message_type="info",
        )
    return done_n, fix_n


def handle_missing_queue_source(
    wq: FlashVSRWorkQueue,
    source_path: str,
    *,
    archive_dirs: Sequence[str],
    output_dirs: Sequence[str],
    prefer_exported: bool = False,
) -> str:
    """
    Resolve a single missing source during a queue pass.

    Returns: 'done' | 'retry' | 'failed'
      done  — existing deliverable or archived source; status set to done
      retry  — path updated to a relocated file; caller should use new path
      failed — truly missing; marked failed
    """
    out = find_matching_deliverable(
        source_path, *output_dirs, prefer_exported=prefer_exported
    )
    if out and os.path.isfile(out):
        wq.set_item_status(source_path, "done", output=out, error=None)
        log(
            f"✅ Recovered (already delivered): {os.path.basename(source_path)} → {out}",
            message_type="finish",
        )
        return "done"

    relocated = find_relocated_source(source_path, *archive_dirs)
    if relocated and os.path.isfile(relocated):
        parent_name = Path(relocated).parent.name.lower()
        if parent_name == "done":
            wq.set_item_status(
                source_path, "done", output=out, error=None, new_path=relocated
            )
            log(
                f"✅ Recovered (already in done/): {os.path.basename(source_path)}",
                message_type="finish",
            )
            return "done"
        wq.set_item_status(
            source_path, "pending", error=None, new_path=relocated
        )
        log(
            f"↪ Relocated source → {relocated}",
            message_type="info",
        )
        return "retry"

    wq.set_item_status(source_path, "failed", error="file not found")
    log(
        f"❌ Missing source (left failed): {os.path.basename(source_path)}",
        message_type="error",
    )
    return "failed"


def run_flashvsr_work_queue(
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    batch_resize_preset,
    enable_chunks,
    chunk_duration,
    progress=gr.Progress(track_tqdm=True),
):
    """Process pending items on the persistent work queue (start / resume). Soft-stop between files."""
    wq = get_flashvsr_work_queue()
    lock = get_exclusive_queue_lock()
    ok, lock_msg = lock.try_acquire("video")
    if not ok:
        log(lock_msg, message_type="warning")
        return None, _queue_busy_html(wq, lock_msg)
    try:
        return _run_flashvsr_work_queue_body(
            wq, mode, model_version, scale, color_fix, tiled_vae, tiled_dit,
            tile_size, tile_overlap, unload_dit, dtype_str, seed, device, fps_override,
            quality, attention_mode, sparse_ratio, kv_ratio, local_range,
            batch_resize_preset, enable_chunks, chunk_duration, progress,
        )
    finally:
        lock.release("video")


def _run_flashvsr_work_queue_body(
    wq,
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    batch_resize_preset,
    enable_chunks,
    chunk_duration,
    progress,
):
    wq.clear_stop()
    ui = get_ui_defaults()
    paths = ensure_workflow_dirs(ui)
    watch_folder = (ui.get("batch_watch_folder") or paths.get("watch") or "").strip()
    source_archive = (
        ui.get("batch_source_archive_dir") or paths.get("pre_scaled") or ""
    ).strip()
    handoff = (ui.get("batch_upscale_handoff_dir") or paths["ready_toolbox"]).strip()

    # Auto-pick up new downloads from the watch folder each Start / Resume
    if watch_folder and os.path.isdir(watch_folder):
        _log_hygiene(
            watch_folder,
            "watch",
            hygiene_scan_folder(
                watch_folder,
                role="watch",
                scale=int(scale or ui.get("scale") or 4),
            ),
        )
        added, skipped = wq.add_folder(watch_folder)
        if added:
            log(
                f"Watch folder: added {added} from {watch_folder}"
                + (f" (skipped {skipped} already queued)" if skipped else ""),
                message_type="info",
            )
        elif skipped:
            log(f"Watch folder: {skipped} already in queue ({watch_folder})", message_type="info")
        else:
            log(f"Watch folder empty or no new videos: {watch_folder}", message_type="info")
    elif watch_folder:
        log(f"Watch folder missing (create it or fix path): {watch_folder}", message_type="warning")

    stuck = wq.reset_stuck_running()
    if stuck:
        log(f"Re-queued {stuck} interrupted (was running) item(s)", message_type="info")

    # Upscaled videos go to Ready for Toolbox (next pipeline step)
    reconcile_work_queue(
        wq,
        archive_dirs=[watch_folder, source_archive, handoff],
        output_dirs=[handoff, paths.get("ready_toolbox", ""), get_output_dir()],
        prefer_exported=False,
        label="Video queue",
    )

    # Step 1 preflight: dedupe queue, drop already-complete outputs, refresh list
    out_dirs = [
        handoff,
        paths.get("ready_toolbox", ""),
        get_output_dir(),
        wq.get_completed_dir() or "",
    ]

    def _find_complete(path: str):
        return find_matching_deliverable(
            path, *out_dirs, prefer_exported=False
        )

    pf = wq.preflight_before_start(
        find_output=_find_complete,
        remove_completed=True,
        requeue_failed=True,
        remove_missing=False,
    )
    if pf.get("dupes") or pf.get("size_dupes") or pf.get("completed_removed") or pf.get("failed_requeued"):
        log(
            "🧹 Step 1 preflight: "
            f"{pf.get('dupes', 0)} path duplicate(s) removed, "
            f"{pf.get('size_dupes', 0)} same-size duplicate(s) skipped, "
            f"{pf.get('completed_removed', 0)} already-complete removed from queue, "
            f"{pf.get('failed_requeued', 0)} failed re-queued · "
            f"{pf.get('pending', 0)} pending left",
            message_type="info",
        )

    pending = wq.pending_items()
    if not pending:
        note = (
            "Queue is empty after preflight (dupes/completed cleared) — drop videos in the watch folder "
            f"({watch_folder or 'set batch_watch_folder'}), then Start / Resume."
        )
        log(note, message_type="warning")
        return None, wq.status_html(note)

    completed_dir = wq.set_fixed_completed_dir(handoff)
    # Progress logs still under app outputs
    log_root = os.path.join(get_output_dir(), "work_queue_video", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_root, exist_ok=True)
    batch_output_dir = log_root
    all_paths = [it["path"] for it in wq.all_items()]
    write_batch_inputs_list(batch_output_dir, all_paths)

    total = len(all_paths)
    pending_count = len(pending)
    log(
        f"Work queue: starting {pending_count} pending of {total} total",
        message_type="info",
    )
    log(f"Upscaled videos → {completed_dir} (Ready for Toolbox)", message_type="info")
    if source_archive:
        log(f"Originals archive (pairing) → {source_archive}", message_type="info")
    if enable_chunks:
        log(f"Chunk mode: {chunk_duration}s segments", message_type="info")
    if batch_resize_preset != "No Resize":
        log(f"Batch resize preset: {batch_resize_preset}", message_type="info")

    last_output_path = None
    fatal_oom = False
    stopped_early = False
    processed_this_run = 0

    for run_i, item in enumerate(list(pending)):
        video_path = item["path"]
        if not os.path.isfile(video_path):
            result = handle_missing_queue_source(
                wq,
                video_path,
                archive_dirs=[watch_folder, source_archive, handoff],
                output_dirs=[completed_dir, handoff, get_output_dir()],
                prefer_exported=False,
            )
            if result == "retry":
                relocated = find_relocated_source(
                    video_path, watch_folder, source_archive, handoff
                )
                if relocated:
                    video_path = relocated
                else:
                    continue
            else:
                continue

        idx, total_q = wq.index_of(video_path)
        label = f"Queue {idx}/{total_q} (this run {run_i + 1}/{pending_count}): {os.path.basename(video_path)}"
        progress((run_i / max(pending_count, 1)), desc=label)
        log(f"\n--- {label} ---", message_type="info")
        wq.set_item_status(video_path, "running")

        if fatal_oom or cuda_context_poisoned(min_free_mb=1500):
            fatal_oom = True
            msg = "Skipped — GPU still exhausted after OOM. Restart app, then Resume."
            log(msg, message_type="warning")
            wq.set_item_status(video_path, "failed", error=msg)
            write_live_batch_progress(
                batch_output_dir,
                total=total_q,
                index=idx - 1,
                source=video_path,
                status="failed",
                error=msg,
                all_sources=all_paths,
            )
            continue

        try:
            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass
                def tqdm(self, iterable, *args, **kwargs):
                    return iterable

            resized_path = apply_batch_resize_preset(
                video_path, batch_resize_preset, scale=scale, progress=DummyProgress()
            )
            process_path = resized_path
            batch_progress = DummyProgress()

            if enable_chunks:
                temp_output_path, _, _, _ = process_video_with_chunks(
                    input_path=process_path,
                    chunk_duration=chunk_duration,
                    mode=mode,
                    model_version=model_version,
                    scale=scale,
                    color_fix=color_fix,
                    tiled_vae=tiled_vae,
                    tiled_dit=tiled_dit,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    unload_dit=unload_dit,
                    dtype_str=dtype_str,
                    seed=seed,
                    device=device,
                    fps_override=fps_override,
                    quality=quality,
                    attention_mode=attention_mode,
                    sparse_ratio=sparse_ratio,
                    kv_ratio=kv_ratio,
                    local_range=local_range,
                    autosave=False,
                    progress=batch_progress,
                )
            else:
                temp_output_path, _, _, _ = run_flashvsr_single(
                    input_path=process_path,
                    mode=mode,
                    model_version=model_version,
                    scale=scale,
                    color_fix=color_fix,
                    tiled_vae=tiled_vae,
                    tiled_dit=tiled_dit,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    unload_dit=unload_dit,
                    dtype_str=dtype_str,
                    seed=seed,
                    device=device,
                    fps_override=fps_override,
                    quality=quality,
                    attention_mode=attention_mode,
                    sparse_ratio=sparse_ratio,
                    kv_ratio=kv_ratio,
                    local_range=local_range,
                    autosave=False,
                    progress=batch_progress,
                )

            if temp_output_path and os.path.exists(temp_output_path):
                filename = os.path.basename(temp_output_path)
                final_path = unique_dest_path(completed_dir, filename)
                shutil.copy(temp_output_path, final_path)
                last_output_path = final_path
                processed_this_run += 1
                # Mark done BEFORE archiving so a crash mid-move cannot requeue a missing path
                wq.set_item_status(video_path, "done", output=final_path)
                archived = archive_original_source(video_path, source_archive)
                log(f"✅ [{idx}/{total_q}] Upscaled → {final_path}", message_type="finish")
                if archived:
                    log(f"   Original archived for pairing → {archived}", message_type="info")
                write_live_batch_progress(
                    batch_output_dir,
                    total=total_q,
                    index=idx - 1,
                    source=video_path,
                    status="done",
                    output=final_path,
                    all_sources=all_paths,
                )
            else:
                wq.set_item_status(video_path, "failed", error="no output")
                log(f"❌ [{idx}/{total_q}] Processing failed", message_type="error")
                write_live_batch_progress(
                    batch_output_dir,
                    total=total_q,
                    index=idx - 1,
                    source=video_path,
                    status="failed",
                    error="processing returned no output",
                    all_sources=all_paths,
                )

        except Exception as e:
            log(f"❌ [{idx}/{total_q}] Error: {e}", message_type="error")
            wq.set_item_status(video_path, "failed", error=str(e))
            write_live_batch_progress(
                batch_output_dir,
                total=total_q,
                index=idx - 1,
                source=video_path,
                status="failed",
                error=str(e),
                all_sources=all_paths,
            )
            if is_cuda_oom(e) or cuda_context_poisoned(min_free_mb=1500):
                fatal_oom = True
                log(
                    "[FlashVSR] Queue pausing remaining after OOM — restart app then Resume.",
                    message_type="error",
                )
        finally:
            release_processing_vram()

        # Soft-stop: finish current (already done), then pause before next
        if wq.stop_requested():
            stopped_early = True
            wq.clear_stop()
            remaining = len(wq.pending_items())
            note = (
                f"⏹ Stopped after finishing file {idx}/{total_q}. "
                f"This run: {processed_this_run} done. Pending left: {remaining}. "
                f"Click Start / Resume when ready."
            )
            log(note, message_type="warning")
            progress(1.0, desc=note)
            return last_output_path, wq.status_html(note)

        if fatal_oom:
            remaining = len(wq.pending_items())
            note = (
                f"⚠️ Paused after OOM at {idx}/{total_q}. "
                f"Restart FlashVSR, then Start / Resume ({remaining} pending)."
            )
            progress(1.0, desc=note)
            return last_output_path, wq.status_html(note)

    remaining = len(wq.pending_items())
    if remaining == 0:
        note = (
            f"✅ Video queue complete — {processed_this_run} finished. "
            f"Upscaled → {completed_dir}. Sort if needed, then Toolbox queue → Ready for CIV."
        )
        log(note, message_type="finish")
    else:
        note = (
            f"Finished this pass ({processed_this_run} → {completed_dir}). "
            f"{remaining} pending — Start / Resume to continue."
        )
        log(note, message_type="info")
    progress(1.0, desc=note)
    return last_output_path, wq.status_html(note)


def run_flashvsr_image_work_queue(
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    create_comparison,
    batch_resize_preset,
    progress=gr.Progress(track_tqdm=True),
):
    """Image upscale queue: watch NEW DOWNLOADS → Ready for CIV/images; originals → Pre Scaled."""
    wq = get_flashvsr_image_queue()
    lock = get_exclusive_queue_lock()
    ok, lock_msg = lock.try_acquire("image")
    if not ok:
        log(lock_msg, message_type="warning")
        return None, _queue_busy_html(wq, lock_msg)
    try:
        return _run_flashvsr_image_work_queue_body(
            wq, mode, model_version, scale, color_fix, tiled_vae, tiled_dit,
            tile_size, tile_overlap, unload_dit, dtype_str, seed, device, fps_override,
            quality, attention_mode, sparse_ratio, kv_ratio, local_range,
            create_comparison, batch_resize_preset, progress,
        )
    finally:
        lock.release("image")


def _run_flashvsr_image_work_queue_body(
    wq,
    mode,
    model_version,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    dtype_str,
    seed,
    device,
    fps_override,
    quality,
    attention_mode,
    sparse_ratio,
    kv_ratio,
    local_range,
    create_comparison,
    batch_resize_preset,
    progress,
):
    wq.clear_stop()
    ui = get_ui_defaults()
    paths = ensure_workflow_dirs(ui)
    watch_folder = (ui.get("batch_watch_folder") or "").strip()
    source_archive = (ui.get("batch_source_archive_dir") or paths["pre_scaled"]).strip()
    handoff = (ui.get("img_upscale_handoff_dir") or paths["img_handoff"]).strip()

    if watch_folder and os.path.isdir(watch_folder):
        added, skipped = wq.add_folder(watch_folder)
        if added:
            log(f"Image watch: added {added} from {watch_folder}", message_type="info")
        elif not skipped:
            log(f"Image watch: no new images in {watch_folder}", message_type="info")

    stuck = wq.reset_stuck_running()
    if stuck:
        log(f"Re-queued {stuck} stuck image job(s)", message_type="info")

    reconcile_work_queue(
        wq,
        archive_dirs=[watch_folder, source_archive, handoff],
        output_dirs=[handoff, paths.get("img_handoff", "")],
        prefer_exported=False,
        label="Image queue",
    )

    def _find_img_complete(path: str):
        return find_matching_deliverable(
            path, handoff, paths.get("img_handoff", ""), prefer_exported=False
        )

    pf = wq.preflight_before_start(
        find_output=_find_img_complete,
        remove_completed=True,
        requeue_failed=True,
        remove_missing=False,
    )
    if pf.get("dupes") or pf.get("size_dupes") or pf.get("completed_removed") or pf.get("failed_requeued"):
        log(
            "🧹 Image queue preflight: "
            f"{pf.get('dupes', 0)} path duplicate(s) removed, "
            f"{pf.get('size_dupes', 0)} same-size duplicate(s) skipped, "
            f"{pf.get('completed_removed', 0)} already-complete removed, "
            f"{pf.get('failed_requeued', 0)} failed re-queued · "
            f"{pf.get('pending', 0)} pending left",
            message_type="info",
        )

    pending = wq.pending_items()
    if not pending:
        note = f"Image queue empty after preflight — drop images in {watch_folder or 'watch folder'}."
        return None, wq.status_html(note)

    completed_dir = wq.set_fixed_completed_dir(handoff)
    log(f"Image upscales → {completed_dir}", message_type="info")
    last_output = None
    processed = 0
    pending_count = len(pending)

    for run_i, item in enumerate(list(pending)):
        image_path = item["path"]
        if not os.path.isfile(image_path):
            result = handle_missing_queue_source(
                wq,
                image_path,
                archive_dirs=[watch_folder, source_archive, handoff],
                output_dirs=[completed_dir, handoff],
                prefer_exported=False,
            )
            if result == "retry":
                relocated = find_relocated_source(
                    image_path, watch_folder, source_archive, handoff
                )
                if relocated:
                    image_path = relocated
                else:
                    continue
            else:
                if result == "done":
                    processed += 1
                continue
        idx, total_q = wq.index_of(image_path)
        label = f"Image {idx}/{total_q} (run {run_i + 1}/{pending_count}): {os.path.basename(image_path)}"
        progress(run_i / max(pending_count, 1), desc=label)
        log(f"\n--- {label} ---", message_type="info")
        wq.set_item_status(image_path, "running")

        try:
            class DummyProgress:
                def __call__(self, *args, **kwargs):
                    pass
                def tqdm(self, iterable, *args, **kwargs):
                    return iterable

            proc = image_path
            # Resize preset mode is NOT the FlashVSR pipeline mode.
            resize_mode, max_width = parse_batch_resize_preset(batch_resize_preset, scale=scale)
            if resize_mode != "none":
                cw, ch = get_image_dimensions(image_path)
                nw, nh, will = calculate_resize_dimensions(
                    cw, ch, max_width=max_width, scale=scale, mode=resize_mode
                )
                if will:
                    proc = resize_input_image(
                        image_path,
                        max_width,
                        scale=scale,
                        progress=DummyProgress(),
                        mode=resize_mode,
                    )
                    log(
                        f"Image resize {cw}×{ch} → {nw}×{nh} "
                        f"(at {scale}× → {nw * int(scale)}×{nh * int(scale)} UHD-safe)",
                        message_type="info",
                    )

            out_path, _, _, _ = run_flashvsr_image(
                image_path=proc,
                mode=mode,
                model_version=model_version,
                scale=scale,
                color_fix=color_fix,
                tiled_vae=tiled_vae,
                tiled_dit=tiled_dit,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                unload_dit=unload_dit,
                dtype_str=dtype_str,
                seed=seed,
                device=device,
                fps_override=fps_override,
                quality=quality,
                attention_mode=attention_mode,
                sparse_ratio=sparse_ratio,
                kv_ratio=kv_ratio,
                local_range=local_range,
                autosave=False,
                create_comparison=create_comparison,
                progress=DummyProgress(),
            )
            if out_path and os.path.exists(out_path):
                final = unique_dest_path(completed_dir, os.path.basename(out_path))
                shutil.copy(out_path, final)
                last_output = final
                processed += 1
                # Mark done before archive so crash recovery never sees a missing source as pending
                wq.set_item_status(image_path, "done", output=final)
                archive_original_source(image_path, source_archive)
                log(f"✅ Image → {final}", message_type="finish")
            else:
                wq.set_item_status(image_path, "failed", error="no output")
        except Exception as e:
            log(f"❌ Image error: {e}", message_type="error")
            wq.set_item_status(image_path, "failed", error=str(e))
        finally:
            release_processing_vram()

        if wq.stop_requested():
            wq.clear_stop()
            note = f"⏹ Stopped after image {idx}/{total_q}. {processed} done this run."
            return last_output, wq.status_html(note)

    remaining = len(wq.pending_items())
    note = (
        f"✅ Image queue done — {processed} this run → {completed_dir}"
        if remaining == 0
        else f"Pass done ({processed}). {remaining} pending."
    )
    progress(1.0, desc=note)
    return last_output, wq.status_html(note)


def run_toolbox_work_queue(progress=gr.Progress(track_tqdm=True)):
    """
    Post-upscale pipeline: scan Ready for Toolbox inbox,
    Frame Adjust 4x + Export → Ready for CIV.
    """
    wq = get_toolbox_work_queue()
    lock = get_exclusive_queue_lock()
    ok, lock_msg = lock.try_acquire("toolbox")
    if not ok:
        log(lock_msg, message_type="warning")
        return None, _queue_busy_html(wq, lock_msg)
    try:
        return _run_toolbox_work_queue_body(wq, progress)
    finally:
        lock.release("toolbox")


def _toolbox_item_timeout_sec() -> int:
    """Wall-clock limit per toolbox item (default 60 min). Env: FLASHVSR_TOOLBOX_ITEM_TIMEOUT."""
    try:
        return max(120, int(os.environ.get("FLASHVSR_TOOLBOX_ITEM_TIMEOUT", "3600")))
    except (TypeError, ValueError):
        return 3600


def _toolbox_max_attempts() -> int:
    try:
        return max(1, int(os.environ.get("FLASHVSR_TOOLBOX_MAX_ATTEMPTS", "3")))
    except (TypeError, ValueError):
        return 3


def _toolbox_error_is_permanent(reason: str, messages: str = "") -> bool:
    blob = f"{reason}\n{messages}".lower()
    needles = (
        "no video stream",
        "audio-only",
        "does not contain any stream",
        "could not load meta information",
        "output file does not contain any stream",
        "already 200+ fps",
        "highfps",
        "over ",
        "fps cap",
    )
    return any(n in blob for n in needles)


def _parse_fps_rate(text: str) -> float:
    text = (text or "").strip()
    if not text or text in ("0/0", "N/A"):
        return 0.0
    try:
        if "/" in text:
            a, b = text.split("/", 1)
            den = float(b)
            return (float(a) / den) if den else 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def probe_file_wh(video_path: str) -> tuple:
    """ffprobe width, height. (0, 0) if unknown / no video."""
    if not video_path or not os.path.isfile(video_path):
        return 0, 0
    try:
        exe = "ffprobe"
        if toolbox_processor and getattr(toolbox_processor, "ffprobe_exe", None):
            exe = toolbox_processor.ffprobe_exe
        out = subprocess.run(
            [
                exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                video_path,
            ],
            capture_output=True, text=True, timeout=12, check=False,
        ).stdout.strip()
        if "x" in out:
            w, h = out.split("x", 1)
            return int(float(w)), int(float(h))
    except Exception:
        pass
    try:
        return get_video_dimensions(video_path)
    except Exception:
        return 0, 0


def probe_file_fps(video_path: str) -> float:
    """ffprobe / imageio FPS. 0 if unknown."""
    if not video_path or not os.path.isfile(video_path):
        return 0.0
    try:
        exe = "ffprobe"
        if toolbox_processor and getattr(toolbox_processor, "ffprobe_exe", None):
            exe = toolbox_processor.ffprobe_exe
        out = subprocess.run(
            [
                exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=12, check=False,
        ).stdout.splitlines()
        for line in out:
            fps = _parse_fps_rate(line)
            if 1.0 <= fps <= 480.0:
                return fps
    except Exception:
        pass
    try:
        with imageio.get_reader(video_path) as reader:
            fps = float((reader.get_meta_data() or {}).get("fps") or 0)
        if 1.0 <= fps <= 480.0:
            return fps
    except Exception:
        pass
    return 0.0


def scale_back_fps(src_path: str, dest_path: str, target_fps: float = 60.0) -> Optional[str]:
    """Write a playable copy at target_fps (frame drop, no RIFE)."""
    if not src_path or not os.path.isfile(src_path):
        return None
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-filter:v", f"fps={int(round(target_fps))}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        dest_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        return dest_path if os.path.isfile(dest_path) else None
    except Exception as e:
        log(f"FPS scale-back failed for {os.path.basename(src_path)}: {e}", message_type="warning")
        return None


def quarantine_high_fps(video_path: str, *, fps: float, scale_to: float = 60.0) -> Optional[str]:
    """Move an already-too-fast file out of the inbox / CIV folder and make a 60fps copy."""
    if not video_path or not os.path.isfile(video_path):
        return None
    dest_dir = os.path.join(os.path.dirname(video_path), "HighFPS")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest = unique_dest_path(dest_dir, os.path.basename(video_path))
        shutil.move(video_path, dest)
    except OSError as e:
        log(f"Could not quarantine high-FPS file: {e}", message_type="warning")
        return None
    playable_dir = os.path.join(dest_dir, f"at_{int(round(scale_to))}fps")
    playable = unique_dest_path(
        playable_dir,
        f"{Path(dest).stem}_at{int(round(scale_to))}fps{Path(dest).suffix}",
    )
    made = scale_back_fps(dest, playable, target_fps=scale_to)
    log(
        f"📦 High-FPS ({fps:.0f}) → {dest}"
        + (f" · playable {int(round(scale_to))}fps copy → {made}" if made else ""),
        message_type="warning",
    )
    return dest


def sweep_high_fps_folder(folder: str, *, floor: float = 160.0, scale_to: float = 60.0) -> int:
    """Move already-exported 200fps+ files out of a deliverable folder."""
    if not folder or not os.path.isdir(folder):
        return 0
    moved = 0
    try:
        names = list(os.listdir(folder))
    except OSError:
        return 0
    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if Path(path).suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
            continue
        stem = Path(name).stem.lower()
        looks_fast = bool(re.search(r"_(1[8-9]\d|2\d\d|3\d\d|4[0-7]\d)fps$", stem))
        fps = probe_file_fps(path)
        if not looks_fast and fps < floor:
            continue
        if fps and fps < floor and not looks_fast:
            continue
        if quarantine_high_fps(path, fps=fps or 240.0, scale_to=scale_to):
            moved += 1
    return moved


def quarantine_novideo_source(video_path: str) -> Optional[str]:
    """Move audio-only / no-video inbox files out of the toolbox queue folder."""
    if not video_path or not os.path.isfile(video_path):
        return None
    dest_dir = os.path.join(os.path.dirname(video_path), "NoVideo")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest = unique_dest_path(dest_dir, os.path.basename(video_path))
        shutil.move(video_path, dest)
        return dest
    except OSError as e:
        log(f"Could not quarantine no-video file: {e}", message_type="warning")
        return None


_HYGIENE_SKIP_DIRS = {
    "novideo", "highfps", "over4k", "bin", "at_60fps", "done", "archive", "work",
}


def _hygiene_video_files(folder: str) -> list:
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    try:
        for p in Path(folder).iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            out.append(str(p))
    except OSError:
        pass
    return out


def _target_hygiene_size(w: int, h: int, *, role: str, scale: int) -> tuple:
    """Return (new_w, new_h, needs_resize)."""
    if w <= 0 or h <= 0:
        return w, h, False
    if role == "watch":
        return calculate_resize_dimensions(w, h, mode="4k_safe", scale=scale)
    lim_w, lim_h = uhd_4k_output_limits(w, h)
    if w <= lim_w and h <= lim_h:
        return w, h, False
    factor = min(lim_w / float(w), lim_h / float(h), 1.0)
    nw = max(2, int(w * factor) // 2 * 2)
    nh = max(2, int(h * factor) // 2 * 2)
    return nw, nh, (nw, nh) != (w, h)


def _ffmpeg_scale_to(src_path: str, dest_path: str, new_w: int, new_h: int) -> Optional[str]:
    """
    FlashVSR-style 4K-safe pre-downscale: cover the target box (no stretch),
    center-crop to the grid-aligned size, high-quality intermediate (CRF 14 / slow).
    Same filter as resize_input_video() so 4× stays inside UHD without crushing detail.
    """
    if not src_path or not os.path.isfile(src_path):
        return None
    if not is_ffmpeg_available():
        log("FFmpeg not available — cannot 4K-safe downscale", message_type="error")
        return None
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    nw, nh = int(new_w), int(new_h)
    vf = (
        f"scale={nw}:{nh}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={nw}:{nh}"
    )
    base = [
        "ffmpeg", "-y", "-i", src_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "slow", "-crf", "14",
        "-pix_fmt", "yuv420p",
        "-map", "0:v:0",
        "-map", "0:a:0?",
    ]
    cmd = base + ["-c:a", "aac", "-b:a", "256k", dest_path]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900)
    except Exception as e:
        log(f"4K-safe encode failed for {os.path.basename(src_path)}: {e}", message_type="error")
        if os.path.isfile(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return None
    if not os.path.isfile(dest_path) or os.path.getsize(dest_path) < 1024:
        return None
    return dest_path


def downscale_replace_uhd(src_path: str, new_w: int, new_h: int) -> Optional[str]:
    """
    Archive the oversize original to Over4K\\ and write a 4K-safe file
    back at the same path so queue rows keep working.
    """
    if not src_path or not os.path.isfile(src_path):
        return None
    if not is_ffmpeg_available():
        log("FFmpeg not available — cannot fix over-4K file", message_type="error")
        return None
    folder = os.path.dirname(src_path)
    archive_dir = os.path.join(folder, "Over4K")
    os.makedirs(archive_dir, exist_ok=True)
    archived = unique_dest_path(archive_dir, os.path.basename(src_path))
    tmp = os.path.join(
        folder,
        f".{Path(src_path).stem}_4kfix_{time.strftime('%H%M%S')}{Path(src_path).suffix}",
    )
    if not _ffmpeg_scale_to(src_path, tmp, new_w, new_h):
        return None
    try:
        shutil.move(src_path, archived)
        os.replace(tmp, src_path)
    except OSError as e:
        log(f"Over-4K replace failed: {e}", message_type="error")
        return None
    log(
        f"🖼 4K-safe pre-downscale {os.path.basename(src_path)} → {new_w}×{new_h} "
        f"(cover+crop, CRF 14 — 4× stays UHD). Original in Over4K\\",
        message_type="finish",
    )
    return src_path


def _inbox_has_file(inbox: str, name: str) -> bool:
    if not inbox or not name:
        return False
    return os.path.isfile(os.path.join(inbox, name))


def hygiene_reclaim_sidecars(
    inbox: str,
    *,
    role: str = "toolbox",
    scale: int = 4,
    high_fps_floor: float = 160.0,
    scale_back_fps_to: float = 60.0,
) -> dict:
    """
    Scan folders hygiene *moves files into* (Over4K, HighFPS) and put a
    processable copy back in the inbox so the queue can pick it up.
    """
    stats = {"reclaimed": 0, "errors": 0}
    if not inbox or not os.path.isdir(inbox):
        return stats
    scale = max(1, int(scale or 4))
    target_fps = int(round(scale_back_fps_to or 60))

    over4k = os.path.join(inbox, "Over4K")
    for src in _hygiene_video_files(over4k):
        name = os.path.basename(src)
        dest = os.path.join(inbox, name)
        if _inbox_has_file(inbox, name):
            continue
        w, h = probe_file_wh(src)
        if w <= 0 or h <= 0:
            continue
        nw, nh, will = _target_hygiene_size(w, h, role=role, scale=scale)
        try:
            if will:
                if not _ffmpeg_scale_to(src, dest, nw, nh):
                    stats["errors"] += 1
                    continue
            else:
                shutil.copy2(src, dest)
            stats["reclaimed"] += 1
            log(
                f"↩️  Reclaimed from Over4K → queue: {name}"
                + (
                    f" ({w}×{h} → {nw}×{nh} 4K-safe so 4× stays UHD)"
                    if will
                    else ""
                ),
                message_type="finish",
            )
        except Exception as e:
            stats["errors"] += 1
            log(f"Reclaim Over4K failed {name}: {e}", message_type="warning")

    high = os.path.join(inbox, "HighFPS")
    playable = os.path.join(high, f"at_{target_fps}fps")
    # Prefer already-scaled playable copies first
    for src in _hygiene_video_files(playable):
        name = os.path.basename(src)
        if _inbox_has_file(inbox, name):
            continue
        dest = os.path.join(inbox, name)
        try:
            shutil.copy2(src, dest)
            stats["reclaimed"] += 1
            log(f"↩️  Reclaimed {target_fps}fps copy → queue: {name}", message_type="finish")
        except Exception as e:
            stats["errors"] += 1
            log(f"Reclaim HighFPS/at_{target_fps} failed {name}: {e}", message_type="warning")

    for src in _hygiene_video_files(high):
        name = os.path.basename(src)
        stem, ext = os.path.splitext(name)
        dest_name = f"{stem}_at{target_fps}fps{ext}"
        if _inbox_has_file(inbox, dest_name) or _inbox_has_file(inbox, name):
            continue
        fps = probe_file_fps(src)
        dest = os.path.join(inbox, dest_name)
        try:
            if fps >= high_fps_floor:
                if not scale_back_fps(src, dest, target_fps=target_fps):
                    stats["errors"] += 1
                    continue
            else:
                # Already playable — send back as-is
                dest = os.path.join(inbox, name)
                if _inbox_has_file(inbox, name):
                    continue
                shutil.copy2(src, dest)
            stats["reclaimed"] += 1
            log(
                f"↩️  Reclaimed from HighFPS → queue: {os.path.basename(dest)}"
                + (f" ({fps:.0f} → {target_fps} fps)" if fps >= high_fps_floor else ""),
                message_type="finish",
            )
        except Exception as e:
            stats["errors"] += 1
            log(f"Reclaim HighFPS failed {name}: {e}", message_type="warning")

    return stats


def hygiene_scan_folder(
    folder: str,
    *,
    role: str = "toolbox",
    scale: int = 4,
    high_fps_floor: float = 160.0,
    scale_back_fps_to: float = 60.0,
) -> dict:
    """
    Scan New Downloads (role=watch) or Ready for Toolbox (role=toolbox)
    and fix/quarantine bad files before the queue picks them up.

    Rules:
      - no video stream → NoVideo\\
      - already 160+ FPS → HighFPS\\ + 60fps playable copy
      - would exceed UHD after ×scale (watch) or already over UHD (toolbox)
        → FlashVSR 4K-safe pre-downscale (lanczos cover + center crop, CRF 14).
        Original archived in Over4K\\
    """
    stats = {"novideo": 0, "highfps": 0, "over4k": 0, "reclaimed": 0, "ok": 0, "errors": 0}
    if not folder or not os.path.isdir(folder):
        return stats
    scale = max(1, int(scale or 4))
    # First: pull fixable files out of Over4K / HighFPS back into this inbox
    rec = hygiene_reclaim_sidecars(
        folder,
        role=role,
        scale=scale,
        high_fps_floor=high_fps_floor,
        scale_back_fps_to=scale_back_fps_to,
    )
    stats["reclaimed"] += int(rec.get("reclaimed") or 0)
    stats["errors"] += int(rec.get("errors") or 0)

    for path in _hygiene_video_files(folder):
        name = os.path.basename(path)
        try:
            w, h = probe_file_wh(path)
            if w <= 0 or h <= 0:
                if quarantine_novideo_source(path):
                    stats["novideo"] += 1
                    log(f"🧹 {name}: no video stream → NoVideo\\", message_type="warning")
                else:
                    stats["errors"] += 1
                continue

            fps = probe_file_fps(path)
            if fps >= high_fps_floor:
                if quarantine_high_fps(path, fps=fps, scale_to=scale_back_fps_to):
                    stats["highfps"] += 1
                else:
                    stats["errors"] += 1
                continue

            nw, nh, will = _target_hygiene_size(w, h, role=role, scale=scale)
            if will and (nw, nh) != (w, h):
                if downscale_replace_uhd(path, nw, nh):
                    stats["over4k"] += 1
                else:
                    stats["errors"] += 1
                continue
            stats["ok"] += 1
        except Exception as e:
            stats["errors"] += 1
            log(f"Hygiene skip {name}: {e}", message_type="warning")
    return stats


def _log_hygiene(folder: str, role: str, stats: dict) -> None:
    if not any(stats.get(k) for k in ("novideo", "highfps", "over4k", "reclaimed", "errors")):
        return
    log(
        f"🧹 Folder hygiene ({role}) {folder}: "
        f"{stats.get('reclaimed', 0)} put back in queue, "
        f"{stats.get('over4k', 0)} 4K-safe pre-downscale, "
        f"{stats.get('highfps', 0)} high-FPS moved, "
        f"{stats.get('novideo', 0)} no-video moved, "
        f"{stats.get('ok', 0)} ok, "
        f"{stats.get('errors', 0)} errors",
        message_type="info",
    )


def _fail_toolbox_item_requeue(
    wq, video_path: str, reason: str, *, max_attempts: int, permanent: bool = False
) -> str:
    """
    Fail this attempt and put the source back at the END of the queue (pending)
    so other jobs run first. Source file is left in place (inbox).
    Returns: requeued | failed_permanent | missing
    """
    if permanent:
        wq.set_item_status(video_path, "failed", error=reason)
        name = os.path.basename(video_path)
        log(f"❌ {name} skipped (will not retry). Reason: {reason}", message_type="error")
        return "failed_permanent"
    result = wq.requeue_to_end(video_path, error=reason, max_attempts=max_attempts)
    name = os.path.basename(video_path)
    if result == "requeued":
        log(
            f"↩️  {name} → end of queue (will retry later). Reason: {reason}",
            message_type="warning",
        )
    elif result == "failed_permanent":
        log(
            f"❌ {name} permanently failed after max attempts. Reason: {reason}",
            message_type="error",
        )
    else:
        log(f"❌ {name}: {reason}", message_type="error")
    return result


def _run_toolbox_work_queue_body(wq, progress):
    global toolbox_processor
    import concurrent.futures

    wq.clear_stop()
    ui = get_ui_defaults()
    paths = ensure_workflow_dirs(ui)
    inbox = (ui.get("tb_inbox_folder") or paths["tb_inbox"]).strip()
    ready_civ = get_toolbox_output_dir()
    source_archive = (ui.get("batch_source_archive_dir") or paths["pre_scaled"]).strip()
    item_timeout = _toolbox_item_timeout_sec()
    max_attempts = _toolbox_max_attempts()

    if toolbox_processor is None:
        toolbox_processor = ToolboxProcessor(True)
    toolbox_processor.output_dir = Path(ready_civ)
    toolbox_processor.autosave_enabled = True

    if inbox and os.path.isdir(inbox):
        _log_hygiene(
            inbox,
            "toolbox",
            hygiene_scan_folder(
                inbox,
                role="toolbox",
                scale=int(ui.get("scale") or 4),
                high_fps_floor=float(ui.get("tb_high_fps_floor") or 160),
                scale_back_fps_to=float(ui.get("tb_scale_back_fps") or 60),
            ),
        )
        added, skipped = wq.add_folder(inbox)
        if added:
            log(f"Toolbox inbox: added {added} from {inbox}", message_type="info")
        elif not skipped:
            log(f"Toolbox inbox empty: {inbox}", message_type="info")
    elif inbox:
        log(f"Toolbox inbox missing: {inbox}", message_type="warning")

    # Stuck "running" items go to the END (not front) so they don't immediately re-block
    stuck = wq.reset_stuck_running(to_end=True, max_attempts=max_attempts)
    if stuck:
        log(
            f"Re-queued {stuck} stuck toolbox job(s) → end of queue "
            f"(source kept; will retry after others)",
            message_type="info",
        )

    # Heal stale rows: exported already in Ready for CIV, or source only in inbox/done
    reconcile_work_queue(
        wq,
        archive_dirs=[inbox, source_archive],
        output_dirs=[ready_civ, paths.get("ready_civ", ""), str(Path(ROOT_DIR) / "outputs" / "toolbox")],
        prefer_exported=True,
        label="Toolbox queue",
    )

    pending = wq.pending_items()
    if not pending:
        note = (
            f"Toolbox queue empty — place upscaled videos in:\n{inbox}\n"
            f"then Start / Resume. Finals go to Ready for CIV."
        )
        return None, wq.status_html(note)

    ops_raw = ui.get("tb_pipeline_ops") or "Frame Adjust,Export"
    selected_ops = [o.strip() for o in str(ops_raw).split(",") if o.strip()]
    if not selected_ops:
        selected_ops = ["Frame Adjust", "Export"]
    fps_mode = ui.get("tb_fps_mode") or "4x Frames"
    max_out_fps = float(ui.get("tb_max_out_fps") or 120)
    high_fps_floor = float(ui.get("tb_high_fps_floor") or 160)
    scale_back_fps_to = float(ui.get("tb_scale_back_fps") or 60)
    os.environ["FLASHVSR_MAX_OUT_FPS"] = str(int(max_out_fps))
    frames_q = int(ui.get("tb_frames_quality") or 95)
    export_q = int(ui.get("tb_export_quality") or 92)
    export_w = int(ui.get("tb_export_max_width") or 3840)
    # Streaming avoids loading full upscaled clips into RAM (main cause of silent hangs)
    use_streaming = ui.get("tb_use_streaming")
    if use_streaming is None:
        use_streaming = True
    else:
        use_streaming = bool(use_streaming)
    # Quality-preserving speed: medium x264 (not slow) + NVENC when available
    export_preset = (ui.get("tb_export_preset") or "medium").strip().lower()
    prefer_nvenc = ui.get("tb_prefer_nvenc")
    if prefer_nvenc is None:
        prefer_nvenc = True
    else:
        prefer_nvenc = bool(prefer_nvenc)

    params = {
        "frame_adjust": {
            "fps_mode": fps_mode,
            "speed_factor": 1.0,
            "use_streaming": use_streaming,
            "output_quality": frames_q,
        },
        "loop": {"loop_type": "loop", "num_loops": 1, "output_quality": frames_q},
        "export": {
            "export_format": "MP4 (H.264)",
            "quality": export_q,
            "max_width": export_w,
            "output_name": "",
            "two_pass": False,
        },
    }

    toolbox_processor.export_preset = export_preset
    toolbox_processor.prefer_nvenc = prefer_nvenc

    completed_dir = wq.set_fixed_completed_dir(ready_civ)
    swept = sweep_high_fps_folder(ready_civ, floor=high_fps_floor, scale_to=scale_back_fps_to)
    if swept:
        log(
            f"High-FPS sweep (Ready for CIV): moved {swept} already-200+ file(s) → HighFPS",
            message_type="warning",
        )

    log(
        f"Toolbox: {selected_ops} | RIFE {fps_mode} "
        f"(stream={use_streaming}, export={export_preset}, nvenc={prefer_nvenc}, "
        f"max_out={int(max_out_fps)}fps, timeout={item_timeout}s, max_try={max_attempts}) → {ready_civ}",
        message_type="info",
    )
    last_out = None
    processed = 0
    requeued = 0
    permanent_fail = 0
    pending_count = len(pending)

    class DummyProgress:
        def __call__(self, *args, **kwargs):
            pass

        def tqdm(self, iterable, *args, **kwargs):
            return iterable

    for run_i, item in enumerate(list(pending)):
        video_path = item["path"]
        if not os.path.isfile(video_path):
            result = handle_missing_queue_source(
                wq,
                video_path,
                archive_dirs=[inbox, source_archive],
                output_dirs=[
                    ready_civ,
                    paths.get("ready_civ", ""),
                    str(Path(ROOT_DIR) / "outputs" / "toolbox"),
                ],
                prefer_exported=True,
            )
            if result == "done":
                processed += 1
                continue
            if result == "retry":
                relocated = find_relocated_source(video_path, inbox, source_archive)
                if relocated:
                    video_path = relocated
                else:
                    permanent_fail += 1
                    continue
            else:
                permanent_fail += 1
                continue

        idx, total_q = wq.index_of(video_path)
        attempt_n = int(item.get("attempts") or 0) + 1
        label = (
            f"Toolbox {idx}/{total_q} (run {run_i + 1}/{pending_count}, "
            f"try {attempt_n}/{max_attempts}): {os.path.basename(video_path)}"
        )
        progress(run_i / max(pending_count, 1), desc=label)
        log(f"\n--- {label} ---", message_type="info")
        src_fps = probe_file_fps(video_path)
        if src_fps >= high_fps_floor:
            q = quarantine_high_fps(video_path, fps=src_fps, scale_to=scale_back_fps_to)
            reason = (
                f"Already {src_fps:.0f} FPS — not 4× RIFE (would be ~{src_fps * 4:.0f}). "
                f"Moved to HighFPS"
                + (f" → {q}" if q else "")
            )
            log(reason, message_type="warning")
            _fail_toolbox_item_requeue(
                wq, video_path, reason, max_attempts=max_attempts, permanent=True
            )
            permanent_fail += 1
            continue
        wq.set_item_status(video_path, "running")
        t0 = time.time()
        result_path, messages = None, ""

        try:
            # Hard wall-clock so a hung RIFE/ffmpeg cannot block the whole queue forever
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    toolbox_processor.process_pipeline,
                    video_path,
                    selected_ops,
                    params,
                    DummyProgress(),
                )
                try:
                    result_path, messages = fut.result(timeout=item_timeout)
                except concurrent.futures.TimeoutError:
                    elapsed = int(time.time() - t0)
                    reason = (
                        f"Timed out after {elapsed}s (limit {item_timeout}s) — "
                        f"source returned to end of queue"
                    )
                    log(
                        f"⏱ TIMEOUT: {os.path.basename(video_path)} ({elapsed}s)",
                        message_type="error",
                    )
                    try:
                        if toolbox_processor and getattr(toolbox_processor, "rife_handler", None):
                            toolbox_processor.rife_handler.unload_model()
                    except Exception:
                        pass
                    res = _fail_toolbox_item_requeue(
                        wq, video_path, reason, max_attempts=max_attempts
                    )
                    if res == "requeued":
                        requeued += 1
                    else:
                        permanent_fail += 1
                    continue

            log(messages, message_type="info")
            if result_path and os.path.exists(result_path):
                # Single final only — autosave may already have written into Ready for CIV
                final = finalize_output_once(result_path, ready_civ)
                # Step 2 name: keep step-1 tags + approximate FPS
                try:
                    fps_est = None
                    try:
                        meta = imageio.get_reader(final).get_meta_data()
                        fps_est = meta.get("fps")
                    except Exception:
                        fps_est = None
                    if not fps_est:
                        base_fps = probe_file_fps(video_path) or 30.0
                        requested = 4 if "4x" in str(fps_mode) else (2 if "2x" in str(fps_mode) else 1)
                        factor = ToolboxProcessor._choose_interp_factor(
                            base_fps, requested, max_out_fps
                        )
                        fps_est = base_fps * factor
                    final = rename_to_step2(
                        final,
                        source_stem=Path(video_path).stem,
                        fps=fps_est,
                        ext=Path(final).suffix or ".mp4",
                    )
                except Exception as e:
                    log(f"Step-2 rename skipped: {e}", message_type="warning")
                last_out = final
                processed += 1
                # Drop leftover RIFE temps for this job (space)
                try:
                    stem_hint = Path(video_path).stem[:40]
                    for folder in (
                        Path(TEMP_DIR) / "toolbox",
                        Path(ROOT_DIR) / "_temp" / "toolbox",
                        Path(ROOT_DIR) / "outputs" / "toolbox",
                    ):
                        if not folder.is_dir():
                            continue
                        for f in folder.glob("*.mp4"):
                            n = f.name
                            if stem_hint and stem_hint[:24] in n and (
                                "_frames_" in n or n.endswith("_2.mp4")
                            ):
                                try:
                                    f.unlink()
                                except OSError:
                                    pass
                except Exception:
                    pass
                # Mark done BEFORE moving step-1 source into Bin
                wq.set_item_status(video_path, "done", output=final)
                # Step-1 upscale file → same folder\Bin\ (user can delete after checking finals)
                binned = move_to_bin(video_path)
                if binned:
                    log(f"🗑 Step-1 moved to Bin → {binned}", message_type="info")
                log(
                    f"✅ Step 2 final → {final} ({int(time.time() - t0)}s)",
                    message_type="finish",
                )
            else:
                reason = "pipeline returned no output"
                msgs = messages if isinstance(messages, str) else ""
                if _toolbox_error_is_permanent(reason, msgs):
                    reason = (
                        "No video stream (audio-only / corrupt). "
                        "RIFE cannot interpolate this file."
                    )
                    q = quarantine_novideo_source(video_path)
                    if q:
                        log(f"📦 Quarantined no-video file → {q}", message_type="warning")
                    res = _fail_toolbox_item_requeue(
                        wq, video_path, reason, max_attempts=max_attempts, permanent=True
                    )
                else:
                    res = _fail_toolbox_item_requeue(
                        wq, video_path, reason, max_attempts=max_attempts
                    )
                if res == "requeued":
                    requeued += 1
                else:
                    permanent_fail += 1
        except Exception as e:
            log(f"❌ Toolbox error: {e}", message_type="error")
            if _toolbox_error_is_permanent(str(e)):
                q = quarantine_novideo_source(video_path)
                if q:
                    log(f"📦 Quarantined no-video file → {q}", message_type="warning")
                res = _fail_toolbox_item_requeue(
                    wq, video_path, str(e), max_attempts=max_attempts, permanent=True
                )
            else:
                res = _fail_toolbox_item_requeue(
                    wq, video_path, str(e), max_attempts=max_attempts
                )
            if res == "requeued":
                requeued += 1
            else:
                permanent_fail += 1
        finally:
            release_processing_vram()

        if wq.stop_requested():
            wq.clear_stop()
            note = (
                f"⏹ Toolbox stopped after {idx}/{total_q}. "
                f"{processed} done, {requeued} requeued → {ready_civ}"
            )
            return last_out, wq.status_html(note)

    remaining = len(wq.pending_items())
    note = (
        f"✅ Toolbox queue complete — {processed} → {ready_civ}"
        f" (requeued {requeued}, permanent fail {permanent_fail})"
        if remaining == 0
        else (
            f"Pass done: {processed} ok, {requeued} sent to end for later, "
            f"{permanent_fail} permanent fail. {remaining} still pending."
        )
    )
    progress(1.0, desc=note)
    return last_out, wq.status_html(note)


def get_video_dimensions(video_path):
    """Get video dimensions quickly. Returns (width, height) or (0, 0) on error."""
    try:
        if not video_path or not os.path.exists(video_path):
            return 0, 0
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        size = meta.get('size', (0, 0))
        width, height = int(size[0]), int(size[1]) if isinstance(size, tuple) else (0, 0)
        reader.close()
        return width, height
    except:
        return 0, 0

def analyze_input_video(video_path):
    """Analyzes video and returns compact HTML display for FlashVSR tab."""
    if not video_path:
        return '<div style="padding: 12px; background: #3d2e0a; border: 1px solid #854d0e; border-radius: 6px; color: #fbbf24;">⚠️ No video provided</div>', 0, 0
    
    try:
        resolved_path = str(Path(video_path).resolve())
        
        # Get file size
        file_size_display = "N/A"
        if os.path.exists(resolved_path):
            size_bytes = os.path.getsize(resolved_path)
            if size_bytes < 1024**2:
                file_size_display = f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                file_size_display = f"{size_bytes/1024**2:.1f} MB"
            else:
                file_size_display = f"{size_bytes/1024**3:.2f} GB"
        
        # Try imageio for quick analysis
        reader = imageio.get_reader(resolved_path)
        meta = reader.get_meta_data()
        
        # Extract info
        duration = meta.get('duration', 0)
        fps = meta.get('fps', 30)
        size = meta.get('size', (0, 0))
        width, height = int(size[0]), int(size[1]) if isinstance(size, tuple) else (0, 0)
        
        # Frame count
        nframes = meta.get('nframes')
        if nframes and nframes != float('inf'):
            frame_count = int(nframes)
        elif duration and fps:
            frame_count = int(duration * fps)
        else:
            frame_count = 0
        
        reader.close()
        
        # Build compact HTML display
        html = f'''
        <div style="padding: 16px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border: 1px solid #667eea40; border-radius: 8px; font-family: 'Segoe UI', sans-serif;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #1a2838 0%, rgba(26, 40, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">RESOLUTION</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{width}×{height}</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e1a38 0%, rgba(30, 26, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FRAMES</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{frame_count}</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a2838 0%, rgba(26, 40, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">DURATION</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{duration:.2f}s @ {fps:.1f} FPS</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e1a38 0%, rgba(30, 26, 56, 0.85) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FILE SIZE</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{file_size_display}</div>
                </div>
            </div>
            <div style="font-size: 0.8em; color: #666; text-align: center; margin-top: 8px;">
                ℹ️ Model requires output frame dimensions in multiples of 128px. We pad input frames to maintain aspect ratio. Padding is removed during upscale processing.
            </div>
        </div>
        '''
        return html, width, height
        
    except Exception as e:
        return f'<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ Error analyzing video: {str(e)}</div>', 0, 0

# UHD 4K frame limits (long edge / short edge)
UHD_4K_LONG = 3840
UHD_4K_SHORT = 2160


def uhd_4k_output_limits(width, height):
    """
    Max *output* size for UHD 4K by orientation:
      landscape / square → 3840×2160 (16:9 class)
      portrait           → 2160×3840 (9:16 class)
    """
    w, h = int(width or 0), int(height or 0)
    if w >= h:
        return UHD_4K_LONG, UHD_4K_SHORT
    return UHD_4K_SHORT, UHD_4K_LONG


def uhd_4k_input_limits(width, height, scale=4):
    """Max *input* size so (input × scale) stays within UHD 4K for that orientation."""
    s = max(1, int(scale or 4))
    out_w, out_h = uhd_4k_output_limits(width, height)
    return out_w / s, out_h / s


def calculate_resize_dimensions(
    current_width,
    current_height,
    max_width=None,
    align_to=2,
    scale=None,
    max_height=None,
    mode=None,
):
    """
    Calculate new dimensions for resize, maintaining aspect ratio.
    Never upsizes — only downsizes if needed.
    Returns (new_width, new_height, will_resize)

    mode:
      - "4k_safe": fit inside the box where (w×scale, h×scale) ≤ UHD 4K
        (3840×2160 landscape or 2160×3840 portrait).
      - "width" / None: legacy max_width; when scale is set, also clamps to 4K-safe box.
      - optional max_height for an explicit box.

    When scale is set, dims are snapped so (dim×scale) lands on model/codec grids.
    """
    cw = int(current_width or 0)
    ch = int(current_height or 0)
    if cw <= 0 or ch <= 0:
        return current_width, current_height, False

    if scale is not None:
        align_to = max(1, int(resize_align_step(scale)))
    else:
        align_to = max(1, int(align_to or 2))

    mode_l = (str(mode).lower() if mode is not None else "") or ""
    if mode_l in ("4k_safe", "4k-safe", "auto_4k") or (
        isinstance(max_width, str) and is_4k_safe_preset(max_width)
    ):
        box_w, box_h = uhd_4k_input_limits(cw, ch, scale=scale or 4)
    else:
        # Legacy width cap; when scale known, also never exceed 4K after upscale
        if max_width is None or max_width == "" or max_width == 0:
            box_w = float(cw)
        else:
            try:
                box_w = float(max_width)
            except (TypeError, ValueError):
                box_w = float(cw)
        if max_height is not None:
            try:
                box_h = float(max_height)
            except (TypeError, ValueError):
                box_h = float(ch)
        elif scale is not None:
            k_w, k_h = uhd_4k_input_limits(cw, ch, scale=scale)
            box_w = min(box_w, k_w)
            box_h = k_h
        else:
            # width-only, no scale: height free (old behavior)
            box_h = float(ch) * (box_w / float(cw)) if cw else float(ch)

    # Fit entire frame inside box (downscale only)
    factor = min(1.0, box_w / float(cw), box_h / float(ch))
    tw = cw * factor
    th = ch * factor

    # Snap down to alignment grid
    new_width = max(align_to, (int(tw) // align_to) * align_to)
    new_height = max(align_to, (int(th) // align_to) * align_to)

    # Keep aspect as close as possible after grid snap (prefer width-driven height)
    if cw > 0:
        aspect = ch / float(cw)
        alt_h = max(align_to, (int(round(new_width * aspect)) // align_to) * align_to)
        # Prefer alt_h if it still fits the box
        if alt_h <= box_h + 0.5:
            new_height = alt_h
        else:
            # height-limited: recompute width from height
            if ch > 0:
                inv = cw / float(ch)
                alt_w = max(align_to, (int(round(new_height * inv)) // align_to) * align_to)
                if alt_w <= box_w + 0.5:
                    new_width = alt_w

    # Final safety: step down if still over box after rounding
    while new_width > box_w + 0.5 and new_width > align_to:
        new_width -= align_to
    while new_height > box_h + 0.5 and new_height > align_to:
        new_height -= align_to

    will_resize = new_width != cw or new_height != ch
    return int(new_width), int(new_height), will_resize

def preview_resize(video_path, max_width, scale=None, mode=None):
    """Generate preview text showing what resize will do."""
    if not video_path:
        return '<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">No video loaded</div>'
    
    current_width, current_height = get_video_dimensions(video_path)
    if current_width == 0:
        return '<div style="padding: 8px; background: #3d2e0a; border: 1px solid #854d0e; border-radius: 4px; color: #fbbf24; font-size: 0.9em; text-align: center;">⚠️ Could not read video dimensions</div>'

    if scale is None:
        scale = get_ui_defaults()["scale"]
    if mode is None and isinstance(max_width, str) and is_4k_safe_preset(max_width):
        mode = "4k_safe"
        max_width = None
    new_width, new_height, will_resize = calculate_resize_dimensions(
        current_width, current_height, max_width, scale=scale, mode=mode
    )
    align = resize_align_step(scale)
    out_w, out_h = new_width * int(scale), new_height * int(scale)
    lim_w, lim_h = uhd_4k_output_limits(current_width, current_height)
    
    # Check if video is small enough to not need tiled DiT (rough threshold)
    # Tiled DiT is mainly beneficial for larger videos that exceed VRAM
    pixels = current_width * current_height
    small_video_threshold = 512 * 512  # ~512p or smaller
    
    if will_resize:
        reduction = ((current_width * current_height - new_width * new_height) / (current_width * current_height)) * 100
        return (
            f'<div style="padding: 8px; background: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac; font-size: 0.9em; text-align: center;">'
            f'{current_width}×{current_height} → {new_width}×{new_height} ({reduction:.0f}% reduction) ✓<br>'
            f'<span style="font-size: 0.85em;">At {scale}× → {out_w}×{out_h} (UHD cap {lim_w}×{lim_h}, {align}px grid)</span></div>'
        )
    else:
        if pixels <= small_video_threshold:
            return f'<div style="padding: 8px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.9em; text-align: center;">{current_width}×{current_height} (no resize needed) ✓ · {scale}× → {out_w}×{out_h}<br><span style=" color: #0c5460; font-size: 0.9em;">💡 Small resolution - consider disabling Tiled DiT for better speed and quality</span></div>'
        else:
            return f'<div style="padding: 8px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.9em; text-align: center;">{current_width}×{current_height} (no resize / already 4K-safe) ✓ · {scale}× → {out_w}×{out_h}</div>'

def resize_input_video(video_path, max_width, scale=4, progress=gr.Progress(), mode=None):
    """
    Resizes video for FlashVSR preprocessing using FFmpeg.
    Never upsizes - only downsizes if needed.
    mode="4k_safe" fits so (size × scale) stays within UHD 4K 16:9 / 9:16.
    Returns path to resized video (or original if no resize needed).
    """
    if not video_path or not os.path.exists(video_path):
        log("No video provided for resize", message_type="warning")
        return video_path
    
    current_width, current_height = get_video_dimensions(video_path)
    if mode is None and isinstance(max_width, str) and is_4k_safe_preset(max_width):
        mode = "4k_safe"
        max_width = None
    new_width, new_height, will_resize = calculate_resize_dimensions(
        current_width, current_height, max_width, scale=scale, mode=mode
    )
    
    if not will_resize:
        log(f"Video is already {current_width}×{current_height}, no resize needed", message_type="info")
        return video_path
    
    if not is_ffmpeg_available():
        log("FFmpeg not available, cannot resize video", message_type="error")
        return video_path
    
    try:
        log(
            f"Resizing video {current_width}×{current_height} → {new_width}×{new_height} "
            f"(center crop, aspect preserved)...",
            message_type="info",
        )
        progress(0.1, desc="Resizing input video...")
        
        # Generate output path in temp directory
        input_basename = os.path.splitext(os.path.basename(video_path))[0]
        input_basename = clean_video_filename(input_basename)  # Clean filename to prevent length issues
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_basename}_resized_{new_width}x{new_height}_{timestamp}.mp4"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Scale to cover target box (no stretch), then center-crop to aligned size
        progress(0.3, desc="Running FFmpeg resize...")
        vf = (
            f"scale={new_width}:{new_height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={new_width}:{new_height}"
        )
        
        # Build FFmpeg command - use map to handle audio gracefully
        # High-quality intermediate: soft pre-encode was compounding "lost detail" before FlashVSR.
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', vf,
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '14',
            '-pix_fmt', 'yuv420p',
            '-map', '0:v:0',  # Map video stream
            '-map', '0:a:0?',  # Map audio stream if it exists (? makes it optional)
            '-c:a', 'aac',
            '-b:a', '256k',
            output_path
        ]
        
        # Run FFmpeg and capture output
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        progress(1.0, desc="Resize complete!")
        log(f"Video resized successfully: {output_path}", message_type="finish")
        return output_path
        
    except subprocess.CalledProcessError as e:
        log(f"FFmpeg error during resize:", message_type="error")
        log(f"Command: {' '.join(e.cmd)}", message_type="error")
        if e.stderr:
            # Print stderr line by line for better readability
            log("FFmpeg stderr output:", message_type="error")
            for line in e.stderr.split('\n'):
                if line.strip():
                    log(f"  {line}", message_type="error")
        return video_path
    except Exception as e:
        log(f"Error resizing video: {e}", message_type="error")
        import traceback
        log(traceback.format_exc(), message_type="error")
        return video_path

def get_video_duration(video_path):
    """Get video duration in seconds. Returns 0 on error."""
    try:
        if not video_path or not os.path.exists(video_path):
            return 0
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        duration = meta.get('duration', 0)
        reader.close()
        return duration
    except:
        return 0

def get_video_fps(video_path):
    """Get video FPS. Returns 30 as default on error."""
    try:
        if not video_path or not os.path.exists(video_path):
            return 30
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = meta.get('fps', 30)
        reader.close()
        return fps
    except:
        return 30

def get_minimum_duration(video_path):
    """Calculate minimum duration needed for FlashVSR (21 frames minimum)."""
    fps = get_video_fps(video_path)
    min_frames = 21
    min_duration = min_frames / fps
    return min_duration

def format_time_mmss(seconds):
    """Format seconds as MM:SS for display."""
    if seconds == 0:
        return "00:00"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def preview_trim(video_path, start_time, end_time):
    """Generate preview text showing what trim operation will do."""
    if not video_path:
        return '<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">No video loaded</div>'
    
    total_duration = get_video_duration(video_path)
    if total_duration == 0:
        return '<div style="padding: 8px; background: #3d2e0a; border: 1px solid #854d0e; border-radius: 4px; color: #fbbf24; font-size: 0.9em; text-align: center;">⚠️ Could not read video duration</div>'
    
    min_duration = get_minimum_duration(video_path)
    
    # Clamp values
    start_time = max(0, min(start_time, total_duration))
    
    # Handle end_time = 0 as "end of video" before clamping
    if end_time == 0:
        end_time = total_duration
    else:
        end_time = max(start_time, min(end_time, total_duration))
    
    # Validate range
    if end_time <= start_time:
        return '<div style="padding: 8px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 4px; color: #fca5a5; font-size: 0.9em; text-align: center;">❌ End time must be after start time</div>'
    
    trim_duration = end_time - start_time
    
    # Check minimum duration (21 frames required by FlashVSR)
    if trim_duration < min_duration:
        fps = get_video_fps(video_path)
        return f'<div style="padding: 8px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 4px; color: #fca5a5; font-size: 0.9em; text-align: center;">❌ Trimmed video too short! Need at least {min_duration:.2f}s (21 frames @ {fps:.1f} FPS)</div>'
    
    # Simple trim mode
    if start_time == 0 and end_time >= total_duration:
        return f'<div style="padding: 8px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.9em; text-align: center;">Processing full video ({total_duration:.1f}s) ✓</div>'
    else:
        return f'<div style="padding: 8px; background: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac; font-size: 0.9em; text-align: center;">Will trim: {start_time:.1f}s → {end_time:.1f}s ({trim_duration:.1f}s) ✓</div>'

def preview_chunk_processing(video_path, chunk_duration):
    """Generate preview showing how many chunks will be created."""
    if not video_path:
        return '<div style="padding: 6px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.85em; text-align: center;">💡 Enable chunk processing for videos that exceed your available VRAM</div>'
    
    duration = get_video_duration(video_path)
    if duration == 0:
        return '<div style="padding: 6px; background: #3d2e0a; border: 1px solid #854d0e; border-radius: 4px; color: #fbbf24; font-size: 0.85em; text-align: center;">⚠️ Could not read video duration</div>'
    
    min_duration = get_minimum_duration(video_path)
    
    # Check if chunk duration is too short
    if chunk_duration < min_duration:
        fps = get_video_fps(video_path)
        return f'<div style="padding: 6px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 4px; color: #fca5a5; font-size: 0.85em; text-align: center;">❌ Chunk duration too short! Need at least {min_duration:.2f}s (21 frames @ {fps:.1f} FPS)</div>'
    
    # Simple chunk calculation - exact boundaries, no redistribution
    fps = get_video_fps(video_path)
    
    # If video fits in one chunk (duration <= chunk_duration), just use single chunk
    if duration <= chunk_duration:
        return f'''<div style="padding: 8px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.85em; text-align: center;">
            📊 Will process as 1 chunk ({duration:.2f}s, {round(duration * fps)} frames)<br>
            Video: {format_time_mmss(duration)} ({duration:.2f}s)
        </div>'''
    
    num_chunks = math.ceil(duration / chunk_duration)
    last_chunk_duration = duration - (chunk_duration * (num_chunks - 1))
    last_chunk_frames = round(last_chunk_duration * fps)  # Use round() not int() for accuracy
    
    warning_note = ""
    if last_chunk_frames < 21:
        warning_note = f'<br><span style="color: #fbbf24;">⚠️ Last chunk only {last_chunk_frames} frames - adjust slider to avoid failure</span>'
        bg_color = "#3d2e0a"
        border_color = "#854d0e"
        text_color = "#fbbf24"
    else:
        bg_color = "#0c2d48"
        border_color = "#1e4a6e"
        text_color = "#7dd3fc"
    
    # Format chunk sizes for display - use .2f for short durations
    last_dur_str = f"{last_chunk_duration:.2f}s" if last_chunk_duration < 1 else f"{last_chunk_duration:.1f}s"
    chunks_desc = f"{num_chunks - 1}x {chunk_duration:.1f}s + 1x {last_dur_str} ({last_chunk_frames} frames)"
    
    return f'''<div style="padding: 8px; background: {bg_color}; border: 1px solid {border_color}; border-radius: 4px; color: {text_color}; font-size: 0.85em; text-align: center;">
        📊 Will create {chunks_desc}<br>
        Video: {format_time_mmss(duration)} ({duration:.2f}s){warning_note}
    </div>'''


def prepare_image_as_frames(image_path, num_frames=21):
    """Duplicate an image 21 times to create a frame folder for processing."""
    if not image_path or not os.path.exists(image_path):
        log("No image provided", message_type="warning")
        return None
    
    try:
        # Create temp folder for frames
        temp_frames_dir = os.path.join(TEMP_DIR, f"image_frames_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        log(f"Preparing image for processing (duplicating {num_frames}x)...", message_type="info")
        
        # Load and save the image 21 times with sequential naming
        img = Image.open(image_path)
        for i in range(num_frames):
            frame_path = os.path.join(temp_frames_dir, f"{i:05d}.png")
            img.save(frame_path)
        
        log(f"Image frames prepared in: {temp_frames_dir}", message_type="finish")
        return temp_frames_dir
        
    except Exception as e:
        log(f"Error preparing image frames: {e}", message_type="error")
        return None

def save_preprocessed_video(video_path, progress=gr.Progress()):
    """Save the current preprocessed video to outputs/preprocessed folder."""
    if not video_path or not os.path.exists(video_path):
        log("No video to save", message_type="warning")
        return
    
    try:
        # Create preprocessed output directory
        output_dir = get_output_dir()
        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        # Generate output filename with timestamp
        input_basename = os.path.splitext(os.path.basename(video_path))[0]
        input_basename = clean_video_filename(input_basename)  # Clean filename to prevent length issues
        timestamp = time.strftime("%H%M%S")
        output_filename = f"{input_basename}_preprocessed_{timestamp}.mp4"
        output_path = os.path.join(preprocessed_dir, output_filename)
        
        log(f"Saving preprocessed video to: {output_path}", message_type="info")
        progress(0.5, desc="Saving preprocessed video...")
        
        # Copy the video file
        shutil.copy(video_path, output_path)
        
        progress(1.0, desc="Save complete!")
        log(f"Preprocessed video saved successfully: {output_path}", message_type="finish")
        
    except Exception as e:
        log(f"Error saving preprocessed video: {e}", message_type="error")

def trim_video(video_path, start_time, end_time, progress=gr.Progress()):
    """Trim video to specified time range using FFmpeg."""
    if not video_path or not os.path.exists(video_path):
        log("No video provided for trim", message_type="warning")
        return video_path
    
    if not is_ffmpeg_available():
        log("FFmpeg not available, cannot trim video", message_type="error")
        return video_path
    
    total_duration = get_video_duration(video_path)
    start_time = max(0, min(start_time, total_duration))
    
    # Handle end_time = 0 as "end of video" before clamping
    if end_time == 0:
        end_time = total_duration
    else:
        end_time = max(start_time, min(end_time, total_duration))
    
    # Validate that end_time is after start_time
    if end_time <= start_time:
        log(f"Invalid trim range: end time ({end_time:.1f}s) must be after start time ({start_time:.1f}s)", message_type="error")
        return video_path
    
    # If no actual trimming needed, return original
    if start_time == 0 and end_time >= total_duration:
        log("No trimming needed - using full video", message_type="info")
        return video_path
    
    try:
        log(f"Trimming video from {start_time:.1f}s to {end_time:.1f}s...", message_type="info")
        progress(0.1, desc="Trimming video...")
        
        input_basename = os.path.splitext(os.path.basename(video_path))[0]
        input_basename = clean_video_filename(input_basename)  # Clean filename to prevent length issues
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_basename}_trim_{start_time:.0f}-{end_time:.0f}s_{timestamp}.mp4"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        duration = end_time - start_time
        
        # Build FFmpeg command for fast, accurate trimming
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),  # Seek to start
            '-i', video_path,
            '-t', str(duration),  # Duration to extract
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-map', '0:v:0',
            '-map', '0:a:0?',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]
        
        progress(0.3, desc="Running FFmpeg trim...")
        
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        progress(1.0, desc="Trim complete!")
        log(f"Video trimmed successfully: {output_path}", message_type="finish")
        return output_path
        
    except subprocess.CalledProcessError as e:
        log(f"FFmpeg error during trim: {e}", message_type="error")
        if e.stderr:
            log("FFmpeg stderr:", message_type="error")
            for line in e.stderr.split('\n')[-10:]:  # Last 10 lines
                if line.strip():
                    log(f"  {line}", message_type="error")
        return video_path
    except Exception as e:
        log(f"Error trimming video: {e}", message_type="error")
        return video_path

def create_video_chunks(video_path, start_time, end_time, chunk_duration, progress=gr.Progress()):
    """Split video into chunks and return list of chunk paths."""
    if not video_path or not os.path.exists(video_path):
        log("No video provided for chunking", message_type="warning")
        return []
    
    if not is_ffmpeg_available():
        log("FFmpeg not available, cannot create chunks", message_type="error")
        return []
    
    total_duration = get_video_duration(video_path)
    start_time = max(0, min(start_time, total_duration))
    
    # Handle end_time = 0 as "end of video" before clamping
    if end_time == 0:
        end_time = total_duration
    else:
        end_time = max(start_time, min(end_time, total_duration))
    
    # Validate range
    if end_time <= start_time:
        log(f"Invalid chunk range: end time ({end_time:.1f}s) must be after start time ({start_time:.1f}s)", message_type="error")
        return []
    
    # Validate chunk duration
    if chunk_duration <= 0:
        log(f"Invalid chunk duration: must be greater than 0", message_type="error")
        return []
    
    trim_duration = end_time - start_time
    fps = get_video_fps(video_path)
    # FlashVSR requires >= 21 frames; merge any leftover shorter than that into the previous chunk.
    min_chunk_duration = max(21 / max(fps, 1e-6), 0.05)

    # Build chunk time ranges, absorbing short tails into the previous segment.
    ranges = []
    t = start_time
    while t < end_time - 1e-6:
        next_t = min(t + chunk_duration, end_time)
        remaining_after = end_time - next_t
        # If the leftover after this chunk would be too short, take everything now.
        if 0 < remaining_after < min_chunk_duration:
            next_t = end_time
        this_dur = next_t - t
        if this_dur < min_chunk_duration and ranges:
            prev_start, _ = ranges.pop()
            ranges.append((prev_start, end_time))
            break
        ranges.append((t, next_t))
        t = next_t

    num_chunks = len(ranges)
    log(
        f"Creating {num_chunks} chunks (max {chunk_duration}s each; "
        f"min segment {min_chunk_duration:.2f}s for 21 frames @ {fps:.1f} FPS)...",
        message_type="info",
    )
    
    chunk_paths = []
    input_basename = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for i, (chunk_start, chunk_end) in enumerate(ranges):
        chunk_dur = chunk_end - chunk_start
        
        progress((i / max(num_chunks, 1)) * 0.8, desc=f"Creating chunk {i+1}/{num_chunks}...")
        
        output_filename = f"{input_basename}_chunk{i+1:03d}_{timestamp}.mp4"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-ss', str(chunk_start),
                '-i', video_path,
                '-t', str(chunk_dur),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-map', '0:v:0',
                '-map', '0:a:0?',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            chunk_paths.append(output_path)
            log(f"Created chunk {i+1}/{num_chunks}: {chunk_start:.1f}s-{chunk_end:.1f}s", message_type="info")
            
        except subprocess.CalledProcessError as e:
            log(f"Error creating chunk {i+1}: {e}", message_type="error")
            continue
    
    progress(1.0, desc=f"Created {len(chunk_paths)} chunks!")
    log(f"Successfully created {len(chunk_paths)} chunks", message_type="finish")
    return chunk_paths

def combine_video_chunks(chunk_paths, output_name_base, progress=gr.Progress()):
    """Combine processed video chunks into a single video."""
    if not chunk_paths:
        log("No chunks to combine", message_type="warning")
        return None
    
    if not is_ffmpeg_available():
        log("FFmpeg not available, cannot combine chunks", message_type="error")
        return None
    
    log(f"Combining {len(chunk_paths)} chunks...", message_type="info")
    progress(0.1, desc="Preparing to combine chunks...")
    
    try:
        # Create a temporary file list for FFmpeg concat
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        concat_list_path = os.path.join(TEMP_DIR, f"concat_list_{timestamp}.txt")
        
        with open(concat_list_path, 'w') as f:
            for chunk_path in chunk_paths:
                # FFmpeg concat requires absolute paths with proper escaping
                abs_path = os.path.abspath(chunk_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        output_filename = f"{output_name_base}_combined_{timestamp}.mp4"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        progress(0.3, desc="Running FFmpeg concat...")
        
        # Use concat demuxer for fast, lossless concatenation
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',  # Copy streams without re-encoding
            output_path
        ]
        
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Clean up concat list file
        try:
            os.remove(concat_list_path)
        except:
            pass
        
        progress(1.0, desc="Chunks combined!")
        log(f"Successfully combined chunks: {output_path}", message_type="finish")
        return output_path
        
    except subprocess.CalledProcessError as e:
        log(f"FFmpeg error during combine: {e}", message_type="error")
        if e.stderr:
            log("FFmpeg stderr:", message_type="error")
            for line in e.stderr.split('\n')[-10:]:
                if line.strip():
                    log(f"  {line}", message_type="error")
        return None
    except Exception as e:
        log(f"Error combining chunks: {e}", message_type="error")
        return None

def process_video_with_chunks(
    input_path, chunk_duration, mode, model_version, scale, color_fix, tiled_vae, tiled_dit,
    tile_size, tile_overlap, unload_dit, dtype_str, seed, device, fps_override,
    quality, attention_mode, sparse_ratio, kv_ratio, local_range, autosave,
    progress=gr.Progress()
):
    """
    Process video in chunks automatically - creates chunks, processes each, and combines.
    This is a wrapper around the main processing function for chunk mode.
    """
    if not input_path or not os.path.exists(input_path):
        log("No input video provided for chunk processing", message_type="error")
        return None, None, None, gr.update(visible=False)
    
    # Log seed for chunk processing
    log(f"Using seed for chunk processing: {seed}", message_type="info")
    
    # Step 1: Create chunks
    log(f"Starting chunk processing mode with {chunk_duration}s chunks...", message_type="info")
    progress(0.05, desc="Creating video chunks...")
    
    total_duration = get_video_duration(input_path)
    chunk_paths = create_video_chunks(input_path, 0, 0, chunk_duration, progress)
    
    if not chunk_paths:
        log("Failed to create chunks", message_type="error")
        return None, None, None, gr.update(visible=False)
    
    num_chunks = len(chunk_paths)
    log(f"Created {num_chunks} chunks, processing each...", message_type="info")
    
    # Step 2: Process each chunk (model reloaded each time for clean state)
    processed_chunks = []
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    input_basename = clean_video_filename(input_basename)  # Clean filename to prevent length issues
    
    for i, chunk_path in enumerate(chunk_paths):
        chunk_progress_start = 0.1 + (i / num_chunks) * 0.8
        chunk_progress_end = 0.1 + ((i + 1) / num_chunks) * 0.8
        
        log(f"Processing chunk {i+1}/{num_chunks}...", message_type="info")
        progress(chunk_progress_start, desc=f"Processing chunk {i+1}/{num_chunks}...")
        
        try:
            # Create a custom progress wrapper that scales to the chunk's progress range
            class ChunkProgress:
                def __init__(self, parent_progress, start, end):
                    self.parent_progress = parent_progress
                    self.start = start
                    self.end = end
                
                def __call__(self, value, desc=None):
                    # Scale the 0-1 progress to the chunk's range
                    scaled_value = self.start + (value * (self.end - self.start))
                    if desc:
                        self.parent_progress(scaled_value, desc=f"Chunk {i+1}/{num_chunks}: {desc}")
                    else:
                        self.parent_progress(scaled_value, desc=f"Processing chunk {i+1}/{num_chunks}...")
            
            chunk_progress = ChunkProgress(progress, chunk_progress_start, chunk_progress_end)
            
            # Process this chunk using the main processing function
            # Seed is already fixed at the start, so all chunks use the same seed
            # Note: create_comparison=False for chunks (comparison only works on full video)
            output_path, _, _, _ = run_flashvsr_single(
                input_path=chunk_path,
                mode=mode,
                model_version=model_version,
                scale=scale,
                color_fix=color_fix,
                tiled_vae=tiled_vae,
                tiled_dit=tiled_dit,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                unload_dit=unload_dit,
                dtype_str=dtype_str,
                seed=seed,  # Use the fixed seed for all chunks
                device=device,
                fps_override=fps_override,
                quality=quality,
                attention_mode=attention_mode,
                sparse_ratio=sparse_ratio,
                kv_ratio=kv_ratio,
                local_range=local_range,
                autosave=False,
                create_comparison=False,  # No comparison for individual chunks
                progress=chunk_progress
            )
            
            if output_path and os.path.exists(output_path):
                processed_chunks.append(output_path)
                log(f"✅ Chunk {i+1}/{num_chunks} processed successfully", message_type="finish")
            else:
                log(f"❌ Failed to process chunk {i+1}/{num_chunks}", message_type="error")
                
        except Exception as e:
            log(f"Error processing chunk {i+1}/{num_chunks}: {e}", message_type="error")
            # After OOM, aggressively free GPU before the next chunk attempts model load.
            if is_cuda_oom(e):
                release_processing_vram()
                log_vram_status(f"chunk-{i+1}-after-oom")
            continue
        finally:
            release_processing_vram()

    # Clean up unprocessed chunks
    for chunk_path in chunk_paths:
        try:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        except:
            pass
    
    if not processed_chunks:
        log("No chunks were successfully processed", message_type="error")
        return None, None, None, gr.update(visible=False)
    
    if len(processed_chunks) < num_chunks:
        log(f"Warning: Only {len(processed_chunks)}/{num_chunks} chunks processed successfully", message_type="warning")
    
    # Step 3: Combine processed chunks
    progress(0.9, desc="Combining processed chunks...")
    log("Combining all processed chunks into final video...", message_type="info")
    
    _, chunk_input_h = get_video_dimensions(input_path)
    combined_path = combine_video_chunks(
        processed_chunks, upscale_combine_stem(input_basename, scale, chunk_input_h * scale), progress
    )
    
    if not combined_path:
        log("Failed to combine chunks", message_type="error")
        # Return first chunk as fallback
        fallback_path = processed_chunks[0] if processed_chunks else None
        fallback_analysis = analyze_output_video(fallback_path) if fallback_path else gr.update(visible=False)
        return fallback_path, fallback_path, None, fallback_analysis
    
    # Clean up individual processed chunks
    for chunk_path in processed_chunks:
        try:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        except:
            pass
    
    # Step 4: Handle audio and final output
    chunk_w = 0
    try:
        chunk_w, _ = get_video_dimensions(input_path)
    except Exception:
        chunk_w = 0
    output_filename = upscale_video_filename(
        input_basename,
        scale,
        chunked=True,
        output_height=int(chunk_input_h or 0) * scale,
        output_width=int(chunk_w or 0) * scale,
    )
    temp_output_path = os.path.join(TEMP_DIR, output_filename)
    
    # Merge audio from original video
    if is_video(input_path):
        progress(0.95, desc="Merging audio...")
        merge_video_with_audio(combined_path, input_path, temp_output_path)
    else:
        shutil.move(combined_path, temp_output_path)
    
    # Autosave if enabled
    output_dir = get_output_dir()
    if autosave:
        final_save_path = os.path.join(output_dir, output_filename)
        shutil.copy(temp_output_path, final_save_path)
        log(f"Chunk processing complete! Auto-saved to: {final_save_path}", message_type="finish")
    else:
        log(f"Chunk processing complete! Use 'Save Output' to save to outputs folder.", message_type="finish")
    
    progress(1.0, desc="Done!")
    
    # Generate output analysis
    output_analysis = analyze_output_video(temp_output_path)
    
    return (
        temp_output_path,
        temp_output_path,
        (input_path, temp_output_path),
        output_analysis
    )

def open_folder(folder_path):
    try:
        if sys.platform == "win32":
            os.startfile(folder_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder_path])
        else:
            subprocess.Popen(["xdg-open", folder_path])
        return f'<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ Opened folder: {folder_path}</div>'
    except Exception as e:
        return f'<div style="padding: 1px; background-color: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 4px; color: #fca5a5;">❌ Error opening folder: {e}</div>'

def save_file(file_path):
    if file_path and os.path.exists(file_path):
        log(f"File saved to: {file_path}", message_type="finish")
    else:
        log(f"File not found or unable to save.", message_type="error")

def handle_start_pipeline(
    active_tab_index, single_video_path, batch_video_paths, batch_folder_path, selected_ops,
    # Frame Adjust params
    fps_mode, speed_factor, frames_use_streaming, frames_quality,
    # Video Loop params
    loop_type, num_loops, loop_quality,
    # Export params
    export_format, quality, max_width, output_name, two_pass,
    progress=gr.Progress()
):
    # Determine input paths based on the active tab
    if active_tab_index == 1:
        # Batch mode - check folder path first, then files
        input_paths = []
        if batch_folder_path and os.path.isdir(batch_folder_path):
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']
            input_paths = [str(f) for f in Path(batch_folder_path).iterdir() 
                          if f.is_file() and f.suffix.lower() in video_extensions]
            input_paths.sort()  # Sort for consistent ordering
        elif batch_video_paths:
            input_paths = [file.name for file in batch_video_paths]
        
        if not input_paths:
            return None, "⚠️ Batch Input tab is active, but no files were provided. Please upload files or specify a valid folder path.", '<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ No input files</div>'
    elif active_tab_index == 0 and single_video_path:
        input_paths = [single_video_path]
    else:
        return None, "⚠️ No input video found in the active tab. Please upload a video.", '<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ No input video</div>'

    if not selected_ops:
        return None, "⚠️ No operations selected. Please check at least one box in 'Pipeline Steps'.", '<div style="padding: 12px; background: #3d2e0a; border: 1px solid #854d0e; border-radius: 6px; color: #fbbf24;">⚠️ No operations selected</div>'

    # Pack parameters for the processor
    params = {
        "frame_adjust": {
            "fps_mode": fps_mode, "speed_factor": speed_factor, "use_streaming": frames_use_streaming, "output_quality": frames_quality
        },
        "loop": {
            "loop_type": loop_type, "num_loops": num_loops, "output_quality": loop_quality
        },
        "export": {
            "export_format": export_format, "quality": quality, "max_width": max_width, "output_name": output_name, "two_pass": two_pass
        }
    }
    
    if len(input_paths) > 1:
        # Batch processing
        final_video, message = toolbox_processor.process_batch(input_paths, selected_ops, params, progress)
        output_analysis = '<div style="padding: 12px; background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 6px; color: #0c5460; text-align: center;">Batch processing complete. Analysis not available for batch mode.</div>'
    else:
        # Single video processing
        temp_video, message = toolbox_processor.process_pipeline(input_paths[0], selected_ops, params, progress)
        final_video = None
        if temp_video:
            if toolbox_processor.autosave_enabled:
                temp_path = Path(temp_video)
                final_path = toolbox_processor.output_dir / temp_path.name
                final_video = toolbox_processor._copy_to_permanent_storage(temp_video, final_path)
                message += f"\n✅ Autosaved result to: {final_path}"
            else:
                final_video = temp_video # Leave in temp folder for manual save
                message += "\nℹ️ Autosave is off. Result is in a temporary folder. Use 'Manual Save' to keep it."
            
            # Analyze output video
            output_analysis = toolbox_processor.analyze_video_html(final_video)
        else:
            output_analysis = '<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ Pipeline failed</div>'

    return final_video, message, output_analysis
    
# Idle state HTML options for save_status display (compact versions)
IDLE_STATES = [
    # Option 1: Compact Gradient
    '''<div style="padding: 1px; text-align: center;">
        <span style="
            font-size: 1.1em;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">FlashVSR+</span>
    </div>'''
]

HEAD_HTML = r"""
<script>
(function () {
  function ensureFloat() {
    var el = document.getElementById('fvsr-tip-float');
    if (!el) {
      el = document.createElement('div');
      el.id = 'fvsr-tip-float';
      document.body.appendChild(el);
    }
    return el;
  }
  function bindTips() {
    var float = ensureFloat();
    var nodes = document.querySelectorAll('span[data-testid="block-info"], .block .info, span.info, p.info');
    nodes.forEach(function (node) {
      if (node.dataset.fvsrTipBound) return;
      var tipText = (node.textContent || '').trim();
      if (!tipText) return;
      node.dataset.fvsrTipBound = '1';
      node.setAttribute('title', tipText);
      var show = function (e) {
        float.textContent = tipText;
        float.classList.add('visible');
        var x = e.clientX + 14;
        var y = e.clientY + 14;
        if (x + 420 > window.innerWidth) x = window.innerWidth - 432;
        if (y + 140 > window.innerHeight) y = e.clientY - 140;
        float.style.left = x + 'px';
        float.style.top = y + 'px';
      };
      var hide = function () { float.classList.remove('visible'); };
      node.addEventListener('mouseenter', show);
      node.addEventListener('mousemove', show);
      node.addEventListener('mouseleave', hide);
    });
  }
  var obs = new MutationObserver(function () { bindTips(); });
  function start() {
    obs.observe(document.body, { childList: true, subtree: true });
    bindTips();
    setInterval(bindTips, 2000);
  }
  if (document.body) start();
  else document.addEventListener('DOMContentLoaded', start);
})();
</script>
"""

css = """
.video-window {
    min-height: 300px !important;
    height: auto !important;
}

.video-window video, .image-window img {
    max-height: 60vh !important;
    object-fit: contain;
    width: 100%;
}
.video-window .source-selection,
.image-window .source-selection {
    display: none !important;
}

/* Monitor / queue text boxes — always dark to match Interstellar / rest of app */
.monitor-box {
    min-width: 0 !important;
}

.monitor-box textarea {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    font-size: 0.85em !important;
    line-height: 1.6 !important;
    padding: 12px !important;
    border-radius: 8px !important;
    border: 1px solid #2d3748 !important;
    background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%) !important;
    color: #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35) !important;
    resize: none !important;
    font-weight: 500 !important;
    caret-color: #7dd3fc !important;
}

.gpu-monitor textarea {
    border-left: 3px solid #667eea !important;
    background: linear-gradient(135deg, #12182a 0%, #1a202c 100%) !important;
    color: #e2e8f0 !important;
}

.cpu-monitor textarea {
    border-left: 3px solid #f5576c !important;
    background: linear-gradient(135deg, #1a1218 0%, #1a202c 100%) !important;
    color: #e2e8f0 !important;
}

.monitor-box textarea:focus {
    outline: none !important;
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25) !important;
}

/* Queue / status HTML panels (override any leftover light inline styles) */
.prose, .gradio-html, .html-container {
    color: #e2e8f0;
}
/* Gradio textboxes used for paths / notes — dark fill when theme leaves them light */
textarea, input[type="text"], input[type="number"], input[type="search"] {
    background-color: #0f1419 !important;
    color: #e2e8f0 !important;
    border-color: #2d3748 !important;
}
textarea::placeholder, input::placeholder {
    color: #64748b !important;
}

/* Machine profile banner */
.fvsr-machine-banner {
    margin: 0 0 12px 0;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #3b4a6b;
    background: linear-gradient(135deg, #1a2332 0%, #243047 100%);
    color: #e8eefc;
    font-size: 0.92em;
    line-height: 1.45;
}
.fvsr-machine-banner strong { color: #9ec1ff; }

span[data-testid="block-info"],
.block .info,
.form .info,
p.info,
span.info {
    cursor: help !important;
}
span[data-testid="block-info"] {
    display: inline-block !important;
    max-width: 100%;
    opacity: 0.9;
    font-size: 0.82em !important;
    line-height: 1.35 !important;
    border-bottom: 1px dashed #7aa2ff66;
}
span[data-testid="block-info"]:hover {
    opacity: 1;
    color: #9ec1ff !important;
}
#fvsr-tip-float {
    display: none;
    position: fixed;
    z-index: 99999;
    max-width: min(420px, 90vw);
    padding: 10px 12px;
    border-radius: 8px;
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #475569;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    font-size: 12.5px;
    line-height: 1.45;
    white-space: pre-wrap;
    pointer-events: none;
}
#fvsr-tip-float.visible { display: block; }
"""
    
def create_ui():
    global toolbox_processor

    config = load_config()
    ui = get_ui_defaults(config)
    ensure_workflow_dirs(ui)
    
    # Initialize toolbox processor with shared config
    if toolbox_processor is None:
        toolbox_processor = ToolboxProcessor(config.get("tb_autosave", True))
        toolbox_processor.output_dir = Path(get_toolbox_output_dir())
    
    # Available Gradio themes
    # Built-in Gradio themes
    BUILTIN_THEMES = {
        "Default": gr.themes.Default(),
        "Soft": gr.themes.Soft(),
        "Monochrome": gr.themes.Monochrome(),
        "Glass": gr.themes.Glass(),
        "Base": gr.themes.Base(),
        "Ocean": gr.themes.Ocean(),
        "Origin": gr.themes.Origin(),
        "Citrus": gr.themes.Citrus(),
    }
    
    # Community themes from Hugging Face Spaces
    COMMUNITY_THEMES = {
        "Miku": "NoCrypt/miku",
        "Interstellar": "Nymbo/Interstellar",
        "xkcd": "gstaff/xkcd",
    }
    
    # Load saved theme preference
    current_theme = config.get("theme", "Default")
    custom_theme_string = config.get("custom_theme", "")
    
    # Determine which theme to use
    selected_theme = None
    if current_theme == "Custom" and custom_theme_string:
        # Try to load custom theme
        try:
            selected_theme = gr.themes.Base.from_hub(custom_theme_string)
        except Exception as e:
            log(f"Failed to load custom theme '{custom_theme_string}': {e}", message_type="warning")
            selected_theme = gr.themes.Default()
    elif current_theme in BUILTIN_THEMES:
        selected_theme = BUILTIN_THEMES[current_theme]
    elif current_theme in COMMUNITY_THEMES:
        try:
            selected_theme = gr.themes.Base.from_hub(COMMUNITY_THEMES[current_theme])
        except Exception as e:
            log(f"Failed to load community theme '{current_theme}': {e}", message_type="warning")
            selected_theme = gr.themes.Default()
    else:
        selected_theme = gr.themes.Default()
    
    # Combine all theme names for dropdown
    ALL_THEME_NAMES = list(BUILTIN_THEMES.keys()) + list(COMMUNITY_THEMES.keys()) + ["Custom"]
    
    with gr.Blocks(css=css, theme=selected_theme, head=HEAD_HTML) as demo:
        output_file_path = gr.State(None)
        completion_status = gr.State(None)

        with gr.Tabs(elem_id="main_tabs") as main_tabs:
            with gr.TabItem("FlashVSR", id=0):
                gr.HTML(
                    value=(
                        f'<div class="fvsr-machine-banner">'
                        f'<strong>Profile loaded:</strong> {MACHINE["profile_name"]}<br>'
                        f'Defaults (clarity): tiny · v1.1 · 4× · tiled DiT 256/48 · chunks 10s · quality 9 · sparse 1.2 · bf16 · sage · unload DiT · resize ≤1024px · models on O:\\MODELS<br>'
                        f'<em>Pre-resize only shrinks wide sources — 1024 keeps more detail than 768. Hover options for guidance. Restart after hard OOM.</em>'
                        f'</div>'
                    )
                )
                # Live map of where each step actually saves (not the old app\\outputs archetype)
                workflow_map = gr.HTML(value=workflow_paths_html())
                with gr.Row():
                    # --- Left-side Column ---                       
                    with gr.Column(scale=1):
                        with gr.Tabs() as flashvsr_input_tabs:
                            with gr.TabItem("Single Video"):
                                input_video = gr.Video(label="Upload Video File", elem_classes="video-window")
                                run_button = gr.Button("Start Processing", variant="primary", size="sm")
                            with gr.TabItem("Batch Video"):
                                flashvsr_batch_input_files = gr.File(
                                    label="Upload videos to add to the work queue",
                                    file_count="multiple",
                                    type="filepath",
                                    file_types=["video"],
                                    height="200px",                            
                                )
                                gr.Markdown(
                                    "**Watch folder** (auto-scanned on Start / Resume) or paste another path:"
                                )
                                batch_folder_path = gr.Textbox(
                                    value=ui.get(
                                        "batch_watch_folder",
                                        r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS",
                                    ),
                                    placeholder=r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS",
                                    label="Watch / folder path",
                                    show_label=True,
                                    info=(
                                        "Step 1 — intake. Start / Resume scans here. "
                                        "Originals → Step 2 (Pre Scaled). Upscaled videos → Step 3 (Ready for Toolbox)."
                                    ),
                                )
                                
                                # Batch resize preset
                                gr.Markdown("---")
                                gr.Markdown('<span style="font-size: 0.9em; color: #666;">📐 **Batch Resize Preset** - Automatically resize videos wider than selected width</span>')
                                batch_resize_preset = gr.Dropdown(
                                    choices=[
                                        "4K-safe (auto)",
                                        "No Resize",
                                        "512px",
                                        "768px",
                                        "1024px",
                                        "1280px",
                                        "1920px",
                                    ],
                                    value=ui["batch_resize_preset"]
                                    if ui.get("batch_resize_preset")
                                    in (
                                        "4K-safe (auto)",
                                        "No Resize",
                                        "512px",
                                        "768px",
                                        "1024px",
                                        "1280px",
                                        "1920px",
                                    )
                                    else "4K-safe (auto)",
                                    label="Pre-downscale (4K-safe)",
                                    info=TIPS["batch_resize"],
                                    interactive=True
                                )

                                gr.Markdown(
                                    '<span style="font-size: 0.9em; color: #555;">'
                                    "<b>Work queue</b> — add files anytime, start when idle, "
                                    "stop after the current video finishes, resume later. "
                                    "List stays until you clear it.</span>"
                                )
                                flashvsr_queue_status = gr.HTML(
                                    value=get_flashvsr_work_queue().status_html()
                                )
                                with gr.Row():
                                    batch_add_queue_btn = gr.Button("➕ Add to Queue", size="sm")
                                    batch_run_button = gr.Button("▶️ Start / Resume Queue", variant="primary", size="sm")
                                    batch_stop_button = gr.Button("⏹ Stop After Current", variant="stop", size="sm")
                                with gr.Row():
                                    batch_requeue_failed_btn = gr.Button("↺ Re-queue Failed", size="sm")
                                    batch_clear_done_btn = gr.Button("Clear Done", size="sm")
                                    batch_clear_all_btn = gr.Button("Clear Entire Queue", size="sm", variant="stop")

                            with gr.TabItem("Group Therapy"):
                                gr.Markdown(
                                    "**Group Therapy** — take **N** files from the original folder, "
                                    "run that group **start to finish** (upscale → RIFE → RIFE → export), "
                                    "then the next N.\n\n"
                                    "After each file finishes: **keep only the original + the end file**. "
                                    "Everything else (resized, `_Upscaled`, RIFE temps) is deleted.\n\n"
                                    "Each pair is tagged **`_PID_xxxxxxxx`** at the end of the filename "
                                    "(and Title metadata — not Media Center tags). "
                                    "Before and After stay **flat folders** so you can compare side by side."
                                )
                                gt_watch_folder = gr.Textbox(
                                    value=ui.get("batch_watch_folder", r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS"),
                                    label="Original folder (intake)",
                                    info="Files are pulled from here in newest-first groups.",
                                )
                                gt_group_size = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=int(ui.get("gt_group_size") or 10),
                                    label="Group size",
                                    info=TIPS["gt_group_size"],
                                )
                                with gr.Row():
                                    gt_before_dir = gr.Textbox(
                                        value=ui.get("gt_before_dir") or ui.get(
                                            "batch_source_archive_dir",
                                            r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos",
                                        ),
                                        label="Before (originals, flat)",
                                        info=TIPS["gt_before_dir"],
                                    )
                                    gt_after_dir = gr.Textbox(
                                        value=ui.get("gt_after_dir") or ui.get(
                                            "toolbox_output_dir",
                                            r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV",
                                        ),
                                        label="After (finals, flat)",
                                        info=TIPS["gt_after_dir"],
                                    )
                                gr.Markdown("Stages for each group (in order):")
                                with gr.Row():
                                    gt_do_upscale = gr.Checkbox(label="1 · Upscale", value=bool(ui.get("gt_do_upscale", True)))
                                    gt_do_rife1 = gr.Checkbox(label="2 · RIFE 2×", value=bool(ui.get("gt_do_rife1", True)))
                                    gt_do_rife2 = gr.Checkbox(label="3 · RIFE 2× again", value=bool(ui.get("gt_do_rife2", True)))
                                    gt_do_export = gr.Checkbox(label="4 · Export / interpolation", value=bool(ui.get("gt_do_export", True)))
                                gt_queue_status = gr.HTML(value=get_group_therapy_queue().status_html())
                                with gr.Row():
                                    gt_add_btn = gr.Button("➕ Scan original folder", size="sm")
                                    gt_run_btn = gr.Button("▶️ Start / Resume Group Therapy", variant="primary", size="sm")
                                    gt_stop_btn = gr.Button("⏹ Stop After Current", variant="stop", size="sm")
                                with gr.Row():
                                    gt_requeue_btn = gr.Button("↺ Re-queue Failed", size="sm")
                                    gt_clear_done_btn = gr.Button("Clear Done", size="sm")
                                    gt_clear_all_btn = gr.Button("Clear Entire Queue", size="sm", variant="stop")
                        
                        # Video Pre-Processing Accordion
                        with gr.Accordion("📊 Video Pre-Processing", open=False):
                            video_analysis_html = gr.HTML(
                                value='<div style="padding: 12px; background: #0f1419; border: 1px solid #2d3748; border-radius: 6px; color: #94a3b8; text-align: center;">Upload video to see analysis</div>'
                            )
                            
                            gr.Markdown("---")
                            
                            # Trim controls in sub-accordion
                            with gr.Accordion("✂️ Trim Video", open=False):
                                gr.Markdown('<span style="font-size: 0.9em; color: #666;">Extract a specific time range from your video</span>')
                                
                                with gr.Row():
                                    trim_start_slider = gr.Slider(
                                        minimum=0,
                                        maximum=60,
                                        step=0.5,
                                        value=0,
                                        label="Start Time (seconds)",
                                        info=TIPS["trim_start"]
                                    )
                                    
                                    trim_end_slider = gr.Slider(
                                        minimum=0,
                                        maximum=60,
                                        step=0.5,
                                        value=0,
                                        label="End Time (seconds)",
                                        info=TIPS["trim_end"]
                                    )
                                trim_preview_html = gr.HTML(
                                    value='<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">Upload video to see trim preview</div>'
                                )
                                
                                trim_button = gr.Button("✂️ Apply Trim", size="sm", variant="primary")
                            
                            # Resize controls in sub-accordion
                            with gr.Accordion("📐 Resize Video", open=False):
                                gr.Markdown('<span style="font-size: 0.9em; color: #666;">Reduce resolution to save VRAM and processing time</span>')
                                
                                resize_max_width_slider = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    value=512,
                                    label="Target Width (pixels)",
                                    info=TIPS["resize_width"],
                                    interactive=True
                                )
                                
                                resize_preview_html = gr.HTML(
                                    value='<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">Upload and analyze video to enable resize</div>'
                                )
                                
                                resize_button = gr.Button("📐 Apply Resize", size="sm", variant="primary")
                            
                            # Save preprocessed video button
                            gr.Markdown(r'<span style="font-size: 0.9em; color: #666;">Saves the processed video to outputs\preprocessed</span>')
                            save_preprocessed_btn = gr.Button("💾 Save Input Video", size="sm", variant="primary")
                            
                            # Hidden state to store current video dimensions and duration
                            current_video_width = gr.State(0)
                            current_video_height = gr.State(0)
                            current_video_duration = gr.State(0)

                                
                        with gr.Group():
                            with gr.Row():
                                mode_radio = gr.Radio(choices=["tiny", "full"], value=ui["mode"], label="Pipeline Mode", info=TIPS["mode"])
                                model_version_radio = gr.Radio(
                                    choices=["v1.0", "v1.1"], 
                                    value=ui["model_version"], 
                                    label="Model Version", 
                                    info=TIPS["model_version"]
                                )
                            with gr.Row():
                                seed_number = gr.Number(value=0, label="Seed", precision=0, info=TIPS["seed"])
                                randomize_seed = gr.Checkbox(label="Randomize Seed", value=ui["randomize_seed"], info=TIPS["randomize_seed"])
                        with gr.Group():
                            with gr.Row():
                                scale_slider = gr.Slider(minimum=2, maximum=4, step=1, value=ui["scale"], label="Upscale Factor", info=TIPS["scale"])
                                tiled_dit_checkbox = gr.Checkbox(label="Enable Tiled DiT", info=TIPS["tiled_dit"], value=ui["tiled_dit"])
                            with gr.Row(visible=ui["tiled_dit"]) as tiled_dit_options:
                                tile_size_slider = gr.Slider(
                                    minimum=64, maximum=512, step=16, value=ui["tile_size"], 
                                    label="Tile Size", 
                                    info=TIPS["tile_size"]
                                )
                                tile_overlap_slider = gr.Slider(
                                    minimum=8, maximum=128, step=8, value=ui["tile_overlap"], 
                                    label="Tile Overlap", 
                                    info=TIPS["tile_overlap"]
                                )
                            # Chunk processing mode
                            with gr.Row():
                                enable_chunk_processing = gr.Checkbox(
                                    label="Process as Chunks [Experimental] ",
                                    value=ui["enable_chunks"],
                                    info=TIPS["enable_chunks"]
                                )
                            with gr.Row(visible=ui["enable_chunks"]) as chunk_settings_row:
                                chunk_duration_slider = gr.Slider(
                                    minimum=1,
                                    maximum=30,
                                    step=0.5,
                                    value=ui["chunk_duration"],
                                    label="Max Chunk Duration (seconds)",
                                    info=TIPS["chunk_duration"]
                                )
                            chunk_preview_display = gr.HTML(
                                value='<div style="padding: 6px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.85em; text-align: center;">💡 Enable chunk processing for videos that exceed your available VRAM</div>',
                                visible=ui["enable_chunks"]
                            )
                                    
                    # --- Right-side Column ---      
                    with gr.Column(scale=1):
                        with gr.Tabs() as flashvsr_output_tab:
                            with gr.TabItem("Processed Video"):                        
                                video_output = gr.Video(label="Output Result", interactive=False, elem_classes="video-window")
                        
                        with gr.Group():
                            with gr.Row():                            
                                save_button = gr.Button("Save Manually 💾", size="sm", variant="primary")
                                send_to_toolbox_btn = gr.Button("Send to Toolbox 🛠️", size="sm")                            
                            with gr.Row():
                                autosave_checkbox = gr.Checkbox(label="Autosave Output", value=config.get("autosave", True), info=TIPS["autosave"])
                                create_comparison_checkbox = gr.Checkbox(label="Create Comparison Video", value=False, info=TIPS["comparison"])
                                clear_on_start_checkbox = gr.Checkbox(label="Clear Temp on Start", value=config.get("clear_temp_on_start", False), info=TIPS["clear_on_start"])
                            with gr.Row():                                
                                open_folder_button = gr.Button("Open Output Folder", size="sm", variant="huggingface")
                                clear_temp_button = gr.Button("⚠️ Clear Temp Files", size="sm", variant="stop")
                        with gr.Row():
                            save_status = gr.HTML(
                                value=random.choice(IDLE_STATES),
                                padding=False
                            )                         
                        with gr.Row():
                            with gr.Column(scale=1, min_width=200):
                                gpu_monitor = gr.Textbox(
                                    lines=4,
                                    container=False,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="monitor-box gpu-monitor"
                                )
                            with gr.Column(scale=1, min_width=200):
                                cpu_monitor = gr.Textbox(
                                    lines=4,
                                    container=False,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="monitor-box cpu-monitor"
                                )
                        
                        # Output Analysis Display
                        video_output_analysis_html = gr.HTML(visible=False)
                                
                # --- Advanced Options ---  
                with gr.Row():
                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                sparse_ratio_slider = gr.Slider(
                                    minimum=0.5, maximum=5.0, step=0.1, value=ui["sparse_ratio"], 
                                    label="Sparse Ratio", 
                                    info=TIPS["sparse_ratio"]
                                )
                                local_range_slider = gr.Slider(
                                    minimum=3, maximum=15, step=2, value=ui["local_range"], 
                                    label="Local Range", 
                                    info=TIPS["local_range"]
                                )
                                quality_slider = gr.Slider(
                                    minimum=1, maximum=10, step=1, value=ui["quality"], 
                                    label="Output Video Quality", 
                                    info=TIPS["quality"]
                                )
                            with gr.Column(scale=1):
                                kv_ratio_slider = gr.Slider(
                                    minimum=1, maximum=8, step=1, value=ui["kv_ratio"], 
                                    label="KV Cache Ratio", 
                                    info=TIPS["kv_ratio"]
                                )
                                fps_number = gr.Number(
                                    value=ui["fps_override"], 
                                    label="Output FPS", 
                                    precision=0, 
                                    info=TIPS["fps_override"]
                                )
                                device_textbox = gr.Textbox(
                                    value=ui["device"], 
                                    label="Device", 
                                    info=TIPS["device"]
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                attention_mode_radio = gr.Radio(
                                    choices=["sage", "block"], 
                                    value=ui["attention_mode"], 
                                    label="Attention Mode", 
                                    info=TIPS["attention_mode"]
                                )
                                dtype_radio = gr.Radio(
                                    choices=["fp16", "bf16"], 
                                    value=ui["dtype"], 
                                    label="Data Type", 
                                    info=TIPS["dtype"]
                                )
                            with gr.Column(scale=1):
                                color_fix_checkbox = gr.Checkbox(
                                    label="Enable Color Fix", 
                                    value=ui["color_fix"], 
                                    info=TIPS["color_fix"]
                                )
                                tiled_vae_checkbox = gr.Checkbox(
                                    label="Enable Tiled VAE", 
                                    value=ui["tiled_vae"], 
                                    info=TIPS["tiled_vae"]
                                )
                                unload_dit_checkbox = gr.Checkbox(
                                    label="Unload DiT Before Decoding", 
                                    value=ui["unload_dit"], 
                                    info=TIPS["unload_dit"]
                                )

                # --- Main Tab's VideoSlider output ---  
                with gr.Row():
                    video_slider_output = VideoSlider(
                        label="Video Comparison",
                        interactive=False,
                        video_mode="preview",
                        show_download_button=False,
                        autoplay=False, 
                        loop=True,
                        height=800,
                        width=1200
                    )  
            
            # --- IMAGE UPSCALING TAB ---
            with gr.TabItem("🖼️ Image Upscaling", id=1):
                with gr.Row():
                    # --- Left Column: Input & Settings ---
                    with gr.Column(scale=1):
                        with gr.Tabs() as img_input_tabs:
                            with gr.TabItem("Single Image"):                      
                                img_input = gr.Image(label="Upload Image File", type="filepath", elem_classes="image-window")
                                img_run_button = gr.Button("Start Processing", variant="primary", size="sm")
                            with gr.TabItem("Batch Image"):
                                img_batch_input_files = gr.File(
                                    label="Upload images to add to the work queue",
                                    file_count="multiple",
                                    type="filepath",
                                    file_types=["image"],
                                    height="200px",
                                )
                                gr.Markdown("**Watch folder** (same NEW DOWNLOADS as video — images auto-picked):")
                                img_batch_folder_path = gr.Textbox(
                                    value=ui.get("batch_watch_folder", r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS"),
                                    placeholder=r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS",
                                    label="Watch / folder path",
                                    show_label=True,
                                    info="Start/Resume scans this folder for images. Upscales → Ready for CIV\\images; originals → Pre Scaled.",
                                )
                                gr.Markdown("---")
                                gr.Markdown('<span style="font-size: 0.9em; color: #666;">📐 **Batch Resize Preset**</span>')
                                img_batch_resize_preset = gr.Dropdown(
                                    choices=[
                                        "4K-safe (auto)",
                                        "No Resize",
                                        "512px",
                                        "768px",
                                        "1024px",
                                        "1280px",
                                        "1920px",
                                    ],
                                    value=ui["batch_resize_preset"]
                                    if ui.get("batch_resize_preset")
                                    in (
                                        "4K-safe (auto)",
                                        "No Resize",
                                        "512px",
                                        "768px",
                                        "1024px",
                                        "1280px",
                                        "1920px",
                                    )
                                    else "4K-safe (auto)",
                                    label="Pre-downscale (4K-safe)",
                                    info=TIPS["batch_resize"],
                                    interactive=True
                                )
                                img_queue_status = gr.HTML(value=get_flashvsr_image_queue().status_html())
                                with gr.Row():
                                    img_add_queue_btn = gr.Button("➕ Add to Queue", size="sm")
                                    img_batch_run_button = gr.Button("▶️ Start / Resume Image Queue", variant="primary", size="sm")
                                    img_stop_queue_btn = gr.Button("⏹ Stop After Current", variant="stop", size="sm")
                                with gr.Row():
                                    img_requeue_failed_btn = gr.Button("↺ Re-queue Failed", size="sm")
                                    img_clear_done_btn = gr.Button("Clear Done", size="sm")
                                    img_clear_all_btn = gr.Button("Clear Entire Queue", size="sm", variant="stop")
                        
                        # Image Pre-Processing Accordion
                        with gr.Accordion("📊 Image Pre-Processing", open=False):
                            img_analysis_html = gr.HTML(
                                value='<div style="padding: 12px; background: #0f1419; border: 1px solid #2d3748; border-radius: 6px; color: #94a3b8; text-align: center;">Upload image to see analysis</div>'
                            )
                            
                            gr.Markdown("---")
                            
                            # Resize controls in sub-accordion
                            with gr.Accordion("📐 Resize Image", open=False):
                                gr.Markdown('<span style="font-size: 0.9em; color: #666;">Reduce resolution to save VRAM and processing time</span>')
                                
                                img_resize_max_width_slider = gr.Slider(
                                    minimum=256,
                                    maximum=2048,
                                    step=64,
                                    value=512,
                                    label="Target Width (pixels)",
                                    info=TIPS["resize_width"],
                                    interactive=True
                                )
                                
                                img_resize_preview_html = gr.HTML(
                                    value='<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">Upload and analyze image to enable resize</div>'
                                )
                                
                                img_resize_button = gr.Button("📐 Apply Resize", size="sm", variant="primary")
                            
                            # Hidden state to store current image dimensions
                            img_current_width = gr.State(0)
                            img_current_height = gr.State(0)
                        
                        # Main Settings
                        with gr.Group():
                            with gr.Row():
                                img_mode = gr.Radio(choices=["tiny", "full"], value=ui["mode"], label="Pipeline Mode", info=TIPS["mode"])
                                img_model_version = gr.Radio(
                                    choices=["v1.0", "v1.1"], 
                                    value=ui["model_version"], 
                                    label="Model Version", 
                                    info=TIPS["model_version"]
                                )
                            with gr.Row():
                                img_seed = gr.Number(value=0, label="Seed", precision=0, info=TIPS["seed"])
                                img_randomize_seed = gr.Checkbox(label="Randomize Seed", value=ui["randomize_seed"], info=TIPS["randomize_seed"])
                        
                        with gr.Group():
                            with gr.Row():
                                img_scale = gr.Slider(minimum=2, maximum=4, step=1, value=ui["scale"], label="Upscale Factor", info=TIPS["img_scale"])
                                img_tiled_dit = gr.Checkbox(label="Enable Tiled DiT", info=TIPS["tiled_dit"], value=ui["tiled_dit"])
                            with gr.Row(visible=ui["tiled_dit"]) as img_tiled_dit_options:
                                img_tile_size = gr.Slider(
                                    minimum=64, maximum=512, step=16, value=ui["tile_size"], 
                                    label="Tile Size", 
                                    info=TIPS["tile_size"]
                                )
                                img_tile_overlap = gr.Slider(
                                    minimum=8, maximum=128, step=8, value=ui["tile_overlap"], 
                                    label="Tile Overlap", 
                                    info=TIPS["tile_overlap"]
                                )
                    
                    # --- Right Column: Output ---
                    with gr.Column(scale=1):
                        with gr.Tabs() as img_output_tabs:
                            with gr.TabItem("Processed Image"):
                                img_output = gr.Image(label="Output Result", interactive=False, elem_classes="image-window")
                            with gr.TabItem("Batch Status"):
                                img_batch_status = gr.Textbox(
                                    label="Batch Processing Status",
                                    lines=15,
                                    max_lines=15,
                                    interactive=False,
                                    show_copy_button=True,
                                    value="Upload images and click 'Start Batch Processing' to begin."
                                )
                        
                        with gr.Group():
                            with gr.Row():
                                img_save_button = gr.Button("Save Manually 💾", size="sm", variant="primary")
                            with gr.Row():
                                img_autosave = gr.Checkbox(label="Autosave Output", value=config.get("autosave", True), info=TIPS["autosave"])
                                img_create_comparison = gr.Checkbox(label="Create Comparison Image", value=False, info=TIPS["comparison"])
                                img_clear_on_start = gr.Checkbox(label="Clear Temp on Start", value=config.get("clear_temp_on_start", False), visible=False, info=TIPS["clear_on_start"])
                            with gr.Row():
                                img_open_folder_button = gr.Button("Open Output Folder", size="sm", variant="huggingface")
                                img_clear_temp_button = gr.Button("⚠️ Clear Temp Files", size="sm", variant="stop")
                        
                        with gr.Row():
                            img_save_status = gr.HTML(
                                value=random.choice(IDLE_STATES),
                                padding=False
                            )
                        
                        with gr.Row():
                            with gr.Column(scale=1, min_width=200):
                                img_gpu_monitor = gr.Textbox(
                                    lines=4,
                                    container=False,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="monitor-box gpu-monitor"
                                )
                            with gr.Column(scale=1, min_width=200):
                                img_cpu_monitor = gr.Textbox(
                                    lines=4,
                                    container=False,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="monitor-box cpu-monitor"
                                )
                        
                        # Output Analysis Display
                        img_output_analysis_html = gr.HTML(visible=False)
                
                # --- Advanced Options ---
                with gr.Row():
                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                img_sparse_ratio = gr.Slider(
                                    minimum=0.5, maximum=5.0, step=0.1, value=ui["sparse_ratio"], 
                                    label="Sparse Ratio", 
                                    info=TIPS["sparse_ratio"]
                                )
                                img_local_range = gr.Slider(
                                    minimum=3, maximum=15, step=2, value=ui["local_range"], 
                                    label="Local Range", 
                                    info=TIPS["local_range"]
                                )
                                img_quality = gr.Slider(
                                    minimum=1, maximum=10, step=1, value=ui["quality"], 
                                    label="Output Image Quality", 
                                    info=TIPS["img_quality"]
                                )
                            with gr.Column(scale=1):
                                img_kv_ratio = gr.Slider(
                                    minimum=1, maximum=8, step=1, value=ui["kv_ratio"], 
                                    label="KV Cache Ratio", 
                                    info=TIPS["kv_ratio"]
                                )
                                img_fps = gr.Number(
                                    value=ui["fps_override"], 
                                    label="Output FPS", 
                                    precision=0, 
                                    info=TIPS["img_fps"],
                                    visible=False
                                )
                                img_device = gr.Textbox(
                                    value=ui["device"], 
                                    label="Device", 
                                    info=TIPS["device"]
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                img_attention_mode = gr.Radio(
                                    choices=["sage", "block"], 
                                    value=ui["attention_mode"], 
                                    label="Attention Mode", 
                                    info=TIPS["attention_mode"]
                                )
                                img_dtype = gr.Radio(
                                    choices=["fp16", "bf16"], 
                                    value=ui["dtype"], 
                                    label="Data Type", 
                                    info=TIPS["dtype"]
                                )
                            with gr.Column(scale=1):
                                img_color_fix = gr.Checkbox(
                                    label="Enable Color Fix", 
                                    value=ui["color_fix"], 
                                    info=TIPS["color_fix"]
                                )
                                img_tiled_vae = gr.Checkbox(
                                    label="Enable Tiled VAE", 
                                    value=ui["tiled_vae"], 
                                    info=TIPS["tiled_vae"]
                                )
                                img_unload_dit = gr.Checkbox(
                                    label="Unload DiT Before Decoding", 
                                    value=ui["unload_dit"], 
                                    info=TIPS["unload_dit"]
                                )
                
                # --- ImageSlider Comparison Window ---
                with gr.Row():
                    img_comparison = gr.ImageSlider(
                        label="Before/After Comparison",
                        interactive=False,
                        elem_classes="image-window"
                    )
                

            # --- TOOLBOX TAB ---
            with gr.TabItem("🛠️ Toolbox", id=2):
                with gr.Row():
                    # --- Left Column: Inputs and Pipeline Control ---
                    with gr.Column(scale=1):
                        # Hidden state to track the active input tab (0=Single, 1=Batch)
                        tb_active_tab_index = gr.Number(value=0, visible=False)
                        
                        with gr.Tabs() as tb_input_tabs:
                            with gr.TabItem("Single Video", id=0):
                                 tb_input_video = gr.Video(label="Toolbox Input Video", autoplay=True, elem_classes="video-window")
                            with gr.TabItem("Batch Video", id=1):
                                tb_batch_input_files = gr.File(
                                    label="Upload Multiple Videos for Batch Processing",
                                    file_count="multiple",
                                    type="filepath",
                                    file_types=["video"],
                                    height="300px",                            
                                )
                                gr.Markdown("**Or** specify a folder path containing videos:")
                                tb_batch_folder_path = gr.Textbox(
                                    placeholder="e.g., C:\\Users\\Videos\\batch",
                                    label="Folder Path",
                                    show_label=False
                                )
                            tb_start_pipeline_btn = gr.Button("🚀 Start Pipeline Processing", variant="primary", size="sm")                              
                            with gr.Group():
                                _tb_ops_default = [
                                    o.strip()
                                    for o in str(ui.get("tb_pipeline_ops") or "Frame Adjust,Export").split(",")
                                    if o.strip()
                                ]
                                tb_pipeline_steps_chkbox = gr.CheckboxGroup(
                                    choices=["Frame Adjust", "Video Loop", "Export"],
                                    value=_tb_ops_default or ["Frame Adjust", "Export"],
                                    show_label=False,
                                    info=TIPS["tb_pipeline"]
                                )
                            with gr.Accordion(
                                "📦 Post-upscale queue (Ready for Toolbox → Ready for CIV)",
                                open=True,
                            ):
                                _tb_inbox_default = ui.get(
                                    "tb_inbox_folder",
                                    r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox",
                                )
                                gr.Markdown(
                                    "<span style='font-size:0.9em;color:#94a3b8;'>"
                                    "Drop upscaled videos in <b>Ready for Toolbox</b>, then Start. "
                                    "Default: <b>4× frames</b> + <b>Export</b> → "
                                    f"<code>{_tb_inbox_default}</code> "
                                    "→ Ready for CIV.</span>"
                                )
                                tb_inbox_path = gr.Textbox(
                                    value=ui.get(
                                        "tb_inbox_folder",
                                        r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox",
                                    ),
                                    label="Toolbox inbox (hand-sort folder)",
                                    info="Place FlashVSR completed upscales here. Start/Resume auto-scans.",
                                )
                                tb_queue_status = gr.HTML(value=get_toolbox_work_queue().status_html())
                                with gr.Row():
                                    tb_queue_add_btn = gr.Button("➕ Scan Inbox → Queue", size="sm")
                                    tb_queue_run_btn = gr.Button(
                                        "▶️ Start / Resume Toolbox Queue",
                                        variant="primary",
                                        size="sm",
                                    )
                                    tb_queue_stop_btn = gr.Button(
                                        "⏹ Stop After Current", variant="stop", size="sm"
                                    )
                                with gr.Row():
                                    tb_queue_requeue_btn = gr.Button("↺ Re-queue Failed", size="sm")
                                    tb_queue_clear_done_btn = gr.Button("Clear Done", size="sm")
                                    tb_queue_clear_all_btn = gr.Button(
                                        "Clear Entire Queue", size="sm", variant="stop"
                                    )
                            
                            # Video Analysis Section
                            tb_analyze_button = gr.Button("📊 Analyze Input Video", size="sm", variant="secondary", visible=False)
                            with gr.Accordion("📊 Input Video Analysis", open=False) as tb_analysis_accordion:
                                tb_input_analysis_html = gr.HTML(
                                    value='<div style="padding: 12px; background: #0f1419; border: 1px solid #2d3748; border-radius: 6px; color: #94a3b8; text-align: center;">Upload video to see analysis</div>'
                                )


                    # --- Right Column: Output and Controls ---
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.TabItem("Processed Video"):
                                processed_video = gr.Video(label="Toolbox Processed Video", interactive=False, elem_classes="video-window")
                        
                        with gr.Row():
                            tb_use_as_input_btn = gr.Button("Use as Input", size="sm", scale=4)
                            initial_autosave_state = toolbox_processor.autosave_enabled
                            tb_manual_save_btn = gr.Button("Save Manually 💾", variant="primary", size="sm", scale=4, visible=not initial_autosave_state)

                        # --- Settings & File Management Group ---
                        with gr.Group():
                            with gr.Row():
                                tb_open_folder_btn = gr.Button("Open Output Folder", size="sm", variant="huggingface")
                                tb_clear_temp_btn = gr.Button("⚠️ Clear Temp Files", size="sm", variant="stop")                                
                            with gr.Row():
                                tb_autosave_checkbox = gr.Checkbox(label="Autosave", scale=1, value=initial_autosave_state, info=TIPS["autosave"])

                        # Output Video Analysis
                        with gr.Row():   
                            with gr.Accordion("📊 Output Video Analysis", open=False) as tb_output_analysis_accordion:                     
                                tb_output_analysis_html = gr.HTML(
                                    value='<div style="padding: 12px; background: #0f1419; border: 1px solid #2d3748; border-radius: 6px; color: #94a3b8; text-align: center;">Process video to see output analysis</div>'
                                )

                        with gr.Row():
                            tb_status_message = gr.Textbox(label="Toolbox Console", lines=8, interactive=False)
                        
                # --- Accordion for operation settings ---
                with gr.Accordion("Operations Settings", open=True):
                    with gr.Tabs():
                        # --- Frame Adjust Tab ---
                        with gr.TabItem("🎞️ Frame Adjust (Speed & Interpolation)"):
                            with gr.Row():
                                gr.Markdown("Adjust video speed and interpolate frames using RIFE AI.")
                            with gr.Row():
                                with gr.Group():
                                    process_fps_mode = gr.Radio(
                                        choices=["No Interpolation", "2x Frames", "4x Frames"],
                                        value=ui.get("tb_fps_mode") or "4x Frames",
                                        label="RIFE Frame Interpolation",
                                        info=TIPS["rife"],
                                    )
                                    process_speed_factor = gr.Slider(
                                        minimum=0.5, maximum=2.0, step=0.05, value=1, label="Adjust Video Speed Factor",
                                        info=TIPS["speed_factor"]
                                    )
                            with gr.Row():
                                frames_output_quality = gr.Slider(
                                    minimum=0, maximum=100, step=5, value=ui["tb_frames_quality"], label="Output Quality",
                                    info=TIPS["frames_quality"]
                                )
                                frames_use_streaming_checkbox = gr.Checkbox(
                                    label="Use Streaming (Low Memory Mode)", value=False,
                                    info=TIPS["rife_stream"]              
                                )                               
                            process_frames_btn = gr.Button("🚀 Process Frames", variant="primary")

                        # --- Loop Tab ---
                        with gr.TabItem("🔄 Video Loop"):
                            with gr.Row():
                                gr.Markdown("Create looped or ping-pong versions of the video.")

                            loop_type_select = gr.Radio(choices=["loop", "ping-pong"], value="loop", label="Loop Type", info=TIPS["loop_type"])
                            with gr.Row():                            
                                num_loops_slider = gr.Slider(
                                    minimum=1, maximum=10, step=1, value=1, label="Number of Loops/Repeats",
                                    info=TIPS["num_loops"]
                                )
                                loop_output_quality = gr.Slider(
                                    minimum=0, maximum=100, step=5, value=85, label="Output Quality",
                                    info=TIPS["loop_quality"]
                                )
                            create_loop_btn = gr.Button("🔁 Create Loop", variant="primary")
                            
                        # --- Export Tab ---
                        with gr.TabItem("📦 Compress, Encode & Export"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    export_format_radio = gr.Radio(
                                        ["MP4 (H.264)", "MP4 (H.265)", "WebM (VP9)", "GIF"], 
                                        value="MP4 (H.264)", 
                                        label="Output Format",
                                        info=TIPS["export_format"]
                                    )
                                    export_quality_slider = gr.Slider(
                                        0, 100, value=ui["tb_export_quality"], step=4, label="Quality",
                                        info=TIPS["export_quality"]
                                    )
                                    export_two_pass = gr.Checkbox(
                                        label="Two-Pass Encoding",
                                        value=False,
                                        visible=False,  # temporarily hiding due to issues with longer videos
                                        info=TIPS["two_pass"]
                                    )
                                with gr.Column(scale=2):
                                    export_resize_slider = gr.Slider(
                                        256, 3840, value=ui["tb_export_max_width"], step=64, label="Max Width (pixels)",
                                        info=TIPS["export_width"]
                                    )
                                    export_name_input = gr.Textbox(
                                        label="Output Filename (optional)",
                                        value="",
                                        placeholder="e.g., my_final_video_for_discord",
                                        info=TIPS["export_name"],
                                                                        )
                            export_video_btn = gr.Button("🚀 Export Video", variant="primary")

                # --- Batch Queue / Resume Chunks (maxed) ---
                with gr.Accordion("📦 Batch Queue / Resume Chunks — never lose a 100-file run again", open=True):
                    gr.Markdown(
                        "### Crash-proof batch workflow\n"
                        "1. **Create queue** from your source folder (auto-skips already finished via stage tags `_1/_2/_3`)\n"
                        "2. **Prepare NEXT chunk** (~20 files) → paste that folder into **FlashVSR → Batch → Folder Path**\n"
                        "3. If it dies mid-run: **Refresh status** (or **Import crashed batch**) → prepare next chunk again\n"
                        "4. FlashVSR batches also write `BATCH_PROGRESS.txt` + `REMAINING.txt` + `INPUTS.txt` in the batch folder"
                    )
                    with gr.Row():
                        bq_source_folder = gr.Textbox(
                            label="Source folder (raw inputs)",
                            placeholder=r"D:\INPUTS\my_100_clips",
                            info="Folder of videos to process (not the outputs).",
                            scale=3,
                        )
                        bq_chunk_size = gr.Slider(
                            minimum=5,
                            maximum=50,
                            step=1,
                            value=20,
                            label="Chunk size",
                            info="Files per work pack. 15–25 is ideal for 4090 OOM recovery.",
                            scale=1,
                        )
                    with gr.Row():
                        bq_output_dirs = gr.Textbox(
                            label="Output folders to scan (one per line)",
                            value="\n".join(
                                [
                                    str(get_output_dir()),
                                    str(get_toolbox_output_dir()),
                                ]
                            ),
                            lines=3,
                            info="Add any batch_* folders or CIV paths. Recursive scan for matches.",
                            scale=3,
                        )
                        with gr.Column(scale=1):
                            bq_target_stage = gr.Radio(
                                choices=["1 upscale", "2 interp", "3 posted"],
                                value="1 upscale",
                                label="Done when stage ≥",
                                info="_1 upscale · _2 RIFE · _3 export",
                            )
                            bq_sort_mode = gr.Dropdown(
                                choices=["mtime", "mtime_asc", "name", "size"],
                                value="mtime",
                                label="Source sort",
                                info="mtime = newest first (recommended). mtime_asc = oldest first. name / size = alphabetical or size.",
                            )
                            bq_link_mode = gr.Dropdown(
                                choices=["auto", "hardlink", "symlink", "copy"],
                                value="auto",
                                label="Chunk file placement",
                                info="auto tries hardlink → symlink → copy.",
                            )
                    with gr.Row():
                        bq_recursive = gr.Checkbox(
                            label="Recursive source scan",
                            value=False,
                            info="Include videos in subfolders of the source path.",
                        )
                        bq_min_mb = gr.Number(
                            value=0,
                            label="Min size (MB)",
                            precision=1,
                            info="Ignore tiny/corrupt stubs. 0 = no filter.",
                        )
                        bq_name = gr.Textbox(
                            label="Queue name (optional)",
                            placeholder="civ_batch_friday",
                            scale=1,
                        )
                    with gr.Row():
                        bq_queue_dropdown = gr.Dropdown(
                            label="Existing queues",
                            choices=[],
                            value=None,
                            allow_custom_value=False,
                            scale=3,
                            info="Saved under outputs/batch_queues — pick to resume.",
                        )
                        bq_chunk_pick = gr.Dropdown(
                            label="Chunk (optional)",
                            choices=[],
                            value=None,
                            allow_custom_value=False,
                            scale=1,
                            info="Leave empty = next pending. Or force a specific pack.",
                        )
                    with gr.Row():
                        bq_create_btn = gr.Button("🆕 Create queue", variant="primary", size="sm")
                        bq_refresh_btn = gr.Button("🔄 Refresh status", size="sm")
                        bq_prepare_btn = gr.Button("📂 Prepare NEXT / selected chunk", variant="primary", size="sm")
                        bq_requeue_fail_btn = gr.Button("♻️ Requeue FAILED", size="sm")
                        bq_rebuild_btn = gr.Button("🧩 Rebuild chunks", size="sm")
                    with gr.Row():
                        bq_import_batch_dir = gr.Textbox(
                            label="Import crashed FlashVSR batch folder",
                            placeholder=r"C:\pinokio\api\FlashVSR_plus_pinokio.git\app\outputs\batch_20260731_120000",
                            info="Folder with BATCH_PROGRESS.json + INPUTS.txt (written automatically now).",
                            scale=3,
                        )
                        bq_import_btn = gr.Button("📥 Import crash → queue", size="sm", scale=1)
                    with gr.Row():
                        bq_reload_list_btn = gr.Button("📋 Reload queue list", size="sm")
                        bq_open_btn = gr.Button("Open queues root", size="sm")
                        bq_open_active_btn = gr.Button("Open active queue folder", size="sm")
                        bq_open_work_btn = gr.Button("Open prepared work folder", size="sm")
                        bq_push_batch_btn = gr.Button("➡️ Push path → FlashVSR Batch folder", size="sm")
                    bq_work_folder = gr.Textbox(
                        label="Prepared chunk folder (FlashVSR Batch → Folder Path)",
                        interactive=True,
                        info="Hardlinks/copies of only unfinished files. Copy this path or use Push button.",
                    )
                    bq_active_id = gr.Textbox(label="Active queue id", interactive=False)
                    bq_status_html = gr.HTML(
                        value='<div style="padding:12px;color:#6c757d">Create a queue, import a crash, or select an existing one.</div>'
                    )
                    bq_console = gr.Textbox(label="Queue console", lines=14, interactive=False)

        
            
        ### --- EVENT HANDLERS --- ###

        def do_sleep(delay_seconds=6):
            """
            Just sleeps. This will be used in the Gradio chain with no outputs 
            to prevent the UI from fading the target component.
            """
            time.sleep(delay_seconds)

        def get_random_idle_state():
            """Returns a random idle state HTML for the save_status display."""
            return random.choice(IDLE_STATES)

        def do_clear():
            """Returns a random idle state HTML instead of empty string."""
            return get_random_idle_state()
        
        def display_status_with_timeout(status_msg):
            """Display status message, sleep, then clear to idle state."""
            # This is a helper to avoid repeating the .then() chain
            # Returns: (status_msg, None, idle_state) for the three steps
            return status_msg
        
        def toggle_tiled_dit_options(is_checked):
            return gr.update(visible=is_checked)
        
        def update_clear_on_start_config(value):
            config = load_config()
            config["clear_temp_on_start"] = value
            save_config(config)
            status = "enabled" if value else "disabled"
            return f'<div style="padding: 1px; background-color: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc;">ℹ️ Clear temp on start: {status}</div>'
        
        def update_autosave_config(value):
            config = load_config()
            config["autosave"] = value
            save_config(config)
            status = "enabled" if value else "disabled"
            return f'<div style="padding: 1px; background-color: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc;">ℹ️ Autosave: {status}</div>'

        tiled_dit_checkbox.change(fn=toggle_tiled_dit_options, inputs=[tiled_dit_checkbox], outputs=[tiled_dit_options])
        
        autosave_checkbox.change(
            fn=update_autosave_config, 
            inputs=[autosave_checkbox], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        clear_on_start_checkbox.change(
            fn=update_clear_on_start_config, 
            inputs=[clear_on_start_checkbox], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        open_folder_button.click(
            fn=lambda: open_folder(get_output_dir()), 
            inputs=[], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        clear_temp_button.click(
            fn=clear_temp_files,
            inputs=[],
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        save_button.click(
            fn=save_file_manually, 
            inputs=[output_file_path], 
            outputs=[save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )

        # Analyze video button handler - updates slider max and preview
        def handle_analyze(video_path):
            html, width, height = analyze_input_video(video_path)
            duration = get_video_duration(video_path)
            
            # Update resize slider maximum to video width (or keep 2048 if video is larger)
            slider_max = min(width, 2048) if width > 0 else 2048
            # Clamp the value between minimum (256) and the new maximum
            # Use 512 as preferred default (matches slider initial value)
            slider_value = max(256, min(512, slider_max))
            
            # Update resize slider - set both value and maximum to prevent reset button errors
            resize_slider_update = gr.update(
                minimum=256,
                maximum=slider_max,
                value=slider_value,
                interactive=(width > 0)
            )
            
            # Update trim sliders based on duration
            trim_start_update = gr.update(
                maximum=duration if duration > 0 else 60,
                value=0,
                interactive=(duration > 0)
            )
            trim_end_update = gr.update(
                maximum=duration if duration > 0 else 60,
                value=0,  # 0 means "end of video"
                interactive=(duration > 0)
            )
            
            # Update previews
            resize_preview = preview_resize(video_path, slider_value)
            trim_preview = preview_trim(video_path, 0, 0)
            
            return html, width, height, duration, resize_slider_update, resize_preview, trim_start_update, trim_end_update, trim_preview
        
        # Update preview when slider changes
        def update_resize_preview(video_path, max_width):
            return preview_resize(video_path, max_width)
        
        resize_max_width_slider.change(
            fn=update_resize_preview,
            inputs=[input_video, resize_max_width_slider],
            outputs=[resize_preview_html]
        )
        
       # When video changes, auto-analyze it and update chunk preview
        def handle_video_change(video_path, chunk_duration):
            if not video_path:
                return (
                    '<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">Upload video to see analysis</div>',
                    0,
                    0,
                    0,
                    gr.update(minimum=256, maximum=2048, value=512, interactive=False),
                    '<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">Upload video to enable resize</div>',
                    gr.update(maximum=60, value=0, interactive=False),
                    gr.update(maximum=60, value=0, interactive=False),
                    # gr.update(maximum=60, value=0, interactive=False),
                    # gr.update(maximum=60, value=0, interactive=False),
                    '<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">Upload video to enable trim</div>',
                    '<div style="padding: 6px; background: #0c2d48; border: 1px solid #1e4a6e; border-radius: 4px; color: #7dd3fc; font-size: 0.85em; text-align: center;">💡 Enable chunk processing for videos that exceed your available VRAM</div>',
                    gr.update(visible=False)  # Hide output analysis when input changes
                )
                
            # Auto-analyze the video
            analysis_results = handle_analyze(video_path)
            # Update chunk preview
            chunk_preview = preview_chunk_processing(video_path, chunk_duration)
            # Hide output analysis when input changes
            return analysis_results + (chunk_preview, gr.update(visible=False))
        
        input_video.change(
            fn=handle_video_change,
            inputs=[input_video, chunk_duration_slider],
            outputs=[
                video_analysis_html, 
                current_video_width, 
                current_video_height, 
                current_video_duration,
                resize_max_width_slider, 
                resize_preview_html,
                trim_start_slider,
                trim_end_slider,
                trim_preview_html,
                chunk_preview_display,
                video_output_analysis_html
            ]
        )
        
        # Update trim preview when parameters change
        def update_trim_preview(video_path, start_time, end_time):
            return preview_trim(video_path, start_time, end_time)
        
        trim_start_slider.change(
            fn=update_trim_preview,
            inputs=[input_video, trim_start_slider, trim_end_slider],
            outputs=[trim_preview_html]
        )
        
        trim_end_slider.change(
            fn=update_trim_preview,
            inputs=[input_video, trim_start_slider, trim_end_slider],
            outputs=[trim_preview_html]
        )
        
        # Apply trim button handler
        def handle_trim_only(video_path, start_time, end_time, progress=gr.Progress()):
            if not video_path:
                return video_path, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            trimmed_path = trim_video(video_path, start_time, end_time, progress)
            
            # Re-analyze the trimmed video
            html, width, height = analyze_input_video(trimmed_path)
            duration = get_video_duration(trimmed_path)
            
            slider_max = min(width, 2048) if width > 0 else 2048
            slider_value = max(256, min(512, slider_max))
            resize_slider_update = gr.update(minimum=256, maximum=slider_max, value=slider_value)
            resize_preview = preview_resize(trimmed_path, slider_value)
            
            trim_start_update = gr.update(maximum=duration if duration > 0 else 60, value=0)
            trim_end_update = gr.update(maximum=duration if duration > 0 else 60, value=0)
            trim_preview = preview_trim(trimmed_path, 0, 0)
            
            return trimmed_path, html, duration, resize_slider_update, resize_preview, trim_start_update, trim_end_update, trim_preview
        
        trim_button.click(
            fn=handle_trim_only,
            inputs=[input_video, trim_start_slider, trim_end_slider],
            outputs=[input_video, video_analysis_html, current_video_duration, resize_max_width_slider, resize_preview_html, trim_start_slider, trim_end_slider, trim_preview_html]
        )

        # Apply resize button handler
        def handle_resize_and_update(video_path, max_width, scale, progress=gr.Progress()):
            resized_path = resize_input_video(video_path, max_width, scale=int(scale), progress=progress)
            
            # After resize, analyze the new video to update sliders
            html, width, height = analyze_input_video(resized_path)
            duration = get_video_duration(resized_path)
            
            resize_slider_max = min(width, 2048) if width > 0 else 2048
            # Clamp value between minimum and maximum
            resize_slider_value = max(256, min(max_width, resize_slider_max))
            
            resize_slider_update = gr.update(minimum=256, maximum=resize_slider_max, value=resize_slider_value)
            resize_preview = preview_resize(resized_path, resize_slider_value, scale=int(scale))
            
            trim_start_update = gr.update(maximum=duration if duration > 0 else 60, value=0)
            trim_end_update = gr.update(maximum=duration if duration > 0 else 60, value=0)
            trim_preview = preview_trim(resized_path, 0, 0)
            
            return resized_path, html, duration, resize_slider_update, resize_preview, trim_start_update, trim_end_update, trim_preview
        
        resize_button.click(
            fn=handle_resize_and_update,
            inputs=[input_video, resize_max_width_slider, scale_slider],
            outputs=[input_video, video_analysis_html, current_video_duration, resize_max_width_slider, resize_preview_html, trim_start_slider, trim_end_slider, trim_preview_html]
        )
        
        # Save preprocessed video button handler
        save_preprocessed_btn.click(
            fn=save_preprocessed_video,
            inputs=[input_video]
        )
        
        # Main processing handler - routes to chunk or normal processing
        def handle_processing(
            input_path, enable_chunks, chunk_duration, mode, model_version, scale, color_fix, tiled_vae,
            tiled_dit, tile_size, tile_overlap, unload_dit, dtype_str, seed, device,
            fps_override, quality, attention_mode, sparse_ratio, kv_ratio, local_range, autosave, create_comparison,
            batch_resize_preset,
        ):
            if input_path:
                input_path = apply_batch_resize_preset(
                    input_path, batch_resize_preset, scale=scale
                )
            if enable_chunks:
                # Use chunk processing mode (comparison not supported in chunk mode)
                return process_video_with_chunks(
                    input_path, chunk_duration, mode, model_version, scale, color_fix, tiled_vae, tiled_dit,
                    tile_size, tile_overlap, unload_dit, dtype_str, seed, device, fps_override,
                    quality, attention_mode, sparse_ratio, kv_ratio, local_range, autosave
                )
            else:
                # Use normal processing
                return run_flashvsr_single(
                    input_path, mode, model_version, scale, color_fix, tiled_vae, tiled_dit, tile_size,
                    tile_overlap, unload_dit, dtype_str, seed, device, fps_override, quality,
                    attention_mode, sparse_ratio, kv_ratio, local_range, autosave, create_comparison
                )
        
        def should_randomize_seed(current_seed, randomize):
            """Generate a new random seed if randomize is checked, otherwise return current seed."""
            if randomize:
                return random.randint(0, 2**32 - 1)
            return current_seed
        
        run_button.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[video_output_analysis_html]
        ).then(
            fn=check_model_status,
            inputs=[model_version_radio],
            outputs=[save_status]
        ).then(
            fn=should_randomize_seed,
            inputs=[seed_number, randomize_seed],
            outputs=[seed_number]
        ).then(
            fn=handle_processing,
            inputs=[
                input_video, enable_chunk_processing, chunk_duration_slider,
                mode_radio, model_version_radio, scale_slider, color_fix_checkbox, tiled_vae_checkbox,
                tiled_dit_checkbox, tile_size_slider, tile_overlap_slider, unload_dit_checkbox,
                dtype_radio, seed_number, device_textbox, fps_number, quality_slider, attention_mode_radio,
                sparse_ratio_slider, kv_ratio_slider, local_range_slider, autosave_checkbox, create_comparison_checkbox,
                batch_resize_preset,
            ],
            outputs=[video_output, output_file_path, video_slider_output, completion_status]
        ).then(
            fn=analyze_output_video,
            inputs=[output_file_path],
            outputs=[video_output_analysis_html]
        ).then(
            fn=lambda status_msg: status_msg,
            inputs=[completion_status],
            outputs=[save_status],
            show_progress=False
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )
        
        # Toggle chunk settings visibility and update preview
        def toggle_chunk_settings(enable_chunks, video_path, chunk_duration):
            if enable_chunks:
                preview = preview_chunk_processing(video_path, chunk_duration)
                return gr.update(visible=True), gr.update(visible=True, value=preview)
            else:
                return gr.update(visible=False), gr.update(visible=False)
        
        enable_chunk_processing.change(
            fn=toggle_chunk_settings,
            inputs=[enable_chunk_processing, input_video, chunk_duration_slider],
            outputs=[chunk_settings_row, chunk_preview_display]
        )
        
        # Update chunk preview when duration changes
        def update_chunk_preview_display(video_path, chunk_duration):
            return preview_chunk_processing(video_path, chunk_duration)
        
        chunk_duration_slider.change(
            fn=update_chunk_preview_display,
            inputs=[input_video, chunk_duration_slider],
            outputs=[chunk_preview_display]
        )

        # --- Persistent work queue (add anytime / soft-stop / resume) ---
        def _collect_batch_paths(batch_files, folder_path):
            paths = []
            if folder_path and os.path.isdir(folder_path):
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']
                paths = [
                    str(f) for f in Path(folder_path).iterdir()
                    if f.is_file() and f.suffix.lower() in video_extensions
                ]
                paths.sort()
            elif batch_files:
                for f in batch_files:
                    p = f if isinstance(f, str) else getattr(f, "name", None)
                    if p:
                        paths.append(p)
            return paths

        def handle_queue_add(batch_files, folder_path):
            wq = get_flashvsr_work_queue()
            paths = []
            if batch_files:
                for f in batch_files:
                    p = f if isinstance(f, str) else getattr(f, "name", None)
                    if p:
                        paths.append(p)
            if folder_path and os.path.isdir(folder_path):
                a, s = wq.add_folder(folder_path)
            else:
                a, s = 0, 0
            if paths:
                a2, s2 = wq.add_paths(paths)
                a += a2
                s += s2
            note = f"Added {a} video(s)" + (f", skipped {s} already in queue" if s else "")
            if a == 0 and s == 0:
                note = "Nothing to add — upload files or set a folder path with videos."
            return wq.status_html(note)

        def handle_queue_stop():
            wq = get_flashvsr_work_queue()
            note = wq.request_stop()
            log(note, message_type="warning")
            return wq.status_html(note)

        def handle_queue_clear_done():
            wq = get_flashvsr_work_queue()
            n = wq.clear_done()
            return wq.status_html(f"Cleared {n} completed item(s).")

        def handle_queue_clear_all():
            wq = get_flashvsr_work_queue()
            n = wq.clear_all()
            return wq.status_html(f"Cleared entire queue ({n} items).")

        def handle_queue_requeue_failed():
            wq = get_flashvsr_work_queue()
            n = wq.requeue_failed()
            return wq.status_html(f"Re-queued {n} failed item(s).")

        def handle_batch_processing(
            mode, model_version, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap,
            unload_dit, dtype_str, seed, device, fps_override, quality, attention_mode,
            sparse_ratio, kv_ratio, local_range, batch_resize_preset, enable_chunks, chunk_duration,
            watch_folder,
        ):
            # Persist watch folder from UI so Start/Resume picks up new downloads
            if watch_folder and str(watch_folder).strip():
                cfg = load_config()
                cfg["batch_watch_folder"] = str(watch_folder).strip()
                save_config(cfg)
            last_video, queue_html = run_flashvsr_work_queue(
                mode, model_version, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap,
                unload_dit, dtype_str, seed, device, fps_override, quality, attention_mode,
                sparse_ratio, kv_ratio, local_range, batch_resize_preset, enable_chunks, chunk_duration
            )
            return last_video, last_video, None, queue_html, queue_html

        batch_add_queue_btn.click(
            fn=handle_queue_add,
            inputs=[flashvsr_batch_input_files, batch_folder_path],
            outputs=[flashvsr_queue_status],
        )
        # queue=False so stop can run while Start/Resume is busy
        batch_stop_button.click(
            fn=handle_queue_stop,
            inputs=[],
            outputs=[flashvsr_queue_status],
            queue=False,
        )
        batch_clear_done_btn.click(
            fn=handle_queue_clear_done,
            inputs=[],
            outputs=[flashvsr_queue_status],
        )
        batch_clear_all_btn.click(
            fn=handle_queue_clear_all,
            inputs=[],
            outputs=[flashvsr_queue_status],
        )
        batch_requeue_failed_btn.click(
            fn=handle_queue_requeue_failed,
            inputs=[],
            outputs=[flashvsr_queue_status],
        )

        batch_run_button.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[video_output_analysis_html]
        ).then(
            fn=check_model_status,
            inputs=[model_version_radio],
            outputs=[save_status]
        ).then(
            fn=should_randomize_seed,
            inputs=[seed_number, randomize_seed],
            outputs=[seed_number]
        ).then(
            fn=handle_batch_processing,
            inputs=[
                mode_radio, model_version_radio, scale_slider, color_fix_checkbox, tiled_vae_checkbox,
                tiled_dit_checkbox, tile_size_slider, tile_overlap_slider, unload_dit_checkbox,
                dtype_radio, seed_number, device_textbox, fps_number, quality_slider, attention_mode_radio,
                sparse_ratio_slider, kv_ratio_slider, local_range_slider, batch_resize_preset,
                enable_chunk_processing, chunk_duration_slider, batch_folder_path,
            ],
            outputs=[video_output, output_file_path, video_slider_output, completion_status, flashvsr_queue_status]
        ).then(
            fn=analyze_output_video,
            inputs=[output_file_path],
            outputs=[video_output_analysis_html]
        ).then(
            fn=lambda status_msg: status_msg,
            inputs=[completion_status],
            outputs=[save_status],
            show_progress=False
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )

        def _gt_q_add(watch):
            wq = get_group_therapy_queue()
            watch = str(watch or "").strip()
            if watch:
                cfg = load_config()
                cfg["batch_watch_folder"] = watch
                save_config(cfg)
            a, s = wq.add_folder(watch) if watch and os.path.isdir(watch) else (0, 0)
            ui = get_ui_defaults()
            gt.assign_groups(wq, int(ui.get("gt_group_size") or 10))
            note = f"Added {a} video(s)" + (f", skipped {s} already in queue" if s else "")
            if a == 0 and s == 0:
                note = "Nothing to add — set the original folder path."
            return wq.status_html(note)

        def _gt_q_stop():
            note = get_group_therapy_queue().request_stop()
            log(note, message_type="warning")
            return get_group_therapy_queue().status_html(note)

        def _gt_q_clear_done():
            n = get_group_therapy_queue().clear_done()
            return get_group_therapy_queue().status_html(f"Cleared {n} completed item(s).")

        def _gt_q_clear_all():
            n = get_group_therapy_queue().clear_all()
            return get_group_therapy_queue().status_html(f"Cleared entire queue ({n} items).")

        def _gt_q_requeue():
            n = get_group_therapy_queue().requeue_failed()
            return get_group_therapy_queue().status_html(f"Re-queued {n} failed item(s).")

        def handle_group_therapy(
            mode, model_version, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap,
            unload_dit, dtype_str, seed, device, fps_override, quality, attention_mode,
            sparse_ratio, kv_ratio, local_range, batch_resize_preset, enable_chunks, chunk_duration,
            group_size, watch_folder, before_dir, after_dir, do_upscale, do_rife1, do_rife2, do_export,
        ):
            last_video, queue_html = run_group_therapy(
                mode, model_version, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap,
                unload_dit, dtype_str, seed, device, fps_override, quality, attention_mode,
                sparse_ratio, kv_ratio, local_range, batch_resize_preset, enable_chunks, chunk_duration,
                group_size, watch_folder, before_dir, after_dir, do_upscale, do_rife1, do_rife2, do_export,
            )
            return last_video, last_video, None, queue_html, queue_html

        gt_add_btn.click(fn=_gt_q_add, inputs=[gt_watch_folder], outputs=[gt_queue_status])
        gt_stop_btn.click(fn=_gt_q_stop, inputs=[], outputs=[gt_queue_status], queue=False)
        gt_clear_done_btn.click(fn=_gt_q_clear_done, outputs=[gt_queue_status])
        gt_clear_all_btn.click(fn=_gt_q_clear_all, outputs=[gt_queue_status])
        gt_requeue_btn.click(fn=_gt_q_requeue, outputs=[gt_queue_status])
        gt_run_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[video_output_analysis_html],
        ).then(
            fn=check_model_status,
            inputs=[model_version_radio],
            outputs=[save_status],
        ).then(
            fn=should_randomize_seed,
            inputs=[seed_number, randomize_seed],
            outputs=[seed_number],
        ).then(
            fn=handle_group_therapy,
            inputs=[
                mode_radio, model_version_radio, scale_slider, color_fix_checkbox, tiled_vae_checkbox,
                tiled_dit_checkbox, tile_size_slider, tile_overlap_slider, unload_dit_checkbox,
                dtype_radio, seed_number, device_textbox, fps_number, quality_slider, attention_mode_radio,
                sparse_ratio_slider, kv_ratio_slider, local_range_slider, batch_resize_preset,
                enable_chunk_processing, chunk_duration_slider,
                gt_group_size, gt_watch_folder, gt_before_dir, gt_after_dir,
                gt_do_upscale, gt_do_rife1, gt_do_rife2, gt_do_export,
            ],
            outputs=[video_output, output_file_path, video_slider_output, completion_status, gt_queue_status],
        ).then(
            fn=analyze_output_video,
            inputs=[output_file_path],
            outputs=[video_output_analysis_html],
        ).then(
            fn=lambda status_msg: status_msg,
            inputs=[completion_status],
            outputs=[save_status],
            show_progress=False,
        )

        def update_monitor():
            gpu_info, cpu_info = SystemMonitor.get_system_info()
            # Return same info for both video and image tabs
            return gpu_info, cpu_info, gpu_info, cpu_info
            
        monitor_timer = gr.Timer(2, active=True)
        monitor_timer.tick(fn=update_monitor, outputs=[gpu_monitor, cpu_monitor, img_gpu_monitor, img_cpu_monitor]) 
        
        def send_to_toolbox(video_path):
            if not video_path:
                return gr.update(), gr.update(), '<div style="padding: 1px; background-color: #3d2e0a; border: 1px solid #854d0e; border-radius: 1px; color: #856404;">⚠️ No video to send!</div>'
            # Switches to tab 2 (Toolbox) and sets the input video value
            return gr.update(selected=2), gr.update(value=video_path), '<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ Video sent to Toolbox!</div>'

        send_to_toolbox_btn.click(
            fn=send_to_toolbox,
            inputs=[output_file_path],
            outputs=[main_tabs, tb_input_video, save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[save_status],
            show_progress="hidden"
        )        

        # --- Image Tab Handlers ---

        # Hidden state for file path
        img_output_path = gr.State(None)
        
        # Toggle tiled DiT settings visibility
        img_tiled_dit.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[img_tiled_dit],
            outputs=[img_tiled_dit_options]
        )
        
        # Image upload - analyze and update UI
        def handle_image_change(image_path):
            if not image_path:
                return (
                    '<div style="padding: 12px; background: #0f1419; border: 1px solid #2d3748; border-radius: 6px; color: #94a3b8; text-align: center;">Upload image to see analysis</div>',
                    0,
                    0,
                    gr.update(minimum=256, maximum=2048, value=512, interactive=False),
                    '<div style="padding: 8px; background: #0f1419; border: 1px solid #2d3748; border-radius: 4px; color: #94a3b8; font-size: 0.9em; text-align: center;">Upload image to enable resize</div>',
                    gr.update(visible=False)  # Hide output analysis when input changes
                )
            
            # Analyze the image
            html, width, height = analyze_input_image(image_path)
            
            # Update resize slider based on image width
            slider_max = min(width, 2048) if width > 0 else 2048
            slider_value = max(256, min(512, slider_max))
            resize_slider_update = gr.update(minimum=256, maximum=slider_max, value=slider_value, interactive=True)
            
            # Update resize preview
            resize_preview = preview_image_resize(image_path, slider_value)
            
            # Hide output analysis when input changes
            return html, width, height, resize_slider_update, resize_preview, gr.update(visible=False)
        
        img_input.change(
            fn=handle_image_change,
            inputs=[img_input],
            outputs=[img_analysis_html, img_current_width, img_current_height, img_resize_max_width_slider, img_resize_preview_html, img_output_analysis_html]
        )
        
        # Update resize preview when slider changes
        img_resize_max_width_slider.change(
            fn=preview_image_resize,
            inputs=[img_input, img_resize_max_width_slider],
            outputs=[img_resize_preview_html]
        )
        
        # Resize button click
        img_resize_button.click(
            fn=resize_input_image,
            inputs=[img_input, img_resize_max_width_slider],
            outputs=[img_input]
        )
        
        # Single image run button click
        def should_randomize_img_seed(img_seed, img_randomize_seed):
            """Generate a new random seed if randomize is checked, otherwise return current seed."""
            if img_randomize_seed:
                return random.randint(0, 2**32 - 1)
            return img_seed
        
        img_run_button.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[img_output_analysis_html]
        ).then(
            fn=check_model_status,
            inputs=[img_model_version],
            outputs=[img_save_status]
        ).then(
            fn=should_randomize_img_seed,
            inputs=[img_seed, img_randomize_seed],
            outputs=[img_seed]
        ).then(
            fn=run_flashvsr_image,
            inputs=[
                img_input, img_mode, img_model_version, img_scale, img_color_fix,
                img_tiled_vae, img_tiled_dit, img_tile_size, img_tile_overlap,
                img_unload_dit, img_dtype, img_seed, img_device, img_fps,
                img_quality, img_attention_mode, img_sparse_ratio, img_kv_ratio,
                img_local_range, img_autosave, img_create_comparison
            ],
            outputs=[img_output, img_output_path, img_comparison, completion_status]
        ).then(
            fn=analyze_output_image,
            inputs=[img_output_path],
            outputs=[img_output_analysis_html]
        ).then(
            fn=lambda status_msg: status_msg,
            inputs=[completion_status],
            outputs=[img_save_status],
            show_progress=False
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[img_save_status],
            show_progress="hidden"
        )
        
        # Image work queue handlers
        def handle_img_queue_add(batch_files, folder_path):
            wq = get_flashvsr_image_queue()
            a, s = 0, 0
            if folder_path and os.path.isdir(folder_path):
                a, s = wq.add_folder(folder_path)
            paths = []
            if batch_files:
                for f in batch_files:
                    p = f if isinstance(f, str) else getattr(f, "name", None)
                    if p:
                        paths.append(p)
            if paths:
                a2, s2 = wq.add_paths(paths)
                a += a2
                s += s2
            note = f"Added {a} image(s)" + (f", skipped {s} dupes" if s else "")
            if a == 0 and s == 0:
                note = "Nothing to add — drop images in the watch folder or upload."
            return wq.status_html(note)

        def handle_img_batch_processing(
            mode, model_version, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap,
            unload_dit, dtype_str, seed, device, fps_override, quality, attention_mode,
            sparse_ratio, kv_ratio, local_range, create_comparison, batch_resize_preset, watch_folder,
        ):
            if watch_folder and str(watch_folder).strip():
                cfg = load_config()
                cfg["batch_watch_folder"] = str(watch_folder).strip()
                save_config(cfg)
            last, html = run_flashvsr_image_work_queue(
                mode, model_version, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap,
                unload_dit, dtype_str, seed, device, fps_override, quality, attention_mode,
                sparse_ratio, kv_ratio, local_range, create_comparison, batch_resize_preset,
            )
            return last, html, html, html

        img_add_queue_btn.click(
            fn=handle_img_queue_add,
            inputs=[img_batch_input_files, img_batch_folder_path],
            outputs=[img_queue_status],
        )
        def _img_q_stop():
            wq = get_flashvsr_image_queue()
            return wq.status_html(wq.request_stop())

        def _img_q_clear_done():
            wq = get_flashvsr_image_queue()
            return wq.status_html(f"Cleared {wq.clear_done()} done.")

        def _img_q_clear_all():
            wq = get_flashvsr_image_queue()
            return wq.status_html(f"Cleared {wq.clear_all()} items.")

        def _img_q_requeue():
            wq = get_flashvsr_image_queue()
            return wq.status_html(f"Re-queued {wq.requeue_failed()} failed.")

        img_stop_queue_btn.click(fn=_img_q_stop, inputs=[], outputs=[img_queue_status], queue=False)
        img_clear_done_btn.click(fn=_img_q_clear_done, outputs=[img_queue_status])
        img_clear_all_btn.click(fn=_img_q_clear_all, outputs=[img_queue_status])
        img_requeue_failed_btn.click(fn=_img_q_requeue, outputs=[img_queue_status])

        img_batch_run_button.click(
            fn=check_model_status,
            inputs=[img_model_version],
            outputs=[img_save_status]
        ).then(
            fn=should_randomize_img_seed,
            inputs=[img_seed, img_randomize_seed],
            outputs=[img_seed]
        ).then(
            fn=handle_img_batch_processing,
            inputs=[
                img_mode, img_model_version, img_scale, img_color_fix,
                img_tiled_vae, img_tiled_dit, img_tile_size, img_tile_overlap,
                img_unload_dit, img_dtype, img_seed, img_device, img_fps,
                img_quality, img_attention_mode, img_sparse_ratio, img_kv_ratio,
                img_local_range, img_create_comparison, img_batch_resize_preset,
                img_batch_folder_path,
            ],
            outputs=[img_output, img_batch_status, img_queue_status, completion_status]
        ).then(
            fn=lambda status_msg: status_msg,
            inputs=[completion_status],
            outputs=[img_save_status],
            show_progress=False
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[img_save_status],
            show_progress="hidden"
        )
        
        # Save button click
        img_save_button.click(
            fn=save_file_manually,
            inputs=[img_output_path],
            outputs=[img_save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[img_save_status],
            show_progress="hidden"
        )
        
        # Open folder button
        def open_images_folder():
            images_folder = os.path.join(get_output_dir(), "images")
            os.makedirs(images_folder, exist_ok=True)
            try:
                if sys.platform == "win32":
                    os.startfile(images_folder)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", images_folder])
                else:
                    subprocess.Popen(["xdg-open", images_folder])
                return f'<div style="padding: 1px; background-color: #14352a; border: 1px solid #166534; border-radius: 4px; color: #86efac;">✅ Opened folder: {images_folder}</div>'
            except Exception as e:
                return f'<div style="padding: 1px; background-color: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 4px; color: #fca5a5;">❌ Error opening folder: {e}</div>'


        img_open_folder_button.click(
            fn=open_images_folder,
            inputs=[],
            outputs=[img_save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[img_save_status],
            show_progress="hidden"
        )
        
        img_clear_temp_button.click(
            fn=clear_temp_files,
            inputs=[],
            outputs=[img_save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[img_save_status],
            show_progress="hidden"
        )
        
        # Autosave checkbox change handler
        img_autosave.change(
            fn=update_autosave_config,
            inputs=[img_autosave],
            outputs=[img_save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[img_save_status],
            show_progress="hidden"
        )
        
        # Clear on start checkbox change handler
        img_clear_on_start.change(
            fn=update_clear_on_start_config,
            inputs=[img_clear_on_start],
            outputs=[img_save_status]
        ).then(
            fn=do_sleep,
            inputs=None,
            outputs=None,
            show_progress="hidden"
        ).then(
            fn=do_clear,
            inputs=None,
            outputs=[img_save_status],
            show_progress="hidden"
        )

        # --- Toolbox Tab Handlers ---
        
        tb_open_folder_btn.click(
            fn=toolbox_processor.open_output_folder, 
            outputs=[tb_status_message]
        )
        
        tb_clear_temp_btn.click(
            fn=lambda: re.sub(r'<[^>]+>', '', clear_temp_files()),  # Strip HTML tags for textbox
            inputs=[],
            outputs=[tb_status_message]
        )
        
        def handle_autosave_toggle(is_enabled):
            # Update toolbox processor
            message = toolbox_processor.set_autosave_mode(is_enabled)
            # Save to shared config
            config = load_config()
            config["tb_autosave"] = is_enabled
            save_config(config)
            return gr.update(visible=not is_enabled), message
        
        tb_autosave_checkbox.change(
            fn=handle_autosave_toggle,
            inputs=[tb_autosave_checkbox],
            outputs=[tb_manual_save_btn, tb_status_message]
        )
    
        def handle_single_operation(operation_func, video_path, status_message, **kwargs):
            if not video_path:
                return None, "⚠️ No input video found.", '<div style="padding: 12px; background: #0f1419; border: 1px solid #2d3748; border-radius: 6px; color: #94a3b8; text-align: center;">Process video to see output analysis</div>'

            temp_video = operation_func(video_path, progress=gr.Progress(), **kwargs)

            if not temp_video or temp_video == video_path:
                return video_path, f"❌ {status_message} failed. Check console.", '<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ Operation failed</div>'

            final_video_path = temp_video
            message = f"✅ {status_message} complete."

            if toolbox_processor.autosave_enabled:
                temp_path = Path(temp_video)
                final_path = toolbox_processor.output_dir / temp_path.name
                final_video_path = toolbox_processor._copy_to_permanent_storage(temp_video, final_path)
                message += f"\n✅ Autosaved result to: {final_path}"
            else:
                message += "\nℹ️ Autosave is off. Result is temporary. Use 'Manual Save'."
            
            # Analyze output video
            output_analysis = toolbox_processor.analyze_video_html(final_video_path)
            
            return final_video_path, message, output_analysis        

        process_frames_btn.click(
            lambda video_path, status, fps, speed, stream, quality: handle_single_operation(toolbox_processor.adjust_frames, video_path, status, fps_mode=fps, speed_factor=speed, use_streaming=stream, output_quality=quality),
            inputs=[tb_input_video, gr.Textbox("Frame Adjustment", visible=False), process_fps_mode, process_speed_factor, frames_use_streaming_checkbox, frames_output_quality],
            outputs=[processed_video, tb_status_message, tb_output_analysis_html]
        )
        
        def handle_create_loop(video_path, loop_type, num_loops, quality, progress=gr.Progress()):
            if not video_path:
                return None, "⚠️ No video provided for loop creation.", '<div style="padding: 12px; background: #0f1419; border: 1px solid #2d3748; border-radius: 6px; color: #94a3b8; text-align: center;">Process video to see output analysis</div>'
            
            output_video = toolbox_processor.create_loop(video_path, loop_type, num_loops, quality, progress)
            
            if output_video:
                message = f"✅ Loop created successfully: {os.path.basename(output_video)}"
                final_video = output_video
                if toolbox_processor.autosave_enabled:
                    temp_path = Path(output_video)
                    final_path = toolbox_processor.output_dir / temp_path.name
                    final_video = toolbox_processor._copy_to_permanent_storage(output_video, final_path)
                    message += f"\n✅ Autosaved to: {final_path}"
                else:
                    message += "\nℹ️ Autosave is off. Use 'Manual Save' to keep it."
                
                # Analyze output video
                output_analysis = toolbox_processor.analyze_video_html(final_video)
                return final_video, message, output_analysis
            else:
                return None, "❌ Loop creation failed. Check console for details.", '<div style="padding: 12px; background: #3f1d1d; border: 1px solid #7f1d1d; border-radius: 6px; color: #fca5a5;">❌ Operation failed</div>'
    
    
        create_loop_btn.click(
            fn=handle_create_loop, 
            inputs=[tb_input_video, loop_type_select, num_loops_slider, loop_output_quality], 
            outputs=[processed_video, tb_status_message, tb_output_analysis_html]
        )
        
        export_video_btn.click(
            lambda video_path, status, format, quality, width, name, two_pass: handle_single_operation(toolbox_processor.export_video, video_path, status, export_format=format, quality=quality, max_width=width, output_name=name, two_pass=two_pass),
            inputs=[tb_input_video, gr.Textbox("Exporting", visible=False), export_format_radio, export_quality_slider, export_resize_slider, export_name_input, export_two_pass],
            outputs=[processed_video, tb_status_message, tb_output_analysis_html]
        )

        def handle_manual_save(video_path_from_player):
            if not video_path_from_player or not os.path.exists(video_path_from_player):
                 return "⚠️ No video in the output player to save."
            
            saved_path = toolbox_processor.save_video_from_any_source(video_path_from_player)
            
            if saved_path:
                return f"✅ Video successfully saved to: {saved_path}"
            else:
                return "❌ An error occurred during save. Check the console for details."

        tb_manual_save_btn.click(
            fn=handle_manual_save,
            inputs=[processed_video], # Takes input directly from the video player
            outputs=[tb_status_message]  # Only needs to update the status message
        )

        # Track which input tab is active (Single vs Batch)
        # The select event passes a SelectData object with an 'index' attribute
        def update_tab_index(evt: gr.SelectData):
            return evt.index
        
        tb_input_tabs.select(
            fn=update_tab_index,
            inputs=[],
            outputs=[tb_active_tab_index]
        )

        # Analyze video button - also opens the accordion
        def analyze_and_open(video_path):
            analysis_html = toolbox_processor.analyze_video_html(video_path)
            return analysis_html, gr.update(open=True)
        
        # Auto-analyze input video when it changes
        tb_input_video.change(
            fn=lambda video_path: toolbox_processor.analyze_video_html(video_path),
            inputs=[tb_input_video],
            outputs=[tb_input_analysis_html]
        )
        
        tb_analyze_button.click(
            fn=analyze_and_open,
            inputs=[tb_input_video],
            outputs=[tb_input_analysis_html, tb_analysis_accordion]
        )

        # Wire up the pipeline button
        tb_start_pipeline_btn.click(
            fn=handle_start_pipeline,
            inputs=[
                tb_active_tab_index,
                tb_input_video,
                tb_batch_input_files,
                tb_batch_folder_path,
                tb_pipeline_steps_chkbox,
                # Frame Adjust params
                process_fps_mode,
                process_speed_factor,
                frames_use_streaming_checkbox,
                frames_output_quality,
                # Video Loop params
                loop_type_select,
                num_loops_slider,
                loop_output_quality,
                # Export params
                export_format_radio,
                export_quality_slider,
                export_resize_slider,
                export_name_input,
                export_two_pass
            ],
            outputs=[processed_video, tb_status_message, tb_output_analysis_html]
        )

        # Toolbox post-upscale work queue (Ready for Toolbox → Ready for CIV)
        def handle_tb_queue_scan(inbox):
            wq = get_toolbox_work_queue()
            if inbox and str(inbox).strip():
                cfg = load_config()
                cfg["tb_inbox_folder"] = str(inbox).strip()
                save_config(cfg)
                inbox = str(inbox).strip()
            a, s = wq.add_folder(inbox) if inbox and os.path.isdir(inbox) else (0, 0)
            note = f"Scanned inbox: +{a}" + (f", {s} already queued" if s else "")
            if a == 0 and s == 0:
                note = f"No new videos in inbox: {inbox}"
            return wq.status_html(note)

        def handle_tb_queue_run(inbox):
            if inbox and str(inbox).strip():
                cfg = load_config()
                cfg["tb_inbox_folder"] = str(inbox).strip()
                save_config(cfg)
            last, html = run_toolbox_work_queue()
            return last, html, html

        tb_queue_add_btn.click(
            fn=handle_tb_queue_scan,
            inputs=[tb_inbox_path],
            outputs=[tb_queue_status],
        )
        def _tb_q_stop():
            wq = get_toolbox_work_queue()
            return wq.status_html(wq.request_stop())

        def _tb_q_clear_done():
            wq = get_toolbox_work_queue()
            return wq.status_html(f"Cleared {wq.clear_done()} done.")

        def _tb_q_clear_all():
            wq = get_toolbox_work_queue()
            return wq.status_html(f"Cleared {wq.clear_all()} items.")

        def _tb_q_requeue():
            wq = get_toolbox_work_queue()
            return wq.status_html(f"Re-queued {wq.requeue_failed()} failed.")

        tb_queue_stop_btn.click(fn=_tb_q_stop, inputs=[], outputs=[tb_queue_status], queue=False)
        tb_queue_clear_done_btn.click(fn=_tb_q_clear_done, outputs=[tb_queue_status])
        tb_queue_clear_all_btn.click(fn=_tb_q_clear_all, outputs=[tb_queue_status])
        tb_queue_requeue_btn.click(fn=_tb_q_requeue, outputs=[tb_queue_status])
        tb_queue_run_btn.click(
            fn=handle_tb_queue_run,
            inputs=[tb_inbox_path],
            outputs=[processed_video, tb_status_message, tb_queue_status],
        )

        # Use as Input button - sends processed video back to input
        def use_as_input(video_path):
            if not video_path:
                return None, "⚠️ No processed video to use as input.", '<div style="padding: 12px; background: #0f1419; border: 1px solid #2d3748; border-radius: 6px; color: #94a3b8; text-align: center;">Upload video to see analysis</div>'
            # Analyze the video being moved to input
            input_analysis = toolbox_processor.analyze_video_html(video_path)
            return video_path, "✅ Processed video loaded as input.", input_analysis
        
        tb_use_as_input_btn.click(
            fn=use_as_input,
            inputs=[processed_video],
            outputs=[tb_input_video, tb_status_message, tb_input_analysis_html]
        )

        # --- Batch Queue / Resume handlers (maxed) ---
        _bq_mgr = BatchQueueManager(ROOT_DIR)

        def _bq_stage_from_ui(label: str) -> int:
            text = str(label or "1")
            if text.startswith("3"):
                return 3
            if text.startswith("2"):
                return 2
            return 1

        def _bq_chunk_choices(data):
            choices = []
            items = data.get("items") or []
            for ch in data.get("chunks") or []:
                left = sum(
                    1
                    for ii in ch.get("item_indices") or []
                    if ii < len(items) and items[ii]["status"] in ("pending", "failed")
                )
                total_c = len(ch.get("item_indices") or [])
                mark = " ▶NEXT" if ch.get("id") == data.get("current_chunk") and left else ""
                choices.append(
                    (
                        f"{ch.get('label')}  [{total_c - left}/{total_c} done, {left} left]  {ch.get('status')}{mark}",
                        ch.get("id"),
                    )
                )
            return choices

        def _bq_choices():
            rows = _bq_mgr.list_queues()
            choices = []
            for r in rows:
                st = r.get("stats") or {}
                left = st.get("remaining", 0)
                total = st.get("total", 0)
                done = st.get("complete", 0) or (
                    (st.get("done", 0) or 0) + (st.get("skipped", 0) or 0)
                )
                nxt = r.get("next_chunk") or "—"
                choices.append(
                    (
                        f"{r.get('name')}  [{done}/{total} · {left} left · next {nxt}]  ·  {r.get('id')}",
                        r.get("id"),
                    )
                )
            return choices

        def bq_reload_list(active_id=None):
            choices = _bq_choices()
            value = (
                active_id
                if active_id and any(c[1] == active_id for c in choices)
                else (choices[0][1] if choices else None)
            )
            return gr.update(choices=choices, value=value)

        def _bq_pack(data, work_folder=""):
            qid = data.get("id", "")
            return (
                bq_reload_list(qid),
                gr.update(choices=_bq_chunk_choices(data), value=data.get("current_chunk")),
                qid,
                _bq_mgr.format_html_report(data),
                _bq_mgr.format_report(data),
                work_folder or "",
            )

        def _bq_err(msg, qid=""):
            return (
                bq_reload_list(qid or None),
                gr.update(choices=[], value=None),
                qid or "",
                f'<div style="padding:12px;color:#f87171">❌ {msg}</div>',
                f"❌ {msg}",
                "",
            )

        def bq_create(
            source_folder,
            output_dirs_text,
            chunk_size,
            stage_label,
            name,
            recursive,
            min_mb,
            sort_mode,
        ):
            try:
                out_dirs = [ln.strip() for ln in str(output_dirs_text or "").splitlines() if ln.strip()]
                min_bytes = int(float(min_mb or 0) * 1024 * 1024)
                data = _bq_mgr.create_queue(
                    source_folder=source_folder,
                    output_dirs=out_dirs,
                    chunk_size=int(chunk_size),
                    target_stage=_bq_stage_from_ui(stage_label),
                    name=name or "",
                    only_pending=True,
                    recursive=bool(recursive),
                    min_bytes=min_bytes,
                    sort_mode=str(sort_mode or "name"),
                    validate=True,
                )
                ch, paths, summary = _bq_mgr.get_next_chunk(data["id"])
                data = _bq_mgr.load(data["id"])
                packed = list(_bq_pack(data))
                packed[4] = _bq_mgr.format_report(data) + "\n\n" + summary
                return tuple(packed)
            except Exception as e:
                return _bq_err(str(e))

        def bq_refresh(queue_id, dropdown_id):
            qid = (queue_id or dropdown_id or "").strip()
            if not qid:
                return _bq_err("Select or create a queue first.")
            try:
                data = _bq_mgr.refresh_status(qid)
                return _bq_pack(data)
            except Exception as e:
                return _bq_err(str(e), qid)

        def bq_prepare(queue_id, dropdown_id, chunk_pick, link_mode):
            qid = (queue_id or dropdown_id or "").strip()
            if not qid:
                return _bq_err("Select or create a queue first.")
            try:
                cid = int(chunk_pick) if chunk_pick is not None and str(chunk_pick) != "" else None
                folder, msg = _bq_mgr.prepare_chunk_folder(
                    qid, chunk_id=cid, link_mode=str(link_mode or "auto")
                )
                data = _bq_mgr.refresh_status(qid)
                packed = list(_bq_pack(data, folder or ""))
                packed[4] = msg + "\n\n" + _bq_mgr.format_report(data)
                return tuple(packed)
            except Exception as e:
                return _bq_err(str(e), qid)

        def bq_requeue_failed(queue_id, dropdown_id):
            qid = (queue_id or dropdown_id or "").strip()
            if not qid:
                return _bq_err("Select or create a queue first.")
            try:
                data = _bq_mgr.requeue_failed(qid)
                return _bq_pack(data)
            except Exception as e:
                return _bq_err(str(e), qid)

        def bq_rebuild(queue_id, dropdown_id, chunk_size):
            qid = (queue_id or dropdown_id or "").strip()
            if not qid:
                return _bq_err("Select or create a queue first.")
            try:
                data = _bq_mgr.rebuild_chunks(qid, chunk_size=int(chunk_size or 20))
                return _bq_pack(data)
            except Exception as e:
                return _bq_err(str(e), qid)

        def bq_import(batch_dir, source_folder, output_dirs_text, chunk_size, stage_label, name):
            try:
                out_dirs = [ln.strip() for ln in str(output_dirs_text or "").splitlines() if ln.strip()]
                data = _bq_mgr.import_batch_progress(
                    batch_dir=batch_dir,
                    source_folder=source_folder,
                    chunk_size=int(chunk_size or 20),
                    target_stage=_bq_stage_from_ui(stage_label),
                    name=name or "",
                    extra_output_dirs=out_dirs,
                )
                packed = list(_bq_pack(data))
                packed[4] = (
                    f"Imported crash from:\n{batch_dir}\n\n" + _bq_mgr.format_report(data)
                )
                return tuple(packed)
            except Exception as e:
                return _bq_err(str(e))

        def _bq_open(path):
            folder = str(path)
            os.makedirs(folder, exist_ok=True)
            try:
                if os.name == "nt":
                    os.startfile(folder)  # type: ignore[attr-defined]
                else:
                    import subprocess as _sp
                    _sp.Popen(["xdg-open", folder])
                return f"Opened: {folder}"
            except Exception as e:
                return f"Path: {folder}\n(Could not open: {e})"

        def bq_open_root():
            return _bq_open(_bq_mgr.queue_root)

        def bq_open_active(queue_id, dropdown_id):
            qid = (queue_id or dropdown_id or "").strip()
            if not qid:
                return "No active queue."
            return _bq_open(_bq_mgr._qdir(qid))

        def bq_open_work(work_folder):
            if not work_folder or not os.path.isdir(str(work_folder)):
                return "No prepared work folder yet — click Prepare chunk first."
            return _bq_open(work_folder)

        def bq_select(dropdown_id):
            qid = (dropdown_id or "").strip()
            if not qid:
                return (
                    gr.update(choices=[], value=None),
                    "",
                    '<div style="padding:12px;color:#6c757d">No queue selected.</div>',
                    "",
                    "",
                )
            try:
                data = _bq_mgr.refresh_status(qid)
                return (
                    gr.update(choices=_bq_chunk_choices(data), value=data.get("current_chunk")),
                    qid,
                    _bq_mgr.format_html_report(data),
                    _bq_mgr.format_report(data),
                    "",
                )
            except Exception as e:
                return gr.update(), qid, f'<div style="padding:12px;color:#f87171">❌ {e}</div>', f"❌ {e}", ""

        def bq_push_to_flashvsr(work_folder):
            """Copy prepared folder into FlashVSR Batch folder path textbox."""
            if not work_folder or not os.path.isdir(str(work_folder)):
                return gr.update(), "No prepared work folder — Prepare a chunk first."
            return str(work_folder), f"Pushed to FlashVSR Batch folder path:\n{work_folder}"

        _bq_out = [
            bq_queue_dropdown,
            bq_chunk_pick,
            bq_active_id,
            bq_status_html,
            bq_console,
            bq_work_folder,
        ]

        bq_create_btn.click(
            fn=bq_create,
            inputs=[
                bq_source_folder,
                bq_output_dirs,
                bq_chunk_size,
                bq_target_stage,
                bq_name,
                bq_recursive,
                bq_min_mb,
                bq_sort_mode,
            ],
            outputs=_bq_out,
        )
        bq_refresh_btn.click(
            fn=bq_refresh,
            inputs=[bq_active_id, bq_queue_dropdown],
            outputs=_bq_out,
        )
        bq_prepare_btn.click(
            fn=bq_prepare,
            inputs=[bq_active_id, bq_queue_dropdown, bq_chunk_pick, bq_link_mode],
            outputs=_bq_out,
        )
        bq_requeue_fail_btn.click(
            fn=bq_requeue_failed,
            inputs=[bq_active_id, bq_queue_dropdown],
            outputs=_bq_out,
        )
        bq_rebuild_btn.click(
            fn=bq_rebuild,
            inputs=[bq_active_id, bq_queue_dropdown, bq_chunk_size],
            outputs=_bq_out,
        )
        bq_import_btn.click(
            fn=bq_import,
            inputs=[
                bq_import_batch_dir,
                bq_source_folder,
                bq_output_dirs,
                bq_chunk_size,
                bq_target_stage,
                bq_name,
            ],
            outputs=_bq_out,
        )
        bq_reload_list_btn.click(fn=lambda: bq_reload_list(), inputs=[], outputs=[bq_queue_dropdown])
        bq_open_btn.click(fn=bq_open_root, inputs=[], outputs=[bq_console])
        bq_open_active_btn.click(
            fn=bq_open_active,
            inputs=[bq_active_id, bq_queue_dropdown],
            outputs=[bq_console],
        )
        bq_open_work_btn.click(fn=bq_open_work, inputs=[bq_work_folder], outputs=[bq_console])
        bq_push_batch_btn.click(
            fn=bq_push_to_flashvsr,
            inputs=[bq_work_folder],
            outputs=[batch_folder_path, bq_console],
        )
        bq_queue_dropdown.change(
            fn=bq_select,
            inputs=[bq_queue_dropdown],
            outputs=[bq_chunk_pick, bq_active_id, bq_status_html, bq_console, bq_work_folder],
        )

        # Theme Selector
        with gr.Accordion("⚙️ Settings", open=False):
            gr.Markdown("### UI Theme")
            with gr.Row():
                theme_dropdown = gr.Dropdown(
                    choices=ALL_THEME_NAMES,
                    value=current_theme,
                    label="Select Theme",
                    info=TIPS["theme"],
                    scale=3
                )
                theme_status = gr.Textbox(label="Status", scale=2, interactive=False, show_label=False)
            
            custom_theme_input = gr.Textbox(
                label="Custom Theme (Hugging Face Space)",
                placeholder="e.g., username/theme-name",
                value=custom_theme_string,
                info=TIPS["custom_theme"],
                visible=(current_theme == "Custom")
            )
            
            apply_theme_btn = gr.Button("Apply Theme", size="sm", variant="primary")

            gr.Markdown("### Process naming (2 steps only)")
            gr.HTML(
                value=(
                    "<div style='padding:10px;background:#0f1419;border:1px solid #2d3748;border-radius:8px;"
                    "color:#e2e8f0;font-size:0.9em;line-height:1.55;'>"
                    "<b style='color:#7dd3fc;'>Step 1 — Upscale</b><br>"
                    "Name: <code style='color:#86efac;'>&lt;original&gt;_&lt;1080p|2K|4K&gt;_&lt;9x16&gt;_Upscaled.mp4</code><br>"
                    "Example: <code>myclip_4K_9x16_Upscaled.mp4</code><br><br>"
                    "<b style='color:#7dd3fc;'>Step 2 — Interp + Export (same Toolbox pass)</b><br>"
                    "Name: <code style='color:#fbbf24;'>&lt;step1&gt;_&lt;30fps&gt;.mp4</code><br>"
                    "Example: <code>myclip_4K_9x16_Upscaled_30fps.mp4</code><br>"
                    "After success, the Step‑1 file is moved to <code>…\\Ready for Toolbox\\Bin\\</code> "
                    "so you can delete intermediates when done."
                    "</div>"
                )
            )
            naming_mode_dropdown = gr.Dropdown(
                label="Legacy name style (optional)",
                choices=[
                    ("Readable 2-step names (default)", "both"),
                    ("Toolbox legacy", "toolbox"),
                    ("VSR legacy", "vsr"),
                ],
                value=str(config.get("naming_mode", "both")),
                info=TIPS["naming_mode"],
            )
            naming_mode_status = gr.Textbox(label="Naming status", interactive=False, show_label=False)
            apply_naming_mode_btn = gr.Button("Apply name style", size="sm", variant="primary")

            gr.Markdown("### Folders (selectable — your current D: paths are fine)")
            gr.HTML(value=workflow_paths_html(config))
            _wp = get_workflow_paths(config)
            step1_watch = gr.Textbox(
                label="Intake / watch (new downloads)",
                value=_wp["batch_watch_folder"],
                info="Where new files land · queues scan on Start / Resume",
            )
            step2_archive = gr.Textbox(
                label="Originals archive (pre-upscale sources for pairing)",
                value=_wp["batch_source_archive_dir"],
                info="Original sources moved here after upscale",
            )
            step3_upscale = gr.Textbox(
                label="Step 1 output — upscaled VIDEOS (Ready for Toolbox)",
                value=_wp["batch_upscale_handoff_dir"],
                info="Upscaled videos land here. After Step 2 they go into Bin\\ under this folder.",
            )
            step4_images = gr.Textbox(
                label="Step 1 output — upscaled IMAGES",
                value=_wp["img_upscale_handoff_dir"],
                info="Images skip toolbox interp · usually Ready for CIV\\images",
            )
            step5_inbox = gr.Textbox(
                label="Step 2 input — Toolbox inbox (usually = Ready for Toolbox)",
                value=_wp["tb_inbox_folder"],
                info="Toolbox reads Step‑1 videos from here",
            )
            step6_final = gr.Textbox(
                label="Step 2 output — Final / Ready for CIV",
                value=_wp["toolbox_output_dir"],
                info="Interp+export finals with _##fps in the name",
            )
            gt_settings_before = gr.Textbox(
                label="Group Therapy — Before (originals, flat folder)",
                value=str(config.get("gt_before_dir") or _wp["batch_source_archive_dir"]),
                info="Originals stay in this folder with _PID_xxxxxxxx at the end of the filename (and Title metadata). Same id as After. No per-song subfolders.",
            )
            gt_settings_after = gr.Textbox(
                label="Group Therapy — After (finals, flat folder)",
                value=str(config.get("gt_after_dir") or _wp["toolbox_output_dir"]),
                info="Finals stay in this folder with _PID_xxxxxxxx at the end of the filename (and Title metadata). Same id as Before. No per-song subfolders.",
            )
            workflow_status = gr.Textbox(label="Pipeline status", interactive=False, show_label=True)
            with gr.Row():
                apply_workflow_btn = gr.Button("💾 Save all pipeline folders", size="sm", variant="primary")
                reset_workflow_btn = gr.Button("Reset steps to defaults", size="sm", variant="secondary")
                refresh_workflow_map_btn = gr.Button("🔄 Refresh map", size="sm")

            # Back-compat aliases used by older event wiring (map to step fields)
            output_dir_input = step3_upscale
            toolbox_output_dir_input = step6_final
            output_dir_status = workflow_status
            toolbox_output_dir_status = workflow_status
        
        def toggle_custom_input(theme_name):
            return gr.update(visible=(theme_name == "Custom"))
        
        theme_dropdown.change(
            fn=toggle_custom_input,
            inputs=[theme_dropdown],
            outputs=[custom_theme_input]
        )
        
        def apply_theme(theme_name, custom_theme):
            config = load_config()
            config["theme"] = theme_name
            if theme_name == "Custom":
                if not custom_theme or not custom_theme.strip():
                    return "⚠️ Please enter a custom theme string (e.g., username/theme-name)"
                config["custom_theme"] = custom_theme.strip()
                save_config(config)
                return f"✅ Custom theme '{custom_theme}' saved! Restart and Refresh the page to apply."
            else:
                config["custom_theme"] = ""
                save_config(config)
                return f"✅ Theme '{theme_name}' saved! Restart and Refresh the page to apply."
        
        apply_theme_btn.click(
            fn=apply_theme,
            inputs=[theme_dropdown, custom_theme_input],
            outputs=[theme_status]
        )

        def apply_naming_mode(mode):
            mode = str(mode or "both").strip().lower()
            if mode not in ("toolbox", "vsr", "both"):
                return "⚠️ Choose toolbox, vsr, or both."
            cfg = load_config()
            cfg["naming_mode"] = mode
            save_config(cfg)
            examples = {
                "toolbox": "myclip_upscaled_x4.mp4",
                "vsr": "UpScale4K_myclip_S_I.mp4",
                "both": "UpScale4K_myclip_upscaled_x4_S_I.mp4",
            }
            return f"✅ Naming mode set to '{mode}'. Example: {examples[mode]}"

        apply_naming_mode_btn.click(
            fn=apply_naming_mode,
            inputs=[naming_mode_dropdown],
            outputs=[naming_mode_status],
        )
        
        def _require_abs(path: str, label: str):
            p = str(path or "").strip()
            if not p:
                return None, f"⚠️ {label}: empty path"
            p = os.path.normpath(p)
            if not os.path.isabs(p):
                return None, f"⚠️ {label}: use an absolute path (e.g. D:\\OUTPUTS\\...)"
            try:
                os.makedirs(p, exist_ok=True)
            except OSError as e:
                return None, f"❌ {label}: cannot create folder — {e}"
            return p, None

        def apply_workflow_folders(s1, s2, s3, s4, s5, s6, gt_before="", gt_after=""):
            """Save all six pipeline steps + Group Therapy Before/After folders (flat PID pairing)."""
            cfg = load_config()
            mapping = [
                ("batch_watch_folder", s1, "Step 1"),
                ("batch_source_archive_dir", s2, "Step 2"),
                ("batch_upscale_handoff_dir", s3, "Step 3"),
                ("img_upscale_handoff_dir", s4, "Step 4"),
                ("tb_inbox_folder", s5, "Step 5"),
                ("toolbox_output_dir", s6, "Step 6"),
                ("gt_before_dir", gt_before or s2, "Group Therapy Before"),
                ("gt_after_dir", gt_after or s6, "Group Therapy After"),
            ]
            saved = []
            for key, raw, label in mapping:
                path, err = _require_abs(raw, label)
                if err:
                    return workflow_paths_html(cfg), err
                cfg[key] = path
                saved.append(f"{label}→{path}")
            # Step 3 is THE upscale save location (single + queue + autosave)
            cfg["output_dir"] = cfg["batch_upscale_handoff_dir"]
            save_config(cfg)
            _apply_toolbox_output_dir()
            ensure_workflow_dirs()
            msg = (
                "✅ Pipeline folders saved. Step 3 (videos) = Ready for Toolbox path. "
                "Restart / refresh if a queue was already mid-run.\n"
                + "\n".join(saved)
            )
            return workflow_paths_html(cfg), msg

        def reset_workflow_folders():
            cfg = load_config()
            for key, default in WORKFLOW_DEFAULTS.items():
                cfg[key] = default
            cfg["output_dir"] = WORKFLOW_DEFAULTS["batch_upscale_handoff_dir"]
            cfg["gt_before_dir"] = WORKFLOW_DEFAULTS["batch_source_archive_dir"]
            cfg["gt_after_dir"] = WORKFLOW_DEFAULTS["toolbox_output_dir"]
            save_config(cfg)
            _apply_toolbox_output_dir()
            ensure_workflow_dirs()
            wp = get_workflow_paths(cfg)
            return (
                wp["batch_watch_folder"],
                wp["batch_source_archive_dir"],
                wp["batch_upscale_handoff_dir"],
                wp["img_upscale_handoff_dir"],
                wp["tb_inbox_folder"],
                wp["toolbox_output_dir"],
                cfg["gt_before_dir"],
                cfg["gt_after_dir"],
                workflow_paths_html(cfg),
                "✅ Reset all steps to FAFO defaults (Step 3 = Ready for Toolbox).",
            )

        def refresh_workflow_map():
            return workflow_paths_html()

        apply_workflow_btn.click(
            fn=apply_workflow_folders,
            inputs=[
                step1_watch, step2_archive, step3_upscale, step4_images, step5_inbox, step6_final,
                gt_settings_before, gt_settings_after,
            ],
            outputs=[workflow_map, workflow_status],
        )
        reset_workflow_btn.click(
            fn=reset_workflow_folders,
            inputs=[],
            outputs=[
                step1_watch,
                step2_archive,
                step3_upscale,
                step4_images,
                step5_inbox,
                step6_final,
                gt_settings_before,
                gt_settings_after,
                workflow_map,
                workflow_status,
            ],
        )
        refresh_workflow_map_btn.click(
            fn=refresh_workflow_map,
            inputs=[],
            outputs=[workflow_map],
        )

        # Footer with author credits
        footer_html = """
        <div style="text-align: center; padding: 10px; margin-top: 20px; font-family: sans-serif;">
            <hr style="border: 0; height: 1px; background: #333; margin-bottom: 10px;">
            <h2 style="margin-bottom: 5px;">FlashVSR: Efficient & High-Quality Video Super-Resolution</h2>
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; font-size: 0.8em; flex-wrap: wrap;">
                <!-- GitHub Badge -->
                <a href="https://github.com/OpenImagingLab/FlashVSR" target="_blank" style="text-decoration: none; display: inline-flex; border-radius: 4px; overflow: hidden;">
                    <span style="background-color: #555; color: white; padding: 4px 8px;">⭐ GitHub</span>
                    <span style="background-color: #24292e; color: white; padding: 4px 8px;">Repository</span>
                </a>
                <!-- Project Page Badge -->
                <a href="http://zhuang2002.github.io/FlashVSR" target="_blank" style="text-decoration: none; display: inline-flex; border-radius: 4px; overflow: hidden;">
                    <span style="background-color: #555; color: white; padding: 4px 8px;">Project</span>
                    <span style="background-color: #4c1; color: white; padding: 4px 8px;">Page</span>
                </a>
                <!-- Hugging Face Model Badge -->
                <a href="https://huggingface.co/JunhaoZhuang/FlashVSR" target="_blank" style="text-decoration: none; display: inline-flex; border-radius: 4px; overflow: hidden;">
                    <span style="background-color: #555; color: white; padding: 4px 8px;">🤗 Hugging Face</span>
                    <span style="background-color: #3b82f6; color: white; padding: 4px 8px;">Model</span>
                </a>
                <!-- Hugging Face Dataset Badge -->
                <a href="https://huggingface.co/datasets/JunhaoZhuang/VSR-120K" target="_blank" style="text-decoration: none; display: inline-flex; border-radius: 4px; overflow: hidden;">
                    <span style="background-color: #555; color: white; padding: 4px 8px;">🤗 Hugging Face</span>
                    <span style="background-color: #ff9a00; color: white; padding: 4px 8px;">Dataset</span>
                </a>
                <!-- arXiv Badge -->
                <a href="https://arxiv.org/abs/2510.12747" target="_blank" style="text-decoration: none; display: inline-flex; border-radius: 4px; overflow: hidden;">
                    <span style="background-color: #555; color: white; padding: 4px 8px;">arXiv</span>
                    <span style="background-color: #b31b1b; color: white; padding: 4px 8px;">2510.12747</span>
                </a>
            </div>
            <p style="margin-top: 10px; font-size: 0.9em; color: #888;">
                Thank you for using FlashVSR! Please visit the project page and consider giving the repository a ⭐ on GitHub.
            </p>
        </div>
        """
        gr.HTML(footer_html)
        
    return demo

if __name__ == "__main__":
    os.makedirs(get_output_dir(), exist_ok=True)
    
    # Check user preference for clearing temp on start
    config = load_config()
    if config.get("clear_temp_on_start", False):
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            log("Temp files cleared on startup.", message_type="info")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Model download now happens on-demand when user starts processing
    # This allows downloading only the version they select (v1.0 or v1.1)
    log("FlashVSR+ WebUI starting...", message_type="info")
    log("Models will be downloaded automatically when you start processing.", message_type="info")
    
    ui = create_ui()
    allowed_paths = get_gradio_allowed_paths()
    log(f"Gradio allowed_paths: {len(allowed_paths)} location(s) (includes toolbox/output drives)", message_type="info")
    launch_kwargs = {"share": False, "allowed_paths": allowed_paths}
    if args.listen:
        launch_kwargs["server_name"] = "0.0.0.0"
        launch_kwargs["server_port"] = args.port
        ui.queue().launch(**launch_kwargs)
    else:
        launch_kwargs["server_port"] = args.port
        ui.queue().launch(**launch_kwargs)