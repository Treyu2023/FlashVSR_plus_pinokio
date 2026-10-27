"""Live session board for the FlashVSR Gradio UI.

Color block bars for the queue, the current batch, and the step in progress.
Accumulates this-process statistics, watches for a WDDM-style stall, and
stores named job profiles (tile, chunk, Group Therapy, quality) separate
from the folder paths.
"""
from __future__ import annotations

import ctypes
import html
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

QUEUE_ORDER = (
    ("group", "Group"),
    ("video", "Video"),
    ("image", "Image"),
    ("toolbox", "Toolbox"),
)
STAGE_ORDER = ("upscale", "rife1", "rife2", "export")
STAGE_LABEL = {
    "upscale": "Upscale",
    "rife1": "RIFE 1",
    "rife2": "RIFE 2",
    "export": "Export",
}
PROFILE_KEYS = (
    "mode",
    "model_version",
    "scale",
    "tile_size",
    "tile_overlap",
    "tiled_dit",
    "tiled_vae",
    "unload_dit",
    "enable_chunks",
    "chunk_duration",
    "quality",
    "sparse_ratio",
    "local_range",
    "kv_ratio",
    "attention_mode",
    "dtype",
    "color_fix",
    "batch_resize_preset",
    "gt_group_size",
    "gt_rife_mode",
    "gt_rife_quality",
    "gt_rife_streaming",
    "gt_export_quality",
    "gt_export_max_width",
    "gt_do_upscale",
    "gt_do_export",
)

_SAMPLE_CACHE: Dict[str, Any] = {"t": 0.0, "data": None}
_IO_PREV: Dict[str, Any] = {"t": 0.0, "read": 0, "write": 0}
_STATS_LOCK = threading.Lock()


def _now(now: Optional[float]) -> float:
    return time.time() if now is None else float(now)


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def _counts(items: List[dict]) -> Dict[str, int]:
    c = {"total": len(items), "pending": 0, "running": 0, "done": 0, "failed": 0}
    for it in items:
        st = str(it.get("status") or "pending")
        if st in c:
            c[st] += 1
        else:
            c["pending"] += 1
    return c


def _load_queues(app_dir: Path) -> Dict[str, dict]:
    out = {}
    root = app_dir / "outputs"
    for name, _label in QUEUE_ORDER:
        data = _read_json(root / f"work_queue_{name}" / "queue.json") or {}
        items = data.get("items") if isinstance(data.get("items"), list) else []
        out[name] = {"data": data, "items": items, "counts": _counts(items)}
    return out


def _active_name(app_dir: Path, queues: Dict[str, dict]) -> str:
    lock = _read_json(app_dir / "outputs" / "ACTIVE_QUEUE.lock") or {}
    name = str(lock.get("queue") or "")
    if name in queues:
        return name
    best = "group"
    best_n = -1
    for key, _label in QUEUE_ORDER:
        pending = queues[key]["counts"]["pending"] + queues[key]["counts"]["running"]
        if pending > best_n:
            best = key
            best_n = pending
    return best


def _heartbeat() -> dict:
    try:
        from src.busy_heartbeat import snapshot
        data = snapshot()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _commit() -> Dict[str, float]:
    """Windows commit charge. Empty dict if the call is unavailable."""
    if os.name != "nt":
        return {}
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return {}
    except Exception:
        return {}
    limit = float(stat.ullTotalPageFile or 0)
    avail = float(stat.ullAvailPageFile or 0)
    used = max(0.0, limit - avail)
    return {
        "phys_total_gb": stat.ullTotalPhys / (1024 ** 3),
        "phys_avail_gb": stat.ullAvailPhys / (1024 ** 3),
        "commit_limit_gb": limit / (1024 ** 3),
        "commit_used_gb": used / (1024 ** 3),
    }


def _query_gpu() -> Dict[str, float]:
    import subprocess
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,power.draw,power.limit,"
        "temperature.gpu,memory.used,memory.total,clocks.mem,clocks.max.mem",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        raw = subprocess.check_output(cmd, encoding="utf-8", timeout=1.2, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    parts = [p.strip() for p in raw.strip().split(",")]
    if len(parts) < 9:
        return {}
    keys = (
        "gpu_util", "mem_util", "power_w", "power_limit_w", "temp_c",
        "vram_used_mib", "vram_total_mib", "mem_clock", "mem_clock_max",
    )
    out: Dict[str, float] = {}
    for key, part in zip(keys, parts):
        try:
            out[key] = float(part)
        except ValueError:
            continue
    return out


def sample_machine(now: Optional[float] = None) -> dict:
    """GPU, commit, and disk-byte rate. Cached for 2 seconds."""
    now_v = _now(now)
    cached = _SAMPLE_CACHE.get("data")
    if cached and now_v - float(_SAMPLE_CACHE.get("t") or 0) < 2.0:
        return dict(cached)
    sample = _query_gpu()
    sample.update(_commit())
    try:
        import psutil
        io = psutil.disk_io_counters()
        if io is not None:
            prev_t = float(_IO_PREV.get("t") or 0)
            if prev_t and now_v > prev_t:
                dt = now_v - prev_t
                sample["disk_mb_s"] = (
                    (io.read_bytes - int(_IO_PREV.get("read") or 0))
                    + (io.write_bytes - int(_IO_PREV.get("write") or 0))
                ) / dt / (1024 ** 2)
            _IO_PREV["t"] = now_v
            _IO_PREV["read"] = io.read_bytes
            _IO_PREV["write"] = io.write_bytes
        ram = psutil.virtual_memory()
        sample.setdefault("phys_total_gb", ram.total / (1024 ** 3))
        sample.setdefault("phys_avail_gb", ram.available / (1024 ** 3))
    except Exception:
        pass
    _SAMPLE_CACHE["t"] = now_v
    _SAMPLE_CACHE["data"] = sample
    return dict(sample)


# Group Therapy walks the whole group through one stage, then the next.
# Each file sits in exactly one bucket, so a later color cannot pass an earlier one:
# saved | RIFE done | upscale done | dark still waiting.
# The three stage colors are the defaults. The board options can replace them.
DEFAULT_STAGE_COLORS = {
    "saved": "#86efac",
    "rife": "#c084fc",
    "upscale": "#38bdf8",
}
STAGE_COLOR_CHOICES = (
    ("Blue", "#38bdf8"),
    ("Sky", "#7dd3fc"),
    ("Purple", "#c084fc"),
    ("Violet", "#a78bfa"),
    ("Green", "#86efac"),
    ("Mint", "#34d399"),
    ("Amber", "#fbbf24"),
    ("Orange", "#fb923c"),
    ("Rose", "#fb7185"),
    ("White", "#f8fafc"),
)
_COLOR_HEXES = {hx for _label, hx in STAGE_COLOR_CHOICES}
_STAGE_STAT_SPECS = (
    ("upscale", "Upscale", "gt_upscale_secs", "gt_upscale_ram_mb", "upscale"),
    ("rife", "RIFE 1", "gt_rife1_secs", "gt_rife1_ram_mb", "rife1"),
    ("rife", "RIFE 2", "gt_rife2_secs", "gt_rife2_ram_mb", "rife2"),
    ("saved", "Saved", "gt_export_secs", "gt_export_ram_mb", "export"),
)
_STAGE_OUTPUT_KEYS = ("gt_export", "gt_rife2", "gt_rife1", "gt_upscale")


def _stage_path(item: dict, key: str) -> bool:
    return bool(str(item.get(key) or "").strip())


def stage_mix(items: List[dict]) -> Dict[str, int]:
    """Furthest finished Group Therapy stage for each file.

    Saved beats RIFE, RIFE beats upscale. A file still inside a stage does not
    count until that stage writes its output path. Failed rows stay out of the
    colors so a dead file does not look like progress.
    """
    mix = {"saved": 0, "rife": 0, "upscale": 0, "wait": 0, "failed": 0, "total": 0}
    rows = [it for it in items if isinstance(it, dict)]
    mix["total"] = len(rows)
    for it in rows:
        if str(it.get("status") or "") == "failed":
            mix["failed"] += 1
            continue
        if str(it.get("status") or "") == "done" or _stage_path(it, "gt_export"):
            mix["saved"] += 1
        elif _stage_path(it, "gt_rife1") or _stage_path(it, "gt_rife2"):
            mix["rife"] += 1
        elif _stage_path(it, "gt_upscale"):
            mix["upscale"] += 1
        else:
            mix["wait"] += 1
    return mix


def _share_blocks(mix: Dict[str, int], width: int) -> Dict[str, int]:
    """Largest-remainder block counts. Saved + RIFE + upscale + rest == width."""
    width = max(0, int(width))
    total = int(mix.get("total") or 0)
    keys = ("saved", "rife", "upscale", "rest")
    weights = [
        int(mix.get("saved") or 0),
        int(mix.get("rife") or 0),
        int(mix.get("upscale") or 0),
        max(0, total - int(mix.get("saved") or 0) - int(mix.get("rife") or 0) - int(mix.get("upscale") or 0)),
    ]
    zeros = {k: 0 for k in keys}
    if width <= 0 or total <= 0 or sum(weights) <= 0:
        return zeros
    raw = [width * w / total for w in weights]
    floors = [int(x) for x in raw]
    left = width - sum(floors)
    order = sorted(
        range(len(weights)),
        key=lambda i: (raw[i] - floors[i], weights[i]),
        reverse=True,
    )
    for i in order:
        if left <= 0:
            break
        if weights[i] <= 0:
            continue
        floors[i] += 1
        left -= 1
    if left > 0:
        floors[3] += left
    return dict(zip(keys, floors))


def _options_path(app_dir: Path) -> Path:
    return Path(app_dir) / "outputs" / "session_board_options.json"


def load_stage_colors(app_dir) -> Dict[str, str]:
    """Upscale, RIFE, and saved colors. Unknown values fall back to the defaults."""
    data = _read_json(_options_path(Path(app_dir))) or {}
    out = {}
    for key, default in DEFAULT_STAGE_COLORS.items():
        raw = str(data.get(key) or "").strip().lower()
        out[key] = raw if raw in _COLOR_HEXES else default
    return out


def save_stage_colors(app_dir, upscale: str, rife: str, saved: str) -> Dict[str, str]:
    """Store a palette choice. A value outside the option list is ignored."""
    colors = load_stage_colors(app_dir)
    incoming = {"upscale": upscale, "rife": rife, "saved": saved}
    for key, value in incoming.items():
        raw = str(value or "").strip().lower()
        if raw in _COLOR_HEXES:
            colors[key] = raw
    _write_json(_options_path(Path(app_dir)), colors)
    return colors


def stage_usage_fields(stage: str, started: float, now: float, ram_mb: float) -> Dict[str, float]:
    """Seconds and RAM to write on a queue row when one stage finishes."""
    prefix = {
        "upscale": "gt_upscale",
        "rife1": "gt_rife1",
        "rife2": "gt_rife2",
        "export": "gt_export",
    }.get(stage or "")
    if not prefix:
        return {}
    try:
        secs = round(max(0.0, float(now) - float(started)), 1)
    except (TypeError, ValueError):
        secs = 0.0
    try:
        ram = round(max(0.0, float(ram_mb)), 1)
    except (TypeError, ValueError):
        ram = 0.0
    return {f"{prefix}_secs": secs, f"{prefix}_ram_mb": ram}


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_ram(mb: Optional[float]) -> str:
    if mb is None or mb <= 0:
        return ""
    if mb >= 1024:
        return f"RAM {mb / 1024:.1f} GB"
    return f"RAM {mb:.0f} MB"


def _process_ram_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def stage_lines(item: dict, now: float, live_ram: Optional[float] = None) -> List[tuple]:
    """(color key, plain line) for each stage that has a finished time, plus the open stage."""
    rows = []
    stage_now = str(item.get("gt_stage") or "").strip().lower()
    running = str(item.get("status") or "") == "running"
    try:
        started = float(item.get("gt_stage_started"))
    except (TypeError, ValueError):
        started = None
    for color_key, label, sec_key, ram_key, stage_id in _STAGE_STAT_SPECS:
        secs = _optional_float(item.get(sec_key))
        ram = _optional_float(item.get(ram_key))
        if secs is not None and secs >= 0:
            bits = [label, _fmt_eta(secs)]
            ram_txt = _fmt_ram(ram)
            if ram_txt:
                bits.append(ram_txt)
            rows.append((color_key, " · ".join(bits)))
        elif running and stage_now == stage_id and started is not None:
            elapsed = max(0.0, float(now) - started)
            bits = [label, f"{_fmt_eta(elapsed)} so far"]
            ram_txt = _fmt_ram(live_ram)
            if ram_txt:
                bits.append(ram_txt)
            rows.append((color_key, " · ".join(bits)))
    return rows


def stage_stats_html(
    items: List[dict],
    colors: Dict[str, str],
    now: float,
    group_id: Any = None,
    live_ram: Optional[float] = None,
) -> str:
    """Per-file stage time and RAM, painted with that stage's bar color."""
    rows = [it for it in items if isinstance(it, dict)]
    if group_id not in (None, "", "—"):
        focused = [
            it for it in rows
            if it.get("gt_group") == group_id or str(it.get("gt_group")) == str(group_id)
        ]
        if focused:
            rows = focused
    shown = []
    for it in rows:
        lines = stage_lines(it, now, live_ram)
        if lines:
            shown.append((it, lines))
    if group_id in (None, "", "—"):
        shown = shown[:12]
    if not shown:
        return ""
    blocks = []
    for it, lines in shown:
        name = Path(str(it.get("path") or "")).name or "file"
        body = "".join(
            f"<div style='color:{colors.get(key, '#e2e8f0')};font-size:14px;font-weight:700'>"
            f"{_esc(text)}</div>"
            for key, text in lines
        )
        blocks.append(
            "<div style='margin-top:6px'>"
            f"<div style='color:#f8fafc;font-size:15px;font-weight:800'>{_esc(name)}</div>"
            f"{body}</div>"
        )
    return (
        "<div style='margin-top:8px;padding:8px 10px;border-radius:8px;"
        "background:#020617;border:1px solid #1e293b'>"
        "<div style='color:#94a3b8;font-size:12px;font-weight:700;letter-spacing:.04em'>"
        "STAGE TIME AND RAM</div>"
        + "".join(blocks)
        + "</div>"
    )


def stage_bar(
    mix: Dict[str, int],
    width: int = 40,
    fill: str = "█",
    empty: str = "░",
    colors: Optional[Dict[str, str]] = None,
) -> str:
    """Stacked file bar. Left to right: saved, RIFE, upscale, dark waiting."""
    width = max(8, int(width))
    total = int(mix.get("total") or 0)
    palette = dict(DEFAULT_STAGE_COLORS)
    if colors:
        palette.update({k: v for k, v in colors.items() if k in palette and v})
    if total <= 0:
        return f"<span style='color:#334155'>{empty * width}</span>"
    blocks = _share_blocks(mix, width)
    parts = []
    for key in ("saved", "rife", "upscale"):
        n = int(blocks.get(key) or 0)
        if n:
            parts.append(f"<span style='color:{palette[key]}'>{fill * n}</span>")
    rest = int(blocks.get("rest") or 0)
    if rest:
        parts.append(f"<span style='color:#334155'>{empty * rest}</span>")
    return "".join(parts)


def stage_pct(mix: Dict[str, int]) -> str:
    """Overall wave percent. Upscale is 1/3, RIFE is 2/3, saved is the whole file."""
    total = int(mix.get("total") or 0)
    if total <= 0:
        return "—"
    score = int(mix.get("saved") or 0) * 3 + int(mix.get("rife") or 0) * 2 + int(mix.get("upscale") or 0)
    return f"{100.0 * score / (total * 3):.0f}%"


def _stage_queue(data: dict, items: List[dict]) -> bool:
    """True when this queue is Group Therapy and stage outputs can move the bar."""
    data = data or {}
    if data.get("gt_stages") or data.get("gt_group_size") or data.get("gt_current_stage"):
        return True
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("gt_stage"):
            return True
        if any(_stage_path(it, key) for key in _STAGE_OUTPUT_KEYS):
            return True
    return False


def glyph_bar(done: int, total: int, width: int = 40, fill: str = "█", tip: str = "▓", empty: str = "░") -> str:
    """Colored block bar. Green is finished, cyan is the piece in progress."""
    width = max(8, int(width))
    total_i = max(0, int(total or 0))
    done_i = max(0, int(done or 0))
    if total_i <= 0:
        return f"<span style='color:#334155'>{empty * width}</span>"
    done_i = min(done_i, total_i)
    filled = int(round(width * done_i / total_i))
    filled = min(width, max(0, filled))
    working = done_i < total_i
    if working and filled >= width:
        filled = width - 1
    tip_n = 1 if working else 0
    rest = max(0, width - filled - tip_n)
    parts = []
    if filled:
        parts.append(f"<span style='color:#86efac'>{fill * filled}</span>")
    if tip_n:
        parts.append(f"<span style='color:#7dd3fc'>{tip}</span>")
    if rest:
        parts.append(f"<span style='color:#334155'>{empty * rest}</span>")
    return "".join(parts)


def _spark(values: List[float]) -> str:
    glyphs = "▁▂▃▄▅▆▇█"
    if not values:
        return "<span style='color:#334155'>········</span>"
    peak = max(values) or 1.0
    med = sorted(values)[len(values) // 2] or peak
    bits = []
    for value in values[-16:]:
        idx = int(round((len(glyphs) - 1) * value / peak))
        idx = max(0, min(len(glyphs) - 1, idx))
        if value >= med * 3:
            color = "#fca5a5"
        elif value >= med * 2:
            color = "#fbbf24"
        else:
            color = "#86efac"
        bits.append(f"<span style='color:{color}'>{glyphs[idx]}</span>")
    return "".join(bits)


def _fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m"


def _avg_rows(stats: dict, session_done: int, elapsed_s: float):
    """Session row, then the all-time total row. Both are always returned."""
    stats = stats or {}
    file_times = [float(x) for x in (stats.get("file_seconds") or []) if float(x) > 1]
    sess_times = [float(x) for x in (stats.get("session_file_seconds") or []) if float(x) > 1]
    avg_file = (sum(file_times) / len(file_times)) if file_times else None
    sess_avg = (sum(sess_times) / len(sess_times)) if sess_times else None
    busy = sum(sess_times) if sess_times else 0.0
    total_n = int(stats.get("lifetime_done") or 0) + int(session_done)
    session_row = (
        f"avg {_fmt_eta(sess_avg)} · session {int(session_done)} done"
        f" · {len(sess_times)} timed · busy {_fmt_eta(busy)} / {_fmt_eta(elapsed_s)}"
    )
    total_row = f"avg {_fmt_eta(avg_file)} · all-time {total_n} done · {len(file_times)} timed"
    return session_row, total_row, avg_file, sess_avg, sess_times, file_times


def _avg_block(
    session_row: str,
    total_row: str,
    session_extra: str = "",
    total_extra: str = "",
    history_html_block: str = "",
) -> str:
    """Session line, running totals, then parked history when this board has it."""
    return (
        "<div style='margin:8px 0;padding:8px 10px;border-radius:8px;"
        "border:1px solid #334155;background:#0b1220;font-size:15px;line-height:1.55;'>"
        f"<div><b style='color:#7dd3fc'>Session</b> &nbsp; {session_row}</div>"
        f"<div style='margin-top:4px'>{session_extra}</div>"
        f"<div style='margin-top:8px'><b style='color:#e2e8f0'>Total</b> &nbsp; {total_row}</div>"
        f"<div style='margin-top:4px'>{total_extra}</div>"
        f"{history_html_block}"
        "</div>"
    )


def _stage_from_section(section: str, stages: List[str]) -> str:
    """Map a heartbeat section onto a Group Therapy stage key. Chunks are not a stage."""
    low = (section or "").strip().lower()
    if not low or low.startswith(("chunk", "loading", "encoding", "tile", "dit", "decode")):
        return ""
    if "rife" in low and ("pass 2" in low or "rife 2" in low or "rife2" in low):
        return "rife2" if "rife2" in stages else ""
    if "rife" in low:
        return "rife1" if "rife1" in stages else ""
    if "export" in low:
        return "export" if "export" in stages else ""
    if "upscale" in low:
        return "upscale" if "upscale" in stages else ""
    return ""


def _pick_stage(stages: List[str], *candidates: str) -> str:
    """Furthest real stage among the queue label, the running row, and the heartbeat."""
    best = ""
    best_i = -1
    for raw in candidates:
        stage = (raw or "").strip().lower()
        if stage not in stages:
            continue
        rank = stages.index(stage)
        if rank >= best_i:
            best, best_i = stage, rank
    return best


def _batch_progress(data: dict, items: List[dict], section: str = "", name: str = "") -> Dict[str, Any]:
    size = int(data.get("gt_group_size") or 0)
    gid = data.get("gt_current_group")
    stages = [s.strip() for s in str(data.get("gt_stages") or "upscale,rife1,rife2,export").split(",") if s.strip()]
    stages = stages or list(STAGE_ORDER)

    def _same_group(it: dict) -> bool:
        if gid in (None, "", "—"):
            return False
        got = it.get("gt_group")
        if got == gid:
            return True
        return str(got) == str(gid)

    members = [it for it in items if _same_group(it)]
    done = sum(1 for it in members if it.get("status") == "done")
    running = next((it for it in members if it.get("status") == "running"), None)
    if running is None:
        running = next((it for it in items if it.get("status") == "running"), None)
    stage = _pick_stage(
        stages,
        str(data.get("gt_current_stage") or ""),
        str((running or {}).get("gt_stage") or ""),
        _stage_from_section(section, stages),
    )
    stage_index = stages.index(stage) if stage in stages else -1
    current_name = Path(str((running or {}).get("path") or "")).name
    file_pos, file_count = file_place(members, stage, name or current_name)
    return {
        "size": size or len(members),
        "group": gid if gid not in (None, "") else "—",
        "stage": stage,
        "stage_label": STAGE_LABEL.get(stage, stage or "—"),
        "stages": stages,
        "stage_index": stage_index,
        "batch_done": done,
        "batch_total": len(members) or (size or 0),
        "current_name": current_name,
        "mix": stage_mix(members),
        "file_pos": file_pos,
        "file_count": file_count or len(members) or (size or 0),
    }


def _empty_stats(now: float) -> dict:
    return {
        "pid": os.getpid(),
        "started": now,
        "session_done": [],
        "session_failed": [],
        "lifetime_done": 0,
        "last_status": {},
        "file_seconds": [],
        "file_seconds_v": 2,
        "session_file_seconds": [],
        "step_seconds": [],
        "open_file": None,
        "last_step_key": None,
        "last_step_advance": now,
        "active_profile": "",
        "recent_files": [],
        "by_cfg": {},
        "session_by_cfg": {},
        "rollup": _empty_rollup(),
        "session_rollup": _empty_rollup(),
    }


def _stats_path(app_dir: Path) -> Path:
    return app_dir / "outputs" / "session_board_stats.json"


# v3 stats: one record per section, keyed from the ACTIVE_QUEUE.lock queue name.
# v4 (FV-UI) keeps the v3 file layout ("v": 3, same buckets) so a v3 reader still
# reads it; the per-file stats sit under the extra top-level key "v4", which v3 ignores.
STATS_V = 4
STATS_FILE_V = 3
V4_KEYS = ("recent_files", "by_cfg", "session_by_cfg")
RECENT_MAX = 20
BUCKETS = ("group", "upscale", "image", "toolbox")
LOCK_TO_BUCKET = {"group": "group", "video": "upscale", "image": "image", "toolbox": "toolbox"}
ARCHIVE_MAX = 50
SESSION_ETA_MIN = 3  # ETA uses the session avg once this many session files exist
ALLTIME_KEYS = (
    "lifetime_done", "file_seconds", "file_seconds_v", "step_seconds",
    "by_cfg", "recent_files", "rollup",
)


def _archive_path(app_dir: Path) -> Path:
    return app_dir / "outputs" / "session_board_stats_archive.json"


def _read_archive(app_dir: Path, store: dict) -> list:
    """Reset archive from its own file, plus any legacy in-file archive (older, first)."""
    data = _read_json(_archive_path(app_dir)) or {}
    arch = data.get("archive") if isinstance(data.get("archive"), list) else []
    return (list(store.get("legacy_archive") or []) + arch)[-ARCHIVE_MAX:]


def _write_archive(app_dir: Path, archive: list) -> None:
    _write_json(_archive_path(app_dir), {"archive": archive[-ARCHIVE_MAX:]})


def _bak_path(app_dir: Path) -> Path:
    return app_dir / "outputs" / "session_board_stats.v2.bak.json"


def _bak_v3_path(app_dir: Path) -> Path:
    return app_dir / "outputs" / "session_board_stats.v3.bak.json"


def _lock_bucket(app_dir: Path) -> Optional[str]:
    """Bucket of the queue that holds the lock. None when no lock: that tick is not recorded."""
    lock = _read_json(app_dir / "outputs" / "ACTIVE_QUEUE.lock") or {}
    return LOCK_TO_BUCKET.get(str(lock.get("queue") or ""))


def _read_store(app_dir: Path) -> dict:
    """v3/v4 store. Flat v2/unversioned record becomes buckets['group']; corrupt/missing is empty.

    A v3 file loads unchanged; v4 extras come from the top-level "v4" key when present.
    """
    raw = _read_json(_stats_path(app_dir)) or {}
    if raw.get("v") == STATS_FILE_V:
        buckets = raw.get("buckets") if isinstance(raw.get("buckets"), dict) else {}
        ext = raw.get("v4") if isinstance(raw.get("v4"), dict) else {}
        ext_b = ext.get("buckets") if isinstance(ext.get("buckets"), dict) else {}
        merged = {}
        for k, v in buckets.items():
            if not isinstance(v, dict):
                continue
            rec = dict(v)
            extra = ext_b.get(k) if isinstance(ext_b.get(k), dict) else {}
            for key in V4_KEYS:
                if key in extra:
                    rec[key] = extra[key]
            merged[k] = rec
        buckets = merged
        legacy = raw.get("archive") if isinstance(raw.get("archive"), list) else []
        profile = str(raw.get("active_profile") or "")
    else:
        buckets = {"group": raw} if raw else {}
        legacy = []
        profile = str(raw.get("active_profile") or "")
    buckets = {k: v for k, v in buckets.items() if k in BUCKETS and isinstance(v, dict)}
    return {"v": STATS_V, "buckets": buckets, "active_profile": profile, "legacy_archive": legacy}


def _write_store(app_dir: Path, store: dict) -> None:
    """Before the first v3/v4 write, keep one copy of the old file. Never overwrite a .bak."""
    path = _stats_path(app_dir)
    bak = _bak_path(app_dir)
    bak3 = _bak_v3_path(app_dir)
    if path.exists() and (not bak.exists() or not bak3.exists()):
        old = _read_json(path)
        if not old or old.get("v") != STATS_FILE_V:
            target = bak  # v2/unversioned/corrupt -> v2.bak
        elif "v4" not in old:
            target = bak3  # plain v3 -> v3.bak, once
        else:
            target = None
        if target is not None and not target.exists():
            try:
                target.write_bytes(path.read_bytes())
            except OSError:
                pass
    if store.get("legacy_archive"):
        # One-time move of an in-file archive (3288a69c layout) to its own file.
        _write_archive(app_dir, _read_archive(app_dir, store))
    buckets = store.get("buckets") or {}
    v3_buckets = {k: {f: x for f, x in v.items() if f not in V4_KEYS} for k, v in buckets.items()}
    ext = {k: {f: v[f] for f in V4_KEYS if f in v} for k, v in buckets.items()}
    payload = {
        "v": STATS_FILE_V,
        "buckets": v3_buckets,
        "active_profile": store.get("active_profile") or "",
        "v4": {"stats_v": STATS_V, "buckets": ext},
    }
    text = json.dumps(payload, indent=2)
    try:
        if path.read_text(encoding="utf-8") == text:
            return  # unchanged: no write
    except (OSError, UnicodeError):
        pass
    _write_json(path, payload)


def _bucket_stats(rec: Optional[dict], now: float) -> dict:
    """Same rules as the flat record had: v2 trim, pid rollover, missing bucket is empty."""
    data = dict(rec or {})
    base = _empty_stats(now)
    if not data:
        return base
    if data.get("file_seconds_v") != 2:
        # One-time trim: entries written before the FV-STOP idle-gap fix carry no marker.
        data["file_seconds"] = []
    if int(data.get("pid") or 0) != os.getpid():
        base["lifetime_done"] = int(data.get("lifetime_done") or 0) + len(data.get("session_done") or [])
        base["active_profile"] = str(data.get("active_profile") or "")
        base["file_seconds"] = list(data.get("file_seconds") or [])[-40:]
        base["step_seconds"] = list(data.get("step_seconds") or [])[-40:]
        base["recent_files"] = list(data.get("recent_files") or [])[-RECENT_MAX:]
        base["by_cfg"] = dict(data.get("by_cfg") or {})
        base["rollup"] = _as_rollup(data.get("rollup"))
        return base
    for key in base:
        if key in data:
            base[key] = data[key]
    base["session_done"] = list(base.get("session_done") or [])
    base["session_failed"] = list(base.get("session_failed") or [])
    base["file_seconds"] = list(base.get("file_seconds") or [])
    base["session_file_seconds"] = list(base.get("session_file_seconds") or [])
    base["step_seconds"] = list(base.get("step_seconds") or [])
    base["last_status"] = dict(base.get("last_status") or {})
    base["recent_files"] = list(base.get("recent_files") or [])[-RECENT_MAX:]
    base["by_cfg"] = dict(base.get("by_cfg") or {})
    base["session_by_cfg"] = dict(base.get("session_by_cfg") or {})
    base["rollup"] = _as_rollup(base.get("rollup"))
    base["session_rollup"] = _as_rollup(base.get("session_rollup"))
    return base


def reset_bucket(app_dir, bucket: str, now: Optional[float] = None) -> str:
    """Archive one section's record, then empty it. Undo with undo_reset."""
    app = Path(app_dir)
    if bucket not in BUCKETS:
        return f"Unknown section: {bucket}"
    with _STATS_LOCK:
        store = _read_store(app)
        old = store["buckets"].get(bucket) or {}
        archive = _read_archive(app, store) + [{"bucket": bucket, "ts": _now(now), "record": old}]
        _write_archive(app, archive)
        store["legacy_archive"] = []
        # All-time only: the current session's numbers stay.
        rec = _bucket_stats(old, _now(now))
        rec.update({
            "lifetime_done": 0, "file_seconds": [], "file_seconds_v": 2,
            "step_seconds": [], "by_cfg": {}, "recent_files": [], "rollup": _empty_rollup(),
        })
        store["buckets"][bucket] = rec
        _write_store(app, store)
    return (
        f"Moved {bucket} running totals into History. "
        "History stays until Unlock history clear is on."
    )


def undo_reset(app_dir, bucket: str) -> str:
    """Restore the newest archived record for this section and drop it from the archive."""
    app = Path(app_dir)
    with _STATS_LOCK:
        store = _read_store(app)
        archive = _read_archive(app, store)
        for i in range(len(archive) - 1, -1, -1):
            entry = archive[i]
            if isinstance(entry, dict) and entry.get("bucket") == bucket:
                saved = dict(entry.get("record") or {})
                rec = dict(store["buckets"].get(bucket) or {})
                for key in ALLTIME_KEYS:
                    if key in saved:
                        rec[key] = saved[key]
                store["buckets"][bucket] = rec
                del archive[i]
                _write_archive(app, archive)
                store["legacy_archive"] = []
                _write_store(app, store)
                return f"Restored {bucket} all-time stats from the last reset."
    return f"No reset to undo for {bucket}."


def new_session(app_dir, bucket: str, now: Optional[float] = None) -> str:
    """Start a fresh session for one section. Done files fold into the all-time count."""
    app = Path(app_dir)
    if bucket not in BUCKETS:
        return f"Unknown section: {bucket}"
    with _STATS_LOCK:
        store = _read_store(app)
        rec = _bucket_stats(store["buckets"].get(bucket), _now(now))
        rec["lifetime_done"] = int(rec.get("lifetime_done") or 0) + len(rec.get("session_done") or [])
        rec.update({
            "session_done": [], "session_failed": [], "session_file_seconds": [],
            "session_by_cfg": {}, "session_rollup": _empty_rollup(), "started": _now(now),
        })
        store["buckets"][bucket] = rec
        _write_store(app, store)
    return f"New {bucket} session started."


def clear_history(app_dir, bucket: str, unlocked: bool = False) -> str:
    """Drop parked eras for one section. Refuses unless the options toggle is on."""
    if bucket not in BUCKETS:
        return f"Unknown section: {bucket}"
    if not unlocked:
        return "History is locked. Turn on Unlock history clear in Stats reset, then press Clear history."
    app = Path(app_dir)
    with _STATS_LOCK:
        store = _read_store(app)
        archive = _read_archive(app, store)
        kept = [e for e in archive if not (isinstance(e, dict) and e.get("bucket") == bucket)]
        removed = len(archive) - len(kept)
        _write_archive(app, kept)
        store["legacy_archive"] = []
        _write_store(app, store)
    if not removed:
        return f"No {bucket} history to clear. Running totals were left as they are."
    return f"Cleared {removed} {bucket} history era(s). Running totals were left as they are."


def _load_stats(app_dir: Path, now: float, bucket: str = "group") -> dict:
    return _bucket_stats(_read_store(app_dir)["buckets"].get(bucket), now)


def _add_file_seconds(stats: dict, secs: float) -> None:
    stats["file_seconds"] = (list(stats.get("file_seconds") or []) + [secs])[-40:]
    stats["session_file_seconds"] = (list(stats.get("session_file_seconds") or []) + [secs])[-500:]


def cfg_key(rec: dict) -> Optional[str]:
    """Comparison key for a file record. None when the record has no settings."""
    if not isinstance(rec, dict) or not rec.get("mode"):
        return None
    tile = rec.get("tile_size") if (rec.get("tiled_dit") or rec.get("tiled_vae")) else 0
    return (
        f"{rec.get('mode')}|{rec.get('model_version')}|x{rec.get('scale')}|{rec.get('dtype_str')}|"
        f"{rec.get('attention_mode')}|t{tile}/{rec.get('tile_overlap')}|chunk{rec.get('chunk_s') or 0}"
    )


_ROLLUP_KEYS = (
    "n", "sum_secs", "n_secs",
    "sum_in", "n_in", "sum_out", "n_out", "sum_pct", "n_pct",
    "sum_steps", "n_steps", "sum_step_s", "n_step_s",
    "sum_frames", "n_frames", "sum_spf", "n_spf", "sum_fps", "n_fps",
    "sum_tiles", "n_tiles", "sum_vram", "n_vram",
    "sum_load", "n_load", "sum_proc", "n_proc",
    "sum_px_in", "sum_px_out", "n_px",
)


def _empty_rollup() -> dict:
    return {key: 0.0 for key in _ROLLUP_KEYS}


def _as_rollup(raw) -> dict:
    base = _empty_rollup()
    if isinstance(raw, dict):
        for key in base:
            try:
                base[key] = float(raw.get(key) or 0)
            except (TypeError, ValueError):
                base[key] = 0.0
    return base


def _add_rolls(left, right) -> dict:
    out = _as_rollup(left)
    extra = _as_rollup(right)
    for key in out:
        out[key] += extra[key]
    return out


def _num_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_num(value: Any) -> str:
    num = _num_or_none(value)
    if num is None:
        return ""
    if num == int(num):
        return str(int(num))
    return f"{num:g}"


def _fmt_bytes(num: Optional[float]) -> str:
    if num is None or num <= 0:
        return "—"
    if num < 1024 ** 2:
        return f"{num / 1024:.0f} KB"
    if num < 1024 ** 3:
        return f"{num / 1024 ** 2:.1f} MB"
    return f"{num / 1024 ** 3:.2f} GB"


def _fmt_pct(num: Optional[float]) -> str:
    if num is None:
        return "—"
    sign = "+" if num > 0 else ""
    return f"{sign}{num:.0f}%"


def _file_steps(entry: dict) -> Optional[float]:
    steps = _num_or_none(entry.get("steps"))
    if steps and steps > 0:
        return steps
    tiles = _num_or_none(entry.get("num_tiles"))
    if tiles and tiles > 0:
        return tiles
    return None


def _byte_part(entry: dict) -> dict:
    """Byte-size slice of a rollup for one file. Empty when the file has no sizes."""
    roll = _empty_rollup()
    groups = set()
    in_b = _num_or_none(entry.get("in_bytes"))
    out_b = _num_or_none(entry.get("out_bytes"))
    if (in_b and in_b > 0) or (out_b and out_b > 0):
        groups.add("bytes")
    if in_b and in_b > 0:
        roll["sum_in"] += in_b
        roll["n_in"] += 1
    if out_b and out_b > 0:
        roll["sum_out"] += out_b
        roll["n_out"] += 1
    if in_b and out_b and in_b > 0:
        roll["sum_pct"] += (out_b - in_b) / in_b * 100.0
        roll["n_pct"] += 1
    return {"roll": roll, "groups": groups}


def _contribute(entry: dict, already: set) -> dict:
    """Add this file's stats, skipping groups already folded in."""
    roll = _empty_rollup()
    groups = set()
    if "core" not in already:
        groups.add("core")
        roll["n"] += 1
        secs = _num_or_none(entry.get("secs"))
        if secs and secs > 0:
            roll["sum_secs"] += secs
            roll["n_secs"] += 1
        steps = _file_steps(entry)
        if steps:
            roll["sum_steps"] += steps
            roll["n_steps"] += 1
            if secs and secs > 0:
                roll["sum_step_s"] += secs / steps
                roll["n_step_s"] += 1
        frames = _num_or_none(entry.get("frames"))
        if frames and frames > 0:
            roll["sum_frames"] += frames
            roll["n_frames"] += 1
        spf = _num_or_none(entry.get("s_per_frame"))
        if spf and spf > 0:
            roll["sum_spf"] += spf
            roll["n_spf"] += 1
        fps = _num_or_none(entry.get("fps_processed"))
        if fps and fps > 0:
            roll["sum_fps"] += fps
            roll["n_fps"] += 1
        tiles = _num_or_none(entry.get("num_tiles"))
        if tiles and tiles > 0:
            roll["sum_tiles"] += tiles
            roll["n_tiles"] += 1
        vram = _vram_peak(entry)
        if vram > 0:
            roll["sum_vram"] += vram
            roll["n_vram"] += 1
        load_s = _num_or_none(entry.get("load_s"))
        if load_s and load_s > 0:
            roll["sum_load"] += load_s
            roll["n_load"] += 1
        proc_s = _num_or_none(entry.get("process_s"))
        if proc_s and proc_s > 0:
            roll["sum_proc"] += proc_s
            roll["n_proc"] += 1
        in_w = _num_or_none(entry.get("in_w"))
        in_h = _num_or_none(entry.get("in_h"))
        out_w = _num_or_none(entry.get("out_w"))
        out_h = _num_or_none(entry.get("out_h"))
        if in_w and in_h and out_w and out_h and in_w * in_h > 0:
            roll["sum_px_in"] += in_w * in_h
            roll["sum_px_out"] += out_w * out_h
            roll["n_px"] += 1
    if "bytes" not in already:
        part = _byte_part(entry)
        roll = _add_rolls(roll, part["roll"])
        groups |= part["groups"]
    return {"roll": roll, "groups": groups}


def _sub_rolls(left, right) -> dict:
    out = _as_rollup(left)
    extra = _as_rollup(right)
    for key in out:
        out[key] -= extra[key]
    return out


def config_tags(rec: dict) -> List[tuple]:
    """Short config chips. Each pair is (label, hover text)."""
    if not isinstance(rec, dict):
        return []
    tags: List[tuple] = []

    def add(short: str, tip: str) -> None:
        if short and all(short != got for got, _tip in tags):
            tags.append((short, tip))

    mode = str(rec.get("mode") or "").strip()
    if mode:
        add(mode, {"tiny": "Tiny pipeline", "full": "Full pipeline", "tiny-long": "Tiny-long pipeline"}.get(mode, f"Pipeline {mode}"))
    version = str(rec.get("model_version") or "").strip()
    if version:
        add(version, f"Model version {version}")
    scale = _compact_num(rec.get("scale"))
    if scale:
        add(f"×{scale}", f"Upscale {scale}×")
    dtype = str(rec.get("dtype_str") or rec.get("dtype") or "").strip()
    if dtype:
        add(dtype, f"Data type {dtype}")
    attn = str(rec.get("attention_mode") or "").strip()
    if attn:
        add(attn, f"Attention mode {attn}")
    tiled = bool(rec.get("tiled_dit") or rec.get("tiled_vae"))
    tile = _compact_num(rec.get("tile_size"))
    if tiled and tile and tile != "0":
        add(f"T{tile}", f"Tile size {tile}")
    overlap = _compact_num(rec.get("tile_overlap"))
    if tiled and overlap:
        add(f"Ov{overlap}", f"Tile overlap {overlap}")
    quality = _compact_num(rec.get("quality"))
    if quality:
        add(f"Q{quality}", f"Output encode quality {quality}")
    chunk = _num_or_none(rec.get("chunk_s") or rec.get("chunk_duration"))
    if chunk and chunk > 0:
        add(f"C{_compact_num(chunk)}", f"Chunks of {chunk:g}s")
    group = _compact_num(rec.get("gt_group_size"))
    if group:
        add(f"G{group}", f"Group size {group}")
    rife = str(rec.get("gt_rife_mode") or "").strip().lower()
    if rife.startswith("4"):
        add("R4", "RIFE 4× frames (two 2× passes)")
    elif rife.startswith("2"):
        add("R2", "RIFE 2× frames (one pass)")
    elif rife.startswith("off") or rife.startswith("no"):
        add("R0", "RIFE off")
    rife_q = _compact_num(rec.get("gt_rife_quality"))
    if rife_q:
        add(f"Rq{rife_q}", f"RIFE quality {rife_q}")
    export_q = _compact_num(rec.get("gt_export_quality"))
    if export_q:
        add(f"Eq{export_q}", f"Export quality {export_q}")
    export_w = _compact_num(rec.get("gt_export_max_width"))
    if export_w:
        add(f"W{export_w}", f"Export max width {export_w}")
    sparse = _compact_num(rec.get("sparse_ratio"))
    if sparse:
        add(f"S{sparse}", f"Sparse ratio {sparse}")
    kv = _compact_num(rec.get("kv_ratio"))
    if kv:
        add(f"K{kv}", f"KV cache ratio {kv}")
    local = _compact_num(rec.get("local_range"))
    if local:
        add(f"L{local}", f"Local range {local}")
    if rec.get("color_fix"):
        add("CF", "Color fix on")
    if rec.get("tiled_dit"):
        add("DiT", "Tiled DiT on")
    if rec.get("tiled_vae"):
        add("VAE", "Tiled VAE on")
    if rec.get("unload_dit"):
        add("Ud", "Unload DiT before decode")
    return tags


def _tag(short: str, tip: str) -> str:
    return (
        f"<span title='{_esc(tip)}' style='display:inline-block;margin:0 4px 4px 0;"
        "padding:1px 7px;border-radius:999px;border:1px solid #334155;background:#0b1220;"
        f"color:#7dd3fc;font-size:11px;cursor:help'>{_esc(short)}</span>"
    )


def tags_html(rec: dict) -> str:
    return "".join(_tag(short, tip) for short, tip in config_tags(rec))


LEGEND = (
    ("tiny / full", "Pipeline mode"),
    ("v1.0 / v1.1", "Model version"),
    ("×4", "Upscale factor"),
    ("bf16 / fp16", "Data type"),
    ("sage / block", "Attention mode"),
    ("T320", "Tile size. T320 means 320px tiles."),
    ("Ov32", "Tile overlap in pixels"),
    ("Q10", "Output encode quality"),
    ("C10.3", "Chunk length in seconds"),
    ("G5", "Group Therapy files per batch"),
    ("R4", "RIFE 4× frames, two 2× passes"),
    ("R2", "RIFE 2× frames, one pass"),
    ("R0", "RIFE off"),
    ("Rq100", "RIFE encode quality"),
    ("Eq100", "Final export quality"),
    ("W3840", "Export max width"),
    ("S1.0", "Sparse ratio"),
    ("K3", "KV cache ratio"),
    ("L7", "Local range"),
    ("CF", "Color fix on"),
    ("DiT", "Tiled DiT on"),
    ("VAE", "Tiled VAE on"),
    ("Ud", "Unload DiT before decode"),
    ("in → out", "Average input file size versus average output file size"),
    ("Δ size", "Average size change, (output − input) / input"),
    ("steps", "Average tiles, or counted steps, per file"),
    ("s/step", "Average seconds per step, file time divided by steps"),
    ("px ×", "Average output pixels divided by input pixels"),
)


def legend_html() -> str:
    """Collapsible-ready legend. Hover any live tag for the same text."""
    rows = "".join(
        f"<tr><td style='{_TD};color:#7dd3fc;white-space:nowrap'>{_esc(short)}</td>"
        f"<td style='{_TD}'>{_esc(tip)}</td></tr>"
        for short, tip in LEGEND
    )
    return (
        "<div style='color:#94a3b8;font-size:12px;line-height:1.45;margin-bottom:6px'>"
        "Hover a tag, a stat, or a progress block to read the full name. "
        "R4 is RIFE 4×. The letters stay short so a finished file can show every knob at once."
        "</div>"
        f"<table style='{_TABLE}'><tr><th style='{_TH}'>Tag</th><th style='{_TH}'>Meaning</th></tr>{rows}</table>"
    )


def _cell(label: str, value: str, tip: str) -> str:
    return (
        f"<span title='{_esc(tip)}' style='display:inline-block;margin:2px 6px 2px 0;"
        "padding:2px 6px;border-radius:6px;background:#111827;border:1px solid #1e293b;cursor:help'>"
        f"<span style='color:#94a3b8'>{_esc(label)}</span> "
        f"<b style='color:#e2e8f0'>{_esc(value)}</b></span>"
    )


def _div(roll: dict, sum_key: str, n_key: str) -> Optional[float]:
    count = float(roll.get(n_key) or 0)
    if count <= 0:
        return None
    return float(roll.get(sum_key) or 0) / count


def rich_html(roll) -> str:
    """Nerd stats for one rollup: sizes, steps, frames, VRAM."""
    roll = _as_rollup(roll)
    avg_s = _div(roll, "sum_secs", "n_secs")
    in_b = _div(roll, "sum_in", "n_in")
    out_b = _div(roll, "sum_out", "n_out")
    pct = _div(roll, "sum_pct", "n_pct")
    steps = _div(roll, "sum_steps", "n_steps")
    step_s = _div(roll, "sum_step_s", "n_step_s")
    frames = _div(roll, "sum_frames", "n_frames")
    spf = _div(roll, "sum_spf", "n_spf")
    fps = _div(roll, "sum_fps", "n_fps")
    tiles = _div(roll, "sum_tiles", "n_tiles")
    vram = _div(roll, "sum_vram", "n_vram")
    load_s = _div(roll, "sum_load", "n_load")
    proc_s = _div(roll, "sum_proc", "n_proc")
    px = None
    if roll["n_px"] and roll["sum_px_in"] > 0:
        px = (roll["sum_px_out"] / roll["n_px"]) / (roll["sum_px_in"] / roll["n_px"])
    cells = [
        _cell("files", str(int(roll["n"])), "Finished files in this average"),
    ]
    if avg_s:
        cells.append(_cell("avg", _fmt_eta(avg_s), "Average time per finished file"))
    if in_b or out_b:
        cells.append(_cell(
            "in → out", f"{_fmt_bytes(in_b)} → {_fmt_bytes(out_b)}",
            "Average input file size versus average output file size",
        ))
    if pct is not None:
        cells.append(_cell("Δ size", _fmt_pct(pct), "Average file-size change, (output − input) / input"))
    if steps:
        cells.append(_cell("steps", f"{steps:.1f}", "Average tiles or counted steps per file"))
    if step_s:
        cells.append(_cell("s/step", f"{step_s:.2f}s", "Average seconds per step, file time divided by steps"))
    if frames:
        cells.append(_cell("frames", f"{frames:.0f}", "Average frame count going into the upscale"))
    if spf:
        cells.append(_cell("s/frame", f"{spf:.2f}", "Average seconds of processing per frame"))
    if fps:
        cells.append(_cell("fps", f"{fps:.2f}", "Average processed frames per second"))
    if tiles:
        cells.append(_cell("tiles", f"{tiles:.1f}", "Average tile count when tiled DiT is on"))
    if px:
        cells.append(_cell("px ×", f"{px:.2f}", "Average output pixel area divided by input pixel area"))
    if vram:
        cells.append(_cell("VRAM", f"{vram:.0f} MB", "Average peak VRAM reserved during the file"))
    if load_s or proc_s:
        cells.append(_cell("load/proc", f"{_fmt_eta(load_s)}/{_fmt_eta(proc_s)}", "Average model-load time versus process time"))
    return "".join(cells)


def _history_eras(archive: list, bucket: str) -> List[dict]:
    eras = []
    for entry in archive or []:
        if not isinstance(entry, dict) or entry.get("bucket") != bucket:
            continue
        rec = entry.get("record") if isinstance(entry.get("record"), dict) else {}
        roll = _as_rollup(rec.get("rollup"))
        if roll["n"] <= 0:
            times = []
            for raw in rec.get("file_seconds") or []:
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if val > 1:
                    times.append(val)
            done = int(rec.get("lifetime_done") or 0)
            if not times and done <= 0:
                continue
            roll["n"] = float(done or len(times))
            roll["sum_secs"] = float(sum(times))
            roll["n_secs"] = float(len(times))
        eras.append({"ts": entry.get("ts"), "roll": roll})
    return eras


def history_html(archive: list, bucket: str) -> str:
    """Parked totals. Clearing them requires the options toggle."""
    eras = _history_eras(archive, bucket)
    if not eras:
        line = "nothing parked yet · locked until Unlock history clear is on"
        body = ""
    else:
        total = _empty_rollup()
        for era in eras:
            total = _add_rolls(total, era["roll"])
        noun = "era" if len(eras) == 1 else "eras"
        line = f"{len(eras)} {noun} · locked · " + rich_html(total)
        rows = []
        for index, era in enumerate(reversed(eras), start=1):
            when = _fmt_eta(None)
            try:
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(era["ts"])))
            except (TypeError, ValueError, OSError, OverflowError):
                when = f"era {index}"
            rows.append([_esc(when), rich_html(era["roll"])])
        body = (
            "<details style='margin-top:6px'>"
            "<summary style='cursor:pointer;color:#c4b5fd;font-size:12px'>History eras</summary>"
            + _table(["when cleared", "parked totals"], rows, "")
            + "</details>"
        )
    return (
        f"<div style='margin-top:6px'><b style='color:#c4b5fd'>History</b> &nbsp; {line}</div>"
        + body
    )


def annotate_settled(app_dir, name: str, **fields) -> str:
    """Patch the finished file with final sizes and config. Adds only new numbers."""
    app = Path(app_dir)
    bucket = _lock_bucket(app) or "group"
    clean = {}
    for key, value in fields.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            if value is not None:
                clean[str(key)] = value
    if not clean or not name:
        return "nothing to note"
    with _STATS_LOCK:
        store = _read_store(app)
        rec = _bucket_stats(store["buckets"].get(bucket), time.time())
        open_file = rec.get("open_file") if isinstance(rec.get("open_file"), dict) else None
        if open_file and open_file.get("name") == name and not open_file.get("recorded"):
            metrics = dict(open_file.get("metrics") or {})
            metrics.update(clean)
            open_file["metrics"] = metrics
            rec["open_file"] = open_file
            store["buckets"][bucket] = rec
            _write_store(app, store)
            return "noted on the open file"
        files = list(rec.get("recent_files") or [])
        for index in range(len(files) - 1, -1, -1):
            entry = files[index]
            if not isinstance(entry, dict) or entry.get("name") != name:
                continue
            folded = set(entry.get("folded") or [])
            try:
                touch_session = float(entry.get("ended") or 0) >= float(rec.get("started") or 0)
            except (TypeError, ValueError):
                touch_session = True
            if "bytes" in folded and ("in_bytes" in clean or "out_bytes" in clean):
                old = _byte_part(entry)["roll"]
                entry.update(clean)
                new = _byte_part(entry)["roll"]
                rec["rollup"] = _add_rolls(_sub_rolls(rec.get("rollup"), old), new)
                if touch_session:
                    rec["session_rollup"] = _add_rolls(_sub_rolls(rec.get("session_rollup"), old), new)
            else:
                entry.update(clean)
                added = _contribute(entry, folded)
                entry["folded"] = sorted(folded | added["groups"])
                rec["rollup"] = _add_rolls(rec.get("rollup"), added["roll"])
                if touch_session:
                    rec["session_rollup"] = _add_rolls(rec.get("session_rollup"), added["roll"])
            if "bytes" in _byte_part(entry)["groups"]:
                entry["folded"] = sorted(set(entry.get("folded") or []) | {"bytes"})
            files[index] = entry
            rec["recent_files"] = files
            store["buckets"][bucket] = rec
            _write_store(app, store)
            return "updated the finished file"
        if open_file and open_file.get("name") == name:
            metrics = dict(open_file.get("metrics") or {})
            metrics.update(clean)
            open_file["metrics"] = metrics
            rec["open_file"] = open_file
            store["buckets"][bucket] = rec
            _write_store(app, store)
            return "noted on the open file"
    return "no matching file yet"


def _vram_peak(rec: dict) -> float:
    try:
        return float(rec.get("vram_reserved_peak_mb") or rec.get("vram_alloc_peak_mb") or 0)
    except (TypeError, ValueError):
        return 0.0


def _fold_cfg(table: dict, key: str, entry: dict) -> dict:
    out = dict(table or {})
    row = dict(out.get(key) or {"n": 0, "sum_spf": 0.0, "vram_peak_mb_max": 0.0, "sum_vram_peak": 0.0})
    vram = _vram_peak(entry)
    row["n"] = int(row.get("n") or 0) + 1
    row["sum_spf"] = float(row.get("sum_spf") or 0) + float(entry["s_per_frame"])
    row["vram_peak_mb_max"] = max(float(row.get("vram_peak_mb_max") or 0), vram)
    row["sum_vram_peak"] = float(row.get("sum_vram_peak") or 0) + vram
    tags = config_tags(entry)
    if tags:
        row["tag_line"] = " ".join(short for short, _tip in tags)
    out[key] = row
    return out


def _record_file(stats: dict, open_file: dict, now: float) -> None:
    """Append the closed file to recent_files and fold it into by_cfg / session_by_cfg."""
    metrics = dict(open_file.get("metrics") or {})
    if not metrics:
        return
    ended = float(metrics.get("finished") or now)
    entry = {k: v for k, v in metrics.items() if k not in ("finished", "ok")}
    entry.update({
        "name": open_file.get("name"),
        "ended": ended,
        "ok": metrics.get("ok"),
        "secs": max(0.0, ended - float(open_file.get("started") or ended)),
        "cfg_key": cfg_key(metrics),
    })
    if not entry.get("steps"):
        tiles = _num_or_none(entry.get("num_tiles"))
        if tiles and tiles > 0:
            entry["steps"] = tiles
    added = _contribute(entry, set())
    entry["folded"] = sorted(added["groups"])
    stats["rollup"] = _add_rolls(stats.get("rollup"), added["roll"])
    stats["session_rollup"] = _add_rolls(stats.get("session_rollup"), added["roll"])
    stats["recent_files"] = (list(stats.get("recent_files") or []) + [entry])[-RECENT_MAX:]
    try:
        spf_ok = entry["cfg_key"] and float(entry.get("s_per_frame") or 0) > 0
    except (TypeError, ValueError):
        spf_ok = False
    if spf_ok:
        stats["by_cfg"] = _fold_cfg(stats.get("by_cfg"), entry["cfg_key"], entry)
        stats["session_by_cfg"] = _fold_cfg(stats.get("session_by_cfg"), entry["cfg_key"], entry)


def _close_file(stats: dict, open_file: dict, now: float) -> None:
    _add_file_seconds(stats, max(0.0, now - float(open_file["started"])))
    if not open_file.get("recorded"):
        _record_file(stats, open_file, now)


def _update_stats(stats: dict, items: List[dict], heartbeat: dict, now: float) -> dict:
    last = dict(stats.get("last_status") or {})
    seen_done = list(stats.get("session_done") or [])
    seen_fail = list(stats.get("session_failed") or [])
    done_set = set(seen_done)
    fail_set = set(seen_fail)
    first_sight = not last and not seen_done and not seen_fail
    for it in items:
        path = str(it.get("path") or "")
        if not path:
            continue
        status = str(it.get("status") or "pending")
        prev = last.get(path)
        if not first_sight and prev != status:
            if status == "done" and path not in done_set:
                seen_done.append(path)
                done_set.add(path)
            elif status == "failed" and path not in fail_set:
                seen_fail.append(path)
                fail_set.add(path)
        last[path] = status
    # Drop paths no longer in this queue so the map stays bounded.
    live = {str(it.get("path") or "") for it in items}
    stats["last_status"] = {k: v for k, v in last.items() if k in live}
    stats["session_done"] = seen_done[-500:]
    stats["session_failed"] = seen_fail[-200:]

    name = str(heartbeat.get("file_name") or "")
    index = heartbeat.get("file_index")
    open_file = stats.get("open_file")
    metrics = heartbeat.get("file_metrics") if isinstance(heartbeat.get("file_metrics"), dict) else {}
    if name:
        key = (name, index)
        if not open_file or (open_file.get("name"), open_file.get("index")) != key:
            if open_file and open_file.get("started"):
                _close_file(stats, open_file, now)
            stats["open_file"] = {"name": name, "index": index, "started": now}
        open_file = stats["open_file"]
        if metrics:
            open_file["metrics"] = dict(metrics)  # refreshed every tick while the file is open
        if metrics.get("finished") and not open_file.get("recorded"):
            _record_file(stats, open_file, now)  # file_finished: close the record with final values
            open_file["recorded"] = True
    elif open_file and open_file.get("started"):
        # No file in progress (between files, stopped, idle): close the open file
        # now so the idle gap is not counted in file_seconds.
        _close_file(stats, open_file, now)
        stats["open_file"] = None
    step_key = (
        heartbeat.get("step_label"),
        heartbeat.get("step_done"),
        heartbeat.get("step_total"),
    )
    if heartbeat.get("step_done") is not None and step_key != stats.get("last_step_key"):
        prev_key = stats.get("last_step_key")
        if prev_key and prev_key[0] == step_key[0] and step_key[1] is not None and prev_key[1] is not None:
            try:
                delta_steps = int(step_key[1]) - int(prev_key[1])
            except (TypeError, ValueError):
                delta_steps = 0
            advanced = stats.get("last_step_advance")
            if delta_steps > 0 and advanced:
                stats["step_seconds"] = (
                    stats["step_seconds"] + [max(0.05, (now - float(advanced)) / delta_steps)]
                )[-40:]
        stats["last_step_key"] = step_key
        stats["last_step_advance"] = now
    return stats


def classify_health(sample: dict, heartbeat: dict, stats: dict, now: float, running: bool) -> dict:
    """Thrash is a stall with a busy-looking GPU, not a full 4090 by itself.

    A healthy tiled job sits near full VRAM. Low power plus a frozen step
    bar is the warning. High commit is a separate page risk.
    """
    gpu = float(sample.get("gpu_util") or 0)
    mem_util = float(sample["mem_util"]) if "mem_util" in sample else None
    power = float(sample["power_w"]) if "power_w" in sample else None
    limit = float(sample["power_limit_w"]) if "power_limit_w" in sample else None
    power_ratio = (power / limit) if power is not None and limit is not None and limit > 1 else None
    commit_used = float(sample.get("commit_used_gb") or 0)
    commit_limit = float(sample.get("commit_limit_gb") or 0)
    commit_ratio = (commit_used / commit_limit) if commit_limit > 1 else 0
    avail = float(sample.get("phys_avail_gb") or 99)
    advanced = stats.get("last_step_advance")
    stalled_for = now - float(now if advanced is None else advanced)
    step_label = str(heartbeat.get("step_label") or heartbeat.get("section") or "")
    steps = list(stats.get("step_seconds") or [])
    slow = False
    if len(steps) >= 4:
        med = sorted(steps)[len(steps) // 2]
        if med > 0 and steps[-1] >= med * 2.5 and steps[-1] >= 8:
            slow = True

    if running and gpu >= 90 and stalled_for >= 25 and power_ratio is not None and power_ratio < 0.40 and mem_util is not None and mem_util < 30:
        return {
            "level": "thrash",
            "title": "THRASH",
            "detail": (
                f"GPU {gpu:.0f}% but {power:.0f}W and the {step_label or 'step'} bar "
                f"has not moved for {int(stalled_for)}s. Let this file finish, then restart "
                "before the next one if the bar stays frozen."
            ),
        }
    if running and (commit_ratio >= 0.82 or avail < 3.5):
        return {
            "level": "watch",
            "title": "PAGE RISK",
            "detail": (
                f"Commit {commit_used:.0f}/{commit_limit:.0f} GB · free RAM {avail:.1f} GB. "
                "Steps can still be healthy. Close other apps before the next group if this stays up."
            ),
        }
    if running and slow:
        return {
            "level": "watch",
            "title": "SLOWING",
            "detail": (
                f"Latest {step_label or 'step'} is {steps[-1]:.1f}s vs a {sorted(steps)[len(steps)//2]:.1f}s median. "
                "A long VAE tile does this and then recovers. Watch the next mark before changing tile size."
            ),
        }
    if running and gpu >= 90 and power_ratio is not None and power_ratio < 0.45 and mem_util is not None and mem_util < 25 and stalled_for < 25:
        return {
            "level": "busy",
            "title": "BUSY",
            "detail": (
                f"VRAM is working and the step bar is moving ({step_label or 'step'}). "
                f"Power is {power:.0f}W. Full VRAM alone is not a stall."
            ),
        }
    if running:
        return {
            "level": "ok",
            "title": "HEALTHY",
            "detail": "Steps are moving. VRAM near full on a 4090 tile is the normal load.",
        }
    return {
        "level": "idle",
        "title": "IDLE",
        "detail": "No file is running. Bars show the queue that will start next.",
    }


def _health_color(level: str) -> str:
    return {
        "ok": "#86efac",
        "busy": "#7dd3fc",
        "watch": "#fbbf24",
        "thrash": "#fca5a5",
        "idle": "#94a3b8",
    }.get(level, "#94a3b8")


def _stage_pips(batch: dict) -> str:
    """Named stages. The one running is marked now; earlier ones are done."""
    stages = batch.get("stages") or list(STAGE_ORDER)
    current = batch.get("stage") or ""
    bits = []
    seen_current = False
    for stage in stages:
        label = STAGE_LABEL.get(stage, stage)
        if stage == current:
            seen_current = True
            bits.append(f"<b style='color:#7dd3fc'>▶ {_esc(label)}</b>")
        elif not seen_current and current:
            bits.append(f"<span style='color:#86efac'>✓ {_esc(label)}</span>")
        else:
            bits.append(f"<span style='color:#64748b'>{_esc(label)}</span>")
    return " &nbsp;·&nbsp; ".join(bits)


def _pct_label(done: int, total: int) -> str:
    total_i = int(total or 0)
    if total_i <= 0:
        return "—"
    return f"{100.0 * min(int(done or 0), total_i) / total_i:.0f}%"


_STAGE_POS_KEY = {
    "upscale": "gt_upscale",
    "rife1": "gt_rife1",
    "rife2": "gt_rife2",
    "export": "gt_export",
}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def file_place(members: List[dict], stage: str, name: str = "") -> tuple:
    """1-based spot of the open file in this group, and how many files are in the group.

    Files that already wrote this stage's output are behind the open file.
    The open file is the running row, or the row whose filename matches.
    """
    rows = [it for it in members if isinstance(it, dict)]
    total = len(rows)
    if total <= 0:
        return 0, 0
    key = _STAGE_POS_KEY.get(stage or "")
    name_l = (name or "").strip().lower()
    finished = 0
    hit = None
    for it in rows:
        if key and _stage_path(it, key):
            finished += 1
        base = Path(str(it.get("path") or "")).name.lower()
        if hit is None and (str(it.get("status") or "") == "running" or (name_l and base == name_l)):
            hit = it
    if hit is None:
        return (min(finished, total) if finished else 0), total
    if key and _stage_path(hit, key):
        pos = finished
    else:
        pos = finished + 1
    return max(1, min(pos, total)), total


def queue_file_place(items: List[dict], beat: dict, name: str = "") -> tuple:
    """1-based spot in the whole queue. The live session index wins when it is set."""
    beat = beat or {}
    idx = _as_int(beat.get("file_index"))
    total = _as_int(beat.get("file_total"))
    if idx > 0 and total > 0:
        return idx, total
    rows = [it for it in items if isinstance(it, dict)]
    total = len(rows)
    if total <= 0:
        return 0, 0
    name_l = (name or "").strip().lower()
    for i, it in enumerate(rows):
        base = Path(str(it.get("path") or "")).name.lower()
        if str(it.get("status") or "") == "running" or (name_l and base == name_l):
            return i + 1, total
    return 0, total


def _file_of(pos: int, total: int) -> str:
    if pos > 0 and total > 0:
        return f"File {int(pos)} of {int(total)}"
    return ""


def now_banner(
    name: str,
    stage_word: str,
    group_pos: int,
    group_total: int,
    queue_pos: int,
    queue_total: int,
    group_id: Any = "",
) -> str:
    """Large plain text: which file, which number out of how many, and the filename."""
    group_line = _file_of(group_pos, group_total)
    queue_line = _file_of(queue_pos, queue_total)
    if group_line:
        head = group_line
    elif queue_line:
        head = queue_line
        queue_line = ""
    else:
        head = "No file running"
    stage = (stage_word or "").strip()
    if stage and stage != "—":
        head += f" · {_esc(stage)}"
    name_html = ""
    if name:
        name_html = (
            "<div style='font-size:20px;font-weight:700;color:#f8fafc;line-height:1.3;"
            f"margin-top:4px;word-break:break-word'>{_esc(name)}</div>"
        )
    detail_bits = []
    gid = group_id if group_id not in (None, "", "—") else ""
    if gid != "":
        detail_bits.append(f"group {_esc(gid)}")
    if queue_line and queue_line != head:
        detail_bits.append(f"queue {queue_line[0].lower()}{queue_line[1:]}")
    detail = ""
    if detail_bits:
        detail = (
            "<div style='font-size:16px;font-weight:700;color:#e2e8f0;margin-top:4px'>"
            + " · ".join(detail_bits)
            + "</div>"
        )
    return (
        "<div style='margin:0 0 10px 0;padding:10px 12px;border-radius:8px;"
        "background:#020617;border:1px solid #334155'>"
        "<div style='font-size:28px;font-weight:800;color:#f8fafc;line-height:1.15'>"
        f"{head}</div>"
        f"{name_html}{detail}</div>"
    )


def _meter(title: str, bar: str, caption: str, pct: str = "", tip: str = "", lead: str = "") -> str:
    pct_html = (
        f"<div style='font-size:22px;font-weight:800;color:#f8fafc;line-height:1;margin-top:2px'>{_esc(pct)}</div>"
        if pct else ""
    )
    lead_html = (
        f"<div style='font-size:16px;font-weight:800;color:#f8fafc;line-height:1.35;margin-top:4px'>{lead}</div>"
        if lead else ""
    )
    hover = f" title='{_esc(tip)}'" if tip else ""
    return (
        f"<div{hover} style='flex:1;min-width:240px;padding:8px 10px;background:#0b1220;"
        "border:1px solid #1e293b;border-radius:8px;cursor:help'>"
        f"<div style='color:#7dd3fc;font-weight:700;font-size:12px;letter-spacing:.04em;'>{_esc(title)}</div>"
        f"{lead_html}"
        f"{pct_html}"
        f"<div style='font-family:Consolas,ui-monospace,monospace;font-size:15px;line-height:1.2;margin:4px 0;'>{bar}</div>"
        f"<div style='color:#cbd5e1;font-size:12px;line-height:1.35;'>{caption}</div>"
        "</div>"
    )


def _chips(queues: Dict[str, dict], active: str) -> str:
    bits = []
    for name, label in QUEUE_ORDER:
        c = queues[name]["counts"]
        left = c["pending"] + c["running"]
        color = "#7dd3fc" if name == active else "#94a3b8"
        border = "#38bdf8" if name == active else "#334155"
        bits.append(
            f"<span style='color:{color};border:1px solid {border};border-radius:999px;"
            f"padding:2px 8px;font-size:12px;'>{_esc(label)} {c['done']}/{c['total']} · left {left}</span>"
        )
    return " ".join(bits)


def render(
    app_dir,
    *,
    heartbeat: Optional[dict] = None,
    sample: Optional[dict] = None,
    now: Optional[float] = None,
    persist: bool = True,
) -> str:
    """HTML for the session board. Safe to call every couple of seconds."""
    app = Path(app_dir)
    now_v = _now(now)
    queues = _load_queues(app)
    active = _active_name(app, queues)
    pack = queues.get(active) or queues["group"]
    items = pack["items"]
    counts = pack["counts"]
    data = pack["data"]
    beat = heartbeat if heartbeat is not None else _heartbeat()
    machine = sample if sample is not None else sample_machine(now_v)
    record_bucket = _lock_bucket(app)
    show_bucket = record_bucket or LOCK_TO_BUCKET.get(active, "group")
    with _STATS_LOCK:
        store = _read_store(app)
        stats = _bucket_stats(store["buckets"].get(show_bucket), now_v)
        if record_bucket:
            stats = _update_stats(stats, items, beat, now_v)
            store["buckets"][record_bucket] = stats
            if persist:
                try:
                    _write_store(app, store)
                except OSError:
                    pass

    running = counts["running"] > 0 or bool(beat.get("file_name"))
    health = classify_health(machine, beat, stats, now_v, running)
    live_name = str(beat.get("file_name") or "").strip()
    batch = _batch_progress(data, items, section=str(beat.get("section") or ""), name=live_name)
    left = counts["pending"] + counts["running"]
    session_done = len(stats.get("session_done") or [])
    session_failed = len(stats.get("session_failed") or [])
    elapsed_s = max(0.0, now_v - float(stats.get("started") or now_v))
    session_row, total_row, avg_file, sess_avg, sess_times, _file_times = _avg_rows(
        stats, session_done, elapsed_s
    )
    archive = _read_archive(app, store)
    session_extra = rich_html(stats.get("session_rollup"))
    total_extra = rich_html(stats.get("rollup"))
    parked = history_html(archive, show_bucket)
    elapsed_h = max(1 / 3600, elapsed_s / 3600)
    per_hour = (session_done / elapsed_h) if session_done else None
    eta_avg = sess_avg if len(sess_times) >= SESSION_ETA_MIN else avg_file
    eta = (eta_avg * max(0, left - (1 if counts["running"] else 0))) if eta_avg else None
    step_done = int(beat.get("step_done") or 0)
    step_total = int(beat.get("step_total") or 0)
    step_label = str(beat.get("step_label") or beat.get("section") or "step")
    step_times = [float(x) for x in (stats.get("step_seconds") or [])]
    current_name = Path(str(batch.get("current_name") or beat.get("file_name") or "")).name
    stage_word = str(batch.get("stage_label") or "").strip()
    if stage_word in ("", "—"):
        stage_word = str(beat.get("section") or "").strip()
    group_pos = _as_int(batch.get("file_pos"))
    group_total = _as_int(batch.get("file_count"))
    queue_pos, queue_total_n = queue_file_place(items, beat, current_name)
    banner = now_banner(
        current_name, stage_word, group_pos, group_total, queue_pos, queue_total_n, batch.get("group"),
    )

    def _lead(pos: int, total_n: int, phrase: str) -> str:
        line = _file_of(pos, total_n)
        bits = []
        if line:
            bits.append(_esc(f"{line} {phrase}".strip()))
        if current_name:
            bits.append(_esc(current_name))
        return "<br>".join(bits)

    file_pos_show, file_total_show = (group_pos, group_total) if group_pos and group_total else (queue_pos, queue_total_n)
    file_lead = _lead(file_pos_show, file_total_show, "")
    batch_lead = _lead(group_pos, group_total, "in this group")
    queue_lead = _lead(queue_pos, queue_total_n, "in the queue")
    profile = str(store.get("active_profile") or stats.get("active_profile") or "")
    live_metrics = beat.get("file_metrics") if isinstance(beat.get("file_metrics"), dict) else {}
    if not config_tags(live_metrics):
        recent = [e for e in (stats.get("recent_files") or []) if isinstance(e, dict)]
        tag_src = recent[-1] if recent else {}
    else:
        tag_src = live_metrics
    tag_row = tags_html(tag_src)
    lamp = _health_color(health["level"])
    stop = " &nbsp; <span style='color:#fca5a5;font-weight:700'>STOP ARMED</span>" if beat.get("stop_armed") else ""
    label = dict(QUEUE_ORDER).get(active, active)

    stage_queue = _stage_queue(data, items)
    colors = load_stage_colors(app)
    queue_mix = stage_mix(items) if stage_queue else None
    batch_mix = batch.get("mix") if isinstance(batch.get("mix"), dict) else stage_mix([])
    if queue_mix:
        file_caption = (
            f"<b style='color:{colors['saved']}'>{queue_mix['saved']}</b> saved"
            f" &nbsp; <b style='color:{colors['rife']}'>{queue_mix['rife']}</b> rife"
            f" &nbsp; <b style='color:{colors['upscale']}'>{queue_mix['upscale']}</b> up"
            f" &nbsp; <b style='color:#fbbf24'>{queue_mix['wait']}</b> waiting"
            f" &nbsp; <b style='color:#e2e8f0'>{queue_mix['total']}</b> total"
            f" &nbsp; <b style='color:#7dd3fc'>+{session_done}</b> this session"
            f" &nbsp; <b style='color:#fca5a5'>{queue_mix['failed']}</b> failed"
        )
    else:
        file_caption = (
            f"<b style='color:#86efac'>{counts['done']}</b> done"
            f" &nbsp; <b style='color:#fbbf24'>{left}</b> left"
            f" &nbsp; <b style='color:#e2e8f0'>{counts['total']}</b> total"
            f" &nbsp; <b style='color:#7dd3fc'>+{session_done}</b> this session"
            f" &nbsp; <b style='color:#fca5a5'>{counts['failed']}</b> failed"
        )
    batch_total = int(batch["batch_total"] or batch["size"] or 0)
    if stage_queue and int(batch_mix.get("total") or 0) > 0:
        batch_caption = (
            f"size <b>{int(batch['size'] or 0)}</b>"
            f" &nbsp; group <b>{_esc(batch['group'])}</b>"
            f" &nbsp; stage <b style='color:#7dd3fc'>{_esc(batch.get('stage_label') or '—')}</b>"
            f" &nbsp; <b style='color:{colors['upscale']}'>{batch_mix['upscale']}</b> up"
            f" &nbsp; <b style='color:{colors['rife']}'>{batch_mix['rife']}</b> rife"
            f" &nbsp; <b style='color:{colors['saved']}'>{batch_mix['saved']}</b> saved"
            f" / {int(batch_mix['total'])}"
        )
    else:
        batch_caption = (
            f"size <b>{int(batch['size'] or 0)}</b>"
            f" &nbsp; group <b>{_esc(batch['group'])}</b>"
            f" &nbsp; stage <b style='color:#7dd3fc'>{_esc(batch.get('stage_label') or '—')}</b>"
            f" &nbsp; <b>{int(batch['batch_done'])}</b>/{batch_total or '—'} settled in this batch"
        )
    if step_total:
        step_bar = glyph_bar(step_done, step_total, 32)
        step_caption = f"{_esc(step_label)} <b>{step_done}/{step_total}</b>"
    else:
        step_bar = glyph_bar(0, 0, 32)
        step_caption = "waiting for the next tile / DiT / decode mark"
    if current_name:
        step_caption = f"<b style='color:#f8fafc'>{_esc(current_name)}</b><br>{step_caption}"

    machine_bits = []
    if machine.get("vram_total_mib"):
        machine_bits.append(
            f"VRAM {machine['vram_used_mib']/1024:.1f}/{machine['vram_total_mib']/1024:.1f} GB"
        )
    if machine.get("power_w"):
        machine_bits.append(f"{machine['power_w']:.0f}W")
    if machine.get("temp_c"):
        machine_bits.append(f"{machine['temp_c']:.0f}°C")
    if machine.get("commit_limit_gb"):
        machine_bits.append(
            f"commit {machine['commit_used_gb']:.0f}/{machine['commit_limit_gb']:.0f} GB"
        )
    if "gpu_util" in machine:
        machine_bits.append(f"GPU {machine['gpu_util']:.0f}%")
    stat_bits = [
        f"rate {per_hour:.1f}/h" if per_hour else "rate —",
        f"ETA {_fmt_eta(eta)}",
        f"step {step_times[-1]:.1f}s" if step_times else "step —",
        f"session fails {session_failed}",
    ]
    stat_bits.extend(machine_bits)
    spark = _spark(step_times)

    return (
        "<div style='padding:12px 14px;margin:0 0 12px 0;border-radius:10px;"
        "border:1px solid #1e293b;background:linear-gradient(180deg,#0f172a,#0b1220);color:#e2e8f0;'>"
        "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px;'>"
        f"<span style='font-weight:800;color:#e2e8f0;letter-spacing:.06em;'>SESSION</span>"
        f"<span style='color:#7dd3fc;font-weight:700'>{_esc(label)}</span>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{lamp};"
        f"box-shadow:0 0 8px {lamp};display:inline-block'></span>"
        f"<span style='color:{lamp};font-weight:800'>{_esc(health['title'])}</span>"
        f"{stop}"
        f"<span style='margin-left:auto;color:#94a3b8;font-size:12px;'>"
        f"{_esc(profile) if profile else 'no job profile applied'}</span>"
        "</div>"
        + banner
        + _avg_block(session_row, total_row, session_extra, total_extra, parked)
        + f"<div style='margin-bottom:8px'>{_chips(queues, active)}</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
        + _meter(
            "THIS FILE",
            step_bar,
            step_caption,
            _pct_label(step_done, step_total),
            "Progress through the file that is open now: tile, DiT, decode, or encode.",
            file_lead,
        )
        + _meter(
            "THIS BATCH",
            stage_bar(batch_mix, 24, colors=colors) if stage_queue and int(batch_mix.get("total") or 0) > 0
            else glyph_bar(int(batch["batch_done"]), batch_total or int(batch["size"] or 0), 24),
            batch_caption,
            stage_pct(batch_mix) if stage_queue and int(batch_mix.get("total") or 0) > 0
            else _pct_label(int(batch["batch_done"]), batch_total or int(batch["size"] or 0)),
            "Current group. Green is saved, purple is RIFE done, blue is upscale done. "
            "Each color is that many files out of the group. A later color stays left of an earlier one.",
            batch_lead,
        )
        + _meter(
            "QUEUE",
            stage_bar(queue_mix, 42, colors=colors) if queue_mix else glyph_bar(counts["done"], counts["total"], 42),
            file_caption,
            stage_pct(queue_mix) if queue_mix else _pct_label(counts["done"], counts["total"]),
            "Whole queue. Green is saved, purple is RIFE done, blue is upscale done. "
            "The bar moves when a file finishes a stage, not only when the group is saved.",
            queue_lead,
        )
        + "</div>"
        + (
            stage_stats_html(
                items, colors, now_v, batch.get("group"), _process_ram_mb(),
            )
            if stage_queue else ""
        )
        + f"<div style='margin-top:8px;font-size:13px;'>{_stage_pips(batch)}</div>"
        + (f"<div style='margin-top:6px'>{tag_row}</div>" if tag_row else "")
        + f"<div style='margin-top:6px;font-family:Consolas,ui-monospace,monospace;font-size:14px;'>"
        f"step time {spark}</div>"
        "<div style='margin-top:4px;color:#cbd5e1;font-size:12px;line-height:1.45;'>"
        + " &nbsp;·&nbsp; ".join(stat_bits)
        + "</div>"
        f"<div style='margin-top:6px;color:{lamp};font-size:13px;line-height:1.4;'>"
        f"{_esc(health['detail'])}</div>"
        "<div style='margin-top:6px;color:#64748b;font-size:11px;line-height:1.45;'>"
        + (
            f"<span style='color:{colors['saved']}'>█</span> saved"
            f" &nbsp; <span style='color:{colors['rife']}'>█</span> rife"
            f" &nbsp; <span style='color:{colors['upscale']}'>█</span> upscale"
            " &nbsp; <span style='color:#334155'>░</span> waiting"
            if stage_queue else
            "█ done &nbsp; ▓ now &nbsp; ░ left"
        )
        + " &nbsp;·&nbsp; file bar █ done &nbsp; ▓ now"
        " &nbsp;·&nbsp; <span style='color:#86efac'>HEALTHY</span> steps moving"
        " &nbsp;·&nbsp; <span style='color:#7dd3fc'>BUSY</span> full GPU, bar still advancing"
        " &nbsp;·&nbsp; <span style='color:#fbbf24'>PAGE / SLOW</span> commit or a long step"
        " &nbsp;·&nbsp; <span style='color:#fca5a5'>THRASH</span> frozen step, low watts, low memory bus"
        "</div>"
        "</div>"
    )


def _num(value: Any, fmt: str = "{:.0f}", suffix: str = "") -> str:
    try:
        if value is None or value == "":
            return "—"
        return fmt.format(float(value)) + suffix
    except (TypeError, ValueError):
        return "—"


_TABLE = "width:100%;border-collapse:collapse;font-size:12px;color:#e2e8f0;"
_TH = "text-align:left;padding:3px 6px;color:#7dd3fc;border-bottom:1px solid #1e293b;"
_TD = "padding:3px 6px;border-bottom:1px solid #111827;"


def _table(head: List[str], rows: List[List[str]], empty: str) -> str:
    if not rows:
        return f"<div style='color:#94a3b8;font-size:12px;padding:6px'>{_esc(empty)}</div>"
    th = "".join(f"<th style='{_TH}'>{_esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td style='{_TD}'>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table style='{_TABLE}'><tr>{th}</tr>{body}</table>"


def recent_files_html(stats: dict) -> str:
    """Pure: the last RECENT_MAX files of one bucket record, newest first."""
    rows = []
    for e in reversed(list((stats or {}).get("recent_files") or [])[-RECENT_MAX:]):
        if not isinstance(e, dict):
            continue
        io = "—"
        if e.get("in_w") and e.get("in_h"):
            io = f"{int(e['in_w'])}x{int(e['in_h'])}"
            if e.get("out_w") and e.get("out_h"):
                io += f"→{int(e['out_w'])}x{int(e['out_h'])}"
        chunks = f"{int(e['chunk_count'])}×{_num(e.get('chunk_s'), '{:g}')}s" if e.get("chunk_count") else "—"
        in_b = _num_or_none(e.get("in_bytes"))
        out_b = _num_or_none(e.get("out_bytes"))
        pct = None
        if in_b and out_b and in_b > 0:
            pct = (out_b - in_b) / in_b * 100.0
        steps = _file_steps(e)
        secs = _num_or_none(e.get("secs"))
        step_s = (secs / steps) if steps and secs and secs > 0 else None
        name = _esc(e.get("name") or "?") + ("" if e.get("ok") is not False else " <span style='color:#fca5a5'>fail</span>")
        chips = tags_html(e)
        if chips:
            name += f"<br>{chips}"
        rows.append([
            name,
            _num(e.get("secs"), "{:.0f}", "s"),
            _num(e.get("s_per_frame"), "{:.2f}"),
            _num(e.get("fps_processed"), "{:.2f}"),
            io,
            f"{_fmt_bytes(in_b)}→{_fmt_bytes(out_b)}",
            _fmt_pct(pct),
            _num(steps, "{:.0f}") if steps else "—",
            _num(step_s, "{:.2f}", "s") if step_s else "—",
            _num(_vram_peak(e) or None, "{:.0f}", " MB"),
            _num(e.get("ram_peak_mb"), "{:.0f}", " MB"),
            f"{_num(e.get('load_s'), '{:.0f}', 's')}/{_num(e.get('process_s'), '{:.0f}', 's')}",
            chunks,
            _num(e.get("num_tiles"), "{:.0f}"),
        ])
    return _table(
        ["name", "s", "s/frame", "fps", "px", "in→out", "Δ size", "steps", "s/step", "VRAM peak", "RAM peak", "load/proc", "chunks", "tiles"],
        rows, "No finished files with stats yet.",
    )


def compare_html(stats: dict) -> str:
    """Pure: averages per cfg_key, this session next to overall."""
    stats = stats or {}
    allc = dict(stats.get("by_cfg") or {})
    sess = dict(stats.get("session_by_cfg") or {})

    def cols(row):
        if not isinstance(row, dict) or not row.get("n"):
            return ["—", "—", "—", "—"]
        n = int(row["n"])
        return [
            str(n),
            _num(float(row.get("sum_spf") or 0) / n, "{:.2f}"),
            _num(float(row.get("sum_vram_peak") or 0) / n, "{:.0f}", " MB"),
            _num(row.get("vram_peak_mb_max"), "{:.0f}", " MB"),
        ]

    keys = sorted(set(allc) | set(sess), key=lambda k: -int((allc.get(k) or sess.get(k) or {}).get("n") or 0))
    rows = []
    for key in keys:
        sample = allc.get(key) if isinstance(allc.get(key), dict) else {}
        if not sample and isinstance(sess.get(key), dict):
            sample = sess.get(key) or {}
        label = f"<code>{_esc(key)}</code>"
        if sample.get("tag_line"):
            label += f"<br><span style='color:#7dd3fc'>{_esc(sample.get('tag_line'))}</span>"
        rows.append([label] + cols(sess.get(key)) + cols(allc.get(key)))
    return _table(
        ["settings", "session n", "s/frame", "VRAM avg", "VRAM max", "overall n", "s/frame", "VRAM avg", "VRAM max"],
        rows, "No settings compared yet.",
    )


def _show_bucket(app: Path, bucket: Optional[str]) -> str:
    if bucket in BUCKETS:
        return bucket
    lock = _lock_bucket(app)
    if lock:
        return lock
    return LOCK_TO_BUCKET.get(_active_name(app, _load_queues(app)), "group")


def render_recent(app_dir, bucket: Optional[str] = None, now: Optional[float] = None) -> str:
    """Recent files table for one bucket. Reads the stats file, never writes."""
    app = Path(app_dir)
    with _STATS_LOCK:
        store = _read_store(app)
    b = _show_bucket(app, bucket)
    return recent_files_html(_bucket_stats(store["buckets"].get(b), _now(now)))


def render_compare(app_dir, bucket: Optional[str] = None, now: Optional[float] = None) -> str:
    """Settings comparison table for one bucket. Reads the stats file, never writes."""
    app = Path(app_dir)
    with _STATS_LOCK:
        store = _read_store(app)
    b = _show_bucket(app, bucket)
    return compare_html(_bucket_stats(store["buckets"].get(b), _now(now)))


def job_step_label(heartbeat: dict, items: Optional[List[dict]] = None) -> str:
    """Pure: queued / loading model / chunk x/y / encoding / done (else the live step)."""
    beat = heartbeat or {}
    metrics = beat.get("file_metrics") if isinstance(beat.get("file_metrics"), dict) else {}
    section = str(beat.get("section") or "").strip()
    low = section.lower()
    if not beat.get("file_name"):
        waiting = any(isinstance(i, dict) and i.get("status") in ("pending", "running") for i in (items or []))
        return "queued" if waiting else "idle"
    if metrics.get("finished"):
        return "done"
    if low.startswith("loading model"):
        return "loading model"
    if low.startswith("encoding"):
        return "encoding"
    if low.startswith("chunk"):
        return section
    return str(beat.get("step_label") or section or "working")


def job_card_html(bucket: str, heartbeat: dict, stats: dict, items: Optional[List[dict]] = None) -> str:
    """Pure: one job card from a heartbeat snapshot, a bucket stats record and its queue items."""
    beat = heartbeat or {}
    stats = stats or {}
    metrics = beat.get("file_metrics") if isinstance(beat.get("file_metrics"), dict) else {}
    label = dict(QUEUE_ORDER).get({v: k for k, v in LOCK_TO_BUCKET.items()}.get(bucket, bucket), bucket)
    step = job_step_label(beat, items)
    name = str(beat.get("file_name") or "")
    f_done = int(beat.get("file_done") or 0)
    f_total = int(beat.get("file_total") or 0)
    f_index = int(beat.get("file_index") or 0)
    title = f"{_esc(name) or '—'}" + (f" &nbsp; <span style='color:#94a3b8'>{f_index}/{f_total}</span>" if f_total else "")
    s_done = int(beat.get("step_done") or 0)
    s_total = int(beat.get("step_total") or 0)
    s_pct = f"{100 * min(s_done, s_total) / s_total:.0f}%" if s_total else "—"
    elapsed = beat.get("step_elapsed")
    s_eta = None
    if s_total and s_done and elapsed:
        s_eta = float(elapsed) / s_done * max(0, s_total - s_done)
    session_done = len(stats.get("session_done") or [])
    started = stats.get("started")
    try:
        elapsed_s = max(0.0, time.time() - float(started)) if started else 0.0
    except (TypeError, ValueError):
        elapsed_s = 0.0
    session_row, total_row, avg_file, sess_avg, sess_times, file_times = _avg_rows(
        stats, session_done, elapsed_s
    )
    session_extra = rich_html(stats.get("session_rollup"))
    total_extra = rich_html(stats.get("rollup"))
    if len(sess_times) >= SESSION_ETA_MIN:
        avg = sess_avg
    elif file_times:
        avg = avg_file
    else:
        avg = None
    f_eta = avg * max(0, f_total - f_done) if (avg and f_total) else None
    vram = _vram_peak(metrics)
    settings = cfg_key(metrics) or "—"
    chips = tags_html(metrics)
    adv = (
        "<div style='margin-top:6px;padding-top:4px;border-top:1px solid #1e293b;"
        "color:#94a3b8;font-size:11px;line-height:1.45'>"
        "<div style='color:#7dd3fc;font-weight:700'>Advanced</div>"
        f"step: {_esc(beat.get('step_label') or '—')} · section: {_esc(beat.get('section') or '—')}"
        f" · tiles: {_num(metrics.get('num_tiles'))} · VRAM so far: {_num(vram or None, '{:.0f}', ' MB')}"
        f"<br>{chips}"
        f"<br>settings: <code>{_esc(settings)}</code></div>"
    )
    return (
        "<div style='border:1px solid #1e293b;border-radius:6px;padding:8px 10px;margin:4px 0;"
        "font-family:ui-monospace,monospace;color:#e2e8f0;font-size:12px'>"
        f"<div><b style='color:#7dd3fc'>{_esc(label)}</b> &nbsp; {title}</div>"
        f"<div style='margin-top:2px'>step: <b>{_esc(step)}</b></div>"
        f"<div>{glyph_bar(s_done, s_total, 30)} {s_pct} · ETA {_fmt_eta(s_eta)}</div>"
        f"<div>files {glyph_bar(f_done, f_total, 30)} {f_done}/{f_total or '—'} · ETA {_fmt_eta(f_eta)}</div>"
        + _avg_block(session_row, total_row, session_extra, total_extra)
        + f"{adv}</div>"
    )


def render_job(app_dir, heartbeat: Optional[dict] = None, now: Optional[float] = None) -> str:
    """Job card for the bucket holding the live ACTIVE_QUEUE.lock. Reads only, never writes."""
    app = Path(app_dir)
    bucket = _lock_bucket(app)
    if not bucket:
        return "<div style='color:#94a3b8;font-size:12px;padding:4px 8px'>No active job.</div>"
    beat = _heartbeat() if heartbeat is None else heartbeat
    with _STATS_LOCK:
        store = _read_store(app)
    stats = _bucket_stats(store["buckets"].get(bucket), _now(now))
    queue = {v: k for k, v in LOCK_TO_BUCKET.items()}.get(bucket, bucket)
    items = _load_queues(app).get(queue, {}).get("items") or []
    return job_card_html(bucket, beat, stats, items)


def _profiles_path(app_dir: Path) -> Path:
    return app_dir / "job_profiles.json"


def _builtin_profiles() -> Dict[str, dict]:
    clarity = {
        "note": "Current 4090 Group Therapy. Tile 320, quality 10, 10.25s chunks, RIFE 4×, groups of 5.",
        "mode": "tiny",
        "model_version": "v1.1",
        "scale": 4,
        "tile_size": 320,
        "tile_overlap": 32,
        "tiled_dit": True,
        "tiled_vae": True,
        "unload_dit": True,
        "enable_chunks": True,
        "chunk_duration": 10.25,
        "quality": 10,
        "sparse_ratio": 1.0,
        "local_range": 7,
        "kv_ratio": 3,
        "attention_mode": "sage",
        "dtype": "bf16",
        "color_fix": True,
        "batch_resize_preset": "4K-safe (auto)",
        "gt_group_size": 5,
        "gt_rife_mode": "4x Frames",
        "gt_rife_quality": 100,
        "gt_rife_streaming": False,
        "gt_export_quality": 100,
        "gt_export_max_width": 3840,
        "gt_do_upscale": True,
        "gt_do_export": True,
    }
    fast = dict(clarity)
    fast.update({
        "note": "Shorter drafts. Smaller tiles, quality 7, 2× RIFE, groups of 8. Same 4K-safe resize.",
        "tile_size": 192,
        "tile_overlap": 16,
        "quality": 7,
        "sparse_ratio": 2.0,
        "chunk_duration": 8.0,
        "gt_group_size": 8,
        "gt_rife_mode": "2x Frames",
        "gt_rife_quality": 90,
        "gt_export_quality": 84,
    })
    max_clarity = dict(clarity)
    max_clarity.update({
        "note": "Slowest clean pass. Tile 320, overlap 48, sparse 1.0, full 4× RIFE and export 100.",
        "tile_overlap": 48,
        "gt_group_size": 3,
        "sparse_ratio": 1.0,
        "quality": 10,
    })
    return {
        "4090 Group Therapy": clarity,
        "4090 Fast draft": fast,
        "4090 Max clarity": max_clarity,
    }


def _load_store(app_dir: Path) -> dict:
    path = _profiles_path(app_dir)
    data = _read_json(path)
    if not data or not isinstance(data.get("profiles"), dict) or not data["profiles"]:
        data = {"active": "", "profiles": _builtin_profiles()}
        try:
            _write_json(path, data)
        except OSError:
            pass
    data.setdefault("profiles", {})
    data.setdefault("active", "")
    return data


def list_profiles(app_dir) -> List[str]:
    store = _load_store(Path(app_dir))
    return sorted(store.get("profiles") or {})


def load_profile(app_dir, name: str) -> dict:
    store = _load_store(Path(app_dir))
    prof = (store.get("profiles") or {}).get(str(name or "").strip())
    return dict(prof) if isinstance(prof, dict) else {}


def save_profile(app_dir, name: str, values: dict, note: str = "") -> List[str]:
    title = str(name or "").strip()
    if not title:
        raise ValueError("Name the profile before saving.")
    store = _load_store(Path(app_dir))
    prof = {k: values.get(k) for k in PROFILE_KEYS if k in values}
    if note:
        prof["note"] = str(note).strip()
    elif isinstance(values.get("note"), str):
        prof["note"] = values["note"]
    store["profiles"][title] = prof
    store["active"] = title
    _write_json(_profiles_path(Path(app_dir)), store)
    _stamp_active(Path(app_dir), title)
    return sorted(store["profiles"])


def delete_profile(app_dir, name: str) -> List[str]:
    title = str(name or "").strip()
    store = _load_store(Path(app_dir))
    store["profiles"].pop(title, None)
    if store.get("active") == title:
        store["active"] = ""
        _stamp_active(Path(app_dir), "")
    _write_json(_profiles_path(Path(app_dir)), store)
    return sorted(store["profiles"])


def mark_active(app_dir, name: str) -> None:
    store = _load_store(Path(app_dir))
    store["active"] = str(name or "").strip()
    _write_json(_profiles_path(Path(app_dir)), store)
    _stamp_active(Path(app_dir), store["active"])


def _stamp_active(app_dir: Path, name: str) -> None:
    with _STATS_LOCK:
        store = _read_store(app_dir)
        store["active_profile"] = name
        try:
            _write_store(app_dir, store)
        except OSError:
            pass


def config_updates(values: dict) -> Dict[str, Any]:
    """Keys that belong in webui_config. Folder paths are left alone."""
    out = {}
    for key in PROFILE_KEYS:
        if key in values and values[key] is not None:
            out[key] = values[key]
    return out
