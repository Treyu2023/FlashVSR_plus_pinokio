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


def _batch_progress(data: dict, items: List[dict]) -> Dict[str, Any]:
    size = int(data.get("gt_group_size") or 0)
    gid = data.get("gt_current_group")
    stage = str(data.get("gt_current_stage") or "")
    stages = [s.strip() for s in str(data.get("gt_stages") or "upscale,rife1,rife2,export").split(",") if s.strip()]
    members = [it for it in items if gid not in (None, "", "—") and it.get("gt_group") == gid]
    done = sum(1 for it in members if it.get("status") == "done")
    running = next((it for it in members if it.get("status") == "running"), None)
    if running is None:
        running = next((it for it in items if it.get("status") == "running"), None)
    stage_index = stages.index(stage) if stage in stages else -1
    return {
        "size": size or len(members),
        "group": gid if gid not in (None, "") else "—",
        "stage": stage,
        "stages": stages or list(STAGE_ORDER),
        "stage_index": stage_index,
        "batch_done": done,
        "batch_total": len(members) or (size or 0),
        "current_name": Path(str((running or {}).get("path") or "")).name,
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
    }


def _stats_path(app_dir: Path) -> Path:
    return app_dir / "outputs" / "session_board_stats.json"


# v3 stats: one record per section, keyed from the ACTIVE_QUEUE.lock queue name.
STATS_V = 3
BUCKETS = ("group", "upscale", "image", "toolbox")
LOCK_TO_BUCKET = {"group": "group", "video": "upscale", "image": "image", "toolbox": "toolbox"}
ARCHIVE_MAX = 50
SESSION_ETA_MIN = 3  # ETA uses the session avg once this many session files exist
ALLTIME_KEYS = ("lifetime_done", "file_seconds", "file_seconds_v", "step_seconds")


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


def _lock_bucket(app_dir: Path) -> Optional[str]:
    """Bucket of the queue that holds the lock. None when no lock: that tick is not recorded."""
    lock = _read_json(app_dir / "outputs" / "ACTIVE_QUEUE.lock") or {}
    return LOCK_TO_BUCKET.get(str(lock.get("queue") or ""))


def _read_store(app_dir: Path) -> dict:
    """v3 store. Flat v2/unversioned record becomes buckets['group']; corrupt/missing is empty."""
    raw = _read_json(_stats_path(app_dir)) or {}
    if raw.get("v") == STATS_V:
        buckets = raw.get("buckets") if isinstance(raw.get("buckets"), dict) else {}
        legacy = raw.get("archive") if isinstance(raw.get("archive"), list) else []
        profile = str(raw.get("active_profile") or "")
    else:
        buckets = {"group": raw} if raw else {}
        legacy = []
        profile = str(raw.get("active_profile") or "")
    buckets = {k: v for k, v in buckets.items() if k in BUCKETS and isinstance(v, dict)}
    return {"v": STATS_V, "buckets": buckets, "active_profile": profile, "legacy_archive": legacy}


def _write_store(app_dir: Path, store: dict) -> None:
    """Before the first v3 write, keep one copy of the old file. Never overwrite the .bak."""
    path = _stats_path(app_dir)
    bak = _bak_path(app_dir)
    if path.exists() and not bak.exists():
        old = _read_json(path)
        if not old or old.get("v") != STATS_V:
            try:
                bak.write_bytes(path.read_bytes())
            except OSError:
                pass
    if store.get("legacy_archive"):
        # One-time move of an in-file archive (3288a69c layout) to its own file.
        _write_archive(app_dir, _read_archive(app_dir, store))
    payload = {"v": STATS_V, "buckets": store.get("buckets") or {}, "active_profile": store.get("active_profile") or ""}
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
        rec.update({"lifetime_done": 0, "file_seconds": [], "file_seconds_v": 2, "step_seconds": []})
        store["buckets"][bucket] = rec
        _write_store(app, store)
    return f"Reset {bucket} all-time stats. Undo last reset brings them back."


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
        rec.update({"session_done": [], "session_failed": [], "session_file_seconds": [], "started": _now(now)})
        store["buckets"][bucket] = rec
        _write_store(app, store)
    return f"New {bucket} session started."


def _load_stats(app_dir: Path, now: float, bucket: str = "group") -> dict:
    return _bucket_stats(_read_store(app_dir)["buckets"].get(bucket), now)


def _add_file_seconds(stats: dict, secs: float) -> None:
    stats["file_seconds"] = (list(stats.get("file_seconds") or []) + [secs])[-40:]
    stats["session_file_seconds"] = (list(stats.get("session_file_seconds") or []) + [secs])[-500:]


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
    if name:
        key = (name, index)
        if not open_file or (open_file.get("name"), open_file.get("index")) != key:
            if open_file and open_file.get("started"):
                _add_file_seconds(stats, max(0.0, now - float(open_file["started"])))
            stats["open_file"] = {"name": name, "index": index, "started": now}
    elif open_file and open_file.get("started"):
        # No file in progress (between files, stopped, idle): close the open file
        # now so the idle gap is not counted in file_seconds.
        _add_file_seconds(stats, max(0.0, now - float(open_file["started"])))
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
    stages = batch.get("stages") or list(STAGE_ORDER)
    current = batch.get("stage") or ""
    bits = []
    seen_current = False
    for stage in stages:
        label = STAGE_LABEL.get(stage, stage)
        if stage == current:
            seen_current = True
            bits.append(f"<span style='color:#7dd3fc;font-weight:700'>▣ {_esc(label)}</span>")
        elif not seen_current and current:
            bits.append(f"<span style='color:#86efac'>■ {_esc(label)}</span>")
        else:
            bits.append(f"<span style='color:#64748b'>□ {_esc(label)}</span>")
    return " &nbsp; ".join(bits)


def _meter(title: str, bar: str, caption: str) -> str:
    return (
        "<div style='flex:1;min-width:240px;padding:8px 10px;background:#0b1220;"
        "border:1px solid #1e293b;border-radius:8px;'>"
        f"<div style='color:#7dd3fc;font-weight:700;font-size:12px;letter-spacing:.04em;'>{_esc(title)}</div>"
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
    batch = _batch_progress(data, items)
    left = counts["pending"] + counts["running"]
    session_done = len(stats.get("session_done") or [])
    session_failed = len(stats.get("session_failed") or [])
    file_times = [float(x) for x in (stats.get("file_seconds") or []) if float(x) > 1]
    avg_file = (sum(file_times) / len(file_times)) if file_times else None
    sess_times = [float(x) for x in (stats.get("session_file_seconds") or []) if float(x) > 1]
    sess_avg = (sum(sess_times) / len(sess_times)) if sess_times else None
    elapsed_s = max(0.0, now_v - float(stats.get("started") or now_v))
    elapsed_h = max(1 / 3600, elapsed_s / 3600)
    per_hour = (session_done / elapsed_h) if session_done else None
    eta_avg = sess_avg if len(sess_times) >= SESSION_ETA_MIN else avg_file
    eta = (eta_avg * max(0, left - (1 if counts["running"] else 0))) if eta_avg else None
    step_done = int(beat.get("step_done") or 0)
    step_total = int(beat.get("step_total") or 0)
    step_label = str(beat.get("step_label") or beat.get("section") or "step")
    step_times = [float(x) for x in (stats.get("step_seconds") or [])]
    current_name = batch.get("current_name") or beat.get("file_name") or ""
    profile = str(store.get("active_profile") or stats.get("active_profile") or "")
    lamp = _health_color(health["level"])
    stop = " &nbsp; <span style='color:#fca5a5;font-weight:700'>STOP ARMED</span>" if beat.get("stop_armed") else ""
    label = dict(QUEUE_ORDER).get(active, active)

    file_caption = (
        f"<b style='color:#86efac'>{counts['done']}</b> done"
        f" &nbsp; <b style='color:#fbbf24'>{left}</b> left"
        f" &nbsp; <b style='color:#e2e8f0'>{counts['total']}</b> total"
        f" &nbsp; <b style='color:#7dd3fc'>+{session_done}</b> this session"
        f" &nbsp; <b style='color:#fca5a5'>{counts['failed']}</b> failed"
    )
    batch_total = int(batch["batch_total"] or batch["size"] or 0)
    batch_caption = (
        f"size <b>{int(batch['size'] or 0)}</b>"
        f" &nbsp; group <b>{_esc(batch['group'])}</b>"
        f" &nbsp; <b>{int(batch['batch_done'])}</b>/{batch_total or '—'} settled in this batch"
    )
    if step_total:
        step_bar = glyph_bar(step_done, step_total, 32)
        step_caption = f"{_esc(step_label)} <b>{step_done}/{step_total}</b>"
    else:
        step_bar = glyph_bar(0, 0, 32)
        step_caption = "waiting for the next tile / DiT / decode mark"
    if current_name:
        step_caption += f"<br><span style='color:#94a3b8'>{_esc(current_name)}</span>"

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
        f"session {session_done} · avg {_fmt_eta(sess_avg)} · busy {_fmt_eta(sum(sess_times))} / {_fmt_eta(elapsed_s)}",
        f"all-time {int(stats.get('lifetime_done') or 0) + session_done} · avg {_fmt_eta(avg_file)}",
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
        f"<div style='margin-bottom:8px'>{_chips(queues, active)}</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
        + _meter("FILES", glyph_bar(counts["done"], counts["total"], 42), file_caption)
        + _meter(
            "BATCH",
            glyph_bar(int(batch["batch_done"]), batch_total or int(batch["size"] or 0), 24),
            batch_caption,
        )
        + _meter("STEP", step_bar, step_caption)
        + "</div>"
        f"<div style='margin-top:8px;font-size:13px;'>{_stage_pips(batch)}</div>"
        f"<div style='margin-top:6px;font-family:Consolas,ui-monospace,monospace;font-size:14px;'>"
        f"step time {spark}</div>"
        "<div style='margin-top:6px;color:#cbd5e1;font-size:12px;line-height:1.45;'>"
        + " &nbsp;·&nbsp; ".join(stat_bits)
        + "</div>"
        f"<div style='margin-top:6px;color:{lamp};font-size:13px;line-height:1.4;'>"
        f"{_esc(health['detail'])}</div>"
        "<div style='margin-top:6px;color:#64748b;font-size:11px;line-height:1.45;'>"
        "█ done &nbsp; ▓ now &nbsp; ░ left"
        " &nbsp;·&nbsp; <span style='color:#86efac'>HEALTHY</span> steps moving"
        " &nbsp;·&nbsp; <span style='color:#7dd3fc'>BUSY</span> full GPU, bar still advancing"
        " &nbsp;·&nbsp; <span style='color:#fbbf24'>PAGE / SLOW</span> commit or a long step"
        " &nbsp;·&nbsp; <span style='color:#fca5a5'>THRASH</span> frozen step, low watts, low memory bus"
        "</div>"
        "</div>"
    )


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
