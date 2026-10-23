"""Thrash alert: score 0..1 that the GPU is spilling VRAM into shared system memory.

Signals (thresholds in toolbox/thrash_config.json, defaults below):
  vram     - VRAM used / total, ramps from vram_pct_start..vram_pct_full.
  power    - only when util >= util_min_pct: watts per util-% falling from
             watts_per_util_start to watts_per_util_full (busy but starved GPU).
  slowdown - last/current step time vs rolling average of earlier steps.
  silent   - seconds since the last step finished.
thrash_score() is pure; StepTracker turns heartbeat snapshots into step timings.
Local only: no network, inline CSS, no scripts.
"""
from __future__ import annotations

import html
import json
import os
import time

DEFAULTS = {
    "vram_pct_start": 85.0, "vram_pct_full": 96.0,
    "util_min_pct": 90.0,
    "watts_per_util_start": 2.5, "watts_per_util_full": 1.0,
    "step_slowdown_start": 2.0, "step_slowdown_full": 6.0,
    "silent_start_s": 120.0, "silent_full_s": 900.0,
    "rolling_steps": 8,
    "weights": {"vram": 0.30, "power": 0.20, "slowdown": 0.25, "silent": 0.25},
    "near_at": 0.45, "thrash_at": 0.75,
}
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "thrash_config.json")


def load_config(path: str = _CONFIG_PATH) -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    except (OSError, ValueError):
        pass
    return cfg


def _ramp(value, start, full) -> float:
    if value is None or start == full:
        return 0.0
    t = (value - start) / (full - start)
    return max(0.0, min(1.0, t))


def thrash_score(r: dict, cfg: dict | None = None) -> tuple[float, list[str]]:
    """r keys (all optional): vram_used_gb, vram_total_gb, util_pct, power_w,
    step_s (current/last step seconds), avg_step_s, silent_s.
    Returns (score 0..1, short reasons for the strongest signals)."""
    cfg = cfg or DEFAULTS
    w = cfg["weights"]
    parts, reasons = {}, []
    used, total = r.get("vram_used_gb"), r.get("vram_total_gb")
    vram_pct = (used / total * 100.0) if used is not None and total else None
    parts["vram"] = _ramp(vram_pct, cfg["vram_pct_start"], cfg["vram_pct_full"])
    if parts["vram"] > 0:
        reasons.append(f"VRAM {vram_pct:.0f}%")
    util, watts = r.get("util_pct"), r.get("power_w")
    if util and watts is not None and util >= cfg["util_min_pct"]:
        wpu = watts / util
        parts["power"] = _ramp(wpu, cfg["watts_per_util_start"], cfg["watts_per_util_full"])
        if parts["power"] > 0:
            reasons.append(f"{watts:.0f} W at {util:.0f}% util")
    else:
        parts["power"] = 0.0
    step, avg = r.get("step_s"), r.get("avg_step_s")
    ratio = (step / avg) if step is not None and avg else None
    parts["slowdown"] = _ramp(ratio, cfg["step_slowdown_start"], cfg["step_slowdown_full"])
    if parts["slowdown"] > 0:
        reasons.append(f"step {ratio:.0f}x slower")
    silent = r.get("silent_s")
    parts["silent"] = _ramp(silent, cfg["silent_start_s"], cfg["silent_full_s"])
    if parts["silent"] > 0:
        reasons.append(f"{silent / 60:.0f} min since last step" if silent >= 120 else f"{silent:.0f}s since last step")
    score = sum(parts[k] * float(w.get(k, 0)) for k in parts)
    return max(0.0, min(1.0, score)), reasons


def level(score: float, cfg: dict | None = None) -> str:
    cfg = cfg or DEFAULTS
    if score >= cfg["thrash_at"]:
        return "thrash"
    return "near" if score >= cfg["near_at"] else "low"


class StepTracker:
    """Feeds on busy_heartbeat.snapshot(); yields step_s, avg_step_s, silent_s."""

    def __init__(self, rolling: int = 8):
        self.rolling = max(2, int(rolling))
        self.key = None
        self.last_t = None
        self.times: list[float] = []

    def update(self, snap: dict, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        done = snap.get("step_done")
        key = (snap.get("file_name"), snap.get("step_label"), snap.get("step_total"))
        if done is None:
            self.key, self.last_t, self.times = None, None, []
            return {}
        if key != self.key:
            self.key, self.last_t, self.times, self.done = key, now, [], done
        elif done != self.done:
            if done > self.done and self.last_t is not None:
                self.times.append((now - self.last_t) / (done - self.done))
                self.times = self.times[-self.rolling:]
            self.done, self.last_t = done, now
        silent = now - self.last_t
        out = {"silent_s": silent}
        if len(self.times) >= 2:
            prev = self.times[:-1]
            avg = sum(prev) / len(prev)
            out["avg_step_s"] = avg
            out["step_s"] = max(self.times[-1], silent)
        return out


_STYLE = """
<style>
#fv-thrash{position:fixed;inset:0;pointer-events:none;z-index:9999}
#fv-thrash .e{position:absolute;opacity:.12;
 background:repeating-linear-gradient(45deg,#000 0 10px,#facc15 10px 20px);
 transition:width 6s linear,height 6s linear;animation:fvt-cyc 6s ease-in-out infinite}
#fv-thrash .t{top:0;left:0;height:6px}#fv-thrash .r{top:0;right:0;width:6px}
#fv-thrash .b{bottom:0;right:0;height:6px}#fv-thrash .l{bottom:0;left:0;width:6px}
@keyframes fvt-cyc{0%,100%{opacity:.06}50%{opacity:.18}}
#fv-thrash.on .e{opacity:.9;animation:fvt-flick .25s steps(1) infinite}
@keyframes fvt-flick{0%{background:repeating-linear-gradient(45deg,#000 0 10px,#facc15 10px 20px)}
 50%{background:repeating-linear-gradient(45deg,#dc2626 0 10px,#fff 10px 20px)}}
#fv-thrash .why{position:absolute;top:10px;left:50%;transform:translateX(-50%);
 background:#7f1d1d;color:#fff;font:600 13px/1.3 system-ui,sans-serif;padding:4px 10px;border-radius:4px}
@media (prefers-reduced-motion: reduce){#fv-thrash .e{animation:none!important;transition:none}
 #fv-thrash.on .e{background:repeating-linear-gradient(45deg,#dc2626 0 10px,#fff 10px 20px)}}
</style>"""


def render_overlay(score: float, reasons: list[str], cfg: dict | None = None, muted: bool = False) -> str:
    """Border fills top -> right -> bottom -> left as score rises toward thrash_at."""
    cfg = cfg or DEFAULTS
    if muted or score <= 0:
        return ""
    lvl = level(score, cfg)
    fill = min(1.0, score / cfg["thrash_at"]) * 4.0
    seg = [max(0.0, min(1.0, fill - i)) * 100 for i in range(4)]
    edges = (f"<div class='e t' style='width:{seg[0]:.1f}%'></div>"
             f"<div class='e r' style='height:{seg[1]:.1f}%'></div>"
             f"<div class='e b' style='width:{seg[2]:.1f}%'></div>"
             f"<div class='e l' style='height:{seg[3]:.1f}%'></div>")
    why = ""
    if lvl == "thrash":
        why = f"<div class='why'>GPU thrash? {html.escape(', '.join(reasons[:3]) or 'score high')}</div>"
    cls = " class='on'" if lvl == "thrash" else ""
    return (f"{_STYLE}<div id='fv-thrash'{cls} data-level='{lvl}' "
            f"data-score='{score:.2f}' role='status'>{edges}{why}</div>")
