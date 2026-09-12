"""Leave GPU headroom so the desktop stays usable while FlashVSR runs.

Toggle ON → nvidia-smi power cap at N% of the card's full-speed limit, and
drop this process to Below-Normal CPU priority.
Toggle OFF (or process exit) → restore the previous power limit and Normal
priority.

Task Manager can still show ~99% GPU: the card uses whatever budget it has.
The cap is watts (same idea as Afterburner power limit), which is what actually
leaves headroom for the desktop.
"""
from __future__ import annotations

import atexit
import os
import subprocess
from typing import Any, Dict, Optional

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_FULL_W: Optional[float] = None
_APPLIED = False
_ATEXIT = False

# Windows priority classes
_NORMAL = 0x00000020
_BELOW_NORMAL = 0x00004000


def _smi(*args: str, timeout: float = 8.0) -> subprocess.CompletedProcess:
    flags = _CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=flags,
    )


def query_power() -> Dict[str, float]:
    """Current / default / min / max power limits in watts."""
    r = _smi(
        "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit",
        "--format=csv,noheader,nounits",
    )
    if r.returncode != 0 or not (r.stdout or "").strip():
        raise RuntimeError((r.stderr or r.stdout or "nvidia-smi power query failed").strip())
    parts = [p.strip() for p in r.stdout.strip().split(",")]
    if len(parts) < 4:
        raise RuntimeError(f"unexpected nvidia-smi power line: {r.stdout!r}")
    cur, default, lo, hi = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    return {"current": cur, "default": default, "min": lo, "max": hi}


def _set_power_w(watts: float) -> float:
    info = query_power()
    lo, hi = info["min"], info["max"]
    target = max(lo, min(hi, round(float(watts), 2)))
    r = _smi("-pl", f"{target:.2f}")
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "nvidia-smi -pl failed").strip())
    return query_power()["current"]


def _set_priority(below_normal: bool) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(),
            _BELOW_NORMAL if below_normal else _NORMAL,
        )
    except Exception:
        pass


def _remember_full_w(info: Optional[Dict[str, float]] = None) -> float:
    global _FULL_W
    if _FULL_W and _FULL_W > 0:
        return _FULL_W
    info = info or query_power()
    # Prefer the higher of current vs default so we restore Afterburner-style
    # raised limits (this 4090 is often 463.5 W vs 450 W default).
    _FULL_W = max(float(info["current"]), float(info["default"]))
    return _FULL_W


def _install_atexit() -> None:
    global _ATEXIT
    if _ATEXIT:
        return
    atexit.register(restore_full)
    _ATEXIT = True


def restore_full() -> str:
    """Undo the cap. Safe to call when already at full speed."""
    global _APPLIED
    _set_priority(False)
    try:
        info = query_power()
        full = _remember_full_w(info)
        if abs(info["current"] - full) >= 0.5:
            now = _set_power_w(full)
        else:
            now = info["current"]
        _APPLIED = False
        return f"Full GPU: {now:.0f} W (no cap)."
    except Exception as e:
        _APPLIED = False
        return f"Could not restore GPU power limit: {e}"


def apply_gpu_headroom(enabled: bool, pct: float = 90) -> str:
    """
    enabled=True  → cap power to pct% of full-speed watts + Below-Normal CPU.
    enabled=False → restore full watts + Normal CPU.
    """
    global _APPLIED
    _install_atexit()
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        pct = 90.0
    pct = max(70.0, min(100.0, pct))
    if not enabled or pct >= 99.5:
        return restore_full()
    try:
        info = query_power()
        full = _remember_full_w(info)
        target = full * (pct / 100.0)
        now = _set_power_w(target)
        _set_priority(True)
        _APPLIED = True
        return (
            f"Multitask cap ON: {now:.0f} W "
            f"({pct:.0f}% of {full:.0f} W full). "
            f"FlashVSR CPU priority Below-Normal. "
            f"Turn off for max speed."
        )
    except Exception as e:
        _set_priority(False)
        _APPLIED = False
        return f"GPU cap failed ({e}). nvidia-smi -pl must work on this driver."


def apply_from_config(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Re-apply whatever webui_config currently says."""
    enabled = False
    pct = 90.0
    if cfg:
        raw = cfg.get("gpu_multitask", False)
        enabled = raw if isinstance(raw, bool) else str(raw).lower() == "true"
        try:
            pct = float(cfg.get("gpu_cap_pct") or 90)
        except (TypeError, ValueError):
            pct = 90.0
        saved = cfg.get("gpu_full_power_w")
        try:
            if saved:
                global _FULL_W
                _FULL_W = float(saved)
        except (TypeError, ValueError):
            pass
    return apply_gpu_headroom(enabled, pct)


def full_power_w() -> Optional[float]:
    return _FULL_W


def status_line() -> str:
    try:
        info = query_power()
        full = _FULL_W or max(info["current"], info["default"])
        return (
            f"GPU power {info['current']:.0f} W "
            f"(full {full:.0f} W, default {info['default']:.0f} W, "
            f"range {info['min']:.0f}–{info['max']:.0f})"
        )
    except Exception as e:
        return f"GPU power unknown ({e})"
