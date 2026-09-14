"""Leave GPU time for the desktop while FlashVSR runs.

Pinokio is not admin, so nvidia-smi -pl cannot change the 4090 watt limit
from this process (Insufficient Permissions). Afterburner / NVIDIA App keep
owning board watts — this module never launches them and never calls -pl.

Multitask ON lowers this process's WDDM GPU scheduling class (Idle or
Below-Normal) so DWM/Chrome can preempt FlashVSR. Cap % picks how
aggressive that class is. CPU Below-Normal is a small extra.
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

# Windows process CPU priority
_NORMAL_CPU = 0x00000020
_BELOW_NORMAL_CPU = 0x00004000

# D3DKMT_SCHEDULINGPRIORITYCLASS
_GPU_IDLE = 0
_GPU_BELOW_NORMAL = 1
_GPU_NORMAL = 2


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
    """Current / default / min / max power limits in watts (read-only)."""
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


def _set_priority(below_normal: bool) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(),
            _BELOW_NORMAL_CPU if below_normal else _NORMAL_CPU,
        )
    except Exception:
        pass


def _set_gpu_sched(cls: int) -> bool:
    """Set this process's WDDM GPU scheduling class. Works without admin."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        gdi32 = ctypes.WinDLL("gdi32")
        fn = gdi32.D3DKMTSetProcessSchedulingPriorityClass
        fn.restype = ctypes.c_long
        fn.argtypes = [wintypes.HANDLE, ctypes.c_int]
        kernel32 = ctypes.WinDLL("kernel32")
        rc = fn(kernel32.GetCurrentProcess(), int(cls))
        return rc == 0
    except Exception:
        return False


def _get_gpu_sched() -> Optional[int]:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        gdi32 = ctypes.WinDLL("gdi32")
        fn = gdi32.D3DKMTGetProcessSchedulingPriorityClass
        fn.restype = ctypes.c_long
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_int)]
        kernel32 = ctypes.WinDLL("kernel32")
        pri = ctypes.c_int()
        rc = fn(kernel32.GetCurrentProcess(), ctypes.byref(pri))
        return int(pri.value) if rc == 0 else None
    except Exception:
        return None


def _gpu_class_name(cls: Optional[int]) -> str:
    return {
        _GPU_IDLE: "Idle",
        _GPU_BELOW_NORMAL: "Below-Normal",
        _GPU_NORMAL: "Normal",
    }.get(cls if cls is not None else _GPU_NORMAL, f"class {cls}")


def _class_for_pct(pct: float) -> int:
    if pct >= 88:
        return _GPU_BELOW_NORMAL
    return _GPU_IDLE


def _remember_full_w(info: Optional[Dict[str, float]] = None) -> float:
    global _FULL_W
    if _FULL_W and _FULL_W > 0:
        return _FULL_W
    info = info or query_power()
    _FULL_W = max(float(info["current"]), float(info["default"]))
    return _FULL_W


def _install_atexit() -> None:
    global _ATEXIT
    if _ATEXIT:
        return
    atexit.register(restore_full)
    _ATEXIT = True


def restore_full() -> str:
    """Undo GPU scheduling this process applied. Never touches nvidia-smi watts."""
    global _APPLIED
    _set_priority(False)
    _set_gpu_sched(_GPU_NORMAL)
    _APPLIED = False
    try:
        info = query_power()
        return (
            f"GPU scheduling Normal. "
            f"Board watts {info['current']:.0f} W "
            f"(Afterburner/driver limit left as-is)."
        )
    except Exception as e:
        return f"GPU scheduling Normal. Watts unread ({e})."


def apply_gpu_headroom(enabled: bool, pct: float = 90) -> str:
    """
    enabled=True  → WDDM GPU Idle/Below-Normal + Below-Normal CPU.
    enabled=False → GPU/CPU Normal. Does not change board watts.
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
    gpu_cls = _class_for_pct(pct)
    gpu_ok = _set_gpu_sched(gpu_cls)
    _set_priority(True)
    _APPLIED = True
    try:
        info = query_power()
        full = _remember_full_w(info)
        watts = (
            f"Board watts {info['current']:.0f} W "
            f"(Afterburner/driver — Pinokio is not admin so nvidia-smi cannot cap watts)."
        )
    except Exception:
        watts = "Board watts unread (Afterburner/driver still own the watt limit)."
        full = _FULL_W
    if not gpu_ok:
        return (
            f"Multitask: could not set GPU scheduling. {watts} "
            f"CPU Below-Normal only."
        )
    return (
        f"Multitask ON: GPU scheduling {_gpu_class_name(gpu_cls)} "
        f"({pct:.0f}% → desktop headroom). {watts}"
        + (f" Full-speed reference {full:.0f} W." if full else "")
    )


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
        gpu = _gpu_class_name(_get_gpu_sched())
        return (
            f"GPU sched {gpu}. Board {info['current']:.0f} W "
            f"(full {full:.0f} W, Afterburner/driver — no nvidia-smi watt cap from Pinokio)"
        )
    except Exception as e:
        gpu = _gpu_class_name(_get_gpu_sched())
        return f"GPU sched {gpu}. Watts unread ({e})"
