"""FlashVSR always runs at Normal GPU/CPU scheduling.

Multitask Idle / Below-Normal WDDM class, CPU Below-Normal, and any
nvidia-smi watt-cap attempts are retired. Pinokio is not admin, so -pl
never worked from this process anyway.

This module only restores Normal so old webui_config / UI state cannot
throttle jobs. GPU scheduling retired — always Normal.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_FULL_W: Optional[float] = None
_ATEXIT = False

_NORMAL_CPU = 0x00000020
_GPU_NORMAL = 2
_PROCESS_SET_INFORMATION = 0x0200
_PROCESS_QUERY_LIMITED = 0x1000


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


def _set_priority_handle(handle, below_normal: bool = False) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(handle, _NORMAL_CPU)
    except Exception:
        pass


def _set_gpu_sched_handle(handle, cls: int) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        gdi32 = ctypes.WinDLL("gdi32")
        fn = gdi32.D3DKMTSetProcessSchedulingPriorityClass
        fn.restype = ctypes.c_long
        fn.argtypes = [wintypes.HANDLE, ctypes.c_int]
        rc = fn(handle, int(cls))
        return rc == 0
    except Exception:
        return False


def _set_priority(below_normal: bool = False) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        _set_priority_handle(kernel32.GetCurrentProcess(), False)
    except Exception:
        pass


def _set_gpu_sched(cls: int) -> bool:
    """Set this process's WDDM GPU scheduling class. Works without admin."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32")
        return _set_gpu_sched_handle(kernel32.GetCurrentProcess(), cls)
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
    return {0: "Idle", 1: "Below-Normal", 2: "Normal"}.get(
        cls if cls is not None else _GPU_NORMAL, f"class {cls}"
    )


def _install_atexit() -> None:
    global _ATEXIT
    if _ATEXIT:
        return
    atexit.register(restore_full)
    _ATEXIT = True


def restore_full() -> str:
    """Undo any GPU/CPU scheduling this process applied. Never touches watts."""
    _install_atexit()
    _set_priority(False)
    _set_gpu_sched(_GPU_NORMAL)
    try:
        info = query_power()
        return (
            "GPU scheduling retired — always Normal. "
            f"Board watts {info['current']:.0f} W "
            f"(Afterburner/driver limit left as-is)."
        )
    except Exception as e:
        return f"GPU scheduling retired — always Normal. Watts unread ({e})."


def restore_pid(pid: int) -> str:
    """Force Normal GPU + CPU scheduling on another process (live FlashVSR)."""
    if os.name != "nt":
        return f"pid {pid}: not Windows"
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(
            _PROCESS_SET_INFORMATION | _PROCESS_QUERY_LIMITED, False, int(pid)
        )
        if not handle:
            return f"pid {pid}: OpenProcess failed ({ctypes.GetLastError()})"
        try:
            gpu_ok = _set_gpu_sched_handle(handle, _GPU_NORMAL)
            _set_priority_handle(handle, False)
        finally:
            kernel32.CloseHandle(handle)
        return (
            f"pid {pid}: GPU scheduling Normal, CPU Normal"
            + ("" if gpu_ok else " (GPU class call failed)")
        )
    except Exception as e:
        return f"pid {pid}: {e}"


def flashvsr_pids() -> List[int]:
    """Python processes whose command line is the Pinokio FlashVSR app."""
    pids: List[int] = []
    if os.name != "nt":
        return pids
    try:
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Where-Object { $_.CommandLine -match 'FlashVSR' } | "
                    "Select-Object -ExpandProperty ProcessId"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=_CREATE_NO_WINDOW,
        )
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
    except Exception:
        pass
    return pids


def restore_flashvsr_processes() -> str:
    pids = flashvsr_pids()
    if not pids:
        return restore_full()
    lines = [restore_pid(p) for p in pids]
    lines.append(restore_full())
    return " | ".join(lines)


def apply_gpu_headroom(enabled: bool = False, pct: float = 100) -> str:
    """Cap/Idle path retired. enabled/pct ignored — always Normal."""
    return restore_full()


def apply_from_config(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Ignore gpu_multitask / gpu_cap_pct. Always Normal."""
    return restore_full()


def full_power_w() -> Optional[float]:
    global _FULL_W
    if _FULL_W and _FULL_W > 0:
        return _FULL_W
    try:
        info = query_power()
        _FULL_W = max(float(info["current"]), float(info["default"]))
    except Exception:
        pass
    return _FULL_W


def status_line() -> str:
    try:
        info = query_power()
        return (
            f"GPU scheduling retired — always Normal. "
            f"Board {info['current']:.0f} W (Afterburner/driver — no watt cap from Pinokio)"
        )
    except Exception as e:
        return f"GPU scheduling retired — always Normal. Watts unread ({e})"


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] in ("--pid", "pid"):
        print(restore_pid(int(sys.argv[2])))
    else:
        print(restore_flashvsr_processes())
