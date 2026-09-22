"""Newline heartbeats so Pinokio's terminal shows FlashVSR is busy.

tqdm uses carriage returns (\\r). Pinokio's log capture only stores whole lines,
so a job can run at 100% GPU for minutes with no new terminal text. This module
prints a fresh line every few seconds and a watchdog line if a single GPU step
goes quiet.
"""
from __future__ import annotations

import os
import sys
import threading
import time

from tqdm import tqdm as _Tqdm


# Soft-stop: Pinokio log repeats a red banner every few status lines while the flag is set.
_ANSI_RED = "\033[91m"
_ANSI_RESET = "\033[0m"
_stop_check = None
_stop_was = False
_lines_since_stop_banner = 0


def set_stop_check(fn=None) -> None:
    """Queue bodies register wq.stop_requested here so heartbeats can see the flag."""
    global _stop_check, _stop_was, _lines_since_stop_banner
    _stop_check = fn
    _stop_was = False
    _lines_since_stop_banner = 0


def stop_armed() -> bool:
    try:
        return bool(_stop_check and _stop_check())
    except Exception:
        return False


def _print_stop_banner() -> None:
    ts = time.strftime("%H:%M:%S")
    msg = (
        f"{_ANSI_RED}⏹ STOP ARMED — finishing this file, then the queue pauses. "
        f"Start / Resume to continue.{_ANSI_RESET}"
    )
    line = f"\n[{ts}] [FlashVSR] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except Exception:
            pass


def tick_stop_banner() -> None:
    """Re-read the stop flag after status lines; remind in red every 5 lines."""
    global _stop_was, _lines_since_stop_banner
    armed = stop_armed()
    if not armed:
        _stop_was = False
        _lines_since_stop_banner = 0
        return
    if not _stop_was:
        _stop_was = True
        _lines_since_stop_banner = 0
        _print_stop_banner()
        return
    _lines_since_stop_banner += 1
    if _lines_since_stop_banner >= 5:
        _lines_since_stop_banner = 0
        _print_stop_banner()


def force_line_buffering() -> None:
    """Make print() show up immediately when stdout is a pipe (Pinokio)."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass
        try:
            stream.flush()
        except Exception:
            pass


# One mark per step when there are few of them (tiles, DiT windows). Big jobs
# (VAE, save) fold into a fixed ring so the line stays short.
_PIE_ONE_EACH = 32
_PIE_RING = 16
_PIE_DONE = "●"
_PIE_LEFT = "○"
_PIE_WORK = ("◉", "◈")
_SPIN = ("◐", "◓", "◑", "◒")

_prog_lock = threading.Lock()
_prog = {"done": None, "total": None, "label": "", "pulse": 0}


def _short_label(desc: str) -> str:
    d = (desc or "").strip()
    if d.startswith("[FlashVSR]"):
        d = d[len("[FlashVSR]"):].strip()
    low = d.lower().replace("...", "").strip()
    if "dit" in low:
        return "DiT"
    if "vae" in low or low == "working":
        return "decode"
    if "tile" in low:
        return "tiles"
    if "sav" in low:
        return "save"
    if "chunk" in low:
        return "chunks"
    if "frame" in low:
        return "frames"
    if "stitch" in low:
        return "stitch"
    return (low.split()[:1] or ["work"])[0][:12]


def render_pie(done: int, total: int, pulse: int = 0) -> str:
    """Solid piece = finished, flashing piece = the one running, hollow = left."""
    total = max(1, int(total))
    done = min(total, max(0, int(done)))
    if total <= _PIE_ONE_EACH:
        slots = total
        filled = done
    else:
        slots = _PIE_RING
        filled = int(round(slots * done / total))
        filled = min(slots, max(0, filled))
    working = done < total
    if working and filled >= slots:
        filled = slots - 1
    marks = []
    for i in range(slots):
        if i < filled:
            marks.append(_PIE_DONE)
        elif working and i == filled:
            marks.append(_PIE_WORK[pulse % 2])
        else:
            marks.append(_PIE_LEFT)
    return "".join(marks)


def _remember_progress(done, total, label: str) -> str:
    with _prog_lock:
        _prog["done"] = done
        _prog["total"] = total
        _prog["label"] = label
        _prog["pulse"] ^= 1
        pulse = _prog["pulse"]
    if total:
        return f"{render_pie(done, total, pulse)}  {label}"
    spin = _SPIN[pulse % len(_SPIN)]
    return f"{spin}  {label}" if label else spin


def _flash_progress(fallback_label: str) -> str:
    """Next blink of the same pie. Used when a GPU step prints nothing for a while."""
    with _prog_lock:
        done = _prog["done"]
        total = _prog["total"]
        label = _prog["label"] or _short_label(fallback_label)
        _prog["pulse"] ^= 1
        pulse = _prog["pulse"]
    if total:
        return f"{render_pie(done or 0, total, pulse)}  {label}"
    spin = _SPIN[pulse % len(_SPIN)]
    return f"{spin}  {label}" if label else spin


def busy(msg: str) -> None:
    """Print a timestamped status line that always becomes a new log row."""
    ts = time.strftime("%H:%M:%S")
    line = f"\n[{ts}] [FlashVSR] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except Exception:
            pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    WATCH.ping(msg)
    tick_stop_banner()


class BusyWatchdog:
    """If processing is active and no log line appears, say so instead of looking frozen."""

    def __init__(self, interval: float = 12.0):
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._phase = "idle"
        self._last = time.time()
        self._t0 = time.time()
        self._depth = 0
        self._stop = threading.Event()
        thread = threading.Thread(
            target=self._loop, name="flashvsr-heartbeat", daemon=True
        )
        thread.start()

    def start(self, phase: str = "working") -> None:
        with self._lock:
            self._depth += 1
            if self._depth == 1:
                self._t0 = time.time()
            self._phase = phase
            self._last = time.time()

    def ping(self, phase: str | None = None) -> None:
        with self._lock:
            self._last = time.time()
            if phase:
                self._phase = phase

    def stop(self) -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                self._phase = "idle"

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                depth = self._depth
                phase = self._phase
                last = self._last
                t0 = self._t0
            if depth <= 0:
                if stop_armed():
                    _print_stop_banner()
                continue
            silent = time.time() - last
            if silent < self.interval:
                continue
            # Reprint the pie with the working piece toggled. The old
            # "still busy" sentence is intentionally not printed.
            ts = time.strftime("%H:%M:%S")
            line = f"\n[{ts}] {_flash_progress(phase)}"
            try:
                print(line, flush=True)
            except Exception:
                pass
            tick_stop_banner()


WATCH = BusyWatchdog(interval=12.0)


class BusySpan:
    """Mark a long GPU/CPU phase and keep the watchdog talking."""

    def __init__(self, phase: str, extra: str = ""):
        self.phase = phase
        self.extra = extra
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.time()
        suffix = f" {self.extra}" if self.extra else ""
        busy(f"{self.phase}...{suffix}")
        WATCH.start(self.phase)
        return self

    def __exit__(self, exc_type, exc, _tb):
        dt = time.time() - self._t0
        if exc_type is not None:
            busy(f"{self.phase} FAILED after {dt:.1f}s: {exc}")
        else:
            busy(f"{self.phase} done ({dt:.1f}s)")
        WATCH.stop()
        return False


class HeartbeatTqdm(_Tqdm):
    """Progress as a pie row. The classic text bar is not written to the log."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 1.0)
        kwargs.setdefault("ncols", 88)
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("dynamic_ncols", False)
        super().__init__(*args, **kwargs)
        self._last_hb = 0.0
        if not self.disable:
            self._emit(force=True, why="start")

    def display(self, msg=None, pos=None):
        return None

    def _emit(self, force: bool = False, why: str = "") -> None:
        if self.disable:
            return
        now = time.time()
        if not force and (now - self._last_hb) < 5.0:
            WATCH.ping(self.desc or "working")
            return
        self._last_hb = now
        label = _short_label(self.desc or "working")
        visual = _remember_progress(self.n or 0, self.total, label)
        ts = time.strftime("%H:%M:%S")
        line = f"\n[{ts}] {visual}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        WATCH.ping(label)
        tick_stop_banner()

    def update(self, n=1):
        result = super().update(n)
        done = bool(self.total) and self.n >= self.total
        self._emit(force=done)
        return result

    def close(self):
        if not self.disable:
            self._emit(force=True, why="done")
        return super().close()
