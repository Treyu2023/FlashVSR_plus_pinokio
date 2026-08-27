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
                continue
            silent = time.time() - last
            if silent < self.interval:
                continue
            total = time.time() - t0
            ts = time.strftime("%H:%M:%S")
            line = (
                f"\n[{ts}] [FlashVSR] still busy — {phase} "
                f"(job {total:.0f}s, {silent:.0f}s since last step; GPU work, not frozen)"
            )
            try:
                print(line, flush=True)
            except Exception:
                pass


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
    """tqdm that also emits a newline snapshot so Pinokio logs move."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 1.0)
        kwargs.setdefault("ncols", 88)
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("dynamic_ncols", False)
        super().__init__(*args, **kwargs)
        self._last_hb = 0.0
        if not self.disable:
            self._emit(force=True, why="start")

    def _emit(self, force: bool = False, why: str = "") -> None:
        if self.disable:
            return
        now = time.time()
        if not force and (now - self._last_hb) < 5.0:
            WATCH.ping(self.desc or "working")
            return
        self._last_hb = now
        total = self.total
        n = self.n
        elapsed = now - self.start_t if self.start_t else 0.0
        rate = (n / elapsed) if elapsed > 0 and n else 0.0
        desc = (self.desc or "working").strip() or "working"
        if desc.startswith("[FlashVSR]"):
            desc = desc[len("[FlashVSR]"):].strip() or "working"
        if total:
            pct = 100.0 * n / total
            remain = ((total - n) / rate) if rate else 0.0
            busy(
                f"{desc}: {n}/{total} ({pct:.0f}%) {rate:.2f}/s "
                f"elapsed {elapsed:.0f}s ETA {remain:.0f}s"
            )
        else:
            busy(f"{desc}: {n} elapsed {elapsed:.0f}s")

    def update(self, n=1):
        result = super().update(n)
        done = bool(self.total) and self.n >= self.total
        self._emit(force=done)
        return result

    def close(self):
        if not self.disable:
            self._emit(force=True, why="done")
        return super().close()
