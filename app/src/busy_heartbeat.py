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


# One 100-piece bar per status line. Each kind of work has its own marks so a
# tiles row, a DiT row, and a decode row are obvious without reading the label.
# done, empty, (work frame A, work frame B)
_PIE_SLOTS = 100
_STYLES = {
    "tiles":  ("█", "░", ("▓", "▒")),
    "dit":    ("●", "○", ("◉", "◈")),
    "decode": ("■", "□", ("▣", "▢")),
    "save":   ("▰", "▱", ("▶", "▷")),
    "stitch": ("✚", "·", ("✦", "✧")),
    "chunks": ("❚", "╌", ("▮", "▯")),
    "frames": ("▲", "△", ("▴", "▵")),
    "image":  ("◆", "⋄", ("◇", "◈")),
    "work":   ("#", "-", (">", "<")),
}
_SPIN = ("◐", "◓", "◑", "◒")

_prog_lock = threading.Lock()
_prog = {"done": None, "total": None, "label": "", "pulse": 0, "start_t": None}
_session = {
    "done": None,
    "total": None,
    "index": None,
    "name": "",
    "section": "",
}


def _short_file(name: str, limit: int = 36) -> str:
    base = os.path.basename(name or "").strip()
    if len(base) <= limit:
        return base
    stem, ext = os.path.splitext(base)
    keep = max(8, limit - len(ext) - 1)
    return stem[:keep] + "…" + ext


def _emit_raw(text: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"\n[{ts}] [FlashVSR] {text}"
    try:
        print(line, flush=True)
    except Exception:
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except Exception:
            pass
    tick_stop_banner()


def chunk_tenths(duration, fps, chunk_seconds) -> float | None:
    """Work in this clip, in chunk-lengths, before display rounding.

    Matches the cutter: a chunk is `chunk_seconds` (10.25). A tail shorter than
    21 frames is pulled into the previous piece, so a 10.02s file stays one
    chunk. Divide by the chunk size; the header rounds that to tenths.
    """
    try:
        step = float(chunk_seconds)
    except (TypeError, ValueError):
        step = 10.25
    if step <= 0:
        step = 10.25
    try:
        dur = float(duration or 0)
    except (TypeError, ValueError):
        return None
    if dur <= 0:
        return None
    try:
        fps_v = float(fps or 0)
    except (TypeError, ValueError):
        fps_v = 0.0
    if fps_v <= 1:
        fps_v = 30.0
    min_tail = max(21.0 / fps_v, 0.05)
    if dur <= step + 1e-6:
        return dur / step
    units = 0.0
    t = 0.0
    end = dur
    while t < end - 1e-6:
        next_t = min(t + step, end)
        remaining_after = end - next_t
        if 0 < remaining_after < min_tail:
            next_t = end
        this = next_t - t
        if this < min_tail and units > 0:
            units += this / step
            break
        units += this / step
        if next_t >= end - 1e-6:
            break
        t = next_t
    return units


def _fmt_tenths(value) -> str:
    return f"{round(float(value), 1):.1f}"


def _file_header(
    done: int, total: int, index: int, name: str, section: str,
    chunks=None, chunks_left=None,
) -> str:
    """One line per file. Files left, then leftover chunks, then this file.

    ÇÇleft is the queue still unfinished, in tenths of a chunk.
    ÇÇhunks is this file only. A 10.02s clip at 10.25s is 1.0.
    """
    total_i = max(1, int(total))
    done_i = max(0, int(done))
    index_i = int(index or 0)
    left = max(0, total_i - done_i)
    bits = []
    if index_i:
        bits.append(f"file {index_i}/{total_i}")
    else:
        bits.append(f"file —/{total_i}")
    bits.append(f"left {left}")
    show_left_chunks = chunks_left is not None
    if show_left_chunks and chunks is not None:
        try:
            if abs(float(chunks_left) - float(chunks)) < 0.05:
                show_left_chunks = False
        except (TypeError, ValueError):
            pass
    if show_left_chunks:
        bits.append(f"ÇÇleft={_fmt_tenths(chunks_left)}")
    if chunks is not None:
        bits.append(f"ÇÇhunks={_fmt_tenths(chunks)}")
    if section:
        bits.append(section)
    if name:
        bits.append(name)
    return "« " + "  ·  ".join(bits) + " »"


def set_session(
    done: int, total: int, index: int, name: str = "", section: str = "",
    chunks=None, chunks_left=None,
) -> None:
    """Print the file line once. Later rows for this file are only the work bar.

    done = files already finished, total = queued, index = 1-based file now running.
    chunks is this file. chunks_left is every file not finished yet, same units.
    """
    short = _short_file(name)
    section = (section or "").strip()
    index_i = int(index) if index else None

    def _f(value):
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    chunks_f = _f(chunks)
    left_f = _f(chunks_left)
    with _prog_lock:
        changed = (index_i, short) != (_session["index"], _session["name"])
        _session["done"] = int(done)
        _session["total"] = int(total) if total else None
        _session["index"] = index_i
        _session["name"] = short
        _session["section"] = section
        total_i = _session["total"]
        done_i = _session["done"]
    if changed and total_i:
        _emit_raw(_file_header(
            done_i or 0, total_i, index_i or 0, short, section, chunks_f, left_f,
        ))


def set_section(section: str) -> str:
    """Swap the stage label. Returns the previous one so a chunk can restore it.

    A real stage change prints one line. It does not repeat the file count.
    """
    section = (section or "").strip()
    with _prog_lock:
        prev = _session["section"]
        _session["section"] = section
        have_file = bool(_session["total"])
    # Only a new chunk is worth a line. Restoring "upscale" after the chunk
    # would repeat the file header's stage.
    if have_file and section and section != (prev or "") and section.lower().startswith("chunk"):
        _emit_raw(f"› {section}")
    return prev


def clear_session() -> None:
    with _prog_lock:
        _session["done"] = None
        _session["total"] = None
        _session["index"] = None
        _session["name"] = ""
        _session["section"] = ""


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


def _style(label: str):
    key = (label or "work").lower()
    return _STYLES.get(key, _STYLES["work"])


def render_pie(done: int, total: int, pulse: int = 0, slots: int = _PIE_SLOTS, label: str = "work") -> str:
    """100 marks. Solid = finished, flashing pair = the piece running, hollow = left.

    `label` picks the glyph set (tiles, DiT, decode, save, …).
    """
    done_g, left_g, work = _style(label)
    total = max(1, int(total))
    done = min(total, max(0, int(done)))
    slots = max(1, int(slots))
    filled = int(round(slots * done / total))
    filled = min(slots, max(0, filled))
    working = done < total
    if working and filled >= slots:
        filled = slots - 1
    marks = []
    for i in range(slots):
        if i < filled:
            marks.append(done_g)
        elif working and i == filled:
            marks.append(work[pulse % 2])
        else:
            marks.append(left_g)
    return "".join(marks)


def _format_counts(done, total, start_t, now: float) -> str:
    """n/total, percent, rate, seconds per step, elapsed, ETA."""
    if done is None or start_t is None:
        return ""
    done_i = int(done)
    elapsed = max(0.0, now - float(start_t))
    rate = (done_i / elapsed) if elapsed > 0 and done_i else 0.0
    if not total:
        return f"{done_i}  elapsed {elapsed:.0f}s"
    total_i = max(1, int(total))
    pct = 100.0 * done_i / total_i
    parts = [f"{done_i}/{total_i} ({pct:.0f}%)"]
    if rate > 0:
        parts.append(f"{rate:.2f}/s")
        parts.append(f"{1.0 / rate:.1f}s/step")
        remain = max(0.0, (total_i - done_i) / rate)
        parts.append(f"elapsed {elapsed:.0f}s")
        parts.append(f"ETA {remain:.0f}s")
    else:
        parts.append(f"elapsed {elapsed:.0f}s")
        parts.append("ETA —" if done_i < total_i else "ETA 0s")
    return "  ".join(parts)


def _progress_text(done, total, label: str, pulse: int, start_t, now: float, silent: float | None = None) -> str:
    """One bar for this step, then the counts. The file line is not repeated."""
    if total:
        visual = render_pie(done or 0, total, pulse, label=label or "work")
    else:
        visual = _SPIN[pulse % len(_SPIN)]
    bits = [visual]
    if label:
        bits.append(label)
    counts = _format_counts(done, total, start_t, now)
    if counts:
        bits.append(counts)
    if silent is not None and silent >= 1:
        bits.append(f"{silent:.0f}s since last step")
    return "  ".join(bits)


def _remember_progress(done, total, label: str, start_t) -> str:
    with _prog_lock:
        _prog["done"] = done
        _prog["total"] = total
        _prog["label"] = label
        _prog["start_t"] = start_t
        _prog["pulse"] ^= 1
        pulse = _prog["pulse"]
    return _progress_text(done, total, label, pulse, start_t, time.time())


def _flash_progress(fallback_label: str, silent: float) -> str:
    """Next blink of the same pie, with the counts still on the line.

    Used when a GPU step prints nothing for a while. The old "still busy"
    sentence stayed noisy; the numbers and the seconds since the last step
    carry that same "not frozen" fact.
    """
    now = time.time()
    with _prog_lock:
        done = _prog["done"]
        total = _prog["total"]
        label = _prog["label"] or _short_label(fallback_label)
        start_t = _prog["start_t"]
        _prog["pulse"] ^= 1
        pulse = _prog["pulse"]
    return _progress_text(done, total, label, pulse, start_t, now, silent=silent)


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
            if depth <= 0:
                if stop_armed():
                    _print_stop_banner()
                continue
            silent = time.time() - last
            if silent < self.interval:
                continue
            # Reprint the pie and the counts. A quiet GPU step used to say
            # "still busy"; the seconds-since-last-step figure says the same thing.
            ts = time.strftime("%H:%M:%S")
            line = f"\n[{ts}] [FlashVSR] {_flash_progress(phase, silent)}"
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
    """Pie row plus count, percent, rate, elapsed, and ETA.

    The classic tqdm text bar is not written to the log (Pinokio only keeps
    whole lines, and a ``\\r`` bar never shows up).
    """

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
        visual = _remember_progress(self.n or 0, self.total, label, self.start_t)
        ts = time.strftime("%H:%M:%S")
        line = f"\n[{ts}] [FlashVSR] {visual}"
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
