"""
Persistent work queues for FlashVSR+ (video / image / toolbox).

- Add files anytime; list survives restarts
- Start / Resume processes pending only
- Soft-stop after current item
- Track progress: file X of Y
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv", ".gif", ".ts", ".mts", ".m2ts"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

ST_PENDING = "pending"
ST_RUNNING = "running"
ST_DONE = "done"
ST_FAILED = "failed"

# Human labels for the exclusive run lock
QUEUE_LABELS = {
    "video": "FlashVSR video queue",
    "image": "FlashVSR image queue",
    "toolbox": "Toolbox post-process queue",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ExclusiveQueueLock:
    """
    Only one work queue may process at a time (video / image / toolbox).
    File-based so the UI can report which queue is active; soft-stop still
    uses each queue's own STOP flag for the active job only.
    """

    def __init__(self, app_dir: str):
        self.root = Path(app_dir) / "outputs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / "ACTIVE_QUEUE.lock"

    def status(self) -> Optional[Dict[str, Any]]:
        if not self.lock_path.is_file():
            return None
        try:
            data = json.loads(self.lock_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    def active_name(self) -> Optional[str]:
        st = self.status()
        return (st or {}).get("queue") if st else None

    def _pid_alive(self, pid: Any) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            # Windows: os.kill(pid, 0) raises if process missing
            os.kill(pid, 0)
            return True
        except (OSError, SystemError):
            return False
        except Exception:
            return False

    def try_acquire(self, queue_name: str) -> Tuple[bool, str]:
        """
        Try to become the sole running queue.
        Returns (ok, message). If not ok, message explains who is busy.
        """
        current = self.status()
        if current:
            who = current.get("queue") or "unknown"
            label = QUEUE_LABELS.get(who, who)
            started = current.get("started") or "?"
            pid = current.get("pid")
            # Crash recovery: clear lock if the process that held it is gone
            if pid is not None and not self._pid_alive(pid):
                self.release()
                current = None
            elif who == queue_name and pid == os.getpid():
                # Same process re-entry after partial failure — take over
                pass
            elif current is not None:
                return (
                    False,
                    f"⏸ Another queue is already running: <b>{label}</b> "
                    f"(since {started}). Wait for it to finish or use "
                    f"<b>Stop After Current</b> on that queue, then start this one. "
                    f"Only one queue runs at a time (your selection).",
                )
        payload = {
            "queue": queue_name,
            "label": QUEUE_LABELS.get(queue_name, queue_name),
            "started": _now_iso(),
            "pid": os.getpid(),
        }
        try:
            _atomic_write_json(self.lock_path, payload)
            return True, f"Running exclusively: {payload['label']}"
        except OSError as e:
            return False, f"Could not acquire queue lock: {e}"

    def release(self, queue_name: Optional[str] = None) -> None:
        """Release lock if held (optionally only if still owned by queue_name)."""
        try:
            if not self.lock_path.is_file():
                return
            if queue_name:
                st = self.status()
                if st and st.get("queue") and st.get("queue") != queue_name:
                    return
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def status_html_snippet(self) -> str:
        st = self.status()
        if not st:
            return (
                "<div style='margin-top:6px;font-size:0.85em;color:#86efac;"
                "background:#14352a;border:1px solid #166534;padding:6px;border-radius:6px;'>"
                "No queue running — you can start video, image, or toolbox.</div>"
            )
        label = st.get("label") or st.get("queue") or "queue"
        return (
            f"<div style='margin-top:6px;font-size:0.85em;color:#fbbf24;background:#3d2e0a;"
            f"border:1px solid #854d0e;padding:6px;border-radius:6px;'>"
            f"🔒 Active now: <b>{label}</b> (started {st.get('started', '?')}). "
            f"Only one queue at a time.</div>"
        )


def _norm(path: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(Path(path).resolve())))
    except OSError:
        return os.path.normcase(os.path.normpath(str(path)))


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".queue_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".txt_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text if text.endswith("\n") else text + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


class FlashVSRWorkQueue:
    """Disk-backed queue: video upscale, image upscale, or toolbox post-process."""

    def __init__(
        self,
        app_dir: str,
        *,
        name: str = "video",
        extensions: Optional[Set[str]] = None,
        label: str = "FlashVSR work queue",
    ):
        self.app_dir = Path(app_dir)
        self.name = name
        self.label = label
        self.extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or VIDEO_EXTS)}
        self.root = self.app_dir / "outputs" / f"work_queue_{name}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue_path = self.root / "queue.json"
        self.stop_flag_path = self.root / "STOP_AFTER_CURRENT.flag"
        self.status_path = self.root / "STATUS.txt"

    def _empty(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "created": _now_iso(),
            "updated": _now_iso(),
            "batch_output_dir": "",
            "completed_dir": "",
            "items": [],
        }

    def load(self) -> Dict[str, Any]:
        if not self.queue_path.is_file():
            return self._empty()
        try:
            data = json.loads(self.queue_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "items" not in data:
                return self._empty()
            return data
        except (json.JSONDecodeError, OSError):
            return self._empty()

    def save(self, data: Dict[str, Any]) -> None:
        data["updated"] = _now_iso()
        data["name"] = self.name
        _atomic_write_json(self.queue_path, data)
        self._write_status_txt(data)

    def _write_status_txt(self, data: Dict[str, Any]) -> None:
        counts = self.counts(data)
        lines = [
            f"{self.label}  updated {data.get('updated', '')}",
            f"Total {counts['total']}  |  done {counts['done']}  |  failed {counts['failed']}  |  pending {counts['pending']}  |  running {counts['running']}",
            f"Batch/output: {data.get('batch_output_dir') or '(not started yet)'}",
            f"Completed: {data.get('completed_dir') or '—'}",
            f"Stop after current: {'YES' if self.stop_requested() else 'no'}",
            "-" * 56,
        ]
        for i, it in enumerate(data.get("items") or []):
            mark = {
                ST_DONE: "OK",
                ST_FAILED: "FAIL",
                ST_RUNNING: "RUN",
                ST_PENDING: "WAIT",
            }.get(it.get("status"), "?")
            name = Path(it.get("path", "")).name
            lines.append(f"[{mark:4}] #{i + 1:03d} {name}")
            if it.get("error"):
                lines.append(f"         ERR: {it['error']}")
        try:
            _atomic_write_text(self.status_path, "\n".join(lines))
        except OSError:
            pass

    @staticmethod
    def counts(data: Dict[str, Any]) -> Dict[str, int]:
        items = data.get("items") or []
        c = {"total": len(items), "pending": 0, "running": 0, "done": 0, "failed": 0}
        for it in items:
            st = it.get("status", ST_PENDING)
            if st in c:
                c[st] += 1
            else:
                c["pending"] += 1
        return c

    def request_stop(self) -> str:
        try:
            self.stop_flag_path.write_text("stop\n", encoding="utf-8")
        except OSError as e:
            return f"⚠️ Could not set stop flag: {e}"
        return (
            "⏹ Stop requested — current item will finish, then the queue pauses. "
            "Click Start / Resume when ready."
        )

    def clear_stop(self) -> None:
        try:
            if self.stop_flag_path.is_file():
                self.stop_flag_path.unlink()
        except OSError:
            pass

    def stop_requested(self) -> bool:
        return self.stop_flag_path.is_file()

    def add_paths(self, paths: Sequence[str]) -> Tuple[int, int]:
        data = self.load()
        existing = {_norm(it["path"]) for it in data.get("items") or [] if it.get("path")}
        added = 0
        skipped = 0
        for p in paths:
            if not p:
                continue
            try:
                ap = str(Path(p).resolve())
            except OSError:
                ap = str(p)
            if not os.path.isfile(ap):
                continue
            if Path(ap).suffix.lower() not in self.extensions:
                continue
            key = _norm(ap)
            if key in existing:
                skipped += 1
                continue
            data.setdefault("items", []).append(
                {
                    "path": ap,
                    "status": ST_PENDING,
                    "output": None,
                    "error": None,
                    "added": _now_iso(),
                    "finished": None,
                }
            )
            existing.add(key)
            added += 1
        if added:
            self.save(data)
        else:
            self._write_status_txt(data)
        return added, skipped

    def add_folder(self, folder: str, *, recursive: bool = False) -> Tuple[int, int]:
        if not folder or not os.path.isdir(folder):
            return 0, 0
        paths: List[str] = []
        root = Path(folder)
        if recursive:
            for f in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
                if f.is_file() and f.suffix.lower() in self.extensions:
                    # skip nested done/archive folders
                    if any(part.lower() in {"done", "archive", "failed", "completed"} for part in f.parts):
                        continue
                    paths.append(str(f))
        else:
            for f in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if f.is_file() and f.suffix.lower() in self.extensions:
                    paths.append(str(f))
        return self.add_paths(paths)

    def clear_done(self) -> int:
        data = self.load()
        before = len(data.get("items") or [])
        data["items"] = [it for it in (data.get("items") or []) if it.get("status") != ST_DONE]
        removed = before - len(data["items"])
        self.save(data)
        return removed

    def clear_all(self) -> int:
        data = self.load()
        n = len(data.get("items") or [])
        data["items"] = []
        data["batch_output_dir"] = ""
        data["completed_dir"] = ""
        self.clear_stop()
        self.save(data)
        return n

    def requeue_failed(self) -> int:
        data = self.load()
        n = 0
        for it in data.get("items") or []:
            if it.get("status") == ST_FAILED:
                it["status"] = ST_PENDING
                it["error"] = None
                it["finished"] = None
                n += 1
        if n:
            self.save(data)
        return n

    def start_new_completed_dir(self, output_root: str, *, prefix: str = "batch") -> str:
        """Fresh completed folder for this Start/Resume session."""
        data = self.load()
        name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
        batch_dir = os.path.join(output_root, name)
        os.makedirs(batch_dir, exist_ok=True)
        completed_dir = os.path.join(batch_dir, "completed")
        os.makedirs(completed_dir, exist_ok=True)
        data["batch_output_dir"] = batch_dir
        data["completed_dir"] = completed_dir
        data["run_started"] = _now_iso()
        self.save(data)
        return completed_dir

    def set_fixed_completed_dir(self, completed_dir: str) -> str:
        """Use a fixed handoff folder (e.g. Ready for Toolbox) as completed destination."""
        os.makedirs(completed_dir, exist_ok=True)
        data = self.load()
        data["batch_output_dir"] = completed_dir
        data["completed_dir"] = completed_dir
        data["run_started"] = _now_iso()
        self.save(data)
        return completed_dir

    def ensure_batch_dir(self, output_root: str) -> str:
        return self.start_new_completed_dir(output_root)

    def get_completed_dir(self) -> str:
        data = self.load()
        completed = (data.get("completed_dir") or "").strip()
        if completed and os.path.isdir(completed):
            return completed
        return ""

    def set_item_status(
        self,
        path: str,
        status: str,
        *,
        output: Optional[str] = None,
        error: Optional[str] = None,
        new_path: Optional[str] = None,
    ) -> None:
        data = self.load()
        key = _norm(path)
        for it in data.get("items") or []:
            if _norm(it.get("path", "")) == key:
                it["status"] = status
                if new_path:
                    it["path"] = new_path
                if output is not None:
                    it["output"] = output
                # Clear error on success / re-open; only set when explicitly provided
                if status in (ST_DONE, ST_PENDING) and error is None:
                    it["error"] = None
                elif error is not None:
                    it["error"] = error
                if status in (ST_RUNNING, ST_PENDING):
                    it["finished"] = None
                if status == ST_RUNNING:
                    it["started_at"] = _now_iso()
                if status in (ST_DONE, ST_FAILED):
                    it["finished"] = _now_iso()
                break
        self.save(data)

    def reconcile_missing_sources(
        self,
        *,
        find_output,
        find_relocated,
    ) -> Tuple[int, int]:
        """
        Repair pending/failed rows whose source path no longer exists.

        find_output(path) -> optional existing deliverable path
        find_relocated(path) -> optional new location of the source file

        Returns (marked_done, path_updated).
        """
        data = self.load()
        items = list(data.get("items") or [])
        done_n = 0
        fix_n = 0
        changed = False

        for it in items:
            status = it.get("status")
            if status not in (ST_PENDING, ST_FAILED):
                continue
            path = (it.get("path") or "").strip()
            if not path:
                continue
            if os.path.isfile(path):
                continue

            # Prefer proving work already finished (export / handoff exists)
            out = None
            try:
                out = find_output(path)
            except Exception:
                out = None
            if out and os.path.isfile(out):
                it["status"] = ST_DONE
                it["output"] = out
                it["error"] = None
                it["finished"] = _now_iso()
                it["reconciled"] = _now_iso()
                it["reconcile_note"] = "source missing; existing output found"
                done_n += 1
                changed = True
                continue

            relocated = None
            try:
                relocated = find_relocated(path)
            except Exception:
                relocated = None
            if relocated and os.path.isfile(relocated):
                # Archived into a done/ folder usually means prior success without
                # a status write — mark done rather than reprocessing.
                parent_name = Path(relocated).parent.name.lower()
                if parent_name == "done":
                    it["status"] = ST_DONE
                    it["path"] = relocated
                    if not it.get("output"):
                        it["output"] = None
                    it["error"] = None
                    it["finished"] = _now_iso()
                    it["reconciled"] = _now_iso()
                    it["reconcile_note"] = "source already archived in done/"
                    done_n += 1
                else:
                    it["path"] = relocated
                    it["status"] = ST_PENDING
                    it["error"] = None
                    it["finished"] = None
                    it["reconciled"] = _now_iso()
                    it["reconcile_note"] = f"path updated → {relocated}"
                    fix_n += 1
                changed = True
                continue

            # Truly gone — keep/make failed so the queue does not spin forever
            if status == ST_PENDING or (status == ST_FAILED and not (it.get("error") or "").strip()):
                it["status"] = ST_FAILED
                it["error"] = "file not found"
                it["finished"] = it.get("finished") or _now_iso()
                changed = True

        if changed:
            data["items"] = items
            self.save(data)
        return done_n, fix_n

    def requeue_to_end(
        self,
        path: str,
        *,
        error: Optional[str] = None,
        max_attempts: int = 3,
    ) -> str:
        """
        Fail this attempt and put the source back at the END of the queue as pending
        so other jobs can run first. After max_attempts, leave as permanent failed.

        Returns: "requeued" | "failed_permanent" | "missing"
        """
        data = self.load()
        key = _norm(path)
        items = list(data.get("items") or [])
        found: Optional[Dict[str, Any]] = None
        keep: List[Dict[str, Any]] = []
        for it in items:
            if _norm(it.get("path", "")) == key:
                found = it
            else:
                keep.append(it)
        if not found:
            return "missing"

        attempts = int(found.get("attempts") or 0) + 1
        found["attempts"] = attempts
        found["requeued_at"] = _now_iso()
        found["finished"] = None
        found["output"] = None
        if error:
            found["error"] = error
            found["last_error"] = error

        if attempts >= max(1, int(max_attempts)):
            found["status"] = ST_FAILED
            found["finished"] = _now_iso()
            if not found.get("error"):
                found["error"] = f"Gave up after {attempts} attempt(s)"
            else:
                found["error"] = f"{found['error']} (gave up after {attempts} attempt(s))"
            keep.append(found)  # keep at end for visibility
            data["items"] = keep
            self.save(data)
            return "failed_permanent"

        found["status"] = ST_PENDING
        # Source file stays where it is (inbox); only queue order changes
        keep.append(found)
        data["items"] = keep
        self.save(data)
        return "requeued"

    def reset_stuck_running(self, *, to_end: bool = True, max_attempts: int = 3) -> int:
        """
        Items left as 'running' after crash/stop are re-queued.
        Default: move them to the END so they do not immediately block the queue again.
        """
        data = self.load()
        items = list(data.get("items") or [])
        stuck = [it for it in items if it.get("status") == ST_RUNNING]
        if not stuck:
            return 0

        if not to_end:
            for it in stuck:
                it["status"] = ST_PENDING
                it["error"] = it.get("error") or "Interrupted — requeued"
                it["attempts"] = int(it.get("attempts") or 0) + 1
                it["requeued_at"] = _now_iso()
                it["finished"] = None
            self.save(data)
            return len(stuck)

        others = [it for it in items if it.get("status") != ST_RUNNING]
        n = 0
        for it in stuck:
            n += 1
            attempts = int(it.get("attempts") or 0) + 1
            it["attempts"] = attempts
            it["requeued_at"] = _now_iso()
            it["finished"] = None
            it["output"] = None
            msg = "Stuck/interrupted while running — moved to end of queue"
            it["error"] = msg
            it["last_error"] = msg
            if attempts >= max(1, int(max_attempts)):
                it["status"] = ST_FAILED
                it["finished"] = _now_iso()
                it["error"] = f"{msg} (gave up after {attempts} attempt(s))"
            else:
                it["status"] = ST_PENDING
            others.append(it)
        data["items"] = others
        self.save(data)
        return n

    def pending_items(self) -> List[Dict[str, Any]]:
        data = self.load()
        return [it for it in (data.get("items") or []) if it.get("status") == ST_PENDING]

    def all_items(self) -> List[Dict[str, Any]]:
        return list(self.load().get("items") or [])

    def index_of(self, path: str) -> Tuple[int, int]:
        items = self.all_items()
        total = len(items)
        key = _norm(path)
        for i, it in enumerate(items):
            if _norm(it.get("path", "")) == key:
                return i + 1, total
        return 0, total

    def status_html(self, note: str = "") -> str:
        data = self.load()
        c = self.counts(data)
        stop = self.stop_requested()
        batch = data.get("batch_output_dir") or "—"
        completed = data.get("completed_dir") or batch
        running = next(
            (it for it in (data.get("items") or []) if it.get("status") == ST_RUNNING),
            None,
        )
        current_line = ""
        if running:
            idx, total = self.index_of(running.get("path", ""))
            current_line = (
                f"<div style='margin-top:6px;font-weight:600;color:#7dd3fc;'>"
                f"▶ Now: <b>{idx}/{total}</b> — {Path(running.get('path','')).name}"
                f"</div>"
            )

        pending_preview = []
        for it in data.get("items") or []:
            if it.get("status") == ST_PENDING:
                pending_preview.append(Path(it.get("path", "")).name)
            if len(pending_preview) >= 8:
                break
        more = max(0, c["pending"] - len(pending_preview))
        preview = ""
        if pending_preview:
            preview = (
                "<div style='margin-top:8px;font-size:0.85em;color:#94a3b8;'>"
                "<b style='color:#7dd3fc;'>Next up:</b> "
                + ", ".join(pending_preview)
                + (f" … +{more} more" if more else "")
                + "</div>"
            )

        note_html = (
            f"<div style='margin-top:8px;padding:6px;background:#3d2e0a;border:1px solid #854d0e;"
            f"border-radius:6px;color:#fbbf24;'>{note}</div>"
            if note
            else ""
        )
        stop_badge = (
            "<span style='background:#3f1d1d;color:#fca5a5;border:1px solid #7f1d1d;"
            "padding:2px 8px;border-radius:4px;font-size:0.85em;'>"
            "⏹ stop after current</span>"
            if stop
            else ""
        )

        # Dark panel — matches Interstellar / dark Gradio (readable light text on dark bg)
        return f"""
<div style="padding:10px;background:#0f1419;border:1px solid #2d3748;border-radius:8px;font-size:0.9em;color:#e2e8f0;">
  <div style="font-weight:600;margin-bottom:6px;color:#7dd3fc;">{self.label}</div>
  <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;color:#e2e8f0;">
    <span><b style="color:#86efac;">{c['done']}</b> done</span>
    <span><b style="color:#fca5a5;">{c['failed']}</b> failed</span>
    <span><b style="color:#fbbf24;">{c['pending']}</b> pending</span>
    <span><b style="color:#7dd3fc;">{c['total']}</b> total</span>
    {stop_badge}
  </div>
  {current_line}
  <div style="margin-top:6px;font-size:0.85em;color:#94a3b8;">Output / handoff: <code style="color:#cbd5e1;background:#1a202c;padding:1px 4px;border-radius:3px;">{completed}</code></div>
  <div style="margin-top:4px;font-size:0.8em;color:#64748b;">Status: <code style="color:#94a3b8;background:#1a202c;padding:1px 4px;border-radius:3px;">{self.status_path}</code></div>
  {preview}
  {note_html}
</div>
"""
