"""
Persistent FlashVSR video work queue.

- Add files anytime (list stays consistent across sessions)
- Start / Resume processes pending only
- Soft-stop: finish the current video, then pause (no mid-file kill)
- Track progress: file X of Y, done / failed / pending
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv", ".gif", ".ts"}

ST_PENDING = "pending"
ST_RUNNING = "running"
ST_DONE = "done"
ST_FAILED = "failed"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
    """Disk-backed queue so you can add work, stop after current, resume later."""

    def __init__(self, app_dir: str):
        self.app_dir = Path(app_dir)
        self.root = self.app_dir / "outputs" / "flashvsr_work_queue"
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue_path = self.root / "queue.json"
        self.stop_flag_path = self.root / "STOP_AFTER_CURRENT.flag"
        self.status_path = self.root / "STATUS.txt"

    def _empty(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "created": _now_iso(),
            "updated": _now_iso(),
            "batch_output_dir": "",
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
        _atomic_write_json(self.queue_path, data)
        self._write_status_txt(data)

    def _write_status_txt(self, data: Dict[str, Any]) -> None:
        counts = self.counts(data)
        lines = [
            f"FlashVSR work queue  updated {data.get('updated', '')}",
            f"Total {counts['total']}  |  done {counts['done']}  |  failed {counts['failed']}  |  pending {counts['pending']}  |  running {counts['running']}",
            f"Batch output: {data.get('batch_output_dir') or '(not started yet)'}",
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
            "⏹ Stop requested — current video will finish, then the queue pauses. "
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
        """Add videos to queue. Returns (added, skipped_dupes)."""
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
            if Path(ap).suffix.lower() not in VIDEO_EXTS:
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
            # still refresh status text
            self._write_status_txt(data)
        return added, skipped

    def add_folder(self, folder: str) -> Tuple[int, int]:
        if not folder or not os.path.isdir(folder):
            return 0, 0
        paths = []
        for f in sorted(Path(folder).iterdir(), key=lambda p: p.name.lower()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                paths.append(str(f))
        return self.add_paths(paths)

    def clear_done(self) -> int:
        data = self.load()
        before = len(data.get("items") or [])
        data["items"] = [
            it for it in (data.get("items") or []) if it.get("status") != ST_DONE
        ]
        removed = before - len(data["items"])
        self.save(data)
        return removed

    def clear_all(self) -> int:
        data = self.load()
        n = len(data.get("items") or [])
        data["items"] = []
        data["batch_output_dir"] = ""
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

    def start_new_completed_dir(self, output_root: str) -> str:
        """
        Create a fresh completed folder for this Start/Resume session
        (same idea as normal FlashVSR batch_YYYYMMDD_HHMMSS folders).
        Previous runs keep their own folders; resume does not mix into the old one.
        """
        data = self.load()
        name = f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
        batch_dir = os.path.join(output_root, name)
        os.makedirs(batch_dir, exist_ok=True)
        # Nested completed/ so outputs are easy to spot next to progress logs
        completed_dir = os.path.join(batch_dir, "completed")
        os.makedirs(completed_dir, exist_ok=True)
        data["batch_output_dir"] = batch_dir
        data["completed_dir"] = completed_dir
        data["run_started"] = _now_iso()
        self.save(data)
        return completed_dir

    def ensure_batch_dir(self, output_root: str) -> str:
        """Backward-compatible alias — always starts a new completed folder."""
        return self.start_new_completed_dir(output_root)

    def get_completed_dir(self) -> str:
        data = self.load()
        completed = (data.get("completed_dir") or "").strip()
        if completed and os.path.isdir(completed):
            return completed
        batch = (data.get("batch_output_dir") or "").strip()
        if batch and os.path.isdir(batch):
            completed = os.path.join(batch, "completed")
            os.makedirs(completed, exist_ok=True)
            data["completed_dir"] = completed
            self.save(data)
            return completed
        return ""

    def set_item_status(
        self,
        path: str,
        status: str,
        *,
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        data = self.load()
        key = _norm(path)
        for it in data.get("items") or []:
            if _norm(it.get("path", "")) == key:
                it["status"] = status
                if output is not None:
                    it["output"] = output
                if error is not None:
                    it["error"] = error
                if status in (ST_DONE, ST_FAILED):
                    it["finished"] = _now_iso()
                break
        self.save(data)

    def reset_stuck_running(self) -> int:
        """Mark interrupted 'running' items as pending (app crash / hard kill)."""
        data = self.load()
        n = 0
        for it in data.get("items") or []:
            if it.get("status") == ST_RUNNING:
                it["status"] = ST_PENDING
                it["error"] = None
                n += 1
        if n:
            self.save(data)
        return n

    def pending_items(self) -> List[Dict[str, Any]]:
        data = self.load()
        return [it for it in (data.get("items") or []) if it.get("status") == ST_PENDING]

    def all_items(self) -> List[Dict[str, Any]]:
        return list(self.load().get("items") or [])

    def index_of(self, path: str) -> Tuple[int, int]:
        """Return (1-based index, total) for path in full queue list."""
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
        completed = data.get("completed_dir") or (os.path.join(batch, "completed") if batch != "—" else "—")
        running = next(
            (it for it in (data.get("items") or []) if it.get("status") == ST_RUNNING),
            None,
        )
        current_line = ""
        if running:
            idx, total = self.index_of(running.get("path", ""))
            current_line = (
                f"<div style='margin-top:6px;font-weight:600;'>"
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
                "<div style='margin-top:8px;font-size:0.85em;color:#555;'>"
                "<b>Next up:</b> "
                + ", ".join(pending_preview)
                + (f" … +{more} more" if more else "")
                + "</div>"
            )

        note_html = (
            f"<div style='margin-top:8px;padding:6px;background:#fff3cd;border-radius:4px;'>{note}</div>"
            if note
            else ""
        )
        stop_badge = (
            "<span style='background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:4px;font-size:0.85em;'>"
            "⏹ stop after current</span>"
            if stop
            else ""
        )

        return f"""
<div style="padding:10px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;font-size:0.9em;">
  <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
    <span><b>{c['done']}</b> done</span>
    <span><b>{c['failed']}</b> failed</span>
    <span><b>{c['pending']}</b> pending</span>
    <span><b>{c['total']}</b> total</span>
    {stop_badge}
  </div>
  {current_line}
  <div style="margin-top:6px;font-size:0.85em;color:#666;">This run completed folder: <code>{completed}</code></div>
  <div style="margin-top:4px;font-size:0.85em;color:#666;">Run folder (logs): <code>{batch}</code></div>
  <div style="margin-top:4px;font-size:0.8em;color:#888;">Queue status: <code>{self.status_path}</code></div>
  {preview}
  {note_html}
</div>
"""
