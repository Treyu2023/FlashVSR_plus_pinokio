"""
BATCH_PROGRESS/INPUTS writers for FlashVSR+ batch runs.

write_live_batch_progress keeps BATCH_PROGRESS.json/.txt, REMAINING.txt and
INPUTS.txt current inside a batch output folder; write_batch_inputs_list
records the inputs at batch start. The old queue UI was removed in FV-UI.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest_", suffix=".json", dir=str(path.parent))
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
            f.write(text)
            if not text.endswith("\n"):
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

# ===================================================================== live batch
def write_live_batch_progress(
    batch_dir: str,
    *,
    total: int,
    index: int,
    source: str,
    status: str,
    output: Optional[str] = None,
    error: Optional[str] = None,
    all_sources: Optional[Sequence[str]] = None,
) -> None:
    """
    Append/update live progress inside a FlashVSR batch output folder.
    Safe to call on every item — never raises into the batch loop.
    """
    try:
        os.makedirs(batch_dir, exist_ok=True)
        progress_path = Path(batch_dir) / "BATCH_PROGRESS.json"
        log_path = Path(batch_dir) / "BATCH_PROGRESS.txt"
        remaining_path = Path(batch_dir) / "REMAINING.txt"
        inputs_path = Path(batch_dir) / "INPUTS.txt"

        if all_sources and not inputs_path.exists():
            _atomic_write_text(inputs_path, "\n".join(str(s) for s in all_sources) + "\n")

        if progress_path.is_file():
            try:
                data = json.loads(progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {"items": []}
        else:
            data = {
                "created": _now_iso(),
                "batch_dir": batch_dir,
                "items": [],
                "version": 2,
            }

        data["updated"] = _now_iso()
        data["total"] = total
        data["last_index"] = index
        if all_sources:
            data["all_sources"] = list(all_sources)

        entry = {
            "index": index,
            "source": source,
            "source_name": Path(source).name,
            "status": status,
            "output": output,
            "error": error,
            "time": _now_iso(),
        }
        items = [it for it in data.get("items", []) if it.get("index") != index]
        items.append(entry)
        items.sort(key=lambda x: x.get("index", 0))
        data["items"] = items
        done = sum(1 for it in items if it.get("status") == "done")
        failed = sum(1 for it in items if it.get("status") == "failed")
        data["done"] = done
        data["failed"] = failed
        data["pending_estimate"] = max(0, total - len(items))

        _atomic_write_json(progress_path, data)

        lines = [
            f"FlashVSR batch progress  updated {data['updated']}",
            f"Folder: {batch_dir}",
            f"Done {done} / tracked {len(items)} / total {total}   failed {failed}",
            "-" * 56,
        ]
        for it in items:
            mark = {"done": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(
                it["status"], it["status"]
            )
            lines.append(
                f"[{mark:4}] #{it['index']+1:03d} {it['source_name']}"
                + (f"  → {Path(it['output']).name}" if it.get("output") else "")
                + (f"  ERR: {it['error']}" if it.get("error") else "")
            )

        seen = {it["index"] for it in items}
        still_idx = [i for i in range(total) if i not in seen]
        still_paths: List[str] = []
        if all_sources:
            for i in still_idx:
                if i < len(all_sources):
                    still_paths.append(str(all_sources[i]))
        if still_idx:
            lines.append("-" * 56)
            lines.append(
                "Not yet attempted (by index): "
                + ", ".join(str(x + 1) for x in still_idx[:60])
                + ("..." if len(still_idx) > 60 else "")
            )
        # also list failed for quick re-queue
        fails = [it for it in items if it.get("status") == "failed"]
        if fails:
            lines.append("-" * 56)
            lines.append("FAILED (re-queue these):")
            for it in fails:
                lines.append(f"  {it.get('source') or it.get('source_name')}")

        _atomic_write_text(log_path, "\n".join(lines) + "\n")

        # REMAINING = not attempted + failed sources
        remaining_list: List[str] = []
        if all_sources:
            for i, src in enumerate(all_sources):
                if i not in seen:
                    remaining_list.append(str(src))
                else:
                    # find status
                    for it in items:
                        if it.get("index") == i and it.get("status") == "failed":
                            remaining_list.append(str(src))
                            break
        else:
            remaining_list = [str(it.get("source")) for it in fails if it.get("source")]
        _atomic_write_text(
            remaining_path, "\n".join(remaining_list) + ("\n" if remaining_list else "")
        )
    except Exception:
        pass


def write_batch_inputs_list(batch_dir: str, sources: Sequence[str]) -> None:
    """Call once at batch start so crashes can be imported cleanly."""
    try:
        os.makedirs(batch_dir, exist_ok=True)
        _atomic_write_text(Path(batch_dir) / "INPUTS.txt", "\n".join(str(s) for s in sources) + "\n")
    except Exception:
        pass
