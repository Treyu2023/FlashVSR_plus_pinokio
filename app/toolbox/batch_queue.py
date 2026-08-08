"""
Batch Queue / Resume Chunks for FlashVSR+ Toolbox (maxed).

Splits a source folder into fixed-size work packs (default 20), tracks done vs
pending by scanning output folders (stage tags _1/_2/_3 + robust name match),
imports crashed FlashVSR BATCH_PROGRESS logs, prepares hardlink/copy work
folders for the next chunk, and keeps atomic manifests so a 100-file batch
that dies at #30 never loses the plot again.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from naming_utils import strip_stage

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv", ".gif", ".ts", ".mts", ".m2ts"}
DEFAULT_CHUNK = 20
MANIFEST_VERSION = 2

ST_PENDING = "pending"
ST_DONE = "done"
ST_FAILED = "failed"
ST_SKIPPED = "skipped"  # already complete when queue was built / refresh

# FlashVSR+ naming residue we strip when recovering original stems
_PROCESS_TOKENS = re.compile(
    r"(?i)("
    r"upscale(?:d)?(?:\d+k|\d+p)?|"
    r"upscaled_x\d+|"
    r"_x[24]|"
    r"flashvsr|"
    r"s_i|"
    r"chunked|"
    r"frames?_[a-z0-9x.+-]+|"
    r"exported_\d+w_\d+q|"
    r"loop_(?:loop|ping-?pong)(?:_\d+x)?|"
    r"comparison|"
    r"resized_\d+x\d+|"
    r"trim|"
    r"preprocessed|"
    r"audiofix|"
    r"combined"
    r")"
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _norm(path: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(Path(path).resolve())))
    except OSError:
        return os.path.normcase(os.path.normpath(str(path)))


def _safe_stem(path_or_name: str) -> str:
    """Normalize to a comparable stem (lowercase, no stage, no process fluff)."""
    name = os.path.basename(str(path_or_name))
    stem, _ = os.path.splitext(name)
    bare, _ = strip_stage(stem)
    bare = re.sub(r"_\d{8}[-_]\d{6}", "", bare)
    bare = re.sub(r"_\d{6}(?=_|$)", "", bare)
    # Drop UpScale4K_ style prefixes
    bare = re.sub(r"(?i)^upscale(?:\d+k|\d+p|qhd|2k-qhd)?_", "", bare)
    bare = re.sub(r"(?i)^wanvideowrapper_", "", bare)
    bare = _PROCESS_TOKENS.sub("", bare)
    bare = re.sub(r"\s+", "_", bare)
    bare = re.sub(r"[^\w.\-]+", "_", bare, flags=re.UNICODE)
    bare = re.sub(r"_+", "_", bare).strip("._-")
    return bare.lower()


def _match_keys(path_or_name: str) -> Set[str]:
    """Keys for source↔output matching.

    IMPORTANT: do NOT strip trailing _01 / _02 counters — that collapses
    hero_clip_01 and hero_clip_02 into the same key and false-matches everything.
    """
    keys: Set[str] = set()
    full = _safe_stem(path_or_name)
    if full:
        keys.add(full)
    raw = Path(str(path_or_name)).stem.lower()
    raw, _ = strip_stage(raw)
    raw = re.sub(r"\s+", "_", raw)
    if raw:
        keys.add(raw)
    return {k for k in keys if k and len(k) >= 2}


def _stem_token_in(src_stem: str, haystack: str) -> bool:
    """True if src_stem appears as a full underscore-delimited token in haystack."""
    if not src_stem or not haystack:
        return False
    if src_stem == haystack:
        return True
    # token boundaries so hero_clip_01 does not match hero_clip_010 / hero_clip_02
    return re.search(rf"(^|_){re.escape(src_stem)}(_|$)", haystack) is not None

def _stage_of_path(path: str) -> Optional[int]:
    _, stage = strip_stage(Path(path).stem)
    return stage


def _file_size(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def _list_videos(
    folder: str,
    *,
    recursive: bool = False,
    max_depth: int = 6,
    min_bytes: int = 0,
    sort_mode: str = "name",
) -> List[str]:
    root = Path(folder)
    if not root.is_dir():
        return []

    files: List[Path] = []
    if not recursive:
        files = [
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
    else:
        root_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).parts) - root_depth
            if depth > max_depth:
                dirnames.clear()
                continue
            dirnames[:] = [
                d
                for d in dirnames
                if d
                not in {
                    ".git",
                    "env",
                    "__pycache__",
                    "node_modules",
                    ".venv",
                    "batch_queues",
                    "_temp",
                }
                and not d.startswith(".")
            ]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in VIDEO_EXTS:
                    files.append(p)

    if min_bytes > 0:
        files = [p for p in files if _file_size(str(p)) >= min_bytes]

    if sort_mode == "mtime":
        files.sort(key=lambda p: (p.stat().st_mtime if p.exists() else 0, p.name.lower()))
    elif sort_mode == "size":
        files.sort(key=lambda p: (_file_size(str(p)), p.name.lower()))
    else:
        files.sort(key=lambda p: p.name.lower())

    out: List[str] = []
    for p in files:
        try:
            out.append(str(p.resolve()))
        except OSError:
            out.append(str(p))
    return out


def _walk_videos(folder: str, max_depth: int = 5) -> List[str]:
    return _list_videos(folder, recursive=True, max_depth=max_depth, sort_mode="name")


def _looks_processed(name: str) -> bool:
    n = name.lower()
    markers = (
        "upscale",
        "upscaled",
        "flashvsr",
        "frames_",
        "exported_",
        "_x2",
        "_x4",
        "s_i",
        "chunked",
        "wanvideowrapper",
    )
    return any(m in n for m in markers)


def _output_matches_source(source_path: str, output_path: str, min_stage: int) -> bool:
    """Heuristic: output is a finished product of this source at >= min_stage."""
    src_keys = _match_keys(source_path)
    if not src_keys:
        return False

    out_stem = Path(output_path).stem
    out_safe = _safe_stem(output_path)
    out_l = out_stem.lower()
    stage = _stage_of_path(output_path)

    if stage is not None and stage < min_stage:
        return False

    # Prefer exact key hits, then token-boundary containment of the full source stem.
    out_keys = _match_keys(output_path)
    shared = src_keys & out_keys
    if shared:
        best = max(len(s) for s in shared)
        if best < 3:
            return False
    else:
        hit = False
        for k in sorted(src_keys, key=len, reverse=True):
            if len(k) < 3:
                continue
            if _stem_token_in(k, out_safe) or _stem_token_in(k, out_l):
                hit = True
                break
        if not hit:
            return False

    if stage is None and min_stage >= 1:
        # unstaged: only accept if clearly processed (legacy outputs)
        if not _looks_processed(out_stem):
            return False
        return min_stage == 1

    return True


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


class BatchQueueManager:
    def __init__(self, app_dir: Optional[str] = None):
        if app_dir is None:
            app_dir = str(Path(__file__).resolve().parent.parent)
        self.app_dir = Path(app_dir)
        self.queue_root = self.app_dir / "outputs" / "batch_queues"
        self.queue_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def _qdir(self, queue_id: str) -> Path:
        return self.queue_root / queue_id

    def _manifest_path(self, queue_id: str) -> Path:
        return self._qdir(queue_id) / "manifest.json"

    def _lock_path(self, queue_id: str) -> Path:
        return self._qdir(queue_id) / ".lock"

    def _history_path(self, queue_id: str) -> Path:
        return self._qdir(queue_id) / "history.log"

    def _append_history(self, queue_id: str, message: str) -> None:
        try:
            line = f"{_now_iso()}  {message}\n"
            with open(self._history_path(queue_id), "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass

    # ------------------------------------------------------------------ lock
    def acquire_lock(self, queue_id: str, owner: str = "ui", ttl_sec: int = 3600) -> bool:
        lp = self._lock_path(queue_id)
        lp.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        if lp.exists():
            try:
                meta = json.loads(lp.read_text(encoding="utf-8"))
                if now - float(meta.get("ts", 0)) < ttl_sec and meta.get("owner") != owner:
                    return False
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        _atomic_write_json(lp, {"owner": owner, "ts": now, "iso": _now_iso()})
        return True

    def release_lock(self, queue_id: str, owner: str = "ui") -> None:
        lp = self._lock_path(queue_id)
        try:
            if lp.exists():
                meta = json.loads(lp.read_text(encoding="utf-8"))
                if meta.get("owner") in (owner, None):
                    lp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except (OSError, json.JSONDecodeError):
            try:
                lp.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------ IO
    def list_queues(self) -> List[Dict[str, Any]]:
        rows = []
        if not self.queue_root.is_dir():
            return rows
        for d in sorted(self.queue_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            man = d / "manifest.json"
            if not man.is_file():
                continue
            try:
                data = json.loads(man.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "id": data.get("id", d.name),
                    "name": data.get("name", d.name),
                    "created": data.get("created"),
                    "updated": data.get("updated"),
                    "source_folder": data.get("source_folder"),
                    "chunk_size": data.get("chunk_size"),
                    "target_stage": data.get("target_stage"),
                    "stats": self._stats(data),
                    "next_chunk": self._next_chunk_label(data),
                }
            )
        return rows

    def load(self, queue_id: str) -> Dict[str, Any]:
        path = self._manifest_path(queue_id)
        if not path.is_file():
            raise FileNotFoundError(f"Queue not found: {queue_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        # migrate v1 → v2 lightly
        data.setdefault("version", 1)
        data.setdefault("action_log", [])
        data.setdefault("options", {})
        return data

    def save(self, data: Dict[str, Any]) -> str:
        queue_id = data["id"]
        qdir = self._qdir(queue_id)
        qdir.mkdir(parents=True, exist_ok=True)
        data["version"] = MANIFEST_VERSION
        data["updated"] = _now_iso()
        path = self._manifest_path(queue_id)

        # rolling backup
        if path.is_file():
            bak_dir = qdir / "backups"
            bak_dir.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                shutil.copy2(path, bak_dir / f"manifest_{stamp}.json")
                # keep last 12 backups
                baks = sorted(bak_dir.glob("manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                for old in baks[12:]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            except OSError:
                pass

        _atomic_write_json(path, data)
        _atomic_write_text(qdir / "STATUS.txt", self.format_report(data))
        self._write_chunk_lists(data)
        self._write_csv(data)
        return str(path)

    def _stats(self, data: Dict[str, Any]) -> Dict[str, int]:
        items = data.get("items") or []
        counts = {
            ST_PENDING: 0,
            ST_DONE: 0,
            ST_FAILED: 0,
            ST_SKIPPED: 0,
            "total": len(items),
            "remaining": 0,
            "complete": 0,
        }
        for it in items:
            st = it.get("status", ST_PENDING)
            counts[st] = counts.get(st, 0) + 1
        counts["complete"] = counts.get(ST_DONE, 0) + counts.get(ST_SKIPPED, 0)
        counts["remaining"] = counts.get(ST_PENDING, 0) + counts.get(ST_FAILED, 0)
        return counts

    def _next_chunk_label(self, data: Dict[str, Any]) -> str:
        for ch in data.get("chunks") or []:
            if any(
                data["items"][i]["status"] in (ST_PENDING, ST_FAILED)
                for i in ch.get("item_indices") or []
                if i < len(data.get("items") or [])
            ):
                return ch.get("label", "?")
        return "—"

    # ------------------------------------------------------------------ index
    def _build_output_index(self, output_dirs: Sequence[str]) -> List[Dict[str, Any]]:
        """Preload all output videos with keys for fast matching."""
        index: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for od in output_dirs:
            if not od or not os.path.isdir(od):
                continue
            for op in _walk_videos(od, max_depth=6):
                n = _norm(op)
                if n in seen:
                    continue
                seen.add(n)
                index.append(
                    {
                        "path": op,
                        "keys": _match_keys(op),
                        "stage": _stage_of_path(op),
                        "stem": Path(op).stem,
                        "size": _file_size(op),
                        "mtime": os.path.getmtime(op) if os.path.isfile(op) else 0,
                    }
                )
        return index

    def find_match(
        self,
        source_path: str,
        output_index: Sequence[Dict[str, Any]],
        min_stage: int,
    ) -> Optional[str]:
        src_keys = _match_keys(source_path)
        if not src_keys:
            return None
        candidates: List[Tuple[int, float, str]] = []
        for ent in output_index:
            path = ent["path"]
            stage = ent.get("stage")
            if stage is not None and stage < min_stage:
                continue
            if not _output_matches_source(source_path, path, min_stage):
                continue
            shared = src_keys & set(ent.get("keys") or [])
            score = 0
            if shared:
                score = max(len(s) for s in shared) * 10
            else:
                # token containment — score by longest source key present
                out_l = (ent.get("stem") or "").lower()
                out_safe = _safe_stem(path)
                for k in sorted(src_keys, key=len, reverse=True):
                    if _stem_token_in(k, out_safe) or _stem_token_in(k, out_l):
                        score = len(k) * 6
                        break
            if stage is not None and stage >= min_stage:
                score += 5 + stage
            if score > 0:
                candidates.append((score, float(ent.get("mtime") or 0), path))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return candidates[0][2]

    def scan_outputs_for_item(
        self,
        source_path: str,
        output_dirs: Sequence[str],
        min_stage: int,
    ) -> Tuple[bool, Optional[str]]:
        idx = self._build_output_index(output_dirs)
        match = self.find_match(source_path, idx, min_stage)
        return (match is not None), match

    # ------------------------------------------------------------------ create
    def create_queue(
        self,
        source_folder: str,
        output_dirs: Sequence[str],
        chunk_size: int = DEFAULT_CHUNK,
        target_stage: int = 1,
        name: str = "",
        only_pending: bool = True,
        *,
        recursive: bool = False,
        min_bytes: int = 0,
        sort_mode: str = "name",
        validate: bool = True,
    ) -> Dict[str, Any]:
        source_folder = os.path.normpath(str(source_folder).strip())
        if not os.path.isdir(source_folder):
            raise ValueError(f"Source folder does not exist: {source_folder}")

        chunk_size = max(1, min(200, int(chunk_size or DEFAULT_CHUNK)))
        target_stage = int(target_stage or 1)
        if target_stage not in (1, 2, 3):
            target_stage = 1
        sort_mode = sort_mode if sort_mode in ("name", "mtime", "size") else "name"

        sources = _list_videos(
            source_folder,
            recursive=bool(recursive),
            min_bytes=int(min_bytes or 0),
            sort_mode=sort_mode,
        )
        if not sources:
            raise ValueError(f"No video files found in: {source_folder}")

        # duplicate detection by stem
        stem_counts: Dict[str, int] = defaultdict(int)
        for s in sources:
            stem_counts[_safe_stem(s)] += 1
        dup_stems = {k for k, c in stem_counts.items() if c > 1 and k}

        out_dirs = [os.path.normpath(d) for d in output_dirs if d and str(d).strip()]
        # always include source-adjacent common outputs if empty
        if not out_dirs:
            out_dirs = [str(self.app_dir / "outputs")]

        output_index = self._build_output_index(out_dirs)

        items = []
        invalid = 0
        for idx, src in enumerate(sources):
            size = _file_size(src)
            exists = os.path.isfile(src)
            if validate and (not exists or size <= 0):
                invalid += 1
                items.append(
                    {
                        "index": idx,
                        "source": src,
                        "stem": _safe_stem(src),
                        "status": ST_FAILED,
                        "output_match": None,
                        "error": "missing or zero-byte source",
                        "size": size,
                        "updated": _now_iso(),
                        "duplicate_stem": _safe_stem(src) in dup_stems,
                    }
                )
                continue

            match = self.find_match(src, output_index, target_stage)
            done = match is not None
            if done and only_pending:
                status = ST_SKIPPED
            elif done:
                status = ST_DONE
            else:
                status = ST_PENDING

            items.append(
                {
                    "index": idx,
                    "source": src,
                    "stem": _safe_stem(src),
                    "status": status,
                    "output_match": match,
                    "error": None,
                    "size": size,
                    "updated": _now_iso(),
                    "duplicate_stem": _safe_stem(src) in dup_stems,
                }
            )

        chunks = self._build_chunks(items, chunk_size)

        queue_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        work_indices = [i for i, it in enumerate(items) if it["status"] in (ST_PENDING, ST_FAILED)]
        data: Dict[str, Any] = {
            "id": queue_id,
            "version": MANIFEST_VERSION,
            "name": name.strip() or f"queue_{Path(source_folder).name}",
            "created": _now_iso(),
            "updated": _now_iso(),
            "source_folder": source_folder,
            "output_dirs": out_dirs,
            "chunk_size": chunk_size,
            "target_stage": target_stage,
            "items": items,
            "chunks": chunks,
            "current_chunk": 0,
            "options": {
                "recursive": bool(recursive),
                "min_bytes": int(min_bytes or 0),
                "sort_mode": sort_mode,
                "only_pending": bool(only_pending),
                "validate": bool(validate),
            },
            "action_log": [
                {
                    "time": _now_iso(),
                    "action": "create",
                    "detail": f"{len(sources)} sources, {len(work_indices)} to process, {len(chunks)} chunks",
                }
            ],
            "notes": (
                f"Target stage _{target_stage}. {len(sources)} sources, "
                f"{sum(1 for it in items if it['status']==ST_SKIPPED)} already done, "
                f"{sum(1 for it in items if it['status']==ST_PENDING)} pending, "
                f"{sum(1 for it in items if it['status']==ST_FAILED)} failed/invalid, "
                f"{len(chunks)} chunks × ≤{chunk_size}."
                + (f" Duplicate stems: {len(dup_stems)}." if dup_stems else "")
                + (f" Invalid sources: {invalid}." if invalid else "")
            ),
        }
        self.save(data)
        self._append_history(queue_id, f"CREATE {data['notes']}")
        return data

    def _build_chunks(self, items: List[Dict[str, Any]], chunk_size: int) -> List[Dict[str, Any]]:
        work_indices = [
            i for i, it in enumerate(items) if it.get("status") in (ST_PENDING, ST_FAILED)
        ]
        chunks = []
        for c_i, start in enumerate(range(0, len(work_indices), chunk_size)):
            slice_idx = work_indices[start : start + chunk_size]
            chunks.append(
                {
                    "id": c_i,
                    "label": f"chunk_{c_i + 1:02d}",
                    "item_indices": slice_idx,
                    "status": ST_PENDING if slice_idx else ST_DONE,
                    "started": None,
                    "finished": None,
                    "work_folder": None,
                }
            )
        return chunks

    def rebuild_chunks(self, queue_id: str, chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """Rebuild chunk packing from current pending/failed set (after big status changes)."""
        data = self.load(queue_id)
        if chunk_size:
            data["chunk_size"] = max(1, min(200, int(chunk_size)))
        data["chunks"] = self._build_chunks(data["items"], int(data["chunk_size"]))
        data["current_chunk"] = 0
        data.setdefault("action_log", []).append(
            {"time": _now_iso(), "action": "rebuild_chunks", "detail": f"size={data['chunk_size']}"}
        )
        self.save(data)
        self._append_history(queue_id, f"REBUILD_CHUNKS size={data['chunk_size']}")
        return data

    # ------------------------------------------------------------------ refresh
    def refresh_status(self, queue_id: str, *, rebuild_if_needed: bool = False) -> Dict[str, Any]:
        data = self.load(queue_id)
        out_dirs = data.get("output_dirs") or []
        min_stage = int(data.get("target_stage") or 1)
        output_index = self._build_output_index(out_dirs)

        changed = 0
        for it in data["items"]:
            prev = it.get("status")
            # always try to find/update match
            match = self.find_match(it["source"], output_index, min_stage)
            if match:
                it["output_match"] = match
                if prev in (ST_PENDING, ST_FAILED):
                    it["status"] = ST_DONE
                    it["error"] = None
                    it["updated"] = _now_iso()
                    changed += 1
                elif prev == ST_SKIPPED:
                    pass
                elif prev == ST_DONE and not os.path.isfile(match):
                    # lost output
                    it["status"] = ST_PENDING
                    it["output_match"] = None
                    it["updated"] = _now_iso()
                    changed += 1
            else:
                if prev in (ST_DONE, ST_SKIPPED):
                    # output disappeared — re-queue
                    it["status"] = ST_PENDING
                    it["output_match"] = None
                    it["updated"] = _now_iso()
                    changed += 1
                # keep FAILED as FAILED unless user requeues

            # validate source still exists
            if not os.path.isfile(it["source"]) and it["status"] == ST_PENDING:
                it["status"] = ST_FAILED
                it["error"] = "source missing on disk"
                it["updated"] = _now_iso()
                changed += 1

        self._update_chunk_statuses(data)
        if rebuild_if_needed and changed:
            # keep existing chunk packing unless empty chunks dominate
            pass
        data.setdefault("action_log", []).append(
            {"time": _now_iso(), "action": "refresh", "detail": f"changed={changed}"}
        )
        # ETA from done timestamps if available
        data["eta"] = self._estimate_eta(data)
        self.save(data)
        if changed:
            self._append_history(queue_id, f"REFRESH changed={changed}")
        return data

    def _update_chunk_statuses(self, data: Dict[str, Any]) -> None:
        items = data.get("items") or []
        for ch in data.get("chunks") or []:
            idxs = [i for i in ch.get("item_indices") or [] if i < len(items)]
            if not idxs:
                ch["status"] = ST_DONE
                continue
            statuses = [items[i]["status"] for i in idxs]
            if all(s in (ST_DONE, ST_SKIPPED) for s in statuses):
                ch["status"] = ST_DONE
                ch["finished"] = ch.get("finished") or _now_iso()
            elif any(s == ST_FAILED for s in statuses) and not any(s == ST_PENDING for s in statuses):
                ch["status"] = ST_FAILED
            elif any(s in (ST_PENDING, ST_FAILED) for s in statuses):
                if ch.get("work_folder") and any(s == ST_PENDING for s in statuses):
                    ch["status"] = ch.get("status") if ch.get("status") == "ready" else ST_PENDING
                else:
                    ch["status"] = ST_PENDING
        # current_chunk pointer
        data["current_chunk"] = 0
        for ch in data.get("chunks") or []:
            idxs = ch.get("item_indices") or []
            if any(
                items[i]["status"] in (ST_PENDING, ST_FAILED)
                for i in idxs
                if i < len(items)
            ):
                data["current_chunk"] = ch["id"]
                break
        else:
            data["current_chunk"] = len(data.get("chunks") or [])

    def _estimate_eta(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rough ETA from timestamps on completed items."""
        times = []
        for it in data.get("items") or []:
            if it.get("status") in (ST_DONE, ST_SKIPPED) and it.get("updated"):
                try:
                    times.append(datetime.fromisoformat(it["updated"]))
                except ValueError:
                    pass
        remaining = self._stats(data).get("remaining", 0)
        if len(times) < 2 or remaining <= 0:
            return {"remaining_items": remaining, "seconds_per_item": None, "eta_iso": None}
        times.sort()
        span = (times[-1] - times[0]).total_seconds()
        # completed count in that span
        done_n = max(1, len(times) - 1)
        spi = max(1.0, span / done_n)
        eta_sec = spi * remaining
        eta = datetime.now().timestamp() + eta_sec
        return {
            "remaining_items": remaining,
            "seconds_per_item": round(spi, 1),
            "eta_seconds": int(eta_sec),
            "eta_iso": datetime.fromtimestamp(eta).isoformat(timespec="seconds"),
        }

    # ------------------------------------------------------------------ chunk ops
    def get_next_chunk(
        self, queue_id: str, include_failed: bool = True
    ) -> Tuple[Dict[str, Any], List[str], str]:
        data = self.refresh_status(queue_id)
        items = data["items"]
        for ch in data["chunks"]:
            paths = []
            for ii in ch["item_indices"]:
                if ii >= len(items):
                    continue
                it = items[ii]
                if it["status"] == ST_PENDING or (include_failed and it["status"] == ST_FAILED):
                    # failed with missing source — skip
                    if it["status"] == ST_FAILED and it.get("error") == "source missing on disk":
                        continue
                    if it["status"] == ST_FAILED and include_failed:
                        paths.append(it["source"])
                    elif it["status"] == ST_PENDING:
                        paths.append(it["source"])
            if paths:
                return ch, paths, self.format_chunk_summary(data, ch, paths)
        return {}, [], self.format_report(data) + "\n\n✅ No pending work — queue complete."

    def get_chunk(self, queue_id: str, chunk_id: int) -> Tuple[Dict[str, Any], List[str]]:
        data = self.load(queue_id)
        for ch in data["chunks"]:
            if ch["id"] == int(chunk_id):
                paths = []
                for ii in ch["item_indices"]:
                    it = data["items"][ii]
                    if it["status"] in (ST_PENDING, ST_FAILED):
                        paths.append(it["source"])
                return ch, paths
        return {}, []

    def prepare_chunk_folder(
        self,
        queue_id: str,
        chunk_id: Optional[int] = None,
        *,
        link_mode: str = "auto",  # auto | hardlink | symlink | copy
        include_failed: bool = True,
        clear_existing: bool = True,
    ) -> Tuple[str, str]:
        data = self.refresh_status(queue_id)
        if not self.acquire_lock(queue_id, owner="prepare"):
            return "", "Queue is locked by another operation — try again in a moment."

        try:
            ch = None
            paths: List[str] = []
            if chunk_id is not None:
                ch, paths = self.get_chunk(queue_id, int(chunk_id))
                if not include_failed:
                    paths = [
                        p
                        for p in paths
                        if next(
                            (
                                it
                                for it in data["items"]
                                if _norm(it["source"]) == _norm(p) and it["status"] == ST_PENDING
                            ),
                            None,
                        )
                    ]
            else:
                ch, paths, _ = self.get_next_chunk(queue_id, include_failed=include_failed)

            if not ch:
                return "", "No pending chunk to prepare."
            if not paths:
                return "", f"Chunk {ch.get('label')} has no pending files."

            dest = self._qdir(queue_id) / "work" / ch["label"]
            if dest.exists() and clear_existing:
                shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)

            linked = copied = symlinked = failed = 0
            errors = []
            for src in paths:
                name = Path(src).name
                target = dest / name
                # avoid name collisions
                if target.exists():
                    stem, ext = target.stem, target.suffix
                    n = 2
                    while target.exists():
                        target = dest / f"{stem}__{n}{ext}"
                        n += 1
                ok, method = self._place_file(src, target, link_mode)
                if not ok:
                    failed += 1
                    errors.append(f"{name}: {method}")
                elif method == "hardlink":
                    linked += 1
                elif method == "symlink":
                    symlinked += 1
                else:
                    copied += 1

            # write README in work folder
            readme = (
                f"FlashVSR Batch Queue work pack\n"
                f"Queue: {data.get('name')} ({queue_id})\n"
                f"Chunk: {ch['label']}\n"
                f"Files: {len(paths)}\n"
                f"Target stage: _{data.get('target_stage')}\n"
                f"Prepared: {_now_iso()}\n\n"
                f"Paste this folder path into FlashVSR → Batch Video → Folder Path, then Start Batch.\n"
                f"When done: Toolbox → Batch Queue → Refresh status.\n\n"
                f"Placement: hardlink={linked} symlink={symlinked} copy={copied} failed={failed}\n"
            )
            if errors:
                readme += "\nErrors:\n" + "\n".join(errors[:20])
            (dest / "README_QUEUE.txt").write_text(readme, encoding="utf-8")
            (dest / "SOURCES.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")

            # update chunk meta
            for c in data["chunks"]:
                if c["id"] == ch["id"]:
                    c["status"] = "ready"
                    c["started"] = c.get("started") or _now_iso()
                    c["work_folder"] = str(dest)
                    break
            data.setdefault("action_log", []).append(
                {
                    "time": _now_iso(),
                    "action": "prepare_chunk",
                    "detail": f"{ch['label']} n={len(paths)} → {dest}",
                }
            )
            self.save(data)
            self._append_history(
                queue_id,
                f"PREPARE {ch['label']} n={len(paths)} hard={linked} sym={symlinked} copy={copied} fail={failed}",
            )

            msg = (
                f"✅ Prepared {ch['label']}: {len(paths)} files\n"
                f"{dest}\n\n"
                f"Placement — hardlinks: {linked}, symlinks: {symlinked}, copies: {copied}, failed: {failed}\n\n"
                f"👉 FlashVSR tab → Batch Video → Folder Path = path above → Start Batch Processing\n"
                f"👉 After run (or crash): Refresh status here — done files drop out of the next chunk\n"
            )
            if errors:
                msg += "\n⚠️ Some files failed to place:\n" + "\n".join(errors[:10])
            return str(dest), msg
        finally:
            self.release_lock(queue_id, owner="prepare")

    def _place_file(self, src: str, dest: Path, link_mode: str) -> Tuple[bool, str]:
        link_mode = (link_mode or "auto").lower()
        modes = {
            "auto": ["hardlink", "symlink", "copy"],
            "hardlink": ["hardlink", "copy"],
            "symlink": ["symlink", "copy"],
            "copy": ["copy"],
        }.get(link_mode, ["hardlink", "symlink", "copy"])

        last_err = ""
        for m in modes:
            try:
                if m == "hardlink":
                    os.link(src, dest)
                    return True, "hardlink"
                if m == "symlink":
                    os.symlink(src, dest)
                    return True, "symlink"
                if m == "copy":
                    shutil.copy2(src, dest)
                    return True, "copy"
            except OSError as e:
                last_err = str(e)
                continue
        return False, last_err or "unknown error"

    # ------------------------------------------------------------------ mark / requeue
    def mark_items(
        self,
        queue_id: str,
        source_paths: Sequence[str],
        status: str,
        error: Optional[str] = None,
        output_match: Optional[str] = None,
    ) -> Dict[str, Any]:
        data = self.load(queue_id)
        wanted = {_norm(p) for p in source_paths}
        n = 0
        for it in data["items"]:
            if _norm(it["source"]) in wanted:
                it["status"] = status
                it["error"] = error
                if output_match:
                    it["output_match"] = output_match
                it["updated"] = _now_iso()
                n += 1
        data.setdefault("action_log", []).append(
            {"time": _now_iso(), "action": "mark_items", "detail": f"{status} x{n}"}
        )
        self.save(data)
        self._append_history(queue_id, f"MARK {status} x{n}")
        return self.refresh_status(queue_id)

    def requeue_failed(self, queue_id: str) -> Dict[str, Any]:
        data = self.load(queue_id)
        n = 0
        for it in data["items"]:
            if it["status"] == ST_FAILED:
                if os.path.isfile(it["source"]):
                    it["status"] = ST_PENDING
                    it["error"] = None
                    it["updated"] = _now_iso()
                    n += 1
        data["chunks"] = self._build_chunks(data["items"], int(data["chunk_size"]))
        data.setdefault("action_log", []).append(
            {"time": _now_iso(), "action": "requeue_failed", "detail": f"n={n}"}
        )
        self.save(data)
        self._append_history(queue_id, f"REQUEUE_FAILED n={n}")
        return self.refresh_status(queue_id)

    def requeue_all_pending_and_failed(self, queue_id: str) -> Dict[str, Any]:
        """Force rebuild chunks for everything not done/skipped."""
        data = self.load(queue_id)
        for it in data["items"]:
            if it["status"] == ST_FAILED and os.path.isfile(it["source"]):
                it["status"] = ST_PENDING
                it["error"] = None
                it["updated"] = _now_iso()
        data["chunks"] = self._build_chunks(data["items"], int(data["chunk_size"]))
        self.save(data)
        return self.refresh_status(queue_id)

    # ------------------------------------------------------------------ import crash
    def import_batch_progress(
        self,
        batch_dir: str,
        source_folder: str,
        *,
        chunk_size: int = DEFAULT_CHUNK,
        target_stage: int = 1,
        name: str = "",
        extra_output_dirs: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build/update a queue from a crashed FlashVSR batch folder that has
        BATCH_PROGRESS.json / BATCH_PROGRESS.txt and INPUTS.txt (if present).
        """
        batch_dir = os.path.normpath(batch_dir)
        if not os.path.isdir(batch_dir):
            raise ValueError(f"Batch folder not found: {batch_dir}")

        progress_json = Path(batch_dir) / "BATCH_PROGRESS.json"
        inputs_txt = Path(batch_dir) / "INPUTS.txt"
        sources: List[str] = []

        if inputs_txt.is_file():
            for line in inputs_txt.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and os.path.isfile(line):
                    sources.append(os.path.normpath(line))

        prog_items = []
        if progress_json.is_file():
            try:
                prog = json.loads(progress_json.read_text(encoding="utf-8"))
                prog_items = prog.get("items") or []
            except json.JSONDecodeError:
                prog_items = []

        # If no INPUTS.txt, recover sources from progress entries
        if not sources and prog_items:
            for it in prog_items:
                sp = it.get("source")
                if sp and os.path.isfile(sp):
                    sources.append(os.path.normpath(sp))

        if not sources:
            # fall back: all videos in source_folder
            if not source_folder or not os.path.isdir(source_folder):
                raise ValueError(
                    "Need INPUTS.txt / BATCH_PROGRESS sources, or a valid source_folder to import."
                )
            sources = _list_videos(source_folder)

        # Map progress status by normalized source
        prog_map: Dict[str, Dict[str, Any]] = {}
        for it in prog_items:
            sp = it.get("source") or ""
            if sp:
                prog_map[_norm(sp)] = it
            sn = it.get("source_name")
            if sn:
                prog_map.setdefault(sn.lower(), it)

        out_dirs = [batch_dir]
        if extra_output_dirs:
            out_dirs.extend([d for d in extra_output_dirs if d])
        if source_folder and os.path.isdir(source_folder):
            # still use source folder listing if inputs incomplete
            pass

        # Create base queue from source_folder or synthetic folder of sources
        if source_folder and os.path.isdir(source_folder):
            data = self.create_queue(
                source_folder=source_folder,
                output_dirs=out_dirs,
                chunk_size=chunk_size,
                target_stage=target_stage,
                name=name or f"import_{Path(batch_dir).name}",
                only_pending=True,
            )
        else:
            # write a virtual source list into a staging folder of hardlinks? simpler: create manually
            raise ValueError("source_folder is required for import when sources are mixed.")

        # Overlay progress statuses
        for it in data["items"]:
            key = _norm(it["source"])
            hit = prog_map.get(key) or prog_map.get(Path(it["source"]).name.lower())
            if not hit:
                continue
            st = (hit.get("status") or "").lower()
            if st == "done":
                it["status"] = ST_DONE
                it["output_match"] = hit.get("output") or it.get("output_match")
                it["error"] = None
                it["updated"] = hit.get("time") or _now_iso()
            elif st == "failed":
                # only fail if not already matched as done by scan
                if it["status"] not in (ST_DONE, ST_SKIPPED):
                    it["status"] = ST_FAILED
                    it["error"] = hit.get("error") or "failed in batch"
                    it["updated"] = hit.get("time") or _now_iso()

        data["chunks"] = self._build_chunks(data["items"], int(data["chunk_size"]))
        data["imported_from"] = batch_dir
        data.setdefault("action_log", []).append(
            {"time": _now_iso(), "action": "import_batch_progress", "detail": batch_dir}
        )
        data["notes"] = (data.get("notes") or "") + f" | Imported from {batch_dir}"
        self.save(data)
        self._append_history(queue_id=data["id"], message=f"IMPORT {batch_dir}")
        return self.refresh_status(data["id"])

    # ------------------------------------------------------------------ side files
    def _write_chunk_lists(self, data: Dict[str, Any]) -> None:
        qdir = self._qdir(data["id"])
        lists_dir = qdir / "chunks"
        lists_dir.mkdir(parents=True, exist_ok=True)
        items = data["items"]
        for ch in data.get("chunks") or []:
            paths = []
            for ii in ch.get("item_indices") or []:
                if ii < len(items) and items[ii]["status"] in (ST_PENDING, ST_FAILED):
                    paths.append(items[ii]["source"])
            list_path = lists_dir / f"{ch['label']}.txt"
            _atomic_write_text(list_path, "\n".join(paths) + ("\n" if paths else ""))
            meta = {
                "chunk": ch["label"],
                "id": ch["id"],
                "count": len(paths),
                "status": ch.get("status"),
                "paths": paths,
                "work_folder": ch.get("work_folder"),
            }
            _atomic_write_json(lists_dir / f"{ch['label']}_meta.json", meta)

        # top-level convenience lists
        pending = [it["source"] for it in items if it["status"] == ST_PENDING]
        failed = [it["source"] for it in items if it["status"] == ST_FAILED]
        done = [it["source"] for it in items if it["status"] in (ST_DONE, ST_SKIPPED)]
        _atomic_write_text(qdir / "PENDING.txt", "\n".join(pending) + ("\n" if pending else ""))
        _atomic_write_text(qdir / "FAILED.txt", "\n".join(failed) + ("\n" if failed else ""))
        _atomic_write_text(qdir / "DONE.txt", "\n".join(done) + ("\n" if done else ""))

    def _write_csv(self, data: Dict[str, Any]) -> None:
        qdir = self._qdir(data["id"])
        path = qdir / "queue.csv"
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "index",
                        "status",
                        "source_name",
                        "source_path",
                        "stem",
                        "output_match",
                        "error",
                        "size",
                        "updated",
                        "duplicate_stem",
                    ]
                )
                for it in data.get("items") or []:
                    w.writerow(
                        [
                            it.get("index"),
                            it.get("status"),
                            Path(it.get("source", "")).name,
                            it.get("source"),
                            it.get("stem"),
                            it.get("output_match"),
                            it.get("error"),
                            it.get("size"),
                            it.get("updated"),
                            it.get("duplicate_stem"),
                        ]
                    )
        except OSError:
            pass

    # ------------------------------------------------------------------ reports
    def format_chunk_summary(self, data: Dict[str, Any], ch: Dict[str, Any], paths: List[str]) -> str:
        stats = self._stats(data)
        lines = [
            f"Queue: {data.get('name')} ({data.get('id')})",
            f"Next: {ch.get('label')}  ({len(paths)} files)   target stage: _{data.get('target_stage')}",
            f"Overall: {stats.get('complete', 0)}/{stats.get('total', 0)} done  |  "
            f"{stats.get('pending', 0)} pending  |  {stats.get('failed', 0)} failed",
            "",
            "Files in this chunk:",
        ]
        for i, p in enumerate(paths, 1):
            lines.append(f"  {i:02d}. {Path(p).name}")
        return "\n".join(lines)

    def format_report(self, data: Dict[str, Any]) -> str:
        stats = self._stats(data)
        eta = data.get("eta") or self._estimate_eta(data)
        lines = [
            "=" * 64,
            f"BATCH QUEUE: {data.get('name')}",
            f"ID: {data.get('id')}   v{data.get('version', 1)}",
            f"Created: {data.get('created')}   Updated: {data.get('updated')}",
            f"Source: {data.get('source_folder')}",
            f"Target stage: _{data.get('target_stage')}   Chunk size: {data.get('chunk_size')}",
            f"Outputs scanned:",
        ]
        for od in data.get("output_dirs") or []:
            lines.append(f"  - {od}")
        if data.get("imported_from"):
            lines.append(f"Imported from: {data.get('imported_from')}")
        lines += [
            "-" * 64,
            f"TOTAL {stats['total']}  |  DONE {stats.get('done', 0)}  |  "
            f"SKIPPED {stats.get('skipped', 0)}  |  "
            f"PENDING {stats.get('pending', 0)}  |  FAILED {stats.get('failed', 0)}  |  "
            f"REMAINING {stats.get('remaining', 0)}",
        ]
        if eta.get("seconds_per_item"):
            lines.append(
                f"ETA ~ {eta.get('eta_seconds')}s  "
                f"({eta.get('seconds_per_item')}s/item)  →  {eta.get('eta_iso')}"
            )
        lines.append("-" * 64)
        for ch in data.get("chunks") or []:
            idxs = ch.get("item_indices") or []
            pending_n = sum(
                1
                for i in idxs
                if i < len(data["items"]) and data["items"][i]["status"] in (ST_PENDING, ST_FAILED)
            )
            done_n = len(idxs) - pending_n
            wf = f"  work={ch['work_folder']}" if ch.get("work_folder") else ""
            lines.append(
                f"  {ch['label']}: {str(ch.get('status')):8}  {done_n}/{len(idxs)} done  "
                f"({pending_n} left){wf}"
            )
        lines.append("-" * 64)
        lines.append("FAILED:")
        fails = [it for it in data.get("items") or [] if it["status"] == ST_FAILED]
        if not fails:
            lines.append("  (none)")
        for it in fails:
            err = f"  ERR: {it['error']}" if it.get("error") else ""
            lines.append(f"  • {Path(it['source']).name}{err}")
        lines.append("-" * 64)
        lines.append("PENDING:")
        pend = [it for it in data.get("items") or [] if it["status"] == ST_PENDING]
        if not pend:
            lines.append("  (none)")
        for it in pend[:80]:
            lines.append(f"  • {Path(it['source']).name}")
        if len(pend) > 80:
            lines.append(f"  … +{len(pend) - 80} more (see PENDING.txt)")
        lines.append("-" * 64)
        lines.append("DONE / SKIPPED (sample):")
        done = [it for it in data.get("items") or [] if it["status"] in (ST_DONE, ST_SKIPPED)]
        for it in done[:40]:
            match = Path(it["output_match"]).name if it.get("output_match") else "?"
            lines.append(f"  [{it['status']}] {Path(it['source']).name}  →  {match}")
        if len(done) > 40:
            lines.append(f"  … +{len(done) - 40} more (see DONE.txt / queue.csv)")
        lines.append("=" * 64)
        lines.append(f"Folder: {self._qdir(data['id'])}")
        return "\n".join(lines)

    def format_html_report(self, data: Dict[str, Any]) -> str:
        stats = self._stats(data)
        done = stats.get("complete", 0)
        total = max(1, stats.get("total", 1))
        pct = int(100 * done / total)
        eta = data.get("eta") or self._estimate_eta(data)
        remaining = stats.get("remaining", 0)

        def esc(s: Any) -> str:
            return (
                str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        chunk_bits = []
        items = data.get("items") or []
        for ch in data.get("chunks") or []:
            left = sum(
                1
                for ii in ch.get("item_indices") or []
                if ii < len(items) and items[ii]["status"] in (ST_PENDING, ST_FAILED)
            )
            total_c = len(ch.get("item_indices") or [])
            st = ch.get("status")
            bg = {
                ST_DONE: "#14532d",
                ST_FAILED: "#7f1d1d",
                "ready": "#1e3a5f",
                ST_PENDING: "#1a2332",
            }.get(st, "#1a2332")
            chunk_bits.append(
                f"<span style='display:inline-block;margin:3px 4px;padding:5px 10px;border-radius:8px;"
                f"background:{bg};border:1px solid #475569;color:#e2e8f0;font-size:0.85em'>"
                f"<b>{esc(ch['label'])}</b> {esc(st)} · {total_c - left}/{total_c}"
                f"{' · NEXT' if left and ch['id']==data.get('current_chunk') else ''}"
                f"</span>"
            )

        # compact status table — failed + next pending first
        ordered = sorted(
            items,
            key=lambda it: (
                0
                if it["status"] == ST_FAILED
                else 1
                if it["status"] == ST_PENDING
                else 2
            ),
        )
        rows = []
        for it in ordered[:120]:
            color = {
                ST_DONE: "#22c55e",
                ST_SKIPPED: "#38bdf8",
                ST_PENDING: "#94a3b8",
                ST_FAILED: "#f87171",
            }.get(it["status"], "#e2e8f0")
            rows.append(
                "<tr>"
                f"<td style='color:{color};font-weight:700;padding:3px 6px'>{esc(it['status'])}</td>"
                f"<td style='padding:3px 6px'>{esc(Path(it['source']).name)}</td>"
                f"<td style='padding:3px 6px;font-size:0.82em;opacity:0.9'>"
                f"{esc(Path(it['output_match']).name) if it.get('output_match') else '—'}"
                f"{(' · ' + esc(it['error'])) if it.get('error') else ''}"
                f"</td></tr>"
            )
        more = ""
        if len(items) > 120:
            more = f"<div style='opacity:0.7;margin-top:6px'>… {len(items) - 120} more rows in queue.csv / STATUS.txt</div>"

        eta_html = ""
        if eta.get("eta_iso"):
            mins = max(1, int((eta.get("eta_seconds") or 0) / 60))
            eta_html = (
                f"<div style='margin:6px 0;opacity:0.9'>ETA ~ <b>{mins} min</b> "
                f"({esc(eta.get('seconds_per_item'))}s/item) → {esc(eta.get('eta_iso'))}</div>"
            )

        next_label = self._next_chunk_label(data)
        return f"""
<div style="padding:14px;border-radius:10px;border:1px solid #334155;
 background:linear-gradient(145deg,#0f172a 0%,#1e293b 55%,#0f172a 100%);
 color:#e2e8f0;font-family:ui-monospace,Consolas,monospace">
  <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:8px">
    <div>
      <div style="font-size:1.1em;font-weight:700">{esc(data.get('name',''))}</div>
      <div style="opacity:0.7;font-size:0.85em">{esc(data.get('id',''))} · stage <b>_{esc(data.get('target_stage'))}</b>
        · chunk {esc(data.get('chunk_size'))}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:1.4em;font-weight:800">{done}/{stats.get('total',0)}</div>
      <div style="opacity:0.75;font-size:0.85em">{remaining} remaining · next <b>{esc(next_label)}</b></div>
    </div>
  </div>
  <div style="height:12px;background:#020617;border-radius:6px;overflow:hidden;margin:8px 0 10px;
    border:1px solid #1e293b">
    <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#6366f1,#22c55e)"></div>
  </div>
  {eta_html}
  <div style="margin:8px 0 12px">{''.join(chunk_bits) if chunk_bits else '<i>No chunks</i>'}</div>
  <div style="max-height:300px;overflow:auto;border:1px solid #1e293b;border-radius:8px">
    <table style="width:100%;border-collapse:collapse;font-size:0.88em">
      <thead><tr style="text-align:left;background:#020617;position:sticky;top:0">
        <th style="padding:6px">Status</th><th style="padding:6px">Source</th><th style="padding:6px">Output / error</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  {more}
  <div style="margin-top:10px;font-size:0.8em;opacity:0.75">
    Manifest: {esc(str(self._manifest_path(data['id'])))}<br>
    Also: STATUS.txt · PENDING.txt · FAILED.txt · DONE.txt · queue.csv · history.log
  </div>
</div>
"""


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
