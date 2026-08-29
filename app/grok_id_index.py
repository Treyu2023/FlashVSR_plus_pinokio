"""
Grok unique-ID catalog for FlashVSR duplicate screening.

Extracts GROK / grok-video / leading X-post IDs from filenames, stores them
with original file size, and flags new downloads whose ID matches exactly
and whose size is within ±2.5% of the cataloged original.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SIZE_TOLERANCE = 0.025  # ±2.5% of original size
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv", ".gif", ".ts", ".mts", ".m2ts"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS

SKIP_DIR_NAMES = {
    "novideo", "highfps", "over4k", "work", "failed", "queue_logs",
    "__pycache__", ".git", "node_modules",
}

# grok-video-<uuid>  (Chrome "(2)" download copies share this id)
_GROK_VIDEO_RE = re.compile(
    r"(grok-video-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})",
    re.I,
)
# GROK-12 / GROK_0042 / grok-99 (underscore after the number is common)
_GROK_NUM_RE = re.compile(r"(?<![a-z0-9])grok[-_]?(\d{2,})(?!\d)", re.I)
# X/Grok post-style leading id: 115668687_tiny_s2_...
_LEADING_NUM_RE = re.compile(r"^(\d{7,})(?:[_-]|$)")

INDEX_VERSION = 1


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def index_path(app_dir: str) -> Path:
    root = Path(app_dir) / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root / "grok_id_index.json"


def log_path(app_dir: str) -> Path:
    root = Path(app_dir) / "outputs" / "queue_logs"
    root.mkdir(parents=True, exist_ok=True)
    return root / "grok_id_scan.jsonl"


def empty_index() -> Dict[str, Any]:
    return {
        "version": INDEX_VERSION,
        "updated": "",
        "ids": {},
        "stats": {"files": 0, "ids": 0},
    }


def load_index(app_dir: str) -> Dict[str, Any]:
    path = index_path(app_dir)
    if not path.is_file():
        return empty_index()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "ids" not in data:
            return empty_index()
        data.setdefault("ids", {})
        data.setdefault("stats", {})
        return data
    except (OSError, json.JSONDecodeError):
        return empty_index()


def save_index(app_dir: str, data: Dict[str, Any]) -> Path:
    data = dict(data or empty_index())
    data["version"] = INDEX_VERSION
    data["updated"] = _now_iso()
    ids = data.get("ids") or {}
    data["stats"] = {
        "ids": len(ids),
        "files": sum(
            1 + len(rec.get("outputs") or [])
            for rec in ids.values()
            if isinstance(rec, dict)
        ),
    }
    path = index_path(app_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def append_scan_log(app_dir: str, rec: Dict[str, Any]) -> None:
    rec = dict(rec)
    rec.setdefault("at", _now_iso())
    try:
        with log_path(app_dir).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def extract_grok_ids(name: str) -> List[str]:
    """Unique ID strings from a filename (order preserved, lowercased)."""
    stem = Path(str(name or "")).name
    found: List[str] = []
    seen: Set[str] = set()

    def add(token: str) -> None:
        t = str(token or "").strip().lower()
        if not t or t in seen:
            return
        seen.add(t)
        found.append(t)

    for m in _GROK_VIDEO_RE.finditer(stem):
        add(m.group(1))
    for m in _GROK_NUM_RE.finditer(stem):
        add("grok-" + m.group(1))
    m = _LEADING_NUM_RE.match(stem)
    if m:
        add(m.group(1))
    return found


def size_within_tolerance(size: int, original: int, tol: float = SIZE_TOLERANCE) -> bool:
    orig = int(original or 0)
    sz = int(size or 0)
    if orig <= 0 or sz <= 0:
        return False
    return abs(sz - orig) / float(orig) <= float(tol)


def _file_size(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def _ensure_id_rec(index: Dict[str, Any], gid: str) -> Dict[str, Any]:
    ids = index.setdefault("ids", {})
    rec = ids.get(gid)
    if not isinstance(rec, dict):
        rec = {
            "id": gid,
            "original_size": 0,
            "original_path": "",
            "outputs": [],
            "first_seen": _now_iso(),
        }
        ids[gid] = rec
    rec.setdefault("outputs", [])
    rec.setdefault("original_size", 0)
    rec.setdefault("original_path", "")
    return rec


def record_file(
    index: Dict[str, Any],
    path: str,
    *,
    as_original: bool,
) -> List[str]:
    """Index one media file. Returns IDs recorded."""
    name = Path(path).name
    ids = extract_grok_ids(name)
    if not ids:
        return []
    sz = _file_size(path)
    ap = os.path.abspath(path)
    for gid in ids:
        rec = _ensure_id_rec(index, gid)
        if as_original:
            if not rec.get("original_size"):
                rec["original_size"] = sz
                rec["original_path"] = ap
            elif size_within_tolerance(sz, int(rec.get("original_size") or 0)):
                rec["original_path"] = rec.get("original_path") or ap
        else:
            outs = rec.setdefault("outputs", [])
            if not any(os.path.normcase(str(o.get("path") or "")) == os.path.normcase(ap) for o in outs if isinstance(o, dict)):
                outs.append({"path": ap, "size": sz, "seen": _now_iso()})
            if not rec.get("original_size") and sz > 0:
                # Output-only catalog: keep size as a hint, but matching still
                # prefers a true original when one is scanned later.
                pass
        rec["last_seen"] = _now_iso()
    return ids


def match_file(path: str, index: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Duplicate if any extracted ID is in the catalog AND size is within ±2.5%
    of that ID's cataloged original size.
    """
    ids = extract_grok_ids(Path(path).name)
    if not ids:
        return None
    sz = _file_size(path)
    catalog = (index or {}).get("ids") or {}
    for gid in ids:
        rec = catalog.get(gid)
        if not isinstance(rec, dict):
            continue
        orig = int(rec.get("original_size") or 0)
        if orig > 0 and size_within_tolerance(sz, orig):
            return {
                "id": gid,
                "original_size": orig,
                "file_size": sz,
                "original_path": rec.get("original_path") or "",
                "outputs": rec.get("outputs") or [],
                "reason": (
                    f"ID {gid} exact + size {sz} within ±2.5% of original {orig}"
                ),
            }
    return None


def iter_media(root: str) -> Iterable[Path]:
    base = Path(root)
    if not base.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIR_NAMES and not d.startswith(".")
        ]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in MEDIA_EXTS:
                yield p


def scan_roots(
    app_dir: str,
    roots: Sequence[Tuple[str, bool]],
    *,
    kind: str = "scan",
) -> Dict[str, Any]:
    """
    roots: list of (folder, as_original).
    as_original=True stores file size as the original for ±2.5% matching.
    """
    index = load_index(app_dir)
    files = 0
    with_ids = 0
    no_id = 0
    errors = 0
    started = time.time()
    for folder, as_original in roots:
        if not folder or not os.path.isdir(folder):
            continue
        for p in iter_media(folder):
            try:
                ids = record_file(index, str(p), as_original=as_original)
            except OSError:
                errors += 1
                continue
            files += 1
            if ids:
                with_ids += 1
            else:
                no_id += 1
    path = save_index(app_dir, index)
    summary = {
        "ok": True,
        "kind": kind,
        "files": files,
        "with_ids": with_ids,
        "no_id": no_id,
        "ids_total": len(index.get("ids") or {}),
        "errors": errors,
        "index": str(path),
        "log": str(log_path(app_dir)),
        "elapsed_s": round(time.time() - started, 2),
        "roots": [r[0] for r in roots if r[0]],
    }
    append_scan_log(app_dir, summary)
    return summary


def screen_folder(app_dir: str, folder: str) -> Dict[str, Any]:
    """Pre-screen new downloads against the catalog. Does not modify files."""
    index = load_index(app_dir)
    dupes: List[Dict[str, Any]] = []
    fresh: List[str] = []
    no_id = 0
    scanned = 0
    started = time.time()
    if folder and os.path.isdir(folder):
        for p in iter_media(folder):
            scanned += 1
            ap = str(p)
            ids = extract_grok_ids(p.name)
            if not ids:
                no_id += 1
                fresh.append(ap)
                continue
            hit = match_file(ap, index)
            if hit:
                dupes.append({
                    "path": ap,
                    "name": p.name,
                    **hit,
                })
            else:
                fresh.append(ap)
    summary = {
        "ok": True,
        "kind": "screen_downloads",
        "folder": folder,
        "scanned": scanned,
        "dupes": len(dupes),
        "fresh": len(fresh),
        "no_id": no_id,
        "ids_in_catalog": len(index.get("ids") or {}),
        "index": str(index_path(app_dir)),
        "log": str(log_path(app_dir)),
        "elapsed_s": round(time.time() - started, 2),
        "dupe_rows": dupes[:200],
    }
    append_scan_log(app_dir, {k: v for k, v in summary.items() if k != "dupe_rows"} | {"dupes_sample": [d.get("name") for d in dupes[:40]]})
    return summary


def html_scan_report(summary: Dict[str, Any]) -> str:
    if summary.get("kind") == "screen_downloads":
        rows = ""
        for d in (summary.get("dupe_rows") or [])[:40]:
            rows += (
                f"<tr><td style='padding:4px 8px;color:#fbbf24;'>{_esc(d.get('name'))}</td>"
                f"<td style='padding:4px 8px;color:#86efac;'>{_esc(d.get('id'))}</td>"
                f"<td style='padding:4px 8px;color:#94a3b8;font-size:0.85em;'>{_esc(d.get('reason'))}</td></tr>"
            )
        more = ""
        n = int(summary.get("dupes") or 0)
        if n > 40:
            more = f"<div style='color:#94a3b8;font-size:0.85em;'>…and {n - 40} more (see { _esc(summary.get('log')) })</div>"
        table = (
            f"<table style='width:100%;border-collapse:collapse;margin-top:8px;font-size:0.88em;'>"
            f"<tr><th align=left>File</th><th align=left>ID</th><th align=left>Why</th></tr>"
            f"{rows or '<tr><td colspan=3 style=\"color:#86efac;padding:6px;\">No ID+size duplicates in this folder.</td></tr>'}"
            f"</table>{more}"
        )
        return (
            "<div style='padding:10px 12px;background:#0b1220;border:1px solid #334155;"
            "border-radius:8px;color:#e2e8f0;font-size:0.9em;line-height:1.45;'>"
            f"<b style='color:#7dd3fc;'>New downloads pre-screen</b><br>"
            f"Scanned <b>{summary.get('scanned', 0)}</b> · "
            f"<span style='color:#f87171;'>dupes {summary.get('dupes', 0)}</span> · "
            f"new {summary.get('fresh', 0)} · no ID {summary.get('no_id', 0)} · "
            f"catalog {summary.get('ids_in_catalog', 0)} IDs<br>"
            f"<code style='color:#94a3b8;font-size:0.8em;'>{_esc(summary.get('folder'))}</code>"
            f"{table}</div>"
        )
    roots = summary.get("roots") or []
    root_html = "".join(
        f"<li style='margin:2px 0;'><code style='color:#cbd5e1;font-size:0.8em;'>{_esc(r)}</code></li>"
        for r in roots
    )
    return (
        "<div style='padding:10px 12px;background:#0b1220;border:1px solid #334155;"
        "border-radius:8px;color:#e2e8f0;font-size:0.9em;line-height:1.45;'>"
        f"<b style='color:#7dd3fc;'>Output / archive ID scan</b><br>"
        f"Files {summary.get('files', 0)} · with IDs {summary.get('with_ids', 0)} · "
        f"no ID {summary.get('no_id', 0)} · catalog now "
        f"<b>{summary.get('ids_total', 0)}</b> unique IDs · {summary.get('elapsed_s', 0)}s<br>"
        f"Log: <code style='color:#94a3b8;font-size:0.8em;'>{_esc(summary.get('log'))}</code>"
        f"<ul style='margin:6px 0 0 1.1em;padding:0;'>{root_html}</ul></div>"
    )


def _esc(text: Any) -> str:
    s = str(text or "")
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
