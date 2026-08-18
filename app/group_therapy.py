"""
Group Therapy — process the original folder in groups of N, stage-by-stage.

Example group size 10:
  10 upscale → 10 RIFE 2× → 10 RIFE 2× → 10 export
  then the next 10 …

After each file finishes the last selected stage:
  keep ONLY the original + the end file
  original → Before / pairing folder
  final    → After / Ready for CIV
  delete resized / upscale / RIFE / toolbox temps
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from flashvsr_work_queue import FlashVSRWorkQueue, VIDEO_EXTS

STAGES = ("upscale", "rife1", "rife2", "export")
STAGE_LABELS = {
    "upscale": "Upscale",
    "rife1": "RIFE pass 1 (2×)",
    "rife2": "RIFE pass 2 (2×)",
    "export": "Export / interpolation",
}
# Last-completed-stage → next stage
_NEXT = {
    None: "upscale",
    "": "upscale",
    "none": "upscale",
    "upscale": "rife1",
    "rife1": "rife2",
    "rife2": "export",
    "export": None,
}


def selected_stages(*, do_upscale: bool, do_rife1: bool, do_rife2: bool, do_export: bool) -> List[str]:
    out: List[str] = []
    if do_upscale:
        out.append("upscale")
    if do_rife1:
        out.append("rife1")
    if do_rife2:
        out.append("rife2")
    if do_export:
        out.append("export")
    return out or ["upscale", "rife1", "rife2", "export"]


def work_root(app_dir: str) -> Path:
    root = Path(app_dir) / "outputs" / "group_therapy" / "work"
    root.mkdir(parents=True, exist_ok=True)
    return root


def stage_dir(app_dir: str, stage: str) -> Path:
    d = work_root(app_dir) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_queue(app_dir: str) -> FlashVSRWorkQueue:
    return FlashVSRWorkQueue(
        app_dir, name="group", extensions=VIDEO_EXTS, label="Group Therapy"
    )


def assign_groups(wq: FlashVSRWorkQueue, group_size: int) -> int:
    """Give pending items a stable gt_group (newest-first already in queue)."""
    group_size = max(1, int(group_size or 10))
    data = wq.load()
    items = list(data.get("items") or [])
    if not items:
        return 0
    assigned = 0
    open_gid = 0
    open_count = 0
    for it in items:
        gid = it.get("gt_group")
        if gid:
            try:
                g = int(gid)
            except (TypeError, ValueError):
                g = 0
            if g > open_gid:
                open_gid = g
                open_count = 1
            elif g == open_gid:
                open_count += 1
    if open_gid and open_count >= group_size:
        open_gid = 0
        open_count = 0
    for it in items:
        if it.get("gt_group"):
            continue
        if it.get("status") not in ("pending", "failed", "running"):
            continue
        if not open_gid or open_count >= group_size:
            open_gid = (open_gid or 0) + 1
            open_count = 0
        it["gt_group"] = open_gid
        it.setdefault("gt_stage", None)
        open_count += 1
        assigned += 1
    if assigned:
        data["gt_group_size"] = group_size
        wq.save(data)
    return assigned


def ordered_groups(items: Sequence[Dict[str, Any]]) -> List[int]:
    seen = []
    for it in items:
        if it.get("status") == "done":
            continue
        try:
            g = int(it.get("gt_group") or 0)
        except (TypeError, ValueError):
            g = 0
        if g and g not in seen:
            seen.append(g)
    return seen


def group_members(items: Sequence[Dict[str, Any]], gid: int) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        try:
            g = int(it.get("gt_group") or 0)
        except (TypeError, ValueError):
            g = 0
        if g == gid:
            out.append(it)
    return out


def input_for_stage(it: Dict[str, Any], stage: str) -> Optional[str]:
    """Best existing file to feed this stage (resume-safe)."""
    chain = {
        "upscale": ["path", "gt_original"],
        "rife1": ["gt_upscale", "path"],
        "rife2": ["gt_rife1", "gt_upscale"],
        "export": ["gt_rife2", "gt_rife1", "gt_upscale"],
    }
    for key in chain.get(stage, []):
        p = (it.get(key) or "").strip()
        if p and os.path.isfile(p):
            return p
    return None


def stage_already_done(it: Dict[str, Any], stage: str) -> bool:
    key = {
        "upscale": "gt_upscale",
        "rife1": "gt_rife1",
        "rife2": "gt_rife2",
        "export": "gt_export",
    }.get(stage)
    if not key:
        return False
    p = (it.get(key) or "").strip()
    return bool(p and os.path.isfile(p))


def last_output(it: Dict[str, Any]) -> Optional[str]:
    for key in ("gt_export", "gt_rife2", "gt_rife1", "gt_upscale"):
        p = (it.get(key) or "").strip()
        if p and os.path.isfile(p):
            return p
    return None


def _safe_delete(path: Optional[str], *, keep: Iterable[str]) -> bool:
    if not path:
        return False
    try:
        ap = os.path.abspath(path)
    except OSError:
        return False
    keep_abs = set()
    for k in keep:
        if not k:
            continue
        try:
            keep_abs.add(os.path.normcase(os.path.abspath(k)))
        except OSError:
            pass
    if os.path.normcase(ap) in keep_abs:
        return False
    if not os.path.isfile(ap):
        return False
    try:
        os.remove(ap)
        return True
    except OSError:
        return False


def place_file(src: str, dest_dir: str) -> str:
    """Move (or copy if cross-device leftover) src into dest_dir. Returns dest path."""
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(src)
    dest = os.path.join(dest_dir, name)
    src_abs = os.path.abspath(src)
    dest_abs = os.path.abspath(dest)
    if os.path.normcase(src_abs) == os.path.normcase(dest_abs):
        return dest_abs
    if os.path.exists(dest_abs):
        if os.path.normcase(src_abs) != os.path.normcase(dest_abs):
            stem, ext = os.path.splitext(name)
            n = 2
            while True:
                cand = os.path.join(dest_dir, f"{stem}_{n}{ext}")
                if not os.path.exists(cand):
                    dest_abs = os.path.abspath(cand)
                    break
                n += 1
    try:
        shutil.move(src_abs, dest_abs)
    except OSError:
        shutil.copy2(src_abs, dest_abs)
        try:
            os.remove(src_abs)
        except OSError:
            pass
    return dest_abs


def settle_pair(
    it: Dict[str, Any],
    *,
    before_dir: str,
    after_dir: str,
    extra_temps: Optional[Sequence[str]] = None,
) -> Tuple[str, str, int]:
    """
    Keep only original + end file in the user matching folders.
    Returns (before_path, after_path, deleted_count).
    """
    original = (it.get("gt_original") or it.get("path") or "").strip()
    if original and not os.path.isfile(original):
        original = (it.get("gt_original") or "").strip()
    end = last_output(it) or ""
    if not end or not os.path.isfile(end):
        raise FileNotFoundError("no end file to settle")

    before_path = ""
    if original and os.path.isfile(original):
        if _path_is_under(original, before_dir):
            before_path = os.path.abspath(original)
        else:
            before_path = place_file(original, before_dir)

    if _path_is_under(end, after_dir):
        after_path = os.path.abspath(end)
    else:
        after_path = place_file(end, after_dir)

    keep = [before_path, after_path]
    deleted = 0
    for key in ("gt_resized", "gt_upscale", "gt_rife1", "gt_rife2", "gt_export"):
        p = it.get(key)
        if p and _safe_delete(p, keep=keep):
            deleted += 1
    for p in extra_temps or []:
        if _safe_delete(p, keep=keep):
            deleted += 1
    return before_path, after_path, deleted


def _path_is_under(path: str, root: str) -> bool:
    try:
        p = os.path.abspath(path)
        r = os.path.abspath(root)
        return os.path.commonpath([p, r]) == r
    except (ValueError, OSError):
        return False


def collect_temp_globs(stem_hint: str, *folders: str) -> List[str]:
    """Find leftover temps that share a stem (resized, frames, toolbox)."""
    found: List[str] = []
    hint = (stem_hint or "")[:24]
    if not hint:
        return found
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        try:
            for f in Path(folder).iterdir():
                if not f.is_file():
                    continue
                n = f.name
                if hint in n and f.suffix.lower() in VIDEO_EXTS | {".jpg", ".jpeg", ".png"}:
                    found.append(str(f))
        except OSError:
            continue
    return found
