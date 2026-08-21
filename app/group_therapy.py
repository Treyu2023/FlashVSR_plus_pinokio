"""
Group Therapy — process the original folder in groups of N, stage-by-stage.

Example group size 10:
  10 upscale → 10 RIFE 2× → 10 RIFE 2× → 10 export
  then the next 10 …

After each file finishes the last selected stage:
  keep ONLY the original + the end file
  original → Before folder (flat), filename ends with _PID_xxxxxxxx
  final    → After folder (flat), same _PID_xxxxxxxx + Title metadata
  delete resized / upscale / RIFE / toolbox temps
  No per-file subfolders. Media Center tags are not touched.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from flashvsr_work_queue import FlashVSRWorkQueue, VIDEO_EXTS

_PAIR_ID_RE = re.compile(r"^[0-9a-f]{8}$")
_PAIR_FOLDER_RE = re.compile(r"^GT-([0-9a-f]{8})__", re.I)
_PID_IN_NAME_RE = re.compile(r"_PID_([0-9a-f]{8})(?:_|$)", re.I)
# Retroactive IDs that we mint (not reused from GT- folders) live in 9xxxxxxx
# so they cannot collide with new auto uuid-hex batches.
_RETRO_PID_PREFIX = "9"
_RETRO_MAP_NAME = "PID_RETRO_MAP.json"
_SKIP_DIR_NAMES = {
    "highfps",
    "novideo",
    "over4k",
    "images",
    "_civ posted",
    "from_toolbox_inbox",
    "bin",
    "crushed_old_fix",
}

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


def make_pair_id() -> str:
    """8-char hex pair id. New batches use random uuid hex (not 9xxxxxxx)."""
    pid = uuid.uuid4().hex[:8]
    # Keep auto ids out of the retro 9xxxxxxx band
    if pid.startswith(_RETRO_PID_PREFIX):
        pid = "a" + pid[1:]
    return pid


def pid_token(pair_id: str) -> str:
    return f"PID_{str(pair_id).lower()}"


def with_pid_name(path_or_name: str, pair_id: str) -> str:
    """Append _PID_xxxxxxxx before the extension (idempotent)."""
    p = Path(str(path_or_name or "clip"))
    stem = _PID_IN_NAME_RE.sub("", p.stem).rstrip("_")
    return f"{stem}_{pid_token(pair_id)}{p.suffix}"


def pair_id_from_name(path_or_name: str) -> Optional[str]:
    text = str(path_or_name or "")
    m = _PID_IN_NAME_RE.search(Path(text).stem)
    if m:
        return m.group(1).lower()
    m2 = _PAIR_FOLDER_RE.search(Path(text).name)
    if m2:
        return m2.group(1).lower()
    m3 = _PAIR_FOLDER_RE.search(Path(text).parent.name)
    if m3:
        return m3.group(1).lower()
    return None


def safe_pair_stem(path_or_name: str) -> str:
    stem = Path(str(path_or_name or "clip")).stem
    stem = _PID_IN_NAME_RE.sub("", stem)
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return (stem or "clip")[:32]


def pair_folder_name(pair_id: str, path_or_name: str) -> str:
    """Legacy helper — pairing is filename PID now, not per-file folders."""
    return pid_token(pair_id)


def ensure_pair_id(it: Dict[str, Any]) -> str:
    raw = str(it.get("gt_pair_id") or "").strip().lower()
    if _PAIR_ID_RE.match(raw):
        it["gt_pair_id"] = raw
        return raw
    for key in ("gt_before", "gt_after", "gt_original", "gt_export", "path"):
        pid = pair_id_from_name(it.get(key) or "")
        if pid:
            it["gt_pair_id"] = pid
            return pid
    pid = make_pair_id()
    it["gt_pair_id"] = pid
    return pid


def stamp_title_pid(path: str, pair_id: str) -> bool:
    """
    Set MP4/MOV Title to PID_xxxxxxxx. Does not touch genre/comment/keywords
    (Media Center tags stay). Filename remains the source of truth.
    """
    if not path or not os.path.isfile(path):
        return False
    title = pid_token(pair_id)
    tmp = str(Path(path).with_name(f".{Path(path).stem}_pidtmp{Path(path).suffix}"))
    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-c", "copy", "-map", "0",
        "-metadata", f"title={title}",
        tmp,
    ]
    try:
        import subprocess
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        if os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
            return True
    except Exception:
        pass
    try:
        if os.path.isfile(tmp):
            os.remove(tmp)
    except OSError:
        pass
    return False


def write_pair_marker(folder: str, *, pair_id: str, role: str, original_name: str, mate_folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    payload = {
        "pair_id": pair_id,
        "role": role,
        "folder": os.path.basename(folder.rstrip("\\/")),
        "original_name": original_name,
        "mate_folder": mate_folder,
    }
    pair_txt = os.path.join(folder, "PAIR.txt")
    pair_json = os.path.join(folder, "pair.json")
    try:
        with open(pair_txt, "w", encoding="utf-8") as f:
            f.write(
                f"pair_id={pair_id}\n"
                f"role={role}\n"
                f"folder={payload['folder']}\n"
                f"original_name={original_name}\n"
                f"mate={mate_folder}\n"
            )
        with open(pair_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except OSError:
        pass


def append_pair_index(root: str, *, pair_id: str, folder: str, role: str, file_path: str) -> None:
    """One-line index so Before/After roots list every pair id."""
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    index = os.path.join(root, "PAIRS.txt")
    line = f"{pair_id}\t{role}\t{folder}\t{file_path}\n"
    try:
        existing = ""
        if os.path.isfile(index):
            existing = Path(index).read_text(encoding="utf-8")
        if pair_id in existing and os.path.basename(folder) in existing and role in existing:
            return
        with open(index, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


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
    counts: Dict[int, int] = {}
    open_gid = 0
    for it in items:
        try:
            g = int(it.get("gt_group") or 0)
        except (TypeError, ValueError):
            g = 0
        if g <= 0:
            continue
        counts[g] = counts.get(g, 0) + 1
        if g > open_gid:
            open_gid = g
    open_count = counts.get(open_gid, 0)
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
        ensure_pair_id(it)
        it["gt_pair_folder"] = pair_folder_name(it["gt_pair_id"], it.get("path") or "clip")
        open_count += 1
        assigned += 1
    # Stamp pair IDs on any older rows that don't have one yet
    for it in items:
        if it.get("status") == "done":
            continue
        if not it.get("gt_pair_id"):
            ensure_pair_id(it)
            it["gt_pair_folder"] = pair_folder_name(it["gt_pair_id"], it.get("path") or "clip")
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
    Keep only original + end file, both in the shared Before/After folders
    (no per-file subfolder). Pairing is `_PID_xxxxxxxx` on the filename
    and Title metadata (not Media Center tags).
    Returns (before_path, after_path, deleted_count).
    """
    original = (it.get("gt_original") or it.get("path") or "").strip()
    if original and not os.path.isfile(original):
        original = (it.get("gt_original") or "").strip()
    end = last_output(it) or ""
    if not end or not os.path.isfile(end):
        raise FileNotFoundError("no end file to settle")

    pair_id = ensure_pair_id(it)
    it["gt_pair_folder"] = pid_token(pair_id)
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    def _place_pid(src: str, dest_dir: str) -> str:
        dest_name = with_pid_name(src, pair_id)
        dest = os.path.join(dest_dir, dest_name)
        src_abs = os.path.abspath(src)
        dest_abs = os.path.abspath(dest)
        if os.path.normcase(src_abs) == os.path.normcase(dest_abs):
            return dest_abs
        if os.path.isfile(dest_abs):
            dest_abs = unique_pid_dest(dest_dir, dest_name)
        try:
            shutil.move(src_abs, dest_abs)
        except OSError:
            shutil.copy2(src_abs, dest_abs)
            try:
                os.remove(src_abs)
            except OSError:
                pass
        stamp_title_pid(dest_abs, pair_id)
        return dest_abs

    before_path = ""
    if original and os.path.isfile(original):
        if _path_is_under(original, before_dir) and pair_id_from_name(original) == pair_id:
            before_path = os.path.abspath(original)
        else:
            before_path = _place_pid(original, before_dir)

    if _path_is_under(end, after_dir) and pair_id_from_name(end) == pair_id:
        after_path = os.path.abspath(end)
    else:
        after_path = _place_pid(end, after_dir)

    append_pair_index(before_dir, pair_id=pair_id, folder=".", role="before", file_path=before_path)
    append_pair_index(after_dir, pair_id=pair_id, folder=".", role="after", file_path=after_path)

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


def unique_pid_dest(dest_dir: str, filename: str) -> str:
    dest = os.path.join(dest_dir, filename)
    if not os.path.exists(dest):
        return os.path.abspath(dest)
    stem, ext = os.path.splitext(filename)
    n = 2
    while True:
        cand = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        if not os.path.exists(cand):
            return os.path.abspath(cand)
        n += 1


def _path_is_under(path: str, root: str) -> bool:
    try:
        p = os.path.abspath(path)
        r = os.path.abspath(root)
        return os.path.commonpath([p, r]) == r
    except (ValueError, OSError):
        return False


def pair_video_in(folder: str) -> Optional[str]:
    """First video inside a pair folder, or None."""
    if not folder or not os.path.isdir(folder):
        return None
    try:
        vids = [
            str(f)
            for f in Path(folder).iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        ]
    except OSError:
        return None
    return vids[0] if vids else None


def load_retro_pid_map(*roots: str) -> Dict[str, str]:
    """old GT pair id → remapped 9xxxxxxx id."""
    mapping: Dict[str, str] = {}
    seen = set()
    candidates = []
    for root in roots:
        if not root:
            continue
        p = Path(root)
        candidates.append(p / _RETRO_MAP_NAME)
        try:
            candidates.append(p.parent / _RETRO_MAP_NAME)
        except OSError:
            pass
    for c in candidates:
        key = os.path.normcase(str(c))
        if key in seen or not c.is_file():
            continue
        seen.add(key)
        try:
            data = json.loads(c.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            old, new = str(k).lower(), str(v).lower()
            if _PAIR_ID_RE.match(old) and _PAIR_ID_RE.match(new):
                mapping[old] = new
    return mapping


def save_retro_pid_map(mapping: Dict[str, str], *roots: str) -> None:
    text = json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    written = set()
    for root in roots:
        if not root:
            continue
        try:
            os.makedirs(root, exist_ok=True)
            dest = Path(root) / _RETRO_MAP_NAME
            key = os.path.normcase(str(dest))
            if key in written:
                continue
            dest.write_text(text, encoding="utf-8")
            written.add(key)
        except OSError:
            continue


def mint_retro_pid(old_id: str, used: set) -> str:
    """Mint a 9xxxxxxx id. Prefer a stable transform of the old GT id."""
    old = str(old_id or "").lower()
    guesses = []
    if _PAIR_ID_RE.match(old):
        guesses.append((_RETRO_PID_PREFIX + old[:7])[:8])
        guesses.append((_RETRO_PID_PREFIX + old[1:])[:8])
    for g in guesses:
        if _PAIR_ID_RE.match(g) and g not in used:
            used.add(g)
            return g
    for _ in range(80):
        cand = _RETRO_PID_PREFIX + uuid.uuid4().hex[:7]
        if cand not in used:
            used.add(cand)
            return cand
    raise RuntimeError("could not mint unique retro PID")


def _pid_search_tokens(pid: str, mapping: Optional[Dict[str, str]] = None) -> List[str]:
    ids = []
    raw = str(pid or "").strip().lower()
    if _PAIR_ID_RE.match(raw):
        ids.append(raw)
    mapping = mapping or {}
    reverse = {v: k for k, v in mapping.items()}
    if raw in mapping:
        ids.append(mapping[raw])
    if raw in reverse:
        ids.append(reverse[raw])
    tokens = []
    for i in ids:
        tok = f"_pid_{i}"
        if tok not in tokens:
            tokens.append(tok)
    return tokens


def find_existing_pair(after_dir: str, it: Dict[str, Any]) -> Optional[str]:
    """If this item already has a finished After file (PID in name), return it."""
    if not after_dir or not os.path.isdir(after_dir):
        return None
    pid = str(it.get("gt_pair_id") or "").strip().lower()
    if not _PAIR_ID_RE.match(pid):
        pid = pair_id_from_name(it.get("path") or "") or ""
    mapping = load_retro_pid_map(after_dir)
    tokens = _pid_search_tokens(pid, mapping)
    try:
        for f in Path(after_dir).iterdir():
            if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
                continue
            stem = f.stem.lower()
            if any(tok in stem for tok in tokens):
                return str(f)
        # Legacy per-file folders (pre-flatten)
        check_ids = [pid] if pid else []
        if pid in mapping:
            check_ids.append(mapping[pid])
        for cid in check_ids:
            prefix = f"gt-{cid}__"
            for d in Path(after_dir).iterdir():
                if d.is_dir() and d.name.lower().startswith(prefix):
                    hit = pair_video_in(str(d))
                    if hit:
                        return hit
    except OSError:
        pass
    return None


def mark_already_paired(wq: FlashVSRWorkQueue, after_dir: str) -> int:
    """Mark queue rows done when their After file already exists (PID in the name)."""
    n = 0
    for it in list(wq.all_items()):
        if it.get("status") == "done":
            continue
        after_p = find_existing_pair(after_dir, it)
        if not after_p:
            continue
        path = it.get("path") or ""
        if not path:
            continue
        wq.update_item(
            path,
            gt_after=after_p,
            gt_export=after_p,
            gt_pair_folder=it.get("gt_pair_folder"),
            gt_pair_id=it.get("gt_pair_id"),
        )
        wq.set_item_status(path, "done", output=after_p)
        n += 1
    return n


def cleanup_empty_work(app_dir: str) -> int:
    """Remove empty Group Therapy work stage folders."""
    root = work_root(app_dir)
    removed = 0
    try:
        for d in root.iterdir():
            if not d.is_dir():
                continue
            try:
                if not any(d.iterdir()):
                    d.rmdir()
                    removed += 1
            except OSError:
                continue
    except OSError:
        return removed
    return removed


_CLIP_VARIANT_RE = re.compile(
    r"^(?P<base>.*?)(?:\s*\((?P<v1>\d+)\)|_+\((?P<v2>\d+)\))(?P<rest>.*)$"
)


def clip_identity(stem: str) -> Tuple[str, str]:
    """
    Distinguish grok-video-UUID vs grok-video-UUID (2) vs (1).
    Returns (base, variant) where variant is '' for the unnumbered file.
    """
    text = Path(str(stem or "")).stem
    m = _CLIP_VARIANT_RE.match(text)
    if m and (m.group("v1") or m.group("v2")):
        base = re.sub(r"[_\s]+$", "", m.group("base") or "").lower()
        var = m.group("v1") or m.group("v2") or ""
        return base, var
    return text.lower().replace(" ", "_"), ""


def flatten_gt_pair_folders(
    *roots: str,
    skip_ids: Optional[Iterable[str]] = None,
    stamp_title: bool = True,
    remap_high_range: bool = True,
) -> Dict[str, Any]:
    """
    Retroactively flatten GT-<id>__name folders into the parent directory.

    Pairing becomes `_PID_<id>` on the filename + Title=PID_<id> (not tags).
    By default, existing GT ids are remapped into the 9xxxxxxx band so they
    cannot collide with new auto uuid-hex batches. The same old→new map is
    used on every root so Before/After stay matched.
    """
    skip = {str(x).lower() for x in (skip_ids or []) if str(x).strip()}
    usable_roots = [os.path.abspath(r) for r in roots if r and os.path.isdir(r)]
    stats: Dict[str, Any] = {
        "folders": 0,
        "moved": 0,
        "stamped": 0,
        "errors": 0,
        "removed_dirs": 0,
        "skipped_running": 0,
        "mapping": 0,
    }
    if not usable_roots:
        return stats

    mapping = load_retro_pid_map(*usable_roots)
    used = set(mapping.values())
    # First pass: collect GT folders and assign ids
    jobs = []  # (root, dirpath, old_id, new_id)
    for root in usable_roots:
        try:
            dirs = [p for p in Path(root).iterdir() if p.is_dir()]
        except OSError:
            continue
        for d in dirs:
            if d.name.lower() in _SKIP_DIR_NAMES or d.name.lower().startswith("batch_"):
                continue
            m = _PAIR_FOLDER_RE.match(d.name)
            if not m:
                continue
            old_id = m.group(1).lower()
            if old_id in skip:
                stats["skipped_running"] += 1
                continue
            if remap_high_range:
                new_id = mapping.get(old_id)
                if not new_id:
                    new_id = mint_retro_pid(old_id, used)
                    mapping[old_id] = new_id
            else:
                new_id = old_id
                used.add(old_id)
            jobs.append((root, d, old_id, new_id))

    if mapping:
        extra_roots = list(usable_roots)
        try:
            extra_roots.append(str(Path(usable_roots[0]).parent))
        except (IndexError, OSError):
            pass
        save_retro_pid_map(mapping, *extra_roots)
        stats["mapping"] = len(mapping)

    to_stamp: List[Tuple[str, str]] = []
    for root, d, old_id, new_id in jobs:
        stats["folders"] += 1
        try:
            files = [f for f in d.iterdir() if f.is_file()]
        except OSError:
            stats["errors"] += 1
            continue
        for f in files:
            if f.suffix.lower() not in VIDEO_EXTS:
                if f.name.lower() in {"pair.txt", "pair.json"}:
                    try:
                        f.unlink()
                    except OSError:
                        pass
                continue
            dest_name = with_pid_name(f.name, new_id)
            dest = Path(root) / dest_name
            if dest.exists():
                dest = Path(unique_pid_dest(root, dest_name))
            try:
                shutil.move(str(f), str(dest))
                append_pair_index(
                    root, pair_id=new_id, folder=".", role="flat", file_path=str(dest)
                )
                to_stamp.append((str(dest), new_id))
                stats["moved"] += 1
            except OSError:
                stats["errors"] += 1
        try:
            leftover = list(d.iterdir())
            if not leftover:
                d.rmdir()
                stats["removed_dirs"] += 1
            elif all(x.name.lower() in {"pair.txt", "pair.json"} for x in leftover):
                for x in leftover:
                    try:
                        x.unlink()
                    except OSError:
                        pass
                try:
                    d.rmdir()
                    stats["removed_dirs"] += 1
                except OSError:
                    pass
        except OSError:
            pass

    if stamp_title:
        for path, pid in to_stamp:
            if stamp_title_pid(path, pid):
                stats["stamped"] += 1
    return stats


def running_pair_ids_from_status(status_path: str) -> List[str]:
    """Pair ids on non-done STATUS lines (WAIT/RUN/FAIL) — skip these while flattening."""
    out = []
    try:
        text = Path(status_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    for line in text.splitlines():
        if "[OK" in line[:10]:
            continue
        m = re.search(r"pair=([0-9a-f]{8})", line, re.I)
        if m:
            out.append(m.group(1).lower())
    return out


def collect_temp_globs(stem_hint: str, *folders: str) -> List[str]:
    """Find leftover temps for THIS clip only — not UUID siblings like (1)/(2)."""
    found: List[str] = []
    ident = clip_identity(stem_hint)
    if not ident[0]:
        return found
    media = VIDEO_EXTS | {".jpg", ".jpeg", ".png"}
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        try:
            for f in Path(folder).iterdir():
                if not f.is_file() or f.suffix.lower() not in media:
                    continue
                if clip_identity(f.stem) == ident:
                    found.append(str(f))
        except OSError:
            continue
    return found


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flatten GT-* pair folders to _PID_ names.")
    parser.add_argument("roots", nargs="*", help="Before/After folders to flatten")
    parser.add_argument("--status", default="", help="Group Therapy STATUS.txt to skip in-progress ids")
    parser.add_argument("--no-stamp", action="store_true", help="Skip Title metadata rewrite")
    parser.add_argument("--keep-ids", action="store_true", help="Reuse original GT ids instead of 9xxxxxxx")
    args = parser.parse_args()
    skip = running_pair_ids_from_status(args.status) if args.status else []
    result = flatten_gt_pair_folders(
        *args.roots,
        skip_ids=skip,
        stamp_title=not args.no_stamp,
        remap_high_range=not args.keep_ids,
    )
    print(json.dumps(result, indent=2))
