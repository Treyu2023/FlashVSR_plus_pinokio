"""
Purge CRF-12 inbox recodes and restore Over4K originals into NEW DOWNLOADS.

- Moves recodes to Post Scaling\\degraded\\crf12_inbox_<stamp>\\
- Moves matching Over4K originals back to the watch folder
- Copies larger Pre Scaled originals when inbox-only grok files look recoded
- Moves already-upscaled (UpScale*/_resized_) files out of the watch folder
  so they are not 4×'d again

Does not delete originals.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

WATCH = Path(r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS")
O4K = WATCH / "Over4K"
PRE = Path(r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos")
DEGRADED = Path(r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\degraded")
VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
PIPE = re.compile(
    r"(?i)(^upscale|_upscaled_|_exported_|_resized_|_chunked|_frames_|upscale\d+k_|upscale\d+p_)"
)


def videos(folder: Path) -> dict[str, Path]:
    out = {}
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO:
            out[p.name] = p
    return out


def unique(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    if not dest.exists():
        return dest
    stem, ext = dest.stem, dest.suffix
    n = 1
    while True:
        cand = dest_dir / f"{stem}__q{n}{ext}"
        if not cand.exists():
            return cand
        n += 1


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recode_dir = DEGRADED / f"crf12_inbox_{stamp}"
    wrong_dir = DEGRADED / f"wrong_stage_in_downloads_{stamp}"
    recode_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    inbox = videos(WATCH)
    o4k = videos(O4K)
    pre = videos(PRE)
    rows = []

    restored_o4k = 0
    restored_pre = 0
    wrong_stage = 0
    errors = 0

    both = sorted(set(inbox) & set(o4k))
    for name in both:
        recode = inbox[name]
        orig = o4k[name]
        qdest = unique(recode_dir, name)
        try:
            shutil.move(str(recode), str(qdest))
            shutil.move(str(orig), str(WATCH / name))
            restored_o4k += 1
            rows.append(
                {
                    "action": "restore_over4k",
                    "name": name,
                    "recode_mb": round(qdest.stat().st_size / 1e6, 2),
                    "orig_mb": round((WATCH / name).stat().st_size / 1e6, 2),
                    "quarantine": str(qdest),
                }
            )
        except OSError as e:
            errors += 1
            rows.append({"action": "error_over4k", "name": name, "error": str(e)})

    inbox = videos(WATCH)
    o4k_left = videos(O4K)
    for name, src in sorted(inbox.items()):
        if name in o4k_left:
            continue
        if PIPE.search(name):
            dest = unique(wrong_dir, name)
            try:
                shutil.move(str(src), str(dest))
                wrong_stage += 1
                rows.append({"action": "wrong_stage", "name": name, "quarantine": str(dest)})
            except OSError as e:
                errors += 1
                rows.append({"action": "error_wrong_stage", "name": name, "error": str(e)})
            continue
        orig = pre.get(name)
        if orig and orig.stat().st_size > src.stat().st_size * 1.2:
            qdest = unique(recode_dir, name)
            try:
                shutil.move(str(src), str(qdest))
                shutil.copy2(str(orig), str(WATCH / name))
                restored_pre += 1
                rows.append(
                    {
                        "action": "restore_prescaled",
                        "name": name,
                        "recode_mb": round(qdest.stat().st_size / 1e6, 2),
                        "orig_mb": round((WATCH / name).stat().st_size / 1e6, 2),
                        "from": str(orig),
                    }
                )
            except OSError as e:
                errors += 1
                rows.append({"action": "error_prescaled", "name": name, "error": str(e)})

    summary = {
        "when": stamp,
        "watch": str(WATCH),
        "restored_from_over4k": restored_o4k,
        "restored_from_prescaled": restored_pre,
        "wrong_stage_moved": wrong_stage,
        "errors": errors,
        "recode_quarantine": str(recode_dir),
        "wrong_stage_quarantine": str(wrong_dir),
        "inbox_videos_now": len(videos(WATCH)),
        "over4k_left": len(videos(O4K)),
        "rows": rows,
    }
    log_path = recode_dir / "RESTORE_LOG.json"
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = recode_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "CRF-12 / whole-folder downscales purged from NEW DOWNLOADS.",
                f"When: {stamp}",
                f"Restored Over4K originals: {restored_o4k}",
                f"Restored larger Pre Scaled copies: {restored_pre}",
                f"Already-upscaled files moved out of watch folder: {wrong_stage}",
                f"Errors: {errors}",
                "Recodes kept here for comparison (not deleted).",
                "Restart FlashVSR in Pinokio, then run Group Therapy / Video queue.",
                "Downscale now happens per file: lanczos, CRF 10, follow source.",
                f"Log: {log_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
