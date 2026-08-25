"""
Quarantine non-96fps Ready-for-CIV (stage 3) videos, match originals,
copy originals back to NEW DOWNLOADS for restage.

Usage:
  python quarantine_degraded_civ.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

CIV = Path(r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV")
DOWNLOADS = Path(r"D:\OUTPUTS\__X_GROK\NEW DOWNLOADS")
PRE = Path(r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos")
DEGRADED = Path(r"D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\degraded")
MATCHED = DEGRADED / "matched"
UNMATCHED = DEGRADED / "unmatched"

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SKIP_DIR = {
    "miniconda", "ffmpeg-env", "ffmpeg-pkgs", "npm", "py", "bluefairy",
    "from_toolbox_inbox", "images", "_civ posted", "degraded", "matched",
    "unmatched", "novideo", "highfps", "bin", "done", "archive", "work",
    "_civit posted",
}
PIPE_HINT = re.compile(
    r"(?i)(_upscaled_|_exported_|_resized_|_chunked|_frames_|upscale\d+k_|upscale\d+p_)"
)
GROK_ID = re.compile(r"(?i)(grok-video-[0-9a-f]{8}(?:-[0-9a-f]{4}){0,4}[0-9a-f]*)")
GROK_IDX = re.compile(r"(?i)grok-video-[0-9a-f-]+[\s_]*\(?(\d+)\)?")
GEN_KEY = re.compile(r"(?i)(generated_video(?:_1080_hd)?(?:[\s_\-]*\(?\d+\)?)?)")
FFPROBE = "ffprobe"


def is_96(fps: float) -> bool:
    return 93.5 <= fps <= 98.5


def probe(path: Path) -> dict:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,color_range",
        "-of", "json", str(path),
    ]
    w = h = 0
    fps = 0.0
    cr = ""
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False).stdout
        st = (json.loads(raw).get("streams") or [{}])[0]
        w = int(st.get("width") or 0)
        h = int(st.get("height") or 0)
        cr = str(st.get("color_range") or "")
        for key in ("avg_frame_rate", "r_frame_rate"):
            val = str(st.get(key) or "")
            if "/" in val:
                a, b = val.split("/", 1)
                try:
                    den = float(b)
                    if den:
                        fps = float(a) / den
                        if fps > 1:
                            break
                except ValueError:
                    pass
            else:
                try:
                    fps = float(val)
                    if fps > 1:
                        break
                except ValueError:
                    pass
    except Exception as e:
        return {"path": str(path), "name": path.name, "w": 0, "h": 0, "fps": 0.0, "err": str(e)}
    return {
        "path": str(path), "name": path.name, "w": w, "h": h,
        "fps": round(fps, 3), "color_range": cr, "bytes": path.stat().st_size,
    }


def grok_key(name: str) -> str:
    m = GROK_ID.search(name)
    return m.group(1).lower().rstrip("-") if m else ""


def grok_idx(name: str) -> str:
    m = GROK_IDX.search(name)
    return m.group(1) if m else ""


def gen_key(name: str) -> str:
    m = GEN_KEY.search(name)
    if not m:
        return ""
    s = m.group(1).lower()
    s = re.sub(r"[\s_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def looks_original(name: str) -> bool:
    return not PIPE_HINT.search(name)


def walk_videos(root: Path):
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in SKIP_DIR and not d.lower().startswith("batch_")
            and not d.lower().startswith("gt-")
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in VIDEO_EXT:
                yield p


def unique_dest(folder: Path, name: str) -> Path:
    dest = folder / name
    if not dest.exists():
        return dest
    stem, ext = os.path.splitext(name)
    n = 2
    while True:
        cand = folder / f"{stem}__q{n}{ext}"
        if not cand.exists():
            return cand
        n += 1


def score_original(p: Path, civ_name: str) -> int:
    name = p.name
    sc = 0
    if looks_original(name):
        sc += 50
    if DOWNLOADS in p.parents or p.parent == DOWNLOADS:
        sc += 30
    if PRE in p.parents or p.parent == PRE:
        sc += 20
    gk = grok_key(civ_name)
    gi = grok_idx(civ_name)
    if gk and gk in name.lower():
        sc += 40
        if grok_key(name) == gk:
            sc += 20
        if gi and grok_idx(name) == gi:
            sc += 80
    try:
        sc += min(20, int(p.stat().st_size / (5 * 1024 * 1024)))
    except OSError:
        pass
    return sc


def index_originals():
    by_grok = {}
    by_gen = {}
    all_orig = []
    for root in (DOWNLOADS, PRE):
        for p in walk_videos(root):
            rec = {"path": p, "name": p.name, "grok": grok_key(p.name), "gen": gen_key(p.name)}
            all_orig.append(rec)
            if rec["grok"]:
                by_grok.setdefault(rec["grok"], []).append(p)
                # prefix buckets for truncated ids
                g = rec["grok"]
                if len(g) >= 18:
                    by_grok.setdefault(g[:18], []).append(p)
            if rec["gen"]:
                by_gen.setdefault(rec["gen"], []).append(p)
    return by_grok, by_gen, all_orig


def pick_original(civ_name: str, by_grok, by_gen):
    gk = grok_key(civ_name)
    cands = []
    if gk:
        cands.extend(by_grok.get(gk, []))
        if len(gk) >= 18:
            cands.extend(by_grok.get(gk[:18], []))
        # prefix match any longer stored id
        for k, ps in by_grok.items():
            if k.startswith(gk) or gk.startswith(k):
                cands.extend(ps)
    gg = gen_key(civ_name)
    if gg:
        cands.extend(by_gen.get(gg, []))
    # unique
    seen = set()
    uniq = []
    for p in cands:
        s = str(p).lower()
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    if not uniq:
        return None
    clean = [p for p in uniq if looks_original(p.name)]
    pool = clean or uniq
    pool.sort(key=lambda p: score_original(p, civ_name), reverse=True)
    return pool[0]


def main():
    DEGRADED.mkdir(parents=True, exist_ok=True)
    MATCHED.mkdir(exist_ok=True)
    UNMATCHED.mkdir(exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    civ_files = [
        p for p in CIV.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXT
    ]
    print(f"Stage 3 top-level videos: {len(civ_files)}")
    print("Probing FPS...")
    probed = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(probe, p): p for p in civ_files}
        for i, fut in enumerate(as_completed(futs), 1):
            probed.append(fut.result())
            if i % 100 == 0:
                print(f"  probed {i}/{len(civ_files)}")

    ok96 = [r for r in probed if is_96(r["fps"])]
    bad = [r for r in probed if not is_96(r["fps"])]
    salvage24 = [
        r for r in bad
        if 22.0 <= r["fps"] <= 26.0 and max(r["w"], r["h"]) >= 1280
    ]

    fps_buckets = {}
    for r in probed:
        bucket = int(round(r["fps"])) if r["fps"] else 0
        fps_buckets[bucket] = fps_buckets.get(bucket, 0) + 1

    print("FPS buckets:", dict(sorted(fps_buckets.items())))
    print(f"Keep in CIV (~96fps): {len(ok96)}")
    print(f"Degraded / restage (not 96fps): {len(bad)}")
    print(f"  of those 24fps scaled-up: {len(salvage24)}")

    print("Indexing originals in Downloads + Pre Scaled...")
    by_grok, by_gen, _all = index_originals()
    print(f"  grok keys: {len(by_grok)}")

    rows = []
    copied = 0
    already_in_dl = 0
    moved_m = 0
    moved_u = 0

    for r in bad:
        orig = pick_original(r["name"], by_grok, by_gen)
        tagged24 = 22.0 <= r["fps"] <= 26.0 and max(r["w"], r["h"]) >= 1280
        rec = {
            **r,
            "salvage_24_upscaled": tagged24,
            "original": str(orig) if orig else "",
            "copied_to_downloads": "",
        }
        src = Path(r["path"])
        if orig:
            dest_s3 = unique_dest(MATCHED, src.name)
            shutil.move(str(src), str(dest_s3))
            rec["quarantine"] = str(dest_s3)
            moved_m += 1
            dl_dest = DOWNLOADS / orig.name
            if dl_dest.exists():
                rec["copied_to_downloads"] = f"already:{dl_dest}"
                already_in_dl += 1
            elif orig.resolve() == dl_dest.resolve():
                rec["copied_to_downloads"] = f"already:{dl_dest}"
                already_in_dl += 1
            else:
                # original may already live in Downloads under another parent
                if DOWNLOADS in orig.parents or orig.parent == DOWNLOADS:
                    rec["copied_to_downloads"] = f"in_downloads:{orig}"
                    already_in_dl += 1
                else:
                    dl_dest = unique_dest(DOWNLOADS, orig.name)
                    shutil.copy2(str(orig), str(dl_dest))
                    rec["copied_to_downloads"] = str(dl_dest)
                    copied += 1
        else:
            dest_s3 = unique_dest(UNMATCHED, src.name)
            shutil.move(str(src), str(dest_s3))
            rec["quarantine"] = str(dest_s3)
            moved_u += 1
        rows.append(rec)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_json = DEGRADED / f"MATCH_LOG_{stamp}.json"
    log_csv = DEGRADED / f"MATCH_LOG_{stamp}.csv"
    log_txt = DEGRADED / "README.txt"
    with log_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "when": stamp,
                "civ": str(CIV),
                "degraded": str(DEGRADED),
                "kept_96fps": len(ok96),
                "quarantined": len(bad),
                "matched": moved_m,
                "unmatched": moved_u,
                "originals_copied_to_downloads": copied,
                "originals_already_in_downloads": already_in_dl,
                "salvage_24_upscaled": len(salvage24),
                "fps_buckets": fps_buckets,
                "rows": rows,
            },
            f,
            indent=2,
        )
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name", "fps", "w", "h", "salvage_24_upscaled",
                "original", "copied_to_downloads", "quarantine",
            ],
        )
        w.writeheader()
        for rec in rows:
            w.writerow({k: rec.get(k, "") for k in w.fieldnames})
    log_txt.write_text(
        "\n".join([
            "Quarantined non-96fps Ready-for-CIV (stage 3) videos.",
            f"When: {stamp}",
            f"Kept in Ready for CIV (~96 fps): {len(ok96)}",
            f"Moved here: {len(bad)}",
            f"  matched (original found) → matched\\ : {moved_m}",
            f"  unmatched → unmatched\\ : {moved_u}",
            f"  24fps but scaled-up (salvage): {len(salvage24)}",
            f"Originals copied into NEW DOWNLOADS: {copied}",
            f"Originals already in NEW DOWNLOADS: {already_in_dl}",
            "Good 96fps files were left in Ready for CIV.",
            "Re-run FlashVSR from NEW DOWNLOADS; 4K-safe downscale is now fit-scale + full chroma/color range.",
            f"Log: {log_csv.name}",
            "",
        ]),
        encoding="utf-8",
    )
    print("--- done ---")
    print(f"kept 96fps in CIV: {len(ok96)}")
    print(f"quarantined: {len(bad)}  matched={moved_m} unmatched={moved_u}")
    print(f"copied originals to downloads: {copied}  already there: {already_in_dl}")
    print(f"24fps scaled-up: {len(salvage24)}")
    print(f"log: {log_txt}")


if __name__ == "__main__":
    main()
