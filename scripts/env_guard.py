#!/usr/bin/env python3
"""
FlashVSR env guard — snapshot custom files, reapply them after Update/Install,
free Windows locks, verify/repair safetensors.

Used by:
  start.js   → preflight  (repair + reapply if stock overwrite detected)
  update.js  → snapshot, stop holders, post_update (always reapply)
  install.js → reapply after a stock `git clone` into app/
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
ENV = APP / "env"
PRESERVE_ROOT = ROOT / "local-preserve"
PRESERVE_LATEST = PRESERVE_ROOT / "latest"
OFFLINE_BACKUPS = Path(r"C:\pinokio\backups\FlashVSR_plus_pinokio")
# Survives a wipe of C:\pinokio. Logon task reads this copy.
EXTERNAL_DIR_NAME = "FlashVSR-preserve"

# Junctions. Reset must rmdir these before fs.rm so the targets stay.
LINKS = (
    ("app/models", Path(r"O:\MODELS")),
    ("app/outputs", Path(r"D:\OUTPUTS\FlashVSR")),
)

# Relative to ROOT — custom + launcher files that must survive Update/Reset.
# Snapshot before pull; reapply after pull/clone. Markers detect stock overwrite.
CRITICAL = [
    "app/webui.py",
    "app/session_board.py",
    "app/stitch_canvas.py",
    "app/src/models/wan_video_dit.py",
    "app/gpu_headroom.py",
    "app/group_therapy.py",
    "app/flashvsr_work_queue.py",
    "app/naming_utils.py",
    "app/grok_id_index.py",
    "app/webui_config",
    "app/webui_config.bak",
    "app/CUSTOMIZATIONS.md",
    "app/toolbox/toolbox.py",
    "app/toolbox/rife_core.py",
    "app/toolbox/batch_queue.py",
    "app/src/pipelines/flashvsr_tiny_long.py",
    "app/src/pipelines/flashvsr_tiny.py",
    "app/src/pipelines/flashvsr_full.py",
    "app/src/busy_heartbeat.py",
    "app/src/models/TCDecoder.py",
    "app/src/models/wan_video_vae.py",
    "app/toolbox/system_monitor.py",
    "app/toolbox/thrash_alert.py",
    "app/toolbox/thrash_config.json",
    "app/requirements.txt",
    "app/Launch-FlashVSR-Plus.bat",
    "app/Launch-FlashVSR-Plus.ps1",
    "PRESERVE.md",
    "scripts/env_guard.py",
    "scripts/backup-to-vidupscaler.ps1",
    "start.js",
    "update.js",
    "install.js",
    "reset.js",
    "pinokio.js",
    "pinokio_meta.json",
    "torch.js",
    "link.js",
    # Gitignored user state. Restore only when the file is missing.
    "app/job_profiles.json",
    # RIFE weights (~12 MB). Restore only when the folder is missing or empty.
    "app/toolbox/model_rife",
    # Fork files that are not in upstream app/. A stock clone leaves its own
    # copy on disk, so markers on webui.py trigger a cascade restore of these.
    ".gitignore",
    "PATCHES.md",
    "README.md",
    "RESTORE.md",
    "UPGRADES.md",
    "pinokio.json",
    "app/.gitignore",
    "app/LICENSE",
    "app/README.md",
    "app/requirements-freeze-after-upgrade.txt",
    "app/requirements-freeze-before-upgrade.txt",
    "app/run.py",
    "app/teaser.jpg",
    "app/test_busy_heartbeat_metrics.py",
    "app/test_grok_id_dupes.py",
    "app/test_pipeline_stage_skip.py",
    "app/test_session_board.py",
    "app/test_thrash_alert.py",
    "app/src/__init__.py",
    "app/src/configs/__init__.py",
    "app/src/configs/model_config.py",
    "app/src/models/__init__.py",
    "app/src/models/model_manager.py",
    "app/src/models/utils.py",
    "app/src/models/sparse_sage/LICENSE.txt",
    "app/src/models/sparse_sage/core.py",
    "app/src/models/sparse_sage/quant_per_block.py",
    "app/src/models/sparse_sage/sparse_int8_attn.py",
    "app/src/pipelines/__init__.py",
    "app/src/pipelines/base.py",
    "app/src/schedulers/__init__.py",
    "app/src/schedulers/flow_match.py",
    "app/src/vram_management/__init__.py",
    "app/src/vram_management/layers.py",
    "app/toolbox/RIFE/IFNet_HDv3.py",
    "app/toolbox/RIFE/RIFE_HDv3.py",
    "app/toolbox/RIFE/__int__.py",
    "app/toolbox/RIFE/loss.py",
    "app/toolbox/RIFE/warplayer.py",
    "scripts/quarantine_degraded_civ.py",
    "scripts/restore_over4k_originals.py",
    "scripts/Restore-After-Core-Reinstall.ps1",
]

# Strings that MUST appear in a live custom file. Missing ⇒ stock overwrite.
MARKERS = {
    "app/webui.py": ("def run_group_therapy", "with_pid_name", "gt_before_dir", "no size recode", "accurate_rnd+full_chroma_int", "orientation_input_box", "_gt_rife_flags", "resolve_resize_encode", "tonemap=hable", "open_media_folder", "open_video_output_folder", "session_board.render", "session_board.render_job", "session_board.load_stage_colors", "thrash_alert.render_overlay", "_log_queue_scan", "def path_textbox", "HeartbeatTqdm", "handle_scan_grok_outputs", "chrome_copy_number", "apply_saved_gpu_headroom", "gpu_multitask_g", "stop_finish_path", "tick_stop_banner", "alloc_stitch_canvases", "GPU scheduling retired", "_enable_fast_gpu_math", '"randomize_seed": (True'),
    "app/stitch_canvas.py": ("class DiskStitchCanvas", "torch.from_file", "release_host_cache"),
    "app/src/models/wan_video_dit.py": ("stream KV clone drops prefix storage",),
    "app/gpu_headroom.py": ("apply_gpu_headroom", "D3DKMTSetProcessSchedulingPriorityClass", "nvidia-smi", "GPU scheduling retired", "always Normal"),
    "app/grok_id_index.py": ("extract_grok_ids", "SIZE_TOLERANCE", "screen_folder", "def version_key"),
    "app/src/busy_heartbeat.py": ("class HeartbeatTqdm", "still busy", "STOP ARMED", "tick_stop_banner", "def snapshot", "def set_file_metrics", "file_metrics"),
    "app/session_board.py": ("def render", "def classify_health", "def save_profile", "load_stage_colors", "gt_upscale_secs", "def now_banner"),
    "app/toolbox/system_monitor.py": ("power.draw", "last_metrics"),
    "app/toolbox/thrash_alert.py": ("def thrash_score", "def render_overlay"),
    "app/toolbox/thrash_config.json": ("thrash_at", "watts_per_util_start"),
    "app/src/pipelines/flashvsr_tiny.py": ("BusySpan", "show_progress_bar=True", "drop_dit_stream_kv", "park stream KV"),
    "app/group_therapy.py": ("stamp_title_pid", "flatten_gt_pair_folders", "pid_token", "item_in_progress", "version_keys_in_folder", "reclaim_watch_folder", "get_after_lookup", "migrate_legacy_pair_sidecars", "_pairs", "looks_like_finished_export", "moved_after", "mark_finished_exports", "peels unstarted siblings"),
    "app/flashvsr_work_queue.py": ("gt_pair_id", "drop_wrong_stage_pending", "class AddResult", "grok_id_dupes", "first_matching_version", "looks_like_finished_export", "seed_pipeline_fields", "lock_holder_matches"),
    "app/naming_utils.py": ("clean_original_stem", "step1_filename"),
    "app/toolbox/toolbox.py": ("_choose_interp_factor", "_has_video_stream"),
    "app/src/pipelines/flashvsr_tiny_long.py": (
        "tiny-long pipeline requires output_path",
    ),
    "scripts/env_guard.py": (
        "def reapply",
        "source_is_good",
        "app/gpu_headroom.py",
        "app/toolbox/thrash_alert.py",
        "FlashVSR-preserve",
        "ensure_links",
        "job_profiles.json",
    ),
    "update.js": ("env_guard.py reapply", "env_guard.py snapshot"),
    "install.js": ("env_guard.py reapply", "mklink /J"),
    "start.js": ("env_guard.py preflight", "PYTHONUNBUFFERED"),
    "pinokio.js": ("AUTOMATICALLY reapply",),
    "reset.js": ("Drop models and outputs junctions before fs.rm",),
}

# Restore these only when missing (no markers to detect overwrite).
CONFIG_FILES = {"app/webui_config.bak", "app/job_profiles.json"}

# Directories restored only when missing or empty (weights, not code).
DIR_IF_MISSING = {"app/toolbox/model_rife"}

# User pipeline paths — stock/empty config gets replaced; live custom config is kept.
MARKERS["app/webui_config"] = ("gt_before_dir=", "gt_after_dir=", "batch_watch_folder=", "resize_kernel=", "gpu_multitask=", "randomize_seed=True", "chunk_duration=10.25")

SAFETENSORS_PIN = "safetensors~=0.6.0"


def _venv_python() -> Path:
    if os.name == "nt":
        return ENV / "Scripts" / "python.exe"
    return ENV / "bin" / "python"


def _venv_uv_env() -> dict:
    """Prefer venv python for subprocesses that reinstall packages."""
    env = os.environ.copy()
    py = _venv_python()
    if py.exists():
        scripts = str(py.parent)
        env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(ENV)
    return env


def log(msg: str) -> None:
    print(f"[env_guard] {msg}", flush=True)


def _copy_rel(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)


def snapshot(also_offline: bool = True) -> Path:
    PRESERVE_LATEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel in CRITICAL:
        src = ROOT / rel
        if not src.exists():
            continue
        # Never walk the models/outputs junctions into O: or D:.
        if rel in {link[0] for link in LINKS}:
            continue
        try:
            _copy_rel(src, PRESERVE_LATEST / rel)
        except OSError as e:
            log(f"snapshot skip {rel}: {e}")
            continue
        copied += 1
    stamp_file = PRESERVE_LATEST / "SNAPSHOT.txt"
    stamp_file.write_text(
        f"snapshot_time={datetime.now().isoformat()}\n"
        f"files_copied={copied}\n"
        f"root={ROOT}\n",
        encoding="utf-8",
    )
    log(f"snapshot → {PRESERVE_LATEST} ({copied} paths)")

    if also_offline and os.name == "nt":
        try:
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            offline = OFFLINE_BACKUPS / stamp
            offline.mkdir(parents=True, exist_ok=True)
            for rel in CRITICAL:
                src = ROOT / rel
                if not src.exists() or rel in {link[0] for link in LINKS}:
                    continue
                try:
                    _copy_rel(src, offline / rel)
                except OSError as e:
                    log(f"offline skip {rel}: {e}")
            (offline / "SNAPSHOT.txt").write_text(
                stamp_file.read_text(encoding="utf-8"), encoding="utf-8"
            )
            log(f"offline backup → {offline}")
        except OSError as e:
            log(f"offline backup skipped: {e}")

    publish_external()
    return PRESERVE_LATEST


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def source_is_good(path: Path, rel: str) -> bool:
    if not path:
        return False
    if path.is_dir():
        if rel not in DIR_IF_MISSING:
            return False
        try:
            return any(path.iterdir())
        except OSError:
            return False
    if not path.is_file():
        return False
    markers = MARKERS.get(rel)
    if not markers:
        return path.stat().st_size > 0
    text = _read_text(path)
    return all(m in text for m in markers)


def live_needs_reapply(rel: str) -> bool:
    dest = ROOT / rel
    if not dest.exists():
        return True
    if rel in CONFIG_FILES:
        return False
    if rel in MARKERS:
        return not source_is_good(dest, rel)
    return False


def _offline_dirs() -> List[Path]:
    if not OFFLINE_BACKUPS.is_dir():
        return []
    try:
        dirs = [p for p in OFFLINE_BACKUPS.iterdir() if p.is_dir()]
    except OSError:
        return []
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def git_blob(rel: str) -> Optional[bytes]:
    posix = rel.replace("\\", "/")
    for spec in (f"mine/main:{posix}", f"HEAD:{posix}"):
        try:
            r = subprocess.run(
                ["git", "show", spec],
                cwd=str(ROOT),
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout:
            return r.stdout
    return None


def git_blob_is_good(rel: str, blob: bytes) -> bool:
    markers = MARKERS.get(rel)
    if not markers:
        return bool(blob)
    try:
        text = blob.decode("utf-8", errors="ignore")
    except Exception:
        return False
    return all(m in text for m in markers)


def _external_root() -> Optional[Path]:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    return Path(base) / "FAFO" / EXTERNAL_DIR_NAME


def _mirror_tree(src: Path, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def _install_restore_task(script: Path) -> None:
    if os.name != "nt" or not script.is_file():
        return
    tr = (
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -File "
        f'"{script}" -Auto'
    )
    try:
        r = subprocess.run(
            [
                "schtasks",
                "/Create",
                "/TN",
                r"FAFO\FlashVSR Restore",
                "/SC",
                "ONLOGON",
                "/RL",
                "LIMITED",
                "/F",
                "/TR",
                tr,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"logon task skipped: {e}")
        _install_startup_cmd(script)
        return
    if r.returncode == 0:
        log(r"logon task FAFO\FlashVSR Restore installed")
    else:
        log(f"logon task skipped: {(r.stderr or r.stdout).strip()[:300]}")
    _install_startup_cmd(script)


def _install_startup_cmd(script: Path) -> None:
    """User Startup folder. schtasks /Create is denied without admin on this PC."""
    appdata = os.environ.get("APPDATA")
    if not appdata or not script.is_file():
        return
    startup = (
        Path(appdata)
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )
    try:
        startup.mkdir(parents=True, exist_ok=True)
        cmd = startup / "FlashVSR-Restore.cmd"
        cmd.write_text(
            "@echo off\r\n"
            "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass "
            f'-File "{script}" -Auto\r\n',
            encoding="utf-8",
        )
        log(f"logon startup → {cmd}")
    except OSError as e:
        log(f"logon startup skipped: {e}")


def publish_external() -> None:
    """Copy the snapshot outside C:\\pinokio and refresh the logon restore task."""
    root = _external_root()
    if root is None:
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
        _mirror_tree(PRESERVE_LATEST, root / "latest")
        script = ROOT / "scripts" / "Restore-After-Core-Reinstall.ps1"
        if script.is_file():
            shutil.copy2(script, root / script.name)
            _install_restore_task(root / script.name)
        log(f"external vault → {root}")
    except OSError as e:
        log(f"external vault skipped: {e}")


def pick_source(rel: str) -> Optional[Path]:
    """Best on-disk copy of a custom file (must pass markers when defined)."""
    candidates = [PRESERVE_LATEST / rel]
    ext = _external_root()
    if ext is not None:
        candidates.append(ext / "latest" / rel)
    for offline in _offline_dirs():
        candidates.append(offline / rel)
    for c in candidates:
        if source_is_good(c, rel):
            return c
    return None


def _write_bytes(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".reapply_tmp")
    tmp.write_bytes(data)
    os.replace(str(tmp), str(dest))


def restore_one(rel: str, *, force: bool = False, cascade: bool = False) -> str:
    """
    Restore one CRITICAL path.
    Returns: 'ok' | 'restored' | 'missing' | 'skipped'
    """
    dest = ROOT / rel
    if rel in DIR_IF_MISSING:
        try:
            populated = dest.exists() and any(dest.iterdir())
        except OSError:
            populated = dest.exists()
        if populated and not force:
            return "ok"
        src = pick_source(rel)
        if src is None or not src.is_dir():
            if not dest.exists():
                log(f"WARNING: no good source for {rel}")
                return "missing"
            return "ok"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
        log(f"reapplied {rel} ← {src}")
        return "restored"

    if rel in CONFIG_FILES:
        need = force or (not dest.exists())
    elif rel in MARKERS:
        need = force or live_needs_reapply(rel)
    else:
        # Unmarked fork files: stock clone still has a file, so replace them
        # only when webui.py itself lost its markers (cascade) or the file is gone.
        need = force or (not dest.exists()) or cascade
    if not need:
        return "ok"

    src = pick_source(rel)
    if src is not None:
        if dest.exists():
            try:
                if _file_digest(src) == _file_digest(dest):
                    return "ok"
            except OSError:
                pass
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        log(f"reapplied {rel} ← {src}")
        return "restored"

    blob = git_blob(rel)
    if blob and git_blob_is_good(rel, blob):
        if dest.exists():
            try:
                if hashlib.sha256(dest.read_bytes()).hexdigest() == hashlib.sha256(blob).hexdigest():
                    return "ok"
            except OSError:
                pass
        _write_bytes(dest, blob)
        log(f"reapplied {rel} ← git mine/HEAD")
        return "restored"

    if dest.exists() and not live_needs_reapply(rel):
        return "ok"
    log(f"WARNING: no good source for {rel}")
    return "missing"


def reapply(*, force: bool = False) -> int:
    """
    Overlay custom files after Pinokio Update/Install/Reset.

    force=True: copy every CRITICAL path from the newest good snapshot.
    force=False: restore missing files, files that lost custom markers, and
                (when webui.py is stock) the rest of the fork overlay.
    Never copies a source that fails MARKERS — a stale snapshot cannot
    clobber Group Therapy / PID pairing.
    """
    cascade = (not force) and live_needs_reapply("app/webui.py")
    if cascade:
        log("stock webui.py detected; restoring the rest of the fork")
    restored = 0
    missing = 0
    ok = 0
    for rel in CRITICAL:
        result = restore_one(rel, force=force, cascade=cascade)
        if result == "restored":
            restored += 1
        elif result == "missing":
            missing += 1
        else:
            ok += 1
    log(f"reapply force={force} restored={restored} already_ok={ok} missing={missing}")
    stamp = PRESERVE_LATEST / "REAPPLY.txt"
    try:
        PRESERVE_LATEST.mkdir(parents=True, exist_ok=True)
        stamp.write_text(
            f"reapply_time={datetime.now().isoformat()}\n"
            f"force={force}\n"
            f"restored={restored}\n"
            f"already_ok={ok}\n"
            f"missing={missing}\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    # missing custom sources is a warning, not a hard fail — Start should still try
    return 0


def restore_webui_config_if_missing() -> None:
    cfg = APP / "webui_config"
    bak = PRESERVE_LATEST / "app" / "webui_config"
    if not cfg.exists() and bak.exists():
        shutil.copy2(bak, cfg)
        log(f"restored missing webui_config from {bak}")


def safetensors_ok() -> bool:
    code = (
        "from safetensors import safe_open; "
        "import safetensors, os, sys; "
        "p = safetensors.__file__; "
        "ok = bool(p) and os.path.isfile(p) and p.replace('\\\\','/').endswith('safetensors/__init__.py'); "
        "print(p or ''); "
        "sys.exit(0 if ok else 1)"
    )
    py = _venv_python()
    if not py.exists():
        log(f"venv python missing: {py}")
        return False
    try:
        r = subprocess.run(
            [str(py), "-c", code],
            cwd=str(APP),
            capture_output=True,
            text=True,
            env=_venv_uv_env(),
            timeout=60,
        )
        out = (r.stdout or "").strip()
        if r.returncode == 0:
            log(f"safetensors OK → {out}")
            return True
        log(f"safetensors BAD (exit {r.returncode}): {(r.stderr or out)[:500]}")
        return False
    except Exception as e:
        log(f"safetensors check failed: {e}")
        return False


def _run(cmd: list[str], check: bool = False) -> int:
    log(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(APP), env=_venv_uv_env())
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r.returncode


def repair_safetensors() -> bool:
    """Force-clean reinstall of pinned safetensors (Windows-safe when unlocked)."""
    site = ENV / "Lib" / "site-packages"
    if not site.exists():
        site = ENV / "lib" / "site-packages"

    log(f"repairing {SAFETENSORS_PIN}")
    # uninstall (uv has no -y)
    _run(["uv", "pip", "uninstall", "safetensors"])
    # force-remove leftovers (namespace package / locked partial installs)
    for pattern in ("safetensors", "safetensors-*.dist-info"):
        if "*" in pattern:
            for p in site.glob(pattern):
                shutil.rmtree(p, ignore_errors=True)
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass
        else:
            p = site / pattern
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    rc = _run(["uv", "pip", "install", SAFETENSORS_PIN])
    if rc != 0:
        log("uv pip install failed")
        return False
    return safetensors_ok()


def flashvsr_python_pids() -> list[int]:
    """PIDs of python.exe whose command line references this FlashVSR tree."""
    if os.name != "nt":
        return []
    needle = str(ROOT).replace("'", "''")
    ps = (
        f"$n = '{needle}'; "
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -and ($_.CommandLine -like \"*$n*\") } | "
        "ForEach-Object { $_.ProcessId }"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=30,
        )
        pids = []
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids
    except Exception as e:
        log(f"pid scan failed: {e}")
        return []


def pyd_unlocked() -> bool:
    pyd = (
        ENV
        / "Lib"
        / "site-packages"
        / "safetensors"
        / "_safetensors_rust.pyd"
    )
    if not pyd.exists():
        pyd = (
            ENV
            / "lib"
            / "site-packages"
            / "safetensors"
            / "_safetensors_rust.pyd"
        )
    if not pyd.exists():
        return True
    try:
        # exclusive open
        with open(pyd, "r+b"):
            pass
        return True
    except OSError as e:
        log(f"pyd still locked: {e}")
        return False


def stop_holders(force: bool = True) -> None:
    """Stop python processes holding this app's venv (Windows .pyd lock)."""
    pids = flashvsr_python_pids()
    if not pids:
        log("no FlashVSR python holders")
        return
    log(f"stopping holders: {pids}")
    if os.name == "nt" and force:
        for pid in pids:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                capture_output=True,
            )
        time.sleep(1.5)
    if not pyd_unlocked():
        log("WARNING: safetensors .pyd still locked after stop — pip upgrade may fail")
    else:
        log("lock clear")


def _norm_target(value: str) -> str:
    v = (value or "").replace("/", "\\")
    for prefix in ("\\\\?\\", "\\??\\"):
        if v.startswith(prefix):
            v = v[len(prefix):]
    if not v:
        return ""
    return os.path.normcase(os.path.normpath(v))


def _is_reparse(path: Path) -> bool:
    try:
        st = path.lstat()
    except OSError:
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(st, "st_file_attributes", 0) & flag)


def _make_junction(link: Path, target: Path) -> bool:
    if os.name != "nt":
        os.symlink(str(target), str(link), target_is_directory=True)
        log(f"linked {link} → {target}")
        return True
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        log(f"mklink failed {link}: {(r.stderr or r.stdout or '').strip()}")
        return False
    log(f"linked {link.relative_to(ROOT)} → {target}")
    return True


def ensure_links() -> None:
    """Recreate models and outputs junctions when they are missing."""
    for rel, target in LINKS:
        dest = ROOT / rel
        if not target.is_dir():
            log(f"link skipped {rel}: target missing {target}")
            continue
        if _is_reparse(dest):
            current = ""
            try:
                current = os.readlink(dest)
            except OSError:
                current = ""
            if _norm_target(current) == _norm_target(str(target)):
                log(f"link ok {rel} → {target}")
            else:
                log(f"link left as-is {rel} → {current or dest}")
            continue
        if dest.exists():
            try:
                populated = any(dest.iterdir())
            except OSError:
                populated = True
            if populated:
                log(f"link skipped {rel}: folder is not empty and is not a junction")
                continue
            try:
                dest.rmdir()
            except OSError as e:
                log(f"link skipped {rel}: {e}")
                continue
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
        _make_junction(dest, target)


def retire_nested_app_git() -> None:
    """A stock Install leaves app/.git. A later Update would pull upstream over the fork."""
    nested = APP / ".git"
    if not nested.exists():
        return
    if live_needs_reapply("app/webui.py"):
        log("nested app/.git kept; webui.py still missing custom markers")
        return
    retired = APP / ".git.stock-retired"
    try:
        if retired.exists():
            shutil.rmtree(retired, ignore_errors=True)
        nested.rename(retired)
    except OSError as e:
        log(f"could not retire nested app/.git: {e}")
        return
    log("retired nested app/.git so Update will not pull stock FlashVSR over this fork")


def status() -> int:
    """Print whether each preserved path would be restored. Does not write."""
    cascade = live_needs_reapply("app/webui.py")
    log(f"sentinel_stock={cascade}")
    needs = 0
    for rel in CRITICAL:
        dest = ROOT / rel
        if rel in DIR_IF_MISSING or rel in CONFIG_FILES:
            state = "missing" if not dest.exists() else "present"
        elif rel in MARKERS:
            state = "needs" if live_needs_reapply(rel) else "ok"
        elif not dest.exists():
            state = "missing"
        elif cascade:
            state = "cascade"
        else:
            state = "ok"
        if state in {"needs", "missing", "cascade"}:
            needs += 1
        print(f"{state}\t{rel}", flush=True)
    for rel, target in LINKS:
        dest = ROOT / rel
        if _is_reparse(dest):
            print(f"link\t{rel} → {target}", flush=True)
        elif dest.exists():
            print(f"link-real\t{rel}", flush=True)
            needs += 1
        else:
            print(f"link-missing\t{rel} → {target}", flush=True)
            needs += 1
    return 1 if needs else 0


def prepare() -> int:
    """Junctions, reapply, and retire a stock nested app/.git. Does not stop webui."""
    ensure_links()
    rc = reapply(force=False)
    retire_nested_app_git()
    restore_webui_config_if_missing()
    return rc


def preflight() -> int:
    """Run before webui.py. Reapply custom files if stock-overwritten, then safetensors."""
    ensure_links()
    reapply(force=False)
    retire_nested_app_git()
    restore_webui_config_if_missing()
    if safetensors_ok():
        return 0
    log("safetensors broken — attempting repair")
    stop_holders()
    if not pyd_unlocked():
        log("ERROR: cannot repair while .pyd is locked. Stop the app and retry.")
        return 2
    if repair_safetensors():
        return 0
    log("ERROR: safetensors repair failed")
    return 1


def post_update() -> int:
    """After git pull + pip: restore missing/stock-overwritten custom files, verify safetensors."""
    ensure_links()
    reapply(force=False)
    retire_nested_app_git()
    restore_webui_config_if_missing()
    if safetensors_ok():
        return 0
    log("post-update safetensors broken — repairing")
    if not pyd_unlocked():
        stop_holders()
    if not pyd_unlocked():
        log("ERROR: still locked after update")
        return 2
    return 0 if repair_safetensors() else 1


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: env_guard.py snapshot|reapply|prepare|status|stop_holders|preflight|post_update|verify|repair",
            file=sys.stderr,
        )
        return 2
    cmd = argv[1]
    if cmd == "snapshot":
        snapshot()
        return 0
    if cmd == "reapply":
        force = "--force" in argv[2:]
        return reapply(force=force)
    if cmd == "prepare":
        return prepare()
    if cmd == "status":
        return status()
    if cmd == "stop_holders":
        stop_holders()
        return 0
    if cmd == "verify":
        return 0 if safetensors_ok() else 1
    if cmd == "repair":
        stop_holders()
        return 0 if repair_safetensors() else 1
    if cmd == "preflight":
        return preflight()
    if cmd == "post_update":
        return post_update()
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
