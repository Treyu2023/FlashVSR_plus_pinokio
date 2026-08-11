#!/usr/bin/env python3
"""
FlashVSR env guard — snapshot custom files, free Windows locks, verify/repair safetensors.

Used by start.js (preflight) and update.js (stop holders + post-update verify).
Does not modify app Python sources.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
ENV = APP / "env"
PRESERVE_ROOT = ROOT / "local-preserve"
PRESERVE_LATEST = PRESERVE_ROOT / "latest"
OFFLINE_BACKUPS = Path(r"C:\pinokio\backups\FlashVSR_plus_pinokio")

# Relative to ROOT — custom + launcher files that must survive Update/Reset recovery
CRITICAL = [
    "app/webui.py",
    "app/flashvsr_work_queue.py",
    "app/naming_utils.py",
    "app/webui_config",
    "app/webui_config.bak",
    "app/CUSTOMIZATIONS.md",
    "app/toolbox/toolbox.py",
    "app/toolbox/rife_core.py",
    "app/toolbox/batch_queue.py",
    "app/requirements.txt",
    "app/Launch-FlashVSR-Plus.bat",
    "app/Launch-FlashVSR-Plus.ps1",
    "start.js",
    "update.js",
    "install.js",
    "reset.js",
    "pinokio.js",
    "pinokio_meta.json",
    "torch.js",
    "link.js",
]

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


def snapshot(also_offline: bool = True) -> Path:
    PRESERVE_LATEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel in CRITICAL:
        src = ROOT / rel
        if not src.exists():
            continue
        dest = PRESERVE_LATEST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
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
                if not src.exists() or src.is_dir():
                    continue
                dest = offline / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            (offline / "SNAPSHOT.txt").write_text(
                stamp_file.read_text(encoding="utf-8"), encoding="utf-8"
            )
            log(f"offline backup → {offline}")
        except OSError as e:
            log(f"offline backup skipped: {e}")

    return PRESERVE_LATEST


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


def preflight() -> int:
    """Run before webui.py. Repair safetensors if import is broken."""
    if safetensors_ok():
        restore_webui_config_if_missing()
        return 0
    log("safetensors broken — attempting repair")
    stop_holders()
    if not pyd_unlocked():
        log("ERROR: cannot repair while .pyd is locked. Stop the app and retry.")
        return 2
    if repair_safetensors():
        restore_webui_config_if_missing()
        return 0
    log("ERROR: safetensors repair failed")
    return 1


def post_update() -> int:
    """After requirements reinstall: verify, repair if needed, restore config."""
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
            "usage: env_guard.py snapshot|stop_holders|preflight|post_update|verify|repair",
            file=sys.stderr,
        )
        return 2
    cmd = argv[1]
    if cmd == "snapshot":
        snapshot()
        return 0
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
