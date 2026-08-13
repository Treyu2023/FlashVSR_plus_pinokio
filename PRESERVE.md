# FlashVSR+ — Preserve / Recovery Guide

How this install is protected against Pinokio **Update**, dirty **pip** upgrades, and accidental **Reset**.

## Remotes

| Remote | URL | Role |
|--------|-----|------|
| `mine` | `https://github.com/Treyu2023/FlashVSR_plus_pinokio.git` | **Your backup** — `main` tracks this |
| `origin` | `https://github.com/ai-anchorite/FlashVSR_plus_pinokio.git` | Upstream launcher (do not force-pull over customizations) |

```powershell
cd C:\pinokio\api\FlashVSR_plus_pinokio.git
git remote -v
git status -sb
# push recovery branch
git push mine main
```

## What Update does now (`update.js`)

1. **Stops** Pinokio `start.js` (releases Windows locks on `.pyd` files)
2. **Snapshots** custom files via `scripts/env_guard.py snapshot`
3. **Kills** any leftover FlashVSR `python.exe` holders
4. `git pull` (launcher monorepo; nested `app/.git` only if present)
5. `uv pip install -r requirements.txt`
6. **Verifies** `from safetensors import safe_open` and **repairs** if broken

## What Start does now (`start.js`)

- Runs `env_guard.py preflight` before `webui.py`
- If safetensors is a broken namespace package, auto-repairs `safetensors~=0.6.0`

## Backup locations

| Location | Survives |
|----------|----------|
| **`C:\pinokio\api\VIDUpscaler`** | **Code milestone mirror** of this install (same product family; FlashVSR is newer/live) |
| `local-preserve/latest/` (in repo, gitignored) | Reset deletes `app/` only — this folder stays |
| `C:\pinokio\backups\FlashVSR_plus_pinokio\<timestamp>\` | Full project delete / reinstall |
| Git remote `mine` | Machine loss (after you push) |

### VIDUpscaler = working milestone backup (not a second live app)

**FlashVSR+ is the only install you should edit and run day-to-day.**  
VIDUpscaler is the same product line kept as a **known-good source snapshot** so you can recover code after a bad update.

```powershell
cd C:\pinokio\api\FlashVSR_plus_pinokio.git
# copy live code → VIDUpscaler (no env/models/outputs)
powershell -ExecutionPolicy Bypass -File .\scripts\backup-to-vidupscaler.ps1 -Commit
```

Restores code only; re-use FlashVSR `app\env` (or reinstall) after recovery.

Critical files in each snapshot: `webui.py`, `flashvsr_work_queue.py`, toolbox modules, `webui_config`, launcher `*.js`, `requirements.txt`, etc.

## Manual commands

```powershell
cd C:\pinokio\api\FlashVSR_plus_pinokio.git\app
.\env\Scripts\Activate.ps1

# snapshot only
python ..\scripts\env_guard.py snapshot

# verify import
python ..\scripts\env_guard.py verify

# force repair safetensors (app must be stopped)
python ..\scripts\env_guard.py repair

# start preflight (same as Start tab)
python ..\scripts\env_guard.py preflight
```

## Do / Don't

**Do**

- Stop the app (Pinokio Stop **or** desktop launcher) before Update
- Commit / push to `mine` after meaningful customization work
- Keep Pinokio Dock field profiles under `%LOCALAPPDATA%\FAFO\PinokioDock\profiles\`

**Don't**

- Run Pinokio **Update** while `webui.py` is running on Windows (locks `_safetensors_rust.pyd`)
- Click **Reset** unless you intend to wipe `app/` (env + code)
- `git pull origin main` if that would discard your customization commits — prefer `mine`

## Recover after bad Update / Reset

### A. Broken safetensors only

```powershell
cd C:\pinokio\api\FlashVSR_plus_pinokio.git\app
.\env\Scripts\Activate.ps1
python ..\scripts\env_guard.py repair
python -c "from safetensors import safe_open; import safetensors; print('OK', safetensors.__file__)"
```

### B. Lost custom Python / config

```powershell
$src = "C:\pinokio\backups\FlashVSR_plus_pinokio"   # pick newest stamp folder
# or: local-preserve\latest
Copy-Item -Force "$src\app\webui.py" "C:\pinokio\api\FlashVSR_plus_pinokio.git\app\webui.py"
# repeat for flashvsr_work_queue.py, toolbox\*.py, webui_config, naming_utils.py, ...
```

### C. Full tree from git

```powershell
cd C:\pinokio\api\FlashVSR_plus_pinokio.git
git fetch mine
git checkout main
git reset --hard mine/main   # only if you accept discarding uncommitted local edits
```

## Related

- `app/CUSTOMIZATIONS.md` — feature changelog for this fork
- Pinokio Dock profiles — separate from git; re-Apply after reinstall
