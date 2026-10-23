# RESTORE: rebuild this FlashVSR fork from scratch

This file is written so a person or an AI agent can rebuild Ryan's FlashVSR install step by step on a new Windows machine, or recover it after a bad Update or Reset. Plain-language background is in [UPGRADES.md](UPGRADES.md). Day-to-day update and reset safety is in [PRESERVE.md](PRESERVE.md). The per-feature changelog is in [app/CUSTOMIZATIONS.md](app/CUSTOMIZATIONS.md).

Rule for this file: every machine-specific drive or folder appears **only** in the [Settings](#settings) table below. Steps refer to setting names, never to literal paths.

## 1. Header

```yaml
repo: Treyu2023/FlashVSR_plus_pinokio
branch: main
remotes:
  mine: https://github.com/Treyu2023/FlashVSR_plus_pinokio.git    # the fork; Update/Install pull from here
  origin: https://github.com/ai-anchorite/FlashVSR_plus_pinokio.git  # stock launcher; read-only, never pulled
upstream_app: ai-anchorite/FlashVSR_plus (stock app code; no longer cloned by install.js)
layout: monorepo            # app/ is tracked inside the launcher repo; there is no app/.git
platform: Windows + NVIDIA  # CUDA 12.8 wheels installed by torch.js; tuned on one 24 GB RTX 4090, 64 GB RAM
python_venv: app/env        # created by Install; never committed
install_dir: <pinokio_root>/api/FlashVSR_plus_pinokio.git   # see Settings: pinokio_root
web_ui_port: 7860           # Gradio, local address only
launcher_entry_points: [install.js, start.js, update.js, reset.js, pinokio.js, torch.js, link.js]
guard_script: scripts/env_guard.py   # snapshot / reapply / preflight / post_update / verify / repair
```

## Settings

All machine-specific values live here. "Ryan's value" is an example from the original machine; on a new machine pick your own folders and put them in the place named in the "Where" column.

| Setting | Where | Meaning | Ryan's value (example) |
|---|---|---|---|
| `pinokio_root` | Pinokio app settings | Folder Pinokio is installed into; the app lives at `<pinokio_root>\api\FlashVSR_plus_pinokio.git` | `C:\pinokio` |
| `batch_watch_folder` | `app/webui_config` | Step 1 intake / watch folder for new downloads | `O:\New Downloads ODrive Media` |
| `batch_source_archive_dir` | `app/webui_config` | Step 2 originals archive ("Pre Scaled") | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos` |
| `batch_upscale_handoff_dir` | `app/webui_config` | Step 3 upscaled videos handed to Toolbox | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox` |
| `img_upscale_handoff_dir` | `app/webui_config` | Step 4 upscaled images | `D:\OUTPUTS\__X_GROK\Images\Upscaled images` |
| `tb_inbox_folder` | `app/webui_config` | Step 5 Toolbox inbox (normally same as step 3) | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox` |
| `toolbox_output_dir` | `app/webui_config` | Step 6 final "Ready for CIV" output | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV` |
| `output_dir` | `app/webui_config` | Where single/batch upscales are saved | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox` |
| `gt_before_dir` | `app/webui_config` | Group Therapy "Before" folder (originals, flat, `_PID_` names) | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos` |
| `gt_after_dir` | `app/webui_config` | Group Therapy "After" folder (finals, flat, `_PID_` names) | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV` |
| `gt_group_size` | `app/webui_config` | Files per Group Therapy batch | `5` |
| `chunk_duration` / `enable_chunks` | `app/webui_config` | Chunk long videos into pieces of N seconds | `10.25` / `True` |
| `tile_size` / `tile_overlap` | `app/webui_config` | DiT/VAE tile size and overlap in px | `320` / `32` |
| `tiled_dit` / `tiled_vae` / `unload_dit` | `app/webui_config` | Memory switches for 24 GB GPUs | `True` / `True` / `True` |
| `quality` / `sparse_ratio` / `local_range` / `kv_ratio` | `app/webui_config` | Model quality knobs | `10` / `1.0` / `7` / `3` |
| `scale` / `mode` / `model_version` / `dtype` / `attention_mode` / `device` | `app/webui_config` | Upscale factor and pipeline | `4` / `tiny` / `v1.1` / `bf16` / `sage` / `cuda:0` |
| `color_fix` / `fps_override` / `randomize_seed` | `app/webui_config` | Output colour fix, fps, seed | `True` / `30` / `True` |
| `batch_resize_preset` | `app/webui_config` | Pre-downscale so input x scale never exceeds UHD 4K | `4K-safe (auto)` |
| `resize_16x9_w` / `_h`, `resize_9x16_w` / `_h` | `app/webui_config` | 4K-safe input sizes per orientation | `960x540` / `540x960` |
| `resize_kernel` / `resize_crf` / `resize_color` | `app/webui_config` | Pre-downscale encoder settings | `lanczos` / `10` / `source` |
| `tb_fps_mode` / `tb_pipeline_ops` / `tb_frames_quality` / `tb_export_quality` / `tb_export_max_width` / `tb_export_preset` / `tb_prefer_nvenc` | `app/webui_config` | Toolbox RIFE and export | `4x Frames` / `Frame Adjust,Export` / `100` / `100` / `3840` / `medium` / `True` |
| `gt_rife_mode` / `gt_rife_quality` / `gt_export_quality` / `gt_export_max_width` / `gt_rife_streaming` | `app/webui_config` | Group Therapy RIFE and export | `4x Frames` / `100` / `100` / `3840` / `False` |
| `gt_do_upscale` / `gt_do_rife1` / `gt_do_rife2` / `gt_do_export` | `app/webui_config` | Which Group Therapy stages run | all `True` |
| `gpu_multitask` / `gpu_cap_pct` / `gpu_full_power_w` | `app/webui_config` | Retired GPU cap; kept so old configs load; always Normal scheduling | `False` / `100` / `463.5` |
| `naming_mode` / `theme` / `custom_theme` / `ui_font` / `ui_font_size` / `ui_scale` | `app/webui_config` | Naming and UI look | `both` / `Interstellar` / empty / `system-ui` / `22` / `90` |
| `clear_temp_on_start` / `autosave` / `tb_autosave` | `app/webui_config` | Housekeeping | `True` / `True` / `True` |
| Gradio allowed paths (literal entries) | `app/webui.py`, `get_gradio_allowed_paths()` | Extra folders Gradio may serve previews from, on top of the folder settings above. Two are hard-coded today (ruling R-5); a later slice may move them into `webui_config`. | `D:\OUTPUTS\__X_GROK` and `O:\New Downloads ODrive Media` |
| `OFFLINE_BACKUPS` | `scripts/env_guard.py` | Offline snapshot folder used by `snapshot` / `reapply` | `C:\pinokio\backups\FlashVSR_plus_pinokio` |
| `-Source` / `-Dest` defaults | `scripts/backup-to-vidupscaler.ps1` | Retired mirror script defaults | `C:\pinokio\api\FlashVSR_plus_pinokio.git` / `C:\pinokio\api\VIDUpscaler` |
| Restage target | `scripts/quarantine_degraded_civ.py`, `scripts/restore_over4k_originals.py` | One-off repair scripts copy originals back to the watch folder | `O:\New Downloads ODrive Media` |
| PinokioDock profiles | outside git | Pinokio Dock field profiles; re-apply after reinstall | `%LOCALAPPDATA%\FAFO\PinokioDock\profiles\` |

`app/webui_config` (66 keys) and `app/webui_config.bak` stay **tracked** by design (ruling R-5): `env_guard.py` uses markers inside `webui_config` to tell a custom config from a stock one, and restores `webui_config.bak` only if it is missing.

## 2. Rebuild steps

a. Install Pinokio and Git for Windows. Have an NVIDIA driver that supports CUDA 12.8.

b. Get the code into `<pinokio_root>\api\FlashVSR_plus_pinokio.git`, either through Pinokio Discover with the fork URL (`https://github.com/Treyu2023/FlashVSR_plus_pinokio`) or with `git clone https://github.com/Treyu2023/FlashVSR_plus_pinokio.git FlashVSR_plus_pinokio.git` inside `<pinokio_root>\api`.

c. Remotes. In the repo folder:
   ```
   git remote -v
   git remote get-url mine || git remote add mine https://github.com/Treyu2023/FlashVSR_plus_pinokio.git
   git remote get-url origin || git remote add origin https://github.com/ai-anchorite/FlashVSR_plus_pinokio.git   # optional, read-only
   git fetch mine
   git branch -u mine/main main
   ```
   If you cloned from the fork, `origin` is the fork; adding `mine` as above is still required because Update and Install name `mine` explicitly.

d. Click **Install** in Pinokio. It: restores `app/` from `HEAD` if `app/webui.py` is missing; adds `mine` if missing and runs `git pull --ff-only mine main`; runs `torch.js` (CUDA wheels into `app/env`); `uv pip install -r requirements.txt`; then `env_guard.py reapply` and `env_guard.py post_update`.

e. Edit `app/webui_config` for this machine: set every folder setting from the Settings table to folders that exist here. If you use folders outside those settings, also update the Gradio allowed-paths entries.

f. Click **Start**. `start.js` runs `env_guard.py preflight` (repairs safetensors and reapplies any overwritten custom file), then `webui.py`. Expect the log line `Gradio allowed_paths: N location(s)` and a local URL on port 7860.

g. First job downloads the FlashVSR model weights into `app/models/`. The RIFE model goes into `app/toolbox/model_rife/` on first Toolbox / Group Therapy use.

h. Restore the items in section 4 (not in git).

## 3. Customized-file inventory

`git diff origin/main..main` covers 66 files. `origin/main` is the stock ai-anchorite launcher, which does not track `app/` at all, so every `app/` file shows as **added**. Kinds used below:

- **Launcher, modified**: stock launcher file with fork edits.
- **New**: written for this fork.
- **App, customized**: FlashVSR_plus app file with fork edits (env_guard `CRITICAL`, usually with markers).
- **App, vendored**: stock FlashVSR_plus app code tracked unchanged so Install can rebuild `app/` from git after Reset.

"Markers" are strings `scripts/env_guard.py` requires in the live file; a missing marker means a stock file overwrote the custom one and `reapply` restores it. The full, authoritative list is the `MARKERS` dict in `env_guard.py`.

### 3.1 Launcher (repo root)

| Path | Kind | What it changes | Markers / guard |
|---|---|---|---|
| `.gitignore` | New | Ignores `app/env/`, `app/models/`, `app/outputs/` (queues and work media), `app/_temp/`, `app/inputs/`, caches, `local-preserve/`, `cache/`, `logs/`, and `app/webui_config.bak-*` (dated config backups; see section 5). | - |
| `PRESERVE.md` | New | Update/Reset safety guide: remotes, what Update/Start do, backup tiers, manual `env_guard` commands, recovery recipes. | CRITICAL |
| `README.md` | Launcher, modified | Adds the "This fork" section with links to these docs; stock text below untouched. | - |
| `RESTORE.md`, `UPGRADES.md` | New | This file and the plain-language upgrade list. | - |
| `install.js` | Launcher, modified | Replaces `git clone` of stock `ai-anchorite/FlashVSR_plus` into `app/` with: (1) only when `app/webui.py` is missing, `git checkout HEAD -- app`; (2) `git remote get-url mine \|\| git remote add mine <fork URL>` then `git pull --ff-only mine main`. Then stock `torch.js`, `uv pip install -r requirements.txt`, and `env_guard.py reapply` + `post_update`. | `env_guard.py reapply`; CRITICAL |
| `update.js` | Launcher, modified | `script.stop start.js` (frees Windows `.pyd` locks); `env_guard.py snapshot` + `stop_holders`; `git remote get-url mine \|\| git remote add mine <fork URL>` then `git pull --ff-only mine main` (was a bare `git pull`); stock nested `app/.git` pull kept but only runs if `app/.git` exists (ruling R-3; never on the monorepo); `env_guard.py reapply`, `uv pip install -r requirements.txt`, `env_guard.py post_update`; notify. | `env_guard.py reapply`, `env_guard.py snapshot`; CRITICAL |
| `start.js` | Launcher, modified | Two steps, both `conda: { skip: true }` (avoids the OpenCL `temp.txt` lock when Pinokio auto-launches) with `PYTHONUNBUFFERED=1` and `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`: (1) venv python `-u ..\scripts\env_guard.py preflight`; (2) venv python `-u webui.py`, waiting for the `http://` URL event. | `env_guard.py preflight`, `PYTHONUNBUFFERED`; CRITICAL |
| `pinokio.js` | Launcher, modified | Update gets a confirm explaining stop/snapshot/pull/reapply; Reset label and confirm warn that it deletes all of `app/` and point to the offline backups setting. | `AUTOMATICALLY reapply`; CRITICAL |
| `pinokio.json` | New | Pinokio title and description, empty plugin menu. | - |
| `reset.js`, `torch.js`, `link.js`, `pinokio_meta.json` | Stock, unchanged | `reset.js` still runs `fs.rm app`. Listed because env_guard snapshots them. | CRITICAL |
| `scripts/env_guard.py` | New | Guard CLI: `snapshot` (copy CRITICAL files to `local-preserve/latest` and `OFFLINE_BACKUPS\<stamp>`), `reapply` (restore any CRITICAL file whose markers are missing, from the first source that has them: `local-preserve/latest`, newest offline backup, `git show mine/main`, `HEAD`), `stop_holders` (kill FlashVSR python holding `_safetensors_rust.pyd`), `verify` / `repair` (pin `safetensors~=0.6.0`), `preflight` (Start), `post_update` (Update/Install). Never applies a stale snapshot that lacks current markers. | `def reapply`, `source_is_good`; CRITICAL |
| `scripts/backup-to-vidupscaler.ps1` | New (retired) | Mirrors code to the old VIDUpscaler copy; retired, git `mine` is the backup. | CRITICAL |
| `scripts/quarantine_degraded_civ.py` | New (one-off) | Moves non-96fps stage-3 outputs to quarantine and copies matching originals back to the watch folder for restaging. | - |
| `scripts/restore_over4k_originals.py` | New (one-off) | Moves CRF-12 inbox recodes aside and restores Over4K originals into the watch folder; never deletes originals. | - |

### 3.2 App: customized files

| Path | What it changes | Markers (sample) |
|---|---|---|
| `app/webui.py` | The Gradio app. Adds: `load_config()` / `get_ui_defaults()` so every control loads its value from `webui_config`; six pipeline-folder settings (`WORKFLOW_DEFAULTS`, Settings > Save all pipeline folders); readable two-step naming and stage tags; Group Therapy tab (`run_group_therapy`) with its own RIFE/export controls; filename PID pairing (`with_pid_name`); watch-folder hygiene (queue preflight, dedupe, skip reasons in `_log_queue_scan`); Grok-ID scan buttons (`handle_scan_grok_outputs`); Chrome "(N)" copy handling (`chrome_copy_number`); per-file HDR probe and `tonemap=hable` to SDR 10-bit Rec.709; fit-scale 4K-safe pre-downscale (`resolve_resize_encode`, `orientation_input_box`, `accurate_rnd+full_chroma_int`, "no size recode"); Open-folder buttons go to media (`open_media_folder`); multi-line path boxes and UI font/zoom (`def path_textbox`); terminal heartbeats (`HeartbeatTqdm`); Stop After Current armed at once (`stop_finish_path`, `tick_stop_banner`); float16 stitch canvases (`alloc_stitch_canvases`); GPU cap retired (`apply_saved_gpu_headroom`, `gpu_multitask_g`, "GPU scheduling retired"); TF32 + cuDNN benchmark (`_enable_fast_gpu_math`); `randomize_seed` default on; OOM auto-retry profiles (user > safe > max_save) with poisoned-VRAM abort; grid-aligned center-crop resize; temp wipe that skips locked files; `get_gradio_allowed_paths()` with the two literal entries in Settings. | `def run_group_therapy`, `with_pid_name`, `gt_before_dir`, `tonemap=hable`, `alloc_stitch_canvases`, `"randomize_seed": (True` |
| `app/group_therapy.py` | New module. Processes originals in groups of N, stage by stage (upscale > RIFE 2x > RIFE 2x > export), then the next N; unstarted groups packed newest first; after each file keeps only original (Before) and final (After), flat, both ending `_PID_xxxxxxxx` plus Title metadata `PID_xxxxxxxx`; deletes temps; reclaims already-delivered watch-folder files; one After-folder index per run (`get_after_lookup`); pair index in `_pairs\pairs.json`; skips/moves finished exports. | `stamp_title_pid`, `flatten_gt_pair_folders`, `get_after_lookup`, `reclaim_watch_folder`, `_pairs` |
| `app/flashvsr_work_queue.py` | New module. Persistent video/image/toolbox queues (`queue.json` under `app/outputs/work_queue_*`): add anytime, resume pending, soft stop, X of Y progress; path and same-byte-size dedupe; Grok-ID dupes; drop wrong-stage pending; skip already-upscaled/exported; seed remaining pipeline stages. | `class AddResult`, `gt_pair_id`, `grok_id_dupes`, `seed_pipeline_fields` |
| `app/grok_id_index.py` | New module. Extracts `grok-video-UUID` / `GROK-##` / leading post IDs from names, catalogs original sizes in `app/outputs/grok_id_index.json`, flags new downloads whose ID matches and size is within 2.5 percent. | `extract_grok_ids`, `SIZE_TOLERANCE`, `screen_folder` |
| `app/naming_utils.py` | New module. Two-step names `<original>_<4K>_<9x16>_Upscaled` then `_<30fps>`; strips legacy `_1/_2/_3` tags. | `clean_original_stem`, `step1_filename` |
| `app/gpu_headroom.py` | New module. Forces Normal GPU/CPU scheduling so old configs cannot throttle jobs (Idle / Below-Normal / nvidia-smi watt cap retired). | `apply_gpu_headroom`, `always Normal` |
| `app/stitch_canvas.py` | New module. Disk-backed fp16 stitch canvas (`DiskStitchCanvas`, `torch.from_file`) mapped only while blending/saving so DiT does not page under WDDM. | `class DiskStitchCanvas`, `release_host_cache` |
| `app/src/busy_heartbeat.py` | New module. Newline status every ~5 s plus a 12 s "still busy" watchdog and red `STOP ARMED` banner in Pinokio's terminal. | `class HeartbeatTqdm`, `STOP ARMED` |
| `app/src/pipelines/flashvsr_tiny.py` | Parks stream KV and LQ on CPU during DiT and copies one block back at a time; TCDecoder loaded only for VAE; heartbeat spans; colour fix in 16-frame slices. | `drop_dit_stream_kv`, `park stream KV` |
| `app/src/pipelines/flashvsr_tiny_long.py` | Refuses `output_path=None` (4k_safe mode bug); `clean_vram` on offload. | `tiny-long pipeline requires output_path` |
| `app/src/pipelines/flashvsr_full.py` | Full pipeline: keeps DiT on the GPU between tiles, drops DiT KV before VAE decode (decode-stall fix), heartbeat spans. | CRITICAL |
| `app/src/models/wan_video_dit.py` | Stream KV clone drops prefix storage so the disk stitch canvas and later tiles do not compete for memory. | `stream KV clone drops prefix storage` |
| `app/src/models/wan_video_vae.py` | Heartbeat progress during VAE encode/decode so long steps show in the terminal. | CRITICAL |
| `app/src/models/TCDecoder.py` | Sequential decode sends finished frames to CPU (no 249-frame GPU stack); heartbeat spans. | CRITICAL |
| `app/toolbox/toolbox.py` | FPS cap, no-video-stream probe, HighFPS skip, medium export preset, NVENC p6/hq with source colour tags. | `_choose_interp_factor`, `_has_video_stream` |
| `app/toolbox/batch_queue.py` | BATCH_PROGRESS/INPUTS writers only (live BATCH_PROGRESS.json/.txt, REMAINING.txt, INPUTS.txt); the old batch queue UI was removed in FV-UI. | CRITICAL |
| `app/toolbox/rife_core.py` | RIFE model loading for Toolbox/Group Therapy; model in `app/toolbox/model_rife/`. | CRITICAL |
| `app/webui_config` | 66-key machine config; see Settings. Tracked (R-5). | `gt_before_dir=`, `batch_watch_folder=`, `randomize_seed=True`, `chunk_duration=10.25` |
| `app/webui_config.bak` | Minimal 5-key fallback config; restored only if missing. | CONFIG_FILES |
| `app/requirements.txt` | Pins (gradio 5.x below 6, numpy 2.x, `safetensors~=0.6.0`, imageio, etc.). `requirements-freeze-before-upgrade.txt` / `-after-upgrade.txt` record the env around an upgrade. | CRITICAL |
| `app/Launch-FlashVSR-Plus.bat` / `.ps1` | Desktop launcher: starts venv `webui.py` on port 7860 (or opens the browser if it is already up). | CRITICAL |
| `app/CUSTOMIZATIONS.md` | Dated changelog of every customization; keep it updated. | CRITICAL |
| `app/.gitignore` | Ignores `env/`, `/models/`, `outputs/`, `_temp/`, `inputs/`, `webui.instance.lock`, `toolbox/model_rife/`, weight files. | - |
| `app/test_grok_id_dupes.py`, `app/test_pipeline_stage_skip.py` | Tests for Chrome-copy screening and stage-skip / finished-export handling. | - |
| `app/run.py` | Command-line entry point with the fork defaults (tile 320/32, float16 stitch, DiT kept on GPU). | - |
| `app/toolbox/system_monitor.py` | GPU/CPU/RAM monitor panel used by the UI. | - |

### 3.3 App: vendored, unchanged

`app/LICENSE`, `app/README.md`, `app/teaser.jpg`, `app/src/__init__.py`, `app/src/configs/*`, `app/src/models/__init__.py`, `app/src/models/model_manager.py`, `app/src/models/utils.py`, `app/src/models/sparse_sage/*`, `app/src/pipelines/__init__.py`, `app/src/pipelines/base.py`, `app/src/schedulers/*`, `app/src/vram_management/*`, `app/toolbox/RIFE/*`. Tracked so that Install can rebuild `app/` from git after Reset.

Source of truth for this inventory: env_guard `CRITICAL` + `MARKERS`, the "Files touched" table in `app/CUSTOMIZATIONS.md`, and `git log origin/main..main --name-only`.

## 4. Not in git (restore separately)

- `.git/info/exclude` entries (agent rule files such as `AGENTS.md`, `CLAUDE.md`, `.clinerules`, and `ENVIRONMENT`) and the Pinokio `ENVIRONMENT` file. These are local only and not carried by a clone.
- `local-preserve/` (env_guard snapshots) and the `OFFLINE_BACKUPS` folder (see Settings).
- PinokioDock profiles (see Settings).
- `app/env` (venv), `app/models` (weights), `app/toolbox/model_rife`, `app/outputs` (queues, logs, Grok-ID index), `app/inputs`.

## 5. .gitignore line `app/webui_config.bak-*`

Added in FV-1 (ruling R-4). Dated config backups such as `app/webui_config.bak-before-defaults-20260814_094631` are machine-local copies of an old `webui_config` and must never be committed. The pattern needs the dash, so the tracked fallback `app/webui_config.bak` is not affected.

## 6. Restore scenarios

- **Bad Update** (custom feature missing, or safetensors import error): stop the app, then from `app` with the venv active run `python ..\scripts\env_guard.py reapply`; for safetensors run `python ..\scripts\env_guard.py repair`. Start runs `preflight` and does both automatically.
- **After Reset**: Reset deletes all of `app/`, including `app/env`, `app/models`, `app/outputs` (queues and logs are gone) and `app/toolbox/model_rife`. Click **Install**: it restores `app/` from `HEAD`, pulls `mine` fast-forward only, rebuilds the venv and reapplies. Then re-check `app/webui_config` against Settings.
- **Update says "Not possible to fast-forward"**: local `main` and `mine/main` have diverged. Update then carries on with pip and reapply on the old code, which is safe but easy to miss. Stop, inspect `git log --oneline main...mine/main`, and decide with the owner. Never `reset --hard` without approval.
- **Machine loss**: clone the fork and follow sections 2 to 4.

## 7. Verification checklist

1. `python ..\scripts\env_guard.py verify` passes.
2. For every file in env_guard `MARKERS`, every marker string is present (`reapply` reports nothing to restore).
3. `git status` is clean apart from known untracked files.
4. Port 7860 is listening and the log shows `Gradio allowed_paths: N location(s)`.
5. UI spot check: 4x, chunks on at 10.25 s, tiled DiT/VAE on, unload DiT on, tile 320/32, quality 10, sparse 1.0, batch resize `4K-safe (auto)`, export `medium`.
