# FlashVSR+ Pinokio — Custom Configuration & Code Changes

**Last updated:** 2026-08-26  
**Install path:** `C:\pinokio\api\FlashVSR_plus_pinokio.git\app`  
**Purpose:** Persistent record of customizations applied outside stock FlashVSR (survives app rollback/reinstall). Re-apply or merge these after updating the Pinokio launcher.

---

## Changelog

| Date | Summary |
|------|---------|
| 2026-08-26 | **UI font / zoom + full path boxes:** Settings → Font & size (font, px, 80–150% zoom). Path fields are multi-line, wrap, and stay selectable so long `D:\\OUTPUTS\\…` strings are not clipped to one line. |
| 2026-08-26 | **Skip reasons split + GT newest-first:** Watch/GT/image/Toolbox scan logs list *already in queue* (still waiting — not dropped), *same-size copies*, *already-upscaled names*, and *sidecar folders* separately. Group Therapy packs unstarted files newest → oldest by source mtime; an in-progress group still finishes first. |
| 2026-08-25 | **Open Output Folder buttons:** Video → Ready for Toolbox (or last saved file). Image → Ready for CIV\\images. Toolbox → Ready for CIV. Never open `app\\outputs\\work_queue_*` JSON/TXT. Queue progress logs moved to `app\\outputs\\queue_logs`. |
| 2026-08-25 | **HDR auto per file (no tagging):** Follow source is still TV/full range. HDR (PQ/HLG/DoVi/mastering metadata) is probed on each file and **hable-tonemapped to SDR 10-bit Rec.709** for FlashVSR (SDR model). 10-bit SDR stays 10-bit. 4K-safe auto (960×540 / 540×960) remains the 4090 OOM-safe max. |
| 2026-08-25 | **Downscale quality knobs (4090 defaults):** Scale kernel **lanczos**, temp CRF **10**, color **Follow source**. Follow source copies TV/full from the file. Always full is range only. Per-file fit-scale; no whole-folder first pass. |
| 2026-08-25 | **GT RIFE/export on that tab:** Group Therapy has its own RIFE Off/2×/4× plus quality/width sliders (Toolbox sliders no longer silently ignored). Defaults: video/image quality **10**, chunk **10.25s**, tile **256**/overlap **32**, RIFE/export quality **100**. |
| 2026-08-25 | **Per-file / per-batch downscale:** watch-folder hygiene no longer recodes the whole inbox first. Resize runs as each file is processed (Group Therapy: current batch only). Separate 16:9 (960×540) and 9:16 (540×960) input sizes = ¼ of UHD 4K. Fit-scale + in_range/out_range; kernel/CRF/color from UI (default lanczos / CRF 10 / follow source). |
| 2026-08-25 | **Fit-scale downscale (no cover-crop):** pre-resize used `increase`+`crop`, which chopped edges and smeared chroma (blocky + color shift). Now FIT to the 4K-safe size with `lanczos+accurate_rnd+full_chroma_int`, CRF 12, tagged bt709 + full (`pc`) range. Toolbox NVENC export: p6/hq, lower CQ, AQ on, same color tags. |
| 2026-08-23 | **Toolbox does not pre-downscale:** Ready-for-Toolbox hygiene no longer recodes over-UHD upscales (that was FlashVSR intake work). Toolbox = RIFE + export only. Over4K originals are restored over the CRF-14 recodes. Video / image / Group Therapy skip already-upscaled files so they are not 4×'d again. |
| 2026-08-21 | **Preserve auto-reapply:** Update / Install / Start run `env_guard.py reapply`. Custom files (Group Therapy, PID pairing, toolbox FPS/no-video, 4K-safe, `webui_config`) are restored if a stock pull/clone overwrote them. Stale snapshots without current markers cannot clobber live code. |
| 2026-08-21 | **Group Therapy pairing:** no per-file `GT-<id>__name` folders. Before/After stay flat; pair id is `_PID_xxxxxxxx` at the end of the filename plus Title metadata (`PID_xxxxxxxx`). Media Center tags are not touched. Retro flatten remaps existing GT folders into the `9xxxxxxx` band so they cannot collide with new auto batches. |
| 2026-08-18 | **Group Therapy:** process originals in groups of N (upscale → RIFE 2× → RIFE 2× → export), then the next N. After each file: keep only original + final in the user Before/After pairing folders; delete resized/upscale/RIFE temps. Image-queue `4k_safe` no longer overwrites pipeline mode (imageio URI: None). |
| 2026-08-13 | **4K-safe pre-downscale:** default `batch_resize_preset=4K-safe (auto)` — math so `(input × scale)` never exceeds UHD 4K: **3840×2160** (16:9) or **2160×3840** (9:16). At 4× that is max **960×540** / **540×960** (grid-aligned). Fixed px presets still clamp inside that box. |
| 2026-08-16 | **Queue size-dedupe:** skip add + preflight drop when another queued/done file has the **same byte size** (renamed copies). Path dedupe still applies. Size 0 ignored. |
| 2026-08-14 | **User defaults:** tile **320** / overlap **32**, sparse **1.0**, local range **7**, output quality **9**. Queue order newest→oldest (mtime). |
| 2026-08-11 | **Clarity defaults:** resize ≤**1024px** (was 768 — was crushing sources), tile **256/48**, quality **9**, sparse **1.2**, chunks **10s**, pre-resize CRF **14**/slow, toolbox export quality **96**/slow. Root cause of soft output: pre-downscale + encode 7; OOM at tile 320 on tall 4× (e.g. 3072×4608). |
| 2026-08-11 | **Preserve stack:** `scripts/env_guard.py` (snapshot / stop holders / safetensors verify+repair); hardened `update.js` (stop Start, snapshot, post-update verify); `start.js` preflight; Update/Reset confirms in `pinokio.js`; offline backups under `C:\pinokio\backups\FlashVSR_plus_pinokio\`; see root `PRESERVE.md`. |
| 2026-07-31 | Toolbox **Batch Queue maxed**: 20-file packs, token-safe stem match, import crashed `batch_*` folders, requeue failed, rebuild chunks, hardlink/symlink/copy work packs, atomic manifests + CSV/PENDING/FAILED/DONE, ETA, push path → FlashVSR Batch. Live `INPUTS.txt` + `BATCH_PROGRESS` + `REMAINING.txt` on every FlashVSR batch. |
| 2026-07-31 | Pipeline stage tags on filenames: `_1` upscale, `_2` RIFE interpolation, `_3` export/posted (always last before extension). |
| 2026-07-31 | RTX 4090 profile defaults (tile 320/40, chunks 12s, resize 768px, quality 7). Machine-aware hover tooltips on all options (`TIPS` + floating tip JS). Desktop launcher `Launch-FlashVSR-Plus.bat`. |
| 2026-07-16 | After hard CUDA OOM: detect poisoned VRAM (&lt;1.5GB free), abort profile retries, skip remaining batch items, more aggressive pipeline drop (no stuck re-init cascade). |
| 2026-07-16 | OOM auto-retry (user → safe → max_save profiles), VRAM free/alloc logging, short final-chunk merge (≥21 frames), safer UI defaults (unload DiT, tile 256, batch resize 512px, chunks on). |
| 2026-07-07 | Initial custom `webui_config`, UI defaults loader, split output paths, naming convention, batch-style resize on single video, grid-aligned center-crop resize (no stretch, no upscale padding). |
| 2026-07-07 | Reverted slow VRAM path (disk tile stitch / per-tile model reload / forced config VRAM override) — kept resize-only approach. |
| 2026-07-07 | Committed customizations to git (`webui.py`, `naming_utils.py`, `webui_config`, `CUSTOMIZATIONS.md`). |

---

## Files touched (in app folder)

| File | Action |
|------|--------|
| `webui_config` | **Created** — user defaults (pipeline folders + Group Therapy) |
| `naming_utils.py` | **Created** — output filename conventions |
| `webui.py` | **Modified** — config, paths, Group Therapy, hygiene, 4K-safe, PID pairing; skip-reason logs; path_textbox + UI font/zoom |
| `group_therapy.py` | **Created** — grouped pipeline + flat `_PID_` pairing; unstarted groups packed newest→oldest |
| `flashvsr_work_queue.py` | **Modified** — size-dedupe, Group Therapy status, skip already-upscaled, AddResult skip reasons |
| `toolbox/toolbox.py` | **Modified** — FPS cap, no-video probe, HighFPS skip |
| `src/pipelines/flashvsr_tiny_long.py` | **Modified** — refuse `output_path=None` (4k_safe mode bug) |
| `../scripts/env_guard.py` | **Modified** — snapshot + **reapply after Update/Install/Start** |
| `../update.js` `../install.js` `../start.js` `../pinokio.js` | **Modified** — call reapply / preflight |

---

## `webui_config` (current values)

```ini
# Clarity profile — see Changelog 2026-08-11
batch_resize_preset=4K-safe (auto)
tile_size=256
tile_overlap=48
quality=9
sparse_ratio=1.2
chunk_duration=10
enable_chunks=True
tiled_dit=True
tiled_vae=True
unload_dit=True
scale=4
mode=tiny
tb_frames_quality=98
tb_export_quality=96
tb_export_preset=slow
```

**RTX 4090 clarity notes:** Prefer **1024px input + tile 256** over **768px + tile 320**. Soft “lost quality” was from pre-downscale discarding pixels the model never sees. If OOM: drop resize to 768px (keep tile 256) or Restart FlashVSR after a hard OOM (VRAM can stick at 0 free).

**Note:** Production files no longer land in `app\outputs`. That folder is only for queue STATUS / temp logs.
Step 3 (upscaled videos) = **Ready for Toolbox**. Configure all steps under ⚙️ Settings → Pipeline folders.

---

## Behaviour summary

### 1. UI defaults from config (`get_ui_defaults`)

Stock FlashVSR only read theme/autosave/output from `webui_config`. Added `get_ui_defaults()` so processing sliders load from config on startup:

- Upscale 4×, tiny, v1.1, chunks 10s, tiled DiT/VAE, unload DiT, tile 256/48, sage, bf16, cuda:0, quality 9, batch resize 1024px, etc.

**Code:** `load_config()` + `_parse_config_value()`, `get_ui_defaults()`, `create_ui()` uses `ui[...]` for control `value=`.

### 2. Pipeline folders (readable steps)

| Step | What | Default path |
|------|------|----------------|
| 1 | Intake / watch | `D:\OUTPUTS\__X_GROK\NEW DOWNLOADS` |
| 2 | Originals archive | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Pre Scaled videos` |
| 3 | After upscale (videos) | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Ready for Toolbox` |
| 4 | After upscale (images) | `...\Ready for CIV\images` |
| 5 | Toolbox inbox | same as step 3 |
| 6 | Final / Ready for CIV | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Post Scaling\Ready for CIV` |

**Name tags:** `_1` upscale · `_2` interpolate · `_3` export  
**Code:** `WORKFLOW_DEFAULTS`, `get_workflow_paths()`, `workflow_paths_html()`, Settings → **Save all pipeline folders**.

### 3. Output naming (`naming_mode=both`)

**File:** `naming_utils.py`

Keeps original stem + upscale metadata. Examples at 4×:

| Mode | Example |
|------|---------|
| `both` | `UpScale4K_my_clip_upscaled_x4_S_I_1.mp4` |
| `toolbox` | `my_clip_upscaled_x4_1.mp4` |
| `vsr` | `UpScale4K_my_clip_S_I_1.mp4` |

Chunked outputs append `_chunked` before the stage. Comparisons: `{stem}_comparison.mp4` (no stage).

#### Pipeline stage tags (at end of name)

| Tag | Meaning | Applied when |
|-----|---------|--------------|
| `_1` | Upscaled | FlashVSR video/image save |
| `_2` | Interpolated | Toolbox Frame Adjust with RIFE 2x/4x |
| `_3` | Exported / ready to post | Toolbox Export |

Stage is always the last token before the extension so you can sort/filter at a glance. Higher stages never demote (e.g. exporting a `_2` becomes `_3`, not `_2_3`).

Preprocessing suffixes (`_resized_`, `_trim_`) stay in the stem; timestamps stripped by `clean_video_filename()`.

**Settings UI:** Output Naming dropdown + Apply button.

### 4. Batch resize on Single Video tab

`apply_batch_resize_preset()` runs before single-video processing (chunk or normal), same as batch tab — uses `batch_resize_preset` from config (512px).

### 5. Grid-aligned resize (center crop, no distortion)

FlashVSR needs **output** dimensions (after upscale) on a **128px** grid.

| Upscale | Input align step |
|---------|------------------|
| 4× | 32px |
| 2× | 64px |

Resize flow (video + image):

1. Compute target box (e.g. `512×608` for 4×).
2. **Scale** proportionally to cover the box.
3. **Center-crop** to exact size — aspect preserved, no stretch, no black-bar padding during upscale.

**Video:** FFmpeg  
`scale=W:H:force_original_aspect_ratio=increase:flags=lanczos,crop=W:H`

**Image:** `center_crop_cover_pil()` in `webui.py`

**Helpers:** `input_align_step()`, `calculate_resize_dimensions(..., scale=)`, `resize_input_video(..., scale=)`.

---

## Intended workflow

1. Drop sources in **Step 1** (NEW DOWNLOADS).
2. **Group Therapy** (recommended for mixed batches): pick group size + Before/After folders, Start. Each group of N runs **upscale → RIFE 2× → RIFE 2× → export**, then the next N. Both sides stay **flat** (no per-file folders). Pairing is `_PID_xxxxxxxx` at the end of the filename plus Title=`PID_xxxxxxxx` (not Media Center tags). Only original + final are kept.
3. Or classic split: Video queue upscales → **Step 3 Ready for Toolbox**; originals → Step 2. Toolbox reads Step 5/3 → export → **Step 6 Ready for CIV**.
4. Images → **Step 4** Ready for CIV\images (skip RIFE).

---

## Reverted / not used (do not re-add without need)

These were tried for RTX 4090 OOMs but **removed** at user request (too slow):

- `vram_settings_from_config()` — forced VRAM toggles from config over Gradio state
- Disk tile stitch for `N > 100` frames
- Per-tile `init_pipeline()` + `release_pipeline()` reload
- Batch auto-chunk from config inside `run_flashvsr_batch()`

If OOM returns:
1. **Restart FlashVSR** in Pinokio (failed runs often leave ~all 24GB reserved until process restart).
2. Confirm UI: **Tiled DiT + Tiled VAE + Unload DiT**, tile **128–256**, batch resize **512px**, chunks on.
   - Do **not** run with tile 320 + Tiled VAE off + Unload DiT off on 24GB (portrait 512×736 4× will OOM at VAE).
3. Code auto-retries OOM with progressive safer profiles (`user` → `safe` → `max_save`), aborts retries/batch when VRAM stays poisoned, and merges final chunks shorter than 21 frames into the previous segment.
4. Portrait/tall clips at 4× are hungrier than landscape 512×288 — prefer **tile 128** if 256 still OOMs after a clean restart.

---

## After stock rollback / Pinokio update

**Automatic.** Pinokio **Update**, **Install**, and **Start** all run `scripts/env_guard.py reapply` / `preflight`. If stock `git pull` / `git clone` overwrote custom files, they are copied back from `local-preserve/latest`, then offline backups, then `git show mine/main`.

If something still looks stock after Start:

```powershell
cd C:\pinokio\api\FlashVSR_plus_pinokio.git\app
.\env\Scripts\Activate.ps1
python ..\scripts\env_guard.py snapshot
python ..\scripts\env_guard.py reapply
```

Key symbols that must exist (reapply uses these as markers):

- `run_group_therapy` / `with_pid_name` in `webui.py`
- `stamp_title_pid` / `flatten_gt_pair_folders` in `group_therapy.py`
- `_choose_interp_factor` in `toolbox.py`
- `gt_before_dir=` in `webui_config`

---

## Manual UI checklist (if config missing)

After restart, verify in UI if anything did not load:

- Upscale **4×**, chunks **on** @ **10.25s**
- Tiled DiT/VAE **on**, unload DiT **on**, tile **256** / overlap **32**
- Advanced: sage, bf16, **cuda:0**, quality **8**
- Batch tab resize preset **512px** (also applies to Single Video via config)

---

## Related paths (reference)

| Item | Path |
|------|------|
| This doc (Pinokio root, persistent) | `C:\pinokio\FlashVSR_plus_pinokio_CUSTOMIZATIONS.md` |
| This doc (in git / app) | `C:\pinokio\api\FlashVSR_plus_pinokio.git\app\CUSTOMIZATIONS.md` |
| App | `C:\pinokio\api\FlashVSR_plus_pinokio.git\app` |
| Earlier fork (heavy custom) | `C:\pinokio\api\VIDUpscaler\app` — not in use |

---

*Update this file whenever customizations change.*