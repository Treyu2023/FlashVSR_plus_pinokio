# FlashVSR+ Pinokio — Custom Configuration & Code Changes

**Last updated:** 2026-07-07  
**Install path:** `C:\pinokio\api\FlashVSR_plus_pinokio.git\app`  
**Purpose:** Persistent record of customizations applied outside stock FlashVSR (survives app rollback/reinstall). Re-apply or merge these after updating the Pinokio launcher.

---

## Changelog

| Date | Summary |
|------|---------|
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
| `webui_config` | **Created** — user defaults (see below) |
| `naming_utils.py` | **Created** — output filename conventions |
| `webui.py` | **Modified** — config loading, paths, naming, resize, settings UI |

**Not modified:** `start.js`, pipeline/model code (`src/`), `toolbox/toolbox.py` (except reading toolbox path from config via `webui.py`).

---

## `webui_config` (current values)

```ini
clear_temp_on_start=True
autosave=True
tb_autosave=True
theme=Interstellar
toolbox_output_dir=D:\OUTPUTS\__X_GROK\Upscaled Videos\Current\Ready for CIV
chunk_duration=12
enable_chunks=True
tiled_dit=True
tiled_vae=True
unload_dit=True
tile_size=320
tile_overlap=40
attention_mode=sage
sparse_ratio=1.5
local_range=11
kv_ratio=3
quality=7
randomize_seed=False
scale=4
mode=tiny
model_version=v1.1
dtype=bf16
color_fix=True
fps_override=30
device=cuda:0
batch_resize_preset=768px
naming_mode=both
```

**RTX 4090 notes:** tile 320 + 768px resize is a speed/quality step up from the ultra-safe 256/512 OOM profile. If you OOM, drop tile to 256 and resize to 512px (unload DiT + chunks stay on).

**Note:** `output_dir` is intentionally **not** set — FlashVSR upscales go to the default app `outputs` folder. Only toolbox final saves use `toolbox_output_dir`.

---

## Behaviour summary

### 1. UI defaults from config (`get_ui_defaults`)

Stock FlashVSR only read theme/autosave/output from `webui_config`. Added `get_ui_defaults()` so processing sliders load from config on startup:

- Upscale 4×, tiny, v1.1, chunks 12s, tiled DiT/VAE, unload DiT, tile 320/40, sage, bf16, cuda:0, quality 7, batch resize 768px, etc.

**Code:** `load_config()` + `_parse_config_value()`, `get_ui_defaults()`, `create_ui()` uses `ui[...]` for control `value=`.

### 2. Split output directories

| Stage | Path |
|-------|------|
| FlashVSR upscaled (working) | `C:\pinokio\api\FlashVSR_plus_pinokio.git\app\outputs` (default) |
| Toolbox complete / export | `D:\OUTPUTS\__X_GROK\Upscaled Videos\Current\Ready for CIV` |

**Code:** `get_toolbox_output_dir()`, `_apply_toolbox_output_dir()`, Settings → separate **FlashVSR Upscale** vs **Toolbox Final Save** fields.

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

1. Batch/single upscale → `app\outputs` (512px-wide center-crop input, 4×, chunks optional).
2. **Send to Toolbox** (or load from outputs).
3. Toolbox export → `D:\OUTPUTS\...\Ready for CIV`.

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

1. Copy or recreate `webui_config` (section above).
2. Copy `naming_utils.py` into `app\`.
3. Merge `webui.py` changes (or restore from backup/git diff). Key symbols to search:
   - `get_ui_defaults`
   - `get_toolbox_output_dir`
   - `apply_batch_resize_preset`
   - `input_align_step`
   - `center_crop_cover_pil`
   - `from naming_utils import`
4. Restart FlashVSR in Pinokio.

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