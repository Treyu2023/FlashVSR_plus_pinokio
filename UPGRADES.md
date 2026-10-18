# What this fork adds, in plain language

## What this is

FlashVSR is an AI video upscaler: it takes a small or soft video and makes a sharper, larger version (here, up to 4K). This copy runs inside [Pinokio](https://pinokio.co/), a one-click app launcher, and is built on ai-anchorite's FlashVSR launcher.

This fork exists because its owner runs FlashVSR as an **unattended batch pipeline**: hundreds of downloaded clips go in one folder, and finished, smoothed, ready-to-post videos come out in another, on a single 24 GB graphics card, without babysitting. Stock FlashVSR is built for one video at a time. Everything below is what was added to make the batch version reliable.

Folder locations are settings, not code. They are listed once in [RESTORE.md, Settings](RESTORE.md#settings). Rebuild steps are in [RESTORE.md](RESTORE.md), and update safety is in [PRESERVE.md](PRESERVE.md).

## 1. Batch pipeline

- **Group Therapy.** Instead of upscaling everything, then smoothing everything, then exporting everything, it takes a small group (5 by default), runs each file through every step (upscale, motion smoothing twice, export), and only then moves on to the next group. If the run stops, finished work is really finished. Controlled by the group size setting and the Group Therapy tab.
- **Pairing by filename.** Each original and its finished version get the same short ID at the end of the filename (`_PID_...`), also written into the video's title. Before and After folders stay flat, with no folder per file.
- **One pair index.** The list of pairs lives in a single `_pairs` file instead of loose text files next to the videos.
- **Stop after current.** Pressing Stop takes effect at once: a red "STOP ARMED" line appears in the terminal, and the current file finishes before the run pauses.
- **Skip finished work.** Files that were already exported or already upscaled are recognised by name and are not processed again.

## 2. Keeping the intake folder clean

- **Queue check before starting.** Duplicates, wrong-stage files and files already in the queue are dropped or explained in the log before any work begins.
- **Duplicate pre-screen.** A catalog of the unique IDs in downloaded filenames, plus the original file size, lets new downloads that are really the same clip be skipped.
- **Browser "(1)" copies.** Chrome's "file (1).mp4" copies of the same clip are treated as one clip; separate takes of an AI clip are kept as separate jobs.
- **Reclaim processed downloads.** Once a clip is delivered, its original is moved to the archive folder and leftover intermediates are removed, so the intake folder only holds work still to do.

## 3. Readable names

- Output names say what happened, for example `myclip_4K_9x16_Upscaled_30fps.mp4`, and a stage tag at the end shows how far a file has gone.

## 4. Picture quality

- **HDR per file.** HDR clips are detected automatically and converted to standard range before upscaling, because the model only understands standard range.
- **Shrink without cropping.** Large inputs are scaled down to fit (never cropped) so the 4x result stays within 4K, keeping full colour detail and the source's colour range.
- **High-quality temporary encodes** (CRF 10) and colours that follow the source file.
- **Grid-aligned sizes.** Inputs are sized to what the model needs, with no stretching and no black bars.

## 5. Speed and stability on a 24 GB card

- The main model stays on the graphics card between tiles instead of being copied back and forth.
- Large temporary data is parked in system memory, one block at a time.
- 4K results are stitched together in a file on disk at half precision instead of filling memory.
- A slowdown that made 10-second clips take 15 minutes in the decode step was fixed.
- Old "GPU sharing" throttles were retired: the app always runs at normal priority.
- Faster defaults with the same quality: tile size 320 and the "medium" export preset.
- If the card runs out of memory, the app retries with safer settings automatically.

## 6. Seeing what it is doing

- The terminal prints a status line every few seconds, plus a "still busy, not frozen" line if one step takes long.
- Progress shows as pie bars with one header per file, and chunk lengths are shown in tenths of a second.

## 7. Safe start and update

- **Self-repair on start.** A known Windows problem that breaks one library file is repaired automatically before the app starts.
- **Custom files protected.** Before an update the custom files are snapshotted; after it, any file a stock update overwrote is put back automatically.
- **Faster, sturdier start.** Pinokio's own environment is skipped on auto-launch (it locked a file and crashed the first start), and the temp-folder wipe skips locked files instead of crashing.
- **Random seed on** by default, and kept after updates.
- **Update and Install follow this fork.** They pull only from this fork, and only when that can be done without merging. After a Reset, Install rebuilds the code from this fork instead of downloading the stock app. Personal backup copies of the settings file are kept out of git.

## 8. Folders are settings

Every folder the pipeline uses is a setting in the app's Settings tab (saved to `app/webui_config`). See the [Settings table](RESTORE.md#settings).

## What was deliberately not changed

The FlashVSR model itself and its weights are stock. All changes are in how files are fed in, scheduled, saved and protected.

## Going back to stock

Install ai-anchorite's original launcher from Pinokio Discover as a separate app. It downloads the stock FlashVSR app, and none of the above applies. To keep this fork but drop one feature, see the matching entry in [app/CUSTOMIZATIONS.md](app/CUSTOMIZATIONS.md).
