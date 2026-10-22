# Local patches on FlashVSR_plus_pinokio (FV-STOP)

Base: 89f0153460057e7893b6985d1a71b80307f191b8 (main = mine/main).
Local commit: <fill SHA after commit>, subject "FV-STOP: non-killing pid check, verified save, startup requeue, idle-free file stats".

| Item | File | Base lines | What |
|---|---|---|---|
| A | app/flashvsr_work_queue.py | 255-269 (_pid_alive), new _win_pid_alive before 339 | no os.kill(pid,0) on Windows |
| 2 | app/webui.py | 4687 (shutil.copy), new _copy_verified after 6563 | temp name + ffprobe duration + os.replace |
| 3 | app/webui.py | new _requeue_stuck_at_startup after 3264, call after 11365 | re-queue 'running' rows at startup |
| 4 | app/session_board.py | after 388 in _update_stats | close open file when no file is running |

Deploy ONLY when every FlashVSR queue is idle (no running item, no stop flag armed). Then restart the app from Pinokio.

## Save before a Pinokio update
    cd C:\pinokio\api\FlashVSR_plus_pinokio.git
    git format-patch -1 <SHA> -o C:\_Archive\fvstop-patches

## Check after any Pinokio update / git pull
    git log -1 --oneline
    git grep -n "os.kill(pid, 0)" -- app/flashvsr_work_queue.py   # must sit after the os.name == "nt" branch
    git grep -n "_copy_verified\|_requeue_stuck_at_startup" -- app/webui.py

## Reapply
    git status                                   # must be clean
    git am --3way C:\_Archive\fvstop-patches\0001-*.patch
    # or: git apply --check <patch>  then  git apply <patch>  and commit

## Rollback
    git revert <SHA>                             # keeps history
    # uncommitted: git apply -R <patch>
    # or: git checkout 89f0153 -- app/flashvsr_work_queue.py app/webui.py app/session_board.py

---
# FV-RESUME (on top of FV-STOP 77299662)

Local commit: <fill SHA after commit>, subject "FV-RESUME: keep finished chunks, untrack runtime config, trim pre-fix file stats".

| Item | File | Base lines (77299662) | What |
|---|---|---|---|
| A | app/webui.py | new helpers before 7130 (process_video_with_chunks); edits after 7159, 7169, at 7221, 7268, after 7296 | reuse finished chunks from app/_temp/chunk_resume/<key>/ |
| B | .gitignore | 15-16 | ignore app/webui_config, app/job_profiles.json |
| B | (index only) | - | `git rm --cached app/webui_config app/job_profiles.json` (local files stay) |
| C | app/session_board.py | 318 (_empty_stats), after 335 (_load_stats) | one-time trim of file_seconds without marker file_seconds_v=2 |
| R1c | app/webui.py | new `_chunk_resume_seed` before `_clear_chunk_resume_dir`; key tuple + call after `_chunk_resume_dir` | seed left out of the resume key; first run saves it to `<key>/seed.json`, resume reuses it (Randomize Seed stays on) |
| FV-PANE | start.js | event regex (base 40); new shell.run before local.set (base 47); url (base 54) | event `/Running on local URL:\s+(http:\/\/\S+)/`, url `{{input.event[1]}}`; wait for HTTP 200 on http://127.0.0.1:7860 (60 x 1 s, app venv python) before the pane opens; on timeout open anyway |

Item B note: a commit that untracks files deletes them from any OTHER clone that pulls it. Back up both files to C:\_Archive\<date>-<worker>\ before any pull of this commit elsewhere.
Save/reapply/rollback: same commands as above, with C:\_Archive\fvresume-patches; for B rollback also run `git add -f app/webui_config app/job_profiles.json`.

---
# FV-STATS (on top of a02b3900)

Local commit: <fill SHA after commit>, subject "FV-STATS: per-section stats buckets (v3) with archived reset/undo".

| Item | File | What |
|---|---|---|
| 1 | app/session_board.py | v3 store {v, buckets, archive, active_profile}; lock→bucket map; v2→group migration with one-time .v2.bak.json; reset_bucket/undo_reset |
| 2 | app/webui.py | collapsed "Stats reset" accordion in the session_board group (Arm → Confirm, Undo) |
| 3 | app/test_session_board.py | migration/.bak, isolation, no-lock skip, reset/undo/cap tests |

Before deploy: queue idle; copy app/outputs/session_board_stats.json to C:\_Archive\<date>-<worker>\.
Rollback: revert the commit (or `git apply -R`), then restore the stats file from app/outputs/session_board_stats.v2.bak.json (or the C:\_Archive copy). v2 code would read a v3 file as an empty flat record, so restore the .bak before restarting.

---
# FV-STATS-2 (on top of 3288a69c)

Local commit: <fill SHA after commit>, subject "FV-STATS-2: per-bucket session stats, archive in its own file, write only on change".

| File | What |
|---|---|
| app/session_board.py | session_file_seconds per bucket; new_session(); session vs all-time line; ETA uses session avg at >= 3 session files; reset/undo touch all-time keys only; archive moved to outputs/session_board_stats_archive.json (written on reset/undo only); stats file written only when its JSON changes |
| app/webui.py | "New session" button in the Stats reset accordion |
| app/test_session_board.py | session, ETA threshold, no-change tick, archive file, in-file archive migration tests |

Before deploy: queue idle; copy app/outputs/session_board_stats*.json to C:\_Archive\<date>-<worker>\.
Rollback: revert (or `git apply -R fvstats2-ALL.diff`), then restore the stats JSON from the C:\_Archive copy (3288a69c expects the archive inside the main file).
