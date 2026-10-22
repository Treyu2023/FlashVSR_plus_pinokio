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
