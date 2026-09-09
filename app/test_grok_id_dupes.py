"""Grok-ID Chrome-copy screening (no catalog required)."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile
import time

import grok_id_index as g
from flashvsr_work_queue import FlashVSRWorkQueue, ST_PENDING


def _age(path: Path) -> None:
    old = time.time() - 120
    os.utime(path, (old, old))


def test_extract_and_rank() -> None:
    gid = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert g.extract_grok_ids(f"{gid} (27).mp4") == [gid]
    assert g.extract_grok_ids(f"{gid}_(27)_Upscaled_96fps_PID_deadbeef.mp4") == [gid]
    assert g.chrome_copy_rank(f"{gid}.mp4") == (0, 0)
    assert g.chrome_copy_rank(f"{gid} (1).mp4")[1] == 1
    assert g.chrome_copy_rank(f"{gid}_(27)_Upscaled.mp4")[1] == 27
    assert g.chrome_copy_number(f"{gid}.mp4") == 0
    assert g.chrome_copy_number(f"{gid} (4).mp4") == 4
    assert g.chrome_copy_number(f"{gid}_(4)_Upscaled_96fps_PID_deadbeef.mp4") == 4
    assert g.version_key(f"{gid} (3).mp4") == (gid, 3)
    assert g.version_key(f"{gid}_Upscaled_96fps.mp4") == (gid, 0)
    winner = g.pick_canonical_path(
        [f"{gid} (4).mp4", f"{gid}.mp4", f"{gid} (1).mp4"]
    )
    assert winner.endswith(f"{gid}.mp4")


def test_add_paths_keeps_distinct_imagine_takes() -> None:
    gid = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        app = root / "app"
        app.mkdir()
        inbox = root / "inbox"
        inbox.mkdir()
        after = root / "after"
        after.mkdir()
        blob = b"x" * (40 * 1024)
        (inbox / f"{gid}.mp4").write_bytes(blob)
        (inbox / f"{gid} (1).mp4").write_bytes(blob + b"1")  # different size = other take
        (inbox / f"{gid} (2).mp4").write_bytes(blob + b"22")
        (after / f"{gid}_Upscaled_96fps_PID_abcd1234.mp4").write_bytes(blob * 4)
        for p in inbox.iterdir():
            _age(p)
        for p in after.iterdir():
            _age(p)
        wq = FlashVSRWorkQueue(str(app), name="group", extensions={".mp4"}, label="t")
        scan = wq.add_folder(str(inbox), known_id_folders=[str(after)])
        # Unnumbered already in After; (1) and (2) are other Imagine takes.
        assert scan.added == 2, scan.summary()
        assert scan.grok_id_dupes >= 1
        names = sorted(Path(it["path"]).name for it in (wq.load().get("items") or []))
        assert names == [f"{gid} (1).mp4", f"{gid} (2).mp4"]

        wq2 = FlashVSRWorkQueue(str(app / "q2"), name="group", extensions={".mp4"}, label="t")
        scan2 = wq2.add_folder(str(inbox))
        assert scan2.added == 3, scan2.summary()
        assert scan2.grok_id_dupes == 0


def test_add_paths_skips_same_size_chrome_copy() -> None:
    gid = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        app = root / "app"
        app.mkdir()
        inbox = root / "inbox"
        inbox.mkdir()
        blob = b"z" * (40 * 1024)
        (inbox / f"{gid}.mp4").write_bytes(blob)
        (inbox / f"{gid} (1).mp4").write_bytes(blob)  # true Chrome re-download
        for p in inbox.iterdir():
            _age(p)
        wq = FlashVSRWorkQueue(str(app), name="group", extensions={".mp4"}, label="t")
        scan = wq.add_folder(str(inbox))
        assert scan.added == 1, scan.summary()
        assert scan.same_size_copy == 1


def test_preflight_keeps_distinct_takes() -> None:
    gid = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        app = root / "app3"
        app.mkdir()
        inbox = root / "inbox"
        inbox.mkdir()
        blob = b"y" * (40 * 1024)
        a = inbox / f"{gid}.mp4"
        b = inbox / f"{gid} (1).mp4"
        c = inbox / f"{gid} (2).mp4"
        a.write_bytes(blob)
        b.write_bytes(blob + b"1")
        c.write_bytes(blob + b"22")
        _age(a)
        _age(b)
        _age(c)
        wq = FlashVSRWorkQueue(str(app), name="group", extensions={".mp4"}, label="t")
        data = wq.load()
        data["items"] = [
            {"path": str(p), "status": ST_PENDING, "output": None, "error": None, "size": p.stat().st_size}
            for p in (a, b, c)
        ]
        wq.save(data)
        stats = wq.preflight_before_start(requeue_failed=False, remove_completed=True)
        assert stats["grok_id_dupes"] == 0, stats
        left = sorted(
            Path(it["path"]).name
            for it in (wq.load().get("items") or [])
            if it.get("status") == ST_PENDING
        )
        assert left == [f"{gid} (1).mp4", f"{gid} (2).mp4", f"{gid}.mp4"]


if __name__ == "__main__":
    test_extract_and_rank()
    test_add_paths_keeps_distinct_imagine_takes()
    test_add_paths_skips_same_size_chrome_copy()
    test_preflight_keeps_distinct_takes()
    print("ok")
