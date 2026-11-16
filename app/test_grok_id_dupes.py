"""Grok-ID Chrome-copy screening (no catalog required)."""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import time

import grok_id_index as g
from flashvsr_work_queue import (
    FlashVSRWorkQueue,
    ST_PENDING,
    sweep_source_duplicates,
)


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


def test_match_file_keeps_other_chrome_take() -> None:
    gid = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        blob = b"q" * (40 * 1024)
        numbered = root / f"{gid} (4).mp4"
        plain = root / f"{gid}.mp4"
        numbered.write_bytes(blob + b"y" * 100)  # within ±2.5%, different take
        plain.write_bytes(blob + b"z" * 50)
        index = {
            "ids": {
                gid: {
                    "id": gid,
                    "original_size": len(blob),
                    "original_path": str(root / f"{gid}.mp4"),
                    "outputs": [],
                }
            }
        }
        assert g.match_file(str(numbered), index) is None
        hit = g.match_file(str(plain), index)
        assert hit is not None and "same take" in hit["reason"]

        legacy = {
            "ids": {
                gid: {
                    "id": gid,
                    "original_size": len(blob),
                    "original_path": "",
                    "outputs": [],
                }
            }
        }
        assert g.match_file(str(numbered), legacy) is None
        assert g.match_file(str(plain), legacy) is not None


def test_add_paths_queues_near_size_other_take() -> None:
    gid = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        app = root / "app"
        (app / "outputs").mkdir(parents=True)
        inbox = root / "inbox"
        inbox.mkdir()
        blob = b"q" * (40 * 1024)
        plain = inbox / f"{gid}.mp4"
        other = inbox / f"{gid} (4).mp4"
        plain.write_bytes(blob)
        other.write_bytes(blob + b"y" * 100)
        _age(plain)
        _age(other)
        index = {
            "ids": {
                gid: {
                    "id": gid,
                    "original_size": len(blob),
                    "original_path": str(plain),
                    "outputs": [],
                }
            }
        }
        (app / "outputs" / "grok_id_index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        wq = FlashVSRWorkQueue(str(app), name="group", extensions={".mp4"}, label="t")
        scan = wq.add_folder(str(inbox))
        names = sorted(Path(it["path"]).name for it in (wq.load().get("items") or []))
        assert names == [f"{gid} (4).mp4"], scan.summary()
        assert scan.added == 1


def test_sweep_parks_done_take_and_keeps_other_chrome() -> None:
    gid = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inbox = root / "inbox"
        before = root / "before"
        inbox.mkdir()
        before.mkdir()
        blob = b"x" * (40 * 1024)
        (before / f"{gid}.mp4").write_bytes(blob)
        redo = inbox / f"{gid}.mp4"
        other = inbox / f"{gid} (2).mp4"
        extra = inbox / f"{gid} (2) extra.mp4"
        redo.write_bytes(blob)
        other.write_bytes(blob + b"22")
        extra.write_bytes(blob + b"22")
        for p in list(inbox.iterdir()) + list(before.iterdir()):
            _age(p)
        app = root / "app"
        app.mkdir()
        wq = FlashVSRWorkQueue(str(app), name="video", extensions={".mp4"}, label="t")
        data = wq.load()
        data["items"] = [{
            "path": str(redo),
            "status": ST_PENDING,
            "output": None,
            "error": None,
            "size": redo.stat().st_size,
        }]
        wq.save(data)
        stats = sweep_source_duplicates(
            str(inbox),
            extensions={".mp4"},
            known_folders=[str(before)],
            archive_folders=[str(before)],
            queue_items=wq.all_items(),
        )
        assert stats["moved"] == 2, stats
        assert not redo.exists()
        left = [p.name for p in inbox.glob("*.mp4")]
        assert left == [other.name] or left == [extra.name], left
        parked = sorted(p.name for p in (inbox / "duplicates").glob("*.mp4"))
        assert f"DUPE_{gid}.mp4" in parked, parked
        assert sum("(2)" in name for name in parked) == 1, parked
        marked = (inbox / "duplicates" / "MARKED.txt").read_text(encoding="utf-8")
        assert "DUPE_" in marked
        assert "same Imagine take" in marked
        assert wq.drop_sources(stats["moved_from"]) == 1
        assert wq.load()["items"] == []


def test_sweep_keeps_other_chrome_take() -> None:
    gid = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inbox = root / "inbox"
        before = root / "before"
        inbox.mkdir()
        before.mkdir()
        blob = b"x" * (40 * 1024)
        (before / f"{gid}.mp4").write_bytes(blob)
        other = inbox / f"{gid} (1).mp4"
        other.write_bytes(blob + b"other-take")
        _age(other)
        _age(before / f"{gid}.mp4")
        stats = sweep_source_duplicates(
            str(inbox),
            extensions={".mp4"},
            known_folders=[str(before)],
            archive_folders=[str(before)],
        )
        assert stats["moved"] == 0, stats
        assert other.exists()
        assert not (inbox / "duplicates").exists()


def test_sweep_trusts_already_done_callback() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inbox = root / "inbox"
        inbox.mkdir()
        name = "interview_take_without_grok_id_long_name.mp4"
        src = inbox / name
        src.write_bytes(b"v" * (40 * 1024))
        _age(src)
        evidence = root / "already_upscaled.mp4"
        evidence.write_bytes(b"o" * (80 * 1024))

        def done(path: str):
            return str(evidence) if name in path else None

        stats = sweep_source_duplicates(
            str(inbox),
            extensions={".mp4"},
            already_done=done,
        )
        assert stats["moved"] == 1, stats
        assert not src.exists()
        assert (inbox / "duplicates" / f"DUPE_{name}").is_file()


if __name__ == "__main__":
    test_extract_and_rank()
    test_add_paths_keeps_distinct_imagine_takes()
    test_add_paths_skips_same_size_chrome_copy()
    test_preflight_keeps_distinct_takes()
    test_match_file_keeps_other_chrome_take()
    test_add_paths_queues_near_size_other_take()
    test_sweep_parks_done_take_and_keeps_other_chrome()
    test_sweep_keeps_other_chrome_take()
    test_sweep_trusts_already_done_callback()
    print("ok")
