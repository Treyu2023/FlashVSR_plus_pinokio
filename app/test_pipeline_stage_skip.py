"""Skip / move already-exported and already-upscaled inbox files."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import group_therapy as gt
from flashvsr_work_queue import (
    FlashVSRWorkQueue,
    ST_PENDING,
    looks_like_finished_export,
    looks_like_rife_output,
    looks_like_upscaled_output,
    seed_pipeline_fields,
)


EXPORT_NAME = "260503_862_3334_73_frames_4xFrames_exported_2368w_85q_005935.mp4"
TINY_NAME = "grok-video-ffffffff-bbbb-cccc-dddd-eeeeeeeeeeee_tiny_s2_20260426-215025.mp4"
RIFE4_NAME = "clip_73_frames_4xFrames.mp4"
RAW_NAME = "grok-video-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.mp4"


def _age(path: Path) -> None:
    old = time.time() - 120
    os.utime(path, (old, old))


def test_classifiers() -> None:
    assert looks_like_finished_export(EXPORT_NAME)
    assert not looks_like_rife_output(EXPORT_NAME)
    assert not looks_like_upscaled_output(EXPORT_NAME)
    assert looks_like_upscaled_output(TINY_NAME)
    assert looks_like_rife_output(RIFE4_NAME)
    assert not looks_like_finished_export(RAW_NAME)
    assert seed_pipeline_fields(TINY_NAME) == {"gt_upscale": TINY_NAME}
    seeded = seed_pipeline_fields(RIFE4_NAME)
    assert seeded["gt_rife1"] == RIFE4_NAME
    assert seeded["gt_rife2"] == RIFE4_NAME
    assert seed_pipeline_fields(EXPORT_NAME) == {}


def test_group_keeps_tiny_skips_export() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        app = root / "app"
        app.mkdir()
        inbox = root / "inbox"
        inbox.mkdir()
        blob = b"x" * (40 * 1024)
        (inbox / EXPORT_NAME).write_bytes(blob)
        (inbox / TINY_NAME).write_bytes(blob + b"tiny")
        (inbox / RAW_NAME).write_bytes(blob + b"raw")
        for p in inbox.iterdir():
            _age(p)
        wq = FlashVSRWorkQueue(str(app), name="group", extensions={".mp4"}, label="t")
        scan = wq.add_folder(str(inbox))
        names = sorted(Path(it["path"]).name for it in (wq.load().get("items") or []))
        assert EXPORT_NAME not in names
        assert TINY_NAME in names
        assert RAW_NAME in names
        assert scan.finished_export == 1
        tiny = next(it for it in wq.load()["items"] if it["path"].endswith(TINY_NAME))
        assert tiny.get("gt_upscale", "").endswith(TINY_NAME)


def test_video_queue_skips_all_pipeline_outputs() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        app = root / "app"
        app.mkdir()
        inbox = root / "inbox"
        inbox.mkdir()
        blob = b"x" * (40 * 1024)
        (inbox / EXPORT_NAME).write_bytes(blob)
        (inbox / TINY_NAME).write_bytes(blob + b"tiny")
        (inbox / RAW_NAME).write_bytes(blob + b"raw")
        for p in inbox.iterdir():
            _age(p)
        wq = FlashVSRWorkQueue(str(app), name="video", extensions={".mp4"}, label="t")
        scan = wq.add_folder(str(inbox))
        names = [Path(it["path"]).name for it in (wq.load().get("items") or [])]
        assert names == [RAW_NAME]
        assert scan.finished_export == 1
        assert scan.already_upscaled == 1


def test_reclaim_moves_export_keeps_tiny() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inbox = root / "inbox"
        before = root / "before"
        after = root / "after"
        inbox.mkdir()
        before.mkdir()
        after.mkdir()
        blob = b"x" * (40 * 1024)
        export = inbox / EXPORT_NAME
        tiny = inbox / TINY_NAME
        export.write_bytes(blob)
        tiny.write_bytes(blob + b"tiny")
        _age(export)
        _age(tiny)
        rec = gt.reclaim_watch_folder(str(inbox), str(before), str(after))
        assert rec["moved_after"] == 1
        assert rec["kept_partial"] == 1
        assert not export.exists()
        assert tiny.exists()
        after_files = list(after.glob("*.mp4"))
        assert len(after_files) == 1
        assert "exported_2368w_85q" in after_files[0].name
        assert "_PID_" in after_files[0].name


def test_mark_finished_exports_queue_row() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        app = root / "app"
        app.mkdir()
        inbox = root / "inbox"
        after = root / "after"
        inbox.mkdir()
        after.mkdir()
        src = inbox / EXPORT_NAME
        src.write_bytes(b"x" * (40 * 1024))
        _age(src)
        wq = FlashVSRWorkQueue(str(app), name="group", extensions={".mp4"}, label="t")
        data = wq.load()
        data["items"] = [
            {
                "path": str(src),
                "status": "running",
                "output": None,
                "error": "Stuck/interrupted while running — moved to end of queue",
                "size": src.stat().st_size,
                "gt_pair_id": "6696319b",
            }
        ]
        wq.save(data)
        stats = gt.mark_finished_exports(wq, str(after))
        assert stats["moved_after"] == 1
        assert stats["marked_done"] == 1
        assert not src.exists()
        row = wq.load()["items"][0]
        assert row["status"] == "done"
        assert os.path.isfile(row["output"])
        assert "PID_6696319b" in row["output"]


if __name__ == "__main__":
    test_classifiers()
    test_group_keeps_tiny_skips_export()
    test_video_queue_skips_all_pipeline_outputs()
    test_reclaim_moves_export_keeps_tiny()
    test_mark_finished_exports_queue_row()
    print("ok")
