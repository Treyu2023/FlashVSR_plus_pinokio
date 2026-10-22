"""Session board bars, session counts, health lamp, and job profiles."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import session_board as sb


def _queue(app: Path, name: str, items: list, **meta) -> None:
    folder = app / "outputs" / f"work_queue_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"items": items, "gt_group_size": 5, "gt_current_group": 2, "gt_current_stage": "upscale", "gt_stages": "upscale,rife1,rife2,export"}
    payload.update(meta)
    (folder / "queue.json").write_text(json.dumps(payload), encoding="utf-8")


def test_bars_and_session_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        items = [
            {"path": r"O:\in\a.mp4", "status": "done", "gt_group": 1},
            {"path": r"O:\in\b.mp4", "status": "running", "gt_group": 2, "gt_stage": "upscale"},
            {"path": r"O:\in\c.mp4", "status": "pending", "gt_group": 2},
            {"path": r"O:\in\d.mp4", "status": "failed", "gt_group": 2},
        ]
        _queue(app, "group", items)
        (app / "outputs" / "ACTIVE_QUEUE.lock").write_text(
            json.dumps({"queue": "group"}), encoding="utf-8"
        )
        html = sb.render(
            app,
            heartbeat={"file_name": "b.mp4", "step_label": "DiT", "step_done": 4, "step_total": 17, "section": "upscale"},
            sample={"gpu_util": 100, "mem_util": 80, "power_w": 280, "power_limit_w": 450, "vram_used_mib": 23000, "vram_total_mib": 24564, "temp_c": 70, "commit_used_gb": 40, "commit_limit_gb": 80, "phys_avail_gb": 20},
            now=1_000.0,
        )
        assert "Group" in html
        assert "+0" in html  # historical done is not this session
        assert "█" in html and "░" in html
        assert "Upscale" in html and "RIFE 1" in html
        items[2]["status"] = "done"
        _queue(app, "group", items)
        html2 = sb.render(
            app,
            heartbeat={"file_name": "b.mp4", "step_label": "DiT", "step_done": 5, "step_total": 17},
            sample={"gpu_util": 100, "power_w": 280, "power_limit_w": 450, "mem_util": 70, "commit_used_gb": 40, "commit_limit_gb": 80, "phys_avail_gb": 20},
            now=1_100.0,
        )
        assert "+1" in html2


def test_health_levels() -> None:
    stall = sb.classify_health(
        {"gpu_util": 100, "mem_util": 5, "power_w": 80, "power_limit_w": 450},
        {"step_label": "DiT"},
        {"last_step_advance": 0, "step_seconds": []},
        now=40,
        running=True,
    )
    assert stall["level"] == "thrash"
    healthy = sb.classify_health(
        {"gpu_util": 100, "mem_util": 90, "power_w": 320, "power_limit_w": 450, "commit_used_gb": 50, "commit_limit_gb": 90, "phys_avail_gb": 12},
        {"step_label": "tiles"},
        {"last_step_advance": 100, "step_seconds": [2, 2, 2, 2.2]},
        now=105,
        running=True,
    )
    assert healthy["level"] in ("ok", "busy")
    page = sb.classify_health(
        {"gpu_util": 50, "mem_util": 40, "power_w": 200, "power_limit_w": 450, "commit_used_gb": 70, "commit_limit_gb": 80, "phys_avail_gb": 8},
        {},
        {"last_step_advance": 100, "step_seconds": []},
        now=102,
        running=True,
    )
    assert page["level"] == "watch"


def test_profiles_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        names = sb.list_profiles(app)
        assert "4090 Group Therapy" in names
        sb.save_profile(app, "night batch", {"tile_size": 256, "quality": 8, "gt_group_size": 4}, note="quiet hours")
        loaded = sb.load_profile(app, "night batch")
        assert loaded["tile_size"] == 256
        assert loaded["note"] == "quiet hours"
        assert "gt_before_dir" not in sb.config_updates(loaded)
        left = sb.delete_profile(app, "night batch")
        assert "night batch" not in left


def _lock(app: Path, name) -> None:
    lock = app / "outputs" / "ACTIVE_QUEUE.lock"
    if name is None:
        if lock.exists():
            lock.unlink()
        return
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"queue": name}), encoding="utf-8")


def _tick(app: Path, name: str, now: float) -> str:
    return sb.render(app, heartbeat={"file_name": name, "file_index": 1}, sample={}, now=now)


def _store(app: Path) -> dict:
    return json.loads((app / "outputs" / "session_board_stats.json").read_text(encoding="utf-8"))


def test_migration_and_bak() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        out = app / "outputs"
        out.mkdir(parents=True)
        flat = {"pid": 1, "lifetime_done": 7, "file_seconds": [100.0, 120.0], "file_seconds_v": 2, "active_profile": "p"}
        (out / "session_board_stats.json").write_text(json.dumps(flat), encoding="utf-8")
        _queue(app, "group", [])
        _lock(app, "group")
        _tick(app, "a.mp4", 1000.0)
        st = _store(app)
        assert st["v"] == 3
        g = st["buckets"]["group"]
        assert g["file_seconds_v"] == 2 and g["file_seconds"] == [100.0, 120.0]
        assert g["lifetime_done"] == 7
        bak = out / "session_board_stats.v2.bak.json"
        assert json.loads(bak.read_text(encoding="utf-8")) == flat
        # Unversioned flat (no marker) still lands in group, trimmed by the v2 rule.
        bak.write_text("KEEP", encoding="utf-8")
        (out / "session_board_stats.json").write_text(json.dumps({"pid": 1, "file_seconds": [9000.0]}), encoding="utf-8")
        assert sb._load_stats(app, 1.0, "group")["file_seconds"] == []
        _tick(app, "a.mp4", 1002.0)
        assert bak.read_text(encoding="utf-8") == "KEEP"  # never overwritten
        # Corrupt file -> empty buckets, .bak untouched.
        (out / "session_board_stats.json").write_text("{not json", encoding="utf-8")
        assert sb._read_store(app)["buckets"] == {}
        _tick(app, "a.mp4", 1004.0)
        assert bak.read_text(encoding="utf-8") == "KEEP"
        assert _store(app)["v"] == 3
        # Missing bucket loads empty.
        assert sb._load_stats(app, 1.0, "toolbox")["file_seconds"] == []


def test_bucket_isolation_and_no_lock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        for q in ("group", "video", "image", "toolbox"):
            _queue(app, q, [])
        _lock(app, "video")
        _tick(app, "v1.mp4", 100.0)
        sb.render(app, heartbeat={}, sample={}, now=160.0)
        before = _store(app)["buckets"]
        assert before["upscale"]["file_seconds"] == [60.0]
        _lock(app, "image")
        _tick(app, "i1.png", 200.0)
        sb.render(app, heartbeat={}, sample={}, now=205.0)
        after = _store(app)["buckets"]
        assert after["image"]["file_seconds"] == [5.0]
        assert after["upscale"] == before["upscale"]
        assert "group" not in after
        # No lock: tick is skipped, nothing written.
        _lock(app, None)
        raw = (app / "outputs" / "session_board_stats.json").read_bytes()
        _tick(app, "x.mp4", 300.0)
        sb.render(app, heartbeat={}, sample={}, now=900.0)
        assert (app / "outputs" / "session_board_stats.json").read_bytes() == raw


def _arch(app: Path) -> list:
    path = app / "outputs" / "session_board_stats_archive.json"
    return json.loads(path.read_text(encoding="utf-8"))["archive"] if path.exists() else []


def test_reset_undo_and_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "group", [])
        _lock(app, "group")
        _tick(app, "a.mp4", 10.0)
        sb.render(app, heartbeat={}, sample={}, now=70.0)
        assert _store(app)["buckets"]["group"]["file_seconds"] == [60.0]
        sb.reset_bucket(app, "group", now=80.0)
        st = _store(app)
        assert "archive" not in st
        assert st["buckets"]["group"]["file_seconds"] == []
        assert st["buckets"]["group"]["session_file_seconds"] == [60.0]  # session survives reset
        assert _arch(app)[-1]["bucket"] == "group"
        assert _arch(app)[-1]["record"]["file_seconds"] == [60.0]
        assert "No reset" in sb.undo_reset(app, "image")
        sb.undo_reset(app, "group")
        assert _store(app)["buckets"]["group"]["file_seconds"] == [60.0]
        assert _arch(app) == []
        for i in range(55):
            sb.reset_bucket(app, "image", now=float(i))
        assert len(_arch(app)) == 50
        assert _arch(app)[0]["ts"] == 5.0
        assert "Unknown" in sb.reset_bucket(app, "nope")


def test_tick_writes_only_on_change() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "group", [])
        _lock(app, "group")
        _tick(app, "a.mp4", 10.0)
        sb.reset_bucket(app, "group", now=11.0)
        main = app / "outputs" / "session_board_stats.json"
        arch = app / "outputs" / "session_board_stats_archive.json"
        raw, mt = main.read_bytes(), main.stat().st_mtime_ns
        araw, amt = arch.read_bytes(), arch.stat().st_mtime_ns
        for t in (12.0, 14.0, 16.0):
            _tick(app, "a.mp4", t)  # same open file: nothing changes
        assert main.read_bytes() == raw and main.stat().st_mtime_ns == mt
        sb.render(app, heartbeat={}, sample={}, now=30.0)  # change: file closes
        assert main.read_bytes() != raw
        assert arch.read_bytes() == araw and arch.stat().st_mtime_ns == amt  # ticks never write archive


def test_in_file_archive_migrates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        out = app / "outputs"
        out.mkdir(parents=True)
        old = {"v": 3, "buckets": {"group": {"file_seconds": [5.0], "file_seconds_v": 2}},
               "archive": [{"bucket": "group", "ts": 1.0, "record": {"file_seconds": [9.0]}}], "active_profile": ""}
        (out / "session_board_stats.json").write_text(json.dumps(old), encoding="utf-8")
        _queue(app, "group", [])
        _lock(app, "group")
        _tick(app, "a.mp4", 10.0)
        assert "archive" not in _store(app)
        assert _arch(app) == old["archive"]
        _tick(app, "b.mp4", 20.0)
        assert _arch(app) == old["archive"]  # moved once, not duplicated
        sb.undo_reset(app, "group")
        assert _store(app)["buckets"]["group"]["file_seconds"] == [9.0]
        assert _arch(app) == []


def test_sessions() -> None:
    import os
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        out = app / "outputs"
        out.mkdir(parents=True)
        hist = {"pid": os.getpid() + 1, "lifetime_done": 40, "file_seconds": [3000.0] * 10, "file_seconds_v": 2,
                "session_file_seconds": [3000.0] * 10, "session_done": ["x"]}
        (out / "session_board_stats.json").write_text(json.dumps({"v": 3, "buckets": {"upscale": hist}}), encoding="utf-8")
        for q in ("video", "image"):
            _queue(app, q, [])
        _lock(app, "video")
        # Restart (stored pid differs): session starts empty, history kept.
        html = _tick(app, "a.mp4", 100.0)
        up = _store(app)["buckets"]["upscale"]
        assert up["session_file_seconds"] == [] and up["session_done"] == []
        assert up["lifetime_done"] == 41 and len(up["file_seconds"]) == 10
        assert "session 0" in html and "all-time 41" in html
        # Session avg ignores history; ETA switches at SESSION_ETA_MIN files.
        t = 100.0
        for i in range(sb.SESSION_ETA_MIN):
            t += 60.0
            _tick(app, f"f{i}.mp4", t)
        up = _store(app)["buckets"]["upscale"]
        assert up["session_file_seconds"] == [60.0] * sb.SESSION_ETA_MIN
        # Per-bucket: image session untouched by video ticks.
        assert sb._load_stats(app, t, "image")["session_file_seconds"] == []
        _lock(app, "image")
        _tick(app, "i.png", t + 1)
        sb.render(app, heartbeat={}, sample={}, now=t + 6)
        assert _store(app)["buckets"]["image"]["session_file_seconds"] == [5.0]
        assert _store(app)["buckets"]["upscale"]["session_file_seconds"] == [60.0] * sb.SESSION_ETA_MIN
        # New session: clears session only, all-time kept.
        sb.new_session(app, "upscale", now=t + 10)
        up = _store(app)["buckets"]["upscale"]
        assert up["session_file_seconds"] == [] and up["started"] == t + 10
        assert len(up["file_seconds"]) == 10 + sb.SESSION_ETA_MIN
        assert sb._load_stats(app, t, "image")["session_file_seconds"] == [5.0]


def test_eta_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "video", [{"path": f"p{i}", "status": "pending"} for i in range(10)])
        _lock(app, "video")
        st = sb._empty_stats(0.0)
        st["file_seconds"] = [1000.0]
        st["session_file_seconds"] = [100.0, 100.0]
        (app / "outputs" / "session_board_stats.json").write_text(json.dumps({"v": 3, "buckets": {"upscale": st}}), encoding="utf-8")
        html = sb.render(app, heartbeat={}, sample={}, now=1.0, persist=False)
        assert "ETA 2h 46m" in html  # 2 session files < 3: all-time avg 1000 s × 10
        st["session_file_seconds"] = [100.0, 100.0, 100.0]
        (app / "outputs" / "session_board_stats.json").write_text(json.dumps({"v": 3, "buckets": {"upscale": st}}), encoding="utf-8")
        html = sb.render(app, heartbeat={}, sample={}, now=1.0, persist=False)
        assert "ETA 16m" in html  # 3 session files: session avg 100 s × 10

if __name__ == "__main__":
    test_bars_and_session_count()
    test_health_levels()
    test_profiles_roundtrip()
    test_migration_and_bak()
    test_bucket_isolation_and_no_lock()
    test_reset_undo_and_cap()
    test_tick_writes_only_on_change()
    test_in_file_archive_migrates()
    test_sessions()
    test_eta_threshold()
    print("session board checks ok")
