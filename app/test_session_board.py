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
        assert "▶ Upscale" in html and "RIFE 1" in html
        assert "stage <b style='color:#7dd3fc'>Upscale</b>" in html
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


_SET = {"mode": "full", "model_version": "v1.1", "scale": 4, "dtype_str": "bf16", "attention_mode": "sage",
        "tiled_dit": True, "tiled_vae": False, "tile_size": 256, "tile_overlap": 24}


def _mtick(app: Path, name: str, now: float, **metrics) -> str:
    hb = {"file_name": name, "file_index": 1, "file_metrics": metrics}
    return sb.render(app, heartbeat=hb, sample={}, now=now)


def test_v3_to_v4_bak_once() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        out = app / "outputs"
        out.mkdir()
        v3 = {"v": 3, "buckets": {"group": {"lifetime_done": 7, "file_seconds": [5.0]}}}
        (out / "session_board_stats.json").write_text(json.dumps(v3), encoding="utf-8")
        _queue(app, "group", [])
        _lock(app, "group")
        _tick(app, "a.mp4", 1.0)
        bak3 = out / "session_board_stats.v3.bak.json"
        assert json.loads(bak3.read_text(encoding="utf-8")) == v3
        assert not (out / "session_board_stats.v2.bak.json").exists()
        st = _store(app)
        assert st["v"] == 3 and st["v4"]["stats_v"] == 4
        assert "recent_files" not in st["buckets"]["group"]  # buckets stay v3-shaped
        assert st["buckets"]["group"]["lifetime_done"] == 7
        bak3.write_text("KEEP", encoding="utf-8")
        _tick(app, "b.mp4", 5.0)
        assert bak3.read_text(encoding="utf-8") == "KEEP"


def test_v2_still_v2_bak() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        out = app / "outputs"
        out.mkdir()
        flat = {"lifetime_done": 3, "file_seconds": [9.0]}
        (out / "session_board_stats.json").write_text(json.dumps(flat), encoding="utf-8")
        _queue(app, "group", [])
        _lock(app, "group")
        _tick(app, "a.mp4", 1.0)
        assert json.loads((out / "session_board_stats.v2.bak.json").read_text(encoding="utf-8")) == flat
        assert not (out / "session_board_stats.v3.bak.json").exists()


def test_v4_unchanged_skip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "group", [])
        _lock(app, "group")
        _mtick(app, "a.mp4", 101.0, **_SET, s_per_frame=1.0)
        main = app / "outputs" / "session_board_stats.json"
        raw, mt = main.read_bytes(), main.stat().st_mtime_ns
        for t in (102.0, 103.0):
            _mtick(app, "a.mp4", t, **_SET, s_per_frame=1.0)
        assert main.read_bytes() == raw and main.stat().st_mtime_ns == mt


def test_cfg_key_and_by_cfg() -> None:
    assert sb.cfg_key(_SET) == "full|v1.1|x4|bf16|sage|t256/24|chunk0"
    assert sb.cfg_key(dict(_SET, tiled_dit=False, chunk_s=10.25)) == "full|v1.1|x4|bf16|sage|t0/24|chunk10.25"
    assert sb.cfg_key({}) is None
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "group", [])
        _lock(app, "group")
        _mtick(app, "a.mp4", 100.0, **_SET, s_per_frame=1.0, vram_reserved_peak_mb=1000)
        _mtick(app, "b.mp4", 110.0, **_SET, s_per_frame=3.0, vram_alloc_peak_mb=3000)
        _mtick(app, "c.mp4", 120.0)  # b closes; c has no metrics
        sb.render(app, heartbeat={}, sample={}, now=130.0)
        g = sb._read_store(app)["buckets"]["group"]
        row = g["by_cfg"][sb.cfg_key(_SET)]
        assert row["n"] == 2 and row["sum_spf"] == 4.0
        assert row["vram_peak_mb_max"] == 3000 and row["sum_vram_peak"] == 4000
        assert g["session_by_cfg"] == g["by_cfg"]
        assert [e["name"] for e in g["recent_files"]] == ["a.mp4", "b.mp4"]
        assert g["recent_files"][0]["secs"] == 10.0
        html = sb.compare_html(g)
        assert "2.00" in html and "3000 MB" in html


def test_recent_cap_and_finished_once() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "group", [])
        _lock(app, "group")
        for i in range(25):
            _mtick(app, f"f{i}.mp4", 100.0 + i * 10, **_SET, s_per_frame=1.0)
        _mtick(app, "last.mp4", 400.0, **_SET, s_per_frame=2.0, finished=405.0, ok=True)
        _mtick(app, "last.mp4", 410.0, **_SET, s_per_frame=2.0, finished=405.0, ok=True)
        sb.render(app, heartbeat={}, sample={}, now=420.0)
        g = sb._read_store(app)["buckets"]["group"]
        names = [e["name"] for e in g["recent_files"]]
        assert len(names) == 20 and names[-1] == "last.mp4" and names.count("last.mp4") == 1
        assert g["recent_files"][-1]["secs"] == 5.0
        assert g["by_cfg"][sb.cfg_key(_SET)]["n"] == 26
        html = sb.recent_files_html(g)
        assert html.count("<tr>") == 21 and "last.mp4" in html


def test_reset_undo_new_session_v4() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "group", [])
        _lock(app, "group")
        _mtick(app, "a.mp4", 100.0, **_SET, s_per_frame=1.0)
        sb.render(app, heartbeat={}, sample={}, now=110.0)
        k = sb.cfg_key(_SET)
        sb.new_session(app, "group", now=111.0)
        g = sb._read_store(app)["buckets"]["group"]
        assert g["session_by_cfg"] == {} and g["by_cfg"][k]["n"] == 1 and len(g["recent_files"]) == 1
        sb.reset_bucket(app, "group", now=112.0)
        g = sb._read_store(app)["buckets"]["group"]
        assert g["by_cfg"] == {} and g["recent_files"] == []
        sb.undo_reset(app, "group")
        g = sb._read_store(app)["buckets"]["group"]
        assert g["by_cfg"][k]["n"] == 1 and len(g["recent_files"]) == 1


def test_tolerant_reads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        out = app / "outputs"
        out.mkdir()
        (out / "session_board_stats.json").write_text(
            json.dumps({"v": 3, "buckets": {"group": {"lifetime_done": 1}}, "v4": "junk"}), encoding="utf-8")
        g = sb._bucket_stats(sb._read_store(app)["buckets"]["group"], 0.0)
        assert g["recent_files"] == [] and g["by_cfg"] == {} and g["session_by_cfg"] == {}
        assert "No finished" in sb.render_recent(app, "group")
        assert "No settings" in sb.render_compare(app, "group")
        assert "—" in sb.recent_files_html({"recent_files": [{"name": "x"}]})
        html = sb.render(app, heartbeat={}, sample={}, now=1.0, persist=False)
        assert ">Session<" in html and ">Total<" in html
        assert "session 0" in html and "all-time 1" in html


def test_stage_colors_and_per_file_time_ram() -> None:
    fields = sb.stage_usage_fields("upscale", 1_000.0, 1_125.0, 22_000.0)
    assert fields["gt_upscale_secs"] == 125.0
    assert fields["gt_upscale_ram_mb"] == 22000.0
    assert sb.stage_usage_fields("nope", 0, 1, 1) == {}

    item = {
        "path": r"O:\in\b clip.mp4",
        "status": "pending",
        "gt_group": 2,
        "gt_upscale": r"O:\stage\b.mp4",
        "gt_upscale_secs": 125.0,
        "gt_upscale_ram_mb": 22000.0,
        "gt_rife1_secs": 80.0,
        "gt_rife1_ram_mb": 9000.0,
    }
    lines = sb.stage_lines(item, now=2_000.0)
    assert lines[0][0] == "upscale" and "RAM 21.5 GB" in lines[0][1]
    assert lines[1][0] == "rife" and "RIFE 1" in lines[1][1]
    colors = {"upscale": "#fbbf24", "rife": "#c084fc", "saved": "#86efac"}
    html = sb.stage_stats_html([item], colors, now=2_000.0, group_id=2)
    assert "b clip.mp4" in html
    assert "color:#fbbf24" in html and "RAM 21.5 GB" in html
    assert "color:#c084fc" in html and "RIFE 1" in html
    running = {
        "path": r"O:\in\c.mp4",
        "status": "running",
        "gt_group": 2,
        "gt_stage": "rife2",
        "gt_stage_started": 1_900.0,
        "gt_upscale_secs": 10.0,
        "gt_upscale_ram_mb": 1000.0,
    }
    live = sb.stage_lines(running, now=2_000.0, live_ram=1500.0)
    assert any(key == "rife" and "so far" in text and "RIFE 2" in text for key, text in live)

    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        saved = sb.save_stage_colors(app, "#fbbf24", "not-a-color", "#86efac")
        assert saved["upscale"] == "#fbbf24"
        assert saved["rife"] == "#c084fc"
        assert sb.load_stage_colors(app)["upscale"] == "#fbbf24"
        _queue(app, "group", [item], gt_current_group=2, gt_current_stage="rife1")
        _lock(app, "group")
        board = sb.render(
            app,
            heartbeat={"file_name": "b clip.mp4", "section": "RIFE pass 1 (2×)"},
            sample={},
            now=2_000.0,
            persist=False,
        )
        assert "STAGE TIME AND RAM" in board
        assert "color:#fbbf24'>█" in board or "color:#fbbf24" in board
        assert "RAM 21.5 GB" in board


def test_plain_text_says_which_file_of_how_many() -> None:
    members = [
        {"path": r"O:\in\a.mp4", "status": "pending", "gt_group": 2, "gt_upscale": r"O:\stage\a.mp4"},
        {"path": r"O:\in\b clip.mp4", "status": "running", "gt_group": 2},
        {"path": r"O:\in\c.mp4", "status": "pending", "gt_group": 2},
        {"path": r"O:\in\d.mp4", "status": "pending", "gt_group": 2},
        {"path": r"O:\in\e.mp4", "status": "pending", "gt_group": 2},
    ]
    assert sb.file_place(members, "upscale", "b clip.mp4") == (2, 5)
    rife_rows = [
        {"path": r"O:\in\a.mp4", "status": "running", "gt_upscale": "u"},
        *[{"path": rf"O:\in\{i}.mp4", "status": "pending", "gt_upscale": "u"} for i in range(4)],
    ]
    assert sb.file_place(rife_rows, "rife1", "a.mp4") == (1, 5)
    assert sb.queue_file_place(members, {"file_index": 7, "file_total": 25}, "b clip.mp4") == (7, 25)

    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "group", members, gt_current_group=2, gt_current_stage="upscale")
        _lock(app, "group")
        html = sb.render(
            app,
            heartbeat={
                "file_name": "b clip.mp4",
                "file_index": 7,
                "file_total": 25,
                "section": "Upscale",
                "step_done": 2,
                "step_total": 10,
            },
            sample={},
            now=1.0,
            persist=False,
        )
        assert "File 2 of 5" in html
        assert "b clip.mp4" in html
        assert "queue file 7 of 25" in html
        assert "File 2 of 5 in this group" in html
        assert "File 7 of 25 in the queue" in html
        # The filename is a visible line, not only a tooltip.
        assert "font-size:20px" in html and "font-size:28px" in html


def test_stage_colors_fill_as_files_finish_each_wave() -> None:
    """One file of five upscaled is a blue fifth. RIFE turns that fifth purple. Save turns it green."""
    waiting = [{"path": f"O:\\in\\{i}.mp4", "status": "pending", "gt_group": 2} for i in range(4)]
    up_only = [{"path": r"O:\in\up.mp4", "status": "pending", "gt_group": 2, "gt_upscale": r"O:\stage\up.mp4"}]
    mix = sb.stage_mix(up_only + waiting)
    assert mix["upscale"] == 1 and mix["rife"] == 0 and mix["saved"] == 0 and mix["total"] == 5
    bar = sb.stage_bar(mix, width=10)
    assert bar.count("█") == 2 and bar.count("░") == 8
    assert "#38bdf8" in bar and "#c084fc" not in bar and "#86efac" not in bar
    # The batch meter is 24 blocks. One file of five is five blue blocks.
    assert sb.stage_bar(mix, width=24).count("█") == 5
    assert sb.stage_pct(mix) == "7%"

    # Still inside upscale: the stage label alone is not a finished file.
    running = sb.stage_mix([{"status": "running", "gt_stage": "upscale", "gt_group": 2}])
    assert running["upscale"] == 0 and running["wait"] == 1

    # Failed work does not paint the bar.
    dead = sb.stage_mix([{"status": "failed", "gt_upscale": r"O:\stage\x.mp4"}])
    assert dead["failed"] == 1 and dead["upscale"] == 0

    rife = [{"path": r"O:\in\r.mp4", "status": "pending", "gt_group": 2, "gt_upscale": "u", "gt_rife1": "r"}]
    saved = [{"path": r"O:\in\s.mp4", "status": "done", "gt_group": 1, "gt_export": "e", "gt_rife1": "r", "gt_upscale": "u"}]
    mixed = sb.stage_mix(saved + rife + up_only + waiting[:2])
    assert mixed == {"saved": 1, "rife": 1, "upscale": 1, "wait": 2, "failed": 0, "total": 5}
    stacked = sb.stage_bar(mixed, width=10)
    assert stacked.index("#86efac") < stacked.index("#c084fc") < stacked.index("#38bdf8")
    assert sb.stage_pct(mixed) == "40%"
    # RIFE 2 with no pass-1 path is still the purple band.
    assert sb.stage_mix([{"status": "pending", "gt_rife2": "r2", "gt_upscale": "u"}])["rife"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        items = saved + rife + up_only + waiting[:2]
        _queue(app, "group", items, gt_current_group=2, gt_current_stage="rife1")
        _lock(app, "group")
        html = sb.render(
            app,
            heartbeat={"file_name": "r.mp4", "section": "RIFE pass 1 (2×)", "step_done": 1, "step_total": 4},
            sample={},
            now=1.0,
            persist=False,
        )
        assert "THIS BATCH" in html and "QUEUE" in html
        assert "#86efac" in html and "#c084fc" in html and "#38bdf8" in html
        batch_at = html.index("THIS BATCH")
        queue_at = html.index("QUEUE")
        batch_html = html[batch_at:queue_at]
        queue_html = html[queue_at:]
        # Current group is 2: one purple, one blue, two waiting. The saved file is group 1.
        assert "color:#c084fc'>█" in batch_html and "color:#38bdf8'>█" in batch_html
        assert "color:#86efac'>█" not in batch_html
        assert batch_html.index("color:#c084fc'>█") < batch_html.index("color:#38bdf8'>█")
        assert "1</b> up" in batch_html and "1</b> rife" in batch_html
        assert queue_html.index("color:#86efac'>█") < queue_html.index("color:#c084fc'>█") < queue_html.index("color:#38bdf8'>█")
        assert "40%" in queue_html


def test_stage_row_follows_rife_and_export() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        (app / "outputs").mkdir()
        (app / "outputs" / "ACTIVE_QUEUE.lock").write_text(
            json.dumps({"queue": "group"}), encoding="utf-8"
        )
        # Queue label still says upscale. The running row and the heartbeat are on RIFE 1.
        _queue(app, "group", [
            {"path": r"O:\in\b.mp4", "status": "running", "gt_group": 2, "gt_stage": "rife1"},
        ])
        html = sb.render(
            app,
            heartbeat={"file_name": "b.mp4", "section": "RIFE pass 1 (2×)", "step_label": "frames"},
            sample={},
            now=1.0,
            persist=False,
        )
        assert "▶ RIFE 1" in html and "✓ Upscale" in html
        assert "RIFE 2" in html and "Export" in html
        assert "stage <b style='color:#7dd3fc'>RIFE 1</b>" in html
        _queue(app, "group", [
            {"path": r"O:\in\b.mp4", "status": "running", "gt_group": 2, "gt_stage": "export"},
        ], gt_current_stage="export")
        html = sb.render(
            app,
            heartbeat={"file_name": "b.mp4", "section": "Export / interpolation"},
            sample={},
            now=2.0,
            persist=False,
        )
        assert "▶ Export" in html and "✓ RIFE 2" in html and "✓ RIFE 1" in html


def test_render_job_steps() -> None:
    lab = sb.job_step_label
    assert lab({}, [{"status": "pending"}]) == "queued"
    assert lab({}, [{"status": "done"}]) == "idle"
    beat = {"file_name": "a.mp4", "file_index": 2, "file_done": 1, "file_total": 4}
    assert lab(dict(beat, section="loading model")) == "loading model"
    assert lab(dict(beat, section="chunk 2/5")) == "chunk 2/5"
    assert lab(dict(beat, section="encoding")) == "encoding"
    assert lab(dict(beat, section="encoding", file_metrics={"finished": 1.0})) == "done"
    assert lab(dict(beat, section="upscale", step_label="tiles")) == "tiles"
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "video", [{"path": "a.mp4", "status": "running"}])
        assert "No active job" in sb.render_job(app, heartbeat=beat)
        _lock(app, "video")
        st = sb._empty_stats(0.0)
        st["file_seconds"] = [100.0]
        (app / "outputs" / "session_board_stats.json").write_text(
            json.dumps({"v": 3, "buckets": {"upscale": st}}), encoding="utf-8")
        raw = (app / "outputs" / "session_board_stats.json").read_bytes()
        html = sb.render_job(app, heartbeat=dict(beat, step_done=5, step_total=20, step_elapsed=50.0,
                                                 file_metrics=dict(_SET, num_tiles=6)))
        assert "a.mp4" in html and "2/4" in html and "25%" in html
        assert "ETA 2m" in html  # step: 50/5*15 = 150 s
        assert "ETA 5m" in html  # files: 100 s x 3 left, all-time avg
        assert ">Session<" in html and ">Total<" in html
        assert "avg —" in html and "avg 1m" in html  # no session samples; all-time is 100s
        assert "all-time 0" in html
        assert "Advanced" in html and "<details" not in html and sb.cfg_key(_SET) in html
        assert (app / "outputs" / "session_board_stats.json").read_bytes() == raw  # read-only
        blank = sb.job_card_html("upscale", {}, {}, [{"status": "pending"}])
        assert "queued" in blank and ">Session<" in blank and ">Total<" in blank


def test_tags_rollup_and_locked_history() -> None:
    assert ("R4", "RIFE 4× frames (two 2× passes)") in sb.config_tags({"gt_rife_mode": "4x Frames"})
    assert sb.config_tags({"gt_rife_mode": "2x Frames"})[0][0] == "R2"
    assert any(short == "T320" for short, _tip in sb.config_tags({"tiled_dit": True, "tile_size": 320, "tile_overlap": 32}))
    legend = sb.legend_html()
    assert "R4" in legend and "RIFE 4×" in legend and "<table" in legend
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        _queue(app, "group", [])
        _lock(app, "group")
        _mtick(
            app, "a.mp4", 100.0, **_SET, s_per_frame=1.0, secs_hint=1,
            in_bytes=1_000_000, out_bytes=4_000_000, num_tiles=8, frames=80,
            in_w=100, in_h=50, out_w=400, out_h=200, gt_rife_mode="4x Frames",
        )
        sb.render(app, heartbeat={}, sample={}, now=110.0)
        g = sb._read_store(app)["buckets"]["group"]
        roll = g["rollup"]
        assert roll["n"] == 1 and roll["n_in"] == 1 and roll["n_out"] == 1
        assert roll["sum_pct"] == 300.0
        assert roll["n_steps"] == 1 and roll["sum_steps"] == 8
        assert "R4" in sb.recent_files_html(g)
        html = sb.render(app, heartbeat={"file_metrics": {"gt_rife_mode": "4x Frames", "mode": "tiny"}}, sample={}, now=120.0, persist=False)
        assert "THIS FILE" in html and "THIS BATCH" in html and "QUEUE" in html
        assert "R4" in html and "History" in html and "nothing parked yet" in html
        assert "in → out" in html
        sb.reset_bucket(app, "group", now=130.0)
        parked = sb.render(app, heartbeat={}, sample={}, now=140.0, persist=False)
        assert "1 era" in parked and "locked" in parked
        assert "History is locked" in sb.clear_history(app, "group", unlocked=False)
        assert "1 era" in sb.render(app, heartbeat={}, sample={}, now=150.0, persist=False)
        assert "Cleared 1" in sb.clear_history(app, "group", unlocked=True)
        assert "nothing parked yet" in sb.render(app, heartbeat={}, sample={}, now=160.0, persist=False)
        _mtick(app, "b.mp4", 200.0, **_SET, s_per_frame=2.0, in_bytes=100, finished=205.0, ok=True)
        sb.render(app, heartbeat={}, sample={}, now=210.0)
        note = sb.annotate_settled(app, "b.mp4", in_bytes=200, out_bytes=800, gt_rife_mode="4x Frames")
        assert note == "updated the finished file"
        g = sb._read_store(app)["buckets"]["group"]
        assert g["rollup"]["n_pct"] == 1
        assert g["rollup"]["sum_in"] == 200
        assert g["recent_files"][-1]["out_bytes"] == 800


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
    test_v3_to_v4_bak_once()
    test_v2_still_v2_bak()
    test_v4_unchanged_skip()
    test_cfg_key_and_by_cfg()
    test_recent_cap_and_finished_once()
    test_reset_undo_new_session_v4()
    test_tolerant_reads()
    test_stage_colors_and_per_file_time_ram()
    test_plain_text_says_which_file_of_how_many()
    test_stage_colors_fill_as_files_finish_each_wave()
    test_stage_row_follows_rife_and_export()
    test_render_job_steps()
    test_tags_rollup_and_locked_history()
    print("session board checks ok")
