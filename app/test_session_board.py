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


if __name__ == "__main__":
    test_bars_and_session_count()
    test_health_levels()
    test_profiles_roundtrip()
    print("session board checks ok")
