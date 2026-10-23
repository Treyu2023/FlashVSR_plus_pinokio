"""busy_heartbeat per-file metrics carrier for the session board (FV-UI)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import sys
import types

try:
    import tqdm  # noqa: F401
except ImportError:  # test box without tqdm: a stub is enough to import the module
    _stub = types.ModuleType("tqdm")
    _stub.tqdm = type("tqdm", (), {})
    sys.modules["tqdm"] = _stub

# Load the module by path: importing the src package pulls in torch.
_spec = importlib.util.spec_from_file_location("busy_heartbeat", Path(__file__).parent / "src" / "busy_heartbeat.py")
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)


def test_metrics_snapshot_and_clear() -> None:
    hb.clear_session()
    assert hb.snapshot()["file_metrics"] == {}
    hb.set_session(0, 3, 1, name="a.mp4", section="upscale")
    hb.set_file_metrics(mode="full", scale=4, s_per_frame=1.5, tiled_dit=True, junk=[1], none=None)
    m = hb.snapshot()["file_metrics"]
    assert m == {"mode": "full", "scale": 4, "s_per_frame": 1.5, "tiled_dit": True, "none": None}
    m["mode"] = "changed"
    assert hb.snapshot()["file_metrics"]["mode"] == "full"  # snapshot is a copy
    hb.file_finished(False)
    m = hb.snapshot()["file_metrics"]
    assert m["ok"] is False and m["finished"] > 0
    hb.set_session(0, 3, 1, name="a.mp4", section="upscale")  # same file: kept
    assert hb.snapshot()["file_metrics"]["mode"] == "full"
    hb.set_session(1, 3, 2, name="b.mp4", section="upscale")  # file change: cleared
    assert hb.snapshot()["file_metrics"] == {}
    hb.set_file_metrics(mode="tiny")
    hb.clear_session()
    assert hb.snapshot()["file_metrics"] == {}


if __name__ == "__main__":
    test_metrics_snapshot_and_clear()
    print("busy heartbeat metrics checks ok")
