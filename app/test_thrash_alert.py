import unittest
from toolbox import thrash_alert as ta

CFG = ta.load_config()
LOW = {"vram_used_gb": 14.0, "vram_total_gb": 24.5, "util_pct": 98, "power_w": 320, "step_s": 30, "avg_step_s": 28, "silent_s": 20}
NEAR = {"vram_used_gb": 23.0, "vram_total_gb": 24.5, "util_pct": 100, "power_w": 170, "step_s": 120, "avg_step_s": 28, "silent_s": 240}
THRASH = {"vram_used_gb": 23.4, "vram_total_gb": 24.5, "util_pct": 100, "power_w": 107, "step_s": 5040, "avg_step_s": 600, "silent_s": 5040}


class TestThrash(unittest.TestCase):
    def test_levels(self):
        for r, want in ((LOW, "low"), (NEAR, "near"), (THRASH, "thrash")):
            s, _ = ta.thrash_score(r, CFG)
            self.assertEqual(ta.level(s, CFG), want, (want, s))

    def test_reason_text(self):
        s, why = ta.thrash_score(THRASH, CFG)
        self.assertTrue(any(w.startswith("VRAM 9") for w in why))
        self.assertTrue(any("x slower" in w for w in why))

    def test_empty_and_bounds(self):
        self.assertEqual(ta.thrash_score({}, CFG)[0], 0.0)
        self.assertLessEqual(ta.thrash_score(THRASH, CFG)[0], 1.0)

    def test_low_util_ignores_power(self):
        s, _ = ta.thrash_score({"util_pct": 20, "power_w": 30}, CFG)
        self.assertEqual(s, 0.0)

    def test_overlay(self):
        self.assertEqual(ta.render_overlay(0.9, ["x"], CFG, muted=True), "")
        h = ta.render_overlay(0.9, ["VRAM 95%"], CFG)
        self.assertIn("class='on'", h); self.assertIn("prefers-reduced-motion", h)
        self.assertNotIn("http", h)

    def test_tracker(self):
        t = ta.StepTracker(8)
        for i, now in enumerate([0, 10, 20, 30]):
            t.update({"step_done": i, "step_total": 10, "step_label": "DiT"}, now)
        out = t.update({"step_done": 3, "step_total": 10, "step_label": "DiT"}, 330)
        self.assertEqual(out["silent_s"], 300)
        self.assertAlmostEqual(out["step_s"] / out["avg_step_s"], 30.0)


if __name__ == "__main__":
    unittest.main()
