"""
Focused tests for the read-only Results Viewer.

Run from the "Denoising ad" folder:

    python -B -m unittest discover -s tests -p "test_results_viewer.py" -v

These tests need pandas and streamlit. They read saved results and only
write small practice files to a temporary folder.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "frontend"))

import app_helpers as helpers


def summary_table(scores):
    rows = [
        {"Noise_Type": "Speckle", "Noise_Level": "High", "Filter": name,
         "PSNR_mean": psnr, "SSIM_mean": ssim}
        for name, psnr, ssim in scores
    ]
    return pd.DataFrame(rows)


def page_images(node):
    children = getattr(node, "children", None)
    if not children:
        return [node] if getattr(node, "type", "") == "image" else []
    return [image for child in children.values() for image in page_images(child)]


class SavedRunTests(unittest.TestCase):
    def test_lists_completed_runs_with_metadata_newest_first(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, status in (("result9", "completed"), ("result10", "completed"),
                                 ("result11", "incomplete"), ("resultX", "completed")):
                run = root / name
                run.mkdir()
                (run / "experiment_results.csv").write_text("x\n", encoding="utf-8")
                (run / "summary.csv").write_text("x\n", encoding="utf-8")
                config = {"run": {"status": status}}
                (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

            no_metadata = root / "result12"
            no_metadata.mkdir()
            (no_metadata / "experiment_results.csv").write_text("x\n", encoding="utf-8")
            (no_metadata / "summary.csv").write_text("x\n", encoding="utf-8")

            self.assertEqual(list(helpers.find_saved_runs(root)), ["result10", "result9"])


class DisplayRuleTests(unittest.TestCase):
    def test_best_filter_uses_saved_average_and_ignores_noisy_baseline(self):
        summary = summary_table([
            ("Noisy (Baseline)", 40.0, 0.9),
            ("Lee Filter", 24.0, 0.7),
            ("TV Filter", 25.0, 0.6),
        ])
        baseline, filters = helpers.get_condition(summary, "Speckle", "High")
        self.assertEqual(baseline["PSNR_mean"], 40.0)
        self.assertEqual(list(filters["Filter"]), ["TV Filter", "Lee Filter"])

    def test_only_differences_below_limit_are_close(self):
        filters = pd.DataFrame({
            "Filter": ["Lee Filter", "TV Filter", "NLM Filter"],
            "PSNR_mean": [24.00, 23.85, 23.75],
        })
        close = helpers.find_close_filters(filters)
        self.assertEqual([name for name, _ in close], ["TV"])
        self.assertAlmostEqual(close[0][1], 0.15)

    def test_counts_images_where_filter_is_best_including_ties(self):
        rows = pd.DataFrame({
            "Image": ["a", "a", "b", "b", "c", "c", "a"],
            "Noise_Type": ["Speckle"] * 7,
            "Noise_Level": ["High"] * 7,
            "Filter": ["Lee Filter", "TV Filter"] * 3 + ["Noisy (Baseline)"],
            "PSNR": [25.0, 24.0, 22.0, 23.0, 21.0, 21.0, 99.0],
        })
        self.assertEqual(
            helpers.count_best_images(rows, "Speckle", "High", "Lee Filter"), (2, 3)
        )

    def test_settings_show_actual_values_and_modes(self):
        cases = [
            ({"resolved": {"patch_size": 5, "h": 0.15}, "h_mode": "explicit"},
             "patch_size=5, h=0.15"),
            ({"resolved": {"patch_size": 5, "h": None}, "h_mode": "automatic"},
             "patch_size=5, h=automatic"),
            ({"resolved": {"window_size": 7, "speckle_variance": 0.04},
              "model": "multiplicative speckle"},
             "window_size=7, speckle_variance=0.04 (speckle model)"),
            ({"resolved": {"window_size": 5, "speckle_variance": None},
              "model": "general adaptive smoother"},
             "window_size=5 (general smoother)"),
        ]
        for entry, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(helpers.format_settings(entry), expected)


class RealResult13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = helpers.find_saved_runs()["result13"]
        cls.summary = helpers.load_summary("result13")
        cls.rows = helpers.load_image_rows("result13")

    def test_all_conditions_match_raw_csv_and_have_images(self):
        conditions = self.config["required_to_rerun"]["conditions"]
        self.assertEqual(len(conditions), 12)

        for condition in conditions:
            noise, level = condition["noise_type"], condition["level"]
            with self.subTest(noise=noise, level=level):
                _, filters = helpers.get_condition(self.summary, noise, level)
                raw = self.rows[
                    (self.rows["Noise_Type"] == noise)
                    & (self.rows["Noise_Level"] == level)
                    & (self.rows["Filter"] != helpers.BASELINE_NAME)
                ].groupby("Filter")["PSNR"].mean()
                self.assertEqual(len(filters), 6)
                self.assertEqual(filters.iloc[0]["Filter"], raw.idxmax())
                self.assertAlmostEqual(filters.iloc[0]["PSNR_mean"], raw.max(), places=9)
                self.assertTrue(
                    helpers.comparison_image_path("result13", noise, level).is_file()
                )

    def test_page_shows_clear_and_close_results(self):
        app = AppTest.from_file(str(PROJECT_DIR / "frontend" / "app.py"), default_timeout=30)
        app.run()
        self.assertEqual(len(app.exception), 0)

        app.selectbox(key="saved_run").set_value("result13").run()
        self.assertEqual(app.selectbox(key="saved_run").value, "result13")

        app.selectbox(key="noise_type").set_value("Salt-and-Pepper")
        app.selectbox(key="noise_level").set_value("Medium").run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Median", app.subheader[0].value)
        main_table = next(t.value for t in app.dataframe if "Settings used" in t.value.columns)
        self.assertEqual(len(main_table), 6)
        self.assertEqual(len(page_images(app.main)), 1)
        self.assertTrue(any("Clear result" in item.value for item in app.success))

        app.selectbox(key="noise_type").set_value("Speckle")
        app.selectbox(key="noise_level").set_value("High").run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Lee", app.subheader[0].value)
        self.assertTrue(any("Close result" in item.value for item in app.warning))


if __name__ == "__main__":
    unittest.main()
