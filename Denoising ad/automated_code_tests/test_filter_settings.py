"""
Tests for choosing each filter's settings: presets, hand edits, and saved records.

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s tests -p "test_filter_settings.py" -v

Real-job tests use real project images and save to a temporary folder.
Page tests replace job starting with a mock, so no experiment runs.
"""

import functools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from test_single_image_job import PROJECT_DIR, make_job, run_real_job

import app_helpers as helpers
import experiment_job as jobs
import job_runner
import main
from saved_runs import needs_saved_runs

SOURCES = ("default", "previous_2_image", "new_12_image")


def settings(preset, **parameters):
    return {"preset": preset, "parameters": parameters}


def param_key(filter_name, name):
    return f"param::{filter_name}::{name}"


class SettingRuleTests(unittest.TestCase):
    def test_every_preset_fits_the_allowed_choices(self):
        for noise_type, levels in main.NOISE_LEVELS.items():
            for level in levels:
                for name, function in main.filters.items():
                    for source in SOURCES:
                        with self.subTest(noise=noise_type, level=level, filter=name, source=source):
                            values = jobs.preset_parameters(source, name, function, noise_type, level)
                            jobs.validate_filter_parameters(name, values)

        self.assertEqual(
            jobs.preset_parameters("default", "NLM Filter", main.filters["NLM Filter"], "Gaussian", "Low"),
            {"patch_size": 5, "patch_distance": 6, "h": None},
        )
        self.assertEqual(
            jobs.preset_parameters("previous_2_image", "Bilateral Filter",
                                   main.filters["Bilateral Filter"], "Gaussian", "Low"),
            {"diameter": 9, "sigma_color": 0.1, "sigma_space": 75},
        )

    def test_invalid_settings_are_rejected_before_start(self):
        cases = {
            "even kernel": {"Median Filter": settings("default", kernel_size=4)},
            "missing value": {"Gaussian Filter": settings("default", kernel_size=5)},
            "conflicting speckle value": {
                "Lee Filter": settings("default", window_size=5, speckle_variance=0.5)
            },
            "true instead of number": {"Median Filter": settings("default", kernel_size=True)},
            "h not offered": {"NLM Filter": settings("default", patch_size=5, patch_distance=6, h=0.13)},
            "unknown preset": {"Median Filter": settings("best_guess", kernel_size=5)},
            "unknown filter": {"Wiener Filter": settings("default", size=3)},
            "no filters": {},
        }
        for label, chosen in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(job_runner, "JOBS_DIR", Path(folder) / "jobs"), \
                    mock.patch.object(job_runner.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    job_runner.start_job(make_job(filters=chosen))
                popen.assert_not_called()
                self.assertIsNone(job_runner.active_job_id())


class MixedSettingsRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.status, _, cls.calls = run_real_job(cls.root, make_job(filters={
            "Gaussian Filter": settings("default", kernel_size=5, sigma=1.0),
            "Median Filter": settings("new_12_image", kernel_size=3),
            "Bilateral Filter": settings("previous_2_image", diameter=9, sigma_color=0.1, sigma_space=75),
            "NLM Filter": settings("new_12_image", patch_size=5, patch_distance=9, h=0.1),
            "Lee Filter": settings("default", window_size=5),
            "TV Filter": settings("new_12_image", weight=0.03),
        }))
        cls.run_dir = cls.root / "results" / "result1"
        cls.config = json.loads((cls.run_dir / "run_config.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_each_filter_receives_its_own_values(self):
        self.assertEqual(self.status["state"], "completed")
        self.assertEqual([(name, kwargs) for name, _, kwargs in self.calls], [
            ("Gaussian Filter", {"kernel_size": 5, "sigma": 1.0}),
            ("Median Filter", {"kernel_size": 3}),
            ("Bilateral Filter", {"diameter": 9, "sigma_color": 0.1, "sigma_space": 75}),
            ("NLM Filter", {"patch_size": 5, "patch_distance": 9, "h": 0.1}),
            ("Lee Filter", {"window_size": 5}),
            ("TV Filter", {"weight": 0.03}),
        ])
        self.assertEqual(len({digest for _, digest, _ in self.calls}), 1)

    def test_metadata_records_preset_and_final_values(self):
        source = self.config["required_to_rerun"]["parameter_source"]
        self.assertEqual(source["mode"], "per_filter")
        nlm = source["filters"]["NLM Filter"]
        self.assertEqual(nlm["preset"], "new_12_image")
        self.assertEqual(nlm["preset_parameters"], {"patch_size": 5, "patch_distance": 6, "h": 0.1})
        self.assertEqual(nlm["parameters"], {"patch_size": 5, "patch_distance": 9, "h": 0.1})
        self.assertTrue(nlm["edited"])
        self.assertFalse(source["filters"]["Median Filter"]["edited"])
        self.assertEqual(source["filters"]["Bilateral Filter"]["preset"], "previous_2_image")

        applied = self.config["required_to_rerun"]["conditions"][0]["filters"]["NLM Filter"]
        self.assertEqual(applied["resolved"], {"patch_size": 5, "patch_distance": 9, "h": 0.1})

    def test_results_view_shows_presets_and_applied_values(self):
        results = self.root / "results"
        st.cache_data.clear()
        with mock.patch.object(helpers, "find_saved_runs",
                               functools.partial(helpers.find_saved_runs, results)), \
                mock.patch.object(helpers, "load_summary",
                                  functools.partial(helpers.load_summary, results_dir=results)), \
                mock.patch.object(helpers, "load_image_rows",
                                  functools.partial(helpers.load_image_rows, results_dir=results)), \
                mock.patch.object(helpers, "comparison_images",
                                  functools.partial(helpers.comparison_images, results_dir=results)):
            app = AppTest.from_file(str(PROJECT_DIR / "frontend" / "app.py"), default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
        st.cache_data.clear()

        self.assertEqual(len(app.exception), 0)
        # The settings table is in the "Details and settings" tab
        table = next(t.value for t in app.dataframe if "Settings used" in t.value.columns).set_index("Filter")
        self.assertEqual(table.loc["NLM", "Settings used"], "patch_size=5, patch_distance=9, h=0.1")
        self.assertEqual(table.loc["NLM", "Preset"], "Optimized settings (tuned on Set12), then custom values")
        # Older runs that used the now-hidden 2-image preset still name it
        self.assertEqual(table.loc["Bilateral", "Preset"], "Old 2-image tuned settings (not recommended)")


@needs_saved_runs('result13')
class BackendAgreementTests(unittest.TestCase):
    def test_unedited_presets_match_full_dataset_scores(self):
        chosen = {
            name: settings("previous_2_image", **jobs.preset_parameters(
                "previous_2_image", name, function, "Speckle", "Low"))
            for name, function in main.filters.items()
        }
        with tempfile.TemporaryDirectory() as folder:
            status, seeds, calls = run_real_job(Path(folder), make_job(
                image="test003.png", noise_type="Speckle", noise_level="Low", filters=chosen))
            rows = pd.read_csv(Path(folder) / "results" / "result1" / "experiment_results.csv")

        self.assertEqual(status["state"], "completed")
        self.assertEqual(seeds, [("Speckle", 0.01, 2242)])
        lee = [kwargs for name, _, kwargs in calls if name == "Lee Filter"][0]
        self.assertEqual(lee, {"window_size": 5, "speckle_variance": 0.01})

        saved = pd.read_csv(PROJECT_DIR / "results" / "result13" / "experiment_results.csv")
        joined = rows.merge(saved, on=["Image", "Noise_Type", "Noise_Level", "Filter"])
        self.assertEqual(len(joined), 7)
        pd.testing.assert_series_equal(joined["PSNR_x"], joined["PSNR_y"], check_names=False, rtol=1e-12)
        pd.testing.assert_series_equal(joined["SSIM_x"], joined["SSIM_y"], check_names=False, rtol=1e-12)


class RunViewTests(unittest.TestCase):
    def open_run_view(self):
        app = AppTest.from_file(str(PROJECT_DIR / "frontend" / "app.py"), default_timeout=30)
        app.session_state["view"] = "results"  # the app now opens on Home
        app.run()
        app.segmented_control(key="view").set_value("run").run()
        app.selectbox(key="run_noise").set_value("Gaussian")
        app.selectbox(key="run_level").set_value("Low").run()
        return app

    def visible_setting_keys(self, app):
        return sorted(box.key for box in app.selectbox if box.key.startswith("param::"))

    def test_one_filter_shows_only_its_own_settings(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None):
            app = self.open_run_view()
            app.selectbox(key="run_filter").set_value("Median Filter").run()
            self.assertEqual(self.visible_setting_keys(app), [param_key("Median Filter", "kernel_size")])

            app.selectbox(key="run_filter").set_value("NLM Filter").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(self.visible_setting_keys(app), [
                param_key("NLM Filter", "h"),
                param_key("NLM Filter", "patch_distance"),
                param_key("NLM Filter", "patch_size"),
            ])

    def test_preset_fills_controls_and_edits_are_sent(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-1") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = self.open_run_view()
            app.selectbox(key="run_filter").set_value("NLM Filter").run()
            self.assertIsNone(app.selectbox(key=param_key("NLM Filter", "h")).value)

            app.selectbox(key="preset::NLM Filter").set_value("new_12_image").run()
            self.assertEqual(app.selectbox(key=param_key("NLM Filter", "h")).value, 0.1)

            app.selectbox(key=param_key("NLM Filter", "patch_distance")).set_value(9).run()
            self.assertIn("Custom values", " ".join(item.value for item in app.caption))
            start.assert_not_called()

            app.button(key="run_experiment").click().run()
            self.assertEqual(len(app.exception), 0)
            sent = start.call_args.args[0]["filters"]
            self.assertEqual(sent, {
                "NLM Filter": settings("new_12_image", patch_size=5, patch_distance=9, h=0.1)
            })

    def test_all_six_filters_keep_independent_settings(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-1") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = self.open_run_view()
            app.selectbox(key="run_filter").set_value("All six filters").run()
            self.assertEqual(
                [expander.label for expander in app.expander if expander.label in
                 ("Gaussian", "Median", "Bilateral", "NLM", "Lee", "TV")],
                ["Gaussian", "Median", "Bilateral", "NLM", "Lee", "TV"],
            )

            # Only Default and Optimized are offered; the old 2-image preset is hidden
            self.assertEqual(list(app.selectbox(key="preset::Bilateral Filter").options),
                             ["Default settings", "Optimized settings (tuned on Set12)"])

            app.selectbox(key=param_key("Median Filter", "kernel_size")).set_value(7).run()
            app.selectbox(key=param_key("TV Filter", "weight")).set_value(0.2).run()
            app.selectbox(key="preset::Bilateral Filter").set_value("new_12_image").run()
            start.assert_not_called()

            app.button(key="run_experiment").click().run()
            sent = start.call_args.args[0]["filters"]
            self.assertEqual(list(sent), list(main.filters))
            self.assertEqual(sent["Median Filter"], settings("default", kernel_size=7))
            self.assertEqual(sent["TV Filter"], settings("default", weight=0.2))
            self.assertEqual(sent["Bilateral Filter"], settings(
                "new_12_image", **jobs.preset_parameters(
                    "new_12_image", "Bilateral Filter", main.filters["Bilateral Filter"], "Gaussian", "Low")))
            self.assertEqual(sent["Gaussian Filter"], settings("default", kernel_size=5, sigma=1.0))

            app.selectbox(key="run_noise").set_value("Speckle").run()
            self.assertEqual(start.call_count, 1)

    def test_lee_shows_speckle_variance_as_read_only(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None):
            app = self.open_run_view()
            app.selectbox(key="run_noise").set_value("Speckle")
            app.selectbox(key="run_level").set_value("Medium")
            app.selectbox(key="run_filter").set_value("Lee Filter").run()
            self.assertEqual(self.visible_setting_keys(app), [param_key("Lee Filter", "window_size")])
            self.assertIn("speckle variance 0.04", " ".join(item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
