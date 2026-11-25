"""
Focused tests for the full experiment: every image under all 12 noise conditions.

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s automated_code_tests -p "test_full_matrix.py" -v

Real-job tests use two real BSD68 images in a temporary folder.
Page tests replace job starting with a mock.
"""

import contextlib
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

from test_filter_settings import settings
from test_full_dataset_job import make_project, run_job_in
from test_single_image_job import PROJECT_DIR

import app_helpers as helpers
import experiment_job as jobs
import job_runner
import main
from saved_runs import needs_saved_runs

APP = str(PROJECT_DIR / "frontend" / "app.py")
CONDITIONS = [(noise, level) for noise, levels in main.NOISE_LEVELS.items() for level in levels]
NLM_VALUES = {"patch_size": 5, "patch_distance": 6, "h": 0.15}


def matrix_job(filters, **changes):
    job = {
        "type": "full_dataset",
        "dataset": "BSD68",
        "image": None,
        "noise_scope": "all",
        "noise_type": None,
        "noise_level": None,
        "filters": filters,
        "save_pictures": True,
    }
    job.update(changes)
    return job


def use_results_folder(results):
    stack = contextlib.ExitStack()
    for name in ("find_saved_runs", "load_summary", "load_image_rows", "comparison_images"):
        stack.enter_context(mock.patch.object(
            helpers, name, functools.partial(getattr(helpers, name), results_dir=results)))
    return stack


class FullMatrixRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        make_project(cls.root)
        cls.chosen = {
            "Median Filter": {"preset": "new_12_image", "parameters": None},
            "NLM Filter": settings("default", **NLM_VALUES),
            "Lee Filter": {"preset": "default", "parameters": None},
        }
        cls.states, cls.seeds, cls.calls = run_job_in(cls.root, matrix_job(cls.chosen))
        cls.results = cls.root / "results"
        cls.run_dir = cls.results / "result1"
        cls.rows = pd.read_csv(cls.run_dir / "experiment_results.csv")
        cls.config = json.loads((cls.run_dir / "run_config.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        st.cache_data.clear()
        cls.folder.cleanup()

    def test_every_image_gets_all_twelve_conditions(self):
        self.assertEqual(self.states[-1], ("completed", {"result": "result1"}))
        self.assertEqual(len(self.rows), 2 * 12 * 4)
        self.assertEqual(
            sorted(set(zip(self.rows["Noise_Type"], self.rows["Noise_Level"]))), sorted(CONDITIONS))
        expected_seeds = [
            42 + image * 1000 + noise * 100 + level
            for image in range(2)
            for noise, levels in enumerate(main.NOISE_LEVELS.values())
            for level in range(len(levels))
        ]
        self.assertEqual([seed for _, _, seed in self.seeds], expected_seeds)

    def test_filters_share_each_noisy_image_and_use_the_right_values(self):
        self.assertEqual(len(self.calls), 2 * 12 * 3)
        for number, (noise, level) in enumerate(CONDITIONS * 2):
            calls = self.calls[number * 3:(number + 1) * 3]
            self.assertEqual(len({digest for _, digest, _ in calls}), 1)
            median, nlm, lee = (kwargs for _, _, kwargs in calls)
            self.assertEqual(median, jobs.preset_parameters(
                "new_12_image", "Median Filter", main.filters["Median Filter"], noise, level))
            self.assertEqual(nlm, NLM_VALUES)
            expected_lee = {"window_size": 5}
            if noise == "Speckle":
                expected_lee["speckle_variance"] = main.NOISE_LEVELS[noise][level]
            self.assertEqual(lee, expected_lee)

    def test_saved_files_metadata_and_progress(self):
        pictures = sorted(p.relative_to(self.run_dir / "comparisons").as_posix()
                          for p in (self.run_dir / "comparisons").rglob("*.jpg"))
        self.assertEqual(len(pictures), 24)
        self.assertIn("speckle_high/001_test003.jpg", pictures)

        summary = helpers.load_summary("result1", results_dir=self.results)
        raw = self.rows.groupby(["Noise_Type", "Noise_Level", "Filter"])["PSNR"].mean()
        self.assertEqual(len(summary), 12 * 4)
        for _, row in summary.iterrows():
            self.assertAlmostEqual(
                row["PSNR_mean"], raw[(row["Noise_Type"], row["Noise_Level"], row["Filter"])], places=12)

        self.assertEqual(self.config["run"]["job"]["noise_scope"], "all")
        self.assertEqual(len(self.config["required_to_rerun"]["conditions"]), 12)
        source = self.config["required_to_rerun"]["parameter_source"]
        median = source["filters_by_condition"]["Gaussian / High"]["Median Filter"]
        self.assertEqual(median["values_from"], "preset for each condition")
        nlm = source["filters_by_condition"]["Speckle / Low"]["NLM Filter"]
        self.assertEqual((nlm["values_from"], nlm["parameters"]), ("same values for every condition", NLM_VALUES))

        progress = [(d["images_done"], d["images_total"]) for _, d in self.states if "images_done" in d]
        self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])

    def test_scores_match_a_single_condition_run(self):
        single = {
            "Median Filter": settings("new_12_image", **jobs.preset_parameters(
                "new_12_image", "Median Filter", main.filters["Median Filter"], "Gaussian", "High")),
            "NLM Filter": settings("default", **NLM_VALUES),
            "Lee Filter": settings("default", window_size=5),
        }
        states, _, _ = run_job_in(self.root, matrix_job(
            single, noise_scope=None, noise_type="Gaussian", noise_level="High"))
        self.assertEqual(states[-1][0], "completed")

        one = pd.read_csv(self.results / states[-1][1]["result"] / "experiment_results.csv")
        joined = one.merge(self.rows, on=["Image", "Noise_Type", "Noise_Level", "Filter"])
        self.assertEqual(len(joined), 2 * 4)
        pd.testing.assert_series_equal(joined["PSNR_x"], joined["PSNR_y"], check_names=False)
        pd.testing.assert_series_equal(joined["SSIM_x"], joined["SSIM_y"], check_names=False)

    def test_results_overview_and_condition_selector(self):
        summary = helpers.load_summary("result1", results_dir=self.results)
        filters_only = summary[summary["Filter"] != helpers.BASELINE_NAME]
        best = filters_only.loc[filters_only.groupby(["Noise_Type", "Noise_Level"])["PSNR_mean"].idxmax()]

        st.cache_data.clear()
        with use_results_folder(self.results):
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("All 12 noise conditions", " ".join(m.value for m in app.markdown))

            # The overview is in the "All 12 conditions" tab
            overview = next(t.value for t in app.dataframe if "Filtering helped?" in t.value.columns)
            self.assertEqual(len(overview), 12)
            for _, row in best.iterrows():
                match = overview[(overview["Noise"] == row["Noise_Type"]) & (overview["Strength"] == row["Noise_Level"])]
                self.assertEqual(match["Best filter"].iloc[0], helpers.short_name(row["Filter"]))

            app.selectbox(key="noise_type").set_value("Speckle")
            app.selectbox(key="noise_level").set_value("High").run()
            expected = best[(best["Noise_Type"] == "Speckle") & (best["Noise_Level"] == "High")]["Filter"].iloc[0]
            self.assertIn(helpers.short_name(expected), app.subheader[0].value)

    def test_image_browser_across_conditions(self):
        st.cache_data.clear()
        with use_results_folder(self.results), mock.patch.object(job_runner, "start_job", return_value=None) as start:
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
            app.selectbox(key="compare_image").set_value("test003.png")
            app.selectbox(key="compare_condition").set_value("Poisson / Low").run()
            self.assertEqual(len(app.exception), 0)
            start.assert_not_called()

        self.assertIn("test003.png with Low Poisson noise", " ".join(i.value for i in app.info))
        tables = [frame.value for frame in app.dataframe]
        image_table = next(t for t in tables if "Best on this image" in t.columns).set_index("Filter")
        expected = self.rows[(self.rows["Image"] == "test003.png") & (self.rows["Noise_Type"] == "Poisson")
                             & (self.rows["Noise_Level"] == "Low")].set_index("Filter")
        self.assertEqual(image_table.loc["Lee", "PSNR (dB)"], f"{expected.loc['Lee Filter', 'PSNR']:.2f}")

        across = next(t for t in tables if "Condition" in t.columns and len(t) == 12)
        self.assertIn("Best filter", across.columns)


class PicturesOffTests(unittest.TestCase):
    def test_no_pictures_but_scores_still_browsable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_project(root)
            states, _, _ = run_job_in(root, matrix_job(
                {"Median Filter": {"preset": "default", "parameters": None}}, save_pictures=False))
            self.assertEqual(states[-1][0], "completed")
            self.assertEqual(list((root / "results" / "result1").rglob("*.jpg")), [])

            st.cache_data.clear()
            with use_results_folder(root / "results"):
                app = AppTest.from_file(APP, default_timeout=30)
                app.session_state["view"] = "results"  # the app now opens on Home
                app.run()
            st.cache_data.clear()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("No comparison pictures were saved", " ".join(c.value for c in app.caption))
        self.assertEqual(list(app.selectbox(key="compare_image").options), ["test001.png", "test003.png"])


class JobRuleTests(unittest.TestCase):
    def test_all_conditions_need_a_full_dataset_job(self):
        with self.assertRaises(ValueError):
            main.prepare_job(matrix_job({"Median Filter": {"preset": "default", "parameters": None}},
                                        type="single_image", image="test001.png"))

    def test_preset_per_condition_is_only_allowed_for_all_conditions(self):
        job = matrix_job({"Median Filter": {"preset": "default", "parameters": None}},
                         noise_scope=None, noise_type="Gaussian", noise_level="Low")
        with self.assertRaises(ValueError):
            jobs.validate_job_filters(job)


@needs_saved_runs('result13', 'result16', 'result17')
class OlderResultsTests(unittest.TestCase):
    def test_saved_runs_still_open(self):
        runs = helpers.find_saved_runs()
        self.assertEqual(len(helpers.comparison_images(runs["result17"], "result17", "Gaussian", "High")), 68)
        st.cache_data.clear()
        for run in ("result17", "result16", "result13"):
            with self.subTest(run=run):
                app = AppTest.from_file(APP, default_timeout=30)
                app.session_state["view"] = "results"  # the app now opens on Home
                app.session_state["saved_run"] = run
                app.run()
                self.assertEqual(len(app.exception), 0)


class RunViewTests(unittest.TestCase):
    def test_full_experiment_choice_sends_an_all_conditions_job(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-1") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
            app.segmented_control(key="view").set_value("run").run()
            self.assertNotIn("run_noise_scope", [r.key for r in app.radio])

            app.radio(key="run_scope").set_value("Full dataset").run()
            app.radio(key="run_noise_scope").set_value("All noise conditions (full experiment)").run()
            keys = [box.key for box in app.selectbox]
            self.assertNotIn("run_noise", keys)
            self.assertNotIn("run_level", keys)

            app.selectbox(key="run_filter").set_value("Median Filter").run()
            app.selectbox(key="preset::Median Filter").set_value("new_12_image").run()
            app.selectbox(key="run_filter").set_value("NLM Filter").run()
            app.checkbox(key="own::NLM Filter").check().run()
            app.selectbox(key="param::NLM Filter::h").set_value(0.15).run()
            app.selectbox(key="run_filter").set_value("All six filters").run()
            start.assert_not_called()

            app.button(key="run_experiment").click().run()
            self.assertEqual(len(app.exception), 0)
            job = start.call_args.args[0]
            self.assertEqual((job["noise_scope"], job["noise_type"], job["noise_level"]), ("all", None, None))
            self.assertTrue(job["save_pictures"])
            self.assertEqual(job["filters"]["Median Filter"], {"preset": "new_12_image", "parameters": None})
            self.assertEqual(job["filters"]["NLM Filter"]["parameters"], NLM_VALUES)

            app.radio(key="run_noise_scope").set_value("One noise condition").run()
            self.assertIn("run_noise", [box.key for box in app.selectbox])
            self.assertEqual(start.call_count, 1)


if __name__ == "__main__":
    unittest.main()
