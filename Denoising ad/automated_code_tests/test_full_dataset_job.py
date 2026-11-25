"""
Tests for full-dataset experiments: running every image in a dataset.

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s automated_code_tests -p "test_full_dataset_job.py" -v

Real-job tests use a temporary dataset with two real BSD68 images and save
to a temporary folder. Page tests replace job starting with a mock.
"""

import contextlib
import functools
import hashlib
import io
import json
import shutil
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
from test_single_image_job import PROJECT_DIR

import app_helpers as helpers
import experiment_job as jobs
import job_runner
import main

CHOSEN = {
    "Median Filter": settings("default", kernel_size=3),
    "NLM Filter": settings("new_12_image", patch_size=5, patch_distance=6, h=0.15),
    "Lee Filter": settings("default", window_size=7),
}


def make_job(**changes):
    job = {
        "type": "full_dataset",
        "dataset": "BSD68",
        "image": None,
        "noise_type": "Salt-and-Pepper",
        "noise_level": "Medium",
        "filters": CHOSEN,
    }
    job.update(changes)
    return job


def make_project(root, images=("test001.png", "test003.png")):
    folder = root / "dataset" / "BSD68"
    folder.mkdir(parents=True)
    for name in images:
        shutil.copy(PROJECT_DIR / "dataset" / "BSD68" / name, folder / name)


def run_job_in(root, job):
    """Runs the real job code with a temporary project folder."""
    job_dir = root / "jobs" / f"job{len(list(root.glob('jobs/*')))}"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")

    seeds, calls, states = [], [], []
    real_noise = main.generate_noisy_image
    real_mark = main.mark_job

    def recording_noise(image, noise_type, value, seed):
        seeds.append((noise_type, value, seed))
        return real_noise(image, noise_type, value, seed)

    def recording_mark(folder, state, **details):
        states.append((state, details))
        real_mark(folder, state, **details)

    def recording_filter(name, function):
        @functools.wraps(function)
        def wrapped(image, **kwargs):
            calls.append((name, hashlib.sha256(image.tobytes()).hexdigest(), kwargs))
            return function(image, **kwargs)
        return wrapped

    recorded = {name: recording_filter(name, f) for name, f in main.filters.items()}

    with mock.patch.multiple(
        main,
        PROJECT_DIR=root,
        DATASET_PATH=main.DATASET_PATH,
        RESULTS_PATH=main.RESULTS_PATH,
        MAX_IMAGES=main.MAX_IMAGES,
        TUNED_PARAMETERS_FILE=main.TUNED_PARAMETERS_FILE,
        JOB_RESULTS_PATH=root / "results",
        generate_noisy_image=recording_noise,
        mark_job=recording_mark,
    ), mock.patch.dict(main.filters, recorded, clear=True), \
            contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        try:
            main.run_job(job_dir / "job.json")
        except SystemExit:
            pass

    return states, seeds, calls


class FullDatasetRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        make_project(cls.root)
        cls.states, cls.seeds, cls.calls = run_job_in(cls.root, make_job())
        cls.run_dir = cls.root / "results" / "result1"
        cls.rows = pd.read_csv(cls.run_dir / "experiment_results.csv")
        cls.config = json.loads((cls.run_dir / "run_config.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_every_image_gets_the_selected_noise_and_filters(self):
        self.assertEqual(self.states[-1], ("completed", {"result": "result1"}))
        self.assertEqual(sorted(set(self.rows["Image"])), ["test001.png", "test003.png"])
        self.assertEqual(set(self.rows["Noise_Type"]), {"Salt-and-Pepper"})
        self.assertEqual(set(self.rows["Noise_Level"]), {"Medium"})
        self.assertEqual(sorted(set(self.rows["Filter"])),
                         ["Lee Filter", "Median Filter", "NLM Filter", "Noisy (Baseline)"])
        # Seed rule: 42 + image_index * 1000 + noise_index (1) * 100 + level_index (1)
        self.assertEqual(self.seeds, [("Salt-and-Pepper", 0.10, 143), ("Salt-and-Pepper", 0.10, 1143)])

    def test_filters_share_each_noisy_image_and_keep_their_settings(self):
        per_image = [self.calls[:3], self.calls[3:]]
        for calls in per_image:
            self.assertEqual([name for name, _, _ in calls],
                             ["Median Filter", "NLM Filter", "Lee Filter"])
            self.assertEqual(len({digest for _, digest, _ in calls}), 1)
            self.assertEqual([kwargs for _, _, kwargs in calls], [
                {"kernel_size": 3},
                {"patch_size": 5, "patch_distance": 6, "h": 0.15},
                {"window_size": 7},
            ])
        self.assertNotEqual(per_image[0][0][1], per_image[1][0][1])

    def test_progress_counts_finished_images(self):
        progress = [
            (details["images_done"], details["images_total"])
            for state, details in self.states if "images_done" in details
        ]
        self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])

    def test_summary_and_metadata_describe_the_whole_run(self):
        summary = helpers.load_summary("result1", results_dir=self.root / "results")
        raw = self.rows.groupby("Filter")[["PSNR", "SSIM", "Execution_Time"]].mean()
        for _, row in summary.iterrows():
            self.assertAlmostEqual(row["PSNR_mean"], raw.loc[row["Filter"], "PSNR"], places=12)
            self.assertAlmostEqual(row["SSIM_mean"], raw.loc[row["Filter"], "SSIM"], places=12)
            self.assertAlmostEqual(row["Execution_Time_mean"],
                                   raw.loc[row["Filter"], "Execution_Time"], places=12)

        needed = self.config["required_to_rerun"]
        self.assertEqual(self.config["run"]["status"], "completed")
        self.assertEqual(self.config["run"]["job"]["type"], "full_dataset")
        self.assertEqual(needed["dataset"]["images_processed"], 2)
        self.assertEqual([i["index"] for i in needed["dataset"]["images"]], [0, 1])
        self.assertTrue(needed["parameter_source"]["filters"]["NLM Filter"]["edited"])
        self.assertEqual(
            sorted(p.relative_to(self.run_dir / "comparisons").as_posix()
                   for p in (self.run_dir / "comparisons").rglob("*.png")),
            ["salt_and_pepper_medium/000_test001.png", "salt_and_pepper_medium/001_test003.png"],
        )

    def test_single_image_run_gives_the_same_scores(self):
        single = make_job(type="single_image", image="test003.png")
        states, seeds, _ = run_job_in(self.root, single)
        self.assertEqual(states[-1], ("completed", {"result": "result2"}))
        self.assertEqual(seeds, [("Salt-and-Pepper", 0.10, 1143)])

        one = pd.read_csv(self.root / "results" / "result2" / "experiment_results.csv")
        full = self.rows[self.rows["Image"] == "test003.png"]
        joined = one.merge(full, on=["Image", "Noise_Type", "Noise_Level", "Filter"])
        self.assertEqual(len(joined), 4)
        pd.testing.assert_series_equal(joined["PSNR_x"], joined["PSNR_y"], check_names=False)
        pd.testing.assert_series_equal(joined["SSIM_x"], joined["SSIM_y"], check_names=False)

    def test_results_view_shows_dataset_averages(self):
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
            app.session_state["open_result"] = {
                "run": "result1", "noise_type": "Salt-and-Pepper", "noise_level": "Medium"
            }
            app.run()
        st.cache_data.clear()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.subheader[0].value.startswith("Best filter by average PSNR:"))
        self.assertIn("averages across 2 images", " ".join(item.value for item in app.caption))
        table = next(t.value for t in app.dataframe if "Average time (ms)" in t.value.columns)
        self.assertEqual(len(table), 3)
        self.assertIn("Average time (ms)", table.columns)
        self.assertEqual(sorted(table["Best PSNR on"].str.endswith(" of 2")), [True] * 3)


class FailureAndDatasetTests(unittest.TestCase):
    def test_empty_dataset_fails_and_is_not_listed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "dataset" / "BSD68").mkdir(parents=True)
            states, _, calls = run_job_in(root, make_job())
            found = helpers.find_saved_runs(root / "results")
        self.assertEqual(states[-1][0], "failed")
        self.assertEqual(calls, [])
        self.assertEqual(found, {})

    def test_result_must_be_saved_as_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            for name, status in (("result1", "completed"), ("result2", "incomplete")):
                (results / name).mkdir()
                (results / name / "run_config.json").write_text(
                    json.dumps({"run": {"status": status}}), encoding="utf-8")
            self.assertTrue(job_runner.result_is_complete("result1", results))
            self.assertFalse(job_runner.result_is_complete("result2", results))
            self.assertFalse(job_runner.result_is_complete("result3", results))

    def test_full_jobs_use_every_image_of_the_real_datasets(self):
        for dataset, count, first, last in (("BSD68", 68, "test001.png", "test068.png"),
                                            ("Set12", 12, "01.png", "12.png")):
            with self.subTest(dataset=dataset), mock.patch.multiple(
                main,
                DATASET_PATH=main.DATASET_PATH,
                RESULTS_PATH=main.RESULTS_PATH,
                MAX_IMAGES=5,
                TUNED_PARAMETERS_FILE=main.TUNED_PARAMETERS_FILE,
            ), contextlib.redirect_stdout(io.StringIO()):
                selection = main.prepare_job(make_job(dataset=dataset))
                names = [name for name, _ in main.load_dataset()]
                self.assertIsNone(selection["image"])
                self.assertEqual((len(names), names[0], names[-1]), (count, first, last))


class RunViewTests(unittest.TestCase):
    def test_full_dataset_choice_hides_image_and_sends_full_job(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-1") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = AppTest.from_file(str(PROJECT_DIR / "frontend" / "app.py"), default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
            app.segmented_control(key="view").set_value("run").run()
            self.assertIn("run_image", [box.key for box in app.selectbox])

            app.radio(key="run_scope").set_value("Full dataset").run()
            self.assertNotIn("run_image", [box.key for box in app.selectbox])
            self.assertIn("BSD68: 68 images will be processed.", " ".join(i.value for i in app.info))

            app.selectbox(key="run_dataset").set_value("Set12").run()
            self.assertIn("Set12: 12 images will be processed.", " ".join(i.value for i in app.info))
            app.selectbox(key="run_noise").set_value("Speckle").run()
            start.assert_not_called()

            app.button(key="run_experiment").click().run()
            self.assertEqual(len(app.exception), 0)
            job = start.call_args.args[0]
            self.assertEqual((job["type"], job["dataset"], job["image"]), ("full_dataset", "Set12", None))

            app.selectbox(key="run_level").set_value("High").run()
            self.assertEqual(start.call_count, 1)

    def test_progress_shows_the_current_image(self):
        running = {"state": "running", "images_done": 23, "images_total": 68}
        with mock.patch.object(job_runner, "active_job_id", return_value="job-1"), \
                mock.patch.object(job_runner, "job_state", return_value=running):
            app = AppTest.from_file(str(PROJECT_DIR / "frontend" / "app.py"), default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Processing image 24 of 68", " ".join(item.value for item in app.info))

    def test_completed_job_without_saved_completion_is_shown_as_failed(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "job_state", return_value={"state": "completed", "result": "result99"}), \
                mock.patch.object(job_runner, "result_is_complete", return_value=False):
            app = AppTest.from_file(str(PROJECT_DIR / "frontend" / "app.py"), default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.session_state["watched_job"] = "job-1"
            app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Status: Failed", " ".join(item.value for item in app.error))
        self.assertEqual(app.segmented_control(key="view").value, "results")
        self.assertNotIn("open_result", app.session_state)


if __name__ == "__main__":
    unittest.main()
