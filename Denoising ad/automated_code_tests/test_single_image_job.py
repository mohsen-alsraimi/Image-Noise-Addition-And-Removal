"""
Tests for one-image experiments started from the frontend.

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s tests -p "test_single_image_job.py" -v

The real-job tests use real project images but save results to a temporary
folder. Page tests replace job starting with a mock, so no experiment runs.
"""

import contextlib
import functools
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "frontend"))

import app_helpers as helpers
import experiment_job as jobs
import job_runner
import main

# Lets tests skip cleanly when a locally saved run is missing
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for saved_runs
from saved_runs import needs_saved_runs


def make_job(**changes):
    job = {
        "type": "single_image",
        "dataset": "BSD68",
        "image": "test001.png",
        "noise_type": "Gaussian",
        "noise_level": "Low",
        "filters": ["Gaussian Filter"],
        "parameter_source": "default",
    }
    job.update(changes)
    return job


def run_real_job(root, job):
    """Runs the real job code and records what reached noise and filters."""
    job_dir = root / "jobs" / f"job{len(list(root.glob('jobs/*')))}"
    job_dir.mkdir(parents=True)
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    seeds, filter_calls = [], []
    real_noise = main.generate_noisy_image

    def recording_noise(image, noise_type, value, seed):
        seeds.append((noise_type, value, seed))
        return real_noise(image, noise_type, value, seed)

    def recording_filter(name, function):
        @functools.wraps(function)
        def wrapped(image, **kwargs):
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            filter_calls.append((name, digest, kwargs))
            return function(image, **kwargs)
        return wrapped

    recorded = {name: recording_filter(name, f) for name, f in main.filters.items()}

    with mock.patch.multiple(
        main,
        DATASET_PATH=main.DATASET_PATH,
        RESULTS_PATH=main.RESULTS_PATH,
        MAX_IMAGES=main.MAX_IMAGES,
        TUNED_PARAMETERS_FILE=main.TUNED_PARAMETERS_FILE,
        JOB_RESULTS_PATH=root / "results",
        generate_noisy_image=recording_noise,
    ), mock.patch.dict(main.filters, recorded, clear=True), \
            contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        try:
            main.run_job(job_path)
        except SystemExit:
            pass

    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    return status, seeds, filter_calls


class RealSingleImageJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        old_run = cls.root / "results" / "result13"
        old_run.mkdir(parents=True)
        (old_run / "keep.txt").write_text("old result", encoding="utf-8")

        cls.status, cls.seeds, cls.calls = run_real_job(cls.root, make_job(
            image="test003.png",
            noise_type="Speckle",
            noise_level="Low",
            filters=list(main.filters),
            parameter_source="previous_2_image",
        ))
        cls.run_dir = cls.root / "results" / "result14"
        cls.rows = pd.read_csv(cls.run_dir / "experiment_results.csv")
        cls.config = json.loads((cls.run_dir / "run_config.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_selected_image_and_noise_are_used(self):
        self.assertEqual(self.status["state"], "completed")
        self.assertEqual(set(self.rows["Image"]), {"test003.png"})
        self.assertEqual(set(self.rows["Noise_Type"]), {"Speckle"})
        self.assertEqual(set(self.rows["Noise_Level"]), {"Low"})
        self.assertEqual(len(self.rows), 7)

    @needs_saved_runs('result13')
    def test_seed_and_scores_match_full_dataset_run(self):
        self.assertEqual(self.seeds, [("Speckle", 0.01, 2242)])
        saved = pd.read_csv(PROJECT_DIR / "results" / "result13" / "experiment_results.csv")
        keys = ["Image", "Noise_Type", "Noise_Level", "Filter"]
        joined = self.rows.merge(saved, on=keys, suffixes=("_new", "_full"))
        self.assertEqual(len(joined), 7)
        pd.testing.assert_series_equal(joined["PSNR_new"], joined["PSNR_full"],
                                       check_names=False, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(joined["SSIM_new"], joined["SSIM_full"],
                                       check_names=False, check_exact=False, rtol=1e-12)

    def test_all_filters_receive_the_same_noisy_image(self):
        self.assertEqual([call[0] for call in self.calls], list(main.filters))
        self.assertEqual(len({call[1] for call in self.calls}), 1)

    def test_new_folder_is_complete_and_old_folder_is_safe(self):
        self.assertEqual(self.status["result"], "result14")
        self.assertEqual(self.config["run"]["status"], "completed")
        self.assertEqual(
            (self.root / "results" / "result13" / "keep.txt").read_text(encoding="utf-8"),
            "old result",
        )
        self.assertTrue((self.run_dir / "summary.csv").is_file())
        self.assertTrue((self.run_dir / "comparisons" / "speckle_low.png").is_file())
        self.assertFalse((self.run_dir / "graphs").exists())

        needed = self.config["required_to_rerun"]
        self.assertEqual(needed["dataset"]["images"][0]["index"], 2)
        self.assertEqual(needed["dataset"]["images"][0]["filename"], "test003.png")
        self.assertEqual([c["seed_offset"] for c in needed["conditions"]], [200])
        self.assertEqual(needed["parameter_source"]["source_id"], "previous_2_image")


class ChoiceTests(unittest.TestCase):
    def test_parameter_sources_send_their_own_settings(self):
        expected = {
            "default": ({}, None),
            "previous_2_image": ({"kernel_size": 5, "sigma": 1.0}, "best_parameters.json"),
            "new_12_image": (
                {"kernel_size": 3, "sigma": 0.5},
                "tuning_runs/set12_12images_corrected/best_parameters.json",
            ),
        }
        for source, (settings, file_name) in expected.items():
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                status, _, calls = run_real_job(
                    Path(folder), make_job(parameter_source=source)
                )
                self.assertEqual(status["state"], "completed")
                self.assertEqual([(c[0], c[2]) for c in calls], [("Gaussian Filter", settings)])
                config = json.loads(
                    (Path(folder) / "results" / "result1" / "run_config.json").read_text(encoding="utf-8")
                )
                saved = config["required_to_rerun"]["parameter_source"]
                self.assertEqual(saved["source_id"], source)
                self.assertEqual(saved["file"], file_name)
                self.assertEqual(saved["label"], jobs.PARAMETER_SOURCES[source]["label"])

    def test_one_filter_on_set12_keeps_image_position(self):
        with tempfile.TemporaryDirectory() as folder:
            status, seeds, calls = run_real_job(Path(folder), make_job(
                dataset="Set12",
                image="05.png",
                noise_type="Speckle",
                noise_level="High",
                filters=["Lee Filter"],
            ))
        self.assertEqual(status["state"], "completed")
        self.assertEqual(seeds, [("Speckle", 0.10, 4244)])
        self.assertEqual([(c[0], c[2]) for c in calls],
                         [("Lee Filter", {"speckle_variance": 0.10})])

    def test_unknown_image_fails_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            status, _, calls = run_real_job(Path(folder), make_job(image="missing.png"))
            found = helpers.find_saved_runs(Path(folder) / "results")
        self.assertEqual(status["state"], "failed")
        self.assertIn("missing.png", status["error"])
        self.assertEqual(calls, [])
        self.assertEqual(found, {})

    def test_second_job_cannot_start(self):
        process = mock.Mock(pid=4321)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(job_runner, "JOBS_DIR", Path(folder) / "jobs"), \
                mock.patch.object(job_runner, "is_process_running", return_value=True), \
                mock.patch.object(job_runner.subprocess, "Popen", return_value=process) as popen:
            first = job_runner.start_job(make_job())
            with self.assertRaises(job_runner.JobAlreadyRunning):
                job_runner.start_job(make_job())
            self.assertEqual(job_runner.active_job_id(), first)
            self.assertEqual(popen.call_count, 1)


class FrontendTests(unittest.TestCase):
    def test_changing_controls_never_starts_a_job(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-1") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = AppTest.from_file(str(PROJECT_DIR / "frontend" / "app.py"), default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
            app.segmented_control(key="view").set_value("run").run()
            self.assertEqual(len(app.exception), 0)

            app.selectbox(key="run_dataset").set_value("Set12").run()
            app.selectbox(key="run_image").set_value("05.png").run()
            app.selectbox(key="run_noise").set_value("Speckle").run()
            app.selectbox(key="run_level").set_value("High").run()
            app.selectbox(key="run_filter").set_value("Lee Filter").run()
            app.selectbox(key="preset::Lee Filter").set_value("new_12_image").run()
            self.assertEqual(len(app.exception), 0)
            start.assert_not_called()

            app.button(key="run_experiment").click().run()
            self.assertEqual(len(app.exception), 0)
            job = make_job(
                dataset="Set12",
                image="05.png",
                noise_type="Speckle",
                noise_level="High",
                filters={"Lee Filter": {"preset": "new_12_image",
                                        "parameters": {"window_size": 7}}},
            )
            del job["parameter_source"]
            start.assert_called_once_with(job)
            self.assertTrue(app.button(key="run_experiment").disabled)

            app.selectbox(key="run_noise").set_value("Gaussian").run()
            self.assertEqual(start.call_count, 1)

    def test_one_filter_result_opens_without_best_filter_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            status, _, _ = run_real_job(root, make_job(
                image="test002.png",
                noise_type="Speckle",
                noise_level="Medium",
                filters=["Lee Filter"],
            ))
            results = root / "results"
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
                    "run": status["result"],
                    "noise_type": "Speckle",
                    "noise_level": "Medium",
                }
                app.run()

            st.cache_data.clear()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.segmented_control(key="view").value, "results")
        self.assertEqual(app.selectbox(key="saved_run").value, status["result"])
        self.assertEqual(app.subheader[0].value, "Filter tested: Lee")
        self.assertEqual(len(app.dataframe[0].value), 1)
        page_text = " ".join(item.value for item in app.caption)  # the verdict notes this in a caption
        self.assertIn("one image", page_text)
        self.assertNotIn("Clear result", " ".join(item.value for item in app.success))


if __name__ == "__main__":
    unittest.main()
