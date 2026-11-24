"""
Focused tests for Stage B: the Urban100_HR and original_png datasets.

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s tests -p "test_datasets.py" -v

Real-job tests copy two images of a dataset into a temporary folder and run
only the Median filter, so they stay small.
"""

import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from test_filter_settings import settings
from test_full_dataset_job import make_job, run_job_in
from test_full_matrix import use_results_folder
from test_single_image_job import PROJECT_DIR

import app_helpers as helpers
import job_runner
import main

APP = str(PROJECT_DIR / "frontend" / "app.py")
EXPECTED_COUNTS = {"BSD68": 68, "Set12": 12, "Urban100_HR": 100, "original_png": 12}


def dataset_names(name):
    return [path.name for path in main.find_image_paths(PROJECT_DIR / "dataset" / name)]


def run_one_image(root, dataset, names, chosen):
    folder = root / "dataset" / dataset
    folder.mkdir(parents=True)
    for name in names:
        shutil.copy(PROJECT_DIR / "dataset" / dataset / name, folder / name)
    job = make_job(type="single_image", dataset=dataset, image=chosen, noise_type="Gaussian",
                   noise_level="Low", filters={"Median Filter": settings("default", kernel_size=3)})
    return run_job_in(root, job)


class DatasetFilesTests(unittest.TestCase):
    def test_four_datasets_with_expected_image_counts(self):
        self.assertEqual(main.JOB_DATASETS, ("BSD68", "Set12", "Urban100_HR", "original_png"))
        for name, count in EXPECTED_COUNTS.items():
            with self.subTest(dataset=name):
                self.assertEqual(len(dataset_names(name)), count)

    def test_tuning_still_uses_only_set12(self):
        # tune_filters.py runs tuning when imported, so only its source is read.
        source = (PROJECT_DIR / "tune_filters.py").read_text(encoding="utf-8")
        self.assertIn('DATASET_PATH = Path("dataset/Set12")', source)
        self.assertNotIn("Urban100", source)
        self.assertNotIn("original_png", source)


class DatasetJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.urban = run_one_image(cls.root, "Urban100_HR", ["img_001.png", "img_002.png"], "img_002.png")
        cls.original = run_one_image(cls.root, "original_png", dataset_names("original_png")[:2],
                                     dataset_names("original_png")[1])
        cls.results = cls.root / "results"
        cls.runs = helpers.find_saved_runs(cls.results)

    @classmethod
    def tearDownClass(cls):
        st.cache_data.clear()
        cls.folder.cleanup()

    def test_selected_dataset_and_image_are_used_with_grayscale_loading(self):
        for result, (states, seeds, _), dataset, image in (
            ("result1", self.urban, "Urban100_HR", "img_002.png"),
            ("result2", self.original, "original_png", dataset_names("original_png")[1]),
        ):
            with self.subTest(dataset=dataset):
                self.assertEqual(states[-1], ("completed", {"result": result}))
                self.assertEqual(seeds, [("Gaussian", 15, 42 + 1 * 1000)])  # image index 1, Gaussian Low

                needed = self.runs[result]["required_to_rerun"]
                self.assertEqual(needed["dataset"]["name"], dataset)
                # The test project is a temporary folder, so only the folder name is checked
                self.assertTrue(needed["dataset"]["location"].endswith(dataset))
                entry = needed["dataset"]["images"][0]
                self.assertEqual((entry["index"], entry["filename"]), (1, image))

                # Same preprocessing as before: grayscale, float32, divided by 255
                gray = cv2.imread(str(PROJECT_DIR / "dataset" / dataset / image), cv2.IMREAD_GRAYSCALE)
                expected = (gray.astype(np.float32) / 255.0)
                self.assertEqual((entry["height"], entry["width"]), gray.shape)
                self.assertEqual(entry["pixel_sha256"], hashlib.sha256(expected.tobytes()).hexdigest())

                rows = pd.read_csv(self.results / result / "experiment_results.csv")
                self.assertEqual(set(rows["Image"]), {image})
                self.assertTrue(rows["Edge_F1"].notna().all())

    def test_urban100_colour_image_becomes_grayscale(self):
        colour = cv2.imread(str(PROJECT_DIR / "dataset" / "Urban100_HR" / "img_002.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(colour.ndim, 3)
        entry = self.runs["result1"]["required_to_rerun"]["dataset"]["images"][0]
        self.assertEqual((entry["height"], entry["width"]), colour.shape[:2])

    def test_results_open_new_datasets_and_flag_original_png(self):
        st.cache_data.clear()
        with use_results_folder(self.results):
            app = AppTest.from_file(APP, default_timeout=30)
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.selectbox(key="saved_run").value, "result2")
            self.assertIn("have not been verified", " ".join(w.value for w in app.warning))

            app.selectbox(key="saved_run").set_value("result1").run()
            self.assertEqual(len(app.exception), 0)
            self.assertNotIn("have not been verified", " ".join(w.value for w in app.warning))
            self.assertIn("Urban100_HR", " ".join(m.value for m in app.markdown))


class RunViewDatasetTests(unittest.TestCase):
    def test_each_dataset_shows_its_count_names_and_notes(self):
        with mock.patch.object(job_runner, "start_job") as start:
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["view"] = "Run Experiment"
            app.run()
            self.assertEqual(list(app.selectbox(key="run_dataset").options), list(EXPECTED_COUNTS))

            for name, count in EXPECTED_COUNTS.items():
                with self.subTest(dataset=name):
                    app.selectbox(key="run_dataset").set_value(name).run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertEqual(list(app.selectbox(key="run_image").options), dataset_names(name))
                    self.assertIn(f"{name} has {count} images.", " ".join(c.value for c in app.caption))
                    warnings = " ".join(w.value for w in app.warning)
                    self.assertEqual("much slower" in warnings, name == "Urban100_HR")
                    self.assertEqual("have not been verified" in warnings, name == "original_png")

            app.radio(key="run_scope").set_value("Full dataset").run()
            self.assertIn("original_png: 12 images will be processed.", " ".join(i.value for i in app.info))
            app.selectbox(key="run_dataset").set_value("Urban100_HR").run()
            self.assertIn("Urban100_HR: 100 images will be processed.", " ".join(i.value for i in app.info))

            app.button(key="run_experiment").click().run()
            job = start.call_args.args[0]
            self.assertEqual((job["type"], job["dataset"], job["image"]), ("full_dataset", "Urban100_HR", None))


if __name__ == "__main__":
    unittest.main()
