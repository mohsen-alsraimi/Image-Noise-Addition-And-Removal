"""
Tests for the edge-preservation scores (Edge Precision, Recall, F1).

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s automated_code_tests -p "test_edge_metrics.py" -v

Real-job tests use one or two real BSD68 images in a temporary folder.
"""

import functools
import sys
import tempfile
import time
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
from test_full_dataset_job import make_job, make_project, run_job_in
from test_full_matrix import matrix_job, use_results_folder
from test_single_image_job import PROJECT_DIR

import app_helpers as helpers
import edge_metrics
import main
from saved_runs import needs_saved_runs

APP = str(PROJECT_DIR / "frontend" / "app.py")


def teammate_edge_metrics(reference_image, processed_image):
    """The teammate's original code (branch main.py), kept here for comparison."""
    reference = (np.clip(reference_image, 0, 1) * 255).astype(np.uint8)
    processed = (np.clip(processed_image, 0, 1) * 255).astype(np.uint8)
    reference_edges = cv2.Canny(reference, 50, 150) > 0
    processed_edges = cv2.Canny(processed, 50, 150) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    reference_dilated = cv2.dilate(reference_edges.astype(np.uint8), kernel) > 0
    processed_dilated = cv2.dilate(processed_edges.astype(np.uint8), kernel) > 0
    reference_count = np.count_nonzero(reference_edges)
    processed_count = np.count_nonzero(processed_edges)
    if reference_count == 0 and processed_count == 0:
        return 1.0, 1.0, 1.0
    precision = np.count_nonzero(processed_edges & reference_dilated) / processed_count if processed_count else 0.0
    recall = np.count_nonzero(reference_edges & processed_dilated) / reference_count if reference_count else 0.0
    f1_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def square(top=20, left=20, size=20, shape=(64, 64)):
    image = np.zeros(shape, np.float32)
    image[top:top + size, left:left + size] = 1.0
    return image


def load(name):
    return cv2.imread(str(PROJECT_DIR / "dataset" / "BSD68" / name), 0).astype(np.float32) / 255.0


class KnownImageTests(unittest.TestCase):
    def test_same_image_keeps_every_edge(self):
        self.assertEqual(edge_metrics.calculate_edge_metrics(square(), square()), (1.0, 1.0, 1.0))

    def test_two_flat_images_have_no_edges_to_lose(self):
        flat = np.full((32, 32), 0.5, np.float32)
        self.assertEqual(edge_metrics.calculate_edge_metrics(flat, flat), (1.0, 1.0, 1.0))

    def test_all_edges_lost(self):
        flat = np.zeros((64, 64), np.float32)
        self.assertEqual(edge_metrics.calculate_edge_metrics(square(), flat), (0.0, 0.0, 0.0))

    def test_one_pixel_shift_is_tolerated_but_three_pixels_is_not(self):
        self.assertEqual(edge_metrics.calculate_edge_metrics(square(), square(left=21)), (1.0, 1.0, 1.0))
        precision, recall, f1_score = edge_metrics.calculate_edge_metrics(square(), square(left=23))
        self.assertLess(f1_score, 1.0)
        self.assertLess(precision, 1.0)
        self.assertLess(recall, 1.0)

    def test_fake_edges_lower_precision_but_not_recall(self):
        reference = square()
        processed = square()
        processed[48:58, 48:58] = 1.0  # a second shape that is not in the clean image
        precision, recall, f1_score = edge_metrics.calculate_edge_metrics(reference, processed)

        real = edge_metrics.find_edges(reference)
        found = edge_metrics.find_edges(processed)
        expected_precision = np.count_nonzero(found & cv2.dilate(real.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) / np.count_nonzero(found)
        self.assertEqual(recall, 1.0)
        self.assertAlmostEqual(precision, expected_precision)
        self.assertLess(precision, 1.0)
        self.assertAlmostEqual(f1_score, 2 * precision / (precision + 1))

    def test_rounding_instead_of_cutting_off(self):
        # 0.9999 * 255 = 254.97: the old conversion gave 254, rounding gives 255
        self.assertEqual(int(edge_metrics.to_uint8(np.float32([0.9999]))[0]), 255)
        self.assertEqual(int(edge_metrics.to_uint8(np.float32([0.998]))[0]), 254)

    def test_same_as_teammate_code_on_whole_number_images(self):
        # Dataset images and salt-and-pepper images hold whole numbers / 255,
        # where cutting off and rounding agree, so the scores must be identical.
        for name in ("test001.png", "test010.png"):
            clean = load(name)
            noisy = main.generate_noisy_image(clean, "Salt-and-Pepper", 0.10, 7)
            blurred = np.round(cv2.GaussianBlur(clean, (5, 5), 1.0) * 255) / 255
            for other in (clean, noisy, blurred.astype(np.float32)):
                self.assertEqual(edge_metrics.calculate_edge_metrics(clean, other),
                                 tuple(float(v) for v in teammate_edge_metrics(clean, other)))


class PipelineTests(unittest.TestCase):
    """One real single-image job: edge scores for the baseline and every filter."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        make_project(cls.root, images=("test001.png", "test003.png"))
        cls.outputs = {}

        def recording(name, function):
            @functools.wraps(function)
            def wrapped(image, **kwargs):
                output = function(image, **kwargs)
                cls.outputs[name] = (image.copy(), output.copy())
                return output
            return wrapped

        recorded = {name: recording(name, f) for name, f in main.filters.items()}
        job = make_job(
            type="single_image", image="test003.png", noise_type="Gaussian", noise_level="Medium",
            filters={
                "Median Filter": settings("default", kernel_size=3),
                "Gaussian Filter": settings("default", kernel_size=5, sigma=1.0),
            },
        )
        with mock.patch.dict(main.filters, recorded, clear=True):
            cls.states, cls.seeds, _ = run_job_in(cls.root, job)

        cls.results = cls.root / "results"
        cls.rows = pd.read_csv(cls.results / "result1" / "experiment_results.csv")
        cls.summary = helpers.load_summary("result1", results_dir=cls.results)
        cls.config = helpers.find_saved_runs(cls.results)["result1"]

    @classmethod
    def tearDownClass(cls):
        st.cache_data.clear()
        cls.folder.cleanup()

    def test_rows_have_edge_scores_for_baseline_and_filters(self):
        self.assertEqual(self.states[-1][0], "completed")
        self.assertEqual(list(self.rows.columns)[:7], [
            "Image", "Noise_Type", "Noise_Level", "Filter", "PSNR", "SSIM", "Execution_Time"])
        self.assertTrue({"Edge_Precision", "Edge_Recall", "Edge_F1"} <= set(self.rows.columns))

        clean = load("test003.png")
        rows = self.rows.set_index("Filter")
        noisy = self.outputs["Median Filter"][0]
        # test003 is image 1 of this two-image dataset; Gaussian is noise 0, Medium is level 1
        self.assertEqual(self.seeds, [("Gaussian", 25, 42 + 1 * 1000 + 0 * 100 + 1)])

        expected = {"Noisy (Baseline)": noisy}
        expected.update({name: output for name, (_, output) in self.outputs.items()})
        for name, image in expected.items():
            with self.subTest(filter=name):
                precision, recall, f1_score = edge_metrics.calculate_edge_metrics(clean, image)
                self.assertEqual(rows.loc[name, "Edge_Precision"], precision)
                self.assertEqual(rows.loc[name, "Edge_Recall"], recall)
                self.assertEqual(rows.loc[name, "Edge_F1"], f1_score)
                # PSNR and SSIM are still the unchanged scikit-image values
                psnr, ssim = main.calculate_metrics(clean, image)
                self.assertEqual(rows.loc[name, "PSNR"], psnr)
                self.assertEqual(rows.loc[name, "SSIM"], ssim)

    def test_summary_and_metadata_record_edge_scores(self):
        for column in ("Edge_Precision_mean", "Edge_Recall_mean", "Edge_F1_mean", "Edge_F1_std"):
            self.assertIn(column, self.summary.columns)
        evaluation = self.config["useful_for_interpretation"]["evaluation"]
        self.assertIn("Edge_F1", evaluation["metrics"])
        edges = evaluation["edge_preservation"]
        self.assertEqual((edges["canny_low_threshold"], edges["canny_high_threshold"],
                          edges["match_tolerance_pixels"]), (50, 150, 1))
        self.assertIn("edge_metrics.py", self.config["required_to_rerun"]["code"]["code_fingerprints"])

    def test_results_shows_edge_f1_and_details(self):
        st.cache_data.clear()
        with use_results_folder(self.results):
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
        self.assertEqual(len(app.exception), 0)

        _, filters = helpers.get_condition(self.summary, "Gaussian", "Medium")
        # Edge scores are in the Details tab, not among the headline numbers
        details = next(t.value for t in app.dataframe if "Edge Recall" in t.value.columns)
        self.assertEqual(list(details["Filter"])[-1], "Noisy (no filter)")
        by_filter = details.set_index("Filter")
        for _, row in filters.iterrows():
            self.assertEqual(by_filter.loc[helpers.short_name(row["Filter"]), "Edge F1"], f"{row['Edge_F1_mean']:.3f}")
        self.assertEqual([metric.label for metric in app.metric][:2], ["PSNR", "SSIM"])
        self.assertNotIn("Edge F1", [metric.label for metric in app.metric])
        self.assertIn("strong noise creates edges almost everywhere", " ".join(w.value for w in app.warning))
        image_table = next(t.value for t in app.dataframe if "Best on this image" in t.value.columns)
        self.assertEqual(image_table.set_index("Filter").loc["Noisy (no filter)", "Edge F1"],
                         f"{self.rows.set_index('Filter').loc['Noisy (Baseline)', 'Edge_F1']:.3f}")

    def test_edge_scores_are_outside_the_timer(self):
        real = main.calculate_edge_metrics

        def slow(reference, processed):
            time.sleep(0.3)
            return real(reference, processed)

        root = Path(self.folder.name) / "timer"
        make_project(root, images=("test003.png",))
        job = make_job(type="single_image", image="test003.png", noise_type="Gaussian",
                       noise_level="Low", filters={"Median Filter": settings("default", kernel_size=3)})
        with mock.patch.object(main, "calculate_edge_metrics", slow):
            states, _, _ = run_job_in(root, job)
        self.assertEqual(states[-1][0], "completed")
        rows = pd.read_csv(root / "results" / "result1" / "experiment_results.csv")
        self.assertLess(rows.loc[rows["Filter"] == "Median Filter", "Execution_Time"].iloc[0], 0.1)


class OverviewMetricTests(unittest.TestCase):
    """A full noise matrix on one image, ranked by PSNR, SSIM or Edge F1."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        make_project(cls.root, images=("test001.png",))
        cls.states, _, _ = run_job_in(cls.root, matrix_job({
            "Median Filter": settings("default", kernel_size=3),
            "Gaussian Filter": settings("default", kernel_size=5, sigma=1.0),
        }, save_pictures=False))
        cls.results = cls.root / "results"
        cls.summary = helpers.load_summary("result1", results_dir=cls.results)

    @classmethod
    def tearDownClass(cls):
        st.cache_data.clear()
        cls.folder.cleanup()

    def test_overview_ranks_by_the_chosen_metric(self):
        self.assertEqual(self.states[-1][0], "completed")
        st.cache_data.clear()
        with use_results_folder(self.results):
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
            self.assertEqual(len(app.exception), 0)
            chooser = app.radio(key="overview_metric")
            self.assertEqual(chooser.value, "PSNR")
            self.assertEqual(list(chooser.options), ["PSNR", "SSIM", "Edge F1"])
            self.assertIn("Winner decided by: PSNR", " ".join(m.value for m in app.markdown))

            for metric, column in (("Edge F1", "Edge_F1_mean"), ("SSIM", "SSIM_mean"), ("PSNR", "PSNR_mean")):
                chooser = app.radio(key="overview_metric").set_value(metric)
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertIn(f"Winner decided by: {metric}", " ".join(m.value for m in app.markdown))
                overview = next(t.value for t in app.dataframe if "Filtering helped?" in t.value.columns)
                filters_only = self.summary[~self.summary["Filter"].isin(helpers.BASELINE_NAMES)]
                best = filters_only.loc[filters_only.groupby(["Noise_Type", "Noise_Level"])[column].idxmax()]
                for _, row in best.iterrows():
                    match = overview[(overview["Noise"] == row["Noise_Type"]) & (overview["Strength"] == row["Noise_Level"])]
                    self.assertEqual(match["Best filter"].iloc[0], helpers.short_name(row["Filter"]), metric)

                    baseline = self.summary[(self.summary["Noise_Type"] == row["Noise_Type"])
                                            & (self.summary["Noise_Level"] == row["Noise_Level"])
                                            & (self.summary["Filter"] == "Noisy (Baseline)")][column].iloc[0]
                    self.assertEqual(match["Filtering helped?"].iloc[0], "Yes" if row[column] > baseline else "No")


class OlderResultsTests(unittest.TestCase):
    @needs_saved_runs('result13')
    def test_older_full_run_shows_not_measured(self):
        # result13: an older 68-image run over all 12 noise conditions, without edge scores
        st.cache_data.clear()
        app = AppTest.from_file(APP, default_timeout=60)
        app.session_state["view"] = "results"  # the app now opens on Home
        app.session_state["open_result"] = {"run": "result13", "noise_type": "Gaussian", "noise_level": "High"}
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(list(app.radio(key="overview_metric").options), ["PSNR", "SSIM"])
        self.assertIn("Edge F1 was not measured", " ".join(c.value for c in app.caption))
        details = next(t.value for t in app.dataframe if "Edge Recall" in t.value.columns)
        self.assertEqual(set(details["Edge F1"]), {helpers.NOT_MEASURED})
        self.assertIn("Edge scores were not measured", " ".join(c.value for c in app.caption))

    def test_filtering_helped_compares_with_the_unfiltered_image(self):
        baseline = pd.Series({"PSNR_mean": 30.0})
        self.assertFalse(helpers.filtering_helped(baseline, pd.DataFrame({"PSNR_mean": [29.0, 25.0]})))
        self.assertFalse(helpers.filtering_helped(baseline, pd.DataFrame({"PSNR_mean": [30.0]})))
        self.assertTrue(helpers.filtering_helped(baseline, pd.DataFrame({"PSNR_mean": [30.5, 25.0]})))


if __name__ == "__main__":
    unittest.main()
