"""
Tests for JPEG compression and glare experiments: functions, jobs, and pages.

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s automated_code_tests -p "test_degradation.py" -v

The real-job test runs all 6 degradation conditions on two BSD68 images with
two fast filters, in a temporary folder.
"""

import json
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

from test_filter_settings import param_key, settings
from test_full_dataset_job import make_project, run_job_in
from test_full_matrix import matrix_job, use_results_folder
from test_single_image_job import PROJECT_DIR

import app_helpers as helpers
import experiment_job as jobs
import image_degradation
import job_runner
import main
import run_view

APP = str(PROJECT_DIR / "frontend" / "app.py")
FILTERS = {
    "Median Filter": settings("default", kernel_size=3),
    "Gaussian Filter": settings("default", kernel_size=5, sigma=1.0),
}
NO_IMPROVEMENT = "No tested filter improved PSNR over the unfiltered degraded image."


def load(name):
    return cv2.imread(str(PROJECT_DIR / "dataset" / "BSD68" / name), 0).astype(np.float32) / 255.0


def teammate_add_glare(image, alpha=0.45, seed=42):
    """The teammate's original add_glare, kept here for comparison."""
    height, width = image.shape
    rng = np.random.default_rng(seed)
    center_x = rng.uniform(0.25, 0.75) * width
    center_y = rng.uniform(0.25, 0.75) * height
    sigma_x = max(width * rng.uniform(0.12, 0.22), 1.0)
    sigma_y = max(height * rng.uniform(0.12, 0.22), 1.0)
    x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    mask = np.exp(-((x_grid - center_x) ** 2 / (2 * sigma_x ** 2) + (y_grid - center_y) ** 2 / (2 * sigma_y ** 2)))
    return np.clip((1 - alpha * mask) * image + alpha * mask, 0, 1).astype(np.float32)


def teammate_add_jpeg(image, quality=60):
    """The teammate's original JPEG function (cuts off decimals)."""
    encoded = cv2.imencode(".jpg", (np.clip(image, 0, 1) * 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


class DegradationFunctionTests(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(main.DEGRADATION_LEVELS, {
            "JPEG Compression": {"Low": 90, "Medium": 60, "High": 30},
            "Glare": {"Low": 0.20, "Medium": 0.45, "High": 0.70},
        })
        # The 12 noise conditions are unchanged and separate
        self.assertEqual(sum(len(levels) for levels in main.NOISE_LEVELS.values()), 12)
        self.assertNotIn("Glare", main.NOISE_LEVELS)

    def test_jpeg_uses_the_selected_quality(self):
        image = load("test001.png")
        scores = []
        for quality in (90, 60, 30):
            output = main.generate_degraded_image(image, "JPEG Compression", quality, None)
            encoded = cv2.imencode(".jpg", np.rint(image * 255).astype(np.uint8),
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            expected = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            np.testing.assert_array_equal(output, expected)
            # Same as the teammate's function on dataset images
            np.testing.assert_array_equal(output, teammate_add_jpeg(image, quality))
            scores.append(main.calculate_metrics(image, output)[0])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_glare_function_is_the_teammates(self):
        image = load("test003.png")
        for alpha, seed in ((0.2, 542), (0.7, 1542)):
            np.testing.assert_array_equal(image_degradation.add_glare(image, alpha, seed),
                                          teammate_add_glare(image, alpha, seed))

    def test_glare_seed_is_per_image_not_per_level(self):
        self.assertEqual([main.glare_seed(index) for index in (0, 1, 67)], [542, 1542, 67542])
        # Noise seeds are unchanged
        self.assertEqual(main.noise_seed(2, 2, 0), 2242)
        self.assertEqual(main.noise_seed(1, 1, 1), 1143)

    def test_glare_spot_stays_and_only_strength_changes(self):
        image = load("test001.png")
        seed = main.glare_seed(0)
        outputs = {level: main.generate_degraded_image(image, "Glare", alpha, seed)
                   for level, alpha in main.DEGRADATION_LEVELS["Glare"].items()}
        changes = {level: output - image for level, output in outputs.items()}

        np.testing.assert_array_equal(outputs["Medium"], main.generate_degraded_image(image, "Glare", 0.45, seed))
        spots = {np.unravel_index(np.argmax(change), change.shape) for change in changes.values()}
        self.assertEqual(len(spots), 1)
        # change = alpha * mask * (1 - image), so it scales exactly with alpha
        np.testing.assert_allclose(changes["High"] * 0.20, changes["Low"] * 0.70, atol=2e-6)
        self.assertGreater(changes["High"].max(), changes["Medium"].max())
        self.assertGreater(changes["Medium"].max(), changes["Low"].max())

        # A different image index moves the spot
        other = main.generate_degraded_image(image, "Glare", 0.45, main.glare_seed(1)) - image
        self.assertNotEqual(np.unravel_index(np.argmax(other), other.shape), spots.pop())


class PresetRuleTests(unittest.TestCase):
    def job(self, preset="default", **changes):
        job = matrix_job({"Median Filter": settings(preset, kernel_size=3)}, experiment="degradation")
        job.update(changes)
        return job

    def test_tuned_presets_are_unavailable_for_degradation(self):
        jobs.validate_job_filters(self.job())
        for preset in ("previous_2_image", "new_12_image"):
            with self.subTest(preset=preset):
                with self.assertRaisesRegex(ValueError, "not available for JPEG compression or glare"):
                    jobs.validate_job_filters(self.job(preset))
                with self.assertRaisesRegex(ValueError, "no values for Glare"):
                    jobs.preset_parameters(preset, "Median Filter", main.filters["Median Filter"], "Glare", "Low")
        # Noise jobs still accept tuned presets
        noise_job = self.job("new_12_image")
        noise_job.pop("experiment")
        jobs.validate_job_filters(noise_job)

    def test_degradation_needs_per_filter_settings(self):
        with self.assertRaisesRegex(ValueError, "per-filter"):
            main.prepare_job(self.job(filters=["Median Filter"], parameter_source="default"))
        with self.assertRaisesRegex(ValueError, "Unknown degradation condition"):
            main.prepare_job(self.job(noise_scope=None, type="single_image", image="x.png",
                                      noise_type="Gaussian", noise_level="Low"))


class DegradationRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        make_project(cls.root)
        cls.calls = []
        real = main.generate_degraded_image

        def recording(image, degradation_type, value, seed):
            cls.calls.append((degradation_type, value, seed))
            return real(image, degradation_type, value, seed)

        with mock.patch.object(main, "generate_degraded_image", recording):
            cls.states, cls.noise_seeds, _ = run_job_in(
                cls.root, matrix_job(FILTERS, experiment="degradation"))

        cls.results = cls.root / "results"
        cls.run_dir = cls.results / "result1"
        cls.rows = pd.read_csv(cls.run_dir / "experiment_results.csv")
        cls.summary = helpers.load_summary("result1", results_dir=cls.results)
        cls.config = json.loads((cls.run_dir / "run_config.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        st.cache_data.clear()
        cls.folder.cleanup()

    def test_all_six_conditions_with_the_right_values_and_seeds(self):
        self.assertEqual(self.states[-1], ("completed", {"result": "result1"}))
        self.assertEqual(self.noise_seeds, [])  # no noise is added in a degradation run
        expected = []
        for image_index in (0, 1):
            expected += [("JPEG Compression", q, None) for q in (90, 60, 30)]
            expected += [("Glare", a, 42 + image_index * 1000 + 500) for a in (0.20, 0.45, 0.70)]
        self.assertEqual(self.calls, expected)

    def test_saved_rows(self):
        self.assertEqual(len(self.rows), 2 * 6 * 3)
        self.assertEqual(set(self.rows["Noise_Type"]), {"JPEG Compression", "Glare"})
        self.assertEqual(set(self.rows["Filter"]), {"Degraded (Baseline)", "Median Filter", "Gaussian Filter"})
        self.assertEqual(set(self.rows["Category"]), {"Image degradation"})
        self.assertTrue(self.rows[["Edge_Precision", "Edge_Recall", "Edge_F1"]].notna().all().all())

        image = load("test003.png")
        for level, quality in main.DEGRADATION_LEVELS["JPEG Compression"].items():
            degraded = image_degradation.add_jpeg_compression(image, quality)
            row = self.rows[(self.rows["Image"] == "test003.png") & (self.rows["Noise_Level"] == level)
                            & (self.rows["Noise_Type"] == "JPEG Compression")
                            & (self.rows["Filter"] == "Degraded (Baseline)")].iloc[0]
            psnr, ssim = main.calculate_metrics(image, degraded)
            self.assertEqual((row["PSNR"], row["SSIM"]), (psnr, ssim))
        self.assertIn("Edge_F1_mean", self.summary.columns)

    def test_metadata(self):
        needed = self.config["required_to_rerun"]
        self.assertEqual(needed["experiment"], "degradation")
        self.assertFalse(needed["noise"]["used_in_this_run"])
        self.assertEqual(len(needed["conditions"]), 6)
        glare = next(c for c in needed["conditions"] if c["noise_type"] == "Glare" and c["level"] == "High")
        jpeg = next(c for c in needed["conditions"] if c["noise_type"] == "JPEG Compression" and c["level"] == "Medium")
        self.assertEqual((glare["category"], glare["glare_strength"], glare["glare_location_seed_offset"]),
                         ("degradation", 0.70, 500))
        self.assertEqual((jpeg["degradation_type"], jpeg["jpeg_quality"], jpeg["glare_location_seed_offset"]),
                         ("JPEG Compression", 60, None))
        self.assertEqual(glare["filters"]["Median Filter"]["resolved"], {"kernel_size": 3})

        policy = needed["degradation"]["seed_policy"]
        self.assertEqual(policy["glare"]["formula"], "base_seed + image_index * 1000 + 500")
        self.assertTrue(policy["glare"]["same_for_all_levels"])
        records = needed["parameter_source"]["filters_by_condition"]["Glare / Low"]
        self.assertEqual(records["Gaussian Filter"]["preset"], "default")
        self.assertEqual(records["Gaussian Filter"]["parameters"], {"kernel_size": 5, "sigma": 1.0})
        self.assertIn("image_degradation.py", needed["code"]["code_fingerprints"])

    def test_pictures(self):
        pictures = sorted(p.relative_to(self.run_dir / "comparisons").as_posix()
                          for p in (self.run_dir / "comparisons").rglob("*.jpg"))
        self.assertEqual(len(pictures), 12)
        self.assertIn("jpeg_compression_high/001_test003.jpg", pictures)
        self.assertIn("glare_low/000_test001.jpg", pictures)

    def test_results_view_is_honest_about_the_baseline(self):
        filters_only = self.summary[self.summary["Filter"] != "Degraded (Baseline)"]
        baseline = self.summary[self.summary["Filter"] == "Degraded (Baseline)"].set_index(["Noise_Type", "Noise_Level"])
        best = filters_only.loc[filters_only.groupby(["Noise_Type", "Noise_Level"])["PSNR_mean"].idxmax()]
        helped = {(r["Noise_Type"], r["Noise_Level"]): r["PSNR_mean"] > baseline.loc[(r["Noise_Type"], r["Noise_Level"]), "PSNR_mean"]
                  for _, r in best.iterrows()}
        not_helped = [condition for condition, value in helped.items() if not value]
        self.assertTrue(not_helped, "expected at least one condition where filtering lowers PSNR")

        st.cache_data.clear()
        with use_results_folder(self.results):
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["view"] = "results"  # the app now opens on Home
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("All 6 conditions", " ".join(t.label for t in app.tabs))
            self.assertIn("Image degradation experiment", " ".join(i.value for i in app.info))
            self.assertIn("All 6 degradation conditions", " ".join(m.value for m in app.markdown))

            overview = next(t.value for t in app.dataframe if "Filtering helped?" in t.value.columns)
            self.assertIn("Degradation", overview.columns)
            self.assertIn("Degraded PSNR (dB)", overview.columns)
            for (noise, level), value in helped.items():
                row = overview[(overview["Degradation"] == noise) & (overview["Strength"] == level)]
                self.assertEqual(row["Filtering helped?"].iloc[0], "Yes" if value else "No")

            noise, level = not_helped[0]
            app.selectbox(key="noise_type").set_value(noise)
            app.selectbox(key="noise_level").set_value(level).run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn(NO_IMPROVEMENT, " ".join(e.value for e in app.error))
            self.assertTrue(app.subheader[0].value.startswith("Ranked first by Average PSNR, but no improvement"))
            image_table = next(t.value for t in app.dataframe if "Best on this image" in t.value.columns)
            self.assertEqual(list(image_table["Filter"])[-1], "Degraded (no filter)")
            self.assertIn("Glare: `base_seed + image_index * 1000 + 500`",
                          " ".join(m.value for m in app.markdown))

            helped_conditions = [condition for condition, value in helped.items() if value]
            if helped_conditions:
                noise, level = helped_conditions[0]
                app.selectbox(key="noise_type").set_value(noise)
                app.selectbox(key="noise_level").set_value(level).run()
                self.assertNotIn(NO_IMPROVEMENT, " ".join(e.value for e in app.error))


class RunViewDegradationTests(unittest.TestCase):
    def open_run_view(self):
        app = AppTest.from_file(APP, default_timeout=30)
        app.session_state["view"] = "results"  # the app now opens on Home
        app.session_state["view"] = "run"
        app.run()
        return app

    def test_choosing_image_degradation(self):
        with mock.patch.object(job_runner, "start_job", return_value=None) as start:
            app = self.open_run_view()
            self.assertEqual(list(app.radio(key="run_test_kind").options), ["Noise", "Image degradation"])
            self.assertEqual(list(app.selectbox(key="run_noise").options),
                             ["Gaussian", "Salt-and-Pepper", "Speckle", "Poisson"])

            # A tuned noise preset chosen earlier must not carry over
            app.selectbox(key="run_filter").set_value("Median Filter").run()
            app.selectbox(key="preset::Median Filter").set_value("new_12_image").run()
            app.radio(key="run_test_kind").set_value("Image degradation").run()
            self.assertEqual(len(app.exception), 0)

            self.assertEqual(list(app.selectbox(key="run_degradation").options), ["JPEG Compression", "Glare"])
            self.assertEqual(list(app.selectbox(key="run_degradation_level").options), ["Low", "Medium", "High"])
            self.assertEqual(list(app.selectbox(key="preset::Median Filter").options), ["Default settings"])
            self.assertEqual(app.selectbox(key="preset::Median Filter").value, "default")
            # The tuned noise value (kernel 3) was replaced by the filter's default
            default = jobs.preset_parameters("default", "Median Filter", main.filters["Median Filter"],
                                             "Glare", "High")
            tuned = jobs.preset_parameters("new_12_image", "Median Filter", main.filters["Median Filter"],
                                           "Gaussian", "Low")
            self.assertNotEqual(default, tuned)
            self.assertEqual(app.selectbox(key=param_key("Median Filter", "kernel_size")).value,
                             default["kernel_size"])
            self.assertIn(run_view.DEFAULT_ONLY_TEXT, [c.value for c in app.caption])

            app.selectbox(key="run_degradation").set_value("Glare")
            app.selectbox(key="run_degradation_level").set_value("High").run()
            self.assertIn("strength 0.7", " ".join(m.value for m in app.markdown))
            app.button(key="run_experiment").click().run()
            job = start.call_args.args[0]
            self.assertEqual((job["experiment"], job["type"], job["noise_type"], job["noise_level"]),
                             ("degradation", "single_image", "Glare", "High"))
            self.assertEqual(job["filters"], {"Median Filter": settings("default", **default)})
            jobs.validate_job_filters(job)

    def test_full_dataset_all_degradation_conditions(self):
        with mock.patch.object(job_runner, "start_job", return_value=None) as start:
            app = self.open_run_view()
            app.radio(key="run_scope").set_value("Full dataset")
            app.radio(key="run_test_kind").set_value("Image degradation").run()
            self.assertEqual(list(app.radio(key="run_degradation_scope").options),
                             [run_view.ONE_DEGRADATION, run_view.ALL_DEGRADATIONS])
            app.radio(key="run_degradation_scope").set_value(run_view.ALL_DEGRADATIONS).run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("All 6 degradation conditions", " ".join(i.value for i in app.info))
            app.button(key="run_experiment").click().run()
            job = start.call_args.args[0]
            self.assertEqual((job["experiment"], job["noise_scope"]), ("degradation", "all"))
            self.assertEqual({entry["preset"] for entry in job["filters"].values()}, {"default"})

            # Back to noise: "All noise conditions" still means 12
            app.radio(key="run_test_kind").set_value("Noise").run()
            self.assertEqual(list(app.radio(key="run_noise_scope").options),
                             [run_view.ONE_CONDITION, run_view.ALL_CONDITIONS])
            app.radio(key="run_noise_scope").set_value(run_view.ALL_CONDITIONS).run()
            self.assertIn("All 12 noise conditions", " ".join(i.value for i in app.info))
            app.button(key="run_experiment").click().run()
            self.assertNotIn("experiment", start.call_args.args[0])  # noise jobs keep their old format
            self.assertEqual(start.call_args.args[0]["noise_scope"], "all")


if __name__ == "__main__":
    unittest.main()
