"""
Tests for the Image Noise Lab website (frontend/app.py).

Run from the "Denoising ad" folder, using an environment that has the
packages in requirements.txt:

    python -B -m unittest discover -s tests -p "test_frontend.py" -v

The tests use Flask's test client, so no browser or server is needed.
They check that the website gives the same numbers as the project's own
code, that bad uploads get a clear error, and that the whole-dataset
mode works from start to finish.
"""

import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "frontend"))

import app as website  # noqa: E402
import main as project  # noqa: E402

FAST_FILTERS = ["Gaussian Filter", "Median Filter", "Lee Filter"]


def png_bytes(height=64, width=48, colour=False):
    """A small test photo, encoded as PNG."""
    rng = np.random.default_rng(0)
    shape = (height, width, 3) if colour else (height, width)
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


class WebsiteTestCase(unittest.TestCase):

    def setUp(self):
        website.app.config["TESTING"] = True
        self.client = website.app.test_client()

    def run_single(self, **fields):
        data = {"dataset": "BSD68", "image": "test001.png", "scope": "one", "noise": "Gaussian",
                "level": "Medium", "seed": "42", "mode": "tuned", "filters": FAST_FILTERS}
        data.update(fields)
        return self.client.post("/api/run-single", data=data, content_type="multipart/form-data")

    def upload(self, name, content):
        return self.run_single(upload=(io.BytesIO(content), name))

    def wait_for(self, job_id, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            status = self.client.get(f"/api/batch/{job_id}").get_json()
            if status["state"] != "running":
                return status
            time.sleep(0.2)
        self.fail("The experiment did not finish in time.")


class PageTests(WebsiteTestCase):

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Image Noise Lab", response.data)

    def test_config_lists_noises_filters_and_datasets(self):
        config = self.client.get("/api/config").get_json()
        self.assertEqual(config["noise_types"], list(project.NOISE_LEVELS))
        self.assertEqual(config["filters"], list(project.filters))
        self.assertEqual(len(config["datasets"]["BSD68"]), 68)
        self.assertEqual(len(config["datasets"]["Set12"]), 12)
        self.assertEqual(config["max_file_mb"], website.MAX_FILE_MB)

    def test_every_filter_setting_slider_matches_a_real_argument(self):
        import inspect
        for name, sliders in website.FILTER_PARAMS.items():
            arguments = inspect.signature(project.filters[name]).parameters
            for slider in sliders:
                self.assertIn(slider["name"], arguments, f"{name}: {slider['name']}")

    def test_dataset_image_and_unknown_dataset(self):
        response = self.client.get("/dataset/BSD68/test001.png")
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(self.client.get("/dataset/Nope/test001.png").status_code, 404)


class SinglePhotoTests(WebsiteTestCase):

    def test_scores_match_the_project_code(self):
        response = self.run_single(seed="7")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        original = website.load_gray(website.DATASETS["BSD68"] / "test001.png")
        noisy = project.generate_noisy_image(original, "Gaussian", 25, 7)
        tuned = project.load_tuned_parameters()
        self.assertAlmostEqual(data["noisy_psnr"], round(project.calculate_metrics(original, noisy)[0], 2))
        for row in data["filters"]:
            function = project.filters[row["filter"]]
            params = project.get_filter_parameters(row["filter"], function, "Gaussian", "Medium", 25, tuned)
            psnr, ssim = project.calculate_metrics(original, function(noisy.copy(), **params))
            self.assertAlmostEqual(row["psnr_exact"], psnr, places=6)
            self.assertAlmostEqual(row["ssim_exact"], ssim, places=6)

    def test_same_seed_gives_same_result_and_new_seed_changes_it(self):
        first = self.run_single(seed="5").get_json()
        again = self.run_single(seed="5").get_json()
        other = self.run_single(seed="6").get_json()
        self.assertEqual(first["noisy"], again["noisy"])
        self.assertNotEqual(first["noisy"], other["noisy"])

    def test_custom_settings_are_used(self):
        data = self.run_single(mode="custom", filters=["Median Filter"],
                               **{"Median Filter::kernel_size": "7"}).get_json()
        self.assertEqual(data["filters"][0]["params"], {"kernel_size": 7})

    def test_custom_noise_strength(self):
        data = self.run_single(noise="Salt-and-Pepper", level="Custom", custom_value="0.3").get_json()
        self.assertEqual(data["value"], 0.3)

    def test_all_noises_and_levels(self):
        data = self.run_single(scope="all").get_json()
        self.assertEqual(data["scope"], "all")
        self.assertEqual(len(data["table"]), 4 * 3)
        self.assertEqual({row["noise"] for row in data["table"]}, set(project.NOISE_LEVELS))
        for row in data["table"]:
            self.assertEqual([f["filter"] for f in row["filters"]], FAST_FILTERS)

    def test_unknown_dataset_photo_is_refused(self):
        response = self.run_single(image="not_a_photo.png")
        self.assertEqual(response.status_code, 400)


class UploadTests(WebsiteTestCase):

    def test_good_colour_photo_is_turned_into_grayscale(self):
        response = self.upload("holiday.png", png_bytes(colour=True))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["size"], "48 × 64")

    def test_big_photo_is_shrunk(self):
        data = self.upload("big.png", png_bytes(height=1500, width=900)).get_json()
        self.assertEqual(data["size"], "614 × 1024")

    def test_wrong_file_type_is_refused(self):
        response = self.upload("notes.txt", b"hello")
        self.assertEqual(response.status_code, 400)
        self.assertIn("PNG, JPG or BMP", response.get_json()["error"])

    def test_renamed_text_file_is_refused(self):
        response = self.upload("fake.png", b"this is not a picture at all")
        self.assertEqual(response.status_code, 400)
        self.assertIn("not really", response.get_json()["error"])

    def test_empty_file_is_refused(self):
        response = self.upload("empty.png", b"")
        self.assertEqual(response.status_code, 400)

    def test_damaged_photo_is_refused(self):
        broken = png_bytes()[:40]  # a real PNG start, but the rest is missing
        response = self.upload("broken.png", broken)
        self.assertEqual(response.status_code, 400)
        self.assertIn("damaged", response.get_json()["error"])

    def test_tiny_photo_is_refused(self):
        response = self.upload("tiny.png", png_bytes(height=8, width=8))
        self.assertEqual(response.status_code, 400)
        self.assertIn("too small", response.get_json()["error"])

    def test_file_over_the_size_limit_is_refused(self):
        too_big = b"\x89PNG\r\n\x1a\n" + bytes(website.MAX_FILE_MB * 1024 * 1024 + 100_000)
        response = self.upload("huge.png", too_big)
        self.assertEqual(response.status_code, 413)
        self.assertIn("too big", response.get_json()["error"])


class WholeDatasetTests(WebsiteTestCase):

    def test_batch_runs_and_matches_main_py_seeds(self):
        body = {"dataset": "Set12", "count": 2, "noises": ["Speckle"], "levels": ["Low", "High"],
                "filters": ["Lee Filter"], "mode": "tuned"}
        job = self.client.post("/api/batch", json=body).get_json()["job"]
        status = self.wait_for(job)
        self.assertEqual(status["state"], "finished")
        self.assertEqual(status["done"], 2 * 1 * 2)
        rows = {(r["level"], r["filter"]): r for r in status["summary"]}
        self.assertEqual(set(rows), {("Low", "Noisy (Baseline)"), ("Low", "Lee Filter"),
                                     ("High", "Noisy (Baseline)"), ("High", "Lee Filter")})

        # Recalculate the Lee filter, High level, with main.py's own seed rule.
        tuned = project.load_tuned_parameters()
        names = website.list_images("Set12")[:2]
        scores = []
        for index, name in enumerate(names):
            original = website.load_gray(website.DATASETS["Set12"] / name)
            seed = project.noise_seed(index, website.NOISE_TYPES.index("Speckle"), 2)
            noisy = project.generate_noisy_image(original, "Speckle", 0.10, seed)
            params = project.get_filter_parameters("Lee Filter", project.filters["Lee Filter"], "Speckle", "High", 0.10, tuned)
            scores.append(project.calculate_metrics(original, project.filters["Lee Filter"](noisy, **params))[0])
        self.assertAlmostEqual(rows[("High", "Lee Filter")]["psnr"], round(float(np.mean(scores)), 2), places=2)

        csv = self.client.get(f"/api/batch/{job}/csv")
        self.assertEqual(csv.status_code, 200)
        self.assertEqual(len(csv.data.decode().strip().splitlines()), 1 + 2 * 2 * 2)

    def test_batch_can_be_stopped(self):
        body = {"dataset": "BSD68", "count": "all", "noises": ["Gaussian"], "levels": ["High"],
                "filters": ["NLM Filter"], "mode": "tuned"}
        job = self.client.post("/api/batch", json=body).get_json()["job"]
        self.client.post(f"/api/batch/{job}/cancel")
        self.assertEqual(self.wait_for(job)["state"], "cancelled")

    def test_bad_batch_requests_are_refused(self):
        self.assertEqual(self.client.post("/api/batch", json={"dataset": "Nope"}).status_code, 400)
        self.assertEqual(self.client.post("/api/batch", json={"filters": []}).status_code, 400)
        self.assertEqual(self.client.get("/api/batch/unknown").status_code, 404)


class SavedResultsTests(WebsiteTestCase):

    def test_unknown_saved_run(self):
        self.assertEqual(self.client.get("/api/saved/result999999").status_code, 404)
        self.assertEqual(self.client.get("/api/saved/secret").status_code, 404)

    def test_saved_run_summary(self):
        config = self.client.get("/api/config").get_json()
        if not config["results"]:
            self.skipTest("No saved results in this copy of the project.")
        name = config["results"][-1]
        data = self.client.get(f"/api/saved/{name}").get_json()
        self.assertGreater(data["images"], 0)
        self.assertTrue(data["summary"])


class AccuracyTests(WebsiteTestCase):

    def test_mse_matches_the_formula_and_psnr(self):
        data = self.run_single(seed="3").get_json()
        original = website.load_gray(website.DATASETS["BSD68"] / "test001.png")
        noisy = project.generate_noisy_image(original, "Gaussian", 25, 3)
        mse = np.mean((original.astype(np.float64) - noisy.astype(np.float64)) ** 2)
        self.assertAlmostEqual(data["noisy_mse"], round(mse, 6), places=6)
        # PSNR (with a peak of 1) must agree with MSE: PSNR = 10 * log10(1 / MSE)
        self.assertAlmostEqual(data["noisy_psnr"], round(10 * np.log10(1 / mse), 2), places=1)

    def test_identical_images_give_infinite_psnr_without_breaking(self):
        image = website.load_gray(website.DATASETS["Set12"] / "01.png")
        result = website.score(image, image.copy())
        self.assertTrue(result["identical"])
        self.assertIsNone(result["psnr"])
        self.assertEqual(result["mse"], 0.0)
        self.assertAlmostEqual(result["ssim"], 1.0, places=3)
        json.dumps(result)  # must be valid JSON (no "Infinity")

    def test_noise_statistics_are_measured(self):
        data = self.run_single(noise="Salt-and-Pepper", level="Medium").get_json()
        # about 10% of the pixels are turned black or white
        self.assertAlmostEqual(data["stats"]["changed_share"], 0.10, delta=0.02)
        gaussian = self.run_single(noise="Gaussian", level="Medium").get_json()
        self.assertAlmostEqual(gaussian["stats"]["measured_sigma"], 25, delta=4)  # a little less because of clipping

    def test_fingerprints_are_stable_and_change_with_the_seed(self):
        a = self.run_single(seed="9").get_json()["run_info"]
        b = self.run_single(seed="9").get_json()["run_info"]
        c = self.run_single(seed="10").get_json()["run_info"]
        self.assertEqual(a["noisy_fingerprint"], b["noisy_fingerprint"])
        self.assertNotEqual(a["noisy_fingerprint"], c["noisy_fingerprint"])
        self.assertEqual(a["image_fingerprint"], c["image_fingerprint"])


class InputCheckTests(WebsiteTestCase):

    def test_even_window_size_is_refused(self):
        response = self.run_single(mode="custom", filters=["Median Filter"], **{"Median Filter::kernel_size": "4"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("odd", response.get_json()["error"])

    def test_setting_outside_its_range_is_refused(self):
        response = self.run_single(mode="custom", filters=["TV Filter"], **{"TV Filter::weight": "9"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("between", response.get_json()["error"])

    def test_setting_that_is_not_a_number_is_refused(self):
        response = self.run_single(mode="custom", filters=["Gaussian Filter"], **{"Gaussian Filter::sigma": "abc"})
        self.assertEqual(response.status_code, 400)

    def test_custom_noise_strength_outside_range_is_refused(self):
        response = self.run_single(noise="Gaussian", level="Custom", custom_value="500")
        self.assertEqual(response.status_code, 400)

    def test_bad_seed_and_unknown_noise_are_refused(self):
        self.assertEqual(self.run_single(seed="-1").status_code, 400)
        self.assertEqual(self.run_single(seed="abc").status_code, 400)
        self.assertEqual(self.run_single(noise="Rain").status_code, 400)


class ReproducibilityTests(WebsiteTestCase):

    def test_run_info_holds_everything_needed_to_repeat(self):
        info = self.run_single(seed="11", mode="default").get_json()["run_info"]
        for key in ("image", "image_fingerprint", "seed", "noise", "level", "noise_value", "settings_mode",
                    "filters", "versions", "noisy_fingerprint"):
            self.assertIn(key, info)
        self.assertEqual(info["seed"], 11)
        self.assertEqual(set(info["filters"]), set(FAST_FILTERS))
        self.assertIn("numpy", info["versions"])

    def test_repeating_the_run_from_its_info_gives_the_same_pixels(self):
        first = self.run_single(seed="21", noise="Speckle", level="High").get_json()
        info = first["run_info"]
        again = self.run_single(seed=str(info["seed"]), noise=info["noise"], level=info["level"]).get_json()
        self.assertEqual(again["run_info"]["noisy_fingerprint"], info["noisy_fingerprint"])
        for a, b in zip(first["filters"], again["filters"]):
            self.assertEqual(a["fingerprint"], b["fingerprint"])
            self.assertEqual(a["psnr"], b["psnr"])


class SweepTests(WebsiteTestCase):

    def sweep(self, **fields):
        data = {"dataset": "Set12", "image": "01.png", "noise": "Salt-and-Pepper", "level": "Medium",
                "seed": "42", "filter": "Median Filter", "param": "kernel_size"}
        data.update(fields)
        return self.client.post("/api/sweep", data=data, content_type="multipart/form-data")

    def test_sweep_tries_every_value_and_picks_the_best(self):
        data = self.sweep().get_json()
        self.assertEqual([r["value"] for r in data["rows"]], [3, 5, 7, 9, 11])
        best = max(data["rows"], key=lambda r: r["psnr"])
        self.assertEqual(data["best"]["value"], best["value"])

    def test_sweep_matches_a_single_run_with_the_same_value(self):
        data = self.sweep(seed="5").get_json()
        row = next(r for r in data["rows"] if r["value"] == 5)
        single = self.run_single(dataset="Set12", image="01.png", noise="Salt-and-Pepper", level="Medium", seed="5",
                                 mode="custom", filters=["Median Filter"], **{"Median Filter::kernel_size": "5"}).get_json()
        self.assertEqual(row["psnr"], single["filters"][0]["psnr"])

    def test_long_ranges_are_limited(self):
        data = self.sweep(filter="Bilateral Filter", param="sigma_space", noise="Gaussian").get_json()
        self.assertLessEqual(len(data["rows"]), website.MAX_SWEEP_VALUES)

    def test_bad_sweep_requests_are_refused(self):
        self.assertEqual(self.sweep(filter="Nope").status_code, 400)
        self.assertEqual(self.sweep(param="nope").status_code, 400)


class RepeatAndSaveTests(WebsiteTestCase):

    def test_repeats_add_new_seeds_and_a_spread(self):
        body = {"dataset": "Set12", "count": 2, "noises": ["Gaussian"], "levels": ["Medium"],
                "filters": ["Gaussian Filter"], "mode": "tuned", "repeats": 3}
        job = self.client.post("/api/batch", json=body).get_json()["job"]
        status = self.wait_for(job)
        self.assertEqual(status["done"], 2 * 3)
        row = next(r for r in status["summary"] if r["filter"] == "Gaussian Filter")
        self.assertEqual(row["runs"], 6)
        self.assertGreater(row["psnr_std"], 0)
        csv = self.client.get(f"/api/batch/{job}/csv").data.decode().strip().splitlines()
        seeds = {line.split(",")[4] for line in csv[1:]}
        self.assertEqual(len(seeds), 2 * 3)  # every image and repeat has its own seed
        config = json.loads(self.client.get(f"/api/batch/{job}/config").data)
        self.assertEqual(config["repeats"], 3)

    def test_too_many_repeats_are_refused(self):
        body = {"dataset": "Set12", "count": 1, "repeats": 99}
        self.assertEqual(self.client.post("/api/batch", json=body).status_code, 400)

    def test_saved_run_appears_under_saved_results(self):
        original = website.RESULTS_DIR
        with tempfile.TemporaryDirectory() as folder:
            website.RESULTS_DIR = Path(folder)
            try:
                body = {"dataset": "Set12", "count": 1, "noises": ["Speckle"], "levels": ["Low"],
                        "filters": ["Lee Filter"], "mode": "tuned", "save": True}
                job = self.client.post("/api/batch", json=body).get_json()["job"]
                status = self.wait_for(job)
                self.assertEqual(status["saved_to"], "result1")
                saved = Path(folder) / "result1"
                for name in ("experiment_results.csv", "summary.csv", "run_config.json"):
                    self.assertTrue((saved / name).exists(), name)
                self.assertIn("result1", self.client.get("/api/config").get_json()["results"])
                opened = self.client.get("/api/saved/result1").get_json()
                self.assertEqual(opened["images"], 1)
                self.assertTrue(opened["has_config"])
            finally:
                website.RESULTS_DIR = original

    def test_old_finished_jobs_are_forgotten(self):
        for index in range(website.MAX_JOBS_KEPT + 5):
            website.JOBS[f"old{index}"] = {"state": "finished", "started": index}
        website.forget_old_jobs()
        self.assertLess(len(website.JOBS), website.MAX_JOBS_KEPT)


if __name__ == "__main__":
    unittest.main()
