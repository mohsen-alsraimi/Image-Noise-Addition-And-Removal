"""
Focused tests for the Lee filter.

Run from the "Denoising ad" folder:

    python -B -m unittest discover -s automated_code_tests -v

The -B flag stops Python from rewriting the .pyc files tracked in this
repository. These tests use small synthetic images and write no files.
"""

import contextlib
import functools
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def install_skimage_stand_in():
    """
    Lets filter.py and main.py import when scikit-image is not installed.

    Only the Lee filter is tested here. NLM and TV are never called, and
    the runner tests do not check SSIM values.
    """

    def unavailable(*args, **kwargs):
        raise NotImplementedError("scikit-image is not installed")

    def psnr(original, processed, data_range=1.0):
        error = np.mean((original.astype(np.float64) - processed) ** 2)
        return float(10 * np.log10(data_range ** 2 / error))

    package = types.ModuleType("skimage")
    restoration = types.ModuleType("skimage.restoration")
    metrics = types.ModuleType("skimage.metrics")
    restoration.denoise_nl_means = unavailable
    restoration.estimate_sigma = unavailable
    restoration.denoise_tv_chambolle = unavailable
    metrics.peak_signal_noise_ratio = psnr
    metrics.structural_similarity = lambda *args, **kwargs: 0.0
    package.restoration = restoration
    package.metrics = metrics
    sys.modules.update({
        "skimage": package,
        "skimage.restoration": restoration,
        "skimage.metrics": metrics,
    })


try:
    import skimage.metrics
    import skimage.restoration
except ImportError:
    install_skimage_stand_in()

import add_noise
import main
from filter import apply_lee_filter


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def box_blur(image, window_size):
    return cv2.blur(image.astype(np.float32), (window_size, window_size))


def previous_lee_behavior(image, window_size=5):
    """Copy of the formula used before speckle_variance was added."""
    image = image.astype(np.float32)
    mean = cv2.blur(image, (window_size, window_size))
    mean_square = cv2.blur(image ** 2, (window_size, window_size))
    local_variance = mean_square - mean ** 2
    noise_variance = np.mean(local_variance)
    weight = local_variance / (local_variance + noise_variance + 1e-10)
    return np.clip(mean + weight * (image - mean), 0, 1)


def step_edge(size=64, low=0.2, high=0.8):
    image = np.full((size, size), low, dtype=np.float32)
    image[:, size // 2:] = high
    return image


def sine_texture(size=128, period=8, amplitude=0.25):
    columns = np.arange(size, dtype=np.float32)
    wave = 0.5 + amplitude * np.sin(2 * np.pi * columns / period)
    return np.tile(wave, (size, 1)).astype(np.float32)


class ConstantImageTests(unittest.TestCase):
    def test_constant_image_stays_constant(self):
        for level in (0.0, 0.25, 0.5, 1.0):
            with self.subTest(level=level):
                image = np.full((32, 32), level, dtype=np.float32)
                output = apply_lee_filter(image, 5, speckle_variance=0.04)
                np.testing.assert_allclose(output, level, atol=1e-6)


class WeightFormulaTests(unittest.TestCase):
    def test_center_pixel_matches_hand_calculation(self):
        # Window values: eight 0.4 pixels and one 0.7 pixel.
        # mean = 3.9/9, variance = 0.72/81, speckle power = 0.1521/81
        # weight = ((0.72 - 0.1521) / 1.01) / 0.72 = 0.780940
        # output = 0.433333 + 0.780940 * (0.7 - 0.433333) = 0.641584
        image = np.full((3, 3), 0.4, dtype=np.float32)
        image[1, 1] = 0.7
        output = apply_lee_filter(image, 3, speckle_variance=0.01)
        self.assertAlmostEqual(float(output[1, 1]), 0.641584, places=5)

    def test_zero_speckle_variance_keeps_image(self):
        image = add_noise.add_speckle_noise(step_edge(), 0.04, seed=3)
        output = apply_lee_filter(image, 5, speckle_variance=0.0)
        np.testing.assert_allclose(output, image, atol=1e-6)

    def test_low_variation_window_returns_local_mean(self):
        # Variance is far below the stated speckle level, so weight is 0.
        image = np.full((3, 3), 0.5, dtype=np.float32)
        image[1, 1] = 0.53
        output = apply_lee_filter(image, 3, speckle_variance=0.10)
        self.assertAlmostEqual(float(output[1, 1]), 4.53 / 9, places=5)


class SpeckleBehaviorTests(unittest.TestCase):
    def test_flat_speckle_is_strongly_reduced(self):
        clean = np.full((256, 256), 0.5, dtype=np.float32)
        for variance in (0.01, 0.04, 0.10):
            with self.subTest(variance=variance):
                noisy = add_noise.add_speckle_noise(clean, variance, seed=7)
                output = apply_lee_filter(noisy, 5, speckle_variance=variance)
                self.assertLess(mse(clean, output), 0.10 * mse(clean, noisy))

    def test_edge_stays_sharper_than_box_blur(self):
        clean = step_edge()
        noisy = add_noise.add_speckle_noise(clean, 0.01, seed=11)
        output = apply_lee_filter(noisy, 5, speckle_variance=0.01)
        band = (slice(None), slice(30, 34))
        self.assertLess(
            mse(clean[band], output[band]),
            0.5 * mse(clean[band], box_blur(noisy, 5)[band])
        )

    def test_texture_survives_better_than_box_blur(self):
        clean = sine_texture()
        noisy = add_noise.add_speckle_noise(clean, 0.01, seed=13)
        output = apply_lee_filter(noisy, 7, speckle_variance=0.01)
        self.assertLess(mse(clean, output), 0.5 * mse(clean, box_blur(noisy, 7)))
        self.assertGreater(float(output.std()), 0.7 * float(clean.std()))

    def test_larger_stated_variance_smooths_more(self):
        noisy = add_noise.add_speckle_noise(step_edge(), 0.04, seed=17)
        distance_from_mean = [
            float(np.mean(np.abs(
                apply_lee_filter(noisy, 5, speckle_variance=value)
                - box_blur(noisy, 5)
            )))
            for value in (0.01, 0.04, 0.10)
        ]
        self.assertGreater(distance_from_mean[0], distance_from_mean[1])
        self.assertGreater(distance_from_mean[1], distance_from_mean[2])


class OutputValidityTests(unittest.TestCase):
    def test_output_is_valid_for_extreme_inputs(self):
        rng = np.random.default_rng(21)
        images = {
            "random": rng.random((40, 60), dtype=np.float32),
            "binary": (rng.random((60, 40)) > 0.5).astype(np.float32),
            "float64": rng.random((33, 47)),
            "black": np.zeros((16, 16), dtype=np.float32),
            "white": np.ones((16, 16), dtype=np.float32),
        }
        for name, image in images.items():
            for window_size in (1, 3, 5, 7):
                for variance in (0.0, 0.01, 0.10, 1e6):
                    with self.subTest(name=name, w=window_size, v=variance):
                        output = apply_lee_filter(
                            image, window_size, speckle_variance=variance
                        )
                        self.assertEqual(output.shape, image.shape)
                        self.assertEqual(output.dtype, np.float32)
                        self.assertTrue(np.isfinite(output).all())
                        self.assertGreaterEqual(float(output.min()), 0.0)
                        self.assertLessEqual(float(output.max()), 1.0)

    def test_output_is_deterministic_and_input_is_unchanged(self):
        noisy = add_noise.add_speckle_noise(sine_texture(), 0.04, seed=23)
        before = noisy.copy()
        first = apply_lee_filter(noisy, 5, speckle_variance=0.04)
        second = apply_lee_filter(noisy, 5, speckle_variance=0.04)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(noisy, before)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((16, 16), 0.5, dtype=np.float32)

    def test_invalid_window_size_is_rejected(self):
        for window_size in (0, -3, 4, 5.0, True, "5", None):
            with self.subTest(window_size=window_size):
                with self.assertRaises((ValueError, TypeError)):
                    apply_lee_filter(self.image, window_size, speckle_variance=0.04)

    def test_invalid_speckle_variance_is_rejected(self):
        for variance in (-0.01, float("nan"), float("inf"), True, "0.04", [0.04]):
            with self.subTest(variance=variance):
                with self.assertRaises((ValueError, TypeError)):
                    apply_lee_filter(self.image, 5, speckle_variance=variance)

    def test_invalid_image_is_rejected(self):
        with_nan = self.image.copy()
        with_nan[0, 0] = np.nan
        with_inf = self.image.copy()
        with_inf[0, 0] = np.inf
        cases = {
            "color": np.zeros((8, 8, 3), dtype=np.float32),
            "one_dimensional": np.zeros(8, dtype=np.float32),
            "empty": np.zeros((0, 8), dtype=np.float32),
            "uint8": np.full((8, 8), 128, dtype=np.uint8),
            "above_one": np.full((8, 8), 1.5, dtype=np.float32),
            "below_zero": np.full((8, 8), -0.1, dtype=np.float32),
            "nan": with_nan,
            "inf": with_inf,
        }
        for name, image in cases.items():
            for variance in (None, 0.04):
                with self.subTest(name=name, variance=variance):
                    with self.assertRaises((ValueError, TypeError)):
                        apply_lee_filter(image, 5, speckle_variance=variance)


class OtherNoiseModelTests(unittest.TestCase):
    def test_without_variance_output_matches_previous_behavior(self):
        clean = sine_texture(64)
        noisy_images = {
            "Gaussian": add_noise.add_gaussian_noise(clean, 25, seed=31),
            "Salt-and-Pepper": add_noise.add_salt_pepper_noise(clean, 0.10, seed=32),
            "Poisson": add_noise.add_poisson_noise(clean, 30, seed=33),
        }
        for name, noisy in noisy_images.items():
            for window_size in (3, 5, 7):
                with self.subTest(name=name, window_size=window_size):
                    np.testing.assert_array_equal(
                        apply_lee_filter(noisy, window_size),
                        previous_lee_behavior(noisy, window_size)
                    )


class ExperimentRunnerTests(unittest.TestCase):
    def run_lee_only(self, parameters_file):
        calls = []
        real_lee = main.filters["Lee Filter"]

        @functools.wraps(real_lee)
        def recording_lee(image, **kwargs):
            calls.append(kwargs)
            return real_lee(image, **kwargs)

        image = sine_texture(32, period=8)

        with mock.patch.dict(main.filters, {"Lee Filter": recording_lee}, clear=True), \
                mock.patch.object(main, "TUNED_PARAMETERS_FILE", parameters_file), \
                mock.patch.object(main, "save_image_comparison"), \
                contextlib.redirect_stdout(io.StringIO()):
            results = main.run_experiments([("synthetic.png", image)])

        return calls, results

    def test_runner_gives_speckle_variance_only_to_speckle(self):
        calls, results = self.run_lee_only(None)

        # Runner order: Gaussian, Salt-and-Pepper, Speckle, Poisson.
        self.assertEqual(len(calls), 12)
        self.assertEqual(
            [call.get("speckle_variance") for call in calls],
            [None, None, None, None, None, None,
             0.01, 0.04, 0.10,
             None, None, None]
        )
        lee_rows = results[results["Filter"] == "Lee Filter"]
        self.assertEqual(len(lee_rows), 12)
        self.assertTrue(np.isfinite(lee_rows["PSNR"]).all())

    def test_runner_combines_saved_window_with_speckle_variance(self):
        parameters_file = PROJECT_DIR / "best_parameters.json"
        saved = json.loads(parameters_file.read_text(encoding="utf-8"))
        calls, _ = self.run_lee_only(parameters_file)

        expected = []
        for noise_type, levels in main.NOISE_LEVELS.items():
            for level_name, noise_value in levels.items():
                parameters = dict(saved[noise_type][level_name]["Lee Filter"])
                if noise_type == "Speckle":
                    parameters["speckle_variance"] = noise_value
                expected.append(parameters)

        self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
