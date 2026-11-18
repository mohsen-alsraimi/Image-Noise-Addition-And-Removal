"""Unit tests for src/noise.py (Gaussian, salt-and-pepper and speckle noise).

Most tests are exact and deterministic (fixed seeds). The few statistical
tests use large images (tens of thousands of pixels) and tolerances several
times larger than the expected random variation, so they are not fragile.

Run from the project root:  python -m unittest -v
"""

import math
import unittest

import numpy as np

from src.metrics import mse
from src.noise import (
    PEPPER_VALUE,
    SALT_VALUE,
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    add_speckle_noise,
    salt_and_pepper_counts,
)
from src.synthetic_image import create_synthetic_test_image

# (function, a typical non-zero strength) - used by the general tests.
NOISE_FUNCTIONS = [
    (add_gaussian_noise, 25),
    (add_salt_and_pepper_noise, 0.10),
    (add_speckle_noise, 0.04),
]


def flat_image(value, height=100, width=100):
    return np.full((height, width), value, dtype=np.uint8)


class TestGeneralBehaviour(unittest.TestCase):
    """Rules every noise function must follow."""

    def setUp(self):
        self.image = create_synthetic_test_image()

    def test_output_shape_dtype_and_range(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                noisy = function(self.image, strength, seed=42)
                self.assertEqual(noisy.shape, self.image.shape)
                self.assertEqual(noisy.ndim, 2)
                self.assertEqual(noisy.dtype, np.uint8)
                self.assertGreaterEqual(int(noisy.min()), 0)
                self.assertLessEqual(int(noisy.max()), 255)

    def test_noise_actually_changes_the_image(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                self.assertFalse(np.array_equal(function(self.image, strength, seed=42), self.image))

    def test_input_image_is_not_modified(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                before = self.image.copy()
                function(self.image, strength, seed=42)
                np.testing.assert_array_equal(self.image, before)

    def test_result_does_not_share_memory_with_input(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                self.assertFalse(np.shares_memory(function(self.image, strength, seed=1), self.image))
                self.assertFalse(np.shares_memory(function(self.image, 0, seed=1), self.image))

    def test_zero_strength_returns_an_equal_copy(self):
        for function, _ in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                result = function(self.image, 0, seed=42)
                np.testing.assert_array_equal(result, self.image)
                self.assertIsNot(result, self.image)

    def test_same_seed_gives_identical_results(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                np.testing.assert_array_equal(
                    function(self.image, strength, seed=42), function(self.image, strength, seed=42)
                )

    def test_numpy_integer_seed_gives_same_result_as_python_int(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                np.testing.assert_array_equal(
                    function(self.image, strength, seed=np.int64(7)), function(self.image, strength, seed=7)
                )

    def test_different_seeds_give_different_noise(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                self.assertFalse(
                    np.array_equal(function(self.image, strength, seed=1), function(self.image, strength, seed=2))
                )

    def test_seed_none_works(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                noisy = function(self.image, strength)  # unpredictable, but must be valid
                self.assertEqual(noisy.shape, self.image.shape)
                self.assertEqual(noisy.dtype, np.uint8)

    def test_global_numpy_random_state_is_not_changed(self):
        np.random.seed(123)
        state_before = np.random.get_state()
        for function, strength in NOISE_FUNCTIONS:
            function(self.image, strength, seed=42)
            function(self.image, strength)  # seed=None too
        state_after = np.random.get_state()
        self.assertEqual(state_before[0], state_after[0])
        np.testing.assert_array_equal(state_before[1], state_after[1])
        self.assertEqual(state_before[2:], state_after[2:])

    def test_accepts_int_and_numpy_float_strengths(self):
        for function, strength in NOISE_FUNCTIONS:
            with self.subTest(function=function.__name__):
                function(self.image, np.float64(strength), seed=1)
                function(self.image, 1 if function is not add_speckle_noise else 0, seed=1)

    def test_invalid_images_are_rejected(self):
        bad_images = [
            ([[0, 1], [2, 3]], TypeError),                    # list, not array
            (np.zeros((4, 4), dtype=np.float64), ValueError),  # wrong dtype
            (np.zeros((4, 4, 3), dtype=np.uint8), ValueError),  # colour (3-D)
            (np.zeros(16, dtype=np.uint8), ValueError),        # 1-D
            (np.zeros((0, 4), dtype=np.uint8), ValueError),    # empty
        ]
        for function, strength in NOISE_FUNCTIONS:
            for bad_image, error in bad_images:
                with self.subTest(function=function.__name__, image=repr(bad_image)[:30]):
                    with self.assertRaises(error):
                        function(bad_image, strength, seed=1)

    def test_invalid_strengths_are_rejected(self):
        bad_strengths = [
            (-1, ValueError),
            (-0.001, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
            (True, TypeError),
            (False, TypeError),
            (np.bool_(True), TypeError),
            ("10", TypeError),
            (None, TypeError),
            (1j, TypeError),
        ]
        for function, _ in NOISE_FUNCTIONS:
            for strength, error in bad_strengths:
                with self.subTest(function=function.__name__, strength=strength):
                    with self.assertRaises(error):
                        function(self.image, strength, seed=1)

    def test_invalid_seeds_are_rejected(self):
        bad_seeds = [(-1, ValueError), (True, TypeError), (3.5, TypeError), ("42", TypeError)]
        for function, strength in NOISE_FUNCTIONS:
            for seed, error in bad_seeds:
                with self.subTest(function=function.__name__, seed=seed):
                    with self.assertRaises(error):
                        function(self.image, strength, seed=seed)


class TestGaussianNoise(unittest.TestCase):
    def test_matches_the_documented_formula_exactly(self):
        # noisy = clip(round(original + N(0, sigma^2)), 0, 255) with the same generator
        image = create_synthetic_test_image()
        noise = np.random.default_rng(7).normal(0.0, 20, size=image.shape)
        expected = np.clip(np.rint(image.astype(np.float64) + noise), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(add_gaussian_noise(image, 20, seed=7), expected)

    def test_noise_has_mean_zero_and_requested_standard_deviation(self):
        # Flat image of 128 with sigma 10: clipping is impossible in practice
        # (it would need a value 12.8 sigma away). 160,000 samples.
        image = flat_image(128, 400, 400)
        difference = add_gaussian_noise(image, 10, seed=1).astype(np.float64) - 128
        self.assertAlmostEqual(difference.mean(), 0.0, delta=0.1)
        # Rounding to whole numbers adds a tiny extra variance (1/12), so std ~ 10.004.
        self.assertAlmostEqual(difference.std(), 10.0, delta=0.15)

    def test_mse_is_close_to_sigma_squared_without_clipping(self):
        image = flat_image(128, 400, 400)
        self.assertAlmostEqual(mse(image, add_gaussian_noise(image, 10, seed=2)), 100.0, delta=2.0)

    def test_bright_pixels_are_clipped_at_255(self):
        image = flat_image(250)
        noisy = add_gaussian_noise(image, 30, seed=0)
        self.assertEqual(int(noisy.max()), 255)
        self.assertGreater(int(np.count_nonzero(noisy == 255)), 0)
        # No wrap-around: without clipping, 250 + 20 = 270 would become 14 in uint8.
        # A real value below 70 would need noise of -6 sigma (practically impossible).
        self.assertGreater(int(noisy.min()), 70)

    def test_dark_pixels_are_clipped_at_0(self):
        image = flat_image(5)
        noisy = add_gaussian_noise(image, 30, seed=0)
        self.assertEqual(int(noisy.min()), 0)
        self.assertGreater(int(np.count_nonzero(noisy == 0)), 0)
        # No wrap-around: without clipping, 5 - 20 = -15 would become 241 in uint8.
        self.assertLess(int(noisy.max()), 185)

    def test_larger_sigma_gives_larger_error(self):
        image = create_synthetic_test_image()
        errors = [mse(image, add_gaussian_noise(image, s, seed=42)) for s in (10, 25, 50)]
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])


class TestSaltAndPepperCounts(unittest.TestCase):
    """The documented rounding rule: selected = floor(amount*N + 0.5); salt = selected // 2."""

    def test_project_values_for_256_by_256_image(self):
        self.assertEqual(salt_and_pepper_counts(65536, 0.02), (1311, 655, 656))  # 1310.72 -> 1311
        self.assertEqual(salt_and_pepper_counts(65536, 0.05), (3277, 1638, 1639))  # 3276.8 -> 3277
        self.assertEqual(salt_and_pepper_counts(65536, 0.10), (6554, 3277, 3277))  # 6553.6 -> 6554

    def test_exact_halves_round_up(self):
        self.assertEqual(salt_and_pepper_counts(10, 0.25), (3, 1, 2))  # 2.5 -> 3
        self.assertEqual(salt_and_pepper_counts(4, 0.125), (1, 0, 1))  # 0.5 -> 1

    def test_below_half_rounds_down(self):
        self.assertEqual(salt_and_pepper_counts(10, 0.24), (2, 1, 1))  # 2.4 -> 2

    def test_zero_and_one(self):
        self.assertEqual(salt_and_pepper_counts(10, 0), (0, 0, 0))
        self.assertEqual(salt_and_pepper_counts(10, 1), (10, 5, 5))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            salt_and_pepper_counts(10, 1.5)
        with self.assertRaises(ValueError):
            salt_and_pepper_counts(0, 0.1)
        with self.assertRaises(TypeError):
            salt_and_pepper_counts(10.0, 0.1)


class TestSaltAndPepperNoise(unittest.TestCase):
    def setUp(self):
        # A flat image of 128 contains no 0 or 255, so every selected pixel
        # is also a visibly changed pixel and we can count them exactly.
        self.image = flat_image(128)  # 10,000 pixels

    def test_exact_counts_follow_the_rounding_rule(self):
        noisy = add_salt_and_pepper_noise(self.image, 0.05, seed=3)
        selected, salt, pepper = salt_and_pepper_counts(self.image.size, 0.05)  # 500, 250, 250
        self.assertEqual(int(np.count_nonzero(noisy == SALT_VALUE)), salt)
        self.assertEqual(int(np.count_nonzero(noisy == PEPPER_VALUE)), pepper)
        self.assertEqual(int(np.count_nonzero(noisy != 128)), selected)

    def test_odd_selection_gives_pepper_the_extra_pixel(self):
        noisy = add_salt_and_pepper_noise(self.image, 0.0503, seed=3)  # 503 selected
        self.assertEqual(int(np.count_nonzero(noisy == SALT_VALUE)), 251)
        self.assertEqual(int(np.count_nonzero(noisy == PEPPER_VALUE)), 252)

    def test_no_duplicate_positions(self):
        # If any position were chosen twice, fewer than `selected` pixels
        # would change. Exactly `selected` changed pixels proves uniqueness.
        for amount in (0.02, 0.3, 0.9):
            with self.subTest(amount=amount):
                noisy = add_salt_and_pepper_noise(self.image, amount, seed=11)
                selected, _, _ = salt_and_pepper_counts(self.image.size, amount)
                self.assertEqual(int(np.count_nonzero(noisy != 128)), selected)

    def test_changed_pixels_are_only_0_or_255_and_others_unchanged(self):
        image = create_synthetic_test_image()  # values 20..240, never 0 or 255
        noisy = add_salt_and_pepper_noise(image, 0.10, seed=42)
        changed = noisy != image
        self.assertTrue(set(np.unique(noisy[changed]).tolist()) <= {0, 255})
        np.testing.assert_array_equal(noisy[~changed], image[~changed])

    def test_amount_one_selects_every_pixel(self):
        noisy = add_salt_and_pepper_noise(self.image, 1.0, seed=5)
        self.assertEqual(int(np.count_nonzero(noisy == 128)), 0)
        self.assertEqual(int(np.count_nonzero(noisy == SALT_VALUE)), 5000)
        self.assertEqual(int(np.count_nonzero(noisy == PEPPER_VALUE)), 5000)

    def test_selected_is_not_the_same_as_changed(self):
        # In an all-white image, salt pixels stay 255 (selected but unchanged);
        # only the pepper pixels visibly change.
        white = flat_image(255)
        noisy = add_salt_and_pepper_noise(white, 0.5, seed=8)
        selected, salt, pepper = salt_and_pepper_counts(white.size, 0.5)  # 5000, 2500, 2500
        self.assertEqual(int(np.count_nonzero(noisy != white)), pepper)
        self.assertLess(pepper, selected)

    def test_positions_are_spread_over_the_image(self):
        noisy = add_salt_and_pepper_noise(self.image, 0.05, seed=4)
        changed_rows = np.nonzero(noisy != 128)[0]
        self.assertGreater(int(np.count_nonzero(changed_rows < 50)), 0)
        self.assertGreater(int(np.count_nonzero(changed_rows >= 50)), 0)

    def test_amount_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            add_salt_and_pepper_noise(self.image, 1.01, seed=1)


class TestSpeckleNoise(unittest.TestCase):
    def test_matches_the_documented_formula_exactly(self):
        image = create_synthetic_test_image()
        noise = np.random.default_rng(9).normal(0.0, math.sqrt(0.04), size=image.shape)
        original = image.astype(np.float64)
        expected = np.clip(np.rint(original + original * noise), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(add_speckle_noise(image, 0.04, seed=9), expected)

    def test_black_pixels_stay_black(self):
        image = flat_image(200)
        image[:, :50] = 0
        noisy = add_speckle_noise(image, 0.1, seed=3)
        self.assertEqual(int(np.count_nonzero(noisy[:, :50])), 0)
        self.assertGreater(int(np.count_nonzero(noisy[:, 50:] != 200)), 0)

    def test_brighter_pixels_change_more(self):
        # Left half 40, right half 160, variance 0.01 (std 0.1):
        # expected change std is 40*0.1 = 4 on the left and 160*0.1 = 16 on the right.
        image = np.empty((200, 400), dtype=np.uint8)
        image[:, :200] = 40
        image[:, 200:] = 160
        change = add_speckle_noise(image, 0.01, seed=6).astype(np.float64) - image
        dark_change, bright_change = change[:, :200], change[:, 200:]
        self.assertGreater(np.abs(bright_change).mean(), 3 * np.abs(dark_change).mean())
        self.assertAlmostEqual(dark_change.std(), 4.0, delta=0.3)
        self.assertAlmostEqual(bright_change.std(), 16.0, delta=0.5)

    def test_bright_pixels_are_clipped_at_255(self):
        # 250 with variance 0.01 -> change std = 250 * 0.1 = 25 levels.
        noisy = add_speckle_noise(flat_image(250), 0.01, seed=0)
        self.assertEqual(int(noisy.max()), 255)
        self.assertGreater(int(np.count_nonzero(noisy == 255)), 0)
        # No wrap-around: values like 270 must become 255, not 14.
        self.assertGreater(int(noisy.min()), 100)

    def test_larger_variance_gives_larger_error(self):
        image = create_synthetic_test_image()
        errors = [mse(image, add_speckle_noise(image, v, seed=42)) for v in (0.01, 0.04, 0.10)]
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])


if __name__ == "__main__":
    unittest.main()
