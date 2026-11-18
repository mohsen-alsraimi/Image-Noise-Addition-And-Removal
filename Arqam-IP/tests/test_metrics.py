"""Unit tests for src/metrics.py (MSE and PSNR).

Expected values are calculated by hand in the comments, so each test can
be checked with a calculator.

Run from the project root:  python -m unittest -v
"""

import math
import unittest

import cv2
import numpy as np

from src.metrics import MAX_PIXEL_VALUE, mse, psnr
from src.synthetic_image import create_synthetic_test_image


class TestMSE(unittest.TestCase):
    def test_identical_images_give_exactly_zero(self):
        image = create_synthetic_test_image()
        self.assertEqual(mse(image, image.copy()), 0.0)

    def test_known_small_example(self):
        # differences: 10-12 = -2, 0, 0, 40-36 = 4
        # squares:     4, 0, 0, 16  -> sum 20 -> MSE = 20 / 4 = 5.0
        original = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        processed = np.array([[12, 20], [30, 36]], dtype=np.uint8)
        self.assertAlmostEqual(mse(original, processed), 5.0)

    def test_completely_different_images(self):
        # every pixel differs by 255 -> MSE = 255^2 = 65025
        black = np.zeros((4, 4), dtype=np.uint8)
        white = np.full((4, 4), 255, dtype=np.uint8)
        self.assertAlmostEqual(mse(black, white), 65025.0)

    def test_every_pixel_off_by_one(self):
        original = np.full((10, 10), 100, dtype=np.uint8)
        self.assertAlmostEqual(mse(original, original + 1), 1.0)

    def test_uint8_subtraction_does_not_wrap_around(self):
        # Plain uint8 arithmetic would compute 10 - 20 = 246 (wrap-around).
        original = np.array([[10]], dtype=np.uint8)
        processed = np.array([[20]], dtype=np.uint8)
        self.assertEqual(int((original - processed)[0, 0]), 246)  # the danger
        self.assertAlmostEqual(mse(original, processed), 100.0)  # (-10)^2 = 100

    def test_is_symmetric(self):
        a = np.array([[0, 50], [100, 200]], dtype=np.uint8)
        b = np.array([[5, 40], [130, 180]], dtype=np.uint8)
        self.assertAlmostEqual(mse(a, b), mse(b, a))

    def test_large_image_sum_does_not_overflow(self):
        # Sum of squares = 1,000,000 * 65025 = 6.5e10, far above the uint32 range.
        black = np.zeros((1000, 1000), dtype=np.uint8)
        white = np.full((1000, 1000), 255, dtype=np.uint8)
        self.assertAlmostEqual(mse(black, white), 65025.0)

    def test_returns_python_float(self):
        result = mse(np.zeros((2, 2), np.uint8), np.ones((2, 2), np.uint8))
        self.assertIs(type(result), float)

    def test_inputs_are_not_modified(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        b = np.array([[4, 3], [2, 1]], dtype=np.uint8)
        a_copy, b_copy = a.copy(), b.copy()
        mse(a, b)
        np.testing.assert_array_equal(a, a_copy)
        np.testing.assert_array_equal(b, b_copy)


class TestMSEInvalidInput(unittest.TestCase):
    def setUp(self):
        self.valid = np.zeros((3, 3), dtype=np.uint8)

    def test_different_dimensions_raise_value_error(self):
        with self.assertRaises(ValueError):
            mse(self.valid, np.zeros((3, 4), dtype=np.uint8))

    def test_transposed_shape_raises_value_error(self):
        # Same number of pixels (6) but a different shape.
        with self.assertRaises(ValueError):
            mse(np.zeros((2, 3), np.uint8), np.zeros((3, 2), np.uint8))

    def test_empty_images_raise_value_error(self):
        empty = np.zeros((0, 0), dtype=np.uint8)
        with self.assertRaises(ValueError):
            mse(empty, empty)

    def test_list_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            mse([[0, 0], [0, 0]], np.zeros((2, 2), np.uint8))

    def test_float_dtype_raises_value_error(self):
        with self.assertRaises(ValueError):
            mse(self.valid, np.zeros((3, 3), dtype=np.float64))

    def test_int64_dtype_raises_value_error(self):
        with self.assertRaises(ValueError):
            mse(np.zeros((3, 3), dtype=np.int64), self.valid)

    def test_colour_3d_array_raises_value_error(self):
        with self.assertRaises(ValueError):
            mse(np.zeros((3, 3, 3), np.uint8), np.zeros((3, 3, 3), np.uint8))

    def test_1d_array_raises_value_error(self):
        with self.assertRaises(ValueError):
            mse(np.zeros(9, np.uint8), np.zeros(9, np.uint8))

    def test_error_message_names_the_bad_argument(self):
        with self.assertRaisesRegex(ValueError, "processed"):
            mse(self.valid, np.zeros((3, 3), dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "original"):
            mse("not an image", self.valid)


class TestPSNR(unittest.TestCase):
    def test_max_pixel_value_is_255(self):
        self.assertEqual(MAX_PIXEL_VALUE, 255)

    def test_identical_images_give_positive_infinity(self):
        image = create_synthetic_test_image()
        result = psnr(image, image.copy())
        self.assertTrue(math.isinf(result))
        self.assertGreater(result, 0)

    def test_completely_different_images_give_zero_db(self):
        # MSE = 65025 -> PSNR = 10 log10(65025 / 65025) = 10 log10(1) = 0 dB
        black = np.zeros((4, 4), dtype=np.uint8)
        white = np.full((4, 4), 255, dtype=np.uint8)
        self.assertAlmostEqual(psnr(black, white), 0.0)

    def test_one_of_100_pixels_fully_wrong_gives_20_db(self):
        # MSE = 65025 / 100 -> PSNR = 10 log10(100) = 20 dB
        original = np.zeros((10, 10), dtype=np.uint8)
        processed = original.copy()
        processed[0, 0] = 255
        self.assertAlmostEqual(psnr(original, processed), 20.0)

    def test_one_of_10000_pixels_fully_wrong_gives_40_db(self):
        # MSE = 65025 / 10000 -> PSNR = 10 log10(10000) = 40 dB
        original = np.zeros((100, 100), dtype=np.uint8)
        processed = original.copy()
        processed[50, 50] = 255
        self.assertAlmostEqual(psnr(original, processed), 40.0)

    def test_every_pixel_off_by_one(self):
        # MSE = 1 -> PSNR = 10 log10(65025) = 48.1308... dB
        original = np.full((10, 10), 100, dtype=np.uint8)
        self.assertAlmostEqual(psnr(original, original + 1), 10 * math.log10(65025))
        self.assertAlmostEqual(psnr(original, original + 1), 48.1308, places=4)

    def test_known_small_example(self):
        # MSE = 5 (see TestMSE) -> PSNR = 10 log10(65025 / 5) = 10 log10(13005)
        original = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        processed = np.array([[12, 20], [30, 36]], dtype=np.uint8)
        self.assertAlmostEqual(psnr(original, processed), 10 * math.log10(13005))

    def test_more_error_means_lower_psnr(self):
        original = np.full((10, 10), 100, dtype=np.uint8)
        self.assertGreater(psnr(original, original + 1), psnr(original, original + 5))

    def test_returns_python_float(self):
        self.assertIs(type(psnr(np.zeros((2, 2), np.uint8), np.ones((2, 2), np.uint8))), float)

    def test_invalid_input_raises_errors(self):
        valid = np.zeros((3, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            psnr(valid, np.zeros((4, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            psnr(np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8))
        with self.assertRaises(TypeError):
            psnr(None, valid)


class TestAgreementWithOpenCV(unittest.TestCase):
    """Compare our PSNR with cv2.PSNR on images that are NOT identical."""

    def test_matches_cv2_psnr_on_random_image_pairs(self):
        rng = np.random.default_rng(seed=0)  # fixed seed -> same test data every run
        for _ in range(5):
            a = rng.integers(0, 256, size=(64, 48), dtype=np.uint8)
            b = rng.integers(0, 256, size=(64, 48), dtype=np.uint8)
            self.assertAlmostEqual(psnr(a, b), cv2.PSNR(a, b), places=6)

    def test_matches_cv2_psnr_on_small_change_to_synthetic_image(self):
        original = create_synthetic_test_image()
        processed = original.copy()
        processed[100:110, 100:110] = 0
        self.assertAlmostEqual(psnr(original, processed), cv2.PSNR(original, processed), places=6)

    def test_cv2_psnr_is_finite_for_identical_images_but_ours_is_infinite(self):
        # OpenCV computes 20*log10(255 / (sqrt(MSE) + DBL_EPSILON)); with MSE = 0
        # this is 20*log10(255 / 2.22e-16) = 361.20 dB, a finite number.
        image = create_synthetic_test_image()
        self.assertAlmostEqual(cv2.PSNR(image, image), 361.20, places=2)
        self.assertEqual(psnr(image, image), math.inf)


if __name__ == "__main__":
    unittest.main()
