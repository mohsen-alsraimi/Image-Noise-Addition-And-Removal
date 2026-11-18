"""Unit tests for src/filters.py (mean, Gaussian, median and bilateral filters).

The "known answer" tests use tiny images whose correct output can be worked
out by hand, for example: a 3x3 average of one bright pixel of 90 surrounded by
zeros is 90 / 9 = 10. The noise-reduction tests use fixed seeds, so they are
fully deterministic.

Run from the project root:  python -m unittest -v
"""

import unittest

import cv2
import numpy as np

from src.filters import (
    BORDER_TYPE,
    MEDIAN_BORDER_TYPE,
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_mean_filter,
    apply_median_filter,
)
from src.metrics import mse
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise
from src.synthetic_image import create_synthetic_test_image

# (filter function, keyword arguments for a typical call) - used by the general tests.
FILTERS = [
    (apply_mean_filter, {"kernel_size": 3}),
    (apply_gaussian_filter, {"kernel_size": 3, "sigma": 0}),
    (apply_median_filter, {"kernel_size": 3}),
    (apply_bilateral_filter, {"diameter": 5, "sigma_color": 50, "sigma_space": 50}),
]

# The filters whose window size parameter is called kernel_size.
KERNEL_FILTERS = [apply_mean_filter, apply_gaussian_filter, apply_median_filter]


def flat_image(value, height=20, width=20):
    return np.full((height, width), value, dtype=np.uint8)


def step_image(left=30, right=120, height=5, width=6):
    """Left half `left`, right half `right`: a vertical edge between columns 2 and 3."""
    image = np.full((height, width), left, dtype=np.uint8)
    image[:, width // 2:] = right
    return image


def impulse_image(value, size=5):
    """All zeros except one pixel in the centre."""
    image = np.zeros((size, size), dtype=np.uint8)
    image[size // 2, size // 2] = value
    return image


class TestGeneralBehaviour(unittest.TestCase):
    """Rules every filter must follow."""

    def setUp(self):
        self.image = create_synthetic_test_image()

    def test_output_shape_and_dtype(self):
        for function, kwargs in FILTERS:
            with self.subTest(function=function.__name__):
                result = function(self.image, **kwargs)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, self.image.shape)
                self.assertEqual(result.dtype, np.uint8)

    def test_non_square_and_tiny_images_keep_their_shape(self):
        for shape in [(7, 11), (11, 7), (1, 1), (1, 9), (9, 1)]:
            image = np.random.default_rng(3).integers(0, 256, shape).astype(np.uint8)
            for function, kwargs in FILTERS:
                with self.subTest(function=function.__name__, shape=shape):
                    result = function(image, **kwargs)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.dtype, np.uint8)

    def test_default_parameters_work(self):
        for function, _ in FILTERS:
            with self.subTest(function=function.__name__):
                self.assertEqual(function(self.image).shape, self.image.shape)

    def test_input_image_is_not_modified(self):
        noisy = add_salt_and_pepper_noise(self.image, 0.05, seed=1)
        for function, kwargs in FILTERS:
            with self.subTest(function=function.__name__):
                before = noisy.copy()
                function(noisy, **kwargs)
                np.testing.assert_array_equal(noisy, before)

    def test_result_does_not_share_memory_with_input(self):
        for function, kwargs in FILTERS:
            with self.subTest(function=function.__name__):
                self.assertFalse(np.shares_memory(function(self.image, **kwargs), self.image))
        # Kernel size 1 changes nothing, but must still return a separate copy.
        for function in KERNEL_FILTERS:
            with self.subTest(function=function.__name__, kernel_size=1):
                self.assertFalse(np.shares_memory(function(self.image, kernel_size=1), self.image))

    def test_same_input_and_settings_give_identical_results(self):
        noisy = add_gaussian_noise(self.image, 25, seed=42)
        for function, kwargs in FILTERS:
            with self.subTest(function=function.__name__):
                np.testing.assert_array_equal(function(noisy, **kwargs), function(noisy, **kwargs))

    def test_constant_image_stays_unchanged(self):
        # Every filter averages or picks values from the window. If all values
        # are equal, the result must be that same value, also at the borders.
        for value in (0, 1, 128, 254, 255):
            image = flat_image(value, 9, 13)
            for function, kwargs in FILTERS:
                with self.subTest(function=function.__name__, value=value):
                    np.testing.assert_array_equal(function(image, **kwargs), image)

    def test_constant_image_stays_unchanged_for_larger_windows(self):
        image = flat_image(77, 15, 15)
        for size in (5, 7, 9):
            for function in KERNEL_FILTERS:
                with self.subTest(function=function.__name__, size=size):
                    np.testing.assert_array_equal(function(image, kernel_size=size), image)
            with self.subTest(function="apply_bilateral_filter", size=size):
                np.testing.assert_array_equal(apply_bilateral_filter(image, diameter=size), image)

    def test_non_contiguous_view_gives_same_result_as_a_copy(self):
        view = self.image[::-1, ::2]  # flipped and every second column: a view, not a copy
        for function, kwargs in FILTERS:
            with self.subTest(function=function.__name__):
                np.testing.assert_array_equal(function(view, **kwargs),
                                              function(np.ascontiguousarray(view), **kwargs))

    def test_results_match_direct_opencv_calls_with_the_documented_borders(self):
        noisy = add_gaussian_noise(self.image, 25, seed=7)
        np.testing.assert_array_equal(apply_mean_filter(noisy, 5),
                                      cv2.blur(noisy, (5, 5), borderType=cv2.BORDER_REFLECT_101))
        np.testing.assert_array_equal(apply_gaussian_filter(noisy, 5, 1.5),
                                      cv2.GaussianBlur(noisy, (5, 5), 1.5, sigmaY=1.5,
                                                       borderType=cv2.BORDER_REFLECT_101))
        np.testing.assert_array_equal(apply_median_filter(noisy, 5), cv2.medianBlur(noisy, 5))
        np.testing.assert_array_equal(apply_bilateral_filter(noisy, 7, 30, 20),
                                      cv2.bilateralFilter(noisy, 7, 30, 20,
                                                          borderType=cv2.BORDER_REFLECT_101))

    def test_border_constants_are_the_documented_ones(self):
        self.assertEqual(BORDER_TYPE, cv2.BORDER_REFLECT_101)
        self.assertEqual(BORDER_TYPE, cv2.BORDER_DEFAULT)  # OpenCV's own default
        self.assertEqual(MEDIAN_BORDER_TYPE, cv2.BORDER_REPLICATE)

    def test_filters_reduce_gaussian_noise_on_a_flat_image(self):
        # On a flat image there are no edges to blur, so every smoothing filter
        # must bring the image closer to the clean original.
        clean = flat_image(128, 200, 200)
        noisy = add_gaussian_noise(clean, 20, seed=0)
        noisy_error = mse(clean, noisy)
        for function, kwargs in FILTERS:
            with self.subTest(function=function.__name__):
                self.assertLess(mse(clean, function(noisy, **kwargs)), noisy_error)


class TestInvalidImages(unittest.TestCase):

    BAD_IMAGES = [
        ([[1, 2], [3, 4]], TypeError),                       # a list, not an array
        (np.zeros((5, 5), dtype=np.float64), ValueError),    # wrong dtype
        (np.zeros((5, 5), dtype=np.uint16), ValueError),     # 16-bit
        (np.zeros((5, 5, 3), dtype=np.uint8), ValueError),   # colour (3-D)
        (np.zeros(5, dtype=np.uint8), ValueError),           # 1-D
        (np.zeros((0, 5), dtype=np.uint8), ValueError),      # empty
        (None, TypeError),
    ]

    def test_invalid_images_are_rejected(self):
        for function, kwargs in FILTERS:
            for image, error in self.BAD_IMAGES:
                with self.subTest(function=function.__name__, image=repr(image)[:30]):
                    with self.assertRaises(error):
                        function(image, **kwargs)


class TestKernelSizeValidation(unittest.TestCase):

    def setUp(self):
        self.image = create_synthetic_test_image(64, 64)

    def test_valid_odd_kernel_sizes_are_accepted(self):
        for function in KERNEL_FILTERS:
            for size in (1, 3, 5, 7, 9, 15):
                with self.subTest(function=function.__name__, size=size):
                    self.assertEqual(function(self.image, kernel_size=size).shape, self.image.shape)

    def test_numpy_integer_kernel_size_gives_same_result_as_int(self):
        for function in KERNEL_FILTERS:
            with self.subTest(function=function.__name__):
                np.testing.assert_array_equal(function(self.image, kernel_size=np.int64(5)),
                                              function(self.image, kernel_size=5))

    def test_kernel_size_one_leaves_the_image_unchanged(self):
        for function in KERNEL_FILTERS:
            with self.subTest(function=function.__name__):
                np.testing.assert_array_equal(function(self.image, kernel_size=1), self.image)

    def test_kernel_larger_than_the_image_still_works(self):
        tiny = np.arange(9, dtype=np.uint8).reshape(3, 3) * 20
        for function in KERNEL_FILTERS:
            with self.subTest(function=function.__name__):
                self.assertEqual(function(tiny, kernel_size=9).shape, (3, 3))

    def test_even_zero_and_negative_kernel_sizes_are_rejected(self):
        for function in KERNEL_FILTERS:
            for size in (0, -1, -3, 2, 4, 10):
                with self.subTest(function=function.__name__, size=size):
                    with self.assertRaisesRegex(ValueError, "kernel_size"):
                        function(self.image, kernel_size=size)

    def test_non_integer_kernel_sizes_are_rejected(self):
        for function in KERNEL_FILTERS:
            for size in (3.0, 3.5, "3", None, True, False, np.bool_(True), [3], 3j):
                with self.subTest(function=function.__name__, size=repr(size)):
                    with self.assertRaisesRegex(TypeError, "kernel_size"):
                        function(self.image, kernel_size=size)


class TestMeanFilter(unittest.TestCase):

    def test_single_bright_pixel_is_spread_over_its_3x3_neighbourhood(self):
        # 90 / 9 = 10 in the 3x3 block around the centre, 0 elsewhere.
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1:4, 1:4] = 10
        np.testing.assert_array_equal(apply_mean_filter(impulse_image(90), 3), expected)

    def test_step_edge_is_blurred_into_intermediate_values(self):
        # Column 2: (30 + 30 + 120) / 3 = 60.  Column 3: (30 + 120 + 120) / 3 = 90.
        result = apply_mean_filter(step_image(30, 120), 3)
        np.testing.assert_array_equal(result[:, 2], np.full(5, 60))
        np.testing.assert_array_equal(result[:, 3], np.full(5, 90))
        np.testing.assert_array_equal(result[:, [0, 1]], np.full((5, 2), 30))
        np.testing.assert_array_equal(result[:, [4, 5]], np.full((5, 2), 120))

    def test_5x5_window_averages_25_pixels(self):
        image = np.zeros((9, 9), dtype=np.uint8)
        image[4, 4] = 250
        self.assertEqual(apply_mean_filter(image, 5)[4, 4], 10)  # 250 / 25

    def test_border_uses_reflect_101(self):
        # Corner pixel 90, rest 0. With BORDER_REFLECT_101 the corner pixel
        # appears once in its own 3x3 window: 90 / 9 = 10.
        # (BORDER_REPLICATE would repeat it 4 times and give 40.)
        image = np.zeros((5, 5), dtype=np.uint8)
        image[0, 0] = 90
        self.assertEqual(apply_mean_filter(image, 3)[0, 0], 10)


class TestGaussianFilter(unittest.TestCase):

    def test_sigma_zero_kernel_3_uses_weights_quarter_half_quarter(self):
        # 2-D weights = outer product of [1/4, 1/2, 1/4]:
        # centre 1/4, direct neighbours 1/8, diagonal neighbours 1/16.
        # With a centre value of 160: 40, 20 and 10.
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1:4, 1:4] = [[10, 20, 10],
                              [20, 40, 20],
                              [10, 20, 10]]
        np.testing.assert_array_equal(apply_gaussian_filter(impulse_image(160), 3, 0), expected)

    def test_default_sigma_matches_opencv_kernel_from_get_gaussian_kernel(self):
        np.testing.assert_allclose(cv2.getGaussianKernel(3, 0).ravel(), [0.25, 0.5, 0.25])

    def test_larger_sigma_smooths_more(self):
        image = impulse_image(255, size=9)
        centre_values = [int(apply_gaussian_filter(image, 7, sigma)[4, 4]) for sigma in (0.5, 1.0, 2.0)]
        self.assertGreater(centre_values[0], centre_values[1])
        self.assertGreater(centre_values[1], centre_values[2])

    def test_gaussian_blurs_an_edge_less_than_the_mean_filter(self):
        # The Gaussian gives the centre more weight than the neighbours,
        # so next to the edge it stays closer to the original value.
        image = step_image(30, 120)
        gaussian = apply_gaussian_filter(image, 3, 0)
        mean = apply_mean_filter(image, 3)
        self.assertEqual(int(gaussian[0, 2]), 53)  # 0.75 * 30 + 0.25 * 120 = 52.5 -> 53
        self.assertLess(abs(int(gaussian[0, 2]) - 30), abs(int(mean[0, 2]) - 30))

    def test_numpy_float_and_int_sigma_are_accepted(self):
        image = create_synthetic_test_image(64, 64)
        np.testing.assert_array_equal(apply_gaussian_filter(image, 5, np.float64(1.0)),
                                      apply_gaussian_filter(image, 5, 1))

    def test_invalid_sigma_values_are_rejected(self):
        image = flat_image(100)
        for sigma in (-1, -0.001, float("nan"), float("inf"), float("-inf")):
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(ValueError, "sigma"):
                    apply_gaussian_filter(image, 3, sigma)
        for sigma in (True, False, np.bool_(False), "1", None, 1j):
            with self.subTest(sigma=repr(sigma)):
                with self.assertRaisesRegex(TypeError, "sigma"):
                    apply_gaussian_filter(image, 3, sigma)


class TestMedianFilter(unittest.TestCase):

    def test_median_of_a_3x3_window(self):
        image = np.array([[10, 20, 30],
                          [40, 50, 60],
                          [70, 80, 90]], dtype=np.uint8)
        self.assertEqual(apply_median_filter(image, 3)[1, 1], 50)  # middle of 9 sorted values

    def test_isolated_salt_and_pepper_pixels_are_removed_completely(self):
        image = flat_image(100, 9, 9)
        image[2, 2] = 255  # salt
        image[6, 6] = 0    # pepper
        np.testing.assert_array_equal(apply_median_filter(image, 3), flat_image(100, 9, 9))

    def test_step_edge_is_kept_perfectly_sharp(self):
        image = step_image(30, 120)
        np.testing.assert_array_equal(apply_median_filter(image, 3), image)

    def test_one_pixel_wide_line_is_erased(self):
        # A thin line is only 3 of 9 pixels in a 3x3 window, so the median
        # ignores it. This is the price of the median filter: fine detail is lost.
        image = flat_image(40, 7, 7)
        image[:, 3] = 240
        np.testing.assert_array_equal(apply_median_filter(image, 3), flat_image(40, 7, 7))

    def test_output_contains_only_values_from_the_input(self):
        image = np.random.default_rng(5).choice([0, 17, 200], size=(30, 30)).astype(np.uint8)
        self.assertTrue(set(np.unique(apply_median_filter(image, 3))) <= {0, 17, 200})

    def test_border_uses_replicate(self):
        # Two salt pixels in the top-left corner of a flat image. With
        # BORDER_REPLICATE the corner window contains 6 copies of 255 out of 9,
        # so the median is 255. (BORDER_REFLECT_101 would give only 3 of 9 -> 100.)
        image = flat_image(100, 5, 5)
        image[0, 0] = 255
        image[0, 1] = 255
        self.assertEqual(apply_median_filter(image, 3)[0, 0], 255)

    def test_kernel_sizes_above_5_work_for_uint8(self):
        # OpenCV allows sizes > 5 only for 8-bit images, which is all we accept.
        image = flat_image(100, 15, 15)
        image[7, 7] = 255
        np.testing.assert_array_equal(apply_median_filter(image, 7), flat_image(100, 15, 15))

    def test_removes_most_salt_and_pepper_noise_on_a_flat_image(self):
        clean = flat_image(128, 200, 200)
        noisy = add_salt_and_pepper_noise(clean, 0.05, seed=0)
        median_error = mse(clean, apply_median_filter(noisy, 3))
        self.assertLess(median_error, 0.02 * mse(clean, noisy))
        # The mean filter spreads every impulse over 9 pixels instead of removing it.
        self.assertGreater(mse(clean, apply_mean_filter(noisy, 3)), median_error)


class TestBilateralFilter(unittest.TestCase):

    def test_strong_edge_is_preserved_when_sigma_color_is_small(self):
        # The two sides differ by 150 levels, far more than sigma_color = 10,
        # so pixels from the other side get (almost) zero weight.
        image = step_image(50, 200, height=9, width=10)
        np.testing.assert_array_equal(apply_bilateral_filter(image, 5, 10, 50), image)

    def test_very_large_sigma_color_blurs_the_edge(self):
        # With sigma_color much larger than the 150-level jump, every neighbour
        # counts, so the filter behaves like an ordinary (Gaussian-weighted) blur.
        image = step_image(50, 200, height=9, width=10)
        result = apply_bilateral_filter(image, 5, 1000, 50)
        self.assertGreater(int(result[4, 4]), 50)
        self.assertLess(int(result[4, 5]), 200)

    def test_smooths_small_noise_on_a_flat_image(self):
        clean = flat_image(128, 100, 100)
        noisy = add_gaussian_noise(clean, 10, seed=3)
        filtered = apply_bilateral_filter(noisy, 5, 50, 50)
        self.assertLess(float(filtered.std()), 0.6 * float(noisy.std()))

    def test_valid_diameters_and_numeric_types_are_accepted(self):
        image = create_synthetic_test_image(64, 64)
        for diameter in (3, 5, 7, 9, np.int64(5)):
            with self.subTest(diameter=diameter):
                self.assertEqual(apply_bilateral_filter(image, diameter, 50, 50).shape, image.shape)
        np.testing.assert_array_equal(apply_bilateral_filter(image, 5, np.float64(50.0), 50.0),
                                      apply_bilateral_filter(image, 5, 50, 50))

    def test_invalid_diameters_are_rejected(self):
        image = flat_image(100)
        # 1 and 2 would silently act like 3 in OpenCV; even values like the next odd one;
        # 0 and negative values would make OpenCV choose the size from sigma_space.
        for diameter in (1, 2, 4, 6, 0, -1, -5):
            with self.subTest(diameter=diameter):
                with self.assertRaisesRegex(ValueError, "diameter"):
                    apply_bilateral_filter(image, diameter, 50, 50)
        for diameter in (5.0, "5", None, True, np.bool_(True)):
            with self.subTest(diameter=repr(diameter)):
                with self.assertRaisesRegex(TypeError, "diameter"):
                    apply_bilateral_filter(image, diameter, 50, 50)

    def test_invalid_sigma_values_are_rejected(self):
        image = flat_image(100)
        for name in ("sigma_color", "sigma_space"):
            # OpenCV would silently replace 0 or negative sigmas by 1.
            for value in (0, 0.0, -1, -50, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        apply_bilateral_filter(image, 5, **{name: value})
            for value in (True, False, "50", None, 1j):
                with self.subTest(name=name, value=repr(value)):
                    with self.assertRaisesRegex(TypeError, name):
                        apply_bilateral_filter(image, 5, **{name: value})


if __name__ == "__main__":
    unittest.main()
