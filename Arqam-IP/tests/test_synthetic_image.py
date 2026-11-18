"""Unit tests for src/synthetic_image.py."""

import unittest

import numpy as np

from src import synthetic_image as synth
from src.image_io import validate_grayscale_image


class TestCreateSyntheticTestImage(unittest.TestCase):
    def setUp(self):
        self.image = synth.create_synthetic_test_image()

    def test_default_shape_and_type(self):
        self.assertEqual(self.image.shape, (256, 256))
        self.assertEqual(self.image.dtype, np.uint8)
        validate_grayscale_image(self.image)

    def test_custom_size(self):
        self.assertEqual(synth.create_synthetic_test_image(100, 150).shape, (100, 150))

    def test_is_deterministic(self):
        np.testing.assert_array_equal(self.image, synth.create_synthetic_test_image())

    def test_avoids_pure_black_and_white(self):
        # Salt-and-pepper noise later uses exactly 0 and 255, so the clean image must not.
        self.assertGreaterEqual(int(self.image.min()), synth.DARK_SQUARE_VALUE)
        self.assertLessEqual(int(self.image.max()), synth.LINE_VALUE)

    def test_contains_each_designed_region(self):
        values = set(np.unique(self.image).tolist())
        for expected in (
            synth.DARK_SQUARE_VALUE,
            synth.BRIGHT_CIRCLE_VALUE,
            synth.FLAT_PATCH_VALUE,
            synth.LINE_BACKGROUND_VALUE,
            synth.LINE_VALUE,
        ):
            self.assertIn(expected, values)

    def test_thin_lines_are_one_pixel_wide(self):
        # Take one row through the line block and check that no two
        # neighbouring pixels are both line pixels.
        row = self.image[int(0.7 * 256), int(0.55 * 256):int(0.92 * 256)]
        is_line = row == synth.LINE_VALUE
        self.assertTrue(is_line.any())
        self.assertFalse(np.any(is_line[:-1] & is_line[1:]))

    def test_too_small_size_raises_value_error(self):
        with self.assertRaises(ValueError):
            synth.create_synthetic_test_image(10, 256)

    def test_non_integer_size_raises_type_error(self):
        with self.assertRaises(TypeError):
            synth.create_synthetic_test_image(256.0, 256)


if __name__ == "__main__":
    unittest.main()
