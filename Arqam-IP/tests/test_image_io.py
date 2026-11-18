"""Unit tests for src/image_io.py.

Every test creates its own small images inside a temporary folder, so the
tests never depend on (or touch) the real input images in data/input.

Run from the project root:  python -m unittest discover -s tests -v
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.image_io import (
    format_image_info,
    get_image_info,
    load_grayscale_image,
    save_grayscale_image,
    validate_grayscale_image,
)


def write_png(path, array):
    """Write an image array to a PNG file (works for non-ASCII paths)."""
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    encoded.tofile(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestLoadGrayscaleImage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        # A tiny 3 x 4 image with known values, including 0 and 255.
        self.known = np.array([[0, 50, 100, 150], [200, 255, 10, 20], [30, 40, 60, 70]], dtype=np.uint8)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_grayscale_png_with_exact_values(self):
        path = self.folder / "known.png"
        write_png(path, self.known)
        image = load_grayscale_image(path)
        self.assertEqual(image.shape, (3, 4))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, self.known)  # PNG is lossless

    def test_accepts_string_path(self):
        path = self.folder / "known.png"
        write_png(path, self.known)
        np.testing.assert_array_equal(load_grayscale_image(str(path)), self.known)

    def test_colour_image_is_converted_to_2d_grayscale(self):
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[0, 0] = (100, 100, 100)  # equal R, G, B -> same gray value
        bgr[0, 1] = (255, 0, 0)      # pure blue (OpenCV uses BGR order)
        bgr[1, 0] = (0, 0, 255)      # pure red
        path = self.folder / "colour.png"
        write_png(path, bgr)

        image = load_grayscale_image(path)
        self.assertEqual(image.shape, (2, 2))
        self.assertEqual(image[0, 0], 100)
        # Y = 0.299 R + 0.587 G + 0.114 B; allow +-1 for codec rounding.
        self.assertAlmostEqual(int(image[0, 1]), round(0.114 * 255), delta=1)
        self.assertAlmostEqual(int(image[1, 0]), round(0.299 * 255), delta=1)

    def test_non_ascii_path_is_supported(self):
        path = self.folder / "görüntü_İ.png"  # Turkish characters
        write_png(path, self.known)
        np.testing.assert_array_equal(load_grayscale_image(path), self.known)

    def test_loading_does_not_modify_the_file(self):
        path = self.folder / "known.png"
        write_png(path, self.known)
        before = file_hash(path)
        load_grayscale_image(path)
        self.assertEqual(file_hash(path), before)

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_grayscale_image(self.folder / "does_not_exist.png")

    def test_directory_raises_value_error(self):
        folder_with_image_name = self.folder / "folder.png"
        folder_with_image_name.mkdir()
        with self.assertRaises(ValueError):
            load_grayscale_image(folder_with_image_name)

    def test_unsupported_extension_raises_value_error(self):
        path = self.folder / "notes.txt"
        path.write_text("hello")
        with self.assertRaises(ValueError):
            load_grayscale_image(path)

    def test_empty_file_raises_value_error(self):
        path = self.folder / "empty.png"
        path.touch()
        with self.assertRaises(ValueError):
            load_grayscale_image(path)

    def test_corrupted_file_raises_value_error(self):
        path = self.folder / "corrupted.png"
        path.write_text("this is not really an image")
        with self.assertRaises(ValueError):
            load_grayscale_image(path)

    def test_wrong_path_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            load_grayscale_image(123)


class TestValidateGrayscaleImage(unittest.TestCase):
    def test_valid_image_passes(self):
        self.assertIsNone(validate_grayscale_image(np.zeros((4, 4), dtype=np.uint8)))

    def test_not_an_array_raises_type_error(self):
        with self.assertRaises(TypeError):
            validate_grayscale_image([[0, 1], [2, 3]])

    def test_colour_array_raises_value_error(self):
        with self.assertRaises(ValueError):
            validate_grayscale_image(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_float_array_raises_value_error(self):
        with self.assertRaises(ValueError):
            validate_grayscale_image(np.zeros((4, 4), dtype=np.float64))

    def test_empty_array_raises_value_error(self):
        with self.assertRaises(ValueError):
            validate_grayscale_image(np.zeros((0, 4), dtype=np.uint8))


class TestGetImageInfo(unittest.TestCase):
    def test_statistics_of_known_image(self):
        image = np.array([[0, 255], [100, 45]], dtype=np.uint8)
        info = get_image_info(image)
        self.assertEqual(info["height"], 2)
        self.assertEqual(info["width"], 2)
        self.assertEqual(info["total_pixels"], 4)
        self.assertEqual(info["dtype"], "uint8")
        self.assertEqual(info["bit_depth"], 8)
        self.assertEqual(info["raw_size_bytes"], 4)
        self.assertEqual(info["min"], 0)
        self.assertEqual(info["max"], 255)
        self.assertAlmostEqual(info["mean"], 100.0)  # (0 + 255 + 100 + 45) / 4
        self.assertAlmostEqual(info["std"], float(np.std([0, 255, 100, 45])))
        self.assertEqual(info["unique_values"], 4)

    def test_mean_does_not_overflow_for_bright_image(self):
        # 255 * 10000 is far above the uint8 maximum; the mean must still be 255.
        image = np.full((100, 100), 255, dtype=np.uint8)
        self.assertAlmostEqual(get_image_info(image)["mean"], 255.0)

    def test_input_image_is_not_modified(self):
        image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        copy = image.copy()
        get_image_info(image)
        np.testing.assert_array_equal(image, copy)

    def test_format_contains_dimensions(self):
        text = format_image_info(get_image_info(np.zeros((3, 5), dtype=np.uint8)))
        self.assertIn("5 x 3", text)


class TestSaveGrayscaleImage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.image = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_then_load_round_trip(self):
        path = save_grayscale_image(self.image, self.folder / "sub" / "out.png")
        np.testing.assert_array_equal(load_grayscale_image(path), self.image)

    def test_refuses_to_overwrite_existing_file(self):
        path = self.folder / "original.png"
        write_png(path, self.image)
        before = file_hash(path)
        with self.assertRaises(FileExistsError):
            save_grayscale_image(np.zeros((3, 4), dtype=np.uint8), path)
        self.assertEqual(file_hash(path), before)  # original untouched

    def test_overwrite_true_replaces_file(self):
        path = self.folder / "out.png"
        save_grayscale_image(self.image, path)
        zeros = np.zeros((3, 4), dtype=np.uint8)
        save_grayscale_image(zeros, path, overwrite=True)
        np.testing.assert_array_equal(load_grayscale_image(path), zeros)

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ValueError):
            save_grayscale_image(np.zeros((3, 4), dtype=np.float32), self.folder / "bad.png")


if __name__ == "__main__":
    unittest.main()
