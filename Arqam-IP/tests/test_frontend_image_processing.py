"""Unit tests for frontend/image_processing.py (Stage 5A: image upload).

Uploaded files are simulated with bytes created in memory by cv2.imencode.
Nothing here starts Streamlit or touches data/input.

Run from the project root:  python -m unittest tests.test_frontend_image_processing -v
"""

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from frontend.image_processing import (
    BILATERAL_DIAMETERS,
    BILATERAL_SIGMA_RANGE,
    DEFAULT_SEED,
    FILTER_OPTIONS,
    FILTER_TYPES,
    GAUSSIAN_FILTER_SIGMA_RANGE,
    KERNEL_SIZES,
    LARGE_IMAGE_PIXELS,
    MAX_FILE_SIZE_BYTES,
    NOISE_OPTIONS,
    NOISE_TYPES,
    UploadedImage,
    apply_selected_filter,
    check_file_name,
    compare_all_filters,
    crop_square,
    describe_filter,
    describe_filter_parameters,
    describe_noise,
    enhance_differences,
    evaluate_image_quality,
    format_mse,
    format_psnr,
    format_signed,
    judge_change,
    make_difference_image,
    psnr_change,
    summarize_difference,
    detect_image_format,
    enlarge_pixels,
    fit_for_display,
    format_file_size,
    get_filter_option,
    get_noise_option,
    is_large_image,
    load_uploaded_image,
    make_noisy_image,
    pixel_fingerprint,
    png_has_alpha,
)
from src.filters import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_mean_filter,
    apply_median_filter,
)
from src.image_io import load_grayscale_image
from src.metrics import mse, psnr
from src.noise import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    add_speckle_noise,
    salt_and_pepper_counts,
)
from src.synthetic_image import create_synthetic_test_image


def encode(extension, array, *params):
    """Encode an array as image-file bytes, like a file a user would upload."""
    ok, encoded = cv2.imencode(extension, array, list(params))
    assert ok
    return encoded.tobytes()


def random_gray(height=40, width=60, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8)


def random_bgr(height=40, width=60, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def add_exif_orientation(jpeg_bytes, orientation):
    """Insert a minimal EXIF block with the given orientation tag into a JPEG.

    Phone cameras store photos sideways and record the rotation in this tag.
    Orientation 6 means "rotate 90 degrees clockwise to display".
    """
    tiff = (
        b"II*\x00"  # little-endian TIFF header
        + struct.pack("<I", 8)  # first directory starts at byte 8
        + struct.pack("<H", 1)  # one entry
        + struct.pack("<HHII", 0x0112, 3, 1, orientation)  # Orientation, SHORT, count 1
        + struct.pack("<I", 0)  # no next directory
    )
    app1 = b"Exif\x00\x00" + tiff
    segment = b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
    return jpeg_bytes[:2] + segment + jpeg_bytes[2:]


class TestValidUploads(unittest.TestCase):
    def test_grayscale_png_is_loaded_exactly(self):
        gray = random_gray()
        result = load_uploaded_image("photo.png", encode(".png", gray))
        self.assertIsInstance(result, UploadedImage)
        self.assertEqual(result.file_format, "PNG")
        self.assertEqual(result.grayscale.dtype, np.uint8)
        self.assertEqual(result.grayscale.ndim, 2)
        np.testing.assert_array_equal(result.grayscale, gray)  # PNG is lossless
        self.assertFalse(result.is_colour)
        self.assertFalse(result.has_alpha)

    def test_dimensions(self):
        result = load_uploaded_image("wide.png", encode(".png", random_gray(height=30, width=70)))
        self.assertEqual((result.height, result.width), (30, 70))
        self.assertEqual(result.grayscale.shape, (30, 70))
        self.assertEqual(result.preview_rgb.shape, (30, 70, 3))
        self.assertEqual(result.pixel_count, 2100)

    def test_colour_png(self):
        bgr = random_bgr()
        result = load_uploaded_image("colour.png", encode(".png", bgr))
        self.assertTrue(result.is_colour)
        self.assertEqual(result.grayscale.shape, (40, 60))
        self.assertEqual(result.preview_rgb.dtype, np.uint8)
        # The preview must be in RGB order (what browsers expect), not OpenCV's BGR.
        np.testing.assert_array_equal(result.preview_rgb, bgr[:, :, ::-1])

    def test_pure_red_pixel_order(self):
        red_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 255
        result = load_uploaded_image("red.png", encode(".png", red_bgr))
        np.testing.assert_array_equal(result.preview_rgb[0, 0], [255, 0, 0])

    def test_gray_image_stored_as_rgb_is_not_colour(self):
        gray = random_gray()
        result = load_uploaded_image("gray_rgb.png", encode(".png", cv2.merge([gray, gray, gray])))
        self.assertFalse(result.is_colour)
        np.testing.assert_array_equal(result.grayscale, gray)

    def test_colour_jpeg(self):
        data = encode(".jpg", random_bgr(), cv2.IMWRITE_JPEG_QUALITY, 90)
        for name in ("photo.jpg", "photo.jpeg", "PHOTO.JPG", "Photo.JpEg"):
            with self.subTest(name=name):
                result = load_uploaded_image(name, data)
                self.assertEqual(result.file_format, "JPEG")
                self.assertEqual(result.grayscale.shape, (40, 60))
                self.assertEqual(result.grayscale.dtype, np.uint8)
                self.assertTrue(result.is_colour)

    def test_grayscale_jpeg_is_not_colour(self):
        result = load_uploaded_image("gray.jpg", encode(".jpg", random_gray()))
        self.assertFalse(result.is_colour)
        self.assertEqual(result.file_format, "JPEG")

    def test_same_grayscale_as_stage1_loader(self):
        """The app and the Stage 1 demos must see exactly the same pixels."""
        cases = {
            "colour.png": encode(".png", random_bgr()),
            "colour.jpg": encode(".jpg", random_bgr(), cv2.IMWRITE_JPEG_QUALITY, 85),
            "gray.png": encode(".png", random_gray()),
            "phone_rotated.jpg": add_exif_orientation(encode(".jpg", random_bgr(height=16, width=32)), 6),
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, data in cases.items():
                with self.subTest(name=name):
                    path = Path(folder) / name
                    path.write_bytes(data)
                    np.testing.assert_array_equal(
                        load_uploaded_image(name, data).grayscale, load_grayscale_image(path)
                    )

    def test_sixteen_bit_png_becomes_uint8(self):
        values = np.random.default_rng(2).integers(0, 65536, (20, 20)).astype(np.uint16)
        result = load_uploaded_image("deep.png", encode(".png", values))
        self.assertEqual(result.grayscale.dtype, np.uint8)
        np.testing.assert_array_equal(result.grayscale, (values >> 8).astype(np.uint8))

    def test_transparent_png_is_flagged(self):
        rgba = np.dstack([random_bgr(), np.zeros((40, 60), dtype=np.uint8)])
        result = load_uploaded_image("logo.png", encode(".png", rgba))
        self.assertTrue(result.has_alpha)
        self.assertEqual(result.grayscale.shape, (40, 60))
        self.assertEqual(result.preview_rgb.shape, (40, 60, 3))

    def test_exif_orientation_is_applied_to_both_images(self):
        data = add_exif_orientation(encode(".jpg", random_bgr(height=16, width=32)), orientation=6)
        result = load_uploaded_image("phone.jpg", data)
        self.assertEqual(result.grayscale.shape, (32, 16))  # rotated: height and width swap
        self.assertEqual(result.preview_rgb.shape, (32, 16, 3))

    def test_extension_and_contents_may_differ(self):
        """The contents decide the format: a PNG saved as '.jpg' is still a PNG."""
        result = load_uploaded_image("renamed.jpg", encode(".png", random_gray()))
        self.assertEqual(result.file_format, "PNG")

    def test_tiny_image(self):
        result = load_uploaded_image("one.png", encode(".png", np.array([[7]], dtype=np.uint8)))
        self.assertEqual(result.grayscale.shape, (1, 1))
        self.assertEqual(int(result.grayscale[0, 0]), 7)

    def test_bytearray_and_memoryview_accepted(self):
        data = encode(".png", random_gray())
        for wrapped in (bytearray(data), memoryview(data)):
            with self.subTest(kind=type(wrapped).__name__):
                self.assertEqual(load_uploaded_image("a.png", wrapped).file_bytes, data)

    def test_deterministic(self):
        data = encode(".jpg", random_bgr())
        first = load_uploaded_image("a.jpg", data)
        second = load_uploaded_image("a.jpg", data)
        np.testing.assert_array_equal(first.grayscale, second.grayscale)
        np.testing.assert_array_equal(first.preview_rgb, second.preview_rgb)


class TestOriginalIsPreserved(unittest.TestCase):
    def setUp(self):
        self.data = encode(".png", random_bgr())
        self.result = load_uploaded_image("photo.png", self.data)

    def test_original_bytes_kept_exactly(self):
        self.assertIsInstance(self.result.file_bytes, bytes)
        self.assertEqual(self.result.file_bytes, self.data)
        self.assertEqual(self.result.file_size, len(self.data))
        self.assertEqual(self.result.file_name, "photo.png")

    def test_later_changes_to_the_input_buffer_do_not_matter(self):
        buffer = bytearray(self.data)
        result = load_uploaded_image("photo.png", buffer)
        buffer[:] = b"\x00" * len(buffer)
        self.assertEqual(result.file_bytes, self.data)

    def test_grayscale_reference_is_read_only(self):
        self.assertFalse(self.result.grayscale.flags.writeable)
        with self.assertRaises(ValueError):
            self.result.grayscale[0, 0] = 0

    def test_preview_is_read_only(self):
        self.assertFalse(self.result.preview_rgb.flags.writeable)
        with self.assertRaises(ValueError):
            self.result.preview_rgb[0, 0, 0] = 0

    def test_preview_and_grayscale_do_not_share_memory(self):
        self.assertFalse(np.shares_memory(self.result.preview_rgb, self.result.grayscale))

    def test_record_cannot_be_reassigned(self):
        with self.assertRaises(AttributeError):
            self.result.grayscale = np.zeros((2, 2), dtype=np.uint8)


class TestInvalidUploads(unittest.TestCase):
    def test_empty_file(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            load_uploaded_image("empty.png", b"")

    def test_wrong_data_type(self):
        for bad in (None, "not bytes", 123, [1, 2, 3], np.zeros(10, dtype=np.uint8)):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    load_uploaded_image("a.png", bad)

    def test_wrong_file_name_type(self):
        for bad in (None, 5, Path("a.png"), b"a.png"):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    load_uploaded_image(bad, encode(".png", random_gray()))

    def test_unsupported_extensions(self):
        data = encode(".png", random_gray())
        for name in ("a.bmp", "a.gif", "a.tif", "a.txt", "a", "a.png.exe", "", "   "):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    load_uploaded_image(name, data)

    def test_other_image_formats_rejected_even_with_png_name(self):
        for extension in (".bmp", ".tif"):
            with self.subTest(extension=extension):
                with self.assertRaisesRegex(ValueError, "not a PNG or JPEG"):
                    load_uploaded_image("disguised.png", encode(extension, random_gray()))

    def test_text_file_renamed_to_png(self):
        with self.assertRaisesRegex(ValueError, "not a PNG or JPEG"):
            load_uploaded_image("notes.png", b"This is a text file, not an image.")

    def test_corrupted_png(self):
        data = b"\x89PNG\r\n\x1a\n" + b"x" * 200  # correct signature, garbage after it
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            load_uploaded_image("broken.png", data)

    def test_truncated_files(self):
        cases = {
            "cut.png": encode(".png", random_bgr()),
            "cut.jpg": encode(".jpg", random_bgr()),
        }
        for name, data in cases.items():
            for fraction in (0.5, 0.9):
                with self.subTest(name=name, fraction=fraction):
                    with self.assertRaisesRegex(ValueError, "could not be decoded"):
                        load_uploaded_image(name, data[: int(len(data) * fraction)])

    def test_file_too_large(self):
        data = encode(".png", random_gray())
        with self.assertRaisesRegex(ValueError, "too large"):
            load_uploaded_image("big.png", data, max_file_bytes=len(data) - 1)
        # exactly at the limit is allowed
        load_uploaded_image("ok.png", data, max_file_bytes=len(data))

    def test_default_file_limit_is_20_mb(self):
        self.assertEqual(MAX_FILE_SIZE_BYTES, 20 * 1024 * 1024)

    def test_decoder_result_is_validated(self):
        """OpenCV always returns 2-D uint8 here, so we fake a wrong result to check
        that validate_grayscale_image (from src/image_io.py) is really called."""
        data = encode(".png", random_gray())
        wrong_result = np.zeros((4, 4), dtype=np.float64)
        with mock.patch("frontend.image_processing._decode", return_value=wrong_result):
            with self.assertRaisesRegex(ValueError, "uint8"):
                load_uploaded_image("a.png", data)

    def test_too_many_pixels(self):
        data = encode(".png", random_gray(height=40, width=60))  # 2400 pixels
        with self.assertRaisesRegex(ValueError, "too large"):
            load_uploaded_image("big.png", data, max_pixels=2399)
        load_uploaded_image("ok.png", data, max_pixels=2400)


class TestSmallHelpers(unittest.TestCase):
    def test_check_file_name_returns_lower_case_extension(self):
        self.assertEqual(check_file_name("My Photo.JPEG"), ".jpeg")
        self.assertEqual(check_file_name("görüntü_İ.png"), ".png")  # Turkish letters are fine

    def test_detect_image_format(self):
        self.assertEqual(detect_image_format(encode(".png", random_gray())), "PNG")
        self.assertEqual(detect_image_format(encode(".jpg", random_gray())), "JPEG")
        with self.assertRaises(ValueError):
            detect_image_format(b"GIF89a....")

    def test_png_has_alpha(self):
        gray_png = encode(".png", random_gray())
        self.assertFalse(png_has_alpha(gray_png))
        self.assertFalse(png_has_alpha(encode(".png", random_bgr())))
        self.assertTrue(png_has_alpha(encode(".png", np.dstack([random_bgr(), random_gray()]))))
        # Colour type 4 (gray + alpha): change only the colour-type byte of the header.
        gray_alpha_header = gray_png[:25] + bytes([4]) + gray_png[26:]
        self.assertTrue(png_has_alpha(gray_alpha_header))
        self.assertFalse(png_has_alpha(encode(".jpg", random_bgr())))
        self.assertFalse(png_has_alpha(b"\x89PNG"))  # too short to contain a header

    def test_is_large_image(self):
        result = load_uploaded_image("a.png", encode(".png", random_gray(height=10, width=10)))
        self.assertFalse(is_large_image(result))
        self.assertTrue(is_large_image(result, threshold=99))
        self.assertFalse(is_large_image(result, threshold=100))
        self.assertEqual(LARGE_IMAGE_PIXELS, 4_000_000)

    def test_format_file_size(self):
        self.assertEqual(format_file_size(0), "0 bytes")
        self.assertEqual(format_file_size(1023), "1023 bytes")
        self.assertEqual(format_file_size(2048), "2.0 KB")
        self.assertEqual(format_file_size(5 * 1024 * 1024), "5.0 MB")


# --------------------------------------------------------------------------
# Stage 5B: noise
# --------------------------------------------------------------------------
DIRECT_NOISE_FUNCTIONS = {
    "gaussian": add_gaussian_noise,
    "salt_and_pepper": add_salt_and_pepper_noise,
    "speckle": add_speckle_noise,
}


class TestNoiseOptions(unittest.TestCase):
    def test_three_types_in_order(self):
        self.assertEqual(NOISE_TYPES, ("gaussian", "salt_and_pepper", "speckle"))

    def test_each_type_uses_the_existing_function(self):
        for noise_type, function in DIRECT_NOISE_FUNCTIONS.items():
            with self.subTest(noise_type=noise_type):
                self.assertIs(NOISE_OPTIONS[noise_type].function, function)

    def test_parameter_names_match_src_noise(self):
        self.assertEqual(NOISE_OPTIONS["gaussian"].parameter, "sigma")
        self.assertEqual(NOISE_OPTIONS["salt_and_pepper"].parameter, "amount")
        self.assertEqual(NOISE_OPTIONS["speckle"].parameter, "variance")

    def test_defaults_are_the_stage3_baselines(self):
        """E002 (sigma 25), E005 (amount 0.05), E008 (variance 0.04), seed 42."""
        self.assertEqual(NOISE_OPTIONS["gaussian"].default, 25)
        self.assertEqual(NOISE_OPTIONS["salt_and_pepper"].default, 0.05)
        self.assertEqual(NOISE_OPTIONS["speckle"].default, 0.04)
        self.assertEqual(DEFAULT_SEED, 42)

    def test_slider_ranges_cover_stage3_values_and_obey_src_noise_rules(self):
        stage3_values = {"gaussian": (10, 25, 50), "salt_and_pepper": (0.02, 0.05, 0.10), "speckle": (0.01, 0.04, 0.10)}
        for noise_type, values in stage3_values.items():
            option = NOISE_OPTIONS[noise_type]
            with self.subTest(noise_type=noise_type):
                self.assertGreaterEqual(option.minimum, 0)  # src/noise.py rejects negative strengths
                self.assertGreater(option.step, 0)
                for value in values + (option.default,):
                    self.assertTrue(option.minimum <= value <= option.maximum)
                    # every Stage 3 value can be reached exactly with the slider step
                    steps = (value - option.minimum) / option.step
                    self.assertAlmostEqual(steps, round(steps), places=9)
        self.assertLessEqual(NOISE_OPTIONS["salt_and_pepper"].maximum, 1.0)  # amount <= 1 in src/noise.py

    def test_get_noise_option_errors(self):
        for bad in ("poisson", "Gaussian", "", "salt and pepper"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "Unknown noise type"):
                    get_noise_option(bad)
        for bad in (None, 1, ["gaussian"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    get_noise_option(bad)


class TestMakeNoisyImage(unittest.TestCase):
    def setUp(self):
        self.image = create_synthetic_test_image()  # 256 x 256, values 20..240
        self.original_copy = self.image.copy()

    def test_exactly_equal_to_direct_calls(self):
        cases = [
            ("gaussian", 25), ("gaussian", 10), ("gaussian", 99),
            ("salt_and_pepper", 0.05), ("salt_and_pepper", 0.3),
            ("speckle", 0.04), ("speckle", 0.25),
        ]
        for noise_type, strength in cases:
            for seed in (0, 42, 12345):
                with self.subTest(noise_type=noise_type, strength=strength, seed=seed):
                    expected = DIRECT_NOISE_FUNCTIONS[noise_type](self.image, strength, seed=seed)
                    np.testing.assert_array_equal(make_noisy_image(self.image, noise_type, strength, seed), expected)

    def test_reproduces_stage3_logged_results(self):
        """The app gives the same noisy images as Stage 3 (MSE values from experiment_log.md)."""
        logged = {("gaussian", 25): 577.9914, ("salt_and_pepper", 0.05): 1058.0422, ("speckle", 0.04): 636.4267}
        for (noise_type, strength), logged_mse in logged.items():
            with self.subTest(noise_type=noise_type):
                noisy = make_noisy_image(self.image, noise_type, strength, 42)
                self.assertAlmostEqual(mse(self.image, noisy), logged_mse, places=4)

    def test_same_settings_same_pixels(self):
        for noise_type in NOISE_TYPES:
            strength = NOISE_OPTIONS[noise_type].default
            with self.subTest(noise_type=noise_type):
                first = make_noisy_image(self.image, noise_type, strength, 42)
                second = make_noisy_image(self.image, noise_type, strength, 42)
                np.testing.assert_array_equal(first, second)
                self.assertEqual(pixel_fingerprint(first), pixel_fingerprint(second))

    def test_different_seeds_give_different_noise(self):
        for noise_type in NOISE_TYPES:
            strength = NOISE_OPTIONS[noise_type].default
            with self.subTest(noise_type=noise_type):
                a = make_noisy_image(self.image, noise_type, strength, 42)
                b = make_noisy_image(self.image, noise_type, strength, 43)
                self.assertFalse(np.array_equal(a, b))

    def test_reference_is_not_changed(self):
        for noise_type in NOISE_TYPES:
            with self.subTest(noise_type=noise_type):
                make_noisy_image(self.image, noise_type, NOISE_OPTIONS[noise_type].maximum, 42)
                np.testing.assert_array_equal(self.image, self.original_copy)

    def test_works_with_read_only_uploaded_reference(self):
        uploaded = load_uploaded_image("synthetic.png", encode(".png", self.image))
        before = uploaded.grayscale.copy()
        for noise_type in NOISE_TYPES:
            with self.subTest(noise_type=noise_type):
                noisy = make_noisy_image(uploaded.grayscale, noise_type, NOISE_OPTIONS[noise_type].default, 42)
                self.assertFalse(np.shares_memory(noisy, uploaded.grayscale))
        np.testing.assert_array_equal(uploaded.grayscale, before)
        self.assertFalse(uploaded.grayscale.flags.writeable)

    def test_output_shape_dtype_and_read_only(self):
        image = random_gray(height=31, width=17)
        for noise_type in NOISE_TYPES:
            with self.subTest(noise_type=noise_type):
                noisy = make_noisy_image(image, noise_type, NOISE_OPTIONS[noise_type].default, 42)
                self.assertEqual(noisy.shape, (31, 17))
                self.assertEqual(noisy.dtype, np.uint8)
                self.assertFalse(noisy.flags.writeable)

    def test_strength_zero_returns_unchanged_copy(self):
        for noise_type in NOISE_TYPES:
            with self.subTest(noise_type=noise_type):
                noisy = make_noisy_image(self.image, noise_type, 0, 42)
                np.testing.assert_array_equal(noisy, self.image)
                self.assertFalse(np.shares_memory(noisy, self.image))

    def test_salt_and_pepper_changes_exact_number_of_pixels(self):
        noisy = make_noisy_image(self.image, "salt_and_pepper", 0.05, 42)
        selected, salt, pepper = salt_and_pepper_counts(self.image.size, 0.05)
        self.assertEqual(int(np.count_nonzero(noisy != self.image)), selected)  # 3,277 for 256 x 256
        self.assertEqual(int(np.count_nonzero(noisy == 255)), salt)
        self.assertEqual(int(np.count_nonzero(noisy == 0)), pepper)

    def test_invalid_noise_type(self):
        with self.assertRaises(ValueError):
            make_noisy_image(self.image, "poisson", 25, 42)
        with self.assertRaises(TypeError):
            make_noisy_image(self.image, None, 25, 42)

    def test_invalid_strengths_are_rejected_by_src_noise(self):
        value_errors = [
            ("gaussian", -1), ("gaussian", float("nan")), ("gaussian", float("inf")),
            ("salt_and_pepper", -0.01), ("salt_and_pepper", 1.5),
            ("speckle", -0.04), ("speckle", float("nan")),
        ]
        for noise_type, strength in value_errors:
            with self.subTest(noise_type=noise_type, strength=strength):
                with self.assertRaises(ValueError):
                    make_noisy_image(self.image, noise_type, strength, 42)
        for noise_type in NOISE_TYPES:
            for strength in ("25", None, True, [1]):
                with self.subTest(noise_type=noise_type, strength=strength):
                    with self.assertRaises(TypeError):
                        make_noisy_image(self.image, noise_type, strength, 42)

    def test_invalid_seeds(self):
        with self.assertRaisesRegex(ValueError, "seed is required"):
            make_noisy_image(self.image, "gaussian", 25, None)
        for noise_type in NOISE_TYPES:
            with self.subTest(noise_type=noise_type):
                with self.assertRaises(ValueError):
                    make_noisy_image(self.image, noise_type, NOISE_OPTIONS[noise_type].default, -1)
                for bad in (1.5, "42", True, np.bool_(False)):
                    with self.assertRaises(TypeError):
                        make_noisy_image(self.image, noise_type, NOISE_OPTIONS[noise_type].default, bad)

    def test_numpy_integer_seed_accepted(self):
        np.testing.assert_array_equal(
            make_noisy_image(self.image, "gaussian", 25, np.int64(42)),
            make_noisy_image(self.image, "gaussian", 25, 42),
        )

    def test_invalid_reference_images(self):
        for bad in (None, [[1, 2]], np.zeros((4, 4), dtype=np.float64), np.zeros((4, 4, 3), dtype=np.uint8)):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    make_noisy_image(bad, "gaussian", 25, 42)


class TestDescribeNoise(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(describe_noise("gaussian", 25), "Gaussian Noise — σ = 25")
        self.assertEqual(describe_noise("salt_and_pepper", 0.05), "Salt-and-Pepper Noise — 5% of pixels")
        self.assertEqual(describe_noise("salt_and_pepper", 0.07), "Salt-and-Pepper Noise — 7% of pixels")
        self.assertEqual(describe_noise("speckle", 0.04), "Speckle Noise — variance = 0.04")

    def test_unknown_type(self):
        with self.assertRaises(ValueError):
            describe_noise("poisson", 1)


class TestPixelFingerprint(unittest.TestCase):
    def test_same_pixels_same_code(self):
        image = random_gray()
        self.assertEqual(pixel_fingerprint(image), pixel_fingerprint(image.copy()))
        self.assertEqual(len(pixel_fingerprint(image)), 12)

    def test_one_pixel_changes_code(self):
        image = random_gray()
        changed = image.copy()
        changed[5, 5] ^= 1
        self.assertNotEqual(pixel_fingerprint(image), pixel_fingerprint(changed))

    def test_shape_is_part_of_the_code(self):
        image = random_gray(height=4, width=6)
        self.assertNotEqual(pixel_fingerprint(image), pixel_fingerprint(image.reshape(6, 4)))

    def test_non_contiguous_view(self):
        image = random_gray()
        view = image[:, ::2]
        self.assertEqual(pixel_fingerprint(view), pixel_fingerprint(view.copy()))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            pixel_fingerprint(np.zeros((3, 3), dtype=np.float32))


class TestDisplayHelpers(unittest.TestCase):
    def test_small_image_is_not_enlarged(self):
        image = random_gray(height=40, width=60)
        shown = fit_for_display(image)
        np.testing.assert_array_equal(shown, image)
        self.assertFalse(np.shares_memory(shown, image))

    def test_portrait_photo_fits_the_box_and_keeps_its_shape(self):
        image = np.zeros((4000, 3000), dtype=np.uint8)  # portrait phone photo
        shown = fit_for_display(image, max_width=560, max_height=420)
        self.assertEqual(shown.shape, (420, 315))  # 3000/4000 = 315/420

    def test_landscape_photo(self):
        shown = fit_for_display(np.zeros((3000, 4000, 3), dtype=np.uint8), max_width=560, max_height=420)
        self.assertEqual(shown.shape, (420, 560, 3))

    def test_very_wide_image(self):
        shown = fit_for_display(np.zeros((10, 5000), dtype=np.uint8), max_width=560, max_height=420)
        self.assertEqual(shown.shape, (1, 560))

    def test_fit_does_not_change_input(self):
        image = random_gray(height=900, width=700)
        before = image.copy()
        fit_for_display(image)
        np.testing.assert_array_equal(image, before)

    def test_crop_centre(self):
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        crop, (top, left) = crop_square(image, 50, 50, 4)
        # centre = round(0.5 * 9) = round(4.5) = 4 (Python rounds .5 to even); window starts 2 earlier
        self.assertEqual((top, left), (2, 2))
        np.testing.assert_array_equal(crop, image[2:6, 2:6])

    def test_crop_corners_stay_inside(self):
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        self.assertEqual(crop_square(image, 0, 0, 4)[1], (0, 0))
        self.assertEqual(crop_square(image, 100, 100, 4)[1], (6, 6))
        self.assertEqual(crop_square(image, 100, 0, 4)[1], (0, 6))

    def test_crop_larger_than_image(self):
        image = random_gray(height=10, width=30)
        crop, (top, left) = crop_square(image, 50, 50, 64)
        self.assertEqual(crop.shape, (10, 10))
        self.assertEqual(top, 0)

    def test_enlarge_pixels(self):
        image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        big = enlarge_pixels(image, 3)
        self.assertEqual(big.shape, (6, 6))
        np.testing.assert_array_equal(big[:3, :3], 1)
        np.testing.assert_array_equal(big[3:, 3:], 4)
        self.assertEqual(set(np.unique(big)), {1, 2, 3, 4})  # no new values
        np.testing.assert_array_equal(enlarge_pixels(image, 1), image)
        with self.assertRaises(ValueError):
            enlarge_pixels(image, 0)


# --------------------------------------------------------------------------
# Stage 5C: filters
# --------------------------------------------------------------------------
DIRECT_FILTER_FUNCTIONS = {
    "mean": apply_mean_filter,
    "gaussian": apply_gaussian_filter,
    "median": apply_median_filter,
    "bilateral": apply_bilateral_filter,
}


class TestFilterOptions(unittest.TestCase):
    def test_four_filters_in_order(self):
        self.assertEqual(FILTER_TYPES, ("mean", "gaussian", "median", "bilateral"))

    def test_each_filter_uses_the_existing_function(self):
        for filter_type, function in DIRECT_FILTER_FUNCTIONS.items():
            with self.subTest(filter_type=filter_type):
                self.assertIs(FILTER_OPTIONS[filter_type].function, function)

    def test_defaults_are_the_stage4_settings(self):
        """Decision D28: kernel 3, Gaussian sigma 0, bilateral 5 / 50 / 50."""
        self.assertEqual(FILTER_OPTIONS["mean"].defaults, {"kernel_size": 3})
        self.assertEqual(FILTER_OPTIONS["gaussian"].defaults, {"kernel_size": 3, "sigma": 0})
        self.assertEqual(FILTER_OPTIONS["median"].defaults, {"kernel_size": 3})
        self.assertEqual(FILTER_OPTIONS["bilateral"].defaults, {"diameter": 5, "sigma_color": 50, "sigma_space": 50})

    def test_defaults_match_the_src_filters_signatures(self):
        """The parameter names and default values are those of src/filters.py."""
        import inspect

        for filter_type, function in DIRECT_FILTER_FUNCTIONS.items():
            with self.subTest(filter_type=filter_type):
                signature = inspect.signature(function).parameters
                src_defaults = {name: p.default for name, p in signature.items() if name != "image"}
                self.assertEqual(FILTER_OPTIONS[filter_type].defaults, src_defaults)

    def test_offered_values_obey_src_filters_rules(self):
        image = random_gray(height=20, width=20)
        for k in KERNEL_SIZES:
            self.assertEqual(k % 2, 1)
            apply_mean_filter(image, k)  # would raise if invalid
        for d in BILATERAL_DIAMETERS:
            self.assertEqual(d % 2, 1)
            self.assertGreaterEqual(d, 3)
        self.assertTrue({3, 5, 7, 9}.issubset(KERNEL_SIZES))
        self.assertIn(5, BILATERAL_DIAMETERS)
        self.assertEqual(GAUSSIAN_FILTER_SIGMA_RANGE[0], 0.0)  # 0 = auto, allowed by src/filters.py
        self.assertGreater(BILATERAL_SIGMA_RANGE[0], 0)  # bilateral sigmas must be > 0
        low, high, _ = BILATERAL_SIGMA_RANGE
        self.assertTrue(low <= 50 <= high)

    def test_get_filter_option_errors(self):
        for bad in ("box", "Median", "", "gaussian_noise"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "Unknown filter type"):
                    get_filter_option(bad)
        for bad in (None, 3, ["mean"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    get_filter_option(bad)


class TestApplySelectedFilter(unittest.TestCase):
    def setUp(self):
        self.reference = create_synthetic_test_image()
        self.reference_copy = self.reference.copy()
        self.noisy = make_noisy_image(self.reference, "gaussian", 25, 42)  # read-only
        self.noisy_copy = self.noisy.copy()

    def test_exactly_equal_to_direct_calls(self):
        cases = [
            ("mean", {"kernel_size": 3}), ("mean", {"kernel_size": 7}), ("mean", {"kernel_size": 15}),
            ("gaussian", {"kernel_size": 3, "sigma": 0.0}), ("gaussian", {"kernel_size": 5, "sigma": 1.5}),
            ("gaussian", {"kernel_size": 9, "sigma": 0.0}), ("gaussian", {"kernel_size": 3, "sigma": 5.0}),
            ("median", {"kernel_size": 3}), ("median", {"kernel_size": 5}), ("median", {"kernel_size": 11}),
            ("bilateral", {"diameter": 5, "sigma_color": 50, "sigma_space": 50}),
            ("bilateral", {"diameter": 9, "sigma_color": 10, "sigma_space": 150}),
            ("bilateral", {"diameter": 3, "sigma_color": 200, "sigma_space": 1}),
        ]
        for filter_type, parameters in cases:
            with self.subTest(filter_type=filter_type, parameters=parameters):
                expected = DIRECT_FILTER_FUNCTIONS[filter_type](self.noisy, **parameters)
                np.testing.assert_array_equal(apply_selected_filter(self.noisy, filter_type, parameters), expected)

    def test_parameters_are_forwarded(self):
        """Different parameter values must reach src/filters.py and give different results."""
        pairs = [
            ("mean", {"kernel_size": 3}, {"kernel_size": 5}),
            ("gaussian", {"kernel_size": 5, "sigma": 0.0}, {"kernel_size": 5, "sigma": 3.0}),
            ("gaussian", {"kernel_size": 3, "sigma": 1.0}, {"kernel_size": 7, "sigma": 1.0}),
            ("median", {"kernel_size": 3}, {"kernel_size": 5}),
            ("bilateral", {"diameter": 5, "sigma_color": 50, "sigma_space": 50}, {"diameter": 9, "sigma_color": 50, "sigma_space": 50}),
            ("bilateral", {"diameter": 5, "sigma_color": 10, "sigma_space": 50}, {"diameter": 5, "sigma_color": 150, "sigma_space": 50}),
            ("bilateral", {"diameter": 9, "sigma_color": 50, "sigma_space": 1}, {"diameter": 9, "sigma_color": 50, "sigma_space": 100}),
        ]
        for filter_type, first, second in pairs:
            with self.subTest(filter_type=filter_type, first=first, second=second):
                a = apply_selected_filter(self.noisy, filter_type, first)
                b = apply_selected_filter(self.noisy, filter_type, second)
                self.assertFalse(np.array_equal(a, b))

    def test_parameters_reach_the_function_unchanged(self):
        calls = []

        def spy(image, **kwargs):
            calls.append((image, kwargs))
            return image.copy()

        parameters = {"diameter": 7, "sigma_color": 12.5, "sigma_space": 99}
        option = FILTER_OPTIONS["bilateral"]
        with mock.patch.dict(FILTER_OPTIONS, {"bilateral": option.__class__(**{**option.__dict__, "function": spy})}):
            apply_selected_filter(self.noisy, "bilateral", parameters)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], self.noisy)  # the noisy image itself, not the reference
        self.assertEqual(calls[0][1], parameters)

    def test_filters_differ_from_each_other(self):
        results = {
            f: pixel_fingerprint(apply_selected_filter(self.noisy, f, FILTER_OPTIONS[f].defaults)) for f in FILTER_TYPES
        }
        self.assertEqual(len(set(results.values())), 4)

    def test_noisy_and_reference_unchanged(self):
        for filter_type in FILTER_TYPES:
            with self.subTest(filter_type=filter_type):
                apply_selected_filter(self.noisy, filter_type, FILTER_OPTIONS[filter_type].defaults)
                np.testing.assert_array_equal(self.noisy, self.noisy_copy)
                np.testing.assert_array_equal(self.reference, self.reference_copy)
        self.assertFalse(self.noisy.flags.writeable)

    def test_output_shape_dtype_read_only_and_new(self):
        noisy = make_noisy_image(random_gray(height=37, width=23), "salt_and_pepper", 0.1, 1)
        for filter_type in FILTER_TYPES:
            with self.subTest(filter_type=filter_type):
                filtered = apply_selected_filter(noisy, filter_type, FILTER_OPTIONS[filter_type].defaults)
                self.assertEqual(filtered.shape, (37, 23))
                self.assertEqual(filtered.dtype, np.uint8)
                self.assertFalse(filtered.flags.writeable)
                self.assertFalse(np.shares_memory(filtered, noisy))

    def test_kernel_size_one_is_a_copy(self):
        filtered = apply_selected_filter(self.noisy, "median", {"kernel_size": 1})
        np.testing.assert_array_equal(filtered, self.noisy)
        self.assertFalse(np.shares_memory(filtered, self.noisy))

    def test_median_removes_most_salt_and_pepper_in_the_flat_patch(self):
        """Known Stage 4 behaviour, now through the frontend helper (flat patch rows 141-229, cols 20-114)."""
        reference = create_synthetic_test_image()
        noisy = make_noisy_image(reference, "salt_and_pepper", 0.05, 42)
        filtered = apply_selected_filter(noisy, "median", {"kernel_size": 3})
        patch = (slice(141, 230), slice(20, 115))
        self.assertLess(mse(reference[patch], filtered[patch]), 0.02 * mse(reference[patch], noisy[patch]))

    def test_invalid_filter_type(self):
        with self.assertRaises(ValueError):
            apply_selected_filter(self.noisy, "box", {"kernel_size": 3})
        with self.assertRaises(TypeError):
            apply_selected_filter(self.noisy, None, {"kernel_size": 3})

    def test_parameters_must_be_a_dict_with_exact_names(self):
        with self.assertRaises(TypeError):
            apply_selected_filter(self.noisy, "mean", 3)
        with self.assertRaises(TypeError):
            apply_selected_filter(self.noisy, "mean", [("kernel_size", 3)])
        bad = [
            ("mean", {}),  # missing
            ("mean", {"kernel_size": 3, "sigma": 1}),  # unknown extra
            ("gaussian", {"kernel_size": 3}),  # missing sigma
            ("median", {"ksize": 3}),  # wrong name
            ("bilateral", {"diameter": 5, "sigma_color": 50}),  # missing sigma_space
            ("bilateral", {"kernel_size": 5, "sigma_color": 50, "sigma_space": 50}),  # mean's name
        ]
        for filter_type, parameters in bad:
            with self.subTest(filter_type=filter_type, parameters=parameters):
                with self.assertRaisesRegex(ValueError, "needs exactly these parameters"):
                    apply_selected_filter(self.noisy, filter_type, parameters)

    def test_invalid_kernel_sizes_rejected_by_src_filters(self):
        for filter_type in ("mean", "median", "gaussian"):
            extra = {"sigma": 0.0} if filter_type == "gaussian" else {}
            for bad in (0, -3, 2, 4, 10):
                with self.subTest(filter_type=filter_type, bad=bad):
                    with self.assertRaises(ValueError):
                        apply_selected_filter(self.noisy, filter_type, {"kernel_size": bad, **extra})
            for bad in (3.0, "3", None, True):
                with self.subTest(filter_type=filter_type, bad=bad):
                    with self.assertRaises(TypeError):
                        apply_selected_filter(self.noisy, filter_type, {"kernel_size": bad, **extra})

    def test_invalid_gaussian_sigma(self):
        for bad in (-0.1, -5, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    apply_selected_filter(self.noisy, "gaussian", {"kernel_size": 3, "sigma": bad})
        for bad in ("1", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    apply_selected_filter(self.noisy, "gaussian", {"kernel_size": 3, "sigma": bad})

    def test_invalid_bilateral_parameters(self):
        good = {"diameter": 5, "sigma_color": 50, "sigma_space": 50}
        value_errors = [
            {"diameter": 1}, {"diameter": 4}, {"diameter": 0}, {"diameter": -5},
            {"sigma_color": 0}, {"sigma_color": -10}, {"sigma_space": 0}, {"sigma_space": float("nan")},
        ]
        for change in value_errors:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    apply_selected_filter(self.noisy, "bilateral", {**good, **change})
        for change in ({"diameter": 5.0}, {"diameter": True}, {"sigma_color": "50"}, {"sigma_space": None}):
            with self.subTest(change=change):
                with self.assertRaises(TypeError):
                    apply_selected_filter(self.noisy, "bilateral", {**good, **change})

    def test_invalid_images(self):
        for bad in (None, np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4, 3), dtype=np.uint8)):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    apply_selected_filter(bad, "mean", {"kernel_size": 3})

    def test_deterministic(self):
        for filter_type in FILTER_TYPES:
            with self.subTest(filter_type=filter_type):
                a = apply_selected_filter(self.noisy, filter_type, FILTER_OPTIONS[filter_type].defaults)
                b = apply_selected_filter(self.noisy, filter_type, FILTER_OPTIONS[filter_type].defaults)
                np.testing.assert_array_equal(a, b)


class TestDescribeFilter(unittest.TestCase):
    def test_short_labels(self):
        self.assertEqual(describe_filter("mean", {"kernel_size": 3}, short=True), "Mean Filter — 3 × 3")
        self.assertEqual(describe_filter("median", {"kernel_size": 5}, short=True), "Median Filter — 5 × 5")
        self.assertEqual(
            describe_filter("gaussian", {"kernel_size": 3, "sigma": 0.0}, short=True),
            "Gaussian Filter — 3 × 3, filter σ = auto",
        )
        self.assertEqual(
            describe_filter("bilateral", {"diameter": 5, "sigma_color": 50, "sigma_space": 50}, short=True),
            "Bilateral Filter — d = 5",
        )

    def test_long_labels(self):
        self.assertEqual(
            describe_filter("bilateral", {"diameter": 5, "sigma_color": 50, "sigma_space": 50}),
            "Bilateral Filter — diameter = 5, sigma color = 50, sigma space = 50",
        )
        self.assertEqual(
            describe_filter("gaussian", {"kernel_size": 5, "sigma": 1.5}), "Gaussian Filter — 5 × 5, filter σ = 1.5"
        )

    def test_filter_and_noise_labels_are_distinct(self):
        noise = describe_noise("gaussian", 25)
        filt = describe_filter("gaussian", {"kernel_size": 3, "sigma": 1.0})
        self.assertIn("Noise", noise)
        self.assertIn("Filter", filt)
        self.assertIn("filter σ", filt)
        self.assertNotIn("Filter", noise)

    def test_unknown_type(self):
        with self.assertRaises(ValueError):
            describe_filter("box", {"kernel_size": 3})


class TestThreeImageCrops(unittest.TestCase):
    def test_same_region_for_reference_noisy_and_filtered(self):
        reference = create_synthetic_test_image()
        noisy = make_noisy_image(reference, "gaussian", 25, 42)
        filtered = apply_selected_filter(noisy, "median", {"kernel_size": 3})
        before = [reference.copy(), noisy.copy(), filtered.copy()]
        for x, y, size in [(50, 50, 64), (0, 100, 16), (100, 0, 128), (37, 81, 32)]:
            with self.subTest(x=x, y=y, size=size):
                results = [crop_square(image, x, y, size) for image in (reference, noisy, filtered)]
                regions = {region for _, region in results}
                self.assertEqual(len(regions), 1)
                top, left = regions.pop()
                for (crop, _), image in zip(results, (reference, noisy, filtered)):
                    np.testing.assert_array_equal(crop, image[top : top + size, left : left + size])
        for image, copy in zip((reference, noisy, filtered), before):
            np.testing.assert_array_equal(image, copy)


# --------------------------------------------------------------------------
# Stage 5D: MSE / PSNR evaluation, four-filter comparison, difference images
# --------------------------------------------------------------------------
import math  # noqa: E402

# The illustrative hand-calculation example (NOT a measurement on a real photo):
HAND_ORIGINAL = np.array([[100, 100, 100, 100]], dtype=np.uint8)
HAND_NOISY = np.array([[120, 80, 110, 90]], dtype=np.uint8)
HAND_FILTERED = np.array([[105, 95, 105, 95]], dtype=np.uint8)

DEFAULT_PARAMETERS = {name: dict(FILTER_OPTIONS[name].defaults) for name in FILTER_TYPES}


def assert_no_nan(test, value):
    """Recursively check that no float in a result is NaN."""
    if isinstance(value, float):
        test.assertFalse(math.isnan(value))
    elif isinstance(value, dict):
        for item in value.values():
            assert_no_nan(test, item)
    elif isinstance(value, list):
        for item in value:
            assert_no_nan(test, item)
    elif isinstance(value, str):
        test.assertNotIn("nan", value.lower())


class TestHandCalculatedExample(unittest.TestCase):
    """MSE = (400 + 400 + 100 + 100) / 4 = 250 and (25 * 4) / 4 = 25;
    PSNR = 10 log10(255^2 / MSE); change = 10 log10(250 / 25) = +10 dB."""

    def test_improvement(self):
        result = evaluate_image_quality(HAND_ORIGINAL, HAND_NOISY, HAND_FILTERED)
        self.assertEqual(result["noisy"]["mse"], 250.0)
        self.assertEqual(result["filtered"]["mse"], 25.0)
        self.assertAlmostEqual(result["noisy"]["psnr"], 10 * math.log10(255**2 / 250), places=10)
        self.assertAlmostEqual(result["noisy"]["psnr"], 24.1514, places=4)
        self.assertAlmostEqual(result["filtered"]["psnr"], 34.1514, places=4)
        self.assertEqual(result["mse_change"], -225.0)
        self.assertAlmostEqual(result["psnr_change"], 10.0, places=10)
        self.assertEqual(result["psnr_change_text"], "+10.00 dB")
        self.assertIsNone(result["psnr_change_note"])
        self.assertEqual(result["verdict"], "closer")

    def test_worse_result(self):
        """MSE 50 -> 450: the 'filter' made the error 9 times larger (-10 log10 9 = -9.54 dB)."""
        noisy = np.array([[110, 90, 100, 100]], dtype=np.uint8)
        filtered = np.array([[130, 70, 100, 100]], dtype=np.uint8)
        result = evaluate_image_quality(HAND_ORIGINAL, noisy, filtered)
        self.assertEqual(result["noisy"]["mse"], 50.0)
        self.assertEqual(result["filtered"]["mse"], 450.0)
        self.assertEqual(result["mse_change"], 400.0)
        self.assertAlmostEqual(result["psnr_change"], -10 * math.log10(9), places=10)
        self.assertEqual(result["psnr_change_text"], "−9.54 dB")
        self.assertEqual(result["verdict"], "farther")

    def test_no_change(self):
        result = evaluate_image_quality(HAND_ORIGINAL, HAND_NOISY, HAND_NOISY.copy())
        self.assertEqual(result["mse_change"], 0.0)
        self.assertEqual(result["psnr_change"], 0.0)
        self.assertEqual(result["psnr_change_text"], "0.00 dB")
        self.assertEqual(result["verdict"], "same")


class TestEvaluateImageQuality(unittest.TestCase):
    def setUp(self):
        self.reference = create_synthetic_test_image()
        self.noisy = make_noisy_image(self.reference, "gaussian", 25, 42)
        self.filtered = apply_selected_filter(self.noisy, "bilateral", DEFAULT_PARAMETERS["bilateral"])

    def test_equal_to_direct_metric_calls(self):
        result = evaluate_image_quality(self.reference, self.noisy, self.filtered)
        self.assertEqual(result["noisy"]["mse"], mse(self.reference, self.noisy))
        self.assertEqual(result["noisy"]["psnr"], psnr(self.reference, self.noisy))
        self.assertEqual(result["filtered"]["mse"], mse(self.reference, self.filtered))
        self.assertEqual(result["filtered"]["psnr"], psnr(self.reference, self.filtered))
        self.assertAlmostEqual(result["mse_change"], mse(self.reference, self.filtered) - mse(self.reference, self.noisy))
        self.assertAlmostEqual(result["psnr_change"], psnr(self.reference, self.filtered) - psnr(self.reference, self.noisy))

    def test_matches_stage4_logged_values(self):
        """E002 (noisy) and E013 (bilateral 5/50/50) in experiment_log.md."""
        result = evaluate_image_quality(self.reference, self.noisy, self.filtered)
        self.assertAlmostEqual(result["noisy"]["mse"], 577.9914, places=4)
        self.assertAlmostEqual(result["noisy"]["psnr"], 20.5116, places=4)
        self.assertAlmostEqual(result["filtered"]["mse"], 87.8501, places=4)
        self.assertAlmostEqual(result["filtered"]["psnr"], 28.6934, places=4)
        self.assertAlmostEqual(result["psnr_change"], 8.1818, places=4)

    def test_reference_is_the_original_not_the_noisy_image(self):
        result = evaluate_image_quality(self.reference, self.noisy, self.filtered)
        self.assertNotEqual(result["filtered"]["mse"], mse(self.noisy, self.filtered))

    def test_inputs_unchanged(self):
        copies = [self.reference.copy(), self.noisy.copy(), self.filtered.copy()]
        evaluate_image_quality(self.reference, self.noisy, self.filtered)
        for image, copy in zip((self.reference, self.noisy, self.filtered), copies):
            np.testing.assert_array_equal(image, copy)

    def test_identical_images(self):
        result = evaluate_image_quality(HAND_ORIGINAL, HAND_ORIGINAL.copy(), HAND_ORIGINAL.copy())
        self.assertEqual(result["noisy"]["mse"], 0.0)
        self.assertEqual(result["noisy"]["psnr"], math.inf)
        self.assertIsNone(result["psnr_change"])  # not inf - inf = NaN
        self.assertEqual(result["psnr_change_text"], "no change (both ∞)")
        self.assertEqual(result["verdict"], "same")
        assert_no_nan(self, result)

    def test_filter_reaches_the_original(self):
        result = evaluate_image_quality(HAND_ORIGINAL, HAND_NOISY, HAND_ORIGINAL.copy())
        self.assertEqual(result["filtered"]["psnr"], math.inf)
        self.assertIsNone(result["psnr_change"])
        self.assertEqual(result["psnr_change_text"], "now identical (∞)")
        self.assertEqual(result["verdict"], "closer")
        assert_no_nan(self, result)

    def test_noise_free_input_made_worse(self):
        result = evaluate_image_quality(HAND_ORIGINAL, HAND_ORIGINAL.copy(), HAND_FILTERED)
        self.assertEqual(result["noisy"]["psnr"], math.inf)
        self.assertIsNone(result["psnr_change"])
        self.assertEqual(result["psnr_change_text"], "worse (was ∞)")
        self.assertEqual(result["verdict"], "farther")
        assert_no_nan(self, result)

    def test_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "filtered image"):
            evaluate_image_quality(self.reference, self.noisy, self.filtered[:-1])
        with self.assertRaisesRegex(ValueError, "noisy image"):
            evaluate_image_quality(self.reference, self.noisy[:, :-1], self.filtered)

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            evaluate_image_quality(self.reference, self.noisy.tolist(), self.filtered)
        with self.assertRaises(ValueError):
            evaluate_image_quality(self.reference, self.noisy, self.filtered.astype(np.float64))
        with self.assertRaises(TypeError):
            evaluate_image_quality(None, self.noisy, self.filtered)


class TestChangeHelpers(unittest.TestCase):
    def test_psnr_change_cases(self):
        self.assertEqual(psnr_change(20.0, 28.5), (8.5, "+8.50 dB", None))
        self.assertEqual(psnr_change(20.0, 17.25)[:2], (-2.75, "−2.75 dB"))
        for before, after, text in [(math.inf, math.inf, "no change (both ∞)"),
                                    (20.0, math.inf, "now identical (∞)"),
                                    (math.inf, 20.0, "worse (was ∞)")]:
            change, short, note = psnr_change(before, after)
            self.assertIsNone(change)
            self.assertEqual(short, text)
            self.assertIn("∞", note)

    def test_judge_change(self):
        self.assertEqual(judge_change(10.0, 5.0), "closer")
        self.assertEqual(judge_change(5.0, 10.0), "farther")
        self.assertEqual(judge_change(5.0, 5.0), "same")
        self.assertEqual(judge_change(0.0, 0.0), "same")

    def test_format_signed_never_shows_minus_zero(self):
        self.assertEqual(format_signed(0.0), "0.00")
        self.assertEqual(format_signed(-0.0), "0.00")
        self.assertEqual(format_signed(8.184), "+8.18")
        self.assertEqual(format_signed(-490.137), "−490.14")
        self.assertEqual(format_signed(0.001), "+<0.01")
        self.assertEqual(format_signed(-0.004), "−<0.01")
        self.assertEqual(format_signed(1.5, "dB"), "+1.50 dB")

    def test_format_mse_and_psnr(self):
        self.assertEqual(format_mse(1216.9823), "1,216.98")
        self.assertEqual(format_mse(0.0), "0.00")
        self.assertEqual(format_psnr(20.5116), "20.51")
        self.assertEqual(format_psnr(math.inf), "∞")


class TestCompareAllFilters(unittest.TestCase):
    def setUp(self):
        self.reference = create_synthetic_test_image()
        self.noisy = make_noisy_image(self.reference, "salt_and_pepper", 0.05, 42)
        self.parameters = {
            "mean": {"kernel_size": 5},
            "gaussian": {"kernel_size": 5, "sigma": 1.2},
            "median": {"kernel_size": 3},
            "bilateral": {"diameter": 7, "sigma_color": 30, "sigma_space": 20},
        }

    def test_rows_equal_direct_filter_and_metric_calls(self):
        result = compare_all_filters(self.reference, self.noisy, self.parameters)
        rows = result["rows"]
        self.assertEqual([r["method"] for r in rows], ["none", "mean", "gaussian", "median", "bilateral"])
        self.assertEqual(rows[0]["label"], "No filtering")
        self.assertEqual(rows[0]["mse"], mse(self.reference, self.noisy))
        self.assertEqual(rows[0]["psnr_change_text"], "baseline")
        direct = {
            "mean": apply_mean_filter(self.noisy, 5),
            "gaussian": apply_gaussian_filter(self.noisy, 5, 1.2),
            "median": apply_median_filter(self.noisy, 3),
            "bilateral": apply_bilateral_filter(self.noisy, 7, 30, 20),
        }
        for row in rows[1:]:
            with self.subTest(method=row["method"]):
                expected = direct[row["method"]]
                np.testing.assert_array_equal(row["image"], expected)
                self.assertEqual(row["mse"], mse(self.reference, expected))
                self.assertEqual(row["psnr"], psnr(self.reference, expected))
                self.assertAlmostEqual(row["psnr_change"], psnr(self.reference, expected) - psnr(self.reference, self.noisy))
                self.assertEqual(row["parameters"], self.parameters[row["method"]])

    def test_parameter_text(self):
        rows = compare_all_filters(self.reference, self.noisy, self.parameters)["rows"]
        self.assertEqual(
            [r["parameter_text"] for r in rows],
            ["—", "5 × 5", "5 × 5, filter σ = 1.2", "3 × 3", "diameter = 7, sigma color = 30, sigma space = 20"],
        )

    def test_every_filter_gets_the_same_noisy_array(self):
        received = []
        real = apply_selected_filter

        def spy(noisy_image, filter_type, parameters):
            received.append((noisy_image, filter_type, dict(parameters)))
            return real(noisy_image, filter_type, parameters)

        with mock.patch("frontend.image_processing.apply_selected_filter", side_effect=spy):
            compare_all_filters(self.reference, self.noisy, self.parameters)
        self.assertEqual([r[1] for r in received], list(FILTER_TYPES))
        for noisy_image, filter_type, parameters in received:
            self.assertIs(noisy_image, self.noisy)  # the very same array, never regenerated
            self.assertEqual(parameters, self.parameters[filter_type])

    def test_best_method_is_the_lowest_mse(self):
        result = compare_all_filters(self.reference, self.noisy, self.parameters)
        lowest = min(r["mse"] for r in result["rows"])
        self.assertEqual(result["best_methods"], [r["method"] for r in result["rows"] if r["mse"] == lowest])
        highest_psnr = max(r["psnr"] for r in result["rows"])
        for method in result["best_methods"]:
            [row] = [r for r in result["rows"] if r["method"] == method]
            self.assertEqual(row["psnr"], highest_psnr)
        self.assertFalse(result["baseline_is_best"])

    def test_no_filter_improves_on_fine_texture(self):
        """Random texture with very weak noise: every filter destroys more detail than it removes noise."""
        reference = random_gray(height=64, width=64, seed=5)
        noisy = make_noisy_image(reference, "gaussian", 2, 1)
        result = compare_all_filters(reference, noisy, DEFAULT_PARAMETERS)
        self.assertEqual(result["best_methods"], ["none"])
        self.assertTrue(result["baseline_is_best"])
        for row in result["rows"][1:]:
            self.assertGreater(row["mse"], result["rows"][0]["mse"])
            self.assertLess(row["psnr_change"], 0)

    def test_all_identical_images_tie_and_no_nan(self):
        reference = np.full((20, 30), 128, dtype=np.uint8)
        noisy = make_noisy_image(reference, "gaussian", 0, 42)  # strength 0: unchanged copy
        result = compare_all_filters(reference, noisy, DEFAULT_PARAMETERS)
        self.assertEqual(result["best_methods"], ["none", *FILTER_TYPES])
        self.assertTrue(result["baseline_is_best"])
        for row in result["rows"][1:]:
            self.assertEqual(row["psnr"], math.inf)
            self.assertEqual(row["psnr_change_text"], "no change (both ∞)")
        assert_no_nan(self, [{k: v for k, v in r.items() if k != "image"} for r in result["rows"]])

    def test_median_reaching_the_original_exactly(self):
        reference = np.full((64, 64), 128, dtype=np.uint8)
        noisy = make_noisy_image(reference, "salt_and_pepper", 0.02, 42)
        result = compare_all_filters(reference, noisy, DEFAULT_PARAMETERS)
        [median] = [r for r in result["rows"] if r["method"] == "median"]
        self.assertEqual(median["mse"], 0.0)
        self.assertEqual(median["psnr_change_text"], "now identical (∞)")
        self.assertEqual(result["best_methods"], ["median"])

    def test_inputs_unchanged_and_deterministic(self):
        reference_copy, noisy_copy = self.reference.copy(), self.noisy.copy()
        first = compare_all_filters(self.reference, self.noisy, self.parameters)
        second = compare_all_filters(self.reference, self.noisy, self.parameters)
        np.testing.assert_array_equal(self.reference, reference_copy)
        np.testing.assert_array_equal(self.noisy, noisy_copy)
        for a, b in zip(first["rows"], second["rows"]):
            np.testing.assert_array_equal(a["image"], b["image"])
            self.assertEqual(a["mse"], b["mse"])
        self.assertIs(first["rows"][0]["image"], self.noisy)

    def test_invalid_filter_parameters(self):
        with self.assertRaises(TypeError):
            compare_all_filters(self.reference, self.noisy, [("mean", {"kernel_size": 3})])
        missing = {k: v for k, v in DEFAULT_PARAMETERS.items() if k != "median"}
        with self.assertRaisesRegex(ValueError, "exactly these keys"):
            compare_all_filters(self.reference, self.noisy, missing)
        with self.assertRaisesRegex(ValueError, "exactly these keys"):
            compare_all_filters(self.reference, self.noisy, {**DEFAULT_PARAMETERS, "box": {"kernel_size": 3}})
        with self.assertRaises(ValueError):
            compare_all_filters(self.reference, self.noisy, {**DEFAULT_PARAMETERS, "mean": {"kernel_size": 4}})

    def test_invalid_images(self):
        with self.assertRaises(ValueError):
            compare_all_filters(self.reference, self.noisy[:-2], DEFAULT_PARAMETERS)
        with self.assertRaises((TypeError, ValueError)):
            compare_all_filters(self.reference, self.noisy.astype(np.int16), DEFAULT_PARAMETERS)

    def test_describe_filter_parameters(self):
        self.assertEqual(describe_filter_parameters("mean", {"kernel_size": 9}), "9 × 9")
        self.assertEqual(describe_filter_parameters("gaussian", {"kernel_size": 3, "sigma": 0}), "3 × 3, filter σ = auto")
        with self.assertRaises(ValueError):
            describe_filter_parameters("box", {"kernel_size": 3})


class TestDifferenceImages(unittest.TestCase):
    def test_exact_absolute_differences(self):
        a = random_gray(seed=11)
        b = random_gray(seed=12)
        expected = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
        np.testing.assert_array_equal(make_difference_image(a, b), expected)
        np.testing.assert_array_equal(make_difference_image(b, a), expected)  # symmetric

    def test_no_uint8_wrap_around(self):
        a = np.array([[0, 255, 0, 255, 10]], dtype=np.uint8)
        b = np.array([[255, 0, 0, 255, 12]], dtype=np.uint8)
        np.testing.assert_array_equal(make_difference_image(a, b), [[255, 255, 0, 0, 2]])
        self.assertEqual(int((a - b)[0, 4]), 254)  # what plain uint8 subtraction would give

    def test_result_new_read_only_and_inputs_unchanged(self):
        a, b = random_gray(seed=1), random_gray(seed=2)
        copies = (a.copy(), b.copy())
        diff = make_difference_image(a, b)
        self.assertFalse(diff.flags.writeable)
        self.assertFalse(np.shares_memory(diff, a) or np.shares_memory(diff, b))
        np.testing.assert_array_equal(a, copies[0])
        np.testing.assert_array_equal(b, copies[1])

    def test_invalid_inputs(self):
        a = random_gray()
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            make_difference_image(a, a[:-1])
        with self.assertRaisesRegex(ValueError, "second image"):
            make_difference_image(a, a.astype(np.float32))
        with self.assertRaises(TypeError):
            make_difference_image(None, a)

    def test_enhancement_uses_one_shared_factor(self):
        d1 = np.array([[0, 10, 51]], dtype=np.uint8)
        d2 = np.array([[0, 5, 20]], dtype=np.uint8)
        (e1, e2), factor = enhance_differences([d1, d2])
        self.assertAlmostEqual(factor, 5.0)  # 255 / 51
        np.testing.assert_array_equal(e1, [[0, 50, 255]])
        np.testing.assert_array_equal(e2, [[0, 25, 100]])
        np.testing.assert_array_equal(d1, [[0, 10, 51]])  # inputs unchanged

    def test_enhancement_of_zero_difference(self):
        zero = np.zeros((3, 3), dtype=np.uint8)
        (enhanced,), factor = enhance_differences([zero])
        self.assertEqual(factor, 1.0)
        np.testing.assert_array_equal(enhanced, zero)

    def test_enhancement_does_not_affect_metrics(self):
        reference = create_synthetic_test_image()
        noisy = make_noisy_image(reference, "speckle", 0.04, 42)
        filtered = apply_selected_filter(noisy, "median", {"kernel_size": 3})
        before = evaluate_image_quality(reference, noisy, filtered)
        diffs = [make_difference_image(reference, noisy), make_difference_image(noisy, filtered)]
        enhance_differences(diffs)
        after = evaluate_image_quality(reference, noisy, filtered)
        self.assertEqual(before["noisy"], after["noisy"])
        self.assertEqual(before["filtered"], after["filtered"])

    def test_summary(self):
        diff = np.array([[0, 0, 7], [0, 255, 0]], dtype=np.uint8)
        self.assertEqual(summarize_difference(diff), {"changed_pixels": 2, "changed_percent": 100 * 2 / 6, "largest": 255})

    def test_noise_difference_counts_salt_and_pepper(self):
        reference = create_synthetic_test_image()
        noisy = make_noisy_image(reference, "salt_and_pepper", 0.05, 42)
        facts = summarize_difference(make_difference_image(reference, noisy))
        self.assertEqual(facts["changed_pixels"], salt_and_pepper_counts(reference.size, 0.05)[0])  # 3,277


if __name__ == "__main__":
    unittest.main()
