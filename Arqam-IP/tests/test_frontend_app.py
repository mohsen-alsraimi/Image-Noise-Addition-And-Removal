"""Smoke tests for the Streamlit app (frontend/app.py), Stage 5A.

Streamlit's own testing tool (streamlit.testing.v1.AppTest) runs the app
script without a browser or web server, and lets us "upload" a file.
These tests only check that our page reacts correctly: they do not test
Streamlit itself. The upload logic is tested in detail in
test_frontend_image_processing.py.

Run from the project root:  python -m unittest tests.test_frontend_app -v
"""

import re
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from frontend.image_processing import pixel_fingerprint
from src.filters import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_mean_filter,
    apply_median_filter,
)
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise, add_speckle_noise
from src.synthetic_image import create_synthetic_test_image

APP_PATH = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")
TIMEOUT_SECONDS = 60


def encode(extension, array):
    ok, encoded = cv2.imencode(extension, array)
    assert ok
    return encoded.tobytes()


def run_app(upload=None):
    """Run the app once; optionally upload (file_name, bytes) and run again."""
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
    app.run()
    if upload is not None:
        file_name, data = upload
        app.sidebar.file_uploader[0].upload(file_name, data).run()
    return app


def all_text(elements):
    return " ".join(str(element.value) for element in elements)


class TestAppWithoutUpload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = run_app()

    def test_runs_without_exception(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)

    def test_title(self):
        self.assertEqual(self.app.title[0].value, "Image Noise Lab — Noise Addition and Removal")

    def test_uploader_accepts_only_png_and_jpeg(self):
        uploader = self.app.sidebar.file_uploader[0]
        self.assertEqual(sorted(t.lstrip(".") for t in uploader.allowed_type), ["jpeg", "jpg", "png"])

    def test_asks_user_to_upload(self):
        self.assertIn("Upload a PNG or JPEG image", all_text(self.app.info))


class TestAppWithValidUploads(unittest.TestCase):
    def test_colour_png(self):
        image = np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8)
        app = run_app(("colour photo.png", encode(".png", image)))
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn("colour", all_text(app.info))
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Width"], "60 px")
        self.assertEqual(metrics["Height"], "40 px")
        self.assertEqual(metrics["Pixels"], "2,400")

    def test_grayscale_jpeg(self):
        image = np.random.default_rng(1).integers(0, 256, (30, 20), dtype=np.uint8)
        app = run_app(("gray.jpg", encode(".jpg", image)))
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn("already grayscale", all_text(app.info))

    def test_transparent_png_shows_warning(self):
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        app = run_app(("logo.png", encode(".png", rgba)))
        self.assertEqual(len(app.exception), 0)
        self.assertIn("transparent", all_text(app.warning))

    def test_large_image_shows_warning(self):
        large = np.zeros((2000, 2100), dtype=np.uint8)  # 4.2 megapixels (compresses to a tiny PNG)
        app = run_app(("large.png", encode(".png", large)))
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn("large image", all_text(app.warning))


class TestAppWithInvalidUploads(unittest.TestCase):
    def check_rejected(self, file_name, data, expected_text):
        app = run_app((file_name, data))
        self.assertEqual(len(app.exception), 0, "the app must not crash")
        self.assertEqual(len(app.error), 1)
        self.assertIn("cannot be used", app.error[0].value)
        self.assertIn(expected_text, app.error[0].value)
        self.assertEqual(len(app.metric), 0, "no image information for a rejected file")

    def test_corrupted_png(self):
        self.check_rejected("broken.png", b"\x89PNG\r\n\x1a\n" + b"x" * 100, "could not be decoded")

    def test_empty_file(self):
        self.check_rejected("empty.png", b"", "empty")

    def test_text_file_renamed_to_png(self):
        self.check_rejected("notes.png", b"hello, this is not an image", "not a PNG or JPEG")

    def test_bmp_file_renamed_to_png(self):
        # A file named "picture.bmp" never reaches our code: Streamlit itself
        # refuses extensions not listed in `type=` (observed with AppTest).
        # A BMP renamed to .png passes that check, so our content check must catch it.
        bmp = encode(".bmp", np.zeros((5, 5), dtype=np.uint8))
        self.check_rejected("picture.png", bmp, "not a PNG or JPEG")


# --------------------------------------------------------------------------
# Stage 5B: noise controls
# --------------------------------------------------------------------------
SYNTHETIC = create_synthetic_test_image()  # 256 x 256, the Stage 3 test image
SYNTHETIC_PNG = encode(".png", SYNTHETIC)


def fingerprint_shown(app):
    """Read the pixel fingerprint of the noisy image from the page."""
    for caption in app.caption:
        match = re.search(r"Pixel fingerprint of the noisy image: `([0-9a-f]{12})`", caption.value)
        if match:
            return match.group(1)
    raise AssertionError("No pixel fingerprint shown on the page.")


def expected_fingerprint(function, strength, seed=42):
    return pixel_fingerprint(function(SYNTHETIC, strength, seed=seed))


def settings_text(app):
    return " ".join(m.value for m in app.markdown if m.value.startswith("**Settings:**"))


class TestNoiseControlsVisibility(unittest.TestCase):
    def test_no_noise_controls_before_upload(self):
        app = run_app()
        self.assertEqual(len(app.sidebar.radio), 0)
        self.assertEqual(len(app.number_input), 0)

    def test_no_noise_controls_after_invalid_upload(self):
        app = run_app(("broken.png", b"\x89PNG\r\n\x1a\n" + b"x" * 100))
        self.assertEqual(len(app.sidebar.radio), 0)
        self.assertEqual(len(app.exception), 0)

    def test_defaults_after_upload(self):
        app = run_app(("synthetic.png", SYNTHETIC_PNG))
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn("2. Add Noise", [h.value for h in app.sidebar.header])
        self.assertEqual(app.radio(key="noise_type").value, "gaussian")
        self.assertEqual(app.radio(key="noise_type").options, ["Gaussian Noise", "Salt-and-Pepper Noise", "Speckle Noise"])
        self.assertEqual(app.slider(key="gaussian_sigma").value, 25)
        self.assertEqual(app.number_input(key="seed").value, 42)
        self.assertEqual(app.number_input(key="seed").label, "Random Seed")
        self.assertIn("Gaussian Noise — σ = 25", settings_text(app))
        self.assertIn("seed = 42", settings_text(app))

    def test_only_the_selected_strength_slider_is_shown(self):
        app = run_app(("synthetic.png", SYNTHETIC_PNG))
        strength_keys = {"gaussian_sigma", "salt_and_pepper_amount", "speckle_variance"}
        for noise_type, expected_key in [
            ("gaussian", "gaussian_sigma"),
            ("salt_and_pepper", "salt_and_pepper_amount"),
            ("speckle", "speckle_variance"),
        ]:
            with self.subTest(noise_type=noise_type):
                app.radio(key="noise_type").set_value(noise_type).run()
                shown = {s.key for s in app.sidebar.slider} & strength_keys
                self.assertEqual(shown, {expected_key})


class TestNoiseResults(unittest.TestCase):
    """The page must show exactly the pixels that src/noise.py produces."""

    def setUp(self):
        self.app = run_app(("synthetic.png", SYNTHETIC_PNG))

    def test_default_gaussian_matches_direct_call(self):
        self.assertEqual(fingerprint_shown(self.app), expected_fingerprint(add_gaussian_noise, 25))

    def test_switching_noise_type_updates_the_result(self):
        gaussian = fingerprint_shown(self.app)
        self.app.radio(key="noise_type").set_value("salt_and_pepper").run()
        self.assertEqual(fingerprint_shown(self.app), expected_fingerprint(add_salt_and_pepper_noise, 0.05))
        self.assertIn("Salt-and-Pepper Noise — 5% of pixels", settings_text(self.app))
        self.assertIn("3,277 of 65,536 pixels (1,638 white, 1,639 black)", all_text(self.app.sidebar.caption))
        self.app.radio(key="noise_type").set_value("speckle").run()
        self.assertEqual(fingerprint_shown(self.app), expected_fingerprint(add_speckle_noise, 0.04))
        self.assertIn("Speckle Noise — variance = 0.04", settings_text(self.app))
        self.assertNotEqual(fingerprint_shown(self.app), gaussian)
        self.assertEqual(len(self.app.exception), 0)

    def test_changing_strength_updates_the_result(self):
        before = fingerprint_shown(self.app)
        self.app.slider(key="gaussian_sigma").set_value(40).run()
        self.assertEqual(fingerprint_shown(self.app), expected_fingerprint(add_gaussian_noise, 40))
        self.assertNotEqual(fingerprint_shown(self.app), before)
        self.assertIn("σ = 40", settings_text(self.app))

    def test_changing_speckle_variance(self):
        self.app.radio(key="noise_type").set_value("speckle").run()
        self.app.slider(key="speckle_variance").set_value(0.1).run()
        self.assertEqual(fingerprint_shown(self.app), expected_fingerprint(add_speckle_noise, 0.1))

    def test_changing_salt_and_pepper_amount(self):
        self.app.radio(key="noise_type").set_value("salt_and_pepper").run()
        self.app.slider(key="salt_and_pepper_amount").set_value(0.1).run()
        self.assertEqual(fingerprint_shown(self.app), expected_fingerprint(add_salt_and_pepper_noise, 0.1))

    def test_seed_changes_pattern_and_is_reproducible(self):
        first = fingerprint_shown(self.app)
        self.app.number_input(key="seed").set_value(7).run()
        self.assertEqual(fingerprint_shown(self.app), expected_fingerprint(add_gaussian_noise, 25, seed=7))
        self.assertNotEqual(fingerprint_shown(self.app), first)
        self.app.number_input(key="seed").set_value(42).run()
        self.assertEqual(fingerprint_shown(self.app), first)

    def test_unrelated_controls_do_not_change_the_noise(self):
        first = fingerprint_shown(self.app)
        self.app.slider(key="zoom_x").set_value(10).run()
        self.assertEqual(fingerprint_shown(self.app), first)
        self.app.select_slider(key="zoom_size").set_value(16).run()
        self.assertEqual(fingerprint_shown(self.app), first)
        self.app.run()  # a plain rerun
        self.assertEqual(fingerprint_shown(self.app), first)

    def test_slider_value_is_remembered_when_switching_type(self):
        self.app.slider(key="gaussian_sigma").set_value(40).run()
        self.app.radio(key="noise_type").set_value("speckle").run()
        self.app.radio(key="noise_type").set_value("gaussian").run()
        self.assertEqual(self.app.slider(key="gaussian_sigma").value, 40)
        self.assertEqual(fingerprint_shown(self.app), expected_fingerprint(add_gaussian_noise, 40))

    def test_negative_seed_shows_safe_error(self):
        self.app.number_input(key="seed").set_value(-1).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 1)
        self.assertIn("Noise could not be added", self.app.error[0].value)
        self.assertIn("non-negative", self.app.error[0].value)

    def test_noise_explanation_follows_selected_type(self):
        def explanation():
            [panel] = [e for e in self.app.expander if e.label == "How does this noise work?"]
            return all_text(panel.markdown)

        self.assertIn("standard deviation", explanation())
        self.app.radio(key="noise_type").set_value("salt_and_pepper").run()
        self.assertIn("salt", explanation())
        self.app.radio(key="noise_type").set_value("speckle").run()
        self.assertIn("times its own brightness", explanation())


# --------------------------------------------------------------------------
# Stage 5C: filter controls
# --------------------------------------------------------------------------
def filtered_fingerprint_shown(app):
    """Read the pixel fingerprint of the filtered image from the page (None if not shown)."""
    for caption in app.caption:
        match = re.search(r"Pixel fingerprint of the filtered image: `([0-9a-f]{12})`", caption.value)
        if match:
            return match.group(1)
    return None


def expected_filtered(filter_function, noise_function=add_gaussian_noise, strength=25, seed=42, **parameters):
    """Fingerprint of filter(noise(reference)) computed directly with src/."""
    noisy = noise_function(SYNTHETIC, strength, seed=seed)
    return pixel_fingerprint(filter_function(noisy, **parameters))


FILTER_KEYS = {
    "mean": {"mean_kernel_size"},
    "gaussian": {"gaussian_kernel_size", "gaussian_filter_sigma"},
    "median": {"median_kernel_size"},
    "bilateral": {"bilateral_diameter", "bilateral_sigma_color", "bilateral_sigma_space"},
}
ALL_FILTER_KEYS = set().union(*FILTER_KEYS.values())


class TestFilterControls(unittest.TestCase):
    def setUp(self):
        self.app = run_app(("synthetic.png", SYNTHETIC_PNG))

    def sidebar_filter_keys(self):
        keys = {w.key for w in list(self.app.sidebar.slider) + list(self.app.sidebar.select_slider)}
        return keys & ALL_FILTER_KEYS

    def test_no_filter_controls_before_upload(self):
        app = run_app()
        self.assertNotIn("3. Remove Noise", [h.value for h in app.sidebar.header])

    def test_defaults(self):
        self.assertIn("3. Remove Noise", [h.value for h in self.app.sidebar.header])
        radio = self.app.radio(key="filter_type")
        self.assertEqual(radio.value, "mean")
        self.assertEqual(radio.options, ["Mean Filter", "Gaussian Filter", "Median Filter", "Bilateral Filter"])
        self.assertEqual(self.app.select_slider(key="mean_kernel_size").value, 3)

    def test_every_filter_renders_and_shows_only_its_controls(self):
        for filter_type, keys in FILTER_KEYS.items():
            with self.subTest(filter_type=filter_type):
                self.app.radio(key="filter_type").set_value(filter_type).run()
                self.assertEqual(len(self.app.exception), 0)
                self.assertEqual(len(self.app.error), 0)
                self.assertEqual(self.sidebar_filter_keys(), keys)
                self.assertIsNotNone(filtered_fingerprint_shown(self.app))

    def test_bilateral_and_gaussian_defaults(self):
        self.app.radio(key="filter_type").set_value("gaussian").run()
        self.assertEqual(self.app.select_slider(key="gaussian_kernel_size").value, 3)
        self.assertEqual(self.app.slider(key="gaussian_filter_sigma").value, 0.0)
        self.app.radio(key="filter_type").set_value("bilateral").run()
        self.assertEqual(self.app.select_slider(key="bilateral_diameter").value, 5)
        self.assertEqual(self.app.slider(key="bilateral_sigma_color").value, 50)
        self.assertEqual(self.app.slider(key="bilateral_sigma_space").value, 50)

    def test_noise_sigma_and_filter_sigma_have_different_labels(self):
        noise_label = self.app.slider(key="gaussian_sigma").label
        self.app.radio(key="filter_type").set_value("gaussian").run()
        filter_label = self.app.slider(key="gaussian_filter_sigma").label
        self.assertNotEqual(noise_label, filter_label)
        self.assertIn("Noise", noise_label)
        self.assertIn("filter", filter_label)

    def test_filter_parameter_remembered_when_switching(self):
        self.app.select_slider(key="mean_kernel_size").set_value(7).run()
        self.app.radio(key="filter_type").set_value("median").run()
        self.app.radio(key="filter_type").set_value("mean").run()
        self.assertEqual(self.app.select_slider(key="mean_kernel_size").value, 7)


class TestFilterResults(unittest.TestCase):
    """The page must show filter(noise(reference)) exactly as src/ computes it."""

    def setUp(self):
        self.app = run_app(("synthetic.png", SYNTHETIC_PNG))
        self.noisy_fp = fingerprint_shown(self.app)

    def select(self, filter_type):
        self.app.radio(key="filter_type").set_value(filter_type).run()

    def test_each_filter_matches_direct_calls(self):
        expected = {
            "mean": expected_filtered(apply_mean_filter, kernel_size=3),
            "gaussian": expected_filtered(apply_gaussian_filter, kernel_size=3, sigma=0),
            "median": expected_filtered(apply_median_filter, kernel_size=3),
            "bilateral": expected_filtered(apply_bilateral_filter, diameter=5, sigma_color=50, sigma_space=50),
        }
        for filter_type, fingerprint in expected.items():
            with self.subTest(filter_type=filter_type):
                self.select(filter_type)
                self.assertEqual(filtered_fingerprint_shown(self.app), fingerprint)

    def test_filter_is_applied_to_the_noisy_image_not_the_reference(self):
        on_reference = pixel_fingerprint(apply_mean_filter(SYNTHETIC, 3))
        self.assertNotEqual(filtered_fingerprint_shown(self.app), on_reference)
        self.assertEqual(filtered_fingerprint_shown(self.app), expected_filtered(apply_mean_filter, kernel_size=3))

    def test_switching_filters_changes_result_but_not_noise(self):
        seen = set()
        for filter_type in ("mean", "gaussian", "median", "bilateral", "mean"):
            self.select(filter_type)
            self.assertEqual(fingerprint_shown(self.app), self.noisy_fp)
            seen.add(filtered_fingerprint_shown(self.app))
        self.assertEqual(len(seen), 4)  # four different results; returning to mean repeats one

    def test_changing_filter_parameters_changes_result_but_not_noise(self):
        changes = [
            ("mean", "mean_kernel_size", 5, "select_slider", expected_filtered(apply_mean_filter, kernel_size=5)),
            ("median", "median_kernel_size", 7, "select_slider", expected_filtered(apply_median_filter, kernel_size=7)),
            ("gaussian", "gaussian_kernel_size", 9, "select_slider", expected_filtered(apply_gaussian_filter, kernel_size=9, sigma=0)),
            ("gaussian", "gaussian_filter_sigma", 2.0, "slider", expected_filtered(apply_gaussian_filter, kernel_size=9, sigma=2.0)),
            ("bilateral", "bilateral_diameter", 9, "select_slider", expected_filtered(apply_bilateral_filter, diameter=9, sigma_color=50, sigma_space=50)),
            ("bilateral", "bilateral_sigma_color", 120, "slider", expected_filtered(apply_bilateral_filter, diameter=9, sigma_color=120, sigma_space=50)),
            ("bilateral", "bilateral_sigma_space", 10, "slider", expected_filtered(apply_bilateral_filter, diameter=9, sigma_color=120, sigma_space=10)),
        ]
        for filter_type, key, value, widget, expected in changes:
            with self.subTest(key=key, value=value):
                self.select(filter_type)
                before = filtered_fingerprint_shown(self.app)
                getattr(self.app, widget)(key=key).set_value(value).run()
                self.assertEqual(len(self.app.exception), 0)
                self.assertEqual(fingerprint_shown(self.app), self.noisy_fp)
                self.assertEqual(filtered_fingerprint_shown(self.app), expected)
                self.assertNotEqual(filtered_fingerprint_shown(self.app), before)

    def test_changing_noise_updates_the_filtered_image(self):
        self.select("median")
        first = filtered_fingerprint_shown(self.app)
        self.app.slider(key="gaussian_sigma").set_value(40).run()
        self.assertEqual(filtered_fingerprint_shown(self.app), expected_filtered(apply_median_filter, strength=40, kernel_size=3))
        self.app.radio(key="noise_type").set_value("salt_and_pepper").run()
        self.assertEqual(
            filtered_fingerprint_shown(self.app),
            expected_filtered(apply_median_filter, add_salt_and_pepper_noise, 0.05, kernel_size=3),
        )
        self.app.number_input(key="seed").set_value(7).run()
        self.assertEqual(
            filtered_fingerprint_shown(self.app),
            expected_filtered(apply_median_filter, add_salt_and_pepper_noise, 0.05, seed=7, kernel_size=3),
        )
        # back to the first settings -> the first result returns
        self.app.number_input(key="seed").set_value(42).run()
        self.app.radio(key="noise_type").set_value("gaussian").run()
        self.app.slider(key="gaussian_sigma").set_value(25).run()
        self.assertEqual(filtered_fingerprint_shown(self.app), first)
        self.assertEqual(fingerprint_shown(self.app), self.noisy_fp)

    def test_labels_and_settings_line(self):
        self.select("bilateral")
        settings = settings_text(self.app)
        self.assertIn("noise = Gaussian Noise — σ = 25", settings)
        self.assertIn("seed = 42", settings)
        self.assertIn("filter = Bilateral Filter — diameter = 5, sigma color = 50, sigma space = 50", settings)
        captions = [c for image in self.app.image for c in image.captions]
        self.assertIn("Original Grayscale", captions)
        self.assertIn("Gaussian Noise — σ = 25", captions)
        self.assertIn("Bilateral Filter — d = 5", captions)
        page_text = " ".join(captions + [settings]).lower()
        self.assertNotIn("restored", page_text)
        self.assertNotIn("clean", page_text)

    def test_zoom_shows_the_same_region_in_all_three_images(self):
        self.select("median")
        self.app.slider(key="zoom_x").set_value(10).run()
        self.app.slider(key="zoom_y").set_value(90).run()
        self.app.select_slider(key="zoom_size").set_value(32).run()
        zoom_captions = [c for image in self.app.image for c in image.captions if "(rows " in c]
        self.assertEqual(len(zoom_captions), 3)
        regions = {re.search(r"\(rows .*\)$", c).group(0) for c in zoom_captions}
        self.assertEqual(len(regions), 1)
        # 256 x 256 image, 32 px window at 10% / 90%: centre row round(0.9 * 255) = round(229.5) = 230,
        # centre column round(0.1 * 255) = round(25.5) = 26 -> rows 214-245, columns 10-41
        self.assertEqual(regions.pop(), "(rows 214–245, columns 10–41)")
        self.assertTrue(zoom_captions[0].startswith("Original Grayscale"))
        self.assertTrue(zoom_captions[1].startswith("Gaussian Noise"))
        self.assertTrue(zoom_captions[2].startswith("Median Filter"))
        self.assertEqual(fingerprint_shown(self.app), self.noisy_fp)

    def test_filter_explanation_follows_selection(self):
        def explanation():
            [panel] = [e for e in self.app.expander if e.label == "How does this filter work?"]
            return all_text(panel.markdown)

        expected_words = {"mean": "plain average", "gaussian": "bell-shaped", "median": "middle", "bilateral": "near **and** similar"}
        for filter_type, words in expected_words.items():
            with self.subTest(filter_type=filter_type):
                self.select(filter_type)
                self.assertIn(words, explanation())
                self.assertIn("**Limitation:**", explanation())

    def test_filter_error_is_shown_safely(self):
        """If the filter fails, the page shows an error but keeps the original and noisy images."""
        with mock.patch("frontend.image_processing.apply_selected_filter", side_effect=ValueError("test failure")):
            self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        # Updated in Stage 5D (was: exactly 1 error). The mocked failure also stops the new
        # four-filter comparison (section 5), which correctly reports its own error. The first
        # error must still be the filter error, and every error must be a safe message.
        self.assertGreaterEqual(len(self.app.error), 1)
        self.assertTrue(all("test failure" in e.value for e in self.app.error))
        self.assertIn("The filter could not be applied", self.app.error[0].value)
        self.assertIn("test failure", self.app.error[0].value)
        self.assertEqual(fingerprint_shown(self.app), self.noisy_fp)  # noisy image still shown
        self.assertIsNone(filtered_fingerprint_shown(self.app))
        captions = [c for image in self.app.image for c in image.captions]
        self.assertIn("Original Grayscale", captions)
        # the controls are kept, and the next normal run works again
        self.assertEqual(self.app.radio(key="filter_type").value, "mean")
        self.app.run()
        self.assertEqual(len(self.app.error), 0)
        self.assertIsNotNone(filtered_fingerprint_shown(self.app))


# --------------------------------------------------------------------------
# Stage 5D: MSE / PSNR, four-filter comparison, difference images
# --------------------------------------------------------------------------
from frontend.image_processing import format_mse, format_psnr  # noqa: E402
from src.metrics import mse, psnr  # noqa: E402


def page_text(app):
    """All visible text of the page (for checks like 'no NaN anywhere')."""
    parts = []
    for group in (app.title, app.subheader, app.header, app.markdown, app.caption, app.info,
                  app.success, app.warning, app.error, app.text):
        parts.extend(str(element.value) for element in group)
    for metric in app.metric:
        parts.extend([metric.label, str(metric.value)])
    for table in app.table:
        parts.append(table.value.to_string())
    for image in app.image:
        parts.extend(image.captions)
    return "\n".join(parts)


def tables(app):
    """(evaluation table, comparison table or None) as pandas DataFrames."""
    frames = [t.value for t in app.table]
    evaluation = next(f for f in frames if "Image" in f.columns)
    comparison = next((f for f in frames if "Method" in f.columns), None)
    return evaluation, comparison


def expected_rows(reference, noisy, parameters):
    """MSE/PSNR text for 'none' + four filters, computed directly with src/."""
    filtered = {
        "No filtering": noisy,
        "Mean Filter": apply_mean_filter(noisy, **parameters["mean"]),
        "Gaussian Filter": apply_gaussian_filter(noisy, **parameters["gaussian"]),
        "Median Filter": apply_median_filter(noisy, **parameters["median"]),
        "Bilateral Filter": apply_bilateral_filter(noisy, **parameters["bilateral"]),
    }
    return {name: (format_mse(mse(reference, image)), format_psnr(psnr(reference, image))) for name, image in filtered.items()}


DEFAULTS = {
    "mean": {"kernel_size": 3},
    "gaussian": {"kernel_size": 3, "sigma": 0},
    "median": {"kernel_size": 3},
    "bilateral": {"diameter": 5, "sigma_color": 50, "sigma_space": 50},
}


class TestEvaluationSection(unittest.TestCase):
    def setUp(self):
        self.app = run_app(("synthetic.png", SYNTHETIC_PNG))
        self.noisy = add_gaussian_noise(SYNTHETIC, 25, seed=42)

    def test_table_equals_direct_metric_calls(self):
        evaluation, _ = tables(self.app)
        filtered = apply_mean_filter(self.noisy, 3)
        self.assertEqual(list(evaluation["MSE ↓"]), [format_mse(mse(SYNTHETIC, self.noisy)), format_mse(mse(SYNTHETIC, filtered))])
        self.assertEqual(list(evaluation["PSNR ↑ (dB)"]), [format_psnr(psnr(SYNTHETIC, self.noisy)), format_psnr(psnr(SYNTHETIC, filtered))])
        self.assertIn("4. Image Quality Evaluation — MSE and PSNR", [s.value for s in self.app.subheader])

    def test_worse_result_is_reported_as_worse(self):
        """Mean 3 x 3 on Gaussian noise: MSE 577.99 -> 945.68 (E002 -> E010)."""
        self.assertNotIn("Numerically closer", all_text(self.app.success))
        warning = all_text(self.app.warning)
        self.assertIn("Numerically farther from the original", warning)
        self.assertIn("577.99 → 945.68", warning)
        metrics = {m.label: m.value for m in self.app.metric}
        self.assertEqual(metrics["MSE change (filtered − noisy)"], "+367.69")
        self.assertEqual(metrics["PSNR change (filtered − noisy)"], "−2.14 dB")

    def test_better_result_is_reported_as_better(self):
        """Bilateral 5/50/50 on Gaussian noise: MSE 577.99 -> 87.85, +8.18 dB (E013)."""
        self.app.radio(key="filter_type").set_value("bilateral").run()
        self.assertIn("Numerically closer to the original", all_text(self.app.success))
        metrics = {m.label: m.value for m in self.app.metric}
        self.assertEqual(metrics["PSNR change (filtered − noisy)"], "+8.18 dB")
        self.assertEqual(metrics["MSE change (filtered − noisy)"], "−490.14")

    def test_noise_free_case_has_no_nan_or_inf_arithmetic(self):
        self.app.slider(key="gaussian_sigma").set_value(0).run()  # noisy == original
        text = page_text(self.app)
        self.assertNotIn("nan", text.lower())
        # no raw Python float infinity ("inf", "-inf"); the word "infinite" in explanations is fine
        self.assertIsNone(re.search(r"(?<![a-z])-?inf(?![a-z])", text.lower()))
        self.assertEqual(text.count("there was no error to remove"), 2)  # once in section 4, once (grouped) in 5
        evaluation, comparison = tables(self.app)
        self.assertEqual(evaluation["PSNR ↑ (dB)"][0], "∞")
        metrics = {m.label: m.value for m in self.app.metric}
        self.assertEqual(metrics["PSNR change (filtered − noisy)"], "worse (was ∞)")
        self.assertEqual(comparison["Best"][0], "★ lowest MSE / highest PSNR")
        self.assertIn("No filter improved the numerical result", all_text(self.app.warning))

    def test_metric_explanation_present(self):
        [panel] = [e for e in self.app.expander if e.label == "How do MSE and PSNR work?"]
        text = all_text(panel.markdown)
        self.assertIn("Squaring", text)
        self.assertIn("infinite", text)
        self.assertIn("illustrative", text)

    def test_results_follow_noise_and_filter_changes(self):
        self.app.radio(key="noise_type").set_value("salt_and_pepper").run()
        self.app.radio(key="filter_type").set_value("median").run()
        self.app.select_slider(key="median_kernel_size").set_value(5).run()
        noisy = add_salt_and_pepper_noise(SYNTHETIC, 0.05, seed=42)
        filtered = apply_median_filter(noisy, 5)
        evaluation, _ = tables(self.app)
        self.assertEqual(list(evaluation["MSE ↓"]), [format_mse(mse(SYNTHETIC, noisy)), format_mse(mse(SYNTHETIC, filtered))])


class TestFourFilterComparison(unittest.TestCase):
    def setUp(self):
        self.app = run_app(("synthetic.png", SYNTHETIC_PNG))

    def check_table(self, reference, noisy, parameters):
        _, comparison = tables(self.app)
        self.assertIsNotNone(comparison)
        expected = expected_rows(reference, noisy, parameters)
        shown = {m.replace(" (selected)", ""): (a, b) for m, a, b in zip(comparison["Method"], comparison["MSE ↓"], comparison["PSNR ↑ (dB)"])}
        self.assertEqual(shown, expected)
        # the "best" mark is on the row(s) with the lowest MSE
        mses = {name: mse(reference, img) for name, img in [
            ("No filtering", noisy),
            ("Mean Filter", apply_mean_filter(noisy, **parameters["mean"])),
            ("Gaussian Filter", apply_gaussian_filter(noisy, **parameters["gaussian"])),
            ("Median Filter", apply_median_filter(noisy, **parameters["median"])),
            ("Bilateral Filter", apply_bilateral_filter(noisy, **parameters["bilateral"])),
        ]}
        lowest = min(mses.values())
        marked = {m.replace(" (selected)", "") for m, best in zip(comparison["Method"], comparison["Best"]) if best}
        self.assertEqual(marked, {name for name, value in mses.items() if value == lowest})
        return comparison

    def test_default_comparison_equals_direct_calls(self):
        self.assertIn("5. Compare All Four Filters", [s.value for s in self.app.subheader])
        self.assertTrue(self.app.toggle(key="compare_all").value)  # small image: on by default
        comparison = self.check_table(SYNTHETIC, add_gaussian_noise(SYNTHETIC, 25, seed=42), DEFAULTS)
        self.assertEqual(comparison["Method"][0], "No filtering")
        self.assertEqual(comparison["PSNR change"][0], "baseline")
        self.assertEqual(comparison["Method"][1], "Mean Filter (selected)")
        self.assertIn("Lowest MSE / highest PSNR for these settings: Bilateral Filter", all_text(self.app.success))

    def test_remembered_parameters_are_used_and_shown(self):
        self.app.select_slider(key="mean_kernel_size").set_value(7).run()
        self.app.radio(key="filter_type").set_value("bilateral").run()
        self.app.select_slider(key="bilateral_diameter").set_value(9).run()
        self.app.radio(key="filter_type").set_value("gaussian").run()
        self.app.slider(key="gaussian_filter_sigma").set_value(1.5).run()
        parameters = {**DEFAULTS, "mean": {"kernel_size": 7}, "gaussian": {"kernel_size": 3, "sigma": 1.5},
                      "bilateral": {"diameter": 9, "sigma_color": 50, "sigma_space": 50}}
        comparison = self.check_table(SYNTHETIC, add_gaussian_noise(SYNTHETIC, 25, seed=42), parameters)
        self.assertEqual(list(comparison["Parameters"]), [
            "—", "7 × 7", "3 × 3, filter σ = 1.5", "3 × 3", "diameter = 9, sigma color = 50, sigma space = 50"])

    def test_comparison_follows_noise_changes(self):
        self.app.radio(key="noise_type").set_value("salt_and_pepper").run()
        self.app.number_input(key="seed").set_value(3).run()
        noisy = add_salt_and_pepper_noise(SYNTHETIC, 0.05, seed=3)
        self.check_table(SYNTHETIC, noisy, DEFAULTS)

    def test_four_images_shown_with_parameters(self):
        captions = [c for image in self.app.image for c in image.captions]
        for expected in ("Mean Filter — 3 × 3", "Gaussian Filter — 3 × 3, filter σ = auto", "Median Filter — 3 × 3",
                         "Bilateral Filter — diameter = 5, sigma color = 50, sigma space = 50 ★"):
            self.assertIn(expected, captions)

    def test_no_filter_best_message(self):
        texture = np.random.default_rng(5).integers(0, 256, (64, 64), dtype=np.uint8)
        app = run_app(("texture.png", encode(".png", texture)))
        app.slider(key="gaussian_sigma").set_value(2).run()
        noisy = add_gaussian_noise(texture, 2, seed=42)
        _, comparison = tables(app)
        self.assertEqual(comparison["Best"][0], "★ lowest MSE / highest PSNR")
        self.assertTrue(all(b == "" for b in comparison["Best"][1:]))
        self.assertIn("No filter improved the numerical result", all_text(app.warning))
        self.assertEqual(dict(zip(comparison["Method"], comparison["MSE ↓"]))["No filtering"],
                         format_mse(mse(texture, noisy)))

    def test_large_image_starts_switched_off(self):
        large = np.zeros((2000, 2100), dtype=np.uint8)
        large[::7] = 200  # some structure; still compresses well
        app = run_app(("large.png", encode(".png", large)))
        self.assertFalse(app.toggle(key="compare_all").value)
        self.assertIsNone(tables(app)[1])
        self.assertIn("Switched off", all_text(app.caption))
        app.toggle(key="compare_all").set_value(True).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIsNotNone(tables(app)[1])

    def test_new_upload_replaces_all_results(self):
        first = tables(self.app)[1].copy()
        other = np.random.default_rng(9).integers(30, 220, (80, 120), dtype=np.uint8)
        self.app.sidebar.file_uploader[0].set_value(("other.png", encode(".png", other), "image/png")).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.metric[0].label, "MSE change (filtered − noisy)")
        comparison = self.check_table(other, add_gaussian_noise(other, 25, seed=42), DEFAULTS)
        self.assertFalse(first.equals(comparison))
        evaluation, _ = tables(self.app)
        self.assertEqual(evaluation["MSE ↓"][0], format_mse(mse(other, add_gaussian_noise(other, 25, seed=42))))

    def test_comparison_does_not_change_the_noise(self):
        before = fingerprint_shown(self.app)
        self.app.toggle(key="compare_all").set_value(False).run()
        self.assertEqual(fingerprint_shown(self.app), before)
        self.app.toggle(key="compare_all").set_value(True).run()
        self.assertEqual(fingerprint_shown(self.app), before)


class TestDifferenceSection(unittest.TestCase):
    def setUp(self):
        self.app = run_app(("synthetic.png", SYNTHETIC_PNG))
        self.app.radio(key="noise_type").set_value("salt_and_pepper").run()
        self.app.radio(key="filter_type").set_value("median").run()

    def panel(self):
        [panel] = [e for e in self.app.expander if e.label == "6. Where Did the Pixels Change? — Difference Images"]
        return panel

    def test_three_difference_images_with_exact_counts(self):
        noisy = add_salt_and_pepper_noise(SYNTHETIC, 0.05, seed=42)
        filtered = apply_median_filter(noisy, 3)
        captions = [c for image in self.panel().image for c in image.captions]
        self.assertEqual(captions, [
            "A. Noise introduced: |Original − Noisy|",
            "B. Filter changes: |Noisy − Filtered|",
            "C. Remaining error after filtering: |Original − Filtered|",
        ])
        facts = all_text(self.panel().caption)
        for a, b in ((SYNTHETIC, noisy), (noisy, filtered), (SYNTHETIC, filtered)):
            changed = int(np.count_nonzero(a != b))
            largest = int(np.abs(a.astype(int) - b.astype(int)).max())
            self.assertIn(f"Changed pixels: {changed:,} ({100 * changed / a.size:.1f}%). Largest difference: {largest}.", facts)
        self.assertIn("It does not prove that the change improved image quality", all_text(self.panel().markdown))

    def test_enhanced_view_does_not_change_metrics(self):
        before_tables = [t.value.copy() for t in self.app.table]
        before_noise = fingerprint_shown(self.app)
        self.app.toggle(key="difference_enhanced").set_value(True).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("Contrast enhanced", all_text(self.panel().caption))
        for before, after in zip(before_tables, [t.value for t in self.app.table]):
            self.assertTrue(before.equals(after))
        self.assertEqual(fingerprint_shown(self.app), before_noise)


class TestWholePageStillWorks(unittest.TestCase):
    def test_every_noise_and_filter_combination_renders_without_nan(self):
        app = run_app(("synthetic.png", SYNTHETIC_PNG))
        for noise_type in ("gaussian", "salt_and_pepper", "speckle"):
            app.radio(key="noise_type").set_value(noise_type).run()
            for filter_type in ("mean", "gaussian", "median", "bilateral"):
                with self.subTest(noise_type=noise_type, filter_type=filter_type):
                    app.radio(key="filter_type").set_value(filter_type).run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertEqual(len(app.error), 0)
                    self.assertEqual(len(app.table), 2)
                    self.assertNotIn("nan", page_text(app).lower())

    def test_constant_image_identical_everywhere(self):
        flat = np.full((40, 50), 128, dtype=np.uint8)
        app = run_app(("flat.png", encode(".png", flat)))
        app.slider(key="gaussian_sigma").set_value(0).run()
        self.assertEqual(len(app.exception), 0)
        metrics = {m.label: m.value for m in app.metric}
        self.assertEqual(metrics["PSNR change (filtered − noisy)"], "no change (both ∞)")
        self.assertIn("No numerical change", all_text(app.info))
        self.assertIn("No filter did better than no filtering", all_text(app.info))
        self.assertNotIn("nan", page_text(app).lower())


if __name__ == "__main__":
    unittest.main()
