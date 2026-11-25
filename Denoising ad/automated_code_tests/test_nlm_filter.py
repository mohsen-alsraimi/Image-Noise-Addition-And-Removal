"""
Focused tests for NLM parameter handling.

Run from the "Denoising ad" folder:

    python -B -m unittest discover -s automated_code_tests -p "test_nlm_filter.py" -v

Most tests replace the scikit-image NLM call with a mock, because they
check which parameters reach it. RealLibraryTests runs real NLM and is
skipped when scikit-image is not installed. No project files are written.
"""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def install_skimage_stand_in():
    """Lets project modules import when scikit-image is not installed."""

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
    REAL_SKIMAGE = True
except ImportError:
    install_skimage_stand_in()
    REAL_SKIMAGE = False

import add_noise
import filter as filter_module
import main
import tune_filters


def test_image():
    image = np.full((32, 32), 0.3, dtype=np.float32)
    image[:, 16:] = 0.7
    return add_noise.add_gaussian_noise(image, 25, seed=42)


class MockedNlmTestCase(unittest.TestCase):
    """Uses an estimated sigma larger than every valid candidate h."""

    def setUp(self):
        library = mock.patch.object(
            filter_module,
            "denoise_nl_means",
            side_effect=lambda image, **kwargs: image.copy()
        )
        estimator = mock.patch.object(
            filter_module, "estimate_sigma", return_value=0.4
        )
        self.library = library.start()
        self.estimator = estimator.start()
        self.addCleanup(mock.patch.stopall)

    def received_h(self):
        return [call.kwargs["h"] for call in self.library.call_args_list]


class ParameterHandlingTests(MockedNlmTestCase):
    def test_explicit_h_values_reach_library_unchanged(self):
        image = test_image()

        filter_module.apply_nlm_filter(image, h=0.1)
        filter_module.apply_nlm_filter(image, h=0.15)

        self.assertEqual(self.received_h(), [0.1, 0.15])
        self.estimator.assert_not_called()
        self.assertEqual(
            {k: v for k, v in self.library.call_args.kwargs.items() if k != "h"},
            {
                "patch_size": 5,
                "patch_distance": 6,
                "channel_axis": None,
                "fast_mode": True,
            }
        )

    def test_omitted_h_keeps_automatic_default(self):
        for estimated, expected in ((0.05, 0.1), (0.25, 0.25)):
            with self.subTest(estimated=estimated):
                self.library.reset_mock()
                self.estimator.return_value = estimated
                filter_module.apply_nlm_filter(test_image())
                self.assertEqual(self.received_h(), [expected])

    def test_output_keeps_shape_and_valid_range(self):
        image = test_image()
        raw = np.linspace(-0.5, 1.5, image.size).reshape(image.shape)
        self.library.side_effect = None
        self.library.return_value = raw

        output = filter_module.apply_nlm_filter(image, h=0.1)

        self.assertEqual(output.shape, image.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(float(output.min()), 0.0)
        self.assertEqual(float(output.max()), 1.0)

    def test_invalid_h_is_rejected_before_filtering(self):
        for h in (0, -0.1, float("nan"), float("inf"), True, "0.1", [0.1]):
            with self.subTest(h=h):
                with self.assertRaises((ValueError, TypeError)):
                    filter_module.apply_nlm_filter(test_image(), h=h)
        self.library.assert_not_called()


class ConnectedCodeTests(MockedNlmTestCase):
    def test_runner_passes_saved_h_for_each_condition(self):
        parameters_file = PROJECT_DIR / "best_parameters.json"
        saved = json.loads(parameters_file.read_text(encoding="utf-8"))
        expected = [
            saved[noise_type][level]["NLM Filter"]["h"]
            for noise_type, levels in main.NOISE_LEVELS.items()
            for level in levels
        ]
        nlm_only = {"NLM Filter": filter_module.apply_nlm_filter}

        with mock.patch.dict(main.filters, nlm_only, clear=True), \
                mock.patch.object(main, "TUNED_PARAMETERS_FILE", parameters_file), \
                mock.patch.object(main, "save_image_comparison"), \
                contextlib.redirect_stdout(io.StringIO()):
            main.run_experiments([("synthetic.png", test_image().astype(np.float32))])

        self.assertEqual(self.received_h(), expected)

    def test_tuning_passes_each_candidate_h(self):
        nlm_only = {"NLM Filter": filter_module.apply_nlm_filter}
        grid = {"NLM Filter": tune_filters.PARAMETER_GRID["NLM Filter"]}

        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(main.filters, nlm_only, clear=True), \
                mock.patch.dict(tune_filters.PARAMETER_GRID, grid, clear=True), \
                mock.patch.object(tune_filters, "load_images",
                                  return_value=[test_image().astype(np.float32)]), \
                mock.patch.object(tune_filters, "OUTPUT_PATH",
                                  Path(folder) / "parameters.json"), \
                mock.patch.object(tune_filters.pd.DataFrame, "to_csv"), \
                contextlib.redirect_stdout(io.StringIO()):
            tune_filters.tune()

        self.assertEqual(self.received_h(), [0.1, 0.15, 0.2] * 12)


@unittest.skipUnless(REAL_SKIMAGE, "scikit-image is not installed")
class RealLibraryTests(unittest.TestCase):
    def test_real_nlm_output_is_valid_and_changes_with_h(self):
        image = test_image()

        weak = filter_module.apply_nlm_filter(image, h=0.05)
        strong = filter_module.apply_nlm_filter(image, h=0.25)

        for output in (weak, strong):
            self.assertEqual(output.shape, image.shape)
            self.assertTrue(np.isfinite(output).all())
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)
        self.assertFalse(np.array_equal(weak, strong))


if __name__ == "__main__":
    unittest.main()
