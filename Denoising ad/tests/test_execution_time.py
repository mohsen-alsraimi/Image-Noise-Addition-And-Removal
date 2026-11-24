"""
Focused tests for execution-time measurement in main.run_experiments.

Run from the "Denoising ad" folder, using an environment that has the
packages in requirements.txt:

    python -B -m unittest discover -s tests -p "test_execution_time.py" -v

A fake clock and fake filters make the timing boundaries exact, so the
results do not depend on computer speed. Output files go to a temporary
folder outside the project.
"""

import contextlib
import inspect
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import main

FILTER_SECONDS = {
    "Gaussian Filter": 0.1,
    "Median Filter": 0.2,
    "Bilateral Filter": 0.3,
    "NLM Filter": 0.4,
    "Lee Filter": 0.5,
    "TV Filter": 0.6,
}
OVERHEAD_SECONDS = 7.0
SPECKLE_LEVELS = {"Speckle": {"Low": 0.01, "Medium": 0.04, "High": 0.10}}


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.events = []

    def perf_counter(self):
        self.events.append("clock")
        return self.now

    def spend(self, event, seconds=OVERHEAD_SECONDS):
        self.events.append(event)
        self.now += seconds


class TrackedArray(np.ndarray):
    clock = None
    copy_event = ""

    def copy(self, *args, **kwargs):
        self.clock.spend(self.copy_event)
        return np.ndarray.copy(self, *args, **kwargs)


class NoisyArray(TrackedArray):
    copy_event = "copy input"


class FilteredArray(TrackedArray):
    copy_event = "copy output"


class TrackedParameters(dict):
    clock = None

    def __getitem__(self, key):
        self.clock.spend("parameter lookup")
        return dict.__getitem__(self, key)


def make_filters(clock):
    def make(name):
        def fake_filter(image, **kwargs):
            clock.spend("filter", FILTER_SECONDS[name])
            return np.array(image).view(FilteredArray)

        return fake_filter

    def fake_lee(image, window_size=5, speckle_variance=None):
        clock.spend("filter", FILTER_SECONDS["Lee Filter"])
        return np.array(image).view(FilteredArray)

    fakes = {name: make(name) for name in FILTER_SECONDS}
    fakes["Lee Filter"] = fake_lee
    return fakes


def run_with_fake_clock():
    clock = FakeClock()
    TrackedArray.clock = clock
    TrackedParameters.clock = clock

    parameters = TrackedParameters({
        "Speckle": {
            level: {name: {} for name in FILTER_SECONDS}
            for level in SPECKLE_LEVELS["Speckle"]
        }
    })
    original = np.full((16, 16), 0.5, dtype=np.float32)

    def fake_noise(image, noise_type, noise_value, seed):
        clock.spend("noise")
        return image.copy().view(NoisyArray)

    def fake_metrics(reference, processed):
        clock.spend("metrics")
        return 20.0, 0.5

    def fake_comparison(*args):
        clock.spend("save comparison")

    real_signature = inspect.signature

    def fake_signature(function):
        clock.spend("signature")
        return real_signature(function)

    with mock.patch.dict(main.filters, make_filters(clock), clear=True), \
            mock.patch.object(main, "NOISE_LEVELS", SPECKLE_LEVELS), \
            mock.patch.object(main, "TUNED_PARAMETERS_FILE",
                              PROJECT_DIR / "best_parameters.json"), \
            mock.patch.object(main.json, "load", return_value=parameters), \
            mock.patch.object(main, "generate_noisy_image", fake_noise), \
            mock.patch.object(main, "calculate_metrics", fake_metrics), \
            mock.patch.object(main, "save_image_comparison", fake_comparison), \
            mock.patch.object(main.inspect, "signature", fake_signature), \
            mock.patch.object(main.time, "perf_counter", clock.perf_counter), \
            contextlib.redirect_stdout(io.StringIO()):
        results = main.run_experiments([("synthetic.png", original)])

    return clock, results


class TimingBoundaryTests(unittest.TestCase):
    def test_timer_surrounds_only_the_filter_call(self):
        clock, _ = run_with_fake_clock()

        clock_positions = [
            index for index, event in enumerate(clock.events)
            if event == "clock"
        ]
        self.assertEqual(len(clock_positions), 36)  # 6 filters x 3 levels x 2

        for start, stop in zip(clock_positions[::2], clock_positions[1::2]):
            self.assertEqual(clock.events[start + 1:stop], ["filter"])

        # The first filter of the first condition, in exact order.
        self.assertEqual(clock.events[:10], [
            "noise", "metrics",
            "parameter lookup", "copy input",
            "clock", "filter", "clock",
            "copy output", "metrics",
            "parameter lookup",
        ])
        lee_start = clock.events.index("signature")
        self.assertEqual(clock.events[lee_start:lee_start + 7], [
            "signature", "copy input",
            "clock", "filter", "clock",
            "copy output", "metrics",
        ])

    def test_stored_time_is_filter_duration_in_seconds(self):
        _, results = run_with_fake_clock()

        for name, seconds in FILTER_SECONDS.items():
            with self.subTest(filter=name):
                stored = results.loc[
                    results["Filter"] == name, "Execution_Time"
                ].to_numpy(dtype=float)
                self.assertEqual(len(stored), 3)
                np.testing.assert_allclose(stored, seconds, atol=1e-9)

        baseline = results.loc[
            results["Filter"] == "Noisy (Baseline)", "Execution_Time"
        ]
        self.assertTrue((baseline == 0).all())


class OutputCompatibilityTests(unittest.TestCase):
    def test_csv_summary_and_time_graph_still_work(self):
        _, results = run_with_fake_clock()

        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(main, "RESULTS_PATH", Path(folder)), \
                mock.patch.object(main, "NOISE_LEVELS", SPECKLE_LEVELS), \
                contextlib.redirect_stdout(io.StringIO()):
            main.save_results(results)
            main.create_graphs(results)

            saved = pd.read_csv(Path(folder) / "experiment_results.csv")
            summary = pd.read_csv(Path(folder) / "summary.csv", header=[0, 1])
            graph = Path(folder) / "graphs" / "speckle_execution_time.png"

            self.assertEqual(list(saved.columns), [
                "Image", "Noise_Type", "Noise_Level", "Filter",
                "PSNR", "SSIM", "Execution_Time",
            ])
            times = saved["Execution_Time"].to_numpy(dtype=float)
            self.assertTrue(np.isfinite(times).all())
            self.assertTrue((times >= 0).all())
            self.assertIn(("Execution_Time", "mean"), summary.columns)
            self.assertGreater(graph.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
