"""
Focused tests for run_config.json.

Run from the "Denoising ad" folder, using an environment that has the
packages in requirements.txt:

    python -B -m unittest discover -s tests -p "test_run_metadata.py" -v

RealRunTests runs the real runner on two tiny synthetic images.
The Git, library, and save-failure tests use mocks.
All output goes to a temporary folder outside the project.
"""

import contextlib
import functools
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import main
import run_metadata

NOISE_LEVELS = {
    "Gaussian": {"Low": 15, "Medium": 25, "High": 50},
    "Speckle": {"Low": 0.01, "Medium": 0.04, "High": 0.10},
}
TUNED_FOR_ONE_LEVEL = {
    "Gaussian Filter": {"kernel_size": 3, "sigma": 0.5},
    "Median Filter": {"kernel_size": 3},
    "Bilateral Filter": {"diameter": 5, "sigma_color": 0.05},
    "NLM Filter": {"h": 0.15},
    "Lee Filter": {"window_size": 3},
    "TV Filter": {"weight": 0.07},
}


def write_small_dataset(folder):
    folder.mkdir()
    rng = np.random.default_rng(5)
    for name, shape in (("a.png", (24, 32)), ("b.png", (32, 24))):
        cv2.imwrite(str(folder / name), rng.integers(40, 210, shape, dtype=np.uint8))


class RealRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        write_small_dataset(root / "TinySet")
        tuned_file = root / "tuned.json"
        tuned = {
            noise: {level: TUNED_FOR_ONE_LEVEL for level in levels}
            for noise, levels in NOISE_LEVELS.items()
        }
        tuned_file.write_text(json.dumps(tuned), encoding="utf-8")

        cls.filter_calls = []
        cls.seeds = []
        real_noise = main.generate_noisy_image

        def recording_noise(image, noise_type, noise_value, seed):
            cls.seeds.append((noise_type, noise_value, seed))
            return real_noise(image, noise_type, noise_value, seed)

        def recording_filter(name, function):
            @functools.wraps(function)
            def wrapped(image, **kwargs):
                cls.filter_calls.append((name, kwargs))
                return function(image, **kwargs)
            return wrapped

        recorded = {
            name: recording_filter(name, function)
            for name, function in main.filters.items()
        }

        with mock.patch.dict(main.filters, recorded, clear=True), \
                mock.patch.object(main, "DATASET_PATH", root / "TinySet"), \
                mock.patch.object(main, "RESULTS_PATH", root / "results"), \
                mock.patch.object(main, "TUNED_PARAMETERS_FILE", tuned_file), \
                mock.patch.object(main, "MAX_IMAGES", None), \
                mock.patch.object(main, "NOISE_LEVELS", NOISE_LEVELS), \
                mock.patch.object(main, "generate_noisy_image", recording_noise), \
                contextlib.redirect_stdout(io.StringIO()):
            main.main()

        cls.run_dir = root / "results" / "result1"
        cls.text = (cls.run_dir / "run_config.json").read_text(encoding="utf-8")
        cls.config = json.loads(cls.text)
        cls.needed = cls.config["required_to_rerun"]

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_run_saves_completed_config_beside_existing_outputs(self):
        self.assertEqual(self.config["run"]["status"], "completed")
        self.assertEqual(self.config["run"]["result_directory"], "result1")
        self.assertEqual(self.config["run"]["mode"], "tuned_parameters")

        saved = pd.read_csv(self.run_dir / "experiment_results.csv")
        self.assertEqual(list(saved.columns), [
            "Image", "Noise_Type", "Noise_Level", "Filter",
            "PSNR", "SSIM", "Execution_Time",
        ])
        self.assertTrue((self.run_dir / "summary.csv").exists())
        self.assertEqual(len(list((self.run_dir / "graphs").glob("*.png"))), 6)
        self.assertNotIn(json.dumps(str(Path.home()))[1:-1], self.text)

    def test_config_records_dataset_and_actual_seeds(self):
        dataset = self.needed["dataset"]
        self.assertEqual(dataset["name"], "TinySet")
        self.assertEqual(dataset["images_processed"], 2)
        self.assertEqual(
            [(i["index"], i["filename"], i["width"], i["height"])
             for i in dataset["images"]],
            [(0, "a.png", 32, 24), (1, "b.png", 24, 32)]
        )

        noise = self.needed["noise"]
        predicted = [
            (c["noise_type"], c["noise_value"],
             noise["seed_policy"]["base_seed"] + image * 1000 + c["seed_offset"])
            for image in range(2)
            for c in self.needed["conditions"]
        ]
        # main() first makes one four-noise preview image.
        self.assertEqual(predicted, self.seeds[4:])

    def test_parameter_snapshot_matches_filter_calls(self):
        recorded = [
            (name, entry["supplied"])
            for _ in range(2)
            for condition in self.needed["conditions"]
            for name, entry in condition["filters"].items()
        ]
        self.assertEqual(recorded, self.filter_calls)

        speckle_medium = next(
            c for c in self.needed["conditions"]
            if (c["noise_type"], c["level"]) == ("Speckle", "Medium")
        )
        lee = speckle_medium["filters"]["Lee Filter"]
        self.assertEqual(lee["resolved"]["speckle_variance"], 0.04)
        self.assertEqual(lee["model"], "multiplicative speckle")
        nlm = speckle_medium["filters"]["NLM Filter"]
        self.assertEqual(nlm["h_mode"], "explicit")
        self.assertEqual(nlm["resolved"]["h"], 0.15)


class DefaultModeTests(unittest.TestCase):
    def test_omitted_nlm_h_is_recorded_as_automatic(self):
        images = [("a.png", np.zeros((8, 8), dtype=np.float32))]

        with mock.patch.object(main, "TUNED_PARAMETERS_FILE", None), \
                mock.patch.object(main, "RESULTS_PATH", Path("result7")), \
                mock.patch.object(run_metadata, "get_git_info", return_value={}):
            config = main.build_current_run_config(images, None)

        self.assertEqual(config["run"]["mode"], "default_parameters")
        nlm = config["required_to_rerun"]["conditions"][0]["filters"]["NLM Filter"]
        self.assertEqual(nlm["supplied"], {})
        self.assertEqual(nlm["h_mode"], "automatic")
        self.assertIsNone(nlm["resolved"]["h"])


class EnvironmentTests(unittest.TestCase):
    def git_info(self, status="", fail=False):
        def fake_run(command, **kwargs):
            if fail:
                raise FileNotFoundError("git")
            self.assertIn("--no-optional-locks", command)
            if "HEAD" in command:
                output = "abc123\n"
            elif "--show-prefix" in command:
                output = "Denoising ad/\n"
            else:
                output = status
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

        with mock.patch.object(run_metadata.subprocess, "run", fake_run):
            return run_metadata.get_git_info()

    def test_git_states_are_distinguished(self):
        modified = " M Denoising ad/main.py\0"
        untracked = "?? Denoising ad/new.py\0"
        generated = (
            "?? Denoising ad/results/result12/summary.csv\0"
            " M Denoising ad/__pycache__/main.cpython-312.pyc\0"
        )
        cases = [
            ("", "clean", True),
            (modified, "modified", False),
            (untracked, "untracked_only", False),
            (modified + untracked, "modified_and_untracked", False),
            (generated, "clean", True),
        ]
        for status, state, matches in cases:
            with self.subTest(state=state):
                info = self.git_info(status)
                self.assertEqual(info["commit"], "abc123")
                self.assertEqual(info["working_tree"], state)
                self.assertIs(info["code_matches_commit"], matches)

    def test_missing_git_and_library_are_reported_as_unknown(self):
        info = self.git_info(fail=True)
        self.assertEqual(info["commit"], "unknown")
        self.assertEqual(info["working_tree"], "unknown")
        self.assertIsNone(info["code_matches_commit"])

        with mock.patch.object(
            run_metadata.importlib, "import_module",
            side_effect=ImportError("missing")
        ):
            versions = run_metadata.get_library_versions()
        self.assertEqual(versions["scikit-image"], "unavailable")


class SaveFailureTests(unittest.TestCase):
    def run_main(self, root, replace):
        write_small_dataset(root / "TinySet")
        output = io.StringIO()
        frame = pd.DataFrame()

        with mock.patch.object(main, "DATASET_PATH", root / "TinySet"), \
                mock.patch.object(main, "RESULTS_PATH", root / "results"), \
                mock.patch.object(main, "TUNED_PARAMETERS_FILE", None), \
                mock.patch.object(main, "MAX_IMAGES", None), \
                mock.patch.object(main, "save_noise_comparison"), \
                mock.patch.object(main, "run_experiments", return_value=frame) as run, \
                mock.patch.object(main, "save_results"), \
                mock.patch.object(main, "create_graphs"), \
                mock.patch.object(run_metadata, "get_git_info", return_value={}), \
                mock.patch.object(run_metadata.os, "replace", replace), \
                contextlib.redirect_stdout(output):
            with self.assertRaises(RuntimeError):
                main.main()

        self.assertNotIn("All experiments completed", output.getvalue())
        return run

    def test_failed_first_save_stops_before_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.run_main(
                Path(folder), mock.Mock(side_effect=OSError("disk full"))
            )
        run.assert_not_called()

    def test_failed_final_save_leaves_run_incomplete(self):
        real_replace = run_metadata.os.replace
        calls = []

        def fail_second_time(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise OSError("disk full")
            real_replace(source, target)

        with tempfile.TemporaryDirectory() as folder:
            run = self.run_main(Path(folder), fail_second_time)
            saved = json.loads(Path(calls[0]).read_text(encoding="utf-8"))

        run.assert_called_once()
        self.assertEqual(saved["run"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
