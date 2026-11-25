
"""
Tests for the image browser: one comparison picture per image in
full-dataset runs, and choosing an image in Results.

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s automated_code_tests -p "test_image_browser.py" -v

The real-job test uses two real BSD68 images in a temporary folder.
"""

import contextlib
import functools
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from test_full_dataset_job import make_job, make_project, run_job_in
from test_single_image_job import PROJECT_DIR

import app_helpers as helpers
import job_runner
import main
from saved_runs import needs_saved_runs

APP = str(PROJECT_DIR / "frontend" / "app.py")


def digest(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def use_results_folder(results):
    """Points the Results helpers at a temporary results folder."""
    stack = contextlib.ExitStack()

    for name in (
        "find_saved_runs",
        "load_summary",
        "load_image_rows",
        "comparison_images",
    ):
        function = getattr(helpers, name)
        stack.enter_context(
            mock.patch.object(
                helpers,
                name,
                functools.partial(function, results_dir=results),
            )
        )

    return stack


class SavedPicturesTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)

        make_project(cls.root)

        cls.originals = {
            name: digest(
                cv2.imread(
                    str(cls.root / "dataset" / "BSD68" / name),
                    0,
                ).astype(np.float32) / 255.0
            )
            for name in ("test001.png", "test003.png")
        }

        cls.outputs = []
        cls.saved = []

        real_save = main.save_image_comparison

        def recording_save(
            original,
            noisy,
            filtered_images,
            noise_type,
            level,
            file_name=None,
            **options,
        ):
            cls.saved.append({
                "file_name": file_name,
                "original": digest(original),
                "noisy": digest(noisy),
                "filtered": {
                    name: digest(image)
                    for name, image in filtered_images.items()
                },
            })

            return real_save(
                original,
                noisy,
                filtered_images,
                noise_type,
                level,
                file_name,
                **options,
            )

        def recording_filter(name, function):

            @functools.wraps(function)
            def wrapped(image, **kwargs):
                output = function(image, **kwargs)

                cls.outputs.append((
                    name,
                    digest(image),
                    digest(output),
                ))

                return output

            return wrapped

        recorded = {
            name: recording_filter(name, function)
            for name, function in main.filters.items()
        }

        with (
            mock.patch.object(
                main,
                "save_image_comparison",
                recording_save,
            ),
            mock.patch.dict(
                main.filters,
                recorded,
                clear=True,
            ),
        ):
            cls.states, _, _ = run_job_in(
                cls.root,
                make_job(),
            )

        cls.results = cls.root / "results"

        cls.rows = pd.read_csv(
            cls.results
            / "result1"
            / "experiment_results.csv"
        )

    @classmethod
    def tearDownClass(cls):
        st.cache_data.clear()
        cls.folder.cleanup()

    def test_each_image_gets_its_own_picture(self):
        self.assertEqual(
            self.states[-1][0],
            "completed",
        )

        pictures = sorted(
            path.relative_to(
                self.results / "result1" / "comparisons"
            ).as_posix()
            for path in (
                self.results / "result1" / "comparisons"
            ).rglob("*.png")
        )

        self.assertEqual(
            pictures,
            [
                "salt_and_pepper_medium/000_test001.png",
                "salt_and_pepper_medium/001_test003.png",
            ],
        )

        self.assertEqual(
            [saved["file_name"] for saved in self.saved],
            [
                "salt_and_pepper_medium/000_test001.png",
                "salt_and_pepper_medium/001_test003.png",
            ],
        )

    def test_pictures_use_the_already_calculated_images(self):
        # Three filters per image, two images, no recomputation.
        self.assertEqual(len(self.outputs), 6)

        for number, (name, saved) in enumerate(
            zip(
                ("test001.png", "test003.png"),
                self.saved,
            )
        ):
            calls = self.outputs[
                number * 3:(number + 1) * 3
            ]

            self.assertEqual(
                saved["original"],
                self.originals[name],
            )

            self.assertEqual(
                {
                    input_digest
                    for _, input_digest, _ in calls
                },
                {saved["noisy"]},
            )

            self.assertEqual(
                saved["filtered"],
                {
                    filter_name: output
                    for filter_name, _, output in calls
                },
            )

    def test_results_lets_me_choose_any_image_and_shows_its_scores(self):
        st.cache_data.clear()

        with (
            use_results_folder(self.results),
            mock.patch.object(
                job_runner,
                "start_job",
                return_value=None,
            ) as start,
        ):
            app = AppTest.from_file(
                APP,
                default_timeout=30,
            )

            # The app now opens on Home.
            app.session_state["view"] = "results"
            app.run()

            self.assertEqual(len(app.exception), 0)

            chooser = app.selectbox(key="compare_image")

            self.assertEqual(
                list(chooser.options),
                ["test001.png", "test003.png"],
            )

            averages = app.dataframe[0].value.copy()

            chooser.set_value("test003.png").run()

            self.assertEqual(len(app.exception), 0)
            start.assert_not_called()

        pd.testing.assert_frame_equal(
            app.dataframe[0].value,
            averages,
        )

        scores = next(
            table.value
            for table in app.dataframe
            if "Best on this image" in table.value.columns
        ).set_index("Filter")

        expected = self.rows[
            self.rows["Image"] == "test003.png"
        ].set_index("Filter")

        for filter_name, row in expected.iterrows():
            label = helpers.short_name(
                filter_name
            ).replace(
                "Noisy (Baseline)",
                "Noisy (no filter)",
            )

            self.assertEqual(
                scores.loc[label, "PSNR (dB)"],
                f"{row['PSNR']:.2f}",
            )

            self.assertEqual(
                scores.loc[label, "SSIM"],
                f"{row['SSIM']:.3f}",
            )

        best = expected.drop(
            "Noisy (Baseline)"
        )["PSNR"].idxmax()

        self.assertEqual(
            scores.loc[
                helpers.short_name(best),
                "Best on this image",
            ],
            "★",
        )

        self.assertIn(
            "test003.png only",
            " ".join(item.value for item in app.info),
        )


@needs_saved_runs("result13", "result14", "result16")
class OlderResultsTests(unittest.TestCase):

    def test_older_runs_list_only_the_pictures_they_saved(self):
        runs = helpers.find_saved_runs()

        # Expectations match the saved local experiments.
        expected = {
            "result13": (
                "Salt-and-Pepper",
                "Medium",
                ["test001.png"],
            ),
            "result14": (
                "Salt-and-Pepper",
                "Medium",
                ["test001.png"],
            ),
            "result16": (
                "Salt-and-Pepper",
                "Medium",
                ["test001.png"],
            ),
        }

        for run, (noise, level, names) in expected.items():
            with self.subTest(run=run):
                pictures = helpers.comparison_images(
                    runs[run],
                    run,
                    noise,
                    level,
                )

                self.assertEqual(
                    list(pictures),
                    names,
                )

                self.assertTrue(
                    all(
                        path.is_file()
                        for path in pictures.values()
                    )
                )

    def test_result16_opens_with_its_one_saved_picture(self):
        st.cache_data.clear()

        app = AppTest.from_file(
            APP,
            default_timeout=30,
        )

        app.session_state["view"] = "results"

        app.session_state["open_result"] = {
            "run": "result16",
            "noise_type": "Salt-and-Pepper",
            "noise_level": "Medium",
        }

        app.run()

        # The page should open without exceptions.
        self.assertEqual(
            len(app.exception),
            0,
        )

        # Only one image was processed, so no image chooser is needed.
        self.assertNotIn(
            "compare_image",
            [box.key for box in app.selectbox],
        )

        self.assertIn(
            "These scores are for one image (test001.png)",
            " ".join(
                caption.value
                for caption in app.caption
            ),
        )

        # Read the real saved rows instead of assuming
        # that every experiment used all six filters.
        saved_rows = helpers.load_image_rows("result16")

        expected_rows = saved_rows[
            (saved_rows["Image"] == "test001.png")
            & (
                saved_rows["Noise_Type"]
                == "Salt-and-Pepper"
            )
            & (
                saved_rows["Noise_Level"]
                == "Medium"
            )
        ]

        self.assertGreater(
            len(expected_rows),
            0,
        )

        # The second dataframe contains per-image scores.
        # Its row count must match the saved experiment.
        self.assertEqual(
            len(app.dataframe[1].value),
            len(expected_rows),
        )

        # Check that the saved comparison picture exists.
        runs = helpers.find_saved_runs()

        pictures = helpers.comparison_images(
            runs["result16"],
            "result16",
            "Salt-and-Pepper",
            "Medium",
        )

        self.assertEqual(
            list(pictures),
            ["test001.png"],
        )

        self.assertTrue(
            pictures["test001.png"].is_file()
        )


if __name__ == "__main__":
    unittest.main()
