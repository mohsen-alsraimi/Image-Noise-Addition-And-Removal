"""
Frontend page tests: navigation between the five pages, "Try an image",
simplified presets and the Results layout.

Run from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -B -m unittest discover -s automated_code_tests -p "test_frontend_pages.py" -v

Starting jobs is mocked; saved results are only read.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from streamlit.testing.v1 import AppTest

from test_filter_settings import param_key, settings
from test_single_image_job import PROJECT_DIR

import app_helpers as helpers
import experiment_job as jobs
import job_runner
import main
from saved_runs import needs_saved_runs

APP = str(PROJECT_DIR / "frontend" / "app.py")


def tuned(filter_name, noise_type, level):
    return jobs.preset_parameters("new_12_image", filter_name, main.filters[filter_name], noise_type, level)


def default(filter_name, noise_type="Gaussian", level="Low"):
    return jobs.preset_parameters("default", filter_name, main.filters[filter_name], noise_type, level)


class NavigationTests(unittest.TestCase):
    def test_opens_on_home_and_buttons_switch_pages(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None):
            app = AppTest.from_file(APP, default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            nav = app.segmented_control(key="view")
            self.assertEqual(nav.value, "home")
            # AppTest shows the labels without their icons
            self.assertEqual(list(nav.options),
                             ["Home", "Try an image", "Run an experiment", "Explore results", "Learn"])
            self.assertIn("See how filters repair a damaged image", [t.value for t in app.title])

            # Each page starts with its own heading (also when no runs are saved yet)
            for button, page, heading in (("home_try", "try", "Try an image"),
                                          ("home_run", "run", "Run an experiment"),
                                          ("home_results", "results", "Explore results"),
                                          ("home_learn", "learn", "Learn the basics")):
                app.segmented_control(key="view").set_value("home").run()
                app.button(key=button).click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.segmented_control(key="view").value, page)
                self.assertEqual([h.value for h in app.header][0], heading)

    def test_learn_page_explains_the_basic_terms(self):
        app = AppTest.from_file(APP, default_timeout=30)
        app.session_state["view"] = "learn"
        app.run()
        text = " ".join(m.value for m in app.markdown)
        for term in ("Noise", "Filter", "Kernel / window size", "PSNR", "SSIM", "Edge F1", "Dataset", "Preset"):
            self.assertIn(f"**{term}:**", text)


class TryImageTests(unittest.TestCase):
    def open_try(self):
        app = AppTest.from_file(APP, default_timeout=30)
        app.session_state["view"] = "try"
        return app.run()

    def test_try_sends_the_same_single_image_job_format(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-try") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = self.open_try()
            self.assertEqual(len(app.exception), 0)
            app.selectbox(key="try_dataset").set_value("Set12").run()
            app.selectbox(key="try_image").set_value("05.png")
            app.selectbox(key="try_noise").set_value("Speckle")
            app.segmented_control(key="try_level").set_value("High").run()
            app.pills(key="try_filters").set_value(["Median Filter", "Lee Filter"]).run()
            app.radio(key="try_preset").set_value("new_12_image").run()
            self.assertEqual(len(app.exception), 0)

            # Optimized values for Speckle / High fill the controls
            self.assertEqual(app.selectbox(key="try::param::Lee Filter::window_size").value,
                             tuned("Lee Filter", "Speckle", "High")["window_size"])
            start.assert_not_called()

            app.button(key="try_run").click().run()
            self.assertEqual(len(app.exception), 0)
            job = start.call_args.args[0]
            self.assertEqual(job, {
                "type": "single_image", "dataset": "Set12", "image": "05.png",
                "noise_type": "Speckle", "noise_level": "High",
                "filters": {
                    "Median Filter": settings("new_12_image", **tuned("Median Filter", "Speckle", "High")),
                    "Lee Filter": settings("new_12_image", **tuned("Lee Filter", "Speckle", "High")),
                },
            })
            jobs.validate_job_filters(job)
            self.assertEqual(app.session_state["try_job"], "job-try")

    def test_hand_edits_are_marked_custom_and_sent(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-try") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = self.open_try()
            app.pills(key="try_filters").set_value(["Median Filter"]).run()
            app.selectbox(key="try::param::Median Filter::kernel_size").set_value(7).run()
            self.assertIn("Custom values", " ".join(c.value for c in app.caption))
            app.button(key="try_run").click().run()
            self.assertEqual(start.call_args.args[0]["filters"], {"Median Filter": settings("default", kernel_size=7)})

    def test_image_damage_offers_default_settings_only(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-try") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = self.open_try()
            app.radio(key="try_preset").set_value("new_12_image").run()
            app.radio(key="try_kind").set_value("Image damage").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(list(app.radio(key="try_preset").options), ["Default settings"])
            app.selectbox(key="try_degradation").set_value("Glare")
            app.pills(key="try_filters").set_value(["Gaussian Filter"]).run()
            app.button(key="try_run").click().run()
            job = start.call_args.args[0]
            self.assertEqual((job["experiment"], job["noise_type"], job["noise_level"]), ("degradation", "Glare", "Medium"))
            self.assertEqual(job["filters"], {"Gaussian Filter": settings("default", **default("Gaussian Filter"))})
            jobs.validate_job_filters(job)

    @needs_saved_runs('result14')
    def test_finished_test_is_shown_on_the_try_page(self):
        # result14 is an existing one-image run; it is only read
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "job_state", return_value={"state": "completed", "result": "result14"}), \
                mock.patch.object(job_runner, "result_is_complete", return_value=True):
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["view"] = "try"
            app.session_state["watched_job"] = "job-try"
            app.session_state["try_job"] = "job-try"
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.segmented_control(key="view").value, "try")
            self.assertEqual(app.session_state["try_result"], "result14")
            self.assertIn("Result of your test", " ".join(m.value for m in app.markdown))
            self.assertIn("Status: Completed", " ".join(s.value for s in app.success))

            app.button(key="try_open_result").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.segmented_control(key="view").value, "results")
            self.assertEqual(app.selectbox(key="saved_run").value, "result14")
        st.cache_data.clear()


class PresetProvenanceTests(unittest.TestCase):
    def open_run(self):
        app = AppTest.from_file(APP, default_timeout=30)
        app.session_state["view"] = "run"
        return app

    def test_hidden_old_preset_is_never_used_silently(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-1") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = self.open_run()
            app.session_state["run_filter"] = "Bilateral Filter"
            app.session_state["preset::Bilateral Filter"] = "previous_2_image"
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.selectbox(key="preset::Bilateral Filter").value, "default")
            app.button(key="run_experiment").click().run()
            sent = start.call_args.args[0]["filters"]["Bilateral Filter"]
            self.assertEqual(sent, settings("default", **default("Bilateral Filter")))

    def test_unedited_optimized_values_follow_the_condition(self):
        with mock.patch.object(job_runner, "active_job_id", return_value=None), \
                mock.patch.object(job_runner, "start_job", return_value="job-1") as start, \
                mock.patch.object(job_runner, "job_state", return_value={"state": "running"}):
            app = self.open_run()
            app.session_state["run_filter"] = "Median Filter"
            app.run()
            app.selectbox(key="preset::Median Filter").set_value("new_12_image").run()

            conditions = [("Gaussian", "Low"), ("Salt-and-Pepper", "High"), ("Speckle", "Medium")]
            for noise, level in conditions:
                app.selectbox(key="run_noise").set_value(noise)
                app.selectbox(key="run_level").set_value(level).run()
                self.assertEqual(app.selectbox(key=param_key("Median Filter", "kernel_size")).value,
                                 tuned("Median Filter", noise, level)["kernel_size"], (noise, level))
            app.button(key="run_experiment").click().run()
            self.assertEqual(start.call_args.args[0]["filters"],
                             {"Median Filter": settings("new_12_image", **tuned("Median Filter", "Speckle", "Medium"))})

            # A hand edit is kept when the condition changes, and marked custom
            other = 7 if tuned("Median Filter", "Speckle", "Medium")["kernel_size"] != 7 else 3
            app.selectbox(key=param_key("Median Filter", "kernel_size")).set_value(other).run()
            app.selectbox(key="run_noise").set_value("Poisson").run()
            self.assertEqual(app.selectbox(key=param_key("Median Filter", "kernel_size")).value, other)
            self.assertIn("Custom values", " ".join(c.value for c in app.caption))
            self.assertIn("(custom values)", " ".join(m.value for m in app.markdown))


@needs_saved_runs('result13')
class ResultsLayoutTests(unittest.TestCase):
    def test_full_run_summary_first_details_in_tabs(self):
        st.cache_data.clear()
        app = AppTest.from_file(APP, default_timeout=60)
        app.session_state["open_result"] = {"run": "result13", "noise_type": "Gaussian", "noise_level": "Medium"}
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([t.label for t in app.tabs], [
            ":material/bar_chart: Compare filters", ":material/grid_view: All 12 conditions",
            ":material/image: Images", ":material/tune: Details and settings",
        ])
        self.assertTrue(app.subheader[0].value.startswith("Best filter by average PSNR:"))
        self.assertIn("Filtering helped.", " ".join(s.value for s in app.success))
        self.assertIn("old 2-image tuned settings", app.selectbox(key="saved_run").format_func("result13"))
        self.assertIn("Old tuned settings", " ".join(w.value for w in app.warning))
        tables = [t.value for t in app.dataframe]
        self.assertTrue(any("PSNR change vs noisy (dB)" in t.columns for t in tables))
        self.assertTrue(any("Settings used" in t.columns for t in tables))
        self.assertTrue(any("Filtering helped?" in t.columns for t in tables))
        st.cache_data.clear()


if __name__ == "__main__":
    unittest.main()
