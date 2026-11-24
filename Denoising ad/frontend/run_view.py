"""
"Run an experiment": choose images, what to test, filters and settings, then
start the normal runner (python main.py --job job.json) in the background.

Also holds the parts shared with "Try an image": the job status line and the
filter-setting controls. This file contains no image processing.
"""

import streamlit as st

import app_helpers as helpers
import experiment_job as jobs
import job_runner
import theme
from explanations import (
    DATASET_NOTES,
    DEGRADATION_TEXT,
    FILTER_GOOD_FOR,
    FILTER_TEXT,
    GLOSSARY,
    NOISE_TEXT,
    PARAMETER_HELP,
    PARAMETER_LABELS,
)
from run_metadata import DEGRADATION_PARAMETER_MEANING, NOISE_PARAMETER_MEANING

ALL_FILTERS = "All six filters"
ONE_IMAGE = "One selected image"
FULL_DATASET = "Full dataset"
ONE_CONDITION = "One noise condition"
ALL_CONDITIONS = "All noise conditions (full experiment)"
NOISE = "Noise"
DEGRADATION = "Image degradation"
ONE_DEGRADATION = "One degradation condition"
ALL_DEGRADATIONS = "All degradation conditions (JPEG and Glare)"

DEFAULT_ONLY_TEXT = (
    "Optimized settings were tuned for noise on Set12 and have not been checked for "
    "JPEG compression or glare, so only Default settings are offered here. "
    "You can still change the values by hand."
)

# Measured size of one compact comparison picture (BSD68 image, six filters)
PICTURE_MB = 0.2


def keep_valid_choice(key, options):
    """Forgets a saved dropdown choice that is no longer available."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def image_size(path):
    """Width and height of a dataset image, read from its header only."""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


# --------------------------------------------------
# JOB STATUS (shown on every page)
# --------------------------------------------------

@st.fragment(run_every="1s")
def watch_running_job(job_id):
    """Checks the job once per second without rerunning the whole page."""
    state = job_runner.job_state(job_id)
    if state["state"] not in jobs.ACTIVE_STATES:
        st.rerun()

    total = state.get("images_total")
    done = state.get("images_done", 0)

    if not total:
        st.info("**Status: Running.** Loading the images...", icon=":material/hourglass_top:")
    elif done < total:
        st.info(f"**Status: Running.** Processing image {done + 1} of {total} ({done} finished).",
                icon=":material/hourglass_top:")
        st.progress(done / total)
    else:
        st.info(f"**Status: Running.** All {total} images processed. Saving the results...",
                icon=":material/hourglass_top:")
        st.progress(1.0)


def show_job_status():
    """Shows the job state. Returns True while an experiment is running."""
    active_job = job_runner.active_job_id()
    if active_job:
        st.session_state["watched_job"] = active_job

    job_id = st.session_state.get("watched_job")

    if job_id:
        state = job_runner.job_state(job_id)

        if state["state"] in jobs.ACTIVE_STATES:
            watch_running_job(job_id)
            return True

        del st.session_state["watched_job"]

        # Trust the saved run_config.json, not only the job status.
        if state["state"] == "completed" and not job_runner.result_is_complete(state.get("result")):
            state = {
                "error": "The runner stopped, but the saved results are not marked complete.",
                "incomplete_result": state.get("result"),
            }

        if state.get("state") == "completed":
            if st.session_state.get("try_job") == job_id:
                # Started from "Try an image": show the result there.
                st.session_state.pop("try_job")
                st.session_state["try_result"] = state["result"]
                # Applied by app.py before the navigation is drawn
                st.session_state["open_view"] = "try"
            else:
                job = jobs.read_json(job_runner.JOBS_DIR / job_id / "job.json")
                st.session_state["open_result"] = {
                    "run": state["result"],
                    "noise_type": job["noise_type"],
                    "noise_level": job["noise_level"],
                }
            st.session_state["finished_message"] = (
                f"**Status: Completed.** Your experiment was saved as {state['result']}."
            )
            st.rerun()

        st.session_state.pop("try_job", None)
        st.session_state["failed_job"] = {
            "id": job_id,
            "error": state.get("error", "Unknown error."),
            "incomplete_result": state.get("incomplete_result"),
        }

    message = st.session_state.pop("finished_message", None)
    if message:
        st.success(message, icon=":material/check_circle:")

    failed = st.session_state.get("failed_job")
    if failed:
        st.error(f"**Status: Failed.** {failed['error']}", icon=":material/error:")
        with st.expander("Technical details"):
            st.write(f"Log folder: `.frontend_jobs/{failed['id']}`")
            if failed["incomplete_result"]:
                st.write(
                    f"`results/{failed['incomplete_result']}` is incomplete, "
                    "so it will not appear in Explore results."
                )
            st.code(job_runner.read_log_tail(failed["id"]) or "No log was saved.", language=None)

    return False


def start(job, try_page=False):
    """Starts a job through job_runner and reports problems in plain words."""
    try:
        job_id = job_runner.start_job(job)
    except job_runner.JobAlreadyRunning:
        st.warning("An experiment is already running. Please wait for it to finish.")
    except ValueError as error:
        st.error(f"These settings cannot be used: {error}")
    except OSError as error:
        st.error(f"The experiment could not start: {error}")
    else:
        st.session_state["watched_job"] = job_id
        if try_page:
            st.session_state["try_job"] = job_id
            st.session_state.pop("try_result", None)
        st.session_state.pop("failed_job", None)
        st.rerun()


# --------------------------------------------------
# FILTER SETTINGS (shared with "Try an image")
# --------------------------------------------------

def setting_key(filter_name, name, prefix=""):
    return f"{prefix}param::{filter_name}::{name}"


def preset_key(filter_name, prefix=""):
    return f"{prefix}preset::{filter_name}"


def show_value(value):
    return "Automatic" if value is None else str(value)


def preset_options(degradation):
    """Default and Optimized for noise; Default only for JPEG and glare."""
    return ["default"] if degradation else list(helpers.VISIBLE_PRESETS)


def current_values(filter_name, prefix=""):
    return {
        name: st.session_state.get(setting_key(filter_name, name, prefix))
        for name in jobs.PARAMETER_CHOICES[filter_name]
    }


def load_preset(filter_name, filter_function, noise_type, noise_level, prefix=""):
    """Copies a preset's values for this condition into the filter's controls."""
    preset = st.session_state[preset_key(filter_name, prefix)]

    try:
        values = jobs.preset_parameters(
            preset, filter_name, filter_function, noise_type, noise_level
        )
    except ValueError as error:
        st.session_state[f"{prefix}preset_error::{filter_name}"] = str(error)
        return

    for name, value in values.items():
        st.session_state[setting_key(filter_name, name, prefix)] = value
    st.session_state[f"{prefix}loaded::{filter_name}"] = (preset, noise_type, noise_level)


def follow_preset(filter_name, filter_function, noise_type, noise_level, prefix=""):
    """Keeps the controls on the preset's values for the chosen condition.

    Values the user changed by hand are kept. Tuned presets can differ by
    noise type and strength, so unchanged values follow the condition.
    """
    preset = st.session_state[preset_key(filter_name, prefix)]
    wanted = (preset, noise_type, noise_level)
    loaded = st.session_state.get(f"{prefix}loaded::{filter_name}")

    # First visit: no values yet
    if any(setting_key(filter_name, name, prefix) not in st.session_state
           for name in jobs.PARAMETER_CHOICES[filter_name]):
        load_preset(filter_name, filter_function, noise_type, noise_level, prefix)
        return
    if loaded == wanted:
        return

    values = current_values(filter_name, prefix)
    if loaded is None or loaded[0] != preset:
        load_preset(filter_name, filter_function, noise_type, noise_level, prefix)
        return

    # Only the condition changed: follow it unless the values were edited.
    try:
        old_values = jobs.preset_parameters(loaded[0], filter_name, filter_function, *loaded[1:])
    except ValueError:
        old_values = None
    if values == old_values:
        load_preset(filter_name, filter_function, noise_type, noise_level, prefix)
    else:
        st.session_state[f"{prefix}loaded::{filter_name}"] = wanted


def show_filter_settings(filter_name, filter_function, noise_type, noise_level, noise_value,
                         degradation=False, prefix="", show_preset=True):
    """Shows one filter's starting preset and its real settings. Returns the choices."""
    key = preset_key(filter_name, prefix)
    choices = jobs.PARAMETER_CHOICES[filter_name]
    options = preset_options(degradation)

    # First visit: start every control from the Default preset.
    if key not in st.session_state:
        st.session_state[key] = "default"
    # A hidden or unavailable preset (the old 2-image preset, or a tuned preset
    # for JPEG/glare) is never used silently: go back to Default values.
    elif st.session_state[key] not in options:
        st.session_state[key] = "default"
        load_preset(filter_name, filter_function, noise_type, noise_level, prefix)

    follow_preset(filter_name, filter_function, noise_type, noise_level, prefix)

    if show_preset:
        st.selectbox(
            "Start from",
            options,
            format_func=helpers.preset_label,
            key=key,
            on_change=load_preset,
            args=(filter_name, filter_function, noise_type, noise_level, prefix),
            persist_state="page",
            help=GLOSSARY["Preset"],
        )
    preset = st.session_state[key]

    error = st.session_state.pop(f"{prefix}preset_error::{filter_name}", None)
    if error:
        st.error(f"This preset could not be loaded: {error}")

    values = {}
    for column, (name, options) in zip(st.columns(len(choices)), choices.items()):
        values[name] = column.selectbox(
            PARAMETER_LABELS.get(name, name),
            options,
            format_func=show_value,
            key=setting_key(filter_name, name, prefix),
            persist_state="page",
            help=f"{PARAMETER_HELP[filter_name][name]} (setting name: {name})",
        )

    if filter_name == "Lee Filter":
        if noise_type == "Speckle":
            st.caption(
                f"Lee also receives speckle variance {noise_value}, taken from the noise "
                "choice above. It is not a separate setting."
            )
        elif degradation:
            st.caption("Lee's speckle model is off, because this is not speckle noise.")
        else:
            st.caption("Lee's speckle model is off, because the noise is not Speckle.")

    label = helpers.preset_label(preset)
    try:
        preset_values = jobs.preset_parameters(
            preset, filter_name, filter_function, noise_type, noise_level
        )
    except ValueError as error:
        st.warning(f"The selected preset is not available: {error}")
    else:
        if values == preset_values:
            st.caption(f"Using {label} for {noise_level} {noise_type}.")
        else:
            st.caption(
                f":orange-badge[Custom values] Changed by hand. "
                f"{label} for {noise_level} {noise_type} would be: "
                f"{helpers.format_settings({'resolved': preset_values})}."
            )
            st.button(
                f"Reset to {label}",
                key=f"{prefix}reload::{filter_name}",
                on_click=load_preset,
                args=(filter_name, filter_function, noise_type, noise_level, prefix),
                icon=":material/undo:",
            )

    return {"preset": preset, "parameters": values}


def show_all_conditions_settings(filter_name, filter_function, noise_levels, degradation=False):
    """Settings for a run over all conditions. Returns the choices."""
    key = preset_key(filter_name)
    choices = jobs.PARAMETER_CHOICES[filter_name]
    conditions = [(noise, level) for noise, levels in noise_levels.items() for level in levels]
    options = preset_options(degradation)

    if st.session_state.get(key) not in options:
        st.session_state[key] = "default"

    preset = st.selectbox(
        "Start from",
        options,
        format_func=helpers.preset_label,
        key=key,
        persist_state="page",
        help=GLOSSARY["Preset"],
    )
    label = helpers.preset_label(preset)

    own_values = st.checkbox(
        f"Use my own values for all {len(conditions)} conditions",
        key=f"own::{filter_name}",
        persist_state="page",
    )

    if not own_values:
        st.caption(
            f"Uses the {label} values saved for each condition. "
            "Optimized settings can differ by noise type and strength."
        )
        try:
            table = [
                {"Condition": f"{noise} / {level}",
                 "Values": helpers.format_settings({"resolved": jobs.preset_parameters(
                     preset, filter_name, filter_function, noise, level)})}
                for noise, level in conditions
            ]
        except ValueError as error:
            st.warning(f"This preset is not available: {error}")
        else:
            with st.expander(f"See the {label} values for each condition"):
                st.dataframe(table, hide_index=True, width="stretch",
                             height=helpers.table_height(len(table)))
        return {"preset": preset, "parameters": None}

    # First use: start the controls from the filter's default values.
    if any(setting_key(filter_name, name) not in st.session_state for name in choices):
        defaults = jobs.preset_parameters("default", filter_name, filter_function, *conditions[0])
        for name, value in defaults.items():
            st.session_state[setting_key(filter_name, name)] = value

    values = {}
    for column, (name, options) in zip(st.columns(len(choices)), choices.items()):
        values[name] = column.selectbox(
            PARAMETER_LABELS.get(name, name), options, format_func=show_value,
            key=setting_key(filter_name, name), persist_state="page",
            help=f"{PARAMETER_HELP[filter_name][name]} (setting name: {name})",
        )

    st.caption(
        f":orange-badge[Custom values] These exact values are used for all {len(conditions)} conditions."
    )
    if filter_name == "Lee Filter" and not degradation:
        st.caption("In the Speckle conditions, Lee also receives that condition's speckle variance.")

    return {"preset": preset, "parameters": values}


def describe_choice(chosen, all_conditions):
    """One filter's settings in the review list."""
    if chosen["parameters"] is None:
        return f"{helpers.preset_label(chosen['preset'])} for each condition"
    text = helpers.format_settings({"resolved": chosen["parameters"]})
    if all_conditions:
        # Own values in an all-conditions run are always custom
        return f"{text} (custom values, every condition)"
    try:
        preset_values = jobs.preset_parameters(
            chosen["preset"], chosen["name"], chosen["function"], *chosen["condition"]
        )
    except ValueError:
        preset_values = None
    source = helpers.preset_label(chosen["preset"]) if chosen["parameters"] == preset_values else "custom values"
    return f"{text} ({source})"


# --------------------------------------------------
# RUN AN EXPERIMENT
# --------------------------------------------------

def show_run_view(is_running):
    # Load the scientific code only when this view is opened.
    import main as backend

    st.header("Run an experiment")
    st.write(
        "Test filters scientifically: on one image or a whole dataset, under one condition "
        "or all of them. Every run is saved and can be opened in Explore results."
    )

    # ---- 1. Images
    with st.container(border=True):
        theme.step(1, "Choose the images",
                   "A dataset is a folder of test images. Test one image, or every image in it.")
        choose_column, preview_column = st.columns([3, 1], gap="large")
        scope = choose_column.radio(
            "Images to process",
            [ONE_IMAGE, FULL_DATASET],
            horizontal=True,
            key="run_scope",
            persist_state="page",
        )
        dataset_column, image_column = choose_column.columns(2)
        dataset = dataset_column.selectbox(
            "Dataset", backend.JOB_DATASETS, key="run_dataset", persist_state="page",
            help=GLOSSARY["Dataset"],
        )

        image_paths = backend.find_image_paths(backend.PROJECT_DIR / "dataset" / dataset)
        image_names = [path.name for path in image_paths]
        dataset_column.caption(f"{dataset} has {len(image_names)} images.")

        if scope == ONE_IMAGE:
            keep_valid_choice("run_image", image_names)
            image = image_column.selectbox(
                "Image", image_names, key="run_image", persist_state="page"
            )
            path = image_paths[image_names.index(image)]
            width, height = image_size(path)
            preview_column.image(str(path), width="stretch")
            preview_column.caption(f"{image}, {width} × {height} pixels")
        else:
            image = None
            image_column.info(f"{dataset}: {len(image_names)} images will be processed.")
            choose_column.caption(
                "Each image gets its own noise from the same seed rule as single-image runs. "
                "This can take several minutes; NLM and TV are the slowest filters."
            )

        if dataset in DATASET_NOTES:
            kind, note = DATASET_NOTES[dataset]
            (st.warning if kind == "warning" else st.info)(note)

    # ---- 2. What to test
    with st.container(border=True):
        theme.step(2, "Choose what to test",
                   "Noise is random grain or dots. Image damage is JPEG blocks or glare. "
                   "Strengths are fixed values, so different runs can be compared.")
        test_kind = st.radio(
            "What to test",
            [NOISE, DEGRADATION],
            horizontal=True,
            key="run_test_kind",
            persist_state="page",
            help="Noise: Gaussian, Salt-and-Pepper, Speckle, Poisson. "
                 "Image degradation: JPEG compression and glare.",
        )
        degradation = test_kind == DEGRADATION
        condition_levels = backend.DEGRADATION_LEVELS if degradation else backend.NOISE_LEVELS
        condition_text = DEGRADATION_TEXT if degradation else NOISE_TEXT
        meanings = DEGRADATION_PARAMETER_MEANING if degradation else NOISE_PARAMETER_MEANING
        one_label, all_label = (ONE_DEGRADATION, ALL_DEGRADATIONS) if degradation else (ONE_CONDITION, ALL_CONDITIONS)
        damaged = "degraded image" if degradation else "noisy image"
        noise_scope = one_label
        save_pictures = True

        if degradation:
            st.info(
                "JPEG compression and glare are not noise. The filters were made for noise, "
                "so they may make these images worse. Results compare every filter with the "
                "unfiltered degraded image."
            )

        if scope == FULL_DATASET:
            noise_scope = st.radio(
                "Degradation conditions" if degradation else "Noise conditions",
                [one_label, all_label],
                horizontal=True,
                key="run_degradation_scope" if degradation else "run_noise_scope",
                persist_state="page",
            )

        all_conditions = noise_scope == all_label
        condition_count = sum(len(levels) for levels in condition_levels.values())

        if all_conditions:
            noise_type = level = None
            levels = {}
            if degradation:
                st.info(
                    f"All {condition_count} degradation conditions: JPEG compression and glare × "
                    "3 strengths. This is a separate experiment from the 12 noise conditions."
                )
            else:
                st.info(
                    f"All {condition_count} noise conditions: 4 noise types × 3 strengths, "
                    "like the full terminal experiment. Every image is tested under each one."
                )
            st.caption("; ".join(
                f"{noise}: " + ", ".join(f"{name} {value}" for name, value in strengths.items())
                for noise, strengths in condition_levels.items()
            ))
            picture_count = len(image_names) * condition_count
            save_pictures = st.checkbox(
                "Save a comparison picture for every image and condition",
                value=True,
                key="run_save_pictures",
                persist_state="page",
            )
            st.caption(
                f"{picture_count} pictures, about {picture_count * PICTURE_MB:.0f} MB. "
                "Saving them adds roughly half a second per picture. "
                "Without pictures, Explore results still shows every score."
            )
        else:
            noise_columns = st.columns(2)
            noise_type = noise_columns[0].selectbox(
                "Degradation type" if degradation else "Noise type",
                list(condition_levels),
                key="run_degradation" if degradation else "run_noise",
                persist_state="page",
            )

            levels = condition_levels[noise_type]
            level = noise_columns[1].selectbox(
                "Damage strength" if degradation else "Noise strength",
                list(levels),
                key="run_degradation_level" if degradation else "run_level",
                persist_state="page",
                help=GLOSSARY["Strength"],
            )
            st.caption(
                f"{condition_text[noise_type]} Saved strength: {levels[level]} "
                f"({meanings[noise_type]}). "
                + ("These settings are fixed; filters only try to repair the damage." if degradation
                   else "Noise settings are fixed; filters only clean it.")
            )

    # ---- 3. Filters
    with st.container(border=True):
        theme.step(3, "Choose filters", GLOSSARY["Filter"])
        filter_choice = st.selectbox(
            "Filters",
            [ALL_FILTERS, *backend.filters],
            format_func=lambda name: name if name == ALL_FILTERS else helpers.short_name(name),
            key="run_filter",
            persist_state="page",
        )
        chosen_names = list(backend.filters) if filter_choice == ALL_FILTERS else [filter_choice]

        if filter_choice == ALL_FILTERS:
            st.caption("Every filter cleans the exact same image, so the comparison is fair.")
        else:
            short = helpers.short_name(filter_choice)
            st.caption(f"{FILTER_TEXT[short]} {FILTER_GOOD_FOR[short]}")

    # ---- 4. Filter settings
    chosen_settings = {}
    with st.container(border=True):
        theme.step(4, "Adjust filter settings (optional)",
                   "Each filter starts from Default settings. Choose Optimized settings to use values "
                   "tuned on Set12, or change any value by hand.")
        if degradation:
            st.caption(DEFAULT_ONLY_TEXT)

        def settings_for(name):
            if all_conditions:
                return show_all_conditions_settings(
                    name, backend.filters[name], condition_levels, degradation
                )
            return show_filter_settings(
                name, backend.filters[name], noise_type, level, levels[level], degradation
            )

        for name in chosen_names:
            if filter_choice == ALL_FILTERS:
                short = helpers.short_name(name)
                with st.expander(short):
                    st.caption(FILTER_GOOD_FOR[short])
                    chosen_settings[name] = settings_for(name)
            else:
                chosen_settings[name] = settings_for(name)

    # ---- 5. Review and run
    with st.container(border=True):
        theme.step(5, "Review and run")
        review = {
            name: {**chosen, "name": name, "function": backend.filters[name],
                   "condition": (noise_type, level)}
            for name, chosen in chosen_settings.items()
        }
        filter_lines = "\n".join(
            f"  - {helpers.short_name(name)}: {describe_choice(chosen, all_conditions)}"
            for name, chosen in review.items()
        )
        same_image = f" Every filter cleans the exact same {damaged}." if len(chosen_names) > 1 else ""
        if image is None:
            image_line = f"- **Images:** all {len(image_names)} images from {dataset}, one after another.\n"
        else:
            image_line = f"- **Image:** `{image}` from {dataset}.\n"
        condition_word = "Image damage" if degradation else "Noise"
        if all_conditions:
            noise_line = (
                f"- **{condition_word}:** all {condition_count} "
                f"{'degradation ' if degradation else ''}conditions, one after another for each image.\n"
            )
        else:
            noise_line = f"- **{condition_word}:** {level} {noise_type}, strength {levels[level]}.\n"
        if degradation:
            noise_line += (
                "- **Compared with:** the unfiltered degraded image. Only Default settings "
                "(or your own edits) are used, because optimized settings were made for noise.\n"
            )
        st.markdown(
            image_line + noise_line +
            f"- **Filters and settings:**{same_image}\n{filter_lines}\n"
            "- **Result:** PSNR, SSIM, edge scores, execution time, and images are saved in a new folder."
        )

        if is_running:
            status = "Running"
        elif st.session_state.get("failed_job"):
            status = "Failed"
        else:
            status = "Ready"
        st.write(f"**Status:** {status}")

        if st.button("Run experiment", type="primary", disabled=is_running, key="run_experiment",
                     icon=":material/play_arrow:"):
            job = {
                "type": "single_image" if scope == ONE_IMAGE else "full_dataset",
                "dataset": dataset,
                "image": image,
                "noise_type": noise_type,
                "noise_level": level,
                "filters": chosen_settings,
            }
            # Noise jobs keep their earlier format; a missing field means "noise".
            if degradation:
                job["experiment"] = "degradation"
            if all_conditions:
                job["noise_scope"] = "all"
                job["save_pictures"] = save_pictures

            start(job)
