"""The Run Experiment view: choose one image and start the normal runner."""

import streamlit as st

import app_helpers as helpers
import experiment_job as jobs
import job_runner
from explanations import FILTER_TEXT, NOISE_TEXT, PARAMETER_HELP
from run_metadata import NOISE_PARAMETER_MEANING

ALL_FILTERS = "All six filters"
ONE_IMAGE = "One selected image"
FULL_DATASET = "Full dataset"
ONE_CONDITION = "One noise condition"
ALL_CONDITIONS = "All noise conditions (full experiment)"

# Measured size of one compact comparison picture (BSD68 image, six filters)
PICTURE_MB = 0.2


def keep_valid_choice(key, options):
    """Forgets a saved dropdown choice that is no longer available."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


@st.fragment(run_every="1s")
def watch_running_job(job_id):
    """Checks the job once per second without rerunning the whole page."""
    state = job_runner.job_state(job_id)
    if state["state"] not in jobs.ACTIVE_STATES:
        st.rerun()

    total = state.get("images_total")
    done = state.get("images_done", 0)

    if not total:
        st.info("**Status: Running.** Loading the images...")
    elif done < total:
        st.info(f"**Status: Running.** Processing image {done + 1} of {total} ({done} finished).")
        st.progress(done / total)
    else:
        st.info(f"**Status: Running.** All {total} images processed. Saving the results...")
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

        st.session_state["failed_job"] = {
            "id": job_id,
            "error": state.get("error", "Unknown error."),
            "incomplete_result": state.get("incomplete_result"),
        }

    message = st.session_state.pop("finished_message", None)
    if message:
        st.success(message)

    failed = st.session_state.get("failed_job")
    if failed:
        st.error(f"**Status: Failed.** {failed['error']}")
        with st.expander("Technical details"):
            st.write(f"Log folder: `.frontend_jobs/{failed['id']}`")
            if failed["incomplete_result"]:
                st.write(
                    f"`results/{failed['incomplete_result']}` is incomplete, "
                    "so it will not appear in Results."
                )
            st.code(job_runner.read_log_tail(failed["id"]) or "No log was saved.", language=None)

    return False


# --------------------------------------------------
# FILTER SETTINGS
# --------------------------------------------------

def setting_key(filter_name, name):
    return f"param::{filter_name}::{name}"


def show_value(value):
    return "Automatic" if value is None else str(value)


def load_preset(filter_name, filter_function, noise_type, noise_level):
    """Copies a preset's values into this filter's controls."""
    preset = st.session_state[f"preset::{filter_name}"]

    try:
        values = jobs.preset_parameters(
            preset, filter_name, filter_function, noise_type, noise_level
        )
    except ValueError as error:
        st.session_state[f"preset_error::{filter_name}"] = str(error)
        return

    for name, value in values.items():
        st.session_state[setting_key(filter_name, name)] = value


def show_filter_settings(filter_name, filter_function, noise_type, noise_level, noise_value):
    """Shows one filter's preset and its real settings. Returns the choices."""
    preset_key = f"preset::{filter_name}"
    choices = jobs.PARAMETER_CHOICES[filter_name]

    # First visit: start every control from the Default preset.
    if preset_key not in st.session_state:
        st.session_state[preset_key] = "default"
    if any(setting_key(filter_name, name) not in st.session_state for name in choices):
        load_preset(filter_name, filter_function, noise_type, noise_level)

    preset = st.selectbox(
        "Load preset",
        list(jobs.PARAMETER_SOURCES),
        format_func=lambda source: jobs.PARAMETER_SOURCES[source]["label"],
        key=preset_key,
        on_change=load_preset,
        args=(filter_name, filter_function, noise_type, noise_level),
        persist_state="page",
    )

    error = st.session_state.pop(f"preset_error::{filter_name}", None)
    if error:
        st.error(f"This preset could not be loaded: {error}")

    values = {}
    for column, (name, options) in zip(st.columns(len(choices)), choices.items()):
        values[name] = column.selectbox(
            name,
            options,
            format_func=show_value,
            key=setting_key(filter_name, name),
            persist_state="page",
        )
        column.caption(PARAMETER_HELP[filter_name][name])

    if filter_name == "Lee Filter":
        if noise_type == "Speckle":
            st.caption(
                f"Lee also receives speckle variance {noise_value}, taken from the noise "
                "choice above. It is not a separate setting."
            )
        else:
            st.caption("Lee's speckle model is off, because the noise is not Speckle.")

    label = jobs.PARAMETER_SOURCES[preset]["label"]
    try:
        preset_values = jobs.preset_parameters(
            preset, filter_name, filter_function, noise_type, noise_level
        )
    except ValueError as error:
        st.warning(f"The selected preset is not available: {error}")
    else:
        if values == preset_values:
            st.caption(f"Same as {label} for {noise_level} {noise_type}.")
        else:
            st.caption(f"**Edited:** differs from {label} for {noise_level} {noise_type}.")
            st.button(
                "Reload preset values",
                key=f"reload::{filter_name}",
                on_click=load_preset,
                args=(filter_name, filter_function, noise_type, noise_level),
            )

    return {"preset": preset, "parameters": values}


def show_all_conditions_settings(filter_name, filter_function, noise_levels):
    """Settings for a run over all noise conditions. Returns the choices."""
    preset_key = f"preset::{filter_name}"
    choices = jobs.PARAMETER_CHOICES[filter_name]
    conditions = [(noise, level) for noise, levels in noise_levels.items() for level in levels]

    st.session_state.setdefault(preset_key, "default")
    preset = st.selectbox(
        "Preset",
        list(jobs.PARAMETER_SOURCES),
        format_func=lambda source: jobs.PARAMETER_SOURCES[source]["label"],
        key=preset_key,
        persist_state="page",
    )
    label = jobs.PARAMETER_SOURCES[preset]["label"]

    own_values = st.checkbox(
        f"Use my own values for all {len(conditions)} conditions",
        key=f"own::{filter_name}",
        persist_state="page",
    )

    if not own_values:
        st.caption(f"Uses the {label} values saved for each noise condition. Tuned presets can differ by condition.")
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
            st.dataframe(table, hide_index=True, width="stretch", height=200)
        return {"preset": preset, "parameters": None}

    # First use: start the controls from the filter's default values.
    if any(setting_key(filter_name, name) not in st.session_state for name in choices):
        defaults = jobs.preset_parameters("default", filter_name, filter_function, *conditions[0])
        for name, value in defaults.items():
            st.session_state[setting_key(filter_name, name)] = value

    values = {}
    for column, (name, options) in zip(st.columns(len(choices)), choices.items()):
        values[name] = column.selectbox(
            name, options, format_func=show_value,
            key=setting_key(filter_name, name), persist_state="page",
        )
        column.caption(PARAMETER_HELP[filter_name][name])

    st.caption(f"These exact values are used for all {len(conditions)} conditions.")
    if filter_name == "Lee Filter":
        st.caption("In the Speckle conditions, Lee also receives that condition's speckle variance.")

    return {"preset": preset, "parameters": values}


# --------------------------------------------------
# RUN EXPERIMENT VIEW
# --------------------------------------------------

def show_run_view(is_running):
    # Load the scientific code only when this view is opened.
    import main as backend

    st.header("Run an experiment")
    st.write(
        "Choose one image or a full dataset, a noise condition, and the filter settings. "
        "Results are saved as a new experiment."
    )

    st.subheader("1. Choose image")
    scope = st.radio(
        "Images to process",
        [ONE_IMAGE, FULL_DATASET],
        horizontal=True,
        key="run_scope",
        persist_state="page",
    )

    image_columns = st.columns(2)
    dataset = image_columns[0].selectbox(
        "Dataset", backend.JOB_DATASETS, key="run_dataset", persist_state="page"
    )

    image_paths = backend.find_image_paths(backend.PROJECT_DIR / "dataset" / dataset)
    image_names = [path.name for path in image_paths]

    if scope == ONE_IMAGE:
        keep_valid_choice("run_image", image_names)
        image = image_columns[1].selectbox(
            "Image", image_names, key="run_image", persist_state="page"
        )
    else:
        image = None
        st.info(f"{dataset}: {len(image_names)} images will be processed.")
        st.caption(
            "Each image gets its own noise from the same seed rule as single-image runs. "
            "This can take several minutes; NLM and TV are the slowest filters."
        )

    st.subheader("2. Choose noise")
    noise_scope = ONE_CONDITION
    save_pictures = True

    if scope == FULL_DATASET:
        noise_scope = st.radio(
            "Noise conditions",
            [ONE_CONDITION, ALL_CONDITIONS],
            horizontal=True,
            key="run_noise_scope",
            persist_state="page",
        )

    all_conditions = noise_scope == ALL_CONDITIONS
    condition_count = sum(len(levels) for levels in backend.NOISE_LEVELS.values())

    if all_conditions:
        noise_type = level = None
        levels = {}
        st.info(
            f"All {condition_count} noise conditions: 4 noise types × 3 strengths, "
            "like the full terminal experiment. Every image is tested under each one."
        )
        st.caption(" · ".join(
            f"{noise}: " + ", ".join(f"{name} {value}" for name, value in strengths.items())
            for noise, strengths in backend.NOISE_LEVELS.items()
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
            "Without pictures, Results still shows every score."
        )
    else:
        noise_columns = st.columns(2)
        noise_type = noise_columns[0].selectbox(
            "Noise type", list(backend.NOISE_LEVELS), key="run_noise", persist_state="page"
        )

        levels = backend.NOISE_LEVELS[noise_type]
        level = noise_columns[1].selectbox(
            "Noise strength", list(levels), key="run_level", persist_state="page"
        )
        st.caption(
            f"{NOISE_TEXT[noise_type]} Saved strength: {levels[level]} "
            f"({NOISE_PARAMETER_MEANING[noise_type]}). Noise settings are fixed; filters only clean it."
        )

    st.subheader("3. Choose filters")
    filter_choice = st.selectbox(
        "Filters",
        [ALL_FILTERS, *backend.filters],
        format_func=lambda name: name if name == ALL_FILTERS else helpers.short_name(name),
        key="run_filter",
        persist_state="page",
    )

    chosen_names = list(backend.filters) if filter_choice == ALL_FILTERS else [filter_choice]
    chosen_settings = {}

    def settings_for(name):
        if all_conditions:
            return show_all_conditions_settings(name, backend.filters[name], backend.NOISE_LEVELS)
        return show_filter_settings(name, backend.filters[name], noise_type, level, levels[level])

    for name in chosen_names:
        if filter_choice == ALL_FILTERS:
            with st.expander(helpers.short_name(name)):
                chosen_settings[name] = settings_for(name)
        else:
            st.caption(FILTER_TEXT[helpers.short_name(name)])
            chosen_settings[name] = settings_for(name)

    def describe_settings(chosen):
        if chosen["parameters"] is None:
            return f"{jobs.PARAMETER_SOURCES[chosen['preset']]['label']} for each condition"
        text = helpers.format_settings({"resolved": chosen["parameters"]})
        return f"{text} for every condition" if all_conditions else text

    st.subheader("What will happen")
    filter_lines = "\n".join(
        f"  - {helpers.short_name(name)}: {describe_settings(chosen)}"
        for name, chosen in chosen_settings.items()
    )
    same_image = " Every filter cleans the exact same noisy image." if len(chosen_names) > 1 else ""
    if image is None:
        image_line = f"- **Images:** all {len(image_names)} images from {dataset}, one after another.\n"
    else:
        image_line = f"- **Image:** `{image}` from {dataset}.\n"
    if all_conditions:
        noise_line = f"- **Noise:** all {condition_count} conditions, one after another for each image.\n"
    else:
        noise_line = f"- **Noise:** {level} {noise_type}, strength {levels[level]}.\n"
    st.markdown(
        image_line + noise_line +
        f"- **Filters and settings:**{same_image}\n{filter_lines}\n"
        "- **Result:** PSNR, SSIM, execution time, and images are saved in a new folder."
    )

    if dataset == "Set12":
        st.info("Set12 is used to choose filter settings, so treat these results as practice.")

    if is_running:
        status = "Running"
    elif st.session_state.get("failed_job"):
        status = "Failed"
    else:
        status = "Ready"
    st.write(f"**Status:** {status}")

    if st.button("Run Experiment", type="primary", disabled=is_running, key="run_experiment"):
        job = {
            "type": "single_image" if scope == ONE_IMAGE else "full_dataset",
            "dataset": dataset,
            "image": image,
            "noise_type": noise_type,
            "noise_level": level,
            "filters": chosen_settings,
        }
        if all_conditions:
            job["noise_scope"] = "all"
            job["save_pictures"] = save_pictures

        try:
            st.session_state["watched_job"] = job_runner.start_job(job)
        except job_runner.JobAlreadyRunning:
            st.warning("An experiment is already running. Please wait for it to finish.")
        except ValueError as error:
            st.error(f"These settings cannot be used: {error}")
        except OSError as error:
            st.error(f"The experiment could not start: {error}")
        else:
            st.session_state.pop("failed_job", None)
            st.rerun()
