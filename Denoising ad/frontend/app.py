"""
Image Noise Lab: a simple page for running and understanding experiments.

Launch from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -m streamlit run frontend\\app.py

The Results view only reads saved files. The Run Experiment view starts
main.py in a separate process, so this page contains no image processing.
"""

import sys
from pathlib import Path

# Do not create Python cache files beside the project code.
sys.dont_write_bytecode = True

# The scientific code (main.py, run_metadata.py, ...) is one folder up.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st

import app_helpers as helpers
from explanations import (
    FILTER_TEXT,
    NOISE_TEXT,
    PSNR_TEXT,
    SSIM_TEXT,
    why_filter_may_work,
)
from run_view import keep_valid_choice, show_job_status, show_run_view


@st.cache_data(show_spinner=False)
def load_tables(run_name):
    return helpers.load_summary(run_name), helpers.load_image_rows(run_name)


st.set_page_config(page_title="Image Noise Lab", layout="wide")

# Open a finished experiment before the dropdowns are created.
if "open_result" in st.session_state:
    finished = st.session_state.pop("open_result")
    st.session_state["view"] = "Results"
    st.session_state["saved_run"] = finished["run"]
    st.session_state["noise_type"] = finished["noise_type"]
    st.session_state["noise_level"] = finished["noise_level"]

st.title("Image Noise Lab")
view = st.radio(
    "View",
    ["Results", "Run Experiment"],
    horizontal=True,
    key="view",
    label_visibility="collapsed",
)
is_running = show_job_status()

if view == "Run Experiment":
    show_run_view(is_running)
    st.stop()

st.write("See which filter cleaned the noisy images best in a saved experiment.")

runs = helpers.find_saved_runs()

if not runs:
    st.warning("No completed experiment with a run_config.json file was found.")
    st.stop()


# --------------------------------------------------
# 1. CHOOSE A RESULT
# --------------------------------------------------

st.header("1. Choose a result")
run_column, noise_column, level_column = st.columns(3)

keep_valid_choice("saved_run", list(runs))
run_name = run_column.selectbox(
    "Saved experiment",
    list(runs),
    format_func=lambda name: helpers.describe_run(name, runs[name]),
    key="saved_run",
)
config = runs[run_name]
conditions = config["required_to_rerun"]["conditions"]

noise_types = list(dict.fromkeys(c["noise_type"] for c in conditions))
keep_valid_choice("noise_type", noise_types)
noise_type = noise_column.selectbox("Noise type", noise_types, key="noise_type")

levels = [c["level"] for c in conditions if c["noise_type"] == noise_type]
keep_valid_choice("noise_level", levels)
level = level_column.selectbox("Noise strength", levels, key="noise_level")

condition = helpers.get_condition_config(config, noise_type, level)
meaning = config["required_to_rerun"]["noise"]["parameter_meaning"][noise_type]
st.caption(
    f"{NOISE_TEXT[noise_type]} Saved strength: {condition['noise_value']} ({meaning})."
)

source_text, uses_old_settings = helpers.describe_parameter_source(config)
if uses_old_settings:
    st.warning(
        f"**Settings used:** {source_text}. Use this run for learning, "
        "but do not treat it as the final project result."
    )
else:
    st.info(f"**Settings used:** {source_text}.")

st.markdown(f"**Run type:** {helpers.describe_run_type(config)}")


# --------------------------------------------------
# 2. THE RESULT
# --------------------------------------------------

summary, image_rows = load_tables(run_name)
baseline, filters = helpers.get_condition(summary, noise_type, level)
best = filters.iloc[0]
best_name = helpers.short_name(best["Filter"])
wins, image_count = helpers.count_best_images(
    image_rows, noise_type, level, best["Filter"]
)
is_one_image = image_count == 1
is_one_filter = len(filters) == 1
first_image = config["required_to_rerun"]["dataset"]["images"][0]["filename"]
psnr_label = "PSNR" if is_one_image else "Average PSNR"
ssim_label = "SSIM" if is_one_image else "Average SSIM"


# --------------------------------------------------
# OVERVIEW (only for runs with several noise conditions)
# --------------------------------------------------

if len(conditions) > 1:
    st.header("Overview: all noise conditions")
    st.write("Each row is one noise condition and the filter with the highest average PSNR there.")

    overview = helpers.condition_overview(summary, image_rows, conditions)
    st.dataframe(overview, hide_index=True, width="stretch", height=helpers.table_height(len(overview)))

    best_counts = overview["Best filter"].value_counts()
    st.caption(
        "Best in: "
        + ", ".join(f"{name} {count} of {len(overview)}" for name, count in best_counts.items())
        + f". 'Close' means the lead is below {helpers.CLOSE_RESULT_DB} dB."
    )

    matrix = helpers.psnr_by_condition(summary, conditions)
    filter_columns = [column for column in matrix.columns if column != "Noisy"]
    st.markdown("**Average PSNR (dB) of every filter in every condition.** The highest in each row is highlighted.")
    st.dataframe(
        matrix.style.format("{:.2f}").highlight_max(
            axis=1, subset=filter_columns,
            props="font-weight: bold; background-color: rgba(46, 160, 67, 0.35)",
        ),
        width="stretch",
        height=helpers.table_height(len(matrix)),
    )
    st.caption("Choose a noise type and strength in section 1 to see one condition in detail below.")

st.header("2. The result")

if is_one_filter:
    st.subheader(f"Filter tested: {best_name}")
elif is_one_image:
    st.subheader(f"Best filter on this image: {best_name}")
else:
    st.subheader(f"Best filter by average PSNR: {best_name}")

if is_one_image:
    st.info(
        f"These scores are for **one image** ({first_image}). "
        "They are not averages across a dataset."
    )
else:
    dataset_name = config["required_to_rerun"]["dataset"]["name"]
    st.info(f"These scores are averages across {image_count} images from {dataset_name}.")

psnr_column, ssim_column, wins_column = st.columns(3)
psnr_column.metric(
    psnr_label,
    f"{best['PSNR_mean']:.2f} dB",
    f"{best['PSNR_mean'] - baseline['PSNR_mean']:+.2f} dB vs noisy image",
)
ssim_column.metric(
    ssim_label,
    f"{best['SSIM_mean']:.3f}",
    f"{best['SSIM_mean'] - baseline['SSIM_mean']:+.3f} vs noisy image",
)

if is_one_image or is_one_filter:
    wins_column.metric("Filters tested", len(filters))
else:
    wins_column.metric("Best PSNR on", f"{wins} of {image_count} images")
    wins_column.caption("This shows whether the filter won on most images or only on average.")

psnr_column.caption(PSNR_TEXT)
ssim_column.caption(SSIM_TEXT)

close_filters = helpers.find_close_filters(filters)

if close_filters:
    close_text = ", ".join(f"{name} ({gap:.2f} dB behind)" for name, gap in close_filters)
    st.warning(
        f"**Close result:** {best_name} is first, but very close to: {close_text}. "
        f"Gaps below {helpers.CLOSE_RESULT_DB} dB are too small to call one filter clearly better here."
    )
elif len(filters) > 1:
    second = filters.iloc[1]
    gap = best["PSNR_mean"] - second["PSNR_mean"]
    st.success(
        f"**Clear result:** {best_name} is {gap:.2f} dB ahead of "
        f"{helpers.short_name(second['Filter'])}."
    )

best_ssim = filters.loc[filters["SSIM_mean"].idxmax(), "Filter"]
if best_ssim != best["Filter"]:
    st.info(
        f"PSNR and SSIM disagree: {helpers.short_name(best_ssim)} has the highest {ssim_label}."
    )


# --------------------------------------------------
# 3. COMPARE THE FILTERS
# --------------------------------------------------

st.header("3. Filter score" if is_one_filter else "3. Compare the filters")
chart_data = filters.assign(Filter=filters["Filter"].map(helpers.short_name))

if not is_one_filter:
    st.bar_chart(
        chart_data,
        x="Filter",
        y="PSNR_mean",
        horizontal=True,
        sort=False,
        x_label="",
        y_label=f"{psnr_label} (dB), higher is better",
        height=300,
    )

if is_one_image:
    st.caption(f"Before filtering, this noisy image scored {baseline['PSNR_mean']:.2f} dB.")
else:
    st.caption(f"Before filtering, the noisy images averaged {baseline['PSNR_mean']:.2f} dB.")

table = chart_data[["Filter"]].copy()
table[f"{psnr_label} (dB)"] = filters["PSNR_mean"].map("{:.2f}".format)
table[ssim_label] = filters["SSIM_mean"].map("{:.3f}".format)

time_label = "Time (ms)" if is_one_image else "Average time (ms)"
table[time_label] = (filters["Execution_Time_mean"] * 1000).map("{:.1f}".format)

if not is_one_image and not is_one_filter:
    table["Best PSNR on"] = [
        "{} of {}".format(*helpers.count_best_images(image_rows, noise_type, level, name))
        for name in filters["Filter"]
    ]

table["Settings used"] = [
    helpers.format_settings(condition["filters"][name]) for name in filters["Filter"]
]

presets = helpers.condition_presets(config, noise_type, level)
if presets:
    table["Preset"] = [helpers.describe_preset(presets.get(name)) for name in filters["Filter"]]

st.dataframe(table, hide_index=True, width="stretch")
st.caption(
    "Time is only the filter call, measured once per image on the computer that ran it. "
    "Small time differences are not meaningful."
    + ("" if is_one_image or is_one_filter else
       " 'Best PSNR on' counts images where that filter scored highest; tied filters all count.")
)


# --------------------------------------------------
# 4. WHY THIS FILTER MAY WORK
# --------------------------------------------------

st.header("4. About this filter" if is_one_filter else "4. Why this filter may work")
st.write(f"**How {best_name} works:** {FILTER_TEXT[best_name]}")

if best["PSNR_mean"] > baseline["PSNR_mean"]:
    st.write(f"**Likely reason here:** {why_filter_may_work(noise_type, best_name)}")
else:
    st.info("This filter did not raise PSNR here, so no reason for success is suggested.")

if condition["filters"].get(best["Filter"], {}).get("model") == "multiplicative speckle":
    st.caption(
        "In this experiment, Lee was given the true speckle strength. "
        "The other filters did not receive that extra information."
    )
st.caption("The scores above are measured. This explanation is the likely reason, not proof.")


# --------------------------------------------------
# 5. SEE THE IMAGES
# --------------------------------------------------

st.header("5. See the images")
picture_noise, picture_level = noise_type, level
image_column = condition_column = st

# Runs with several noise conditions: choose the image and the condition here.
if len(conditions) > 1:
    image_column, condition_column = st.columns(2)
    labels = [helpers.condition_label(c["noise_type"], c["level"]) for c in conditions]
    keep_valid_choice("compare_condition", labels)
    st.session_state.setdefault("compare_condition", helpers.condition_label(noise_type, level))
    picture_label = condition_column.selectbox("Noise condition", labels, key="compare_condition")
    picture_noise, picture_level = picture_label.split(" / ")

pictures = helpers.comparison_images(config, run_name, picture_noise, picture_level)

# Without saved pictures, the per-image scores can still be shown.
manifest = config["required_to_rerun"]["dataset"]["images"]
image_names = list(pictures) or [entry["filename"] for entry in manifest]

if len(image_names) > 1:
    keep_valid_choice("compare_image", image_names)
    chosen_image = image_column.selectbox("Choose image", image_names, key="compare_image")
else:
    chosen_image = image_names[0]

if not pictures:
    st.caption("No comparison pictures were saved for this condition, so only the scores are shown.")
elif len(pictures) == 1 and not is_one_image:
    st.caption(
        f"Only {chosen_image} has a saved picture in this experiment. "
        "Newer full-dataset runs save a picture for every image."
    )

if not is_one_image and len(conditions) > 1:
    st.info(
        f"The picture and tables below are for {chosen_image} with {picture_level} "
        f"{picture_noise} noise. Sections 2 to 4 show averages across all {image_count} images."
    )
elif not is_one_image:
    st.info(
        f"The picture and table below are for {chosen_image} only. "
        f"Section 2 shows the averages across all {image_count} images."
    )

if chosen_image in pictures:
    st.image(
        str(pictures[chosen_image]),
        caption=f"Saved comparison for {chosen_image} ({picture_level} {picture_noise}).",
    )

st.dataframe(
    helpers.image_scores(image_rows, picture_noise, picture_level, chosen_image),
    hide_index=True,
    width="stretch",
)

if len(conditions) > 1:
    st.markdown(f"**{chosen_image} in every noise condition** (PSNR in dB, higher is better)")
    st.dataframe(
        helpers.image_across_conditions(image_rows, chosen_image, conditions),
        hide_index=True,
        width="stretch",
        height=helpers.table_height(len(conditions)),
    )


# --------------------------------------------------
# RUN DETAILS
# --------------------------------------------------

with st.expander("How this experiment was made"):
    needed = config["required_to_rerun"]
    environment = config["useful_for_interpretation"]["environment"]
    git = needed["code"]["git"]
    image_word = "image" if needed["dataset"]["images_processed"] == 1 else "images"
    st.write(f"**Dataset:** {needed['dataset']['name']} ({needed['dataset']['images_processed']} {image_word})")
    st.write(f"**Started:** {config['run']['started_at']}")
    st.write(f"**Random seed rule:** `{needed['noise']['seed_policy']['formula']}`")
    st.write(f"**Code version:** `{git.get('commit', 'unknown')[:7]}` ({git.get('working_tree', 'unknown')})")
    st.write(f"**Note:** {git.get('note', 'No Git note was saved.')}")
    st.write(f"**Python:** {environment['python']}")
    st.write(
        "**Libraries:** "
        + ", ".join(f"{name} {version}" for name, version in environment["libraries"].items())
    )
