"""
"Explore results": read a saved experiment and explain it, summary first.

This page only reads files saved by experiments (run_config.json,
experiment_results.csv, summary.csv and comparison pictures). It never runs
filters or calculates scores.
"""

import pandas as pd
import streamlit as st

import app_helpers as helpers
import theme
from explanations import (
    CONDITION_TEXT,
    DATASET_NOTES,
    EDGE_CAUTION,
    EDGE_DETAIL_TEXT,
    EDGE_TEXT,
    FILTER_TEXT,
    PSNR_TEXT,
    SSIM_TEXT,
    why_filter_may_work,
)
from run_view import keep_valid_choice


@st.cache_data(show_spinner=False)
def load_tables(run_name):
    return helpers.load_summary(run_name), helpers.load_image_rows(run_name)


def open_result(run_name, noise_type, level):
    """Button callback: show this saved run in Explore results."""
    st.session_state["open_result"] = {"run": run_name, "noise_type": noise_type, "noise_level": level}


class Condition:
    """The scores of one condition in one saved run, ready to display."""

    def __init__(self, config, summary, rows, noise_type, level):
        self.config, self.summary, self.rows = config, summary, rows
        self.noise_type, self.level = noise_type, level
        self.baseline, self.filters = helpers.get_condition(summary, noise_type, level)
        self.best = self.filters.iloc[0]
        self.best_name = helpers.short_name(self.best["Filter"])
        self.wins, self.image_count = helpers.count_best_images(rows, noise_type, level, self.best["Filter"])
        self.is_one_image = self.image_count == 1
        self.is_one_filter = len(self.filters) == 1
        self.helped = helpers.filtering_helped(self.baseline, self.filters)
        self.unfiltered, self.unfiltered_text, self.baseline_label = helpers.unfiltered_words(config)
        self.psnr_label = "PSNR" if self.is_one_image else "Average PSNR"
        self.ssim_label = "SSIM" if self.is_one_image else "Average SSIM"


def show_verdict(result, key="verdict"):
    """Best filter, and separately: did any filter improve the image?"""
    r = result
    with st.container(border=True, key=key):
        if r.is_one_filter:
            st.subheader(f"Filter tested: {r.best_name}")
        elif not r.helped:
            st.subheader(f"Ranked first by {r.psnr_label}, but no improvement: {r.best_name}")
        elif r.is_one_image:
            st.subheader(f"Best filter on this image: {r.best_name}")
        else:
            st.subheader(f"Best filter by average PSNR: {r.best_name}")

        gain = r.best["PSNR_mean"] - r.baseline["PSNR_mean"]
        if r.helped:
            st.success(
                f"**Filtering helped.** {r.best_name} raised PSNR by {gain:.2f} dB compared "
                f"with the unfiltered {r.unfiltered_text}.",
                icon=":material/trending_up:",
            )
        else:
            st.error(f"**No tested filter improved PSNR over the unfiltered {r.unfiltered_text}.**",
                     icon=":material/trending_down:")
            st.caption(
                (f"{r.best_name} lowered PSNR. " if r.is_one_filter else
                 f"{r.best_name} is listed first only because it lost the least PSNR. ")
                + f"Here, leaving the {r.unfiltered_text} unfiltered gave the highest PSNR."
            )

        psnr_column, ssim_column, third_column = st.columns(3)
        psnr_column.metric(
            r.psnr_label, f"{r.best['PSNR_mean']:.2f} dB",
            f"{gain:+.2f} dB vs {r.unfiltered.lower()}",
            help=PSNR_TEXT.replace("**", ""),
        )
        ssim_column.metric(
            r.ssim_label, f"{r.best['SSIM_mean']:.3f}",
            f"{r.best['SSIM_mean'] - r.baseline['SSIM_mean']:+.3f} vs {r.unfiltered.lower()}",
            help=SSIM_TEXT.replace("**", ""),
        )
        if r.is_one_image or r.is_one_filter:
            third_column.metric("Filters compared", len(r.filters))
        else:
            third_column.metric(
                "Best PSNR on (images)", f"{r.wins} of {r.image_count}",
                help="On how many images this filter had the highest PSNR. "
                     "Shows whether it won on most images or only on average.",
            )

        if r.is_one_image:
            first_image = r.config["required_to_rerun"]["dataset"]["images"][0]["filename"]
            st.caption(f"These scores are for one image ({first_image}). They are not averages across a dataset.")
        else:
            dataset_name = r.config["required_to_rerun"]["dataset"]["name"]
            st.caption(f"These scores are averages across {r.image_count} images from {dataset_name}.")

        close_filters = helpers.find_close_filters(r.filters)
        if close_filters:
            close_text = ", ".join(f"{name} ({gap:.2f} dB behind)" for name, gap in close_filters)
            st.warning(
                f"**Close result:** {r.best_name} is first, but very close to: {close_text}. "
                f"Gaps below {helpers.CLOSE_RESULT_DB} dB are too small to call one filter clearly better here."
            )
        elif len(r.filters) > 1:
            second = r.filters.iloc[1]
            st.success(
                f"**Clear result:** {r.best_name} is {r.best['PSNR_mean'] - second['PSNR_mean']:.2f} dB "
                f"ahead of {helpers.short_name(second['Filter'])}."
            )

        best_ssim = r.filters.loc[r.filters["SSIM_mean"].idxmax(), "Filter"]
        if best_ssim != r.best["Filter"]:
            st.info(f"PSNR and SSIM disagree: {helpers.short_name(best_ssim)} has the highest {r.ssim_label}.")


def score_table(r):
    """PSNR, SSIM, change versus the unfiltered image, and time, best PSNR first."""
    table = pd.DataFrame({"Filter": [helpers.short_name(name) for name in r.filters["Filter"]]})
    table[f"{r.psnr_label} (dB)"] = r.filters["PSNR_mean"].map("{:.2f}".format).values
    table[r.ssim_label] = r.filters["SSIM_mean"].map("{:.3f}".format).values
    table[f"PSNR change vs {r.unfiltered.lower()} (dB)"] = (
        r.filters["PSNR_mean"] - r.baseline["PSNR_mean"]).map("{:+.2f}".format).values
    time_label = "Time (ms)" if r.is_one_image else "Average time (ms)"
    table[time_label] = (r.filters["Execution_Time_mean"] * 1000).map("{:.1f}".format).values
    if not r.is_one_image and not r.is_one_filter:
        table["Best PSNR on"] = [
            "{} of {}".format(*helpers.count_best_images(r.rows, r.noise_type, r.level, name))
            for name in r.filters["Filter"]
        ]
    return table


def show_compare_tab(r):
    if not r.is_one_filter:
        chart_data = r.filters.assign(Filter=r.filters["Filter"].map(helpers.short_name))
        psnr_chart, ssim_chart = st.columns(2)
        with psnr_chart:
            st.markdown(f"**{r.psnr_label}** (dB, higher is better)")
            st.bar_chart(chart_data, x="Filter", y="PSNR_mean", horizontal=True, sort="-PSNR_mean",
                         x_label="", y_label="", height=260)
        with ssim_chart:
            st.markdown(f"**{r.ssim_label}** (0 to 1, higher is better)")
            st.bar_chart(chart_data, x="Filter", y="SSIM_mean", horizontal=True, sort="-SSIM_mean",
                         x_label="", y_label="", height=260)

    before = "this" if r.is_one_image else "the"
    verb = "scored" if r.is_one_image else "averaged"
    plural = "" if r.is_one_image else "s"
    st.caption(
        f"Before filtering, {before} {r.unfiltered_text}{plural} {verb} {r.baseline['PSNR_mean']:.2f} dB "
        f"PSNR and {r.baseline['SSIM_mean']:.3f} SSIM."
    )

    st.dataframe(score_table(r), hide_index=True, width="stretch")
    st.caption(
        "Time is only the filter call, measured once per image on the computer that ran it. "
        "Small time differences are not meaningful."
        + ("" if r.is_one_image or r.is_one_filter else
           " 'Best PSNR on' counts images where that filter scored highest; tied filters all count.")
    )


def show_conditions_tab(r, conditions, kind):
    summary, rows, config = r.summary, r.rows, r.config
    has_edges = helpers.has_edge_metrics(summary)
    metric_options = list(helpers.METRICS) if has_edges else ["PSNR", "SSIM"]
    keep_valid_choice("overview_metric", metric_options)
    metric = st.radio(
        "Rank the filters by",
        metric_options,
        horizontal=True,
        key="overview_metric",
        help="PSNR is the default ranking. SSIM and Edge F1 can choose a different winner.",
    )
    if not has_edges:
        st.caption("Edge F1 was not measured in this older experiment.")

    metric_name = helpers.METRICS[metric][1]
    st.write(
        f"Each row is one condition and the filter with the highest average **{metric}** there. "
        f"**Winner decided by: {metric}.** 'Filtering helped?' compares that filter with the "
        f"unfiltered {r.unfiltered_text}."
    )

    overview = helpers.condition_overview(
        summary, rows, conditions, metric,
        kind="Degradation" if kind == "degradation" else "Noise", unfiltered=r.unfiltered,
    )
    st.dataframe(overview, hide_index=True, width="stretch", height=helpers.table_height(len(overview)))

    best_counts = overview["Best filter"].value_counts()
    st.caption(
        f"Best {metric} in: "
        + ", ".join(f"{name} {count} of {len(overview)}" for name, count in best_counts.items())
        + (f". 'Close' means the lead is below {helpers.CLOSE_RESULT_DB} dB." if metric == "PSNR" else ".")
    )
    if (overview["Filtering helped?"] == "No").any():
        st.warning(
            f"In {(overview['Filtering helped?'] == 'No').sum()} of {len(overview)} conditions, "
            f"no tested filter improved {metric} over the unfiltered {r.unfiltered_text}."
        )
    if metric == "Edge F1":
        st.caption(EDGE_CAUTION)

    matrix = helpers.metric_by_condition(summary, conditions, metric, r.unfiltered)
    filter_columns = [column for column in matrix.columns if column != r.unfiltered]
    st.markdown(f"**Average {metric_name} of every filter in every condition.** The highest in each row is highlighted.")
    st.dataframe(
        matrix.style.format(helpers.METRICS[metric][2]).highlight_max(
            axis=1, subset=filter_columns,
            props="font-weight: bold; background-color: rgba(46, 160, 67, 0.35)",
        ),
        width="stretch",
        height=helpers.table_height(len(matrix)),
    )
    st.caption(f"Choose a {kind} type and strength at the top of the page to see one condition in detail.")


def show_images_tab(r, run_name, conditions):
    config, is_degradation = r.config, helpers.is_degradation_run(r.config)
    picture_noise, picture_level = r.noise_type, r.level
    image_column = condition_column = st

    # Runs with several conditions: choose the image and the condition here.
    if len(conditions) > 1:
        image_column, condition_column = st.columns(2)
        labels = [helpers.condition_label(c["noise_type"], c["level"]) for c in conditions]
        keep_valid_choice("compare_condition", labels)
        st.session_state.setdefault("compare_condition", helpers.condition_label(r.noise_type, r.level))
        picture_label = condition_column.selectbox(
            "Degradation condition" if is_degradation else "Noise condition",
            labels, key="compare_condition",
        )
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
        image_column.markdown(f"**Image:** {chosen_image}")

    if not pictures:
        st.caption("No comparison pictures were saved for this condition, so only the scores are shown.")
    elif len(pictures) == 1 and not r.is_one_image:
        st.caption(
            f"Only {chosen_image} has a saved picture in this experiment. "
            "Newer full-dataset runs save a picture for every image."
        )

    if not r.is_one_image and len(conditions) > 1:
        st.info(
            f"The picture and tables below are for {chosen_image} with {picture_level} "
            f"{picture_noise}{'' if is_degradation else ' noise'}. "
            f"The summary above shows averages across all {r.image_count} images."
        )
    elif not r.is_one_image:
        st.info(
            f"The picture and table below are for {chosen_image} only. "
            f"The summary above shows the averages across all {r.image_count} images."
        )

    if chosen_image in pictures:
        st.image(
            str(pictures[chosen_image]),
            caption=f"Saved comparison for {chosen_image} ({picture_level} {picture_noise}). "
                    "Each panel shows its PSNR and SSIM.",
            width="stretch",
        )

    st.markdown(f"**Scores for {chosen_image}**")
    st.dataframe(
        helpers.image_scores(r.rows, picture_noise, picture_level, chosen_image, r.baseline_label),
        hide_index=True,
        width="stretch",
    )

    if len(conditions) > 1:
        st.markdown(f"**{chosen_image} in every condition** (PSNR in dB, higher is better)")
        st.dataframe(
            helpers.image_across_conditions(r.rows, chosen_image, conditions, r.unfiltered),
            hide_index=True,
            width="stretch",
            height=helpers.table_height(len(conditions)),
        )


def show_details_tab(r, condition):
    config = r.config

    st.markdown("##### Exact filter settings")
    table = pd.DataFrame({
        "Filter": [helpers.short_name(name) for name in r.filters["Filter"]],
        "Settings used": [helpers.format_settings(condition["filters"][name]) for name in r.filters["Filter"]],
    })
    presets = helpers.condition_presets(config, r.noise_type, r.level)
    if presets:
        table["Preset"] = [helpers.describe_preset(presets.get(name)) for name in r.filters["Filter"]]
    st.dataframe(table, hide_index=True, width="stretch")

    st.markdown("##### About this filter" if r.is_one_filter else "##### Why this filter may work")
    st.write(f"**How {r.best_name} works:** {FILTER_TEXT[r.best_name]}")
    if r.best["PSNR_mean"] > r.baseline["PSNR_mean"]:
        st.write(f"**Likely reason here:** {why_filter_may_work(r.noise_type, r.best_name)}")
    else:
        st.info("This filter did not raise PSNR here, so no reason for success is suggested.")
    if condition["filters"].get(r.best["Filter"], {}).get("model") == "multiplicative speckle":
        st.caption(
            "In this experiment, Lee was given the true speckle strength. "
            "The other filters did not receive that extra information."
        )
    st.caption("The scores are measured. This explanation is the likely reason, not proof.")

    st.markdown("##### Edge preservation (advanced)")
    if helpers.has_edge_metrics(r.summary):
        st.markdown(EDGE_TEXT)
        st.markdown(EDGE_DETAIL_TEXT)
        st.dataframe(helpers.edge_details(r.baseline, r.filters, r.baseline_label),
                     hide_index=True, width="stretch")
        st.warning(EDGE_CAUTION)
    else:
        st.caption("Edge scores were not measured in this older experiment.")
        st.dataframe(helpers.edge_details(r.baseline, r.filters, r.baseline_label),
                     hide_index=True, width="stretch")

    st.markdown("##### How this experiment was made")
    needed = config["required_to_rerun"]
    environment = config["useful_for_interpretation"]["environment"]
    git = needed["code"]["git"]
    image_word = "image" if needed["dataset"]["images_processed"] == 1 else "images"
    st.write(f"**Dataset:** {needed['dataset']['name']} ({needed['dataset']['images_processed']} {image_word})")
    st.write(f"**Started:** {config['run']['started_at']}")
    st.write(f"**Random seed rule:** {helpers.seed_rule(config)}")
    if helpers.is_degradation_run(config):
        st.write(
            "**Degradation settings:** "
            + "; ".join(
                f"{name}: " + ", ".join(f"{lvl} {value}" for lvl, value in strengths.items())
                + f" ({needed['degradation']['parameter_meaning'][name]})"
                for name, strengths in needed["degradation"]["levels"].items()
            )
        )
    edge_settings = config["useful_for_interpretation"]["evaluation"].get("edge_preservation")
    if edge_settings:
        st.write(
            f"**Edge scores:** Canny thresholds {edge_settings['canny_low_threshold']} and "
            f"{edge_settings['canny_high_threshold']}, matched within "
            f"{edge_settings['match_tolerance_pixels']} pixel of the clean image's edges."
        )
    else:
        st.write("**Edge scores:** not measured in this experiment.")
    st.write(f"**Code version:** `{git.get('commit', 'unknown')[:7]}` ({git.get('working_tree', 'unknown')})")
    st.write(f"**Note:** {git.get('note', 'No Git note was saved.')}")
    st.write(f"**Python:** {environment['python']}")
    st.write(
        "**Libraries:** "
        + ", ".join(f"{name} {version}" for name, version in environment["libraries"].items())
    )
    st.caption(f"Everything above is read from results/{config['run']['result_directory']}/run_config.json.")


def show_results_view():
    st.header("Explore results")
    runs = helpers.find_saved_runs()

    if not runs:
        st.info("No saved experiments yet. Results appear here after you run a test or an experiment.",
                icon=":material/folder_open:")
        st.button("Try an image", on_click=lambda: st.session_state.update(view="try"), key="empty_try")
        return

    # ---- Choose what to look at
    with st.container(border=True):
        run_column, noise_column, level_column = st.columns([2.2, 1, 1])
        keep_valid_choice("saved_run", list(runs))
        run_name = run_column.selectbox(
            "Saved experiment",
            list(runs),
            format_func=lambda name: helpers.describe_run(name, runs[name]),
            key="saved_run",
            help="Newest first. Only runs that finished and saved their settings are listed.",
        )
        config = runs[run_name]
        conditions = config["required_to_rerun"]["conditions"]
        is_degradation = helpers.is_degradation_run(config)
        kind = "degradation" if is_degradation else "noise"

        noise_types = list(dict.fromkeys(c["noise_type"] for c in conditions))
        keep_valid_choice("noise_type", noise_types)
        noise_type = noise_column.selectbox(
            "Degradation type" if is_degradation else "Noise type", noise_types, key="noise_type"
        )

        levels = [c["level"] for c in conditions if c["noise_type"] == noise_type]
        keep_valid_choice("noise_level", levels)
        level = level_column.selectbox(
            "Damage strength" if is_degradation else "Noise strength", levels, key="noise_level"
        )

        condition = helpers.get_condition_config(config, noise_type, level)
        meaning = helpers.condition_meaning(config, noise_type)
        st.caption(f"{CONDITION_TEXT[noise_type]} Saved strength: {condition['noise_value']} ({meaning}).")
        st.markdown(f"**Run type:** {helpers.describe_run_type(config)}")

    summary, image_rows = load_tables(run_name)
    result = Condition(config, summary, image_rows, noise_type, level)
    needed = config["required_to_rerun"]

    # ---- Summary facts
    facts = st.columns(5)
    theme.fact(facts[0], "Dataset", needed["dataset"]["name"])
    theme.fact(facts[1], "Images", needed["dataset"]["images_processed"])
    theme.fact(facts[2], "Degradation" if is_degradation else "Noise",
               f"{level} {noise_type} ({condition['noise_value']})")
    theme.fact(facts[3], "Filters compared", len(result.filters))
    theme.fact(facts[4], "Winner decided by", "PSNR (higher is better)")

    if is_degradation:
        st.info(
            "**Image degradation experiment.** JPEG compression and glare are not noise, and the "
            "filters were made for noise. Every filter is compared with the unfiltered degraded "
            "image, and filtering may make the image worse."
        )

    source_text, uses_old_settings = helpers.describe_parameter_source(config)
    if uses_old_settings:
        st.warning(
            f"**Settings used:** {source_text}. Use this run for learning, "
            "but do not treat it as the final project result."
        )
    else:
        st.caption(f"**Settings used:** {source_text}.")

    if needed["dataset"]["name"] == "original_png":
        st.warning(DATASET_NOTES["original_png"][1])

    # ---- The verdict for the chosen condition
    show_verdict(result)

    # ---- Details in tabs
    tab_names = [":material/bar_chart: Compare filters"]
    if len(conditions) > 1:
        tab_names.append(f":material/grid_view: All {len(conditions)} conditions")
    tab_names += [":material/image: Images", ":material/tune: Details and settings"]
    tabs = dict(zip(["compare", *(["conditions"] if len(conditions) > 1 else []), "images", "details"],
                    st.tabs(tab_names)))

    with tabs["compare"]:
        show_compare_tab(result)
    if "conditions" in tabs:
        with tabs["conditions"]:
            show_conditions_tab(result, conditions, kind)
    with tabs["images"]:
        show_images_tab(result, run_name, conditions)
    with tabs["details"]:
        show_details_tab(result, condition)


def show_result_brief(run_name):
    """A short view of one saved run, for the Try an image page."""
    runs = helpers.find_saved_runs()
    if run_name not in runs:
        st.warning(f"{run_name} is not available any more.")
        return
    config = runs[run_name]
    first = config["required_to_rerun"]["conditions"][0]
    summary, image_rows = load_tables(run_name)
    result = Condition(config, summary, image_rows, first["noise_type"], first["level"])

    show_verdict(result, key="try_verdict")
    st.dataframe(score_table(result), hide_index=True, width="stretch")
    st.button(
        "Open in Explore results",
        key="try_open_result",
        on_click=open_result,
        args=(run_name, first["noise_type"], first["level"]),
        icon=":material/open_in_new:",
    )
    st.caption(f"Saved as {run_name}. Explore results also shows edge scores, exact settings and run details.")


def show_result_picture(run_name):
    """The saved comparison picture of a one-image run, at full width."""
    runs = helpers.find_saved_runs()
    if run_name not in runs:
        return
    config = runs[run_name]
    first = config["required_to_rerun"]["conditions"][0]
    pictures = helpers.comparison_images(config, run_name, first["noise_type"], first["level"])
    if pictures:
        name, path = next(iter(pictures.items()))
        st.markdown("#### Compare the images")
        st.image(str(path), width="stretch",
                 caption=f"{name} with {first['level']} {first['noise_type']}. Each panel shows its "
                         "PSNR and SSIM; the noisy or damaged image is the second panel.")
