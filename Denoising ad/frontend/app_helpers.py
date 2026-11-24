"""
Read-only helpers for the Results Viewer.

These functions only read files saved by experiments. They never run
filters, calculate PSNR or SSIM, or write anything.
"""

import json
from pathlib import Path

import pandas as pd

from experiment_job import PREVIOUS_PARAMETERS_SHA256 as OLD_PARAMETERS_SHA256
from experiment_job import PROJECT_DIR
from run_metadata import comparison_file_name

RESULTS_DIR = PROJECT_DIR / "results"
BASELINE_NAME = "Noisy (Baseline)"
DEGRADED_BASELINE_NAME = "Degraded (Baseline)"
BASELINE_NAMES = (BASELINE_NAME, DEGRADED_BASELINE_NAME)

# A display rule, not a scientific test: smaller gaps are labelled "close".
CLOSE_RESULT_DB = 0.2

NOT_MEASURED = "Not measured"

# Names shown in the interface for the saved presets. Only the label changes:
# the preset id, its file and its values are exactly what experiment_job.py uses.
PRESET_LABELS = {
    "default": "Default settings",
    "new_12_image": "Optimized settings (tuned on Set12)",
    "previous_2_image": "Old 2-image tuned settings (not recommended)",
}

# Presets offered when starting a new experiment. The old 2-image preset is
# hidden, but older saved runs that used it still show its name.
VISIBLE_PRESETS = ("default", "new_12_image")


def preset_label(source_id):
    return PRESET_LABELS.get(source_id, source_id)

# Metrics the overview can rank by: saved column, label, number format.
METRICS = {
    "PSNR": ("PSNR", "PSNR (dB)", "{:.2f}"),
    "SSIM": ("SSIM", "SSIM", "{:.3f}"),
    "Edge F1": ("Edge_F1", "Edge F1", "{:.3f}"),
}


def is_baseline(filter_name):
    return filter_name in BASELINE_NAMES


def has_edge_metrics(table):
    """True if this run saved edge scores (older runs did not)."""
    return "Edge_F1" in table.columns or "Edge_F1_mean" in table.columns


def show_number(value, number_format="{:.3f}"):
    """Formats a saved score, or says it was not measured."""
    if value is None or pd.isna(value):
        return NOT_MEASURED
    return number_format.format(value)


def find_saved_runs(results_dir=RESULTS_DIR):
    """Returns completed runs with metadata, newest first."""
    runs = {}

    for folder in Path(results_dir).glob("result*"):
        needed = ("run_config.json", "experiment_results.csv", "summary.csv")

        if not folder.name[6:].isdigit():
            continue
        if not all((folder / name).is_file() for name in needed):
            continue

        try:
            config = json.loads(
                (folder / "run_config.json").read_text(encoding="utf-8")
            )
            if config["run"]["status"] == "completed":
                runs[folder.name] = config
        except (OSError, ValueError, KeyError, TypeError):
            continue

    newest_first = sorted(runs, key=lambda name: int(name[6:]), reverse=True)
    return {name: runs[name] for name in newest_first}


def load_summary(run_name, results_dir=RESULTS_DIR):
    """Reads the averages that the experiment saved in summary.csv."""
    summary = pd.read_csv(
        Path(results_dir) / run_name / "summary.csv", header=[0, 1]
    )
    summary.columns = [
        first if second.startswith("Unnamed") else f"{first}_{second}"
        for first, second in summary.columns
    ]
    return summary


def load_image_rows(run_name, results_dir=RESULTS_DIR):
    """Reads the per-image scores saved in experiment_results.csv."""
    return pd.read_csv(Path(results_dir) / run_name / "experiment_results.csv")


def short_name(filter_name):
    return filter_name.replace(" Filter", "")


def get_condition(summary, noise_type, level, metric="PSNR"):
    """Returns the unfiltered baseline and the filters ranked by a metric's average."""
    rows = summary[
        (summary["Noise_Type"] == noise_type)
        & (summary["Noise_Level"] == level)
    ]
    baseline = rows[rows["Filter"].isin(BASELINE_NAMES)].iloc[0]
    filters = rows[~rows["Filter"].isin(BASELINE_NAMES)]
    filters = filters.sort_values(f"{METRICS[metric][0]}_mean", ascending=False)
    return baseline, filters.reset_index(drop=True)


def filtering_helped(baseline, filters, metric="PSNR"):
    """Did any filter score higher than the unfiltered image? (best average vs baseline)"""
    column = f"{METRICS[metric][0]}_mean"
    return bool(filters[column].max() > baseline[column])


def edge_details(baseline, filters, baseline_label="Noisy (no filter)"):
    """Edge precision, recall and F1 of every filter, with the unfiltered image last."""
    table = []
    for _, row in [*filters.iterrows(), (None, baseline)]:
        table.append({
            "Filter": baseline_label if is_baseline(row["Filter"]) else short_name(row["Filter"]),
            "Edge Precision": show_number(row.get("Edge_Precision_mean")),
            "Edge Recall": show_number(row.get("Edge_Recall_mean")),
            "Edge F1": show_number(row.get("Edge_F1_mean")),
        })
    return pd.DataFrame(table)


def find_close_filters(filters):
    """Returns filters within CLOSE_RESULT_DB of the best filter."""
    best_psnr = filters.iloc[0]["PSNR_mean"]
    close = []

    for _, row in filters.iloc[1:].iterrows():
        gap = best_psnr - row["PSNR_mean"]
        if gap < CLOSE_RESULT_DB:
            close.append((short_name(row["Filter"]), gap))

    return close


def count_best_images(rows, noise_type, level, filter_name, metric="PSNR"):
    """Counts images where this filter has the highest score (ties count)."""
    column = METRICS[metric][0]
    chosen = rows[
        (rows["Noise_Type"] == noise_type)
        & (rows["Noise_Level"] == level)
        & (~rows["Filter"].isin(BASELINE_NAMES))
    ]
    best_for_image = chosen.groupby("Image")[column].transform("max")
    winners = chosen[
        (chosen["Filter"] == filter_name) & (chosen[column] == best_for_image)
    ]
    return winners["Image"].nunique(), chosen["Image"].nunique()


def get_condition_config(config, noise_type, level):
    for condition in config["required_to_rerun"]["conditions"]:
        if condition["noise_type"] == noise_type and condition["level"] == level:
            return condition
    raise KeyError(f"{noise_type} / {level} is not in this run.")


def format_settings(entry):
    """Turns one filter's saved settings into readable text."""
    parts = []

    for name, value in entry["resolved"].items():
        if name == "h" and value is None:
            parts.append("h=automatic")
        elif value is not None:
            parts.append(f"{name}={value}")

    text = ", ".join(parts)

    if entry.get("model") == "multiplicative speckle":
        text += " (speckle model)"
    elif entry.get("model") == "general adaptive smoother":
        text += " (general smoother)"

    return text


def describe_parameter_source(config):
    """Returns plain text about the settings, and whether they are old."""
    source = config["required_to_rerun"]["parameter_source"]

    if source["mode"] == "per_filter":
        return (
            "Chosen separately for each filter. The Details tab shows each "
            "filter's preset and exact values",
            False,
        )

    if source["mode"] != "tuned":
        return "Default settings built into each filter", False

    if source.get("file_sha256") == OLD_PARAMETERS_SHA256:
        return (
            "Old tuned settings from best_parameters.json, chosen using "
            "only 2 Set12 images before the Lee and NLM fixes",
            True,
        )

    if source.get("label"):
        return f"{source['label']} from {source['file']}", False

    return f"Tuned settings from {source['file']}", False


def is_degradation_run(config):
    """True for JPEG / glare experiments. Older runs are all noise runs."""
    return config["required_to_rerun"].get("experiment", "noise") == "degradation"


def unfiltered_words(config):
    """(table column, sentence text, row label) for the image before filtering."""
    if is_degradation_run(config):
        return "Degraded", "degraded image", "Degraded (no filter)"
    return "Noisy", "noisy image", "Noisy (no filter)"


def condition_meaning(config, condition_type):
    """What the saved strength number means, e.g. 'JPEG quality ...'."""
    needed = config["required_to_rerun"]
    block = needed["degradation"] if is_degradation_run(config) else needed["noise"]
    return block["parameter_meaning"][condition_type]


def seed_rule(config):
    """The random seed rule that applies to this run, in plain text."""
    needed = config["required_to_rerun"]
    if is_degradation_run(config):
        glare = needed["degradation"]["seed_policy"]["glare"]
        return (
            f"Glare: `{glare['formula']}` (same spot for every strength). "
            "JPEG compression is not random."
        )
    return f"`{needed['noise']['seed_policy']['formula']}`"


def describe_run(run_name, config):
    dataset = config["required_to_rerun"]["dataset"]
    source = config["required_to_rerun"]["parameter_source"]

    if source["mode"] == "per_filter":
        settings = "settings chosen per filter"
    elif source["mode"] != "tuned":
        settings = "default settings"
    elif source.get("file_sha256") == OLD_PARAMETERS_SHA256:
        settings = "old 2-image tuned settings"
    elif source.get("source_id") == "new_12_image":
        settings = "optimized settings (tuned on Set12)"
    else:
        settings = "tuned settings"

    count = dataset["images_processed"]
    images = dataset["images"][0]["filename"] if count == 1 else f"{count} images"
    conditions = len(config["required_to_rerun"]["conditions"])
    if is_degradation_run(config):
        noise = f"all {conditions} degradation conditions" if conditions == 6 else (
            "1 degradation condition" if conditions == 1 else f"{conditions} degradation conditions"
        )
    else:
        noise = "all 12 noise conditions" if conditions == 12 else (
            "1 noise condition" if conditions == 1 else f"{conditions} noise conditions"
        )
    return f"{run_name}: {dataset['name']}, {images}, {noise}, {settings}"


def describe_preset(record):
    """Plain text for where one filter's values started."""
    if not record:
        return "Unknown"
    label = PRESET_LABELS.get(record.get("preset"), record["preset_label"])
    if record["edited"]:
        return f"{label}, then custom values"
    return label


def comparison_images(config, run_name, noise_type, level, results_dir=RESULTS_DIR):
    """Returns the saved comparison pictures for one condition: {image name: path}.

    Newer full-dataset runs save one picture per image. Older runs saved one
    picture per condition, for the first image only. Missing pictures are skipped.
    """
    folder = Path(results_dir) / run_name / "comparisons"
    manifest = config["required_to_rerun"]["dataset"]["images"]
    pictures = {}

    # Full experiments save small .jpg pictures; other runs save .png
    for entry in manifest:
        for extension in (".png", ".jpg"):
            path = folder / comparison_file_name(
                noise_type, level, entry["index"], entry["filename"], extension
            )
            if path.is_file():
                pictures[entry["filename"]] = path

    old_style = comparison_image_path(run_name, noise_type, level, results_dir=results_dir)
    first_image = manifest[0]["filename"]
    if old_style.is_file() and first_image not in pictures:
        pictures = {first_image: old_style, **pictures}

    return pictures


def image_scores(rows, noise_type, level, image_name, baseline_label="Noisy (no filter)"):
    """One image's saved scores, best PSNR first, with the unfiltered image last."""
    chosen = rows[
        (rows["Noise_Type"] == noise_type)
        & (rows["Noise_Level"] == level)
        & (rows["Image"] == image_name)
    ]
    filters = chosen[~chosen["Filter"].isin(BASELINE_NAMES)].sort_values("PSNR", ascending=False)
    noisy = chosen[chosen["Filter"].isin(BASELINE_NAMES)]
    both = pd.concat([filters, noisy])
    edge_f1 = both["Edge_F1"] if "Edge_F1" in both.columns else [None] * len(both)

    table = pd.DataFrame({
        "Filter": [short_name(name) for name in filters["Filter"]] + [baseline_label] * len(noisy),
        "PSNR (dB)": [f"{value:.2f}" for value in both["PSNR"]],
        "SSIM": [f"{value:.3f}" for value in both["SSIM"]],
    })
    # Older runs did not measure edges; their tables simply leave the column out
    if "Edge_F1" in both.columns:
        table["Edge F1"] = [show_number(value) for value in edge_f1]

    if len(filters) > 1:
        best = filters["PSNR"].max()
        table["Best on this image"] = ["★" if value == best else "" for value in filters["PSNR"]] + [""] * len(noisy)

    return table


def condition_label(noise_type, level):
    return f"{noise_type} / {level}"


def describe_run_type(config):
    """One plain sentence: how many images and noise conditions this run used."""
    needed = config["required_to_rerun"]
    images = needed["dataset"]["images_processed"]
    image_text = "1 image" if images == 1 else f"{images} images"
    conditions = needed["conditions"]
    kind = "degradation" if is_degradation_run(config) else "noise"

    if len(conditions) > 1:
        return (
            f"All {len(conditions)} {kind} conditions on {image_text} from "
            f"{needed['dataset']['name']}"
            + (" (image degradation experiment, separate from noise)." if kind == "degradation"
               else " (full experiment).")
        )

    only = conditions[0]
    return (
        f"One {kind} condition ({only['level']} {only['noise_type']}) on {image_text} "
        f"from {needed['dataset']['name']}."
    )


def condition_presets(config, noise_type, level):
    """Each filter's preset record for one condition, or None for older runs."""
    source = config["required_to_rerun"]["parameter_source"]
    if "filters_by_condition" in source:
        return source["filters_by_condition"].get(condition_label(noise_type, level))
    return source.get("filters")


def condition_overview(summary, rows, conditions, metric="PSNR", kind="Noise", unfiltered="Noisy"):
    """One row per condition: which filter had the best average score for one metric.

    "Filtering helped?" is a separate question: did that best filter score
    higher than the unfiltered image at all?
    """
    column, label, number_format = METRICS[metric]
    unit = " (dB)" if metric == "PSNR" else ""
    table = []

    for condition in conditions:
        noise_type, level = condition["noise_type"], condition["level"]
        baseline, filters = get_condition(summary, noise_type, level, metric)
        best = filters.iloc[0]
        wins, total = count_best_images(rows, noise_type, level, best["Filter"], metric)
        best_score, baseline_score = best[f"{column}_mean"], baseline[f"{column}_mean"]

        row = {
            kind: noise_type,
            "Strength": level,
            "Best filter": short_name(best["Filter"]),
            f"Average {metric}{unit}": number_format.format(best_score),
            f"{unfiltered} {metric}{unit}": number_format.format(baseline_score),
            "Filtering helped?": "Yes" if best_score > baseline_score else "No",
        }

        lead = best_score - filters.iloc[1][f"{column}_mean"] if len(filters) > 1 else None
        row[f"Lead{unit}"] = "-" if lead is None else number_format.format(lead)
        # The "close" rule is defined in dB, so it is used only for PSNR.
        if metric == "PSNR":
            row["Result"] = (
                "Only one filter" if lead is None else
                "Close" if lead < CLOSE_RESULT_DB else "Clear"
            )
        row["Best on"] = f"{wins} of {total} images"
        table.append(row)

    return pd.DataFrame(table)


def metric_by_condition(summary, conditions, metric="PSNR", unfiltered="Noisy"):
    """Average score of every filter in every condition (rows = conditions)."""
    column = METRICS[metric][0]
    rows = {}
    for condition in conditions:
        noise_type, level = condition["noise_type"], condition["level"]
        baseline, filters = get_condition(summary, noise_type, level, metric)
        scores = {unfiltered: baseline[f"{column}_mean"]}
        scores.update({
            short_name(f): p for f, p in zip(filters["Filter"], filters[f"{column}_mean"])
        })
        rows[condition_label(noise_type, level)] = scores

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Condition"

    # Keep the filters in the runner's order, not in score order
    order = [unfiltered] + [short_name(name) for name in summary["Filter"].drop_duplicates()
                            if not is_baseline(name)]
    return table[[column for column in order if column in table.columns]]


def psnr_by_condition(summary, conditions):
    """Average PSNR of every filter in every condition (rows = conditions)."""
    return metric_by_condition(summary, conditions, "PSNR")


def table_height(rows):
    """Height that shows every row of a small table without scrolling."""
    return 38 + 35 * rows


def image_across_conditions(rows, image_name, conditions, unfiltered="Noisy"):
    """One image's PSNR in every condition, with its best filter."""
    table = []

    for condition in conditions:
        noise_type, level = condition["noise_type"], condition["level"]
        chosen = rows[
            (rows["Image"] == image_name)
            & (rows["Noise_Type"] == noise_type)
            & (rows["Noise_Level"] == level)
        ]
        scores = dict(zip(chosen["Filter"], chosen["PSNR"]))
        noisy = None
        for name in BASELINE_NAMES:
            noisy = scores.pop(name, noisy)
        row = {"Condition": condition_label(noise_type, level)}
        row[unfiltered] = f"{noisy:.2f}" if noisy is not None else "-"
        row.update({short_name(name): f"{value:.2f}" for name, value in scores.items()})
        row["Best filter"] = short_name(max(scores, key=scores.get)) if scores else "-"
        table.append(row)

    return pd.DataFrame(table)


def comparison_image_path(run_name, noise_type, level, results_dir=RESULTS_DIR):
    """Finds the comparison picture saved by the experiment."""
    noise = noise_type.lower().replace("-", "_").replace(" ", "_")
    return (
        Path(results_dir) / run_name / "comparisons"
        / f"{noise}_{level.lower()}.png"
    )
