
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import time
import json
import inspect
import traceback
from pathlib import Path

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from add_noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    add_poisson_noise
)

from filter import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_bilateral_filter,
    apply_nlm_filter,
    apply_lee_filter,
    apply_tv_filter
)

from edge_metrics import calculate_edge_metrics

from image_degradation import (
    add_jpeg_compression,
    add_glare
)

from experiment_job import (
    PARAMETER_SOURCES,
    describe_preset_use,
    mark_job,
    parameter_file,
    preset_parameters,
    read_job,
    validate_filter_parameters,
    validate_job_filters
)

from run_metadata import (
    PROJECT_DIR,
    build_run_config,
    comparison_file_name,
    mark_run_completed,
    save_run_config
)


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

DATASET_PATH = Path("dataset/BSD68")
RESULTS_PATH = Path("results")

BASE_SEED = 42
MAX_IMAGES = None# Set to None to process the full dataset

TUNED_PARAMETERS_FILE = Path("best_parameters.json")

# Used only by frontend jobs (python main.py --job job.json).
# Urban100_HR and original_png come from the teammate branch
# feature/image_degradation-different_datasets-edge_preservation.
# Parameter tuning still uses only Set12 (see tune_filters.py).
JOB_DATASETS = ("BSD68", "Set12", "Urban100_HR", "original_png")
JOB_RESULTS_PATH = PROJECT_DIR / "results"

NOISE_LEVELS = {
    "Gaussian": {
        "Low": 15,
        "Medium": 25,
        "High": 50
    },

    "Salt-and-Pepper": {
        "Low": 0.05,
        "Medium": 0.10,
        "High": 0.20
    },

    "Speckle": {
        "Low": 0.01,
        "Medium": 0.04,
        "High": 0.10
    },

    "Poisson": {
        "Low": 100,
        "Medium": 30,
        "High": 10
    }
}

# Image damage that is not noise (from the teammate branch). Kept apart from
# NOISE_LEVELS, so "all noise conditions" still means the 12 noise conditions.
# Only frontend jobs with "experiment": "degradation" use these.
DEGRADATION_LEVELS = {
    "JPEG Compression": {   # JPEG quality: lower means stronger damage
        "Low": 90,
        "Medium": 60,
        "High": 30
    },

    "Glare": {              # glare strength alpha: higher means stronger
        "Low": 0.20,
        "Medium": 0.45,
        "High": 0.70
    }
}

# Glare uses slot 5 of the seed rule (noise types use slots 0-3, JPEG is 4
# but needs no seed). The level is not part of the glare seed.
GLARE_SEED_OFFSET = 500

NOISY_BASELINE = "Noisy (Baseline)"
DEGRADED_BASELINE = "Degraded (Baseline)"


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def find_image_paths(dataset_path):

    return sorted(
        list(dataset_path.rglob("*.png")) +
        list(dataset_path.rglob("*.jpg")) +
        list(dataset_path.rglob("*.bmp"))
    )


def load_dataset():
    images = []

    image_paths = find_image_paths(DATASET_PATH)

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is not None:
            image = image.astype(np.float32) / 255.0
            images.append((image_path.name, image))

    if len(images) == 0:
        raise ValueError(
            f"No images found in {DATASET_PATH}"
        )

    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    print(f"Loaded {len(images)} images.")

    return images


# --------------------------------------------------
# 3. NOISE GENERATION
# --------------------------------------------------

def generate_noisy_image(image, noise_type, noise_level, seed):

    if noise_type == "Gaussian":
        return add_gaussian_noise(image, noise_level, seed)

    elif noise_type == "Salt-and-Pepper":
        return add_salt_pepper_noise(image, noise_level, seed)

    elif noise_type == "Speckle":
        return add_speckle_noise(image, noise_level, seed)

    elif noise_type == "Poisson":
        return add_poisson_noise(
         image, noise_level, seed )


def generate_degraded_image(image, degradation_type, value, seed):
    """JPEG compression (value = quality) or glare (value = strength)."""

    if degradation_type == "JPEG Compression":
        return add_jpeg_compression(image, value)

    elif degradation_type == "Glare":
        return add_glare(image, value, seed)

    raise ValueError(f"Unknown degradation: {degradation_type}")



# --------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------

def calculate_metrics(original_image, processed_image):

    psnr = peak_signal_noise_ratio(
        original_image,
        processed_image,
        data_range=1.0
    )

    ssim = structural_similarity(
        original_image,
        processed_image,
        data_range=1.0
    )

    return psnr, ssim


def edge_columns(original_image, processed_image):
    """Edge-preservation scores for one result row (see edge_metrics.py)."""

    precision, recall, f1_score = calculate_edge_metrics(
        original_image,
        processed_image
    )

    return {
        "Edge_Precision": precision,
        "Edge_Recall": recall,
        "Edge_F1": f1_score
    }


# --------------------------------------------------
# 5. FILTERS
# --------------------------------------------------

filters = {
    "Gaussian Filter": apply_gaussian_filter,
    "Median Filter": apply_median_filter,
    "Bilateral Filter": apply_bilateral_filter,
    "NLM Filter": apply_nlm_filter,
    "Lee Filter": apply_lee_filter,
    "TV Filter": apply_tv_filter
}


# --------------------------------------------------
# 6. RUN EXPERIMENTS
# --------------------------------------------------

def load_tuned_parameters():

    if TUNED_PARAMETERS_FILE is None:
        return None

    with open(TUNED_PARAMETERS_FILE, encoding="utf-8") as file:
        return json.load(file)


def noise_seed(image_index, noise_index, level_index):

    return (
        BASE_SEED
        + image_index * 1000
        + noise_index * 100
        + level_index
    )


def glare_seed(image_index):
    """One seed per image, the same for Low, Medium and High.

    The seed places the glare spot, so only the strength changes between
    levels. JPEG compression is not random and needs no seed.
    """

    return BASE_SEED + image_index * 1000 + GLARE_SEED_OFFSET


def get_filter_parameters(filter_name, filter_function, noise_type,
                          level_name, noise_value, tuned_parameters):

    params = {}

    if tuned_parameters is not None:
        params = tuned_parameters[
            noise_type
        ][level_name][filter_name].copy()

    if (
        filter_name == "Lee Filter"
        and noise_type == "Speckle"
        and "speckle_variance"
        in inspect.signature(filter_function).parameters
    ):
        params["speckle_variance"] = noise_value

    return params


def run_experiments(images, tuned_parameters=None, selection=None):

    results = []

    if tuned_parameters is None:
        tuned_parameters = load_tuned_parameters()

    # Noise experiments (default) or image degradation experiments.
    # Both use the same filters, scores, timer and saving below.
    is_degradation = (
        selection is not None
        and selection.get("experiment") == "degradation"
    )
    condition_levels = DEGRADATION_LEVELS if is_degradation else NOISE_LEVELS
    baseline_name = DEGRADED_BASELINE if is_degradation else NOISY_BASELINE
    category = "Image degradation" if is_degradation else "Noise"

    noise_types = list(condition_levels.keys())

    active_filters = filters
    report_progress = None
    if selection is not None:
        active_filters = {name: filters[name] for name in selection["filters"]}
        report_progress = selection.get("report_progress")

    # A selection with image None means "every image in the dataset".
    images_total = sum(
        1 for name, _ in images
        if selection is None or selection["image"] in (None, name)
    )
    images_done = 0
    if report_progress:
        report_progress(images_done, images_total)

    # Skip unselected items instead of removing them, so every image and
    # noise condition keeps its original position and random seed.
    for image_index, (image_name, original_image) in enumerate(images):

        if selection is not None and selection["image"] not in (None, image_name):
            continue

        print(f"\nProcessing: {image_name}")

        for noise_index, noise_type in enumerate(noise_types):

            for level_index, (level_name, noise_value) in enumerate(
                condition_levels[noise_type].items()
            ):

                # A selection with condition None means "all conditions".
                if (
                    selection is not None
                    and selection["condition"] is not None
                    and (noise_type, level_name) != selection["condition"]
                ):
                    continue

                # Generate one damaged image for all filters. In degradation
                # experiments "noisy_image" holds the JPEG or glare image.
                if is_degradation:
                    seed = glare_seed(image_index) if noise_type == "Glare" else None

                    noisy_image = generate_degraded_image(
                        original_image,
                        noise_type,
                        noise_value,
                        seed
                    )
                else:
                    seed = noise_seed(image_index, noise_index, level_index)

                    noisy_image = generate_noisy_image(
                        original_image,
                        noise_type,
                        noise_value,
                        seed
                    )

                # Evaluate the damaged image before filtering
                psnr, ssim = calculate_metrics(
                    original_image,
                    noisy_image
                )

                baseline_row = {
                    "Image": image_name,
                    "Noise_Type": noise_type,
                    "Noise_Level": level_name,
                    "Filter": baseline_name,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "Execution_Time": 0,
                    **edge_columns(original_image, noisy_image)
                }

                if is_degradation:
                    baseline_row["Category"] = category

                results.append(baseline_row)
                filtered_images = {}

                # Apply every filter to the same noisy image
                for filter_name, filter_function in active_filters.items():

                    params = get_filter_parameters(
                        filter_name,
                        filter_function,
                        noise_type,
                        level_name,
                        noise_value,
                        tuned_parameters
                    )

                    filter_input = noisy_image.copy()

                    # Time only the filter call
                    start_time = time.perf_counter()
                    filtered_image = filter_function(
                            filter_input,
                            **params
                    )
                    execution_time = time.perf_counter() - start_time

                    filtered_images[filter_name] = filtered_image.copy()

                    psnr, ssim = calculate_metrics(
                        original_image,
                        filtered_image
                    )

                    filter_row = {
                        "Image": image_name,
                        "Noise_Type": noise_type,
                        "Noise_Level": level_name,
                        "Filter": filter_name,
                        "PSNR": psnr,
                        "SSIM": ssim,
                        "Execution_Time": execution_time,
                        **edge_columns(original_image, filtered_image)
                    }

                    if is_degradation:
                        filter_row["Category"] = category

                    results.append(filter_row)

                # Degradation pictures say "Degraded" instead of "Noisy".
                # Noise runs call save_image_comparison exactly as before.
                picture_options = {"kind": "degradation"} if is_degradation else {}

                # Full-dataset jobs: one picture for every image, made from
                # the images already filtered above.
                if selection is not None and selection["image"] is None:
                    # Runs over all conditions save smaller JPEG pictures,
                    # because they save 12 pictures for every image.
                    compact = selection["condition"] is None

                    if selection.get("save_pictures", True):
                        save_image_comparison(
                            original_image,
                            noisy_image,
                            filtered_images,
                            noise_type,
                            level_name,
                            comparison_file_name(
                                noise_type, level_name, image_index, image_name,
                                extension=".jpg" if compact else ".png"
                            ),
                            compact=compact,
                            **picture_options
                        )

                # Other runs: one example picture, for the first image
                elif image_index == 0 or selection is not None:
                    save_image_comparison(
                        original_image,
                        noisy_image,
                        filtered_images,
                        noise_type,
                        level_name,
                        **picture_options
                    )

        images_done += 1
        if report_progress:
            report_progress(images_done, images_total)

    return pd.DataFrame(results)



def save_image_comparison(original, noisy, filtered_images,
                          noise_type, noise_level, file_name=None,
                          compact=False, kind="noise"):

    output_dir = RESULTS_PATH / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    # In degradation experiments the damaged image is JPEG or glare, not noise
    images = [
        ("Original", original),
        ("Degraded" if kind == "degradation" else "Noisy", noisy)
    ]

    images.extend(filtered_images.items())

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (name, image) in enumerate(images):

        axes[i].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")

        if name == "Original":
            axes[i].set_title("Original Image")
        else:
            psnr, ssim = calculate_metrics(original, image)

            axes[i].set_title(
                f"{name}\n"
                f"PSNR: {psnr:.2f} dB\n"
                f"SSIM: {ssim:.3f}",
                fontsize=10
            )

    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"{noise_type} - {noise_level}" if kind == "degradation"
        else f"{noise_type} Noise - {noise_level}",
        fontsize=16
    )

    plt.tight_layout()

    filename = (
        f"{noise_type.lower().replace('-', '_').replace(' ', '_')}"
        f"_{noise_level.lower()}.png"
    )

    # A given file_name may include a subfolder, e.g. one per noise condition
    path = output_dir / (file_name or filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    if compact:
        # Same layout, smaller file: about 0.2 MB instead of 1.5 MB
        plt.savefig(path, dpi=100, pil_kwargs={"quality": 85})
    else:
        plt.savefig(path, dpi=150)
    plt.close()


def save_noise_comparison(original_image, image_name):
    """Compare all four noise types on the same original image."""

    noise_settings = [
        ("Gaussian", 25),
        ("Salt-and-Pepper", 0.10),
        ("Speckle", 0.04),
        ("Poisson", 30)
    ]

    images = [("Original", original_image)]

    for noise_type, noise_value in noise_settings:
        noisy_image = generate_noisy_image(
            original_image,
            noise_type,
            noise_value,
            BASE_SEED
        )

        images.append((noise_type, noisy_image))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, (name, image) in zip(axes, images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

        if name == "Original":
            ax.set_title("Original")
        else:
            psnr, ssim = calculate_metrics(original_image, image)

            ax.set_title(
                f"{name}\nPSNR: {psnr:.2f} dB\n"
                f"SSIM: {ssim:.3f}",
                fontsize=11
            )

    output_dir = RESULTS_PATH / "noise_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{Path(image_name).stem}_noise_comparison.png",
        dpi=150
    )

    plt.close()

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save individual image results
    results.to_csv(
        RESULTS_PATH / "experiment_results.csv",
        index=False
    )

    # Calculate average and standard deviation
    summary = results.groupby(
        ["Noise_Type", "Noise_Level", "Filter"]
    ).agg({
        "PSNR": ["mean", "std"],
        "SSIM": ["mean", "std"],
        "Execution_Time": ["mean", "std"],
        "Edge_Precision": ["mean", "std"],
        "Edge_Recall": ["mean", "std"],
        "Edge_F1": ["mean", "std"]
    }).reset_index()

    summary.to_csv(
        RESULTS_PATH / "summary.csv",
        index=False
    )

    print("\nResults saved successfully.")


# --------------------------------------------------
# 8. CREATE GRAPHS
# --------------------------------------------------

def create_graphs(results):

    graph_path = RESULTS_PATH / "graphs"
    graph_path.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("PSNR", "PSNR (dB)"),
        ("SSIM", "SSIM"),
        ("Execution_Time", "Execution Time (seconds)"),
        ("Edge_Precision", "Edge Precision"),
        ("Edge_Recall", "Edge Recall"),
        ("Edge_F1", "Edge F1-score")
    ]

    level_order = ["Low", "Medium", "High"]

    for noise_type in NOISE_LEVELS:

        noise_results = results[
            results["Noise_Type"] == noise_type
        ]

        for metric, ylabel in metrics:

            summary = noise_results.groupby(
                ["Noise_Level", "Filter"]
            )[metric].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))

            if metric == "Execution_Time":
                filter_names = list(filters.keys())
                x = np.arange(len(level_order))
                bar_width = 0.8 / len(filter_names)
                all_values = []

                for filter_index, filter_name in enumerate(filter_names):
                    filter_results = (
                        summary[summary["Filter"] == filter_name]
                        .set_index("Noise_Level")
                        .reindex(level_order)
                    )
                    values = filter_results[metric].to_numpy(dtype=float)
                    all_values.extend(values[np.isfinite(values) & (values > 0)])
                    positions = x - 0.4 + bar_width * (filter_index + 0.5)
                    bars = ax.bar(positions, values, bar_width, label=filter_name)

                    for bar, value in zip(bars, values):
                        if np.isfinite(value) and value > 0:
                            ax.annotate(
                                f"{value:.4g}s",
                                (bar.get_x() + bar.get_width() / 2, value),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                rotation=45
                            )

                if all_values:
                    y_floor = min(all_values) * 0.5
                    y_top = max(all_values) * 4
                    for bar in ax.patches:
                        bar.set_y(y_floor)
                        bar.set_height(max(bar.get_height() - y_floor, y_floor * 0.01))
                    ax.set_yscale("log")
                    ax.set_ylim(y_floor, y_top)

                ax.set_xticks(x, level_order)
                ax.set_ylabel("Execution Time (seconds, log scale)")
            else:
                for filter_name in ["Noisy (Baseline)"] + list(filters.keys()):
                    filter_results = (
                        summary[summary["Filter"] == filter_name]
                        .set_index("Noise_Level")
                        .reindex(level_order)
                    )
                    ax.plot(
                        level_order,
                        filter_results[metric],
                        marker="o",
                        label=filter_name
                    )

                ax.set_ylabel(ylabel)

            ax.set_title(f"{metric} Comparison - {noise_type}")
            ax.set_xlabel("Noise Level")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            filename = (
                f"{noise_type.lower().replace('-', '_')}"
                f"_{metric.lower()}.png"
            )

            plt.savefig(graph_path / filename)
            plt.close()

    print("Graphs created successfully.")


# --------------------------------------------------
# 9. MAIN FUNCTION
# --------------------------------------------------

def next_results_path():
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    run_numbers = []
    for path in RESULTS_PATH.iterdir():
        if path.is_dir() and path.name.startswith("result"):
            suffix = path.name[6:]
            if suffix.isdigit():
                run_numbers.append(int(suffix))

    next_number = max(run_numbers, default=0) + 1

    # Creating the folder reserves it, so two runs can never share it.
    while True:
        path = RESULTS_PATH / f"result{next_number}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            next_number += 1


def build_current_run_config(images, tuned_parameters, selection=None):

    chosen_images = images
    chosen_filters = filters
    image_indices = None
    job = None
    parameter_label = None
    parameter_sources = None
    parameter_sources_by_condition = None

    if selection is not None:
        if selection["image"] is not None:
            image_indices = [
                index for index, (name, _) in enumerate(images)
                if name == selection["image"]
            ]
            chosen_images = [images[index] for index in image_indices]
        chosen_filters = {name: filters[name] for name in selection["filters"]}
        job = selection["job"]
        parameter_sources = selection.get("parameter_sources")
        parameter_sources_by_condition = selection.get("parameter_sources_by_condition")
        if parameter_sources is None and parameter_sources_by_condition is None:
            parameter_label = PARAMETER_SOURCES[job["parameter_source"]]["label"]

    return build_run_config(
        images=chosen_images,
        dataset_path=DATASET_PATH,
        results_path=RESULTS_PATH,
        max_images=MAX_IMAGES,
        noise_levels=NOISE_LEVELS,
        base_seed=BASE_SEED,
        filters=chosen_filters,
        tuned_parameters=tuned_parameters,
        tuned_file=TUNED_PARAMETERS_FILE,
        get_seed=noise_seed,
        get_parameters=get_filter_parameters,
        degradation_levels=DEGRADATION_LEVELS,
        glare_seed_offset=GLARE_SEED_OFFSET,
        image_indices=image_indices,
        job=job,
        parameter_label=parameter_label,
        parameter_sources=parameter_sources,
        parameter_sources_by_condition=parameter_sources_by_condition
    )


def main(selection=None):

    images = load_dataset()

    if selection is not None and selection["image"] not in [
        None, *(name for name, _ in images)
    ]:
        raise ValueError(
            f"Image {selection['image']} was not found in {DATASET_PATH.name}."
        )

    tuned_parameters = load_tuned_parameters()

    # Settings chosen per filter in the frontend replace any file.
    if selection is not None and selection.get("filter_parameters") is not None:
        tuned_parameters = selection["filter_parameters"]

    global RESULTS_PATH
    RESULTS_PATH = next_results_path()

    # Save settings first, so a failed save stops the run early
    run_config = build_current_run_config(images, tuned_parameters, selection)
    save_run_config(RESULTS_PATH, run_config)

    if selection is None:
        save_noise_comparison(
        images[0][1],
        images[0][0]
        )

    results = run_experiments(images, tuned_parameters, selection)

    save_results(results)

    if selection is None:
        create_graphs(results)

    mark_run_completed(RESULTS_PATH, run_config)

    print("\nAll experiments completed!")


# --------------------------------------------------
# 10. FRONTEND JOBS
# --------------------------------------------------

def prepare_job(job):
    """Checks a frontend job and points the normal runner at its settings."""

    global DATASET_PATH, RESULTS_PATH, MAX_IMAGES, TUNED_PARAMETERS_FILE

    if job.get("type") not in ("single_image", "full_dataset"):
        raise ValueError(f"Unknown job type: {job.get('type')}")

    # A full-dataset job uses every image, so it has no single image name.
    image = job.get("image") if job["type"] == "single_image" else None
    if job["type"] == "single_image" and not image:
        raise ValueError("Choose an image for a single-image job.")

    if job["dataset"] not in JOB_DATASETS:
        raise ValueError(f"Unknown dataset: {job['dataset']}")

    # "noise" (also older jobs without this field) or "degradation"
    experiment = job.get("experiment", "noise")
    if experiment not in ("noise", "degradation"):
        raise ValueError(f"Unknown experiment: {experiment}")

    condition_levels = DEGRADATION_LEVELS if experiment == "degradation" else NOISE_LEVELS

    if experiment == "degradation" and isinstance(job["filters"], list):
        raise ValueError("Degradation experiments need per-filter settings.")

    # "all" runs every condition of this experiment at every level:
    # the 12 noise conditions, or the 6 degradation conditions.
    all_conditions = job.get("noise_scope") == "all"

    if all_conditions:
        if job["type"] != "full_dataset" or isinstance(job["filters"], list):
            raise ValueError(
                "All noise conditions need a full-dataset job with per-filter settings."
            )
        conditions = [
            (noise_type, level) for noise_type, levels in condition_levels.items()
            for level in levels
        ]
    else:
        if job["noise_level"] not in condition_levels.get(job["noise_type"], {}):
            raise ValueError(
                f"Unknown {experiment} condition: {job['noise_type']} / {job['noise_level']}"
            )
        conditions = [(job["noise_type"], job["noise_level"])]

    unknown_filters = [name for name in job["filters"] if name not in filters]
    if not job["filters"] or unknown_filters:
        raise ValueError(f"Invalid filter selection: {list(job['filters'])}")

    selected_filters = [name for name in filters if name in job["filters"]]
    filter_parameters = None
    parameter_sources = None
    parameter_sources_by_condition = None

    if isinstance(job["filters"], list):
        # Older jobs: one parameter source for every filter.
        TUNED_PARAMETERS_FILE = parameter_file(job["parameter_source"])
    else:
        # Newer jobs: every filter brings its own checked values.
        validate_job_filters(job)
        filter_parameters = {}
        parameter_sources_by_condition = {}

        for noise_type, noise_level in conditions:
            chosen = {}
            records = {}

            for name in selected_filters:
                entry = job["filters"][name]
                values = entry["parameters"]

                # None: use this preset's saved values for this condition
                if values is None:
                    values = preset_parameters(
                        entry["preset"], name, filters[name], noise_type, noise_level
                    )

                values = validate_filter_parameters(name, values)
                chosen[name] = values
                records[name] = describe_preset_use(
                    entry["preset"], name, filters[name], noise_type, noise_level, values
                )
                if all_conditions:
                    records[name]["values_from"] = (
                        "preset for each condition" if entry["parameters"] is None
                        else "same values for every condition"
                    )

            filter_parameters.setdefault(noise_type, {})[noise_level] = chosen
            parameter_sources_by_condition[f"{noise_type} / {noise_level}"] = records

        if not all_conditions:
            parameter_sources = records
            parameter_sources_by_condition = None

        TUNED_PARAMETERS_FILE = None

    DATASET_PATH = PROJECT_DIR / "dataset" / job["dataset"]
    RESULTS_PATH = JOB_RESULTS_PATH
    MAX_IMAGES = None

    return {
        "image": image,
        "experiment": experiment,
        "condition": None if all_conditions else conditions[0],
        "filters": selected_filters,
        "filter_parameters": filter_parameters,
        "parameter_sources": parameter_sources,
        "parameter_sources_by_condition": parameter_sources_by_condition,
        "save_pictures": job.get("save_pictures", True),
        "job": job,
    }


def run_job(job_path):
    """Runs one saved job and records whether it finished."""

    job_dir = Path(job_path).resolve().parent
    mark_job(job_dir, "running")

    def report_progress(images_done, images_total):
        mark_job(job_dir, "running", images_done=images_done, images_total=images_total)

    try:
        selection = prepare_job(read_job(job_path))
        selection["report_progress"] = report_progress
        main(selection)
    except Exception as error:
        traceback.print_exc()
        incomplete = (
            RESULTS_PATH.name if RESULTS_PATH.parent == JOB_RESULTS_PATH else None
        )
        mark_job(job_dir, "failed", error=str(error), incomplete_result=incomplete)
        raise SystemExit(1)

    mark_job(job_dir, "completed", result=RESULTS_PATH.name)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--job":
        run_job(sys.argv[2])
    else:
        main()
