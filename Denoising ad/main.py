
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from pathlib import Path

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from add_noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise
)
from image_degradation import add_glare, add_jpeg_compression

from filter import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_bilateral_filter,
    apply_nlm_filter,
    apply_lee_filter
)


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

DATASETS_PATH = Path("dataset")
RESULTS_PATH = Path("results")

BASE_SEED = 42
MAX_IMAGES = 2  # Set to None to process the full dataset

EDGE_CANNY_LOW_THRESHOLD = 50
EDGE_CANNY_HIGH_THRESHOLD = 150
EDGE_MATCH_TOLERANCE = 1

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
    }
}

JPEG_QUALITY_LEVELS = {
    "Low": 90,
    "Medium": 60,
    "High": 30
}

GLARE_ALPHA_LEVELS = {
    "Low": 0.20,
    "Medium": 0.45,
    "High": 0.70
}


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset(dataset_path):
    images = []

    image_paths = sorted(
        list(dataset_path.rglob("*.png")) +
        list(dataset_path.rglob("*.jpg")) +
        list(dataset_path.rglob("*.bmp"))
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is not None:
            image = image.astype(np.float32) / 255.0
            images.append((image_path.name, image))

    if len(images) == 0:
        raise ValueError(
            f"No images found in {dataset_path}"
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


def calculate_edge_metrics(reference_image, processed_image):
    reference = (np.clip(reference_image, 0, 1) * 255).astype(np.uint8)
    processed = (np.clip(processed_image, 0, 1) * 255).astype(np.uint8)

    reference_edges = cv2.Canny(
        reference,
        EDGE_CANNY_LOW_THRESHOLD,
        EDGE_CANNY_HIGH_THRESHOLD
    ) > 0
    processed_edges = cv2.Canny(
        processed,
        EDGE_CANNY_LOW_THRESHOLD,
        EDGE_CANNY_HIGH_THRESHOLD
    ) > 0

    kernel_size = 2 * EDGE_MATCH_TOLERANCE + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    reference_dilated = cv2.dilate(
        reference_edges.astype(np.uint8), kernel
    ) > 0
    processed_dilated = cv2.dilate(
        processed_edges.astype(np.uint8), kernel
    ) > 0

    reference_count = np.count_nonzero(reference_edges)
    processed_count = np.count_nonzero(processed_edges)

    if reference_count == 0 and processed_count == 0:
        return 1.0, 1.0, 1.0

    matched_processed = np.count_nonzero(processed_edges & reference_dilated)
    precision = matched_processed / processed_count if processed_count else 0.0

    matched_reference = np.count_nonzero(reference_edges & processed_dilated)
    recall = matched_reference / reference_count if reference_count else 0.0

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score


# --------------------------------------------------
# 5. FILTERS
# --------------------------------------------------

filters = {
    "Gaussian Filter": apply_gaussian_filter,
    "Median Filter": apply_median_filter,
    "Bilateral Filter": apply_bilateral_filter,
    "NLM Filter": apply_nlm_filter,
    "Lee Filter": apply_lee_filter
}


# --------------------------------------------------
# 6. RUN EXPERIMENTS
# --------------------------------------------------

def run_experiments(images, dataset_name):

    results = []

    noise_types = list(NOISE_LEVELS.keys())

    for image_index, (image_name, original_image) in enumerate(images):

        print(f"\nProcessing: {image_name}")

        for noise_index, noise_type in enumerate(noise_types):

            for level_index, (level_name, noise_value) in enumerate(
                NOISE_LEVELS[noise_type].items()
            ):

                # Generate one noisy image for all filters
                seed = (
                    BASE_SEED
                    + image_index * 1000
                    + noise_index * 100
                    + level_index
                )

                noisy_image = generate_noisy_image(
                    original_image,
                    noise_type,
                    noise_value,
                    seed
                )

                # Evaluate noisy image before filtering
                psnr, ssim = calculate_metrics(
                    original_image,
                    noisy_image
                )
                edge_precision, edge_recall, edge_f1 = calculate_edge_metrics(
                    original_image,
                    noisy_image
                )

                results.append({
                    "Dataset": dataset_name,
                    "Experiment_Category": "Noise",
                    "Image": image_name,
                    "Noise_Type": noise_type,
                    "Degradation_Type": "",
                    "Level": level_name,
                    "Noise_Level": level_name,
                    "Filter": "Noisy (Baseline)",
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "Edge_Precision": edge_precision,
                    "Edge_Recall": edge_recall,
                    "Edge_F1": edge_f1,
                    "Execution_Time": 0
                })

                # Apply every filter to the same noisy image
                for filter_name, filter_function in filters.items():

                    start_time = time.perf_counter()

                    filtered_image = filter_function(
                        noisy_image.copy()
                    )

                    execution_time = time.perf_counter() - start_time

                    psnr, ssim = calculate_metrics(
                        original_image,
                        filtered_image
                    )
                    edge_precision, edge_recall, edge_f1 = calculate_edge_metrics(
                        original_image,
                        filtered_image
                    )

                    results.append({
                        "Dataset": dataset_name,
                        "Experiment_Category": "Noise",
                        "Image": image_name,
                        "Noise_Type": noise_type,
                        "Degradation_Type": "",
                        "Level": level_name,
                        "Noise_Level": level_name,
                        "Filter": filter_name,
                        "PSNR": psnr,
                        "SSIM": ssim,
                        "Edge_Precision": edge_precision,
                        "Edge_Recall": edge_recall,
                        "Edge_F1": edge_f1,
                        "Execution_Time": execution_time
                    })

    return pd.DataFrame(results)


def save_sample_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image_uint8):
        raise ValueError(f"Could not save sample image to {path}")


def run_degradation_experiments(images, dataset_name, run_name, sample_root):
    results = []
    degradation_levels = {
        "JPEG Compression": JPEG_QUALITY_LEVELS,
        "Glare": GLARE_ALPHA_LEVELS
    }

    for image_index, (image_name, original_image) in enumerate(images):
        print(f"\nDegradation experiments: {image_name}")

        for degradation_type, levels in degradation_levels.items():
            for level_index, (level_name, value) in enumerate(levels.items()):
                seed = BASE_SEED + image_index * 1000 + level_index

                if degradation_type == "JPEG Compression":
                    degraded_image = add_jpeg_compression(original_image, value)
                    baseline_name = "JPEG Baseline"
                    sample_folder = "jpeg"
                else:
                    degraded_image = add_glare(original_image, value, seed)
                    baseline_name = "Glare Baseline"
                    sample_folder = "glare"

                psnr, ssim = calculate_metrics(original_image, degraded_image)
                edge_precision, edge_recall, edge_f1 = calculate_edge_metrics(
                    original_image, degraded_image
                )
                results.append({
                    "Dataset": dataset_name,
                    "Experiment_Category": "Image Degradation",
                    "Image": image_name,
                    "Noise_Type": "",
                    "Degradation_Type": degradation_type,
                    "Level": level_name,
                    "Noise_Level": "",
                    "Filter": baseline_name,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "Edge_Precision": edge_precision,
                    "Edge_Recall": edge_recall,
                    "Edge_F1": edge_f1,
                    "Execution_Time": 0
                })

                save_samples = image_index == 0 and level_name == "Medium"
                if save_samples:
                    sample_prefix = f"{run_name}_{image_name.rsplit('.', 1)[0]}"
                    sample_dir = sample_root / sample_folder
                    save_sample_image(
                        original_image,
                        sample_dir / f"{sample_prefix}_original.png"
                    )
                    save_sample_image(
                        degraded_image,
                        sample_dir / f"{sample_prefix}_degraded.png"
                    )

                for filter_name, filter_function in filters.items():
                    start_time = time.perf_counter()
                    filtered_image = filter_function(degraded_image.copy())
                    execution_time = time.perf_counter() - start_time
                    psnr, ssim = calculate_metrics(original_image, filtered_image)
                    edge_precision, edge_recall, edge_f1 = calculate_edge_metrics(
                        original_image, filtered_image
                    )
                    results.append({
                        "Dataset": dataset_name,
                        "Experiment_Category": "Image Degradation",
                        "Image": image_name,
                        "Noise_Type": "",
                        "Degradation_Type": degradation_type,
                        "Level": level_name,
                        "Noise_Level": "",
                        "Filter": filter_name,
                        "PSNR": psnr,
                        "SSIM": ssim,
                        "Edge_Precision": edge_precision,
                        "Edge_Recall": edge_recall,
                        "Edge_F1": edge_f1,
                        "Execution_Time": execution_time
                    })

                    if save_samples:
                        save_sample_image(
                            filtered_image,
                            sample_dir / f"{sample_prefix}_{filter_name.lower().replace(' ', '_')}.png"
                        )

    return pd.DataFrame(results)


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
        [
            "Dataset",
            "Experiment_Category",
            "Noise_Type",
            "Degradation_Type",
            "Level",
            "Filter"
        ]
    ).agg({
        "PSNR": ["mean", "std"],
        "SSIM": ["mean", "std"],
        "Edge_Precision": ["mean", "std"],
        "Edge_Recall": ["mean", "std"],
        "Edge_F1": ["mean", "std"],
        "Execution_Time": ["mean", "std"]
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
        ("Edge_Precision", "Edge Precision"),
        ("Edge_Recall", "Edge Recall"),
        ("Edge_F1", "Edge F1-score"),
        ("Execution_Time", "Execution Time (seconds)")
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

            dataset_name = str(results["Dataset"].iloc[0])
            filename = (
                f"{dataset_name.lower()}_"
                f"{noise_type.lower().replace('-', '_')}"
                f"_{metric.lower()}.png"
            )

            plt.savefig(graph_path / filename)
            plt.close()

    print("Graphs created successfully.")


def create_degradation_graphs(results):
    graph_path = RESULTS_PATH / "graphs"
    graph_path.mkdir(parents=True, exist_ok=True)

    dataset_name = str(results["Dataset"].iloc[0])
    degradation_settings = {
        "JPEG Compression": {
            "levels": list(JPEG_QUALITY_LEVELS),
            "x_label": "JPEG Quality",
            "x_ticks": [str(value) for value in JPEG_QUALITY_LEVELS.values()],
            "baseline": "JPEG Baseline",
            "file_prefix": "jpeg_compression"
        },
        "Glare": {
            "levels": list(GLARE_ALPHA_LEVELS),
            "x_label": "Glare Intensity (alpha)",
            "x_ticks": [f"{value:.2f}" for value in GLARE_ALPHA_LEVELS.values()],
            "baseline": "Glare Baseline",
            "file_prefix": "glare"
        }
    }
    metrics = [
        ("PSNR", "PSNR (dB)"),
        ("SSIM", "SSIM"),
        ("Edge_Precision", "Edge Precision"),
        ("Edge_Recall", "Edge Recall"),
        ("Edge_F1", "Edge F1-score"),
        ("Execution_Time", "Execution Time (seconds)")
    ]

    for degradation_type, settings in degradation_settings.items():
        degradation_results = results[
            (results["Experiment_Category"] == "Image Degradation")
            & (results["Degradation_Type"] == degradation_type)
        ]
        if degradation_results.empty:
            continue

        for metric, ylabel in metrics:
            summary = degradation_results.groupby(
                ["Level", "Filter"]
            )[metric].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            level_order = settings["levels"]

            if metric == "Execution_Time":
                filter_names = list(filters.keys())
                x = np.arange(len(level_order))
                bar_width = 0.8 / len(filter_names)
                all_values = []

                for filter_index, filter_name in enumerate(filter_names):
                    filter_results = (
                        summary[summary["Filter"] == filter_name]
                        .set_index("Level")
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

                ax.set_xticks(x, settings["x_ticks"])
                ax.set_ylabel("Execution Time (seconds, log scale)")
            else:
                filter_names = [settings["baseline"]] + list(filters.keys())
                for filter_name in filter_names:
                    filter_results = (
                        summary[summary["Filter"] == filter_name]
                        .set_index("Level")
                        .reindex(level_order)
                    )
                    ax.plot(
                        settings["x_ticks"],
                        filter_results[metric],
                        marker="o",
                        label=filter_name
                    )
                ax.set_ylabel(ylabel)

            ax.set_title(f"{dataset_name} - {degradation_type} - {metric}")
            ax.set_xlabel(settings["x_label"])
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            filename = (
                f"{dataset_name.lower()}_{settings['file_prefix']}_"
                f"{metric.lower()}.png"
            )
            fig.savefig(graph_path / filename)
            plt.close(fig)

    print("Degradation graphs created successfully.")


# --------------------------------------------------
# 9. MAIN FUNCTION
# --------------------------------------------------

def next_results_path(dataset_name):
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    run_numbers = []
    for path in RESULTS_PATH.iterdir():
        prefix = f"{dataset_name}_result"
        if path.is_dir() and path.name.startswith(prefix):
            suffix = path.name[len(prefix):]
            if suffix.isdigit():
                run_numbers.append(int(suffix))

    return RESULTS_PATH / f"{dataset_name}_result{max(run_numbers, default=0) + 1}"


def main():
    dataset_paths = sorted(
        path for path in DATASETS_PATH.iterdir()
        if path.is_dir()
    ) if DATASETS_PATH.exists() else []

    if not dataset_paths:
        raise ValueError(f"No dataset folders found in {DATASETS_PATH}")

    global RESULTS_PATH
    base_results_path = RESULTS_PATH

    for dataset_path in dataset_paths:
        print(f"\n=== Dataset: {dataset_path.name} ===")
        images = load_dataset(dataset_path)
        RESULTS_PATH = base_results_path
        RESULTS_PATH = next_results_path(dataset_path.name)
        run_name = RESULTS_PATH.name
        noise_results = run_experiments(images, dataset_path.name)
        degradation_results = run_degradation_experiments(
            images,
            dataset_path.name,
            run_name,
            base_results_path / "sample_images"
        )
        results = pd.concat(
            [noise_results, degradation_results],
            ignore_index=True
        )
        save_results(results)
        create_graphs(results)
        create_degradation_graphs(results)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
