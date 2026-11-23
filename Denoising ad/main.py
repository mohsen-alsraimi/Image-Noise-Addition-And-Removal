
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


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

DATASET_PATH = Path("dataset/BSD68")
RESULTS_PATH = Path("results")

BASE_SEED = 42
MAX_IMAGES = 1 # Set to None to process the full dataset

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


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset():
    images = []

    image_paths = sorted(
        list(DATASET_PATH.rglob("*.png")) +
        list(DATASET_PATH.rglob("*.jpg")) +
        list(DATASET_PATH.rglob("*.bmp"))
    )

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

def run_experiments(images):

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

                results.append({
                    "Image": image_name,
                    "Noise_Type": noise_type,
                    "Noise_Level": level_name,
                    "Filter": "Noisy (Baseline)",
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "Execution_Time": 0
                })
                filtered_images = {}

                # Apply every filter to the same noisy image
                for filter_name, filter_function in filters.items():

                    start_time = time.perf_counter()

                    filtered_image = filter_function(
                        noisy_image.copy()
                    )

                    filtered_images[filter_name] = filtered_image.copy()
                    execution_time = time.perf_counter() - start_time

                    psnr, ssim = calculate_metrics(
                        original_image,
                        filtered_image
                    )

                    results.append({
                        "Image": image_name,
                        "Noise_Type": noise_type,
                        "Noise_Level": level_name,
                        "Filter": filter_name,
                        "PSNR": psnr,
                        "SSIM": ssim,
                        "Execution_Time": execution_time
                    })
                if image_index == 0:
                    save_image_comparison(
                        original_image,
                        noisy_image,
                        filtered_images,
                        noise_type,
                        level_name
                    )

    return pd.DataFrame(results)



def save_image_comparison(original, noisy, filtered_images,
                          noise_type, noise_level):

    output_dir = RESULTS_PATH / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        ("Original", original),
        ("Noisy", noisy)
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
        f"{noise_type} Noise - {noise_level}",
        fontsize=16
    )

    plt.tight_layout()

    filename = (
        f"{noise_type.lower().replace('-', '_').replace(' ', '_')}"
        f"_{noise_level.lower()}.png"
    )

    plt.savefig(output_dir / filename, dpi=150)
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

    return RESULTS_PATH / f"result{max(run_numbers, default=0) + 1}"


def main():

    images = load_dataset()

    global RESULTS_PATH
    RESULTS_PATH = next_results_path()

    save_noise_comparison(
    images[0][1],
    images[0][0]
    ) 

    results = run_experiments(images)

    save_results(results)

    create_graphs(results)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
