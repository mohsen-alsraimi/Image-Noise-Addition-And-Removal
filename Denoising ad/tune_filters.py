
import json
import inspect
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from main import (
    NOISE_LEVELS,
    BASE_SEED,
    filters,
    generate_noisy_image,
    calculate_metrics,
)

DATASET_PATH = Path("dataset/Set12")
OUTPUT_PATH = Path("best_parameters.json")
MAX_TUNING_IMAGES = None # Quick test; None = all images

PARAMETER_GRID = {
    "Gaussian Filter": [
        {"kernel_size": 3, "sigma": 0.5},
        {"kernel_size": 5, "sigma": 1.0},
        {"kernel_size": 7, "sigma": 1.5},
    ],
    "Median Filter": [
        {"kernel_size": 3},
        {"kernel_size": 5},
        {"kernel_size": 7},
    ],
    "Bilateral Filter": [
        {"diameter": 5, "sigma_color": 0.05},
        {"diameter": 9, "sigma_color": 0.1},
        {"diameter": 9, "sigma_color": 0.2},
    ],
    "NLM Filter": [
        {"h": 0.1},
        {"h": 0.15},
        {"h": 0.2},
    ],
    "Lee Filter": [
        {"window_size": 3},
        {"window_size": 5},
        {"window_size": 7},
    ],
    "TV Filter": [
        {"weight": 0.03},
        {"weight": 0.07},
        {"weight": 0.1},
    ],
}


def load_images():
    paths = sorted(
        p for p in DATASET_PATH.rglob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )

    if MAX_TUNING_IMAGES is not None:
        paths = paths[:MAX_TUNING_IMAGES]

    images = []

    for path in paths:
        image = cv2.imread(
            str(path), cv2.IMREAD_GRAYSCALE
        )
        if image is not None:
            images.append(
                image.astype(np.float32) / 255.0
            )

    if not images:
        raise ValueError("No Set12 images found.")

    print(f"Loaded {len(images)} tuning images")
    return images


def tune():
    images = load_images()
    best_parameters = {}
    records = []

    for noise_index, (noise_type, levels) in enumerate(
        NOISE_LEVELS.items()
    ):
        best_parameters[noise_type] = {}

        for level_index, (level_name, noise_value) in enumerate(
            levels.items()
        ):
            best_parameters[noise_type][level_name] = {}

            # Generate the same noisy inputs for all candidates.
            noisy_images = []

            for i, original in enumerate(images):
                seed = (
                    BASE_SEED + i * 1000
                    + noise_index * 100 + level_index
                )
                noisy_images.append(
                    generate_noisy_image(
                        original, noise_type, noise_value, seed
                    )
                )

            for filter_name, filter_function in filters.items():
                highest_psnr = float("-inf")
                selected = None

                for parameters in PARAMETER_GRID[filter_name]:
                    scores = []
                    ssim_scores = []

                    for original, noisy in zip(images, noisy_images):
                        kwargs = parameters.copy()

                        # Supply the known variance for simulated
                        # speckle noise if this Lee implementation
                        # accepts it.
                        if (
                            filter_name == "Lee Filter"
                            and noise_type == "Speckle"
                            and "speckle_variance"
                            in inspect.signature(filter_function).parameters
                        ):
                            kwargs["speckle_variance"] = noise_value

                        restored = filter_function(
                            noisy.copy(), **kwargs
                        )

                        psnr, ssim = calculate_metrics(
                            original, restored
                        )
                        scores.append(psnr)
                        ssim_scores.append(ssim)

                    average_psnr = float(np.mean(scores))
                    average_ssim = float(np.mean(ssim_scores))

                    records.append({
                        "Noise_Type": noise_type,
                        "Noise_Level": level_name,
                        "Filter": filter_name,
                        "Parameters": json.dumps(parameters),
                        "Mean_PSNR": average_psnr,
                        "Mean_SSIM": average_ssim,
                    })

                    if average_psnr > highest_psnr:
                        highest_psnr = average_psnr
                        selected = parameters.copy()

                best_parameters[noise_type][level_name][
                    filter_name
                ] = selected

                print(
                    f"{noise_type} / {level_name} / "
                    f"{filter_name}: {selected} "
                    f"PSNR={highest_psnr:.2f}"
                )

    # Save all candidate results
    pd.DataFrame(records).to_csv(
        "tuning_results.csv", index=False
    )

    # Save the best parameter settings
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(best_parameters, f, indent=4)

    print("\nTuning completed successfully!")


if __name__ == "__main__":
    tune()
