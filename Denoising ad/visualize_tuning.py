
import json
import inspect
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from main import (
    BASE_SEED,
    NOISE_LEVELS,
    filters,
    generate_noisy_image,
    calculate_metrics,
)

DATASET = Path("dataset/BSD68")
PARAMETERS_FILE = Path("best_parameters.json")
OUTPUT = Path("results/tuning_visual_comparison")

# Select the image to compare
IMAGE_INDEX = 0

# Change these to explore other noise conditions
NOISE_TYPE = "Salt-and-Pepper"
NOISE_LEVEL = "Medium"


def apply_filter(function, name, noisy, params,
                 noise_type, noise_value):

    kwargs = params.copy()

    if (
        name == "Lee Filter"
        and noise_type == "Speckle"
        and "speckle_variance"
        in inspect.signature(function).parameters
    ):
        kwargs["speckle_variance"] = noise_value

    return function(noisy.copy(), **kwargs)


def main():
    with PARAMETERS_FILE.open(encoding="utf-8") as f:
        tuned_params = json.load(f)

    image_paths = sorted(DATASET.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError("BSD68 dataset not found")

    image_path = image_paths[IMAGE_INDEX]

    original = cv2.imread(
        str(image_path), cv2.IMREAD_GRAYSCALE
    ).astype(np.float32) / 255.0

    # Match the seed formula used in main.py
    noise_index = list(NOISE_LEVELS).index(NOISE_TYPE)
    level_index = list(
        NOISE_LEVELS[NOISE_TYPE]
    ).index(NOISE_LEVEL)

    seed = (
        BASE_SEED
        + IMAGE_INDEX * 1000
        + noise_index * 100
        + level_index
    )

    noise_value = NOISE_LEVELS[
        NOISE_TYPE
    ][NOISE_LEVEL]

    noisy = generate_noisy_image(
        original, NOISE_TYPE, noise_value, seed
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for filter_name, function in filters.items():

        # Original fixed parameters
        before = apply_filter(
            function, filter_name, noisy,
            {}, NOISE_TYPE, noise_value
        )

        # Selected tuning parameters
        params = tuned_params[
            NOISE_TYPE
        ][NOISE_LEVEL][filter_name]

        after = apply_filter(
            function, filter_name, noisy,
            params, NOISE_TYPE, noise_value
        )

        before_psnr, before_ssim = calculate_metrics(
            original, before
        )
        after_psnr, after_ssim = calculate_metrics(
            original, after
        )

        images = [
            ("Original", original),
            ("Noisy", noisy),
            ("Before Tuning", before),
            ("After Tuning", after),
        ]

        fig, axes = plt.subplots(
            1, 4, figsize=(16, 5)
        )

        for ax, (label, img) in zip(axes, images):
            ax.imshow(
                img, cmap="gray",
                vmin=0, vmax=1
            )
            ax.axis("off")
            ax.set_title(label)

        axes[2].set_title(
            f"Before Tuning\n"
            f"PSNR: {before_psnr:.2f} dB\n"
            f"SSIM: {before_ssim:.3f}"
        )

        axes[3].set_title(
            f"After Tuning\n"
            f"PSNR: {after_psnr:.2f} dB\n"
            f"SSIM: {after_ssim:.3f}"
        )

        fig.suptitle(
            f"{filter_name} | {NOISE_TYPE} | "
            f"{NOISE_LEVEL}\n"
            f"PSNR change: "
            f"{after_psnr - before_psnr:+.2f} dB"
        )

        fig.tight_layout()

        filename = (
            f"{image_path.stem}_"
            f"{NOISE_TYPE.lower().replace(' ', '_')}_"
            f"{NOISE_LEVEL.lower()}_"
            f"{filter_name.lower().replace(' ', '_')}.png"
        )

        fig.savefig(
            OUTPUT / filename, dpi=150
        )
        plt.close(fig)

        print(
            f"{filter_name}: "
            f"{before_psnr:.2f} -> "
            f"{after_psnr:.2f} dB"
        )

    print("Visual comparisons saved to:", OUTPUT)


if __name__ == "__main__":
    main()
