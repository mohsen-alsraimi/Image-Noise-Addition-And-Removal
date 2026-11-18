"""Stage 3 demonstration: add three types of noise at three strengths.

Run from the project root folder (IP):

    python -m demos.demo_noise                  # print the results table
    python -m demos.demo_noise --show           # also show the comparison figure
    python -m demos.demo_noise --save-figure    # save results/figures/stage3_noise_examples.png
    python -m demos.demo_noise --save-outputs   # save noisy PNGs + CSV table in results/

What it does:
1. Loads the synthetic test image (the clean original).
2. Adds Gaussian (sigma 10/25/50), salt-and-pepper (2/5/10%) and speckle
   (variance 0.01/0.04/0.10) noise, always with the same fixed seed (42).
3. Measures MSE and PSNR of every noisy image against the original.

No filtering is done here. These noisy images are the BASELINES that the
filters of Stage 4 will later try to improve. The original is never changed.
"""

import argparse
import csv
import datetime
import sys
from pathlib import Path

import numpy as np

from demos.demo_image_loading import prepare_synthetic_image
from src.image_io import load_grayscale_image, save_grayscale_image
from src.metrics import mse, psnr
from src.noise import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    add_speckle_noise,
    salt_and_pepper_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_PATH = RESULTS_DIR / "figures" / "stage3_noise_examples.png"
TABLE_PATH = RESULTS_DIR / "tables" / "stage3_noise_results.csv"
NOISY_DIR = RESULTS_DIR / "noisy"

SEED = 42  # fixed seed -> every run produces exactly the same noisy images

# (noise type, parameter name, symbol for labels, function, three strengths)
NOISE_SETTINGS = [
    ("gaussian", "sigma", "σ", add_gaussian_noise, [10, 25, 50]),
    ("salt_and_pepper", "amount", "amount", add_salt_and_pepper_noise, [0.02, 0.05, 0.10]),
    ("speckle", "variance", "variance", add_speckle_noise, [0.01, 0.04, 0.10]),
]
DISPLAY_NAMES = {"gaussian": "Gaussian", "salt_and_pepper": "Salt-and-pepper", "speckle": "Speckle"}


def run_experiments(original, image_name):
    """Create all nine noisy images and measure them. Returns a list of result dicts."""
    results = []
    number = 1
    for noise_type, parameter, symbol, function, strengths in NOISE_SETTINGS:
        for strength in strengths:
            noisy = function(original, strength, seed=SEED)
            row = {
                "experiment_id": f"E{number:03d}",
                "group_id": f"G{number:03d}",
                "image": image_name,
                "height": original.shape[0],
                "width": original.shape[1],
                "noise_type": noise_type,
                "parameter": parameter,
                "symbol": symbol,
                "strength": strength,
                "seed": SEED,
                "filter": "none (noisy baseline)",
                "mse": mse(original, noisy),
                "psnr_db": psnr(original, noisy),
                "changed_pixels": int(np.count_nonzero(noisy != original)),
                "pixels_at_0_or_255": int(np.count_nonzero((noisy == 0) | (noisy == 255))),
                "output_file": f"{Path(image_name).stem}__{noise_type}_{parameter}{strength}__seed{SEED}.png",
                "noisy_image": noisy,
            }
            if noise_type == "salt_and_pepper":
                selected, salt, pepper = salt_and_pepper_counts(original.size, strength)
                row.update(sp_selected=selected, sp_salt=salt, sp_pepper=pepper)
            results.append(row)
            number += 1
    return results


def print_table(results, total_pixels):
    print(f"\nSeed = {SEED} for every experiment. Image size: {total_pixels:,} pixels.")
    header = f"{'ID':<6}{'Noise':<17}{'Strength':<16}{'MSE':>10}{'PSNR (dB)':>11}{'changed px':>12}{'at 0/255':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        strength = f"{r['symbol']} = {r['strength']}"
        print(f"{r['experiment_id']:<6}{DISPLAY_NAMES[r['noise_type']]:<17}{strength:<16}"
              f"{r['mse']:>10.2f}{r['psnr_db']:>11.2f}{r['changed_pixels']:>12,}{r['pixels_at_0_or_255']:>10,}")

    print("\nSalt-and-pepper: selected vs changed pixels")
    for r in results:
        if r["noise_type"] == "salt_and_pepper":
            print(f"  {r['experiment_id']}: amount {r['strength']} x {total_pixels:,} = "
                  f"{r['strength'] * total_pixels:.2f} -> {r['sp_selected']:,} selected "
                  f"({r['sp_salt']:,} salt + {r['sp_pepper']:,} pepper); changed: {r['changed_pixels']:,}")
    print("  (The synthetic image contains no 0 or 255, so here every selected pixel is also changed.)")


def print_theory_check(results, original):
    """Compare measured MSE with the value the noise model predicts WITHOUT clipping.

    Gaussian: E[n^2] = sigma^2.   Speckle: E[(I*n)^2] = variance * mean(I^2).
    (Rounding to whole numbers adds about 1/12 = 0.08, which is negligible here.)
    If the measured MSE is clearly lower, clipping at 0 and 255 removed part of the error.
    """
    mean_square_intensity = float(np.mean(original.astype(np.float64) ** 2))
    print("\nCheck: measured MSE vs. MSE predicted by the noise model without clipping")
    for r in results:
        if r["noise_type"] == "gaussian":
            predicted = float(r["strength"]) ** 2
            formula = f"sigma^2 = {r['strength']}^2"
        elif r["noise_type"] == "speckle":
            predicted = float(r["strength"]) * mean_square_intensity
            formula = f"variance x mean(I^2) = {r['strength']} x {mean_square_intensity:.1f}"
        else:
            continue
        print(f"  {r['experiment_id']}: predicted {predicted:8.2f} ({formula}); measured {r['mse']:8.2f}; "
              f"measured/predicted = {r['mse'] / predicted:.3f}")


def save_outputs(results):
    """Save every noisy image as PNG and the results table as CSV."""
    today = datetime.date.today().isoformat()
    for r in results:
        # Results are regenerated deterministically, so replacing them is allowed.
        save_grayscale_image(r["noisy_image"], NOISY_DIR / r["output_file"], overwrite=True)
    columns = ["experiment_id", "group_id", "date", "image", "height", "width", "noise_type",
               "parameter", "strength", "seed", "filter", "mse", "psnr_db", "changed_pixels",
               "pixels_at_0_or_255", "sp_selected", "sp_salt", "sp_pepper", "output_file"]
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLE_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r, date=today, mse=f"{r['mse']:.4f}", psnr_db=f"{r['psnr_db']:.4f}")
            writer.writerow(row)
    print(f"\nSaved {len(results)} noisy images to: {NOISY_DIR}")
    print(f"Saved results table to: {TABLE_PATH}")


def make_figure(original, results):
    """3 rows (one per noise type) x 4 columns (original + three strengths)."""
    import matplotlib.pyplot as plt

    # "constrained" layout keeps titles from overlapping the images above them.
    figure, axes = plt.subplots(3, 4, figsize=(13, 11), layout="constrained")
    for row_index, (noise_type, _, _, _, _) in enumerate(NOISE_SETTINGS):
        row_results = [r for r in results if r["noise_type"] == noise_type]
        axes[row_index, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 0].set_title("Original (clean)", fontsize=10)
        for column, r in enumerate(row_results, start=1):
            axes[row_index, column].imshow(r["noisy_image"], cmap="gray", vmin=0, vmax=255)
            axes[row_index, column].set_title(
                f"{r['experiment_id']} {DISPLAY_NAMES[noise_type]}, {r['symbol']} = {r['strength']}\n"
                f"MSE = {r['mse']:.1f},  PSNR = {r['psnr_db']:.2f} dB",
                fontsize=9.5,
            )
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(f"Stage 3: noise added to the synthetic test image (seed = {SEED}, no filtering)", fontsize=12)
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add Gaussian, salt-and-pepper and speckle noise and measure it.")
    parser.add_argument("--show", action="store_true", help="display the comparison figure")
    parser.add_argument("--save-figure", action="store_true", help=f"save the figure to {FIGURE_PATH}")
    parser.add_argument("--save-outputs", action="store_true", help="save noisy PNG images and a CSV table in results/")
    args = parser.parse_args(argv)

    image_path = prepare_synthetic_image()
    original = load_grayscale_image(image_path)
    before = original.copy()

    results = run_experiments(original, image_path.name)
    print_table(results, original.size)
    print_theory_check(results, original)

    if not np.array_equal(original, before):
        raise RuntimeError("The original image was modified - this must never happen.")
    print("\nOriginal image unchanged: True")

    if args.save_outputs:
        save_outputs(results)
    if args.show or args.save_figure:
        import matplotlib.pyplot as plt

        figure = make_figure(original, results)
        if args.save_figure:
            FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {FIGURE_PATH}")
        if args.show:
            plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
