"""Stage 4 demonstration: remove noise with four filters and compare them fairly.

Run from the project root folder (IP):

    python -m demos.demo_filters                  # print the comparison tables
    python -m demos.demo_filters --show           # also show the comparison figure
    python -m demos.demo_filters --save-figure    # save results/figures/stage4_filter_comparison.png
    python -m demos.demo_filters --save-outputs   # save filtered PNGs + CSV table in results/

What it does, for each of two noise cases:
1. Starts from the clean synthetic test image (the original).
2. Creates ONE noisy image with a fixed seed (42). These are exactly the
   Stage 3 baselines G002 (Gaussian, sigma = 25) and G005 (salt-and-pepper,
   amount = 0.05); the demo checks this against the saved Stage 3 PNG if it exists.
3. Applies all four filters to that SAME noisy image.
4. Measures MSE and PSNR of the noisy image and of every filtered image
   against the clean original, and prints the change in PSNR.

Two extra tables help to explain the numbers:
* MSE by region: the flat patch (no edges: pure noise removal) and the block
  of 1-pixel lines (fine detail), because one region can dominate the
  whole-image MSE.
* Each filter applied to the CLEAN image, to show how much each filter
  changes an image that has no noise at all (the "cost" of filtering).

Nothing in data/input or results/noisy is changed. The original is never modified.
"""

import argparse
import csv
import datetime
import sys
from pathlib import Path

import numpy as np

from demos.demo_image_loading import prepare_synthetic_image
from src.filters import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_mean_filter,
    apply_median_filter,
)
from src.image_io import load_grayscale_image, save_grayscale_image
from src.metrics import mse, psnr
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_PATH = RESULTS_DIR / "figures" / "stage4_filter_comparison.png"
TABLE_PATH = RESULTS_DIR / "tables" / "stage4_filter_results.csv"
FILTERED_DIR = RESULTS_DIR / "filtered"
NOISY_DIR = RESULTS_DIR / "noisy"  # read only: used to compare with the Stage 3 baselines

SEED = 42  # same seed as Stage 3 -> the noisy images are identical to G002 and G005

# One noisy image per case. Each tuple:
# (noise type, parameter, symbol, function, strength, Stage 3 group, Stage 3 baseline ID)
NOISE_CASES = [
    ("gaussian", "sigma", "σ", add_gaussian_noise, 25, "G002", "E002"),
    ("salt_and_pepper", "amount", "amount", add_salt_and_pepper_noise, 0.05, "G005", "E005"),
]

# The four filters and their settings. Each tuple:
# (filter name, function, keyword arguments, short label used in file names)
# * kernel size 3 (3x3 window) is the smallest window that smooths; it is the
#   same size for the mean, Gaussian and median filters so they compare fairly.
# * Gaussian sigma = 0: OpenCV chooses sigma from the kernel size (weights 1/4, 1/2, 1/4).
# * Bilateral d = 5, sigmaColor = sigmaSpace = 50: OpenCV's documentation suggests
#   d = 5 for fast filtering, equal sigmas "for simplicity", and says sigmas
#   below 10 have little effect while above 150 the image looks "cartoonish".
#   50 lies between these limits. These are reasonable starting values, not tuned ones.
FILTER_SETTINGS = [
    ("mean", apply_mean_filter, {"kernel_size": 3}, "k3"),
    ("gaussian", apply_gaussian_filter, {"kernel_size": 3, "sigma": 0}, "k3_sigma0"),
    ("median", apply_median_filter, {"kernel_size": 3}, "k3"),
    ("bilateral", apply_bilateral_filter, {"diameter": 5, "sigma_color": 50, "sigma_space": 50},
     "d5_sc50_ss50"),
]
FIRST_EXPERIMENT_NUMBER = 10  # E001-E009 are the Stage 3 noise baselines

NOISE_NAMES = {"gaussian": "Gaussian", "salt_and_pepper": "Salt-and-pepper"}
FILTER_NAMES = {"mean": "Mean", "gaussian": "Gaussian", "median": "Median", "bilateral": "Bilateral"}


def describe_parameters(kwargs):
    return ", ".join(f"{name}={value}" for name, value in kwargs.items())


def synthetic_regions(original):
    """Return two rectangular regions of the synthetic image as (row slice, column slice).

    * "flat patch": constant 128, no edges -> shows pure noise removal.
    * "line block": 1-pixel lines (240) on a dark block (40) -> shows loss of fine detail.
    The fractions are the ones used in src/synthetic_image.py. A check below
    makes sure they still match the image, so they cannot silently drift apart.
    """
    height, width = original.shape

    def r(fraction):
        return int(round(fraction * height))

    def c(fraction):
        return int(round(fraction * width))

    regions = {
        "flat patch": (slice(r(0.55), r(0.90)), slice(c(0.08), c(0.45))),
        "line block": (slice(r(0.55), r(0.90)), slice(c(0.55), c(0.92))),
    }
    if set(np.unique(original[regions["flat patch"]])) != {128} or \
            set(np.unique(original[regions["line block"]])) != {40, 240}:
        raise RuntimeError("Region positions do not match the synthetic image.")
    return regions


def region_mse(original, processed, region):
    """MSE inside one rectangular region only (reuses the project's mse function)."""
    return mse(original[region], processed[region])


def make_noisy_images(original, image_name):
    """Create the two noisy images and compare them with the saved Stage 3 baselines."""
    cases = []
    for noise_type, parameter, symbol, function, strength, group, baseline_id in NOISE_CASES:
        noisy = function(original, strength, seed=SEED)
        stem = f"{Path(image_name).stem}__{noise_type}_{parameter}{strength}__seed{SEED}"
        saved_path = NOISY_DIR / f"{stem}.png"
        if saved_path.is_file():
            same = np.array_equal(load_grayscale_image(saved_path), noisy)
            check = f"identical to saved Stage 3 file {saved_path.name}: {same}"
        else:
            check = "Stage 3 file not found (run: python -m demos.demo_noise --save-outputs)"
        cases.append({
            "noise_type": noise_type, "parameter": parameter, "symbol": symbol,
            "strength": strength, "group_id": group, "baseline_id": baseline_id,
            "stem": stem, "noisy_image": noisy, "baseline_check": check,
            "noisy_mse": mse(original, noisy), "noisy_psnr": psnr(original, noisy),
        })
    return cases


def run_filters(original, cases, image_name):
    """Apply every filter to the same noisy image of each case. Returns result dicts."""
    results = []
    number = FIRST_EXPERIMENT_NUMBER
    for case in cases:
        noisy = case["noisy_image"]
        noisy_before = noisy.copy()
        for filter_name, function, kwargs, label in FILTER_SETTINGS:
            filtered = function(noisy, **kwargs)
            filtered_psnr = psnr(original, filtered)
            results.append({
                "experiment_id": f"E{number:03d}",
                "group_id": case["group_id"],
                "baseline_id": case["baseline_id"],
                "image": image_name,
                "noise_type": case["noise_type"],
                "parameter": case["parameter"],
                "symbol": case["symbol"],
                "strength": case["strength"],
                "seed": SEED,
                "filter": filter_name,
                "filter_parameters": describe_parameters(kwargs),
                "mse": mse(original, filtered),
                "psnr_db": filtered_psnr,
                "noisy_mse": case["noisy_mse"],
                "noisy_psnr_db": case["noisy_psnr"],
                "psnr_change_db": filtered_psnr - case["noisy_psnr"],
                "output_file": f"{case['stem']}__{filter_name}_{label}.png",
                "filtered_image": filtered,
            })
            number += 1
        # Fairness check: every filter must have received exactly the same noisy image.
        if not np.array_equal(noisy, noisy_before):
            raise RuntimeError("A filter modified the noisy image - the comparison would be unfair.")
    return results


def print_tables(cases, results):
    print(f"\nSeed = {SEED}. Every filter in a group is applied to the SAME noisy image.")
    for case in cases:
        print(f"  {case['group_id']} ({NOISE_NAMES[case['noise_type']]}, {case['symbol']} = {case['strength']}): "
              f"{case['baseline_check']}")

    header = (f"{'ID':<6}{'Group':<7}{'Noise':<17}{'Filter':<11}{'Parameters':<40}"
              f"{'MSE':>10}{'PSNR (dB)':>11}{'vs noisy':>10}")
    for case in cases:
        print(f"\n{NOISE_NAMES[case['noise_type']]} noise, {case['symbol']} = {case['strength']}")
        print(header)
        print("-" * len(header))
        print(f"{case['baseline_id']:<6}{case['group_id']:<7}{NOISE_NAMES[case['noise_type']]:<17}"
              f"{'none':<11}{'(noisy baseline from Stage 3)':<40}"
              f"{case['noisy_mse']:>10.2f}{case['noisy_psnr']:>11.2f}{'':>10}")
        for r in results:
            if r["group_id"] == case["group_id"]:
                print(f"{r['experiment_id']:<6}{r['group_id']:<7}{NOISE_NAMES[r['noise_type']]:<17}"
                      f"{FILTER_NAMES[r['filter']]:<11}{r['filter_parameters']:<40}"
                      f"{r['mse']:>10.2f}{r['psnr_db']:>11.2f}{r['psnr_change_db']:>+10.2f}")
    print("\n'vs noisy' = PSNR(filtered) - PSNR(noisy). Positive: the filter brought the image")
    print("closer to the original. Negative: the filtered image is further away than the noisy one.")


def print_region_tables(original, cases, results, regions):
    """Where does the error come from? MSE inside the flat patch and inside the line block."""
    flat, lines = regions["flat patch"], regions["line block"]
    flat_pixels = original[flat].size
    line_pixels = original[lines].size
    print("\nMSE by region (same images as above). Flat patch = pure noise removal (no edges);")
    print("line block = fine detail (1-pixel lines). Lower is better.")
    for case in cases:
        print(f"\n{NOISE_NAMES[case['noise_type']]}, {case['symbol']} = {case['strength']}   "
              f"(flat patch {flat_pixels:,} px, line block {line_pixels:,} px, whole image {original.size:,} px)")
        print(f"{'ID':<6}{'Filter':<11}{'flat patch':>12}{'line block':>12}{'whole image':>13}")
        noisy = case["noisy_image"]
        print(f"{case['baseline_id']:<6}{'none':<11}{region_mse(original, noisy, flat):>12.2f}"
              f"{region_mse(original, noisy, lines):>12.2f}{case['noisy_mse']:>13.2f}")
        for r in results:
            if r["group_id"] == case["group_id"]:
                image = r["filtered_image"]
                print(f"{r['experiment_id']:<6}{FILTER_NAMES[r['filter']]:<11}"
                      f"{region_mse(original, image, flat):>12.2f}{region_mse(original, image, lines):>12.2f}"
                      f"{r['mse']:>13.2f}")


def clean_image_check(original, regions):
    """Apply each filter to the CLEAN image: how much does filtering alone change it?"""
    rows = []
    for number, (filter_name, function, kwargs, _) in enumerate(FILTER_SETTINGS, start=1):
        filtered = function(original, **kwargs)
        rows.append({
            "check_id": f"FC{number:03d}",
            "filter": filter_name,
            "filter_parameters": describe_parameters(kwargs),
            "mse": mse(original, filtered),
            "psnr_db": psnr(original, filtered),
            "changed_pixels": int(np.count_nonzero(filtered != original)),
            "line_block_mse": region_mse(original, filtered, regions["line block"]),
        })
    print("\nFilter applied to the CLEAN image (no noise): the change caused by filtering alone")
    print(f"{'ID':<7}{'Filter':<11}{'Parameters':<40}{'MSE':>9}{'PSNR (dB)':>11}{'changed px':>12}"
          f"{'line block MSE':>16}")
    for row in rows:
        print(f"{row['check_id']:<7}{FILTER_NAMES[row['filter']]:<11}{row['filter_parameters']:<40}"
              f"{row['mse']:>9.2f}{row['psnr_db']:>11.2f}{row['changed_pixels']:>12,}"
              f"{row['line_block_mse']:>16.2f}")
    return rows


def save_outputs(results):
    """Save every filtered image as PNG and the results table as CSV.

    Only Stage 4 files are written (results/filtered and a new CSV);
    the Stage 3 noisy images and CSV are never touched.
    """
    today = datetime.date.today().isoformat()
    for r in results:
        # Results are regenerated deterministically, so replacing them is allowed.
        save_grayscale_image(r["filtered_image"], FILTERED_DIR / r["output_file"], overwrite=True)
    columns = ["experiment_id", "group_id", "baseline_id", "date", "image", "noise_type", "parameter",
               "strength", "seed", "filter", "filter_parameters", "mse", "psnr_db", "noisy_mse",
               "noisy_psnr_db", "psnr_change_db", "output_file"]
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLE_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(dict(
                r, date=today, mse=f"{r['mse']:.4f}", psnr_db=f"{r['psnr_db']:.4f}",
                noisy_mse=f"{r['noisy_mse']:.4f}", noisy_psnr_db=f"{r['noisy_psnr_db']:.4f}",
                psnr_change_db=f"{r['psnr_change_db']:.4f}",
            ))
    print(f"\nSaved {len(results)} filtered images to: {FILTERED_DIR}")
    print(f"Saved results table to: {TABLE_PATH}")


def make_figure(original, cases, results):
    """One row per noise case: original, noisy, and the four filtered images."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(cases), 2 + len(FILTER_SETTINGS), figsize=(19, 7.4),
                                layout="constrained")
    for row, case in enumerate(cases):
        panels = [("Original (clean)", original),
                  (f"{case['baseline_id']} Noisy: {NOISE_NAMES[case['noise_type']]}, "
                   f"{case['symbol']} = {case['strength']}\n"
                   f"MSE = {case['noisy_mse']:.1f}, PSNR = {case['noisy_psnr']:.2f} dB", case["noisy_image"])]
        for r in results:
            if r["group_id"] == case["group_id"]:
                panels.append((f"{r['experiment_id']} {FILTER_NAMES[r['filter']]} ({r['filter_parameters']})\n"
                               f"MSE = {r['mse']:.1f}, PSNR = {r['psnr_db']:.2f} dB ({r['psnr_change_db']:+.2f})",
                               r["filtered_image"]))
        for column, (title, image) in enumerate(panels):
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row, column].set_title(title, fontsize=8)
            axes[row, column].axis("off")
    figure.suptitle(f"Stage 4: four filters applied to the same noisy image (seed = {SEED}); "
                    "PSNR change vs. the noisy image in brackets", fontsize=11)
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare mean, Gaussian, median and bilateral filters.")
    parser.add_argument("--show", action="store_true", help="display the comparison figure")
    parser.add_argument("--save-figure", action="store_true", help=f"save the figure to {FIGURE_PATH}")
    parser.add_argument("--save-outputs", action="store_true",
                        help="save filtered PNG images and a CSV table in results/")
    args = parser.parse_args(argv)

    image_path = prepare_synthetic_image()
    original = load_grayscale_image(image_path)
    before = original.copy()

    cases = make_noisy_images(original, image_path.name)
    results = run_filters(original, cases, image_path.name)
    print_tables(cases, results)
    regions = synthetic_regions(original)
    print_region_tables(original, cases, results, regions)
    clean_image_check(original, regions)

    if not np.array_equal(original, before):
        raise RuntimeError("The original image was modified - this must never happen.")
    print("\nOriginal image unchanged: True")

    if args.save_outputs:
        save_outputs(results)
    if args.show or args.save_figure:
        import matplotlib.pyplot as plt

        figure = make_figure(original, cases, results)
        if args.save_figure:
            FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {FIGURE_PATH}")
        if args.show:
            plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
