"""Stage 2 demonstration: understanding MSE and PSNR.

Run from the project root folder (IP):

    python -m demos.demo_metrics                 # print results
    python -m demos.demo_metrics --show          # also show the images
    python -m demos.demo_metrics --save-figure   # save figure to results/figures/

What it does:
Part 1  A tiny 2 x 2 example, calculated step by step (check it by hand).
Part 2  Loads the synthetic test image and makes two controlled, modified copies:
          A) 10 single pixels in the flat gray patch changed from 128 to 178,
          B) every pixel made 10 levels brighter.
        Then it prints MSE and PSNR for each (and cv2.PSNR for comparison).
Part 3  Compares the image with an identical copy: MSE = 0, PSNR = infinity.

These modifications are NOT noise models. They are simple, known changes used
only to check and understand the metrics. The original image is never changed.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from demos.demo_image_loading import prepare_synthetic_image
from src.image_io import load_grayscale_image
from src.metrics import MAX_PIXEL_VALUE, mse, psnr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "stage2_metric_validation.png"

# Case A: 10 known pixels inside the flat patch (value 128) of the 256x256 image.
CHANGED_ROW = 185
CHANGED_COLUMNS = list(range(30, 120, 9))  # 30, 39, ..., 111 -> 10 pixels
CHANGE_AMOUNT_A = 50
# Case B: brightness shift applied to every pixel.
BRIGHTNESS_SHIFT_B = 10


def format_psnr(value):
    return "infinity" if math.isinf(value) else f"{value:.4f} dB"


def part1_hand_example():
    print("=" * 70)
    print("PART 1: a 2 x 2 example you can calculate by hand")
    print("=" * 70)
    original = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    processed = np.array([[12, 20], [30, 36]], dtype=np.uint8)
    difference = original.astype(np.float64) - processed.astype(np.float64)

    print("original  =", original.tolist())
    print("processed =", processed.tolist())
    print("difference (original - processed) =", difference.astype(int).tolist())
    print("squared difference                =", (difference ** 2).astype(int).tolist())
    total = float((difference ** 2).sum())
    print(f"sum = {total:.0f},  N = {original.size} pixels,  MSE = {total:.0f} / {original.size} = {mse(original, processed)}")
    print(f"PSNR = 10 * log10({MAX_PIXEL_VALUE}^2 / {mse(original, processed)}) "
          f"= 10 * log10({MAX_PIXEL_VALUE ** 2 / mse(original, processed):.0f}) = {format_psnr(psnr(original, processed))}")
    # Subtracting whole uint8 arrays wraps around SILENTLY (no warning, no error).
    wrapped = original - processed
    print("\nWhy float64? Subtracting the uint8 arrays directly gives:", wrapped.tolist())
    print("  10 - 12 became", int(wrapped[0, 0]), "instead of -2 (wrap-around: -2 + 256 = 254),")
    print("  and NumPy gives no warning. Squaring 254 instead of -2 would ruin the MSE.")


def make_case_a(original):
    modified = original.copy()  # never change the original array
    for column in CHANGED_COLUMNS:
        modified[CHANGED_ROW, column] = int(original[CHANGED_ROW, column]) + CHANGE_AMOUNT_A
    return modified


def make_case_b(original):
    shifted = original.astype(np.int16) + BRIGHTNESS_SHIFT_B  # int16 avoids uint8 wrap-around
    clipped_pixels = int(np.count_nonzero(shifted > MAX_PIXEL_VALUE))
    return np.clip(shifted, 0, MAX_PIXEL_VALUE).astype(np.uint8), clipped_pixels


def part2_controlled_changes(original):
    print("\n" + "=" * 70)
    print("PART 2: controlled changes to the synthetic test image")
    print("=" * 70)
    case_a = make_case_a(original)
    case_b, clipped = make_case_b(original)

    old_values = sorted({int(original[CHANGED_ROW, c]) for c in CHANGED_COLUMNS})
    print(f"Case A: {len(CHANGED_COLUMNS)} pixels in row {CHANGED_ROW}, columns {CHANGED_COLUMNS}")
    print(f"        original value(s) {old_values} -> +{CHANGE_AMOUNT_A}; all other pixels unchanged")
    print(f"Case B: every pixel +{BRIGHTNESS_SHIFT_B} (pixels that had to be clipped at 255: {clipped})")

    rows = []
    print(f"\n{'Case':<34}{'changed px':>11}{'MSE':>12}{'PSNR (ours)':>16}{'cv2.PSNR':>12}")
    for label, modified in (("A) 10 pixels changed by +50", case_a), ("B) all pixels brighter by +10", case_b)):
        changed = int(np.count_nonzero(modified != original))
        error, ours, theirs = mse(original, modified), psnr(original, modified), cv2.PSNR(original, modified)
        rows.append((label, changed, error, ours, theirs))
        print(f"{label:<34}{changed:>11,}{error:>12.4f}{format_psnr(ours):>16}{theirs:>9.4f} dB")

    n = original.size
    print(f"\nCheck by hand: A) MSE = {len(CHANGED_COLUMNS)} x {CHANGE_AMOUNT_A}^2 / {n} = "
          f"{len(CHANGED_COLUMNS) * CHANGE_AMOUNT_A ** 2 / n:.4f};   "
          f"B) MSE = {BRIGHTNESS_SHIFT_B}^2 = {BRIGHTNESS_SHIFT_B ** 2}")
    return case_a, case_b, rows


def part3_identical(original):
    print("\n" + "=" * 70)
    print("PART 3: identical images")
    print("=" * 70)
    copy = original.copy()
    print(f"MSE          = {mse(original, copy)}")
    print(f"PSNR (ours)  = {format_psnr(psnr(original, copy))}")
    print(f"cv2.PSNR     = {cv2.PSNR(original, copy):.4f} dB  "
          "(finite: OpenCV adds a tiny epsilon to avoid dividing by zero)")


def make_figure(original, case_a, case_b, rows):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    panels = [("Original (synthetic)", original, "reference")]
    for (label, _, error, value, _), image in zip(rows, (case_a, case_b)):
        panels.append((label, image, f"MSE = {error:.2f},  PSNR = {format_psnr(value)}"))
    for axis, (title, image, subtitle) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"{title}\n{subtitle}", fontsize=10)
        axis.axis("off")
    figure.suptitle("Stage 2 metric validation (controlled changes, not noise experiments)", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.92))  # leave room at the top for the main title
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Demonstrate MSE and PSNR on controlled image changes.")
    parser.add_argument("--show", action="store_true", help="display the original and modified images")
    parser.add_argument("--save-figure", action="store_true", help=f"save the figure to {FIGURE_PATH}")
    args = parser.parse_args(argv)

    part1_hand_example()

    image_path = prepare_synthetic_image()
    original = load_grayscale_image(image_path)
    before = original.copy()

    case_a, case_b, rows = part2_controlled_changes(original)
    part3_identical(original)

    assert np.array_equal(original, before), "the original image must never change"
    print("\nOriginal image unchanged: True")

    if args.show or args.save_figure:
        import matplotlib.pyplot as plt

        figure = make_figure(original, case_a, case_b, rows)
        if args.save_figure:
            FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {FIGURE_PATH}")
        if args.show:
            plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
