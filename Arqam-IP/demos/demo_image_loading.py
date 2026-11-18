"""Stage 1 demonstration: load a grayscale image and show its pixel information.

Run from the project root folder (IP):

    python -m demos.demo_image_loading                      # synthetic test image
    python -m demos.demo_image_loading data/input/photo.jpg # your own image
    python -m demos.demo_image_loading --show               # also open a window

What it does:
1. If no image path is given, it generates the synthetic test image and saves
   it to data/input/synthetic_test_image.png (only if that file does not exist).
2. Loads the image from disk as 8-bit grayscale.
3. Prints dimensions and pixel statistics.
4. Prints a small 5 x 5 block of raw pixel values, showing that an image
   really is a matrix of numbers I(x, y).
5. With --show, displays the image and its intensity histogram.

The input file is only read. It is never modified.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.image_io import format_image_info, get_image_info, load_grayscale_image, save_grayscale_image
from src.synthetic_image import create_synthetic_test_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_IMAGE_PATH = PROJECT_ROOT / "data" / "input" / "synthetic_test_image.png"
PATCH_SIZE = 5


def prepare_synthetic_image():
    """Create the synthetic test image on disk if it does not exist yet."""
    if SYNTHETIC_IMAGE_PATH.exists():
        print(f"Using existing synthetic image: {SYNTHETIC_IMAGE_PATH}")
    else:
        save_grayscale_image(create_synthetic_test_image(), SYNTHETIC_IMAGE_PATH)
        print(f"Generated synthetic image and saved it to: {SYNTHETIC_IMAGE_PATH}")
    return SYNTHETIC_IMAGE_PATH


def print_pixel_patch(image, top, left):
    """Print a PATCH_SIZE x PATCH_SIZE block of pixel values starting at (top, left)."""
    patch = image[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
    print(f"\n{PATCH_SIZE} x {PATCH_SIZE} pixel values starting at row {top}, column {left}:")
    print("        " + " ".join(f"c{left + j:<4}" for j in range(patch.shape[1])))
    for i, row in enumerate(patch):
        print(f"r{top + i:<5}  " + " ".join(f"{value:5d}" for value in row))


def show_image_and_histogram(image, title):
    """Display the image and a histogram of its 256 intensity levels."""
    import matplotlib.pyplot as plt  # imported here so the text demo works without a display

    figure, (ax_image, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_image.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax_image.set_title(title)
    ax_image.axis("off")

    counts = np.bincount(image.ravel(), minlength=256)
    ax_hist.bar(np.arange(256), counts, width=1.0, color="0.3")
    ax_hist.set_title("Intensity histogram")
    ax_hist.set_xlabel("Pixel intensity (0 = black, 255 = white)")
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_xlim(0, 255)
    figure.tight_layout()
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load a grayscale image and print pixel information.")
    parser.add_argument("image", nargs="?", help="path to an image file (default: synthetic test image)")
    parser.add_argument("--show", action="store_true", help="display the image and its histogram")
    args = parser.parse_args(argv)

    image_path = Path(args.image) if args.image else prepare_synthetic_image()

    try:
        image = load_grayscale_image(image_path)
    except (FileNotFoundError, ValueError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"\nLoaded: {image_path}")
    print(format_image_info(get_image_info(image)))

    # Show a block from the middle of the image.
    height, width = image.shape
    print_pixel_patch(image, top=(height - PATCH_SIZE) // 2, left=(width - PATCH_SIZE) // 2)

    if args.show:
        show_image_and_histogram(image, title=image_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
