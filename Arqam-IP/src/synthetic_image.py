"""Generate a synthetic grayscale test image with known content.

Why a synthetic image?
* We know exactly what every pixel should be, so it is a perfect "clean"
  reference image for measuring noise and filter quality.
* It needs no download, so there are no source or licensing questions.
* It is deterministic: the same parameters always give the same pixels.

The image is designed to contain the structures that make filters behave
differently:
* a smooth horizontal gradient (slow intensity change),
* flat regions (one constant intensity), where noise is easy to see,
* sharp straight and curved edges (a square and a circle),
* very fine detail: 1-pixel-wide vertical lines.

Intensities deliberately avoid exactly 0 and 255 (the darkest value is 20,
the brightest 240). Later, this makes salt-and-pepper noise pixels (which
are exactly 0 or 255) easy to tell apart from original pixels.
"""

import numpy as np

MIN_SIZE = 64  # smaller images would make the shapes too tiny to be useful

# Intensity values used for each region (0 = black, 255 = white).
GRADIENT_START = 50
GRADIENT_END = 200
DARK_SQUARE_VALUE = 20
BRIGHT_CIRCLE_VALUE = 235
FLAT_PATCH_VALUE = 128
LINE_BACKGROUND_VALUE = 40
LINE_VALUE = 240
LINE_SPACING = 4  # one 1-pixel line every 4 columns


def create_synthetic_test_image(height=256, width=256):
    """Return a deterministic uint8 grayscale test image of shape (height, width)."""
    for name, value in (("height", height), ("width", width)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if value < MIN_SIZE:
            raise ValueError(f"{name} must be at least {MIN_SIZE} pixels, got {value}.")

    # Work in float64 while drawing, convert to uint8 once at the end.
    # 1) Background: a horizontal gradient, the same on every row.
    row = np.linspace(GRADIENT_START, GRADIENT_END, width)
    image = np.tile(row, (height, 1))

    # Helper to convert a fraction of the image size into a pixel index.
    def r(fraction):
        return int(round(fraction * height))

    def c(fraction):
        return int(round(fraction * width))

    # 2) Dark square in the upper-left area (straight, sharp edges).
    image[r(0.10):r(0.42), c(0.08):c(0.40)] = DARK_SQUARE_VALUE

    # 3) Bright circle in the upper-right area (curved edges).
    #    A pixel (y, x) is inside the circle if (y-cy)^2 + (x-cx)^2 <= radius^2.
    cy, cx = r(0.26), c(0.70)
    radius = 0.16 * min(height, width)
    ys, xs = np.ogrid[:height, :width]
    inside_circle = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    image[inside_circle] = BRIGHT_CIRCLE_VALUE

    # 4) Flat mid-gray patch in the lower-left area (constant intensity).
    image[r(0.55):r(0.90), c(0.08):c(0.45)] = FLAT_PATCH_VALUE

    # 5) Fine detail in the lower-right area: thin bright lines on a dark block.
    top, bottom, left, right = r(0.55), r(0.90), c(0.55), c(0.92)
    image[top:bottom, left:right] = LINE_BACKGROUND_VALUE
    image[top:bottom, left:right:LINE_SPACING] = LINE_VALUE

    return np.round(image).astype(np.uint8)
