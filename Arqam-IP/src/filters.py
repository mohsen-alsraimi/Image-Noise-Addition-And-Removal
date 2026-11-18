import math
import numbers

import cv2
import numpy as np

from src.image_io import validate_grayscale_image


BORDER_TYPE = cv2.BORDER_REFLECT_101


MEDIAN_BORDER_TYPE = cv2.BORDER_REPLICATE


# --------------------------------------------------------------------------
# Parameter validation helpers
# --------------------------------------------------------------------------
def _validate_odd_size(value, name, minimum=1):
    """Check a window size: an odd integer >= minimum. Booleans are rejected."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    if value % 2 == 0:
        # An odd size gives the window an exact centre pixel (e.g. 3 = 1 + centre + 1).
        examples = ", ".join(str(minimum + 2 * step) for step in range(3))
        raise ValueError(f"{name} must be odd ({examples}, ...), got {value}.")
    return int(value)


def _validate_sigma(value, name, allow_zero):
    """Check a sigma value: a finite real number > 0 (or >= 0 if allow_zero)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number (int or float), got {type(value).__name__}.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if allow_zero and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    if not allow_zero and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return float(value)


# --------------------------------------------------------------------------
# Mean (box) filter
# --------------------------------------------------------------------------
def apply_mean_filter(image, kernel_size=3):
    """Replace every pixel by the average of its kernel_size x kernel_size neighbourhood.

    kernel_size: odd integer >= 1 (3 means a 3x3 window of 9 pixels).
                 1 leaves the image unchanged; larger values smooth more.
    Every pixel in the window has the same weight 1 / kernel_size^2.
    Border: BORDER_REFLECT_101 (see BORDER_TYPE).
    """
    validate_grayscale_image(image)
    kernel_size = _validate_odd_size(kernel_size, "kernel_size")
    return cv2.blur(image, (kernel_size, kernel_size), borderType=BORDER_TYPE)


# --------------------------------------------------------------------------
# Gaussian filter
# --------------------------------------------------------------------------
def apply_gaussian_filter(image, kernel_size=3, sigma=0):
    """Weighted average where the weights follow a bell (Gaussian) curve.

    kernel_size: odd integer >= 1, the window size.
    sigma:       standard deviation of the bell curve in pixels (finite, >= 0).
                 Larger sigma = flatter bell = stronger smoothing.
                 0 means "let OpenCV choose sigma from kernel_size". For
                 kernel_size 3 OpenCV then uses the fixed weights [1/4, 1/2, 1/4]
                 in each direction (checked in the OpenCV source and in our tests).
    The same sigma is used horizontally and vertically.
    Border: BORDER_REFLECT_101 (see BORDER_TYPE).
    """
    validate_grayscale_image(image)
    kernel_size = _validate_odd_size(kernel_size, "kernel_size")
    sigma = _validate_sigma(sigma, "sigma", allow_zero=True)
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,  # passed explicitly, as the OpenCV documentation recommends
        borderType=BORDER_TYPE,
    )


# --------------------------------------------------------------------------
# Median filter
# --------------------------------------------------------------------------
def apply_median_filter(image, kernel_size=3):
    """Replace every pixel by the MIDDLE value of its sorted neighbourhood.

    kernel_size: odd integer >= 1. For uint8 images OpenCV supports every odd
                 size (sizes above 5 work only for uint8, which is all we accept).
                 1 leaves the image unchanged.
    The output only contains values that already exist in the window, so a
    single extreme pixel (salt 255 or pepper 0) is simply dropped, not averaged in.
    Border: always BORDER_REPLICATE inside OpenCV (see MEDIAN_BORDER_TYPE).
    """
    validate_grayscale_image(image)
    kernel_size = _validate_odd_size(kernel_size, "kernel_size")
    return cv2.medianBlur(image, kernel_size)


# --------------------------------------------------------------------------
# Bilateral filter
# --------------------------------------------------------------------------
def apply_bilateral_filter(image, diameter=5, sigma_color=50, sigma_space=50):
    """Edge-preserving smoothing: average pixels that are both NEAR and SIMILAR.

    Each neighbour's weight = closeness weight x similarity weight:
    diameter:    odd integer >= 3, the width of the square neighbourhood in pixels.
                 (OpenCV uses radius = diameter // 2, with at least radius 1, so
                 an even diameter would silently act like the next odd one, and
                 1 or 2 would act like 3. We reject those to avoid confusion.)
    sigma_color: positive number, in intensity levels (0-255 scale). Neighbours
                 whose value differs by much more than sigma_color get almost no
                 weight, which is what keeps edges sharp. Larger = more mixing.
    sigma_space: positive number, in pixels. Controls how quickly the weight
                 falls with distance inside the window. Larger = far neighbours
                 count more.
    OpenCV would silently replace sigma values <= 0 by 1, so we reject them.
    Border: BORDER_REFLECT_101 (see BORDER_TYPE).
    """
    validate_grayscale_image(image)
    diameter = _validate_odd_size(diameter, "diameter", minimum=3)
    sigma_color = _validate_sigma(sigma_color, "sigma_color", allow_zero=False)
    sigma_space = _validate_sigma(sigma_space, "sigma_space", allow_zero=False)
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space, borderType=BORDER_TYPE)
