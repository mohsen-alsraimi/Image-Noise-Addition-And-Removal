
import numbers

import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle


def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Applies Gaussian Blur to the image.
    """
    filtered_image = cv2.GaussianBlur(
        image.astype(np.float32),
        (kernel_size, kernel_size),
        sigma
    )

    return np.clip(filtered_image, 0, 1)


def apply_median_filter(image, kernel_size=5):
    """
    Applies Median Blur to the image.
    """

    image_uint8 = (
        np.clip(image, 0, 1) * 255
    ).round().astype(np.uint8)

    """
    filtered_image = cv2.medianBlur(
        image.astype(np.float32),
        kernel_size
    )
    """

    filtered_uint8 = cv2.medianBlur(
        image_uint8,
        kernel_size
    )

    filtered_image = filtered_uint8.astype(np.float32) / 255.0

    return filtered_image


def apply_bilateral_filter(image, diameter=9, sigma_color=0.1, sigma_space=75):
    """
    Applies Bilateral Filter to the image.
    """
    filtered_image = cv2.bilateralFilter(
        image.astype(np.float32),
        diameter,
        sigma_color,
        sigma_space
    )

    return np.clip(filtered_image, 0, 1)


def apply_nlm_filter(image, patch_size=5, patch_distance=6, h=None):
    """
    Applies Non-Local Means denoising.

    h: Filtering strength, in the same 0-1 units as the image.
       A supplied value is used exactly.
       If None, h = max(0.1, estimated noise sigma).
    """
    if h is None:
        estimated_sigma = np.mean(estimate_sigma(image, channel_axis=None))
        h = max(0.1, estimated_sigma)
    elif isinstance(h, (bool, np.bool_)) or not isinstance(h, numbers.Real):
        raise TypeError("h must be a positive number or None.")
    elif not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive.")

    filtered_image = denoise_nl_means(
        image,
        h=h,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
        fast_mode=True
    )

    return np.clip(filtered_image, 0, 1)


def _validate_lee_inputs(image, window_size, speckle_variance):
    """
    Rejects inputs the Lee filter cannot process correctly.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array.")

    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("image must be a floating-point array in [0, 1].")

    # NaN makes both comparisons False, so it is rejected here too.
    if not (image.min() >= 0 and image.max() <= 1):
        raise ValueError("image values must be finite and in [0, 1].")

    if isinstance(window_size, (bool, np.bool_)) or not isinstance(
        window_size, numbers.Integral
    ):
        raise TypeError("window_size must be an integer.")

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    if speckle_variance is None:
        return

    if isinstance(speckle_variance, (bool, np.bool_)) or not isinstance(
        speckle_variance, numbers.Real
    ):
        raise TypeError("speckle_variance must be a number or None.")

    if not np.isfinite(speckle_variance) or speckle_variance < 0:
        raise ValueError("speckle_variance must be finite and non-negative.")


def apply_lee_filter(image, window_size=5, speckle_variance=None):
    """
    Applies Lee Filter for multiplicative Speckle noise.

    Noise model: noisy = original * (1 + noise), where the noise has
    zero mean and variance speckle_variance. The weight is the best
    linear estimate for this model:

        signal_variance = max(0, local_variance - v * mean^2) / (1 + v)
        weight = signal_variance / ((1 + v) * signal_variance + v * mean^2)
        filtered = mean + weight * (noisy - mean)

    speckle_variance: Known speckle variance. If None, the speckle model
                      is not applied and the filter acts as a general
                      adaptive smoother for other noise types.
    """
    _validate_lee_inputs(image, window_size, speckle_variance)

    kernel = (window_size, window_size)

    if speckle_variance is None:
        image = image.astype(np.float32)

        mean = cv2.blur(image, kernel)
        mean_square = cv2.blur(image ** 2, kernel)

        local_variance = mean_square - mean ** 2

        noise_variance = np.mean(local_variance)

        weight = local_variance / (local_variance + noise_variance + 1e-10)

        filtered_image = mean + weight * (image - mean)

        return np.clip(filtered_image, 0, 1)

    image = image.astype(np.float64)

    mean = cv2.blur(image, kernel)
    mean_square = cv2.blur(image ** 2, kernel)

    local_variance = np.maximum(mean_square - mean ** 2, 0)
    speckle_power = speckle_variance * mean ** 2

    signal_variance = (
        np.maximum(local_variance - speckle_power, 0)
        / (1 + speckle_variance)
    )
    total_variance = (1 + speckle_variance) * signal_variance + speckle_power

    # Where total variance is zero the window is flat, so use the mean.
    weight = np.divide(
        signal_variance,
        total_variance,
        out=np.zeros_like(signal_variance),
        where=total_variance > 0
    )

    filtered_image = mean + weight * (image - mean)

    return np.clip(filtered_image.astype(np.float32), 0, 1)


def apply_tv_filter(image, weight=0.1):
    """
    Applies Total Variation denoising.

    weight: Controls smoothing strength.
            Higher weight = stronger smoothing.
    """

    image = image.astype(np.float32)

    filtered_image = denoise_tv_chambolle(
        image,
        weight=weight,
        channel_axis=None
    )

    return np.clip(filtered_image, 0, 1)
