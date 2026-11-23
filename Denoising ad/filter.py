
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
    filtered_image = cv2.medianBlur(
        image.astype(np.float32),
        kernel_size
    )

    return np.clip(filtered_image, 0, 1)


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


def apply_nlm_filter(image, patch_size=5, patch_distance=6, h=0.1):
    """
    Applies Non-Local Means denoising.
    """
    estimated_sigma = np.mean(estimate_sigma(image, channel_axis=None))

    filtered_image = denoise_nl_means(
        image,
        h=max(h, estimated_sigma),
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
        fast_mode=True
    )

    return np.clip(filtered_image, 0, 1)


def apply_lee_filter(image, window_size=5):
    """
    Applies Lee Filter for multiplicative Speckle noise.
    """
    image = image.astype(np.float32)

    mean = cv2.blur(image, (window_size, window_size))
    mean_square = cv2.blur(image ** 2, (window_size, window_size))

    local_variance = mean_square - mean ** 2

    noise_variance = np.mean(local_variance)

    weight = local_variance / (local_variance + noise_variance + 1e-10)

    filtered_image = mean + weight * (image - mean)

    return np.clip(filtered_image, 0, 1)


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
