
import numpy as np


def add_gaussian_noise(image, sigma, seed=42):
    """
    Adds Gaussian noise to an image.

    sigma: Standard deviation of the noise.
    """
    rng = np.random.default_rng(seed)

    noise = rng.normal(0, sigma / 255.0, image.shape)

    noisy_image = image + noise

    return np.clip(noisy_image, 0, 1)


def add_salt_pepper_noise(image, amount, seed=42):
    """
    Adds Salt-and-Pepper noise to an image.

    amount: Percentage of pixels affected (0-1).
    """
    rng = np.random.default_rng(seed)

    noisy_image = image.copy()

    random_values = rng.random(image.shape)

    noisy_image[random_values < amount / 2] = 0
    noisy_image[random_values > 1 - amount / 2] = 1

    return noisy_image


def add_speckle_noise(image, variance, seed=42):
    """
    Adds multiplicative Speckle noise to an image.

    variance: Variance of the noise.
    """
    rng = np.random.default_rng(seed)

    noise = rng.normal(0, np.sqrt(variance), image.shape)

    noisy_image = image + image * noise

    return np.clip(noisy_image, 0, 1)
