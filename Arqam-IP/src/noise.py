import math
import numbers

import numpy as np

from src.image_io import validate_grayscale_image

SALT_VALUE = 255  # white
PEPPER_VALUE = 0  # black
MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255


# --------------------------------------------------------------------------
# Input validation helpers
# --------------------------------------------------------------------------
def _validate_strength(value, name, maximum=None):
    """Check a noise-strength parameter: a finite real number >= 0 (and <= maximum)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number (int or float), got {type(value).__name__}.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}.")
    return float(value)


def _validate_seed(seed):
    """Check the seed: None, or a non-negative integer (booleans are rejected)."""
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be None or a non-negative integer, got {type(seed).__name__}.")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")
    return int(seed)


def _to_uint8(values):
    """Round float values to whole numbers, clip to [0, 255], convert to uint8."""
    return np.clip(np.rint(values), MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8)


# --------------------------------------------------------------------------
# Gaussian noise
# --------------------------------------------------------------------------
def add_gaussian_noise(image, sigma, seed=None):

    validate_grayscale_image(image)
    sigma = _validate_strength(sigma, "sigma")
    seed = _validate_seed(seed)
    if sigma == 0:
        return image.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    return _to_uint8(image.astype(np.float64) + noise)


# --------------------------------------------------------------------------
# Salt-and-pepper noise
# --------------------------------------------------------------------------
def salt_and_pepper_counts(total_pixels, amount):

    if isinstance(total_pixels, (bool, np.bool_)) or not isinstance(total_pixels, numbers.Integral):
        raise TypeError("total_pixels must be an integer.")
    if total_pixels <= 0:
        raise ValueError(f"total_pixels must be positive, got {total_pixels}.")
    amount = _validate_strength(amount, "amount", maximum=1.0)

    selected = min(int(math.floor(amount * total_pixels + 0.5)), int(total_pixels))
    salt = selected // 2
    pepper = selected - salt
    return selected, salt, pepper


def add_salt_and_pepper_noise(image, amount, seed=None):
    validate_grayscale_image(image)
    selected, salt, _ = salt_and_pepper_counts(image.size, amount)
    seed = _validate_seed(seed)
    noisy = image.copy()
    if selected == 0:
        return noisy

    rng = np.random.default_rng(seed)
    # Choose `selected` different positions out of 0 .. (number of pixels - 1).
    # These "flat" positions count pixels row by row; unravel_index turns
    # each one back into a (row, column) pair.
    flat_positions = rng.choice(image.size, size=selected, replace=False)
    rows, columns = np.unravel_index(flat_positions, image.shape)
    # The positions are in random order, so the first `salt` of them are a
    # random half: they become salt, and the rest become pepper.
    noisy[rows[:salt], columns[:salt]] = SALT_VALUE
    noisy[rows[salt:], columns[salt:]] = PEPPER_VALUE
    return noisy


# --------------------------------------------------------------------------
# Speckle noise
# --------------------------------------------------------------------------
def add_speckle_noise(image, variance, seed=None):
    
    validate_grayscale_image(image)
    variance = _validate_strength(variance, "variance")
    seed = _validate_seed(seed)
    if variance == 0:
        return image.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=math.sqrt(variance), size=image.shape)
    original = image.astype(np.float64)
    return _to_uint8(original + original * noise)
