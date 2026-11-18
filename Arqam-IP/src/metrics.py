import math

import numpy as np

from src.image_io import validate_grayscale_image

# Largest possible pixel value for 8-bit images (2**8 - 1).
MAX_PIXEL_VALUE = 255


def _validate_image_pair(original, processed):
    """Check that both inputs are valid 8-bit grayscale images of the same size."""
    for name, image in (("original", original), ("processed", processed)):
        try:
            validate_grayscale_image(image)
        except (TypeError, ValueError) as error:
            # Re-raise the same error type, but say which argument was wrong.
            raise type(error)(f"{name} image: {error}") from error

    if original.shape != processed.shape:
        raise ValueError(
            f"Images must have the same dimensions: original {original.shape} "
            f"vs processed {processed.shape}."
        )


def mse(original, processed):
    """Return the Mean Squared Error between two grayscale images.

        MSE = (1 / N) * sum( (original - processed)^2 )

    where N is the number of pixels. 0.0 means the images are identical.
    Larger values mean larger differences. Inputs are not modified.
    """
    _validate_image_pair(original, processed)

    difference = original.astype(np.float64) - processed.astype(np.float64)
    squared_difference = difference ** 2
    return float(np.mean(squared_difference))


def psnr(original, processed):
    
    error = mse(original, processed)  # validates the inputs too
    if error == 0.0:
        return math.inf
    return 10.0 * math.log10(MAX_PIXEL_VALUE ** 2 / error)
