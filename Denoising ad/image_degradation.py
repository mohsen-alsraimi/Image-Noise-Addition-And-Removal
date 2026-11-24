import cv2
import numpy as np


def add_jpeg_compression(image, quality=60):
    """Compress and decode a grayscale image using JPEG."""
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    success, encoded = cv2.imencode(
        ".jpg",
        image_uint8,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    if not success:
        raise ValueError("JPEG encoding failed.")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return decoded.astype(np.float32) / 255.0


def add_glare(image, alpha=0.45, seed=42):
    """Blend a reproducible elliptical Gaussian light hotspot into an image."""
    height, width = image.shape
    rng = np.random.default_rng(seed)

    # Choose the hotspot location and spread relative to the image dimensions.
    center_x = rng.uniform(0.25, 0.75) * width
    center_y = rng.uniform(0.25, 0.75) * height
    sigma_x = max(width * rng.uniform(0.12, 0.22), 1.0)
    sigma_y = max(height * rng.uniform(0.12, 0.22), 1.0)

    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    mask = np.exp(
        -((x_grid - center_x) ** 2 / (2 * sigma_x ** 2)
          + (y_grid - center_y) ** 2 / (2 * sigma_y ** 2))
    )

    glare_image = (1 - alpha * mask) * image + alpha * mask
    return np.clip(glare_image, 0, 1).astype(np.float32)
