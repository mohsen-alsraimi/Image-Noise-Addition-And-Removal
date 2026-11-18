"""Loading, validating, describing and saving 8-bit grayscale images.

Key ideas (see docs/report_notes.md, section 3 "Methods"):

* A grayscale image is a 2-D matrix I(x, y). Each element is one pixel.
* In an 8-bit image each pixel is a whole number from 0 (black) to 255 (white).
* In NumPy this is an array with shape (height, width) and dtype uint8.

Why we read files with NumPy + cv2.imdecode instead of cv2.imread:
on Windows, cv2.imread cannot open paths that contain non-ASCII characters
(for example Turkish letters such as "ü" or "İ"). We tested this in our
environment: cv2.imread returned None for such a path. Reading the raw bytes
with numpy.fromfile and decoding them with cv2.imdecode works for any path.
"""

from pathlib import Path

import cv2
import numpy as np

# File types we accept. All are standard formats that OpenCV can decode.
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _to_path(path):
    """Convert a str or Path into a Path, rejecting any other type."""
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Image path must be a str or pathlib.Path, got {type(path).__name__}.")
    return Path(path)


def _check_extension(path):
    """Raise ValueError if the file extension is not a supported image type."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{path.suffix}'. Supported types: {supported}.")


def validate_grayscale_image(image):
    """Check that `image` is a valid 8-bit grayscale image.

    A valid image is a non-empty 2-D NumPy array of type uint8.
    Raises TypeError or ValueError with a clear message otherwise.
    Returns None when the image is valid.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be a NumPy array, got {type(image).__name__}.")
    if image.ndim != 2:
        raise ValueError(
            f"Grayscale image must be 2-D (height, width), got {image.ndim}-D with shape {image.shape}."
        )
    if image.size == 0:
        raise ValueError("Image is empty (it has zero pixels).")
    if image.dtype != np.uint8:
        raise ValueError(f"Image must have dtype uint8 (values 0-255), got {image.dtype}.")


def load_grayscale_image(path):
    """Load an image file and return it as an 8-bit grayscale NumPy array.

    Colour images are converted to grayscale by OpenCV during decoding,
    using the luminance weights Y = 0.299*R + 0.587*G + 0.114*B
    (individual values can differ by 1 because of rounding inside the codec).

    The file on disk is only read, never changed.

    Raises:
        TypeError: path is not a str or Path.
        FileNotFoundError: the file does not exist.
        ValueError: the path is not a file, the type is unsupported,
            the file is empty, or the contents cannot be decoded.
    """
    path = _to_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    _check_extension(path)
    if path.stat().st_size == 0:
        # OpenCV crashes with an assertion error on an empty buffer,
        # so we check for this case ourselves and give a clear message.
        raise ValueError(f"Image file is empty (0 bytes): {path}")

    raw_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image (file may be corrupted): {path}")

    validate_grayscale_image(image)
    return image


def save_grayscale_image(image, path, overwrite=False):
    """Save an 8-bit grayscale image to `path`.

    By default this refuses to replace an existing file. This protects
    original input images from being overwritten by mistake.
    Parent folders are created if needed. Returns the saved Path.
    """
    validate_grayscale_image(image)
    path = _to_path(path)
    _check_extension(path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {path}")

    success, encoded = cv2.imencode(path.suffix.lower(), image)
    if not success:
        raise ValueError(f"OpenCV could not encode the image as '{path.suffix}'.")

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)  # works for non-ASCII paths, like np.fromfile
    return path


def get_image_info(image):
    """Return basic facts and pixel statistics about a grayscale image.

    Statistics are calculated in float64 so that summing many uint8
    values cannot overflow. The input image is not modified.
    """
    validate_grayscale_image(image)
    height, width = image.shape
    pixels = image.astype(np.float64)

    return {
        "height": height,
        "width": width,
        "total_pixels": height * width,
        "dtype": str(image.dtype),
        "bit_depth": image.dtype.itemsize * 8,
        "raw_size_bytes": image.nbytes,  # 1 byte per pixel for uint8
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "unique_values": int(np.unique(image).size),
    }


def format_image_info(info):
    """Turn the dictionary from get_image_info into readable text lines."""
    return "\n".join(
        [
            f"Dimensions       : {info['width']} x {info['height']} pixels (width x height)",
            f"NumPy shape      : ({info['height']}, {info['width']})  -> (rows, columns)",
            f"Total pixels     : {info['total_pixels']:,}",
            f"Data type        : {info['dtype']} ({info['bit_depth']}-bit, values 0-255)",
            f"Raw size         : {info['raw_size_bytes']:,} bytes (1 byte per pixel)",
            f"Min / Max        : {info['min']} / {info['max']}",
            f"Mean intensity   : {info['mean']:.2f}",
            f"Std deviation    : {info['std']:.2f}",
            f"Distinct levels  : {info['unique_values']} of 256 possible",
        ]
    )
