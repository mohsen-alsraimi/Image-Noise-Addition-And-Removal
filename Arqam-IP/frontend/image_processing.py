"""Small helpers that connect the Streamlit app (frontend/app.py) to src/.

This module does NOT import Streamlit, so it can be tested with unittest
without starting a web server. It also contains NO image-processing
algorithms of its own:

* Decoding uses OpenCV exactly like src.image_io.load_grayscale_image
  (cv2.imdecode with cv2.IMREAD_GRAYSCALE), but works on the uploaded bytes
  in memory instead of on a file path. Nothing is written to disk.
* Validation of the result uses src.image_io.validate_grayscale_image.

* Noise is added by calling the existing functions in src/noise.py
  (Stage 5B). This module only chooses WHICH function to call.
* Filters are applied by calling the existing functions in src/filters.py
  (Stage 5C), in the same way.
* MSE and PSNR come from the existing functions in src/metrics.py
  (Stage 5D). The original grayscale image is always the reference.
"""

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePath

import cv2
import numpy as np

from src.filters import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_mean_filter,
    apply_median_filter,
)
from src.image_io import validate_grayscale_image
from src.metrics import mse, psnr
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise, add_speckle_noise

# --------------------------------------------------------------------------
# Upload rules
# --------------------------------------------------------------------------
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")
UPLOAD_TYPES = ("png", "jpg", "jpeg")  # the same list, in the form st.file_uploader wants

# Limits that keep the app responsive and memory use reasonable.
# 20 MB is far more than a normal phone photo (a 12-megapixel JPEG is usually 2-6 MB).
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
# 25 megapixels covers typical phone (12 MP) and camera (24 MP) photos.
MAX_PIXELS = 25_000_000
# Above 4 megapixels (about 2300 x 1700) the app shows a "may be slow" note.
LARGE_IMAGE_PIXELS = 4_000_000

# The first bytes of a file tell us its real format, whatever its name says.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
# In a PNG, byte 25 (inside the IHDR header) is the "colour type".
# Types 4 (gray + alpha) and 6 (RGB + alpha) contain transparency.
PNG_COLOUR_TYPE_OFFSET = 25
PNG_ALPHA_COLOUR_TYPES = (4, 6)


@dataclass(frozen=True)
class UploadedImage:
    """Everything the app needs to know about one uploaded image.

    file_bytes  : the uploaded file, byte for byte (never changed).
    file_format : "PNG" or "JPEG", detected from the file contents.
    preview_rgb : the image as decoded in colour (height x width x 3, RGB),
                  used ONLY for display. Read-only.
    grayscale   : the clean grayscale reference (height x width, uint8)
                  that all processing will use. Read-only.
    is_colour   : True if the red, green and blue channels are not all equal.
    has_alpha   : True if a PNG stores transparency (it is ignored).
    """

    file_name: str
    file_bytes: bytes
    file_format: str
    preview_rgb: np.ndarray
    grayscale: np.ndarray
    is_colour: bool
    has_alpha: bool

    @property
    def height(self):
        return self.grayscale.shape[0]

    @property
    def width(self):
        return self.grayscale.shape[1]

    @property
    def pixel_count(self):
        return self.grayscale.size

    @property
    def file_size(self):
        return len(self.file_bytes)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_file_name(file_name):
    """Check the file name and return its extension in lower case (e.g. '.jpg')."""
    if not isinstance(file_name, str):
        raise TypeError(f"File name must be a string, got {type(file_name).__name__}.")
    if not file_name.strip():
        raise ValueError("File name is empty.")
    extension = PurePath(file_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        shown = extension if extension else "no extension"
        raise ValueError(f"Unsupported file type ({shown}). Please upload one of: {allowed}.")
    return extension


def detect_image_format(data):
    """Return "PNG" or "JPEG" by looking at the first bytes of the file.

    The file name can be wrong, so the contents decide. Any other format
    (for example BMP or GIF, or a text file renamed to .png) is rejected.
    """
    if data.startswith(PNG_SIGNATURE):
        return "PNG"
    if data.startswith(JPEG_SIGNATURE):
        return "JPEG"
    raise ValueError("The file is not a PNG or JPEG image (its contents do not match either format).")


def png_has_alpha(data):
    """True if a PNG file's header says it stores transparency (colour type 4 or 6)."""
    return (
        data.startswith(PNG_SIGNATURE)
        and len(data) > PNG_COLOUR_TYPE_OFFSET
        and data[PNG_COLOUR_TYPE_OFFSET] in PNG_ALPHA_COLOUR_TYPES
    )


def _decode(data, flag):
    """Decode image bytes with OpenCV; return None if they cannot be decoded."""
    try:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
    except cv2.error:
        return None


def _read_only(array):
    """Mark an array as read-only, so no later step can change it by accident."""
    array.setflags(write=False)
    return array


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
def load_uploaded_image(file_name, data, max_file_bytes=MAX_FILE_SIZE_BYTES, max_pixels=MAX_PIXELS):
    """Check and decode an uploaded PNG/JPEG file, entirely in memory.

    Steps:
      1. Check the file name (.png, .jpg or .jpeg) and the data type.
      2. Reject empty files and files larger than `max_file_bytes`.
      3. Check that the contents really are PNG or JPEG.
      4. Decode to grayscale with cv2.IMREAD_GRAYSCALE, exactly like
         src.image_io.load_grayscale_image (Stage 1), and validate the
         result with validate_grayscale_image (2-D, uint8, not empty).
      5. Reject images with more than `max_pixels` pixels.
      6. Decode again in colour, only for displaying the original.

    Both decodes apply the EXIF orientation of phone photos, so the colour
    preview and the grayscale image always have the same shape.

    Returns an UploadedImage. Raises TypeError or ValueError with a message
    that can be shown to the user.
    """
    check_file_name(file_name)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"File contents must be bytes, got {type(data).__name__}.")
    data = bytes(data)  # an unchangeable copy of exactly what was uploaded

    if len(data) == 0:
        raise ValueError("The file is empty (0 bytes).")
    if len(data) > max_file_bytes:
        raise ValueError(
            f"The file is too large ({format_file_size(len(data))}). "
            f"The limit is {format_file_size(max_file_bytes)}."
        )

    file_format = detect_image_format(data)

    grayscale = _decode(data, cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise ValueError(f"The {file_format} file could not be decoded. It may be damaged or incomplete.")
    validate_grayscale_image(grayscale)

    if grayscale.size > max_pixels:
        height, width = grayscale.shape
        raise ValueError(
            f"The image is too large ({width} x {height} = {grayscale.size:,} pixels). "
            f"The limit is {max_pixels:,} pixels. Please resize it and upload it again."
        )

    preview_bgr = _decode(data, cv2.IMREAD_COLOR)
    if preview_bgr is None or preview_bgr.shape[:2] != grayscale.shape:
        raise ValueError("The colour version of the image could not be decoded consistently.")

    # OpenCV stores colour as Blue-Green-Red; browsers expect Red-Green-Blue.
    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    blue, green, red = cv2.split(preview_bgr)
    is_colour = not (np.array_equal(blue, green) and np.array_equal(green, red))

    return UploadedImage(
        file_name=file_name,
        file_bytes=data,
        file_format=file_format,
        preview_rgb=_read_only(preview_rgb),
        grayscale=_read_only(grayscale),
        is_colour=is_colour,
        has_alpha=png_has_alpha(data),
    )


# --------------------------------------------------------------------------
# Noise (Stage 5B): choose and call the existing functions in src/noise.py
# --------------------------------------------------------------------------
DEFAULT_SEED = 42  # the same seed as every Stage 3 and Stage 4 experiment


@dataclass(frozen=True)
class NoiseOption:
    """One noise type: which src/noise.py function to call, and its slider settings.

    The defaults are the Stage 3 baselines (E002, E005, E008). The slider
    ranges cover the Stage 3 values (and more) and stay inside the
    rules of src/noise.py.
    """

    label: str
    function: Callable
    parameter: str  # the name of the strength argument in src/noise.py
    default: float
    minimum: float
    maximum: float
    step: float
    explanation: str


NOISE_OPTIONS = {
    "gaussian": NoiseOption(
        label="Gaussian Noise",
        function=add_gaussian_noise,
        parameter="sigma",
        default=25,
        minimum=0,
        maximum=100,
        step=1,
        explanation=(
            "Gaussian noise adds random brightness changes to pixels. "
            "A larger sigma usually creates stronger noise."
        ),
    ),
    "salt_and_pepper": NoiseOption(
        label="Salt-and-Pepper Noise",
        function=add_salt_and_pepper_noise,
        parameter="amount",
        default=0.05,
        minimum=0.0,
        maximum=0.5,
        step=0.01,
        explanation=(
            "Salt-and-Pepper noise randomly changes some pixels to black or white. "
            "A larger amount affects more pixels."
        ),
    ),
    "speckle": NoiseOption(
        label="Speckle Noise",
        function=add_speckle_noise,
        parameter="variance",
        default=0.04,
        minimum=0.0,
        maximum=0.25,
        step=0.01,
        explanation=(
            "Speckle noise changes pixel brightness by an amount related to the original "
            "brightness. Larger variance usually produces stronger noise."
        ),
    ),
}
NOISE_TYPES = tuple(NOISE_OPTIONS)  # ("gaussian", "salt_and_pepper", "speckle")


def get_noise_option(noise_type):
    """Return the NoiseOption for a noise type name, or raise a clear error."""
    if not isinstance(noise_type, str):
        raise TypeError(f"noise_type must be a string, got {type(noise_type).__name__}.")
    if noise_type not in NOISE_OPTIONS:
        raise ValueError(f"Unknown noise type '{noise_type}'. Choose one of: {', '.join(NOISE_TYPES)}.")
    return NOISE_OPTIONS[noise_type]


def make_noisy_image(reference_image, noise_type, strength, seed):
    """Add noise to a copy of the clean reference image and return the noisy image.

    This only selects and calls the existing function from src/noise.py:
        "gaussian"        -> add_gaussian_noise(image, sigma, seed)
        "salt_and_pepper" -> add_salt_and_pepper_noise(image, amount, seed)
        "speckle"         -> add_speckle_noise(image, variance, seed)
    The strength and seed are passed on unchanged, and src/noise.py checks them
    (for example: no negative values, amount at most 1, seed a whole number >= 0).

    One extra rule for the app: a seed is required (None is refused), because
    without a seed the noise would change every time the page reruns.

    The reference image is never changed (src/noise.py always works on a copy).
    The result is a new uint8 image with the same shape, marked read-only so the
    same noisy image can later be given to every filter without being changed.
    """
    option = get_noise_option(noise_type)
    if seed is None:
        raise ValueError("A seed is required, so that the same settings always give the same noise.")
    noisy = option.function(reference_image, strength, seed=seed)
    return _read_only(noisy)


def describe_noise(noise_type, strength):
    """A short label for the noisy image, e.g. 'Gaussian Noise — σ = 25'."""
    option = get_noise_option(noise_type)
    if noise_type == "gaussian":
        return f"{option.label} — σ = {strength:g}"
    if noise_type == "salt_and_pepper":
        return f"{option.label} — {strength * 100:g}% of pixels"
    return f"{option.label} — variance = {strength:g}"


# --------------------------------------------------------------------------
# Filters (Stage 5C): choose and call the existing functions in src/filters.py
# --------------------------------------------------------------------------
# Window sizes offered in the app. src/filters.py accepts every odd size >= 1
# (>= 3 for the bilateral diameter). Size 1 would leave the image unchanged,
# so the app starts at 3. Even 15 takes well under a second on a
# 12-megapixel photo (measured in Stage 5C).
KERNEL_SIZES = (3, 5, 7, 9, 11, 13, 15)
BILATERAL_DIAMETERS = (3, 5, 7, 9, 11, 13, 15)
# Slider ranges (minimum, maximum, step) for the filter sigmas.
GAUSSIAN_FILTER_SIGMA_RANGE = (0.0, 5.0, 0.1)  # 0 = OpenCV chooses from the kernel size
BILATERAL_SIGMA_RANGE = (1, 200, 1)  # src/filters.py needs > 0


@dataclass(frozen=True)
class FilterOption:
    """One filter: which src/filters.py function to call, its parameters and texts.

    `defaults` are the Stage 4 demo settings (decision D28): kernel 3,
    Gaussian sigma 0, bilateral 5 / 50 / 50. Its keys are exactly the
    parameter names of the src/filters.py function.
    """

    label: str
    function: Callable
    defaults: dict
    explanation: str
    benefit: str
    limitation: str


FILTER_OPTIONS = {
    "mean": FilterOption(
        label="Mean Filter",
        function=apply_mean_filter,
        defaults={"kernel_size": 3},
        explanation=(
            "The Mean filter replaces each pixel with the average of nearby pixels. "
            "It can reduce random noise but may blur edges and fine details."
        ),
        benefit="Simple smoothing: random ups and downs are averaged out.",
        limitation="Blurs edges and fine details. A black or white pixel is spread into a gray blob, not removed.",
    ),
    "gaussian": FilterOption(
        label="Gaussian Filter",
        function=apply_gaussian_filter,
        defaults={"kernel_size": 3, "sigma": 0.0},
        explanation=(
            "The Gaussian filter uses a weighted average of nearby pixels, giving more "
            "importance to pixels closer to the center."
        ),
        benefit="Reduces random brightness variations, a little more gently than the Mean filter.",
        limitation="Still softens edges and fine details.",
    ),
    "median": FilterOption(
        label="Median Filter",
        function=apply_median_filter,
        defaults={"kernel_size": 3},
        explanation=(
            "The Median filter replaces a pixel with the middle value of its neighborhood. "
            "It is especially useful for removing isolated black and white pixels."
        ),
        benefit="Often very effective against Salt-and-Pepper noise, while keeping strong edges.",
        limitation="Can remove thin lines and small features, because they are a minority in the window.",
    ),
    "bilateral": FilterOption(
        label="Bilateral Filter",
        function=apply_bilateral_filter,
        defaults={"diameter": 5, "sigma_color": 50, "sigma_space": 50},
        explanation="The Bilateral filter smooths similar nearby pixels while trying to preserve strong edges.",
        benefit="Smooths flat areas while trying to keep edges sharp.",
        limitation=(
            "Slower than the others, and it may leave strong impulse noise: a single white "
            "pixel looks like an \"edge\", so it is not mixed with its neighbours."
        ),
    ),
}
FILTER_TYPES = tuple(FILTER_OPTIONS)  # ("mean", "gaussian", "median", "bilateral")


def get_filter_option(filter_type):
    """Return the FilterOption for a filter name, or raise a clear error."""
    if not isinstance(filter_type, str):
        raise TypeError(f"filter_type must be a string, got {type(filter_type).__name__}.")
    if filter_type not in FILTER_OPTIONS:
        raise ValueError(f"Unknown filter type '{filter_type}'. Choose one of: {', '.join(FILTER_TYPES)}.")
    return FILTER_OPTIONS[filter_type]


def apply_selected_filter(noisy_image, filter_type, parameters):
    """Apply one of the four existing filters to the NOISY image and return the result.

    This only selects and calls the existing function from src/filters.py:
        "mean"      -> apply_mean_filter(noisy, kernel_size)
        "gaussian"  -> apply_gaussian_filter(noisy, kernel_size, sigma)
        "median"    -> apply_median_filter(noisy, kernel_size)
        "bilateral" -> apply_bilateral_filter(noisy, diameter, sigma_color, sigma_space)

    `parameters` is a dict that must contain exactly the parameter names of
    that function (so no setting silently falls back to a default). The values
    are passed on unchanged, and src/filters.py checks them (odd window sizes,
    sigma >= 0 or > 0, and so on).

    The noisy image is never changed (src/filters.py always returns a new array).
    The result is a new uint8 image with the same shape, marked read-only.
    """
    option = get_filter_option(filter_type)
    if not isinstance(parameters, dict):
        raise TypeError(f"parameters must be a dict, got {type(parameters).__name__}.")
    expected = set(option.defaults)
    given = set(parameters)
    if given != expected:
        missing = ", ".join(sorted(expected - given)) or "none"
        unknown = ", ".join(sorted(map(str, given - expected))) or "none"
        raise ValueError(
            f"{option.label} needs exactly these parameters: {', '.join(sorted(expected))} "
            f"(missing: {missing}; unknown: {unknown})."
        )
    filtered = option.function(noisy_image, **parameters)
    return _read_only(filtered)


def describe_filter(filter_type, parameters, short=False):
    """A label for the filtered image.

    short=True : for the image caption, e.g. 'Bilateral Filter — d = 5'.
    short=False: for the settings line, e.g.
                 'Bilateral Filter — diameter = 5, sigma color = 50, sigma space = 50'.
    """
    option = get_filter_option(filter_type)
    if filter_type == "bilateral" and short:
        return f"{option.label} — d = {parameters['diameter']}"
    return f"{option.label} — {describe_filter_parameters(filter_type, parameters)}"


def describe_filter_parameters(filter_type, parameters):
    """The parameters as text, e.g. '3 × 3, filter σ = auto' or
    'diameter = 5, sigma color = 50, sigma space = 50'."""
    get_filter_option(filter_type)
    if filter_type == "bilateral":
        return (
            f"diameter = {parameters['diameter']}, sigma color = {parameters['sigma_color']:g}, "
            f"sigma space = {parameters['sigma_space']:g}"
        )
    k = parameters["kernel_size"]
    text = f"{k} × {k}"
    if filter_type == "gaussian":
        sigma = parameters["sigma"]
        text += ", filter σ = auto" if sigma == 0 else f", filter σ = {sigma:g}"
    return text


# --------------------------------------------------------------------------
# Evaluation (Stage 5D): MSE and PSNR from src/metrics.py
# --------------------------------------------------------------------------
# Every score compares an image with the ORIGINAL grayscale reference, using
# the full-resolution arrays (never the shrunk display copies).


def score_against_reference(reference, image, name):
    """Return {"mse": ..., "psnr": ...} for `image` compared with the reference.

    Both numbers come from src/metrics.py, which also checks that both are
    2-D uint8 images of the same size. `name` is added to any error message.
    """
    try:
        return {"mse": mse(reference, image), "psnr": psnr(reference, image)}
    except (TypeError, ValueError) as error:
        raise type(error)(f"{name}: {error}") from error


def format_signed(value, unit=""):
    """'+8.18', '-490.14', '0.00'; never '-0.00'. Tiny non-zero values show as '+<0.01'."""
    if value == 0:
        text = "0.00"
    elif round(value, 2) == 0:
        text = "+<0.01" if value > 0 else "−<0.01"
    else:
        text = f"{value:+.2f}".replace("-", "−")
    return f"{text} {unit}".strip()


def format_mse(value):
    return f"{value:,.2f}"


def format_psnr(value):
    """PSNR with two decimals, or '∞' for identical images (MSE = 0)."""
    return "∞" if math.isinf(value) else f"{value:.2f}"


def psnr_change(before, after):
    """PSNR change after - before, safe for infinity.

    Returns (change, short_text, explanation):
      * both finite      -> (after - before, '+8.18 dB', None)
      * otherwise change is None (it is not a finite number, and inf - inf
        would be NaN), and short_text / explanation say what happened.
    PSNR is infinite only when the image is identical to the reference.
    """
    if math.isinf(before) and math.isinf(after):
        return None, "no change (both ∞)", (
            "Both images are identical to the original (MSE = 0, PSNR = ∞), so nothing changed."
        )
    if math.isinf(after):
        return None, "now identical (∞)", (
            "The filtered image is identical to the original (MSE = 0, PSNR = ∞). "
            "The improvement cannot be written as a finite number of dB."
        )
    if math.isinf(before):
        return None, "worse (was ∞)", (
            "The noisy image was identical to the original (MSE = 0, PSNR = ∞), so there was no "
            "error to remove. Filtering added error, so the change cannot be written as a finite "
            "number of dB."
        )
    change = after - before
    return change, format_signed(change, "dB"), None


def judge_change(before_mse, after_mse):
    """'closer', 'farther' or 'same', comparing errors against the original.

    Uses MSE (always a finite number), so infinity never causes problems.
    Lower MSE always means higher PSNR, so both metrics agree.
    """
    if after_mse < before_mse:
        return "closer"
    if after_mse > before_mse:
        return "farther"
    return "same"


def evaluate_image_quality(reference, noisy, filtered):
    """Compare the noisy and the filtered image with the original reference.

    Returns a dict:
        noisy, filtered : {"mse", "psnr"} each, from src/metrics.py
        mse_change      : filtered MSE - noisy MSE (negative = less error)
        psnr_change     : filtered PSNR - noisy PSNR in dB, or None if not finite
        psnr_change_text, psnr_change_note : text from psnr_change()
        verdict         : "closer", "farther" or "same" (filtered vs noisy)
    No image is changed.
    """
    noisy_scores = score_against_reference(reference, noisy, "noisy image")
    filtered_scores = score_against_reference(reference, filtered, "filtered image")
    change, change_text, change_note = psnr_change(noisy_scores["psnr"], filtered_scores["psnr"])
    return {
        "noisy": noisy_scores,
        "filtered": filtered_scores,
        "mse_change": filtered_scores["mse"] - noisy_scores["mse"],
        "psnr_change": change,
        "psnr_change_text": change_text,
        "psnr_change_note": change_note,
        "verdict": judge_change(noisy_scores["mse"], filtered_scores["mse"]),
    }


def compare_all_filters(reference, noisy, filter_parameters):
    """Apply all four filters to the SAME noisy image and score everything.

    filter_parameters: dict with one entry per filter type, e.g.
        {"mean": {"kernel_size": 3}, "gaussian": {...}, "median": {...}, "bilateral": {...}}
    Every filter gets exactly the same `noisy` array (it is never regenerated),
    and every score uses the same `reference`.

    Returns a dict:
        rows : list of dicts, first the "No filtering" baseline, then the four
               filters, each with method, label, parameters (dict or None),
               parameter_text, image, mse, psnr, psnr_change,
               psnr_change_text, psnr_change_note
        best_methods     : the method(s) with the lowest MSE (= highest PSNR),
                           the baseline "none" included
        baseline_is_best : True if no filter beats "No filtering"
    """
    if not isinstance(filter_parameters, dict):
        raise TypeError(f"filter_parameters must be a dict, got {type(filter_parameters).__name__}.")
    if set(filter_parameters) != set(FILTER_TYPES):
        raise ValueError(f"filter_parameters must have exactly these keys: {', '.join(FILTER_TYPES)}.")

    baseline = score_against_reference(reference, noisy, "noisy image")
    rows = [
        {
            "method": "none",
            "label": "No filtering",
            "parameters": None,
            "parameter_text": "—",
            "image": noisy,
            **baseline,
            "psnr_change": None,
            "psnr_change_text": "baseline",
            "psnr_change_note": None,
        }
    ]
    for filter_type in FILTER_TYPES:
        parameters = filter_parameters[filter_type]
        filtered = apply_selected_filter(noisy, filter_type, parameters)
        scores = score_against_reference(reference, filtered, FILTER_OPTIONS[filter_type].label)
        change, change_text, change_note = psnr_change(baseline["psnr"], scores["psnr"])
        rows.append(
            {
                "method": filter_type,
                "label": FILTER_OPTIONS[filter_type].label,
                "parameters": dict(parameters),
                "parameter_text": describe_filter_parameters(filter_type, parameters),
                "image": filtered,
                **scores,
                "psnr_change": change,
                "psnr_change_text": change_text,
                "psnr_change_note": change_note,
            }
        )

    lowest = min(row["mse"] for row in rows)
    best_methods = [row["method"] for row in rows if row["mse"] == lowest]
    return {"rows": rows, "best_methods": best_methods, "baseline_is_best": "none" in best_methods}


# --------------------------------------------------------------------------
# Difference images (Stage 5D): visualisation only, not a quality metric
# --------------------------------------------------------------------------
def make_difference_image(image_a, image_b):
    """Return |image_a - image_b| for every pixel, as a new read-only uint8 image.

    cv2.absdiff works on the uint8 values without wrapping around
    (|0 - 255| = 255, not 1), so every value is the exact absolute difference.
    Black (0) = no change; brighter = larger change.
    """
    for name, image in (("first image", image_a), ("second image", image_b)):
        try:
            validate_grayscale_image(image)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{name}: {error}") from error
    if image_a.shape != image_b.shape:
        raise ValueError(f"Images must have the same dimensions: {image_a.shape} vs {image_b.shape}.")
    return _read_only(cv2.absdiff(image_a, image_b))


def enhance_differences(differences):
    """Contrast-stretch difference images FOR DISPLAY ONLY.

    All images are multiplied by the SAME factor 255 / (largest difference in
    any of them), so they stay comparable with each other: the largest
    difference becomes white. Returns (enhanced_images, factor).
    If there is no difference at all, the factor is 1 and nothing changes.
    The input images are not changed, and the results must never be used
    for MSE or PSNR.
    """
    peak = max(int(d.max()) for d in differences)
    if peak == 0:
        return [d.copy() for d in differences], 1.0
    factor = 255.0 / peak
    enhanced = [np.rint(d.astype(np.float64) * factor).astype(np.uint8) for d in differences]
    return enhanced, factor


def summarize_difference(difference):
    """Simple facts about a difference image: changed pixels and the largest change."""
    changed = int(np.count_nonzero(difference))
    return {
        "changed_pixels": changed,
        "changed_percent": 100.0 * changed / difference.size,
        "largest": int(difference.max()),
    }


def pixel_fingerprint(image):
    """A short code computed from every pixel value (the first 12 characters of SHA-256).

    Two images with the same shape and identical pixels always get the same
    code; changing even one pixel gives a different code (except with
    negligible probability). The app shows it so users can SEE that the same
    settings and seed reproduce exactly the same noisy image.
    """
    validate_grayscale_image(image)
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------
# Display helpers (they change only what is SHOWN, never the processed images)
# --------------------------------------------------------------------------
# A compact box for each image of the side-by-side comparison on a typical
# laptop screen (three images in a row since Stage 5C).
DISPLAY_MAX_WIDTH = 420
DISPLAY_MAX_HEIGHT = 340
# The zoomed crop is enlarged to about this many screen pixels (16 x 16 -> 256 x 256).
ZOOM_DISPLAY_SIZE = 256


def fit_for_display(image, max_width=DISPLAY_MAX_WIDTH, max_height=DISPLAY_MAX_HEIGHT):
    """Return a copy shrunk to fit inside max_width x max_height (never enlarged).

    The aspect ratio is kept. Shrinking uses cv2.INTER_AREA, which averages
    neighbouring pixels; this makes fine noise look weaker, which is why the
    app also offers a 1:1 zoom of the real pixels.
    Works for grayscale (2-D) and colour (3-D) arrays.
    """
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return image.copy()
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))  # OpenCV wants (width, height)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def crop_square(image, centre_x_percent, centre_y_percent, size):
    """Cut a size x size square around a point given in percent of the width/height.

    The square is moved inward if it would leave the image, and made smaller
    if the image itself is smaller. Returns (crop, (top, left)).
    """
    height, width = image.shape[:2]
    size = int(min(size, height, width))
    centre_x = round(centre_x_percent / 100 * (width - 1))
    centre_y = round(centre_y_percent / 100 * (height - 1))
    left = min(max(centre_x - size // 2, 0), width - size)
    top = min(max(centre_y - size // 2, 0), height - size)
    return image[top : top + size, left : left + size], (top, left)


def enlarge_pixels(image, factor):
    """Show each pixel as a factor x factor block (nearest neighbour).

    No new values are invented: every displayed value is an original pixel value.
    """
    if factor < 1:
        raise ValueError(f"factor must be at least 1, got {factor}.")
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def is_large_image(uploaded, threshold=LARGE_IMAGE_PIXELS):
    """True if the image has more pixels than `threshold` (processing may be slow)."""
    return uploaded.pixel_count > threshold


def format_file_size(num_bytes):
    """Turn a number of bytes into readable text, e.g. 2048 -> '2.0 KB'."""
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"
