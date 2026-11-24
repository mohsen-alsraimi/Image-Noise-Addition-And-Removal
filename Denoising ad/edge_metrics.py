"""
Edge-preservation scores: does a processed image keep the clean image's edges?

Adapted from the teammate branch
feature/image_degradation-different_datasets-edge_preservation (main.py,
calculate_edge_metrics). The method and settings are unchanged:

1. Canny (OpenCV, thresholds 50 and 150) finds the edge pixels of the clean
   image and of the processed image.
2. An edge pixel counts as matched if the other image has an edge pixel
   within 1 pixel (a 3x3 elliptical neighbourhood).
3. Precision: share of the processed image's edge pixels that are real.
   Recall: share of the clean image's edge pixels that were kept.
   F1: 2 * precision * recall / (precision + recall).

One correction: images are converted to 0-255 by rounding instead of
cutting off the decimals. For dataset images (whole numbers divided by 255)
both give the same values; for filtered images, rounding is the correct
conversion.
"""

import cv2
import numpy as np

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
MATCH_TOLERANCE_PIXELS = 1

EDGE_METRIC_SETTINGS = {
    "edge_detector": "OpenCV cv2.Canny, default 3x3 Sobel aperture, L1 gradient, no extra smoothing",
    "canny_low_threshold": CANNY_LOW_THRESHOLD,
    "canny_high_threshold": CANNY_HIGH_THRESHOLD,
    "match_tolerance_pixels": MATCH_TOLERANCE_PIXELS,
    "match_neighbourhood": "3x3 elliptical structuring element (dilation)",
    "conversion": "clip to 0-1, multiply by 255, round to the nearest whole number",
    "reference": "Canny edges of the clean original image",
    "edge_precision": "processed edge pixels with a clean edge within the tolerance / all processed edge pixels",
    "edge_recall": "clean edge pixels with a processed edge within the tolerance / all clean edge pixels",
    "edge_f1": "2 * precision * recall / (precision + recall); 0 when both are 0",
    "no_edges": "1, 1, 1 when neither image has edges; precision 0 when only the processed image has none",
    "source": "teammate branch feature/image_degradation-different_datasets-edge_preservation",
}

EDGE_METRIC_LIMITATIONS = [
    "Edge scores use fixed Canny thresholds (50, 150) without smoothing, so fine texture also counts as edges.",
    "Strong noise creates edge pixels almost everywhere. With a 1-pixel tolerance many of them land next "
    "to real edges by chance, so an unfiltered noisy image can get a high Edge Recall and Edge F1. "
    "Read Edge F1 together with Edge Precision, PSNR, and SSIM.",
]


def to_uint8(image):
    """Converts a 0-1 image to 0-255 whole numbers, rounding to the nearest."""
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def find_edges(image):
    """Canny edges of a 0-1 grayscale image, as a True/False map."""
    return cv2.Canny(to_uint8(image), CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD) > 0


def _widen(edges):
    size = 2 * MATCH_TOLERANCE_PIXELS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(edges.astype(np.uint8), kernel) > 0


def calculate_edge_metrics(reference_image, processed_image):
    """Returns (edge precision, edge recall, edge F1) against the clean image."""
    reference_edges = find_edges(reference_image)
    processed_edges = find_edges(processed_image)

    reference_count = np.count_nonzero(reference_edges)
    processed_count = np.count_nonzero(processed_edges)

    if reference_count == 0 and processed_count == 0:
        return 1.0, 1.0, 1.0

    matched_processed = np.count_nonzero(processed_edges & _widen(reference_edges))
    precision = matched_processed / processed_count if processed_count else 0.0

    matched_reference = np.count_nonzero(reference_edges & _widen(processed_edges))
    recall = matched_reference / reference_count if reference_count else 0.0

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1_score)
