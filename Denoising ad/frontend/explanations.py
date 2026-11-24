"""Plain-English text for the Image Noise Lab pages. It contains no result numbers."""

# Short definitions, used for tooltips and on the "Learn" page
GLOSSARY = {
    "Image": "A grid of pixels. Each pixel here is a brightness from 0 (black) to 1 (white).",
    "Noise": "Unwanted random changes in pixel brightness, like grain in a dark photo.",
    "Image damage": "Changes that are not random noise, such as JPEG blocks or a bright glare spot.",
    "Filter": "An algorithm that looks at each pixel and its neighbours and tries to undo the noise.",
    "Kernel / window size": "How many neighbouring pixels the filter looks at. Bigger is smoother but blurrier.",
    "Strength": "How much noise or damage is added: Low, Medium or High. The exact values are fixed.",
    "Baseline": "The noisy or damaged image before any filter. A filter only helped if it beats this.",
    "PSNR": "How close the pixel values are to the original, in decibels (dB). Higher is better.",
    "SSIM": "How similar shapes and textures are to the original, from 0 to 1. Higher is better.",
    "Edge F1": "How well object outlines agree with the original. Useful, but misleading under heavy noise.",
    "Dataset": "A folder of test images. Experiments run on one image or on all of them.",
    "Preset": "A saved set of filter values. You can start from one and still change any value.",
    "Seed": "A fixed number that makes the random noise repeatable, so results can be checked.",
}

# Short names for filter settings, shown as control labels
PARAMETER_LABELS = {
    "kernel_size": "Window size",
    "sigma": "Blur amount (sigma)",
    "diameter": "Neighbourhood width",
    "sigma_color": "Brightness similarity",
    "sigma_space": "Distance reach",
    "patch_size": "Patch size",
    "patch_distance": "Search distance",
    "h": "Smoothing strength (h)",
    "window_size": "Window size",
    "weight": "Smoothing strength",
}

PSNR_TEXT = (
    "**PSNR** measures how close the pixel values are to the clean image. "
    "It is measured in decibels (dB), and **higher is better**. "
    "A gain of about 3 dB means the squared pixel error was cut roughly in half."
)

SSIM_TEXT = (
    "**SSIM** measures how well shapes, edges, and textures were kept. "
    "It is usually between 0 and 1, and **closer to 1 is better**."
)

EDGE_TEXT = (
    "**Edge F1** checks whether the outlines of objects survived. Edges are found in the "
    "clean image and in the result with the Canny detector. It is between 0 and 1, and "
    "**higher is better**."
)

EDGE_DETAIL_TEXT = (
    "**Edge Precision:** of the edges in the result, how many are real edges? Noise creates "
    "fake edges, which lowers it. **Edge Recall:** of the real edges, how many were kept? "
    "Blurring removes edges, which lowers it. **Edge F1** balances the two. An edge counts "
    "as kept if it is within 1 pixel of the real one (Canny thresholds 50 and 150)."
)

EDGE_CAUTION = (
    "Careful: strong noise creates edges almost everywhere, and some of them land on real "
    "edges by chance. So an unfiltered image can get a high Edge Recall and Edge F1 without "
    "being clean. Read Edge F1 together with Edge Precision, PSNR, and SSIM."
)

DATASET_NOTES = {
    "Set12": ("info", "Set12 is used to choose filter settings, so treat these results as practice."),
    "Urban100_HR": (
        "warning",
        "Urban100_HR has large colour photos of buildings (about 0.6 to 1.2 megapixels, "
        "roughly 5 times more pixels than a BSD68 image). They are converted to grayscale "
        "like every other image. Full-dataset experiments will be much slower, and the "
        "saved pictures and folders will be larger.",
    ),
    "original_png": (
        "warning",
        "original_png: 12 grayscale images (1.22 megapixels each) added by a teammate. "
        "Their origin and intended purpose have not been verified yet. Check them before "
        "using these results in a research report.",
    ),
}

NOISE_TEXT = {
    "Gaussian": "Gaussian noise adds a small random change to every pixel, like fine grain.",
    "Salt-and-Pepper": "Salt-and-Pepper noise turns some random pixels fully white or black.",
    "Speckle": "Speckle noise multiplies pixels by a random amount, so bright areas get stronger noise.",
    "Poisson": "Poisson noise imitates low light, where the noise depends on image brightness.",
}

DEGRADATION_TEXT = {
    "JPEG Compression": (
        "JPEG compression makes the file smaller by throwing away fine detail, which can "
        "create 8×8 blocks and ripples near edges. Lower quality means stronger damage."
    ),
    "Glare": (
        "Glare adds a soft bright spot, like light reflecting on a lens. For one image the "
        "spot is always in the same place; only its brightness changes with the strength."
    ),
}

# Text for every condition type a run can contain
CONDITION_TEXT = {**NOISE_TEXT, **DEGRADATION_TEXT}

FILTER_TEXT = {
    "Gaussian": "The Gaussian filter takes a weighted average, with nearby pixels counting most.",
    "Median": "The Median filter sorts nearby pixels and keeps the middle value.",
    "Bilateral": "The Bilateral filter averages pixels that are both nearby and similar in brightness.",
    "NLM": "Non-Local Means averages pixels from small image patches that look similar.",
    "Lee": "The Lee filter smooths flat areas more and keeps areas with strong detail.",
    "TV": "Total Variation removes small ups and downs while trying to keep strong edges.",
}

# One line per filter: what it is good at, for beginners choosing filters
FILTER_GOOD_FOR = {
    "Gaussian": "Fast and simple; reduces fine grain but blurs edges.",
    "Median": "Removes isolated black and white dots (salt-and-pepper).",
    "Bilateral": "Smooths flat areas while keeping strong edges sharp.",
    "NLM": "High quality on grain and texture; the slowest filter.",
    "Lee": "Made for speckle, where noise grows with brightness.",
    "TV": "Flattens grain and keeps big edges; can look a little cartoon-like.",
}

WHY_TEXT = {
    ("Salt-and-Pepper", "Median"): (
        "White and black dots are extreme values. When nearby pixels are sorted, "
        "those dots move to the ends, so the middle value usually ignores them."
    ),
    ("Speckle", "Lee"): (
        "Lee is designed for noise that grows with brightness. It estimates how "
        "much variation is expected from speckle and smooths less where real detail remains."
    ),
    ("Speckle", "TV"): (
        "TV can flatten small speckle changes while keeping larger brightness edges."
    ),
    ("Speckle", "NLM"): (
        "NLM can average repeated patches, which reduces random speckle when similar areas exist."
    ),
    ("Gaussian", "Gaussian"): (
        "Random grain goes up and down, so weighted averaging cancels much of it. "
        "The cost is some blur."
    ),
    ("Gaussian", "TV"): (
        "Grain creates many small ups and downs. TV reduces them while trying to keep strong edges."
    ),
    ("Gaussian", "NLM"): (
        "Averaging similar patches can cancel random grain with less blur than a simple window."
    ),
    ("Poisson", "Gaussian"): (
        "Strong low-light noise behaves much like random grain, so averaging can reduce it."
    ),
    ("Poisson", "TV"): (
        "TV can reduce small random brightness changes while trying to keep strong edges."
    ),
    ("Poisson", "NLM"): (
        "Averaging similar patches can reduce random low-light noise."
    ),
}


PARAMETER_HELP = {
    "Gaussian Filter": {
        "kernel_size": "Width of the square window, in pixels. Bigger looks further.",
        "sigma": "How fast weights fade from the center. Bigger means more blur.",
    },
    "Median Filter": {
        "kernel_size": "Width of the window whose middle value is kept. Bigger removes larger dots but more detail.",
    },
    "Bilateral Filter": {
        "diameter": "Width of the neighbourhood, in pixels.",
        "sigma_color": "How different in brightness (0-1) a neighbour can be and still be averaged.",
        "sigma_space": "How much distance matters, in pixels. Bigger lets far pixels count more.",
    },
    "NLM Filter": {
        "patch_size": "Size of the small squares compared to find similar areas.",
        "patch_distance": "How far to search for similar squares. Bigger is slower.",
        "h": "How strongly similar squares are averaged. Bigger means smoother. Automatic estimates it.",
    },
    "Lee Filter": {
        "window_size": "Width of the local area used to measure brightness and variation.",
    },
    "TV Filter": {
        "weight": "Denoising strength. Bigger gives a flatter, smoother result.",
    },
}


def why_filter_may_work(noise_type, filter_name):
    """Gives the likely reason. The saved scores remain the measured evidence."""
    specific = WHY_TEXT.get((noise_type, filter_name))
    if specific:
        return specific

    damage = "damage" if noise_type in DEGRADATION_TEXT else "noise"
    return (
        f"{FILTER_TEXT[filter_name]} With the saved settings, that approach "
        f"raised the PSNR for this {damage}."
    )
