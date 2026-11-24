"""Plain-English text for the Results Viewer. It contains no result numbers."""

PSNR_TEXT = (
    "**PSNR** measures how close the pixel values are to the clean image. "
    "It is measured in decibels (dB), and **higher is better**. "
    "A gain of about 3 dB means the squared pixel error was cut roughly in half."
)

SSIM_TEXT = (
    "**SSIM** measures how well shapes, edges, and textures were kept. "
    "It is usually between 0 and 1, and **closer to 1 is better**."
)

NOISE_TEXT = {
    "Gaussian": "Gaussian noise adds a small random change to every pixel, like fine grain.",
    "Salt-and-Pepper": "Salt-and-Pepper noise turns some random pixels fully white or black.",
    "Speckle": "Speckle noise multiplies pixels by a random amount, so bright areas get stronger noise.",
    "Poisson": "Poisson noise imitates low light, where the noise depends on image brightness.",
}

FILTER_TEXT = {
    "Gaussian": "The Gaussian filter takes a weighted average, with nearby pixels counting most.",
    "Median": "The Median filter sorts nearby pixels and keeps the middle value.",
    "Bilateral": "The Bilateral filter averages pixels that are both nearby and similar in brightness.",
    "NLM": "Non-Local Means averages pixels from small image patches that look similar.",
    "Lee": "The Lee filter smooths flat areas more and keeps areas with strong detail.",
    "TV": "Total Variation removes small ups and downs while trying to keep strong edges.",
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

    return (
        f"{FILTER_TEXT[filter_name]} With the saved settings, that approach "
        "raised the PSNR for this noise."
    )
