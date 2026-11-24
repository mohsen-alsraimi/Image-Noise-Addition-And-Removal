# Image Noise Addition and Removal
## Experiment Documentation

### 1. Project Objective

Compare image-denoising algorithms across different
noise types and intensities.

Evaluate the impact of parameter tuning on image
quality, structural similarity, and execution time.

### 2. Datasets

**BSD68**
- 68 grayscale images
- Used for final testing and evaluation

**Set12**
- 12 grayscale benchmark images
- Used to select filter parameters
- Separate from the final BSD68 test dataset

### 3. Noise Algorithms

| Noise | Low | Medium | High |
|---|---|---|---|
| Gaussian (sigma) | 15 | 25 | 50 |
| Salt-and-Pepper (fraction) | 0.05 | 0.10 | 0.20 |
| Speckle (variance) | 0.01 | 0.04 | 0.10 |
| Poisson (peak) | 100 | 30 | 10 |

Images are normalized to the range [0, 1].

A fixed base random seed (42) makes the simulated
noise reproducible for the same image and condition.

### 4. Denoising Algorithms

1. Gaussian Filter
2. Median Filter
3. Bilateral Filter
4. Non-Local Means (NLM)
5. Lee Filter
6. Total Variation (TV)

### 5. Parameter Tuning

The script `tune_filters.py` tests candidate
parameters on Set12.

The selection metric is average PSNR.

The tuning grid is:

- Gaussian:
  (3, 0.5), (5, 1.0), (7, 1.5)
  for (kernel_size, sigma)

- Median:
  kernel_size = 3, 5, 7

- Bilateral:
  (diameter, sigma_color) =
  (5, 0.05), (9, 0.1), (9, 0.2)

- NLM:
  h = 0.1, 0.15, 0.2

- Lee:
  window_size = 3, 5, 7

- TV:
  weight = 0.03, 0.07, 0.1

Other arguments retain their function defaults.

The selected settings are saved to
`best_parameters.json`.

### 6. Evaluation Metrics

PSNR:
Measures pixel reconstruction accuracy.
Higher is better.

SSIM:
Measures structural similarity to the clean image.
Higher is better.

Execution Time:
Measures filtering runtime in seconds.
Lower is better.

### 7. Experimental Procedure

1. Load clean grayscale images.
2. Generate simulated noisy images.
3. Apply all six filters using default parameters.
4. Tune parameters separately using Set12.
5. Apply tuned parameters to the same noisy inputs.
6. Evaluate both configurations on BSD68.
7. Compare PSNR, SSIM, and execution time.
8. Generate comparison graphs and images.

### 8. Reproduction

Run the commands from the Denoising ad folder.

Install dependencies:

    py -m pip install -r requirements.txt

Run parameter tuning:

    py tune_filters.py

In main.py, set MAX_IMAGES = None.

For the original experiment, set:

    TUNED_PARAMETERS_FILE = None

Then run:

    py main.py

For the tuned experiment, set:

    TUNED_PARAMETERS_FILE = Path("best_parameters.json")

Then run:

    py main.py

Compare the resulting folders:

    py compare_tuning.py resultX resultY

Generate before-and-after images:

    py visualize_tuning.py

Replace resultX and resultY with the actual
original and tuned folder names.

### 9. Reproducibility Notes

- Use all 12 Set12 images for final tuning.
- Do not select parameters using BSD68.
- Use identical noise settings and random seeds
  for original and tuned comparisons.
- Keep the filter implementations unchanged
  between the two evaluation runs.
- Record the Python and dependency versions.
- The Lee filter uses known simulated speckle
  variance when supported by its implementation.
- The NLM implementation enforces a minimum h
  based on its estimated noise level.
- Runtime may vary with hardware and system load.

### 10. Limitations

The parameter search covers only a small grid.

Selecting the highest PSNR does not guarantee
the highest SSIM or fastest execution.

Results may vary across different datasets.

For these reasons, conclusions should be based
on the independent BSD68 evaluation rather
than the tuning results alone.

### 2. Datasets

#### BSD68 (Berkeley Segmentation Dataset 68)

- Contains 68 grayscale natural images.
- Used for final testing and evaluation of the denoising algorithms.
- Derived from the Berkeley Segmentation Dataset.
- Widely used as a benchmark for image denoising.

**References:**
- [Berkeley Segmentation Dataset — Official Website](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- [BSD68 Dataset — GitHub](https://github.com/smartboy110/denoising-datasets)

#### Set12 (Standard Image Denoising Dataset)

- Contains 12 grayscale benchmark images.
- Includes images of different sizes: 256 × 256 and 512 × 512 pixels.
- Used for selecting optimal filter parameters.
- Kept separate from BSD68 to avoid tuning on the final evaluation dataset.

**References:**
- [Set12 Dataset — GitHub (FFDNet)](https://github.com/cszn/FFDNet/tree/master/testsets/Set12)
- [Set12 Dataset — HyperAI](https://hyper.ai/en/datasets/17513)

**Dataset separation:** Set12 is used for parameter tuning, while BSD68 is used for independent evaluation of the original and tuned filters.

### 11. References

1. Martin, D., Fowlkes, C., Tal, D., & Malik, J. (2001). *A Database of Human Segmented Natural Images and Its Application to Evaluating Segmentation Algorithms and Measuring Ecological Statistics.* Proceedings of ICCV. [Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).

2. Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising.* IEEE Transactions on Image Processing, 26(7), 3142–3155. [DnCNN Repository](https://github.com/cszn/DnCNN).

3. Zhang, K., Zuo, W., & Zhang, L. (2018). *FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising.* IEEE Transactions on Image Processing, 27(9), 4608–4622. [Set12 Benchmark Images](https://github.com/cszn/FFDNet/tree/master/testsets/Set12).