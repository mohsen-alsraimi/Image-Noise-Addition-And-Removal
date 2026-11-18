# Experiment Log

_Last updated: 2026-11-16 (after Stage 4)_

Rules:
* Every entry must come from an actual program run. Never type results in by hand.
* Every number in the report's Results section must be traceable to an entry here.
* Filters compared in one experiment group must use the **same** noisy image (same image, noise type, intensity and seed).
* The noisy image itself is always recorded as the **baseline** (filter = "none").

## Entry format

| Field | Meaning |
|---|---|
| Experiment ID | `E###`, for example E001. One ID per (image, noise, intensity, seed, filter, parameters). |
| Group ID | `G###`. All filters applied to the same noisy image share a group. |
| Date | YYYY-MM-DD of the run |
| Input image | Filename in `data/input/` |
| Noise type | gaussian / salt_and_pepper / speckle |
| Noise intensity | For example σ = 25, p = 0.05, variance = 0.04 |
| Random seed | Integer used for noise generation |
| Filter | none (baseline) / mean / gaussian / median / bilateral |
| Filter parameters | For example ksize = 3; or d = 9, sigmaColor = 75, sigmaSpace = 75 |
| MSE | Compared with the clean original |
| PSNR (dB) | Compared with the clean original |
| Saved image paths | Paths in `results/noisy/` and `results/filtered/` |
| Source of numbers | Script and command that produced the values, plus the CSV file |
| Observations | What we saw (visual and numerical) |

## Table template

| ID | Group | Date | Image | Noise | Intensity | Seed | Filter | Params | MSE | PSNR (dB) | Output | Observations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Entries: noise experiments (E###)

### Stage 3: noise addition, noisy baselines (filter = none)
These are real noise experiments, but **no filter was applied**. Each one creates
one noisy image (group G###) that Stage 4 filters will later be applied to.

* **Date:** 2026-11-16
* **Command:** `python -m demos.demo_noise --save-figure --save-outputs` (project root)
* **Code:** `src/noise.py` (`add_gaussian_noise`, `add_salt_and_pepper_noise`, `add_speckle_noise`); metrics from `src/metrics.py`
* **Environment:** Python 3.12.10, NumPy 2.5.5 (Generator `default_rng`, PCG64), OpenCV 5.0.0, Windows 11
* **Input image:** `data/input/synthetic_test_image.png` (256 × 256 = 65,536 pixels, uint8, values 20–240). The program confirmed "Original image unchanged: True".
* **Random seed:** 42 for every experiment (each call creates its own generator from seed 42).
* **Machine-readable copy:** `results/tables/stage3_noise_results.csv` (also contains the dates and salt/pepper counts)
* **Reproducibility check:** a second run produced byte-identical PNG and CSV files (SHA-256), and a saved PNG reloads to exactly the array the function returns.

| ID | Group | Image | Noise | Strength | Seed | Filter | MSE | PSNR (dB) | Changed px | At 0/255 | Output (`results/noisy/`) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E001 | G001 | synthetic_test_image.png (256×256) | gaussian | σ = 10 | 42 | none | 99.5965 | 28.1484 | 62,782 | 456 | synthetic_test_image__gaussian_sigma10__seed42.png |
| E002 | G002 | synthetic_test_image.png (256×256) | gaussian | σ = 25 | 42 | none | 577.9914 | 20.5116 | 64,457 | 3,713 | synthetic_test_image__gaussian_sigma25__seed42.png |
| E003 | G003 | synthetic_test_image.png (256×256) | gaussian | σ = 50 | 42 | none | 2053.9173 | 15.0050 | 64,984 | 8,693 | synthetic_test_image__gaussian_sigma50__seed42.png |
| E004 | G004 | synthetic_test_image.png (256×256) | salt_and_pepper | amount = 0.02 | 42 | none | 413.3608 | 21.9675 | 1,311 | 1,311 | synthetic_test_image__salt_and_pepper_amount0.02__seed42.png |
| E005 | G005 | synthetic_test_image.png (256×256) | salt_and_pepper | amount = 0.05 | 42 | none | 1058.0422 | 17.8858 | 3,277 | 3,277 | synthetic_test_image__salt_and_pepper_amount0.05__seed42.png |
| E006 | G006 | synthetic_test_image.png (256×256) | salt_and_pepper | amount = 0.10 | 42 | none | 2083.0331 | 14.9438 | 6,554 | 6,554 | synthetic_test_image__salt_and_pepper_amount0.1__seed42.png |
| E007 | G007 | synthetic_test_image.png (256×256) | speckle | variance = 0.01 | 42 | none | 171.4604 | 25.7892 | 61,713 | 1,680 | synthetic_test_image__speckle_variance0.01__seed42.png |
| E008 | G008 | synthetic_test_image.png (256×256) | speckle | variance = 0.04 | 42 | none | 636.4267 | 20.0933 | 63,627 | 3,086 | synthetic_test_image__speckle_variance0.04__seed42.png |
| E009 | G009 | synthetic_test_image.png (256×256) | speckle | variance = 0.10 | 42 | none | 1483.1099 | 16.4191 | 64,330 | 4,468 | synthetic_test_image__speckle_variance0.1__seed42.png |

Salt-and-pepper counts (rounding rule: selected = ⌊amount × N + 0.5⌋; salt = ⌊selected/2⌋; pepper = rest):
* E004: 0.02 × 65,536 = 1310.72 → 1,311 selected = 655 salt + 656 pepper; changed 1,311
* E005: 0.05 × 65,536 = 3276.80 → 3,277 selected = 1,638 salt + 1,639 pepper; changed 3,277
* E006: 0.10 × 65,536 = 6553.60 → 6,554 selected = 3,277 salt + 3,277 pepper; changed 6,554

**Figure:** `results/figures/stage3_noise_examples.png` (300 dpi, all nine experiments plus the original).

**Observations:**
* E001–E003: PSNR falls 28.15 → 20.51 → 15.00 dB as σ grows. Measured MSE / σ² = 0.996, 0.925, 0.822, so clipping at 0/255 increasingly cuts off the noise (pixels at 0/255: 456 → 8,693). The unchanged pixels in E001 (2,754) are exactly those whose noise was smaller than ±0.5.
* E004–E006: only the selected pixels change, but each by a large amount, so MSE is high for few pixels. E006 (14.94 dB) ≈ E003 (15.00 dB) in PSNR, but E006 looks far less degraded overall: the shapes and lines stay recognisable.
* E007–E009: noise is visibly stronger on bright regions (circle, right side) than on dark ones (dark square, left side). Measured MSE / predicted (variance × mean(I²) = variance × 18983.8) = 0.903, 0.838, 0.781, an even stronger clipping effect than Gaussian, because the largest changes occur at the brightest pixels.

### Stage 4: noise removal, four filters on the same noisy image (E010–E017)
Each group uses **one** noisy image, and all four filters are applied to that same array. The baselines are the Stage 3 entries E002 (G002) and E005 (G005); they are not given new IDs.

* **Date:** 2026-11-16
* **Command:** `python -m demos.demo_filters --save-figure --save-outputs` (project root)
* **Code:** `src/filters.py` (`apply_mean_filter`, `apply_gaussian_filter`, `apply_median_filter`, `apply_bilateral_filter`), noise from `src/noise.py`, metrics from `src/metrics.py`
* **Environment:** Python 3.12.10, NumPy 2.5.5, OpenCV 5.0.0, Windows 11
* **Input image:** `data/input/synthetic_test_image.png` (256 × 256). The program confirmed "Original image unchanged: True". It also checks internally that no filter modified the noisy image.
* **Noisy images:** recreated with seed 42. The program confirmed both are identical to the saved Stage 3 files: G002 = `synthetic_test_image__gaussian_sigma25__seed42.png`: True; G005 = `synthetic_test_image__salt_and_pepper_amount0.05__seed42.png`: True. The Stage 3 files were only read (SHA-256 unchanged).
* **Machine-readable copy:** `results/tables/stage4_filter_results.csv`
* **Reproducibility check:** a second run produced byte-identical filtered PNGs, CSV and figure (SHA-256).
* **Border handling:** mean, Gaussian and bilateral use `BORDER_REFLECT_101`; median uses `BORDER_REPLICATE` (fixed inside OpenCV).

| ID | Group | Baseline | Noise | Filter | Parameters | MSE | PSNR (dB) | ΔPSNR vs noisy (dB) | Output (`results/filtered/`) |
|---|---|---|---|---|---|---|---|---|---|
| E002 | G002 | – | gaussian σ = 25 | none (baseline) | – | 577.9914 | 20.5116 | – | (Stage 3, `results/noisy/`) |
| E010 | G002 | E002 | gaussian σ = 25 | mean | kernel_size = 3 | 945.6829 | 18.3733 | −2.1382 | synthetic_test_image__gaussian_sigma25__seed42__mean_k3.png |
| E011 | G002 | E002 | gaussian σ = 25 | gaussian | kernel_size = 3, sigma = 0 | 588.3290 | 20.4346 | −0.0770 | synthetic_test_image__gaussian_sigma25__seed42__gaussian_k3_sigma0.png |
| E012 | G002 | E002 | gaussian σ = 25 | median | kernel_size = 3 | 1216.9818 | 17.2780 | −3.2336 | synthetic_test_image__gaussian_sigma25__seed42__median_k3.png |
| E013 | G002 | E002 | gaussian σ = 25 | bilateral | diameter = 5, sigma_color = 50, sigma_space = 50 | 87.8501 | 28.6934 | +8.1818 | synthetic_test_image__gaussian_sigma25__seed42__bilateral_d5_sc50_ss50.png |
| E005 | G005 | – | salt_and_pepper amount = 0.05 | none (baseline) | – | 1058.0422 | 17.8858 | – | (Stage 3, `results/noisy/`) |
| E014 | G005 | E005 | salt_and_pepper amount = 0.05 | mean | kernel_size = 3 | 1007.2358 | 18.0995 | +0.2137 | synthetic_test_image__salt_and_pepper_amount0.05__seed42__mean_k3.png |
| E015 | G005 | E005 | salt_and_pepper amount = 0.05 | gaussian | kernel_size = 3, sigma = 0 | 672.0491 | 19.8568 | +1.9710 | synthetic_test_image__salt_and_pepper_amount0.05__seed42__gaussian_k3_sigma0.png |
| E016 | G005 | E005 | salt_and_pepper amount = 0.05 | median | kernel_size = 3 | 1271.0915 | 17.0890 | −0.7967 | synthetic_test_image__salt_and_pepper_amount0.05__seed42__median_k3.png |
| E017 | G005 | E005 | salt_and_pepper amount = 0.05 | bilateral | diameter = 5, sigma_color = 50, sigma_space = 50 | 861.1188 | 18.7802 | +0.8944 | synthetic_test_image__salt_and_pepper_amount0.05__seed42__bilateral_d5_sc50_ss50.png |

**MSE by region (same run, printed by the demo).**
* **Flat patch:** rows 141–229, columns 20–114, all 128; 8,455 px. It has no edges inside it, so it mostly shows pure noise removal. Its outermost pixels are an exception: their filter windows reach outside the patch.
* **Line block:** rows 141–229, columns 141–235, 24 one-pixel lines of 240 on 40; 8,455 px, which is 12.9% of the image. It shows the loss of fine detail.

| ID | Filter | Gaussian σ = 25: flat patch | line block | whole image |
|---|---|---|---|---|
| E002 | none | 629.93 | 520.13 | 577.99 |
| E010 | mean | 80.14 | 6713.03 | 945.68 |
| E011 | gaussian | 94.16 | 3889.52 | 588.33 |
| E012 | median | 107.46 | 8664.46 | 1216.98 |
| E013 | bilateral | 88.01 | 119.86 | 87.85 |

| ID | Filter | Salt-and-pepper 0.05: flat patch | line block | whole image |
|---|---|---|---|---|
| E005 | none | 830.67 | 1181.32 | 1058.04 |
| E014 | mean | 102.34 | 6775.90 | 1007.24 |
| E015 | gaussian | 125.27 | 4039.72 | 672.05 |
| E016 | median | 2.66 | 9826.72 | 1271.09 |
| E017 | bilateral | 458.25 | 998.52 | 861.12 |

**Figure:** `results/figures/stage4_filter_comparison.png` (300 dpi; row 1 = Gaussian noise, row 2 = salt-and-pepper; columns: original, noisy, mean, Gaussian, median, bilateral; MSE, PSNR and ΔPSNR in the titles).

**Observations (from this run and the figure):**
* **Line block:** in the figure, the 1-pixel lines are erased by the median filter (the block becomes flat gray), faded by mean and Gaussian, and kept by the bilateral filter. This one region (12.9% of the pixels) decides most of the whole-image ranking.
* **Flat patch, Gaussian noise:** all four filters reduce the error to 13–17% of the noisy value. The mean filter (80.14) is lowest and the median (107.46) highest.
* **Flat patch, salt-and-pepper:** the median removes almost everything (830.67 → 2.66, −99.7%). The bilateral filter removes the least (→ 458.25, −44.8%), and in the figure most impulses are still visible.
* **E016 has a lower whole-image PSNR than the noisy image (−0.80 dB),** although the median removed the impulses almost everywhere. The reason is that the median erased the lines (line-block MSE 9826.72).
* **E013 (bilateral, Gaussian noise)** is the only setting that improves both regions. It gives the largest PSNR gain (+8.18 dB).

### Filter cost on the clean image (FC###): not a noise removal experiment
Each filter applied to the **clean** original (no noise), to measure how much filtering alone changes it. Same command and run as above.

| ID | Filter | Parameters | MSE | PSNR (dB) | Changed px | Line-block MSE |
|---|---|---|---|---|---|---|
| FC001 | mean | kernel_size = 3 | 879.34 | 18.69 | 9,378 | 6657.56 |
| FC002 | gaussian | kernel_size = 3, sigma = 0 | 497.07 | 21.17 | 9,114 | 3758.35 |
| FC003 | median | kernel_size = 3 | 1263.88 | 17.11 | 2,563 | 9793.13 |
| FC004 | bilateral | diameter = 5, sigma_color = 50, sigma_space = 50 | 3.44 | 42.76 | 21,041 | 4.00 |

**Share of the clean-image MSE coming from the line block** (calculated from the logged values: line-block MSE × 8,455 / 65,536 ÷ whole MSE):
* mean: 97.7%
* Gaussian: 97.5%
* median: ≈ 100% (1263.44 of 1263.88)
* bilateral: 15%

**Observation:** the bilateral filter changes the most pixels (21,041) but by tiny amounts (MSE 3.44). The median changes the fewest (2,563), but those are mainly the line pixels, changed by 200 levels.

### Later: full noise removal experiments
_Stage 6 (experiment runner) will apply the filters, with several settings, to all nine baselines G001–G009._

---

## Separate section: metric validation runs (MV###), Stage 2

These are **not** noise removal experiments. They are controlled, hand-checkable
changes used only to check that MSE and PSNR are calculated correctly.
No noise model and no filter were involved.

* **Date:** 2026-11-16
* **Command:** `python -m demos.demo_metrics --save-figure` (run from the project root; the same numbers were also printed by a run without `--save-figure`)
* **Code:** `src/metrics.py` (`mse`, `psnr`), compared with `cv2.PSNR` (OpenCV 5.0.0)
* **Input image:** `data/input/synthetic_test_image.png` (256 × 256, uint8; see report_notes §2). The original array was checked to be unchanged after the run ("Original image unchanged: True").
* **Random seed:** not applicable (all changes are deterministic)

| ID | Original | Processed (controlled change) | Changed pixels | MSE | PSNR (ours) | cv2.PSNR | Hand calculation | Observation |
|---|---|---|---|---|---|---|---|---|
| MV001 | 2×2 array [[10,20],[30,40]] | [[12,20],[30,36]] | 2 / 4 | 5.0 | 41.1411 dB | not run | (4+0+0+16)/4 = 5; 10·log₁₀(65025/5) = 41.1411 | Direct uint8 subtraction gave [[254,0],[0,4]] (silent wrap-around) |
| MV002 | synthetic_test_image.png | Case A: row 185, columns 30, 39, …, 111 (10 pixels), 128 → 178 | 10 / 65,536 | 0.3815 | 52.3162 dB | 52.3162 dB | 10·50²/65536 = 0.3815 | Dots clearly visible, yet PSNR is high |
| MV003 | synthetic_test_image.png | Case B: every pixel +10 (0 pixels clipped) | 65,536 / 65,536 | 100.0000 | 28.1308 dB | 28.1308 dB | 10² = 100; 10·log₁₀(650.25) = 28.1308 | Looks almost unchanged, yet PSNR is much lower than case A |
| MV004 | synthetic_test_image.png | Identical copy | 0 / 65,536 | 0.0 | infinity | 361.2020 dB | MSE = 0 → PSNR = ∞ | OpenCV's epsilon gives a finite value |

**Saved output:** `results/figures/stage2_metric_validation.png` (300 dpi; original, case A, case B with MSE/PSNR in the titles). Git ignores it, so regenerate it with the command above.
