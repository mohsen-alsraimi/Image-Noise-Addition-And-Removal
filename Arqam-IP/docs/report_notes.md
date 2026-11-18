# Report Notes

_Last updated: 2026-11-17 (after Stage 5D)_

These are **preparation notes**, not the report. Each section separates:

* **Confirmed:** facts from course material, verified sources, or our own code and tests.
* **Proposed:** planned methodology, not implemented yet.
* **Implemented:** working code that has been tested.
* **Not yet collected:** results that do not exist yet. Never fill these in without real program runs.

---

## 0. Professor's template: verified locally

`course_material/template.docx` was read on 2026-11-16 (text, tables, page setup and images extracted with a read-only script).

### Confirmed: structure and instructions in the template
| Section in template | Instruction text in the template |
|---|---|
| Abstract | Example paragraph only (about image processing in daily life) |
| Table of contents | A manually typed table: 1 Introduction, 2 Datasets, 3 Methods, 4 Results and discussion, 5 Conclusion, with page numbers |
| 1 Introduction | Example paragraph only |
| 2. Datasets (if applicable) | "If you use image data." |
| 3. Methods | "Please use draw.io for ploting", followed by an example figure with the caption "Figure1: Block diagram of the proposed work [1]." |
| 4. Results and discussion | "Simulation results should be here", followed by two example images |
| 5. Conclusion | "Summarized the whole project" |
| References | "Please put a least five Reference", followed by one example reference in IEEE style, numbered "[1]." in a two-column table |
| Index | "Please put your source code here if you it is applicable" |

### Confirmed: formatting facts read from the file
* Page size US Letter (8.5 × 11 in), all margins 1 inch.
* Theme body font **Aptos**, default size 12 pt. Some text uses 10 pt.
* Paragraph alignments used: justified and centred.
* Section titles are bold text in the Normal style, not Word heading styles. The table of contents is a typed table, not an automatic Word TOC.
* Figure caption pattern: "Figure1: <description> [ref]." placed with the figure.
* **The example reference is IEEE style** (authors, "title," venue, vol., pp., year, doi), so our provisional IEEE choice matches the template.

### Confirmed: template content that is NOT ours
* The abstract and introduction paragraphs are example text.
* The three images are examples: a block diagram for a speech-dataset pipeline (dashed rounded group boxes, icons, arrows), and two sampling demonstrations. They show the expected *style* only.
* Example reference [1] is a speech-detection paper and is not related to our topic.

### Not specified in the template
Page limit, deadline, number of figures, font requirements for the student's text, and heading numbering style (the template itself mixes "1 Introduction", "2.Datasets" and "3. Methods"). Do not invent rules for these. Ask the professor if needed.

> **Course context (lecture 1, PDF page 2):** Final exam 40%, Project 60%.
> The project part is Project work 15%, Written report 25%, Oral presentation 10%,
> Class contribution 10%.

---

## Abstract
* **Confirmed:** nothing yet.
* **Not yet collected:** main findings. Write the abstract only after the experiments (Stage 7).
* Material to collect: one-sentence problem, method (noise types, filters, metrics), key numerical finding, conclusion.

## Table of Contents
* The template uses a typed table (see section 0). Fill it in at the end.

---

## 1. Introduction

### Confirmed (lecture 1, "Simple Introduction to Digital Image Processing")
* A signal is a function that carries information. An image is a **2-D signal f(x, y)** (PDF p. 4).
* An image is a 2-D collection of **pixels**. Each pixel stores an intensity or colour value (PDF p. 5).
* **Pixel** = picture element, the smallest addressable element of a digital image (PDF p. 10).
* An 8-bit grayscale pixel uses 1 byte. The values range from **0 (black) to 255 (white)** (PDF pp. 6–8).
* Pixels in rows and columns form an **intensity matrix I(x, y)**. Example: a 512×512 image has 262,144 pixels, which is 256 KiB of raw 8-bit storage (PDF p. 9).
* "Spatial" refers to position (x, y) in the image (PDF p. 11).
* Image types: grayscale (1 value per pixel), RGB (3 values), RGBA (4 values, alpha = transparency) (PDF p. 13).
* Image processing changes an input image into a useful output image, for example by improving quality, **removing noise** or finding edges (PDF p. 14).
* Workflow: Acquire → Enhance → Analyze → Output (PDF p. 15).
* Sampling decides how many pixels represent the image. Quantization decides how many intensity levels are available (PDF pp. 17–18).
* **"Noise is unwanted variation in an image. Filtering can reduce noise and produce a cleaner image."** (PDF p. 20, "Noise Removal")
* Spatial filtering: "A small filter moves across the image. It uses neighboring pixels to smooth, sharpen, or detect patterns." The example uses a 3×3 neighbourhood (PDF p. 21).
* Applications listed: medical imaging, security, satellites, photography, industry, biology (PDF p. 27).

### Confirmed (verified literature, for the "how do we measure quality" part)
* Wang and Bovik [V5]: MSE "remains the standard criterion for the assessment of signal quality and fidelity" and is "the method of choice for comparing competing signal processing methods", even though it "has been widely criticized for serious shortcomings, especially when dealing with perceptually important signals such as speech and images."

### Confirmed (verified literature): why image noise matters
* **Where noise comes from.** Fan et al. [V8]: "Owing to the influence of environment, transmission channel, and other factors, images are inevitably contaminated by noise during acquisition, compression, and transmission, leading to distortion and loss of image information." Also: "the images captured by modern cameras are inevitably degraded by noise, which leads to deteriorated visual image quality."
* **Why it is a problem for later processing.** Fan et al. [V8]: "With the presence of noise, possible subsequent image processing tasks, such as video processing, image analysis, and tracking, are adversely affected. Therefore, image denoising plays an important role in modern image processing systems."
* **Impulse (salt-and-pepper) noise.** Ma and Nie [V9]: "Digital images can be corrupted by impulse noise due to bit errors in the process of image acquisition and transmission." It "can not only significantly deteriorate the quality of an image, but also bring some difficulty to the subsequent image analysis such as image segmentation, edge detection and classification."
* **Speckle noise.** Yu et al. [V10]: "ultrasound images are inevitably corrupted by speckle noise due to the coherent imaging mechanism", which "reduces the sharpness of image details and complicates the diagnosis". Denoising as "a pre-processing step … will also benefit image post-processing tasks such as image segmentation, classification and registration."
* **The core difficulty of denoising.** Fan et al. [V8]: "since noise, edge, and texture are high frequency components, it is difficult to distinguish them in the process of denoising and the denoised images could inevitably lose some details." Also: "Normally, spatial filters eliminate noise to a reasonable extent but at the cost of image blurring, which in turn loses sharp edges." The Stage 4 results show exactly this trade-off on our test image (§4.2).
* **What a good denoiser should do.** Fan et al. [V8]: "flat areas should be smooth, edges should be protected without blurring, textures should be preserved, and new artifacts should not be generated." This list matches the regions of our synthetic image (flat patch, square/circle edges, line pattern).

### Proposed (still to write)
* Project objectives (from `project_overview.md`).
* Optional: more real-world examples, each needing its own verified source.

---

## 2. Datasets
Template instruction: "If you use image data." We do, so this section is required.

### Confirmed / Implemented
**Image D1: `synthetic_test_image.png`**
| Field | Value |
|---|---|
| Source | Generated by our own code: `src/synthetic_image.py`, function `create_synthetic_test_image()` |
| Location | `data/input/synthetic_test_image.png` (created by the demo; it can always be regenerated) |
| Licence | Our own work, no restrictions |
| Dimensions | 256 × 256 pixels (measured by `demos/demo_image_loading.py` on 2026-11-16) |
| Colour format | 8-bit grayscale (`uint8`), PNG (lossless) |
| Pixel statistics (measured) | min 20, max 240, mean 120.59, std 66.64, 155 distinct intensity levels |
| Preprocessing | None. It is created directly as grayscale. |
| Why selected | We know every pixel exactly, and it contains the structures that make filters behave differently (listed below). |

Regions (positions are fractions of the image height/width):
* Background: horizontal gradient from 50 (left) to 200 (right), a smooth change.
* Dark square, value 20: rows 0.10–0.42, columns 0.08–0.40. It has straight sharp edges.
* Bright circle, value 235: centre (0.26 h, 0.70 w), radius 0.16 × size. It has curved edges.
* Flat patch, value 128: rows 0.55–0.90, columns 0.08–0.45. Constant intensity, so noise is easy to see.
* Fine detail: 1-pixel-wide lines (value 240) every 4 columns on a dark block (value 40), rows 0.55–0.90, columns 0.55–0.92.
* Values avoid exactly 0 and 255, so salt-and-pepper noise pixels can be identified later.

How D1 is used in Stage 3 (confirmed by the actual runs):
* D1 is the **clean reference** for every noise experiment (E001–E009). All MSE/PSNR values are measured against it.
* Because D1 contains no 0 or 255, every salt-and-pepper pixel that is *selected* is also *visibly changed* (measured: 1,311 / 3,277 / 6,554 selected = changed).
* The dark square (value 20) next to the bright circle (value 235) makes the brightness dependence of speckle noise easy to see.
* The values 20 and 240 lie close to the limits 0 and 255, so strong Gaussian/speckle noise is clipped there. This is visible in the results (see §4.1).
* Derived noisy images are stored in `results/noisy/` with names like `synthetic_test_image__gaussian_sigma25__seed42.png`. They can be regenerated exactly, so Git ignores them.

How D1 is used in Stage 4 (confirmed by the actual runs):
* The Stage 4 demo also measures MSE inside two regions of D1. Each is 8,455 px.
  * The **flat patch** (rows 141–229, columns 20–114): noise removal without edges.
  * The **line block** (rows 141–229, columns 141–235; 24 one-pixel lines): loss of fine detail.
  * The demo checks that these regions still contain exactly the expected values (128, or 40 and 240).
* The 1-pixel lines turned out to dominate the whole-image MSE of the smoothing filters (§4.2). This is an important property of D1 that must be stated whenever its filter results are reported.

### Proposed
**Image D2: student's own photograph** (not yet provided). Record these fields:
| Field | Value |
|---|---|
| Filename | _TBD_ |
| Source | Taken by the student (record device and date) |
| Licence | Own work |
| Original dimensions / format | _TBD (measure with the demo)_ |
| Preprocessing | Conversion to grayscale during loading (Y = 0.299R + 0.587G + 0.114B [V1]). Whether to resize is _undecided_. |
| Why selected | Real-world content (textures, natural edges) to complement the synthetic image |

Notes for D2: avoid identifiable people (privacy). Check whether phone photos
appear rotated after loading (EXIF orientation); this is not tested yet.

**Possible D3:** a standard test image, used only with a verified source and licence.

---

## 3. Methods
Template instruction: "Please use draw.io for ploting" (block diagram, see 3.5).

### 3.1 Image loading and representation (Stage 1)
* **Implemented:** `src/image_io.py`
  * `load_grayscale_image(path)`: reads the file bytes (`numpy.fromfile` [V3]) and decodes them as grayscale (`cv2.imdecode` with `IMREAD_GRAYSCALE` [V2]). The result is a 2-D `uint8` NumPy array of shape (height, width), the intensity matrix I(x, y).
  * Colour conversion: OpenCV documents the luminance formula **Y = 0.299·R + 0.587·G + 0.114·B** [V1]. With `IMREAD_GRAYSCALE` the conversion happens inside the image codec [V2]. In our test, a pure green pixel gave 149 through decoding and 150 through `cv2.cvtColor`, so results can differ by 1 because of rounding.
  * Validation: clear errors for a wrong path type, a missing file, a folder, an unsupported extension, an empty file, or undecodable content.
  * `validate_grayscale_image(image)`: the image must be a non-empty 2-D `uint8` array.
  * `get_image_info(image)`: height, width, total pixels, dtype, bit depth, raw size in bytes, min, max, mean, standard deviation, and number of distinct intensity levels.
    * Mean: μ = (1/N) Σ I(x, y)
    * Standard deviation: σ = √[(1/N) Σ (I(x, y) − μ)²] (population form, NumPy default)
    * Computed in `float64` to avoid `uint8` overflow.
  * `save_grayscale_image(image, path, overwrite=False)`: refuses to overwrite existing files by default.
* **Implemented:** `src/synthetic_image.py`, which builds the D1 image with NumPy array slicing and a circle mask (y − cy)² + (x − cx)² ≤ r².
* **Design decisions:** see D1–D8 in `project_overview.md`.
* **Limitations:** grayscale only; 16-bit images are converted to 8-bit by the decoder.
* **EXIF orientation (tested in Stage 5A):** a JPEG with EXIF orientation 6 ("rotate 90°") and stored size 32×16 was loaded as 16 wide × 32 high. This held both for `load_grayscale_image` and for the web interface's in-memory decoding, so phone photos appear upright.

### 3.2 Image quality metrics: MSE and PSNR (Stage 2)

**Implemented:** `src/metrics.py`. Tests: `tests/test_metrics.py` (31 tests).

**Mean Squared Error**

  MSE = (1 / (M·N)) · Σₓ Σᵧ [ I(x, y) − K(x, y) ]²

| Symbol | Meaning |
|---|---|
| I(x, y) | Pixel of the **original** (clean reference) image at row x, column y |
| K(x, y) | Pixel of the **processed** image (noisy or filtered) at the same position |
| M, N | Number of rows and columns; M·N = total number of pixels |
| Σ | Sum over all pixel positions |

* MSE = 0 means the images are identical. Larger MSE means larger average difference.
* The unit is "intensity levels squared". For 8-bit images the largest possible value is 255² = 65025.
* Squaring makes every difference positive and penalises large differences much more than small ones.

**Peak Signal-to-Noise Ratio**

  PSNR = 10 · log₁₀( MAX² / MSE )   [dB]

| Symbol | Meaning |
|---|---|
| MAX | Largest possible pixel value. For 8-bit images MAX = 2⁸ − 1 = **255** |
| MSE | Mean squared error from above |
| dB | Decibels, a logarithmic scale for ratios: every +10 dB means the error power is 10 times smaller |

* This is the same formula documented by OpenCV [V4] ("R is the maximum integer value of depth (e.g. 255 in the case of CV_8U data)") and by Wang and Bovik [V5] ("L is the dynamic range of allowable image pixel intensities").
* Higher PSNR means the processed image is closer to the original. Identical images have MSE = 0, so PSNR is infinite.
* PSNR is a rescaled MSE. Wang and Bovik [V5]: PSNR "is useful if images having different dynamic ranges are being compared, but otherwise contains no new information relative to the MSE." We report both because PSNR in dB is the common way to present results.

**Why float64 arithmetic**
* Images are `uint8` (0–255). In NumPy, uint8 array subtraction **wraps around silently**: 10 − 12 gives 254, not −2 (−2 + 256 = 254). We observed this, with no warning, in `demos/demo_metrics.py`. NumPy documents that "the fixed size of NumPy numeric types may cause overflow errors when a value requires more memory than available in the data type" [V6].
* Squared differences can reach 65025, and their sum over a 1000×1000 image reaches 6.5 × 10¹⁰. Both are far too large for uint8.
* Therefore both images are converted to `float64` **before** subtracting. A test proves that the uint8 version would give the wrong answer (a deliberately buggy uint8 version failed 6 of our tests).

**Implementation (how the code works)**
1. `_validate_image_pair`: both inputs must be valid 8-bit grayscale images (reusing Stage 1's `validate_grayscale_image`) with the **same shape**. Errors name the wrong argument ("original" or "processed").
2. `mse`: convert both to float64, subtract, square, average with `np.mean`, and return a Python `float`.
3. `psnr`: call `mse` (which also validates). If MSE is exactly 0.0, return `math.inf`; otherwise return 10·log₁₀(255² / MSE).
4. No new libraries: only NumPy and Python's `math`.

**Design decision: explicit infinity instead of cv2.PSNR**
* The OpenCV source code (`modules/core/src/norm.dispatch.cpp`, 4.x and 5.x branches) computes `20*log10(R/(diff+DBL_EPSILON))`, where diff = √MSE and DBL_EPSILON ≈ 2.22 × 10⁻¹⁶. Adding this tiny number avoids division by zero, but for identical images it gives 20·log₁₀(255 / 2.22×10⁻¹⁶) = **361.20 dB**, a finite number. We measured exactly this value.
* We return true infinity, so "identical" is never confused with "very good". It also prints clearly as "infinity" in tables.
* For non-identical images our PSNR equals `cv2.PSNR` to at least 6 decimal places (5 random image pairs and a modified synthetic image, tested).

**How the metrics will be used later (proposed, Stages 6–7)**
* For every noise setting: PSNR(original, noisy) is the **baseline**. Then PSNR(original, filtered) for each filter, always on the same noisy image.
* A filter helps if its PSNR is higher (MSE lower) than the noisy baseline. Filters are compared by their PSNR under identical conditions.

**Limitations**
* MSE/PSNR compare pixel by pixel. They ignore *where* the errors are and *what kind* they are. Wang and Bovik [V5] show that distorted versions of an image can have "nearly identical" MSE "even though the same images present dramatically (and obviously) different visual quality."
* Our own validation shows the same effect (section 4.0): a uniform +10 brightness shift (hard to notice) scored a much **lower** PSNR (28.13 dB) than 10 clearly visible white dots (52.32 dB).
* PSNR needs a clean reference image. This is fine for our controlled experiments, but not possible for real photos where no clean version exists.
* A possible extension is SSIM [C3, bibliographic data verified], which is designed to compare structure.

### 3.3 Noise models (Stage 3)

**Implemented:** `src/noise.py`. Tests: `tests/test_noise.py` (38 tests). Demo: `demos/demo_noise.py`.

#### 3.3.0 Rules shared by all three noise functions
* **Input:** a valid 2-D `uint8` image (checked with Stage 1's `validate_grayscale_image`), a strength parameter, and an optional `seed`.
* **Output:** a **new** 2-D `uint8` array. The input is never modified (tested, including that the result shares no memory with the input).
* **Strength validation:** it must be a real number (int or float), finite (not NaN or ±∞) and ≥ 0. Booleans are rejected (`True` would otherwise count as 1). Salt-and-pepper `amount` must also be ≤ 1. Strength 0 returns an unchanged copy.
* **Random numbers:** `numpy.random.default_rng(seed)` creates a private Generator for each call [V7]. NumPy documents that "Our RNGs are deterministic sequences and can be reproduced by specifying a seed integer" [V7], and that the old `np.random.*` functions use "a single global RandomState instance" [V7]. We never touch that global state (tested).
  * Same image + strength + seed always gives exactly the same output (tested; reruns of the demo produced byte-identical PNG files).
  * The seed must be `None` or a non-negative integer, as NumPy requires [V7]. We reject booleans ourselves, because NumPy would accept `True` as a seed (we observed this).
* **Back to uint8 (Gaussian and speckle):** the noisy values are calculated in `float64`, then (1) rounded to the nearest whole number with `np.rint` (exact halves go to the even number, e.g. 2.5 → 2), (2) clipped to [0, 255], and (3) converted to `uint8`.
  * Without clipping, `uint8` conversion would wrap around (270 → 14). A deliberately buggy wrap-around version failed our clipping tests.
  * Without rounding, `astype` would *truncate* (cut off the decimals), which biases every value downwards. A buggy truncating version failed our mean/standard-deviation test.
  * **Consequence of clipping:** the noise near 0 and 255 is cut off, so the real error is *smaller* than the noise model predicts (measured in §4.1).

#### 3.3.1 Gaussian noise: `add_gaussian_noise(image, sigma, seed=None)`
* **Definition:** additive noise. An independent random value from a normal (bell-shaped) distribution is added to every pixel. This is the standard additive white Gaussian noise (AWGN) model of the denoising literature: "y = x + n, where y is the observed noisy image, x is the unknown clean image, and n represents additive white Gaussian noise (AWGN) with standard deviation σ" [V8].
* **Model:** g(x, y) = f(x, y) + n(x, y),  n(x, y) ~ N(0, σ²), then rounded and clipped to [0, 255].

| Symbol | Meaning |
|---|---|
| f(x, y) | original (clean) pixel |
| g(x, y) | noisy pixel |
| n(x, y) | random noise value for this pixel, independent of all other pixels |
| N(0, σ²) | normal distribution with mean 0 and variance σ² |
| σ (sigma) | standard deviation in intensity levels (0–255 scale) |

* **Parameter σ:** controls **how much** pixels change, **not how many**. Almost every pixel changes (E001: 62,782 of 65,536 = 95.8%; the rest received noise that rounded to 0). About 68% of noise values lie within ±σ and about 95% within ±2σ (property of the normal distribution). Values used: σ = 10, 25, 50.
* **Implementation:** `rng.normal(loc=0.0, scale=sigma, size=image.shape)` [V7, documented as "Draw random samples from a normal (Gaussian) distribution", with `scale` = "Standard deviation"], added to the float64 image, then rounded, clipped and converted.
* **Verified by tests:** on a flat 400×400 image of 128 with σ = 10, the measured noise mean is ≈ 0 and its standard deviation ≈ 10. MSE ≈ σ² when there is no clipping. The output exactly matches the formula.
* **Limitations:** a simplified model. Real camera noise also depends on brightness and can be correlated between neighbouring pixels. Clipping makes the effective noise smaller than σ near black and white.

#### 3.3.2 Salt-and-pepper (impulse) noise: `add_salt_and_pepper_noise(image, amount, seed=None)`
* **Definition:** a kind of impulse noise where "each corrupted pixel takes either the maximum or minimum gray level" [V9]: 255 = salt (white), 0 = pepper (black). All other pixels stay **exactly** unchanged.
* **Model:**
  g(x, y) = 255 if (x, y) ∈ S_salt;  0 if (x, y) ∈ S_pepper;  f(x, y) otherwise,
  where S_salt and S_pepper are disjoint random sets of pixel positions.
* **Parameter `amount`:** the fraction of pixels **selected** (0.02 = 2%). Values used: 0.02, 0.05, 0.10.
* **Rounding rule (our design, tested):**
  * selected = round-half-up(amount × N) = ⌊amount × N + 0.5⌋, where N is the number of pixels
  * salt = ⌊selected / 2⌋; pepper = selected − salt (pepper gets the extra pixel when the count is odd)
  * Our 256×256 image: 0.02 → 1310.72 → **1,311** (655 + 656); 0.05 → 3276.8 → **3,277** (1,638 + 1,639); 0.10 → 6553.6 → **6,554** (3,277 + 3,277).
* **Implementation:** `rng.choice(N, size=selected, replace=False)` picks `selected` *different* flat positions [V7: `replace` = "Whether the sample is with or without replacement"]. `np.unravel_index` converts them to (row, column). The positions come in random order, so the first `salt` become 255 and the rest 0.
* **Selected vs. visibly changed:** a selected pixel that already had the value it is given does not change (255 + salt stays 255; 0 + pepper stays 0). So changed ≤ selected. In a test with an all-white image and amount 0.5: 5,000 selected, but only the 2,500 pepper pixels changed. In our synthetic image (no 0 or 255) the two counts are equal.
* **Design choice vs. other libraries:** scikit-image decides for **each pixel independently** whether it is corrupted (a Bernoulli trial with probability `amount`) [V11], so the number of noisy pixels is only *approximately* amount × N. We select an **exact** number without replacement, so every run with the same amount corrupts exactly the same number of pixels. That makes the experiments easier to compare and to check by hand.
* **Limitations:** real impulse noise can also take random values (not only 0/255). Ma and Nie call this "random-valued noise" [V9]. That type is not modelled here.

#### 3.3.3 Speckle (multiplicative) noise: `add_speckle_noise(image, variance, seed=None)`
* **Definition:** noise whose size is proportional to the pixel's own brightness ("signal-dependent" [V10]).
* **Model:** g(x, y) = f(x, y) + f(x, y) · n(x, y),  n ~ N(0, variance), then rounded and clipped.
  * This is the same form scikit-image calls 'speckle': "out = image + n * image, where n is Gaussian noise with specified mean & variance" [V11].
  * Yu et al. [V10] write the more general form u = v + v^γ · η, with η zero-mean Gaussian with variance σ². Our model is the case **γ = 1**. They note that γ = 0.5 is often used for ultrasound images.
* **Parameter `variance`:** the variance of n, with no unit (it is relative to brightness). The standard deviation of n is √variance: 0.01 → 0.1 (±10% of the pixel value), 0.04 → 0.2, 0.10 → 0.316. Values used: 0.01, 0.04, 0.10.
* **Why brightness matters:** the change f·n has standard deviation f·√variance. With variance 0.01, a pixel of 40 changes by about ±4 levels and a pixel of 160 by about ±16 (tested: measured 4.0 and 16.0). A pixel of 0 never changes (tested).
* **Implementation:** `rng.normal(0, sqrt(variance), shape)`, multiplied by the float64 original, added to it, then rounded, clipped and converted.
* **Limitations:** this is a *simplified* multiplicative model, **not** a complete physical model of radar or ultrasound speckle. Real speckle comes from coherent imaging and is described by other distributions (e.g. Fisher–Tippett or Gamma, as noted in [V10]), may use γ ≠ 1 [V10], and is spatially correlated. Our noise is independent from pixel to pixel.

### 3.4 Noise-removal filters (Stage 4)

**Implemented:** `src/filters.py`. Tests: `tests/test_filters.py` (43 tests). Demo: `demos/demo_filters.py`.

All four are **spatial filters**: a small window moves over the image, and each output pixel is computed from the pixels in the window around it. Lecture 1 (PDF p. 21) describes this as "A small filter moves across the image. It uses neighboring pixels to smooth, sharpen, or detect patterns." Fan et al. [V8] divide spatial filters into linear filters (mean, Gaussian) and non-linear filters (median, bilateral).

#### 3.4.0 Rules shared by all four filter functions (implemented and tested)
* **Input:** a valid 2-D `uint8` image, checked with Stage 1's `validate_grayscale_image`.
* **Output:** a **new** `uint8` array of the same shape. The input is never modified (tested, also for kernel size 1, where the result must still be a separate copy).
* **Wrappers, not reimplementations:** each function validates its parameters and then calls one OpenCV function [V12]. A test compares every wrapper with a direct OpenCV call and requires identical output.
* **Window sizes:** odd integers (3 = 3×3 window), so the window has a centre pixel.
  * The mean, Gaussian and median `kernel_size` must be ≥ 1, where 1 means "no change".
  * The bilateral `diameter` must be ≥ 3.
  * Booleans and floats are rejected.
  * Note: OpenCV's `cv2.blur` would silently accept an even size (observed), so we check ourselves.
* **Border handling (what happens at the image edge):** part of the window lies outside the image, so the missing pixels must be invented.
  * Mean, Gaussian and bilateral use **`BORDER_REFLECT_101`**: mirror the image without repeating the edge pixel ("gfedcb|abcdefgh|gfedcba" [V12]). This is OpenCV's default; we pass it explicitly.
  * The median filter has no border option. OpenCV "uses #BORDER_REPLICATE internally" [V12]: the edge pixel is repeated ("aaaaaa|abcdefgh").
  * Both behaviours are confirmed by known-answer tests on tiny images.
* **Deterministic:** the same input and settings always give the same output (tested). No randomness is involved.

#### 3.4.1 Mean (box) filter: `apply_mean_filter(image, kernel_size=3)` → `cv2.blur`
* **Idea (simple English):** replace every pixel by the **average** of the pixels in its window.
* **Formula:** g(x, y) = (1 / k²) · Σ f(x + i, y + j) over the k × k window; every weight = 1/k² [V12: "normalized box filter"].
* **Parameter:** `kernel_size` k. Larger k = more smoothing and more blur.
* **Advantages:** simplest filter; fast; very effective against small random noise in flat regions.
  * Theory (our derivation from basic statistics, for independent noise): averaging 9 values divides the noise variance by 9.
* **Disadvantages:** blurs edges and thin lines, because pixels from both sides of an edge are averaged. Fan et al. [V8]: mean filtering "has been adopted for Gaussian noise reduction, however, it can over-smooth images with high noise". An impulse (255 or 0) is not removed but spread into a gray 3×3 blob.
* **Expected behaviour:** good for Gaussian noise in flat areas, poor for salt-and-pepper and for fine detail.
* **Tested known answers:**
  * one pixel of 90 among zeros → 10 in its 3×3 neighbourhood (90/9)
  * step edge 30 | 120 → 60 and 90 next to the edge
  * corner pixel → 10 with REFLECT_101 (REPLICATE would give 40)

#### 3.4.2 Gaussian filter: `apply_gaussian_filter(image, kernel_size=3, sigma=0)` → `cv2.GaussianBlur`
* **Idea:** a **weighted** average. Pixels near the centre count more than pixels far away; the weights follow a bell curve. Tomasi and Manduchi [V13]: "Gaussian low-pass filtering computes a weighted average of pixel values in the neighborhood, in which the weights decrease with distance from the neighborhood center."
* **Weights:** G_i ∝ exp(−(i − (k−1)/2)² / (2σ²)), normalised to sum to 1 [V12]. The 2-D weights are the product of a horizontal and a vertical 1-D kernel.
* **Parameters:**
  * `kernel_size` k: the window size.
  * `sigma` σ, in pixels: the width of the bell. Larger σ = stronger smoothing (tested).
  * `sigma = 0`: OpenCV chooses σ from k [V12]. For k = 3 the OpenCV source uses the fixed weights **[1/4, 1/2, 1/4]**, so the 3×3 weights are 1/4 (centre), 1/8 (direct neighbours) and 1/16 (diagonal neighbours). We confirmed this with a test (an impulse of 160 gives exactly 40 / 20 / 10).
  * We pass the same σ for x and y, as OpenCV recommends [V12].
* **Advantages:** smoother, more natural blur than the mean filter; it blurs edges a little less at the same kernel size, because the centre pixel has more weight (tested on a step edge: 53 instead of 60 next to the edge).
* **Disadvantages:** still a linear filter, so it still blurs edges and lines [V13: edges "are consequently blurred by linear low-pass filtering"]. With k = 3 it removes less noise than the mean filter.
  * Theory: the noise variance is multiplied by Σw² = 0.375² ≈ 0.141 instead of 1/9 ≈ 0.111.
* **Expected behaviour:** like the mean filter, slightly less smoothing and slightly less blur.

#### 3.4.3 Median filter: `apply_median_filter(image, kernel_size=3)` → `cv2.medianBlur`
* **Idea:** sort the pixels in the window and take the **middle** value. Ma and Nie [V9]: "The standard median filter uniformly replaces the gray level of each noise corrupted pixel by the median gray level of its neighborhood pixels."
* **Parameter:** `kernel_size` (odd ≥ 1). For uint8 images every odd size is supported; sizes above 5 work only for 8-bit images [V12], which is all we accept.
* **Advantages:**
  * It is the classic answer to salt-and-pepper noise: "one of the most popular nonlinear filters for the removal of salt-and-pepper noise" [V9].
  * An isolated 0 or 255 is an extreme value, so it is never the middle value. It is dropped, not averaged in (tested: an isolated salt and pepper pixel disappear completely).
  * It keeps straight step edges perfectly sharp (tested).
  * It never creates new gray values: the output contains only values from the window (tested).
* **Disadvantages:**
  * It **erases details thinner than half the window**: a 1-pixel line is only 3 of 9 values in a 3×3 window, so the median ignores it (tested).
  * Ma and Nie [V9]: effective "in the case of low noise density"; for high densities larger windows are needed, which "leads edges and details to be blurred".
  * Expected to be less effective than averaging for Gaussian noise. Our reasoning, not from a cited source: averaging combines all 9 values, while the median keeps only the middle one. This is consistent with the flat patch in §4.2 (107.46 vs 80.14).
* **Expected behaviour:** excellent for salt-and-pepper in flat areas; destroys thin lines; moderate for Gaussian noise.

#### 3.4.4 Bilateral filter: `apply_bilateral_filter(image, diameter=5, sigma_color=50, sigma_space=50)` → `cv2.bilateralFilter`
* **Idea:** an average of pixels that are **near and similar**. Tomasi and Manduchi [V13]: "It replaces the pixel value at x with an average of similar and nearby pixel values." Each neighbour gets two weights multiplied together:
  * a **closeness** weight, which falls with distance (like the Gaussian filter, controlled by `sigma_space`)
  * a **similarity** weight, which falls with the difference in intensity (controlled by `sigma_color`)
* **Why it keeps edges:** on the bright side of an edge, the dark pixels are very different, so they get almost no weight. "The filter replaces the bright pixel at the center by an average of the bright pixels in its vicinity, and essentially ignores the dark pixels" [V13]. In flat regions all neighbours are similar, so it "acts essentially as a standard domain filter, averaging away the small … differences … caused by noise" [V13].
* **Parameters [V12]:**
  * `diameter` d: the size of the square neighbourhood. In OpenCV the radius = d // 2, at least 1 (source code), so even values and 1 or 2 do not mean what they say. We accept only odd d ≥ 3.
  * `sigma_color`, in intensity levels: neighbours that differ by much more than this are ignored. Larger = more mixing.
  * `sigma_space`, in pixels: how fast the weight falls with distance. Larger = far neighbours count more.
  * OpenCV silently replaces sigmas ≤ 0 by 1 (observed), so we require > 0.
* **Our setting d = 5, sigma_color = sigma_space = 50:**
  * OpenCV: "it is recommended to use d=5 for real-time applications". "For simplicity, you can set the 2 sigma values to be the same". Values "< 10" have little effect, values "> 150" make the image "cartoonish" [V12].
  * 50 is a middle value between those limits. It was **not** tuned for the best PSNR.
* **Advantages:** "edge-preserving, and noise-reducing" [V8]; keeps edges and thin lines.
  * Tested: a 50 | 200 step edge is unchanged with sigma_color = 10, but blurred with sigma_color = 1000.
* **Disadvantages:**
  * Slow: "very slow compared to most filters" [V12]; "One issue concerning the bilateral filter is its efficiency" [V8].
  * It cannot tell an impulse from an edge. A salt pixel differs from its neighbours by more than sigma_color, so the neighbours get little weight and the impulse largely **survives** (our expectation from the formula, confirmed in §4.2).
  * More parameters to choose.
* **Expected behaviour:** good for Gaussian noise while keeping edges and lines; weak against salt-and-pepper.

#### 3.4.5 Summary table (theory and documentation; measured results are in §4.2)
| Filter | Type | Main parameter(s) | Expected strength | Expected weakness |
|---|---|---|---|---|
| Mean | linear | kernel_size | Gaussian noise in flat areas | blurs edges and lines; spreads impulses |
| Gaussian | linear | kernel_size, sigma | Gaussian noise, slightly less blur than mean | still blurs edges and lines |
| Median | non-linear | kernel_size | salt-and-pepper noise; keeps step edges | erases thin lines and fine detail |
| Bilateral | non-linear | diameter, sigma_color, sigma_space | Gaussian noise while keeping edges and lines | slow; impulses survive |

**Limitations of the filter module:** grayscale uint8 only; square windows only (OpenCV also allows rectangles); one border mode per filter, not configurable; no adaptive or switching median filters (those are mentioned in [V9] but out of scope).

### 3.5 Block diagram workflow (for draw.io, Stage 8)
Status of each box: [I] = implemented, [P] = proposed. The template's example diagram uses dashed rounded group boxes, icons and arrows, with the caption "Figure1: Block diagram of the proposed work".
```
[I] Input image file (data/input)
      │  load_grayscale_image + validation
      ▼
[I] Clean grayscale image I (uint8 matrix)  ───────────────┐
      │  add noise (Gaussian σ / salt-and-pepper amount /    │ reference
      │  speckle variance, fixed seed)            [I]        │
      ▼                                                     │
[I] Noisy image  ──► [I] Filter (mean / Gaussian / median / bilateral, parameters)
      │                         │
      │ baseline                ▼
      │                  [I] Filtered image
      ▼                         ▼
[I] Metrics vs. clean image: MSE, PSNR  ◄───────────────────┘
      ▼
[I] Results table (CSV) + comparison figure (PNG) for the Stage 4 demo
[P] Full results table and figures for all settings (Stage 6)
```
Interactive path (Stage 5): [I] photo upload in the web interface → [I] clean grayscale reference →
[I] noise (5B) → [I] filter (5C) → [I] MSE/PSNR, four-filter comparison, difference images (5D) → [P] downloads (5E).

### 3.6 Interactive web interface "Image Noise Lab" (Stage 5, Streamlit)
Purpose: let a user try the same pipeline on their **own photo** in a browser, without writing code.
It is built in five sub-stages (5A–5E). **5A (upload), 5B (noise), 5C (filters) and 5D (evaluation) are implemented so far.**

* **Implemented (5A):** `frontend/app.py` (page) and `frontend/image_processing.py` (helpers, no Streamlit import).
* **Processing workflow of an upload (5A):**
  1. The user chooses a PNG/JPEG file in the sidebar. Streamlit rejects other extensions and files over 20 MB.
  2. `load_uploaded_image(file_name, data)` checks:
     * the extension
     * that the file is not empty
     * the size limit (20 MB)
     * the file signature (the first bytes must be the PNG signature `89 50 4E 47 0D 0A 1A 0A`, or the JPEG signature `FF D8 FF`)
  3. The bytes are decoded **in memory** with `cv2.imdecode(..., IMREAD_GRAYSCALE)` [V2]. This is the same conversion as `load_grayscale_image` (§3.1), so the formula Y = 0.299·R + 0.587·G + 0.114·B applies [V1].
  4. The result is checked with `validate_grayscale_image` (2-D, `uint8`, not empty), then against the 25-megapixel limit.
  5. A colour decode (`IMREAD_COLOR`, converted from BGR to RGB) is made **only for display**. "Colour" means the red, green and blue channels are not all equal.
  6. The original bytes, the read-only grayscale reference and the read-only preview are stored together in one frozen record. Later steps (5B–5D) will use the grayscale reference as the clean image I for noise, filtering and metrics.
* **What the page shows (5A):**
  * the uploaded image and the grayscale reference side by side, at equal size
  * width, height, number of pixels and file size
  * pixel statistics from the existing `get_image_info`
  * notes for colour photos, transparent PNGs and large images
  * a clear error for damaged, empty or non-PNG/JPEG files
* **Display detail:** images are sent to the browser as PNG. Streamlit's default would re-encode them as JPEG with quality 90 (read in its source code), which changes pixel values. Streamlit also shrinks images wider than 1460 px **for display only**.
* **Verification (5A):**
  * 38 helper tests and 12 page tests with Streamlit's `AppTest`, which runs the page without a browser.
  * A mutation check: 14 deliberately broken versions of the helper were all detected.
  * A real server start: the health check answered "ok" on localhost.
  * The visual look in a browser has not been checked automatically.
* **Implemented (5B): connection to the noise models (§3.3).**
  * The sidebar section "2. Add Noise" appears after a valid upload. It has a noise-type selector, **one** strength slider for the selected type, and a "Random Seed" box (default 42).
  * The slider defaults are the Stage 3 baselines (E002, E005, E008):

    | Noise | Parameter in `src/noise.py` | Slider range (step) | Default |
    |---|---|---|---|
    | Gaussian | `sigma` | 0–100 (1) | 25 |
    | Salt-and-pepper | `amount` | 0–0.50 (0.01), displayed as % | 0.05 |
    | Speckle | `variance` | 0.00–0.25 (0.01) | 0.04 |

  * `make_noisy_image(reference, noise_type, strength, seed)` looks up the function in a table and calls `add_gaussian_noise`, `add_salt_and_pepper_noise` or `add_speckle_noise` from §3.3. It passes the strength and seed on **unchanged**, so the existing checks in `src/noise.py` apply.
    * The only extra rule: the app requires a seed (`None` is refused).
    * The result is a new, read-only `uint8` image of the same size. The clean reference is never changed.
  * For salt-and-pepper, the sidebar also shows the exact pixel counts from the existing `salt_and_pepper_counts` (for the 256×256 synthetic image at 5%: 3,277 pixels = 1,638 white + 1,639 black).
  * "How does this noise work?" explains the selected noise in simple words, with the same formulas as §3.3.
* **Reproducibility (5B):**
  * Streamlit reruns the whole script after every control change. The noise is generated again each time, always with the chosen seed (D14). So the same image, type, strength and seed give **identical pixels**, and there is no cache that could become stale.
  * The page shows a *pixel fingerprint*: the first 12 hex characters of SHA-256 over the image shape and pixels.
  * Page tests confirm all of the following:
    * the fingerprint equals that of a direct `src/noise.py` call for all three types
    * it changes with type, strength and seed
    * returning to seed 42 restores it
    * moving the zoom controls or a plain rerun does not change it
  * Helper tests reproduce the logged Stage 3 MSE values of E002, E005 and E008 (577.9914, 1058.0422, 636.4267) from the frontend helper. This was a test run, not a new experiment.
  * Measured speed: a full page rerun for a 12-megapixel photo took about 0.8 s (AppTest, this computer).
* **Display (5B):**
  * "Original Grayscale" and the noisy image are shown side by side, both shrunk the same way (`cv2.INTER_AREA`, aspect ratio kept, never enlarged) to fit 560 × 420 px.
  * Because shrinking averages pixels and hides noise, the zoom panel cuts the same 16–128 px square from both full-size images. It enlarges them by repeating each pixel, so no new values are created.
  * The colour upload moved into an expandable panel, so the main comparison is grayscale against grayscale.
* **Verification (5B):**
  * 74 helper tests and 26 page tests (both counts include 5A).
  * A mutation check: 13 deliberately broken versions of the new code were all detected.
  * A real server start on localhost.
  * The browser appearance is not verified automatically.
* **Implemented (5C): connection to the filters (§3.4).**
  * **Pipeline of the interactive app (every rerun):**
    ```
    uploaded file ─► grayscale reference f (read-only)
                  ─► make_noisy_image(f, noise, strength, seed)  ─► noisy image (read-only)
                  ─► apply_selected_filter(noisy, filter, parameters) ─► filtered image (read-only)
    ```
    The filter always receives the **noisy** image. The clean reference is only displayed (and in 5D it will be the reference for MSE/PSNR).
  * The sidebar section "3. Remove Noise" has a filter selector and only the controls of the selected filter. The defaults are the Stage 4 settings (D28):

    | Filter | Function in `src/filters.py` | Controls in the app | Default |
    |---|---|---|---|
    | Mean | `apply_mean_filter(image, kernel_size)` | kernel 3–15, odd | 3 |
    | Gaussian | `apply_gaussian_filter(image, kernel_size, sigma)` | kernel 3–15; filter σ 0.0–5.0 (0 = OpenCV chooses) | 3, 0 |
    | Median | `apply_median_filter(image, kernel_size)` | kernel 3–15, odd | 3 |
    | Bilateral | `apply_bilateral_filter(image, diameter, sigma_color, sigma_space)` | diameter 3–15, odd; sigmas 1–200 | 5, 50, 50 |

  * `apply_selected_filter` looks up the function in a table. It requires exactly that function's parameter names, passes the values on unchanged (so the existing checks in §3.4.0 apply), and returns a new, read-only `uint8` image of the same size.
  * The main view shows **Original Grayscale | noisy | filtered** with one settings line (noise, seed, filter). The noisy and filtered images each have a pixel fingerprint; the page labels these as identifiers, not quality measures.
  * The filtered image is described as filtered or noise-reduced, never as "clean" or "restored".
  * The zoom panel shows the **same** region of all three images. "How does this filter work?" gives a simple explanation, the main benefit and limitation, and the formula of the selected filter. It does not claim that any filter is best.
* **Reproducibility (5C):** the noise is generated again from the same seed before the filter runs. So changing only the filter or its parameters leaves the noisy pixels identical. Page tests confirm that:
  * the noisy fingerprint stays the same for all four filters and for changes of every filter parameter
  * the filtered fingerprint equals that of a direct `src/` computation filter(noise(reference)) for every filter and for 7 parameter changes
  * it follows changes of noise strength, type and seed, and the first result returns when the first settings return
* **Performance (5C, measured, not an experiment result):**
  * **Setup:** Windows 11, Python 3.12.10, OpenCV 5.0.0 with 12 threads, a 4000 × 3000 test image.
  * **Filters alone:** mean, Gaussian and bilateral (d = 5) took 4–12 ms; bilateral d = 15 took 92 ms; median took 4 ms at 3 × 3 and 401–439 ms at 9 × 9 to 15 × 15.
  * **Full page rerun** (decode, Gaussian noise, filter, display) took 0.66–1.06 s. So the app processes at full resolution without caching.
* **Verification (5C):**
  * 100 helper tests and 41 page tests (both counts include 5A–5B).
  * A mutation check: 13 deliberately broken versions of the helper and the page were all detected. These included filtering the clean reference, noise that depends on the filter choice, and a zoom crop from a different region.
  * A real server start on localhost.
  * The browser appearance is not verified automatically.
* **Implemented (5D): evaluation in the app.**
  * **How MSE and PSNR are used:**
    * Every score is computed by `mse()` and `psnr()` from §3.2 (`src/metrics.py`). It always compares an image with the **original grayscale reference f**, on the full-resolution arrays, never on the shrunk display copies.
    * Section 4 shows MSE and PSNR for f vs noisy and f vs filtered. It also shows MSE change = MSE(filtered) − MSE(noisy) and PSNR change = PSNR(filtered) − PSNR(noisy).
    * The verdict ("numerically closer / farther / no change") compares the two MSE values. MSE is always finite, and lower MSE always means higher PSNR.
    * Noisy-vs-filtered is **never** used as a quality score, because the noisy image is not the truth.
  * **Infinity:**
    * Identical images have MSE = 0 and PSNR = ∞, shown as "∞".
    * When a PSNR change would be ∞ − ∞ (NaN) or infinite, it is not calculated. The page explains the case in words: "no change (both ∞)", "now identical (∞)" or "worse (was ∞)".
  * **Fair four-filter comparison:**
    * Section 5 applies all four filters to the **same noisy array**. It is passed four times as the same object, and a spy test checks this. The noise is generated once per run from the selected type, strength and seed.
    * Every row is scored against the same reference, and a "No filtering" baseline row is included.
    * Each filter uses the parameters shown in the table: the current sidebar settings for that filter, or its Stage 4 default. Nothing is tuned automatically.
    * The row(s) with the lowest MSE are marked. If the baseline is best, the page says that no filter improved the numerical result. The page states that the ranking holds only for this image and these settings.
  * **Difference images (section 6, visualisation only):**
    * |f − noisy|, |noisy − filtered| and |f − filtered|, computed with `cv2.absdiff`, which gives the exact absolute difference for uint8 with no wrap-around.
    * An optional contrast enhancement multiplies all three images by one shared factor. It is never used for metrics.
    * The page states that a difference image shows *where* pixels changed, not whether quality improved.
  * **What the metrics can and cannot say:** the reference is the uploaded photo in grayscale. Camera noise, blur or JPEG artifacts already in that photo count as "correct". The numbers measure only the artificial noise and the filtering. PSNR does not always agree with human visual judgement (see §4.0–4.1).
* **Illustrative example vs real results (keep them apart in the report):**
  * The worked example in "How do MSE and PSNR work?" and in the tests is **illustrative**, not a measurement:
    * original 100, 100, 100, 100; noisy 120, 80, 110, 90; filtered 105, 95, 105, 95
    * MSE 250 → 25 and PSNR 24.15 → 34.15 dB (+10.00 dB)
  * It was confirmed with the real `mse()`/`psnr()` functions. A second illustrative example shows a "worse" case: MSE 50 → 450 (−9.54 dB).
  * Neither belongs in the experiment log.
  * Real results on the synthetic image remain those of §4.2 (E010–E017). The app reproduces them exactly: helper tests check E002 = 577.9914 / 20.5116 dB and E013 = 87.8501 / 28.6934 dB (+8.1818 dB).
* **Performance (5D, measured on this computer, 12 CPU threads; not an experiment result):**

  | Image | Full rerun, comparison off | Comparison on |
  |---|---|---|
  | 256 × 256 synthetic | 0.07 s | 0.07 s |
  | 4000 × 3000 photo | 1.37 s | 3.01 s (defaults), 3.88 s (kernel/diameter 15) |

  So the comparison starts switched off for photos above 4 MP. Each MSE+PSNR pair costs about 0.33 s at 12 MP, partly because `psnr()` recomputes `mse()` internally.
* **Verification (5D):**
  * 136 helper tests and 59 page tests (both counts include 5A–5C).
  * A mutation check: 16 of 17 deliberately broken versions were detected. The one survivor did not actually change the page's behaviour.
  * A real server start on localhost.
  * The browser appearance is not verified automatically.
* **Proposed (5E):** downloads (noisy image, filtered image, metrics CSV) and final polish.
* **Screenshots / figures needed for the report (Not yet collected; to be taken by the student after the final sub-stage, never mocked up):**
  1. The start page (sidebar with upload, "How the lab works").
  2. A colour photo uploaded: grayscale reference with dimensions, and the "Uploaded photo (original colours)" panel.
  3. The error message for a damaged file.
  4. The noisy image for each noise type (Gaussian σ = 25, salt-and-pepper 5%, speckle 0.04) with the sidebar visible.
  5. The zoom panel showing single noisy pixels.
  6. The same settings with seed 42 twice (equal fingerprints) and with a different seed (different fingerprint).
  7. Original / noisy / filtered for salt-and-pepper 5% with the **median** filter, and for Gaussian σ = 25 with the **bilateral** filter.
  8. The three-image zoom panel showing the median filter removing single white and black pixels.
  9. One example where the noisy fingerprint stays the same while two different filters are selected.
  10. Section 4 (MSE/PSNR table, changes and verdict) for one case where the filter helps (e.g. median on salt-and-pepper) and one where it makes the numbers worse (e.g. mean 3 × 3 on Gaussian σ = 25 on the synthetic image).
  11. Section 5, the four-filter table with the baseline row and the ★ mark, plus the row of four filtered images.
  12. Section 6, the three difference images, raw and contrast-enhanced.
  13. One case where "No filtering" is best (for example a fine-texture photo with weak noise), to show that filters can make the numbers worse.
* **Design decisions:** D31–D64 in `project_overview.md`.

---

## 4. Results and Discussion
Template instruction: "Simulation results should be here".

### 4.0 Preliminary metric validation (Stage 2): NOT noise removal results
These numbers only check that MSE and PSNR are calculated correctly. **No noise was added and no filter was applied.** Source: `python -m demos.demo_metrics`, run on 2026-11-16 (log IDs MV001–MV004 in `experiment_log.md`).

| ID | Comparison (original vs …) | Changed pixels | MSE | PSNR (ours) | cv2.PSNR | Hand check |
|---|---|---|---|---|---|---|
| MV001 | 2×2 example [[10,20],[30,40]] vs [[12,20],[30,36]] | 2 of 4 | 5.0 | 41.1411 dB | not run | 20/4 = 5; 10·log₁₀(13005) |
| MV002 | Synthetic image, case A: 10 pixels 128 → 178 | 10 of 65,536 | 0.3815 | 52.3162 dB | 52.3162 dB | 10·50²/65536 = 0.3815 |
| MV003 | Synthetic image, case B: every pixel +10 | 65,536 of 65,536 | 100.0000 | 28.1308 dB | 28.1308 dB | 10² = 100 |
| MV004 | Synthetic image vs identical copy | 0 | 0.0 | infinity | 361.2020 dB | MSE = 0 |

Figure (for report and presentation, if useful): `results/figures/stage2_metric_validation.png` (300 dpi, generated with `--save-figure`; regenerate any time).

Observations (from these real runs):
* Our MSE and PSNR match the hand calculations and `cv2.PSNR` (to 4 decimals here; to 6 decimals in the automated tests).
* Case B looks almost the same as the original to a human (slightly brighter), but scores far worse (28.13 dB) than case A (52.32 dB), where 10 bright dots are visible. This demonstrates the limitation described in [V5]: PSNR measures numerical difference, not visual importance.
* For identical images, our PSNR is infinity while OpenCV reports 361.20 dB.

### 4.1 Noise addition results (Stage 3): noisy baselines, NO filtering yet
These are the first real noise experiments. **No filter has been applied**, so
nothing here says anything about noise removal. These values are the
**baselines** that Stage 4 filters must beat.

**Setup:** image D1 (`synthetic_test_image.png`, 256 × 256 = 65,536 pixels, uint8). Seed 42 for every experiment. Command: `python -m demos.demo_noise --save-figure --save-outputs`, run on 2026-11-16 with Python 3.12.10, NumPy 2.5.5 and OpenCV 5.0.0. Source of every number: `results/tables/stage3_noise_results.csv` (log IDs E001–E009).

| ID | Noise | Strength | MSE | PSNR (dB) | Changed pixels | Pixels at 0 or 255 |
|---|---|---|---|---|---|---|
| E001 | Gaussian | σ = 10 | 99.60 | 28.15 | 62,782 | 456 |
| E002 | Gaussian | σ = 25 | 577.99 | 20.51 | 64,457 | 3,713 |
| E003 | Gaussian | σ = 50 | 2053.92 | 15.00 | 64,984 | 8,693 |
| E004 | Salt-and-pepper | amount = 0.02 | 413.36 | 21.97 | 1,311 | 1,311 |
| E005 | Salt-and-pepper | amount = 0.05 | 1058.04 | 17.89 | 3,277 | 3,277 |
| E006 | Salt-and-pepper | amount = 0.10 | 2083.03 | 14.94 | 6,554 | 6,554 |
| E007 | Speckle | variance = 0.01 | 171.46 | 25.79 | 61,713 | 1,680 |
| E008 | Speckle | variance = 0.04 | 636.43 | 20.09 | 63,627 | 3,086 |
| E009 | Speckle | variance = 0.10 | 1483.11 | 16.42 | 64,330 | 4,468 |

**Model check: measured MSE vs. the value predicted without clipping** (printed by the same demo run)
| ID | Predicted MSE | Formula | Measured | Measured / predicted |
|---|---|---|---|---|
| E001 | 100.00 | σ² | 99.60 | 0.996 |
| E002 | 625.00 | σ² | 577.99 | 0.925 |
| E003 | 2500.00 | σ² | 2053.92 | 0.822 |
| E007 | 189.84 | variance × mean(I²) = 0.01 × 18983.8 | 171.46 | 0.903 |
| E008 | 759.35 | 0.04 × 18983.8 | 636.43 | 0.838 |
| E009 | 1898.38 | 0.10 × 18983.8 | 1483.11 | 0.781 |

**Figure:** `results/figures/stage3_noise_examples.png` (300 dpi; rows = Gaussian / salt-and-pepper / speckle; columns = original + three strengths; MSE and PSNR in each title). A good candidate for the report.

**Observations (based only on these runs):**
1. **Stronger noise → higher MSE, lower PSNR, for all three types.** Every step up in strength lowered PSNR (Gaussian 28.15 → 20.51 → 15.00 dB; salt-and-pepper 21.97 → 17.89 → 14.94 dB; speckle 25.79 → 20.09 → 16.42 dB). Visually the shapes become harder to see: at σ = 50 the edges of the square and the 1-pixel lines are still visible but much less clear.
2. **Gaussian and speckle change almost every pixel; salt-and-pepper changes only the selected few.** Gaussian σ = 10 changed 95.8% of pixels (the rest got noise smaller than ±0.5, which rounds to no change; checked: exactly those 2,754 pixels). Salt-and-pepper 10% changed exactly 6,554 pixels (10.0%).
3. **Salt-and-pepper creates a large error with few pixels.** Each corrupted pixel jumps to 0 or 255, an error of up to 240 levels in our image (a pixel of 240 set to pepper 0). So 2% salt-and-pepper (MSE 413.36) gives a larger MSE than Gaussian σ = 10 applied to almost every pixel (MSE 99.60).
4. **Same PSNR, different appearance.** E003 (Gaussian σ = 50, 15.00 dB) and E006 (salt-and-pepper 10%, 14.94 dB) have almost the same PSNR. In the figure, however, E006 keeps 90% of the pixels exactly correct and the shapes and lines stay clearly recognisable, while E003 looks uniformly grainy. This is another example of the PSNR limitation in [V5] and §3.2.
5. **Speckle depends on brightness, as the model predicts.** In the figure, speckle is strong on the bright circle (235) and the bright right side, and barely visible on the dark square (20) and dark left side. Gaussian noise looks the same everywhere.
6. **Clipping reduces the measured error.** For Gaussian σ = 10 the measured MSE is 99.6% of σ², so almost no clipping. At σ = 50 it is only 82.2%, because values pushed below 0 or above 255 are cut off. The number of pixels at exactly 0 or 255 grows from 456 to 8,693. Speckle shows the effect more strongly (90.3% → 78.1%) because its largest changes happen at the brightest pixels, which are closest to 255.
7. **Reproducibility confirmed:** running the demo a second time produced byte-identical PNG and CSV files (SHA-256 compared), and a saved noisy PNG reloads to exactly the array the noise function produces.

**Limitations of these results:** one synthetic image only, one seed (42), and no real photograph yet. The numbers describe this image and seed. Other images or seeds will give somewhat different values. PSNR comparisons across noise types should be read together with the images (observation 4).

### 4.2 Noise removal results (Stage 4): first comparison, one setting per filter
**Setup:**
* Image D1, seed 42, two noise cases. Each case uses **one** noisy image, and all four filters are applied to that same array.
  * G002: Gaussian σ = 25, baseline E002
  * G005: salt-and-pepper amount = 0.05, baseline E005
* The program confirmed that both noisy images are identical to the saved Stage 3 files.
* Filters: mean k = 3; Gaussian k = 3, σ = 0 (auto); median k = 3; bilateral d = 5, σ_color = σ_space = 50.
* Command: `python -m demos.demo_filters --save-figure --save-outputs`, run on 2026-11-16 with Python 3.12.10, NumPy 2.5.5 and OpenCV 5.0.0.
* Source of every number: `results/tables/stage4_filter_results.csv` and the demo output (log IDs E010–E017, FC001–FC004).

**Whole-image results**
| ID | Noise | Filter | MSE | PSNR (dB) | ΔPSNR vs noisy (dB) |
|---|---|---|---|---|---|
| E002 | Gaussian σ = 25 | none (noisy) | 577.99 | 20.51 | – |
| E010 | Gaussian σ = 25 | mean k3 | 945.68 | 18.37 | −2.14 |
| E011 | Gaussian σ = 25 | Gaussian k3 | 588.33 | 20.43 | −0.08 |
| E012 | Gaussian σ = 25 | median k3 | 1216.98 | 17.28 | −3.23 |
| E013 | Gaussian σ = 25 | bilateral d5/50/50 | 87.85 | 28.69 | **+8.18** |
| E005 | Salt-and-pepper 0.05 | none (noisy) | 1058.04 | 17.89 | – |
| E014 | Salt-and-pepper 0.05 | mean k3 | 1007.24 | 18.10 | +0.21 |
| E015 | Salt-and-pepper 0.05 | Gaussian k3 | 672.05 | 19.86 | **+1.97** |
| E016 | Salt-and-pepper 0.05 | median k3 | 1271.09 | 17.09 | −0.80 |
| E017 | Salt-and-pepper 0.05 | bilateral d5/50/50 | 861.12 | 18.78 | +0.89 |

**MSE by region.** Both regions are 8,455 px.
* **Flat patch:** the constant 128 area, with no edges inside it.
* **Line block:** the 1-pixel lines, 12.9% of the image.

| Filter | Gaussian σ = 25: flat patch | line block | S&P 0.05: flat patch | line block |
|---|---|---|---|---|
| none (noisy) | 629.93 | 520.13 | 830.67 | 1181.32 |
| mean k3 | 80.14 | 6713.03 | 102.34 | 6775.90 |
| Gaussian k3 | 94.16 | 3889.52 | 125.27 | 4039.72 |
| median k3 | 107.46 | 8664.46 | **2.66** | 9826.72 |
| bilateral | 88.01 | **119.86** | 458.25 | 998.52 |

**Filters applied to the clean image (no noise)**
| ID | Filter | MSE | PSNR (dB) | Changed px | Line-block MSE |
|---|---|---|---|---|---|
| FC001 | mean k3 | 879.34 | 18.69 | 9,378 | 6657.56 |
| FC002 | Gaussian k3 | 497.07 | 21.17 | 9,114 | 3758.35 |
| FC003 | median k3 | 1263.88 | 17.11 | 2,563 | 9793.13 |
| FC004 | bilateral | 3.44 | 42.76 | 21,041 | 4.00 |

**Figure:** `results/figures/stage4_filter_comparison.png`, a strong candidate for the report.

**Observations (based only on these runs):**
1. **Not every filter improves the image.**
   * With Gaussian noise, three of four filters gave a *lower* whole-image PSNR than the noisy image: mean −2.14 dB, Gaussian −0.08 dB, median −3.23 dB.
   * With salt-and-pepper, the median filter gave −0.80 dB.
   * Only the bilateral filter improved both cases.
2. **The line block decides the whole-image ranking.** On the clean image, 97.7% (mean), 97.5% (Gaussian) and ≈ 100% (median) of the filtering error comes from the line block, which is only 12.9% of the pixels.
   * A 3×3 window always contains a line pixel next to darker pixels, so mean and Gaussian smear the lines and the median removes them completely.
   * In the figure, the median-filtered line block is flat gray in both rows.
   * This matches the trade-off in [V8]: spatial filters remove noise "at the cost of image blurring".
3. **In the flat patch, every filter removes most of the noise**, which is the behaviour the theory predicts.
   * Gaussian noise: mean 80.14 (12.7% of the noisy 629.93), bilateral 88.01 (14.0%), Gaussian 94.16 (14.9%), median 107.46 (17.1%).
   * Theory for independent noise: mean 1/9 = 11.1%, Gaussian k3 Σw² = 14.1%.
   * The measured ratios are slightly higher. Possible reasons, not separately measured: the outermost patch pixels also average in neighbouring gradient values, and rounding and clipping play a part.
4. **The median filter is by far the best against salt-and-pepper where there is no fine detail.** In the flat patch: 830.67 → 2.66 (−99.7%), against mean −87.7% and Gaussian −84.9%.
   * In the figure the median result looks almost perfectly clean except for the line block.
   * This agrees with [V9] ("one of the most popular nonlinear filters for the removal of salt-and-pepper noise").
5. **Mean and Gaussian do not remove impulses; they spread them.** In the figure, each salt or pepper pixel becomes a faint gray 3×3 blob.
6. **The bilateral filter keeps edges and lines, but not against impulses.**
   * Gaussian noise: the best result in the whole image (28.69 dB) and the only filter that also improved the line block (520.13 → 119.86).
   * On the clean image it changed almost nothing (MSE 3.44).
   * Salt-and-pepper: the flat-patch error was only halved (830.67 → 458.25), and in the figure most impulses are still visible. An impulse differs from its neighbours by far more than sigma_color = 50, so it is treated like an edge and kept. This matches the expectation in §3.4.4.
7. **PSNR again depends on image content.** The Gaussian filter has the best whole-image PSNR for salt-and-pepper (+1.97 dB), yet the median result clearly looks better almost everywhere. On an image without 1-pixel lines, the ranking would very likely be different (an expectation, to be tested with the student's photo in Stage 6).

**Limitations of these results:**
* One synthetic image, one seed, two noise cases, and one parameter setting per filter (kernel 3, bilateral 5/50/50).
* The bilateral parameters were not tuned. The line pattern (period 4 px) is a deliberately hard case for 3×3 windows.
* No conclusions about other kernel sizes, noise levels or real photographs can be drawn yet. These need the experiment runner (Stage 6).

## 5. Conclusion
Template instruction: "Summarized the whole project".
* **Not yet collected.** Write it after Stage 7.
* **Achievements so far (engineering, not final conclusions):**
  * reliable image loading with validation and Unicode path support
  * a verified MSE/PSNR implementation
  * three reproducible noise models
  * four validated filter wrappers
  * a web interface that safely uploads and converts a user's photo (Stage 5A) and adds reproducible noise to it with the existing noise models (Stage 5B), then filters it with the existing filters (Stage 5C), and evaluates and compares the filters with MSE/PSNR and difference images (Stage 5D)
  * 339 passing unit tests
* **Findings so far that may feed the conclusion:**
  * From §4.0–4.1: PSNR does not capture the *type* of degradation (salt-and-pepper 10% and Gaussian σ = 50 have equal PSNR but look very different). Clipping makes strong noise measure lower than its model predicts.
  * From §4.2, preliminary, one setting per filter: the median filter almost completely removes salt-and-pepper noise in flat regions but erases 1-pixel lines. The bilateral filter preserves edges and lines and works well for Gaussian noise but leaves impulses. Smoothing filters can *lower* the whole-image PSNR when the image contains fine detail.

## References
See `references.md`. Style: IEEE numbered (confirmed by the template's example reference). Template: "Please put a least five Reference". **13 sources are verified so far (V1–V13): 5 peer-reviewed papers (V5, V8, V9, V10, V13) and 8 official technical documentation / source-code references.**

## Index (source code)
Template: "Please put your source code here if you it is applicable".
Current source files: `src/image_io.py`, `src/synthetic_image.py`, `src/metrics.py`, `src/noise.py`, `src/filters.py`, `demos/demo_image_loading.py`, `demos/demo_metrics.py`, `demos/demo_noise.py`, `demos/demo_filters.py`, `frontend/app.py`, `frontend/image_processing.py`.
Final listing to be prepared at the end.
