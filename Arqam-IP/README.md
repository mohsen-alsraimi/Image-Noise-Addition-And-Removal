# Image Noise Addition and Removal

University Image Processing course project. The goal is to add controlled
noise (Gaussian, salt-and-pepper, speckle) to clean grayscale images, remove it
with classic filters (mean, Gaussian, median, bilateral), and compare the
filters fairly using MSE and PSNR.

**Current status: Stages 1–4 are complete.** They cover image loading,
validation, pixel statistics, the image quality metrics MSE and PSNR,
three reproducible noise models (Gaussian, salt-and-pepper, speckle), and
four noise-removal filters (mean, Gaussian, median, bilateral).

**Stage 5 (in progress): Image Noise Lab, a Streamlit web interface.**
Stages 5A–5D are complete. You can upload a PNG/JPEG photo, see its grayscale version,
add Gaussian, salt-and-pepper or speckle noise with a reproducible seed, and remove it with the
mean, Gaussian, median or bilateral filter. You can then measure the result with MSE and PSNR,
compare all four filters on the same noisy image, and see difference images.
Downloads (5E) come next.
The full experiment runner comes in a later stage.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\ahsan\OneDrive\Desktop\IP
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # only needed on a new machine
```

All commands below are run from the project root folder (`IP`).

## Run the Stage 1 demonstration

```powershell
python -m demos.demo_image_loading                         # synthetic test image
python -m demos.demo_image_loading data\input\my_photo.jpg  # your own image
python -m demos.demo_image_loading --show                  # also show image + histogram
```

## Run the Stage 2 demonstration (MSE and PSNR)

```powershell
python -m demos.demo_metrics                # print a step-by-step example and results
python -m demos.demo_metrics --show         # also show original vs modified images
python -m demos.demo_metrics --save-figure  # save results\figures\stage2_metric_validation.png
```

## Run the Stage 3 demonstration (adding noise)

```powershell
python -m demos.demo_noise                  # print the MSE/PSNR table for 9 noisy images
python -m demos.demo_noise --show           # also show the comparison figure
python -m demos.demo_noise --save-figure    # save results\figures\stage3_noise_examples.png
python -m demos.demo_noise --save-outputs   # save noisy PNGs (results\noisy) + CSV (results\tables)
```

All noise uses the fixed seed 42, so every run gives exactly the same images.

## Run the Stage 4 demonstration (removing noise with filters)

```powershell
python -m demos.demo_filters                  # print the comparison tables
python -m demos.demo_filters --show           # also show the comparison figure
python -m demos.demo_filters --save-figure    # save results\figures\stage4_filter_comparison.png
python -m demos.demo_filters --save-outputs   # save filtered PNGs (results\filtered) + CSV (results\tables)
```

The demo recreates two Stage 3 noisy images (Gaussian σ = 25 and
salt-and-pepper 5%, seed 42), applies all four filters to the **same** noisy
image, and prints MSE and PSNR for each one. It never changes the Stage 3 files.

## Run the Image Noise Lab web app (Stage 5)

The app needs Streamlit, which is listed in `requirements.txt`. On a new
machine, `pip install -r requirements.txt` installs it into the existing `.venv`.
No other virtual environment is needed.

```powershell
cd C:\Users\ahsan\OneDrive\Desktop\IP
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py
```

Streamlit prints a local address (normally http://localhost:8501) and opens it in your
browser. Press `Ctrl+C` in PowerShell to stop the app.

**First launch:** Streamlit may ask for an email address. You can leave it empty and press Enter.

**Using it:**
1. **Upload (sidebar, section 1):** choose a PNG or JPEG file (at most 20 MB and 25 megapixels).
   Colour photos are converted to grayscale, because our algorithms work on grayscale images.
2. **Add noise (sidebar, section 2, appears after a valid upload):**
   * Choose **Gaussian**, **Salt-and-Pepper** or **Speckle** noise. Only the slider for the chosen type is shown.
   * Set its strength:

     | Noise | Slider | Range (step) | Default |
     |---|---|---|---|
     | Gaussian | σ (sigma) | 0–100 (1) | 25 |
     | Salt-and-Pepper | amount, shown as % of pixels | 0–50% (1%) | 5% |
     | Speckle | variance | 0.00–0.25 (0.01) | 0.04 |

     The defaults are the Stage 3 baselines E002, E005 and E008.
   * Set the **Random Seed** (default 42). The same image, settings and seed always give
     exactly the same noisy image.
3. **Remove noise (sidebar, section 3):** choose a filter. Only its own controls are shown:

   | Filter | Controls (range) | Default |
   |---|---|---|
   | Mean | kernel size 3 × 3 … 15 × 15 (odd) | 3 × 3 |
   | Gaussian | kernel size 3 × 3 … 15 × 15; **Gaussian filter σ** 0.0–5.0 (0 = OpenCV chooses) | 3 × 3, σ = 0 |
   | Median | kernel size 3 × 3 … 15 × 15 | 3 × 3 |
   | Bilateral | diameter 3–15 (odd); sigma color 1–200; sigma space 1–200 | 5, 50, 50 |

   * The defaults are the Stage 4 demo settings.
   * The *Gaussian filter σ* (blur width) is a different thing from the *noise σ* in section 2.
   * The filter is always applied to the **noisy** image. Changing the filter or its settings never
     changes the noisy image.
4. **Main page: three-image comparison**
   * **Original Grayscale | noisy image | filtered image** are shown side by side, at the same
     compact size (aspect ratio kept). On a narrow window Streamlit stacks them vertically.
   * Below them is one settings line (noise, seed, filter) and two **pixel fingerprints**: one for
     the noisy image and one for the filtered image.
     * A fingerprint is a short code that changes when any pixel changes, so you can check that
       returning to the same settings gives the identical image.
     * A fingerprint is **not** a quality measure.
   * The filtered image is "noise reduced", not "restored". Filtering does not recover the exact original.
5. **Expandable panels:**
   * "Inspect details": zoom into the real pixels of all three images, always at the same position.
   * "How does this noise work?" and "How does this filter work?": a simple explanation, the main
     benefit and limitation, and the formula for the selected noise or filter.
   * "Uploaded photo (original colours)", plus the grayscale statistics.
6. **4. Image Quality Evaluation — MSE and PSNR:**
   * A table compares the noisy image and the filtered image with the **original grayscale
     image**, which is always the reference. The numbers come from `src/metrics.py`, computed
     on the full-size images.
   * Below it are the **MSE change** (filtered − noisy; negative = less error) and the
     **PSNR change** (filtered − noisy, in dB; positive = numerically closer to the original).
   * A short message then says whether the filter made the image numerically closer, farther
     or unchanged. A filter can make the numbers **worse**, and the page says so.
   * If an image is identical to the original, its PSNR is shown as ∞. A change that is not a
     finite number is explained in words instead (never "NaN").
   * "How do MSE and PSNR work?" explains both formulas with a small worked example
     (illustrative numbers, not a measurement).
7. **5. Compare All Four Filters:**
   * Switch on "Run all four filters". It starts on automatically for images up to 4 megapixels.
   * All four filters are applied to **exactly the same noisy image**. Each uses the settings
     currently chosen for it in the sidebar, or its default if you never changed it.
   * The table lists "No filtering" (the baseline) and the four filters, with their parameters,
     MSE, PSNR and PSNR change. ★ marks the lowest MSE / highest PSNR, and the baseline can win.
   * The four filtered images are shown in a row below the table.
   * The ranking is true only for the current image and settings; no filter is best in general.
8. **6. Where Did the Pixels Change? — Difference Images** (expandable). Three images show
   |A − B| for every pixel; black = no change, brighter = larger change:
   * A: original vs noisy (the noise added)
   * B: noisy vs filtered (what the filter changed)
   * C: original vs filtered (the error that remains)

   The raw view shows the exact differences. The "Contrast-enhanced view" multiplies all three by
   the same factor so small changes become visible. It is for viewing only and never used for
   MSE or PSNR. A difference image shows *where* pixels changed, not whether the change was an
   improvement.

The metrics measure similarity to **your uploaded photo in grayscale**. If the photo already had
camera noise or JPEG artifacts, those count as "correct". The numbers measure only the artificial
noise and the filtering.

Damaged, empty, oversized or non-PNG/JPEG files show a clear error message instead of crashing.
An invalid seed (for example a negative number) shows an error message too. If a filter fails,
an error is shown and the original and noisy images stay visible.

**Speed** (measured on the development computer with 12 CPU threads, full page update):

| Image | Comparison off | Comparison on |
|---|---|---|
| 256 × 256 synthetic image | 0.07 s | 0.07 s |
| 4000 × 3000 photo | about 1.4 s | 3.0–3.9 s |

That is why the four-filter comparison starts switched off for photos above 4 megapixels.
Processing and metrics always use the full resolution.
Images are processed in memory only. Nothing is saved to disk when you change a control.

**How it connects to the project code:**
* `frontend/app.py` contains only the interface.
* `frontend/image_processing.py` holds four kinds of helpers:
  * **Upload:** checks the file and decodes it in the same way as
    `src/image_io.load_grayscale_image`, then validates it with `src/image_io.validate_grayscale_image`.
  * **Noise:** `make_noisy_image()` picks and calls the existing function in `src/noise.py`
    (`add_gaussian_noise`, `add_salt_and_pepper_noise` or `add_speckle_noise`). It passes the
    strength and seed on unchanged.
  * **Filters:** `apply_selected_filter()` picks and calls the existing function in `src/filters.py`
    (`apply_mean_filter`, `apply_gaussian_filter`, `apply_median_filter` or
    `apply_bilateral_filter`). It passes the parameters on unchanged.
  * **Evaluation:** `evaluate_image_quality()` and `compare_all_filters()` call `mse()` and
    `psnr()` from `src/metrics.py`. `make_difference_image()` uses OpenCV's `cv2.absdiff`
    (exact |A − B| for 8-bit values) and is for display only.
* No algorithm is copied into `frontend/`.
* `.streamlit/config.toml` keeps the app on this computer only (localhost) and turns off
  Streamlit's anonymous usage statistics.

## Run the tests

```powershell
python -m unittest -v
```

## Folder structure

| Folder | Purpose |
|---|---|
| `src/` | Image processing code (no interface code) |
| `frontend/` | Streamlit web interface (Image Noise Lab) and its small helper module |
| `.streamlit/` | Streamlit settings (no usage statistics) |
| `demos/` | Small scripts that demonstrate one feature |
| `tests/` | Unit tests (Python's built-in `unittest`) |
| `experiments/` | Experiment runner (later stage) |
| `data/input/` | Original input images, which are never modified |
| `results/` | Generated outputs (ignored by Git) |
| `docs/` | Report notes, experiment log, references, presentation notes |
| `course_material/` | Professor's template and lecture slides |
