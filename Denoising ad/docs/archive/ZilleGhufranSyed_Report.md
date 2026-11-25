# Image Noise Addition and Removal

*Comparing classic filters with MSE and PSNR*

**Individual Report – Image Processing Course Project**

**Zill e Ghufran Syed** · Student ID: 230504510

Group members: Ahmet Dogan (220504026), Arqam Waqar (2305045610), Mohsen Abdulmalik (220504534)

November 2026

## Abstract

Digital photos often contain “noise”: random grain or dots that hide details. This project builds a small, fully tested Python toolkit that adds three common types of noise to a clean grayscale image (Gaussian, salt-and-pepper and speckle) and then tries to remove it with four classic filters (mean, Gaussian, median and bilateral). Every result is scored against the clean original with two standard numbers, the Mean Squared Error (MSE) and the Peak Signal-to-Noise Ratio (PSNR). All random noise uses a fixed seed, so every experiment can be repeated exactly. On our synthetic test image, the bilateral filter gave the best result for Gaussian noise (PSNR rose from 20.51 dB to 28.69 dB), while the median filter removed 99.7% of the salt-and-pepper error in flat areas but also erased one-pixel-wide lines. The main lesson is that no filter is best in every case: each one trades noise removal against loss of detail. The project also includes “Image Noise Lab”, a web app where anyone can upload a photo and try the whole process without writing code.

## Table of contents

1. [Introduction](#1-introduction)
2. [Datasets](#2-datasets)
3. [Methods](#3-methods)
4. [Results and discussion](#4-results-and-discussion)
5. [Conclusion](#5-conclusion)
- [References](#references)
- [Index (source code)](#index)

## 1. Introduction

A digital image is simply a grid of tiny squares called pixels. In a grayscale image each pixel holds one number from 0 (black) to 255 (white), so a picture is really a big table of numbers \[1\]. Image processing means changing those numbers to make the picture more useful, for example by making it clearer or by removing noise.

**Noise** is unwanted random change in the pixel values. It appears when a photo is taken in low light, when a signal is sent over a bad connection, or when a scanner or sensor is imperfect. Noise is not only ugly: it also makes later tasks such as edge detection, object recognition or medical diagnosis harder \[2\]. Removing it is therefore one of the most basic steps in image processing.

The hard part is that noise, edges and fine textures all look like “sudden changes” to a computer. A filter that removes noise can easily remove real detail too and make the image blurry \[2\]. This project studies that trade-off in a controlled, measurable way.

### Aim and objectives

The aim is to compare classic noise-removal filters fairly. The objectives were to:

- load images safely and turn them into 0–255 grayscale pixel grids;
- add three realistic types of noise in a controlled and repeatable way;
- remove the noise with four standard filters;
- score every result against the clean original with MSE and PSNR \[3\];
- build a simple web app so the experiments can be tried on any photo.

## 2. Datasets

Instead of downloading a large dataset, we created our own **synthetic test image** with code (src/synthetic_image.py). Because we generate it ourselves, we know the exact “true” value of every pixel, which makes the scoring completely reliable. The image is 256 × 256 pixels (65,536 pixels) and contains four areas, each chosen to test something different:

**Table 1: Areas of the synthetic test image and why they are there**

| **Area**                      | **What it looks like**                  | **What it tests**                     |
|-------------------------------|-----------------------------------------|---------------------------------------|
| Background                    | Smooth left-to-right shading (50 → 200) | Gradual changes                       |
| Dark square and bright circle | Sharp straight and curved edges         | Does the filter keep edges sharp?     |
| Flat gray patch               | One constant value (128)                | Pure noise removal, no detail to lose |
| Line block                    | Thin 1-pixel white lines                | Does the filter destroy fine detail?  |

We also avoided the exact values 0 and 255 in the clean image, so that every salt-and-pepper dot added later is a real, visible change. In addition, the web app (Section 3.5) accepts any PNG or JPEG photo; colour photos are converted to grayscale with the standard formula Gray = 0.299·Red + 0.587·Green + 0.114·Blue \[1\].

## 3. Methods

### 3.1 Overall workflow

Figure 1 shows the full process. The clean grayscale image is kept aside as the reference. Noise is added to a copy, a filter is applied to the noisy copy, and both the noisy and the filtered images are compared with the clean reference. Comparing the noisy image first gives a “baseline” score; a filter only helps if it beats it.

![block diagram](images/block_diagram.png)

*Figure 1: Block diagram of the proposed work.*

### 3.2 Adding noise

Each noise function uses a fixed random “seed” (42). A seed works like a recipe number for the random generator, so the same settings always produce exactly the same noisy image \[4\]. This makes the comparison fair.

**Table 2: The three noise types**

| **Noise**       | **In plain words**                                          | **Real-life example**                  | **Strength setting**      |
|-----------------|-------------------------------------------------------------|----------------------------------------|---------------------------|
| Gaussian        | A small random amount is added to every pixel (fine grain). | Low-light phone photos                 | σ = 10, 25, 50            |
| Salt-and-pepper | A few random pixels become pure white or pure black.        | Transmission errors, dusty scans \[5\] | 2%, 5%, 10% of pixels     |
| Speckle         | Grain that grows with brightness: bright areas get noisier. | Ultrasound and radar images \[6\]      | variance 0.01, 0.04, 0.10 |

### 3.3 Removing noise with filters

All four filters work the same basic way: a small window (for example 3 × 3 pixels) slides over the image, and each pixel is replaced by a value calculated from its neighbours \[1\], \[7\]. They differ in how that value is chosen:

**Table 3: The four filters**

| **Filter** | **How it works (simply)**                                           | **Good at**                     | **Weak at**                  |
|------------|---------------------------------------------------------------------|---------------------------------|------------------------------|
| Mean       | Average of the window.                                              | Gaussian grain in flat areas    | Blurs edges and lines        |
| Gaussian   | Weighted average: close pixels count more.                          | Gaussian grain, softer blur     | Still blurs edges            |
| Median     | Sorts the window and takes the middle value.                        | Salt-and-pepper dots \[5\]      | Erases very thin lines       |
| Bilateral  | Averages only pixels that are near AND similar in brightness \[8\]. | Grain while keeping edges sharp | Slower; leaves isolated dots |

### 3.4 Measuring quality: MSE and PSNR

To judge a result we compare it pixel by pixel with the clean original \[3\]:

- **MSE (Mean Squared Error)** = the average of (original pixel − processed pixel)² over all pixels. 0 means identical; **lower is better**.
- **PSNR (Peak Signal-to-Noise Ratio)** = 10 · log₁₀(255² / MSE), measured in decibels (dB). It is the same information on a log scale; **higher is better**. Every +10 dB means ten times less error.

Small example: if four original pixels are 100, 100, 100, 100 and the noisy ones are 120, 80, 110, 90, the errors are 20, −20, 10, −10, so MSE = (400 + 400 + 100 + 100) / 4 = 250 and PSNR = 10 · log₁₀(65025 / 250) ≈ 24.15 dB. Calculations are done with decimal numbers rather than 0–255 integers, because integer subtraction would “wrap around” (10 − 12 would give 254 instead of −2).

### 3.5 Tools, web app and testing

The project is written in Python. OpenCV provides the filters \[7\], NumPy handles the pixel arrays and random numbers \[4\], Matplotlib draws the figures, and Streamlit \[9\] powers the web app “Image Noise Lab” (Figure 2). In the app a user uploads a photo, picks a noise type and a filter with simple controls, and immediately sees the original, noisy and filtered images side by side together with the MSE/PSNR scores, a ranking of all four filters and “difference images” showing where pixels changed. The code is checked by **339 automated tests**, all of which pass.

![web app](images/web_app.png)

*Figure 2: The Image Noise Lab web app: salt-and-pepper noise (5%) removed with a 3 × 3 median filter.*

## 4. Results and discussion

### 4.1 Effect of the noise

Figure 3 and Table 5 show the synthetic image with each noise type at three strengths. As expected, stronger noise always gives a higher MSE and a lower PSNR.

![noise examples](images/noise_examples.png)

*Figure 3: The three noise types at three strengths (seed 42, no filtering).*

**Table 4: Noise results (no filter applied)**

| **Noise**                       | **Weak** | **Medium** | **Strong** |
|---------------------------------|----------|------------|------------|
| Gaussian (σ = 10 / 25 / 50)     | 28.15 dB | 20.51 dB   | 15.00 dB   |
| Salt-and-pepper (2% / 5% / 10%) | 21.97 dB | 17.89 dB   | 14.94 dB   |
| Speckle (0.01 / 0.04 / 0.10)    | 25.79 dB | 20.09 dB   | 16.42 dB   |

Two observations stand out. First, salt-and-pepper noise causes a large error with very few pixels, because each damaged pixel jumps all the way to black or white. Second, Gaussian σ = 50 and salt-and-pepper 10% have almost the same PSNR (≈15 dB) yet look completely different: the shapes remain easy to see under salt-and-pepper noise. This shows a known weakness of PSNR: it measures numerical difference, not what the human eye notices \[3\].

### 4.2 Effect of the filters

All four filters were applied to the **same** noisy image for two cases: Gaussian noise (σ = 25) and salt-and-pepper noise (5%). Every filter used a 3 × 3 window (bilateral: diameter 5, both sigmas 50).

![filter comparison](images/filter_comparison.png)

*Figure 4: All four filters on the same noisy images. Top: Gaussian noise; bottom: salt-and-pepper noise.*

**Table 5: PSNR of the whole image after filtering (higher is better; change from the noisy image in brackets)**

| **Filter**         | **Gaussian noise σ = 25** | **Salt-and-pepper 5%** |
|--------------------|---------------------------|------------------------|
| None (noisy image) | 20.51 dB                  | 17.89 dB               |
| Mean               | 18.37 dB (−2.14)          | 18.10 dB (+0.21)       |
| Gaussian           | 20.43 dB (−0.08)          | **19.86 dB (+1.97)**   |
| Median             | 17.28 dB (−3.23)          | 17.09 dB (−0.80)       |
| Bilateral          | **28.69 dB (+8.18)**      | 18.78 dB (+0.89)       |

**What the results mean:**

- **Bilateral was the clear winner for Gaussian noise** (+8.18 dB). It smooths the grain but refuses to mix pixels across an edge, so the edges and thin lines stay sharp.
- **Median is the classic cure for salt-and-pepper noise.** In the flat gray patch it cut the error by 99.7% (MSE 830.67 → 2.66), and the image looks almost perfectly clean. However, it also wiped out the 1-pixel lines, which pulled its whole-image score down.
- **Smoothing can make the score worse.** Mean, Gaussian and median filters lowered the PSNR for Gaussian noise because they blurred the thin lines more than they removed noise. In flat areas, on the other hand, every filter removed most of the noise.
- **Mean and Gaussian filters do not remove dots, they spread them** into faint gray blobs, and the bilateral filter leaves most dots in place because it treats them like edges.

These results only cover one test image, one seed and one setting per filter, so they should be read as a clear demonstration of each filter’s character rather than a final ranking.

## 5. Conclusion

This project built a complete, tested pipeline for adding noise to images, removing it with four classic filters and measuring the result with MSE and PSNR, plus a web app that makes the process easy to explore. The experiments confirmed what the theory predicts: the **median filter** is excellent against salt-and-pepper noise, the **bilateral filter** is the best choice for Gaussian grain when edges matter, and the simple **mean and Gaussian filters** remove grain but blur detail. Most importantly, **no single filter is best for every image and every noise type**; the right choice depends on what kind of noise is present and what details must be kept. We also saw that PSNR does not always agree with what the eye sees. Future work could test real photographs and more filter settings, and add a structure-aware quality measure such as SSIM \[10\].

## References

1. R. C. Gonzalez and R. E. Woods, Digital Image Processing, 4th ed. New York, NY, USA: Pearson, 2018.
2. L. Fan, F. Zhang, H. Fan, and C. Zhang, “Brief review of image denoising techniques,” Visual Computing for Industry, Biomedicine, and Art, vol. 2, no. 1, Art. no. 7, 2019, doi: [10.1186/s42492-019-0016-7](https://doi.org/10.1186/s42492-019-0016-7)
3. Z. Wang and A. C. Bovik, “Mean squared error: Love it or leave it? A new look at signal fidelity measures,” IEEE Signal Processing Magazine, vol. 26, no. 1, pp. 98–117, Jan. 2009, doi: [10.1109/MSP.2008.930649](https://doi.org/10.1109/MSP.2008.930649)
4. NumPy Developers, “Random sampling (numpy.random),” NumPy Manual. \[Online\]. Available: <https://numpy.org/doc/stable/reference/random/index.html>
5. H. Ma and Y. Nie, “A two-stage filter for removing salt-and-pepper noise using noise detector based on characteristic difference parameter and adaptive directional mean filter,” PLOS ONE, vol. 13, no. 10, Art. no. e0205736, Oct. 2018, doi: [10.1371/journal.pone.0205736](https://doi.org/10.1371/journal.pone.0205736)
6. H. Yu, M. Ding, X. Zhang, and J. Wu, “PCANet based nonlocal means method for speckle noise removal in ultrasound images,” PLOS ONE, vol. 13, no. 10, Art. no. e0205390, Oct. 2018, doi: [10.1371/journal.pone.0205390](https://doi.org/10.1371/journal.pone.0205390)
7. OpenCV, “Image filtering,” OpenCV Documentation. \[Online\]. Available: <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html>
8. C. Tomasi and R. Manduchi, “Bilateral filtering for gray and color images,” in Proc. 6th Int. Conf. Computer Vision (ICCV), Bombay, India, 1998, pp. 839–846, doi: [10.1109/ICCV.1998.710815](https://doi.org/10.1109/ICCV.1998.710815)
9. Streamlit, “Streamlit documentation.” \[Online\]. Available: <https://docs.streamlit.io>
10. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Processing, vol. 13, no. 4, pp. 600–612, Apr. 2004, doi: [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)

## Index

Full source code: <https://github.com/arqam709/image-processing>. The main files are:

**Table 6: Source code files**

| **File**                                      | **What it does**                                                       |
|-----------------------------------------------|------------------------------------------------------------------------|
| src/image_io.py                               | Loads and saves images safely; converts to grayscale; pixel statistics |
| src/synthetic_image.py                        | Creates the 256 × 256 synthetic test image                             |
| src/noise.py                                  | Adds Gaussian, salt-and-pepper and speckle noise (fixed seed)          |
| src/filters.py                                | Mean, Gaussian, median and bilateral filters (OpenCV)                  |
| src/metrics.py                                | MSE and PSNR calculations                                              |
| demos/\*.py                                   | Scripts that reproduce every table and figure in this report           |
| frontend/app.py, frontend/image_processing.py | The Image Noise Lab web app                                            |
| tests/\*.py                                   | 339 automated tests                                                    |

**Example – the quality measures (src/metrics.py, simplified):**

```python
import math
import numpy as np

def mse(original, processed):
    # use decimals so that 10 - 12 = -2 (not 254)
    a = original.astype(np.float64)
    b = processed.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def psnr(original, processed):
    error = mse(original, processed)
    if error == 0.0:
        return math.inf            # identical images
    return 10 * math.log10(255 ** 2 / error)
```
