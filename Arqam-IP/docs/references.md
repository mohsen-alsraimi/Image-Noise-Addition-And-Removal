# References

_Last updated: 2026-11-16 (after Stage 4)_

Citation style: **IEEE numbered**. The template's own example reference is
in IEEE style (authors, "title," venue, vol., pp., year, doi), so this is consistent.
Template requirement: "Please put a least five Reference".
Final numbers [1], [2], ... are assigned in order of first citation when the
report is written. Until then, verified sources use IDs V1, V2, ...

**Status: 13 verified sources (V1–V13).**
* **Academic, peer-reviewed (5):** V5, V8, V9, V10, V13 (V13: content checked in the authors' HTML version, see its entry).
* **Official technical documentation / source code (8):** V1, V2, V3, V4, V6, V7, V11, V12.

No textbook is verified yet: we could not inspect the contents of Gonzalez and
Woods online (see section D).

---

## A. Verified sources

Verification means we opened an official source and confirmed that the content
we rely on is really there.

### [V1] OpenCV: colour conversions (RGB → Gray formula)
* **IEEE draft:** OpenCV, "Color conversions," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
* **Author/organization:** OpenCV team · **Year:** documentation for the 4.x branch (no single publication year)
* **What we verified:** the formula "RGB[A] to Gray: Y ← 0.299·R + 0.587·G + 0.114·B".
* **How verified (2026-11-16):** the docs.opencv.org page blocked automated access (HTTP 403, bot check), so we read the same text in OpenCV's official source file `modules/imgproc/doc/colors.markdown` (4.x branch, https://github.com/opencv/opencv), which the documentation page is generated from. **Action for the student:** open the docs URL in a browser once to confirm it displays the formula.
* **Note:** our installed version is OpenCV 5.0.0; the text was checked on the 4.x branch. Our own tests confirm the behaviour in our environment.
* **Supports:** grayscale conversion during image loading.
* **Cite in:** 2 Datasets (preprocessing), 3 Methods (3.1).

### [V2] OpenCV: image file reading and writing (imgcodecs)
* **IEEE draft:** OpenCV, "Image file reading and writing," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
* **Author/organization:** OpenCV team · **Year:** 4.x documentation
* **What we verified:** `IMREAD_GRAYSCALE`: "always convert image to the single channel grayscale image (codec internal conversion)". `imdecode`: "Reads an image from a buffer in memory"; if decoding fails, it "returns an empty matrix". `imread`: "If the image cannot be read, the function returns an empty matrix."
* **How verified (2026-11-16):** same situation as V1. We read the official header `modules/imgcodecs/include/opencv2/imgcodecs.hpp` (4.x branch), which the documentation is generated from. Please confirm in a browser.
* **Supports:** image loading, error handling, and why grayscale values can differ by 1 from `cvtColor`.
* **Cite in:** 3 Methods (3.1).

### [V3] NumPy: numpy.fromfile
* **IEEE draft:** NumPy Developers, "numpy.fromfile," *NumPy v2.5 Manual*. [Online]. Available: https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html
* **Author/organization:** NumPy Developers · **Year:** v2.5 manual
* **What we verified (2026-11-16):** the page loads (title "numpy.fromfile — NumPy v2.5 Manual") and describes the function as "Construct an array from data in a text or binary file."
* **Supports:** reading image bytes so that non-ASCII Windows paths work.
* **Cite in:** 3 Methods (3.1).

### [V4] OpenCV: cv::PSNR (core module, operations on arrays)
* **IEEE draft:** OpenCV, "Operations on arrays: PSNR," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/d2/de8/group__core__array.html
* **Author/organization:** OpenCV team · **Year:** 4.x documentation
* **What we verified (2026-11-16):**
  * Documentation text in the official header `modules/core/include/opencv2/core.hpp` (4.x): "PSNR = 10 · log10(R² / MSE) where R is the maximum integer value of depth (e.g. 255 in the case of CV_8U data) and MSE is the mean squared error between the two arrays."
  * Implementation in `modules/core/src/norm.dispatch.cpp` (identical in the 4.x and 5.x branches): `return 20*log10(R/(diff+DBL_EPSILON));` with `diff = sqrt(MSE)`. This explains the finite 361.20 dB for identical images.
* **How verified:** docs.opencv.org blocks automated access, so the text was read in the official GitHub repository. The exact docs page URL above (group "core_array") was **not** opened. **Action for the student:** open it in a browser and confirm that it shows PSNR. If not, cite the GitHub source file instead.
* **Supports:** PSNR formula with MAX = 255; why `cv2.PSNR` is finite for identical images; validating our implementation against OpenCV.
* **Cite in:** 3 Methods (3.2).

### [V5] Wang & Bovik (2009): MSE and its limitations (peer-reviewed)
* **IEEE:** Z. Wang and A. C. Bovik, "Mean squared error: Love it or leave it? A new look at signal fidelity measures," *IEEE Signal Processing Magazine*, vol. 26, no. 1, pp. 98–117, Jan. 2009, doi: 10.1109/MSP.2008.930649.
* **How verified (2026-11-16):** bibliographic data from the Crossref DOI registry (api.crossref.org). Content checked in the author-hosted PDF (https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf).
* **Quotes we rely on (from the PDF):**
  * "PSNR = 10 log10 (L² / MSE), where L is the dynamic range of allowable image pixel intensities."
  * PSNR "is useful if images having different dynamic ranges are being compared, but otherwise contains no new information relative to the MSE."
  * "The MSE has many attractive features: 1) It is simple. It is parameter free and inexpensive to compute…"
  * MSE "remains the standard criterion for the assessment of signal quality and fidelity" and "is the method of choice for comparing competing signal processing methods".
  * "the MSE values … of several of the distorted images are nearly identical …, even though the same images present dramatically (and obviously) different visual quality."
* **Supports:** definitions of MSE and PSNR, why they are widely used, and their main limitation.
* **Cite in:** 1 Introduction, 3 Methods (3.2), 4 Results (discussion of metric limits).

### [V6] NumPy: data types and overflow
* **IEEE draft:** NumPy Developers, "Data types," *NumPy v2.5 Manual*. [Online]. Available: https://numpy.org/doc/stable/user/basics.types.html
* **What we verified (2026-11-16):** page title "Data types — NumPy v2.5 Manual". Section "Overflow errors": "The fixed size of NumPy numeric types may cause overflow errors when a value requires more memory than available in the data type."
* **Supports:** why uint8 arithmetic is unsafe and why we convert to float64.
* **Cite in:** 3 Methods (3.1, 3.2).

### [V7] NumPy: random sampling (Generator, default_rng, normal, choice) [technical documentation]
* **IEEE draft:** NumPy Developers, "Random sampling," *NumPy v2.5 Manual*. [Online]. Available: https://numpy.org/doc/stable/reference/random/index.html (see also "Random Generator," https://numpy.org/doc/stable/reference/random/generator.html)
* **What we verified (2026-11-16), quotes from these pages:**
  * Random sampling page: "Our RNGs are deterministic sequences and can be reproduced by specifying a seed integer to derive its initial state."
  * Same page: "The convenience Functions in numpy.random are still aliases to the methods on a single global RandomState instance."
  * Random Generator page, `default_rng(seed=None)`: "Construct a new Generator with the default BitGenerator (PCG64)." "If None, then fresh, unpredictable entropy will be pulled from the OS. If an int or array_like[ints] is passed, then all values must be non-negative…"
  * `Generator.normal(loc=0.0, scale=1.0, size=None)`: "Draw random samples from a normal (Gaussian) distribution." `scale`: "Standard deviation (spread or 'width') of the distribution. Must be non-negative." (page: …/generated/numpy.random.Generator.normal.html)
  * `Generator.choice`: `replace`: "Whether the sample is with or without replacement." (page: …/generated/numpy.random.Generator.choice.html)
* **Note:** the URL …/generated/numpy.random.default_rng.html returned HTTP 404; `default_rng` is documented on the Random Generator page instead.
* **Supports:** reproducible seeded noise, avoiding global state, Gaussian sampling, sampling positions without replacement.
* **Cite in:** 3 Methods (3.3).

### [V8] Fan et al. (2019): review of image denoising (peer-reviewed, open access CC-BY 4.0) [academic]
* **IEEE:** L. Fan, F. Zhang, H. Fan, and C. Zhang, "Brief review of image denoising techniques," *Visual Computing for Industry, Biomedicine, and Art*, vol. 2, no. 1, Art. no. 7, 2019, doi: 10.1186/s42492-019-0016-7.
* **How verified (2026-11-16):** bibliographic data from Crossref. Full text read on the publisher's site (https://vciba.springeropen.com/articles/10.1186/s42492-019-0016-7).
* **Quotes we rely on:**
  * "images are inevitably contaminated by noise during acquisition, compression, and transmission, leading to distortion and loss of image information."
  * "With the presence of noise, possible subsequent image processing tasks, such as video processing, image analysis, and tracking, are adversely affected. Therefore, image denoising plays an important role in modern image processing systems."
  * "y = x + n (1) where y is the observed noisy image, x is the unknown clean image, and n represents additive white Gaussian noise (AWGN) with standard deviation σ"
  * "since noise, edge, and texture are high frequency components, it is difficult to distinguish them in the process of denoising and the denoised images could inevitably lose some details."
  * Also mentions PSNR/SSIM as evaluation metrics.
* **Filter quotes (re-read on 2026-11-16 for Stage 4, section "Spatial domain filtering"):**
  * "The major challenges for image denoising are as follows: flat areas should be smooth, edges should be protected without blurring, textures should be preserved, and new artifacts should not be generated."
  * "Originally, linear filters were adopted to remove noise in the spatial domain, but they fail to preserve image textures."
  * "Mean filtering [14] has been adopted for Gaussian noise reduction, however, it can over-smooth images with high noise [15]."
  * "By using non-linear filters, such as median filtering [14, 18] and weighted median filtering [19], noise can be suppressed without any identification."
  * "As a non-linear, edge-preserving, and noise-reducing smoothing filter, Bilateral filtering [10] is widely used for image denoising." "One issue concerning the bilateral filter is its efficiency." (Their reference [10] is Tomasi and Manduchi 1998 = our V13.)
  * "Normally, spatial filters eliminate noise to a reasonable extent but at the cost of image blurring, which in turn loses sharp edges."
* **Supports:** why noise matters; the Gaussian noise model; the denoising trade-off; the general properties of mean, median and bilateral filtering.
* **Cite in:** 1 Introduction, 3 Methods (3.3.1, 3.4).

### [V9] Ma and Nie (2018): salt-and-pepper noise (peer-reviewed, open access) [academic]
* **IEEE:** H. Ma and Y. Nie, "A two-stage filter for removing salt-and-pepper noise using noise detector based on characteristic difference parameter and adaptive directional mean filter," *PLOS ONE*, vol. 13, no. 10, Art. no. e0205736, Oct. 2018, doi: 10.1371/journal.pone.0205736.
* **How verified (2026-11-16):** Crossref plus full text on journals.plos.org.
* **Quotes we rely on:**
  * "Digital images can be corrupted by impulse noise due to bit errors in the process of image acquisition and transmission"
  * "Salt-and-pepper noise and random-valued noise are the two common types of impulse noise. The salt-and-pepper noise can corrupt an image where each corrupted pixel takes either the maximum or minimum gray level."
  * It "can not only significantly deteriorate the quality of an image, but also bring some difficulty to the subsequent image analysis such as image segmentation, edge detection and classification"
  * Also defines PSNR with MAX = 255 for gray images and MSE between the original noise-free image and the restored image (consistent with our §3.2).
* **Median filter quotes (re-read on 2026-11-16 for Stage 4, Introduction):**
  * "the standard median filter is one of the most popular nonlinear filters for the removal of salt-and-pepper noise in terms of its good denoising capability and computational efficiency"
  * "The standard median filter uniformly replaces the gray level of each noise corrupted pixel by the median gray level of its neighborhood pixels."
  * "Hence, the standard median filter is effective in the case of low noise density."
  * "While the noise density is higher than 50%, the size of filtering window should to be enlarged to suppress serious noise which leads edges and details to be blurred." (quoted as written)
* **Supports:** the salt-and-pepper definition, why impulse noise matters, that random-valued noise is a separate type, and why the median filter is the standard choice for salt-and-pepper noise.
* **Cite in:** 1 Introduction, 3 Methods (3.3.2, 3.4).

### [V10] Yu et al. (2018): speckle noise in ultrasound (peer-reviewed, open access) [academic]
* **IEEE:** H. Yu, M. Ding, X. Zhang, and J. Wu, "PCANet based nonlocal means method for speckle noise removal in ultrasound images," *PLOS ONE*, vol. 13, no. 10, Art. no. e0205390, Oct. 2018, doi: 10.1371/journal.pone.0205390.
* **How verified (2026-11-16):** Crossref plus full text on journals.plos.org. Equation 1 is an image on the page; we downloaded and read it: **u(x, y) = v(x, y) + v(x, y)^γ · η(x, y)**.
* **Quotes we rely on:**
  * "ultrasound images are inevitably corrupted by speckle noise due to the coherent imaging mechanism from the scatters"; "Such a noise reduces the sharpness of image details and complicates the diagnosis of the tiny structure of lesions."
  * "as a pre-processing step, denoising will also benefit image post-processing tasks such as image segmentation, classification and registration."
  * "speckle noise has a random granular pattern. The distribution of speckle noise is signal dependent and is governed by Fisher-Tippett distribution or Gamma distribution"
  * "v(x, y) is the clean image, u(x, y) is the noisy image, η is a Gaussian noise distribution with zero-mean and variance σ², and the factor γ is related to the ultrasound devices and additional processing. Extensive studies indicate that γ = 0.5 can be used to model speckle noise in the ultrasound images"
* **Supports:** the speckle model (ours is the γ = 1 case), signal dependence, and the limitation that our model is simplified.
* **Cite in:** 1 Introduction, 3 Methods (3.3.3).

### [V11] scikit-image: `random_noise` (v0.24.0 source and docstring) [technical documentation]
* **IEEE draft:** scikit-image developers, "skimage.util.random_noise," *scikit-image v0.24.0*, source file `skimage/util/noise.py`. [Online]. Available: https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/util/noise.py
* **What we verified (2026-11-16)** (read from the official repository at tag v0.24.0; scikit-image is **not** installed in our project):
  * 'speckle': "Multiplicative noise using `out = image + n * image`, where `n` is Gaussian noise with specified mean & variance." Implementation: `noise = rng.normal(mean, var ** 0.5, image.shape); out = image + image * noise`.
  * 's&p': "Replaces random pixels with either 1 or `low_val`". `amount`: "Proportion of image pixels to replace with noise on range [0, 1]". `salt_vs_pepper` default 0.5 ("equal amounts").
  * Implementation detail: s&p uses independent Bernoulli trials per pixel (`_bernoulli(p, image.shape, …)`), so the number of noisy pixels is random, unlike our exact count.
  * 'gaussian': "Gaussian-distributed additive noise."
* **Supports:** our speckle model is the same form as an established library; our salt-and-pepper design choice differs on purpose.
* **Cite in:** 3 Methods (3.3.2, 3.3.3).

### [V12] OpenCV: image filtering (blur, GaussianBlur, medianBlur, bilateralFilter, getGaussianKernel, border types) [technical documentation]
* **IEEE draft:** OpenCV, "Image filtering," *OpenCV Documentation*. [Online]. Available: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
* **Author/organization:** OpenCV team · **Year:** documentation for the 4.x/5.x branches
* **How verified (2026-11-16):** docs.opencv.org blocks automated access (see V1), so we read the official source files on GitHub (https://github.com/opencv/opencv), **5.x branch** (our installed version is 5.0.0), and compared with 4.x:
  * documentation comments in `modules/imgproc/include/opencv2/imgproc.hpp`
  * border types in `modules/core/include/opencv2/core/base.hpp`
  * implementation details in `modules/imgproc/src/smooth.dispatch.cpp`, `bilateral_filter.dispatch.cpp` and `median_blur.dispatch.cpp`
  * **The docs URL above was not opened. Action for the student:** open it in a browser and check that it shows these functions.
* **Quotes we rely on (imgproc.hpp, 5.x):**
  * `blur`: "Blurs an image using the normalized box filter." Kernel K = 1/(ksize.width·ksize.height) × a matrix of ones.
  * `GaussianBlur`: "ksize.width and ksize.height can differ but they both must be positive and odd." sigmaY: "if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height … it is recommended to specify all of ksize, sigmaX, and sigmaY."
  * `getGaussianKernel`: G_i = α·exp(−(i − (ksize−1)/2)² / (2·sigma²)), with α chosen so that ΣG_i = 1; "If it is non-positive, it is computed from ksize as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`."
  * `medianBlur`: "The median filter uses #BORDER_REPLICATE internally to cope with border pixels". ksize: "it must be odd and positive, for example: 1, 3, 5, 7 ... (ksize = 1 returns the image unchanged)". "when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U." (The 4.x header instead says "odd and greater than 1"; our 5.0.0 behaves as the 5.x text says: tested.)
  * `bilateralFilter`: "bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is very slow compared to most filters." "For simplicity, you can set the 2 sigma values to be the same. If they are small (< 10), the filter will not have much effect, whereas if they are large (> 150), they will have a very strong effect, making the image look "cartoonish"." "Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering." d: "Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace." sigmaColor: "A larger value of the parameter means that farther colors within the pixel neighborhood … will be mixed together". sigmaSpace: "A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough". It links to Tomasi and Manduchi's page (V13).
  * Border types (base.hpp): `BORDER_REPLICATE` "aaaaaa|abcdefgh|hhhhhhh"; `BORDER_REFLECT_101` "gfedcb|abcdefgh|gfedcba"; `BORDER_DEFAULT` "same as BORDER_REFLECT_101".
* **Source-code facts (not stated in the documentation text):**
  * Bilateral: `radius = (d <= 0) ? cvRound(sigmaSpace * 1.5) : d / 2; return std::max(radius, 1);` so d = 1, 2, 3 give the same result, and an even d acts like d + 1 (both also observed by us).
  * Gaussian: with sigma ≤ 0 and kernel size 3, 5, 7 or 9, fixed "bit-exact" kernels are used, e.g. [0.25, 0.5, 0.25] for size 3, instead of the sigma formula. We measured that `GaussianBlur(…, (3, 3), 0)` and `GaussianBlur(…, (3, 3), 0.8)` differ by up to 5 levels on a random image.
  * Median: `CV_Assert(ksize % 2 == 1 …)` and `if (ksize <= 1 …) copy`, i.e. even sizes raise an error and size 1 copies the image.
* **Supports:** how each filter works, what the parameters mean, the border behaviour, and our validation rules and default settings.
* **Cite in:** 3 Methods (3.4).

### [V13] Tomasi and Manduchi (1998): the bilateral filter [academic, conference paper]
* **IEEE:** C. Tomasi and R. Manduchi, "Bilateral filtering for gray and color images," in *Proc. 6th Int. Conf. Computer Vision (ICCV)*, 1998, pp. 839–846, doi: 10.1109/ICCV.1998.710815.
* **How verified (2026-11-16):**
  * **Bibliographic data:** Crossref (title, authors, container "Sixth International Conference on Computer Vision (IEEE Cat. No.98CH36271)", pages 839–846). Crossref has no issue date; the year 1998 comes from the DOI and the conference name.
  * **Content:** the PDF on R. Manduchi's university page (https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf, 8 pages) downloaded, but our simple text extractor found no readable text layer. We therefore checked the content in the authors' HTML version "Bilateral Filtering for Gray and Color Images" (https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html). This is the page linked from OpenCV's `bilateralFilter` documentation (V12).
  * **Action for the student:** before final citation, open the PDF and confirm that the quotes also appear there, or cite the HTML page separately.
* **Quotes we rely on (HTML version):**
  * "Gaussian low-pass filtering computes a weighted average of pixel values in the neighborhood, in which the weights decrease with distance from the neighborhood center."
  * "The assumption of slow spatial variations fails at edges, which are consequently blurred by linear low-pass filtering."
  * "Bilateral filtering is a simple, non-iterative scheme for edge-preserving smoothing."
  * "It replaces the pixel value at x with an average of similar and nearby pixel values."
  * "In smooth regions, pixel values in a small neighborhood are similar to each other, and the bilateral filter acts essentially as a standard domain filter, averaging away the small, weakly correlated differences between pixel values caused by noise."
  * At a sharp boundary, "the filter replaces the bright pixel at the center by an average of the bright pixels in its vicinity, and essentially ignores the dark pixels."
* **Supports:** the idea of the bilateral filter (closeness × similarity), why linear filters blur edges, why bilateral filtering preserves edges.
* **Cite in:** 3 Methods (3.4.4), 4 Results (discussion of the bilateral results).

---

## B. Course material (cite only if the professor allows lecture slides)

### [C-L1] Lecture 1 slides
* **Draft:** K. Zaman, "Simple Introduction to Digital Image Processing," lecture slides, course 1400002073 Image Processing.
* **File:** `course_material/İP lecture 1.pdf`
* **Supports:** basic definitions (pixel, f(x, y), 8-bit grayscale, noise, spatial filtering).
* **Note:** this does **not** count toward the five references unless the professor confirms it does. The year is not printed on the extracted slide text.

---

## C. Bibliographic data verified, content NOT yet checked

The DOI and details were confirmed in Crossref on 2026-11-16, but we have not read
the paper text, so we must not cite these for any specific claim yet.

| ID | IEEE reference | Possible use | What is still needed |
|---|---|---|---|
| C1 | Q. Huynh-Thu and M. Ghanbari, "Scope of validity of PSNR in image/video quality assessment," *Electronics Letters*, vol. 44, no. 13, pp. 800–801, Jun. 2008, doi: 10.1049/el:20080522. | When PSNR comparisons are valid | Read the paper (closed access; try the university library) |
| C2 | A. Hore and D. Ziou, "Image quality metrics: PSNR vs. SSIM," in *Proc. 20th Int. Conf. Pattern Recognition (ICPR)*, 2010, pp. 2366–2369, doi: 10.1109/ICPR.2010.579. | Relation between PSNR and SSIM | Read the paper (closed access) |
| C3 | Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Trans. Image Processing*, vol. 13, no. 4, pp. 600–612, Apr. 2004, doi: 10.1109/TIP.2003.819861. | SSIM (only if we add it) | Read the paper if SSIM is added |
| C4 | H. Hwang and R. A. Haddad, "Adaptive median filters: new algorithms and results," *IEEE Trans. Image Processing*, vol. 4, no. 4, pp. 499–502, Apr. 1995, doi: 10.1109/83.370679. | Impulse noise, median filtering (Stage 4) | Closed access; read via the library |
| C5 | J.-S. Lee, "Digital image enhancement and noise filtering by use of local statistics," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. PAMI-2, no. 2, pp. 165–168, Mar. 1980, doi: 10.1109/TPAMI.1980.4766994. | Multiplicative noise model (classic) | Closed access |
| C6 | V. S. Frost, J. A. Stiles, K. S. Shanmugan, and J. C. Holtzman, "A model for radar images and its application to adaptive digital filtering of multiplicative noise," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. PAMI-4, no. 2, pp. 157–166, Mar. 1982, doi: 10.1109/TPAMI.1982.4767223. | Multiplicative speckle in radar | Closed access (also cited by V10) |
| C7 | J. W. Goodman, "Some fundamental properties of speckle," *J. Optical Society of America*, vol. 66, no. 11, p. 1145, Nov. 1976, doi: 10.1364/JOSA.66.001145. | Physics of speckle (only if needed) | Not read |

---

## D. Candidates: NOT yet verified (do not cite)

| Candidate | Would support | Report section | When / status |
|---|---|---|---|
| R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, Pearson (4th ed.) | Noise models, spatial filters | 1, 3 | Contents not accessible online. Verify from a library copy if the professor prefers a textbook. |
| MATLAB `imnoise` documentation (how another tool defines speckle) | Comparison of speckle conventions | 3 | mathworks.com returned HTTP 403 to automated access; not verified |
