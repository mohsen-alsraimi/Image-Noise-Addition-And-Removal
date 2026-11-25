# Image Noise Addition and Removal

**Image Noise Lab — Image Processing Course Project**

## Project Overview

Image Noise Lab is a Python-based application developed to study how noise affects digital images and how different denoising algorithms can improve image quality.

The application allows users to add noise to images, apply denoising filters, and compare the results using numerical measurements and visual examples.

A web interface developed with Streamlit makes it possible to perform experiments without writing Python code.

## Features

- Add four noise types: Gaussian, Salt-and-Pepper, Speckle, and Poisson.
- Select three intensity levels: Low, Medium, and High.
- Apply six denoising filters: Gaussian, Median, Bilateral, Non-Local Means (NLM), Lee, and Total Variation (TV).
- Simulate JPEG compression and glare.
- Compare default and Set12-optimized filter parameters for supported noise experiments.
- Evaluate results using PSNR, SSIM, edge-preservation metrics, and execution time.
- Run experiments on individual images or complete datasets.
- Explore saved results using images, tables, and graphs.

## Datasets

| Dataset | Images | Purpose |
|---|---:|---|
| Set12 | 12 | Filter parameter optimization |
| BSD68 | 68 | Main denoising evaluation |
| Urban100_HR | 100 | Evaluation on detailed urban images |
| original_png | 12 | Additional experiments |

All images are converted to grayscale before processing. The origin of the `original_png` collection has not yet been verified.

## Technologies

The project uses Python, Streamlit, NumPy, OpenCV, scikit-image, SciPy, Pandas, and Matplotlib.

## Installation

Clone the repository and open the project directory:

```powershell
git clone https://github.com/mohsen-alsraimi/Image-Noise-Addition-And-Removal.git
cd Image-Noise-Addition-And-Removal
cd "Denoising ad"
```

Create a Python virtual environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Web Application

From the `Denoising ad` directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

Open `http://localhost:8501` in your browser.

The website contains five pages: Home, Try an Image, Run an Experiment, Explore Results, and Learn.

## Run Backend Experiments

```powershell
.\.venv\Scripts\python.exe main.py
```

The backend processes the configured dataset and saves experimental measurements, comparison images, and graphs.

## Automated Testing

Run all automated tests from the `Denoising ad` directory:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s automated_code_tests -p "test*.py" -v
```

The final integration check on November 25, 2026, completed **132 automated tests successfully**, with no failures or skipped tests.

These tests verify software functionality. Scientific performance conclusions are based on separately recorded experiment results.

## Project Structure

```text
Denoising ad/
├── frontend/                # Streamlit application
├── dataset/                 # Image datasets
├── automated_code_tests/    # Software tests
├── docs/                    # Documentation
├── results/                 # Experiment outputs
├── tuning_runs/             # Parameter tuning records
├── add_noise.py             # Noise generation
├── filter.py                # Denoising algorithms
├── image_degradation.py     # JPEG and glare simulation
├── edge_metrics.py          # Edge preservation
├── experiment_job.py        # Experiment configuration
├── main.py                  # Experiment processing
├── tune_filters.py          # Parameter optimization
└── requirements.txt         # Python dependencies
```

## Documentation

Additional information is available in:

- `Denoising ad/docs/FRONTEND_USER_GUIDE.md`
- `Denoising ad/docs/PROJECT_STRUCTURE.md`
- `Denoising ad/docs/archive/` — Historical project and team documentation.

## Project Objective

The main objective is to understand and compare classical image-denoising algorithms under different noise conditions, examining image quality, structural similarity, edge preservation, and computational performance.