# Project Structure — Image Noise Lab

This guide explains where the main files are and what they do.

## Main folders

All folders below are inside **`Denoising ad/`**.

| Folder | What it contains |
|---|---|
| `frontend/` | The website and its five pages. |
| `dataset/` | Images we use for testing filters: BSD68, Set12, Urban100_HR and original_png. |
| `automated_code_tests/` | Tests that check whether our **code works correctly**. These are **not** our research experiments. |
| `results/` | Pictures, tables and scores from experiments we run. |
| `tuning_runs/` | Saved filter settings tested on Set12. |
| `docs/` | Guides for teammates. |
| `.frontend_jobs/` | Temporary files used while the website runs experiments. |
| `.venv/` | Python packages installed on your computer. |

## Important Python files

| File | What it does |
|---|---|
| `main.py` | Runs the image experiments and saves results. |
| `add_noise.py` | Adds Gaussian, Salt-and-Pepper, Speckle and Poisson noise. |
| `image_degradation.py` | Adds JPEG compression and glare. |
| `filter.py` | Contains our six image-cleaning filters. |
| `edge_metrics.py` | Measures how well image edges are kept. |
| `experiment_job.py` | Checks experiment settings before running them. |
| `run_metadata.py` | Saves details about each experiment. |
| `tune_filters.py` | Finds good filter settings using Set12 images. |
| `compare_tuning.py` | Compares saved tuning results. |
| `visualize_tuning.py` | Makes pictures comparing filter settings. |
| `requirements.txt` | Python packages needed by the project. |
| `ReadMe.md` | Quick project information. |

The **`frontend/`** folder contains the website pages. Start the website with `frontend/app.py`.

## How the project works

1. Open the website and choose an image, noise and filters.
2. The website asks `main.py` to run the experiment.
3. `main.py` adds noise or damage, applies filters and calculates scores.
4. The output is saved in `results/resultN/`.
5. Open **Explore results** on the website to see the pictures, PSNR, SSIM and other scores.

## Important difference

- **`automated_code_tests/`** checks that our software works. It does not contain our research findings.
- **`results/`** contains the actual outputs of our image experiments.
- **`dataset/Set12/`** was used to choose optimized filter settings. BSD68, Urban100_HR and original_png can be used for evaluation; the source of original_png still needs checking.

## How to start the website

Open a terminal inside **`Denoising ad/`** and run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

Then open `http://localhost:8501`.

For step-by-step instructions, see `docs/FRONTEND_USER_GUIDE.md`.
