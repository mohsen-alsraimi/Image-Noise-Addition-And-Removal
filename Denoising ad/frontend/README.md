# Image Noise Lab (frontend)

A simple website for the denoising project. It uses the project's own code
(`add_noise.py`, `filter.py`, `main.py`), so every score matches the experiments.

## What you can do

| Tab | What it does |
|---|---|
| **Try one photo** | Pick a BSD68 / Set12 photo or upload your own, add one noise (Low / Medium / High / custom strength) or all noises and levels, run one or all filters with recommended, default or your own settings, and compare the results side by side. Click a result to compare it with a slider or zoom into the real pixels. |
| **Whole dataset** | Run the experiment on some or all photos, with the noises, levels and filters you choose. Shows average PSNR / SSIM / time as charts and a table, and lets you download the CSV. Uses the same seeds as `main.py`. |
| **Find best setting** | Pick a photo, a noise and one filter setting (for example the NLM strength h). The same noisy photo is cleaned with many values of that setting and a chart shows which value scores best. |
| **Saved results** | Open any `results/resultN` folder made by `main.py` and see its charts, table and comparison pictures. |
| **How it works** | Plain-language explanations of every noise type, filter and score. |

Rest the mouse on any button for 1.5 seconds to see what it does.

**Accuracy and reproducibility**

- Every result shows PSNR, SSIM and MSE, always measured on the full-size images. A result identical to the clean photo shows PSNR "∞".
- The page shows what the noise really did (share of changed pixels, measured noise size).
- Each run shows its seed, settings, software versions and a "fingerprint" (a short code made from every pixel), and can be downloaded as JSON and CSV. Running the same settings again gives the same fingerprints.
- Whole-dataset runs can be repeated with up to 5 different seeds; the table then shows the average ± the spread. Repeat 1 uses exactly the same seeds as `main.py`.
- Whole-dataset runs can be saved to `results/resultN` (experiment_results.csv, summary.csv, run_config.json), so they also appear under Saved results.
- Wrong inputs (an even window size, a value out of range, a bad seed or file) get a clear message instead of an error.

## How to run it on your computer

You need Python 3.10 or newer. Open a terminal **in the `Denoising ad` folder**.

**Windows (PowerShell)**

```
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe frontend\app.py
```

**Mac / Linux**

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python frontend/app.py
```

Then open **http://127.0.0.1:5000** in your browser. Press `Ctrl + C` in the terminal to stop it.

## Files

- `app.py` – the small Flask server. It loads the photos, calls the project's noise and filter functions and sends the results to the page.
- `templates/index.html` – the page layout.
- `static/style.css` – the look of the page.
- `static/app.js` – the buttons, charts, comparison slider and tooltips.
- `static/chart.umd.min.js` – the Chart.js library (included so the page works without internet).

## Tests

```
python -B -m unittest discover -s tests -p "test_frontend.py" -v
```

42 tests check that the website gives the same scores as the project code (PSNR, SSIM and MSE), that infinite PSNR is handled, that the same seed gives the same pixels, that bad uploads and bad settings get a clear message, that the setting search picks the real best value, and that whole-dataset runs (with repeats and saving) work and match main.py.

Notes: uploads must be PNG, JPG or BMP and at most 20 MB. Uploaded photos are turned into grayscale and shrunk to at most 1024 pixels on the longest side to keep things fast.
"Recommended" settings come from `best_parameters.json`, the same file `main.py` uses.
