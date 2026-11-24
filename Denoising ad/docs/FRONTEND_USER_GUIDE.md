# Image Noise Lab — Frontend User Guide

This guide explains how to open and use our Image Noise Lab website.

## 1. What does the website do?

The website lets us add noise or image damage to a picture, apply filters, and compare the results. It helps us see which filter works best.

We use **six filters**: Gaussian, Median, Bilateral, NLM, Lee, and TV.

## 2. How to start the website

Open a terminal in the **`Denoising ad`** folder.

**First time only:**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Start the website:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

Open **http://localhost:8501** in your browser. Press **Ctrl+C** in the terminal to stop the website.

## 3. What are the five pages?

| Page | What you can do |
| --- | --- |
| **Home** | Understand the project and choose where to start. |
| **Try an image** | Test noise and filters on one image. |
| **Run an experiment** | Test one image or a full dataset with more settings. |
| **Explore results** | View saved images, scores, and filter comparisons. |
| **Learn** | Read simple explanations of the terms. |

## 4. Try your first image

1. Open **Try an image**.
2. Choose **BSD68** and select an image.
3. Select **Noise → Salt-and-Pepper → Medium**.
4. Choose the **Median** filter and **Default settings**.
5. Click **Run this test**.
6. Look at the original, noisy, and filtered images. Open **Explore results** for more details.

## 5. Run a bigger experiment

Open **Run an experiment** and choose:

1. **Images:** one image or a full dataset.
2. **What to test:** a noise type or image damage, with a strength.
3. **Filters:** one filter or all six.
4. **Settings:** Default, Optimized (for noise), or your own values.
5. Click **Run experiment**.

You can also test **all 12 noise conditions** (four noise types × three strengths) or **all six image-damage conditions** (JPEG and Glare × three strengths).

## 6. Available datasets

| Dataset | Images | Use |
| --- | ---: | --- |
| BSD68 | 68 | Testing denoising performance. |
| Set12 | 12 | Used to tune filter settings. |
| Urban100_HR | 100 | Testing on larger building images. |
| original_png | 12 | Additional images; their source still needs checking. |

## 7. What do the scores mean?

- **PSNR:** Measures pixel-level quality. **Higher is better.**
- **SSIM:** Measures how similar the image looks to the original. **Closer to 1 is better.**
- **Execution time:** How long the filter took.
- **Edge F1:** Compares detected edges. It can sometimes be misleading, so check PSNR and SSIM too.

**Important:** The best filter in a comparison might still make the image worse than the unfiltered noisy image. Check the **“Filtering helped”** message.

## 8. Where are results saved?

Each test creates a new folder inside **`results/`**, such as `results/result24/`.

Open **Explore results** to see saved comparisons, graphs, scores, and settings. Generated results normally stay on your computer and are not uploaded to GitHub.

## 9. Common problems

| Problem | What to do |
| --- | --- |
| Missing Python package | Run the installation command from Section 2. |
| Website does not start | Make sure the terminal is inside `Denoising ad`. |
| Port 8501 is busy | Stop the other Streamlit app or use `--server.port 8502`. |
| Experiment is slow | Try one image first; large datasets and NLM/TV take longer. |

For information about the code and folders, see **`docs/PROJECT_STRUCTURE.md`**.
