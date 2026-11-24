to run this program run the following in order:
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py

## Documentation

- docs/FRONTEND_USER_GUIDE.md: install, launch and use the Image Noise Lab website.
- docs/PROJECT_STRUCTURE.md: which files are the frontend, the science, the runner,
  the tuning scripts, the datasets and the generated results.
- tests/README.md: the automated tests and how to run them:
  .\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

## Frontend

Open the Image Noise Lab page from this folder:
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py

## Datasets

dataset/BSD68 (68 images), dataset/Set12 (12 images), dataset/Urban100_HR (100 images) and
dataset/original_png (12 images). Every image is read as grayscale and divided by 255.
Urban100_HR and original_png come from the teammate branch
feature/image_degradation-different_datasets-edge_preservation.
- Urban100_HR has large colour photos (about 0.6 to 1.2 megapixels). They are converted to
  grayscale. Full-dataset runs on it are much slower and save larger pictures.
- original_png: the origin and intended purpose of these images have not been verified yet.
  Check them before using their results in a research report.
- Parameter tuning (tune_filters.py) still uses only Set12.

## Edge-preservation scores (edge_metrics.py)

Every result row also saves Edge_Precision, Edge_Recall and Edge_F1 (method from the teammate
branch). Canny (thresholds 50 and 150) finds edges in the clean image and in the result; an
edge counts as kept if the other image has an edge within 1 pixel.
- Edge Precision: of the edges in the result, how many are real.
- Edge Recall: of the real edges, how many were kept.
- Edge F1: the balance of the two.
They are calculated after the timer stops. Limitation: strong noise creates edges almost
everywhere, so an unfiltered noisy image can still get a high Edge Recall and Edge F1.

## Image degradation: JPEG compression and glare (image_degradation.py)

A separate kind of experiment ("Image degradation" in the Run Experiment view), kept apart
from the 12 noise conditions:
- JPEG Compression: quality Low 90, Medium 60, High 30 (lower quality means more damage).
- Glare: strength Low 0.20, Medium 0.45, High 0.70.
Glare seed policy: seed = 42 + image_index * 1000 + 500. The seed only decides where the
glare spot is and how wide it is, so the spot stays in the same place for Low, Medium and
High of one image and only the strength changes. (The teammate's version added the level to
the seed, which moved the spot between levels.) JPEG compression is not random.
Every filter is compared with the unfiltered degraded image. The filters were made for noise,
so they can make these images worse; the Results view says so when that happens. Tuned
presets were chosen for noise on Set12, so only Default parameters are offered for JPEG and
glare.
