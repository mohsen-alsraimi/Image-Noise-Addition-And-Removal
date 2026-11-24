to run this program run the following in order:
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py

Place each dataset in its own folder under `dataset/`. For example:

dataset/
  x/
  y/
  z/

The program processes each dataset folder separately. Results are saved under
`results/` using the dataset folder name and run number, such as
`results/x_result1`, `results/y_result1`, and `results/z_result1`. On the next
run, each dataset gets its next available number (for example, `x_result2`).

Each result row also includes Edge Precision, Edge Recall, and Edge F1. These
compare Canny edge maps from the clean and processed images, allowing a
one-pixel position tolerance. The shared Canny thresholds and tolerance are
configured near the top of `main.py`.

The program also runs JPEG Compression (quality 90, 60, and 30) and a
synthetic Gaussian-hotspot glare experiment (intensity 0.20, 0.45, and 0.70).
Their results are labeled as `Image Degradation` in the CSV files, with
separate graphs for each dataset and degradation type. A representative
medium-level example and its five filter outputs are saved under
`results/sample_images/jpeg/` and `results/sample_images/glare/`.
