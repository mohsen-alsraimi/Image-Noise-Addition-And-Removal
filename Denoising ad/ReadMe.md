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
`results/` using the dataset folder name and run number

Each result row also includes Edge Precision, Edge Recall, and Edge F1.

The program also runs JPEG Compression  and a
synthetic glare experiment 
Their results are labeled as `Image Degradation` in the CSV files, with
separate graphs for each dataset and degradation type. 


In `results/sample_images/jpeg/` and `results/sample_images/glare/` it shows how images look under these masks.
