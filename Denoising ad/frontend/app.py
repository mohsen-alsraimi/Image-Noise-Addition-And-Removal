"""
Image Noise Lab - a simple web frontend for the denoising project.

It reuses the project's own code (add_noise.py, filter.py and main.py),
so every number shown here is calculated exactly like the experiments.

Run it from the "Denoising ad" folder:
    python frontend/app.py
then open http://127.0.0.1:5000 in your browser.
"""

import base64
import inspect
import io
import sys
import threading
import time
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no pop-up windows, we only make images for the browser

import cv2
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, Response

# Make "import main" work no matter where the app is started from.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import main as project  # noqa: E402  (the team's experiment code)

app = Flask(__name__)
MAX_FILE_MB = 20
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024 + 64 * 1024  # 20 MB photo + the form fields

DATASETS = {"BSD68": PROJECT_DIR / "dataset" / "BSD68", "Set12": PROJECT_DIR / "dataset" / "Set12"}
LEVELS = ["Low", "Medium", "High"]
NOISE_TYPES = list(project.NOISE_LEVELS.keys())
FILTER_NAMES = list(project.filters.keys())
MAX_UPLOAD_SIDE = 1024  # big photos are shrunk so the page stays fast
MIN_UPLOAD_SIDE = 16    # filters need at least a few pixels around each pixel
MAX_UPLOAD_PIXELS = 40_000_000  # refuse giant images that would use too much memory
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
# The first bytes of each file type ("magic numbers"), so a renamed text file is caught.
FILE_SIGNATURES = {b"\x89PNG\r\n\x1a\n": "PNG", b"\xff\xd8\xff": "JPEG", b"BM": "BMP"}

# Range of the "custom strength" slider for each noise type.
NOISE_CUSTOM = {
    "Gaussian": {"min": 1, "max": 100, "step": 1, "unit": "σ (out of 255)"},
    "Salt-and-Pepper": {"min": 0.01, "max": 0.5, "step": 0.01, "unit": "share of pixels"},
    "Speckle": {"min": 0.005, "max": 0.3, "step": 0.005, "unit": "variance"},
    "Poisson": {"min": 2, "max": 200, "step": 1, "unit": "peak (lower = noisier)"},
}

# Sliders for the "custom settings" mode. The names match filter.py.
FILTER_PARAMS = {
    "Gaussian Filter": [
        {"name": "kernel_size", "label": "Window size", "min": 3, "max": 15, "step": 2, "odd": True,
         "tip": "How many neighbours are mixed. Bigger window = smoother but blurrier."},
        {"name": "sigma", "label": "Blur strength (σ)", "min": 0.1, "max": 5, "step": 0.1,
         "tip": "How fast the weights drop away from the centre. Bigger = more blur."}],
    "Median Filter": [
        {"name": "kernel_size", "label": "Window size", "min": 3, "max": 11, "step": 2, "odd": True,
         "tip": "How many neighbours are sorted. Bigger removes more dots but loses thin details."}],
    "Bilateral Filter": [
        {"name": "diameter", "label": "Neighbourhood size", "min": 3, "max": 15, "step": 2, "odd": True,
         "tip": "How far the filter looks around each pixel."},
        {"name": "sigma_color", "label": "Similarity limit", "min": 0.01, "max": 0.5, "step": 0.01,
         "tip": "How different a neighbour's brightness may be and still count. Bigger = more smoothing across edges."},
        {"name": "sigma_space", "label": "Distance limit", "min": 1, "max": 150, "step": 1,
         "tip": "How quickly far-away neighbours stop counting."}],
    "NLM Filter": [
        {"name": "h", "label": "Strength (h)", "min": 0.02, "max": 0.4, "step": 0.01,
         "tip": "How strongly similar patches are averaged. Should be close to the noise level."},
        {"name": "patch_size", "label": "Patch size", "min": 3, "max": 9, "step": 2, "odd": True,
         "tip": "Size of the small squares that are compared with each other."},
        {"name": "patch_distance", "label": "Search distance", "min": 2, "max": 10, "step": 1,
         "tip": "How far away the filter searches for similar patches. Bigger = slower."}],
    "Lee Filter": [
        {"name": "window_size", "label": "Window size", "min": 3, "max": 15, "step": 2, "odd": True,
         "tip": "Area used to decide if a spot is flat (smooth it) or an edge (keep it)."}],
    "TV Filter": [
        {"name": "weight", "label": "Flattening strength", "min": 0.01, "max": 0.5, "step": 0.01,
         "tip": "How much small ups and downs are flattened. Bigger = smoother, cartoon-like."}],
}

JOBS = {}  # running and finished whole-dataset experiments


# ---------------------------------------------------------------- helpers

def to_png_data_url(image):
    """Turns a 0-1 grayscale image into something the browser can show."""
    ok, buffer = cv2.imencode(".png", (np.clip(image, 0, 1) * 255).round().astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(buffer).decode()


def list_images(dataset):
    folder = DATASETS[dataset]
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp"))


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image.astype(np.float32) / 255.0


def default_params(filter_name):
    """The default values written in filter.py."""
    signature = inspect.signature(project.filters[filter_name])
    values = {}
    for name, parameter in signature.parameters.items():
        if name in ("image", "speckle_variance"):
            continue
        values[name] = parameter.default
    return values


def params_for(filter_name, mode, noise_type, level, noise_value, custom):
    """Chooses the filter settings: default, tuned (best_parameters.json) or the user's own."""
    tuned = project.load_tuned_parameters() if mode == "tuned" and level in LEVELS else None
    params = project.get_filter_parameters(filter_name, project.filters[filter_name],
                                           noise_type, level, noise_value, tuned)
    if mode == "custom":
        for key, value in (custom.get(filter_name) or {}).items():
            spec = next((p for p in FILTER_PARAMS[filter_name] if p["name"] == key), None)
            if spec is None:
                continue
            value = float(value)
            if spec.get("odd") or spec["step"] >= 1:
                value = int(round(value))
            params[key] = value
    return params


def run_filters(original, noisy, filter_names, mode, noise_type, level, noise_value, custom, with_images=True):
    """Applies the chosen filters to the same noisy image and scores them."""
    rows = []
    for name in filter_names:
        params = params_for(name, mode, noise_type, level, noise_value, custom)
        start = time.perf_counter()
        filtered = project.filters[name](noisy.copy(), **params)
        seconds = time.perf_counter() - start
        psnr, ssim = project.calculate_metrics(original, filtered)
        row = {"filter": name, "psnr": round(float(psnr), 2), "ssim": round(float(ssim), 3),
               "psnr_exact": float(psnr), "ssim_exact": float(ssim), "seconds": seconds,
               "time_ms": round(seconds * 1000, 1),
               "params": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()
                          if k != "speckle_variance"}}
        if with_images:
            row["image"] = to_png_data_url(filtered)
            row["diff"] = to_png_data_url(np.clip(np.abs(filtered - original) * 4, 0, 1))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- pages

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def config():
    tuned_available = project.TUNED_PARAMETERS_FILE is not None and (PROJECT_DIR / project.TUNED_PARAMETERS_FILE).exists()
    return jsonify({
        "noise_types": NOISE_TYPES,
        "noise_levels": project.NOISE_LEVELS,
        "noise_custom": NOISE_CUSTOM,
        "filters": FILTER_NAMES,
        "filter_params": FILTER_PARAMS,
        "defaults": {name: default_params(name) for name in FILTER_NAMES},
        "datasets": {name: list_images(name) for name in DATASETS},
        "tuned_available": tuned_available,
        "max_file_mb": MAX_FILE_MB,
        "results": sorted([p.name for p in (PROJECT_DIR / "results").glob("result*") if (p / "summary.csv").exists()],
                          key=lambda n: int(n[6:]) if n[6:].isdigit() else 0),
    })


@app.route("/dataset/<dataset>/<name>")
def dataset_image(dataset, name):
    if dataset not in DATASETS:
        abort(404)
    return send_from_directory(DATASETS[dataset], name)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": f"This file is too big. Please use a photo smaller than {MAX_FILE_MB} MB."}), 413


def check_upload(file_name, data):
    """Checks an uploaded file and explains clearly what is wrong with it."""
    if not file_name.lower().endswith(ALLOWED_EXTENSIONS):
        raise ValueError("Please upload a PNG, JPG or BMP photo.")
    if len(data) == 0:
        raise ValueError("This file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"This file is too big. Please use a photo smaller than {MAX_FILE_MB} MB.")
    if not any(data.startswith(signature) for signature in FILE_SIGNATURES):
        raise ValueError("This file is not really a PNG, JPG or BMP photo (only its name looks like one).")


def read_source_image():
    """Gets the clean image: a dataset photo or an uploaded photo."""
    upload = request.files.get("upload")
    if upload and upload.filename:
        raw = upload.read()
        check_upload(upload.filename, raw)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("This photo seems to be damaged, so it cannot be opened. Please try another file.")
        height, width = image.shape
        if min(height, width) < MIN_UPLOAD_SIDE:
            raise ValueError(f"This photo is too small. It must be at least {MIN_UPLOAD_SIDE} × {MIN_UPLOAD_SIDE} pixels.")
        if height * width > MAX_UPLOAD_PIXELS:
            raise ValueError("This photo has too many pixels. Please use a smaller photo.")
        scale = MAX_UPLOAD_SIDE / max(height, width)
        if scale < 1:
            image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return image.astype(np.float32) / 255.0, upload.filename
    dataset = request.form.get("dataset", "BSD68")
    name = request.form.get("image", "")
    if dataset not in DATASETS or name not in list_images(dataset):
        raise ValueError("Unknown image.")
    return load_gray(DATASETS[dataset] / name), f"{dataset}/{name}"


@app.route("/api/run-single", methods=["POST"])
def run_single():
    """One image: one noise (or every noise and level) and the chosen filters."""
    try:
        original, label = read_source_image()
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    form = request.form
    filter_names = [f for f in form.getlist("filters") if f in FILTER_NAMES] or FILTER_NAMES
    mode = form.get("mode", "tuned")
    custom = {}
    for name in FILTER_NAMES:
        values = {p["name"]: form.get(f"{name}::{p['name']}") for p in FILTER_PARAMS[name]}
        custom[name] = {k: v for k, v in values.items() if v not in (None, "")}
    seed = int(form.get("seed", project.BASE_SEED))

    if form.get("scope") == "all":
        # Every noise type at every level: a big table, no images for each filter.
        table = []
        for noise_type in NOISE_TYPES:
            for level in LEVELS:
                value = project.NOISE_LEVELS[noise_type][level]
                noisy = project.generate_noisy_image(original, noise_type, value, seed)
                psnr, ssim = project.calculate_metrics(original, noisy)
                rows = run_filters(original, noisy, filter_names, mode, noise_type, level, value, custom, with_images=False)
                table.append({"noise": noise_type, "level": level, "value": value,
                              "noisy_psnr": round(float(psnr), 2), "noisy_ssim": round(float(ssim), 3),
                              "noisy_image": to_png_data_url(noisy), "filters": rows})
        return jsonify({"scope": "all", "label": label, "original": to_png_data_url(original), "table": table})

    noise_type = form.get("noise", "Gaussian")
    level = form.get("level", "Medium")
    value = float(form.get("custom_value")) if level == "Custom" else project.NOISE_LEVELS[noise_type][level]
    noisy = project.generate_noisy_image(original, noise_type, value, seed)
    psnr, ssim = project.calculate_metrics(original, noisy)
    rows = run_filters(original, noisy, filter_names, mode, noise_type, level, value, custom)
    return jsonify({
        "scope": "one", "label": label, "noise": noise_type, "level": level, "value": value,
        "original": to_png_data_url(original), "noisy": to_png_data_url(noisy),
        "noisy_psnr": round(float(psnr), 2), "noisy_ssim": round(float(ssim), 3),
        "size": f"{original.shape[1]} × {original.shape[0]}", "filters": rows,
    })


# ---------------------------------------------------------------- whole dataset

def batch_worker(job_id, dataset, count, noises, levels, filter_names, mode):
    job = JOBS[job_id]
    images = [(name, load_gray(DATASETS[dataset] / name)) for name in list_images(dataset)][:count]
    job["total"] = len(images) * len(noises) * len(levels)
    rows = []
    for image_index, (name, original) in enumerate(images):
        for noise_type in noises:
            noise_index = NOISE_TYPES.index(noise_type)
            for level in levels:
                if job["cancel"]:
                    job["state"] = "cancelled"
                    return
                level_index = LEVELS.index(level)
                value = project.NOISE_LEVELS[noise_type][level]
                # Same seed rule as main.py, so the numbers match the team's experiments.
                seed = project.noise_seed(image_index, noise_index, level_index)
                noisy = project.generate_noisy_image(original, noise_type, value, seed)
                psnr, ssim = project.calculate_metrics(original, noisy)
                rows.append({"Image": name, "Noise_Type": noise_type, "Noise_Level": level,
                             "Filter": "Noisy (Baseline)", "PSNR": psnr, "SSIM": ssim, "Execution_Time": 0.0})
                for row in run_filters(original, noisy, filter_names, mode, noise_type, level, value, {}, with_images=False):
                    rows.append({"Image": name, "Noise_Type": noise_type, "Noise_Level": level,
                                 "Filter": row["filter"], "PSNR": row["psnr_exact"], "SSIM": row["ssim_exact"],
                                 "Execution_Time": row["seconds"]})
                job["done"] += 1
                job["current"] = f"{name} · {noise_type} · {level}"
    job["rows"] = rows
    job["state"] = "finished"


def summarise(frame):
    """Average PSNR / SSIM / time for every noise, level and filter."""
    grouped = frame.groupby(["Noise_Type", "Noise_Level", "Filter"]).agg(
        PSNR=("PSNR", "mean"), SSIM=("SSIM", "mean"), Time=("Execution_Time", "mean"), Images=("Image", "nunique"))
    return [{"noise": n, "level": l, "filter": f, "psnr": round(r.PSNR, 2), "ssim": round(r.SSIM, 3),
             "time_ms": round(r.Time * 1000, 2), "images": int(r.Images)}
            for (n, l, f), r in grouped.iterrows()]


@app.route("/api/batch", methods=["POST"])
def start_batch():
    data = request.get_json()
    dataset = data.get("dataset", "BSD68")
    if dataset not in DATASETS:
        return jsonify({"error": "Unknown dataset."}), 400
    available = len(list_images(dataset))
    count = available if data.get("count") in (None, "all") else max(1, min(int(data["count"]), available))
    noises = [n for n in data.get("noises", NOISE_TYPES) if n in NOISE_TYPES]
    levels = [l for l in data.get("levels", LEVELS) if l in LEVELS]
    filter_names = [f for f in data.get("filters", FILTER_NAMES) if f in FILTER_NAMES]
    if not (noises and levels and filter_names):
        return jsonify({"error": "Pick at least one noise, one level and one filter."}), 400
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"state": "running", "done": 0, "total": 1, "current": "", "cancel": False, "rows": None,
                    "started": time.time(), "dataset": dataset}
    threading.Thread(target=batch_worker, daemon=True,
                     args=(job_id, dataset, count, noises, levels, filter_names, data.get("mode", "tuned"))).start()
    return jsonify({"job": job_id})


@app.route("/api/batch/<job_id>")
def batch_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    answer = {k: job[k] for k in ("state", "done", "total", "current")}
    answer["seconds"] = round(time.time() - job["started"])
    if job["state"] == "finished":
        answer["summary"] = summarise(pd.DataFrame(job["rows"]))
    return jsonify(answer)


@app.route("/api/batch/<job_id>/cancel", methods=["POST"])
def batch_cancel(job_id):
    if job_id in JOBS:
        JOBS[job_id]["cancel"] = True
    return jsonify({"ok": True})


@app.route("/api/batch/<job_id>/csv")
def batch_csv(job_id):
    job = JOBS.get(job_id)
    if not job or job["rows"] is None:
        return jsonify({"error": "No results yet"}), 404
    return Response(pd.DataFrame(job["rows"]).to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=results_{job_id}.csv"})


@app.route("/api/saved/<name>")
def saved_result(name):
    """Opens an experiment that main.py already saved in the results folder."""
    folder = PROJECT_DIR / "results" / name
    csv_path = folder / "experiment_results.csv"
    if not name.startswith("result") or not csv_path.exists():
        return jsonify({"error": "Not found"}), 404
    frame = pd.read_csv(csv_path)
    pictures = sorted(p.name for p in (folder / "comparisons").glob("*.png")) if (folder / "comparisons").exists() else []
    return jsonify({"summary": summarise(frame), "images": int(frame["Image"].nunique()), "pictures": pictures})


@app.route("/results-file/<name>/<path:filename>")
def saved_picture(name, filename):
    return send_from_directory(PROJECT_DIR / "results" / name, filename)


if __name__ == "__main__":
    print("\n  Image Noise Lab is running. Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
