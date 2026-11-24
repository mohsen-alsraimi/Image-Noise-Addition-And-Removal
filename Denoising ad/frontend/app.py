"""
Image Noise Lab - a simple web frontend for the denoising project.

It reuses the project's own code (add_noise.py, filter.py and main.py),
so every number shown here is calculated exactly like the experiments.

Run it from the "Denoising ad" folder:
    python frontend/app.py
then open http://127.0.0.1:5000 in your browser.
"""

import base64
import hashlib
import inspect
import json
import platform
import sys
import threading
import time
import uuid
from datetime import datetime
from importlib import metadata
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
RESULTS_DIR = PROJECT_DIR / "results"  # where main.py (and this website, if asked) saves experiments
LEVELS = ["Low", "Medium", "High"]
NOISE_TYPES = list(project.NOISE_LEVELS.keys())
FILTER_NAMES = list(project.filters.keys())
MAX_UPLOAD_SIDE = 1024  # big photos are shrunk so the page stays fast
MIN_UPLOAD_SIDE = 16    # filters need at least a few pixels around each pixel
MAX_UPLOAD_PIXELS = 40_000_000  # refuse giant images that would use too much memory
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
# The first bytes of each file type ("magic numbers"), so a renamed text file is caught.
FILE_SIGNATURES = {b"\x89PNG\r\n\x1a\n": "PNG", b"\xff\xd8\xff": "JPEG", b"BM": "BMP"}
MAX_REPEATS = 5        # whole-dataset runs can be repeated with new seeds
MAX_SWEEP_VALUES = 12  # "find the best setting" tries at most this many values
MAX_JOBS_KEPT = 20     # old finished experiments are forgotten so memory does not grow

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


def score(original, image):
    """PSNR, SSIM and MSE against the clean image, always on the full-size arrays.

    PSNR and SSIM come from main.calculate_metrics (the experiment code).
    MSE = average of (clean - result)^2, in 0-1 units, calculated in float64.
    Identical images have MSE 0 and an infinite PSNR; JSON cannot hold
    infinity, so PSNR is then None and "identical" is True.
    """
    mse = float(np.mean((original.astype(np.float64) - image.astype(np.float64)) ** 2))
    psnr, ssim = project.calculate_metrics(original, image)
    identical = mse == 0.0
    return {"psnr": None if identical else round(float(psnr), 2), "ssim": round(float(ssim), 3),
            "mse": round(mse, 6), "identical": identical,
            "psnr_exact": None if identical else float(psnr), "ssim_exact": float(ssim)}


def fingerprint(image):
    """A short code made from every pixel: the same image always gives the same code."""
    pixels = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes())
    return digest.hexdigest()[:12]


def noise_stats(original, noisy):
    """What the noise really did: share of changed pixels and the measured size of the change."""
    change = noisy.astype(np.float64) - original.astype(np.float64)
    changed = np.abs(change) * 255 >= 0.5  # at least half a grey level
    return {"changed_pixels": int(changed.sum()), "changed_share": round(float(changed.mean()), 4),
            "measured_sigma": round(float(change.std() * 255), 2)}


def library_versions():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "opencv-python", "scikit-image", "pandas", "flask"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def default_params(filter_name):
    """The default values written in filter.py."""
    signature = inspect.signature(project.filters[filter_name])
    values = {}
    for name, parameter in signature.parameters.items():
        if name in ("image", "speckle_variance"):
            continue
        values[name] = parameter.default
    return values


def check_setting(filter_name, key, raw):
    """Turns a custom slider value into a number and refuses values the filter cannot use."""
    spec = next((p for p in FILTER_PARAMS[filter_name] if p["name"] == key), None)
    if spec is None:
        raise ValueError(f"{filter_name} has no setting called '{key}'.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{filter_name}: '{spec['label']}' must be a number.")
    if not np.isfinite(value) or not (spec["min"] <= value <= spec["max"]):
        raise ValueError(f"{filter_name}: '{spec['label']}' must be between {spec['min']} and {spec['max']}.")
    if spec.get("odd") or spec["step"] >= 1:
        if value != int(value):
            raise ValueError(f"{filter_name}: '{spec['label']}' must be a whole number.")
        value = int(value)
        if spec.get("odd") and value % 2 == 0:
            raise ValueError(f"{filter_name}: '{spec['label']}' must be an odd number (3, 5, 7, ...).")
    return value


def check_noise_value(noise_type, raw):
    spec = NOISE_CUSTOM[noise_type]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError("The custom noise strength must be a number.")
    if not np.isfinite(value) or not (spec["min"] <= value <= spec["max"]):
        raise ValueError(f"The custom {noise_type} strength must be between {spec['min']} and {spec['max']}.")
    return value


def check_seed(raw):
    try:
        seed = int(raw)
    except (TypeError, ValueError):
        raise ValueError("The random seed must be a whole number.")
    if seed < 0:
        raise ValueError("The random seed must be 0 or larger.")
    return seed


def params_for(filter_name, mode, noise_type, level, noise_value, custom):
    """Chooses the filter settings: default, tuned (best_parameters.json) or the user's own."""
    tuned = project.load_tuned_parameters() if mode == "tuned" and level in LEVELS else None
    params = project.get_filter_parameters(filter_name, project.filters[filter_name],
                                           noise_type, level, noise_value, tuned)
    if mode == "custom":
        for key, value in (custom.get(filter_name) or {}).items():
            params[key] = check_setting(filter_name, key, value)
    return params


def run_filters(original, noisy, filter_names, mode, noise_type, level, noise_value, custom, with_images=True):
    """Applies the chosen filters to the same noisy image and scores them."""
    rows = []
    for name in filter_names:
        params = params_for(name, mode, noise_type, level, noise_value, custom)
        start = time.perf_counter()
        filtered = project.filters[name](noisy.copy(), **params)
        seconds = time.perf_counter() - start
        row = {"filter": name, "seconds": seconds, "time_ms": round(seconds * 1000, 1),
               "params": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()
                          if k != "speckle_variance"}}
        row.update(score(original, filtered))
        if with_images:
            row["image"] = to_png_data_url(filtered)
            row["diff"] = to_png_data_url(np.clip(np.abs(filtered - original) * 4, 0, 1))
            row["fingerprint"] = fingerprint(filtered)
        rows.append(row)
    return rows


def read_custom_settings(form):
    custom = {}
    for name in FILTER_NAMES:
        values = {p["name"]: form.get(f"{name}::{p['name']}") for p in FILTER_PARAMS[name]}
        custom[name] = {k: v for k, v in values.items() if v not in (None, "")}
    return custom


# ---------------------------------------------------------------- pages

@app.route("/")
def index():
    return render_template("index.html")


def saved_runs():
    if not RESULTS_DIR.exists():
        return []
    names = [p.name for p in RESULTS_DIR.glob("result*") if (p / "experiment_results.csv").exists()]
    return sorted(names, key=lambda n: int(n[6:]) if n[6:].isdigit() else 0)


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
        "max_repeats": MAX_REPEATS,
        "results": saved_runs(),
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
        resized = scale < 1
        if resized:
            image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        note = f"shrunk from {width} × {height}" if resized else ""
        return image.astype(np.float32) / 255.0, upload.filename, "upload", note
    dataset = request.form.get("dataset", "BSD68")
    name = request.form.get("image", "")
    if dataset not in DATASETS or name not in list_images(dataset):
        raise ValueError("Unknown image.")
    return load_gray(DATASETS[dataset] / name), f"{dataset}/{name}", "dataset", ""


def read_noise_choice(form):
    noise_type = form.get("noise", "Gaussian")
    if noise_type not in NOISE_TYPES:
        raise ValueError("Unknown noise type.")
    level = form.get("level", "Medium")
    if level == "Custom":
        return noise_type, level, check_noise_value(noise_type, form.get("custom_value"))
    if level not in LEVELS:
        raise ValueError("Unknown noise strength.")
    return noise_type, level, project.NOISE_LEVELS[noise_type][level]


@app.route("/api/run-single", methods=["POST"])
def run_single():
    """One image: one noise (or every noise and level) and the chosen filters."""
    form = request.form
    try:
        original, label, source, source_note = read_source_image()
        seed = check_seed(form.get("seed", project.BASE_SEED))
        filter_names = [f for f in form.getlist("filters") if f in FILTER_NAMES] or FILTER_NAMES
        mode = form.get("mode", "tuned")
        if mode not in ("tuned", "default", "custom"):
            raise ValueError("Unknown settings mode.")
        custom = read_custom_settings(form)
        if mode == "custom":
            for name in filter_names:  # check every custom value before doing any work
                for key, value in custom[name].items():
                    check_setting(name, key, value)

        run_info = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "image": label, "image_source": source, "image_note": source_note,
            "image_size": f"{original.shape[1]} × {original.shape[0]}",
            "image_fingerprint": fingerprint(original), "seed": seed, "settings_mode": mode,
            "versions": library_versions(),
        }

        if form.get("scope") == "all":
            # Every noise type at every level: a big table, no images for each filter.
            table = []
            for noise_type in NOISE_TYPES:
                for level in LEVELS:
                    value = project.NOISE_LEVELS[noise_type][level]
                    noisy = project.generate_noisy_image(original, noise_type, value, seed)
                    rows = run_filters(original, noisy, filter_names, mode, noise_type, level, value, custom, with_images=False)
                    entry = {"noise": noise_type, "level": level, "value": value,
                             "noisy_image": to_png_data_url(noisy), "filters": rows,
                             "stats": noise_stats(original, noisy)}
                    entry.update({"noisy_" + k: v for k, v in score(original, noisy).items()})
                    table.append(entry)
            return jsonify({"scope": "all", "label": label, "original": to_png_data_url(original),
                            "table": table, "run_info": run_info})

        noise_type, level, value = read_noise_choice(form)
        noisy = project.generate_noisy_image(original, noise_type, value, seed)
        rows = run_filters(original, noisy, filter_names, mode, noise_type, level, value, custom)
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    run_info.update({"noise": noise_type, "level": level, "noise_value": value,
                     "noisy_fingerprint": fingerprint(noisy),
                     "filters": {r["filter"]: {"settings": r["params"], "fingerprint": r["fingerprint"]} for r in rows}})
    answer = {"scope": "one", "label": label, "noise": noise_type, "level": level, "value": value,
              "original": to_png_data_url(original), "noisy": to_png_data_url(noisy),
              "size": run_info["image_size"], "filters": rows, "stats": noise_stats(original, noisy),
              "run_info": run_info}
    answer.update({"noisy_" + k: v for k, v in score(original, noisy).items()})
    return jsonify(answer)


@app.route("/api/sweep", methods=["POST"])
def run_sweep():
    """Tries one filter setting at many values on the same noisy photo, to find the best value."""
    form = request.form
    try:
        original, label, _, _ = read_source_image()
        seed = check_seed(form.get("seed", project.BASE_SEED))
        noise_type, level, value = read_noise_choice(form)
        name = form.get("filter", "")
        if name not in FILTER_NAMES:
            raise ValueError("Please choose a filter.")
        key = form.get("param", "")
        spec = next((p for p in FILTER_PARAMS[name] if p["name"] == key), None)
        if spec is None:
            raise ValueError("Please choose a setting to test.")
        values = np.arange(spec["min"], spec["max"] + spec["step"] / 2, spec["step"])
        if len(values) > MAX_SWEEP_VALUES:  # spread the tries evenly over the whole range
            values = np.linspace(spec["min"], spec["max"], MAX_SWEEP_VALUES)
        # snap every value onto the slider's steps, counted from the minimum (so window sizes stay odd)
        snapped = {round(spec["min"] + round((v - spec["min"]) / spec["step"]) * spec["step"], 4) for v in values}
        values = sorted(check_setting(name, key, v) for v in snapped)
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    noisy = project.generate_noisy_image(original, noise_type, value, seed)
    base = project.get_filter_parameters(name, project.filters[name], noise_type, level, value,
                                         project.load_tuned_parameters() if level in LEVELS else None)
    rows = []
    for v in values:
        params = dict(base, **{key: v})
        start = time.perf_counter()
        filtered = project.filters[name](noisy.copy(), **params)
        seconds = time.perf_counter() - start
        row = {"value": v, "time_ms": round(seconds * 1000, 1)}
        row.update(score(original, filtered))
        rows.append(row)
    best = max(rows, key=lambda r: r["psnr_exact"] if r["psnr_exact"] is not None else float("inf"))
    best_params = dict(base, **{key: best["value"]})
    best_image = project.filters[name](noisy.copy(), **best_params)
    noisy_score = score(original, noisy)
    return jsonify({"label": label, "filter": name, "param": key, "param_label": spec["label"],
                    "noise": noise_type, "level": level, "noise_value": value, "seed": seed,
                    "other_settings": {k: v for k, v in base.items() if k not in (key, "speckle_variance")},
                    "rows": rows, "best": best, "noisy_psnr": noisy_score["psnr"],
                    "noisy": to_png_data_url(noisy), "best_image": to_png_data_url(best_image)})


# ---------------------------------------------------------------- whole dataset

def batch_worker(job_id, dataset, count, noises, levels, filter_names, mode, repeats, save):
    job = JOBS[job_id]
    images = [(name, load_gray(DATASETS[dataset] / name)) for name in list_images(dataset)][:count]
    job["total"] = len(images) * len(noises) * len(levels) * repeats
    rows = []
    for repeat in range(repeats):
        for image_index, (name, original) in enumerate(images):
            for noise_type in noises:
                noise_index = NOISE_TYPES.index(noise_type)
                for level in levels:
                    if job["cancel"]:
                        job["state"] = "cancelled"
                        return
                    level_index = LEVELS.index(level)
                    value = project.NOISE_LEVELS[noise_type][level]
                    # Repeat 1 uses exactly main.py's seed rule, so it matches the team's experiments.
                    # Every extra repeat shifts the seed, which gives new random noise.
                    seed = project.noise_seed(image_index, noise_index, level_index) + repeat * 100_000
                    noisy = project.generate_noisy_image(original, noise_type, value, seed)
                    base = {"Image": name, "Noise_Type": noise_type, "Noise_Level": level,
                            "Repeat": repeat + 1, "Seed": seed}
                    s = score(original, noisy)
                    rows.append(dict(base, Filter="Noisy (Baseline)", PSNR=s["psnr_exact"], SSIM=s["ssim_exact"],
                                     MSE=s["mse"], Execution_Time=0.0))
                    for row in run_filters(original, noisy, filter_names, mode, noise_type, level, value, {}, with_images=False):
                        rows.append(dict(base, Filter=row["filter"], PSNR=row["psnr_exact"], SSIM=row["ssim_exact"],
                                         MSE=row["mse"], Execution_Time=row["seconds"]))
                    job["done"] += 1
                    job["current"] = f"{name} · {noise_type} · {level}" + (f" · repeat {repeat + 1}" if repeats > 1 else "")
    job["rows"] = rows
    if save:
        job["saved_to"] = save_run(job, rows)
    job["state"] = "finished"


def next_result_folder():
    numbers = [int(n[6:]) for n in saved_runs() if n[6:].isdigit()]
    numbers += [int(p.name[6:]) for p in RESULTS_DIR.glob("result*") if p.name[6:].isdigit()]
    return RESULTS_DIR / f"result{max(numbers, default=0) + 1}"


def save_run(job, rows):
    """Saves the experiment like main.py does, so it also appears under 'Saved results'."""
    folder = next_result_folder()
    folder.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "experiment_results.csv", index=False)
    frame.groupby(["Noise_Type", "Noise_Level", "Filter"]).agg(
        {"PSNR": ["mean", "std"], "SSIM": ["mean", "std"], "Execution_Time": ["mean", "std"]}
    ).reset_index().to_csv(folder / "summary.csv", index=False)
    (folder / "run_config.json").write_text(json.dumps(job["config"], indent=2))
    return folder.name


def summarise(frame):
    """Average (and spread) of PSNR / SSIM / MSE / time for every noise, level and filter."""
    has_mse = "MSE" in frame.columns
    rows = []
    for (n, l, f), g in frame.groupby(["Noise_Type", "Noise_Level", "Filter"]):
        psnr = g["PSNR"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "noise": n, "level": l, "filter": f,
            "psnr": round(float(psnr.mean()), 2) if len(psnr) else None,
            "psnr_std": round(float(psnr.std()), 2) if len(psnr) > 1 else 0.0,
            "ssim": round(float(g["SSIM"].mean()), 3),
            "ssim_std": round(float(g["SSIM"].std()), 3) if len(g) > 1 else 0.0,
            "mse": round(float(g["MSE"].mean()), 6) if has_mse else None,
            "time_ms": round(float(g["Execution_Time"].mean()) * 1000, 2),
            "images": int(g["Image"].nunique()), "runs": int(len(g)),
        })
    return rows


def forget_old_jobs():
    finished = [k for k, j in JOBS.items() if j["state"] != "running"]
    for job_id in sorted(finished, key=lambda k: JOBS[k]["started"])[:max(0, len(JOBS) - MAX_JOBS_KEPT + 1)]:
        del JOBS[job_id]


@app.route("/api/batch", methods=["POST"])
def start_batch():
    data = request.get_json(silent=True) or {}
    dataset = data.get("dataset", "BSD68")
    if dataset not in DATASETS:
        return jsonify({"error": "Unknown dataset."}), 400
    available = len(list_images(dataset))
    try:
        count = available if data.get("count") in (None, "all") else max(1, min(int(data["count"]), available))
        repeats = int(data.get("repeats", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "The number of photos and repeats must be whole numbers."}), 400
    if not 1 <= repeats <= MAX_REPEATS:
        return jsonify({"error": f"Repeats must be between 1 and {MAX_REPEATS}."}), 400
    noises = [n for n in data.get("noises", NOISE_TYPES) if n in NOISE_TYPES]
    levels = [l for l in data.get("levels", LEVELS) if l in LEVELS]
    filter_names = [f for f in data.get("filters", FILTER_NAMES) if f in FILTER_NAMES]
    mode = data.get("mode", "tuned")
    if mode not in ("tuned", "default"):
        return jsonify({"error": "Unknown settings mode."}), 400
    if not (noises and levels and filter_names):
        return jsonify({"error": "Pick at least one noise, one level and one filter."}), 400

    forget_old_jobs()
    job_id = uuid.uuid4().hex[:8]
    config = {
        "created": datetime.now().isoformat(timespec="seconds"), "made_by": "Image Noise Lab website",
        "dataset": dataset, "images": list_images(dataset)[:count], "noises": noises, "levels": levels,
        "noise_values": {n: {l: project.NOISE_LEVELS[n][l] for l in levels} for n in noises},
        "filters": filter_names, "settings_mode": mode,
        "settings_file": str(project.TUNED_PARAMETERS_FILE) if mode == "tuned" else None,
        "repeats": repeats, "seed_rule": "main.noise_seed(image_index, noise_index, level_index) + (repeat - 1) * 100000",
        "base_seed": project.BASE_SEED, "versions": library_versions(),
    }
    JOBS[job_id] = {"state": "running", "done": 0, "total": 1, "current": "", "cancel": False, "rows": None,
                    "started": time.time(), "dataset": dataset, "config": config, "saved_to": None}
    threading.Thread(target=batch_worker, daemon=True,
                     args=(job_id, dataset, count, noises, levels, filter_names, mode, repeats,
                           bool(data.get("save")))).start()
    return jsonify({"job": job_id})


@app.route("/api/batch/<job_id>")
def batch_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    answer = {k: job[k] for k in ("state", "done", "total", "current", "saved_to")}
    answer["seconds"] = round(time.time() - job["started"])
    if job["state"] == "finished":
        answer["summary"] = summarise(pd.DataFrame(job["rows"]))
        answer["config"] = job["config"]
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


@app.route("/api/batch/<job_id>/config")
def batch_config(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return Response(json.dumps(job["config"], indent=2), mimetype="application/json",
                    headers={"Content-Disposition": f"attachment; filename=run_config_{job_id}.json"})


@app.route("/api/saved/<name>")
def saved_result(name):
    """Opens an experiment that main.py (or this website) saved in the results folder."""
    if not name.startswith("result") or name not in saved_runs():
        return jsonify({"error": "Not found"}), 404
    folder = RESULTS_DIR / name
    frame = pd.read_csv(folder / "experiment_results.csv")
    pictures = sorted(p.name for p in (folder / "comparisons").glob("*.png")) if (folder / "comparisons").exists() else []
    config_file = folder / "run_config.json"
    return jsonify({"summary": summarise(frame), "images": int(frame["Image"].nunique()), "pictures": pictures,
                    "has_config": config_file.exists()})


@app.route("/results-file/<name>/<path:filename>")
def saved_picture(name, filename):
    if name not in saved_runs():
        abort(404)
    return send_from_directory(RESULTS_DIR / name, filename)


if __name__ == "__main__":
    print("\n  Image Noise Lab is running. Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
