"""
Saves the settings of each experiment run to run_config.json.

"required_to_rerun" holds what is needed to repeat the run.
"useful_for_interpretation" holds context for reading its results.
"""

import contextlib
import hashlib
import importlib
import inspect
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from edge_metrics import EDGE_METRIC_LIMITATIONS, EDGE_METRIC_SETTINGS

CONFIG_NAME = "run_config.json"
PROJECT_DIR = Path(__file__).resolve().parent
CODE_FILES = (
    "add_noise.py", "filter.py", "main.py", "run_metadata.py",
    "edge_metrics.py", "image_degradation.py",
)
MAX_LISTED_FILES = 100

LIBRARY_MODULES = {
    "numpy": "numpy",
    "opencv": "cv2",
    "scikit-image": "skimage",
    "scipy": "scipy",
    "PyWavelets": "pywt",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
}

NOISE_PARAMETER_MEANING = {
    "Gaussian": "standard deviation on the 0-255 scale",
    "Salt-and-Pepper": "fraction of pixels affected",
    "Speckle": "variance of the multiplicative noise",
    "Poisson": "peak photon count (lower means stronger noise)",
}

DEGRADATION_PARAMETER_MEANING = {
    "JPEG Compression": "JPEG quality, 0-100 (lower means stronger damage)",
    "Glare": "glare strength alpha, 0-1 (higher means stronger)",
}

DEGRADATION_SETTING_NAME = {"JPEG Compression": "jpeg_quality", "Glare": "glare_strength"}

DEGRADATION_METHODS = {
    "JPEG Compression": (
        "Round to 0-255, encode with cv2.imencode('.jpg', IMWRITE_JPEG_QUALITY=quality), "
        "decode as grayscale, divide by 255. Not random."
    ),
    "Glare": (
        "Blend an elliptical Gaussian light spot: (1 - alpha * mask) * image + alpha * mask, "
        "clipped to 0-1. The seed places the spot (centre at 25-75% of the width and height) "
        "and sets its spread (12-22% of the width and height)."
    ),
}


def _current_time():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "unknown"


def _project_relative(path):
    """Returns a location without exposing personal folder names."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return f"outside project: {resolved.name}"


def _run_git(*arguments):
    # --no-optional-locks keeps this check from updating Git's index.
    command = ["git", "--no-optional-locks", "-C", str(PROJECT_DIR), *arguments]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=True
    ).stdout


def comparison_file_name(noise_type, noise_level, image_index, image_name,
                         extension=".png"):
    """Where a full-dataset run saves one image's comparison picture.

    The dataset position makes the name unique, even if two images share
    a name stem, e.g. "salt_and_pepper_medium/004_test005.png".
    """
    condition = f"{noise_type.lower().replace('-', '_').replace(' ', '_')}_{noise_level.lower()}"
    return f"{condition}/{image_index:03d}_{Path(image_name).stem}{extension}"


def get_git_info():
    """Describes the code state without changing anything in Git."""
    try:
        commit = _run_git("rev-parse", "HEAD").strip()
        prefix = _run_git("rev-parse", "--show-prefix").strip()
        status = _run_git(
            "status", "--porcelain=v1", "-z",
            "--untracked-files=all", "--no-renames"
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": "unknown",
            "working_tree": "unknown",
            "code_matches_commit": None,
            "note": "Git information was not available.",
        }

    modified, untracked = [], []
    generated_files = 0

    for entry in status.split("\0"):
        if not entry:
            continue

        code, path = entry[:2], entry[3:]

        # Run outputs and Python caches are not experiment code.
        if "__pycache__/" in path or path.startswith(f"{prefix}results/"):
            generated_files += 1
        elif code == "??":
            untracked.append(path)
        else:
            modified.append(path)

    if modified and untracked:
        working_tree = "modified_and_untracked"
    elif modified:
        working_tree = "modified"
    elif untracked:
        working_tree = "untracked_only"
    else:
        working_tree = "clean"

    is_clean = working_tree == "clean"

    return {
        "commit": commit,
        "working_tree": working_tree,
        "code_matches_commit": is_clean,
        "modified_file_count": len(modified),
        "modified_files": modified[:MAX_LISTED_FILES],
        "untracked_file_count": len(untracked),
        "untracked_files": untracked[:MAX_LISTED_FILES],
        "generated_files_ignored": generated_files,
        "note": (
            "The commit identifies the code used."
            if is_clean else
            "The commit alone does not identify the code used. "
            "Compare code_fingerprints as well."
        ),
    }


def get_library_versions():
    versions = {}

    for label, module_name in LIBRARY_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            versions[label] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[label] = "unavailable"

    return versions


def _describe_filter(name, function, supplied):
    parameters = list(inspect.signature(function).parameters.values())[1:]
    defaults = {
        parameter.name: parameter.default
        for parameter in parameters
        if parameter.default is not inspect.Parameter.empty
    }

    entry = {"supplied": supplied, "resolved": {**defaults, **supplied}}

    if name == "NLM Filter":
        if supplied.get("h") is None:
            entry["h_mode"] = "automatic"
            entry["h_rule"] = "max(0.1, estimated noise sigma), per image"
        else:
            entry["h_mode"] = "explicit"

    if name == "Lee Filter":
        entry["model"] = (
            "multiplicative speckle"
            if supplied.get("speckle_variance") is not None
            else "general adaptive smoother"
        )

    return entry


def build_run_config(images, dataset_path, results_path, max_images,
                     noise_levels, base_seed, filters, tuned_parameters,
                     tuned_file, get_seed, get_parameters, image_indices=None,
                     job=None, parameter_label=None, parameter_sources=None,
                     parameter_sources_by_condition=None, degradation_levels=None,
                     glare_seed_offset=None):
    """Builds the record using the runner's own seed and parameter rules.

    Degradation jobs (JPEG and glare) list degradation conditions instead
    of noise conditions. For them "noise_type" holds the degradation type,
    so the Results view can read both kinds the same way.
    """
    conditions = []
    experiment = (job or {}).get("experiment", "noise")
    is_degradation = experiment == "degradation"
    condition_levels = degradation_levels if is_degradation else noise_levels

    if image_indices is None:
        image_indices = range(len(images))

    for noise_index, (noise_type, levels) in enumerate(condition_levels.items()):
        for level_index, (level_name, noise_value) in enumerate(levels.items()):
            # A job over all noise conditions keeps every condition.
            if (
                job is not None
                and job.get("noise_scope") != "all"
                and (noise_type, level_name) != (job["noise_type"], job["noise_level"])
            ):
                continue

            if is_degradation:
                record = {
                    "category": "degradation",
                    "noise_type": noise_type,
                    "level": level_name,
                    "noise_value": noise_value,
                    "degradation_type": noise_type,
                    DEGRADATION_SETTING_NAME[noise_type]: noise_value,
                    "glare_location_seed_offset": (
                        glare_seed_offset if noise_type == "Glare" else None
                    ),
                }
            else:
                record = {
                    "category": "noise",
                    "noise_type": noise_type,
                    "level": level_name,
                    "noise_value": noise_value,
                    "noise_index": noise_index,
                    "level_index": level_index,
                    "seed_offset": get_seed(0, noise_index, level_index) - base_seed,
                }

            conditions.append({
                **record,
                "filters": {
                    name: _describe_filter(
                        name,
                        function,
                        get_parameters(
                            name, function, noise_type, level_name,
                            noise_value, tuned_parameters
                        )
                    )
                    for name, function in filters.items()
                },
            })

    is_tuned = tuned_parameters is not None

    config = {
        "schema_version": 1,
        "run": {
            "result_directory": Path(results_path).name,
            "status": "incomplete",
            "started_at": _current_time(),
            "finished_at": None,
            "mode": "tuned_parameters" if is_tuned else "default_parameters",
        },
        "required_to_rerun": {
            "dataset": {
                "name": Path(dataset_path).name,
                "location": _project_relative(dataset_path),
                "max_images_setting": max_images,
                "images_processed": len(images),
                "preprocessing": (
                    "cv2.imread grayscale, converted to float32, divided by 255"
                ),
                "pixel_hash": "SHA-256 of each loaded float32 image",
                "images": [
                    {
                        "index": index,
                        "filename": name,
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    }
                    for index, (name, image) in zip(image_indices, images)
                ],
            },
            "noise": {
                "levels": noise_levels,
                "parameter_meaning": {
                    noise_type: NOISE_PARAMETER_MEANING.get(noise_type, "unknown")
                    for noise_type in noise_levels
                },
                "seed_policy": {
                    "base_seed": base_seed,
                    "formula": (
                        "base_seed + image_index * 1000 "
                        "+ noise_index * 100 + level_index"
                    ),
                    "from_this_file": "base_seed + image_index * 1000 + seed_offset",
                },
                "used_in_this_run": not is_degradation,
            },
            "experiment": experiment,
            "filters_executed": list(filters),
            "parameter_source": {
                "mode": "tuned" if is_tuned else "function_defaults",
                "file": _project_relative(tuned_file) if is_tuned and tuned_file else None,
                "file_sha256": _file_sha256(tuned_file) if is_tuned and tuned_file else None,
                "values": "The actual values are recorded in conditions below.",
            },
            "conditions": conditions,
            "code": {
                "git": get_git_info(),
                "code_fingerprints": {
                    name: _file_sha256(PROJECT_DIR / name) for name in CODE_FILES
                },
            },
        },
        "useful_for_interpretation": {
            "environment": {
                "python": platform.python_version(),
                "libraries": get_library_versions(),
                "operating_system": platform.system(),
                "os_release": platform.release(),
                "machine": platform.machine(),
            },
            "evaluation": {
                "metrics": ["PSNR", "SSIM", "Edge_Precision", "Edge_Recall", "Edge_F1"],
                "image_data_range": [0.0, 1.0],
                "psnr": "scikit-image peak_signal_noise_ratio, data_range=1.0",
                "ssim": "scikit-image structural_similarity defaults, data_range=1.0",
                "edge_preservation": EDGE_METRIC_SETTINGS,
                "edge_scores_scope": (
                    "Calculated after the timer stops, for the unfiltered baseline and every filter"
                ),
                "execution_time_unit": "seconds",
                "execution_time_scope": (
                    "Only the filter function call, including work inside it"
                ),
                "aggregation": (
                    "Mean and sample standard deviation across images, per "
                    "noise type, noise level, and filter"
                ),
                "noisy_baseline_time": "0 is a placeholder, not a measurement",
            },
            "limitations": [
                "Results may differ across computers and library versions.",
                "Timing is one measurement per image and depends on system load.",
                "OpenCV may use multiple threads, which can affect timing.",
                "NLM estimates noise inside the timed call only when h is omitted.",
                *EDGE_METRIC_LIMITATIONS,
            ],
        },
    }

    if job is not None:
        config["run"]["job"] = job

    if is_degradation:
        config["required_to_rerun"]["degradation"] = {
            "levels": degradation_levels,
            "parameter_meaning": {
                name: DEGRADATION_PARAMETER_MEANING.get(name, "unknown")
                for name in degradation_levels
            },
            "methods": DEGRADATION_METHODS,
            "seed_policy": {
                "jpeg_compression": "Not random, so no seed is used.",
                "glare": {
                    "base_seed": base_seed,
                    "offset": glare_seed_offset,
                    "formula": f"base_seed + image_index * 1000 + {glare_seed_offset}",
                    "same_for_all_levels": True,
                    "reason": (
                        "The seed places the glare spot. One seed per image keeps the spot "
                        "in the same place for Low, Medium and High, so only the strength changes."
                    ),
                },
            },
            "baseline": "Degraded (Baseline): the JPEG or glare image before any filter",
            "source": (
                "Adapted from the teammate branch "
                "feature/image_degradation-different_datasets-edge_preservation. "
                "Its glare seed (base_seed + image_index * 1000 + level_index) moved the "
                "spot between levels; it was replaced by the per-image seed above."
            ),
        }
        config["useful_for_interpretation"]["limitations"].append(
            "The filters were designed for noise. JPEG blocks and glare are different "
            "damage, so a filter can lower PSNR compared with the unfiltered degraded image."
        )

    if parameter_sources_by_condition is not None:
        config["run"]["mode"] = "per_filter_parameters"
        config["required_to_rerun"]["parameter_source"] = {
            "mode": "per_filter",
            "values": "Each filter's preset and final values, for every noise condition.",
            "filters_by_condition": parameter_sources_by_condition,
        }
    elif parameter_sources is not None:
        config["run"]["mode"] = "per_filter_parameters"
        config["required_to_rerun"]["parameter_source"] = {
            "mode": "per_filter",
            "values": "Each filter's preset and final values are listed below.",
            "filters": parameter_sources,
        }
    elif job is not None:
        config["required_to_rerun"]["parameter_source"].update(
            source_id=job["parameter_source"],
            label=parameter_label,
        )

    return config


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot save {type(value).__name__} in {CONFIG_NAME}.")


def save_run_config(result_directory, config):
    """Writes the file completely, or stops the run with a clear error."""
    path = Path(result_directory) / CONFIG_NAME
    temporary = path.with_name(f"{CONFIG_NAME}.tmp")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, allow_nan=False, default=_json_value)
            file.write("\n")

        os.replace(temporary, path)

    except (OSError, TypeError, ValueError) as error:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)

        raise RuntimeError(
            f"Could not save {path.name}. This run is NOT complete."
        ) from error


def mark_run_completed(result_directory, config):
    config["run"]["status"] = "completed"
    config["run"]["finished_at"] = _current_time()
    save_run_config(result_directory, config)
