"""
Describes a frontend experiment job: its settings, presets and status files.

main.py uses this file to read and check a job (python main.py --job job.json).
The frontend uses it to fill the filter controls. Starting and watching the
background process lives in frontend/job_runner.py.

This file contains no image-processing code.
"""

import hashlib
import inspect
import json
import os
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ACTIVE_FILE_NAME = "active_job.txt"
ACTIVE_STATES = ("starting", "running")

# Fingerprint of the first best_parameters.json, tuned on only 2 Set12 images
# before the Lee and NLM fixes.
PREVIOUS_PARAMETERS_SHA256 = (
    "527fcf91d52f51286a031dca1faa354f5cbbe504cc2158963740581b1bd8dd5b"
)

PARAMETER_SOURCES = {
    "default": {
        "label": "Default parameters",
        "file": None,
        "explanation": "Uses the settings built into each filter, with no tuning.",
    },
    "previous_2_image": {
        "label": "Previous 2-image tuned parameters",
        "file": "best_parameters.json",
        "explanation": (
            "Uses the older settings chosen from only 2 Set12 images. "
            "Useful for comparison, but not recommended for final results."
        ),
    },
    "new_12_image": {
        "label": "New 12-image tuned parameters",
        "file": "tuning_runs/set12_12images_corrected/best_parameters.json",
        "explanation": (
            "Uses the newer settings chosen from all 12 Set12 images "
            "with the corrected Lee and NLM filters."
        ),
    },
}


# Values offered for each filter setting. Every value in the saved presets
# is included. NLM h=None means "estimate h from the image".
PARAMETER_CHOICES = {
    "Gaussian Filter": {"kernel_size": (3, 5, 7, 9), "sigma": (0.5, 1.0, 1.5, 2.0)},
    "Median Filter": {"kernel_size": (3, 5, 7)},
    "Bilateral Filter": {
        "diameter": (5, 7, 9),
        "sigma_color": (0.05, 0.1, 0.2, 0.3),
        "sigma_space": (25, 50, 75),
    },
    "NLM Filter": {
        "patch_size": (3, 5, 7),
        "patch_distance": (3, 6, 9),
        "h": (None, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25),
    },
    "Lee Filter": {"window_size": (3, 5, 7, 9)},
    "TV Filter": {"weight": (0.03, 0.05, 0.07, 0.1, 0.15, 0.2)},
}


def read_json(path):
    """Returns saved JSON, or None if it is missing or being replaced."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path, data):
    """Replaces a JSON file in one step, so readers never see half a file."""
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Windows briefly blocks replacing a file while another process reads it.
    for _ in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.05)

    os.replace(temporary, path)


def read_job(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parameter_file(source_id):
    """Returns the checked parameter file, or None for default settings."""
    if source_id not in PARAMETER_SOURCES:
        raise ValueError(f"Unknown parameter source: {source_id}")

    relative_path = PARAMETER_SOURCES[source_id]["file"]
    if relative_path is None:
        return None

    path = PROJECT_DIR / relative_path
    if not path.is_file():
        raise ValueError(f"Parameter file was not found: {relative_path}")

    if source_id == "previous_2_image":
        if hashlib.sha256(path.read_bytes()).hexdigest() != PREVIOUS_PARAMETERS_SHA256:
            raise ValueError(
                "best_parameters.json is no longer the previous 2-image file."
            )

    if source_id == "new_12_image":
        tuning = read_json(path.with_name("tuning_config.json")) or {}
        if (
            tuning.get("run", {}).get("status") != "completed"
            or tuning.get("dataset", {}).get("images_evaluated") != 12
        ):
            raise ValueError("The 12-image tuning run is missing or incomplete.")

    return path


def validate_filter_parameters(filter_name, values):
    """Accepts only the offered values. Returns them in a fixed order."""
    if filter_name not in PARAMETER_CHOICES:
        raise ValueError(f"Unknown filter: {filter_name}")

    choices = PARAMETER_CHOICES[filter_name]
    if not isinstance(values, dict) or set(values) != set(choices):
        raise ValueError(f"{filter_name} needs exactly these settings: {', '.join(choices)}.")

    for name, value in values.items():
        if isinstance(value, bool) or value not in choices[name]:
            raise ValueError(f"{filter_name}: {name}={value} is not an allowed value.")

    return {name: values[name] for name in choices}


def preset_parameters(source_id, filter_name, filter_function, noise_type, noise_level):
    """Returns a full set of values for one filter from a preset.

    Tuned files store only the tuned settings, so the remaining settings
    come from the filter's own defaults, exactly as the runner uses them.
    """
    path = parameter_file(source_id)
    defaults = inspect.signature(filter_function).parameters
    values = {name: defaults[name].default for name in PARAMETER_CHOICES[filter_name]}

    if path is not None:
        saved = json.loads(path.read_text(encoding="utf-8"))

        # Tuned files only hold noise conditions (tuned on Set12).
        # Never fall back to other values for JPEG, glare, or anything else.
        if noise_type not in saved:
            raise ValueError(
                f"{PARAMETER_SOURCES[source_id]['label']} were tuned for noise on Set12 "
                f"and have no values for {noise_type}. Use Default parameters."
            )

        values.update(saved[noise_type][noise_level][filter_name])

    return values


def describe_preset_use(source_id, filter_name, filter_function, noise_type,
                        noise_level, values):
    """Records where a filter's values started and whether they were edited."""
    path = parameter_file(source_id)
    preset_values = preset_parameters(
        source_id, filter_name, filter_function, noise_type, noise_level
    )
    return {
        "preset": source_id,
        "preset_label": PARAMETER_SOURCES[source_id]["label"],
        "preset_file": PARAMETER_SOURCES[source_id]["file"],
        "preset_file_sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest() if path else None
        ),
        "preset_parameters": preset_values,
        "parameters": values,
        "edited": values != preset_values,
    }


def validate_job_filters(job):
    """Checks per-filter settings. Older jobs with a filter list are unchanged."""
    chosen = job.get("filters")
    if isinstance(chosen, list):
        return

    if not isinstance(chosen, dict) or not chosen:
        raise ValueError("Choose at least one filter.")

    all_conditions = job.get("noise_scope") == "all"
    is_degradation = job.get("experiment") == "degradation"

    for filter_name, entry in chosen.items():
        if not isinstance(entry, dict) or entry.get("preset") not in PARAMETER_SOURCES:
            raise ValueError(f"{filter_name} needs a known preset and its values.")

        # Tuned presets were chosen for noise only, so they are not offered
        # for JPEG compression or glare.
        if is_degradation and entry["preset"] != "default":
            raise ValueError(
                f"{filter_name}: {PARAMETER_SOURCES[entry['preset']]['label']} were tuned "
                "for noise on Set12 and are not available for JPEG compression or glare. "
                "Use Default parameters (you can still edit the values)."
            )

        # In a run over all noise conditions, "parameters": None means
        # "use the preset's own saved values for each condition".
        if all_conditions and entry.get("parameters") is None:
            if filter_name not in PARAMETER_CHOICES:
                raise ValueError(f"Unknown filter: {filter_name}")
            continue

        validate_filter_parameters(filter_name, entry.get("parameters"))


def mark_job(job_dir, state, **details):
    """Saves the job state and releases the one-job lock when it ends."""
    job_dir = Path(job_dir)
    status = {
        "state": state,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **details,
    }
    write_json(job_dir / "status.json", status)

    if state not in ACTIVE_STATES:
        active_file = job_dir.parent / ACTIVE_FILE_NAME
        try:
            if active_file.read_text(encoding="utf-8").strip() == job_dir.name:
                active_file.unlink()
        except OSError:
            pass
