
import hashlib
import json
import inspect
import os
import platform
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from main import (
    NOISE_LEVELS,
    BASE_SEED,
    filters,
    generate_noisy_image,
    calculate_metrics,
)
from run_metadata import (
    CODE_FILES,
    PROJECT_DIR,
    get_git_info,
    get_library_versions,
)

DATASET_PATH = Path("dataset/Set12")
OUTPUT_PATH = Path("best_parameters.json")
TUNING_RUNS_PATH = Path("tuning_runs")
MAX_TUNING_IMAGES = 12 # Quick test; None = all images

PARAMETER_GRID = {
    "Gaussian Filter": [
        {"kernel_size": 3, "sigma": 0.5},
        {"kernel_size": 5, "sigma": 1.0},
        {"kernel_size": 7, "sigma": 1.5},
    ],
    "Median Filter": [
        {"kernel_size": 3},
        {"kernel_size": 5},
        {"kernel_size": 7},
    ],
    "Bilateral Filter": [
        {"diameter": 5, "sigma_color": 0.05},
        {"diameter": 9, "sigma_color": 0.1},
        {"diameter": 9, "sigma_color": 0.2},
    ],
    "NLM Filter": [
        {"h": 0.1},
        {"h": 0.15},
        {"h": 0.2},
    ],
    "Lee Filter": [
        {"window_size": 3},
        {"window_size": 5},
        {"window_size": 7},
    ],
    "TV Filter": [
        {"weight": 0.03},
        {"weight": 0.07},
        {"weight": 0.1},
    ],
}


def load_images(return_names=False):
    paths = sorted(
        p for p in DATASET_PATH.rglob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )

    if MAX_TUNING_IMAGES is not None:
        paths = paths[:MAX_TUNING_IMAGES]

    images = []
    names = []

    for path in paths:
        image = cv2.imread(
            str(path), cv2.IMREAD_GRAYSCALE
        )
        if image is not None:
            images.append(
                image.astype(np.float32) / 255.0
            )
            names.append(path.name)

    if not images:
        raise ValueError("No Set12 images found.")

    print(f"Loaded {len(images)} tuning images")

    if return_names:
        return images, names

    return images


def create_tuning_run_directory(image_count):
    """Creates a new run folder and never reuses an existing one."""
    TUNING_RUNS_PATH.mkdir(parents=True, exist_ok=True)
    base_name = f"set12_{image_count}images_corrected"
    run_number = 1

    while True:
        name = base_name if run_number == 1 else f"{base_name}_run{run_number}"
        try:
            path = TUNING_RUNS_PATH / name
            path.mkdir()
            return path
        except FileExistsError:
            run_number += 1


def current_time():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_tuning_config(output_dir, images, image_names):
    condition_count = sum(len(levels) for levels in NOISE_LEVELS.values())
    candidates_per_condition = sum(
        len(PARAMETER_GRID[name]) for name in filters
    )
    candidate_evaluations = condition_count * candidates_per_condition

    return {
        "schema_version": 1,
        "run": {
            "directory": Path(output_dir).name,
            "status": "incomplete",
            "every_condition_completed": False,
            "started_at": current_time(),
            "finished_at": None,
            "activated_for_experiments": False,
            "note": (
                "These parameters are saved separately. The experiment "
                "runner still reads the project-level best_parameters.json."
            ),
        },
        "dataset": {
            "name": Path(DATASET_PATH).name,
            "max_tuning_images_setting": MAX_TUNING_IMAGES,
            "images_evaluated": len(images),
            "preprocessing": (
                "cv2.imread grayscale, converted to float32, divided by 255"
            ),
            "images": [
                {
                    "index": index,
                    "filename": name,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
                for index, (name, image) in enumerate(zip(image_names, images))
            ],
        },
        "noise": {
            "levels": NOISE_LEVELS,
            "seed_policy": {
                "base_seed": BASE_SEED,
                "formula": (
                    "base_seed + image_index * 1000 "
                    "+ noise_index * 100 + level_index"
                ),
                "noise_order": list(NOISE_LEVELS),
            },
        },
        "filters": list(filters),
        "candidate_grid": {name: PARAMETER_GRID[name] for name in filters},
        "selection": {
            "metric": "highest mean PSNR across all tuning images",
            "scope": "chosen separately per noise type, level, and filter",
            "ties": "the earlier candidate in the grid is kept",
        },
        "expected_counts": {
            "noise_conditions": condition_count,
            "candidates_per_condition": candidates_per_condition,
            "candidate_evaluations": candidate_evaluations,
            "image_evaluations": candidate_evaluations * len(images),
        },
        "environment": {
            "python": platform.python_version(),
            "libraries": get_library_versions(),
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
        },
        "code": {
            "git": get_git_info(),
            "code_fingerprints": {
                name: hashlib.sha256(
                    (PROJECT_DIR / name).read_bytes()
                ).hexdigest()
                for name in (*CODE_FILES, "tune_filters.py")
            },
        },
    }


def save_tuning_config(output_dir, config):
    path = Path(output_dir) / "tuning_config.json"
    temporary = path.with_name("tuning_config.json.tmp")

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, allow_nan=False)
        file.write("\n")

    os.replace(temporary, path)


def finish_tuning_run(output_dir, config, best_parameters, records,
                      per_image_records):
    output_dir = Path(output_dir)
    expected = config["expected_counts"]

    pd.DataFrame(records).to_csv(
        output_dir / "tuning_results.csv", index=False
    )
    pd.DataFrame(per_image_records).to_csv(
        output_dir / "tuning_per_image.csv", index=False
    )

    with (output_dir / "best_parameters.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(best_parameters, file, indent=4)

    complete = (
        len(records) == expected["candidate_evaluations"]
        and len(per_image_records) == expected["image_evaluations"]
        and all(np.isfinite(r["Mean_PSNR"]) for r in records)
        and all(np.isfinite(r["PSNR"]) for r in per_image_records)
        and all(
            selected is not None
            for levels in best_parameters.values()
            for level in levels.values()
            for selected in level.values()
        )
    )

    config["run"]["every_condition_completed"] = complete
    config["run"]["status"] = "completed" if complete else "incomplete"
    config["run"]["finished_at"] = current_time()
    save_tuning_config(output_dir, config)

    if not complete:
        raise RuntimeError(f"Tuning run is incomplete: {output_dir}")

    print(f"\nNew tuning results saved to: {output_dir}")


def tune(new_run=False):
    output_dir = None
    run_config = None
    image_names = None
    per_image_records = []

    if new_run:
        images, image_names = load_images(return_names=True)

        if (
            MAX_TUNING_IMAGES is not None
            and len(images) != MAX_TUNING_IMAGES
        ):
            raise ValueError(
                f"Expected {MAX_TUNING_IMAGES} readable tuning images, "
                f"found {len(images)}."
            )

        output_dir = create_tuning_run_directory(len(images))
        run_config = build_tuning_config(output_dir, images, image_names)
        save_tuning_config(output_dir, run_config)
    else:
        images = load_images()

    best_parameters = {}
    records = []

    for noise_index, (noise_type, levels) in enumerate(
        NOISE_LEVELS.items()
    ):
        best_parameters[noise_type] = {}

        for level_index, (level_name, noise_value) in enumerate(
            levels.items()
        ):
            best_parameters[noise_type][level_name] = {}

            # Generate the same noisy inputs for all candidates.
            noisy_images = []

            for i, original in enumerate(images):
                seed = (
                    BASE_SEED + i * 1000
                    + noise_index * 100 + level_index
                )
                noisy_images.append(
                    generate_noisy_image(
                        original, noise_type, noise_value, seed
                    )
                )

            for filter_name, filter_function in filters.items():
                highest_psnr = float("-inf")
                selected = None

                for parameters in PARAMETER_GRID[filter_name]:
                    scores = []
                    ssim_scores = []

                    for original, noisy in zip(images, noisy_images):
                        kwargs = parameters.copy()

                        # Supply the known variance for simulated
                        # speckle noise if this Lee implementation
                        # accepts it.
                        if (
                            filter_name == "Lee Filter"
                            and noise_type == "Speckle"
                            and "speckle_variance"
                            in inspect.signature(filter_function).parameters
                        ):
                            kwargs["speckle_variance"] = noise_value

                        restored = filter_function(
                            noisy.copy(), **kwargs
                        )

                        psnr, ssim = calculate_metrics(
                            original, restored
                        )
                        scores.append(psnr)
                        ssim_scores.append(ssim)

                    average_psnr = float(np.mean(scores))
                    average_ssim = float(np.mean(ssim_scores))

                    records.append({
                        "Noise_Type": noise_type,
                        "Noise_Level": level_name,
                        "Filter": filter_name,
                        "Parameters": json.dumps(parameters),
                        "Mean_PSNR": average_psnr,
                        "Mean_SSIM": average_ssim,
                    })

                    if new_run:
                        for image_name, psnr, ssim in zip(
                            image_names, scores, ssim_scores
                        ):
                            per_image_records.append({
                                "Noise_Type": noise_type,
                                "Noise_Level": level_name,
                                "Filter": filter_name,
                                "Parameters": json.dumps(parameters),
                                "Image": image_name,
                                "PSNR": psnr,
                                "SSIM": ssim,
                            })

                    if average_psnr > highest_psnr:
                        highest_psnr = average_psnr
                        selected = parameters.copy()

                best_parameters[noise_type][level_name][
                    filter_name
                ] = selected

                print(
                    f"{noise_type} / {level_name} / "
                    f"{filter_name}: {selected} "
                    f"PSNR={highest_psnr:.2f}"
                )

    if new_run:
        finish_tuning_run(
            output_dir,
            run_config,
            best_parameters,
            records,
            per_image_records
        )
    else:
        # Save all candidate results
        pd.DataFrame(records).to_csv(
            "tuning_results.csv", index=False
        )

        # Save the best parameter settings
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(best_parameters, f, indent=4)

    print("\nTuning completed successfully!")


if __name__ == "__main__":
    tune(new_run=True)
