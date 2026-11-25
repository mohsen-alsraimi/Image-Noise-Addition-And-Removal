# Image Noise Addition and Removal — My Changes

## Overview

I worked on fixing problems in our Image Noise Addition and Removal project. My main changes were improving the Lee filter, fixing the NLM parameter handling, correcting the execution-time calculation, and making the experiments easier to reproduce.

I also updated the parameter tuning process and ran experiments to verify the changes.

## 1. Lee Filter — `filter.py`

The old Lee filter was not handling speckle noise correctly. It used a general noise estimate that did not properly consider how speckle noise depends on pixel brightness.

I updated the Lee filter to use local mean, local variance, and the speckle noise variance.

I also added input validation to reject invalid values.

For other noise types, the filter still supports general adaptive smoothing.

**Testing:** 16 focused tests passed. The corrected filter generally improved PSNR on the tested speckle-noise images.

## 2. NLM Filter — `filter.py`

Previously, the NLM filter could ignore the selected `h` parameter because the code replaced it with the estimated noise level when that value was higher.

I fixed this so that an explicitly provided `h` value is used directly.

When no `h` is provided, the filter still uses its automatic noise estimation behavior.

**Testing:** Six focused mock tests and one real-library test passed.

## 3. Execution-Time Measurement — `main.py`

The experiment timer previously included extra work, such as parameter lookup and image copying.

I moved the timer so that it measures only the denoising filter function call.

The filter outputs, PSNR, and SSIM calculations were not changed.

**Testing:** Three focused tests passed, and 72 checked filter outputs remained identical.

## 4. Experiment Metadata — `main.py`, `run_metadata.py`

Previously, experiment results did not save enough information about how they were generated.

I added automatic `run_config.json` recording for new experiments.

It saves information such as the dataset, noise settings, filter parameters, random seeds, library versions, and Git state.

This makes it easier to understand and reproduce experiment results.

**Testing:** Fourteen relevant tests passed.



**Results:**
- 12 Set12 images processed.
- 216 parameter candidates evaluated across the noise conditions.
- 2,592 successful image evaluations.
- No failed evaluations.
- Seven of the 72 selected parameter choices differed from the previous tuning.

The new results are saved under `tuning_runs/`.

The original `best_parameters.json` remains unchanged, and the new tuning parameters are not automatically active.

## 6. BSD68 Experiments

I ran a one-image experiment and a full BSD68 experiment after the fixes.

- `result12`: One-image demonstration.
- `result13`: Full BSD68 experiment using 68 images, four noise models, three intensity levels, and six denoising filters.

The full experiment produced 4,896 filter evaluations and 816 noisy-baseline records.

These runs still use the previous root tuning parameters, so they should not be considered the final evaluation of the newly tuned settings.

## 7. Focused Tests

I added focused tests for:

- Lee filter behavior and input validation.
- NLM parameter handling.
- Execution-time boundaries.
- Experiment metadata and runner integration.

I followed a focused-testing approach instead of rerunning the entire test suite after every change.

## 8. Remaining Work

- Review whether the parameter search ranges should be expanded.
- Decide when to activate the new 12-image tuned parameters.
- Improve handling of corrupted images.
- Improve input validation for other noise functions.
- Review graph labels and presentation.
- Run final experiments after the parameter settings are finalized.

## Notes

The original datasets and historical experiment results were preserved. The new tuning results were saved separately.

This file documents the changes made during my development work. It does not mean that the remaining issues are already fixed.