
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: py compare_tuning.py result8 result9"
    )

results_root = Path("results")
before_dir = results_root / sys.argv[1]
after_dir = results_root / sys.argv[2]

before = pd.read_csv(
    before_dir / "experiment_results.csv"
)
after = pd.read_csv(
    after_dir / "experiment_results.csv"
)

keys = ["Image", "Noise_Type", "Noise_Level", "Filter"]

# Confirm both runs contain exactly the same test cases.
if before.duplicated(keys).any() or after.duplicated(keys).any():
    raise ValueError("Duplicate experiment rows found.")

if set(map(tuple, before[keys].to_numpy())) != set(
    map(tuple, after[keys].to_numpy())
):
    raise ValueError("The two runs contain different test cases.")

merged = before.merge(
    after,
    on=keys,
    suffixes=("_Before", "_After"),
    validate="one_to_one"
)

# Verify that the noisy inputs are identical.
baseline = merged[
    merged["Filter"] == "Noisy (Baseline)"
]

if not np.allclose(
    baseline["PSNR_Before"],
    baseline["PSNR_After"],
    atol=1e-8
):
    raise ValueError("Baseline PSNR differs between runs.")

output = results_root / "tuning_comparison"
output.mkdir(exist_ok=True)

# Calculate the differences.
for metric in ["PSNR", "SSIM", "Execution_Time"]:
    merged[f"{metric}_Change"] = (
        merged[f"{metric}_After"]
        - merged[f"{metric}_Before"]
    )

merged.to_csv(output / "comparison.csv", index=False)

# Exclude the noisy baseline from filter comparisons.
filtered = merged[
    merged["Filter"] != "Noisy (Baseline)"
]

summary = filtered.groupby(
    ["Noise_Type", "Noise_Level", "Filter"],
    as_index=False
)[[
    "PSNR_Before", "PSNR_After", "PSNR_Change",
    "SSIM_Before", "SSIM_After", "SSIM_Change",
    "Execution_Time_Before", "Execution_Time_After",
    "Execution_Time_Change"
]].mean()

summary.to_csv(output / "summary.csv", index=False)

for noise_type, group in summary.groupby("Noise_Type"):
    levels = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, level in zip(axes, levels):
        subset = group[
            group["Noise_Level"] == level
        ].sort_values("Filter")

        x = np.arange(len(subset))
        width = 0.38

        ax.bar(
            x - width / 2,
            subset["PSNR_Before"],
            width,
            label="Before"
        )
        ax.bar(
            x + width / 2,
            subset["PSNR_After"],
            width,
            label="After"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            subset["Filter"].str.replace(" Filter", ""),
            rotation=45,
            ha="right"
        )
        ax.set_title(level)
        ax.set_ylabel("Average PSNR (dB)")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"{noise_type}: Before vs After Tuning")
    fig.tight_layout()
    fig.savefig(
        output / f"{noise_type.lower().replace(' ', '_')}_psnr.png",
        dpi=150
    )
    plt.close(fig)

# Overall PSNR improvements by filter.
overall = filtered.groupby(
    "Filter"
)["PSNR_Change"].mean().sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(overall.index, overall.values)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Average PSNR change (dB)")
ax.set_title("Parameter Tuning: Improvement by Filter")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(output / "psnr_improvement.png", dpi=150)
plt.close(fig)

print("Comparison complete!")
print("Results saved to:", output)
