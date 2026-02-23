"""
Plot grouped bars for E0 evaluation (static vs burst per mode/policy).

Inputs:
  results_E0/E0_runs.csv
Outputs:
  results_E0/E0_delivery.png
  results_E0/E0_goodput.png
  results_E0/E0_p95.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out: Path):
    modes = ["light", "medium", "heavy"]
    policies = ["sp", "ecmp", "PPO_CANDIDATE", "HIER_STAGE3"]
    variants = ["static", "burst"]
    colors = {"static": "#7aa6c2", "burst": "#c27a7a"}

    fig, ax = plt.subplots(figsize=(20, 4))
    width = 0.18
    group_gap = 0.6
    policy_gap = 0.1
    x_positions = []
    x_labels = []
    group_centers = []

    x = 0.0
    first_label = {v: True for v in variants}
    for mode in modes:
        group_start = x
        for pol in policies:
            sub = df[(df["mode"] == mode) & (df["policy"] == pol)]
            for i, var in enumerate(variants):
                vals = sub[sub["variant"] == var][metric]
                mean_val = vals.mean()
                err = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                ax.bar(
                    x + i * width,
                    mean_val,
                    width=width,
                    color=colors[var],
                    yerr=err,
                    capsize=3,
                    label=var if first_label[var] else None,
                )
                first_label[var] = False
            x += width * len(variants) + policy_gap
            x_positions.append(x - width - policy_gap / 2)
            x_labels.append(pol.upper())
        group_end = x - policy_gap
        group_centers.append((group_start + group_end - policy_gap) / 2)
        x += group_gap  # gap between mode groups

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0)
    for center, mode in zip(group_centers, modes):
        ax.text(center, ax.get_ylim()[1] * 1.01, mode, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot grouped bars for E0 metrics.")
    parser.add_argument("--runs", type=Path, default=Path("results_E0/E0_runs.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results_E0"))
    args = parser.parse_args()

    df = pd.read_csv(args.runs)
    plot_metric(df, "delivery_ratio", "Delivery ratio", args.outdir / "E0_delivery.png")
    plot_metric(df, "goodput_mbps", "Goodput (Mbps)", args.outdir / "E0_goodput.png")
    plot_metric(df, "p95_latency_ms", "p95 latency (ms)", args.outdir / "E0_p95.png")
    print(f"Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()
