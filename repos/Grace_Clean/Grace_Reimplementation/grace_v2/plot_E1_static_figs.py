"""
Plot E1 static scalability (no burst) for SP/ECMP/HIER across flows and modes.

Input: results_E1_static/E1_static_runs.csv
Outputs: results_E1_static/E1_static_delivery.png, E1_static_goodput.png, E1_static_p95.png, E1_static_goodput_per_flow.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out: Path):
    modes = ["light", "medium", "heavy"]
    flows_levels = sorted(df["flows"].unique())
    policies = ["SP", "ECMP", "HIER_STAGE3"]
    fig, axes = plt.subplots(1, len(modes), figsize=(12, 4), sharey=True)
    width = 0.25
    policy_gap = 0.05
    group_gap = 0.5
    for ax, mode in zip(axes, modes):
        sub = df[df["mode"] == mode]
        x_positions = []
        x_labels = []
        x = 0.0
        for f in flows_levels:
            for pol in policies:
                s = sub[(sub["flows"] == f) & (sub["policy"] == pol)]
                mean_val = s[metric].mean()
                err = 1.96 * s[metric].std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
                ax.bar(x, mean_val, width=width, yerr=err, capsize=3, label=pol if (f == flows_levels[0] and mode == modes[0]) else None)
                x += width + policy_gap
                x_positions.append(x - width - policy_gap / 2)
                x_labels.append(f"{pol}")
            x += group_gap
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(mode)
        ax.set_ylabel(ylabel)
    axes[0].legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot E1 static scalability.")
    parser.add_argument("--runs", type=Path, default=Path("results_E1_static/E1_static_runs.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results_E1_static"))
    args = parser.parse_args()

    df = pd.read_csv(args.runs)
    df["goodput_per_flow"] = df["goodput_mbps"] / df["flows"]
    plot_metric(df, "delivery_ratio", "Delivery ratio", args.outdir / "E1_static_delivery.png")
    plot_metric(df, "goodput_mbps", "Goodput (Mbps)", args.outdir / "E1_static_goodput.png")
    plot_metric(df, "p95_latency_ms", "p95 latency (ms)", args.outdir / "E1_static_p95.png")
    plot_metric(df, "goodput_per_flow", "Goodput per flow (Mbps)", args.outdir / "E1_static_goodput_per_flow.png")
    print(f"Saved static plots to {args.outdir}")


if __name__ == "__main__":
    main()
