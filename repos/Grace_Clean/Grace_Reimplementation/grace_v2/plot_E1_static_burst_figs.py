"""
Plot E1 scalability results (F=3/6/9/12) for static vs burst heavy.

Input: results_E1/E1_runs.csv
Outputs: results_E1/E1_delivery.png, E1_goodput.png, E1_p95.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out: Path):
    flows_levels = sorted(df["flows"].unique())
    variants = ["static", "burst"]
    colors = {"static": "#7aa6c2", "burst": "#c27a7a"}
    policies = ["SP", "ECMP", "PPO_CANDIDATE", "HIER_STAGE3"]

    fig, axes = plt.subplots(1, len(policies), figsize=(16, 4), sharey=True)
    for ax, pol in zip(axes, policies):
        sub = df[df["policy"] == pol]
        x = np.arange(len(flows_levels))
        width = 0.35
        for i, var in enumerate(variants):
            vals = []
            errs = []
            for f in flows_levels:
                s = sub[(sub["flows"] == f) & (sub["variant"] == var)][metric]
                vals.append(s.mean())
                errs.append(1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)
            ax.bar(x + (i - 0.5) * width, vals, width=width, yerr=errs, capsize=3, color=colors[var], label=var if pol == "SP" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([str(f) for f in flows_levels])
        ax.set_title(pol)
        ax.set_xlabel("Flows (F)")
        ax.set_ylabel(ylabel)
    axes[0].legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot E1 scalability grouped bars.")
    parser.add_argument("--runs", type=Path, default=Path("results_E1/E1_runs.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results_E1"))
    args = parser.parse_args()

    df = pd.read_csv(args.runs)
    # Per-flow goodput and fairness
    df["goodput_per_flow"] = df["goodput_mbps"] / df["flows"]
    # If needed, Jain's fairness would require per-flow stats; here we approximate using aggregate if provided per-flow in windows.
    plot_metric(df, "delivery_ratio", "Delivery ratio", args.outdir / "E1_delivery.png")
    plot_metric(df, "goodput_mbps", "Goodput (Mbps)", args.outdir / "E1_goodput.png")
    plot_metric(df, "p95_latency_ms", "p95 latency (ms)", args.outdir / "E1_p95.png")
    plot_metric(df, "goodput_per_flow", "Goodput per flow (Mbps)", args.outdir / "E1_goodput_per_flow.png")
    print(f"Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()
