"""
Plot entropy-based diagnostics for E1 scalability (HIER only).

Inputs:
  results_E1/E1_windows.jsonl

Outputs:
  results_E1/E1_entropy_timeseries.png   # F=6,12 static vs burst
  results_E1/E1_entropy_box.png         # box/violin of mean entropy per run vs flows (static/burst)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEASURE_START = 500.0
BURST_START = 1700.0
BURST_END = 2300.0


def load_windows(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_timeseries(df: pd.DataFrame, out: Path):
    # Only HIER, F in {6,12}, variants static/burst
    targets = [6, 12]
    variants = ["static", "burst"]
    colors = {"static": "#7aa6c2", "burst": "#c27a7a"}
    fig, axes = plt.subplots(1, len(targets), figsize=(10, 4), sharey=True)
    for ax, flows in zip(axes, targets):
        sub_f = df[(df["flows"] == flows) & (df["policy"] == "HIER_STAGE3")]
        for var in variants:
            sub = sub_f[sub_f["variant"] == var]
            grouped = sub.groupby("window").agg(
                t_start_ms=("t_start_ms", "mean"),
                mean_ent=("cluster_entropy_window", "mean"),
                std_ent=("cluster_entropy_window", "std"),
                count=("cluster_entropy_window", "count"),
            )
            if grouped.empty:
                continue
            ax.plot(grouped["t_start_ms"], grouped["mean_ent"], color=colors[var], label=var if flows == targets[0] else None)
            if (grouped["std_ent"].notna()).any():
                ax.fill_between(
                    grouped["t_start_ms"],
                    grouped["mean_ent"] - grouped["std_ent"],
                    grouped["mean_ent"] + grouped["std_ent"],
                    color=colors[var],
                    alpha=0.2,
                )
        ax.axvline(BURST_START, color="k", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(BURST_END, color="k", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"F={flows}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Cluster entropy")
    axes[0].legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_box(df: pd.DataFrame, out: Path):
    # Mean entropy per run (seed) over measure windows
    hier = df[df["policy"] == "HIER_STAGE3"]
    records = []
    for (flows, variant, seed), grp in hier.groupby(["flows", "variant", "seed"]):
        mean_ent = grp["cluster_entropy_window"].mean()
        records.append({"flows": flows, "variant": variant, "mean_entropy": mean_ent})
    box_df = pd.DataFrame(records)
    if box_df.empty:
        return
    flows_levels = sorted(box_df["flows"].unique())
    variants = ["static", "burst"]
    fig, axes = plt.subplots(1, len(variants), figsize=(10, 4), sharey=True)
    for ax, var in zip(axes, variants):
        sub = box_df[box_df["variant"] == var]
        data = [sub[sub["flows"] == f]["mean_entropy"].dropna() for f in flows_levels]
        ax.boxplot(data, positions=range(len(flows_levels)))
        ax.set_xticks(range(len(flows_levels)))
        ax.set_xticklabels([str(f) for f in flows_levels])
        ax.set_title(f"{var}")
        ax.set_xlabel("Flows (F)")
        ax.set_ylabel("Mean cluster entropy")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot entropy diagnostics for E1.")
    parser.add_argument("--windows", type=Path, default=Path("results_E1/E1_windows.jsonl"))
    parser.add_argument("--outdir", type=Path, default=Path("results_E1"))
    args = parser.parse_args()

    df = load_windows(args.windows)
    # Ensure numeric
    for col in ["cluster_entropy_window", "t_start_ms", "flows"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    plot_timeseries(df, args.outdir / "E1_entropy_timeseries.png")
    plot_box(df, args.outdir / "E1_entropy_box.png")
    print(f"Saved entropy plots to {args.outdir}")


if __name__ == "__main__":
    main()
