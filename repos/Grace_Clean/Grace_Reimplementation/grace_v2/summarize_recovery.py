"""
Summarize burst impact/recovery metrics and plot recovery time bars.

Inputs:
  results_E0/E0_recovery.csv (per-seed records from analyze_recovery.py)
Outputs:
  results_E0/E0_recovery_table.csv   # mean/std per mode/policy
  results_E0/E0_recovery_table.md    # markdown-friendly table
  results_E0/E0_recovery_time.png    # bar chart of recovery_time_delivery_ms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_mean_std(series: pd.Series, precision: int = 3) -> str:
    vals = series.dropna()
    if vals.empty:
        return "N/A"
    return f"{vals.mean():.{precision}f} \u00b1 {vals.std(ddof=1):.{precision}f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize recovery metrics.")
    parser.add_argument("--recovery", type=Path, default=Path("results_E0/E0_recovery.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results_E0"))
    args = parser.parse_args()

    df = pd.read_csv(args.recovery)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    for (mode, policy), grp in df.groupby(["mode", "policy"]):
        records.append(
            {
                "mode": mode,
                "policy": policy,
                "min_delivery_during_burst": format_mean_std(grp["min_delivery_during_burst"], precision=4),
                "max_p95_during_burst": format_mean_std(grp["max_p95_during_burst"], precision=4),
                "recovery_time_delivery_ms": format_mean_std(grp["recovery_time_delivery_ms"], precision=3),
            }
        )
    table_df = pd.DataFrame(records)
    table_csv = outdir / "E0_recovery_table.csv"
    table_md = outdir / "E0_recovery_table.md"
    table_df.to_csv(table_csv, index=False)
    # Markdown-friendly table
    table_df.to_markdown(table_md, index=False)

    # Plot recovery time bars
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    modes = ["light", "medium", "heavy"]
    policies = ["SP", "ECMP", "HIER_STAGE3"]
    for ax, mode in zip(axes, modes):
        sub = df[df["mode"] == mode]
        means = []
        errs = []
        labels = []
        for pol in policies:
            vals = sub[sub["policy"].str.upper() == pol]["recovery_time_delivery_ms"].dropna()
            if vals.empty:
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(vals.mean())
                errs.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
            labels.append(pol)
        x = np.arange(len(policies))
        ax.bar(x, means, yerr=errs, capsize=3, color=["#777", "#5a8", "#c55"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(mode)
        ax.set_ylabel("Recovery time (ms)")
    fig.suptitle("Recovery time to pre-burst delivery level")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(outdir / "E0_recovery_time.png", dpi=200)
    plt.close(fig)

    print(f"[Recovery] table -> {table_csv}")
    print(f"[Recovery] markdown -> {table_md}")
    print(f"[Recovery] plot -> {outdir / 'E0_recovery_time.png'}")


if __name__ == "__main__":
    main()
