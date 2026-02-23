"""
Plots for ablation results in acceptance_runs_ablation.csv.

Generates:
1) p95 latency vs policy (faceted by mode)
2) reward vs policy (faceted by mode)
3) delivery vs policy (faceted by mode)
4) cluster_top1_frac (Stage1/2 vs Stage3) if cluster hist is available in acceptance_runs.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8")


def load_runs(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def bar_facet(df: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path):
    modes = ["light", "medium", "heavy", "burst"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.ravel()
    for ax, mode in zip(axes_flat, modes):
        sub = df[df["mode"] == mode]
        if sub.empty:
            ax.set_visible(False)
            continue
        means = sub.groupby("policy")[metric].mean()
        stds = sub.groupby("policy")[metric].std()
        n = sub.groupby("policy")[metric].count().clip(lower=1)
        ci = 1.96 * stds / np.sqrt(n)
        policies = means.index.tolist()
        ax.bar(policies, means.values, yerr=ci.values, capsize=4, alpha=0.7)
        ax.set_title(mode)
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cluster_top1(out: Path):
    # Use acceptance_runs.csv (HIER) if available; look at policy stage1_2 and stage3
    runs_path = Path("checkpoints_v2/acceptance_runs.csv")
    if not runs_path.exists():
        return
    df = pd.read_csv(runs_path)
    if "cluster_hist" not in df.columns:
        return
    df = df[df["policy"] == "trained"] if "policy" in df.columns else df
    modes = ["light", "medium", "heavy", "burst"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, mode in zip(axes.ravel(), modes):
        sub = df[df["mode"] == mode]
        if sub.empty:
            ax.set_visible(False)
            continue
        pair_to_frac = {}
        for _, row in sub.iterrows():
            chist = json.loads(row["cluster_hist"])
            for pair, hist in chist.items():
                total = sum(hist.values()) or 1
                top_cid, top_cnt = max(hist.items(), key=lambda kv: kv[1])
                pair_to_frac.setdefault(pair, []).append(top_cnt / total)
        labels = list(pair_to_frac.keys())
        vals = [np.mean(v) for v in pair_to_frac.values()]
        ax.bar(labels, vals, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_title(mode)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Top1 cluster frac")
    fig.suptitle("HIER cluster preference (top1 frac)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    runs_path = Path("checkpoints_v2/acceptance_runs_ablation.csv")
    df = load_runs(runs_path)
    out_dir = Path("checkpoints_v2")

    bar_facet(df, "p95_latency_ms", "p95 latency (ms)", "p95 latency vs policy", out_dir / "ablation_bar_p95.png")
    bar_facet(df, "reward_total", "Reward", "Reward vs policy", out_dir / "ablation_bar_reward.png")
    bar_facet(df, "delivery_ratio", "Delivery ratio", "Delivery vs policy", out_dir / "ablation_bar_delivery.png")

    # Optional cluster preference plot (uses main acceptance_runs.csv)
    plot_cluster_top1(out_dir / "ablation_cluster_pref.png")

    print("Ablation plots saved to checkpoints_v2/*.png")


if __name__ == "__main__":
    main()

