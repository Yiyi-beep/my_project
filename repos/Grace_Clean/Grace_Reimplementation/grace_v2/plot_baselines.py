"""
Plot comparison charts for OSPF/SP, ECMP, and HIER using acceptance runs.

Generates:
- checkpoints_v2/plot_pareto.png
- checkpoints_v2/plot_bar_metrics.png
- checkpoints_v2/plot_latency_cdf.png
- checkpoints_v2/plot_cluster_pref.png (HIER only; optional if cluster_top1 present)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8")


def load_runs(path: Path):
    df = pd.read_csv(path)
    return df


def plot_bar_with_ci(df: pd.DataFrame, metric: str, ylabel: str, out: Path):
    # Expect columns: policy, mode, metric
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    modes = ["light", "medium", "heavy", "burst"]
    axes_flat = axes.ravel()
    for ax, mode in zip(axes_flat, modes):
        sub = df[df["mode"] == mode]
        means = sub.groupby("policy")[metric].mean()
        stds = sub.groupby("policy")[metric].std()
        n = sub.groupby("policy")[metric].count().clip(lower=1)
        ci = 1.96 * stds / np.sqrt(n)
        policies = means.index.tolist()
        ax.bar(policies, means.values, yerr=ci.values, capsize=4, alpha=0.7)
        ax.set_title(mode)
        ax.set_ylabel(ylabel)
    fig.suptitle(f"{metric} by policy (with 95% CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_latency_cdf(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    modes = ["heavy", "burst"]
    for mode in modes:
        sub = df[df["mode"] == mode]
        for policy in sub["policy"].unique():
            vals = sub[sub["policy"] == policy]["p95_latency_ms"].values
            if len(vals) == 0:
                continue
            sorted_vals = np.sort(vals)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf, label=f"{policy}-{mode}")
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF (heavy & burst)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pareto(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"ospf": "o", "ecmp": "s", "hier": "^"}
    for policy in df["policy"].unique():
        sub = df[df["policy"] == policy]
        ax.scatter(sub["p95_latency_ms"], sub["goodput_mbps"], label=policy, marker=markers.get(policy, "x"), alpha=0.8)
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("goodput (Mbps)")
    ax.set_title("Pareto: p95 latency vs goodput")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cluster_pref(runs_path: Path, out: Path):
    # This requires cluster_hist info; if not available in baselines runs, skip.
    # We can try to read acceptance_runs.csv (HIER only) for cluster_hist JSON.
    try:
        df = pd.read_csv("checkpoints_v2/acceptance_runs.csv")
    except FileNotFoundError:
        return
    df = df[df["policy"] == "trained"] if "policy" in df.columns else df
    if "cluster_hist" not in df.columns:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    modes = ["light", "medium", "heavy", "burst"]
    for ax, mode in zip(axes.ravel(), modes):
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        # aggregate top1 frac per pair
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
        ax.set_ylabel("Top cluster frac")
    fig.suptitle("HIER cluster preference (top1 frac)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    runs_path = Path("checkpoints_v2/acceptance_runs_baselines.csv")
    df = load_runs(runs_path)

    # Bar charts per metric
    out_dir = Path("checkpoints_v2")
    plot_bar_with_ci(df, "p95_latency_ms", "p95 latency (ms)", out_dir / "plot_bar_p95.png")
    plot_bar_with_ci(df, "goodput_mbps", "Goodput (Mbps)", out_dir / "plot_bar_goodput.png")
    plot_bar_with_ci(df, "delivery_ratio", "Delivery ratio", out_dir / "plot_bar_delivery.png")
    plot_bar_with_ci(df, "reward_total", "Reward", out_dir / "plot_bar_reward.png")

    # Latency CDF for heavy/burst
    plot_latency_cdf(df, out_dir / "plot_latency_cdf.png")

    # Pareto scatter
    plot_pareto(df, out_dir / "plot_pareto.png")

    # Cluster preference (HIER only)
    plot_cluster_pref(runs_path, out_dir / "plot_cluster_pref.png")

    print("Plots saved to checkpoints_v2/*.png")


if __name__ == "__main__":
    main()

