import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POLICIES = ["SP", "ECMP", "HIER_STAGE3"]
COLOR = {
    "SP": "#1f77b4",
    "ECMP": "#ff7f0e",
    "HIER_STAGE3": "#2ca02c",
}

def mean_ci95(x: pd.Series):
    x = x.dropna().astype(float)
    n = len(x)
    if n == 0:
        return np.nan, 0.0
    mean = x.mean()
    if n == 1:
        return mean, 0.0
    sem = x.std(ddof=1) / np.sqrt(n)
    ci = 1.96 * sem
    return mean, ci

def plot_metric(df: pd.DataFrame, value_col: str, ylabel: str, outpath: str, transform=None):
    d = df.copy()
    if transform is not None:
        d[value_col] = transform(d)

    # 只保留我们关心的策略，避免出现奇怪 policy
    d = d[d["policy"].isin(POLICIES)].copy()

    # flows 排序
    flows_levels = sorted(d["flows"].dropna().unique().tolist())
    modes = ["light", "medium", "heavy"]
    modes = [m for m in modes if m in set(d["mode"].unique())]  # 只画存在的

    # 聚合：mode/flows/policy -> mean + ci95
    rows = []
    for (mode, flows, policy), g in d.groupby(["mode", "flows", "policy"]):
        mean, ci = mean_ci95(g[value_col])
        rows.append((mode, flows, policy, mean, ci))
    agg = pd.DataFrame(rows, columns=["mode", "flows", "policy", "mean", "ci95"])

    fig, axes = plt.subplots(1, len(modes), figsize=(14, 4), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    x = np.arange(len(flows_levels))
    width = 0.24

    for ax, mode in zip(axes, modes):
        a = agg[agg["mode"] == mode]

        for i, policy in enumerate(POLICIES):
            y = []
            e = []
            for f in flows_levels:
                r = a[(a["flows"] == f) & (a["policy"] == policy)]
                if len(r) == 0:
                    y.append(np.nan); e.append(0.0)
                else:
                    y.append(float(r["mean"].iloc[0]))
                    e.append(float(r["ci95"].iloc[0]))
            ax.bar(x + (i - 1) * width, y, width=width, label=policy,
                   color=COLOR[policy], yerr=e, capsize=3)

        ax.set_title(mode)
        ax.set_xticks(x)
        ax.set_xticklabels([str(f) for f in flows_levels])
        ax.set_xlabel("Concurrent flows (F)")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel(ylabel)

    # 把 legend 放到图外顶部，不挡图
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(POLICIES), bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Path to E1_static_runs.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for figures")
    args = ap.parse_args()

    df = pd.read_csv(args.runs)

    # 基础检查
    needed = {"flows","mode","policy","delivery_ratio","goodput_mbps","p95_latency_ms"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in runs CSV: {sorted(missing)}")

    out_delivery = os.path.join(args.outdir, "E1_static_delivery_clean.png")
    out_goodput = os.path.join(args.outdir, "E1_static_goodput_clean.png")
    out_gpf = os.path.join(args.outdir, "E1_static_goodput_per_flow_clean.png")
    out_p95 = os.path.join(args.outdir, "E1_static_p95_clean.png")

    plot_metric(df, "delivery_ratio", "Delivery ratio", out_delivery)
    plot_metric(df, "goodput_mbps", "Goodput (Mbps)", out_goodput)
    plot_metric(df, "goodput_mbps", "Goodput per flow (Mbps/flow)", out_gpf,
                transform=lambda d: d["goodput_mbps"] / d["flows"])
    plot_metric(df, "p95_latency_ms", "p95 latency (ms)", out_p95)

    print("Wrote:")
    print(" ", out_delivery)
    print(" ", out_goodput)
    print(" ", out_gpf)
    print(" ", out_p95)

if __name__ == "__main__":
    main()
