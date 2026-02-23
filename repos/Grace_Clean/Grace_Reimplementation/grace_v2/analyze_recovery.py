"""
Compute simple recovery stats from E0_windows.jsonl.

Inputs:
  results_E0/E0_windows.jsonl
Outputs:
  results_E0/E0_recovery.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

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


def compute_recovery(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (mode, policy, seed), grp in df.groupby(["mode", "policy", "seed"]):
        burst = grp[grp["variant"] == "burst"]
        if burst.empty:
            continue

        pre = burst[burst["t_start_ms"] < BURST_START]
        during = burst[(burst["t_start_ms"] >= BURST_START) & (burst["t_start_ms"] < BURST_END)]
        post = burst[burst["t_start_ms"] >= BURST_END]

        pre_del_mean = pre["delivery_ratio_window"].mean() if not pre.empty else None
        pre_goodput_mean = pre["goodput_mbps_window"].mean() if not pre.empty else None

        min_del_burst = during["delivery_ratio_window"].min() if not during.empty else None
        max_p95_burst = during["p95_latency_ms_window"].max() if not during.empty else None

        # recovery time: first window >= pre_mean (if pre_mean exists)
        rec_del = None
        if pre_del_mean is not None and not post.empty:
            recovered = post[post["delivery_ratio_window"] >= pre_del_mean]
            if not recovered.empty:
                rec_del = recovered["t_start_ms"].min() - BURST_END

        rec_goodput = None
        if pre_goodput_mean is not None and not post.empty:
            recovered = post[post["goodput_mbps_window"] >= pre_goodput_mean]
            if not recovered.empty:
                rec_goodput = recovered["t_start_ms"].min() - BURST_END

        records.append(
            {
                "mode": mode,
                "policy": policy,
                "seed": seed,
                "min_delivery_during_burst": min_del_burst,
                "max_p95_during_burst": max_p95_burst,
                "recovery_time_delivery_ms": rec_del,
                "recovery_time_goodput_ms": rec_goodput,
            }
        )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Analyze recovery from E0_windows.jsonl.")
    parser.add_argument("--windows", type=Path, default=Path("results_E0/E0_windows.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results_E0/E0_recovery.csv"))
    args = parser.parse_args()

    df = load_windows(args.windows)
    rec = compute_recovery(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(args.out, index=False)
    print(f"[Recovery] saved -> {args.out}")


if __name__ == "__main__":
    main()
