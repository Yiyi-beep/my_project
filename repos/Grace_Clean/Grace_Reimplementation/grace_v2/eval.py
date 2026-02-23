"""
Acceptance / evaluation script for the v2 hierarchical router.

It compares a trained checkpoint against a simple fixed-cluster baseline
across all modes, runs multiple seeds, and writes per-run metrics plus
aggregated averages to CSV.
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import json

# Allow execution via `python grace_v2/eval.py`
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from grace_v2 import config
from grace_v2.bandit import PairModeLinUCB
from grace_v2.path_clustering import PathClusterManager, nsfnet_edges, build_nsfnet_graph
from grace_v2.ppo_agent import PPOAgent
from grace_v2.simulator import HierarchicalRoutingSimulator
from grace_v2.metrics import MetricsRecorder
from grace_v2.traffic import TrafficManager
from grace_v2.logging_utils import append_jsonl
from ns.port.port import Port
from ns.port.wire import Wire
import zlib
import simpy


def load_trained_agents(ckpt_dir: Path) -> tuple[PathClusterManager, PairModeLinUCB, PPOAgent]:
    clusters_path = ckpt_dir / "clusters.json"
    cm = PathClusterManager.load(clusters_path) if clusters_path.exists() else PathClusterManager()
    if not cm.clusters:
        cm.build_all(config.FLOW_PAIRS)

    ppo = PPOAgent()
    ppo_ckpt = ckpt_dir / "stage3_full_ppo.pth"
    if ppo_ckpt.exists():
        ppo.load(str(ppo_ckpt))

    bandit = PairModeLinUCB()
    bandit_ckpt = ckpt_dir / "stage3_full_bandit.pkl"
    if bandit_ckpt.exists():
        try:
            with open(bandit_ckpt, "rb") as f:
                bandit = pickle.load(f)
        except Exception:
            # Old checkpoint might be incompatible; fall back to fresh bandit.
            bandit = PairModeLinUCB()

    return cm, bandit, ppo


def run_suite(sim: HierarchicalRoutingSimulator, seeds: List[int], modes: List[str], fixed_cluster_id: int | None):
    rows = []
    for mode in modes:
        for seed in seeds:
            result = sim.run_episode(mode=mode, stage="bandit", seed=seed, fixed_cluster_id=fixed_cluster_id)
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "delivery_ratio": result.delivery_ratio,
                    "p95_latency_ms": result.p95_latency_ms,
                    "ppo_loss": result.ppo_loss,
                    "all_same_ratio": result.all_same_ratio,
                    "fallback_strict": result.fallback_counts.get("strict", 0),
                    "fallback_fallback1": result.fallback_counts.get("fallback1", 0),
                    "fallback_fallback2": result.fallback_counts.get("fallback2", 0),
                    "cluster_hist": json.dumps({f"{k[0]}-{k[1]}": v for k, v in result.cluster_hist.items()}),
                }
            )
    return rows


def aggregate(rows: List[dict]) -> dict:
    if not rows:
        return {}
    delivery = [r["delivery_ratio"] for r in rows]
    p95s = [r["p95_latency_ms"] for r in rows if r["p95_latency_ms"] is not None]
    goodputs = [r.get("goodput_mbps") for r in rows if r.get("goodput_mbps") is not None]
    rewards = [r.get("reward_total") for r in rows if r.get("reward_total") is not None]
    return {
        "delivery_ratio_avg": float(np.mean(delivery)),
        "p95_latency_ms_avg": float(np.mean(p95s)) if p95s else None,
        "goodput_mbps_avg": float(np.mean(goodputs)) if goodputs else None,
        "reward_avg": float(np.mean(rewards)) if rewards else None,
    }


def main():
    ckpt_dir = Path("checkpoints_v2")
    out_dir = ckpt_dir
    out_dir.mkdir(exist_ok=True)

    # Load trained agents
    cm, bandit, ppo = load_trained_agents(ckpt_dir)

    # Config: modes/seeds
    modes = list(config.MODES_PPS.keys())
    seeds = list(range(5))  # 5 runs per mode for acceptance

    # Trained policy
    sim_trained = HierarchicalRoutingSimulator(cluster_manager=cm, bandit=bandit, ppo_agent=ppo)
    trained_rows = run_suite(sim_trained, seeds, modes, fixed_cluster_id=None)

    # Baseline: fixed cluster 0 (e.g., shortest-path style)
    sim_baseline = HierarchicalRoutingSimulator(cluster_manager=cm, bandit=PairModeLinUCB(), ppo_agent=PPOAgent())
    baseline_rows = run_suite(sim_baseline, seeds, modes, fixed_cluster_id=0)

    # Write per-run results
    per_run_csv = out_dir / "acceptance_runs.csv"
    with open(per_run_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "mode",
                "seed",
                "delivery_ratio",
                "p95_latency_ms",
                "ppo_loss",
                "all_same_ratio",
                "fallback_strict",
                "fallback_fallback1",
                "fallback_fallback2",
                "cluster_hist",
            ],
        )
        writer.writeheader()
        for r in trained_rows:
            writer.writerow({"policy": "trained", **r})
        for r in baseline_rows:
            writer.writerow({"policy": "fixed0", **r})

    # Aggregates
    agg = {
        "trained": aggregate(trained_rows),
        "fixed0": aggregate(baseline_rows),
    }
    agg_csv = out_dir / "acceptance_summary.csv"
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "delivery_ratio_avg", "p95_latency_ms_avg"])
        for name, vals in agg.items():
            writer.writerow([name, vals.get("delivery_ratio_avg", ""), vals.get("p95_latency_ms_avg", "")])

    print(f"[Eval] Per-run metrics -> {per_run_csv}")
    print(f"[Eval] Summary -> {agg_csv}")


if __name__ == "__main__":
    main()
