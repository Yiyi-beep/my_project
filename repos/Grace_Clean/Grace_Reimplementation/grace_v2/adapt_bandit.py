"""
Online bandit adaptation for new (src,dst) pairs with fixed PPS tiers.

Usage (from repo root):
  python -m grace_v2.adapt_bandit --seeds 0 1 2 --episodes 20 --modes light medium heavy burst

This:
- Loads stage3 PPO (frozen) and stage3 bandit as initialization.
- Uses the new FLOW_PAIRS from config.py (ensure you already updated it).
- Sets PPS tiers to: light=400, medium=420, heavy=460, burst=575.
- Runs bandit-only updates (PPO frozen) for the given modes/seeds.
- Saves adapted bandit to checkpoints_v2/stage3_adapt_bandit.pkl (default).

Notes:
- Burst here is constant 575 pps (no stepped burst schedule); use static adaptation then evaluate burst separately.
- Clusters will be rebuilt if clusters.json is absent.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from grace_v2 import config
from grace_v2.bandit import PairModeLinUCB
from grace_v2.path_clustering import PathClusterManager
from grace_v2.ppo_agent import PPOAgent
from grace_v2.simulator import HierarchicalRoutingSimulator


# Override PPS tiers for adaptation (update per calibration).
NEW_PPS = {"light": 400, "medium": 420, "heavy": 480, "burst": 540}


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_clusters(cm: PathClusterManager):
    missing = [pair for pair in config.FLOW_PAIRS if pair not in cm.clusters]
    if missing:
        try:
            for pair in missing:
                cm.build_pair(pair)
        except Exception:
            cm.build_all(config.FLOW_PAIRS)


def load_cm(path: Path) -> PathClusterManager:
    cm = PathClusterManager()
    if path.exists():
        try:
            cm = PathClusterManager.load(path)
        except Exception:
            cm = PathClusterManager()
    if not cm.clusters:
        cm.build_all(config.FLOW_PAIRS)
    ensure_clusters(cm)
    return cm


def load_bandit(path: Path) -> PairModeLinUCB:
    bandit = PairModeLinUCB()
    if path.exists():
        try:
            with path.open("rb") as f:
                bandit = pickle.load(f)
        except Exception:
            bandit = PairModeLinUCB()
    return bandit


def load_ppo(path: Path) -> PPOAgent:
    ppo = PPOAgent()
    if path.exists():
        try:
            ppo.load(str(path))
        except Exception:
            pass
    return ppo


def adapt_bandit(seeds, modes, episodes, bandit_init: Path, bandit_out: Path, clusters_path: Path, ppo_ckpt: Path):
    # Override PPS tiers in-memory for this run
    config.MODES_PPS.update(NEW_PPS)

    cm = load_cm(clusters_path)
    bandit = load_bandit(bandit_init)
    ppo = load_ppo(ppo_ckpt)
    sim = HierarchicalRoutingSimulator(cluster_manager=cm, bandit=bandit, ppo_agent=ppo)

    for seed in seeds:
        for mode in modes:
            for ep in range(episodes):
                ep_seed = seed * 1000 + ep
                set_seed(ep_seed)
                sim.run_episode(mode=mode, stage="bandit", seed=ep_seed, disable_updates=False)

    # Save adapted artifacts
    bandit_out.parent.mkdir(parents=True, exist_ok=True)
    with bandit_out.open("wb") as f:
        pickle.dump(sim.bandit, f)
    if hasattr(cm, "save"):
        cm.save(clusters_path)

    print(f"[adapt] done. bandit -> {bandit_out}")
    print(f"[adapt] clusters -> {clusters_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Online bandit adaptation for new flow pairs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to adapt with")
    parser.add_argument("--modes", type=str, nargs="+", default=["light", "medium", "heavy"], help="Modes to adapt")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per mode per seed")
    parser.add_argument(
        "--bandit-init",
        type=Path,
        default=Path("checkpoints_v2/stage3_full_bandit.pkl"),
        help="Initial bandit checkpoint",
    )
    parser.add_argument(
        "--bandit-out",
        type=Path,
        default=Path("checkpoints_v2/stage3_adapt_bandit.pkl"),
        help="Where to save adapted bandit",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("checkpoints_v2/clusters.json"),
        help="Cluster file (rebuilt if missing)",
    )
    parser.add_argument(
        "--ppo",
        type=Path,
        default=Path("checkpoints_v2/stage3_full_ppo.pth"),
        help="Stage3 PPO checkpoint (frozen)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    adapt_bandit(args.seeds, args.modes, args.episodes, args.bandit_init, args.bandit_out, args.clusters, args.ppo)


if __name__ == "__main__":
    main()
