
"""
Adapt bandit per concurrency level (F=3/6/9/12) using stepped heavy schedule (base->burst->base),
matching E1 eval behavior. PPO frozen; bandit updates enabled during adaptation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import simpy

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.getcwd(), "ns.py"))

from ns.packet.packet import Packet

from grace_v2 import config
from grace_v2.bandit import PairModeLinUCB
from grace_v2.metrics import MetricsRecorder
from grace_v2.path_clustering import PathClusterManager
from grace_v2.ppo_agent import PPOAgent
from grace_v2.simulator import HierarchicalRoutingSimulator

MEASURE_START = config.WARMUP_MS
MEASURE_END = MEASURE_START + config.MEASURE_MS
TOTAL_END = config.episode_total_ms()
WINDOW_MS = config.WINDOW_MS
NUM_WINDOWS = int((MEASURE_END - MEASURE_START) / WINDOW_MS)
BASE_PPS = 480.0
BURST_PPS = 580.0
BURST_START = 1700.0
BURST_END = 2300.0


def set_seed(seed: int):
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
    # Deduplicate to avoid repeated build on replicated pairs
    missing = list(dict.fromkeys(missing))
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


def replicate_flows(base_pairs: List[Tuple[int, int]], factor: int) -> List[Tuple[int, int]]:
    flows: List[Tuple[int, int]] = []
    for _ in range(factor):
        flows.extend(base_pairs)
    return flows


class ScheduledFlow:
    def __init__(self, env, src, dst, fid, schedule, metrics=None):
        self.env = env
        self.src = src
        self.dst = dst
        self.fid = fid
        self.schedule = schedule
        self.metrics = metrics
        self.cluster_id = 0
        self.out = None
        self.action = env.process(self.run())

    def run(self):
        for start, end, pps in self.schedule:
            if self.env.now < start:
                yield self.env.timeout(start - self.env.now)
            while self.env.now < end:
                if pps <= 0:
                    break
                rate_per_ms = pps / 1000.0
                iat = random.expovariate(rate_per_ms) if rate_per_ms > 0 else (end - self.env.now)
                pkt = Packet(
                    self.env.now,
                    float(config.PACKET_SIZE_BYTES),
                    0,
                    src=self.src,
                    dst=self.dst,
                    flow_id=self.fid,
                )
                pkt.ttl = config.TTL_HOPS
                pkt.cluster_id = self.cluster_id
                pkt.mode = "heavy"
                in_measure = MEASURE_START <= self.env.now < MEASURE_END
                pkt.generated_in_measure = in_measure
                if in_measure:
                    pkt.window_id = int((self.env.now - MEASURE_START) // WINDOW_MS)
                    if self.metrics:
                        self.metrics.on_generate(pkt, self.env.now)
                if self.out:
                    self.out.put(pkt)
                yield self.env.timeout(iat)
                if self.env.now >= end:
                    break


def adapt_episode(sim: HierarchicalRoutingSimulator, flows: List[Tuple[int, int]], schedule, seed: int):
    set_seed(seed)
    env = simpy.Environment()
    metrics = MetricsRecorder(window_ms=WINDOW_MS)
    nodes, _ = sim._build_topology(env, metrics)

    controllers = {}
    for fid, (src, dst) in enumerate(flows):
        gen = ScheduledFlow(env, src, dst, fid, schedule, metrics)
        controllers[fid] = gen
        gen.out = nodes[src]

    cap_per_mode = sim.cap_profile.get("heavy", {})
    pair_k_eff = {pair: max(len(clusters), 1) for pair, clusters in sim.cluster_manager.clusters.items()}
    last_context: Dict[int, List[float]] = {
        fid: sim._make_context("heavy", None, cap_per_mode) for fid in controllers
    }
    window_records: Dict[int, Dict[int, Tuple[List[float], int]]] = {fid: {} for fid in controllers}

    # Window loop with bandit select/update
    for w in range(NUM_WINDOWS):
        for fid, (src, dst) in enumerate(flows):
            pair = (src, dst)
            k_eff = pair_k_eff.get(pair, config.N_CLUSTERS)
            max_arm = max(k_eff - 1, 0)
            ctx = last_context.get(fid, sim._make_context("heavy", None, cap_per_mode))
            cid, _ = sim.bandit.select(pair, "heavy", ctx, k_eff)
            cid = min(cid, max_arm)
            controllers[fid].cluster_id = cid
            window_records[fid][w] = (ctx, cid)
        env.run(until=MEASURE_START + (w + 1) * WINDOW_MS)
        for fid in controllers:
            summary = metrics.get_window_summary(fid, w)
            last_context[fid] = sim._make_context("heavy", summary, cap_per_mode)

    env.run(until=TOTAL_END)
    # Update bandit with rewards
    for w in range(NUM_WINDOWS):
        for fid, (src, dst) in enumerate(flows):
            r = metrics.bandit_reward(fid, w, cap_per_mode)
            if r is not None:
                ctx, cid = window_records.get(fid, {}).get(w, (None, None))
                if ctx is None or cid is None:
                    continue
                k_eff = pair_k_eff.get((src, dst), config.N_CLUSTERS)
                sim.bandit.update((src, dst), "heavy", cid, ctx, r, k_eff)


def adapt_for_F(factor: int, seeds: List[int], episodes: int, bandit_init: Path, bandit_out: Path, clusters_path: Path, ppo_ckpt: Path):
    orig_pairs = list(config.FLOW_PAIRS)
    orig_modes_pps = dict(config.MODES_PPS)
    try:
        config.FLOW_PAIRS = replicate_flows(orig_pairs, factor)
        config.MODES_PPS.update({"heavy": BASE_PPS})
        F = len(config.FLOW_PAIRS)
        # Use per-F clusters to avoid confusion
        clusters_file = clusters_path.parent / f"clusters_F{F}.json"
        cm = load_cm(clusters_file)
        bandit = PairModeLinUCB()
        if bandit_init.exists():
            try:
                with bandit_init.open("rb") as f:
                    bandit = pickle.load(f)
            except Exception:
                bandit = PairModeLinUCB()
        ppo = PPOAgent()
        if ppo_ckpt.exists():
            ppo.load(str(ppo_ckpt))

        sim = HierarchicalRoutingSimulator(cluster_manager=cm, bandit=bandit, ppo_agent=ppo)
        static_sched = [(0.0, MEASURE_END, BASE_PPS)]
        burst_sched = [(0.0, BURST_START, BASE_PPS), (BURST_START, BURST_END, BURST_PPS), (BURST_END, MEASURE_END, BASE_PPS)]

        for seed in seeds:
            for ep in range(episodes):
                ep_seed = seed * 1000 + ep
                set_seed(ep_seed)
                adapt_episode(sim, config.FLOW_PAIRS, static_sched, ep_seed)
                adapt_episode(sim, config.FLOW_PAIRS, burst_sched, ep_seed)

        bandit_out.parent.mkdir(parents=True, exist_ok=True)
        with bandit_out.open("wb") as f:
            pickle.dump(sim.bandit, f)
        cm.save(clusters_file)
        print(f"[adapt F={len(config.FLOW_PAIRS)}] saved bandit -> {bandit_out}, clusters -> {clusters_file}")
    finally:
        config.FLOW_PAIRS = orig_pairs
        config.MODES_PPS = orig_modes_pps


def main():
    parser = argparse.ArgumentParser(description="Adapt bandit per concurrency level (stepped heavy) for E1 scalability.")
    parser.add_argument("--factors", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--bandit-init", type=Path, default=Path("checkpoints_v2/stage3_full_bandit.pkl"))
    parser.add_argument("--clusters", type=Path, default=Path("checkpoints_v2/clusters.json"))
    parser.add_argument("--ppo", type=Path, default=Path("checkpoints_v2/stage3_full_ppo.pth"))
    parser.add_argument("--outdir", type=Path, default=Path("checkpoints_v2"))
    args = parser.parse_args()

    for factor in args.factors:
        F = factor * len(config.FLOW_PAIRS)
        out_path = args.outdir / f"stage3_adapt_bandit_F{F}.pkl"
        adapt_for_F(factor, args.seeds, args.episodes, args.bandit_init, out_path, args.clusters, args.ppo)


if __name__ == "__main__":
    main()
