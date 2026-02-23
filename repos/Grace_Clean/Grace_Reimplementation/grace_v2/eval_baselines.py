"""
Baseline evaluation for OSPF (single shortest-path), ECMP, and frozen HIER.

Outputs:
- checkpoints_v2/acceptance_runs_baselines.csv
- checkpoints_v2/acceptance_summary_baselines.csv (per policy, per mode)
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import simpy

# Allow running as a script
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ns.port.port import Port
from ns.port.wire import Wire

from grace_v2 import config
from grace_v2.bandit import PairModeLinUCB
from grace_v2.capnorm import load_cap_profile
from grace_v2.metrics import MetricsRecorder
from grace_v2.path_clustering import PathClusterManager, build_nsfnet_graph, nsfnet_edges
from grace_v2.ppo_agent import PPOAgent
from grace_v2.simulator import HierarchicalRoutingSimulator
from grace_v2.traffic import TrafficManager


def load_trained_agents(ckpt_dir: Path) -> tuple[PathClusterManager, PairModeLinUCB, PPOAgent]:
    """Load trained cluster manager, bandit, and PPO."""
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
            bandit = PairModeLinUCB()

    return cm, bandit, ppo


def shortest_path_tables(graph) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[Tuple[int, int, int], int]]:
    """Return lexicographically smallest shortest path and next-hop table."""
    paths = {}
    nh = {}
    for src in graph.nodes():
        for dst in graph.nodes():
            if src == dst:
                continue
            try:
                all_sp = list(nx.all_shortest_paths(graph, src, dst))
            except nx.NetworkXNoPath:
                all_sp = []
            all_sp = [list(p) for p in all_sp]
            all_sp.sort()
            if not all_sp:
                continue
            path = list(all_sp[0])
            paths[(src, dst)] = path
            for i, u in enumerate(path[:-1]):
                v = path[i + 1]
                nh[(src, dst, u)] = v
    return paths, nh


def dist_map(graph) -> Dict[int, Dict[int, int]]:
    return dict(nx.all_pairs_shortest_path_length(graph))


class BaselineRouter:
    def __init__(self, env, node_id: int, neighbors: List[int], policy: str, nh_table=None, dist_tbl=None, metrics=None):
        self.env = env
        self.node_id = node_id
        self.policy = policy
        self.nh_table = nh_table or {}
        self.dist_tbl = dist_tbl or {}
        self.metrics = metrics
        self.ports: List[Port] = []
        self.neighbor_ids = sorted(neighbors)
        self.neighbor_map = {}
        for idx, neigh in enumerate(self.neighbor_ids):
            port = Port(env, rate=config.PORT_RATE_BIT_PER_MS, qlimit=config.BUFFER_PACKETS, limit_bytes=False)
            self.ports.append(port)
            self.neighbor_map[idx] = neigh
        self.local_sink = None

    def _ospf_next(self, packet):
        return self.nh_table.get((packet.src, packet.dst, self.node_id))

    def _ecmp_next(self, packet):
        dist_u = self.dist_tbl.get(self.node_id, {})
        du = dist_u.get(packet.dst, 1e9)
        nhs = []
        for idx, neigh in self.neighbor_map.items():
            dv = self.dist_tbl.get(neigh, {}).get(packet.dst, 1e9)
            if dv == du - 1:
                nhs.append(neigh)
        if not nhs:
            return None
        nhs_sorted = sorted(nhs)
        key = f"{packet.flow_id}-{self.node_id}-{packet.dst}".encode()
        idx = zlib.crc32(key) % len(nhs_sorted)
        return nhs_sorted[idx]

    def _fallback(self, packet):
        dist_u = self.dist_tbl.get(self.node_id, {})
        du = dist_u.get(packet.dst, 1e9)
        for neigh in sorted(self.neighbor_map.values()):
            dv = self.dist_tbl.get(neigh, {}).get(packet.dst, 1e9)
            if dv < du:
                return neigh
        return sorted(self.neighbor_map.values())[0] if self.neighbor_map else None

    def put(self, packet):
        if packet.dst == self.node_id:
            if self.metrics:
                self.metrics.on_delivery(packet, self.env.now)
            if self.local_sink:
                self.local_sink.put(packet)
            return
        if hasattr(packet, "ttl"):
            packet.ttl -= 1
            if packet.ttl <= 0:
                if self.metrics:
                    self.metrics.on_drop(packet, self.env.now)
                return
        if not self.neighbor_map:
            if self.metrics:
                self.metrics.on_drop(packet, self.env.now)
            return
        if self.policy == "ospf":
            nxt = self._ospf_next(packet)
        else:
            nxt = self._ecmp_next(packet)
        if nxt is None:
            nxt = self._fallback(packet)
        if nxt is None:
            if self.metrics:
                self.metrics.on_drop(packet, self.env.now)
            return
        port_idx = None
        for idx, neigh in self.neighbor_map.items():
            if neigh == nxt:
                port_idx = idx
                break
        if port_idx is None:
            if self.metrics:
                self.metrics.on_drop(packet, self.env.now)
            return
        self.ports[port_idx].put(packet)


def build_baseline_topology(env, policy: str, nh_table=None, dist_tbl=None, metrics=None):
    graph = build_nsfnet_graph()
    nodes = {}
    wires = []
    for n in graph.nodes():
        neighs = list(graph.neighbors(n))
        nodes[n] = BaselineRouter(env, n, neighs, policy=policy, nh_table=nh_table, dist_tbl=dist_tbl, metrics=metrics)
    for u, v, delay in nsfnet_edges():
        wire_uv = Wire(env, delay_dist=lambda d=delay: d)
        wire_vu = Wire(env, delay_dist=lambda d=delay: d)
        wires.extend([wire_uv, wire_vu])
        idx_u = sorted(list(graph.neighbors(u))).index(v)
        idx_v = sorted(list(graph.neighbors(v))).index(u)
        nodes[u].ports[idx_u].out = wire_uv
        nodes[v].ports[idx_v].out = wire_vu
        wire_uv.out = nodes[v]
        wire_vu.out = nodes[u]
    return nodes, wires


def run_baseline(policy: str, mode: str, seed: int, cap_profile: Dict[str, Dict[str, float]]):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    graph = build_nsfnet_graph()
    dist_tbl = dist_map(graph)
    nh_table = None
    if policy == "ospf":
        _, nh_table = shortest_path_tables(graph)

    metrics = MetricsRecorder(window_ms=config.WINDOW_MS)
    nodes, _ = build_baseline_topology(env, policy=policy, nh_table=nh_table, dist_tbl=dist_tbl, metrics=metrics)
    tm = TrafficManager(env)

    measure_start = config.WARMUP_MS
    measure_end = measure_start + config.MEASURE_MS
    duration_ms = measure_end
    for fid, (src, dst) in enumerate(config.FLOW_PAIRS):
        gen, controller = tm.add_flow(
            flow_id=fid,
            src=src,
            dst=dst,
            pps=config.MODES_PPS[mode],
            duration_ms=duration_ms,
            measure_start=measure_start,
            measure_end=measure_end,
            window_ms=config.WINDOW_MS,
            initial_cluster=0,
            mode=mode,
            metrics=metrics,
        )
        controller.set_mode(mode)
        gen.out = nodes[src]
        nodes[dst].local_sink = None

    env.run(until=config.episode_total_ms())
    overall = metrics.overall_metrics()
    total_gen = overall["total_generated"]
    total_del = overall["total_delivered"]
    delivery_ratio = overall["delivery_ratio"]
    p95 = overall["p95_latency_ms"]
    p50 = overall["p50_latency_ms"]
    avg_lat = overall.get("avg_latency_ms")

    total_bits = 0.0
    num_windows = int(config.MEASURE_MS // config.WINDOW_MS)
    for fid_stats in metrics.window_stats.values():
        for w, stats in fid_stats.items():
            if 0 <= w < num_windows:
                total_bits += stats.delivered_bits
    goodput_mbps = (total_bits / (config.MEASURE_MS / 1000.0)) / 1e6 if total_bits else 0.0

    rewards = []
    caps = cap_profile.get(mode, {})
    num_windows = int(config.MEASURE_MS // config.WINDOW_MS)
    for fid in range(len(config.FLOW_PAIRS)):
        for w in range(num_windows):
            r = metrics.bandit_reward(fid, w, caps)
            if r is not None:
                rewards.append(r)
    reward_total = float(np.mean(rewards)) if rewards else None
    undelivered = max(total_gen - total_del, 0)

    return {
        "policy": policy,
        "mode": mode,
        "seed": seed,
        "delivery_ratio": delivery_ratio,
        "p95_latency_ms": p95,
        "goodput_mbps": goodput_mbps,
        "reward_total": reward_total,
        "avg_latency_ms": avg_lat,
        "p50_latency_ms": p50,
        "undelivered": undelivered,
    }


def run_hier_inference(cm, bandit, ppo, modes, seeds, cap_profile):
    sim = HierarchicalRoutingSimulator(cluster_manager=cm, bandit=bandit, ppo_agent=ppo, cap_profile=cap_profile)
    rows = []
    for mode in modes:
        for seed in seeds:
            res = sim.run_episode(mode=mode, stage="eval", seed=seed, fixed_cluster_id=None)
            rewards = list(res.reward_trace.values())
            reward_total = float(np.mean(rewards)) if rewards else None
            rows.append(
                {
                    "policy": "hier",
                    "mode": mode,
                    "seed": seed,
                    "delivery_ratio": res.delivery_ratio,
                    "p95_latency_ms": res.p95_latency_ms,
                    "goodput_mbps": res.goodput_mbps,
                    "reward_total": reward_total,
                    "avg_latency_ms": res.overall_metrics.get("avg_latency_ms"),
                    "p50_latency_ms": res.overall_metrics.get("p50_latency_ms"),
                    "undelivered": max(
                        res.overall_metrics.get("total_generated", 0) - res.overall_metrics.get("total_delivered", 0), 0
                    ),
                }
            )
    return rows


def main():
    cap_profile = load_cap_profile()
    modes = list(config.MODES_PPS.keys())
    seeds = list(range(5))

    cm, bandit, ppo = load_trained_agents(Path("checkpoints_v2"))
    hier_rows = run_hier_inference(cm, bandit, ppo, modes, seeds, cap_profile)

    baseline_rows = []
    for policy in ["ospf", "ecmp"]:
        for mode in modes:
            for seed in seeds:
                baseline_rows.append(run_baseline(policy, mode, seed, cap_profile))

    rows = hier_rows + baseline_rows
    runs_path = Path("checkpoints_v2/acceptance_runs_baselines.csv")
    with runs_path.open("w", newline="") as f:
        import csv

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "mode",
                "seed",
                "delivery_ratio",
                "p95_latency_ms",
                "goodput_mbps",
                "reward_total",
                "avg_latency_ms",
                "p50_latency_ms",
                "undelivered",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Aggregates per policy, per mode
    agg = {}
    for policy in set(r["policy"] for r in rows):
        for mode in modes:
            agg_rows = [r for r in rows if r["policy"] == policy and r["mode"] == mode]
            if not agg_rows:
                continue
            agg[(policy, mode)] = {
                "delivery_ratio_avg": float(np.mean([r["delivery_ratio"] for r in agg_rows])),
                "p95_latency_ms_avg": float(np.mean([r["p95_latency_ms"] for r in agg_rows])),
                "goodput_mbps_avg": float(np.mean([r["goodput_mbps"] for r in agg_rows])),
                "reward_avg": float(np.mean([r["reward_total"] for r in agg_rows if r["reward_total"] is not None]))
                if any(r["reward_total"] is not None for r in agg_rows)
                else None,
            }
    summary_path = Path("checkpoints_v2/acceptance_summary_baselines.csv")
    with summary_path.open("w", newline="") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow(["policy", "mode", "delivery_ratio_avg", "p95_latency_ms_avg", "goodput_mbps_avg", "reward_avg"])
        for (policy, mode), v in agg.items():
            writer.writerow([policy, mode, v["delivery_ratio_avg"], v["p95_latency_ms_avg"], v["goodput_mbps_avg"], v["reward_avg"]])

    print(f"[Eval] runs -> {runs_path}")
    print(f"[Eval] summary -> {summary_path}")


if __name__ == "__main__":
    main()
