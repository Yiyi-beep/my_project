"""
E1: Scale concurrent main flows (F) and compare SP/ECMP/HIER under heavy static and heavy burst.

Flows: base_pairs = config.FLOW_PAIRS ([(0,4),(1,4),(3,12)])
F ∈ {3,6,9,12} by duplicating base_pairs.

Outputs:
  results_E1/E1_runs.csv
  results_E1/E1_windows.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import simpy

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.getcwd(), "ns.py"))

from ns.packet.packet import Packet
from ns.port.monitor import PortMonitor
from ns.port.port import Port
from ns.port.wire import Wire

from grace_v2 import config
from grace_v2.bandit import PairModeLinUCB
from grace_v2.logging_utils import append_jsonl
from grace_v2.metrics import MetricsRecorder
from grace_v2.path_clustering import PathClusterManager, build_nsfnet_graph, nsfnet_edges
from grace_v2.ppo_agent import PPOAgent
from grace_v2.simulator import HierarchicalRoutingSimulator

MEASURE_START = 500.0
MEASURE_END = 3500.0
TOTAL_END = 4500.0
BURST_START = 1700.0
BURST_END = 2300.0
WINDOW_MS = config.WINDOW_MS
NUM_WINDOWS = int((MEASURE_END - MEASURE_START) / WINDOW_MS)
assert abs((MEASURE_END - MEASURE_START) / WINDOW_MS - NUM_WINDOWS) < 1e-6, "MEASURE range not divisible by WINDOW_MS"

BASE_PPS = 480.0
BURST_PPS = 580.0


def dist_map(graph) -> Dict[int, Dict[int, int]]:
    return dict(nx.all_pairs_shortest_path_length(graph))


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


class BaselineRouter:
    def __init__(self, env, node_id: int, neighbors: List[int], policy: str, dist_tbl=None, metrics=None):
        self.env = env
        self.node_id = node_id
        self.policy = policy
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

    def _sp_next(self, packet):
        dist_u = self.dist_tbl.get(self.node_id, {})
        du = dist_u.get(packet.dst, 1e9)
        for neigh in sorted(self.neighbor_map.values()):
            dv = self.dist_tbl.get(neigh, {}).get(packet.dst, 1e9)
            if dv == du - 1:
                return neigh
        return None

    def _ecmp_next(self, packet):
        dist_u = self.dist_tbl.get(self.node_id, {})
        du = dist_u.get(packet.dst, 1e9)
        nhs = []
        for neigh in self.neighbor_map.values():
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
        nxt = self._sp_next(packet) if self.policy == "sp" else self._ecmp_next(packet)
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


class ScheduledPacketGenerator:
    def __init__(
        self,
        env,
        src: int,
        dst: int,
        flow_id: int,
        schedule: List[Tuple[float, float, float]],
        measure_start: float,
        measure_end: float,
        metrics: MetricsRecorder | None = None,
        mode: str = "heavy",
        arrival_process: str = "poisson",
    ):
        self.env = env
        self.src = src
        self.dst = dst
        self.flow_id = flow_id
        self.schedule = schedule
        self.measure_start = measure_start
        self.measure_end = measure_end
        self.metrics = metrics
        self.mode = mode
        self.arrival_process = arrival_process
        self.packet_counter = 0
        self.out = None
        self.cluster_id = None
        self.action = env.process(self.run())

    def run(self):
        for start, end, pps in self.schedule:
            t = self.env.now
            if t < start:
                yield self.env.timeout(start - t)
            while self.env.now < end:
                if pps <= 0:
                    break
                if self.arrival_process == "poisson":
                    rate_per_ms = pps / 1000.0
                    iat = random.expovariate(rate_per_ms) if rate_per_ms > 0 else (end - self.env.now)
                else:
                    iat = 1000.0 / pps
                self.packet_counter += 1
                pkt = Packet(
                    self.env.now,
                    float(config.PACKET_SIZE_BYTES),
                    self.packet_counter,
                    src=self.src,
                    dst=self.dst,
                    flow_id=self.flow_id,
                )
                pkt.ttl = config.TTL_HOPS
                pkt.packet_id = self.packet_counter
                in_measure = self.measure_start <= self.env.now < self.measure_end
                pkt.generated_in_measure = in_measure
                if in_measure:
                    pkt.window_id = int((self.env.now - self.measure_start) // WINDOW_MS)
                    if self.metrics:
                        self.metrics.on_generate(pkt, self.env.now)
                if self.cluster_id is not None:
                    pkt.cluster_id = self.cluster_id
                pkt.mode = self.mode
                if self.out:
                    self.out.put(pkt)
                yield self.env.timeout(iat)
                if self.env.now >= end:
                    break


def build_topology(env, policy: str, metrics: MetricsRecorder):
    graph = build_nsfnet_graph()
    dist_tbl = dist_map(graph)
    nodes = {}
    wires = []
    for n in graph.nodes():
        neighs = list(graph.neighbors(n))
        nodes[n] = BaselineRouter(env, n, neighs, policy=policy, dist_tbl=dist_tbl, metrics=metrics)
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
    monitors = []
    step = 10.0
    for node in nodes.values():
        for port in node.ports:
            m = PortMonitor(env, port, dist=lambda s=step: s, pkt_in_service_included=True)
            setattr(m, "sample_step_ms", step)
            monitors.append(m)
    return nodes, wires, monitors


def _entropy_from_hist(hist: Dict[int, int]) -> float | None:
    if not hist:
        return None
    total = sum(hist.values())
    if total <= 0:
        return None
    ent = 0.0
    for v in hist.values():
        p = v / total
        if p > 0:
            ent -= p * np.log(p)
    return ent


def aggregate_window_stats(metrics: MetricsRecorder, reward_trace=None, choices_log=None):
    series = []
    for w in range(NUM_WINDOWS):
        latencies = []
        delivered = 0
        generated = 0
        bits = 0.0
        for fid_stats in metrics.window_stats.values():
            if w in fid_stats:
                ws = fid_stats[w]
                delivered += ws.delivered
                generated += ws.generated
                bits += ws.delivered_bits
                latencies.extend(ws.latencies_ms)
        p95 = float(np.percentile(latencies, 95)) if latencies else None
        goodput_win = (bits / (WINDOW_MS / 1000.0)) / 1e6 if bits else 0.0
        delivery = delivered / generated if generated > 0 else 0.0
        reward_win = None
        if reward_trace:
            win_rewards = [rv for (fid, ww), rv in reward_trace.items() if ww == w]
            reward_win = float(np.mean(win_rewards)) if win_rewards else None
        top_frac = None
        chist = None
        if choices_log and w in choices_log:
            cnt = Counter(choices_log[w].values())
            total = sum(cnt.values())
            top_frac = max(cnt.values()) / total if total > 0 else None
            chist = dict(cnt)
            ent = _entropy_from_hist(chist)
        else:
            ent = None
        series.append(
            {
                "window": w,
                "t_start_ms": MEASURE_START + w * WINDOW_MS,
                "p95_latency_ms_window": p95,
                "delivery_ratio_window": delivery,
                "goodput_mbps_window": goodput_win,
                "reward_window": reward_win,
                "cluster_top1_frac_window": top_frac,
                "cluster_entropy_window": ent,
                "cluster_hist_window": chist,
            }
        )
    return series


def burst_window_metrics(metrics: MetricsRecorder, burst_start: float | None, burst_end: float | None):
    if burst_start is None or burst_end is None:
        return {"delivery_burst": None, "p95_latency_burst": None, "goodput_burst_mbps": None, "service_ratio_bits": None}
    win_indices = []
    for w in range(NUM_WINDOWS):
        t_start = MEASURE_START + w * WINDOW_MS
        t_end = t_start + WINDOW_MS
        if max(t_start, burst_start) < min(t_end, burst_end):
            win_indices.append(w)
    if not win_indices:
        return {"delivery_burst": None, "p95_latency_burst": None, "goodput_burst_mbps": None, "service_ratio_bits": None}
    delivered = generated = 0
    bits_del = 0.0
    latencies = []
    for fid_stats in metrics.window_stats.values():
        for w in win_indices:
            if w in fid_stats:
                ws = fid_stats[w]
                delivered += ws.delivered
                generated += ws.generated
                bits_del += ws.delivered_bits
                latencies.extend(ws.latencies_ms)
    delivery = delivered / generated if generated > 0 else None
    p95 = float(np.percentile(latencies, 95)) if latencies else None
    dur_s = max((burst_end - burst_start) / 1000.0, 1e-9)
    goodput = (bits_del / dur_s) / 1e6 if bits_del else 0.0
    offered_bits = generated * config.PACKET_SIZE_BYTES * 8.0
    service_ratio = (bits_del / offered_bits) if offered_bits > 0 else None
    return {
        "delivery_burst": delivery,
        "p95_latency_burst": p95,
        "goodput_burst_mbps": goodput,
        "service_ratio_bits": service_ratio,
    }


def run_baseline(policy: str, flows: List[Tuple[int, int]], variant: str, schedule: List[Tuple[float, float, float]], seed: int):
    set_seed(seed)
    env = simpy.Environment()
    metrics = MetricsRecorder(window_ms=WINDOW_MS)
    nodes, wires, monitors = build_topology(env, policy, metrics)

    for fid, (src, dst) in enumerate(flows):
        gen = ScheduledPacketGenerator(
            env,
            src=src,
            dst=dst,
            flow_id=fid,
            schedule=schedule,
            measure_start=MEASURE_START,
            measure_end=MEASURE_END,
            metrics=metrics,
            mode="heavy",
            arrival_process="poisson",
        )
        gen.out = nodes[src]

    env.run(until=TOTAL_END)

    overall = metrics.overall_metrics()
    total_bits = 0.0
    for fid_stats in metrics.window_stats.values():
        for w, stats in fid_stats.items():
            if 0 <= w < NUM_WINDOWS:
                total_bits += stats.delivered_bits
    goodput_mbps = (total_bits / ((MEASURE_END - MEASURE_START) / 1000.0)) / 1e6 if total_bits else 0.0
    windows = aggregate_window_stats(metrics)
    burst_extra = burst_window_metrics(metrics, BURST_START, BURST_END) if variant == "burst" else {
        "delivery_burst": None,
        "p95_latency_burst": None,
        "goodput_burst_mbps": None,
        "service_ratio_bits": None,
    }

    return {
        "topology": "NSFNET",
        "flow_pairs_id": "_".join([f"{a}-{b}" for a, b in config.FLOW_PAIRS]),
        "flows": len(flows),
        "variant": variant,
        "base_pps": BASE_PPS,
        "burst_pps": BURST_PPS,
        "burst_start_ms": BURST_START if variant == "burst" else None,
        "burst_end_ms": BURST_END if variant == "burst" else None,
        "policy": policy.upper(),
        "seed": seed,
        "delivery_ratio": overall["delivery_ratio"],
        "goodput_mbps": goodput_mbps,
        "p95_latency_ms": overall["p95_latency_ms"],
        **burst_extra,
        "windows": windows,
    }


def load_trained_stage3():
    cm_path = Path("checkpoints_v2/clusters.json")
    cm = PathClusterManager.load(cm_path) if cm_path.exists() else PathClusterManager()
    if not cm.clusters:
        cm.build_all(config.FLOW_PAIRS)
    else:
        missing = [pair for pair in config.FLOW_PAIRS if pair not in cm.clusters]
        if missing:
            try:
                for pair in missing:
                    cm.build_pair(pair)
            except Exception:
                cm.build_all(config.FLOW_PAIRS)
        cm.save(cm_path)
    ppo = PPOAgent()
    ppo_ckpt = Path("checkpoints_v2/stage3_full_ppo.pth")
    if ppo_ckpt.exists():
        ppo.load(str(ppo_ckpt))
    bandit = PairModeLinUCB()
    bandit_ckpt = Path("checkpoints_v2/stage3_adapt_bandit.pkl")
    if not bandit_ckpt.exists():
        bandit_ckpt = Path("checkpoints_v2/stage3_full_bandit.pkl")
    if bandit_ckpt.exists():
        try:
            with bandit_ckpt.open("rb") as f:
                bandit = pickle.load(f)
        except Exception:
            bandit = PairModeLinUCB()
    return cm, bandit, ppo


def run_hier(flows: List[Tuple[int, int]],
             variant: str,
             schedule: List[Tuple[float, float, float]],
             seed: int,
             sim: HierarchicalRoutingSimulator,
             policy_name: str = "HIER_STAGE3",
             use_bandit: bool = True,):
    set_seed(seed)
    env = simpy.Environment()
    metrics = MetricsRecorder(window_ms=WINDOW_MS)
    nodes, wires = sim._build_topology(env, metrics)

    cluster_choices_log: Dict[int, Dict[int, int]] = {}
    controllers = {}
    for fid, (src, dst) in enumerate(flows):
        gen = ScheduledPacketGenerator(
            env,
            src=src,
            dst=dst,
            flow_id=fid,
            schedule=schedule,
            measure_start=MEASURE_START,
            measure_end=MEASURE_END,
            metrics=metrics,
            mode="heavy",
            arrival_process="poisson",
        )
        controllers[fid] = gen
        gen.out = nodes[src]

    cap_per_mode = sim.cap_profile.get("heavy", {})
    pair_k_eff = {pair: max(len(clusters), 1) for pair, clusters in sim.cluster_manager.clusters.items()}
    last_context: Dict[int, List[float]] = {
        fid: sim._make_context("heavy", None, cap_per_mode) for fid in controllers
    }

    for w in range(NUM_WINDOWS):
        window_choices = {}
        for fid, (src, dst) in enumerate(flows):
            pair = (src, dst)
            k_eff = pair_k_eff.get(pair, config.N_CLUSTERS)
            max_arm = max(k_eff - 1, 0)
            ctx = last_context.get(fid, sim._make_context("heavy", None, cap_per_mode))
            if use_bandit:
              cid, _ = sim.bandit.select(pair, "heavy", ctx, k_eff)
              cid = min(cid, max_arm)
            else:
              cid = 0  # PPO_CANDIDATE 不做 bandit；candidate 模式下 Router 会忽略 cluster_id
            controllers[fid].cluster_id = cid

            window_choices[fid] = cid
        cluster_choices_log[w] = window_choices
        env.run(until=MEASURE_START + (w + 1) * WINDOW_MS)
        for fid in controllers:
            summary = metrics.get_window_summary(fid, w)
            last_context[fid] = sim._make_context("heavy", summary, cap_per_mode)

    env.run(until=TOTAL_END)

    reward_trace: Dict[Tuple[int, int], float] = {}
    if use_bandit:
        for w in range(NUM_WINDOWS):
            for fid in controllers:
                r = metrics.bandit_reward(fid, w, cap_per_mode)
                if r is not None:
                    reward_trace[(fid, w)] = r
        windows = aggregate_window_stats(metrics, reward_trace=reward_trace, choices_log=cluster_choices_log)
    else:
        windows = aggregate_window_stats(metrics)

    burst_extra = burst_window_metrics(metrics, BURST_START, BURST_END) if variant == "burst" else {
        "delivery_burst": None,
        "p95_latency_burst": None,
        "goodput_burst_mbps": None,
        "service_ratio_bits": None,
    }
    overall = metrics.overall_metrics()
    # --- compute average goodput over the MEASURE interval ---
    total_bits = 0.0
    for fid_stats in metrics.window_stats.values():
        for w, stats in fid_stats.items():
            if 0 <= w < NUM_WINDOWS:
                total_bits += stats.delivered_bits

    dur_s = (MEASURE_END - MEASURE_START) / 1000.0
    goodput_mbps = (total_bits / dur_s) / 1e6 if total_bits else 0.0



    return {
        "topology": "NSFNET",
        "flow_pairs_id": "_".join([f"{a}-{b}" for a, b in config.FLOW_PAIRS]),
        "flows": len(flows),
        "variant": variant,
        "base_pps": BASE_PPS,
        "burst_pps": BURST_PPS,
        "burst_start_ms": BURST_START if variant == "burst" else None,
        "burst_end_ms": BURST_END if variant == "burst" else None,
        "policy": policy_name,
        "seed": seed,
        "delivery_ratio": overall["delivery_ratio"],
        "goodput_mbps": goodput_mbps,
        "p95_latency_ms": overall["p95_latency_ms"],
        **burst_extra,
        "windows": windows,
    }


def replicate_flows(base_pairs: List[Tuple[int, int]], factor: int) -> List[Tuple[int, int]]:
    flows = []
    for i in range(factor):
        flows.extend(base_pairs)
    return flows


def main():
    parser = argparse.ArgumentParser(description="E1 eval: scalability with concurrent flows.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--outdir", type=Path, default=Path("results_E1"))
    args = parser.parse_args()

    seeds = args.seeds
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    factors = [1, 2, 3, 4]  # corresponding to F=3,6,9,12
    base_pairs = config.FLOW_PAIRS
    cm, bandit, ppo = load_trained_stage3()
    sim = HierarchicalRoutingSimulator(cluster_manager=cm, bandit=bandit, ppo_agent=ppo)
    sim_candidate = HierarchicalRoutingSimulator(
        cluster_manager=cm,
        bandit=bandit,
        ppo_agent=ppo,
        router_mask_mode="candidate",
    )

    runs = []
    windows_log = outdir / "E1_windows.jsonl"
    if windows_log.exists():
        windows_log.unlink()

    for factor in factors:
        flows = replicate_flows(base_pairs, factor)
        for variant in ["static", "burst"]:
            if variant == "burst":
                schedule = [
                    (0.0, BURST_START, BASE_PPS),
                    (BURST_START, BURST_END, BURST_PPS),
                    (BURST_END, MEASURE_END, BASE_PPS),
                ]
            else:
                schedule = [(0.0, MEASURE_END, BASE_PPS)]
            for seed in seeds:
                for policy in ["sp", "ecmp"]:
                    res = run_baseline(policy, flows, variant, schedule, seed)
                    runs.append(res)
                    for wrec in res["windows"]:
                        append_jsonl(
                            windows_log,
                            {"policy": res["policy"], "flows": len(flows), "variant": variant, "seed": seed, **wrec},
                        )
                res = run_hier(flows, variant, schedule, seed, sim)
                runs.append(res)
                for wrec in res["windows"]:
                    append_jsonl(
                        windows_log,
                        {"policy": res["policy"], "flows": len(flows), "variant": variant, "seed": seed, **wrec},
                    )
                res = run_hier(
                    flows, variant, schedule, seed, sim_candidate,
                    policy_name="PPO_CANDIDATE",
                    use_bandit=False,
                )
                runs.append(res)
                for wrec in res["windows"]:
                    append_jsonl(
                        windows_log,
                        {"policy": res["policy"], "flows": len(flows), "variant": variant, "seed": seed, **wrec},
                    )
    


    runs_path = outdir / "E1_runs.csv"
    with runs_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topology",
                "flow_pairs_id",
                "flows",
                "variant",
                "base_pps",
                "burst_pps",
                "burst_start_ms",
                "burst_end_ms",
                "policy",
                "seed",
                "delivery_ratio",
                "goodput_mbps",
                "p95_latency_ms",
                "delivery_burst",
                "p95_latency_burst",
                "goodput_burst_mbps",
                "service_ratio_bits",
            ],
        )
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    print(f"[Eval E1] runs -> {runs_path}")
    print(f"[Eval E1] windows -> {windows_log}")


if __name__ == "__main__":
    main()
