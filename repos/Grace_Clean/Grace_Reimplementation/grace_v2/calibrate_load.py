"""
Load calibration sweep for SP/ECMP to align pps for light/medium/heavy/burst.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import simpy

# Ensure package importable when run as a script
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.getcwd(), "ns.py"))

from ns.port.port import Port
from ns.port.wire import Wire
from ns.packet.packet import Packet
from ns.port.monitor import PortMonitor

from grace_v2 import config
from grace_v2.metrics import MetricsRecorder
from grace_v2.path_clustering import build_nsfnet_graph, nsfnet_edges


def dist_map(graph) -> Dict[int, Dict[int, int]]:
    return dict(nx.all_pairs_shortest_path_length(graph))


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
        best = None
        for neigh in sorted(self.neighbor_map.values()):
            dv = self.dist_tbl.get(neigh, {}).get(packet.dst, 1e9)
            if dv == du - 1:
                best = neigh
                break
        return best

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
        if self.policy == "sp":
            nxt = self._sp_next(packet)
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


class ConstPacketGenerator:
    def __init__(
        self,
        env,
        src: int,
        dst: int,
        flow_id: int,
        pps: float,
        duration_ms: float,
        measure_start: float,
        measure_end: float,
        metrics=None,
    ):
        self.env = env
        self.src = src
        self.dst = dst
        self.flow_id = flow_id
        self.pps = pps
        self.duration_ms = duration_ms
        self.measure_start = measure_start
        self.measure_end = measure_end
        self.packet_counter = 0
        self.out = None
        self.metrics = metrics
        self.action = env.process(self.run())

    def run(self):
        iat = 1000.0 / self.pps
        # emit first packet at t=0
        while self.env.now < self.duration_ms:
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
            pkt.generated_in_measure = self.measure_start <= self.env.now < self.measure_end
            if pkt.generated_in_measure:
                pkt.window_id = int((self.env.now - self.measure_start) // config.WINDOW_MS)
                if self.metrics:
                    self.metrics.on_generate(pkt, self.env.now)
            if self.out:
                self.out.put(pkt)
            yield self.env.timeout(iat)
            if self.env.now >= self.duration_ms:
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
    step = 1.0  # ms sampling for queue size
    for node in nodes.values():
        for port in node.ports:
            m = PortMonitor(env, port, dist=lambda s=step: s, pkt_in_service_included=True)
            setattr(m, "sample_step_ms", step)
            monitors.append(m)
    return nodes, wires, monitors


def run_single(policy: str, pps: float, seed: int, flows: List[Tuple[int, int]]):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    metrics = MetricsRecorder(window_ms=config.WINDOW_MS)
    nodes, wires, monitors = build_topology(env, policy, metrics)

    measure_start = config.WARMUP_MS
    measure_end = measure_start + config.MEASURE_MS
    duration_ms = measure_end

    generators = []
    for fid, (src, dst) in enumerate(flows):
        gen = ConstPacketGenerator(
            env,
            src=src,
            dst=dst,
            flow_id=fid,
            pps=pps,
            duration_ms=duration_ms,
            measure_start=measure_start,
            measure_end=measure_end,
            metrics=metrics,
        )
        gen.out = nodes[src]
        generators.append(gen)

    env.run(until=config.episode_total_ms())

    overall = metrics.overall_metrics()
    total_gen = overall["total_generated"]
    total_del = overall["total_delivered"]
    delivery_ratio = overall["delivery_ratio"]
    p95 = overall["p95_latency_ms"]
    p50 = overall.get("p50_latency_ms")
    avg_lat = overall.get("avg_latency_ms")
    num_windows = int(math.ceil(config.MEASURE_MS / config.WINDOW_MS))

    total_bits = 0.0
    for fid_stats in metrics.window_stats.values():
        for w, stats in fid_stats.items():
            if 0 <= w < num_windows:
                total_bits += stats.delivered_bits
    goodput_mbps = (total_bits / (config.MEASURE_MS / 1000.0)) / 1e6 if total_bits else 0.0

    # Queue stats filter measure window
    samples = []
    for m in monitors:
        times = getattr(m, "times", None) or getattr(m, "timestamps", None)
        sizes = getattr(m, "sizes_byte", [])
        if times is None and hasattr(m, "sample_step_ms"):
            step = getattr(m, "sample_step_ms")
            times = [i * step for i in range(len(sizes))]
        if times is not None and len(times) == len(sizes):
            for t, val in zip(times, sizes):
                if measure_start <= t < measure_end:
                    samples.append(val)
        else:
            samples.extend(sizes)
    max_queue_bytes = max(samples) if samples else 0
    avg_queue_bytes = float(np.mean(samples)) if samples else 0.0

    undelivered = max(total_gen - total_del, 0)

    return {
        "policy": policy,
        "pps": pps,
        "seed": seed,
        "delivery_ratio": delivery_ratio,
        "p95_latency_ms": p95,
        "p50_latency_ms": p50,
        "avg_latency_ms": avg_lat,
        "goodput_mbps": goodput_mbps,
        "max_queue_bytes": max_queue_bytes,
        "avg_queue_bytes": avg_queue_bytes,
        "undelivered": undelivered,
    }


def select_levels(df: List[dict]) -> Dict[str, Dict[str, float]]:
    """Heuristic selection of light/medium/heavy/burst based on ECMP stats."""
    levels = {}
    by_pps = {}
    for row in df:
        pps = row["pps"]
        by_pps.setdefault(pps, []).append(row)
    agg = []
    for pps, rows in by_pps.items():
        delivery = np.mean([r["delivery_ratio"] for r in rows])
        p95_vals = [r["p95_latency_ms"] for r in rows if r["p95_latency_ms"] is not None]
        p95 = float(np.mean(p95_vals)) if p95_vals else None
        goodput = np.mean([r["goodput_mbps"] for r in rows])
        maxq = np.mean([r["max_queue_bytes"] for r in rows])
        agg.append({"pps": pps, "delivery": delivery, "p95": p95, "goodput": goodput, "maxq": maxq})
    agg.sort(key=lambda x: x["pps"])

    min_p95 = min([a["p95"] for a in agg if a["p95"] is not None], default=None)

    # Light: delivery>=0.99 and p95 near best (<=1.2x min)
    light_candidates = [
        a for a in agg if a["delivery"] >= 0.99 and (min_p95 is None or (a["p95"] or 0) <= 1.2 * min_p95)
    ]
    if light_candidates:
        levels["light"] = light_candidates[-1]

    # Medium: delivery>=0.99 and p95 significantly higher (>=1.5x light_p95)
    if "light" in levels:
        light_p95 = levels["light"]["p95"] or (min_p95 or 0)
        med_candidates = [a for a in agg if a["delivery"] >= 0.99 and (a["p95"] or 0) >= 1.5 * light_p95]
        if med_candidates:
            # pick the highest pps in this stable-but-higher-latency band
            levels["medium"] = med_candidates[-1]

    # Heavy: delivery still decent (>=0.8) and p95 grows further
    if "medium" in levels:
        med_p95 = levels["medium"]["p95"] or (min_p95 or 0)
        heavy_candidates = [a for a in agg if a["delivery"] >= 0.8 and (a["p95"] or 0) >= 1.3 * med_p95]
        if heavy_candidates:
            # choose the highest pps that meets heavy criteria to be closer to the knee
            levels["heavy"] = heavy_candidates[-1]
    else:
        heavy_candidates = [a for a in agg if a["delivery"] >= 0.8]
        if heavy_candidates:
            levels["heavy"] = heavy_candidates[-1]

    # Burst: suggest one step above heavy if available
    if "heavy" in levels:
        heavy_pps = levels["heavy"]["pps"]
        higher = [a for a in agg if a["pps"] > heavy_pps]
        if higher:
            levels["burst"] = higher[0]

    # Recommend burst multipliers (for Grace-style burst-by-rate)
    burst_k = [1.25, 1.5, 1.75]

    return {"ecmp_levels": levels, "agg": agg, "recommended_burst_k": burst_k}


def main():
    parser = argparse.ArgumentParser(description="Calibrate load levels via SP/ECMP sweep.")
    parser.add_argument("--pairs", type=str, help="Path to JSON/CSV of flow pairs [[src,dst],...]", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to use")
    parser.add_argument(
        "--pps", type=int, nargs="+", default=[300, 340, 380, 400, 420, 440, 460, 500, 560], help="pps grid"
    )
    args = parser.parse_args()

    flows = config.FLOW_PAIRS
    if args.pairs:
        path = Path(args.pairs)
        if path.suffix.lower() == ".json":
            flows = [tuple(x) for x in json.loads(path.read_text())]
        else:
            import pandas as pd

            df_pairs = pd.read_csv(path, header=None)
            flows = [tuple(map(int, row)) for row in df_pairs.values.tolist()]

    policies = ["sp", "ecmp"]
    rows = []
    for policy in policies:
        for pps in args.pps:
            for seed in args.seeds:
                rows.append(run_single(policy, pps, seed, flows))

    out_path = Path("checkpoints_v2/calibration_scan.csv")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "pps",
                "seed",
                "delivery_ratio",
                "p95_latency_ms",
                "p50_latency_ms",
                "avg_latency_ms",
                "goodput_mbps",
                "max_queue_bytes",
                "avg_queue_bytes",
                "undelivered",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    ecmp_rows = [r for r in rows if r["policy"] == "ecmp"]
    summary = select_levels(ecmp_rows)
    summary_path = Path("checkpoints_v2/calibration_summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Calib] saved -> {out_path}")
    print(f"[Calib] summary -> {summary_path}")


if __name__ == "__main__":
    main()
