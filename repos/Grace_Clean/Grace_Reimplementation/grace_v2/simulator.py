"""
SimPy-based simulator for hierarchical routing and baselines.
"""

from __future__ import annotations

import json
import os
import random
import sys
import zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import simpy

# Ensure ns.py is importable before importing ns modules.
sys.path.append(os.path.join(os.getcwd(), "ns.py"))
from ns.port.wire import Wire

from . import config
from .bandit import PairModeLinUCB
from .capnorm import load_cap_profile
from .metrics import MetricsRecorder
from .path_clustering import PathClusterManager, build_nsfnet_graph, nsfnet_edges
from .ppo_agent import PPOAgent
from .router import HierarchicalRouter
from .traffic import TrafficManager


@dataclass
class EpisodeResult:
    mode: str
    stage: str
    delivery_ratio: float
    p95_latency_ms: float | None
    reward_trace: Dict[Tuple[int, int], float]
    ppo_loss: float | dict | None
    cluster_hist: Dict[Tuple[int, int], Dict[int, int]]
    all_same_ratio: float
    fallback_counts: Dict[str, int]
    overall_metrics: dict
    goodput_mbps: float


class HierarchicalRoutingSimulator:
    def __init__(
        self,
        *,
        cluster_manager: PathClusterManager | None = None,
        bandit: PairModeLinUCB | None = None,
        ppo_agent: PPOAgent | None = None,
        router_mask_mode: str = "cluster", 
        cap_profile: Dict[str, Dict[str, float]] | None = None,
    ):
        self.graph = build_nsfnet_graph()
        self.edges = nsfnet_edges()
        self.cluster_manager = cluster_manager or PathClusterManager(self.graph)
        if not self.cluster_manager.clusters:
            self.cluster_manager.build_all(config.FLOW_PAIRS)
        self.bandit = bandit or PairModeLinUCB()
        self.ppo_agent = ppo_agent or PPOAgent()
        self.router_mask_mode = router_mask_mode
        self.cap_profile = cap_profile or load_cap_profile()

    def _build_topology(self, env, metrics: MetricsRecorder):
        nodes = {}
        neighbor_lists: Dict[int, List[int]] = {}
        for n in self.graph.nodes():
            neighbors = sorted(list(self.graph.neighbors(n)))
            neighbor_lists[n] = neighbors
            nodes[n] = HierarchicalRouter(
                env,
                n,
                neighbors,
                port_rate=config.PORT_RATE_BIT_PER_MS,
                buffer_packets=config.BUFFER_PACKETS,
                cluster_manager=self.cluster_manager,
                ppo_agent=self.ppo_agent,
                mask_mode=self.router_mask_mode,
                metrics=metrics,
            )

        wires = []
        for u, v, delay in self.edges:
            wire_uv = Wire(env, delay_dist=lambda d=delay: d)
            wire_vu = Wire(env, delay_dist=lambda d=delay: d)
            wires.extend([wire_uv, wire_vu])

            u_idx = neighbor_lists[u].index(v)
            v_idx = neighbor_lists[v].index(u)

            nodes[u].ports[u_idx].out = wire_uv
            nodes[v].ports[v_idx].out = wire_vu
            wire_uv.out = nodes[v]
            wire_vu.out = nodes[u]

        return nodes, wires

    def _make_context(self, mode: str, summary: dict | None, caps: Dict[str, float]) -> List[float]:
        mode_vec = [0.0, 0.0, 0.0, 0.0]
        mode_idx = {"light": 0, "medium": 1, "heavy": 2, "burst": 3}.get(mode, 0)
        mode_vec[mode_idx] = 1.0
        if not summary:
            return mode_vec + [0.0, 0.0, 0.0]

        latency = summary.get("avg_latency_ms") or 0.0
        delivery = summary.get("delivery_ratio") or 0.0
        goodput_bits = summary.get("goodput_bits") or 0.0

        L_cap = caps.get("L_cap", None)
        T_cap = caps.get("T_cap_bits", None)
        lat_norm = float(np.clip(latency / L_cap, 0.0, 1.0)) if L_cap else float(latency)
        goodput_norm = float(np.clip(goodput_bits / T_cap, 0.0, 1.0)) if T_cap else float(goodput_bits)

        return mode_vec + [lat_norm, delivery, goodput_norm]

    def run_episode(
        self,
        mode: str,
        stage: str,
        seed: int = 0,
        fixed_cluster_id: int | None = None,
        *,
        force_random_cluster: bool = False,
        random_cluster_seed: int | None = None,
        disable_updates: bool = False,
    ) -> EpisodeResult:
        random.seed(seed)
        np.random.seed(seed)

        env = simpy.Environment()
        metrics = MetricsRecorder(window_ms=config.WINDOW_MS)
        nodes, _ = self._build_topology(env, metrics)

        # Traffic setup
        measure_start = config.WARMUP_MS
        measure_end = measure_start + config.MEASURE_MS
        duration_ms = measure_end  # stop generating before drain
        tm = TrafficManager(env)
        controllers = {}
        pair_k_eff: Dict[Tuple[int, int], int] = {}
        for pair, clusters in self.cluster_manager.clusters.items():
            pair_k_eff[pair] = max(len(clusters), 1)
        cap_per_mode = self.cap_profile.get(mode, {})
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
            controllers[fid] = controller
            gen.out = nodes[src]

        # Warmup
        env.run(until=measure_start)

        last_context: Dict[int, List[float]] = {
            fid: self._make_context(mode, None, cap_per_mode) for fid in controllers
        }
        reward_trace: Dict[Tuple[int, int], float] = {}
        num_windows = int(config.MEASURE_MS // config.WINDOW_MS)
        cluster_choice_hist: Dict[Tuple[int, int], Dict[int, int]] = {}
        same_cluster_count = 0
        window_history: List[Dict[int, Tuple[int, List[float], Tuple[int, int], int]]] = []
        rand_seed = random_cluster_seed if random_cluster_seed is not None else seed

        for w in range(num_windows):
            window_choices: Dict[int, Tuple[int, List[float], Tuple[int, int], int]] = {}
            for fid, controller in controllers.items():
                ctx = last_context.get(fid, self._make_context(mode, None, cap_per_mode))
                pair = config.FLOW_PAIRS[fid]
                k_eff = pair_k_eff.get(pair, config.N_CLUSTERS)
                max_arm = max(k_eff - 1, 0)
                if fixed_cluster_id is not None:
                    cid = min(fixed_cluster_id, max_arm)
                elif force_random_cluster:
                    h = zlib.crc32(f"{rand_seed}-{fid}-{w}".encode())
                    cid = h % (k_eff if k_eff > 0 else 1)
                elif stage == "pretrain":
                    cid = random.randint(0, max_arm)
                else:
                    cid, _ = self.bandit.select(pair, mode, ctx, k_eff)
                controller.set_cluster(cid)
                window_choices[fid] = (cid, ctx, pair, k_eff)
                cluster_choice_hist.setdefault(pair, {}).setdefault(cid, 0)
                cluster_choice_hist[pair][cid] += 1

            window_end = measure_start + (w + 1) * config.WINDOW_MS
            env.run(until=window_end)

            chosen = [cid for cid, *_ in window_choices.values()]
            if chosen and len(set(chosen)) == 1:
                same_cluster_count += 1

            window_history.append(window_choices)

            for fid in controllers:
                summary = metrics.get_window_summary(fid, w)
                last_context[fid] = self._make_context(mode, summary, cap_per_mode)

        # Drain remaining packets and settle rewards
        env.run(until=config.episode_total_ms())
        for w, choices in enumerate(window_history):
            for fid, (cid, ctx, pair, k_eff) in choices.items():
                reward = metrics.bandit_reward(fid, w, cap_per_mode)
                if reward is not None:
                    reward_trace[(fid, w)] = reward
                if reward is not None and stage in ("bandit", "joint") and fixed_cluster_id is None and not disable_updates:
                    self.bandit.update(pair, mode, cid, ctx, reward, k_eff)
                summary = metrics.get_window_summary(fid, w)
                last_context[fid] = self._make_context(mode, summary, cap_per_mode)

        ppo_loss = None
        if stage in ("pretrain", "joint") and not disable_updates:
            ppo_loss = self.ppo_agent.update()
        else:
            self.ppo_agent.completed_trajectories.clear()
        if disable_updates:
            self.ppo_agent.reset_rollout()

        overall = metrics.overall_metrics()
        measure_seconds = config.MEASURE_MS / 1000.0
        total_bits = 0.0
        for fid_stats in metrics.window_stats.values():
            for w, stats in fid_stats.items():
                if 0 <= w < num_windows:
                    total_bits += stats.delivered_bits
        goodput_mbps = (total_bits / measure_seconds) / 1e6 if measure_seconds > 0 else 0.0
        all_same_ratio = same_cluster_count / num_windows if num_windows > 0 else 0.0
        return EpisodeResult(
            mode=mode,
            stage=stage,
            delivery_ratio=overall["delivery_ratio"],
            p95_latency_ms=overall["p95_latency_ms"],
            reward_trace=reward_trace,
            ppo_loss=ppo_loss,
            cluster_hist=cluster_choice_hist,
            all_same_ratio=all_same_ratio,
            fallback_counts=metrics.fallback_counts,
            overall_metrics=overall,
            goodput_mbps=goodput_mbps,
        )
