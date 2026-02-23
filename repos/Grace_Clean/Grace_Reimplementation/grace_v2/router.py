"""
Routing node that applies masked PPO decisions under cluster constraints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ns.port.port import Port

from . import config
from .ppo_agent import PPOAgent
from .path_clustering import PathClusterManager


class HierarchicalRouter:
    def __init__(
        self,
        env,
        node_id: int,
        neighbors: List[int],
        *,
        port_rate: float,
        buffer_packets: int,
        cluster_manager: PathClusterManager,
        ppo_agent: PPOAgent,
        mask_mode: str = "cluster",
        metrics=None,

    ):
        self.env = env
        self.node_id = node_id
        self.cluster_manager = cluster_manager
        self.ppo_agent = ppo_agent
        self.metrics = metrics
        self.mask_mode = mask_mode


        self.ports: List[Port] = []
        self.neighbor_map: Dict[int, int] = {}
        self.local_sink = None


        for idx, neigh in enumerate(sorted(neighbors)):
            port = Port(
                env,
                rate=port_rate,
                qlimit=buffer_packets,
                limit_bytes=False,
                element_id=f"{node_id}_{idx}",
            )
            self.ports.append(port)
            self.neighbor_map[idx] = neigh

    # --- Core helpers ---
    def _dist(self, a: int, b: int) -> float:
        return float(self.cluster_manager.dist_matrix.get(a, {}).get(b, config.MAX_PORTS))

    def _build_state_and_mask(self, packet) -> Tuple[np.ndarray, np.ndarray, List[int], str]:
        if self.mask_mode == "candidate":
          allowed = self.cluster_manager.get_candidate_next_hops(packet.src, packet.dst, self.node_id)
        else:
            # --- compat: older Packet may not have cluster_id ---
            if not hasattr(packet, "cluster_id"):
                # best-effort: if cluster_manager knows how to compute it, use that
                if hasattr(self.cluster_manager, "get_cluster_id"):
                    packet.cluster_id = self.cluster_manager.get_cluster_id(packet.src, packet.dst)
                elif hasattr(self.cluster_manager, "assign_cluster_id"):
                    packet.cluster_id = self.cluster_manager.assign_cluster_id(packet.src, packet.dst)
                else:
                    # fallback: single cluster
                    packet.cluster_id = 0
            # --- end compat ---
            allowed = self.cluster_manager.get_allowed_next_hops(
                packet.src, packet.dst, packet.cluster_id, self.node_id
            )
        allowed_set = set(allowed)

        current_dist = self._dist(self.node_id, packet.dst)
        mask = np.zeros(config.MAX_PORTS, dtype=np.float32)
        features: List[float] = []
        strategy = "strict"
        for idx in range(config.MAX_PORTS):
            if idx < len(self.ports):
                neighbor = self.neighbor_map[idx]
                # mask fallback 1: allow if in cluster mask
                if neighbor in allowed_set:
                    mask[idx] = 1.0
                # mask fallback 2: allow progress to shorter dist when cluster mask empty
            else:
                features.extend([1.0, 0.0, 1.0, 0.0])  # padding
                continue

            port = self.ports[idx]
            queue_ratio = min(len(port.store.items) / float(config.BUFFER_PACKETS), 1.0)
            util = 1.0 if port.busy else 0.0
            neighbor_dist = self._dist(neighbor, packet.dst)
            dist_norm = min(neighbor_dist / self.cluster_manager.network_diameter, 1.0)
            delta = max(neighbor_dist - current_dist, 0.0) / self.cluster_manager.network_diameter

            features.extend([queue_ratio, util, dist_norm, delta])

        # Fallbacks when strict mask empty
        if mask.sum() == 0 and len(self.ports) > 0:
            strategy = "fallback1"
            for idx in range(len(self.ports)):
                neighbor = self.neighbor_map[idx]
                if self._dist(neighbor, packet.dst) < current_dist:
                    mask[idx] = 1.0
        if mask.sum() == 0 and len(self.ports) > 0:
            strategy = "fallback2"
            for idx in range(len(self.ports)):
                neighbor = self.neighbor_map[idx]
                if getattr(packet, "prev_node", None) is not None and neighbor == packet.prev_node:
                    continue
                mask[idx] = 1.0

        # Global features (compat for older Packet)
        mode_one_hot = [0.0, 0.0, 0.0, 0.0]
        mode = getattr(packet, "mode", "light")
        # (optional) write back so downstream code can rely on it
        packet.mode = mode
        mode_idx = {"light": 0, "medium": 1, "heavy": 2, "burst": 3}.get(mode, 0)
        mode_one_hot[mode_idx] = 1.0

        ttl = getattr(packet, "ttl", config.TTL_HOPS)
        ttl_norm = max(ttl, 0) / float(config.TTL_HOPS)
        full_state = np.array(features + mode_one_hot + [ttl_norm], dtype=np.float32)
        return full_state, mask.astype(np.float32), allowed, strategy

    def _record_drop(self, packet, reward: float):
        traj_id = (packet.flow_id, packet.packet_id)
        self.ppo_agent.finalize_trajectory(traj_id, reward)
        if self.metrics:
            self.metrics.on_drop(packet, self.env.now)

    def put(self, packet):
        # Deliver locally
        if packet.dst == self.node_id:
            traj_id = (packet.flow_id, packet.packet_id)
            self.ppo_agent.finalize_trajectory(traj_id, config.TERMINAL_REWARD["delivered"])
            if self.metrics:
                self.metrics.on_delivery(packet, self.env.now)
            if self.local_sink:
                self.local_sink.put(packet)
            return

        # TTL guard
        if hasattr(packet, "ttl"):
            packet.ttl -= 1
            if packet.ttl <= 0:
                self._record_drop(packet, config.TERMINAL_REWARD["dropped"])
                return

        state, mask, allowed, strategy = self._build_state_and_mask(packet)
        if self.metrics:
            self.metrics.on_mask_strategy(strategy)

        action_idx, logprob = self.ppo_agent.select_action(state, mask)
        if action_idx >= len(self.ports):
            action_idx = len(self.ports) - 1
        target_port = self.ports[action_idx]
        next_hop = self.neighbor_map[action_idx]
        packet.prev_node = self.node_id

        # Compute step reward
        feature_offset = action_idx * config.PER_PORT_FEATURES
        queue_ratio = state[feature_offset]
        util = state[feature_offset + 1]
        delta = state[feature_offset + 3]
        step_reward = (
            -config.PPO_STEP_WEIGHTS["queue"] * queue_ratio
            -config.PPO_STEP_WEIGHTS["util"] * util
            -config.PPO_STEP_WEIGHTS["delta"] * max(0.0, delta)
        )

        # If link exists, forward; otherwise treat as drop
        if target_port.out:
            traj_id = (packet.flow_id, packet.packet_id)
            self.ppo_agent.store_transition(traj_id, state, action_idx, logprob, step_reward, False, mask)
            target_port.put(packet)
            packet.last_action_node = self.node_id
            packet.last_action = action_idx
        else:
            self._record_drop(packet, config.TERMINAL_REWARD["dropped"])
