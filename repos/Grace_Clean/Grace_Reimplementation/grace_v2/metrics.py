"""
Window-level metrics collection and bandit reward computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config


@dataclass
class WindowStats:
    generated: int = 0
    delivered: int = 0
    drops: int = 0
    delivered_bits: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)

    def summary(self) -> dict:
        delivery_ratio = self.delivered / self.generated if self.generated > 0 else 0.0
        loss_rate = (self.generated - self.delivered) / self.generated if self.generated > 0 else 0.0
        avg_latency = float(np.mean(self.latencies_ms)) if self.latencies_ms else None
        goodput_bits = self.delivered_bits  # bits delivered within the window
        return {
            "generated": self.generated,
            "delivered": self.delivered,
            "drops": self.drops,
            "delivery_ratio": delivery_ratio,
            "loss_rate": max(loss_rate, 0.0),
            "avg_latency_ms": avg_latency,
            "goodput_bits": goodput_bits,
        }


class MetricsRecorder:
    def __init__(self, window_ms: float):
        self.window_ms = window_ms
        self.window_stats: Dict[int, Dict[int, WindowStats]] = {}
        self.all_latencies: Dict[int, List[float]] = {}
        self.generated_total: Dict[int, int] = {}
        self.delivered_total: Dict[int, int] = {}
        self.fallback_counts = {"strict": 0, "fallback1": 0, "fallback2": 0}

    # --- Event hooks ---
    def on_generate(self, packet, now: float):
        if not getattr(packet, "generated_in_measure", False):
            return
        flow_stats = self.window_stats.setdefault(packet.flow_id, {})
        stats = flow_stats.setdefault(packet.window_id, WindowStats())
        stats.generated += 1
        self.generated_total[packet.flow_id] = self.generated_total.get(packet.flow_id, 0) + 1

    def on_delivery(self, packet, now: float):
        if getattr(packet, "generated_in_measure", False):
            self.delivered_total[packet.flow_id] = self.delivered_total.get(packet.flow_id, 0) + 1
            latency_ms = (now - packet.time)
            self.all_latencies.setdefault(packet.flow_id, []).append(latency_ms)
            flow_stats = self.window_stats.setdefault(packet.flow_id, {})
            stats = flow_stats.setdefault(packet.window_id, WindowStats())
            stats.delivered += 1
            stats.delivered_bits += float(packet.size) * 8.0
            stats.latencies_ms.append(latency_ms)

    def on_drop(self, packet, now: float):
        if getattr(packet, "generated_in_measure", False):
            flow_stats = self.window_stats.setdefault(packet.flow_id, {})
            stats = flow_stats.setdefault(packet.window_id, WindowStats())
            stats.drops += 1

    def on_mask_strategy(self, strategy: str):
        if strategy in self.fallback_counts:
            self.fallback_counts[strategy] += 1

    # --- Queries ---
    def get_window_summary(self, flow_id: int, window_id: int) -> Optional[dict]:
        flow_stats = self.window_stats.get(flow_id, {})
        stats = flow_stats.get(window_id)
        if not stats:
            return None
        return stats.summary()

    def bandit_reward(self, flow_id: int, window_id: int, cap_profile: dict) -> Optional[float]:
        summary = self.get_window_summary(flow_id, window_id)
        if summary is None:
            return None

        lat = summary["avg_latency_ms"]
        goodput_bits = summary["goodput_bits"]
        loss = summary["loss_rate"]

        # Cap profile expects per-mode entries
        caps = cap_profile
        f_L = 0.0
        if lat is not None:
            L_cap = caps.get("L_cap", None)
            if L_cap:
                f_L = np.clip(1.0 - lat / L_cap, 0.0, 1.0)

        T_cap = caps.get("T_cap_bits", None)
        f_T = np.clip(goodput_bits / T_cap, 0.0, 1.0) if T_cap else 0.0

        P_cap = max(caps.get("P_cap", 0.05), 1e-3)
        f_P = 1.0 - np.clip(loss / max(P_cap, 1e-6), 0.0, 1.0)

        reward = (
            config.REWARD_WEIGHTS["latency"] * f_L
            + config.REWARD_WEIGHTS["goodput"] * f_T
            + config.REWARD_WEIGHTS["loss"] * f_P
        )
        return float(reward)

    def window_goodput_bps(self, summary: dict) -> float:
        return summary["goodput_bits"] * (1000.0 / self.window_ms)

    def overall_metrics(self) -> dict:
        total_gen = sum(self.generated_total.values()) if self.generated_total else 0
        total_del = sum(self.delivered_total.values()) if self.delivered_total else 0
        delivery_ratio = total_del / total_gen if total_gen > 0 else 0.0
        latencies = [lat for lst in self.all_latencies.values() for lat in lst]
        avg_lat = float(np.mean(latencies)) if latencies else None
        p50 = float(np.percentile(latencies, 50)) if latencies else None
        p95 = float(np.percentile(latencies, 95)) if latencies else None
        return {
            "total_generated": total_gen,
            "total_delivered": total_del,
            "delivery_ratio": delivery_ratio,
            "avg_latency_ms": avg_lat,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "fallback_counts": self.fallback_counts.copy(),
        }
