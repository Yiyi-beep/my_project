"""
Global configuration for the v2 hierarchical intelligent routing stack.

All time values are expressed in milliseconds to stay consistent with the
`ns.py` simulator, where port rate is configured in bit/ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# --- Topology / traffic defaults ---
PACKET_SIZE_BYTES = 1500
PORT_RATE_BIT_PER_MS = 10_000.0  # 10 Mbps
PROP_DELAY_MS = 10.0
BUFFER_BYTES = 512 * 1024
# Port in ns.py uses packet-based limit; 512KB / 1500B ≈ 349 packets.
BUFFER_PACKETS = 349
TTL_HOPS = 32

# Default flow pairs for training/eval; override here when switching scenarios.
# Current combo chosen to create ECMP≠SP divergence.
FLOW_PAIRS: List[Tuple[int, int]] = [(0, 4), (1, 4), (3, 12)]

MODES_PPS: Dict[str, int] = {
    "light": 380,
    "medium": 420,
    "heavy": 460,
    "burst": 560,
}

# --- Timing (ms) ---
WARMUP_MS = 500.0
MEASURE_MS = 3000.0
DRAIN_MS = 1000.0
WINDOW_MS = 100.0  # 0.1s decision window for the bandit


# --- Clustering ---
K_PATHS = 20
N_CLUSTERS = 6

# --- Bandit ---
LINUCB_ALPHA = 1.0  # exploration strength
CONTEXT_DIM = 7  # 4 (mode one-hot) + 3 (avg latency, delivery ratio, goodput)

# --- PPO ---
MAX_PORTS = 4  # NSFNET node degree upper bound
PER_PORT_FEATURES = 4  # queue_ratio, utilization, dist_norm, delta_dist_norm
GLOBAL_FEATURES = 4 + 1  # mode one-hot + ttl_remaining
STATE_DIM = MAX_PORTS * PER_PORT_FEATURES + GLOBAL_FEATURES
ACTIONS = MAX_PORTS
PPO_CFG = {
    "lr": 3e-4,
    "gamma": 0.7,
    "eps_clip": 0.2,
    "k_epochs": 4,
}

# --- Rewards ---
REWARD_WEIGHTS = {"latency": 0.5, "goodput": 0.4, "loss": 0.1}
PPO_STEP_WEIGHTS = {"queue": 0.6, "util": 0.4, "delta": 0.2}
TERMINAL_REWARD = {"delivered": 1.0, "dropped": -1.0}

# --- Training schedule ---
CURRICULUM = [
    ("stage0_pretrain", {"light": 1.0}),
    ("stage1_mix", {"light": 0.7, "medium": 0.3}),
    ("stage2_heavy", {"light": 0.4, "medium": 0.4, "heavy": 0.2}),
    # burst：把 burst 权重拉高
    ("stage3_full", {"light": 0.15, "medium": 0.20, "heavy": 0.15, "burst": 0.50}),
]

# Number of episodes per stage; can be tuned by the runner.
STAGE_EPISODES = {
    "stage0_pretrain": 60,
    "stage1_mix": 120,
    "stage2_heavy": 200,
    "stage3_full": 200,
}


def episode_total_ms() -> float:
    return WARMUP_MS + MEASURE_MS + DRAIN_MS


def cap_summary_path() -> Path:
    """Default path to the baseline calibration summary."""
    return Path("calibration_fixed_summary.json")
