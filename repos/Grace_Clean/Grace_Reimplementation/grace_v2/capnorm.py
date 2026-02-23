"""
Cap-normalization utilities for bandit rewards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from . import config


def _closest_entry(target_pps: int, aggregated: list):
    candidates = [row for row in aggregated if "pps_per_flow" in row]
    if not candidates:
        return None
    candidates.sort(key=lambda r: abs(r["pps_per_flow"] - target_pps))
    return candidates[0]


def load_cap_profile(summary_path: Path | str = None) -> Dict[str, Dict[str, float]]:
    path = Path(summary_path) if summary_path else config.cap_summary_path()
    if not path.exists():
        # Fallback constants if calibration is missing.
        print(f"[CapNorm] Missing {path}, using conservative defaults.")
        return {
            "light": {"L_cap": 100.0, "T_cap_bits": 1e6, "P_cap": 0.01},
            "medium": {"L_cap": 150.0, "T_cap_bits": 1e6, "P_cap": 0.02},
            "heavy": {"L_cap": 250.0, "T_cap_bits": 8e5, "P_cap": 0.05},
            "burst": {"L_cap": 350.0, "T_cap_bits": 7e5, "P_cap": 0.1},
        }

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    aggregated = payload.get("aggregated", [])

    profile = {}
    for mode, pps in config.MODES_PPS.items():
        entry = _closest_entry(pps, aggregated)
        if not entry:
            continue

        # Prefer ECMP numbers when available
        if isinstance(entry, dict) and "strategy" in entry:
            # Already closest; if both strategies exist for same pps, pick ECMP
            same_pps = [r for r in aggregated if r["pps_per_flow"] == entry["pps_per_flow"]]
            ecmp = [r for r in same_pps if r.get("strategy") == "ecmp"]
            if ecmp:
                entry = ecmp[0]

        L_cap = float(entry.get("p95_latency_ms") or 1.0)
        throughput_pps = float(entry.get("throughput_pps") or 0.0)
        T_cap_bits = throughput_pps * config.PACKET_SIZE_BYTES * 8.0 * (config.WINDOW_MS / 1000.0)
        loss_rate = max(1.0 - float(entry.get("delivery_ratio", 1.0)), 1e-3)

        profile[mode] = {"L_cap": L_cap, "T_cap_bits": T_cap_bits, "P_cap": loss_rate}

    return profile

