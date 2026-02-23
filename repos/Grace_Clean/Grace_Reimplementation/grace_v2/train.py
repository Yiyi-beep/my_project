"""
Training runner for the new hierarchical intelligent routing stack.

It keeps the v2 code isolated from the legacy project by using its own
package (`grace_v2`). The runner follows the three-stage schedule:
1) PPO pre-train with uniform cluster sampling
2) Bandit training with frozen PPO
3) Joint fine-tune with small PPO updates
"""

from __future__ import annotations

import csv
import os
import random
import sys
import datetime
from pathlib import Path
from typing import Dict

import numpy as np

# Allow running as a script (python grace_v2/train.py) by ensuring package import works.
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from grace_v2 import config
from grace_v2.bandit import PairModeLinUCB
from grace_v2.capnorm import load_cap_profile
from grace_v2.logging_utils import append_jsonl
from grace_v2.path_clustering import PathClusterManager
from grace_v2.ppo_agent import PPOAgent
from grace_v2.simulator import HierarchicalRoutingSimulator


def sample_mode(weights: Dict[str, float]) -> str:
    modes = list(weights.keys())
    probs = list(weights.values())
    return random.choices(modes, weights=probs, k=1)[0]


def stage_phase(stage_name: str) -> str:
    """Map curriculum stage to training phase."""
    if stage_name == "stage0_pretrain":
        return "pretrain"
    if stage_name in ("stage1_mix", "stage2_heavy"):
        return "bandit"
    return "joint"


def main():
    random.seed(42)
    np.random.seed(42)

    run_id = os.environ.get("RUN_ID") or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = Path("checkpoints_v2")
    ckpt_dir.mkdir(exist_ok=True)
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    cap_profile = load_cap_profile()
    cluster_mgr = PathClusterManager()
    cluster_mgr.build_all(config.FLOW_PAIRS)
    cluster_mgr.save(ckpt_dir / "clusters.json")

    bandit = PairModeLinUCB()
    ppo = PPOAgent()
    sim = HierarchicalRoutingSimulator(
        cluster_manager=cluster_mgr,
        bandit=bandit,
        ppo_agent=ppo,
        cap_profile=cap_profile,
    )

    csv_path = ckpt_dir / "training_metrics_v2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "stage",
                "phase",
                "mode",
                "delivery_ratio",
                "p95_latency_ms",
                "ppo_loss",
            ]
        )

    episode_idx = 0
    update_counter = 0
    for stage_name, mode_weights in config.CURRICULUM:
        phase = stage_phase(stage_name)
        episodes = config.STAGE_EPISODES.get(stage_name, 0)
        print(f"[Train] Stage {stage_name} ({phase}) for {episodes} episodes...")
        for i in range(episodes):
            mode = sample_mode(mode_weights)
            result = sim.run_episode(mode, phase, seed=episode_idx)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        episode_idx + 1,
                        stage_name,
                        phase,
                        mode,
                        f"{result.delivery_ratio:.4f}",
                        f"{result.p95_latency_ms:.2f}" if result.p95_latency_ms else "",
                        f"{result.ppo_loss['loss']:.4f}" if isinstance(result.ppo_loss, dict) else "",
                    ]
                )
            # episode-level JSONL log
            total_gen = result.overall_metrics.get("total_generated", 0)
            total_del = result.overall_metrics.get("total_delivered", 0)
            delivery_ratio = result.overall_metrics.get("delivery_ratio", 0.0)
            avg_lat = result.overall_metrics.get("avg_latency_ms")
            p50 = result.overall_metrics.get("p50_latency_ms")
            p95 = result.overall_metrics.get("p95_latency_ms")
            strict = result.fallback_counts.get("strict", 0)
            fb1 = result.fallback_counts.get("fallback1", 0)
            fb2 = result.fallback_counts.get("fallback2", 0)
            total_masks = strict + fb1 + fb2 if (strict + fb1 + fb2) > 0 else 1
            strict_ratio = strict / total_masks
            fallback2_ratio = fb2 / total_masks
            reward_vals = list(result.reward_trace.values())
            reward_total = float(np.mean(reward_vals)) if reward_vals else None
            cluster_top1 = {}
            for pair, hist in result.cluster_hist.items():
                total = sum(hist.values()) or 1
                top_cid, top_cnt = max(hist.items(), key=lambda kv: kv[1])
                cluster_top1[f"{pair[0]}-{pair[1]}"] = {"cid": top_cid, "frac": round(top_cnt / total, 3)}
            undelivered = max(total_gen - total_del, 0)
            record = {
                "run_id": run_id,
                "stage": stage_name,
                "episode_idx": episode_idx + 1,
                "seed": episode_idx,
                "mode": mode,
                "generated_pkts_total": total_gen,
                "delivered_pkts_total": total_del,
                "delivery_ratio_total": delivery_ratio,
                "avg_latency_ms_total": avg_lat,
                "p50_latency_ms_total": p50,
                "p95_latency_ms_total": p95,
                "goodput_mbps_total": result.goodput_mbps,
                "reward_total": reward_total,
                "cluster_top1": cluster_top1,
                "all_same_ratio": result.all_same_ratio,
                "strict_ratio": strict_ratio,
                "fallback2_ratio": fallback2_ratio,
                "undelivered_pkts_total": undelivered,
                "mask_counts_scope": "all_time",
            }
            append_jsonl(logs_dir / "episode.jsonl", record)
            episode_idx += 1

            # Save checkpoints at boundaries
            if (i + 1) == episodes:
                ppo.save(str(ckpt_dir / f"{stage_name}_ppo.pth"))
                # bandit uses numpy matrices -> save as pickle
                import pickle

                with open(ckpt_dir / f"{stage_name}_bandit.pkl", "wb") as fb:
                    pickle.dump(bandit, fb)

            # PPO update logging (if an update just happened)
            if result.ppo_loss and isinstance(result.ppo_loss, dict):
                update_stats = result.ppo_loss
                update_record = {
                    "run_id": run_id,
                    "stage": stage_name,
                    "episode_idx": episode_idx,
                    "update_idx": update_counter,
                    "policy_loss": update_stats.get("policy_loss"),
                    "value_loss": update_stats.get("value_loss"),
                    "entropy": update_stats.get("entropy"),
                    "approx_kl": update_stats.get("approx_kl"),
                }
                append_jsonl(logs_dir / "ppo_update.jsonl", update_record)
                update_counter += 1

    print(f"[Train] Finished. Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
