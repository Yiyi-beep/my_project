"""
Masked PPO agent for per-hop routing.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden = 128
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state: np.ndarray, mask: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
        probs = self.actor(state_t)
        masked = probs * mask_t
        total = masked.sum()
        if total.item() <= 0:
            masked = torch.ones_like(masked) / float(len(masked))
        else:
            masked = masked / total
        dist = torch.distributions.Categorical(masked)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions, masks):
        probs = self.actor(states)
        masked = probs * masks
        totals = masked.sum(dim=1, keepdim=True)
        # If a row mask is all zero, fall back to uniform to avoid NaN entropy/logprob.
        zero_mask = (totals <= 1e-8).squeeze(1)
        if zero_mask.any():
            # set uniform on zero rows
            masked[zero_mask] = 1.0 / masked.shape[1]
            totals = masked.sum(dim=1, keepdim=True)
        masked = masked / (totals + 1e-8)
        dist = torch.distributions.Categorical(masked)
        action_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states)
        return action_logprobs, values, entropy


class PPOAgent:
    def __init__(
        self,
        state_dim: int = config.STATE_DIM,
        action_dim: int = config.ACTIONS,
        lr: float = config.PPO_CFG["lr"],
        gamma: float = config.PPO_CFG["gamma"],
        K_epochs: int = config.PPO_CFG["k_epochs"],
        eps_clip: float = config.PPO_CFG["eps_clip"],
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse = nn.MSELoss()

        self.active_trajectories: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        self.completed_trajectories: List[List[dict]] = []

    def select_action(self, state: np.ndarray, mask: np.ndarray):
        with torch.no_grad():
            action, logprob = self.policy_old.act(state, mask)
        return action, logprob.item()

    def store_transition(
        self,
        traj_id: Tuple[int, int],
        state: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        mask: np.ndarray,
    ):
        self.active_trajectories[traj_id].append(
            {"s": state, "a": action, "lp": logprob, "r": reward, "d": done, "mask": mask}
        )
        if done:
            self.completed_trajectories.append(self.active_trajectories.pop(traj_id, []))

    def finalize_trajectory(self, traj_id: Tuple[int, int], terminal_reward: float):
        traj = self.active_trajectories.get(traj_id)
        if not traj:
            self.active_trajectories.pop(traj_id, None)
            return
        traj[-1]["r"] += terminal_reward
        traj[-1]["d"] = True
        self.completed_trajectories.append(self.active_trajectories.pop(traj_id, []))

    def update(self):
        MIN_TRAJ = 50
        if len(self.completed_trajectories) < MIN_TRAJ:
            return None

        # Flatten trajectories
        states = []
        actions = []
        old_logprobs = []
        returns = []
        masks = []

        for traj in self.completed_trajectories:
            R = 0.0
            for step in reversed(traj):
                R = step["r"] + self.gamma * R
                states.insert(0, step["s"])
                actions.insert(0, step["a"])
                old_logprobs.insert(0, step["lp"])
                returns.insert(0, R)
                masks.insert(0, step["mask"])

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_logprobs_t = torch.tensor(old_logprobs, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        masks_t = torch.tensor(np.array(masks), dtype=torch.float32, device=device)

        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)

        total_loss = 0.0
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_kl = 0.0
        for _ in range(self.K_epochs):
            logprobs, values, entropy = self.policy.evaluate(states_t, actions_t, masks_t)
            values = values.squeeze()
            advantages = returns_t - values.detach()
            ratios = torch.exp(logprobs - old_logprobs_t)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.5 * self.mse(values, returns_t)
            loss = policy_loss + value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            total_loss += loss.mean().item()
            last_policy_loss = policy_loss.mean().item()
            last_value_loss = value_loss.mean().item()
            last_entropy = entropy.mean().item()
            with torch.no_grad():
                last_kl = (old_logprobs_t - logprobs).mean().item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.completed_trajectories = []
        # avoid memory leak if something stuck
        if len(self.active_trajectories) > 10000:
            self.active_trajectories.clear()

        return {
            "loss": total_loss / self.K_epochs,
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "approx_kl": last_kl,
        }

    def reset_rollout(self):
        """Clear rollout buffers (useful for eval to avoid accumulation)."""
        self.active_trajectories.clear()
        self.completed_trajectories = []

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        self.policy.load_state_dict(torch.load(path, map_location=device))
        self.policy_old.load_state_dict(self.policy.state_dict())
