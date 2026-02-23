"""
Contextual bandit (LinUCB) with per-mode buckets.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from . import config


class LinUCBSolver:
    def __init__(self, n_arms: int, n_features: int, alpha: float):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def select_action(self, context: np.ndarray) -> Tuple[int, List[float]]:
        ctx = context.reshape(-1, 1)
        scores = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mean = (theta.T @ ctx).item()
            ci = self.alpha * np.sqrt(ctx.T @ A_inv @ ctx).item()

            scores.append(mean + ci)
        best = int(np.argmax(scores))
        return best, scores

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        ctx = context.reshape(-1, 1)
        self.A[arm] += ctx @ ctx.T
        self.b[arm] += reward * ctx


class PairModeLinUCB:
    """
    Manages LinUCB instances keyed by (src,dst,mode) to avoid arm semantics collision
    across different flow pairs. Each solver is sized by that pair's effective
    cluster count (k_eff).
    """

    def __init__(self, n_features: int = config.CONTEXT_DIM, alpha: float = config.LINUCB_ALPHA):
        self.n_features = n_features
        self.alpha = alpha
        # {(src,dst): {mode: LinUCBSolver}}
        self.solvers: Dict[Tuple[int, int], Dict[str, LinUCBSolver]] = {}

    def _get_solver(self, pair: Tuple[int, int], mode: str, k_eff: int) -> LinUCBSolver:
        if k_eff <= 0:
            k_eff = 1
        pair_map = self.solvers.setdefault(pair, {})
        solver = pair_map.get(mode)
        # Re-create solver if missing or arm size changed
        if solver is None or solver.n_arms != k_eff:
            solver = LinUCBSolver(k_eff, self.n_features, self.alpha)
            pair_map[mode] = solver
        return solver

    def select(self, pair: Tuple[int, int], mode: str, context: List[float], k_eff: int) -> Tuple[int, List[float]]:
        solver = self._get_solver(pair, mode, k_eff)
        ctx = np.array(context, dtype=float)
        return solver.select_action(ctx)

    def update(self, pair: Tuple[int, int], mode: str, arm: int, context: List[float], reward: float, k_eff: int) -> None:
        solver = self._get_solver(pair, mode, k_eff)
        solver.update(arm, np.array(context, dtype=float), reward)
