"""Gradient Field Theory (GFT) implementation.

Agents move in behavioral space following affinity gradients,
forming groups through continuous attraction dynamics.
"""

from typing import List, Dict, Any, Optional

import numpy as np
from scipy.spatial.distance import cdist

from .base import BehaviorTheory, Agent, Group, TheoryParameters


class GFT(BehaviorTheory):
    """Gradient Field Theory: continuous dynamics in behavioral space.

    Agents experience attractive forces toward similar others and drift
    toward equilibrium positions. Groups emerge as spatial clusters.
    """

    def __init__(
        self,
        params: TheoryParameters,
        k: float = 0.1,
        sigma: float = 1.0,
        affinity_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__(params, affinity_matrix=affinity_matrix)
        self.k = k  # Attraction strength
        self.sigma = sigma  # Interaction range
        self.positions: Optional[np.ndarray] = None

    def initialize_agents(self, agents: List[Agent]):
        self.agents = agents
        self.positions = np.array([a.features.copy() for a in agents], dtype=float)

    def step(self, dt: float):
        """Move agents along gradient field (vectorized)."""
        # diffs[i,j] = positions[j] - positions[i], shape (n, n, d)
        diffs = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        # dists[i,j] = ||positions[j] - positions[i]||, shape (n, n)
        dists = np.linalg.norm(diffs, axis=2)

        # Avoid division by zero on diagonal
        safe_dists = np.where(dists > 0, dists, 1.0)

        # Gaussian force magnitudes: k * exp(-d^2 / (2*sigma^2))
        force_mags = self.k * np.exp(-(dists ** 2) / (2 * self.sigma ** 2))
        # Zero out self-interaction
        np.fill_diagonal(force_mags, 0.0)

        # Unit direction vectors scaled by force magnitude
        # forces[i] = sum_j force_mag[i,j] * (pos[j] - pos[i]) / |pos[j] - pos[i]|
        forces = np.sum(
            (force_mags / safe_dists)[:, :, np.newaxis] * diffs, axis=1
        )

        self.positions += forces * dt
        self.current_time += dt

    def get_groups(self) -> List[Group]:
        """Cluster agents based on position proximity."""
        groups = []
        n = len(self.agents)
        assigned = [False] * n
        group_id = 0

        for i in range(n):
            if assigned[i]:
                continue

            group_members = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue

                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.sigma:
                    group_members.append(j)
                    assigned[j] = True

            groups.append(Group(id=group_id, members=group_members))
            group_id += 1

        return groups

    def get_state(self) -> Dict[str, Any]:
        return {
            "positions": self.positions.copy(),
            "k": self.k,
            "sigma": self.sigma,
        }
