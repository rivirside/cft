"""Thermodynamic Social Theory (TST) implementation.

Social systems minimize free energy like physical systems. Uses the Potts model
(generalization of Ising) with Metropolis-Hastings Monte Carlo sampling.
"""

from typing import List, Dict, Any, Optional

import numpy as np

from .base import BehaviorTheory, Agent, Group, TheoryParameters
from ..affinity import compute_affinity_matrix


class TST(BehaviorTheory):
    """Thermodynamic Social Theory: statistical mechanics of group formation.

    Agents are modeled as spins in a Potts model. Group assignments evolve via
    Metropolis-Hastings Monte Carlo, minimizing free energy F = E - TS.
    Temperature controls the disorder: high T = fluid groups, low T = frozen.
    """

    def __init__(
        self,
        params: TheoryParameters,
        temperature: float = 1.0,
        n_groups_max: int = 5,
        cooling_rate: float = 0.0,
        sweeps_per_step: int = 10,
        affinity_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__(params, affinity_matrix=affinity_matrix)
        self.temperature = temperature
        self.n_groups_max = n_groups_max
        self.cooling_rate = cooling_rate  # 0 = no annealing
        self.sweeps_per_step = sweeps_per_step  # MC sweeps per time step

        self.spins: Optional[np.ndarray] = None  # Group assignments
        self.coupling_matrix: Optional[np.ndarray] = None  # J_ij
        self._rng = np.random.default_rng(params.random_seed)
        self._energy_history: List[float] = []

    def initialize_agents(self, agents: List[Agent]):
        self.agents = agents
        n = len(agents)

        # Random initial group assignments
        self.spins = self._rng.integers(0, self.n_groups_max, size=n)

        # Coupling matrix from affinity
        if self._affinity_matrix is not None:
            self.coupling_matrix = self._affinity_matrix.copy()
        else:
            self.coupling_matrix = compute_affinity_matrix(
                agents, metric="euclidean", n_features=self.params.n_features
            )

    def step(self, dt: float):
        """Metropolis-Hastings Monte Carlo step."""
        n = len(self.agents)

        for _ in range(self.sweeps_per_step):
            # One sweep: try flipping each agent
            order = self._rng.permutation(n)
            for i in order:
                old_spin = self.spins[i]
                new_spin = self._rng.integers(0, self.n_groups_max)
                if new_spin == old_spin:
                    continue

                # Compute energy change (vectorized over j)
                couplings = self.coupling_matrix[i]
                old_same = (self.spins == old_spin).astype(float)
                new_same = (self.spins == new_spin).astype(float)
                old_same[i] = 0.0  # exclude self
                new_same[i] = 0.0
                delta_e = -np.dot(couplings, new_same - old_same)

                # Metropolis acceptance
                if delta_e <= 0:
                    self.spins[i] = new_spin
                elif self.temperature > 0:
                    if self._rng.random() < np.exp(-delta_e / self.temperature):
                        self.spins[i] = new_spin

        # Optional cooling
        if self.cooling_rate > 0:
            self.temperature = max(0.01, self.temperature * (1.0 - self.cooling_rate * dt))

        self._energy_history.append(self._compute_energy())
        self.current_time += dt

    def _compute_energy(self) -> float:
        """Compute total Potts model energy (vectorized)."""
        # same_spin[i,j] = 1 if spins[i] == spins[j], else 0
        same_spin = (self.spins[:, np.newaxis] == self.spins[np.newaxis, :]).astype(float)
        # Upper triangle only to avoid double-counting
        upper = np.triu(same_spin, k=1)
        return -np.sum(self.coupling_matrix * upper)

    def _compute_entropy(self) -> float:
        """Compute configurational entropy from group size distribution."""
        n = len(self.agents)
        _, counts = np.unique(self.spins, return_counts=True)
        probs = counts / n
        return -np.sum(probs * np.log(probs + 1e-10))

    def get_groups(self) -> List[Group]:
        groups = {}
        for i, spin in enumerate(self.spins):
            groups.setdefault(int(spin), []).append(i)

        return [
            Group(id=gid, members=members) for gid, members in sorted(groups.items())
        ]

    def get_state(self) -> Dict[str, Any]:
        energy = self._compute_energy()
        entropy = self._compute_entropy()
        return {
            "spins": self.spins.copy(),
            "temperature": self.temperature,
            "energy": energy,
            "entropy": entropy,
            "free_energy": energy - self.temperature * entropy,
            "n_groups": len(np.unique(self.spins)),
            "energy_history": list(self._energy_history),
        }
