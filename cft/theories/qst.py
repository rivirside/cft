"""Quantum Social Theory (QST) implementation.

Agents exist in superposition of behavioral states until "measured"
through interaction. Uses mean-field approximation for tractability.

Quantum-Inspired Approximations
================================
This module borrows mathematical machinery from quantum mechanics to model
social uncertainty and correlated decision-making. Key correspondences and
deliberate simplifications:

1. **Mean-field approximation**: True quantum systems of n particles require a
   joint state in a 2^n-dimensional Hilbert space. We instead give each agent
   its own n_states-dimensional state vector (amplitudes). This is the standard
   mean-field / product-state approximation from condensed matter physics. It
   makes the model O(n * n_states) in memory instead of O(n_states^n), but it
   cannot represent genuine multi-party entanglement.

2. **Entanglement as classical correlation**: Real quantum entanglement creates
   non-local correlations that violate Bell inequalities. Our "entanglement
   matrix" is a classical proxy: a pairwise correlation strength ∈ [0,1] that
   determines the probability of correlated collapse during measurement. This
   is closer to classical Bayesian correlation than true quantum entanglement,
   but it captures the key behavioral prediction: highly-interacting agents
   tend to "decide" together.

3. **Hamiltonian evolution**: In real QM, state evolution follows the
   Schrödinger equation iℏ ∂|ψ⟩/∂t = H|ψ⟩ with a Hermitian Hamiltonian.
   We use a simplified linear influence rule: ψ_i += dt * Σ_j α_ij * ψ_j,
   followed by renormalization. This is a first-order Euler step of a
   dissipative (non-unitary) evolution - it converges to consensus rather
   than oscillating. The 0.1 prefactor controls the coupling strength.

4. **Measurement as Born-rule sampling**: The probability of collapsing to
   state k is P(k) = |a_k|², matching the Born rule. Post-measurement, the
   state collapses to a basis vector - this is standard projective measurement.
   The extension to correlated collapse (entangled partners collapse together)
   is our social-theory addition.

5. **Decoherence**: Real decoherence arises from uncontrolled environmental
   interaction and drives density matrices toward diagonal form. We model it
   as additive complex Gaussian noise on amplitudes, with scale ∝ √n (larger
   groups are noisier). This is a phenomenological choice, not derived from
   a Lindblad master equation.
"""

from typing import List, Dict, Any, Optional

import numpy as np

from .base import BehaviorTheory, Agent, Group, TheoryParameters
from ..affinity import compute_affinity_matrix


class QST(BehaviorTheory):
    """Quantum Social Theory: probabilistic group membership with measurement effects.

    Each agent has a state vector of complex amplitudes across possible group states.
    Entanglement correlates agents through interaction. Measurement collapses
    superposition into definite group assignments.

    Uses mean-field approximation (per-agent state vectors) rather than full
    2^n joint quantum state, keeping it tractable for n > 20.
    """

    def __init__(
        self,
        params: TheoryParameters,
        n_states: int = 5,
        measurement_strength: float = 0.8,
        entanglement_rate: float = 0.1,
        decoherence_rate: float = 0.01,
        affinity_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__(params, affinity_matrix=affinity_matrix)
        self.n_states = n_states
        self.measurement_strength = measurement_strength
        self.entanglement_rate = entanglement_rate
        self.decoherence_rate = decoherence_rate

        self.amplitudes: Optional[np.ndarray] = None  # (n_agents, n_states) complex
        self.entanglement_matrix: Optional[np.ndarray] = None  # (n_agents, n_agents)
        self.affinity_matrix: Optional[np.ndarray] = None
        self._measured_groups: Optional[List[Group]] = None
        self._rng = np.random.default_rng(params.random_seed)

    def initialize_agents(self, agents: List[Agent]):
        self.agents = agents
        n = len(agents)

        # Initialize amplitudes: each agent in superposition biased by features
        self.amplitudes = np.zeros((n, self.n_states), dtype=complex)
        for i, agent in enumerate(agents):
            # Use features to bias initial state
            raw = np.abs(agent.features[: self.n_states]) if len(agent.features) >= self.n_states else np.concatenate([np.abs(agent.features), np.ones(self.n_states - len(agent.features))])
            raw = raw + 0.1  # avoid zeros
            probs = raw / raw.sum()
            self.amplitudes[i] = np.sqrt(probs).astype(complex)

        # Normalize
        self._normalize_all()

        # Initialize entanglement matrix
        self.entanglement_matrix = np.zeros((n, n))

        # Compute affinity matrix if not provided
        if self._affinity_matrix is not None:
            self.affinity_matrix = self._affinity_matrix
        else:
            self.affinity_matrix = compute_affinity_matrix(
                agents, metric="euclidean", n_features=self.params.n_features
            )

    def _normalize_all(self):
        """Ensure all state vectors are normalized."""
        norms = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2, axis=1, keepdims=True))
        norms = np.maximum(norms, 1e-10)
        self.amplitudes /= norms

    def step(self, dt: float):
        """Evolve quantum states: Hamiltonian evolution + entanglement + decoherence.

        Approximation notes:
        - Step 1 is a first-order Euler integration of a dissipative (non-unitary)
          Hamiltonian. Unlike true Schrödinger evolution, states converge rather
          than oscillate. See module docstring, point 3.
        - Step 2 uses classical pairwise correlation as an entanglement proxy.
          See module docstring, point 2.
        - Step 3 uses phenomenological noise rather than Lindblad dynamics.
          See module docstring, point 5.
        """
        n = len(self.agents)

        # 1. Hamiltonian evolution (vectorized)
        # Analogy: ψ_i += dt * Σ_j α_ij * ψ_j (dissipative, not unitary)
        # influence[i] = sum_j affinity[i,j] * amplitudes[j] (excluding self)
        affinity_no_self = self.affinity_matrix.copy()
        np.fill_diagonal(affinity_no_self, 0.0)
        influence = affinity_no_self @ self.amplitudes  # (n, n) @ (n, n_states)
        self.amplitudes = self.amplitudes + dt * 0.1 * influence
        self._normalize_all()

        # 2. Entanglement update (vectorized)
        # Classical proxy: E_ij grows with affinity * state overlap * dt
        # overlap[i,j] = |⟨ψ_i|ψ_j⟩| (inner product of per-agent states)
        overlap = np.abs(np.conj(self.amplitudes) @ self.amplitudes.T)
        positive_affinity = np.maximum(self.affinity_matrix, 0.0)
        delta = self.entanglement_rate * positive_affinity * overlap * dt
        new_entanglement = np.minimum(1.0, self.entanglement_matrix + delta)
        # Enforce symmetry and zero diagonal
        new_entanglement = np.triu(new_entanglement, k=1)
        new_entanglement = new_entanglement + new_entanglement.T
        self.entanglement_matrix = new_entanglement

        # 3. Decoherence: additive complex Gaussian noise, scale ∝ √n
        # Phenomenological - not derived from a Lindblad master equation
        noise_scale = self.decoherence_rate * np.sqrt(n) * dt
        noise_real = self._rng.normal(0, noise_scale / np.sqrt(2), self.amplitudes.shape)
        noise_imag = self._rng.normal(0, noise_scale / np.sqrt(2), self.amplitudes.shape)
        self.amplitudes += noise_real + 1j * noise_imag
        self._normalize_all()

        self._measured_groups = None  # invalidate cached measurement
        self.current_time += dt

    def measure(self, agent_ids: Optional[List[int]] = None) -> Dict[int, int]:
        """Collapse agent states through measurement (Born rule + correlated collapse).

        Sampling follows the Born rule: P(state=k) = |a_k|². This is the one
        piece that is genuinely quantum-faithful. Post-measurement collapse to
        a basis vector is standard projective measurement.

        Correlated collapse for entangled agents is our social extension:
        if E_ij > random(), agent j collapses to the same state as agent i.
        This models the empirical observation that tightly-coupled individuals
        tend to converge on the same group decision.

        Returns mapping of agent_id -> group_state.
        """
        n = len(self.agents)
        if agent_ids is None:
            agent_ids = list(range(n))

        assignments = {}
        measured = set()

        for idx in agent_ids:
            if idx in measured:
                continue

            # Sample from probability distribution
            probs = np.abs(self.amplitudes[idx]) ** 2
            probs = probs / probs.sum()  # renormalize for safety
            state = self._rng.choice(self.n_states, p=probs)
            assignments[idx] = state
            measured.add(idx)

            # Collapse: set amplitude to basis state
            self.amplitudes[idx] = np.zeros(self.n_states, dtype=complex)
            self.amplitudes[idx, state] = 1.0

            # Correlated collapse for entangled agents
            for j in range(n):
                if j in measured or j == idx:
                    continue
                if self.entanglement_matrix[idx, j] > self._rng.random():
                    # Entangled agent collapses to same state
                    assignments[j] = state
                    measured.add(j)
                    self.amplitudes[j] = np.zeros(self.n_states, dtype=complex)
                    self.amplitudes[j, state] = 1.0

        return assignments

    def get_groups(self) -> List[Group]:
        """Return group configuration using expectation (most probable state per agent)."""
        n = len(self.agents)
        probs = np.abs(self.amplitudes) ** 2
        assignments = np.argmax(probs, axis=1)

        state_to_members = {}
        for i in range(n):
            state = int(assignments[i])
            state_to_members.setdefault(state, []).append(i)

        return [
            Group(id=state, members=members)
            for state, members in sorted(state_to_members.items())
        ]

    def measure_groups(self) -> List[Group]:
        """Return group configuration by collapsing superposition (stochastic).

        Unlike get_groups(), this mutates state - amplitudes collapse to basis states.
        Entangled agents are correlated in their collapse.
        """
        assignments = self.measure()
        state_to_members = {}
        for agent_idx, state in assignments.items():
            state_to_members.setdefault(state, []).append(agent_idx)

        return [
            Group(id=state, members=members)
            for state, members in sorted(state_to_members.items())
        ]

    def get_state(self) -> Dict[str, Any]:
        probs = np.abs(self.amplitudes) ** 2
        return {
            "amplitudes": self.amplitudes.copy(),
            "probabilities": probs,
            "entanglement_matrix": self.entanglement_matrix.copy(),
            "n_states": self.n_states,
            "avg_entanglement": np.mean(
                self.entanglement_matrix[np.triu_indices(len(self.agents), k=1)]
            )
            if len(self.agents) > 1
            else 0.0,
        }
