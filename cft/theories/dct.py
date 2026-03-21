"""Dual-Context Theory (DCT) implementation.

Each agent exists simultaneously in two coupled spaces:

- **Context space** (proximity): who you're socially near, co-located with,
  or share a communication channel with. Evolves fast.
- **Alignment space**: your values, opinions, personality. Evolves slowly.

The two spaces couple bidirectionally:

- Alignment drives *seeking*: you move toward people you agree with (G_a -> G_p).
- Proximity drives *conformity*: being near someone shifts your alignment (G_p -> G_a).

Groups form only where both layers agree. High alignment but no proximity
means no group (online friends you never meet). High proximity but no
alignment means forced coexistence (coworkers you disagree with) - tension
that resolves when the constraint lifts.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .base import BehaviorTheory, Agent, Group, TheoryParameters
from ..affinity import compute_affinity_matrix


# ---------------------------------------------------------------------------
# Trait maps: derive per-agent mu/lam from feature vectors
# ---------------------------------------------------------------------------

class TraitMap:
    """Map agent features to behavioral parameters (mu, lam).

    A trait map is a function ``(agents) -> (mu_array, lam_array)`` that
    derives per-agent seeking and conformity rates from their feature
    vectors and metadata.

    Built-in presets are available via :meth:`from_preset`.
    """

    def __init__(
        self,
        mu_fn: Callable[[List[Agent]], np.ndarray],
        lam_fn: Callable[[List[Agent]], np.ndarray],
    ):
        self.mu_fn = mu_fn
        self.lam_fn = lam_fn

    def compute(self, agents: List[Agent]) -> tuple:
        """Return (mu_array, lam_array) for the given agents."""
        return self.mu_fn(agents), self.lam_fn(agents)

    @classmethod
    def from_indices(
        cls,
        mu_index: int,
        lam_index: int,
        mu_scale: float = 1.0,
        lam_scale: float = 0.5,
        mu_baseline: float = 0.1,
        lam_baseline: float = 0.01,
    ) -> "TraitMap":
        """Create a trait map from feature vector indices.

        mu[i] = baseline + scale * sigmoid(features[mu_index])
        lam[i] = baseline + scale * sigmoid(features[lam_index])

        For the default SocialSimulator feature layout [4 MBTI, n opinions, 1 influence]:
        - Index 0 = E/I dimension (extraversion). High -> high seeking.
        - Index 3 = J/P dimension (perceiving). High -> low rigidity -> high conformity.
        - Index -1 = influence. High influence -> high seeking, low conformity.
        """
        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

        def mu_fn(agents):
            vals = np.array([a.features[mu_index] for a in agents], dtype=float)
            return mu_baseline + mu_scale * _sigmoid(vals)

        def lam_fn(agents):
            vals = np.array([a.features[lam_index] for a in agents], dtype=float)
            return lam_baseline + lam_scale * _sigmoid(vals)

        return cls(mu_fn=mu_fn, lam_fn=lam_fn)

    @classmethod
    def from_metadata(
        cls,
        mu_key: str,
        lam_key: str,
        mu_default: float = 0.3,
        lam_default: float = 0.05,
    ) -> "TraitMap":
        """Create a trait map from agent metadata fields."""
        def mu_fn(agents):
            return np.array(
                [a.metadata.get(mu_key, mu_default) for a in agents], dtype=float
            )

        def lam_fn(agents):
            return np.array(
                [a.metadata.get(lam_key, lam_default) for a in agents], dtype=float
            )

        return cls(mu_fn=mu_fn, lam_fn=lam_fn)

    @classmethod
    def from_preset(cls, name: str) -> "TraitMap":
        """Built-in presets for common personality models.

        Presets:
        - ``"mbti"``: E/I drives seeking (extraverts seek), J/P drives
          conformity (perceivers conform more). Uses SocialSimulator's
          default feature layout [E/I, S/N, T/F, J/P, opinions..., influence].
        - ``"influence"``: influence score drives seeking (high-influence
          agents actively seek), inverse drives conformity (low-influence
          agents conform more).
        """
        if name == "mbti":
            # E/I = index 0, J/P = index 3
            return cls.from_indices(mu_index=0, lam_index=3,
                                    mu_scale=0.6, lam_scale=0.3,
                                    mu_baseline=0.1, lam_baseline=0.01)
        elif name == "influence":
            # influence = last feature
            def mu_fn(agents):
                vals = np.array([a.features[-1] for a in agents], dtype=float)
                return 0.1 + 0.8 * np.clip(vals, 0, 1)

            def lam_fn(agents):
                vals = np.array([a.features[-1] for a in agents], dtype=float)
                return 0.01 + 0.4 * (1.0 - np.clip(vals, 0, 1))

            return cls(mu_fn=mu_fn, lam_fn=lam_fn)
        else:
            raise ValueError(f"Unknown trait map preset: {name!r}. Use: mbti, influence")


class DCT(BehaviorTheory):
    """Dual-Context Theory: groups require both proximity and alignment.

    Parameters
    ----------
    params : TheoryParameters
        Shared simulation parameters.
    mu : float, array-like, or None
        Seeking rate. Scalar broadcasts to all agents, array sets per-agent.
        Ignored if ``trait_map`` is provided (trait_map derives mu from features).
    lam : float, array-like, or None
        Conformity rate. Same broadcasting rules as mu.
        Ignored if ``trait_map`` is provided.
    trait_map : TraitMap, str, or None
        Derives per-agent mu and lam from agent features/metadata.
        Pass a TraitMap instance, or a preset name ("mbti", "influence").
        When set, the mu/lam constructor args are ignored.
    noise : float
        Context noise - random encounter intensity each step.
    threshold : float
        Group formation threshold on effective affinity.
    context_scale : float
        Gaussian kernel width for proximity affinity.
    affinity_matrix : ndarray, optional
        Pre-computed affinity matrix for initial alignment.
    proximity_matrix : ndarray, optional
        Pre-computed proximity matrix (n x n). When provided, used as the
        initial proximity affinity instead of deriving from alignment + noise.
        Allows proximity and alignment to come from different data sources.
    alignment_features : ndarray, optional
        Explicit (n x d) alignment vectors. When provided, used instead of
        agent.features for the alignment layer. This separates the alignment
        representation from the general feature vector.
    """

    def __init__(
        self,
        params: TheoryParameters,
        mu: Any = 0.3,
        lam: Any = 0.05,
        trait_map: Union[TraitMap, str, None] = None,
        noise: float = 0.1,
        threshold: float = 0.3,
        context_scale: float = 1.5,
        affinity_matrix: Optional[np.ndarray] = None,
        proximity_matrix: Optional[np.ndarray] = None,
        alignment_features: Optional[np.ndarray] = None,
    ):
        super().__init__(params, affinity_matrix=affinity_matrix)
        self._mu_init = mu
        self._lam_init = lam
        if isinstance(trait_map, str):
            trait_map = TraitMap.from_preset(trait_map)
        self._trait_map = trait_map
        self.noise = noise
        self.threshold = threshold
        self.context_scale = context_scale
        self._proximity_matrix_init = proximity_matrix
        self._alignment_features_init = alignment_features

        # Per-agent vectors; built in initialize_agents once n is known.
        self.mu: Optional[np.ndarray] = None
        self.lam: Optional[np.ndarray] = None

        self.context_pos: Optional[np.ndarray] = None
        self.alignment: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(params.random_seed)

        self._proximity_affinity: Optional[np.ndarray] = None
        self._alignment_affinity: Optional[np.ndarray] = None
        self._effective_affinity: Optional[np.ndarray] = None

    @staticmethod
    def _broadcast_param(value, n: int) -> np.ndarray:
        """Convert scalar or array-like to a per-agent vector of length n."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr))
        if len(arr) != n:
            raise ValueError(
                f"Per-agent parameter has length {len(arr)}, expected {n}"
            )
        return arr.copy()

    def initialize_agents(self, agents: List[Agent]):
        self.agents = agents
        n = len(agents)

        # --- Derive per-agent mu/lam ---
        if self._trait_map is not None:
            self.mu, self.lam = self._trait_map.compute(agents)
        else:
            self.mu = self._broadcast_param(self._mu_init, n)
            self.lam = self._broadcast_param(self._lam_init, n)

        # --- Build alignment layer ---
        if self._alignment_features_init is not None:
            self.alignment = np.array(self._alignment_features_init, dtype=float).copy()
        elif self._affinity_matrix is not None:
            # Derive alignment vectors from pre-computed affinity via spectral embedding.
            d = min(self.params.n_features, n - 1) if self.params.n_features else min(3, n - 1)
            eigvals, eigvecs = np.linalg.eigh(self._affinity_matrix)
            self.alignment = eigvecs[:, -d:].copy() * np.sqrt(np.maximum(eigvals[-d:], 0.0))
        else:
            self.alignment = np.array([a.features.copy() for a in agents], dtype=float)

        d = self.alignment.shape[1]

        # --- Build context/proximity layer ---
        if self._proximity_matrix_init is not None:
            # Derive context positions from proximity via spectral embedding,
            # so agents who are proximate start near each other.
            prox = np.array(self._proximity_matrix_init, dtype=float)
            p_eigvals, p_eigvecs = np.linalg.eigh(prox)
            self.context_pos = p_eigvecs[:, -d:].copy() * np.sqrt(
                np.maximum(p_eigvals[-d:], 0.0)
            )
        else:
            # Default: context starts as alignment + noise
            self.context_pos = self.alignment.copy() + self._rng.normal(0, self.noise * 2, (n, d))

        self._update_affinities()

    def _update_affinities(self):
        """Recompute proximity, alignment, and effective affinity matrices."""
        # Proximity: Gaussian kernel on context-space distance
        diffs = self.context_pos[:, np.newaxis, :] - self.context_pos[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        self._proximity_affinity = np.exp(-dists_sq / (2.0 * self.context_scale ** 2))
        np.fill_diagonal(self._proximity_affinity, 1.0)

        # Alignment: cosine similarity
        norms = np.linalg.norm(self.alignment, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = self.alignment / norms
        self._alignment_affinity = np.clip(normalized @ normalized.T, -1.0, 1.0)
        np.fill_diagonal(self._alignment_affinity, 1.0)

        # Effective affinity: product of proximity and rescaled alignment.
        # Rescale alignment from [-1, 1] to [0, 1] so the product works as a gate:
        # you need BOTH proximity AND alignment for a high effective affinity.
        a_gate = (self._alignment_affinity + 1.0) / 2.0
        self._effective_affinity = self._proximity_affinity * a_gate
        np.fill_diagonal(self._effective_affinity, 1.0)

    def step(self, dt: float):
        n = len(self.agents)
        d = self.context_pos.shape[1]

        # --- Layer 1: proximity update (fast timescale) ---
        # Seeking: agents move toward others they align with.
        # mu is per-agent: mu[i] controls how strongly agent i seeks.
        ctx_diffs = self.context_pos[np.newaxis, :, :] - self.context_pos[:, np.newaxis, :]
        ctx_dists = np.linalg.norm(ctx_diffs, axis=2)
        safe_dists = np.where(ctx_dists > 0, ctx_dists, 1.0)

        # Weight by per-agent seeking rate and positive alignment affinity
        seek_weights = self.mu[:, np.newaxis] * np.maximum(self._alignment_affinity, 0.0)
        np.fill_diagonal(seek_weights, 0.0)

        forces = np.sum(
            (seek_weights / safe_dists)[:, :, np.newaxis] * ctx_diffs, axis=1
        )

        # Random encounter noise
        noise_vec = self._rng.normal(0, self.noise, (n, d))
        self.context_pos += (forces + noise_vec) * dt

        # --- Layer 2: alignment update (slow timescale) ---
        # Conformity: alignment drifts toward proximate others' alignment.
        # lam is per-agent: lam[i] controls how much agent i conforms.
        align_diffs = self.alignment[np.newaxis, :, :] - self.alignment[:, np.newaxis, :]

        conform_weights = self.lam[:, np.newaxis] * self._proximity_affinity
        np.fill_diagonal(conform_weights, 0.0)

        drift = np.sum(conform_weights[:, :, np.newaxis] * align_diffs, axis=1)
        self.alignment += drift * dt

        # --- Recompute affinities for next step / group detection ---
        self._update_affinities()
        self.current_time += dt

    def get_groups(self) -> List[Group]:
        """Greedy clique cover on effective affinity (same algorithm as CFT)."""
        n = len(self.agents)
        assigned = [False] * n
        groups = []
        group_id = 0

        for i in range(n):
            if assigned[i]:
                continue

            members = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                compatible = all(
                    self._effective_affinity[j, m] >= self.threshold for m in members
                )
                if compatible:
                    members.append(j)
                    assigned[j] = True

            groups.append(Group(id=group_id, members=members))
            group_id += 1

        return groups

    def get_state(self) -> Dict[str, Any]:
        # Layer tension: mean absolute difference between proximity and
        # rescaled alignment. High tension = the two layers disagree about
        # who should be grouped with whom.
        a_gate = (self._alignment_affinity + 1.0) / 2.0
        n = len(self.agents)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        tension = float(np.mean(np.abs(self._proximity_affinity[mask] - a_gate[mask])))

        return {
            "context_positions": self.context_pos.copy(),
            "alignment": self.alignment.copy(),
            "proximity_affinity": self._proximity_affinity.copy(),
            "alignment_affinity": self._alignment_affinity.copy(),
            "effective_affinity": self._effective_affinity.copy(),
            "tension": tension,
            "mu": self.mu.copy(),
            "lam": self.lam.copy(),
            "threshold": self.threshold,
            "n_groups": len(self.get_groups()),
        }
