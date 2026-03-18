"""Synthetic social simulator: generate MiroFish-format data without external dependencies.

Produces agent profiles (MBTI, opinions, influence) and timestamped interaction logs
(follows, likes, reposts, comments) driven entirely by numpy. The output is consumable
by MiroFishAdapter without modification.

Basic usage::

    from cft.simulator import SocialSimulator

    sim = SocialSimulator(n_agents=40, scenario="clustered", k=3, T=30, seed=42)
    adapter = sim.to_adapter()
    agents = adapter.load_agents()
    adapter.load_interactions()
    affinity = adapter.compute_affinity_matrix()
    communities = adapter.extract_ground_truth_groups()
"""

import copy
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# MBTI types and their dimension vectors (reused from mirofish._MBTI_MAP)
_MBTI_MAP: Dict[str, float] = {
    "E": 1.0, "I": -1.0,
    "S": 1.0, "N": -1.0,
    "T": 1.0, "F": -1.0,
    "J": 1.0, "P": -1.0,
}

MBTI_TYPES: List[str] = [
    "ENFP", "ENFJ", "ENTP", "ENTJ",
    "ESFP", "ESFJ", "ESTP", "ESTJ",
    "INFP", "INFJ", "INTP", "INTJ",
    "ISFP", "ISFJ", "ISTP", "ISTJ",
]

# Positive action types and their sampling weights
_POS_ACTIONS = ["follow", "like", "repost", "pos_comment"]
_POS_WEIGHTS = np.array([0.35, 0.30, 0.25, 0.10])

# Timestamp step size
_STEP_MINUTES = 5


def _mbti_vec(mbti: str) -> np.ndarray:
    """Convert 4-letter MBTI type to feature vector in {-1, 1}^4."""
    return np.array([_MBTI_MAP[c] for c in mbti.upper()], dtype=float)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_arr(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SocialSimulator:
    """Generate synthetic MiroFish-format social data.

    Agents are characterized by MBTI personality type, opinion vector, and influence
    score. Interaction probability between agents is driven by affinity (MBTI + opinion
    similarity), enabling controllable clustering structure.

    Parameters
    ----------
    n_agents : int
        Number of agents (default 50).
    n_opinions : int
        Dimension of each agent's opinion vector (default 3).
    scenario : str
        One of ``"random"``, ``"clustered"``, ``"polarized"``, ``"hierarchical"``.
    T : int
        Number of simulation timesteps (default 20).
    interaction_rate : float
        Expected interactions per agent per timestep (default 2.0).
    alpha : float
        Blend weight for MBTI similarity vs. opinion similarity (0=opinions only,
        1=MBTI only, default 0.5).
    beta : float
        Sigmoid sharpness; higher values → stronger clustering (default 2.0).
    gamma : float
        Influence amplification factor (default 0.5).
    pos_threshold : float
        Raw affinity above which positive actions are used (default 0.2).
    neg_threshold : float
        Raw affinity below which neg_comment is used (default -0.2).
    seed : int or None
        NumPy RNG seed for reproducibility.
    **scenario_kwargs
        Scenario-specific parameters:

        - ``clustered``: ``k`` (int, groups, default 3), ``cluster_purity`` (float,
          default 0.8), ``opinion_noise`` (float, default 0.3)
        - ``polarized``: ``bias_strength`` (float, default 0.8),
          ``camp_sizes`` (tuple[int,int] | None, default equal split)
        - ``hierarchical``: ``n_influencers`` (int, default 3)
    """

    def __init__(
        self,
        n_agents: int = 50,
        n_opinions: int = 3,
        scenario: str = "random",
        T: int = 20,
        interaction_rate: float = 2.0,
        alpha: float = 0.5,
        beta: float = 2.0,
        gamma: float = 0.5,
        pos_threshold: float = 0.2,
        neg_threshold: float = -0.2,
        seed: Optional[int] = None,
        **scenario_kwargs: Any,
    ):
        valid_scenarios = ("random", "clustered", "polarized", "hierarchical")
        if scenario not in valid_scenarios:
            raise ValueError(f"Unknown scenario {scenario!r}. Choose from {valid_scenarios}.")

        self.n_agents = n_agents
        self.n_opinions = n_opinions
        self.scenario = scenario
        self.T = T
        self.interaction_rate = interaction_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.seed = seed
        self.scenario_kwargs = scenario_kwargs

        self._rng: Optional[np.random.Generator] = None
        self._agent_dicts: Optional[List[Dict]] = None
        self._interaction_dicts: Optional[List[Dict]] = None
        self._affinity_matrix: Optional[np.ndarray] = None
        self._tmpdir: Optional[str] = None
        self._generated = False

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self) -> "SocialSimulator":
        """Run the simulation. Idempotent: call again to regenerate. Returns self."""
        self._rng = np.random.default_rng(self.seed)

        self._agent_dicts = self._build_agents()
        self._affinity_matrix = self._build_affinity_matrix(self._agent_dicts)
        self._interaction_dicts = self._generate_interactions(
            self._agent_dicts, self._affinity_matrix
        )
        self._generated = True
        return self

    def to_adapter(self):
        """Write JSONL files to a managed temp directory and return a MiroFishAdapter.

        The temp directory is cleaned up by calling ``cleanup()``.

        Returns
        -------
        MiroFishAdapter
        """
        from .integrations.mirofish import MiroFishAdapter

        if not self._generated:
            self.generate()

        if self._tmpdir is None or not Path(self._tmpdir).exists():
            self._tmpdir = tempfile.mkdtemp(prefix="cft_sim_")

        self._write_jsonl(Path(self._tmpdir))
        return MiroFishAdapter(self._tmpdir)

    def write_to_dir(self, path: Union[str, Path]):
        """Write profiles.jsonl and actions.jsonl to *path* and return a MiroFishAdapter.

        Parameters
        ----------
        path : str or Path
            Destination directory (must exist).

        Returns
        -------
        MiroFishAdapter
        """
        from .integrations.mirofish import MiroFishAdapter

        if not self._generated:
            self.generate()

        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(dest)
        return MiroFishAdapter(str(dest))

    def to_dataframes(self):
        """Return ``(agents, interactions_df)`` directly without writing to disk.

        Requires pandas (``pip install cft[mirofish]``).

        Returns
        -------
        tuple[List[Agent], pandas.DataFrame]
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframes(). "
                "Install with: pip install cft[mirofish]"
            )

        from .integrations.mirofish import MiroFishAdapter

        if not self._generated:
            self.generate()

        # Use adapter's parser on the dicts directly by writing to a temp adapter
        adapter = self.to_adapter()
        agents = adapter.load_agents()
        df = pd.DataFrame(self._interaction_dicts)
        return agents, df

    def cleanup(self) -> None:
        """Remove the managed temp directory created by ``to_adapter()``."""
        if self._tmpdir and Path(self._tmpdir).exists():
            shutil.rmtree(self._tmpdir)
            self._tmpdir = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    @property
    def n_features(self) -> int:
        """Feature dimension per agent: 4 MBTI + n_opinions + 1 influence."""
        return 4 + self.n_opinions + 1

    # ── Private: JSONL I/O ───────────────────────────────────────────────────

    def _write_jsonl(self, dest: Path) -> None:
        with open(dest / "profiles.jsonl", "w") as f:
            for agent in self._agent_dicts:
                f.write(json.dumps(agent) + "\n")
        with open(dest / "actions.jsonl", "w") as f:
            for action in self._interaction_dicts:
                f.write(json.dumps(action) + "\n")

    # ── Private: Simulation Core ─────────────────────────────────────────────

    def _build_agents(self) -> List[Dict]:
        """Dispatch to scenario factory."""
        dispatch = {
            "random": self._scenario_random,
            "clustered": self._scenario_clustered,
            "polarized": self._scenario_polarized,
            "hierarchical": self._scenario_hierarchical,
        }
        return dispatch[self.scenario]()

    def _build_affinity_matrix(self, agent_dicts: List[Dict]) -> np.ndarray:
        """Compute raw (n, n) affinity from MBTI + opinion similarity."""
        n = len(agent_dicts)
        mbti_vecs = np.array([_mbti_vec(a["mbti"]) for a in agent_dicts])
        opinion_vecs = np.array([a["opinions"] for a in agent_dicts], dtype=float)

        # MBTI similarity: dot product / 4  → [-1, 1]
        mbti_sim = (mbti_vecs @ mbti_vecs.T) / 4.0

        # Opinion similarity: cosine  → [-1, 1]
        norms = np.linalg.norm(opinion_vecs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = opinion_vecs / norms
        opinion_sim = normalized @ normalized.T

        raw = self.alpha * mbti_sim + (1.0 - self.alpha) * opinion_sim
        np.fill_diagonal(raw, 1.0)
        return raw

    def _generate_interactions(
        self,
        agent_dicts: List[Dict],
        affinity: np.ndarray,
    ) -> List[Dict]:
        """Generate interaction records over T timesteps."""
        n = len(agent_dicts)
        influences = np.array([a["influence"] for a in agent_dicts])
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Interaction probability matrix: p_interact * influence amplification
        p_interact = _sigmoid_arr(self.beta * affinity)
        np.fill_diagonal(p_interact, 0.0)
        # Scale by interaction_rate / (n - 1)
        p_ij = p_interact * (1.0 + self.gamma * influences[:, np.newaxis])
        p_ij *= self.interaction_rate / max(n - 1, 1)
        # Clamp to valid probability range
        p_ij = np.clip(p_ij, 0.0, 1.0)
        np.fill_diagonal(p_ij, 0.0)

        interactions: List[Dict] = []

        for t in range(self.T):
            ts = (base_time + timedelta(minutes=t * _STEP_MINUTES)).isoformat()

            # Sample interactions: for each ordered pair (i, j), Bernoulli(p_ij[i,j])
            draws = self._rng.random((n, n))
            active = draws < p_ij  # shape (n, n), active[i,j] = i initiates action toward j
            np.fill_diagonal(active, False)

            initiators, targets = np.where(active)
            for i, j in zip(initiators, targets):
                aff = affinity[i, j]
                if aff > self.pos_threshold:
                    # Positive: weighted sample from follow/like/repost/pos_comment
                    action = self._rng.choice(_POS_ACTIONS, p=_POS_WEIGHTS / _POS_WEIGHTS.sum())
                elif aff < self.neg_threshold:
                    action = "neg_comment"
                else:
                    action = "like"

                interactions.append({
                    "timestamp": ts,
                    "agent_i": int(i),
                    "agent_j": int(j),
                    "action": str(action),
                })

        return interactions

    # ── Scenario Factories ────────────────────────────────────────────────────

    def _make_agent(
        self,
        agent_id: int,
        mbti: str,
        opinions: List[float],
        influence: float,
    ) -> Dict:
        return {
            "id": agent_id,
            "mbti": mbti.upper(),
            "opinions": [float(x) for x in opinions],
            "influence": float(np.clip(influence, 0.0, 1.0)),
        }

    def _scenario_random(self) -> List[Dict]:
        """Uniform random agents: no structure."""
        agents = []
        for i in range(self.n_agents):
            mbti = MBTI_TYPES[self._rng.integers(0, len(MBTI_TYPES))]
            opinions = self._rng.standard_normal(self.n_opinions).tolist()
            influence = float(self._rng.uniform(0.1, 1.0))
            agents.append(self._make_agent(i, mbti, opinions, influence))
        return agents

    def _scenario_clustered(self) -> List[Dict]:
        """k clusters with configurable purity and opinion spread."""
        k = self.scenario_kwargs.get("k", 3)
        cluster_purity = self.scenario_kwargs.get("cluster_purity", 0.8)
        opinion_noise = self.scenario_kwargs.get("opinion_noise", 0.3)

        # One dominant MBTI per cluster
        cluster_mbtis = [MBTI_TYPES[self._rng.integers(0, len(MBTI_TYPES))] for _ in range(k)]
        # One opinion center per cluster (well-separated)
        centers = self._rng.standard_normal((k, self.n_opinions))
        # Normalize centers and scale to have distance ~4 apart
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        centers = centers / norms * 2.0

        agents = []
        for i in range(self.n_agents):
            cluster = i % k
            # MBTI: modal type with probability cluster_purity, else random
            if self._rng.random() < cluster_purity:
                mbti = cluster_mbtis[cluster]
            else:
                mbti = MBTI_TYPES[self._rng.integers(0, len(MBTI_TYPES))]
            # Opinion: near cluster center
            opinions = (centers[cluster] + self._rng.normal(0, opinion_noise, self.n_opinions)).tolist()
            influence = float(self._rng.uniform(0.1, 1.0))
            agents.append(self._make_agent(i, mbti, opinions, influence))
        return agents

    def _scenario_polarized(self) -> List[Dict]:
        """Two opposing camps with MBTI and opinion polarization."""
        bias_strength = self.scenario_kwargs.get("bias_strength", 0.8)
        camp_sizes = self.scenario_kwargs.get("camp_sizes", None)

        if camp_sizes is None:
            n_camp_a = self.n_agents // 2
            n_camp_b = self.n_agents - n_camp_a
        else:
            n_camp_a, n_camp_b = camp_sizes
            if n_camp_a + n_camp_b != self.n_agents:
                raise ValueError(
                    f"camp_sizes {camp_sizes} must sum to n_agents={self.n_agents}"
                )

        # Camp A: extrovert/sensing leaning (E_S__), camp B: introvert/intuiting leaning (I_N__)
        camp_a_archetype = "ESFJ"
        camp_b_archetype = "INTP"
        camp_a_opinion = np.ones(self.n_opinions)
        camp_b_opinion = -np.ones(self.n_opinions)

        agents = []
        for camp, n_camp, archetype, opinion_center in [
            (0, n_camp_a, camp_a_archetype, camp_a_opinion),
            (1, n_camp_b, camp_b_archetype, camp_b_opinion),
        ]:
            archetype_vec = _mbti_vec(archetype)
            for j in range(n_camp):
                agent_id = len(agents)
                # MBTI: mix of archetype and random
                if self._rng.random() < bias_strength:
                    mbti = archetype
                else:
                    mbti = MBTI_TYPES[self._rng.integers(0, len(MBTI_TYPES))]
                # Opinion: near camp center with noise scaled by (1 - bias_strength)
                noise_scale = 1.0 - bias_strength * 0.5
                opinions = (opinion_center + self._rng.normal(0, noise_scale, self.n_opinions)).tolist()
                influence = float(self._rng.uniform(0.1, 1.0))
                agents.append(self._make_agent(agent_id, mbti, opinions, influence))

        return agents

    def _scenario_hierarchical(self) -> List[Dict]:
        """n_influencers high-influence hubs with followers assigned to their nearest hub."""
        n_influencers = self.scenario_kwargs.get("n_influencers", 3)
        if n_influencers >= self.n_agents:
            n_influencers = max(1, self.n_agents // 2)

        # Influencers: high influence, random MBTI, spread opinion centers
        inf_opinions = self._rng.standard_normal((n_influencers, self.n_opinions))
        # Scale to be well-separated
        norms = np.linalg.norm(inf_opinions, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        inf_opinions = inf_opinions / norms * 2.0
        inf_mbtis = [MBTI_TYPES[self._rng.integers(0, len(MBTI_TYPES))] for _ in range(n_influencers)]

        agents = []
        # Add influencers
        for k in range(n_influencers):
            influence = float(self._rng.uniform(0.7, 1.0))
            agents.append(self._make_agent(k, inf_mbtis[k], inf_opinions[k].tolist(), influence))

        # Add followers: each gets assigned to nearest influencer by opinion
        for i in range(n_influencers, self.n_agents):
            # Random opinion close to a random influencer
            leader = self._rng.integers(0, n_influencers)
            opinions = (inf_opinions[leader] + self._rng.normal(0, 0.4, self.n_opinions)).tolist()
            # Followers tend to share MBTI with their leader
            if self._rng.random() < 0.6:
                mbti = inf_mbtis[leader]
            else:
                mbti = MBTI_TYPES[self._rng.integers(0, len(MBTI_TYPES))]
            influence = float(self._rng.uniform(0.1, 0.5))
            agents.append(self._make_agent(i, mbti, opinions, influence))

        return agents

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def iso_from_step(step: int, base: str = "2024-01-01T00:00:00") -> str:
        """Convert an integer timestep to an ISO 8601 timestamp string.

        Parameters
        ----------
        step : int
            Timestep index (0-based).
        base : str
            Base datetime string (default: "2024-01-01T00:00:00").

        Returns
        -------
        str
            ISO 8601 datetime string for the given step.
        """
        dt = datetime.fromisoformat(base) + timedelta(minutes=step * _STEP_MINUTES)
        return dt.isoformat()
