"""Information Cascade Theory (ICT) implementation.

Group formation driven by information flow and processing limits.
Groups grow when mutual information exceeds maintenance cost,
and split when communication overhead becomes too high.
"""

from typing import List, Dict, Any, Optional

import numpy as np

from .base import BehaviorTheory, Agent, Group, TheoryParameters
from ..affinity import compute_affinity_matrix


class ICT(BehaviorTheory):
    """Information Cascade Theory: information-driven group formation.

    Agents share knowledge through bandwidth-limited communication.
    Groups form around shared information, growing through cascades
    and splitting when too large for effective communication.
    """

    def __init__(
        self,
        params: TheoryParameters,
        bandwidth: int = 4,
        info_decay: float = 0.05,
        maintenance_cost: float = 0.1,
        cascade_threshold: float = 0.3,
        share_rate: float = 0.1,
        min_cascade_neighbors: int = 2,
        forced_move_threshold: float = 0.3,
        affinity_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__(params, affinity_matrix=affinity_matrix)
        self.bandwidth = bandwidth  # Max connections per agent per step
        self.info_decay = info_decay  # Rate of knowledge loss
        self.maintenance_cost = maintenance_cost  # Per-member group cost
        self.cascade_threshold = cascade_threshold  # Min similarity to count as cascade neighbor
        self.share_rate = share_rate  # Knowledge blending rate per communication
        self.min_cascade_neighbors = min_cascade_neighbors  # Neighbors needed to trigger cascade
        self.forced_move_threshold = forced_move_threshold  # Info gain to move without cascade

        self.knowledge: Optional[np.ndarray] = None  # (n_agents, n_features)
        self.affinity_matrix: Optional[np.ndarray] = None
        self.group_assignments: Optional[np.ndarray] = None  # Agent -> group ID
        self._next_group_id: int = 0
        self._rng = np.random.default_rng(params.random_seed)

    def initialize_agents(self, agents: List[Agent]):
        self.agents = agents
        n = len(agents)

        # Each agent starts knowing their own features
        self.knowledge = np.array([a.features.copy() for a in agents], dtype=float)

        # Everyone starts unassigned (each in their own group)
        self.group_assignments = np.arange(n)
        self._next_group_id = n

        # Compute affinity matrix if not provided
        if self._affinity_matrix is not None:
            self.affinity_matrix = self._affinity_matrix
        else:
            self.affinity_matrix = compute_affinity_matrix(
                agents, metric="euclidean", n_features=self.params.n_features
            )

    def step(self, dt: float):
        """One step: communicate, update knowledge, form/split groups."""
        n = len(self.agents)

        # 1. Communication round: each agent talks to top-bandwidth partners
        for i in range(n):
            # Rank others by affinity, pick top bandwidth
            affinities = self.affinity_matrix[i].copy()
            affinities[i] = -np.inf  # exclude self
            partners = np.argsort(affinities)[-self.bandwidth :]

            for j in partners:
                if affinities[j] <= 0:
                    continue
                # Share knowledge: blend toward partner's knowledge
                share_rate = self.share_rate * affinities[j] * dt
                self.knowledge[i] += share_rate * (self.knowledge[j] - self.knowledge[i])

        # 2. Knowledge decay
        self.knowledge *= 1.0 - self.info_decay * dt

        # 3. Compute knowledge similarity (cosine) as mutual information proxy
        norms = np.linalg.norm(self.knowledge, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-10)
        normalized = self.knowledge / safe_norms
        knowledge_similarity = normalized @ normalized.T
        np.fill_diagonal(knowledge_similarity, 0.0)

        # 4. Group formation via cascade dynamics
        self._update_groups(knowledge_similarity)

        self.current_time += dt

    def _update_groups(self, similarity: np.ndarray):
        """Update group assignments based on knowledge similarity."""
        n = len(self.agents)

        # Snapshot current groups (read-only during move calculation)
        groups = {}
        for i in range(n):
            gid = self.group_assignments[i]
            groups.setdefault(gid, []).append(i)

        # Collect proposed moves first, then apply (avoids mutating during iteration)
        moves = []  # (agent_idx, old_gid, new_gid)

        for i in range(n):
            current_group = self.group_assignments[i]

            best_group = current_group
            best_info_gain = 0

            for gid, members in groups.items():
                if gid == current_group:
                    continue

                # Info gain = avg similarity to group members - maintenance cost
                sims = [similarity[i, m] for m in members if m != i]
                if not sims:
                    continue
                avg_sim = np.mean(sims)
                info_gain = avg_sim - self.maintenance_cost * len(members)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_group = gid

            # Cascade check: if enough of my current neighbors are in the target group
            if best_group != current_group and best_info_gain > 0:
                target_members = groups[best_group]
                neighbor_count = sum(
                    1 for m in target_members if similarity[i, m] > self.cascade_threshold
                )

                if neighbor_count >= self.min_cascade_neighbors or best_info_gain > self.forced_move_threshold:
                    moves.append((i, current_group, best_group))

        # Apply moves
        for i, old_gid, new_gid in moves:
            self.group_assignments[i] = new_gid

        # Rebuild groups after moves
        groups = {}
        for i in range(n):
            gid = self.group_assignments[i]
            groups.setdefault(gid, []).append(i)

        # Split oversized groups using similarity-based ordering
        max_viable_size = self.bandwidth * 3
        for gid, members in list(groups.items()):
            if len(members) > max_viable_size:
                # Sort members by avg internal similarity for coherent splits
                member_sims = []
                for m in members:
                    avg_s = np.mean([similarity[m, o] for o in members if o != m]) if len(members) > 1 else 0.0
                    member_sims.append((avg_s, m))
                member_sims.sort(reverse=True)
                sorted_members = [m for _, m in member_sims]

                for chunk_start in range(0, len(sorted_members), max_viable_size):
                    chunk = sorted_members[chunk_start : chunk_start + max_viable_size]
                    new_gid = self._next_group_id
                    self._next_group_id += 1
                    for m in chunk:
                        self.group_assignments[m] = new_gid

    def get_groups(self) -> List[Group]:
        groups = {}
        for i, gid in enumerate(self.group_assignments):
            groups.setdefault(int(gid), []).append(i)

        return [
            Group(id=gid, members=members) for gid, members in sorted(groups.items())
        ]

    def get_state(self) -> Dict[str, Any]:
        groups = self.get_groups()
        return {
            "knowledge": self.knowledge.copy(),
            "group_assignments": self.group_assignments.copy(),
            "n_groups": len(groups),
            "avg_group_size": np.mean([len(g.members) for g in groups]) if groups else 0,
            "bandwidth": self.bandwidth,
        }
