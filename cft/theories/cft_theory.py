"""Consensus-Fracture Theory (CFT) implementation.

Groups form when all pairwise affinities within exceed a consensus threshold.
Uses greedy clique-based clustering for tractable group discovery.
"""

from typing import List, Dict, Any, Optional

import numpy as np

from .base import BehaviorTheory, Agent, Group, TheoryParameters
from ..affinity import compute_affinity_matrix


class CFT(BehaviorTheory):
    """Consensus-Fracture Theory: threshold-based binary group formation.

    Agents form groups where every pair has affinity >= threshold.
    Groups are stable, boundaries are sharp, convergence is fast.
    """

    # Above this threshold, use exact solver (Bron-Kerbosch clique cover).
    # Below it, use greedy. Exact is O(3^(n/3)) worst case.
    EXACT_SOLVER_LIMIT = 15

    def __init__(
        self,
        params: TheoryParameters,
        threshold: float = 0.6,
        affinity_matrix: Optional[np.ndarray] = None,
        affinity_metric: str = "euclidean",
        solver: str = "auto",
    ):
        super().__init__(params, affinity_matrix=affinity_matrix)
        self.threshold = threshold
        self.affinity_matrix = affinity_matrix
        self.affinity_metric = affinity_metric
        self.solver = solver  # "auto", "greedy", or "exact"
        self.groups: List[Group] = []
        self._groups_formed = False

    def initialize_agents(self, agents: List[Agent]):
        self.agents = agents
        self._groups_formed = False
        if self.affinity_matrix is None:
            self.affinity_matrix = compute_affinity_matrix(
                agents, metric=self.affinity_metric, n_features=self.params.n_features
            )

    def step(self, dt: float):
        # CFT reaches equilibrium quickly - form groups on first step
        if not self._groups_formed:
            n = len(self.agents)
            use_exact = (
                self.solver == "exact"
                or (self.solver == "auto" and n <= self.EXACT_SOLVER_LIMIT)
            )
            if use_exact:
                self._form_groups_exact()
            else:
                self._form_groups_greedy()
            self._groups_formed = True
        self.current_time += dt

    def _form_groups_greedy(self):
        """Greedy group formation: find cliques meeting threshold.

        Order-dependent - processes agents by index. Fast (O(n²)) but may
        not find the minimum number of groups.
        """
        n = len(self.agents)
        assigned = [False] * n
        self.groups = []
        group_id = 0

        for i in range(n):
            if assigned[i]:
                continue

            group_members = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue

                # Check if j is compatible with all current members
                compatible = all(
                    self.affinity_matrix[j, m] >= self.threshold for m in group_members
                )

                if compatible:
                    group_members.append(j)
                    assigned[j] = True

            self.groups.append(Group(id=group_id, members=group_members))
            group_id += 1

    def _form_groups_exact(self):
        """Exact group formation: minimum clique cover of the threshold graph.

        Finds all maximal cliques via Bron-Kerbosch, then selects the minimum
        set of cliques that covers all agents (minimum clique cover = minimum
        coloring of the complement graph). Uses greedy set-cover heuristic on
        the maximal cliques, which is optimal for small n.

        Only practical for n <= ~15 due to exponential worst-case clique count.
        """
        n = len(self.agents)

        # Build adjacency: edge (i,j) exists iff affinity >= threshold
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.affinity_matrix[i, j] >= self.threshold:
                    adj[i][j] = True
                    adj[j][i] = True

        # Bron-Kerbosch with pivoting to find all maximal cliques
        maximal_cliques = []

        def bron_kerbosch(r: set, p: set, x: set):
            if not p and not x:
                maximal_cliques.append(frozenset(r))
                return
            # Pivot: choose vertex in P ∪ X with most neighbors in P
            pivot = max(p | x, key=lambda v: sum(1 for u in p if adj[v][u]))
            for v in list(p - {u for u in range(n) if adj[pivot][u]}):
                neighbors_v = {u for u in range(n) if adj[v][u]}
                bron_kerbosch(r | {v}, p & neighbors_v, x & neighbors_v)
                p.remove(v)
                x.add(v)

        bron_kerbosch(set(), set(range(n)), set())

        # Greedy set cover: pick largest uncovered clique repeatedly
        uncovered = set(range(n))
        self.groups = []
        group_id = 0

        # Sort cliques by size descending for greedy
        maximal_cliques.sort(key=len, reverse=True)

        while uncovered:
            # Find clique covering most uncovered agents
            best_clique = None
            best_cover = 0
            for clique in maximal_cliques:
                cover = len(clique & uncovered)
                if cover > best_cover:
                    best_cover = cover
                    best_clique = clique

            if best_clique is None:
                # Remaining agents are singletons (no edges)
                for agent in uncovered:
                    self.groups.append(Group(id=group_id, members=[agent]))
                    group_id += 1
                break

            covered = best_clique & uncovered
            self.groups.append(Group(id=group_id, members=sorted(covered)))
            uncovered -= covered
            group_id += 1

    def get_groups(self) -> List[Group]:
        return self.groups

    def get_state(self) -> Dict[str, Any]:
        return {
            "affinity_matrix": self.affinity_matrix,
            "threshold": self.threshold,
            "n_groups": len(self.groups),
        }
