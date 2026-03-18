"""Utilities for comparing predictions across theories.

Provides multiple comparison metrics:
- Jaccard similarity (agent-level)
- Normalized Mutual Information (NMI) for partition comparison
- Partition agreement score (fraction of agent pairs co-grouped identically)
- Theory-specific diagnostics (e.g., QST entanglement, TST temperature)
"""

from typing import List, Dict, Any, Optional
import math

import numpy as np

from .theories.base import Group, BehaviorTheory


class TheoryComparator:
    """Compare group structures and predictions across different theories."""

    # ── Partition comparison metrics ──────────────────────────────────────

    @staticmethod
    def compare_group_structures(
        groups1: List[Group],
        groups2: List[Group],
        metric: str = "jaccard",
    ) -> float:
        """Compare two group structures using the specified metric.

        Args:
            groups1: First partition.
            groups2: Second partition.
            metric: One of "jaccard", "nmi", "pair_agreement".

        Returns:
            Similarity score in [0, 1]. Higher = more similar.
        """
        if metric == "jaccard":
            return TheoryComparator._jaccard(groups1, groups2)
        elif metric == "nmi":
            return TheoryComparator._nmi(groups1, groups2)
        elif metric == "pair_agreement":
            return TheoryComparator._pair_agreement(groups1, groups2)
        else:
            raise ValueError(f"Unknown metric: {metric!r}. Use: jaccard, nmi, pair_agreement")

    @staticmethod
    def _jaccard(groups1: List[Group], groups2: List[Group]) -> float:
        """Agent-level Jaccard: for each agent, Jaccard of its group in each partition."""
        all_agents = set()
        for g in groups1 + groups2:
            all_agents.update(g.members)

        if not all_agents:
            return 0.0

        similarity_sum = 0
        for agent in all_agents:
            g1 = [g for g in groups1 if agent in g.members]
            g2 = [g for g in groups2 if agent in g.members]

            if g1 and g2:
                members1 = set(g1[0].members)
                members2 = set(g2[0].members)

                intersection = len(members1.intersection(members2))
                union = len(members1.union(members2))

                if union > 0:
                    similarity_sum += intersection / union

        return similarity_sum / len(all_agents)

    @staticmethod
    def _nmi(groups1: List[Group], groups2: List[Group]) -> float:
        """Normalized Mutual Information between two partitions.

        NMI = 2 * I(U;V) / (H(U) + H(V)), where U and V are the two
        label assignments. Returns 1.0 for identical partitions, 0.0 for
        independent ones.
        """
        all_agents = set()
        for g in groups1 + groups2:
            all_agents.update(g.members)
        n = len(all_agents)
        if n == 0:
            return 0.0

        # Build label vectors
        label1 = {}
        for gid, g in enumerate(groups1):
            for m in g.members:
                label1[m] = gid
        label2 = {}
        for gid, g in enumerate(groups2):
            for m in g.members:
                label2[m] = gid

        agents = sorted(all_agents)
        labels_a = [label1.get(a, -1) for a in agents]
        labels_b = [label2.get(a, -1) for a in agents]

        # Contingency matrix
        unique_a = sorted(set(labels_a))
        unique_b = sorted(set(labels_b))
        map_a = {v: i for i, v in enumerate(unique_a)}
        map_b = {v: i for i, v in enumerate(unique_b)}

        contingency = np.zeros((len(unique_a), len(unique_b)))
        for la, lb in zip(labels_a, labels_b):
            contingency[map_a[la], map_b[lb]] += 1

        # Marginals
        row_sums = contingency.sum(axis=1)
        col_sums = contingency.sum(axis=0)

        # Mutual information
        mi = 0.0
        for i in range(len(unique_a)):
            for j in range(len(unique_b)):
                if contingency[i, j] > 0:
                    mi += contingency[i, j] / n * math.log(
                        n * contingency[i, j] / (row_sums[i] * col_sums[j])
                    )

        # Entropies
        h_a = -sum(p / n * math.log(p / n) for p in row_sums if p > 0)
        h_b = -sum(p / n * math.log(p / n) for p in col_sums if p > 0)

        if h_a + h_b == 0:
            return 1.0  # both are single-group → identical
        return 2.0 * mi / (h_a + h_b)

    @staticmethod
    def _pair_agreement(groups1: List[Group], groups2: List[Group]) -> float:
        """Fraction of agent pairs where both partitions agree on co-grouping.

        For each pair (i,j), checks if "i and j in same group" is consistent
        across both partitions. Returns fraction of agreeing pairs.
        """
        all_agents = sorted(set(m for g in groups1 + groups2 for m in g.members))
        n = len(all_agents)
        if n < 2:
            return 1.0

        # Build group-lookup dicts
        lookup1 = {}
        for gid, g in enumerate(groups1):
            for m in g.members:
                lookup1[m] = gid
        lookup2 = {}
        for gid, g in enumerate(groups2):
            for m in g.members:
                lookup2[m] = gid

        agree = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = all_agents[i], all_agents[j]
                same1 = lookup1.get(a) == lookup1.get(b)
                same2 = lookup2.get(a) == lookup2.get(b)
                if same1 == same2:
                    agree += 1
                total += 1

        return agree / total if total > 0 else 1.0

    # ── Multi-theory analysis ────────────────────────────────────────────

    @staticmethod
    def analyze_predictions(
        results: Dict[str, List[Dict[str, Any]]],
        metric: str = "jaccard",
    ) -> Dict[str, Any]:
        """Analyze predictions from multiple theories.

        Args:
            results: {theory_name: history_list} from run_simulation().
            metric: Comparison metric for cross-theory agreement.
        """
        analysis = {
            "convergence_times": {},
            "final_group_counts": {},
            "stability_scores": {},
            "cross_theory_agreement": {},
        }

        theory_names = list(results.keys())

        for theory_name, history in results.items():
            if not history:
                continue

            converged = False
            convergence_time = 0

            for i in range(1, len(history)):
                groups_prev = history[i - 1]["groups"]
                groups_curr = history[i]["groups"]

                if TheoryComparator._groups_equal(groups_prev, groups_curr):
                    convergence_time = history[i]["time"]
                    converged = True
                    break

            analysis["convergence_times"][theory_name] = (
                convergence_time if converged else None
            )
            analysis["final_group_counts"][theory_name] = len(history[-1]["groups"])

            # Stability: fraction of timesteps where groups didn't change
            if len(history) > 1:
                stable_steps = sum(
                    1
                    for i in range(1, len(history))
                    if TheoryComparator._groups_equal(
                        history[i - 1]["groups"], history[i]["groups"]
                    )
                )
                analysis["stability_scores"][theory_name] = stable_steps / (len(history) - 1)

        # Cross-theory agreement
        for i, theory1 in enumerate(theory_names):
            for theory2 in theory_names[i + 1 :]:
                if results[theory1] and results[theory2]:
                    groups1 = results[theory1][-1]["groups"]
                    groups2 = results[theory2][-1]["groups"]

                    similarity = TheoryComparator.compare_group_structures(
                        groups1, groups2, metric=metric
                    )
                    analysis["cross_theory_agreement"][f"{theory1}-{theory2}"] = similarity

        return analysis

    @staticmethod
    def theory_diagnostics(theory: BehaviorTheory) -> Dict[str, Any]:
        """Extract theory-specific diagnostic information.

        Returns a dict of diagnostics relevant to the theory type.
        Works with any BehaviorTheory subclass - returns basic info for
        unknown types and richer diagnostics for known theories.
        """
        state = theory.get_state()
        groups = theory.get_groups()

        diagnostics: Dict[str, Any] = {
            "theory_type": type(theory).__name__,
            "n_agents": len(theory.agents),
            "n_groups": len(groups),
            "group_sizes": [len(g.members) for g in groups],
            "time": theory.current_time,
        }

        # Theory-specific enrichment
        theory_type = type(theory).__name__

        if theory_type == "QST":
            diagnostics["avg_entanglement"] = state.get("avg_entanglement", 0.0)
            diagnostics["n_states"] = state.get("n_states")
            if "probabilities" in state:
                # Entropy of average probability distribution
                avg_probs = state["probabilities"].mean(axis=0)
                avg_probs = avg_probs[avg_probs > 0]
                diagnostics["avg_state_entropy"] = float(-np.sum(avg_probs * np.log(avg_probs + 1e-10)))

        elif theory_type == "TST":
            diagnostics["temperature"] = state.get("temperature")
            diagnostics["energy"] = state.get("energy")
            diagnostics["entropy"] = state.get("entropy")
            diagnostics["free_energy"] = state.get("free_energy")

        elif theory_type == "ICT":
            diagnostics["bandwidth"] = state.get("bandwidth")
            diagnostics["avg_group_size"] = state.get("avg_group_size")

        elif theory_type == "GFT":
            if "positions" in state:
                positions = state["positions"]
                # Average inter-agent distance
                from scipy.spatial.distance import pdist
                if len(positions) > 1:
                    diagnostics["avg_distance"] = float(np.mean(pdist(positions)))
                    diagnostics["position_spread"] = float(np.std(positions))

        return diagnostics

    @staticmethod
    def _groups_equal(groups1: List[Group], groups2: List[Group]) -> bool:
        """Check if two group configurations are identical."""
        if len(groups1) != len(groups2):
            return False

        def group_signature(g):
            return tuple(sorted(g.members))

        sigs1 = sorted([group_signature(g) for g in groups1])
        sigs2 = sorted([group_signature(g) for g in groups2])

        return sigs1 == sigs2
