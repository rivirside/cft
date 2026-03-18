"""Prediction tournament: run multiple theories, score against ground truth.

Implements the quantitative comparison metrics from EXPERIMENTS.md:
- PAS (Prediction Accuracy Score)
- DFI (Dynamic Fidelity Index)
- PSS (Parameter Sensitivity Score)
- CTAI (Cross-Theory Agreement Index)
"""

from typing import List, Dict, Any, Optional, Type
import copy

import numpy as np

from .theories.base import BehaviorTheory, Agent, Group, TheoryParameters
from .comparator import TheoryComparator


class PredictionTournament:
    """Run multiple theories on shared agents and score their predictions.

    Usage::

        agents = [Agent(id=i, features=rng.standard_normal(3)) for i in range(20)]
        params = TheoryParameters(n_agents=20, n_features=3, random_seed=42)

        tournament = PredictionTournament(agents, params)
        tournament.add_theory("CFT", CFT, threshold=0.5)
        tournament.add_theory("GFT", GFT, k=0.1, sigma=1.0)
        tournament.add_theory("TST", TST, temperature=0.5)

        results = tournament.run(t_max=20.0, dt=1.0)
        scores = tournament.score(ground_truth_groups)
    """

    def __init__(self, agents: List[Agent], params: TheoryParameters):
        self.agents = agents
        self.params = params
        self._theories: Dict[str, BehaviorTheory] = {}
        self._theory_kwargs: Dict[str, dict] = {}
        self._histories: Dict[str, List[Dict[str, Any]]] = {}

    def add_theory(
        self,
        name: str,
        theory_class: Type[BehaviorTheory],
        **kwargs,
    ):
        """Register a theory for the tournament.

        Args:
            name: Display name (e.g., "CFT-0.5").
            theory_class: The BehaviorTheory subclass.
            **kwargs: Arguments passed to theory_class.__init__ (after params).
        """
        theory = theory_class(self.params, **kwargs)
        theory.initialize_agents(self.agents)
        self._theories[name] = theory
        self._theory_kwargs[name] = kwargs

    def run(self, t_max: float, dt: float) -> Dict[str, List[Dict[str, Any]]]:
        """Run all theories and collect histories.

        Returns:
            {theory_name: history_list} where each history entry has
            "time", "groups", and "state" keys.
        """
        self._histories = {}
        for name, theory in self._theories.items():
            self._histories[name] = theory.run_simulation(t_max=t_max, dt=dt)
        return self._histories

    def score(
        self,
        ground_truth: List[Group],
        metric: str = "nmi",
    ) -> Dict[str, Dict[str, float]]:
        """Score each theory against ground truth groups.

        Returns:
            {theory_name: {"pas": float, "nmi": float, "n_groups_predicted": int, ...}}
        """
        scores = {}
        for name, theory in self._theories.items():
            predicted = theory.get_groups()
            scores[name] = self._compute_pas(predicted, ground_truth, metric)
        return scores

    def freeze_and_predict(
        self,
        t_freeze: float,
        t_predict: float,
        dt: float,
        ground_truth_at_predict: List[Group],
        metric: str = "nmi",
    ) -> Dict[str, Dict[str, float]]:
        """Temporal prediction: run to t_freeze, then predict at t_predict.

        This tests whether theories can extrapolate future group structure
        from partial observation.

        Args:
            t_freeze: Time at which theories are "frozen" (observed data ends).
            t_predict: Time at which prediction is evaluated.
            dt: Timestep.
            ground_truth_at_predict: Known groups at t_predict.
            metric: Comparison metric.

        Returns:
            {theory_name: scores_dict}
        """
        scores = {}
        for name, theory_class_and_kwargs in self._theory_kwargs.items():
            # Get the theory class from the existing instance
            theory_class = type(self._theories[name])
            kwargs = self._theory_kwargs[name]

            # Fresh instance
            theory = theory_class(self.params, **kwargs)
            theory.initialize_agents(self.agents)
            theory.run_simulation(t_max=t_predict, dt=dt)

            predicted = theory.get_groups()
            scores[name] = self._compute_pas(predicted, ground_truth_at_predict, metric)
            scores[name]["t_freeze"] = t_freeze
            scores[name]["t_predict"] = t_predict

        return scores

    def compute_dfi(
        self,
        ground_truth_history: List[List[Group]],
        metric: str = "nmi",
    ) -> Dict[str, float]:
        """Dynamic Fidelity Index: how well each theory tracks temporal evolution.

        DFI = (1/T) * sum_t similarity(predicted(t), observed(t))

        Args:
            ground_truth_history: List of group configs at each timestep,
                aligned with the simulation history length.
            metric: Comparison metric.

        Returns:
            {theory_name: dfi_score}
        """
        dfi_scores = {}
        for name, history in self._histories.items():
            T = min(len(history), len(ground_truth_history))
            if T == 0:
                dfi_scores[name] = 0.0
                continue

            total_sim = 0.0
            for t in range(T):
                predicted = history[t]["groups"]
                observed = ground_truth_history[t]
                total_sim += TheoryComparator.compare_group_structures(
                    predicted, observed, metric=metric
                )
            dfi_scores[name] = total_sim / T

        return dfi_scores

    def compute_pss(
        self,
        theory_name: str,
        param_name: str,
        base_value: float,
        perturbation: float = 0.1,
        t_max: float = 10.0,
        dt: float = 1.0,
        metric: str = "nmi",
    ) -> float:
        """Parameter Sensitivity Score for a single theory/parameter.

        PSS = 1 - std(similarities) where similarities are computed between
        the base run and ±perturbation runs.

        Higher PSS = more robust (less sensitive to parameter changes).

        Args:
            theory_name: Name of registered theory.
            param_name: Parameter to perturb (e.g., "threshold", "temperature").
            base_value: Base parameter value.
            perturbation: Fractional perturbation (default 10%).
            t_max: Simulation time.
            dt: Timestep.
            metric: Comparison metric.

        Returns:
            PSS score in [0, 1].
        """
        theory_class = type(self._theories[theory_name])
        base_kwargs = dict(self._theory_kwargs[theory_name])

        # Run base
        base_theory = theory_class(self.params, **base_kwargs)
        base_theory.initialize_agents(self.agents)
        base_theory.run_simulation(t_max=t_max, dt=dt)
        base_groups = base_theory.get_groups()

        # Run perturbations
        similarities = []
        for factor in [1 - perturbation, 1 + perturbation]:
            perturbed_kwargs = dict(base_kwargs)
            perturbed_kwargs[param_name] = base_value * factor

            perturbed = theory_class(self.params, **perturbed_kwargs)
            perturbed.initialize_agents(self.agents)
            perturbed.run_simulation(t_max=t_max, dt=dt)

            sim = TheoryComparator.compare_group_structures(
                base_groups, perturbed.get_groups(), metric=metric
            )
            similarities.append(sim)

        # PSS = 1 - std(similarities to base)
        return 1.0 - float(np.std(similarities))

    def compute_ctai(self, metric: str = "nmi") -> float:
        """Cross-Theory Agreement Index.

        CTAI = mean pairwise similarity across all theory pairs.
        When theories agree, confidence in the prediction is higher.

        Returns:
            CTAI score in [0, 1].
        """
        names = list(self._theories.keys())
        if len(names) < 2:
            return 1.0

        similarities = []
        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                groups1 = self._theories[name1].get_groups()
                groups2 = self._theories[name2].get_groups()
                sim = TheoryComparator.compare_group_structures(
                    groups1, groups2, metric=metric
                )
                similarities.append(sim)

        n_agreeing = sum(1 for s in similarities if s > 0.5)
        agreement_strength = float(np.mean(similarities))
        return (n_agreeing / len(similarities)) * agreement_strength

    def rankings(self, ground_truth: List[Group], metric: str = "nmi") -> List[Dict[str, Any]]:
        """Return theories ranked by prediction accuracy.

        Returns:
            List of dicts sorted by score descending, each with
            "rank", "theory", "score", "n_groups", "diagnostics".
        """
        scores = self.score(ground_truth, metric=metric)
        ranked = []
        for name, score_dict in scores.items():
            ranked.append({
                "theory": name,
                "score": score_dict["similarity"],
                "n_groups_predicted": score_dict["n_groups_predicted"],
                "n_groups_actual": score_dict["n_groups_actual"],
                "diagnostics": TheoryComparator.theory_diagnostics(self._theories[name]),
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        for i, entry in enumerate(ranked):
            entry["rank"] = i + 1

        return ranked

    def results_dict(self) -> Dict[str, Any]:
        """Return all results as a plain dict (pandas-free).

        Returns dict with "histories", "final_groups", "diagnostics" keys.
        """
        result = {
            "histories": self._histories,
            "final_groups": {},
            "diagnostics": {},
        }
        for name, theory in self._theories.items():
            result["final_groups"][name] = theory.get_groups()
            result["diagnostics"][name] = TheoryComparator.theory_diagnostics(theory)
        return result

    @staticmethod
    def _compute_pas(
        predicted: List[Group],
        ground_truth: List[Group],
        metric: str,
    ) -> Dict[str, float]:
        """Prediction Accuracy Score components.

        PAS = (group_count_accuracy + partition_similarity + size_accuracy) / 3
        """
        n_pred = len(predicted)
        n_true = len(ground_truth)

        # 1. Partition similarity (using specified metric)
        similarity = TheoryComparator.compare_group_structures(
            predicted, ground_truth, metric=metric
        )

        # 2. Group count accuracy: 1 - |n_pred - n_true| / max(n_pred, n_true, 1)
        group_count_accuracy = 1.0 - abs(n_pred - n_true) / max(n_pred, n_true, 1)

        # 3. Size distribution accuracy: compare sorted group sizes
        pred_sizes = sorted([len(g.members) for g in predicted], reverse=True)
        true_sizes = sorted([len(g.members) for g in ground_truth], reverse=True)

        # Pad shorter list with zeros
        max_len = max(len(pred_sizes), len(true_sizes))
        pred_sizes += [0] * (max_len - len(pred_sizes))
        true_sizes += [0] * (max_len - len(true_sizes))

        total_agents = max(sum(true_sizes), 1)
        size_error = sum(abs(p - t) for p, t in zip(pred_sizes, true_sizes))
        size_accuracy = 1.0 - size_error / (2 * total_agents)

        pas = (group_count_accuracy + similarity + size_accuracy) / 3.0

        return {
            "pas": pas,
            "similarity": similarity,
            "group_count_accuracy": group_count_accuracy,
            "size_accuracy": size_accuracy,
            "n_groups_predicted": n_pred,
            "n_groups_actual": n_true,
        }
