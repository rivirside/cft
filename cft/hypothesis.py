"""Hypothesis testing framework for multi-theory group formation analysis.

Wraps SocialSimulator + MiroFishAdapter + PredictionTournament into a concise
interface for running theory comparison experiments, parameter sweeps, temporal
prediction tests, and named theoretical claims.

Usage::

    from cft.simulator import SocialSimulator
    from cft.hypothesis import HypothesisTester

    sim = SocialSimulator(n_agents=40, scenario="clustered", k=3, T=30, seed=42)
    ht = HypothesisTester(simulator=sim)

    # Which theory best predicts the social simulation?
    print(ht.compare_theories()["rankings"])

    # How does clustering strength affect theory performance?
    results = ht.parameter_sweep("beta", [0.5, 1.0, 2.0, 4.0, 8.0])

    # Can theories predict future group structure from early interactions?
    print(ht.temporal_prediction(t_freeze=15, t_predict=30)["scores"])

    # Assert a named theoretical claim
    print(ht.test_claim("all_theories_agree_on_clustered", beta=6.0, ctai_threshold=0.6))
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .theories.base import Agent, Group, TheoryParameters
from .theories import CFT, GFT, QST, ICT, TST
from .tournament import PredictionTournament
from .comparator import TheoryComparator
from .simulator import SocialSimulator

# Default theory configurations used when none are specified
DEFAULT_THEORIES: Dict[str, Dict[str, Any]] = {
    "CFT": {"class": CFT, "threshold": 0.6},
    "GFT": {"class": GFT, "k": 0.1, "sigma": 1.0},
    "QST": {"class": QST, "n_states": 5},
    "ICT": {"class": ICT, "bandwidth": 3},
    "TST": {"class": TST, "temperature": 1.0, "sweeps_per_step": 10},
}

# Named claim identifiers
CLAIM_NAMES = (
    "cft_fewer_groups_than_gft_on_polarized",
    "tst_phase_transition",
    "all_theories_agree_on_clustered",
    "theory_x_best_on_scenario",
)


class HypothesisTester:
    """Test hypotheses about group formation theories against synthetic social data.

    Parameters
    ----------
    simulator : SocialSimulator, optional
        Pre-configured simulator. Used as the default for all method calls. If not
        provided, callers must pass a simulator to each method individually.
    theory_configs : dict, optional
        Mapping of ``{name: {"class": TheoryClass, **kwargs}}``. Defaults to all
        five theories with standard parameters (see ``DEFAULT_THEORIES``).
    t_max : float
        Theory simulation duration (default 10.0).
    dt : float
        Theory simulation timestep (default 1.0).
    metric : str
        Partition comparison metric: ``"nmi"``, ``"jaccard"``, or
        ``"pair_agreement"`` (default ``"nmi"``).
    """

    def __init__(
        self,
        simulator: Optional[SocialSimulator] = None,
        theory_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        t_max: float = 10.0,
        dt: float = 1.0,
        metric: str = "nmi",
    ):
        self.simulator = simulator
        self.theory_configs = theory_configs if theory_configs is not None else copy.deepcopy(DEFAULT_THEORIES)
        self.t_max = t_max
        self.dt = dt
        self.metric = metric

    # ── (a) Theory Comparison ─────────────────────────────────────────────────

    def compare_theories(
        self,
        simulator: Optional[SocialSimulator] = None,
        use_temporal_split: bool = False,
        split_fraction: float = 0.5,
        n_runs: int = 1,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Run all theories against simulated social data and score them.

        Parameters
        ----------
        simulator : SocialSimulator, optional
            Override the instance-level simulator.
        use_temporal_split : bool
            If True, split interactions by time: early half builds the affinity
            matrix, late half defines ground-truth communities. Fixes the
            circular ground-truth problem (issue #23). Default False.
        split_fraction : float
            Fraction of interactions used for training when use_temporal_split
            is True (default 0.5).
        n_runs : int
            Number of independent simulation runs. When > 1, each run uses a
            different random seed (base_seed + run_idx). Returns mean ± std
            similarity and a Wilcoxon signed-rank p-value (issue #27).
        significance_level : float
            Alpha threshold for declaring the top-ranked theory significantly
            better than the second (default 0.05). Only used when n_runs >= 5.

        Returns
        -------
        dict with keys:
            Single run: ``"scores"``, ``"rankings"``, ``"ctai"``, ``"agents"``,
            ``"ground_truth"``, ``"affinity_matrix"``, ``"temporal_split"``.

            Multi-run (n_runs > 1): ``"n_runs"``, ``"scores_per_run"``,
            ``"mean_similarity"``, ``"std_similarity"``, ``"rankings"``,
            ``"wilcoxon_pvalue"``, ``"winner_significant"``, ``"temporal_split"``.
        """
        if n_runs > 1:
            return self._compare_theories_multi_run(
                simulator=simulator,
                use_temporal_split=use_temporal_split,
                split_fraction=split_fraction,
                n_runs=n_runs,
                significance_level=significance_level,
            )

        sim = self._resolve_simulator(simulator)
        adapter = sim.to_adapter()
        agents = adapter.load_agents()
        adapter.load_interactions()

        if use_temporal_split:
            import pandas as pd
            df = adapter._interactions.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df_sorted = df.sort_values("timestamp")
            cutoff_idx = int(len(df_sorted) * split_fraction)
            train_df = df_sorted.iloc[:cutoff_idx]
            eval_df = df_sorted.iloc[cutoff_idx:]
            affinity = adapter.compute_affinity_matrix(train_df)
            ground_truth = adapter.extract_ground_truth_groups(eval_df)
        else:
            affinity = adapter.compute_affinity_matrix()
            ground_truth = adapter.extract_ground_truth_groups()

        tournament = self._run_tournament(agents, affinity, ground_truth)

        return {
            "scores": tournament.score(ground_truth, metric=self.metric),
            "rankings": tournament.rankings(ground_truth, metric=self.metric),
            "ctai": tournament.compute_ctai(metric=self.metric),
            "agents": agents,
            "ground_truth": ground_truth,
            "affinity_matrix": affinity,
            "temporal_split": use_temporal_split,
        }

    # ── (b) Parameter Sweep ───────────────────────────────────────────────────

    def parameter_sweep(
        self,
        sweep_param: str,
        sweep_values: List[Any],
        fixed_kwargs: Optional[Dict[str, Any]] = None,
        theories: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Sweep a SocialSimulator parameter and run theories for each value.

        Parameters
        ----------
        sweep_param : str
            Name of a ``SocialSimulator.__init__`` parameter (e.g., ``"beta"``,
            ``"n_agents"``, ``"scenario"``).
        sweep_values : list
            Values to test.
        fixed_kwargs : dict, optional
            Other simulator kwargs held constant across the sweep. If an instance
            simulator is set, its config is used as a base and these override it.
        theories : dict, optional
            Override ``self.theory_configs`` for this sweep.

        Returns
        -------
        list[dict]
            One entry per sweep value, each with:
            ``"param_value"``, ``"scores"``, ``"ctai"``, ``"n_groups_gt"``
        """
        base_kwargs = self._simulator_base_kwargs()
        if fixed_kwargs:
            base_kwargs.update(fixed_kwargs)

        results = []
        for value in sweep_values:
            kwargs = dict(base_kwargs)
            kwargs[sweep_param] = value

            sim = SocialSimulator(**kwargs)
            adapter = sim.to_adapter()
            agents = adapter.load_agents()
            adapter.load_interactions()
            affinity = adapter.compute_affinity_matrix()
            ground_truth = adapter.extract_ground_truth_groups()

            tournament = self._run_tournament(agents, affinity, ground_truth, theories)

            results.append({
                "param_value": value,
                "scores": tournament.score(ground_truth, metric=self.metric),
                "ctai": tournament.compute_ctai(metric=self.metric),
                "n_groups_gt": len(ground_truth),
            })
            sim.cleanup()

        return results

    # ── (c) Temporal Prediction ───────────────────────────────────────────────

    def temporal_prediction(
        self,
        t_freeze: int,
        t_predict: int,
        simulator: Optional[SocialSimulator] = None,
        theories: Optional[Dict[str, Dict[str, Any]]] = None,
        n_runs: int = 1,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Test whether theories can predict future group structure from early interactions.

        Splits interactions at ``t_freeze``: interactions before the freeze are used to
        compute the affinity matrix; interactions after are used to detect ground-truth
        communities. Each theory predicts group structure at ``t_predict``.

        Parameters
        ----------
        t_freeze : int
            Timestep cutoff for training data (0-indexed). Must be < t_predict and
            < simulator.T.
        t_predict : int
            Timestep at which to evaluate predictions (ground truth). Must be
            <= simulator.T.
        simulator : SocialSimulator, optional
            Override instance-level simulator.
        theories : dict, optional
            Override ``self.theory_configs``.
        n_runs : int
            Number of independent simulation runs (issue #27). When > 1, returns
            aggregated mean ± std and Wilcoxon p-value (default 1).
        significance_level : float
            Alpha threshold for Wilcoxon test (default 0.05).

        Returns
        -------
        dict with keys:
            Single run: ``"scores"``, ``"t_freeze"``, ``"t_predict"``,
            ``"n_train_interactions"``, ``"n_eval_interactions"``.

            Multi-run: ``"n_runs"``, ``"scores_per_run"``, ``"mean_similarity"``,
            ``"std_similarity"``, ``"rankings"``, ``"wilcoxon_pvalue"``,
            ``"winner_significant"``, ``"t_freeze"``, ``"t_predict"``,
            ``"n_train_interactions"``, ``"n_eval_interactions"``.
        """
        if n_runs > 1:
            return self._temporal_prediction_multi_run(
                t_freeze=t_freeze,
                t_predict=t_predict,
                simulator=simulator,
                theories=theories,
                n_runs=n_runs,
                significance_level=significance_level,
            )

        sim = self._resolve_simulator(simulator)

        if t_freeze < 0:
            raise ValueError(f"t_freeze must be >= 0, got {t_freeze}")
        if t_predict <= t_freeze:
            raise ValueError(f"t_predict ({t_predict}) must be > t_freeze ({t_freeze})")
        if t_freeze >= sim.T:
            raise ValueError(
                f"t_freeze ({t_freeze}) must be < simulator.T ({sim.T})"
            )

        freeze_iso = SocialSimulator.iso_from_step(t_freeze)
        predict_iso = SocialSimulator.iso_from_step(t_predict)

        adapter = sim.to_adapter()
        agents = adapter.load_agents()
        adapter.load_interactions()

        import pandas as pd
        all_df = adapter._interactions.copy()
        ts_col = pd.to_datetime(all_df["timestamp"])
        freeze_ts = pd.to_datetime(freeze_iso)
        predict_ts = pd.to_datetime(predict_iso)

        n_train = int((ts_col < freeze_ts).sum())
        n_eval = int(((ts_col >= freeze_ts) & (ts_col < predict_ts)).sum())

        train_df = all_df[ts_col < freeze_ts]
        eval_df = all_df[(ts_col >= freeze_ts) & (ts_col < predict_ts)]

        affinity = adapter.compute_affinity_matrix(train_df)
        ground_truth = adapter.extract_ground_truth_groups(eval_df)

        tournament = self._run_tournament(agents, affinity, ground_truth, theories)
        scores = tournament.score(ground_truth, metric=self.metric)

        sim.cleanup()

        return {
            "scores": scores,
            "t_freeze": t_freeze,
            "t_predict": t_predict,
            "n_train_interactions": n_train,
            "n_eval_interactions": n_eval,
        }

    # ── (d) Named Claims ──────────────────────────────────────────────────────

    def test_claim(
        self,
        claim: str,
        simulator: Optional[SocialSimulator] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Test a named theoretical assertion.

        Parameters
        ----------
        claim : str
            One of:

            - ``"cft_fewer_groups_than_gft_on_polarized"``
              CFT produces fewer groups than GFT on polarized data.
              kwargs: ``n_trials`` (int, default 5), ``n_agents`` (int, default 40),
              ``bias_strength`` (float, default 0.8)

            - ``"tst_phase_transition"``
              TST group count varies significantly with temperature, indicating a
              phase transition. kwargs: ``n_temps`` (int, default 8),
              ``n_agents`` (int, default 30), ``temp_range`` (tuple, default (0.1, 3.0))

            - ``"all_theories_agree_on_clustered"``
              On strongly clustered data, CTAI > ctai_threshold.
              kwargs: ``ctai_threshold`` (float, default 0.6), ``beta`` (float, default 4.0),
              ``n_agents`` (int, default 40), ``k`` (int, default 3)

            - ``"theory_x_best_on_scenario"``
              Named theory achieves highest PAS on a named scenario.
              kwargs: ``theory`` (str, required), ``scenario`` (str, required),
              ``n_trials`` (int, default 5), ``n_agents`` (int, default 40)

        simulator : SocialSimulator, optional
            Override instance-level simulator (only used by claims that don't
            build their own).
        **kwargs
            Claim-specific arguments (see above).

        Returns
        -------
        dict with keys ``"passed"`` (bool) and ``"evidence"`` (dict).
        """
        dispatch = {
            "cft_fewer_groups_than_gft_on_polarized": self._claim_cft_fewer_groups,
            "tst_phase_transition": self._claim_tst_phase_transition,
            "all_theories_agree_on_clustered": self._claim_all_agree_clustered,
            "theory_x_best_on_scenario": self._claim_theory_x_best,
        }
        if claim not in dispatch:
            raise ValueError(
                f"Unknown claim {claim!r}. Valid claims: {list(dispatch.keys())}"
            )
        return dispatch[claim](**kwargs)

    # ── Claim Implementations ─────────────────────────────────────────────────

    def _claim_cft_fewer_groups(
        self,
        n_trials: int = 5,
        n_agents: int = 40,
        bias_strength: float = 0.8,
    ) -> Dict[str, Any]:
        """CFT produces fewer groups than GFT on polarized data."""
        cft_cfg = {"class": CFT, "threshold": 0.6}
        gft_cfg = {"class": GFT, "k": 0.1, "sigma": 1.0}
        params = TheoryParameters(n_agents=n_agents, n_features=4 + 3 + 1, random_seed=42)

        cft_counts = []
        gft_counts = []

        for trial in range(n_trials):
            sim = SocialSimulator(
                n_agents=n_agents,
                scenario="polarized",
                bias_strength=bias_strength,
                T=20,
                seed=trial,
            )
            adapter = sim.to_adapter()
            agents = adapter.load_agents()
            adapter.load_interactions()
            affinity = adapter.compute_affinity_matrix()

            n_feat = len(agents[0].features)
            params_t = TheoryParameters(n_agents=len(agents), n_features=n_feat, random_seed=42)

            cft = CFT(params_t, threshold=0.6, affinity_matrix=affinity)
            cft.initialize_agents(agents)
            cft.run_simulation(self.t_max, self.dt)
            cft_counts.append(len(cft.get_groups()))

            gft = GFT(params_t, k=0.1, sigma=1.0, affinity_matrix=affinity)
            gft.initialize_agents(agents)
            gft.run_simulation(self.t_max, self.dt)
            gft_counts.append(len(gft.get_groups()))

            sim.cleanup()

        cft_mean = float(np.mean(cft_counts))
        gft_mean = float(np.mean(gft_counts))
        passed = cft_mean < gft_mean

        return {
            "passed": passed,
            "evidence": {
                "cft_group_counts": cft_counts,
                "gft_group_counts": gft_counts,
                "cft_mean": cft_mean,
                "gft_mean": gft_mean,
                "n_trials": n_trials,
                "bias_strength": bias_strength,
            },
        }

    def _claim_tst_phase_transition(
        self,
        n_temps: int = 8,
        n_agents: int = 30,
        temp_range: Tuple[float, float] = (0.1, 3.0),
    ) -> Dict[str, Any]:
        """TST group count varies significantly with temperature."""
        temperatures = np.linspace(temp_range[0], temp_range[1], n_temps).tolist()

        sim = SocialSimulator(n_agents=n_agents, scenario="random", T=15, seed=99)
        adapter = sim.to_adapter()
        agents = adapter.load_agents()
        adapter.load_interactions()
        affinity = adapter.compute_affinity_matrix()

        n_feat = len(agents[0].features)
        params = TheoryParameters(n_agents=n_agents, n_features=n_feat, random_seed=42)

        group_counts = []
        for temp in temperatures:
            tst = TST(params, temperature=temp, sweeps_per_step=10, affinity_matrix=affinity)
            tst.initialize_agents(agents)
            tst.run_simulation(self.t_max, self.dt)
            group_counts.append(len(tst.get_groups()))

        sim.cleanup()

        variance = float(np.var(group_counts))
        # Phase transition is indicated by non-trivial variance in group count
        threshold = 0.5
        passed = variance > threshold

        return {
            "passed": passed,
            "evidence": {
                "temperatures": temperatures,
                "group_counts": group_counts,
                "variance": variance,
                "variance_threshold": threshold,
            },
        }

    def _claim_all_agree_clustered(
        self,
        ctai_threshold: float = 0.6,
        beta: float = 4.0,
        n_agents: int = 40,
        k: int = 3,
    ) -> Dict[str, Any]:
        """All theories agree (CTAI > threshold) on strongly clustered data."""
        sim = SocialSimulator(
            n_agents=n_agents,
            scenario="clustered",
            k=k,
            beta=beta,
            T=20,
            seed=7,
        )
        adapter = sim.to_adapter()
        agents = adapter.load_agents()
        adapter.load_interactions()
        affinity = adapter.compute_affinity_matrix()
        ground_truth = adapter.extract_ground_truth_groups()

        tournament = self._run_tournament(agents, affinity, ground_truth)
        ctai = tournament.compute_ctai(metric=self.metric)

        sim.cleanup()
        passed = ctai >= ctai_threshold

        return {
            "passed": passed,
            "evidence": {
                "ctai": ctai,
                "ctai_threshold": ctai_threshold,
                "beta": beta,
                "n_agents": n_agents,
                "k": k,
                "n_groups_gt": len(ground_truth),
            },
        }

    def _claim_theory_x_best(
        self,
        theory: str,
        scenario: str,
        n_trials: int = 5,
        n_agents: int = 40,
    ) -> Dict[str, Any]:
        """Named theory achieves highest PAS on a named scenario."""
        if theory not in self.theory_configs:
            raise ValueError(
                f"Theory {theory!r} not in theory_configs. "
                f"Available: {list(self.theory_configs.keys())}"
            )

        trial_winners = []
        trial_scores = []

        for trial in range(n_trials):
            sim = SocialSimulator(
                n_agents=n_agents,
                scenario=scenario,
                T=20,
                seed=trial,
            )
            adapter = sim.to_adapter()
            agents = adapter.load_agents()
            adapter.load_interactions()
            affinity = adapter.compute_affinity_matrix()
            ground_truth = adapter.extract_ground_truth_groups()

            tournament = self._run_tournament(agents, affinity, ground_truth)
            scores = tournament.score(ground_truth, metric=self.metric)
            rankings = tournament.rankings(ground_truth, metric=self.metric)

            winner = rankings[0]["theory"]
            trial_winners.append(winner)
            trial_scores.append({name: s["similarity"] for name, s in scores.items()})

            sim.cleanup()

        win_count = sum(1 for w in trial_winners if w == theory)
        passed = win_count > n_trials // 2

        return {
            "passed": passed,
            "evidence": {
                "theory": theory,
                "scenario": scenario,
                "n_trials": n_trials,
                "win_count": win_count,
                "trial_winners": trial_winners,
                "trial_scores": trial_scores,
            },
        }

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _compare_theories_multi_run(
        self,
        simulator=None,
        use_temporal_split: bool = False,
        split_fraction: float = 0.5,
        n_runs: int = 5,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Run compare_theories n_runs times with varying seeds; return aggregated stats."""
        import pandas as pd

        base_kwargs = self._get_simulator_kwargs(simulator)
        base_seed = base_kwargs.get("seed") or 0

        scores_per_run: List[Dict[str, float]] = []
        for run_idx in range(n_runs):
            kwargs = dict(base_kwargs)
            kwargs["seed"] = base_seed + run_idx

            sim = SocialSimulator(**kwargs)
            adapter = sim.to_adapter()
            agents = adapter.load_agents()
            adapter.load_interactions()

            if use_temporal_split:
                df = adapter._interactions.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df_sorted = df.sort_values("timestamp")
                cutoff_idx = int(len(df_sorted) * split_fraction)
                train_df = df_sorted.iloc[:cutoff_idx]
                eval_df = df_sorted.iloc[cutoff_idx:]
                affinity = adapter.compute_affinity_matrix(train_df)
                ground_truth = adapter.extract_ground_truth_groups(eval_df)
            else:
                affinity = adapter.compute_affinity_matrix()
                ground_truth = adapter.extract_ground_truth_groups()

            tournament = self._run_tournament(agents, affinity, ground_truth)
            run_scores = tournament.score(ground_truth, metric=self.metric)
            scores_per_run.append({name: s["similarity"] for name, s in run_scores.items()})
            sim.cleanup()

        theory_names = list(scores_per_run[0].keys())
        mean_sim = {n: float(np.mean([r[n] for r in scores_per_run])) for n in theory_names}
        std_sim = {n: float(np.std([r[n] for r in scores_per_run])) for n in theory_names}
        rankings = sorted(theory_names, key=lambda n: mean_sim[n], reverse=True)

        wilcoxon_pvalue = None
        winner_significant = False
        if len(rankings) >= 2 and n_runs >= 5:
            try:
                from scipy.stats import wilcoxon as _wilcoxon
                top1 = [r[rankings[0]] for r in scores_per_run]
                top2 = [r[rankings[1]] for r in scores_per_run]
                _, pvalue = _wilcoxon(top1, top2, alternative="greater")
                wilcoxon_pvalue = float(pvalue)
                winner_significant = pvalue < significance_level
            except Exception:
                pass

        return {
            "n_runs": n_runs,
            "scores_per_run": scores_per_run,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "rankings": rankings,
            "wilcoxon_pvalue": wilcoxon_pvalue,
            "winner_significant": winner_significant,
            "temporal_split": use_temporal_split,
        }

    def _temporal_prediction_multi_run(
        self,
        t_freeze: int,
        t_predict: int,
        simulator=None,
        theories=None,
        n_runs: int = 5,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Run temporal_prediction n_runs times; return aggregated statistics."""
        import pandas as pd

        base_kwargs = self._get_simulator_kwargs(simulator)
        base_seed = base_kwargs.get("seed") or 0
        freeze_iso = SocialSimulator.iso_from_step(t_freeze)
        predict_iso = SocialSimulator.iso_from_step(t_predict)

        scores_per_run: List[Dict[str, float]] = []
        n_train_list: List[int] = []
        n_eval_list: List[int] = []

        for run_idx in range(n_runs):
            kwargs = dict(base_kwargs)
            kwargs["seed"] = base_seed + run_idx

            sim = SocialSimulator(**kwargs)
            adapter = sim.to_adapter()
            agents = adapter.load_agents()
            adapter.load_interactions()

            all_df = adapter._interactions.copy()
            ts_col = pd.to_datetime(all_df["timestamp"])
            freeze_ts = pd.to_datetime(freeze_iso)
            predict_ts = pd.to_datetime(predict_iso)

            train_df = all_df[ts_col < freeze_ts]
            eval_df = all_df[(ts_col >= freeze_ts) & (ts_col < predict_ts)]
            n_train_list.append(int(len(train_df)))
            n_eval_list.append(int(len(eval_df)))

            affinity = adapter.compute_affinity_matrix(train_df)
            ground_truth = adapter.extract_ground_truth_groups(eval_df)

            tournament = self._run_tournament(agents, affinity, ground_truth, theories)
            run_scores = tournament.score(ground_truth, metric=self.metric)
            scores_per_run.append({name: s["similarity"] for name, s in run_scores.items()})
            sim.cleanup()

        theory_names = list(scores_per_run[0].keys())
        mean_sim = {n: float(np.mean([r[n] for r in scores_per_run])) for n in theory_names}
        std_sim = {n: float(np.std([r[n] for r in scores_per_run])) for n in theory_names}
        rankings = sorted(theory_names, key=lambda n: mean_sim[n], reverse=True)

        wilcoxon_pvalue = None
        winner_significant = False
        if len(rankings) >= 2 and n_runs >= 5:
            try:
                from scipy.stats import wilcoxon as _wilcoxon
                top1 = [r[rankings[0]] for r in scores_per_run]
                top2 = [r[rankings[1]] for r in scores_per_run]
                _, pvalue = _wilcoxon(top1, top2, alternative="greater")
                wilcoxon_pvalue = float(pvalue)
                winner_significant = pvalue < significance_level
            except Exception:
                pass

        return {
            "n_runs": n_runs,
            "scores_per_run": scores_per_run,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "rankings": rankings,
            "wilcoxon_pvalue": wilcoxon_pvalue,
            "winner_significant": winner_significant,
            "t_freeze": t_freeze,
            "t_predict": t_predict,
            "n_train_interactions": int(np.mean(n_train_list)) if n_train_list else 0,
            "n_eval_interactions": int(np.mean(n_eval_list)) if n_eval_list else 0,
        }

    def _get_simulator_kwargs(self, simulator=None) -> Dict[str, Any]:
        """Extract constructor kwargs from given or instance simulator."""
        sim = simulator or self.simulator
        if sim is None:
            return {"n_agents": 40, "scenario": "random", "T": 20, "seed": None}
        return {
            "n_agents": sim.n_agents,
            "n_opinions": sim.n_opinions,
            "scenario": sim.scenario,
            "T": sim.T,
            "interaction_rate": sim.interaction_rate,
            "alpha": sim.alpha,
            "beta": sim.beta,
            "gamma": sim.gamma,
            "pos_threshold": sim.pos_threshold,
            "neg_threshold": sim.neg_threshold,
            "seed": sim.seed,
            **sim.scenario_kwargs,
        }

    def _run_tournament(
        self,
        agents: List[Agent],
        affinity: np.ndarray,
        ground_truth: List[Group],
        theories: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> PredictionTournament:
        """Build and run a PredictionTournament. Returns the tournament instance."""
        theories = copy.deepcopy(theories or self.theory_configs)
        n_feat = len(agents[0].features)
        params = TheoryParameters(n_agents=len(agents), n_features=n_feat, random_seed=42)

        tournament = PredictionTournament(agents, params)
        for name, config in theories.items():
            cfg = dict(config)
            theory_cls = cfg.pop("class")
            tournament.add_theory(name, theory_cls, affinity_matrix=affinity, **cfg)

        tournament.run(t_max=self.t_max, dt=self.dt)
        return tournament

    def _resolve_simulator(self, override: Optional[SocialSimulator]) -> SocialSimulator:
        """Return override if given, else instance simulator; error if neither."""
        sim = override or self.simulator
        if sim is None:
            raise ValueError(
                "No simulator provided. Pass a SocialSimulator to the method or "
                "set HypothesisTester(simulator=...)."
            )
        if not sim._generated:
            sim.generate()
        return sim

    def _simulator_base_kwargs(self) -> Dict[str, Any]:
        """Extract constructor kwargs from instance simulator, or return defaults."""
        if self.simulator is None:
            return {"n_agents": 40, "scenario": "random", "T": 20, "seed": None}
        sim = self.simulator
        return {
            "n_agents": sim.n_agents,
            "n_opinions": sim.n_opinions,
            "scenario": sim.scenario,
            "T": sim.T,
            "interaction_rate": sim.interaction_rate,
            "alpha": sim.alpha,
            "beta": sim.beta,
            "gamma": sim.gamma,
            "pos_threshold": sim.pos_threshold,
            "neg_threshold": sim.neg_threshold,
            "seed": sim.seed,
            **sim.scenario_kwargs,
        }
