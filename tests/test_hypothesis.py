"""Tests for HypothesisTester."""

import pytest
import numpy as np

from cft import CFT, GFT, TST
from cft.simulator import SocialSimulator
from cft.hypothesis import HypothesisTester, DEFAULT_THEORIES

# Lightweight configs to keep tests fast
_FAST_THEORIES = {
    "CFT": {"class": CFT, "threshold": 0.6},
    "GFT": {"class": GFT, "k": 0.1, "sigma": 1.0},
    "TST": {"class": TST, "temperature": 1.0, "sweeps_per_step": 5},
}

_FAST_SIM_KWARGS = dict(n_agents=15, T=8, seed=42)


def _fast_sim(**overrides):
    kwargs = {**_FAST_SIM_KWARGS, **overrides}
    return SocialSimulator(**kwargs)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_theory_configs(self):
        ht = HypothesisTester()
        assert set(ht.theory_configs.keys()) == set(DEFAULT_THEORIES.keys())

    def test_custom_theory_configs(self):
        ht = HypothesisTester(theory_configs=_FAST_THEORIES)
        assert set(ht.theory_configs.keys()) == {"CFT", "GFT", "TST"}

    def test_no_simulator_raises_on_compare(self):
        ht = HypothesisTester(theory_configs=_FAST_THEORIES)
        with pytest.raises(ValueError, match="No simulator"):
            ht.compare_theories()


# ---------------------------------------------------------------------------
# compare_theories
# ---------------------------------------------------------------------------


class TestCompareTheories:
    def setup_method(self):
        self.sim = _fast_sim()
        self.ht = HypothesisTester(
            simulator=self.sim,
            theory_configs=_FAST_THEORIES,
            t_max=3.0,
            dt=1.0,
        )

    def test_returns_all_keys(self):
        result = self.ht.compare_theories()
        assert "scores" in result
        assert "rankings" in result
        assert "ctai" in result
        assert "agents" in result
        assert "ground_truth" in result
        assert "affinity_matrix" in result

    def test_scores_has_all_theories(self):
        result = self.ht.compare_theories()
        assert set(result["scores"].keys()) == {"CFT", "GFT", "TST"}

    def test_pas_in_range(self):
        result = self.ht.compare_theories()
        for name, s in result["scores"].items():
            assert 0.0 <= s["pas"] <= 1.0, f"{name} PAS={s['pas']} out of [0,1]"

    def test_ctai_in_range(self):
        result = self.ht.compare_theories()
        assert 0.0 <= result["ctai"] <= 1.0

    def test_rankings_sorted_descending(self):
        result = self.ht.compare_theories()
        scores = [r["score"] for r in result["rankings"]]
        assert scores == sorted(scores, reverse=True)

    def test_rankings_have_rank_field(self):
        result = self.ht.compare_theories()
        for i, r in enumerate(result["rankings"]):
            assert r["rank"] == i + 1

    def test_override_simulator(self):
        sim2 = _fast_sim(seed=99)
        result = self.ht.compare_theories(simulator=sim2)
        assert "scores" in result

    def test_single_theory(self):
        ht = HypothesisTester(
            simulator=self.sim,
            theory_configs={"CFT": {"class": CFT, "threshold": 0.6}},
            t_max=2.0,
            dt=1.0,
        )
        result = ht.compare_theories()
        assert list(result["scores"].keys()) == ["CFT"]

    def test_theory_configs_not_mutated(self):
        """Deep-copy ensures repeated calls don't corrupt theory_configs."""
        configs_before = {k: dict(v) for k, v in self.ht.theory_configs.items()}
        self.ht.compare_theories()
        self.ht.compare_theories()
        for k, v in configs_before.items():
            assert "class" in self.ht.theory_configs[k], f"{k} lost 'class' key"


# ---------------------------------------------------------------------------
# parameter_sweep
# ---------------------------------------------------------------------------


class TestParameterSweep:
    def setup_method(self):
        self.ht = HypothesisTester(
            theory_configs=_FAST_THEORIES,
            t_max=3.0,
            dt=1.0,
        )
        self.fixed = dict(n_agents=12, scenario="clustered", k=2, T=6, seed=10)

    def test_returns_correct_length(self):
        results = self.ht.parameter_sweep("beta", [0.5, 2.0, 4.0], fixed_kwargs=self.fixed)
        assert len(results) == 3

    def test_param_values_preserved(self):
        values = [0.5, 2.0, 4.0]
        results = self.ht.parameter_sweep("beta", values, fixed_kwargs=self.fixed)
        for r, v in zip(results, values):
            assert r["param_value"] == v

    def test_each_result_has_required_keys(self):
        results = self.ht.parameter_sweep("beta", [1.0, 2.0], fixed_kwargs=self.fixed)
        for r in results:
            assert "param_value" in r
            assert "scores" in r
            assert "ctai" in r
            assert "n_groups_gt" in r

    def test_ctai_values_in_range(self):
        results = self.ht.parameter_sweep("beta", [0.5, 3.0], fixed_kwargs=self.fixed)
        for r in results:
            assert 0.0 <= r["ctai"] <= 1.0

    def test_sweep_scenario(self):
        fixed_no_scenario = dict(n_agents=12, T=6, seed=5)
        results = self.ht.parameter_sweep(
            "scenario",
            ["random", "polarized"],
            fixed_kwargs=fixed_no_scenario,
        )
        assert len(results) == 2
        assert results[0]["param_value"] == "random"
        assert results[1]["param_value"] == "polarized"

    def test_sweep_inherits_simulator_base(self):
        sim = SocialSimulator(**self.fixed)
        ht = HypothesisTester(
            simulator=sim,
            theory_configs=_FAST_THEORIES,
            t_max=3.0,
            dt=1.0,
        )
        results = ht.parameter_sweep("beta", [1.0, 3.0])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# temporal_prediction
# ---------------------------------------------------------------------------


class TestTemporalPrediction:
    def setup_method(self):
        self.sim = SocialSimulator(n_agents=15, T=20, seed=3)
        self.ht = HypothesisTester(
            simulator=self.sim,
            theory_configs=_FAST_THEORIES,
            t_max=3.0,
            dt=1.0,
        )

    def test_returns_required_keys(self):
        result = self.ht.temporal_prediction(t_freeze=5, t_predict=15)
        assert "scores" in result
        assert "t_freeze" in result
        assert "t_predict" in result
        assert "n_train_interactions" in result
        assert "n_eval_interactions" in result

    def test_t_values_preserved(self):
        result = self.ht.temporal_prediction(t_freeze=6, t_predict=14)
        assert result["t_freeze"] == 6
        assert result["t_predict"] == 14

    def test_train_less_than_eval_when_early_freeze(self):
        result = self.ht.temporal_prediction(t_freeze=3, t_predict=18)
        # Training on first 3 steps, evaluating on 3-18
        assert result["n_train_interactions"] < result["n_eval_interactions"]

    def test_pas_in_range(self):
        result = self.ht.temporal_prediction(t_freeze=5, t_predict=15)
        for name, s in result["scores"].items():
            assert 0.0 <= s["pas"] <= 1.0

    def test_t_freeze_negative_raises(self):
        with pytest.raises(ValueError, match="t_freeze"):
            self.ht.temporal_prediction(t_freeze=-1, t_predict=10)

    def test_t_predict_not_gt_t_freeze_raises(self):
        with pytest.raises(ValueError, match="t_predict"):
            self.ht.temporal_prediction(t_freeze=10, t_predict=5)

    def test_t_freeze_ge_T_raises(self):
        with pytest.raises(ValueError, match="t_freeze"):
            self.ht.temporal_prediction(t_freeze=25, t_predict=30)


# ---------------------------------------------------------------------------
# test_claim
# ---------------------------------------------------------------------------


class TestTestClaim:
    def setup_method(self):
        self.ht = HypothesisTester(theory_configs=_FAST_THEORIES, t_max=3.0, dt=1.0)

    def test_unknown_claim_raises(self):
        with pytest.raises(ValueError, match="Unknown claim"):
            self.ht.test_claim("nonexistent_claim")

    def test_all_agree_clustered_passes_low_threshold(self):
        # With any clustered data, CTAI > very low threshold should pass
        result = self.ht.test_claim(
            "all_theories_agree_on_clustered",
            ctai_threshold=0.001,
            beta=4.0,
            n_agents=15,
        )
        assert result["passed"] is True

    def test_all_agree_clustered_fails_high_threshold(self):
        # No scenario produces CTAI > 0.99
        result = self.ht.test_claim(
            "all_theories_agree_on_clustered",
            ctai_threshold=0.99,
            beta=0.1,
            n_agents=15,
        )
        assert result["passed"] is False

    def test_claim_returns_evidence(self):
        result = self.ht.test_claim(
            "all_theories_agree_on_clustered",
            ctai_threshold=0.5,
            n_agents=15,
        )
        assert "passed" in result
        assert "evidence" in result
        assert "ctai" in result["evidence"]

    def test_tst_phase_transition_returns_evidence(self):
        result = self.ht.test_claim(
            "tst_phase_transition",
            n_temps=4,
            n_agents=15,
            temp_range=(0.1, 2.0),
        )
        assert "passed" in result
        assert "temperatures" in result["evidence"]
        assert "group_counts" in result["evidence"]
        assert len(result["evidence"]["temperatures"]) == 4

    def test_cft_fewer_groups_returns_evidence(self):
        result = self.ht.test_claim(
            "cft_fewer_groups_than_gft_on_polarized",
            n_trials=2,
            n_agents=15,
        )
        assert "cft_group_counts" in result["evidence"]
        assert "gft_group_counts" in result["evidence"]
        assert len(result["evidence"]["cft_group_counts"]) == 2

    def test_theory_x_best_unknown_theory_raises(self):
        with pytest.raises(ValueError, match="Theory"):
            self.ht.test_claim(
                "theory_x_best_on_scenario",
                theory="NONEXISTENT",
                scenario="random",
            )

    def test_theory_x_best_returns_evidence(self):
        result = self.ht.test_claim(
            "theory_x_best_on_scenario",
            theory="CFT",
            scenario="random",
            n_trials=2,
            n_agents=12,
        )
        assert "win_count" in result["evidence"]
        assert "trial_winners" in result["evidence"]
        assert len(result["evidence"]["trial_winners"]) == 2


# ---------------------------------------------------------------------------
# Issue #23 - temporal split in compare_theories
# ---------------------------------------------------------------------------


class TestCompareTheoriesTemporalSplit:
    def setup_method(self):
        self.sim = _fast_sim(T=16, n_agents=15)
        self.ht = HypothesisTester(
            simulator=self.sim,
            theory_configs=_FAST_THEORIES,
            t_max=3.0,
            dt=1.0,
        )

    def test_returns_temporal_split_key_false(self):
        result = self.ht.compare_theories(use_temporal_split=False)
        assert result["temporal_split"] is False

    def test_returns_temporal_split_key_true(self):
        result = self.ht.compare_theories(use_temporal_split=True)
        assert result["temporal_split"] is True

    def test_temporal_split_runs_without_error(self):
        result = self.ht.compare_theories(use_temporal_split=True, split_fraction=0.5)
        assert "scores" in result
        assert set(result["scores"].keys()) == {"CFT", "GFT", "TST"}

    def test_temporal_split_pas_in_range(self):
        result = self.ht.compare_theories(use_temporal_split=True)
        for name, s in result["scores"].items():
            assert 0.0 <= s["pas"] <= 1.0, f"{name} PAS out of range"

    def test_split_fraction_respected(self):
        # With a very early split, ground truth is richer than affinity
        result = self.ht.compare_theories(
            use_temporal_split=True, split_fraction=0.3
        )
        assert "scores" in result

    def test_theory_configs_not_mutated_across_temporal_calls(self):
        self.ht.compare_theories(use_temporal_split=True)
        self.ht.compare_theories(use_temporal_split=True)
        for k in self.ht.theory_configs:
            assert "class" in self.ht.theory_configs[k]


# ---------------------------------------------------------------------------
# Issue #27 - n_runs statistical validation
# ---------------------------------------------------------------------------


class TestNRuns:
    def setup_method(self):
        self.sim = _fast_sim(T=10, n_agents=12)
        self.ht = HypothesisTester(
            simulator=self.sim,
            theory_configs=_FAST_THEORIES,
            t_max=2.0,
            dt=1.0,
        )

    def test_multi_run_returns_n_runs_key(self):
        result = self.ht.compare_theories(n_runs=3)
        assert result["n_runs"] == 3

    def test_multi_run_scores_per_run_length(self):
        result = self.ht.compare_theories(n_runs=3)
        assert len(result["scores_per_run"]) == 3

    def test_multi_run_mean_std_keys(self):
        result = self.ht.compare_theories(n_runs=3)
        assert "mean_similarity" in result
        assert "std_similarity" in result
        assert set(result["mean_similarity"].keys()) == {"CFT", "GFT", "TST"}
        assert set(result["std_similarity"].keys()) == {"CFT", "GFT", "TST"}

    def test_multi_run_mean_in_range(self):
        result = self.ht.compare_theories(n_runs=3)
        for name, v in result["mean_similarity"].items():
            assert 0.0 <= v <= 1.0, f"{name} mean_similarity={v} out of range"

    def test_multi_run_rankings_is_list(self):
        result = self.ht.compare_theories(n_runs=3)
        assert isinstance(result["rankings"], list)
        assert set(result["rankings"]) == {"CFT", "GFT", "TST"}

    def test_multi_run_temporal_split_key(self):
        result = self.ht.compare_theories(n_runs=3, use_temporal_split=True)
        assert result["temporal_split"] is True

    def test_multi_run_wilcoxon_none_for_small_n(self):
        # Wilcoxon requires n_runs >= 5
        result = self.ht.compare_theories(n_runs=3)
        assert result["wilcoxon_pvalue"] is None

    def test_temporal_prediction_multi_run(self):
        sim = SocialSimulator(n_agents=12, T=20, seed=7)
        ht = HypothesisTester(
            simulator=sim,
            theory_configs=_FAST_THEORIES,
            t_max=2.0,
            dt=1.0,
        )
        result = ht.temporal_prediction(t_freeze=5, t_predict=15, n_runs=3)
        assert result["n_runs"] == 3
        assert len(result["scores_per_run"]) == 3
        assert result["t_freeze"] == 5
        assert result["t_predict"] == 15
        assert "mean_similarity" in result
        assert "std_similarity" in result


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline_runs_without_error(self):
        """Regression guard: full compare_theories on polarized data."""
        sim = SocialSimulator(n_agents=20, scenario="polarized", T=15, seed=0)
        ht = HypothesisTester(
            simulator=sim,
            theory_configs=_FAST_THEORIES,
            t_max=4.0,
            dt=1.0,
        )
        result = ht.compare_theories()
        assert len(result["scores"]) == 3
        assert all(0.0 <= s["pas"] <= 1.0 for s in result["scores"].values())
        assert result["affinity_matrix"].shape == (20, 20)
