"""Tests for PredictionTournament."""

import numpy as np
import pytest

from cft import (
    PredictionTournament, Agent, Group, TheoryParameters,
    CFT, GFT, TST, QST, ICT,
)


@pytest.fixture
def tournament_setup():
    """Create agents, params, and a ground truth for tournament testing."""
    rng = np.random.default_rng(42)
    agents = [Agent(id=i, features=rng.standard_normal(3)) for i in range(15)]
    params = TheoryParameters(n_agents=15, n_features=3, random_seed=42)
    # Ground truth: 3 groups of 5
    ground_truth = [
        Group(id=0, members=[0, 1, 2, 3, 4]),
        Group(id=1, members=[5, 6, 7, 8, 9]),
        Group(id=2, members=[10, 11, 12, 13, 14]),
    ]
    return agents, params, ground_truth


class TestPredictionTournament:
    def test_add_and_run(self, tournament_setup):
        agents, params, _ = tournament_setup
        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        t.add_theory("GFT", GFT, k=0.1, sigma=1.0)

        histories = t.run(t_max=5.0, dt=1.0)
        assert "CFT" in histories
        assert "GFT" in histories
        assert len(histories["CFT"]) > 0

    def test_score_returns_pas(self, tournament_setup):
        agents, params, ground_truth = tournament_setup
        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        t.run(t_max=5.0, dt=1.0)

        scores = t.score(ground_truth)
        assert "CFT" in scores
        assert "pas" in scores["CFT"]
        assert "similarity" in scores["CFT"]
        assert "group_count_accuracy" in scores["CFT"]
        assert "size_accuracy" in scores["CFT"]
        assert 0.0 <= scores["CFT"]["pas"] <= 1.0

    def test_score_perfect_match(self):
        """Scoring against own output should give high PAS."""
        rng = np.random.default_rng(99)
        agents = [Agent(id=i, features=rng.standard_normal(3)) for i in range(10)]
        params = TheoryParameters(n_agents=10, n_features=3, random_seed=99)

        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        t.run(t_max=2.0, dt=1.0)

        # Score against own output
        own_groups = t._theories["CFT"].get_groups()
        scores = t.score(own_groups)
        assert scores["CFT"]["pas"] == pytest.approx(1.0, abs=1e-10)

    def test_rankings(self, tournament_setup):
        agents, params, ground_truth = tournament_setup
        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        t.add_theory("TST", TST, temperature=0.5, sweeps_per_step=5)
        t.run(t_max=5.0, dt=1.0)

        ranked = t.rankings(ground_truth)
        assert len(ranked) == 2
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2
        assert ranked[0]["score"] >= ranked[1]["score"]
        assert "diagnostics" in ranked[0]

    def test_compute_ctai(self, tournament_setup):
        agents, params, _ = tournament_setup
        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        t.add_theory("TST", TST, temperature=0.5, sweeps_per_step=5)
        t.run(t_max=5.0, dt=1.0)

        ctai = t.compute_ctai()
        assert 0.0 <= ctai <= 1.0

    def test_compute_pss(self, tournament_setup):
        agents, params, _ = tournament_setup
        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        t.run(t_max=5.0, dt=1.0)

        pss = t.compute_pss("CFT", "threshold", 0.5, perturbation=0.1)
        assert 0.0 <= pss <= 1.0

    def test_compute_dfi(self, tournament_setup):
        agents, params, _ = tournament_setup
        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        histories = t.run(t_max=5.0, dt=1.0)

        # Use theory's own history as ground truth (should give DFI ≈ 1.0)
        gt_history = [h["groups"] for h in histories["CFT"]]
        dfi = t.compute_dfi(gt_history)
        assert dfi["CFT"] == pytest.approx(1.0, abs=1e-10)

    def test_results_dict(self, tournament_setup):
        agents, params, _ = tournament_setup
        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        t.run(t_max=2.0, dt=1.0)

        result = t.results_dict()
        assert "histories" in result
        assert "final_groups" in result
        assert "diagnostics" in result
        assert "CFT" in result["final_groups"]

    def test_all_five_theories(self, tournament_setup):
        """Smoke test: all 5 theories in a tournament."""
        agents, params, ground_truth = tournament_setup
        t = PredictionTournament(agents, params)
        t.add_theory("CFT", CFT, threshold=0.5)
        t.add_theory("GFT", GFT, k=0.1, sigma=1.0)
        t.add_theory("QST", QST, n_states=5)
        t.add_theory("ICT", ICT, bandwidth=3)
        t.add_theory("TST", TST, temperature=0.5, sweeps_per_step=5)

        t.run(t_max=5.0, dt=1.0)
        scores = t.score(ground_truth)
        assert len(scores) == 5

        ranked = t.rankings(ground_truth)
        assert len(ranked) == 5
        assert ranked[0]["rank"] == 1
