"""Tests for TheoryComparator."""

import numpy as np
import pytest

from cft import TheoryComparator, Group, CFT, GFT, QST, TST, Agent, TheoryParameters


class TestJaccard:
    def test_identical_groups(self):
        groups = [Group(id=0, members=[0, 1, 2]), Group(id=1, members=[3, 4])]
        sim = TheoryComparator.compare_group_structures(groups, groups, metric="jaccard")
        assert sim == 1.0

    def test_completely_different_groups(self):
        g1 = [Group(id=0, members=[0, 1]), Group(id=1, members=[2, 3])]
        g2 = [Group(id=0, members=[0, 2]), Group(id=1, members=[1, 3])]
        sim = TheoryComparator.compare_group_structures(g1, g2, metric="jaccard")
        assert 0.0 < sim < 1.0

    def test_empty_groups(self):
        sim = TheoryComparator.compare_group_structures([], [], metric="jaccard")
        assert sim == 0.0


class TestNMI:
    def test_identical_partitions(self):
        groups = [Group(id=0, members=[0, 1, 2]), Group(id=1, members=[3, 4])]
        nmi = TheoryComparator.compare_group_structures(groups, groups, metric="nmi")
        assert nmi == pytest.approx(1.0, abs=1e-10)

    def test_single_group_vs_single_group(self):
        """Both have one group → NMI = 1.0."""
        g1 = [Group(id=0, members=[0, 1, 2, 3])]
        g2 = [Group(id=0, members=[0, 1, 2, 3])]
        nmi = TheoryComparator.compare_group_structures(g1, g2, metric="nmi")
        assert nmi == pytest.approx(1.0, abs=1e-10)

    def test_perfect_split_vs_single(self):
        """Two groups vs one group → NMI = 0."""
        g1 = [Group(id=0, members=[0, 1]), Group(id=1, members=[2, 3])]
        g2 = [Group(id=0, members=[0, 1, 2, 3])]
        nmi = TheoryComparator.compare_group_structures(g1, g2, metric="nmi")
        assert nmi == pytest.approx(0.0, abs=1e-10)

    def test_empty_partitions(self):
        nmi = TheoryComparator.compare_group_structures([], [], metric="nmi")
        assert nmi == 0.0


class TestPairAgreement:
    def test_identical_partitions(self):
        groups = [Group(id=0, members=[0, 1, 2]), Group(id=1, members=[3, 4])]
        score = TheoryComparator.compare_group_structures(groups, groups, metric="pair_agreement")
        assert score == 1.0

    def test_completely_swapped(self):
        """Swap two agents between groups - some pairs disagree."""
        g1 = [Group(id=0, members=[0, 1]), Group(id=1, members=[2, 3])]
        g2 = [Group(id=0, members=[0, 2]), Group(id=1, members=[1, 3])]
        score = TheoryComparator.compare_group_structures(g1, g2, metric="pair_agreement")
        # Pairs: (0,1) same in g1, diff in g2; (0,2) diff in g1, same in g2;
        # (0,3) diff/diff=agree; (1,2) diff/diff=agree; (1,3) diff/same;
        # (2,3) same/diff → 2 agree out of 6
        assert score == pytest.approx(2 / 6, abs=1e-10)

    def test_single_agent(self):
        g = [Group(id=0, members=[0])]
        score = TheoryComparator.compare_group_structures(g, g, metric="pair_agreement")
        assert score == 1.0


class TestInvalidMetric:
    def test_unknown_metric_raises(self):
        groups = [Group(id=0, members=[0])]
        with pytest.raises(ValueError, match="Unknown metric"):
            TheoryComparator.compare_group_structures(groups, groups, metric="invalid")


class TestGroupsEqual:
    def test_identical(self):
        g = [Group(id=0, members=[0, 1, 2]), Group(id=1, members=[3, 4])]
        assert TheoryComparator._groups_equal(g, g)

    def test_different_ids(self):
        g1 = [Group(id=0, members=[0, 1]), Group(id=1, members=[2, 3])]
        g2 = [Group(id=5, members=[0, 1]), Group(id=9, members=[2, 3])]
        assert TheoryComparator._groups_equal(g1, g2)

    def test_not_equal(self):
        g1 = [Group(id=0, members=[0, 1]), Group(id=1, members=[2, 3])]
        g2 = [Group(id=0, members=[0, 2]), Group(id=1, members=[1, 3])]
        assert not TheoryComparator._groups_equal(g1, g2)


class TestAnalyzePredictions:
    def test_basic_analysis(self):
        results = {
            "A": [
                {"time": 1, "groups": [Group(id=0, members=[0, 1, 2])]},
                {"time": 2, "groups": [Group(id=0, members=[0, 1, 2])]},
            ],
            "B": [
                {"time": 1, "groups": [Group(id=0, members=[0, 1]), Group(id=1, members=[2])]},
                {"time": 2, "groups": [Group(id=0, members=[0, 1, 2])]},
            ],
        }
        analysis = TheoryComparator.analyze_predictions(results)
        assert "convergence_times" in analysis
        assert "final_group_counts" in analysis
        assert "cross_theory_agreement" in analysis
        assert "stability_scores" in analysis
        assert analysis["final_group_counts"]["A"] == 1
        assert analysis["final_group_counts"]["B"] == 1
        # A converges immediately, B converges at step 2
        assert analysis["convergence_times"]["A"] == 2
        assert analysis["stability_scores"]["A"] == 1.0

    def test_nmi_metric_in_analysis(self):
        results = {
            "X": [{"time": 1, "groups": [Group(id=0, members=[0, 1, 2])]}],
            "Y": [{"time": 1, "groups": [Group(id=0, members=[0, 1, 2])]}],
        }
        analysis = TheoryComparator.analyze_predictions(results, metric="nmi")
        assert analysis["cross_theory_agreement"]["X-Y"] == pytest.approx(1.0, abs=1e-10)


class TestTheoryDiagnostics:
    def test_cft_diagnostics(self, small_agents, small_params):
        theory = CFT(small_params, threshold=0.5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=1.0, dt=1.0)
        diag = TheoryComparator.theory_diagnostics(theory)
        assert diag["theory_type"] == "CFT"
        assert diag["n_agents"] == 10
        assert diag["n_groups"] > 0
        assert len(diag["group_sizes"]) == diag["n_groups"]

    def test_qst_diagnostics(self, small_agents, small_params):
        theory = QST(small_params, n_states=5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=2.0, dt=1.0)
        diag = TheoryComparator.theory_diagnostics(theory)
        assert diag["theory_type"] == "QST"
        assert "avg_entanglement" in diag
        assert "avg_state_entropy" in diag

    def test_tst_diagnostics(self, small_agents, small_params):
        theory = TST(small_params, temperature=1.0)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=2.0, dt=1.0)
        diag = TheoryComparator.theory_diagnostics(theory)
        assert diag["theory_type"] == "TST"
        assert "temperature" in diag
        assert "energy" in diag
        assert "free_energy" in diag

    def test_gft_diagnostics(self, small_agents, small_params):
        theory = GFT(small_params, k=0.1, sigma=1.0)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=2.0, dt=1.0)
        diag = TheoryComparator.theory_diagnostics(theory)
        assert diag["theory_type"] == "GFT"
        assert "avg_distance" in diag
        assert "position_spread" in diag
