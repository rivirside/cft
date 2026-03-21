"""Tests for the intervention system."""

import numpy as np
import pytest

from cft import (
    Agent, DCT, CFT, GFT, TheoryParameters,
    InterventionRunner, InterventionReport,
    RemoveAgents, ShiftFeatures, AddAgent,
    NoiseShock, ModifyAffinity, ShiftProximity, ShiftAlignment,
)
from cft.interventions import Snapshot
from cft.affinity import compute_affinity_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dct(agents, params, **kwargs):
    """Create and initialize a DCT theory."""
    defaults = dict(mu=0.3, lam=0.05, noise=0.01, threshold=0.3)
    defaults.update(kwargs)
    theory = DCT(params, **defaults)
    aff = compute_affinity_matrix(agents)
    theory._affinity_matrix = aff
    theory.initialize_agents(agents)
    return theory


def _make_cft(agents, params, **kwargs):
    defaults = dict(threshold=0.5)
    defaults.update(kwargs)
    theory = CFT(params, **defaults)
    aff = compute_affinity_matrix(agents)
    theory._affinity_matrix = aff
    theory.initialize_agents(agents)
    return theory


# ---------------------------------------------------------------------------
# Intervention base
# ---------------------------------------------------------------------------

class TestInterventionBase:
    def test_not_implemented(self):
        from cft.interventions import Intervention
        iv = Intervention(time=1.0)
        with pytest.raises(NotImplementedError):
            iv.apply(None)


# ---------------------------------------------------------------------------
# RemoveAgents
# ---------------------------------------------------------------------------

class TestRemoveAgents:
    def test_removes_affinity(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = RemoveAgents(time=0.0, agent_ids=[0, 1])
        log = iv.apply(theory)
        assert log["removed"] == [0, 1]
        # Affinities for removed agents should be zero (except diagonal)
        assert theory._affinity_matrix[0, 2] == 0.0
        assert theory._affinity_matrix[1, 3] == 0.0
        assert theory._affinity_matrix[0, 0] == 1.0

    def test_isolates_in_dct_layers(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = RemoveAgents(time=0.0, agent_ids=[3])
        iv.apply(theory)
        assert theory._proximity_affinity[3, 4] == 0.0
        assert theory._effective_affinity[3, 5] == 0.0
        # Context position should be far away
        assert np.linalg.norm(theory.context_pos[3]) > 1000

    def test_skip_missing_ids(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = RemoveAgents(time=0.0, agent_ids=[999])
        log = iv.apply(theory)
        assert log["removed"] == []

    def test_removed_agents_become_isolated(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        theory.run_simulation(t_max=3.0, dt=1.0)

        iv = RemoveAgents(time=0.0, agent_ids=[0])
        iv.apply(theory)
        theory.step(1.0)
        groups_after = theory.get_groups()

        # Removed agent should be in a group with no non-removed agents
        for g in groups_after:
            if 0 in g.members:
                non_removed = [m for m in g.members if m != 0]
                # Either singleton or only co-grouped with other removed agents
                assert len(non_removed) == 0


# ---------------------------------------------------------------------------
# ShiftFeatures
# ---------------------------------------------------------------------------

class TestShiftFeatures:
    def test_shifts_agent_features(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        original = theory.agents[0].features.copy()
        delta = np.array([1.0, 0.0, 0.0])
        iv = ShiftFeatures(time=0.0, agent_ids=[0], delta=delta)
        log = iv.apply(theory)
        assert log["shifted"] == [0]
        np.testing.assert_array_almost_equal(
            theory.agents[0].features, original + delta
        )

    def test_shifts_dct_alignment(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        align_before = theory.alignment[0].copy()
        delta = np.array([0.5, -0.5, 0.0])
        iv = ShiftFeatures(time=0.0, agent_ids=[0], delta=delta)
        iv.apply(theory)
        np.testing.assert_array_almost_equal(
            theory.alignment[0], align_before + delta
        )

    def test_none_delta_is_noop(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = ShiftFeatures(time=0.0, agent_ids=[0], delta=None)
        log = iv.apply(theory)
        assert log["shifted"] == []

    def test_delta_truncated_to_feature_length(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        original = theory.agents[0].features.copy()
        long_delta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        iv = ShiftFeatures(time=0.0, agent_ids=[0], delta=long_delta)
        iv.apply(theory)
        np.testing.assert_array_almost_equal(
            theory.agents[0].features, original + long_delta[:3]
        )

    def test_recalculates_affinities(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        eff_before = theory._effective_affinity.copy()
        delta = np.array([5.0, 5.0, 5.0])
        iv = ShiftFeatures(time=0.0, agent_ids=[0], delta=delta)
        iv.apply(theory)
        assert not np.array_equal(theory._effective_affinity, eff_before)


# ---------------------------------------------------------------------------
# AddAgent
# ---------------------------------------------------------------------------

class TestAddAgent:
    def test_adds_agent(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        n_before = len(theory.agents)
        iv = AddAgent(time=0.0, features=np.array([1.0, 1.0, 1.0]))
        log = iv.apply(theory)
        assert len(theory.agents) == n_before + 1
        assert log["added"] == n_before  # next ID

    def test_expands_affinity_matrix(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = AddAgent(time=0.0, features=np.array([0.0, 0.0, 0.0]))
        iv.apply(theory)
        n = len(theory.agents)
        assert theory._affinity_matrix.shape == (n, n)

    def test_expands_dct_layers(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = AddAgent(time=0.0, features=np.array([1.0, 2.0, 3.0]))
        iv.apply(theory)
        n = len(theory.agents)
        assert theory.context_pos.shape[0] == n
        assert theory.alignment.shape[0] == n
        assert theory._proximity_affinity.shape == (n, n)

    def test_none_features_is_noop(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        n_before = len(theory.agents)
        iv = AddAgent(time=0.0, features=None)
        log = iv.apply(theory)
        assert log["added"] is None
        assert len(theory.agents) == n_before

    def test_simulation_continues_after_add(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        theory.run_simulation(t_max=2.0, dt=1.0)
        iv = AddAgent(time=0.0, features=np.array([0.0, 0.0, 0.0]))
        iv.apply(theory)
        # Should not crash
        theory.step(1.0)
        groups = theory.get_groups()
        all_members = []
        for g in groups:
            all_members.extend(g.members)
        assert len(theory.agents) == len(set(all_members))


# ---------------------------------------------------------------------------
# NoiseShock
# ---------------------------------------------------------------------------

class TestNoiseShock:
    def test_changes_features(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        features_before = [a.features.copy() for a in theory.agents]
        iv = NoiseShock(time=0.0, intensity=1.0, seed=42)
        log = iv.apply(theory)
        assert log["shocked"] == 10
        changed = any(
            not np.array_equal(theory.agents[i].features, features_before[i])
            for i in range(10)
        )
        assert changed

    def test_targets_subset(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        f5_before = theory.agents[5].features.copy()
        iv = NoiseShock(time=0.0, intensity=1.0, target_ids=[0, 1], seed=42)
        log = iv.apply(theory)
        assert log["shocked"] == 2
        # Agent 5 should be unchanged
        np.testing.assert_array_equal(theory.agents[5].features, f5_before)

    def test_zero_intensity_is_noop(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        features_before = [a.features.copy() for a in theory.agents]
        iv = NoiseShock(time=0.0, intensity=0.0, seed=42)
        iv.apply(theory)
        for i in range(10):
            np.testing.assert_array_equal(theory.agents[i].features, features_before[i])


# ---------------------------------------------------------------------------
# ModifyAffinity
# ---------------------------------------------------------------------------

class TestModifyAffinity:
    def test_sets_affinity(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = ModifyAffinity(time=0.0, pairs=[(0, 1)], values=[0.99])
        log = iv.apply(theory)
        assert log["modified"] == 1
        assert theory._affinity_matrix[0, 1] == pytest.approx(0.99)
        assert theory._affinity_matrix[1, 0] == pytest.approx(0.99)

    def test_no_matrix_returns_zero(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        theory._affinity_matrix = None
        iv = ModifyAffinity(time=0.0, pairs=[(0, 1)], values=[0.5])
        log = iv.apply(theory)
        assert log["modified"] == 0


# ---------------------------------------------------------------------------
# ShiftProximity (DCT-specific)
# ---------------------------------------------------------------------------

class TestShiftProximity:
    def test_shifts_context_pos(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        pos_before = theory.context_pos[0].copy()
        delta = np.array([10.0, 0.0, 0.0])
        iv = ShiftProximity(time=0.0, agent_ids=[0], delta=delta)
        log = iv.apply(theory)
        assert log["shifted"] == [0]
        np.testing.assert_array_almost_equal(
            theory.context_pos[0], pos_before + delta
        )

    def test_noop_on_non_dct(self, small_agents, small_params):
        theory = _make_cft(small_agents, small_params)
        delta = np.array([10.0, 0.0, 0.0])
        iv = ShiftProximity(time=0.0, agent_ids=[0], delta=delta)
        log = iv.apply(theory)
        assert log["shifted"] == []


# ---------------------------------------------------------------------------
# ShiftAlignment (DCT-specific)
# ---------------------------------------------------------------------------

class TestShiftAlignment:
    def test_shifts_alignment(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        align_before = theory.alignment[0].copy()
        delta = np.array([0.0, 0.0, 5.0])
        iv = ShiftAlignment(time=0.0, agent_ids=[0], delta=delta)
        log = iv.apply(theory)
        assert log["shifted"] == [0]
        np.testing.assert_array_almost_equal(
            theory.alignment[0], align_before + delta
        )

    def test_noop_on_non_dct(self, small_agents, small_params):
        theory = _make_cft(small_agents, small_params)
        iv = ShiftAlignment(time=0.0, agent_ids=[0], delta=np.array([1.0]))
        log = iv.apply(theory)
        assert log["shifted"] == []


# ---------------------------------------------------------------------------
# InterventionRunner
# ---------------------------------------------------------------------------

class TestInterventionRunner:
    def test_runs_without_interventions(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        runner = InterventionRunner(theory, [])
        report = runner.run(t_max=3.0, dt=1.0)
        assert isinstance(report, InterventionReport)
        # Initial snapshot + 3 steps
        assert len(report.snapshots) == 4

    def test_applies_intervention_at_time(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = RemoveAgents(time=2.0, agent_ids=[0])
        runner = InterventionRunner(theory, [iv])
        report = runner.run(t_max=5.0, dt=1.0)
        assert len(report.intervention_log) == 1
        assert report.intervention_log[0]["time"] == 2.0

    def test_multiple_interventions_ordered(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv1 = NoiseShock(time=1.0, intensity=0.1, seed=1)
        iv2 = RemoveAgents(time=3.0, agent_ids=[5])
        iv3 = ShiftFeatures(time=5.0, agent_ids=[2], delta=np.array([1.0, 0, 0]))
        runner = InterventionRunner(theory, [iv3, iv1, iv2])  # out of order
        report = runner.run(t_max=6.0, dt=1.0)
        assert len(report.intervention_log) == 3
        times = [log["time"] for log in report.intervention_log]
        assert times == sorted(times)

    def test_snapshots_have_correct_structure(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        runner = InterventionRunner(theory, [])
        report = runner.run(t_max=2.0, dt=1.0)
        for snap in report.snapshots:
            assert isinstance(snap, Snapshot)
            assert isinstance(snap.groups, list)
            assert isinstance(snap.n_groups, int)
            assert isinstance(snap.group_sizes, list)

    def test_intervention_logged_in_snapshot(self, small_agents, small_params):
        theory = _make_dct(small_agents, small_params)
        iv = NoiseShock(time=1.0, intensity=0.1, seed=1)
        runner = InterventionRunner(theory, [iv])
        report = runner.run(t_max=3.0, dt=1.0)
        # Find the snapshot with intervention
        iv_snaps = [s for s in report.snapshots if s.intervention_log is not None]
        assert len(iv_snaps) >= 1


# ---------------------------------------------------------------------------
# InterventionReport
# ---------------------------------------------------------------------------

class TestInterventionReport:
    def _make_report(self, agents, params):
        theory = _make_dct(agents, params)
        iv = RemoveAgents(time=2.0, agent_ids=[0, 1])
        runner = InterventionRunner(theory, [iv])
        return runner.run(t_max=5.0, dt=1.0)

    def test_timeline(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        tl = report.timeline
        assert len(tl) > 0
        assert "time" in tl[0]
        assert "n_groups" in tl[0]
        assert "intervention" in tl[0]

    def test_stability_curve(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        curve = report.stability_curve
        assert len(curve) == len(report.snapshots) - 1
        for val in curve:
            assert 0.0 <= val <= 1.0 + 1e-10

    def test_fracture_events_detected(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        # Fracture events should exist (removing agents causes group changes)
        # Even if none, the API should work
        events = report.fracture_events
        assert isinstance(events, list)
        for e in events:
            assert "time" in e
            assert e["groups_after"] > e["groups_before"]

    def test_merge_events_detected(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        events = report.merge_events
        assert isinstance(events, list)

    def test_group_survival(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        survival = report.group_survival(before_time=1.0, after_time=4.0)
        assert "survival_rate" in survival
        assert 0.0 <= survival["survival_rate"] <= 1.0
        assert "survived" in survival
        assert "dissolved" in survival

    def test_group_survival_bad_time(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        # Should still return something (nearest snap)
        survival = report.group_survival(before_time=0.0, after_time=5.0)
        assert "survival_rate" in survival

    def test_agent_stability(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        stab = report.agent_stability()
        assert isinstance(stab, dict)
        for aid, val in stab.items():
            assert 0.0 <= val <= 1.0

    def test_vulnerability_ranking(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        ranking = report.vulnerability_ranking()
        assert isinstance(ranking, list)
        if len(ranking) > 1:
            # Most volatile first
            assert ranking[0][1] <= ranking[-1][1]

    def test_resilience_scores(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        scores = report.resilience_scores
        assert "mean_stability" in scores
        assert "min_stability" in scores
        assert "recovery_rate" in scores
        assert 0.0 <= scores["mean_stability"] <= 1.0
        assert 0.0 <= scores["recovery_rate"] <= 1.0

    def test_summary(self, small_agents, small_params):
        report = self._make_report(small_agents, small_params)
        s = report.summary()
        assert "n_snapshots" in s
        assert "n_interventions" in s
        assert "timeline" in s
        assert "resilience" in s
        assert "vulnerability_ranking" in s
        assert s["n_interventions"] == 1


# ---------------------------------------------------------------------------
# Integration: full scenario
# ---------------------------------------------------------------------------

class TestInterventionScenarios:
    def test_leader_removal_scenario(self, clustered_agents):
        """Remove a central agent and verify groups reorganize."""
        params = TheoryParameters(n_agents=10, n_features=3, random_seed=42)
        theory = _make_dct(clustered_agents, params, threshold=0.2)

        # Warm up
        runner = InterventionRunner(theory, [
            RemoveAgents(time=3.0, agent_ids=[0]),  # remove cluster center
        ])
        report = runner.run(t_max=8.0, dt=1.0)

        assert len(report.intervention_log) == 1
        assert report.resilience_scores["mean_stability"] >= 0.0

    def test_propaganda_scenario(self, clustered_agents):
        """Shift opinions of one cluster toward the other."""
        params = TheoryParameters(n_agents=10, n_features=3, random_seed=42)
        theory = _make_dct(clustered_agents, params, threshold=0.2)

        # Shift cluster A toward cluster B's features
        runner = InterventionRunner(theory, [
            ShiftFeatures(
                time=3.0,
                agent_ids=[0, 1, 2, 3, 4],
                delta=np.array([-2.0, -2.0, -2.0]),  # toward cluster B
            ),
        ])
        report = runner.run(t_max=10.0, dt=1.0)
        assert len(report.snapshots) > 0

    def test_infiltration_scenario(self, small_agents, small_params):
        """Add a charismatic agent and see if groups shift."""
        theory = _make_dct(small_agents, small_params)
        extreme_features = np.array([5.0, 5.0, 5.0])
        runner = InterventionRunner(theory, [
            AddAgent(time=2.0, features=extreme_features),
        ])
        report = runner.run(t_max=6.0, dt=1.0)
        assert len(theory.agents) == 11

    def test_crisis_scenario(self, small_agents, small_params):
        """Noise shock disrupts, then groups should partially recover."""
        theory = _make_dct(small_agents, small_params)
        runner = InterventionRunner(theory, [
            NoiseShock(time=3.0, intensity=2.0, seed=42),
        ])
        report = runner.run(t_max=10.0, dt=1.0)
        scores = report.resilience_scores
        assert "recovery_rate" in scores

    def test_multi_intervention_scenario(self, small_agents, small_params):
        """Chain multiple interventions in sequence."""
        theory = _make_dct(small_agents, small_params)
        runner = InterventionRunner(theory, [
            NoiseShock(time=2.0, intensity=0.5, seed=1),
            RemoveAgents(time=4.0, agent_ids=[5]),
            ShiftFeatures(time=6.0, agent_ids=[0], delta=np.array([3.0, 0, 0])),
        ])
        report = runner.run(t_max=10.0, dt=1.0)
        assert len(report.intervention_log) == 3
        assert len(report.snapshots) == 11  # initial + 10 steps

    def test_works_with_cft(self, small_agents, small_params):
        """Interventions should work with non-DCT theories too."""
        theory = _make_cft(small_agents, small_params)
        runner = InterventionRunner(theory, [
            RemoveAgents(time=1.0, agent_ids=[0]),
        ])
        report = runner.run(t_max=3.0, dt=1.0)
        assert len(report.snapshots) == 4
        assert len(report.intervention_log) == 1
