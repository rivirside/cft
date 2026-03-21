"""Tests for DCT advanced features: TraitMap, separate sources, sustained interventions."""

import numpy as np
import pytest

from cft import (
    Agent, DCT, TheoryParameters, InterventionRunner,
    SustainedShift, SustainedNoise, SustainedAffinityBias, TraitMap,
)
from cft.interventions import SustainedIntervention
from cft.affinity import compute_affinity_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agents_with_varied_features(n=10, seed=42):
    """Agents with diverse features (simulates MBTI + opinions + influence)."""
    rng = np.random.default_rng(seed)
    agents = []
    for i in range(n):
        # [E/I, S/N, T/F, J/P, opinion1, opinion2, opinion3, influence]
        features = np.concatenate([
            rng.choice([-1.0, 1.0], size=4),  # MBTI binary
            rng.standard_normal(3),             # opinions
            [rng.uniform(0.1, 1.0)],            # influence
        ])
        agents.append(Agent(id=i, features=features, metadata={"mbti": "ENTJ"}))
    return agents


def _params(n=10):
    return TheoryParameters(n_agents=n, n_features=8, random_seed=42)


def _init_dct(agents, params, **kwargs):
    defaults = dict(noise=0.01, threshold=0.3)
    defaults.update(kwargs)
    theory = DCT(params, **defaults)
    aff = compute_affinity_matrix(agents)
    theory._affinity_matrix = aff
    theory.initialize_agents(agents)
    return theory


# ===========================================================================
# Feature 1: TraitMap
# ===========================================================================

class TestTraitMapFromIndices:
    def test_produces_per_agent_vectors(self):
        agents = _agents_with_varied_features()
        tm = TraitMap.from_indices(mu_index=0, lam_index=3)
        mu, lam = tm.compute(agents)
        assert len(mu) == 10
        assert len(lam) == 10

    def test_values_are_positive(self):
        agents = _agents_with_varied_features()
        tm = TraitMap.from_indices(mu_index=0, lam_index=3)
        mu, lam = tm.compute(agents)
        assert np.all(mu > 0)
        assert np.all(lam > 0)

    def test_different_features_produce_different_rates(self):
        agents = _agents_with_varied_features()
        tm = TraitMap.from_indices(mu_index=0, lam_index=3)
        mu, lam = tm.compute(agents)
        # Since MBTI are +/-1, sigmoid gives different values
        assert not np.allclose(mu, mu[0])

    def test_scale_affects_range(self):
        agents = _agents_with_varied_features()
        tm_low = TraitMap.from_indices(mu_index=0, lam_index=3, mu_scale=0.1)
        tm_high = TraitMap.from_indices(mu_index=0, lam_index=3, mu_scale=2.0)
        mu_low, _ = tm_low.compute(agents)
        mu_high, _ = tm_high.compute(agents)
        assert np.std(mu_high) > np.std(mu_low)


class TestTraitMapFromMetadata:
    def test_reads_metadata_keys(self):
        agents = [
            Agent(id=0, features=np.zeros(3), metadata={"seeking": 0.8, "conformity": 0.1}),
            Agent(id=1, features=np.zeros(3), metadata={"seeking": 0.2, "conformity": 0.9}),
        ]
        tm = TraitMap.from_metadata(mu_key="seeking", lam_key="conformity")
        mu, lam = tm.compute(agents)
        assert mu[0] == pytest.approx(0.8)
        assert mu[1] == pytest.approx(0.2)
        assert lam[0] == pytest.approx(0.1)
        assert lam[1] == pytest.approx(0.9)

    def test_uses_defaults_for_missing_keys(self):
        agents = [Agent(id=0, features=np.zeros(3))]
        tm = TraitMap.from_metadata(mu_key="x", lam_key="y", mu_default=0.5, lam_default=0.1)
        mu, lam = tm.compute(agents)
        assert mu[0] == pytest.approx(0.5)
        assert lam[0] == pytest.approx(0.1)


class TestTraitMapPresets:
    def test_mbti_preset(self):
        agents = _agents_with_varied_features()
        tm = TraitMap.from_preset("mbti")
        mu, lam = tm.compute(agents)
        assert len(mu) == 10
        assert np.all(mu > 0)

    def test_influence_preset(self):
        agents = _agents_with_varied_features()
        tm = TraitMap.from_preset("influence")
        mu, lam = tm.compute(agents)
        # High influence -> high mu, low lam
        # Low influence -> low mu, high lam
        high_inf_idx = np.argmax([a.features[-1] for a in agents])
        low_inf_idx = np.argmin([a.features[-1] for a in agents])
        assert mu[high_inf_idx] > mu[low_inf_idx]
        assert lam[high_inf_idx] < lam[low_inf_idx]

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown trait map"):
            TraitMap.from_preset("nonexistent")


class TestDCTWithTraitMap:
    def test_trait_map_overrides_mu_lam(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params, trait_map="mbti")
        # mu/lam should vary per agent
        assert not np.allclose(theory.mu, theory.mu[0])

    def test_trait_map_string_works(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params, trait_map="influence")
        assert theory.mu is not None
        assert len(theory.mu) == 10

    def test_simulation_runs_with_trait_map(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params, trait_map="mbti")
        theory.run_simulation(t_max=5.0, dt=1.0)
        groups = theory.get_groups()
        assert len(groups) > 0

    def test_trait_map_instance_works(self):
        agents = _agents_with_varied_features()
        params = _params()
        tm = TraitMap.from_indices(mu_index=0, lam_index=-1,
                                   mu_scale=0.5, lam_scale=0.3)
        theory = _init_dct(agents, params, trait_map=tm)
        assert not np.allclose(theory.mu, theory.mu[0])


# ===========================================================================
# Feature 2: Separate proximity/alignment sources
# ===========================================================================

class TestSeparateSources:
    def test_proximity_matrix_sets_initial_positions(self):
        agents = _agents_with_varied_features()
        params = _params()
        n = len(agents)
        # Create a proximity matrix where agents 0-4 are close, 5-9 are close
        prox = np.eye(n) * 0.5
        for i in range(5):
            for j in range(5):
                prox[i, j] = 0.9
        for i in range(5, 10):
            for j in range(5, 10):
                prox[i, j] = 0.9
        np.fill_diagonal(prox, 1.0)

        theory = _init_dct(agents, params, proximity_matrix=prox)
        # Agents in same cluster should start closer in context space
        intra = np.mean([np.linalg.norm(theory.context_pos[i] - theory.context_pos[j])
                         for i in range(5) for j in range(i + 1, 5)])
        inter = np.mean([np.linalg.norm(theory.context_pos[i] - theory.context_pos[j])
                         for i in range(5) for j in range(5, 10)])
        assert intra < inter

    def test_alignment_features_override_agent_features(self):
        agents = _agents_with_varied_features()
        params = _params()
        n = len(agents)
        custom_align = np.random.default_rng(99).standard_normal((n, 4))
        theory = _init_dct(agents, params, alignment_features=custom_align)
        # Alignment should match what we provided, not agent.features
        np.testing.assert_array_almost_equal(theory.alignment, custom_align)

    def test_both_sources_independent(self):
        """Proximity from interactions, alignment from opinions."""
        agents = _agents_with_varied_features()
        params = _params()
        n = len(agents)

        # Proximity: random interaction pattern
        rng = np.random.default_rng(42)
        prox = rng.uniform(0, 1, (n, n))
        prox = (prox + prox.T) / 2
        np.fill_diagonal(prox, 1.0)

        # Alignment: opinion features only (indices 4-6)
        align = np.array([a.features[4:7] for a in agents])

        theory = _init_dct(agents, params, proximity_matrix=prox,
                           alignment_features=align)
        theory.run_simulation(t_max=5.0, dt=1.0)
        groups = theory.get_groups()
        assert len(groups) > 0

    def test_default_behavior_unchanged(self):
        """Without new params, behavior is identical to before."""
        agents = _agents_with_varied_features()
        params = _params()
        t1 = _init_dct(agents, params)
        t2 = _init_dct(agents, params)
        t1.run_simulation(t_max=3.0, dt=1.0)
        t2.run_simulation(t_max=3.0, dt=1.0)
        np.testing.assert_array_equal(t1.context_pos, t2.context_pos)


# ===========================================================================
# Feature 3: Sustained interventions
# ===========================================================================

class TestSustainedInterventionBase:
    def test_not_implemented(self):
        siv = SustainedIntervention(start=1.0, end=5.0)
        with pytest.raises(NotImplementedError):
            siv.apply_step(None, 1.0)

    def test_is_active(self):
        siv = SustainedIntervention(start=2.0, end=5.0)
        assert not siv.is_active(0.0, 1.0)  # t=0, next=1 < start=2
        assert siv.is_active(1.0, 1.0)       # t=1, next=2 >= start=2
        assert siv.is_active(3.0, 1.0)       # within range
        assert siv.is_active(4.0, 1.0)       # t=4, next=5 >= start, t=4 < end=5
        assert not siv.is_active(5.0, 1.0)   # t=5, t >= end=5


class TestSustainedShift:
    def test_applies_every_active_step(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params)
        feat_before = theory.agents[0].features.copy()

        delta = np.array([0.1, 0, 0, 0, 0, 0, 0, 0])
        runner = InterventionRunner(theory, sustained=[
            SustainedShift(start=2.0, end=5.0, agent_ids=[0], delta_per_step=delta),
        ])
        report = runner.run(t_max=6.0, dt=1.0)

        # is_active fires when start <= t+dt, so t=1,2,3,4 (4 applications)
        sustained_logs = [l for l in report.intervention_log
                          if l.get("intervention_class") == "SustainedShift"]
        assert len(sustained_logs) == 4

    def test_cumulative_effect(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params)
        feat_before = theory.agents[0].features[-1].copy()

        delta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        runner = InterventionRunner(theory, sustained=[
            SustainedShift(start=1.0, end=4.0, agent_ids=[0], delta_per_step=delta),
        ])
        runner.run(t_max=5.0, dt=1.0)
        # After multiple steps of delta, feature should have shifted
        assert theory.agents[0].features[-1] != feat_before

    def test_no_effect_outside_range(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params)

        delta = np.array([0.5, 0, 0, 0, 0, 0, 0, 0])
        runner = InterventionRunner(theory, sustained=[
            SustainedShift(start=10.0, end=20.0, agent_ids=[0], delta_per_step=delta),
        ])
        report = runner.run(t_max=5.0, dt=1.0)

        sustained_logs = [l for l in report.intervention_log
                          if l.get("intervention_class") == "SustainedShift"]
        assert len(sustained_logs) == 0


class TestSustainedNoise:
    def test_applies_noise_each_step(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params)

        runner = InterventionRunner(theory, sustained=[
            SustainedNoise(start=1.0, end=4.0, intensity=0.5, seed=42),
        ])
        report = runner.run(t_max=5.0, dt=1.0)

        # is_active fires when start <= t+dt, so t=0,1,2,3 (4 applications)
        noise_logs = [l for l in report.intervention_log
                      if l.get("intervention_class") == "SustainedNoise"]
        assert len(noise_logs) == 4

    def test_targets_subset(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params)

        runner = InterventionRunner(theory, sustained=[
            SustainedNoise(start=0.0, end=3.0, intensity=0.5,
                           target_ids=[0, 1], seed=42),
        ])
        report = runner.run(t_max=3.0, dt=1.0)
        for log in report.intervention_log:
            if log.get("intervention_class") == "SustainedNoise":
                assert log["shocked"] == 2


class TestSustainedAffinityBias:
    def test_biases_affinity(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params)
        aff_before = theory._affinity_matrix[0, 1]

        runner = InterventionRunner(theory, sustained=[
            SustainedAffinityBias(start=0.0, end=5.0,
                                  pairs=[(0, 1)], bias_per_step=[0.05]),
        ])
        runner.run(t_max=5.0, dt=1.0)
        # After 5 steps of +0.05 bias, affinity should have increased
        # (clamped to [-1, 1])
        assert theory._affinity_matrix[0, 1] >= aff_before


class TestMixedInterventions:
    def test_point_and_sustained_together(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params)

        from cft import RemoveAgents
        runner = InterventionRunner(
            theory,
            interventions=[RemoveAgents(time=3.0, agent_ids=[9])],
            sustained=[SustainedShift(
                start=1.0, end=6.0, agent_ids=[0],
                delta_per_step=np.array([0.1, 0, 0, 0, 0, 0, 0, 0]),
            )],
        )
        report = runner.run(t_max=7.0, dt=1.0)

        point_logs = [l for l in report.intervention_log
                      if l.get("intervention_class") == "RemoveAgents"]
        sustained_logs = [l for l in report.intervention_log
                          if l.get("intervention_class") == "SustainedShift"]
        assert len(point_logs) == 1
        assert len(sustained_logs) == 6  # is_active fires at t=0,1,2,3,4,5

    def test_resilience_scores_with_sustained(self):
        agents = _agents_with_varied_features()
        params = _params()
        theory = _init_dct(agents, params)

        runner = InterventionRunner(theory, sustained=[
            SustainedNoise(start=3.0, end=6.0, intensity=1.0, seed=42),
        ])
        report = runner.run(t_max=10.0, dt=1.0)
        scores = report.resilience_scores
        assert "mean_stability" in scores
        assert "recovery_rate" in scores
