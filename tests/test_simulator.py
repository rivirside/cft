"""Tests for SocialSimulator."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cft.simulator import SocialSimulator, MBTI_TYPES, _mbti_vec

# ---------------------------------------------------------------------------
# MBTI helpers
# ---------------------------------------------------------------------------


class TestMBTIVec:
    def test_shape(self):
        v = _mbti_vec("ENFP")
        assert v.shape == (4,)

    def test_values(self):
        v = _mbti_vec("ENFP")
        np.testing.assert_array_equal(v, [1.0, -1.0, -1.0, -1.0])

    def test_istj(self):
        np.testing.assert_array_equal(_mbti_vec("ISTJ"), [-1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_agents(self):
        a = SocialSimulator(n_agents=15, T=5, seed=42).generate()
        b = SocialSimulator(n_agents=15, T=5, seed=42).generate()
        assert a._agent_dicts == b._agent_dicts

    def test_same_seed_same_interactions(self):
        a = SocialSimulator(n_agents=15, T=5, seed=42).generate()
        b = SocialSimulator(n_agents=15, T=5, seed=42).generate()
        assert a._interaction_dicts == b._interaction_dicts

    def test_different_seeds_different_output(self):
        a = SocialSimulator(n_agents=15, T=5, seed=0).generate()
        b = SocialSimulator(n_agents=15, T=5, seed=1).generate()
        # Very unlikely to produce identical outputs
        assert a._agent_dicts != b._agent_dicts


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------


class TestSchemaValidity:
    def setup_method(self):
        self.sim = SocialSimulator(n_agents=12, T=8, seed=7).generate()

    def test_agent_count(self):
        assert len(self.sim._agent_dicts) == 12

    def test_agent_has_required_fields(self):
        for agent in self.sim._agent_dicts:
            assert "id" in agent
            assert "mbti" in agent
            assert "opinions" in agent
            assert "influence" in agent

    def test_agent_ids_unique_and_sequential(self):
        ids = [a["id"] for a in self.sim._agent_dicts]
        assert sorted(ids) == list(range(12))

    def test_mbti_valid(self):
        for agent in self.sim._agent_dicts:
            assert agent["mbti"] in MBTI_TYPES

    def test_opinions_is_list_of_floats(self):
        for agent in self.sim._agent_dicts:
            assert isinstance(agent["opinions"], list)
            assert all(isinstance(x, float) for x in agent["opinions"])
            assert len(agent["opinions"]) == 3  # default n_opinions

    def test_influence_in_range(self):
        for agent in self.sim._agent_dicts:
            assert 0.0 <= agent["influence"] <= 1.0

    def test_no_self_interactions(self):
        for act in self.sim._interaction_dicts:
            assert act["agent_i"] != act["agent_j"]

    def test_interaction_fields(self):
        known_actions = {"follow", "like", "repost", "pos_comment", "neg_comment"}
        for act in self.sim._interaction_dicts:
            assert "timestamp" in act
            assert "agent_i" in act
            assert "agent_j" in act
            assert act["action"] in known_actions

    def test_iso_timestamps_parseable(self):
        from datetime import datetime

        for act in self.sim._interaction_dicts[:20]:
            # Must not raise
            datetime.fromisoformat(act["timestamp"])

    def test_agent_ids_in_interactions(self):
        valid_ids = {a["id"] for a in self.sim._agent_dicts}
        for act in self.sim._interaction_dicts:
            assert act["agent_i"] in valid_ids
            assert act["agent_j"] in valid_ids


# ---------------------------------------------------------------------------
# Scale properties
# ---------------------------------------------------------------------------


class TestScale:
    def test_interaction_count_positive(self):
        sim = SocialSimulator(n_agents=10, T=5, interaction_rate=2.0, seed=0).generate()
        assert len(sim._interaction_dicts) > 0

    def test_interaction_count_scales_with_T(self):
        low = SocialSimulator(n_agents=15, T=5, seed=42).generate()
        high = SocialSimulator(n_agents=15, T=20, seed=42).generate()
        assert len(high._interaction_dicts) > len(low._interaction_dicts)

    def test_interaction_count_scales_with_rate(self):
        low = SocialSimulator(n_agents=15, T=10, interaction_rate=0.5, seed=42).generate()
        high = SocialSimulator(n_agents=15, T=10, interaction_rate=4.0, seed=42).generate()
        assert len(high._interaction_dicts) > len(low._interaction_dicts)

    def test_n_features_property(self):
        sim = SocialSimulator(n_opinions=5, seed=0)
        assert sim.n_features == 4 + 5 + 1  # MBTI + opinions + influence


# ---------------------------------------------------------------------------
# Scenarios: structural properties
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_random_scenario_runs(self):
        sim = SocialSimulator(n_agents=20, scenario="random", T=5, seed=1).generate()
        assert len(sim._agent_dicts) == 20

    def test_clustered_scenario_runs(self):
        sim = SocialSimulator(n_agents=20, scenario="clustered", k=3, T=5, seed=2).generate()
        assert len(sim._agent_dicts) == 20

    def test_polarized_scenario_runs(self):
        sim = SocialSimulator(n_agents=20, scenario="polarized", T=5, seed=3).generate()
        assert len(sim._agent_dicts) == 20

    def test_hierarchical_scenario_runs(self):
        sim = SocialSimulator(n_agents=20, scenario="hierarchical", n_influencers=3, T=5, seed=4).generate()
        assert len(sim._agent_dicts) == 20

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            SocialSimulator(scenario="unknown")

    def test_polarized_intra_camp_affinity_higher(self):
        """Agents in same camp have higher raw affinity than cross-camp pairs."""
        sim = SocialSimulator(
            n_agents=20, scenario="polarized", bias_strength=0.95, T=5, seed=5
        ).generate()
        aff = sim._affinity_matrix
        n_a = 10  # first camp
        # Intra-camp A mean affinity (off-diagonal)
        intra = aff[:n_a, :n_a]
        mask = ~np.eye(n_a, dtype=bool)
        intra_mean = intra[mask].mean()
        # Inter-camp mean affinity
        inter_mean = aff[:n_a, n_a:].mean()
        assert intra_mean > inter_mean

    def test_hierarchical_influencers_have_high_influence(self):
        """First n_influencers agents have influence > 0.7."""
        sim = SocialSimulator(
            n_agents=15, scenario="hierarchical", n_influencers=3, T=5, seed=6
        ).generate()
        for agent in sim._agent_dicts[:3]:
            assert agent["influence"] >= 0.7

    def test_polarized_camp_sizes(self):
        sim = SocialSimulator(
            n_agents=20, scenario="polarized", camp_sizes=(12, 8), T=5, seed=8
        ).generate()
        assert len(sim._agent_dicts) == 20

    def test_polarized_invalid_camp_sizes_raises(self):
        sim = SocialSimulator(n_agents=20, scenario="polarized", camp_sizes=(5, 5), seed=0)
        with pytest.raises(ValueError, match="camp_sizes"):
            sim.generate()


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------


class TestAdapterIntegration:
    def test_to_adapter_returns_adapter(self):
        from cft.integrations.mirofish import MiroFishAdapter

        sim = SocialSimulator(n_agents=10, T=5, seed=0)
        adapter = sim.to_adapter()
        assert isinstance(adapter, MiroFishAdapter)
        sim.cleanup()

    def test_adapter_loads_agents(self):
        sim = SocialSimulator(n_agents=10, T=5, seed=0)
        adapter = sim.to_adapter()
        agents = adapter.load_agents()
        assert len(agents) == 10
        sim.cleanup()

    def test_adapter_loads_interactions(self):
        pd = pytest.importorskip("pandas")
        sim = SocialSimulator(n_agents=10, T=5, seed=0)
        adapter = sim.to_adapter()
        adapter.load_agents()
        df = adapter.load_interactions()
        assert len(df) == len(sim._interaction_dicts)
        sim.cleanup()

    def test_adapter_affinity_symmetric(self):
        sim = SocialSimulator(n_agents=10, T=5, seed=0)
        adapter = sim.to_adapter()
        adapter.load_agents()
        adapter.load_interactions()
        aff = adapter.compute_affinity_matrix()
        np.testing.assert_array_almost_equal(aff, aff.T)
        sim.cleanup()

    def test_adapter_affinity_diagonal_one(self):
        sim = SocialSimulator(n_agents=10, T=5, seed=0)
        adapter = sim.to_adapter()
        adapter.load_agents()
        adapter.load_interactions()
        aff = adapter.compute_affinity_matrix()
        np.testing.assert_array_equal(np.diag(aff), np.ones(10))
        sim.cleanup()

    def test_adapter_ground_truth_covers_all_agents(self):
        sim = SocialSimulator(n_agents=15, T=8, seed=1)
        adapter = sim.to_adapter()
        adapter.load_agents()
        adapter.load_interactions()
        groups = adapter.extract_ground_truth_groups()
        all_members = set()
        for g in groups:
            all_members.update(g.members)
        assert all_members == set(range(15))
        sim.cleanup()

    def test_write_to_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = SocialSimulator(n_agents=8, T=5, seed=2)
            adapter = sim.write_to_dir(tmp)
            assert (Path(tmp) / "profiles.jsonl").exists()
            assert (Path(tmp) / "actions.jsonl").exists()
            agents = adapter.load_agents()
            assert len(agents) == 8

    def test_cleanup_removes_tmpdir(self):
        sim = SocialSimulator(n_agents=8, T=5, seed=3)
        sim.to_adapter()
        tmpdir = sim._tmpdir
        assert Path(tmpdir).exists()
        sim.cleanup()
        assert not Path(tmpdir).exists()

    def test_feature_dimensions_match(self):
        """Agent features from adapter match simulator.n_features."""
        sim = SocialSimulator(n_agents=10, n_opinions=4, T=5, seed=4)
        adapter = sim.to_adapter()
        agents = adapter.load_agents()
        assert agents[0].features.shape[0] == sim.n_features  # 4 + 4 + 1 = 9
        sim.cleanup()


# ---------------------------------------------------------------------------
# iso_from_step utility
# ---------------------------------------------------------------------------


class TestIsoFromStep:
    def test_step_0(self):
        assert SocialSimulator.iso_from_step(0) == "2024-01-01T00:00:00"

    def test_step_1(self):
        # 5 minutes per step
        assert SocialSimulator.iso_from_step(1) == "2024-01-01T00:05:00"

    def test_ordering(self):
        t0 = SocialSimulator.iso_from_step(0)
        t5 = SocialSimulator.iso_from_step(5)
        assert t0 < t5  # lexicographic comparison works for ISO 8601
