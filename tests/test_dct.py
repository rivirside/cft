"""Tests for Dual-Context Theory."""

import numpy as np

from cft import DCT, TheoryParameters, Agent


class TestDCTInitialization:
    def test_creates_both_spaces(self, small_agents, small_params):
        theory = DCT(small_params, mu=0.3, lam=0.05)
        theory.initialize_agents(small_agents)
        assert theory.context_pos is not None
        assert theory.alignment is not None
        assert theory.context_pos.shape == (10, 3)
        assert theory.alignment.shape == (10, 3)

    def test_alignment_starts_as_features(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        for i, agent in enumerate(small_agents):
            np.testing.assert_array_equal(theory.alignment[i], agent.features)

    def test_context_pos_differs_from_alignment(self, small_agents, small_params):
        """Context starts as alignment + noise, so shouldn't be identical."""
        theory = DCT(small_params, noise=0.1)
        theory.initialize_agents(small_agents)
        assert not np.array_equal(theory.context_pos, theory.alignment)

    def test_affinity_matrices_computed(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        assert theory._proximity_affinity is not None
        assert theory._alignment_affinity is not None
        assert theory._effective_affinity is not None
        assert theory._proximity_affinity.shape == (10, 10)

    def test_pre_computed_affinity_matrix(self, small_agents, small_params):
        """When affinity_matrix is provided, alignment is derived from it."""
        n = len(small_agents)
        fake_aff = np.eye(n) * 0.5 + 0.5
        theory = DCT(small_params, affinity_matrix=fake_aff)
        theory.initialize_agents(small_agents)
        # Alignment should be derived via spectral embedding, not raw features
        assert theory.alignment.shape[0] == n


class TestDCTDynamics:
    def test_step_updates_positions(self, small_agents, small_params):
        theory = DCT(small_params, mu=0.3, lam=0.05, noise=0.1)
        theory.initialize_agents(small_agents)
        pos_before = theory.context_pos.copy()
        theory.step(1.0)
        assert not np.array_equal(theory.context_pos, pos_before)

    def test_step_updates_alignment(self, small_agents, small_params):
        theory = DCT(small_params, mu=0.3, lam=0.2, noise=0.0)
        theory.initialize_agents(small_agents)
        align_before = theory.alignment.copy()
        theory.step(1.0)
        # With lam > 0 and noise=0, alignment should drift
        assert not np.array_equal(theory.alignment, align_before)

    def test_zero_conformity_preserves_alignment(self, small_agents, small_params):
        """With lam=0, alignment should not change."""
        theory = DCT(small_params, mu=0.3, lam=0.0, noise=0.0)
        theory.initialize_agents(small_agents)
        align_before = theory.alignment.copy()
        theory.step(1.0)
        np.testing.assert_array_almost_equal(theory.alignment, align_before)

    def test_time_advances(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        assert theory.current_time == 0.0
        theory.step(1.0)
        assert theory.current_time == 1.0
        theory.step(0.5)
        assert theory.current_time == 1.5


class TestDCTGroups:
    def test_all_agents_assigned(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=5.0, dt=1.0)
        groups = theory.get_groups()
        all_members = []
        for g in groups:
            all_members.extend(g.members)
        assert sorted(all_members) == list(range(10))

    def test_no_duplicate_assignments(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=5.0, dt=1.0)
        groups = theory.get_groups()
        all_members = []
        for g in groups:
            all_members.extend(g.members)
        assert len(all_members) == len(set(all_members))

    def test_clustered_agents_form_groups(self, clustered_agents):
        """Two well-separated clusters should produce two groups."""
        params = TheoryParameters(n_agents=10, n_features=3, random_seed=42)
        theory = DCT(params, mu=0.3, lam=0.05, noise=0.01, threshold=0.2)
        theory.initialize_agents(clustered_agents)
        theory.run_simulation(t_max=10.0, dt=1.0)
        groups = theory.get_groups()
        # Should find roughly 2 groups (clusters are very well separated)
        assert len(groups) <= 5  # generous upper bound


class TestDCTAffinities:
    def test_proximity_affinity_symmetric(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.step(1.0)
        np.testing.assert_array_almost_equal(
            theory._proximity_affinity, theory._proximity_affinity.T
        )

    def test_alignment_affinity_symmetric(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.step(1.0)
        np.testing.assert_array_almost_equal(
            theory._alignment_affinity, theory._alignment_affinity.T
        )

    def test_effective_affinity_bounded(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.step(1.0)
        assert np.all(theory._effective_affinity >= 0.0)
        assert np.all(theory._effective_affinity <= 1.0 + 1e-10)

    def test_diagonal_is_one(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.step(1.0)
        np.testing.assert_array_almost_equal(
            np.diag(theory._effective_affinity), 1.0
        )


class TestDCTState:
    def test_state_keys(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.step(1.0)
        state = theory.get_state()
        expected_keys = {
            "context_positions", "alignment", "proximity_affinity",
            "alignment_affinity", "effective_affinity", "tension",
            "mu", "lam", "threshold", "n_groups",
        }
        assert set(state.keys()) == expected_keys

    def test_tension_is_finite(self, small_agents, small_params):
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=5.0, dt=1.0)
        state = theory.get_state()
        assert np.isfinite(state["tension"])
        assert state["tension"] >= 0.0

    def test_state_returns_copies(self, small_agents, small_params):
        """Mutating returned state should not affect theory internals."""
        theory = DCT(small_params)
        theory.initialize_agents(small_agents)
        theory.step(1.0)
        state = theory.get_state()
        state["context_positions"][:] = 999.0
        assert not np.any(theory.context_pos == 999.0)


class TestDCTDeterminism:
    def test_same_seed_same_result(self, small_agents):
        params1 = TheoryParameters(n_agents=10, n_features=3, random_seed=7)
        params2 = TheoryParameters(n_agents=10, n_features=3, random_seed=7)
        t1 = DCT(params1, mu=0.3, lam=0.05, noise=0.1)
        t2 = DCT(params2, mu=0.3, lam=0.05, noise=0.1)
        t1.initialize_agents(small_agents)
        t2.initialize_agents(small_agents)
        t1.run_simulation(t_max=5.0, dt=1.0)
        t2.run_simulation(t_max=5.0, dt=1.0)
        np.testing.assert_array_equal(t1.context_pos, t2.context_pos)
        np.testing.assert_array_equal(t1.alignment, t2.alignment)

    def test_different_seeds_differ(self, small_agents):
        params1 = TheoryParameters(n_agents=10, n_features=3, random_seed=1)
        params2 = TheoryParameters(n_agents=10, n_features=3, random_seed=2)
        t1 = DCT(params1, mu=0.3, lam=0.05, noise=0.1)
        t2 = DCT(params2, mu=0.3, lam=0.05, noise=0.1)
        t1.initialize_agents(small_agents)
        t2.initialize_agents(small_agents)
        t1.run_simulation(t_max=5.0, dt=1.0)
        t2.run_simulation(t_max=5.0, dt=1.0)
        assert not np.array_equal(t1.context_pos, t2.context_pos)


class TestDCTCoupling:
    """Tests for the bidirectional coupling between layers."""

    def test_high_seeking_clusters_context(self):
        """High mu should cause context positions to cluster around aligned agents."""
        # Two groups with opposing features
        agents = []
        for i in range(5):
            agents.append(Agent(id=i, features=np.array([1.0, 1.0, 0.0])))
        for i in range(5):
            agents.append(Agent(id=5 + i, features=np.array([-1.0, -1.0, 0.0])))

        params = TheoryParameters(n_agents=10, n_features=3, random_seed=42)
        theory = DCT(params, mu=1.0, lam=0.0, noise=0.01, context_scale=1.5)
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=10.0, dt=0.5)

        # Context positions of group A should be closer to each other than to group B
        pos = theory.context_pos
        intra_a = np.mean([np.linalg.norm(pos[i] - pos[j]) for i in range(5) for j in range(i + 1, 5)])
        inter = np.mean([np.linalg.norm(pos[i] - pos[j]) for i in range(5) for j in range(5, 10)])
        assert intra_a < inter

    def test_high_conformity_homogenizes_alignment(self):
        """High lam with proximate agents should reduce alignment variance."""
        # All agents start close in context space but with diverse opinions
        rng = np.random.default_rng(42)
        agents = [Agent(id=i, features=rng.standard_normal(3)) for i in range(8)]

        params = TheoryParameters(n_agents=8, n_features=3, random_seed=42)
        theory = DCT(params, mu=0.0, lam=0.5, noise=0.0, context_scale=5.0)
        theory.initialize_agents(agents)

        # Force all context positions close together
        theory.context_pos = np.zeros_like(theory.context_pos) + rng.normal(0, 0.1, theory.context_pos.shape)
        theory._update_affinities()

        var_before = np.var(theory.alignment)
        theory.run_simulation(t_max=20.0, dt=0.5)
        var_after = np.var(theory.alignment)

        assert var_after < var_before
