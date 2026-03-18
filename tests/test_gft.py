"""Tests for Gradient Field Theory."""

import numpy as np

from cft import GFT, Agent, TheoryParameters


class TestGFT:
    def test_initialization(self, small_agents, small_params):
        theory = GFT(small_params, k=0.1, sigma=1.0)
        theory.initialize_agents(small_agents)
        assert theory.positions is not None
        assert theory.positions.shape == (10, 3)

    def test_positions_change(self, small_agents, small_params):
        """Positions should change after stepping."""
        theory = GFT(small_params, k=0.2, sigma=1.5)
        theory.initialize_agents(small_agents)
        initial_positions = theory.positions.copy()
        theory.step(1.0)
        assert not np.allclose(theory.positions, initial_positions)

    def test_attraction(self):
        """Two nearby agents should move closer together."""
        agents = [
            Agent(id=0, features=np.array([0.0, 0.0, 0.0])),
            Agent(id=1, features=np.array([1.0, 0.0, 0.0])),
        ]
        params = TheoryParameters(n_agents=2, n_features=3)
        theory = GFT(params, k=0.5, sigma=2.0)
        theory.initialize_agents(agents)

        initial_dist = np.linalg.norm(theory.positions[0] - theory.positions[1])
        theory.run_simulation(t_max=10.0, dt=0.5)
        final_dist = np.linalg.norm(theory.positions[0] - theory.positions[1])

        assert final_dist < initial_dist

    def test_group_detection(self, small_agents, small_params):
        theory = GFT(small_params, k=0.2, sigma=1.5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=20.0, dt=1.0)
        groups = theory.get_groups()
        assert len(groups) > 0
        all_members = []
        for g in groups:
            all_members.extend(g.members)
        assert sorted(all_members) == list(range(10))

    def test_clustered_convergence(self, clustered_agents):
        """Two clear clusters should remain separated."""
        params = TheoryParameters(n_agents=10, n_features=3)
        theory = GFT(params, k=0.2, sigma=1.0)
        theory.initialize_agents(clustered_agents)
        theory.run_simulation(t_max=20.0, dt=1.0)
        groups = theory.get_groups()
        # Should maintain separation between the two clusters
        assert len(groups) >= 2
