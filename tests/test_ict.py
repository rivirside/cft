"""Tests for Information Cascade Theory."""

import numpy as np

from cft import ICT, TheoryParameters


class TestICT:
    def test_initialization(self, small_agents, small_params):
        theory = ICT(small_params, bandwidth=4)
        theory.initialize_agents(small_agents)
        assert theory.knowledge is not None
        assert theory.knowledge.shape == (10, 3)

    def test_knowledge_changes(self, small_agents, small_params):
        """Knowledge should evolve through communication."""
        theory = ICT(small_params, bandwidth=4)
        theory.initialize_agents(small_agents)
        initial_knowledge = theory.knowledge.copy()
        theory.step(1.0)
        assert not np.allclose(theory.knowledge, initial_knowledge)

    def test_all_agents_assigned(self, small_agents, small_params):
        theory = ICT(small_params, bandwidth=4)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=10.0, dt=1.0)
        groups = theory.get_groups()
        all_members = []
        for g in groups:
            all_members.extend(g.members)
        assert sorted(all_members) == list(range(10))

    def test_bandwidth_limits_group_size(self):
        """Groups shouldn't grow beyond bandwidth * 3 after splitting."""
        from cft import Agent

        params = TheoryParameters(n_agents=30, n_features=3, random_seed=42)
        rng = np.random.default_rng(42)
        agents = [Agent(id=i, features=rng.standard_normal(3)) for i in range(30)]

        theory = ICT(params, bandwidth=3)
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=20.0, dt=1.0)
        groups = theory.get_groups()
        max_size = max(len(g.members) for g in groups)
        # bandwidth * 3 = 9 is the split threshold
        assert max_size <= theory.bandwidth * 3

    def test_group_formation_occurs(self, small_agents, small_params):
        """Should form fewer groups than agents over time."""
        theory = ICT(small_params, bandwidth=4)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=20.0, dt=1.0)
        groups = theory.get_groups()
        assert len(groups) < 10  # Should consolidate from initial 10 singletons
