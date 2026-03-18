"""Tests for Quantum Social Theory."""

import numpy as np

from cft import QST, TheoryParameters


class TestQST:
    def test_initialization(self, small_agents, small_params):
        theory = QST(small_params, n_states=5)
        theory.initialize_agents(small_agents)
        assert theory.amplitudes is not None
        assert theory.amplitudes.shape == (10, 5)

    def test_normalization_preserved(self, small_agents, small_params):
        """State vectors should remain normalized after steps."""
        theory = QST(small_params, n_states=5)
        theory.initialize_agents(small_agents)

        for _ in range(5):
            theory.step(1.0)
            norms = np.sqrt(np.sum(np.abs(theory.amplitudes) ** 2, axis=1))
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_measurement_collapses_state(self, small_agents, small_params):
        """After measurement, agent should be in a definite state."""
        theory = QST(small_params, n_states=5)
        theory.initialize_agents(small_agents)
        theory.step(1.0)

        theory.measure(agent_ids=[0])
        probs = np.abs(theory.amplitudes[0]) ** 2
        # One probability should be 1, rest 0
        assert np.max(probs) == 1.0
        assert np.sum(probs > 0.5) == 1

    def test_get_groups_deterministic(self, small_agents, small_params):
        """get_groups() should give same result on repeated calls."""
        theory = QST(small_params, n_states=5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=5.0, dt=1.0)

        g1 = theory.get_groups()
        g2 = theory.get_groups()
        assert [tuple(g.members) for g in g1] == [tuple(g.members) for g in g2]

    def test_measure_groups_assigns_all(self, small_agents, small_params):
        """measure_groups() should assign all agents and collapse state."""
        theory = QST(small_params, n_states=5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=5.0, dt=1.0)

        groups = theory.measure_groups()
        all_members = []
        for g in groups:
            all_members.extend(g.members)
        assert sorted(all_members) == list(range(10))

        # After measurement, all amplitudes should be collapsed (one component = 1)
        for i in range(10):
            probs = np.abs(theory.amplitudes[i]) ** 2
            assert np.max(probs) > 0.99

    def test_all_agents_assigned(self, small_agents, small_params):
        theory = QST(small_params, n_states=5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=5.0, dt=1.0)
        groups = theory.get_groups()
        all_members = []
        for g in groups:
            all_members.extend(g.members)
        assert sorted(all_members) == list(range(10))

    def test_entanglement_grows(self, small_agents, small_params):
        """Entanglement should increase over time for agents with positive affinity."""
        theory = QST(small_params, n_states=5, entanglement_rate=0.5)
        theory.initialize_agents(small_agents)
        initial_entanglement = theory.entanglement_matrix.sum()
        theory.run_simulation(t_max=10.0, dt=1.0)
        final_entanglement = theory.entanglement_matrix.sum()
        assert final_entanglement > initial_entanglement
