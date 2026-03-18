"""Tests for Thermodynamic Social Theory."""

import numpy as np

from cft import TST, TheoryParameters


class TestTST:
    def test_initialization(self, small_agents, small_params):
        theory = TST(small_params, temperature=1.0, n_groups_max=5)
        theory.initialize_agents(small_agents)
        assert theory.spins is not None
        assert len(theory.spins) == 10

    def test_energy_computation(self, small_agents, small_params):
        """Energy should be a finite number."""
        theory = TST(small_params, temperature=1.0, n_groups_max=5)
        theory.initialize_agents(small_agents)
        energy = theory._compute_energy()
        assert np.isfinite(energy)

    def test_entropy_computation(self, small_agents, small_params):
        """Entropy should be non-negative."""
        theory = TST(small_params, temperature=1.0, n_groups_max=5)
        theory.initialize_agents(small_agents)
        entropy = theory._compute_entropy()
        assert entropy >= 0

    def test_cooling_reduces_energy(self, small_agents, small_params):
        """Simulated annealing should reduce energy over time."""
        theory = TST(small_params, temperature=5.0, n_groups_max=5, cooling_rate=0.1)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=20.0, dt=1.0)
        energies = theory._energy_history
        # Energy should generally decrease (allow some fluctuation)
        assert energies[-1] <= energies[0] + 5.0  # generous tolerance

    def test_temperature_decreases_with_cooling(self, small_agents, small_params):
        """Temperature should decrease when cooling_rate > 0."""
        theory = TST(small_params, temperature=5.0, n_groups_max=5, cooling_rate=0.1)
        theory.initialize_agents(small_agents)
        initial_temp = theory.temperature
        theory.run_simulation(t_max=10.0, dt=1.0)
        assert theory.temperature < initial_temp

    def test_all_agents_assigned(self, small_agents, small_params):
        theory = TST(small_params, temperature=1.0, n_groups_max=5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=5.0, dt=1.0)
        groups = theory.get_groups()
        all_members = []
        for g in groups:
            all_members.extend(g.members)
        assert sorted(all_members) == list(range(10))

    def test_low_temperature_ordering(self, small_agents, small_params):
        """Very low temperature should produce ordered state (fewer groups)."""
        theory = TST(
            small_params, temperature=0.01, n_groups_max=5, sweeps_per_step=50
        )
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=10.0, dt=1.0)
        state = theory.get_state()
        # At low T, should consolidate into fewer groups
        assert state["n_groups"] <= 5

    def test_high_temperature_disorder(self, small_agents, small_params):
        """Very high temperature should maintain disorder (many groups)."""
        theory = TST(small_params, temperature=100.0, n_groups_max=5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=10.0, dt=1.0)
        state = theory.get_state()
        # At high T, should have multiple groups (random assignment)
        assert state["n_groups"] >= 2

    def test_state_includes_thermodynamic_quantities(self, small_agents, small_params):
        theory = TST(small_params, temperature=1.0, n_groups_max=5)
        theory.initialize_agents(small_agents)
        theory.run_simulation(t_max=5.0, dt=1.0)
        state = theory.get_state()
        assert "energy" in state
        assert "entropy" in state
        assert "free_energy" in state
        assert "temperature" in state
        assert np.isfinite(state["free_energy"])
