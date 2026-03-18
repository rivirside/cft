"""Analytically solvable test cases with known-correct answers.

These tests use hand-crafted scenarios where the correct group assignment
can be derived mathematically, not just checked for plausibility.
"""

import numpy as np
import pytest

from cft import Agent, TheoryParameters, CFT, GFT, TST
from cft.affinity import compute_affinity_matrix


def make_agents(positions):
    """Create agents from a list of feature vectors."""
    return [Agent(id=i, features=np.array(f, dtype=float)) for i, f in enumerate(positions)]


class TestCFTAnalytical:
    """CFT with a known affinity matrix - group assignments are deterministic."""

    def test_two_cliques_with_known_affinity(self):
        """Two cliques: {0,1,2} and {3,4} with threshold=0.5.

        Affinity matrix (symmetric):
            0    1    2    3    4
        0  1.0  0.8  0.7  0.1  0.2
        1  0.8  1.0  0.9  0.0  0.1
        2  0.7  0.9  1.0  0.2  0.3
        3  0.1  0.0  0.2  1.0  0.6
        4  0.2  0.1  0.3  0.6  1.0

        With threshold=0.5:
        - {0,1}: 0.8 >= 0.5 ✓
        - {0,1,2}: min(0.8, 0.7, 0.9) = 0.7 >= 0.5 ✓
        - {0,1,2,3}: 0.1 < 0.5 for (0,3) ✗
        - {3,4}: 0.6 >= 0.5 ✓
        Expected: groups = [{0,1,2}, {3,4}]
        """
        affinity = np.array([
            [1.0, 0.8, 0.7, 0.1, 0.2],
            [0.8, 1.0, 0.9, 0.0, 0.1],
            [0.7, 0.9, 1.0, 0.2, 0.3],
            [0.1, 0.0, 0.2, 1.0, 0.6],
            [0.2, 0.1, 0.3, 0.6, 1.0],
        ])
        agents = make_agents([[0]] * 5)  # features don't matter with pre-computed affinity
        params = TheoryParameters(n_agents=5, n_features=1, random_seed=0)

        theory = CFT(params, threshold=0.5, affinity_matrix=affinity)
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=1.0, dt=1.0)

        groups = theory.get_groups()
        group_sets = [set(g.members) for g in groups]
        assert {0, 1, 2} in group_sets
        assert {3, 4} in group_sets
        assert len(groups) == 2

    def test_three_singletons_low_affinity(self):
        """All pairwise affinities below threshold -> everyone is a singleton."""
        affinity = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ])
        agents = make_agents([[0]] * 3)
        params = TheoryParameters(n_agents=3, n_features=1, random_seed=0)

        theory = CFT(params, threshold=0.5, affinity_matrix=affinity)
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=1.0, dt=1.0)

        groups = theory.get_groups()
        assert len(groups) == 3
        for g in groups:
            assert len(g.members) == 1

    def test_complete_graph_one_group(self):
        """All affinities above threshold -> one big group."""
        n = 4
        affinity = np.ones((n, n)) * 0.9
        np.fill_diagonal(affinity, 1.0)
        agents = make_agents([[0]] * n)
        params = TheoryParameters(n_agents=n, n_features=1, random_seed=0)

        theory = CFT(params, threshold=0.5, affinity_matrix=affinity)
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=1.0, dt=1.0)

        groups = theory.get_groups()
        assert len(groups) == 1
        assert set(groups[0].members) == {0, 1, 2, 3}

    def test_chain_graph_greedy_behavior(self):
        """Chain: 0-1-2-3 where only adjacent pairs have high affinity.

        Affinity:
            0    1    2    3
        0  1.0  0.9  0.1  0.0
        1  0.9  1.0  0.8  0.1
        2  0.1  0.8  1.0  0.9
        3  0.0  0.1  0.9  1.0

        CFT's greedy algorithm processes in order:
        - i=0: start group with {0}, check j=1: α(0,1)=0.9≥0.5 → {0,1}
               check j=2: α(0,2)=0.1 < 0.5 → skip
               check j=3: α(0,3)=0.0 < 0.5 → skip
        - i=2 (next unassigned): start group with {2}, check j=3: α(2,3)=0.9≥0.5 → {2,3}
        Expected: [{0,1}, {2,3}]
        """
        affinity = np.array([
            [1.0, 0.9, 0.1, 0.0],
            [0.9, 1.0, 0.8, 0.1],
            [0.1, 0.8, 1.0, 0.9],
            [0.0, 0.1, 0.9, 1.0],
        ])
        agents = make_agents([[0]] * 4)
        params = TheoryParameters(n_agents=4, n_features=1, random_seed=0)

        theory = CFT(params, threshold=0.5, affinity_matrix=affinity)
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=1.0, dt=1.0)

        groups = theory.get_groups()
        group_sets = [set(g.members) for g in groups]
        assert {0, 1} in group_sets
        assert {2, 3} in group_sets


class TestGFTAnalytical:
    """GFT with symmetric configurations where equilibrium is predictable."""

    def test_two_particles_attract_to_midpoint(self):
        """Two agents at (-1,0) and (1,0) should converge toward (0,0)."""
        agents = make_agents([[-1.0, 0.0], [1.0, 0.0]])
        params = TheoryParameters(n_agents=2, n_features=2, random_seed=0)

        theory = GFT(params, k=0.5, sigma=5.0)
        theory.initialize_agents(agents)

        # Run enough steps to converge
        theory.run_simulation(t_max=50.0, dt=0.5)

        # Both agents should be near the midpoint
        midpoint = np.array([0.0, 0.0])
        for i in range(2):
            dist_to_mid = np.linalg.norm(theory.positions[i] - midpoint)
            assert dist_to_mid < 0.1, f"Agent {i} at {theory.positions[i]}, expected near {midpoint}"

    def test_symmetric_triangle_converges_to_centroid(self):
        """Three agents at vertices of equilateral triangle converge to centroid."""
        h = np.sqrt(3) / 2
        agents = make_agents([[0.0, 1.0], [-h, -0.5], [h, -0.5]])
        params = TheoryParameters(n_agents=3, n_features=2, random_seed=0)

        theory = GFT(params, k=0.5, sigma=5.0)
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=50.0, dt=0.5)

        centroid = np.array([0.0, 0.0])
        for i in range(3):
            dist = np.linalg.norm(theory.positions[i] - centroid)
            assert dist < 0.15, f"Agent {i} at {theory.positions[i]}, expected near centroid"

    def test_two_clusters_stay_separate(self):
        """Two tight clusters far apart should remain separate with small sigma."""
        agents = make_agents([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],   # cluster A near origin
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],  # cluster B far away
        ])
        params = TheoryParameters(n_agents=6, n_features=2, random_seed=0)

        theory = GFT(params, k=0.5, sigma=0.5)  # small sigma = short range
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=20.0, dt=0.5)

        groups = theory.get_groups()
        group_sets = [set(g.members) for g in groups]
        assert {0, 1, 2} in group_sets
        assert {3, 4, 5} in group_sets


class TestTSTAnalytical:
    """TST with engineered coupling matrices where ground state is known."""

    def test_ferromagnetic_low_temperature(self):
        """Uniform positive coupling at T≈0 → all agents in same group.

        With J_ij > 0 for all pairs and T→0, the Potts ground state has
        all spins aligned (one group).
        """
        n = 6
        coupling = np.ones((n, n)) * 1.0
        np.fill_diagonal(coupling, 0.0)

        agents = make_agents([[0.0]] * n)
        params = TheoryParameters(n_agents=n, n_features=1, random_seed=42)

        theory = TST(
            params,
            temperature=0.01,  # near zero
            n_groups_max=4,
            sweeps_per_step=50,
            affinity_matrix=coupling,
        )
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=10.0, dt=1.0)

        groups = theory.get_groups()
        # At T≈0 with uniform positive coupling, should converge to 1 group
        assert len(groups) == 1

    def test_antiferromagnetic_two_groups(self):
        """Two-block structure with negative inter-block coupling.

        Agents 0,1,2 coupled positively (+1). Agents 3,4,5 coupled positively (+1).
        Inter-block coupling is negative (-1). At low T, ground state is two groups.
        """
        n = 6
        coupling = np.zeros((n, n))
        # Positive intra-block coupling
        for i in range(3):
            for j in range(i + 1, 3):
                coupling[i, j] = coupling[j, i] = 1.0
        for i in range(3, 6):
            for j in range(i + 1, 6):
                coupling[i, j] = coupling[j, i] = 1.0
        # Negative inter-block coupling
        for i in range(3):
            for j in range(3, 6):
                coupling[i, j] = coupling[j, i] = -1.0

        agents = make_agents([[0.0]] * n)
        params = TheoryParameters(n_agents=n, n_features=1, random_seed=42)

        theory = TST(
            params,
            temperature=0.01,
            n_groups_max=4,
            sweeps_per_step=50,
            affinity_matrix=coupling,
        )
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=10.0, dt=1.0)

        groups = theory.get_groups()
        assert len(groups) == 2
        group_sets = [set(g.members) for g in groups]
        assert {0, 1, 2} in group_sets
        assert {3, 4, 5} in group_sets

    def test_energy_is_optimal_at_ground_state(self):
        """After annealing, energy should be at or near the theoretical minimum.

        For n agents all in one group with uniform coupling J:
        E_min = -J * n*(n-1)/2
        """
        n = 4
        J = 1.0
        coupling = np.full((n, n), J)
        np.fill_diagonal(coupling, 0.0)

        agents = make_agents([[0.0]] * n)
        params = TheoryParameters(n_agents=n, n_features=1, random_seed=42)

        theory = TST(
            params,
            temperature=0.01,
            n_groups_max=3,
            sweeps_per_step=100,
            affinity_matrix=coupling,
        )
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=20.0, dt=1.0)

        state = theory.get_state()
        theoretical_min = -J * n * (n - 1) / 2  # = -6.0
        assert state["energy"] == pytest.approx(theoretical_min, abs=0.1)


class TestAffinityAnalytical:
    """Affinity computations with hand-verifiable inputs."""

    def test_euclidean_two_agents_known_distance(self):
        """Agents at (0,0) and (1,0) with n_features=2.

        dist = 1.0, max_dist = sqrt(2)
        affinity = 1 - 1/sqrt(2) ≈ 0.2929
        """
        agents = make_agents([[0.0, 0.0], [1.0, 0.0]])
        A = compute_affinity_matrix(agents, metric="euclidean", n_features=2)
        expected = 1.0 - 1.0 / np.sqrt(2)
        assert A[0, 1] == pytest.approx(expected, abs=1e-10)

    def test_cosine_known_angle(self):
        """Agents at (1,0) and (1,1). Angle = 45°, cos(45°) = 1/sqrt(2)."""
        agents = make_agents([[1.0, 0.0], [1.0, 1.0]])
        A = compute_affinity_matrix(agents, metric="cosine")
        expected = 1.0 / np.sqrt(2)
        assert A[0, 1] == pytest.approx(expected, abs=1e-10)

    def test_correlation_anticorrelated(self):
        """Perfectly anti-correlated features → affinity = -1."""
        agents = make_agents([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        A = compute_affinity_matrix(agents, metric="correlation")
        assert A[0, 1] == pytest.approx(-1.0, abs=1e-10)
