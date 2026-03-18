"""Tests for shared affinity computation module."""

import numpy as np
import pytest

from cft import Agent
from cft.affinity import compute_affinity_matrix


@pytest.fixture
def identical_agents():
    """Three agents with identical features."""
    return [Agent(id=i, features=np.array([1.0, 2.0, 3.0])) for i in range(3)]


@pytest.fixture
def orthogonal_agents():
    """Three agents with orthogonal feature vectors."""
    return [
        Agent(id=0, features=np.array([1.0, 0.0, 0.0])),
        Agent(id=1, features=np.array([0.0, 1.0, 0.0])),
        Agent(id=2, features=np.array([0.0, 0.0, 1.0])),
    ]


class TestAffinityProperties:
    """Properties that hold for all metrics."""

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "correlation", "probabilistic"])
    def test_diagonal_is_one(self, small_agents, metric):
        A = compute_affinity_matrix(small_agents, metric=metric)
        np.testing.assert_allclose(np.diag(A), 1.0)

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "correlation", "probabilistic"])
    def test_symmetric(self, small_agents, metric):
        A = compute_affinity_matrix(small_agents, metric=metric)
        np.testing.assert_allclose(A, A.T, atol=1e-10)

    @pytest.mark.parametrize("metric", ["cosine", "correlation", "probabilistic"])
    def test_values_in_range(self, small_agents, metric):
        """Cosine, correlation, probabilistic are bounded to [-1, 1]."""
        A = compute_affinity_matrix(small_agents, metric=metric)
        assert A.min() >= -1.0 - 1e-10
        assert A.max() <= 1.0 + 1e-10

    def test_euclidean_can_exceed_minus_one(self, small_agents):
        """Euclidean affinity can go below -1 when distances exceed sqrt(n_features)."""
        A = compute_affinity_matrix(small_agents, metric="euclidean")
        # Diagonal is 1, off-diagonal can be anything <= 1
        np.testing.assert_allclose(np.diag(A), 1.0)
        assert A.max() <= 1.0 + 1e-10

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "correlation", "probabilistic"])
    def test_shape(self, small_agents, metric):
        A = compute_affinity_matrix(small_agents, metric=metric)
        n = len(small_agents)
        assert A.shape == (n, n)


class TestEuclideanAffinity:
    def test_identical_agents_max_affinity(self, identical_agents):
        A = compute_affinity_matrix(identical_agents, metric="euclidean")
        np.testing.assert_allclose(A, 1.0)

    def test_closer_agents_higher_affinity(self):
        agents = [
            Agent(id=0, features=np.array([0.0, 0.0])),
            Agent(id=1, features=np.array([0.1, 0.0])),  # close to 0
            Agent(id=2, features=np.array([1.0, 0.0])),  # far from 0
        ]
        A = compute_affinity_matrix(agents, metric="euclidean")
        assert A[0, 1] > A[0, 2]


class TestCosineAffinity:
    def test_identical_directions_max_affinity(self):
        agents = [
            Agent(id=0, features=np.array([1.0, 0.0])),
            Agent(id=1, features=np.array([2.0, 0.0])),  # same direction, different magnitude
        ]
        A = compute_affinity_matrix(agents, metric="cosine")
        np.testing.assert_allclose(A[0, 1], 1.0, atol=1e-10)

    def test_orthogonal_zero_affinity(self, orthogonal_agents):
        A = compute_affinity_matrix(orthogonal_agents, metric="cosine")
        np.testing.assert_allclose(A[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(A[0, 2], 0.0, atol=1e-10)


class TestCorrelationAffinity:
    def test_perfectly_correlated(self):
        agents = [
            Agent(id=0, features=np.array([1.0, 2.0, 3.0])),
            Agent(id=1, features=np.array([2.0, 4.0, 6.0])),  # perfectly correlated
        ]
        A = compute_affinity_matrix(agents, metric="correlation")
        np.testing.assert_allclose(A[0, 1], 1.0, atol=1e-10)

    def test_single_feature_falls_back_to_cosine(self):
        agents = [
            Agent(id=0, features=np.array([1.0])),
            Agent(id=1, features=np.array([2.0])),
        ]
        A_corr = compute_affinity_matrix(agents, metric="correlation")
        A_cos = compute_affinity_matrix(agents, metric="cosine")
        np.testing.assert_allclose(A_corr, A_cos)


class TestProbabilisticAffinity:
    def test_identical_agents_high_affinity(self, identical_agents):
        """Identical agents should have high (but not necessarily 1.0) affinity.

        The probabilistic metric clamps p to [0.01, 0.99], so identical agents
        get p=0.99, which maps to ~0.86 after logit+arctan.
        """
        A = compute_affinity_matrix(identical_agents, metric="probabilistic")
        # Off-diagonal should be high positive
        assert A[0, 1] > 0.8

    def test_far_agents_low_affinity(self):
        agents = [
            Agent(id=0, features=np.array([0.0, 0.0])),
            Agent(id=1, features=np.array([10.0, 10.0])),
        ]
        A = compute_affinity_matrix(agents, metric="probabilistic")
        assert A[0, 1] < 0.0  # far apart -> negative affinity


class TestInvalidMetric:
    def test_unknown_metric_raises(self, small_agents):
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_affinity_matrix(small_agents, metric="invalid")
