"""Shared test fixtures."""

import numpy as np
import pytest

from cft import Agent, TheoryParameters


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_agents(rng):
    """10 agents with 3 features, deterministic."""
    return [Agent(id=i, features=rng.standard_normal(3)) for i in range(10)]


@pytest.fixture
def large_agents(rng):
    """30 agents with 3 features, deterministic."""
    return [Agent(id=i, features=rng.standard_normal(3)) for i in range(30)]


@pytest.fixture
def clustered_agents():
    """Two clear clusters for predictable grouping."""
    agents = []
    # Cluster A: near origin
    for i in range(5):
        agents.append(Agent(id=i, features=np.array([0.1 * i, 0.1 * i, 0.0])))
    # Cluster B: far away
    for i in range(5):
        agents.append(Agent(id=5 + i, features=np.array([10.0 + 0.1 * i, 10.0 + 0.1 * i, 0.0])))
    return agents


@pytest.fixture
def small_params():
    return TheoryParameters(n_agents=10, n_features=3, random_seed=42)


@pytest.fixture
def large_params():
    return TheoryParameters(n_agents=30, n_features=3, random_seed=42)
