"""Tests for base classes."""

import numpy as np

from cft import Agent, Group, TheoryParameters


class TestAgent:
    def test_creation(self):
        a = Agent(id=0, features=np.array([1.0, 2.0]))
        assert a.id == 0
        assert len(a.features) == 2
        assert a.metadata == {}

    def test_equality(self):
        a1 = Agent(id=0, features=np.array([1.0]))
        a2 = Agent(id=0, features=np.array([2.0]))  # same id, different features
        a3 = Agent(id=1, features=np.array([1.0]))
        assert a1 == a2
        assert a1 != a3

    def test_hashable(self):
        a1 = Agent(id=0, features=np.array([1.0]))
        a2 = Agent(id=1, features=np.array([1.0]))
        s = {a1, a2}
        assert len(s) == 2

    def test_metadata_default_not_shared(self):
        a1 = Agent(id=0, features=np.array([1.0]))
        a2 = Agent(id=1, features=np.array([1.0]))
        a1.metadata["key"] = "val"
        assert "key" not in a2.metadata


class TestGroup:
    def test_creation(self):
        g = Group(id=0, members=[0, 1, 2])
        assert len(g.members) == 3
        assert g.properties == {}

    def test_properties_default_not_shared(self):
        g1 = Group(id=0, members=[0])
        g2 = Group(id=1, members=[1])
        g1.properties["key"] = "val"
        assert "key" not in g2.properties


class TestTheoryParameters:
    def test_creation(self):
        p = TheoryParameters(n_agents=10, n_features=3)
        assert p.n_agents == 10
        assert p.random_seed is None
        assert p.custom_params == {}
