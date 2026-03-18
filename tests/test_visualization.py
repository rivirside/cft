"""Tests for visualization module.

These are smoke tests - they verify that plots are created without errors,
not that they look correct (visual regression is out of scope).
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from cft import Agent, Group, TheoryParameters, CFT, GFT
from cft.visualization import (
    plot_groups,
    plot_affinity_matrix,
    plot_theory_comparison,
    plot_convergence,
    plot_parameter_sweep,
    animate_simulation,
)


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def viz_agents():
    rng = np.random.default_rng(42)
    return [Agent(id=i, features=rng.standard_normal(3)) for i in range(15)]


@pytest.fixture
def viz_params():
    return TheoryParameters(n_agents=15, n_features=3, random_seed=42)


@pytest.fixture
def viz_groups():
    return [
        Group(id=0, members=[0, 1, 2, 3, 4]),
        Group(id=1, members=[5, 6, 7, 8, 9]),
        Group(id=2, members=[10, 11, 12, 13, 14]),
    ]


class TestPlotGroups:
    def test_basic(self, viz_agents, viz_groups):
        fig = plot_groups(viz_groups, viz_agents)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self, viz_agents, viz_groups):
        fig = plot_groups(viz_groups, viz_agents, title="Custom Title")
        assert fig.axes[0].get_title() == "Custom Title"

    def test_with_existing_axes(self, viz_agents, viz_groups):
        fig, ax = plt.subplots()
        result = plot_groups(viz_groups, viz_agents, ax=ax)
        assert result is fig

    def test_2d_features(self):
        agents = [Agent(id=i, features=np.array([float(i), float(i)])) for i in range(5)]
        groups = [Group(id=0, members=[0, 1, 2]), Group(id=1, members=[3, 4])]
        fig = plot_groups(groups, agents)
        assert isinstance(fig, plt.Figure)

    def test_1d_features(self):
        agents = [Agent(id=i, features=np.array([float(i)])) for i in range(5)]
        groups = [Group(id=0, members=[0, 1, 2]), Group(id=1, members=[3, 4])]
        fig = plot_groups(groups, agents)
        assert isinstance(fig, plt.Figure)


class TestPlotAffinityMatrix:
    def test_basic(self):
        matrix = np.random.default_rng(0).standard_normal((10, 10))
        matrix = (matrix + matrix.T) / 2
        fig = plot_affinity_matrix(matrix)
        assert isinstance(fig, plt.Figure)

    def test_with_groups(self):
        matrix = np.random.default_rng(0).standard_normal((10, 10))
        matrix = (matrix + matrix.T) / 2
        groups = [Group(id=0, members=[0, 1, 2, 3, 4]), Group(id=1, members=[5, 6, 7, 8, 9])]
        fig = plot_affinity_matrix(matrix, groups=groups)
        assert isinstance(fig, plt.Figure)


class TestPlotTheoryComparison:
    def test_two_theories(self, viz_agents, viz_params):
        cft = CFT(viz_params, threshold=0.5)
        cft.initialize_agents(viz_agents)
        cft_history = cft.run_simulation(t_max=2.0, dt=1.0)

        gft = GFT(viz_params, k=0.1, sigma=1.0)
        gft.initialize_agents(viz_agents)
        gft_history = gft.run_simulation(t_max=2.0, dt=1.0)

        fig = plot_theory_comparison({"CFT": cft_history, "GFT": gft_history}, viz_agents)
        assert isinstance(fig, plt.Figure)


class TestPlotConvergence:
    def test_basic(self, viz_agents, viz_params):
        cft = CFT(viz_params, threshold=0.5)
        cft.initialize_agents(viz_agents)
        history = cft.run_simulation(t_max=5.0, dt=1.0)

        fig = plot_convergence({"CFT": history})
        assert isinstance(fig, plt.Figure)


class TestPlotParameterSweep:
    def test_cft_threshold(self, viz_agents, viz_params):
        fig = plot_parameter_sweep(
            CFT, "threshold", [0.1, 0.3, 0.5, 0.7, 0.9],
            viz_agents, viz_params, t_max=2.0, dt=1.0,
        )
        assert isinstance(fig, plt.Figure)


class TestAnimateSimulation:
    def test_returns_animation(self, viz_agents, viz_params):
        cft = CFT(viz_params, threshold=0.5)
        cft.initialize_agents(viz_agents)
        history = cft.run_simulation(t_max=3.0, dt=1.0)

        anim = animate_simulation(history, viz_agents)
        assert isinstance(anim, matplotlib.animation.FuncAnimation)
