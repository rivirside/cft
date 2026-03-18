"""Visualization utilities for CFT group formation theories.

All functions return matplotlib figure objects for customization.
Optional plotly backend for interactive Jupyter exploration.
"""

from typing import List, Dict, Any, Optional, Type

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from .theories.base import BehaviorTheory, Agent, Group, TheoryParameters

# Consistent color palette across all plots
PALETTE = list(mcolors.TABLEAU_COLORS.values())


def _get_colors(n: int) -> List[str]:
    """Get n colors from the palette, cycling if needed."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _reduce_to_2d(features: np.ndarray) -> np.ndarray:
    """Reduce features to 2D via PCA if needed."""
    if features.shape[1] <= 2:
        if features.shape[1] == 1:
            return np.column_stack([features, np.zeros(len(features))])
        return features

    # Simple PCA (no sklearn dependency)
    centered = features - features.mean(axis=0)
    cov = centered.T @ centered / (len(features) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top 2 components (eigenvalues are ascending)
    top2 = eigenvectors[:, -2:][:, ::-1]
    return centered @ top2


def plot_groups(
    groups: List[Group],
    agents: List[Agent],
    title: str = "Group Structure",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """2D scatter plot of agents colored by group membership.

    Uses PCA for dimensionality reduction if features > 2.
    """
    features = np.array([a.features for a in agents])
    coords = _reduce_to_2d(features)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    colors = _get_colors(len(groups))

    for idx, group in enumerate(groups):
        members = group.members
        ax.scatter(
            coords[members, 0],
            coords[members, 1],
            c=colors[idx],
            label=f"Group {group.id} ({len(members)})",
            s=60,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1" if features.shape[1] > 2 else "Feature 1")
    ax.set_ylabel("PC2" if features.shape[1] > 2 else "Feature 2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig


def plot_affinity_matrix(
    matrix: np.ndarray,
    groups: Optional[List[Group]] = None,
    title: str = "Affinity Matrix",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Heatmap of affinity matrix with optional group structure overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    # Reorder by group membership if groups provided
    if groups:
        order = []
        for g in groups:
            order.extend(g.members)
        matrix = matrix[np.ix_(order, order)]

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    fig.colorbar(im, ax=ax, label="Affinity")

    # Draw group boundaries
    if groups:
        cumsum = 0
        for g in groups[:-1]:
            cumsum += len(g.members)
            ax.axhline(y=cumsum - 0.5, color="black", linewidth=1.5)
            ax.axvline(x=cumsum - 0.5, color="black", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")

    return fig


def plot_theory_comparison(
    results: Dict[str, List[Dict[str, Any]]],
    agents: List[Agent],
    title: str = "Theory Comparison",
) -> plt.Figure:
    """Side-by-side panels showing final group structure from each theory."""
    n_theories = len(results)
    fig, axes = plt.subplots(1, n_theories, figsize=(5 * n_theories, 5))
    if n_theories == 1:
        axes = [axes]

    features = np.array([a.features for a in agents])
    coords = _reduce_to_2d(features)

    for ax, (theory_name, history) in zip(axes, results.items()):
        groups = history[-1]["groups"] if history else []
        colors = _get_colors(len(groups))

        for idx, group in enumerate(groups):
            members = group.members
            ax.scatter(
                coords[members, 0],
                coords[members, 1],
                c=colors[idx],
                s=50,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_title(f"{theory_name}\n({len(groups)} groups)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_convergence(
    results: Dict[str, List[Dict[str, Any]]],
    title: str = "Convergence",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot group count over time for each theory."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    colors = _get_colors(len(results))

    for (theory_name, history), color in zip(results.items(), colors):
        times = [h["time"] for h in history]
        n_groups = [len(h["groups"]) for h in history]
        ax.plot(times, n_groups, label=theory_name, color=color, linewidth=2, marker="o", markersize=4)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Groups")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig


def plot_parameter_sweep(
    theory_class: Type[BehaviorTheory],
    param_name: str,
    param_values: List[float],
    agents: List[Agent],
    params: TheoryParameters,
    t_max: float = 10.0,
    dt: float = 1.0,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Sweep a parameter and plot number of groups vs parameter value."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    n_groups_list = []
    for val in param_values:
        kwargs = {param_name: val}
        theory = theory_class(params, **kwargs)
        theory.initialize_agents(agents)
        theory.run_simulation(t_max=t_max, dt=dt)
        n_groups_list.append(len(theory.get_groups()))

    ax.plot(param_values, n_groups_list, "o-", color=PALETTE[0], linewidth=2, markersize=6)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Number of Groups")
    ax.set_title(title or f"{theory_class.__name__}: {param_name} sweep")
    ax.grid(True, alpha=0.3)

    return fig


def animate_simulation(
    history: List[Dict[str, Any]],
    agents: List[Agent],
    title: str = "Simulation",
    interval: int = 500,
) -> FuncAnimation:
    """Animate group formation over time.

    Returns a FuncAnimation object. In Jupyter, it renders as HTML5 video.
    For file export, call `anim.save("output.gif", writer="pillow")`.
    """
    features = np.array([a.features for a in agents])
    coords = _reduce_to_2d(features)

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        step = history[frame]
        groups = step["groups"]
        colors = _get_colors(len(groups))

        for idx, group in enumerate(groups):
            members = group.members
            ax.scatter(
                coords[members, 0],
                coords[members, 1],
                c=colors[idx],
                s=60,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_title(f"{title} - t={step['time']:.1f} ({len(groups)} groups)")
        ax.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=False)
    return anim
