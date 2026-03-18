"""Shared affinity computation utilities.

Supports multiple metrics from the paper: Euclidean, cosine, correlation, probabilistic.
All functions return symmetric matrices with values in [-1, 1].
"""

from typing import List, Optional

import numpy as np
from scipy.spatial.distance import cdist

from .theories.base import Agent


def compute_affinity_matrix(
    agents: List[Agent],
    metric: str = "euclidean",
    n_features: Optional[int] = None,
) -> np.ndarray:
    """Compute pairwise affinity matrix from agent feature vectors.

    Args:
        agents: List of agents with feature vectors.
        metric: One of "euclidean", "cosine", "correlation", "probabilistic".
        n_features: Number of features (used for euclidean normalization).
            If None, inferred from agent feature vectors.

    Returns:
        Symmetric (n_agents, n_agents) matrix with values in [-1, 1].
    """
    features = np.array([a.features for a in agents], dtype=float)
    n = len(agents)

    if n_features is None:
        n_features = features.shape[1]

    if metric == "euclidean":
        return _euclidean_affinity(features, n_features)
    elif metric == "cosine":
        return _cosine_affinity(features)
    elif metric == "correlation":
        return _correlation_affinity(features)
    elif metric == "probabilistic":
        return _probabilistic_affinity(features)
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Use: euclidean, cosine, correlation, probabilistic")


def _euclidean_affinity(features: np.ndarray, n_features: int) -> np.ndarray:
    """Bipolar affinity from normalized Euclidean distance.

    α_ij = 1 - d_ij / max_d, mapped to [-1, 1] via 2α - 1.
    Simplified: α_ij = 1 - ||x_i - x_j|| / sqrt(n_features).
    """
    dists = cdist(features, features, metric="euclidean")
    affinity = 1.0 - dists / np.sqrt(n_features)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _cosine_affinity(features: np.ndarray) -> np.ndarray:
    """Cosine similarity, naturally in [-1, 1]."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = features / norms
    affinity = normalized @ normalized.T
    np.fill_diagonal(affinity, 1.0)
    return np.clip(affinity, -1.0, 1.0)


def _correlation_affinity(features: np.ndarray) -> np.ndarray:
    """Pearson correlation between feature vectors, naturally in [-1, 1]."""
    n = len(features)
    if features.shape[1] < 2:
        # Correlation undefined for 1 feature; fall back to cosine
        return _cosine_affinity(features)

    # Center each agent's features
    centered = features - features.mean(axis=1, keepdims=True)
    stds = np.linalg.norm(centered, axis=1, keepdims=True)
    stds = np.maximum(stds, 1e-10)
    standardized = centered / stds

    affinity = standardized @ standardized.T
    np.fill_diagonal(affinity, 1.0)
    return np.clip(affinity, -1.0, 1.0)


def _probabilistic_affinity(features: np.ndarray) -> np.ndarray:
    """Probabilistic affinity via logit transform of interaction probability.

    Uses feature similarity as proxy for cooperative probability p_ij,
    then maps: α_ij = (2/π) * arctan(log(p_ij / (1 - p_ij))).
    """
    # Compute similarity as proxy for cooperation probability
    dists = cdist(features, features, metric="euclidean")
    max_dist = np.maximum(dists.max(), 1e-10)
    p = 1.0 - dists / max_dist  # probability in [0, 1]

    # Clamp away from 0 and 1 to avoid log(0)
    p = np.clip(p, 0.01, 0.99)

    # Logit transform mapped to [-1, 1]
    logit = np.log(p / (1.0 - p))
    affinity = (2.0 / np.pi) * np.arctan(logit)

    np.fill_diagonal(affinity, 1.0)
    return affinity
