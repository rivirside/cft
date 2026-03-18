"""Base classes for the multi-theory framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np


@dataclass
class Agent:
    """Represents an individual agent in the system."""

    id: int
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


@dataclass
class Group:
    """Represents a group of agents."""

    id: int
    members: List[int]  # Agent IDs
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TheoryParameters:
    """Base parameters that all theories might use."""

    n_agents: int
    n_features: int
    random_seed: Optional[int] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)


class BehaviorTheory(ABC):
    """Abstract base class for theories of human behavior.

    All theories implement the same interface: initialize agents, step forward
    in time, and report current group structure. This enables direct comparison
    across different theoretical frameworks.
    """

    def __init__(self, params: TheoryParameters, affinity_matrix: Optional[np.ndarray] = None):
        self.params = params
        self.agents: List[Agent] = []
        self.history: List[Dict[str, Any]] = []
        self.current_time: float = 0.0
        self._affinity_matrix = affinity_matrix

    @abstractmethod
    def initialize_agents(self, agents: List[Agent]):
        """Initialize the theory with a list of agents."""
        pass

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the simulation by time dt."""
        pass

    @abstractmethod
    def get_groups(self) -> List[Group]:
        """Return current group configuration."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return full current state for analysis."""
        pass

    def run_simulation(self, t_max: float, dt: float = 1.0) -> List[Dict[str, Any]]:
        """Run simulation and return history."""
        history = []
        while self.current_time < t_max:
            self.step(dt)
            snapshot = {
                "time": self.current_time,
                "groups": self.get_groups(),
                "state": self.get_state(),
            }
            history.append(snapshot)
            self.history.append(snapshot)
        return history
