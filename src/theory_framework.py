"""
Multi-Theory Framework for Human Behavior Simulation

This module provides abstract base classes and utilities for implementing
and comparing different theories of human group formation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


@dataclass
class Agent:
    """Represents an individual agent in the system."""
    id: int
    features: np.ndarray  # Position in feature space
    metadata: Dict[str, Any] = None
    

@dataclass  
class Group:
    """Represents a group of agents."""
    id: int
    members: List[int]  # Agent IDs
    properties: Dict[str, Any] = None
    

@dataclass
class TheoryParameters:
    """Base parameters that all theories might use."""
    n_agents: int
    n_features: int
    random_seed: Optional[int] = None
    custom_params: Dict[str, Any] = None


class BehaviorTheory(ABC):
    """Abstract base class for theories of human behavior."""
    
    def __init__(self, params: TheoryParameters):
        self.params = params
        self.agents: List[Agent] = []
        self.history: List[Dict[str, Any]] = []
        self.current_time: float = 0.0
        
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
            history.append({
                'time': self.current_time,
                'groups': self.get_groups(),
                'state': self.get_state()
            })
        return history


class CFT(BehaviorTheory):
    """Consensus-Fracture Theory implementation."""
    
    def __init__(self, params: TheoryParameters, threshold: float = 0.6):
        super().__init__(params)
        self.threshold = threshold
        self.affinity_matrix = None
        
    def initialize_agents(self, agents: List[Agent]):
        self.agents = agents
        # Calculate affinity matrix
        n = len(agents)
        self.affinity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # Simple affinity: negative distance in feature space
                dist = np.linalg.norm(agents[i].features - agents[j].features)
                affinity = 1.0 - dist / np.sqrt(self.params.n_features)
                self.affinity_matrix[i, j] = affinity
                self.affinity_matrix[j, i] = affinity
                
    def step(self, dt: float):
        # CFT reaches equilibrium quickly
        if self.current_time == 0:
            # Find maximal cliques meeting threshold
            # Simplified: greedy clustering
            self._form_groups()
        self.current_time += dt
        
    def _form_groups(self):
        """Simplified group formation algorithm."""
        n = len(self.agents)
        assigned = [False] * n
        self.groups = []
        group_id = 0
        
        for i in range(n):
            if assigned[i]:
                continue
                
            # Start new group with agent i
            group_members = [i]
            assigned[i] = True
            
            # Add compatible agents
            for j in range(i+1, n):
                if assigned[j]:
                    continue
                    
                # Check if j is compatible with all current members
                compatible = True
                for m in group_members:
                    if self.affinity_matrix[j, m] < self.threshold:
                        compatible = False
                        break
                        
                if compatible:
                    group_members.append(j)
                    assigned[j] = True
                    
            self.groups.append(Group(id=group_id, members=group_members))
            group_id += 1
            
    def get_groups(self) -> List[Group]:
        return self.groups
        
    def get_state(self) -> Dict[str, Any]:
        return {
            'affinity_matrix': self.affinity_matrix,
            'threshold': self.threshold,
            'n_groups': len(self.groups)
        }


class GFT(BehaviorTheory):
    """Gradient Field Theory implementation."""
    
    def __init__(self, params: TheoryParameters, k: float = 0.1, sigma: float = 1.0):
        super().__init__(params)
        self.k = k  # Attraction strength  
        self.sigma = sigma  # Interaction range
        self.positions = None
        
    def initialize_agents(self, agents: List[Agent]):
        self.agents = agents
        # Start with agents at their feature positions
        self.positions = np.array([a.features for a in agents])
        
    def step(self, dt: float):
        """Move agents along gradient field."""
        n = len(self.agents)
        forces = np.zeros_like(self.positions)
        
        # Calculate pairwise forces
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                # Vector from i to j
                r_ij = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(r_ij)
                
                if dist > 0:
                    # Attractive force with Gaussian decay
                    force_mag = self.k * np.exp(-dist**2 / (2 * self.sigma**2))
                    forces[i] += force_mag * r_ij / dist
                    
        # Update positions
        self.positions += forces * dt
        self.current_time += dt
        
    def get_groups(self) -> List[Group]:
        """Cluster agents based on position proximity."""
        # Simple distance-based clustering
        groups = []
        n = len(self.agents)
        assigned = [False] * n
        group_id = 0
        
        for i in range(n):
            if assigned[i]:
                continue
                
            # Find all agents within clustering distance
            group_members = [i]
            assigned[i] = True
            
            for j in range(i+1, n):
                if assigned[j]:
                    continue
                    
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.sigma:
                    group_members.append(j)
                    assigned[j] = True
                    
            groups.append(Group(id=group_id, members=group_members))
            group_id += 1
            
        return groups
        
    def get_state(self) -> Dict[str, Any]:
        return {
            'positions': self.positions.copy(),
            'k': self.k,
            'sigma': self.sigma
        }


class TheoryComparator:
    """Utilities for comparing predictions across theories."""
    
    @staticmethod
    def compare_group_structures(groups1: List[Group], groups2: List[Group]) -> float:
        """Calculate similarity between two group structures."""
        # Simplified: Jaccard similarity of group memberships
        all_agents = set()
        for g in groups1 + groups2:
            all_agents.update(g.members)
            
        similarity_sum = 0
        for agent in all_agents:
            # Find groups containing this agent
            g1 = [g for g in groups1 if agent in g.members]
            g2 = [g for g in groups2 if agent in g.members]
            
            if g1 and g2:
                # Compare co-membership
                members1 = set(g1[0].members)
                members2 = set(g2[0].members)
                
                intersection = len(members1.intersection(members2))
                union = len(members1.union(members2))
                
                if union > 0:
                    similarity_sum += intersection / union
                    
        return similarity_sum / len(all_agents) if all_agents else 0
        
    @staticmethod
    def analyze_predictions(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze predictions from multiple theories."""
        analysis = {
            'convergence_times': {},
            'final_group_counts': {},
            'stability_scores': {},
            'cross_theory_agreement': {}
        }
        
        theory_names = list(results.keys())
        
        for theory_name, history in results.items():
            if not history:
                continue
                
            # Convergence time (when groups stop changing)
            converged = False
            convergence_time = 0
            
            for i in range(1, len(history)):
                groups_prev = history[i-1]['groups']
                groups_curr = history[i]['groups']
                
                if TheoryComparator._groups_equal(groups_prev, groups_curr):
                    convergence_time = history[i]['time']
                    converged = True
                    break
                    
            analysis['convergence_times'][theory_name] = convergence_time if converged else None
            analysis['final_group_counts'][theory_name] = len(history[-1]['groups'])
            
        # Cross-theory agreement
        for i, theory1 in enumerate(theory_names):
            for theory2 in theory_names[i+1:]:
                if results[theory1] and results[theory2]:
                    groups1 = results[theory1][-1]['groups']
                    groups2 = results[theory2][-1]['groups']
                    
                    similarity = TheoryComparator.compare_group_structures(groups1, groups2)
                    analysis['cross_theory_agreement'][f"{theory1}-{theory2}"] = similarity
                    
        return analysis
        
    @staticmethod
    def _groups_equal(groups1: List[Group], groups2: List[Group]) -> bool:
        """Check if two group configurations are identical."""
        if len(groups1) != len(groups2):
            return False
            
        # Sort groups by size and member IDs for comparison
        def group_signature(g):
            return tuple(sorted(g.members))
            
        sigs1 = sorted([group_signature(g) for g in groups1])
        sigs2 = sorted([group_signature(g) for g in groups2])
        
        return sigs1 == sigs2


# Example usage
if __name__ == "__main__":
    # Create test agents
    np.random.seed(42)
    n_agents = 20
    n_features = 3
    
    agents = [
        Agent(id=i, features=np.random.randn(n_features))
        for i in range(n_agents)
    ]
    
    # Initialize parameters
    params = TheoryParameters(n_agents=n_agents, n_features=n_features)
    
    # Run simulations
    theories = {
        'CFT': CFT(params, threshold=0.5),
        'GFT': GFT(params, k=0.2, sigma=1.5)
    }
    
    results = {}
    for name, theory in theories.items():
        theory.initialize_agents(agents)
        history = theory.run_simulation(t_max=10.0, dt=0.5)
        results[name] = history
        
        print(f"\n{name} Results:")
        print(f"Final groups: {len(history[-1]['groups'])}")
        for g in history[-1]['groups']:
            print(f"  Group {g.id}: {len(g.members)} members")
    
    # Compare results
    print("\nComparison Analysis:")
    analysis = TheoryComparator.analyze_predictions(results)
    for key, value in analysis.items():
        print(f"{key}: {value}")