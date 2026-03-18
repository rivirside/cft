"""CFT: Multi-theory framework for modeling group formation and fracture dynamics."""

__version__ = "0.1.0"

from .theories import Agent, Group, TheoryParameters, BehaviorTheory, CFT, GFT, QST, ICT, TST
from .affinity import compute_affinity_matrix
from .comparator import TheoryComparator
from .tournament import PredictionTournament
from .simulator import SocialSimulator
from .hypothesis import HypothesisTester
from .inference import (
    MCMCInference,
    MCMCResult,
    ParameterSpec,
    compare_theories_by_evidence,
    DEFAULT_WEIGHT_SPECS,
    DEFAULT_THEORY_PARAM_SPECS,
)

__all__ = [
    "Agent",
    "Group",
    "TheoryParameters",
    "BehaviorTheory",
    "CFT",
    "GFT",
    "QST",
    "ICT",
    "TST",
    "TheoryComparator",
    "PredictionTournament",
    "compute_affinity_matrix",
    "SocialSimulator",
    "HypothesisTester",
    "MCMCInference",
    "MCMCResult",
    "ParameterSpec",
    "compare_theories_by_evidence",
    "DEFAULT_WEIGHT_SPECS",
    "DEFAULT_THEORY_PARAM_SPECS",
]
