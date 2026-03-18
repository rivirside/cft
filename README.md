# CFT: Multi-Theory Group Formation Framework

A Python library for modeling how groups form, stabilize, and fracture. Implements five competing theories of collective behavior and provides tools to compare their predictions, run statistical validation, and infer parameters from data.

## Quick Start

```bash
pip install -e .
```

```python
from cft import SocialSimulator, HypothesisTester

# Generate synthetic social data and compare all five theories
sim = SocialSimulator(n_agents=40, scenario="clustered", k=3, T=30, seed=42)
ht = HypothesisTester(simulator=sim)

result = ht.compare_theories()
print(result["rankings"])   # sorted by prediction accuracy
print(result["ctai"])       # cross-theory agreement index
```

## Five Theories

| Theory | Mechanism | Key Parameter | Best For |
|--------|-----------|---------------|----------|
| **CFT** | Threshold cliques | `threshold` | Sharp boundaries, stable groups |
| **GFT** | Force-based clustering | `k`, `sigma` | Continuous drift, fuzzy boundaries |
| **QST** | Quantum-inspired superposition | `n_states` | Uncertainty, observer effects |
| **ICT** | Information cascades | `bandwidth` | Communication-limited settings |
| **TST** | Potts model (statistical mechanics) | `temperature` | Phase transitions, large systems |

## Scenarios

`SocialSimulator` generates synthetic interaction data in four regimes:

```python
from cft import SocialSimulator

# Random - uniform MBTI, Gaussian opinions
sim = SocialSimulator(n_agents=50, scenario="random", T=20, seed=0)

# Clustered - k MBTI+opinion clusters
sim = SocialSimulator(n_agents=50, scenario="clustered", k=4, T=20, seed=0)

# Polarized - two opposing camps
sim = SocialSimulator(n_agents=50, scenario="polarized", bias_strength=0.9, T=20, seed=0)

# Hierarchical - influencers + followers
sim = SocialSimulator(n_agents=50, scenario="hierarchical", n_influencers=5, T=20, seed=0)
```

## Theory Comparison

```python
from cft import HypothesisTester

ht = HypothesisTester(simulator=sim)

# Fix circular ground-truth bias by using a temporal train/eval split (#23)
result = ht.compare_theories(use_temporal_split=True, split_fraction=0.5)

# Statistical validation across multiple independent runs (#27)
result = ht.compare_theories(n_runs=10, significance_level=0.05)
print(result["mean_similarity"])   # {theory: mean NMI}
print(result["wilcoxon_pvalue"])   # None if n_runs < 5

# Predict future group structure from early interactions
result = ht.temporal_prediction(t_freeze=10, t_predict=25)

# Sweep a simulator parameter
results = ht.parameter_sweep("beta", [0.5, 1.0, 2.0, 4.0, 8.0])

# Assert named theoretical claims
result = ht.test_claim("all_theories_agree_on_clustered", beta=6.0, ctai_threshold=0.6)
print(result["passed"])
```

## MCMC Parameter Inference

Instead of setting interaction weights and theory parameters by hand, infer them from data:

```python
from cft import MCMCInference, compare_theories_by_evidence, CFT, GFT
from cft.integrations.mirofish import MiroFishAdapter

adapter = MiroFishAdapter("sim_dir")
adapter.load_agents()
adapter.load_interactions()

# Infer posterior over interaction weights for CFT (#21)
mcmc = MCMCInference(adapter, CFT, seed=42)
result = mcmc.infer_weights(n_samples=2000, burn_in=500)
print(result.map_estimate)    # best-fit weights
print(result.posterior_mean)  # posterior average
print(result.posterior_std)   # uncertainty

# Infer theory-specific parameters (#22)
from cft import DEFAULT_THEORY_PARAM_SPECS
result = mcmc.infer_theory_params(DEFAULT_THEORY_PARAM_SPECS["CFT"])

# Joint inference over weights + parameters
result = mcmc.infer_joint(param_specs=DEFAULT_THEORY_PARAM_SPECS["CFT"])

# Complexity-penalised theory ranking via marginal likelihood (#24)
log_ml = compare_theories_by_evidence(adapter, {"CFT": CFT, "GFT": GFT})
print(log_ml)  # higher = better evidence (implicit Occam's razor)
```

## MiroFish / OASIS Integration

Load data from a MiroFish-Offline (OASIS format) simulation directory directly:

```python
from cft.integrations.mirofish import MiroFishAdapter

# Normalized profiles.jsonl + actions.jsonl format
adapter = MiroFishAdapter("/path/to/simulation")
adapter.load_agents()
adapter.load_interactions()

# OASIS event-log format (from MiroFish-Offline / camel-oasis) (#26)
adapter = MiroFishAdapter.from_oasis_dir("/path/to/oasis_sim")
# adapter has agents and interactions pre-loaded

affinity = adapter.compute_affinity_matrix()
communities = adapter.extract_ground_truth_groups()
adapter.cleanup_oasis()  # remove temp dir if created automatically
```

```bash
pip install -e ".[mirofish]"  # adds pandas, networkx
```

## Low-Level API

```python
from cft import Agent, TheoryParameters, CFT, PredictionTournament
import numpy as np

rng = np.random.default_rng(42)
agents = [Agent(id=i, features=rng.normal(0, 1, 3)) for i in range(20)]
params = TheoryParameters(n_agents=20, n_features=3, random_seed=42)

tournament = PredictionTournament(agents, params)
tournament.add_theory("CFT", CFT, threshold=0.5)
tournament.add_theory("GFT", GFT, k=0.3, sigma=2.0)
tournament.run(t_max=10.0, dt=1.0)
scores = tournament.score(ground_truth)
```

## Visualization

```python
from cft.visualization import plot_groups, plot_theory_comparison, plot_convergence

plot_groups(cft.get_groups(), agents)
plot_theory_comparison(histories, agents)
plot_convergence(histories)
```

## Notebooks

- [`quickstart.ipynb`](notebooks/quickstart.ipynb) - 20 agents, CFT + GFT, visualize
- [`theory_comparison.ipynb`](notebooks/theory_comparison.ipynb) - All 5 theories, tournament, parameter sweeps
- [`mirofish_demo.ipynb`](notebooks/mirofish_demo.ipynb) - Load real simulation data, predict, score

## Scoring Metrics

`PredictionTournament` scores theories with:

- **PAS** (Prediction Accuracy Score) - group count + partition similarity + size accuracy
- **DFI** (Dynamic Fidelity Index) - temporal evolution accuracy
- **PSS** (Parameter Sensitivity Score) - stability under parameter perturbation
- **CTAI** (Cross-Theory Agreement Index) - inter-theory consensus

## Known Limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) for documented methodological caveats and their status.

## Installation

```bash
git clone https://github.com/rivirside/cft.git
cd cft
pip install -e ".[dev]"   # includes pytest, ruff
pytest                     # 276+ tests
```

Optional extras: `[viz]` (plotly), `[mirofish]` (pandas, networkx), `[all]` (everything).

## Repository Structure

```
cft/
  theories/          # CFT, GFT, QST, ICT, TST implementations
  affinity.py        # Shared affinity computation (4 metrics)
  comparator.py      # TheoryComparator with NMI, Jaccard, pair agreement
  tournament.py      # PredictionTournament with PAS/DFI/PSS/CTAI
  simulator.py       # SocialSimulator (4 scenarios, no external deps)
  hypothesis.py      # HypothesisTester (compare, sweep, temporal, claims)
  inference.py       # MCMCInference over weights + theory parameters
  visualization.py   # Plotting and animation
  integrations/
    mirofish.py      # MiroFish/OASIS adapter
notebooks/           # Jupyter notebooks
tests/               # 276+ tests
LIMITATIONS.md       # Known methodological caveats
```

## License

MIT
