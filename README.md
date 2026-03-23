# CFT: Computational Social Dynamics

A Python framework for modeling how human groups form, shift, and fracture. Simulate interventions, measure resilience, and test which social theories best predict collective behavior.

## Install

```bash
pip install -e .
```

## Core Ideas

**CFT** (Coalition Formation Theory) models groups as threshold-based cliques in affinity space. **DCT** (Dual-Context Theory) extends this with two coupled layers - proximity (who you're near) and alignment (what you believe) - where each agent has individual seeking and conformity rates.

Four additional theories (GFT, QST, ICT, TST) serve as competing baselines for comparison.

## Quick Start

```python
from cft import SocialSimulator, HypothesisTester

sim = SocialSimulator(n_agents=40, scenario="clustered", k=3, T=30, seed=42)
ht = HypothesisTester(simulator=sim)

result = ht.compare_theories()
print(result["rankings"])
```

## Six Theories

| Theory | Mechanism | Key Param | Best For |
|--------|-----------|-----------|----------|
| **CFT** | Threshold cliques | `threshold` | Sharp boundaries, stable groups |
| **DCT** | Dual-context layers | `mu`, `lam` | Code-switching, belief vs proximity |
| **GFT** | Force-based clustering | `k`, `sigma` | Continuous drift, fuzzy boundaries |
| **QST** | Quantum superposition | `n_states` | Uncertainty, observer effects |
| **ICT** | Information cascades | `bandwidth` | Communication-limited settings |
| **TST** | Potts model | `temperature` | Phase transitions, large systems |

## Interventions

Model perturbations and measure how groups respond:

```python
from cft import DCT, InterventionRunner, InterventionReport
from cft import RemoveAgents, ShiftFeatures, NoiseShock, SustainedNoise
from cft import Agent, TheoryParameters
import numpy as np

params = TheoryParameters(n_agents=30, n_features=4, random_seed=42)
agents = [Agent(id=i, features=np.random.default_rng(42).normal(0, 1, 4)) for i in range(30)]

theory = DCT(params, mu=0.3, lam=0.05)
theory.initialize_agents(agents)

runner = InterventionRunner(
    theory,
    interventions=[
        RemoveAgents(time=5.0, agent_ids=[0, 1]),       # remove leaders
        ShiftFeatures(time=10.0, agent_ids=[5,6,7], delta=np.array([0,0,0,2])),  # propaganda
    ],
    sustained=[
        SustainedNoise(start=15.0, end=25.0, sigma=0.5),  # crisis period
    ],
)
snapshots = runner.run(t_max=30.0, dt=1.0)

report = InterventionReport(snapshots, runner.log)
print(report.summary())
print(report.resilience_scores)
print(report.vulnerability_ranking())
```

**7 point-in-time interventions:** RemoveAgents, ShiftFeatures, AddAgent, NoiseShock, ModifyAffinity, ShiftProximity, ShiftAlignment

**3 sustained interventions:** SustainedShift, SustainedNoise, SustainedAffinityBias

**Resilience analysis:** stability curves, fracture/merge detection, group survival, vulnerability ranking

## TraitMap

Derive behavioral parameters from agent personality features instead of setting them globally:

```python
from cft import DCT, TraitMap

# From MBTI: extraversion -> seeking rate, judging/perceiving -> conformity
trait_map = TraitMap.from_preset("mbti")

theory = DCT(params, trait_map=trait_map)
theory.initialize_agents(agents)
# Each agent now has individual mu and lam derived from their features
```

## DCT: Dual-Context Theory

DCT models two coupled social layers:
- **Proximity** (fast) - who you're physically near, shifts quickly
- **Alignment** (slow) - what you believe, changes gradually

Groups form at the intersection. Agents can be near people they disagree with (workplace) or aligned with people far away (online communities).

```python
theory = DCT(
    params,
    mu=0.3,              # seeking rate (or per-agent array)
    lam=0.05,            # conformity rate (or per-agent array)
    proximity_matrix=P,  # optional: separate proximity data
    alignment_features=A, # optional: separate alignment data
)
```

## MCMC Inference

Infer parameters from data instead of guessing:

```python
from cft import MCMCInference, compare_theories_by_evidence, CFT, GFT

mcmc = MCMCInference(adapter, CFT, seed=42)
result = mcmc.infer_weights(n_samples=2000, burn_in=500)
print(result.map_estimate)

# Complexity-penalized theory ranking
log_ml = compare_theories_by_evidence(adapter, {"CFT": CFT, "GFT": GFT, "DCT": DCT})
```

## AutoResearch

An automated experiment pipeline using GitHub Actions and the Claude API. The agent reads the experiment queue, runs simulations, interprets results, and logs findings.

```bash
# Trigger manually
gh workflow run research.yml
```

See `research/queue.yaml` for pending experiments and `research/log.md` for results.

## Scenarios

`SocialSimulator` generates synthetic data in four regimes: `random`, `clustered`, `polarized`, `hierarchical`.

## Scoring

`PredictionTournament` scores theories with:
- **PAS** - Prediction Accuracy Score (group count + partition similarity)
- **DFI** - Dynamic Fidelity Index (temporal evolution accuracy)
- **PSS** - Parameter Sensitivity Score (stability under perturbation)
- **CTAI** - Cross-Theory Agreement Index (inter-theory consensus)

## Project Structure

```
cft/
  theories/          # CFT, GFT, QST, ICT, TST, DCT
  interventions.py   # 10 intervention types, runner, resilience report
  affinity.py        # 4 affinity metrics
  comparator.py      # NMI, Jaccard, pair agreement
  tournament.py      # PredictionTournament
  simulator.py       # SocialSimulator (4 scenarios)
  hypothesis.py      # HypothesisTester
  inference.py       # MCMC parameter inference
  visualization.py   # Plotting and animation
  integrations/
    mirofish.py      # MiroFish/OASIS adapter
docs/                # Interactive docs site
research/            # AutoResearch pipeline
tests/               # 371+ tests
```

## Installation

```bash
git clone https://github.com/rivirside/cft.git
cd cft
pip install -e ".[dev]"
pytest
```

Optional extras: `[viz]` (plotly), `[mirofish]` (pandas, networkx), `[all]` (everything).

## License

MIT
