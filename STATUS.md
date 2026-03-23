# CFT Project Status

Last updated: 2026-03-21

## Current State

**371 passing tests** | **6 theories** | **pip-installable** | **docs site live**

### What's built

| Module | Description | Status |
|--------|-------------|--------|
| `cft/theories/` | 6 theories: CFT, GFT, QST, ICT, TST, DCT | Done |
| `cft/theories/dct.py` | Dual-Context Theory with per-agent mu/lam, TraitMap, separate sources | Done |
| `cft/interventions.py` | 7 point-in-time + 3 sustained interventions, InterventionRunner, resilience analysis | Done |
| `cft/affinity.py` | 4 affinity metrics (euclidean, cosine, correlation, probabilistic) | Done |
| `cft/comparator.py` | Jaccard, NMI, pair agreement, cross-theory analysis | Done |
| `cft/tournament.py` | PredictionTournament with PAS, DFI, PSS, CTAI scoring | Done |
| `cft/simulator.py` | SocialSimulator (random, clustered, polarized, hierarchical scenarios) | Done |
| `cft/hypothesis.py` | HypothesisTester: compare, sweep, temporal prediction, claims | Done |
| `cft/inference.py` | MCMC weight + parameter inference, marginal likelihood comparison | Done |
| `cft/integrations/mirofish.py` | MiroFish/OASIS adapter, temporal split, Louvain ground truth | Done |
| `cft/visualization.py` | plot_groups, plot_affinity_matrix, animate_simulation, parameter sweeps | Done |
| `docs/` | Interactive docs site with theory visualizations | Done |
| `research/` | AutoResearch pipeline (GitHub Actions + Claude API) | Done |

### Key architecture

- All theories share `BehaviorTheory` base class with `initialize_agents()`, `step()`, `get_groups()`, `get_state()`
- All theories accept optional `affinity_matrix` parameter for external data sources
- DCT is the primary theory (originated by project author); others serve as competing baselines
- InterventionRunner replaces `run_simulation()` loop when perturbations are needed
- TraitMap decouples agent personality schema from behavioral parameters

## Next Steps

### High priority

- [ ] **Continuous personality features** (issue #25) - replace binary MBTI {-1, +1} with Big Five (OCEAN) continuous scores or learned embeddings from interaction data
- [ ] **DCT + MiroFish integration** - run DCT on real AI-agent simulation data, derive TraitMap from interaction patterns, compare against other theories
- [ ] **Intervention scenario library** - pre-built scenarios for common questions: leader removal, propaganda campaigns, echo chamber formation, platform algorithm effects
- [ ] **Multi-run DCT validation** - run DCT with `n_runs=10` and Wilcoxon test against other theories on all 4 simulator scenarios

### Medium priority

- [ ] **DCT interactive visualization** - add DCT to `docs/explore.html` with dual-layer visualization showing proximity and alignment spaces side-by-side
- [ ] **Intervention visualization** - timeline plots showing stability curve, fracture/merge events, group survival over time
- [ ] **TraitMap from real data** - learn mu/lam mappings from observed agent behavior (who seeks, who conforms) rather than assumed personality dimensions
- [ ] **Notebook updates** - add notebooks demonstrating intervention system, DCT, TraitMap, sustained interventions

### Lower priority

- [ ] **Scalability** - profile and optimize for n > 100 agents (DCT spectral embedding is O(n^3))
- [ ] **More AutoResearch cycles** - trigger research pipeline with DCT parameter sweeps and intervention experiments
- [ ] **Cross-theory intervention comparison** - run same intervention on all 6 theories, compare resilience scores
- [ ] **Export formats** - save simulation state, intervention reports, and resilience metrics to JSON/CSV for external analysis

## Architecture Decisions

- DCT uses spectral embedding to derive initial positions from proximity/affinity matrices
- Sustained interventions use `is_active(current_time, dt)` check where activation happens when `start <= current_time + dt` (fires one step before start time)
- TraitMap uses sigmoid transform for index-based mappings, raw values for metadata-based
- InterventionReport computes NMI between consecutive snapshots for stability curve
- All theories are deterministic given `random_seed` (QST measurement mode is the exception)
