# Contributing to CFT

## Setup

```bash
git clone https://github.com/rivirside/cft.git
cd cft
pip install -e ".[all]"
pytest
```

## Development Workflow

1. Fork and clone the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

## Code Standards

- **Style**: Follow PEP 8, line length 100 (enforced by ruff)
- **Types**: Use type annotations for public APIs
- **Tests**: Add tests for new functionality. Run `pytest` before submitting.
- **Dependencies**: Core code uses numpy/scipy/matplotlib only. Optional deps (pandas, networkx, plotly) must be lazy-imported.

## Adding a New Theory

1. Create `cft/theories/your_theory.py` subclassing `BehaviorTheory`
2. Implement: `initialize_agents()`, `step()`, `get_groups()`, `get_state()`
3. Accept optional `affinity_matrix` parameter for external affinity data
4. Use `compute_affinity_matrix()` from `cft.affinity` if computing your own
5. Add tests in `tests/test_your_theory.py`
6. Export from `cft/theories/__init__.py` and `cft/__init__.py`

## Project Structure

```
cft/
  theories/          # Theory implementations (subclass BehaviorTheory)
  affinity.py        # Shared affinity computation
  comparator.py      # Group structure comparison metrics
  tournament.py      # PredictionTournament scoring system
  visualization.py   # Plotting functions
  integrations/      # External data adapters
tests/               # pytest test suite
notebooks/           # Jupyter demos
```

## Reporting Issues

Use [GitHub Issues](https://github.com/rivirside/cft/issues). Include:
- What you expected vs. what happened
- Minimal reproduction steps
- Python version and OS
