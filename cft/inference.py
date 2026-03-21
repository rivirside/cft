"""MCMC-based inference over interaction weights and theory parameters.

Addresses three methodological limitations:

- **#21** - Interaction weights (follow, like, repost, …) are inferred from
  data via MCMC rather than set by hand.
- **#22** - Theory parameters (CFT threshold, GFT k/sigma, …) are inferred
  from data rather than set manually.
- **#24** - Theory comparison accounts for model complexity via marginal
  likelihood (Bayes factor); theories with more free parameters are
  automatically penalised by integrating over parameter uncertainty.

Usage::

    from cft.integrations.mirofish import MiroFishAdapter
    from cft.inference import MCMCInference, compare_theories_by_evidence
    from cft.theories import CFT, GFT

    adapter = MiroFishAdapter("sim_dir")
    adapter.load_agents()
    adapter.load_interactions()

    # Infer posterior over interaction weights for CFT
    mcmc = MCMCInference(adapter, CFT, seed=42)
    result = mcmc.infer_weights(n_samples=2000, burn_in=500)
    print(result.map_estimate)   # best-fit weights
    print(result.posterior_mean) # posterior average

    # Complexity-penalised theory ranking
    log_ml = compare_theories_by_evidence(adapter, {"CFT": CFT, "GFT": GFT})
    print(log_ml)  # higher = better evidence
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from scipy.special import logsumexp

from .theories.base import BehaviorTheory, Agent, Group, TheoryParameters
from .comparator import TheoryComparator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpec:
    """Specification for a single parameter in MCMC inference.

    Parameters
    ----------
    name : str
        Parameter name (must match the kwarg accepted by the theory or the
        interaction weight key in DEFAULT_WEIGHTS).
    low : float
        Hard lower bound (reflection boundary).
    high : float
        Hard upper bound (reflection boundary).
    initial : float
        Starting value for the Markov chain.
    """

    name: str
    low: float
    high: float
    initial: float


@dataclass
class MCMCResult:
    """Result of a completed MCMC inference run.

    Attributes
    ----------
    chain : np.ndarray
        Post-burn-in samples, shape ``(n_samples, n_params)``.
    log_likelihoods : np.ndarray
        Log-likelihood evaluated at each sample, shape ``(n_samples,)``.
    param_names : list[str]
        Names corresponding to columns in ``chain``.
    acceptance_rate : float
        Fraction of proposals accepted during post-burn-in sampling.
    log_marginal_likelihood : float
        ``logsumexp(log_likelihoods) - log(n_samples)`` - an estimate of the
        log marginal likelihood (used for Bayes-factor theory comparison).
    posterior_mean : dict
        Mean of each parameter across the chain.
    posterior_std : dict
        Standard deviation of each parameter across the chain.
    map_estimate : dict
        Parameter values at the highest-likelihood sample.
    """

    chain: np.ndarray
    log_likelihoods: np.ndarray
    param_names: List[str]
    acceptance_rate: float
    log_marginal_likelihood: float
    posterior_mean: Dict[str, float]
    posterior_std: Dict[str, float]
    map_estimate: Dict[str, float]


# ---------------------------------------------------------------------------
# Default parameter specifications
# ---------------------------------------------------------------------------

DEFAULT_WEIGHT_SPECS: List[ParameterSpec] = [
    ParameterSpec("follow",      0.0,  1.0,  0.3),
    ParameterSpec("like",        0.0,  1.0,  0.2),
    ParameterSpec("repost",      0.0,  1.0,  0.4),
    ParameterSpec("pos_comment", 0.0,  1.0,  0.3),
    ParameterSpec("neg_comment", -1.0, 0.0, -0.3),
]

DEFAULT_THEORY_PARAM_SPECS: Dict[str, List[ParameterSpec]] = {
    "CFT": [ParameterSpec("threshold",   0.1,  0.99, 0.6)],
    "GFT": [
        ParameterSpec("k",     0.01, 2.0, 0.1),
        ParameterSpec("sigma", 0.1,  5.0, 1.0),
    ],
    "TST": [ParameterSpec("temperature", 0.05, 5.0, 1.0)],
    "QST": [ParameterSpec("n_states",    2.0, 20.0, 5.0)],
    "ICT": [ParameterSpec("bandwidth",   1.0, 10.0, 3.0)],
    "DCT": [
        ParameterSpec("mu",        0.01, 1.0, 0.3),
        ParameterSpec("lam",       0.001, 0.5, 0.05),
        ParameterSpec("threshold", 0.05, 0.9, 0.3),
    ],
}


# ---------------------------------------------------------------------------
# MCMCInference
# ---------------------------------------------------------------------------


class MCMCInference:
    """Metropolis-Hastings MCMC for inferring weights and/or theory parameters.

    Uses NMI as a likelihood proxy::

        log_likelihood(θ) = NMI(predicted_groups, held_out_groups)
                            / likelihood_temperature

    The held-out communities are derived from the later ``1 - split_fraction``
    of interactions (temporal split), while the affinity matrix is built from
    the earlier ``split_fraction``. This avoids the circular ground-truth
    problem (issue #23) during inference.

    Parameters
    ----------
    adapter : MiroFishAdapter
        Pre-loaded adapter (``load_agents()`` and ``load_interactions()`` must
        have been called).
    theory_class : type
        The :class:`~cft.theories.base.BehaviorTheory` subclass to evaluate.
    split_fraction : float
        Fraction of interactions used for training the affinity matrix; the
        remainder provides the held-out ground truth (default 0.5).
    t_max : float
        Theory simulation duration (default 10.0).
    dt : float
        Theory simulation timestep (default 1.0).
    likelihood_temperature : float
        Scales NMI into log-likelihood. Smaller values sharpen the posterior
        but may cause slow mixing (default 0.1).
    seed : int, optional
        Random seed for the MCMC sampler.
    """

    def __init__(
        self,
        adapter,
        theory_class: Type[BehaviorTheory],
        split_fraction: float = 0.5,
        t_max: float = 10.0,
        dt: float = 1.0,
        likelihood_temperature: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.adapter = adapter
        self.theory_class = theory_class
        self.split_fraction = split_fraction
        self.t_max = t_max
        self.dt = dt
        self.likelihood_temperature = likelihood_temperature
        self.rng = np.random.default_rng(seed)

        self._agents: List[Agent] = adapter._agents or []
        self._train_df, self._eval_df = self._split_interactions()

    # ── Interaction splitting ─────────────────────────────────────────────────

    def _split_interactions(self):
        """Split adapter interactions into train/eval DataFrames."""
        import pandas as pd

        df = self.adapter._interactions.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_sorted = df.sort_values("timestamp")
        cutoff = int(len(df_sorted) * self.split_fraction)
        return df_sorted.iloc[:cutoff].copy(), df_sorted.iloc[cutoff:].copy()

    # ── Likelihood evaluation ─────────────────────────────────────────────────

    def _log_likelihood(
        self,
        weights: Dict[str, float],
        theory_kwargs: Dict[str, Any],
    ) -> float:
        """Compute log-likelihood = NMI(predicted, held-out) / temperature."""
        try:
            affinity = self.adapter.compute_affinity_matrix(
                self._train_df, weights=weights
            )
            ground_truth = self.adapter.extract_ground_truth_groups(
                self._eval_df, weights=weights
            )
            if not ground_truth:
                return -1.0 / self.likelihood_temperature

            n = len(self._agents)
            n_feat = len(self._agents[0].features)
            params = TheoryParameters(
                n_agents=n, n_features=n_feat, random_seed=42
            )

            # Coerce integer-valued parameters
            tkw = dict(theory_kwargs)
            if "n_states" in tkw:
                tkw["n_states"] = max(2, int(round(tkw["n_states"])))
            if "bandwidth" in tkw:
                tkw["bandwidth"] = max(1, int(round(tkw["bandwidth"])))

            theory = self.theory_class(params, affinity_matrix=affinity, **tkw)
            theory.initialize_agents(self._agents)
            theory.run_simulation(t_max=self.t_max, dt=self.dt)

            predicted = theory.get_groups()
            nmi = TheoryComparator.compare_group_structures(
                predicted, ground_truth, metric="nmi"
            )
            return float(nmi) / self.likelihood_temperature

        except Exception as exc:
            logger.debug("Log-likelihood evaluation failed: %s", exc)
            return -10.0 / self.likelihood_temperature

    # ── Reflection boundary ───────────────────────────────────────────────────

    @staticmethod
    def _reflect(theta: np.ndarray, specs: List[ParameterSpec]) -> np.ndarray:
        """Fold-over reflection to keep θ within [low, high] for each spec."""
        theta = theta.copy()
        for i, spec in enumerate(specs):
            lo, hi = spec.low, spec.high
            width = hi - lo
            x = (theta[i] - lo) % (2.0 * width)
            if x > width:
                x = 2.0 * width - x
            theta[i] = lo + x
        return theta

    # ── Core MH sampler ───────────────────────────────────────────────────────

    def _run_mcmc(
        self,
        specs: List[ParameterSpec],
        log_lik_fn,
        n_samples: int,
        burn_in: int,
        step_size: float,
    ) -> MCMCResult:
        """Core Metropolis-Hastings sampling loop.

        Parameters
        ----------
        specs : list[ParameterSpec]
            Parameter specifications (bounds + initial values).
        log_lik_fn : callable
            Function ``(theta: np.ndarray) -> float`` returning log-likelihood.
        n_samples : int
            Post-burn-in samples to collect.
        burn_in : int
            Steps to discard before collecting.
        step_size : float
            Standard deviation of Gaussian proposal.

        Returns
        -------
        MCMCResult
        """
        n_params = len(specs)
        theta = np.array([s.initial for s in specs], dtype=float)
        current_ll = log_lik_fn(theta)

        chain = np.zeros((n_samples, n_params))
        log_liks = np.zeros(n_samples)
        n_accepted = 0

        for step in range(burn_in + n_samples):
            proposal = theta + self.rng.normal(0.0, step_size, n_params)
            proposal = self._reflect(proposal, specs)
            proposal_ll = log_lik_fn(proposal)

            if np.log(self.rng.uniform() + 1e-300) < (proposal_ll - current_ll):
                theta = proposal
                current_ll = proposal_ll
                if step >= burn_in:
                    n_accepted += 1

            if step >= burn_in:
                idx = step - burn_in
                chain[idx] = theta
                log_liks[idx] = current_ll

        acceptance_rate = n_accepted / n_samples
        log_ml = float(logsumexp(log_liks) - np.log(n_samples))

        names = [s.name for s in specs]
        posterior_mean = {names[i]: float(np.mean(chain[:, i])) for i in range(n_params)}
        posterior_std = {names[i]: float(np.std(chain[:, i])) for i in range(n_params)}
        map_idx = int(np.argmax(log_liks))
        map_estimate = {names[i]: float(chain[map_idx, i]) for i in range(n_params)}

        return MCMCResult(
            chain=chain,
            log_likelihoods=log_liks,
            param_names=names,
            acceptance_rate=acceptance_rate,
            log_marginal_likelihood=log_ml,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            map_estimate=map_estimate,
        )

    # ── Public inference methods ──────────────────────────────────────────────

    def infer_weights(
        self,
        weight_specs: Optional[List[ParameterSpec]] = None,
        fixed_theory_params: Optional[Dict[str, Any]] = None,
        n_samples: int = 2000,
        burn_in: int = 500,
        step_size: float = 0.05,
    ) -> MCMCResult:
        """Infer posterior over interaction weights (addresses issue #21).

        Parameters
        ----------
        weight_specs : list[ParameterSpec], optional
            Specifications for each weight. Defaults to
            :data:`DEFAULT_WEIGHT_SPECS`.
        fixed_theory_params : dict, optional
            Theory constructor kwargs held fixed during inference.
        n_samples : int
            Post-burn-in samples.
        burn_in : int
            Burn-in steps to discard.
        step_size : float
            Gaussian proposal standard deviation.

        Returns
        -------
        MCMCResult
        """
        specs = weight_specs or DEFAULT_WEIGHT_SPECS
        theory_kwargs = fixed_theory_params or {}

        def log_lik_fn(theta: np.ndarray) -> float:
            weights = {specs[i].name: float(theta[i]) for i in range(len(specs))}
            return self._log_likelihood(weights, theory_kwargs)

        return self._run_mcmc(specs, log_lik_fn, n_samples, burn_in, step_size)

    def infer_theory_params(
        self,
        param_specs: List[ParameterSpec],
        fixed_weights: Optional[Dict[str, float]] = None,
        n_samples: int = 2000,
        burn_in: int = 500,
        step_size: float = 0.05,
    ) -> MCMCResult:
        """Infer posterior over theory parameters (addresses issue #22).

        Parameters
        ----------
        param_specs : list[ParameterSpec]
            Specifications for each theory parameter. Use
            :data:`DEFAULT_THEORY_PARAM_SPECS` as a starting point.
        fixed_weights : dict, optional
            Interaction weights held fixed. Defaults to
            ``DEFAULT_WEIGHTS`` from the MiroFish adapter.
        n_samples : int
            Post-burn-in samples.
        burn_in : int
            Burn-in steps.
        step_size : float
            Gaussian proposal standard deviation.

        Returns
        -------
        MCMCResult
        """
        from .integrations.mirofish import DEFAULT_WEIGHTS

        weights = fixed_weights or dict(DEFAULT_WEIGHTS)

        def log_lik_fn(theta: np.ndarray) -> float:
            theory_kwargs = {
                param_specs[i].name: float(theta[i])
                for i in range(len(param_specs))
            }
            return self._log_likelihood(weights, theory_kwargs)

        return self._run_mcmc(param_specs, log_lik_fn, n_samples, burn_in, step_size)

    def infer_joint(
        self,
        weight_specs: Optional[List[ParameterSpec]] = None,
        param_specs: Optional[List[ParameterSpec]] = None,
        n_samples: int = 4000,
        burn_in: int = 1000,
        step_size: float = 0.03,
    ) -> MCMCResult:
        """Joint inference over interaction weights **and** theory parameters.

        Parameters
        ----------
        weight_specs : list[ParameterSpec], optional
            Defaults to :data:`DEFAULT_WEIGHT_SPECS`.
        param_specs : list[ParameterSpec], optional
            Theory parameter specs. If None, only weights are inferred (same
            as :meth:`infer_weights`).
        n_samples : int
            Post-burn-in samples (default 4000; larger for joint space).
        burn_in : int
            Burn-in steps (default 1000).
        step_size : float
            Gaussian proposal standard deviation.

        Returns
        -------
        MCMCResult
        """
        w_specs = weight_specs or DEFAULT_WEIGHT_SPECS
        p_specs = param_specs or []
        all_specs = w_specs + p_specs
        n_weights = len(w_specs)

        def log_lik_fn(theta: np.ndarray) -> float:
            weights = {w_specs[i].name: float(theta[i]) for i in range(n_weights)}
            theory_kwargs = {
                p_specs[i].name: float(theta[n_weights + i])
                for i in range(len(p_specs))
            }
            return self._log_likelihood(weights, theory_kwargs)

        return self._run_mcmc(all_specs, log_lik_fn, n_samples, burn_in, step_size)

    def log_marginal_likelihood(
        self,
        n_samples: int = 2000,
        burn_in: int = 500,
    ) -> float:
        """Estimate log marginal likelihood for this theory (addresses issue #24).

        Integrates over interaction-weight uncertainty using
        :data:`DEFAULT_WEIGHT_SPECS`. The result can be compared across
        theories as an implicit Occam's razor: a more complex theory must fit
        the data better to overcome the larger volume of its parameter space.

        Parameters
        ----------
        n_samples : int
            MCMC samples.
        burn_in : int
            Burn-in steps.

        Returns
        -------
        float
            ``logsumexp(log_likelihoods) - log(n_samples)``.
        """
        result = self.infer_weights(n_samples=n_samples, burn_in=burn_in)
        return result.log_marginal_likelihood


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def compare_theories_by_evidence(
    adapter,
    theory_classes: Dict[str, Type[BehaviorTheory]],
    n_samples: int = 2000,
    burn_in: int = 500,
    split_fraction: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Compare theories by marginal likelihood (complexity-penalised, issue #24).

    For each theory, runs MCMC over interaction weights and estimates the
    log marginal likelihood. Higher is better; the ratio of two log marginal
    likelihoods is the log Bayes factor.

    Parameters
    ----------
    adapter : MiroFishAdapter
        Pre-loaded adapter.
    theory_classes : dict
        ``{theory_name: TheoryClass}`` mapping.
    n_samples : int
        MCMC samples per theory (default 2000).
    burn_in : int
        MCMC burn-in per theory (default 500).
    split_fraction : float
        Train/eval split fraction (default 0.5).
    seed : int, optional
        Base random seed; theory ``i`` uses ``seed + i``.

    Returns
    -------
    dict
        ``{theory_name: log_marginal_likelihood}`` sorted best-first.
    """
    results: Dict[str, float] = {}
    for i, (name, cls) in enumerate(theory_classes.items()):
        theory_seed = (seed or 0) + i
        mcmc = MCMCInference(
            adapter=adapter,
            theory_class=cls,
            split_fraction=split_fraction,
            seed=theory_seed,
        )
        log_ml = mcmc.log_marginal_likelihood(n_samples=n_samples, burn_in=burn_in)
        results[name] = log_ml
        logger.info("Theory %s  log-marginal-likelihood: %.4f", name, log_ml)

    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
