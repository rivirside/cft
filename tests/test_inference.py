"""Tests for cft.inference - MCMC weight/parameter inference (issues #21, #22, #24)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cft import CFT, GFT, TST
from cft.integrations.mirofish import MiroFishAdapter
from cft.inference import (
    MCMCInference,
    MCMCResult,
    ParameterSpec,
    DEFAULT_WEIGHT_SPECS,
    DEFAULT_THEORY_PARAM_SPECS,
    compare_theories_by_evidence,
)

# Skip if optional deps missing
pd = pytest.importorskip("pandas")
nx = pytest.importorskip("networkx")

FIXTURES = Path(__file__).parent / "fixtures"

# Use tiny n_samples/burn_in so tests run fast
_FAST_MCMC = dict(n_samples=50, burn_in=20, step_size=0.05)


def _make_adapter():
    adapter = MiroFishAdapter(FIXTURES)
    adapter.load_agents()
    adapter.load_interactions()
    return adapter


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------


class TestParameterSpec:
    def test_fields(self):
        spec = ParameterSpec("threshold", 0.1, 0.99, 0.6)
        assert spec.name == "threshold"
        assert spec.low == 0.1
        assert spec.high == 0.99
        assert spec.initial == 0.6

    def test_default_weight_specs_count(self):
        assert len(DEFAULT_WEIGHT_SPECS) == 5

    def test_default_weight_specs_names(self):
        names = {s.name for s in DEFAULT_WEIGHT_SPECS}
        assert names == {"follow", "like", "repost", "pos_comment", "neg_comment"}

    def test_default_theory_param_specs_keys(self):
        assert set(DEFAULT_THEORY_PARAM_SPECS.keys()) == {"CFT", "GFT", "TST", "QST", "ICT"}


# ---------------------------------------------------------------------------
# MCMCInference construction
# ---------------------------------------------------------------------------


class TestMCMCInferenceInit:
    def test_construction(self):
        adapter = _make_adapter()
        mcmc = MCMCInference(adapter, CFT, seed=0)
        assert mcmc.theory_class is CFT
        assert len(mcmc._agents) == 8

    def test_train_eval_split(self):
        adapter = _make_adapter()
        mcmc = MCMCInference(adapter, CFT, split_fraction=0.5, seed=0)
        n_total = len(adapter._interactions)
        n_train = len(mcmc._train_df)
        n_eval = len(mcmc._eval_df)
        assert n_train + n_eval == n_total
        assert n_train > 0
        assert n_eval > 0

    def test_custom_split_fraction(self):
        adapter = _make_adapter()
        mcmc = MCMCInference(adapter, CFT, split_fraction=0.3, seed=0)
        n_total = len(adapter._interactions)
        assert len(mcmc._train_df) == int(n_total * 0.3)


# ---------------------------------------------------------------------------
# MCMCResult fields
# ---------------------------------------------------------------------------


class TestInferWeights:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.mcmc = MCMCInference(self.adapter, CFT, seed=42)

    def test_returns_mcmc_result(self):
        result = self.mcmc.infer_weights(**_FAST_MCMC)
        assert isinstance(result, MCMCResult)

    def test_chain_shape(self):
        result = self.mcmc.infer_weights(**_FAST_MCMC)
        assert result.chain.shape == (_FAST_MCMC["n_samples"], len(DEFAULT_WEIGHT_SPECS))

    def test_param_names_match_specs(self):
        result = self.mcmc.infer_weights(**_FAST_MCMC)
        assert result.param_names == [s.name for s in DEFAULT_WEIGHT_SPECS]

    def test_acceptance_rate_in_range(self):
        result = self.mcmc.infer_weights(**_FAST_MCMC)
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_posterior_mean_keys(self):
        result = self.mcmc.infer_weights(**_FAST_MCMC)
        assert set(result.posterior_mean.keys()) == {s.name for s in DEFAULT_WEIGHT_SPECS}

    def test_map_estimate_keys(self):
        result = self.mcmc.infer_weights(**_FAST_MCMC)
        assert set(result.map_estimate.keys()) == {s.name for s in DEFAULT_WEIGHT_SPECS}

    def test_log_marginal_likelihood_finite(self):
        result = self.mcmc.infer_weights(**_FAST_MCMC)
        assert np.isfinite(result.log_marginal_likelihood)

    def test_chain_respects_bounds(self):
        result = self.mcmc.infer_weights(**_FAST_MCMC)
        for i, spec in enumerate(DEFAULT_WEIGHT_SPECS):
            assert np.all(result.chain[:, i] >= spec.low)
            assert np.all(result.chain[:, i] <= spec.high)

    def test_deterministic_with_same_seed(self):
        a1 = MCMCInference(self.adapter, CFT, seed=7)
        a2 = MCMCInference(self.adapter, CFT, seed=7)
        r1 = a1.infer_weights(**_FAST_MCMC)
        r2 = a2.infer_weights(**_FAST_MCMC)
        np.testing.assert_array_equal(r1.chain, r2.chain)

    def test_different_seeds_give_different_chains(self):
        a1 = MCMCInference(self.adapter, CFT, seed=1)
        a2 = MCMCInference(self.adapter, CFT, seed=2)
        r1 = a1.infer_weights(**_FAST_MCMC)
        r2 = a2.infer_weights(**_FAST_MCMC)
        assert not np.array_equal(r1.chain, r2.chain)


# ---------------------------------------------------------------------------
# infer_theory_params
# ---------------------------------------------------------------------------


class TestInferTheoryParams:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.mcmc = MCMCInference(self.adapter, CFT, seed=99)
        self.cft_specs = DEFAULT_THEORY_PARAM_SPECS["CFT"]

    def test_returns_mcmc_result(self):
        result = self.mcmc.infer_theory_params(self.cft_specs, **_FAST_MCMC)
        assert isinstance(result, MCMCResult)

    def test_chain_shape(self):
        result = self.mcmc.infer_theory_params(self.cft_specs, **_FAST_MCMC)
        assert result.chain.shape == (_FAST_MCMC["n_samples"], len(self.cft_specs))

    def test_threshold_stays_in_bounds(self):
        result = self.mcmc.infer_theory_params(self.cft_specs, **_FAST_MCMC)
        assert np.all(result.chain[:, 0] >= 0.1)
        assert np.all(result.chain[:, 0] <= 0.99)

    def test_log_marginal_likelihood_finite(self):
        result = self.mcmc.infer_theory_params(self.cft_specs, **_FAST_MCMC)
        assert np.isfinite(result.log_marginal_likelihood)


# ---------------------------------------------------------------------------
# infer_joint
# ---------------------------------------------------------------------------


class TestInferJoint:
    def test_joint_chain_wider(self):
        """Joint inference has n_weights + n_params columns."""
        adapter = _make_adapter()
        mcmc = MCMCInference(adapter, CFT, seed=0)
        cft_specs = DEFAULT_THEORY_PARAM_SPECS["CFT"]
        result = mcmc.infer_joint(param_specs=cft_specs, **_FAST_MCMC)
        expected_cols = len(DEFAULT_WEIGHT_SPECS) + len(cft_specs)
        assert result.chain.shape == (_FAST_MCMC["n_samples"], expected_cols)

    def test_no_param_specs_same_as_infer_weights(self):
        """infer_joint with no param_specs == infer_weights."""
        adapter = _make_adapter()
        mcmc = MCMCInference(adapter, CFT, seed=5)
        r_joint = mcmc.infer_joint(param_specs=[], **_FAST_MCMC)
        assert r_joint.chain.shape[1] == len(DEFAULT_WEIGHT_SPECS)


# ---------------------------------------------------------------------------
# log_marginal_likelihood
# ---------------------------------------------------------------------------


class TestLogMarginalLikelihood:
    def test_returns_float(self):
        adapter = _make_adapter()
        mcmc = MCMCInference(adapter, CFT, seed=0)
        lml = mcmc.log_marginal_likelihood(
            n_samples=_FAST_MCMC["n_samples"],
            burn_in=_FAST_MCMC["burn_in"],
        )
        assert isinstance(lml, float)
        assert np.isfinite(lml)


# ---------------------------------------------------------------------------
# compare_theories_by_evidence
# ---------------------------------------------------------------------------


class TestCompareTheoriesByEvidence:
    def test_returns_all_theories(self):
        adapter = _make_adapter()
        results = compare_theories_by_evidence(
            adapter,
            {"CFT": CFT, "GFT": GFT},
            n_samples=_FAST_MCMC["n_samples"],
            burn_in=_FAST_MCMC["burn_in"],
            seed=0,
        )
        assert set(results.keys()) == {"CFT", "GFT"}

    def test_sorted_descending(self):
        adapter = _make_adapter()
        results = compare_theories_by_evidence(
            adapter,
            {"CFT": CFT, "GFT": GFT},
            n_samples=_FAST_MCMC["n_samples"],
            burn_in=_FAST_MCMC["burn_in"],
            seed=0,
        )
        values = list(results.values())
        assert values == sorted(values, reverse=True)

    def test_values_finite(self):
        adapter = _make_adapter()
        results = compare_theories_by_evidence(
            adapter,
            {"CFT": CFT},
            n_samples=_FAST_MCMC["n_samples"],
            burn_in=_FAST_MCMC["burn_in"],
            seed=0,
        )
        assert np.isfinite(results["CFT"])
