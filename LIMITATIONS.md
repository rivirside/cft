# Known Limitations

This document tracks known methodological and implementation limitations. Each item links to a GitHub issue for tracking.

---

## Methodology

### [#21] Arbitrary interaction weights
The affinity matrix is built from interaction counts weighted by hardcoded values (`follow=0.3`, `like=0.2`, `repost=0.4`, `pos_comment=0.3`, `neg_comment=-0.3`). These were chosen by intuition, not inferred from data. The theory comparison results are sensitive to these choices.

**Proposed fix:** Use MCMC to infer weights from a given population (see issue for design notes).

---

### [#22] Arbitrary theory parameters
Theory parameters (`CFT threshold=0.6`, `GFT k=0.1, sigma=1.0`, `TST temperature=1.0`) are set manually. A theory may appear to "win" simply because its parameters happened to fit the data well, not because it is the better model.

**Proposed fix:** Joint Bayesian inference over interaction weights and theory parameters; or at minimum a parameter sweep reporting best achievable NMI per theory.

---

### [#23] Circular ground truth
The affinity matrix and the Louvain ground truth are both derived from the same interaction dataset. This means we are measuring how well each theory re-derives the same clustering that the preprocessing already found - not whether the theory predicts genuinely independent structure.

**Proposed fix:** Temporal split - compute affinity from early interactions, define ground truth from late interactions. `temporal_prediction()` exists for this but is not used in the primary `compare_theories()` workflow.

---

### [#24] Theory comparison does not account for model complexity
A theory with more free parameters has more opportunity to fit the data. The current NMI-based ranking does not penalise complexity, so a more flexible theory can win by overfitting.

**Proposed fix:** Compare marginal likelihoods (integrating over parameter uncertainty via MCMC), or apply a penalty such as BIC/AIC.

---

## Implementation

### [#25] MBTI features are binary and coarse
MBTI personality types are encoded as four `{-1, +1}` binary dimensions. This is a crude representation that discards within-type variation and treats all INTJs as identical.

**Proposed fix:** Use continuous Big-Five (OCEAN) scores if available, or learn a continuous embedding from interaction data.

---

### [#26] MiroFish output requires manual conversion
The real MiroFish simulation produces OASIS-format event logs (CSV profiles, JSONL event streams with platform-specific schemas). The `MiroFishAdapter` expects a normalised `profiles.jsonl` + `actions.jsonl` format. Bridging the two currently requires hand-written conversion code.

**Proposed fix:** Add a `MiroFishAdapter.from_oasis_dir()` constructor that handles the OASIS event log format directly.

---

### [#27] Single simulation run - results may not generalise
Theory comparison results come from a single simulation with a fixed random seed. Stochastic theories (TST uses Metropolis-Hastings) and stochastic community detection (Louvain) mean rankings can vary across runs.

**Proposed fix:** Run multiple seeds, report mean ± std NMI per theory, and test for statistical significance before declaring a winner.

---
