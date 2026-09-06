# Architecture: how the code maps onto the paper

This document is the single source of truth for "where is theorem X implemented." If the
code and the paper ever disagree, this mapping is what a reviewer or reader should use to
locate the relevant lines quickly. (See `docs/CROSSWALK.md` for the finer-grained,
per-table/per-figure version of this same idea, mirroring the paper's own Appendix~J.2.)

## The core reduction: the sorted filtration

Everything in this codebase follows from one idea, implemented once in
`src/bands/sorted_filtration.py`: sorting items by a score is a function of the covariates
alone, so conditioning on the sort order leaves the losses conditionally independent. That
single module is what every other file builds on:

```
src/bands/sorted_filtration.py
    └── build_filtration(scores, tie_breaker_seed) -> permutation π
    └── realized_and_empirical_curves(losses, eta, π) -> (R̄_k, R̂_k)
```

## Theorem-by-theorem map

| Paper object | File | Function |
|---|---|---|
| Definition 1 (Uniform validity) | `src/bands/validity.py` | `is_uniformly_valid` |
| Theorem 1 (Uniform risk–coverage band) | `src/bands/uniform_band.py` | `compute` |
| Eq. (menu) — the $\lambda_j$-menu construction | `src/bands/uniform_band.py` | `_lambda_menu` |
| Corollary 1 (Explicit closed-form band) | `src/bands/uniform_band.py` | `explicit_form` |
| Theorem 2 (Iterated-logarithm lower bound) | `src/bands/lil_lower_bound.py` | `homogeneous_world_violation_rate` |
| Theorem 3 (Population band) | `src/bands/population_band.py` | `compute` |
| Corollary 2 (Feasibility) | `src/bands/population_band.py` | `feasibility_threshold` |
| Corollary 3 (Joint risk–coverage certificate) | `src/bands/population_band.py` | `coverage_floor` |
| Theorem 4 (Block-robust band) | `src/bands/block_robust.py` | `compute_block_band` |
| Bhatia–Davis step in Theorem 4's proof | `src/bands/block_robust.py` | `block_variance_bound` |
| Lemma 1 (max-with-any-valid-bound patch) | `src/baselines/wsr_betting.py` | `patched_upper_bound` |
| DKW embedding (Appendix A.3 / Remark 1) | `src/bands/population_band.py` | `_dkw_embedding` |

## Baselines (`src/baselines/`)

Every baseline is implemented from scratch against its own original reference — none of them
call an external statistical package — precisely so that implementation-specific failure
modes (like the WSR silent-margin bisection defect audited in Appendix D.3) are ones we can
actually find and fix, rather than ones hidden inside someone else's library:

- `clopper_pearson.py` — exact binomial interval at a fixed $k$.
- `hoeffding.py`, `empirical_bernstein.py` — concentration-based fixed-$k$ bounds.
- `wsr_betting.py` — the betting/WSR upper confidence bound, its bisection-search
  implementation, the silent-margin defect, and the Lemma 1 patch.
- `learn_then_test.py` — Bonferroni-corrected grid certification (10/100/1000 points) and the
  fixed-sequence variant.
- `conformal_risk_control.py` — the in-expectation guarantee used as a non-high-probability
  reference point in the utility comparison.
- `transfer_corrected.py` — the population-transfer-corrected Clopper–Pearson/Hoeffding
  variants used to isolate transfer failure from selection failure (Section 5.3).

## Scoring (`src/scoring/`)

One file per score family, all sharing the same interface (`score(model, batch) -> np.ndarray`)
so any of the seven can be swapped into any evaluation harness without touching the harness:
`maxprob.py`, `entropy.py`, `seqlogprob.py`, `hidden_centroid.py`, `hidden_knn.py`,
`self_consistency.py`, `verbalized.py`.

## Evaluation harnesses (`src/eval/`)

- `tier_protocols.py` — Tier A (exact $\eta$), Tier B (leave-one-out estimated $\eta$), Tier C
  (held-out measurement) as three interchangeable loss-generation strategies over the same
  underlying scoring pipeline.
- `selection_rules.py` — the four post-hoc selection rules ($r_1, r_2', r_3, r_{\mathrm{adv}}$).
- `cluster_bootstrap.py` — the setting-level (not replication-level) bootstrap confidence
  intervals from Appendix D.4.
- `vision_tier_a.py` — the ResNet-18 / CIFAR-10 / CIFAR-10-C harness, kept separate from the
  LLM harness since it shares the Tier-A *protocol* but none of the scoring code.

## Why the layout is this way

Every experimental arm in the paper is a *combination* of (a) a loss-generation tier, (b) a
score family, (c) a certificate, and (d) sometimes a selection rule. Rather than one script
per arm with duplicated logic, `src/eval/` harnesses take all four as parameters, and
`scripts/*.py` are thin, one-command wrappers that pick a specific combination and a config
file — see `docs/REPRODUCING.md`. This is also why every property test in `tests/` targets a
`src/bands/` or `src/baselines/` function directly rather than an end-to-end script: the
theorem-level guarantee doesn't depend on which tier or score produced the losses, so it
shouldn't need tier- or score-specific tests to check it.
