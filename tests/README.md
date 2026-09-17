# `tests/` — property-based tests for every theorem implementation

62 tests across 10 files, all passing. Each file targets one module in `src/bands/` or
`src/baselines/` (plus supporting modules in `src/eval/`, `src/data/`, `src/scoring/`,
`src/utils/`) and, wherever the paper's own theorem or table gives a checkable property or
number, tests it directly:

| File | Checks |
|---|---|
| `test_uniform_band.py` | Theorem 1, Corollary 1 — including an exact match to the paper's own worked example (widths 0.034 / 0.118) and a regression test for a real overflow bug found during development |
| `test_lil_lower_bound.py` | Theorem 2 — violation rate climbs monotonically as the band is shrunk below its proven width |
| `test_population_band.py` | Theorem 3, Corollaries 2-3, the DKW embedding — 6 of 8 reference numbers matched exactly against the paper's own tables |
| `test_block_robust.py` | Theorem 4 — item-level certification breaks under induced correlation, block-level does not |
| `test_wsr_betting.py` | The betting bound, its documented silent-margin defect, and the Lemma 1 patch — including a direct empirical reproduction of the defect firing far more often on heterogeneous than i.i.d. data |
| `test_baselines.py` | Clopper-Pearson, Hoeffding, empirical Bernstein, Learn-then-Test, conformal risk control against known reference values |
| `test_cluster_bootstrap.py` | The setting-level bootstrap CI correction (Appendix D.4) |
| `test_determinism.py` | Seed derivation is deterministic and collision-free across arms/replications |
| `test_self_checks.py` | The three self-checks correctly *fail* when given deliberately non-deterministic input, not just pass on well-behaved input |
| `test_scoring_and_eval.py` | The seven score families, pool construction, the Gaussian copula, selection rules, Tier A/C protocols |

Run with `pytest` (or `pytest --no-cov` for a faster run without the coverage report) from
the repository root.
