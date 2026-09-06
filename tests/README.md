# `tests/` — property-based tests for every theorem implementation

> **Status:** populated in Step 2 of the release, alongside `src/`.

Each file targets one module in `src/bands/` or `src/baselines/` and, wherever the paper's
own theorem gives a checkable property (coverage, monotonicity, a known closed-form special
case), tests that property directly against synthetic data with a known ground truth —
mirroring the same falsification logic Section 5 of the paper itself uses, at unit-test scale
and speed. Planned: `test_uniform_band.py`, `test_lil_lower_bound.py`,
`test_population_band.py`, `test_block_robust.py`, `test_wsr_betting.py`,
`test_baselines_agree_with_reference_values.py`, `test_determinism.py`.

Run with `pytest` from the repository root once Step 2 lands.
