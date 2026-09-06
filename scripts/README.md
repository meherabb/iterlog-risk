# `scripts/` — one-command entry points

> **Status:** populated in Step 3 of the release.

Thin CLI wrappers around `src/eval/` harnesses — no experimental logic lives here, only
argument parsing, config loading, and calling into `src/`. See
[`../docs/REPRODUCING.md`](../docs/REPRODUCING.md) for the full list with expected
hardware/time and the exact table/figure each one produces, and
[`../docs/CROSSWALK.md`](../docs/CROSSWALK.md) for the mapping back to the paper's own
notebook-cell crosswalk (Appendix J.2).

Planned scripts: `run_synthetic_validity.py`, `run_tier_a.py`, `run_tier_b.py`,
`run_deployment.py`, `run_wsr_audit.py`, `run_necessity_sweep.py`,
`run_block_correlation.py`, `run_group_conditional.py`, `run_exchangeability.py`,
`run_scaling_experiment.py`, `make_all_figures.py`.
