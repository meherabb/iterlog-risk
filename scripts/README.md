# `scripts/` — one-command entry points

Thin CLI wrappers around `src/bands/`, `src/baselines/`, and `src/eval/` — no experimental
logic lives here, only argument parsing, config loading, and calling into `src/`. See
[`../docs/REPRODUCING.md`](../docs/REPRODUCING.md) for the full list with expected
hardware/time and the exact table/figure each one produces, and
[`../docs/CROSSWALK.md`](../docs/CROSSWALK.md) for the mapping back to the paper's own
notebook-cell crosswalk (Appendix J.2).

| Script | Runs in this environment? | Verified |
|---|---|---|
| `run_synthetic_validity.py` | Yes, fully (CPU-only) | 0/1,000 violations, CI [0.0, 0.0037] -- exact match to Table 3; feasibility table exact match to Table 2 |
| `run_necessity_sweep.py` | Yes, fully (CPU-only) | Reproduces the paper's own necessity-table rates closely |
| `run_block_correlation.py` | Yes, fully (CPU-only) | Item-level breaks under correlation, block-level stays at 0.000 -- reproduces Table 13's pattern |
| `run_wsr_audit.py --synthetic` | Yes, fully (CPU-only) | i.i.d. bug rate 0.013, matching the paper's "~0.01" |
| `run_scaling_experiment.py` | Yes, fully (pure arithmetic) | Power-calculation rates within ~1 point of Appendix I's own numbers |
| `make_all_figures.py` | Yes, fully (CPU-only) | Regenerates 3 figures from the scripts above; verified against real generated output |
| `run_tier_a.py` | `--manifest-only` yes; full run needs GPU + HF Hub | Core arm logic (`run_one_arm`) tested against synthetic (η, scores) |
| `run_tier_b.py` | `--smoke-test` yes; full run needs GPU + HF Hub | Leave-one-out logic tested against synthetic decode data |
| `run_deployment.py` | `--smoke-test` yes; full run needs `run_tier_a.py`'s real output | Selection rules + cluster bootstrap tested against synthetic curves |
| `run_group_conditional.py` | `--smoke-test` yes; full run needs GPU + HF Hub | Worst-subject comparison tested against a synthetic multi-subject population |
| `run_exchangeability.py` | `--smoke-test` yes; full run needs CIFAR-10/-C + trained ResNet-18 | Reproduces the paper's own severity-0-to-1 validity cliff almost exactly |

Every script accepts `--config path/to/config.yaml` (defaults live in `configs/`, one file
per script) and writes its output into `results/` using the checkpoint-naming convention in
`docs/CROSSWALK.md`.
