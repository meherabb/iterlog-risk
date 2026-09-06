# `configs/` — one YAML per experimental arm

Each file here corresponds one-to-one with a script in `scripts/` and an arm named in
`docs/REPRODUCING.md`:

| Config | Consumed by | Paper reference |
|---|---|---|
| `synthetic.yaml` | `scripts/run_synthetic_validity.py` | Appendix D, Table 3 |
| `tier_a.yaml` | `scripts/run_tier_a.py` | Section 5.1, Tables 4/10 |
| `tier_b.yaml` | `scripts/run_tier_b.py` | Section 5.1, Table 5 |
| `deployment.yaml` | `scripts/run_deployment.py` | Section 5.3/5.6, Tables 1/6/12 |
| `wsr_audit.yaml` | `scripts/run_wsr_audit.py` | Appendix D.3, Table 11 |
| `necessity.yaml` | `scripts/run_necessity_sweep.py` | Section 5.2, Table 14 |
| `block_correlation.yaml` | `scripts/run_block_correlation.py` | Section 5.4, Table 13 |
| `group_conditional.yaml` | `scripts/run_group_conditional.py` | Appendix F, Table 17 |
| `exchangeability.yaml` | `scripts/run_exchangeability.py` | Section 5.7, Table 7 |
| `scaling.yaml` | `scripts/run_scaling_experiment.py` | Appendix I |

All hyperparameter values in these configs match Appendix J.1 (Complete hyperparameter and
reproducibility reference) exactly, with one correction found and fixed during development:
`scaling.yaml`'s `selected_above_threshold: 2000` (not the full `held_out_size: 4000`) is
the value consistent with the paper's own reported standard error of 0.0107 -- see
`scripts/run_scaling_experiment.py`'s docstring for the arithmetic showing why the full
held-out size cannot reproduce that number.

