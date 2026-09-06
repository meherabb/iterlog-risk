# Crosswalk: paper items &rarr; repository scripts

This is the repository-side mirror of the paper's own Appendix~J.2 (Reproducibility
crosswalk). The paper traces every figure and table to a notebook cell and checkpoint file
from the original research notebook; this table traces the same 26 items to the
**public, re-runnable script** in this repository that reproduces them, so a reader can move
between "the paper says X came from notebook cell Y" and "here is the equivalent, readable,
version-controlled code."

| Paper item | Paper's notebook cell (Appendix J.2) | This repo's script | Output written to |
|---|---|---|---|
| Table 1 | `s7-stats2 / s7-verdicts2` | `scripts/run_deployment.py` | `results/e4_*.npz` |
| Figure 1 | `fig8-necessity` | `scripts/run_necessity_sweep.py` | `results/v3_necessity.npz` |
| Figure 2 | `fig-block-corr` | `scripts/run_block_correlation.py` | `results/blockcorr_*.npz` |
| Table 2 | `s6-feas` | `scripts/run_synthetic_validity.py --feasibility` | `results/feas_table.npz` |
| Figure 3 | `fig-feas-phase` | `scripts/run_synthetic_validity.py --feasibility --plot` | `results/feas_phase.npz` |
| Table 3 | `s5-synth` | `scripts/run_synthetic_validity.py` | `results/v1_synthetic.npz` |
| CIFAR Tier-A row (Appendix D) | `s8-e6` | `scripts/run_tier_a.py --include-cifar` | `results/e6_clean_tierA.npz` |
| Table 9 | `s7-verdicts2` | `scripts/run_deployment.py --cluster-bootstrap` | `results/e4_*.npz`, `results/e4t2u_*.npz` |
| Figure 4 | `s7-verdicts2 (ext.)` | `scripts/run_deployment.py --cluster-bootstrap --plot` | `results/e4_*.npz`, `results/e4t2u_*.npz` |
| Table 10 | `s7-verdicts2 (ext.)` | `scripts/run_tier_a.py --full-breakdown` | `results/e2_*.npz` |
| Table 11 | `s7-wsr-audit / s7-verdicts2` | `scripts/run_wsr_audit.py` | `results/wsr_audit_*.npz` |
| Table 12 | `s7-cluster-boot` | `scripts/run_deployment.py --cluster-bootstrap` | `results/e4_*.npz` |
| Table 13 | `s9-blockcorr-full` | `scripts/run_block_correlation.py --full-table` | `results/blockcorr_*.npz` |
| Table 14 | `s9-necscale-full` | `scripts/run_necessity_sweep.py --full-table` | `results/necscale_*.npz` |
| Table 15 | `s0-datasets` | `scripts/run_tier_a.py --manifest-only` | `results/dataset_manifest.json` |
| Table 16 | `s0-models` | `scripts/run_tier_a.py --manifest-only` | `results/model_manifest.json` |
| Table 17 | `s11-group-mmlu` | `scripts/run_group_conditional.py` | `results/mmlu_group_*.npz` |
| Figure 9 | `fig-realdata-bands` | `scripts/run_tier_a.py --plot` | `results/e2_*.npz` |
| Figure 5 | `5348d9b0-...d828` | `scripts/make_all_figures.py --figure band` | `results/curves_store.npz` |
| Figure 8 | `fig-price-unif` | `scripts/run_scaling_experiment.py --plot` | `results/bandwidth_nsweep.npz` |
| Figure 6 | `fig-robust-stress` | `scripts/run_exchangeability.py --plot` | `results/e6_*.npz`, `results/v1_margins.npz` |
| Figure 7 | `7ec39717-...fc2` | `scripts/run_deployment.py --plot` | `results/e4_*.npz`, `results/e4ext_*.npz` |
| Figure 10 | `fig-fragility-spectrum` | `scripts/run_deployment.py --plot` | `results/e4_*.npz` |
| Figure 11 | `s7-transfer-iso` | `scripts/run_deployment.py --isolate-transfer` | `results/e4_transfer_*.npz` |
| Figure 12 | `fig-utility-pareto` | `scripts/run_deployment.py --plot` | `results/e4_*.npz` |
| Table 18 | `s10-pack2` | *(this repository's `environment.yml` + `configs/*.yaml`)* | `results/run_manifest.json` |

## Reading this table

- The **paper's notebook cell** column is exactly what appears in the paper's Appendix~J.2 —
  it refers to a cell in the original research notebook the experiments were run in, not to
  anything in this repository.
- The **script** column is what actually ships here: readable, tested, version-controlled
  Python, organized by experimental arm rather than by notebook cell order.
- Every script writes into `results/`, which is gitignored (see `results/README.md`) — this
  repository ships code, not bundled outputs.
- Table 19 is the paper's own crosswalk table (the one this document mirrors) and has no
  corresponding script — it's an appendix artifact of the paper itself.

If you find a place where this table and the paper's Appendix~J.2 disagree, that's a bug in
this repository's documentation, not in the paper — please open an issue citing both.
