# Reproducing the paper, table by table

This guide follows the paper's own organization (Appendix~D onward uses named experimental
arms — V1/E1 through E6, plus several post-hoc-audit arms added after the original
submission; we kept that exact naming as the organizing principle here). Every arm below
lists: the script that produces it, the config it reads, its expected wall-clock time on the
paper's reference hardware (a single NVIDIA Tesla T4, 16GB VRAM), and which table(s)/figure(s)
in the paper it backs.

Before running anything, see [`../README.md#installation`](../README.md#installation) and
make sure `python -m src.data.fetch_all --dry-run` reports all four MCQ datasets (ARC-Easy,
ARC-Challenge, OpenBookQA, CommonsenseQA), MMLU, TriviaQA, and CIFAR-10 as reachable — these
are pulled from the public `datasets` library / `torchvision` and are not vendored here.

## Master seed and determinism

Every arm derives its per-replication seed deterministically from a single master seed via a
counter-based generator, so any individual replication — identified by `(arm_id, rep_index)`
— is reproducible in isolation without re-running the whole arm:

```bash
export ITERLOG_MASTER_SEED=20260727
```

`cudnn.deterministic=True` is set repo-wide (`src/utils/determinism.py`); expect a modest
speed penalty relative to a non-deterministic run.

## Tier A — exact conditional risk (Section 5.1, Appendix D.2)

```bash
python scripts/run_tier_a.py --config configs/tier_a.yaml
```
- **Produces:** Table 4 (summary), Table 10 (all 32 LLM arms individually), Figure 5
  (real-data bands), the score-family contrast numbers in Appendix G.
- **What it does:** scores Pythia-410M, Qwen2.5-1.5B-Instruct, and Phi-3.5-mini-instruct on
  the 24,000-item pool (ARC-Easy + ARC-Challenge + OpenBookQA + CommonsenseQA) and on MMLU,
  across the five score families that run in Tier A (max softmax probability, predictive
  entropy, sequence log-probability, hidden-state centroid distance, hidden-state $k$-NN),
  plus verbalized confidence on the two instruct models. The *loss* for each item is a
  simulated Bernoulli draw from the model's own predictive distribution — see
  `src/eval/tier_protocols.py::tier_a_loss` — so the conditional risk is exact and any
  violation falsifies Theorem 1 outright.
- **Hardware/time:** dominates the paper's GPU budget; plan for several hours on a single
  T4-class GPU for the full 32-arm sweep. Runs happily on Kaggle's or Colab's free-tier T4.
- **Also runs:** the CIFAR-10 matched-control arm (`--include-cifar`), which reuses the same
  planted-loss construction on ResNet-18 (`src/eval/vision_tier_a.py`).

## Tier B — genuine decoding noise (Section 5.1, Table 5)

```bash
python scripts/run_tier_b.py --config configs/tier_b.yaml
```
- **Produces:** Table 5.
- **What it does:** decodes 32 genuine samples per question at $T{=}1$ on TriviaQA for the two
  instruct models, and checks each real decode against the leave-one-out $\hat\eta$ curve
  built from the other 31 (`src/eval/tier_protocols.py::tier_b_loo`).

## Deployment protocol — E4 (Section 5.3, 5.6; Tables 1, 6, 12; Figures 7, 11, 12)

```bash
python scripts/run_deployment.py --config configs/deployment.yaml
```
- **Produces:** Table 1 (headline overrun rates), Table 6 (utility/feasibility), Table 12
  (margin/slack quantiles), Figures 7 (per-setting heatmap), 11 (fragility spectrum), 12
  (utility Pareto front).
- **What it does:** implements all four selection rules ($r_1$ = argmin, $r_2'$ = closest to
  target level, $r_3$ = pre-registered fixed $k_0$, $r_{\mathrm{adv}}$ = adversarial) in
  `src/eval/selection_rules.py`, and the transfer-corrected Clopper–Pearson/Hoeffding
  variants in `src/baselines/transfer_corrected.py`.
- **Also runs:** the cluster-robust (setting-level bootstrap) re-analysis of the same output
  (`--cluster-bootstrap`, Table 12/Appendix D.4) and the population-transfer isolation
  (`--isolate-transfer`, Figure 11).

## WSR / betting-bound audit (Appendix D.3, Table 11)

```bash
python scripts/run_wsr_audit.py --config configs/wsr_audit.yaml
```
- **Produces:** Table 11 (before/after the one-line patch).
- **What it does:** runs the calibration-side audit at a single pre-registered $k_0$ with no
  selection and no population transfer, reproduces the silent-margin bisection defect, and
  applies the max-with-Clopper–Pearson patch (`src/baselines/wsr_betting.py`, see the
  `Lemma 1` docstring on `patched_upper_bound`).

## Necessity sweep — the LIL lower bound is not slack (Section 5.2, Table 14, Figure 1)

```bash
python scripts/run_necessity_sweep.py --config configs/necessity.yaml
```
- **CPU-only.** Sweeps sub-LIL width fractions $c \in \{1, 0.75, 0.5, 0.25\}$ across four
  calibration sizes and confirms the violation rate climbs monotonically with $n$ at fixed
  $c < 1$, tracking $\ln\ln n$ (Theorem 2).

## Block-correlation sweep (Section 5.4, Table 13, Figure 2)

```bash
python scripts/run_block_correlation.py --config configs/block_correlation.yaml
```
- **CPU-only.** Induces within-block correlation via a Gaussian copula
  (`src/data/copula.py`) on two independent block structures (the LLM pool's four source
  datasets; CIFAR-10's ten classes) and compares item-level vs. block-level certification
  (Theorem 4).

## Group-conditional validity (Appendix F, Table 17)

```bash
python scripts/run_group_conditional.py --config configs/group_conditional.yaml
```
Marginal vs. Bonferroni-corrected bands across all 57 MMLU subjects.

## Exchangeability stress test (Section 5.7, Table 7, Figure 6)

```bash
python scripts/run_exchangeability.py --config configs/exchangeability.yaml
```
Calibrates on clean CIFAR-10, deploys across all five CIFAR-10-C severities.

## Synthetic validity and ablations — V1/E1 (Appendix D, Table 3)

```bash
python scripts/run_synthetic_validity.py --config configs/synthetic.yaml
```
**CPU-only, under a minute.** The planted-risk world used for the core validity check, plus
the score-quantization, geometric-base, and $\delta$-allocation ablations of the band's own
construction.

## Full scaling experiment (Appendix I)

```bash
python scripts/run_scaling_experiment.py --config configs/scaling.yaml
```
The $n_{\mathrm{cal}}=20{,}000$ arm on Phi-3.5-mini, reported as confounded by an undersized
holdout — this script also emits the power calculation in Appendix I, not just the raw rates.

## Regenerating every figure from existing result files

If you've already run the arms above and just want to re-render figures without re-running
experiments:

```bash
python scripts/make_all_figures.py --results-dir results/
```

## Self-checks

Three self-checks run automatically at the start of every script above and abort the run on
failure (`src/utils/self_checks.py`): an RNG-stream reproducibility check, a forward-pass
determinism check on the loaded model, and a full-pipeline determinism check that runs the
same seed twice and compares arrays element-wise. You can also run them standalone:

```bash
python -m src.utils.self_checks --all
```
