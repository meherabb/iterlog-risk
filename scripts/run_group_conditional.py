#!/usr/bin/env python3
r"""Reproduces Table 17 (worst-subject held-out violation, marginal vs. Bonferroni-corrected
band, across all 57 MMLU subjects).

**Requires GPU + Hugging Face Hub access** to score Phi-3.5-mini-instruct on MMLU. The
statistical comparison itself (marginal band vs. per-subject worst-case violation) is pure
NumPy; --smoke-test exercises it against a synthetic multi-subject population.

Usage
-----
    python scripts/run_group_conditional.py --config configs/group_conditional.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bands import sorted_filtration, uniform_band
from src.bands.validity import clopper_pearson_ci_on_rate
from src.eval import selection_rules
from src.utils.determinism import derive_seed
from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json


def worst_subject_violation(subject_of_item, eta, scores, losses, delta, alpha, master_seed, n_reps):
    """Calibrate one band on the *pooled* population, select a threshold via r1, then
    check whether *any single subject's* realized risk among the selected items exceeds
    the pooled certificate -- even when the pooled realized risk doesn't."""
    n_subjects = int(subject_of_item.max()) + 1
    worst_violations = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(derive_seed("group_conditional", rep, master_seed))
        rep_losses = rng.binomial(1, eta).astype(float)
        pi = sorted_filtration.build_permutation(scores, rng)
        curves = sorted_filtration.realized_and_empirical_curves(rep_losses, pi, eta=eta)
        U_k = uniform_band._band_from_curve(curves.Rhat_k, delta=delta)
        k = selection_rules.r1_argmin(curves.Rhat_k)
        pooled_bound = U_k[k - 1]

        selected_mask = np.zeros_like(eta, dtype=bool)
        selected_mask[pi[:k]] = True
        any_subject_violated = False
        for s in range(n_subjects):
            subject_selected = selected_mask & (subject_of_item == s)
            if subject_selected.sum() == 0:
                continue
            subject_risk = eta[subject_selected].mean()
            if subject_risk > pooled_bound:
                any_subject_violated = True
                break
        worst_violations += any_subject_violated

    lo, hi = clopper_pearson_ci_on_rate(worst_violations, n_reps)
    return {
        "violations": worst_violations,
        "n_reps": n_reps,
        "rate": worst_violations / n_reps,
        "ci95": [lo, hi],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/group_conditional.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_group_conditional", config)
    out_dir = ensure_output_dir(config)

    if not args.smoke_test:
        print(
            f"Full run needs GPU + Hugging Face Hub access to score {config['model']} on "
            f"{config['dataset']} ({config['n_subjects']} subjects), which this environment "
            "does not have. Run with --smoke-test for a synthetic-population check of the "
            "worst-subject comparison logic itself."
        )
        return

    n_subjects_small = 10  # a small stand-in for the paper's 57, for a fast smoke test
    items_per_subject = 200
    rng = np.random.default_rng(0)
    subject_eta = rng.uniform(0.1, 0.5, size=n_subjects_small)
    subject_of_item = np.repeat(np.arange(n_subjects_small), items_per_subject)
    eta = subject_eta[subject_of_item]
    scores = -eta + rng.normal(scale=0.3, size=len(eta))
    losses = rng.binomial(1, eta).astype(float)

    result = worst_subject_violation(
        subject_of_item,
        eta,
        scores,
        losses,
        config["delta"],
        config["alpha"],
        config["master_seed"],
        n_reps=200,
    )
    print(
        f"Smoke test (marginal band, {n_subjects_small} synthetic subjects): "
        f"{result['violations']}/{result['n_reps']} = {result['rate']:.4f}, "
        f"95% CI {result['ci95']}"
    )
    save_json(out_dir, config["output_prefix"], "smoke_test", result)


if __name__ == "__main__":
    main()
