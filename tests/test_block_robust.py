"""Tests for src/bands/block_robust.py -- Theorem 4."""

from __future__ import annotations

import numpy as np
import pytest

from src.bands import block_robust
from src.bands.validity import is_violated
from src.data.copula import correlated_losses


def test_block_variance_bound_bhatia_davis():
    """Var(Z) <= mu(1-mu) for Z in [0,1]; tight (equality) only at mu in {0, 1} for a
    degenerate Z, and at mu=0.5 gives the maximum possible bound, 0.25."""
    assert block_robust.block_variance_bound(0.5) == pytest.approx(0.25)
    assert block_robust.block_variance_bound(0.0) == 0.0
    assert block_robust.block_variance_bound(1.0) == 0.0
    mu = np.linspace(0, 1, 1000)
    bound = block_robust.block_variance_bound(mu)
    assert np.all(bound <= 0.25 + 1e-12)
    assert np.all(bound >= 0.0)


def test_assign_blocks_remaps_arbitrary_labels():
    labels = np.array(["arc_easy", "mmlu", "arc_easy", "csqa", "mmlu"])
    assignment = block_robust.assign_blocks(labels)
    assert assignment.n_blocks == 3
    # same original label -> same remapped id
    assert assignment.block_of_item[0] == assignment.block_of_item[2]
    assert assignment.block_of_item[1] == assignment.block_of_item[4]


def test_block_level_curves_unweighted_average():
    """Remark 'What block-count k certifies': block_losses[j] must be the *unweighted*
    within-block mean, not an item-weighted quantity -- checked directly against unequal
    block sizes."""
    losses = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # block 0: [1,0,0] -> mean 1/3; block 1: [0,0] -> mean 0
    block_of_item = np.array([0, 0, 0, 1, 1])
    assignment = block_robust.BlockAssignment(block_of_item=block_of_item, n_blocks=2)
    block_scores = np.array([1.0, 0.0])
    block_losses, _, _ = block_robust.block_level_curves(losses, block_scores, assignment)
    assert block_losses[0] == pytest.approx(1 / 3)
    assert block_losses[1] == pytest.approx(0.0)


def test_item_level_breaks_but_block_level_holds_under_induced_correlation():
    """The paper's own core demonstration (Section 5.4 / Table 13): induce within-block
    correlation and confirm item-level certification degrades while block-level does not.
    Smaller-scale than the paper's own experiment (which uses the full 24,000-item LLM pool
    and CIFAR-10's 10 classes) so this runs in CI in a few seconds; the full-scale
    reproduction is scripts/run_block_correlation.py.
    """
    from src.bands import sorted_filtration, uniform_band

    n_blocks, items_per_block, n_reps, delta = 4, 1500, 60, 0.05
    n = n_blocks * items_per_block

    def run(rho, seed_offset):
        item_violations = block_violations = 0
        for rep in range(n_reps):
            rng = np.random.default_rng((4242, rep, seed_offset))
            block_eta_true = rng.uniform(0.1, 0.4, size=n_blocks)
            block_of_item = np.repeat(np.arange(n_blocks), items_per_block)
            eta = block_eta_true[block_of_item]
            scores = rng.normal(size=n) - 2 * eta
            losses = correlated_losses(eta, block_of_item, rho, rng)

            pi = sorted_filtration.build_permutation(scores, rng)
            curves = sorted_filtration.realized_and_empirical_curves(losses, pi, eta=eta)
            U_k_item = uniform_band._band_from_curve(curves.Rhat_k, delta=delta)
            if is_violated(curves.Rbar_k, U_k_item, k_min=100).violated:
                item_violations += 1

            assignment = block_robust.assign_blocks(block_of_item)
            block_scores = np.array([scores[block_of_item == b].mean() for b in range(n_blocks)])
            block_losses, _, block_eta_obs = block_robust.block_level_curves(
                losses, block_scores, assignment, eta=eta
            )
            res = block_robust.compute(
                block_losses, block_scores, delta=delta, k_min=1, rng=rng, block_eta=block_eta_obs
            )
            if is_violated(res.Rbar_k, res.U_k, k_min=1).violated:
                block_violations += 1
        return item_violations / n_reps, block_violations / n_reps

    item_rate_indep, block_rate_indep = run(rho=0.0, seed_offset=1)
    item_rate_corr, block_rate_corr = run(rho=0.7, seed_offset=2)

    assert item_rate_indep <= 0.05 + 0.05, "item-level should be roughly valid with no correlation"
    assert (
        item_rate_corr > item_rate_indep
    ), "item-level certification must degrade once within-block correlation is induced"
    assert block_rate_indep == 0.0
    assert block_rate_corr == 0.0, (
        "block-level certification must stay valid regardless of within-block correlation "
        "-- this is the entire content of Theorem 4"
    )
