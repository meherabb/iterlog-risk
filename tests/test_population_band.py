"""Tests for src/bands/population_band.py -- Theorem 3, Corollaries 2-3, DKW embedding."""

from __future__ import annotations

import numpy as np
import pytest

from src.bands import population_band as pb


def test_epsilon_n_matches_paper_remark():
    """Remark 'Implemented constant': at n=2000, delta=0.05, the released-code constant is
    0.0304 against the theorem-correct 0.0329."""
    assert pb.epsilon_n_as_implemented(2000, 0.05) == pytest.approx(0.0304, abs=1e-4)
    assert pb.epsilon_n(2000, 0.05) == pytest.approx(0.0329, abs=2e-3)
    assert pb.epsilon_n(2000, 0.05) > pb.epsilon_n_as_implemented(2000, 0.05), (
        "the theorem-correct constant must be larger (more conservative) than the "
        "released-code one, per the paper's own remark"
    )


@pytest.mark.parametrize(
    "n,expected",
    [(500, 0.033), (2000, 0.067), (20000, 0.089), (50000, 0.093)],
)
def test_feasibility_threshold_matches_paper_table(n, expected):
    """Appendix B's feasibility table (Corollary 2), using the paper's own implemented
    constant, alpha=0.10, delta=0.05, gamma=1 (the best-case, coverage -> 1 threshold)."""
    threshold = pb.feasibility_threshold(alpha=0.10, n=n, delta=0.05, gamma=1.0, use_paper_constant=True)
    assert threshold == pytest.approx(expected, abs=1e-3)


def test_feasibility_tightens_with_smaller_gamma():
    """Corollary 2: target coverage gamma < 1 is strictly tighter by a factor 1/gamma on
    the epsilon term, so a lower target coverage should require a *lower* Rhat_k threshold
    to be feasible."""
    thr_full_coverage = pb.feasibility_threshold(0.10, n=2000, delta=0.05, gamma=1.0)
    thr_half_coverage = pb.feasibility_threshold(0.10, n=2000, delta=0.05, gamma=0.5)
    assert thr_half_coverage < thr_full_coverage


def test_dkw_embedding_exact_equality():
    """The defining property the embedding buys (paper's proof of Theorem 3): for every
    threshold t, {i : W_i >= phi(t)} equals exactly {i : s(X_i) >= t, ell_i = 1}, with no
    restriction on t needed -- checked directly here rather than merely asserted."""
    rng = np.random.default_rng(7)
    n = 3000
    scores = rng.normal(size=n) * 5
    losses = rng.binomial(1, 0.3, size=n).astype(float)
    W = pb._dkw_embedding(scores, losses)

    for t in np.linspace(scores.min() - 1, scores.max() + 1, 25):
        lhs = set(np.where(W >= pb.bijection_phi(t))[0])
        rhs = set(np.where((scores >= t) & (losses == 1))[0])
        assert lhs == rhs, f"embedding equality failed at t={t}"


def test_dkw_embedding_iid_by_construction():
    """Each W_i is a fixed function of (X_i, ell_i) alone (Remark 'Two earlier attempts'):
    unlike a sample-dependent construction (e.g. one keyed on max|s(X_i)| over the whole
    sample), permuting the input order must not change any individual W_i's value."""
    rng = np.random.default_rng(3)
    n = 500
    scores = rng.normal(size=n)
    losses = rng.binomial(1, 0.4, size=n).astype(float)
    W = pb._dkw_embedding(scores, losses)

    perm = rng.permutation(n)
    W_permuted_input = pb._dkw_embedding(scores[perm], losses[perm])
    assert np.allclose(W[perm], W_permuted_input)


def test_population_band_dominates_and_is_nan_when_undefined():
    rng = np.random.default_rng(1)
    n = 2000
    losses = rng.binomial(1, 0.15, size=n).astype(float)
    Rhat_k = np.cumsum(losses) / np.arange(1, n + 1)
    result = pb.compute(Rhat_k, n=n, delta=0.05)

    defined_mask = result.defined
    assert np.all(result.U_pop[defined_mask] >= Rhat_k[defined_mask] - 1e-9)
    assert np.all(np.isnan(result.U_pop[~defined_mask]))
    # phi_hat <= eps only possible at very small k -- confirm at least the very first entry
    assert not result.defined[0] or result.phi_hat[0] > result.eps
