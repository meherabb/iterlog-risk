"""Tests for src/bands/uniform_band.py -- Theorem 1 and Corollary 1."""

from __future__ import annotations

import numpy as np
import pytest

from src.bands import uniform_band
from src.bands.validity import clopper_pearson_ci_on_rate, is_violated


def test_a_of_lambda_hits_three_quarters_exactly_at_one():
    """a(lambda) <= 3/4 throughout (0, 1], with equality exactly at lambda=1 -- the paper is
    careful this is non-strict, since the menu's own boundary case (j in {0,1,2,3} at
    delta=0.05) genuinely reaches it."""
    assert uniform_band.a_of_lambda(1.0) == pytest.approx(0.75, abs=1e-12)
    lam = np.linspace(1e-6, 1.0, 10_000)
    assert np.all(uniform_band.a_of_lambda(lam) <= 0.75 + 1e-12)


def test_lambda_menu_delta_j_sums_converge_to_delta():
    """The infinite sum over j of delta_j equals delta exactly (Basel-sum identity); any
    finite truncation necessarily falls slightly short (never over), which only makes the
    resulting band more conservative, never invalid. Checked here as a convergence
    property, not a fixed-j_max exact-equality (which would be mathematically wrong to
    expect)."""
    delta = 0.05
    sums = [uniform_band.lambda_menu(delta, j_max=j)["delta_j"].sum() for j in (60, 600, 6000)]
    assert all(s <= delta for s in sums), "truncation must never exceed the delta budget"
    assert sums[0] < sums[1] < sums[2] < delta
    assert sums[2] == pytest.approx(delta, abs=1e-4)


def test_lambda_menu_no_overflow_at_large_j_max():
    """Regression test for a real overflow bug found during development: b**j must not be
    computed directly (it overflows float64 for j_max in the thousands at b=2), since the
    resulting inf/nan silently corrupted lambda_j. Computing in log-space fixes this.

    Underflow (lambda_j collapsing to exactly 0.0 for enormous j, where it's supposed to
    vanish) is the mathematically correct limit, not a bug, so only overflow/invalid ops
    are treated as failures here."""
    with np.errstate(over="raise", invalid="raise"):
        menu = uniform_band.lambda_menu(delta=0.05, j_max=50_000)
    assert np.all(np.isfinite(menu["lambda_j"]))
    assert np.all(np.isfinite(menu["a_lambda_j"]))


def test_explicit_form_matches_paper_worked_example():
    """At n=20,000, k=2,000, Rhat_k=0.10, delta=0.05: the paper reports a width of 0.118 for
    the explicit (Corollary 1) form and 0.034 for the exact min-form (Theorem 1) -- both
    reproduced here to within rounding."""
    n = 20_000
    Rhat_full = np.zeros(n)
    Rhat_full[1999] = 0.10  # k=2000 -> index 1999

    exact_width = uniform_band._band_from_curve(Rhat_full, delta=0.05)[1999] - 0.10
    explicit_width = uniform_band.explicit_form(Rhat_full, delta=0.05)[1999] - 0.10

    assert exact_width == pytest.approx(0.034, abs=2e-3)
    assert explicit_width == pytest.approx(0.118, abs=2e-3)


def test_explicit_form_is_nan_outside_its_validity_domain():
    """Corollary 1 only applies when k*Rhat_k >= 2*ell(k); small k or Rhat_k near 0 should
    return nan rather than a number presented as if it were a valid bound."""
    n = 100
    Rhat_full = np.full(n, 1e-6)  # essentially zero risk -> far outside the validity domain
    result = uniform_band.explicit_form(Rhat_full, delta=0.05)
    assert np.isnan(result[9])  # k=10, certainly outside the domain at this Rhat_k


def test_band_always_dominates_empirical_curve():
    """U_k >= Rhat_k always, by construction (a(lambda) < 1 and the additive term is
    non-negative) -- this should hold for *any* input, not just realistic risk curves."""
    rng = np.random.default_rng(0)
    Rhat_k = rng.uniform(0, 1, size=5000)
    U_k = uniform_band._band_from_curve(Rhat_k, delta=0.05)
    assert np.all(U_k >= Rhat_k - 1e-12)


@pytest.mark.parametrize("p", [0.05, 0.15, 0.4])
def test_empirical_coverage_on_homogeneous_synthetic_world(p, rng):
    """The core falsification check the paper's own Table 3 runs: homogeneous world, score
    independent of loss, delta-uniform validity should hold empirically. Uses a smaller
    n and fewer replications than the paper's own n=20,000/1,000-rep validity table so this
    test runs in CI in a few seconds; the full-scale version is
    scripts/run_synthetic_validity.py (see docs/REPRODUCING.md)."""
    n, n_reps, delta, k_min = 3_000, 300, 0.05, 100
    n_violations = 0
    for rep in range(n_reps):
        r = np.random.default_rng((20260727, rep, round(p * 1000)))
        losses = r.binomial(1, p, size=n).astype(float)
        scores = r.normal(size=n)
        eta = np.full(n, p)
        result = uniform_band.compute(losses, scores, delta=delta, k_min=k_min, rng=r, eta=eta)
        if is_violated(result.Rbar_k, result.U_k, k_min=k_min).violated:
            n_violations += 1

    lo, hi = clopper_pearson_ci_on_rate(n_violations, n_reps)
    assert hi <= delta + 0.02, (
        f"violation rate {n_violations}/{n_reps} (CI upper {hi:.4f}) is implausibly high "
        f"for a delta={delta} certificate"
    )
