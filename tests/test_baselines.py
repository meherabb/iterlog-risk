"""Tests for the standard concentration-inequality baselines, checked against known
reference values (either the paper's own reported numbers or textbook closed forms)."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import (
    clopper_pearson,
    conformal_risk_control,
    empirical_bernstein,
    hoeffding,
    learn_then_test,
)


def test_hoeffding_half_width_matches_paper():
    """Section 5.3's arithmetic explanation: at n_cal=2000, Hoeffding's half-width at
    k0=1000 is 0.039."""
    assert hoeffding.half_width(1000, 0.05) == pytest.approx(0.039, abs=1e-3)


def test_clopper_pearson_boundary_case():
    """S_k == k (every item so far a loss) must return exactly 1.0, not attempt the beta
    inverse (which is degenerate there)."""
    assert clopper_pearson.upper_bound(S_k=10, k=10, delta=0.05) == 1.0


def test_clopper_pearson_zero_successes_gives_known_small_bound():
    """At 0 successes out of a large n, the exact CP upper bound should sit close to the
    well-known approximation 1 - delta**(1/n) for small delta/large n."""
    n = 1000
    delta = 0.05
    exact = clopper_pearson.upper_bound(S_k=0, k=n, delta=delta)
    approx = 1 - delta ** (1 / n)
    assert exact == pytest.approx(approx, rel=0.05)


def test_clopper_pearson_upper_bound_always_at_least_the_rate():
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = rng.integers(10, 1000)
        S_k = rng.integers(0, k + 1)
        bound = clopper_pearson.upper_bound(int(S_k), int(k), delta=0.05)
        assert bound >= S_k / k - 1e-9


def test_empirical_bernstein_tighter_than_hoeffding_at_low_variance():
    """At a base rate far from 0.5 (low empirical variance), empirical Bernstein should
    generally out-perform Hoeffding, which ignores the variance entirely -- checked as an
    average-case property over many draws, not a per-draw guarantee."""
    rng = np.random.default_rng(3)
    k0, delta = 2000, 0.05
    eb_widths, hoeff_width = [], hoeffding.half_width(k0, delta)
    for _ in range(30):
        losses = rng.binomial(1, 0.05, size=k0).astype(float)  # low base rate -> low variance
        eb_bound = empirical_bernstein.upper_bound(losses, delta)
        eb_widths.append(eb_bound - losses.mean())
    assert np.mean(eb_widths) < hoeff_width


def test_learn_then_test_grid_only_certifies_grid_points():
    rng = np.random.default_rng(1)
    n = 5000
    Rhat_k = np.cumsum(rng.binomial(1, 0.2, size=n)) / np.arange(1, n + 1)
    result = learn_then_test.certify_grid(Rhat_k, grid_size=10, delta=0.05)
    assert len(result.grid_k) <= 10
    assert np.all(result.upper_bounds >= Rhat_k[result.grid_k - 1] - 1e-9)


def test_learn_then_test_fixed_sequence_selects_a_certified_point():
    rng = np.random.default_rng(1)
    n = 5000
    Rhat_k = np.cumsum(rng.binomial(1, 0.1, size=n)) / np.arange(1, n + 1)
    k = learn_then_test.certify_fixed_sequence(Rhat_k, delta=0.05, alpha=0.2)
    assert k is not None
    bound = clopper_pearson.upper_bound_at_fixed_k(Rhat_k, k, delta=0.05)
    assert bound <= 0.2 + 1e-9


def test_learn_then_test_fixed_sequence_returns_none_when_infeasible():
    """If even the full calibration set (k=n) can't be certified at alpha, no smaller
    prefix can either (fewer samples -> a looser, not tighter, Clopper-Pearson bound at the
    same empirical rate), so the walk should find nothing and return None."""
    Rhat_k = np.full(1000, 0.9)
    k = learn_then_test.certify_fixed_sequence(Rhat_k, delta=0.05, alpha=0.01)
    assert k is None


def test_conformal_risk_control_returns_zero_when_infeasible():
    """If even Rhat_k at k=1 already exceeds alpha with the RCPS-style +1 correction, no k
    should be selectable."""
    Rhat_k = np.full(100, 0.9)
    k = conformal_risk_control.select_threshold(Rhat_k, alpha=0.1)
    assert k == 0


def test_transfer_corrected_cp_is_at_least_as_wide_as_plain_cp():
    """Adding a population-transfer allowance on top of the calibration-side bound should
    never make the certificate *tighter* than plain Clopper-Pearson at the same point --
    the whole purpose of Section 5.3's isolation is that transfer costs something extra."""
    from src.baselines import transfer_corrected

    rng = np.random.default_rng(1)
    n, k0, delta = 2000, 1000, 0.05
    Rhat_k = np.cumsum(rng.binomial(1, 0.2, size=n)) / np.arange(1, n + 1)

    plain_cp = clopper_pearson.upper_bound_at_fixed_k(Rhat_k, k0, delta)
    corrected = transfer_corrected.clopper_pearson_transfer_corrected(Rhat_k, k0, n, delta)
    assert corrected >= plain_cp - 1e-9


def test_transfer_corrected_hoeffding_is_at_least_as_wide_as_plain_hoeffding():
    from src.baselines import transfer_corrected

    rng = np.random.default_rng(1)
    n, k0, delta = 2000, 1000, 0.05
    Rhat_k = np.cumsum(rng.binomial(1, 0.2, size=n)) / np.arange(1, n + 1)

    plain_hoeff = hoeffding.upper_bound(Rhat_k[k0 - 1], k0, delta)
    corrected = transfer_corrected.hoeffding_transfer_corrected(Rhat_k, k0, n, delta)
    assert corrected >= plain_hoeff - 1e-9


def test_transfer_corrected_narrows_as_n_grows():
    """The transfer allowance shrinks as the calibration set grows (it's an
    O(n^{-1/2}) term), so at a *fixed coverage level* the corrected bound should tighten
    toward the uncorrected one as n increases. (Coverage, not k0 itself, must be held fixed
    here: fixing k0 as n grows would let phi_hat = k0/n collapse toward the epsilon_n
    denominator, confounding the comparison with a coverage effect instead of isolating n.)
    """
    from src.baselines import transfer_corrected

    delta = 0.05
    coverage = 0.5
    gaps = []
    for n in (2000, 20000, 200000):
        k0 = int(n * coverage)
        Rhat_k = np.full(n, 0.2)
        plain = clopper_pearson.upper_bound_at_fixed_k(Rhat_k, k0, delta)
        corrected = transfer_corrected.clopper_pearson_transfer_corrected(Rhat_k, k0, n, delta)
        gaps.append(corrected - plain)
    assert gaps[0] > gaps[1] > gaps[2], "the transfer gap should shrink as n grows at fixed coverage"
