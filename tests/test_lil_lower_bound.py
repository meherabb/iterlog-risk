"""Tests for src/bands/lil_lower_bound.py -- Theorem 2."""

from __future__ import annotations

import numpy as np
import pytest

from src.bands import lil_lower_bound as lil


def test_lil_envelope_matches_hand_computation():
    k = np.array([100.0])
    p = 0.15
    expected = np.sqrt(2 * p * (1 - p) * np.log(np.log(100.0)) / 100.0)
    assert lil.lil_envelope(k, p)[0] == pytest.approx(expected)


def test_lil_envelope_nan_below_e():
    """The theorem is asymptotic; ln ln k <= 0 for k <= e, where the rate is undefined."""
    result = lil.lil_envelope(np.array([1.0, 2.0]), p=0.1)
    assert np.all(np.isnan(result))


def test_shrunk_band_reduces_to_original_at_c_equals_one():
    Rhat_k = np.array([0.1, 0.2, 0.3])
    U_k = np.array([0.15, 0.28, 0.4])
    assert np.allclose(lil.shrunk_band(Rhat_k, U_k, c=1.0), U_k)


def test_shrunk_band_reduces_to_empirical_curve_at_c_equals_zero():
    Rhat_k = np.array([0.1, 0.2, 0.3])
    U_k = np.array([0.15, 0.28, 0.4])
    assert np.allclose(lil.shrunk_band(Rhat_k, U_k, c=0.0), Rhat_k)


def test_necessity_sweep_violation_rate_increases_as_c_decreases():
    """The core qualitative claim of Theorem 2, made empirical (Section 5.2): shrinking the
    band by a constant factor c < 1 should break it increasingly often. Uses a smaller n
    and fewer reps than the paper's own n=20,000/1,000-rep table for CI speed; the full-scale
    reproduction is scripts/run_necessity_sweep.py.

    Reference point: at the paper's own full scale, c=1.0 gives 0 violations while
    c=0.25 gives roughly 59% -- we don't assert exact rates here (this repository's RNG
    stream necessarily differs from the original notebook's), only that the *direction* and
    *monotonicity* the theorem predicts holds.
    """
    results = lil.necessity_sweep(
        n=5_000, p=0.15, delta=0.05, k_min=100, c_values=(1.0, 0.5, 0.25), n_reps=150
    )
    rates = [r.violation_rate for r in results]
    assert rates[0] == 0.0, "c=1.0 (the actual band) should never be violated"
    assert rates[0] <= rates[1] <= rates[2], "violation rate must not decrease as c shrinks"
    assert rates[2] > rates[0], "c=0.25 must break strictly more often than c=1.0"
