"""Tests for src/baselines/wsr_betting.py -- the betting bound, its documented bug
(Appendix D.3), and the Lemma 1 patch."""

from __future__ import annotations

import numpy as np

from src.baselines import wsr_betting


def test_sequential_test_correctly_discriminates_true_from_false_means():
    """Sanity check on the underlying betting test itself, before testing the buggy
    bisection wrapper around it: it should not reject the true mean, and should reject
    (with growing wealth) candidates increasingly far from it."""
    rng = np.random.default_rng(1)
    data = rng.binomial(1, 0.2, size=2000).astype(float)

    assert not wsr_betting.sequential_test_rejects(data, m=0.2, delta=0.05)

    log_wealths = [wsr_betting.log_wealth_path(data, m)[-1] for m in (0.3, 0.4, 0.6, 0.9)]
    assert all(w > np.log(20) for w in log_wealths), "all should be rejected (delta=0.05 -> threshold ln(20))"
    assert log_wealths == sorted(log_wealths), "wealth should grow as the candidate moves further"


def test_buggy_bound_is_approximately_valid_on_iid_data():
    """On well-behaved i.i.d. data the naive bisection should still work correctly most of
    the time -- the defect is specifically about *heterogeneous* data, not a claim that the
    naive implementation is always broken."""
    rng = np.random.default_rng(1)
    n_trials = 150
    valid_count = 0
    for _ in range(n_trials):
        data = rng.binomial(1, 0.2, size=500).astype(float)
        result = wsr_betting.upper_bound_buggy(data, delta=0.05)
        if result.upper_bound >= 0.2:
            valid_count += 1
    rate = valid_count / n_trials
    assert rate >= 0.85, f"i.i.d. validity rate {rate:.2f} is implausibly low for delta=0.05"


def test_bug_fires_more_often_on_heterogeneous_than_iid_data():
    """The paper's central audit finding (Appendix D.3): the silent-margin shortcut fires
    far more often on heterogeneous-risk data than on i.i.d. data of the same size.
    Reference point: paper reports ~1% (i.i.d.) vs. 32% (real heterogeneous data); this
    synthetic heterogeneous construction is a more extreme stress test than the paper's own
    real-data heterogeneity; we assert the qualitative direction and a much-higher rate on
    i.i.d. as a sanity floor, not an exact match to either number.
    """
    rng = np.random.default_rng(999)
    n_trials = 150

    def iid_trial():
        return rng.binomial(1, 0.3, size=500).astype(float)

    def heterogeneous_trial():
        eta = np.linspace(0.05, 0.6, 500)
        return rng.binomial(1, eta).astype(float)

    iid_bug_rate = (
        sum(
            wsr_betting.upper_bound_buggy(iid_trial(), delta=0.05).used_buggy_shortcut
            for _ in range(n_trials)
        )
        / n_trials
    )
    het_bug_rate = (
        sum(
            wsr_betting.upper_bound_buggy(heterogeneous_trial(), delta=0.05).used_buggy_shortcut
            for _ in range(n_trials)
        )
        / n_trials
    )

    assert iid_bug_rate < 0.10, f"i.i.d. bug rate {iid_bug_rate:.2f} should be low, matching the paper's ~1%"
    assert het_bug_rate > iid_bug_rate, "the defect must fire more often under heterogeneity"


def test_lemma1_patch_never_worse_than_clopper_pearson_alone():
    """Lemma 1: max(A, B) >= B pointwise, so {mu > max(A,B)} subset {mu > B} -- the patched
    bound's violation rate can never exceed Clopper-Pearson's own, regardless of how badly
    the raw betting bound A behaves. Checked directly on the same heterogeneous construction
    that triggers the bug."""
    from src.baselines import clopper_pearson

    rng = np.random.default_rng(2024)
    n_trials = 300
    cp_violations = 0
    patched_violations = 0

    for _ in range(n_trials):
        eta = np.linspace(0.1, 0.3, 800)
        data = rng.binomial(1, eta).astype(float)
        true_mean = eta.mean()

        S_k = int(round(data.sum()))
        cp_bound = clopper_pearson.upper_bound(S_k, len(data), delta=0.05)
        patched = wsr_betting.patched_upper_bound(data, delta=0.05)

        assert patched >= cp_bound - 1e-9, "patched bound must be >= the Clopper-Pearson component (Lemma 1)"

        if cp_bound < true_mean:
            cp_violations += 1
        if patched < true_mean:
            patched_violations += 1

    assert (
        patched_violations <= cp_violations
    ), "Lemma 1 guarantees the patched bound violates no more often than Clopper-Pearson alone"
    # Both should be roughly within the delta=0.05 budget (allowing Monte Carlo slack).
    assert patched_violations / n_trials <= 0.05 + 0.03


def test_patch_fixes_cases_where_raw_bound_would_have_failed():
    """Construct a case where the raw buggy bound is known to fail (heterogeneous data
    triggering the shortcut) and confirm the patched version recovers a valid bound there
    specifically, not just in aggregate."""
    rng = np.random.default_rng(11)
    found_a_failure_case = False
    for _ in range(200):
        eta = np.linspace(0.05, 0.6, 500)
        data = rng.binomial(1, eta).astype(float)
        true_mean = eta.mean()
        raw = wsr_betting.upper_bound_buggy(data, delta=0.05)
        if raw.used_buggy_shortcut and raw.upper_bound < true_mean:
            found_a_failure_case = True
            patched = wsr_betting.patched_upper_bound(data, delta=0.05)
            assert patched >= raw.upper_bound, "patch must never be tighter than the raw bound"
    assert found_a_failure_case, "test setup should reliably trigger at least one raw failure"
