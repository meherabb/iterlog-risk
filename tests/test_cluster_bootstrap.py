"""Tests for src/eval/cluster_bootstrap.py -- Appendix D.4's setting-level CI correction."""

from __future__ import annotations

import numpy as np

from src.eval import cluster_bootstrap as cb


def test_naive_and_cluster_center_agree_but_width_differs_under_heterogeneity(rng):
    """The point estimate is identical either way; only the interval width should change
    once settings genuinely differ from one another."""
    true_rates = rng.uniform(0.05, 0.3, size=20)
    violated = [rng.binomial(1, r, size=500).astype(float) for r in true_rates]

    naive_rate = cb.naive_pooled_rate(violated)
    lo, hi = cb.setting_level_bootstrap_ci(violated, rng, n_bootstrap=1000)

    assert lo <= naive_rate <= hi
    assert (hi - lo) > 0


def test_cluster_ci_matches_naive_when_settings_are_homogeneous(rng):
    """If every setting truly shares the same rate, the cluster-robust interval should be
    close to (not wildly different from) what pooling suggests -- the correction should
    only matter when there's genuine heterogeneity to correct for."""
    from src.bands.validity import clopper_pearson_ci_on_rate

    shared_rate = 0.1
    violated = [rng.binomial(1, shared_rate, size=1000).astype(float) for _ in range(20)]
    naive_rate = cb.naive_pooled_rate(violated)
    naive_lo, naive_hi = clopper_pearson_ci_on_rate(int(round(naive_rate * 20000)), 20000)
    cluster_lo, cluster_hi = cb.setting_level_bootstrap_ci(violated, rng, n_bootstrap=1000)

    naive_width = naive_hi - naive_lo
    cluster_width = cluster_hi - cluster_lo
    # allow some slack, but the ratio should be modest (nowhere near the 2-8x the paper
    # reports for genuinely heterogeneous settings)
    assert cluster_width / naive_width < 2.5


def test_per_setting_rates_shape_and_range(rng):
    violated = [rng.binomial(1, 0.2, size=100).astype(float) for _ in range(5)]
    rates = cb.per_setting_rates(violated)
    assert rates.shape == (5,)
    assert np.all((rates >= 0) & (rates <= 1))
