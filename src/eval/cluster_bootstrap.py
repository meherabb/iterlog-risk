r"""Setting-level (cluster-robust) bootstrap confidence intervals (Appendix D.4).

The naive Clopper-Pearson interval on a pooled violation rate treats every replication as
an independent observation -- but within one (model, group, score) *setting*, all
replications share the same :math:`\eta` vector, the same score ordering, and the same
underlying dataset; only the loss realization is redrawn. The unit of independent
replication is really the *setting*, not the individual replication. This module resamples
settings, with replacement, rather than replications, giving intervals that correctly widen
to reflect genuine across-setting heterogeneity (the paper reports this understates
uncertainty by 2-8x depending on the certificate).
"""

from __future__ import annotations

import numpy as np


def setting_level_bootstrap_ci(
    violated_by_setting: list[np.ndarray],
    rng: np.random.Generator,
    n_bootstrap: int = 10_000,
    conf: float = 0.95,
) -> tuple[float, float]:
    r"""Cluster-robust CI on a pooled violation rate.

    Parameters
    ----------
    violated_by_setting:
        A list with one boolean/0-1 array per setting, each array holding that setting's
        own replication-level violation indicators (arrays may differ in length across
        settings).
    rng:
        Seeded generator.
    n_bootstrap:
        Number of bootstrap resamples of *settings* (with replacement) -- the paper uses
        10,000.
    conf:
        Confidence level.

    Returns
    -------
    (lower, upper)
        The percentile bootstrap interval on the pooled violation rate.
    """
    n_settings = len(violated_by_setting)
    boot_rates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.integers(0, n_settings, size=n_settings)
        pooled = np.concatenate([violated_by_setting[i] for i in chosen])
        boot_rates[b] = pooled.mean()

    alpha = 1.0 - conf
    lower = float(np.quantile(boot_rates, alpha / 2.0))
    upper = float(np.quantile(boot_rates, 1.0 - alpha / 2.0))
    return lower, upper


def naive_pooled_rate(violated_by_setting: list[np.ndarray]) -> float:
    """The naive pooled rate, for direct comparison against the cluster-robust interval's
    center -- the point estimate is the same either way; only the *interval width* differs.
    """
    pooled = np.concatenate(violated_by_setting)
    return float(pooled.mean())


def per_setting_rates(violated_by_setting: list[np.ndarray]) -> np.ndarray:
    """Each setting's own violation rate -- what the paper's Appendix D.4 uses to
    demonstrate genuine heterogeneity (e.g. "CP's own per-setting rate ranges from 0.066 to
    0.320 across the 24 settings"), as distinct from sampling noise around one shared rate.
    """
    return np.array([np.mean(v) for v in violated_by_setting])
