r"""Exact binomial (Clopper-Pearson) upper confidence bound, at a single fixed :math:`k`.

Valid only at a threshold/prefix length fixed *before* the calibration data are seen
(Section 2, "The trouble is in how the threshold actually gets picked") -- used throughout
the paper as the tightest of the fixed-:math:`k` baselines, and correspondingly the one that
breaks hardest under post-hoc selection (Table 1: selection alone costs it 12.7 percentage
points).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def upper_bound(S_k: int, k: int, delta: float) -> float:
    r"""The exact one-sided $(1-\delta)$ upper confidence bound on a Binomial proportion.

    Standard Clopper-Pearson identity: the upper bound is the value $p_U$ solving
    $\Pr[\mathrm{Binomial}(k, p_U) \le S_k] = \delta$, given in closed form via the inverse
    incomplete beta function as $p_U = \mathrm{Beta}^{-1}(1-\delta;\ S_k+1,\ k-S_k)$, with
    the boundary case $p_U = 1$ when $S_k = k$ (every item so far has been a loss).

    Parameters
    ----------
    S_k:
        Number of losses observed among the first $k$ items (an integer count, *not* the
        rate -- pass ``round(k * Rhat_k)`` if starting from a rate).
    k:
        Sample size (the fixed prefix length / calibration size).
    delta:
        Failure probability.
    """
    if S_k == k:
        return 1.0
    return float(stats.beta.ppf(1.0 - delta, S_k + 1, k - S_k))


def upper_bound_at_fixed_k(Rhat_k: np.ndarray, k0: int, delta: float) -> float:
    """Convenience: evaluate :func:`upper_bound` at one fixed prefix length ``k0`` directly
    from the empirical curve, rounding ``k0 * Rhat_k[k0-1]`` to the nearest integer loss
    count (the curve is exact in the paper's own simulated-loss experiments, so this
    rounding is a no-op there; it only matters if ``Rhat_k`` came from some other source).
    """
    S_k0 = int(round(k0 * Rhat_k[k0 - 1]))
    return upper_bound(S_k0, k0, delta)
