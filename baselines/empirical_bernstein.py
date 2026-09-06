r"""Maurer-Pontil empirical Bernstein bound, at a single fixed :math:`k`.

.. math::
    U_{k_0} = \hat R_{k_0} + \sqrt{2 \hat V_{k_0} \ln(2/\delta) / k_0}
        + \frac{7 \ln(2/\delta)}{3(k_0 - 1)},

where :math:`\hat V_{k_0}` is the sample variance of the :math:`k_0` losses. Self-normalizing
in the same spirit as Theorem 1's own bound (the variance bound :math:`\eta(1-\eta) \le \eta`
that lets :math:`U_k` be computed from :math:`\hat R_k` alone is the identical trick, just
applied at a single fixed :math:`k` here rather than uniformly).
"""

from __future__ import annotations

import numpy as np


def upper_bound(losses_first_k: np.ndarray, delta: float) -> float:
    """Empirical Bernstein bound from the raw ``0/1`` losses of the first ``k`` items.

    Parameters
    ----------
    losses_first_k:
        Shape ``(k,)``, the losses among the top-``k`` (by score) items. Needs the raw
        per-item losses, not just the summary rate, since it uses the *sample variance*.
    delta:
        Failure probability.
    """
    losses_first_k = np.asarray(losses_first_k, dtype=float)
    k = losses_first_k.shape[0]
    if k < 2:
        raise ValueError("empirical Bernstein needs at least 2 observations")
    Rhat = losses_first_k.mean()
    V_hat = losses_first_k.var(ddof=1)  # sample variance, matching Maurer-Pontil's own convention
    log_term = np.log(2.0 / delta)
    return float(Rhat + np.sqrt(2.0 * V_hat * log_term / k) + 7.0 * log_term / (3.0 * (k - 1)))
