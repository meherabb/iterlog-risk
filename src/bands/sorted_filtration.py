r"""The sorted filtration (paper Section 3, "Setup").

The single idea everything else in this package builds on: the permutation that sorts
items by score depends only on the covariates (and a set of independent tie-breakers), so
conditioning on it leaves the losses conditionally independent. Reading the sorted sequence
prefix by prefix turns the risk-coverage curve into a martingale problem (Eq. "mart" in the
paper), which is what lets anytime-valid concentration apply directly in ``uniform_band.py``.

Notation follows the paper exactly:
    - :math:`s(X_i)` -- an arbitrary score, larger meaning more confident.
    - :math:`\zeta_i \sim \mathrm{Unif}(0,1)` -- i.i.d. tie-breakers, independent of everything
      else.
    - :math:`\pi` -- the permutation sorting the pairs :math:`(s(X_i), \zeta_i)` in decreasing
      lexicographic order.
    - :math:`\bar R_k = \frac{1}{k}\sum_{i=1}^k \eta_{\pi(i)}` -- the *realized* (transductive)
      risk-coverage curve. Requires the true conditional risk :math:`\eta`; only available
      when :math:`\eta` is known or estimated (Tier A / Tier B in the paper's terminology).
    - :math:`\hat R_k = \frac{1}{k}\sum_{i=1}^k \ell_{\pi(i)}` -- the *empirical* risk-coverage
      curve. Always computable from observed losses alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def build_permutation(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sort items by decreasing score, breaking ties with independent uniform draws.

    Implements :math:`\\pi`: the permutation sorting :math:`(s(X_i), \\zeta_i)` pairs in
    decreasing lexicographic order.

    Parameters
    ----------
    scores:
        Shape ``(n,)``. Any real-valued score; larger means more confident. No assumption
        is made about calibration, monotonicity in the true risk, or informativeness --
        the theorem's guarantee does not depend on the score being any good.
    rng:
        A ``numpy.random.Generator`` used only to draw the tie-breakers. Pass a
        deterministically-seeded generator (see ``src/utils/determinism.py``) for
        reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` integer array. ``pi[0]`` is the index (into ``scores``) of the
        highest-scoring item, ``pi[1]`` the second-highest, and so on.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    tie_breakers = rng.uniform(0.0, 1.0, size=n)
    # np.lexsort sorts ascending, keyed on the LAST array first; negate both keys to get
    # decreasing order on (score, tie_breaker) with score as the primary key.
    pi = np.lexsort((-tie_breakers, -scores))
    return pi


@dataclass(frozen=True)
class RiskCoverageCurves:
    """The realized and empirical risk-coverage curves, and their supporting quantities.

    Attributes
    ----------
    Rbar_k:
        The realized (transductive) curve, :math:`\\bar R_k`. ``None`` if ``eta`` was not
        supplied to :func:`realized_and_empirical_curves` (i.e. the true conditional risk is
        unknown, as in a genuine Tier C deployment).
    Rhat_k:
        The empirical curve, :math:`\\hat R_k`. Always present.
    S_k:
        :math:`S_k = k \\hat R_k`, the raw loss count among the top-``k`` items. Kept as its
        own field because the martingale :math:`N_k = k\\bar R_k - S_k` (Eq. "mart") is stated
        directly in terms of it.
    pi:
        The permutation used, as returned by :func:`build_permutation`.
    """

    Rhat_k: np.ndarray
    S_k: np.ndarray
    pi: np.ndarray
    Rbar_k: np.ndarray | None = None


def realized_and_empirical_curves(
    losses: np.ndarray,
    pi: np.ndarray,
    eta: np.ndarray | None = None,
) -> RiskCoverageCurves:
    """Compute :math:`\\hat R_k` (and :math:`\\bar R_k`, if ``eta`` is given) for every prefix.

    Parameters
    ----------
    losses:
        Shape ``(n,)``, values in ``{0, 1}`` (or ``[0, 1]`` for the bounded-loss extension
        the paper notes in Section 3). The *observed* loss for each item, in original
        (pre-sort) order.
    pi:
        The permutation from :func:`build_permutation`.
    eta:
        Shape ``(n,)``, the true conditional risk :math:`\\eta(X_i)` for each item, in
        original order. Only available when it is known exactly (Tier A) or estimated
        (Tier B); omit it for a genuine deployment where it is never observed (Tier C).

    Returns
    -------
    RiskCoverageCurves
        ``Rhat_k[k-1]`` is :math:`\\hat R_k` for prefix length ``k`` (1-indexed in the paper,
        0-indexed here as is standard for arrays).
    """
    losses = np.asarray(losses, dtype=float)
    sorted_losses = losses[pi]
    k = np.arange(1, len(losses) + 1)
    S_k = np.cumsum(sorted_losses)
    Rhat_k = S_k / k

    Rbar_k = None
    if eta is not None:
        eta = np.asarray(eta, dtype=float)
        sorted_eta = eta[pi]
        Rbar_k = np.cumsum(sorted_eta) / k

    return RiskCoverageCurves(Rhat_k=Rhat_k, S_k=S_k, pi=pi, Rbar_k=Rbar_k)
