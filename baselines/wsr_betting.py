r"""The betting/WSR upper confidence bound (Waudby-Smith & Ramdas), its bisection-search
implementation, the silent-margin defect the paper audits (Appendix D.3), and the Lemma 1
patch that fixes it.

Betting-based confidence sequences invert a sequential test: for a candidate mean
:math:`m`, the wealth process :math:`K_t(m) = \prod_{i=1}^t (1 + \lambda_i (X_i - m))` is a
nonnegative martingale under :math:`\mathbb E[X_i] = m`, so by Ville's inequality
:math:`\Pr[\exists t : K_t(m) \geq 1/\delta] \leq \delta`. The confidence set at time
:math:`t` is :math:`\{m : K_s(m) < 1/\delta \text{ for all } s \leq t\}`, and the upper bound
is its right endpoint, found here by bisection.

**The bug, reproduced deliberately.** Our own bisection search begins at the sample mean
and, if the sequential test already rejects right there, returns that starting point
directly -- zero margin -- rather than refining further (Appendix D.3, "The betting bound
before and after the patch"). This fires far more often on real, heterogeneous-risk
(score-sorted, non-stationary) data than on i.i.d. synthetic data of the same size, because
the wealth process is more likely to have already spiked past :math:`1/\delta` by the time
the naive search starts at a point that is itself downstream of that spike.

**The patch (Lemma 1).** :math:`\max(A, B)` is :math:`\delta`-valid whenever :math:`B` is,
regardless of what :math:`A` is or how it was computed -- so taking the max of the (possibly
buggy) betting bound with an exact Clopper-Pearson bound at the same :math:`\delta` is
provably valid, with no new derivation needed. See :func:`patched_upper_bound`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.baselines import clopper_pearson


def _predictable_lambda(mean_so_far: float, var_so_far: float, m: float, cap: float = 0.5) -> float:
    r"""A simple predictable betting fraction, signed and scaled by how far the running
    mean sits from the candidate ``m``: $\lambda \propto (\hat\mu_{t-1} - m) / \hat\sigma^2_{t-1}$,
    capped so $1 + \lambda(X - m) > 0$ always holds for $X \in [0, 1]$.

    The sign is what actually does the work: if the running mean sits *below* ``m`` (``m``
    is too high a guess), $\lambda < 0$ so the bet profits exactly when a new draw is also
    below ``m`` -- growing the wealth process when the data keeps disconfirming ``m``, and
    shrinking it if a draw surprises upward. This is a plug-in ("aGRAPA"-style) tuning, not
    the only valid choice -- the point of this module isn't to pin down one canonical
    tuning, but to have a genuine, valid, reasonably-powered betting test to invert, so the
    *defect* being reproduced is a property of the bisection-search wrapper, not an artifact
    of a degenerate test.
    """
    var_floor = max(var_so_far, 1e-4)
    lam = (mean_so_far - m) / var_floor
    return float(np.clip(lam, -cap, cap))


def log_wealth_path(data: np.ndarray, m: float) -> np.ndarray:
    r"""$\ln K_t(m)$ for $t = 1, \dots, n$, using the predictable mixture of
    :func:`_predictable_lambda` re-estimated online from the data seen so far (never
    peeking ahead -- $\lambda_i$ depends only on $X_1, \dots, X_{i-1}$, keeping $K_t(m)$ a
    genuine martingale under $\mathbb E[X]=m$).
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    log_wealth = np.empty(n)
    running_sum = 0.0
    running_sumsq = 0.0
    log_k = 0.0
    for t in range(1, n + 1):
        if t == 1:
            lam = 0.0  # no history yet; open with a neutral (zero) bet
        else:
            mean_so_far = running_sum / (t - 1)
            var_so_far = running_sumsq / (t - 1) - mean_so_far**2
            lam = _predictable_lambda(mean_so_far, var_so_far, m)
        increment = 1.0 + lam * (data[t - 1] - m)
        increment = max(increment, 1e-12)  # numerical floor; should not bind for valid lambda
        log_k += np.log(increment)
        log_wealth[t - 1] = log_k
        running_sum += data[t - 1]
        running_sumsq += data[t - 1] ** 2
    return log_wealth


def sequential_test_rejects(data: np.ndarray, m: float, delta: float) -> bool:
    """Does the wealth process for candidate mean ``m`` ever cross $\\ln(1/\\delta)$?"""
    return bool(np.any(log_wealth_path(data, m) >= np.log(1.0 / delta)))


@dataclass(frozen=True)
class WSRResult:
    upper_bound: float
    used_buggy_shortcut: bool


def upper_bound_buggy(data: np.ndarray, delta: float, search_hi: float = 1.0, tol: float = 1e-3) -> WSRResult:
    r"""The upper confidence bound via the paper's own (buggy) bisection implementation.

    Deliberately reproduces the defect under audit: the search begins at the sample mean
    and, if the test already rejects there, returns the sample mean itself -- a zero-margin
    "bound" that is *not* actually a valid upper confidence bound on the true mean, since
    nothing has certified that the true mean is not above it.
    """
    sample_mean = float(np.clip(data.mean(), 0.0, 1.0))

    if sequential_test_rejects(data, sample_mean, delta):
        # The bug: return the starting point directly instead of searching upward from it.
        return WSRResult(upper_bound=sample_mean, used_buggy_shortcut=True)

    lo, hi = sample_mean, search_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if sequential_test_rejects(data, mid, delta):
            hi = mid
        else:
            lo = mid
    return WSRResult(upper_bound=hi, used_buggy_shortcut=False)


def upper_bound_fixed(data: np.ndarray, delta: float, search_hi: float = 1.0, tol: float = 1e-3) -> float:
    """The same bisection search, but searching outward from ``search_hi`` downward to find
    the *smallest* non-rejected point, rather than starting from (and trusting) the sample
    mean -- the straightforward, non-buggy fix, kept separate from :func:`patched_upper_bound`
    below so both remediation strategies (fix the search vs. patch the output with Lemma 1)
    are available and separately testable.
    """
    sample_mean = float(np.clip(data.mean(), 0.0, 1.0))
    lo, hi = sample_mean, search_hi
    # Always bisect properly, regardless of whether the test rejects at the sample mean.
    if not sequential_test_rejects(data, hi, delta):
        return hi  # even the top of the search range isn't rejected
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if sequential_test_rejects(data, mid, delta):
            hi = mid
        else:
            lo = mid
    return hi


def patched_upper_bound(data: np.ndarray, delta: float, **kwargs) -> float:
    r"""Lemma 1's patch: $\max(A, B)$ is $\delta$-valid whenever $B$ is.

    Here $A$ is the (possibly buggy) betting bound and $B$ is an exact Clopper-Pearson
    bound at the same $\delta$ -- valid regardless of what the betting computation does,
    failure modes not yet found included. This is the paper's own one-line patch and needs
    no new derivation: see ``tests/test_wsr_betting.py::test_lemma1_patch_is_valid`` for a
    direct empirical check that the max-patched bound is never violated, even when the raw
    buggy bound is.
    """
    data = np.asarray(data, dtype=float)
    a = upper_bound_buggy(data, delta, **kwargs).upper_bound
    k = data.shape[0]
    S_k = int(round(data.sum()))
    b = clopper_pearson.upper_bound(S_k, k, delta)
    return max(a, b)
