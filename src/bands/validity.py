r"""Uniform validity (paper Definition 1).

    A band :math:`\{U_k\}_{k \geq k_{\min}}`, with :math:`U_k` a measurable function of
    :math:`(\hat R_1, \dots, \hat R_k)`, is :math:`\delta`-uniformly valid if
    :math:`\Pr[\bar R_k \leq U_k \text{ for all } k \geq k_{\min}] \geq 1 - \delta`.

This module doesn't compute anything a theorem proves -- it's the falsification harness:
given one realized trajectory of :math:`(\bar R_k, U_k)`, did the band hold everywhere it was
supposed to? Every validity table in the paper (Tables 3, 4, 5, 10, ...) is built by calling
:func:`is_violated` once per replication and averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViolationResult:
    """Outcome of checking one band trajectory against Definition 1.

    Attributes
    ----------
    violated:
        ``True`` if :math:`\\bar R_k > U_k` for at least one :math:`k \\geq k_{\\min}`.
    first_violation_k:
        The smallest such ``k`` (1-indexed, matching the paper), or ``None`` if never
        violated.
    max_excess:
        :math:`\\max_{k \\ge k_{\\min}} (\\bar R_k - U_k)`. Negative (or zero) exactly when
        not violated; this is what Appendix D.1's margin/slack analysis reports per
        replication.
    """

    violated: bool
    first_violation_k: int | None
    max_excess: float


def is_violated(Rbar_k: np.ndarray, U_k: np.ndarray, k_min: int = 100) -> ViolationResult:
    """Check Definition 1 for a single realized trajectory.

    Parameters
    ----------
    Rbar_k:
        Shape ``(n,)``, 0-indexed array where entry ``k - 1`` is :math:`\\bar R_k`.
    U_k:
        Shape ``(n,)``, same indexing, the band's value at each prefix length.
    k_min:
        The minimum prefix length the band (and every verdict in the paper) is evaluated
        from -- Appendix J.1 fixes this at 100 throughout.

    Returns
    -------
    ViolationResult
    """
    Rbar_k = np.asarray(Rbar_k, dtype=float)
    U_k = np.asarray(U_k, dtype=float)
    if Rbar_k.shape != U_k.shape:
        raise ValueError(f"shape mismatch: Rbar_k {Rbar_k.shape} vs U_k {U_k.shape}")
    n = Rbar_k.shape[0]
    if k_min > n:
        return ViolationResult(violated=False, first_violation_k=None, max_excess=-np.inf)

    # k is 1-indexed in the paper; array index k_min - 1 corresponds to k = k_min.
    excess = Rbar_k[k_min - 1 :] - U_k[k_min - 1 :]
    max_excess = float(np.max(excess))
    if max_excess <= 0:
        return ViolationResult(violated=False, first_violation_k=None, max_excess=max_excess)

    first_idx = int(np.argmax(excess > 0))
    first_k = first_idx + k_min
    return ViolationResult(violated=True, first_violation_k=first_k, max_excess=max_excess)


def violation_rate(results: list[ViolationResult]) -> float:
    """Fraction of replications violated -- the single number every validity table reports."""
    if not results:
        raise ValueError("no replications given")
    return sum(r.violated for r in results) / len(results)


def clopper_pearson_ci_on_rate(n_violations: int, n_reps: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson CI on a violation *rate* itself.

    Used to attach the "exact 95% CI" column that accompanies every violation-rate number
    in the paper's tables (e.g. Table 4's "[0.0000, 0.0037]" per-arm intervals). This is a
    thin, explicit wrapper around the beta-quantile identity so callers don't need to know
    that trick -- it is *not* the paper's own certificate (see ``src/baselines/clopper_pearson.py``
    for that); it is only used here, internally, to report uncertainty on a violation rate
    computed from a finite number of replications.
    """
    from scipy import stats

    alpha = 1 - conf
    lower = 0.0 if n_violations == 0 else stats.beta.ppf(alpha / 2, n_violations, n_reps - n_violations + 1)
    upper = (
        1.0
        if n_violations == n_reps
        else stats.beta.ppf(1 - alpha / 2, n_violations + 1, n_reps - n_violations)
    )
    return float(lower), float(upper)
