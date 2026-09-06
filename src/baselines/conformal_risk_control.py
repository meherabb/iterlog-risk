r"""Conformal risk control (Angelopoulos et al.): an in-expectation, not high-probability,
risk guarantee.

Reported in the paper's utility comparison (Section 5.6, Table 6) as a distinct category
from the high-probability methods -- $\mathbb E[\hat R_{\hat k}] \le \alpha$ is a materially
weaker promise than $\Pr[\hat R_{\hat k} \le \alpha] \ge 1-\delta$, so it is not a peer
comparison to Theorem 1 or 3, only a useful reference point for "how much does giving up the
high-probability guarantee buy you."
"""

from __future__ import annotations

import numpy as np


def select_threshold(Rhat_k: np.ndarray, alpha: float) -> int:
    r"""The conformal-risk-control selection rule: the largest $k$ such that a
    Robbins-Lai-style adjusted average keeps the *expected* risk at coverage below
    ``alpha``. Implemented here via the standard RCPS-style correction
    $\tilde R_k = \frac{k \hat R_k + 1}{k + 1}$ (a small, deterministic upward adjustment
    that makes the in-expectation guarantee hold exactly, rather than merely
    asymptotically), returning the largest $k$ with $\tilde R_k \le \alpha$.
    """
    n = Rhat_k.shape[0]
    k = np.arange(1, n + 1, dtype=float)
    adjusted = (k * Rhat_k + 1.0) / (k + 1.0)
    feasible = np.where(adjusted <= alpha)[0]
    if feasible.size == 0:
        return 0
    return int(feasible[-1] + 1)  # convert 0-indexed array position back to a prefix length
