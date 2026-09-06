r"""The four post-hoc selection rules probed in the deployment protocol (Section 5.3).

Every certificate in ``src/baselines/`` is only as trustworthy as the rule used to pick a
prefix length from the calibration curve -- these four rules are what the deployment
experiment (Table 1) applies each certificate under.
"""

from __future__ import annotations

import numpy as np


def r1_argmin(Rhat_k: np.ndarray) -> int:
    r"""$r_1 = \operatorname{argmin}_k \hat R_k$ -- pick the empirically best-looking prefix.

    The rule built specifically to exploit whatever optimism a certificate's slack allows
    (paper's own description of $r_{\mathrm{adv}}$ aside, $r_1$ is already the most
    aggressive of the three *non-adversarial* rules).
    """
    return int(np.argmin(Rhat_k)) + 1  # +1: convert 0-indexed position to a 1-indexed prefix length


def r2_prime_target_level(Rhat_k: np.ndarray, alpha: float) -> int:
    r"""$r_2' = \operatorname{argmin}_k |\hat R_k - \alpha|$ -- the prefix closest to a
    target risk level.

    Supersedes the paper's originally pre-registered $r_2$ (largest $k$ with
    $\hat R_k \le \alpha$), which had no solution in 88.8% of replications for the paper's
    own models (Appendix C) since $\hat R_k$ never dips below $\alpha$ at all for those
    weaker models -- $r_2'$ is defined in every replication by construction.
    """
    return int(np.argmin(np.abs(Rhat_k - alpha))) + 1


def r3_fixed(n: int) -> int:
    r"""$r_3 = n_{\mathrm{cal}} / 2$ -- a *pre-registered*, fixed prefix length with no
    selection involved at all. The control condition: any failure here is not about
    selection, it's about the certificate itself (or population transfer -- see
    ``src/baselines/transfer_corrected.py``)."""
    return n // 2


def r_adversarial(Rhat_k: np.ndarray, certify_fn) -> int:
    r"""$r_{\mathrm{adv}}$: for a *specific* certificate, pick whichever $k$ minimizes that
    certificate's own reported bound -- i.e. hand the adversary knowledge of the certificate
    itself, not just the empirical curve.

    Parameters
    ----------
    Rhat_k:
        The empirical curve.
    certify_fn:
        A callable ``k -> float`` returning the certificate's upper bound at prefix length
        ``k`` (1-indexed) -- e.g. a closure over
        ``lambda k: clopper_pearson.upper_bound_at_fixed_k(Rhat_k, k, delta)``.
    """
    n = Rhat_k.shape[0]
    bounds = np.array([certify_fn(k) for k in range(1, n + 1)])
    return int(np.argmin(bounds)) + 1
