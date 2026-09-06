r"""Hoeffding's inequality, at a single fixed :math:`k`.

.. math::
    U_{k_0} = \hat R_{k_0} + \sqrt{\ln(1/\delta) / (2 k_0)}.

The paper's headline deployment finding (Section 5.3) is that Hoeffding survives post-hoc
selection at the paper's own scale ($n_{\mathrm{cal}}=2{,}000$) not because it's uniformly
valid -- it isn't -- but because it happens to carry more slack than the specific attack
(post-hoc argmin selection) can exploit at that sample size: half-width $0.039$ at
$k_0=1{,}000$ against an exploitable optimism of about $0.02$. See
``src/eval/selection_rules.py`` for the selection rules used to probe exactly this.
"""

from __future__ import annotations

import numpy as np


def upper_bound(Rhat_k0: float, k0: int, delta: float) -> float:
    """Hoeffding's upper confidence bound at one fixed prefix length."""
    return float(Rhat_k0 + np.sqrt(np.log(1.0 / delta) / (2.0 * k0)))


def half_width(k0: int, delta: float) -> float:
    """The additive term alone, i.e. $U_{k_0} - \\hat R_{k_0}$ -- useful for the
    "how much slack does this certificate carry" comparisons the paper makes directly
    (Section 5.3's arithmetic explanation of why Hoeffding survives selection but
    Clopper-Pearson doesn't, at the paper's own scale)."""
    return float(np.sqrt(np.log(1.0 / delta) / (2.0 * k0)))
