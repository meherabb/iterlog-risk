r"""Transfer-corrected Clopper-Pearson and Hoeffding: give the fixed-:math:`k` baselines the
same population-transfer allowance Theorem 3 pays for, at a *single* threshold.

Used in Section 5.3 ("Two failures, disentangled") to isolate how much of Clopper-Pearson's
and Hoeffding's deployment failure comes from the *transfer* gap (certifying the calibration
prefix's risk rather than the population risk above a fixed threshold) versus from *selection*
(choosing that threshold after seeing the data). Both baselines certify the calibration
prefix by construction; giving them a single-point (not uniform-over-:math:`t`) concentration
allowance on :math:`\hat\phi` and :math:`\hat\rho` directly answers "how much of the failure
was ever about selection at all."
"""

from __future__ import annotations

import numpy as np

from src.baselines import clopper_pearson, hoeffding


def _single_point_epsilon(n: int, delta: float) -> float:
    r"""A single-point (not uniform-over-$t$) Hoeffding-style transfer allowance: split
    $\delta$ across the two one-sided bounds needed (upper on $\hat\rho$, lower on
    $\hat\phi$), so each uses budget $\delta/2$, giving
    $\varepsilon = \sqrt{\ln(2/\delta)/(2n)}$ per side.
    """
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * n)))


def clopper_pearson_transfer_corrected(Rhat_k: np.ndarray, k0: int, n: int, delta: float) -> float:
    """Clopper-Pearson at a fixed $k_0$, with the population-transfer allowance added on top
    of the calibration-prefix bound (rather than treating the calibration prefix's risk as
    if it already *were* the population risk above the corresponding threshold)."""
    eps = _single_point_epsilon(n, delta / 2.0)  # half the budget for the CP step itself
    S_k0 = int(round(k0 * Rhat_k[k0 - 1]))
    cp_bound = clopper_pearson.upper_bound(S_k0, k0, delta / 2.0)
    phi_hat = k0 / n
    # Transfer: rho <= rho_hat + eps and phi >= phi_hat - eps, both already folded into the
    # calibration-side CP bound via rho_hat = phi_hat * Rhat_k; apply the ratio-form correction
    # directly, matching Theorem 3's own U^pop functional form.
    rho_hat = phi_hat * cp_bound
    return float((rho_hat + eps) / max(phi_hat - eps, 1e-12))


def hoeffding_transfer_corrected(Rhat_k: np.ndarray, k0: int, n: int, delta: float) -> float:
    """Hoeffding at a fixed $k_0$, with the same style of transfer correction applied."""
    eps = _single_point_epsilon(n, delta / 2.0)
    hoeff_bound = hoeffding.upper_bound(Rhat_k[k0 - 1], k0, delta / 2.0)
    phi_hat = k0 / n
    rho_hat = phi_hat * hoeff_bound
    return float((rho_hat + eps) / max(phi_hat - eps, 1e-12))
