r"""The population, threshold-indexed band (Theorem 3), feasibility (Corollary 2), and the
joint risk-coverage certificate (Corollary 3).

Deployment fixes a score *threshold* :math:`t`, not a rank. Let
:math:`\phi(t) = \Pr[s(X) \geq t]` and :math:`\rho(t) = \Pr[s(X) \geq t,\ \ell = 1]`, so the
population selective risk is :math:`R(t) = \rho(t)/\phi(t)`, with empirical counterparts
:math:`\hat\phi(t) = k/n` and :math:`\hat\rho(t) = S_k/n` on a calibration set of size
:math:`n`, where :math:`k = |\{i : s(X_i) \geq t\}|`.

Theorem 3 (Population band). With :math:`\varepsilon_n(\delta) = \sqrt{\ln(4/\delta)/(2n)}`,
with probability at least :math:`1-\delta`, simultaneously for every threshold
:math:`t \in \mathbb{R}` with :math:`\hat\phi(t) > \varepsilon_n(\delta)`,

.. math::
    R(t) \;\leq\; U^{\mathrm{pop}}(t) \;:=\;
        \frac{\hat\rho(t) + \varepsilon_n(\delta)}{\hat\phi(t) - \varepsilon_n(\delta)}.

Unlike Theorem 1, this transfers to *fresh* data at a cost of :math:`O(n^{-1/2})` uniformly
over :math:`t` -- a DKW/Massart argument handles :math:`\hat\phi` directly (a VC class of
index 1); :math:`\hat\rho` needs the per-item embedding in :func:`_dkw_embedding` since it is
a sub-distribution function, not a distribution function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def epsilon_n(n: int, delta: float) -> float:
    r"""$\varepsilon_n(\delta) = \sqrt{\ln(4/\delta) / (2n)}$ -- the DKW transfer cost.

    Note (Remark, "Implemented constant"): the paper's *released code* uses the smaller,
    single-event constant $\sqrt{\ln(2/\delta)/(2n)}$ rather than the two-event union bound's
    $\sqrt{\ln(4/\delta)/(2n)}$; the theorem as stated in the paper uses the correct, larger
    constant, and every validity conclusion the paper reports is therefore conservative
    (a larger $\varepsilon_n$ only widens the band). This function implements the
    theorem-correct constant; use ``epsilon_n_as_implemented`` below to match the paper's
    own (slightly optimistic, but empirically still valid) released numbers exactly.
    """
    return float(np.sqrt(np.log(4.0 / delta) / (2.0 * n)))


def epsilon_n_as_implemented(n: int, delta: float) -> float:
    r"""$\sqrt{\ln(2/\delta)/(2n)}$ -- matches the paper's own released-code constant exactly.

    At $n=2{,}000$: $0.0304$ (paper: "$0.0304$ against $0.0329$"); see
    ``tests/test_population_band.py::test_epsilon_n_matches_paper_remark``.
    """
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * n)))


@dataclass(frozen=True)
class PopulationBandResult:
    U_pop: np.ndarray
    phi_hat: np.ndarray
    rho_hat: np.ndarray
    eps: float
    defined: np.ndarray  # True where phi_hat > eps, i.e. the band is meaningful there


def compute(
    Rhat_k: np.ndarray, n: int, delta: float, use_paper_constant: bool = False
) -> PopulationBandResult:
    r"""Theorem 3's band, evaluated at every observed prefix $k = 1, \dots, n$ as a stand-in
    for "every threshold" (since on a finite sample, distinct thresholds only ever produce
    the finitely many distinct prefixes of the score-sorted sequence).

    Parameters
    ----------
    Rhat_k:
        Shape ``(n,)``, the empirical curve from a *calibration* set of size $n$ (see
        :mod:`src.bands.sorted_filtration`).
    n:
        Calibration-set size. Must equal ``len(Rhat_k)``.
    delta:
        Failure probability budget.
    use_paper_constant:
        If ``True``, use :func:`epsilon_n_as_implemented` (the paper's released-code
        constant) instead of the theorem-correct :func:`epsilon_n`. Default ``False``
        (theorem-correct, conservative).
    """
    Rhat_k = np.asarray(Rhat_k, dtype=float)
    if Rhat_k.shape[0] != n:
        raise ValueError(f"n={n} must equal len(Rhat_k)={Rhat_k.shape[0]}")

    eps = epsilon_n_as_implemented(n, delta) if use_paper_constant else epsilon_n(n, delta)
    k = np.arange(1, n + 1, dtype=float)
    phi_hat = k / n
    rho_hat = (k * Rhat_k) / n  # = S_k / n

    defined = phi_hat > eps
    with np.errstate(divide="ignore", invalid="ignore"):
        U_pop = (rho_hat + eps) / (phi_hat - eps)
    U_pop = np.where(defined, U_pop, np.nan)

    return PopulationBandResult(U_pop=U_pop, phi_hat=phi_hat, rho_hat=rho_hat, eps=eps, defined=defined)


def feasibility_threshold(
    alpha: float, n: int, delta: float, gamma: float = 1.0, use_paper_constant: bool = False
) -> float:
    r"""Corollary 2: the largest $\hat R_k$ at which $U^{\mathrm{pop}}(t) \le \alpha$ is still
    achievable at coverage $\gamma = \hat\phi(t)$.

    .. math::
        U^{\mathrm{pop}}(t) \le \alpha
        \iff
        \hat R_k \le \alpha - (1+\alpha)\,\varepsilon_n(\delta)/\gamma.

    The necessary condition $\hat R_k \le \alpha - (1+\alpha)\varepsilon_n(\delta)$
    (``gamma=1``) is the tightest-possible version, attained only as coverage $\to 1$; any
    finite target coverage $\gamma < 1$ is strictly tighter by a factor $1/\gamma$ on the
    second term, exactly as the paper's Appendix B (Table 2) tabulates.

    At $\alpha=0.10, \delta=0.05, n=2{,}000, \gamma=1$ (the paper's own implemented
    constant) this returns $\approx 0.067$; see
    ``tests/test_population_band.py::test_feasibility_matches_paper_table``.
    """
    eps = epsilon_n_as_implemented(n, delta) if use_paper_constant else epsilon_n(n, delta)
    return alpha - (1.0 + alpha) * eps / gamma


def coverage_floor(phi_hat: np.ndarray, n: int, delta: float, use_paper_constant: bool = False) -> np.ndarray:
    r"""Corollary 3 (joint risk-coverage certificate): $\phi(t) \geq \hat\phi(t) - \varepsilon_n(\delta)$,
    on the *same* event as Theorem 3 -- this costs nothing extra since the same two-sided
    DKW bound already controls $|\hat\phi(t) - \phi(t)|$ and Theorem 3 only used one
    direction of it.
    """
    eps = epsilon_n_as_implemented(n, delta) if use_paper_constant else epsilon_n(n, delta)
    return np.asarray(phi_hat, dtype=float) - eps


def _dkw_embedding(scores: np.ndarray, losses: np.ndarray) -> np.ndarray:
    r"""The per-item embedding behind the DKW extension to $\hat\rho$ (paper's Appendix A.3).

    Constructs :math:`W_i = \phi(s(X_i))` if :math:`\ell_i = 1`, else :math:`-1`, using the
    increasing bijection :math:`\phi(u) = u/(1+|u|)` mapping :math:`\mathbb{R} \to (-1, 1)`.
    Built entirely *per item* (no reference to the rest of the sample -- unlike two earlier,
    rejected attempts the paper documents in Remark "Two earlier attempts at this embedding"),
    so :math:`(W_i)` are i.i.d. by construction and DKW applies to them with no modification.

    The defining property this buys -- verified directly in
    ``tests/test_population_band.py::test_dkw_embedding_exact_equality`` -- is that for
    *every* real threshold :math:`t`, :math:`\{i : W_i \geq \phi(t)\}` equals exactly
    :math:`\{i : s(X_i) \geq t,\ \ell_i = 1\}`, with no restriction on :math:`t` needed.
    """
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)
    phi_s = scores / (1.0 + np.abs(scores))
    return np.where(losses == 1, phi_s, -1.0)


def bijection_phi(u: np.ndarray | float) -> np.ndarray | float:
    """The bijection $\\phi(u) = u / (1 + |u|)$ used by :func:`_dkw_embedding`, exposed
    separately so callers can map a threshold $t$ to $\\phi(t)$ for the equality check
    above."""
    u = np.asarray(u, dtype=float)
    return u / (1.0 + np.abs(u))
