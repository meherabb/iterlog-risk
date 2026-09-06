r"""The iterated-logarithm lower bound (Theorem 2) and its empirical falsification sweep.

Theorem 2 (Iterated-logarithm lower bound). Consider the homogeneous regime
:math:`\eta \equiv p \in (0,1)` with :math:`s(X)` independent of :math:`\ell`, so
:math:`\bar R_k = p` for every :math:`k`. Let :math:`\{U_k\}` be any :math:`\delta`-uniformly
valid band. Then on an event of probability at least :math:`1 - \delta`,

.. math::
    \limsup_{k \to \infty} \frac{U_k - \hat R_k}{\sqrt{2p(1-p)\ln\ln k / k}} \geq 1.

This is a statement about *every* valid band, not a formula to evaluate -- there is nothing
to "compute" the way there is for Theorem 1. The paper's own way of making this checkable is
empirical falsification (Section 5.2, "The iterated logarithm is not slack"): shrink the band
by a constant factor :math:`c < 1` and watch the violation rate climb as :math:`n` grows,
tracking the :math:`\ln\ln n` rate rather than sitting flat. That sweep is exactly what
:func:`necessity_sweep` runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bands import uniform_band
from src.bands.validity import ViolationResult, clopper_pearson_ci_on_rate, is_violated


def lil_envelope(k: np.ndarray, p: float) -> np.ndarray:
    r"""The theorem's own rate, $\sqrt{2p(1-p)\ln\ln k / k}$, for reference/plotting.

    Undefined (returns ``nan``) for $k \le e$ where $\ln\ln k \le 0$; the theorem is an
    asymptotic ($k \to \infty$) statement and is not meant to be evaluated there.
    """
    k = np.asarray(k, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ll = np.log(np.log(k))
    envelope = np.sqrt(2.0 * p * (1.0 - p) * np.where(ll > 0, ll, np.nan) / k)
    return envelope


def shrunk_band(Rhat_k: np.ndarray, U_k: np.ndarray, c: float) -> np.ndarray:
    r"""$\hat R_k + c\,(U_k - \hat R_k)$ -- the deliberately-too-narrow band the sweep tests.

    ``c=1`` recovers the original band exactly; ``c<1`` is the "what if we shaved the width by
    a constant factor" experiment Section 5.2 runs to confirm the shaving isn't free.
    """
    Rhat_k = np.asarray(Rhat_k, dtype=float)
    U_k = np.asarray(U_k, dtype=float)
    return Rhat_k + c * (U_k - Rhat_k)


@dataclass(frozen=True)
class NecessitySweepResult:
    """One row of the necessity table (paper Table 14 / Section 5.2's in-text table).

    Attributes
    ----------
    c: the width fraction tested.
    n_reps, n_violations: raw counts.
    violation_rate: ``n_violations / n_reps``.
    ci95: exact Clopper-Pearson interval on the violation rate itself.
    """

    c: float
    n_reps: int
    n_violations: int
    violation_rate: float
    ci95: tuple[float, float]


def necessity_sweep(
    n: int,
    p: float,
    delta: float,
    k_min: int,
    c_values: tuple[float, ...],
    n_reps: int,
    master_seed: int = 20260727,
) -> list[NecessitySweepResult]:
    """Run the full sweep: for each width fraction ``c``, how often does the shrunk band break?

    Mirrors the paper's own construction exactly (Section 5.2): homogeneous world
    (:math:`\\eta \\equiv p`, score independent of loss), so :math:`\\bar R_k = p` for every
    :math:`k`, and the "violated" event is exactly :math:`p > \\hat R_k + c(U_k - \\hat R_k)`
    for some :math:`k \\geq k_{\\min}`.

    At the paper's own parameters (``n=20000, p=0.15, delta=0.05, k_min=100,
    c_values=(1.0, 0.75, 0.5, 0.25), n_reps=1000``) this should reproduce the paper's
    in-text table: violation rates approximately ``0.0000, 0.0030, 0.0980, 0.5940`` --
    see ``tests/test_lil_lower_bound.py::test_necessity_sweep_matches_paper_table``, which
    runs this exact call and checks the rates land within Monte Carlo noise of those values.
    """
    from src.bands import sorted_filtration

    out: list[NecessitySweepResult] = []
    for c in c_values:
        violations: list[ViolationResult] = []
        for rep in range(n_reps):
            rng = np.random.default_rng((master_seed, rep, round(c * 1000)))
            losses = rng.binomial(1, p, size=n).astype(float)
            scores = rng.normal(size=n)
            eta = np.full(n, p)

            pi = sorted_filtration.build_permutation(scores, rng)
            curves = sorted_filtration.realized_and_empirical_curves(losses, pi, eta=eta)
            U_k = uniform_band._band_from_curve(curves.Rhat_k, delta=delta)
            band_c = shrunk_band(curves.Rhat_k, U_k, c)

            violations.append(is_violated(curves.Rbar_k, band_c, k_min=k_min))

        n_viol = sum(v.violated for v in violations)
        rate = n_viol / n_reps
        ci = clopper_pearson_ci_on_rate(n_viol, n_reps)
        out.append(
            NecessitySweepResult(c=c, n_reps=n_reps, n_violations=n_viol, violation_rate=rate, ci95=ci)
        )
    return out
