r"""The uniform risk-coverage band (Theorem 1) and its explicit closed form (Corollary 1).

Theorem 1 (Uniform risk-coverage band). With probability at least :math:`1-\delta`,
simultaneously for every :math:`k \geq 1`,

.. math::
    \bar R_k \;\leq\; U_k \;:=\; \min_{j \geq 0}
        \frac{\hat R_k + \ell_j / (\lambda_j k)}{1 - a(\lambda_j)}.

built from the :math:`\lambda_j`-menu (Eq. "menu" in the paper):

.. math::
    \delta_j = \frac{6\delta}{\pi^2 (j+1)^2}, \qquad
    \ell_j = \ln(1/\delta_j), \qquad
    \lambda_j = \min\{1,\ \sqrt{2\ell_j / b^{\,j}}\},

with :math:`a(\lambda) = \frac{\lambda}{2(1 - \lambda/3)} \le 3/4` and geometric base
:math:`b = 2` (the paper's default; see Appendix J.1's ``geometric base`` ablation).

Proof sketch (full derivation in the paper's Appendix A.1): Freedman's supermartingale
construction applied to the martingale :math:`N_k = k\bar R_k - S_k` at a fixed
:math:`\lambda`, combined with Ville's maximal inequality, gives a bound at parameter
:math:`\lambda` holding for every :math:`k` at once with failure probability
:math:`\delta'`. Applying this at each :math:`\lambda_j` with budget :math:`\delta_j` and
union-bounding over :math:`j` spends exactly :math:`\delta` in total; the minimum over
:math:`j` then preserves validity at every :math:`k`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bands import sorted_filtration


def a_of_lambda(lam: np.ndarray | float) -> np.ndarray | float:
    r"""$a(\lambda) = \lambda / (2(1 - \lambda/3))$, so $a(\lambda) \le 3/4$ on $(0, 1]$.

    Equality holds exactly at :math:`\lambda = 1` (the paper is careful to state this as a
    non-strict bound after an earlier draft used a strict inequality that the construction's
    own boundary case, :math:`j \in \{0,1,2,3\}` at :math:`\delta=0.05`, violates).
    """
    lam = np.asarray(lam, dtype=float)
    return lam / (2.0 * (1.0 - lam / 3.0))


def lambda_menu(delta: float, j_max: int = 60, b: float = 2.0) -> dict[str, np.ndarray]:
    """Build the $\\lambda_j$-menu for $j = 0, \\dots,$ ``j_max`` (Eq. "menu").

    Parameters
    ----------
    delta:
        The overall failure probability budget, split geometrically across $j$ via the
        polynomial allocation $\\delta_j \\propto 1/(j+1)^2$ (so $\\sum_j \\delta_j = \\delta$
        exactly, by the Basel-sum identity $\\sum 1/(j+1)^2 = \\pi^2/6$).
    j_max:
        Truncation point. The menu is tuned so $\\lambda_j$ is tightest around
        $k \\approx b^j$; $j_{\\max}=60$ with $b=2$ covers prefixes up to $k \\approx 10^{18}$,
        far beyond anything in the paper's experiments (largest $n$ used is 50,000).
    b:
        Geometric base. The paper uses $b=2$ throughout except in the ablation reported in
        Appendix D ("synthetic validity and malpractice arms"), which also tries $b \\in
        \\{e, 1.5\\}$.

    Returns
    -------
    dict with keys ``delta_j``, ``ell_j``, ``lambda_j``, ``a_lambda_j``, each shape
    ``(j_max + 1,)``.
    """
    j = np.arange(0, j_max + 1, dtype=float)
    delta_j = 6.0 * delta / (np.pi**2 * (j + 1.0) ** 2)
    ell_j = np.log(1.0 / delta_j)
    # sqrt(2*ell_j / b^j) computed in log-space to avoid overflow in b**j for large j_max
    # (irrelevant for the library's default j_max=60, but keeps this robust for callers who
    # pass a much larger truncation point).
    log_ratio = np.log(2.0 * ell_j) - j * np.log(b)
    lambda_j = np.minimum(1.0, np.exp(0.5 * log_ratio))
    return {
        "delta_j": delta_j,
        "ell_j": ell_j,
        "lambda_j": lambda_j,
        "a_lambda_j": a_of_lambda(lambda_j),
    }


@dataclass(frozen=True)
class BandResult:
    """A computed band together with the curves it was built from.

    ``Rbar_k`` is ``None`` unless the true conditional risk was supplied (see
    :func:`compute`'s ``eta`` argument) -- it is only used to *check* validity
    (:mod:`src.bands.validity`), never to compute :math:`U_k` itself, which by construction
    (Definition 1) is a measurable function of the empirical curve alone.
    """

    U_k: np.ndarray
    Rhat_k: np.ndarray
    pi: np.ndarray
    Rbar_k: np.ndarray | None = None


def compute(
    losses: np.ndarray,
    scores: np.ndarray,
    delta: float = 0.05,
    k_min: int = 100,
    rng: np.random.Generator | None = None,
    eta: np.ndarray | None = None,
    j_max: int = 60,
    b: float = 2.0,
) -> BandResult:
    """End-to-end: sort by score, then compute $U_k$ for every prefix (Theorem 1).

    Parameters
    ----------
    losses, scores:
        Shape ``(n,)`` each, in original (pre-sort) order.
    delta:
        Failure probability budget, $\\delta \\in (0, 1)$.
    k_min:
        Minimum prefix length the band is reported from; entries below it are still
        computed (for continuity) but the paper never certifies below $k_{\\min}=100$.
    rng:
        Generator for tie-breaking (see :func:`src.bands.sorted_filtration.build_permutation`).
        A fresh default generator is created if omitted -- pass an explicit, seeded one for
        reproducibility.
    eta:
        Optional true conditional risk, for validity checking against $\\bar R_k$
        (Tier A / Tier B only; omit for a genuine deployment).
    j_max, b:
        Passed through to :func:`lambda_menu`.

    Returns
    -------
    BandResult
    """
    if rng is None:
        rng = np.random.default_rng()
    pi = sorted_filtration.build_permutation(scores, rng)
    curves = sorted_filtration.realized_and_empirical_curves(losses, pi, eta=eta)
    U_k = _band_from_curve(curves.Rhat_k, delta=delta, j_max=j_max, b=b)
    return BandResult(U_k=U_k, Rhat_k=curves.Rhat_k, pi=pi, Rbar_k=curves.Rbar_k)


def _band_from_curve(Rhat_k: np.ndarray, delta: float, j_max: int = 60, b: float = 2.0) -> np.ndarray:
    r"""The pure, curve-in/band-out core of Theorem 1 -- everything above is convenience.

    Vectorized over both $k$ (the prefix length, one entry per element of ``Rhat_k``) and
    $j$ (the menu index): for each $j$, the candidate bound
    $[\hat R_k + \ell_j/(\lambda_j k)] / [1 - a(\lambda_j)]$ is a simple function of $k$, so
    stacking all $j_{\max}+1$ candidates into a $(j_{\max}+1) \times n$ array and taking the
    elementwise minimum over the first axis gives $U_k = \min_j (\cdot)$ directly, with no
    Python-level loop over $k$.
    """
    Rhat_k = np.asarray(Rhat_k, dtype=float)
    n = Rhat_k.shape[0]
    k = np.arange(1, n + 1, dtype=float)

    menu = lambda_menu(delta, j_max=j_max, b=b)
    ell_j = menu["ell_j"][:, None]  # (j_max+1, 1)
    lambda_j = menu["lambda_j"][:, None]
    a_j = menu["a_lambda_j"][:, None]

    candidate = (Rhat_k[None, :] + ell_j / (lambda_j * k[None, :])) / (1.0 - a_j)
    return np.min(candidate, axis=0)


def explicit_form(Rhat_k: np.ndarray, delta: float) -> np.ndarray:
    r"""Corollary 1's closed form: a looser but explicit alternative to :func:`_band_from_curve`.

    .. math::
        \ell(k) = \ln\!\left(\frac{\pi^2 (\lceil \log_2 2k \rceil + 1)^2}{6\delta}\right),
        \qquad
        U^{\mathrm c}_k = \hat R_k + 4\sqrt{\ell(k)\hat R_k / k} + 8\ell(k)/k,

    valid whenever $k \hat R_k \geq 2\ell(k)$; entries not satisfying this are set to
    ``np.nan`` rather than silently returning an invalid number, since the corollary's
    guarantee simply does not apply there (small $k$, or $\hat R_k$ very close to 0).

    At the paper's own worked example -- $n=20{,}000$, $k=2{,}000$, $\hat R_k=0.10$,
    $\delta=0.05$ -- this returns a width of $0.118$, against $0.034$ for the exact
    min-form (:func:`_band_from_curve`); reproducing that comparison is
    ``tests/test_uniform_band.py::test_explicit_form_matches_paper_worked_example``.
    """
    Rhat_k = np.asarray(Rhat_k, dtype=float)
    n = Rhat_k.shape[0]
    k = np.arange(1, n + 1, dtype=float)

    ell_k = np.log(np.pi**2 * (np.ceil(np.log2(2 * k)) + 1) ** 2 / (6.0 * delta))
    band = Rhat_k + 4.0 * np.sqrt(ell_k * Rhat_k / k) + 8.0 * ell_k / k

    valid = (k * Rhat_k) >= (2.0 * ell_k)
    return np.where(valid, band, np.nan)
