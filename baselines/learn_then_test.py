r"""Learn-then-Test (Angelopoulos et al.): Bonferroni-corrected certification over a
pre-registered, finite grid of candidate thresholds.

The closest comparison point to this paper's own contribution (Section "Related work"):
LTT certifies a pre-registered grid with a multiplicity correction, tighter than the
uniform band at its own grid points -- but the grid has to be fixed *before* seeing the
data and certifies nothing between points, which is exactly the gap Theorem 1 closes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.baselines import clopper_pearson


@dataclass(frozen=True)
class LTTResult:
    grid_k: np.ndarray  # the pre-registered prefix lengths (grid points), sorted ascending
    upper_bounds: np.ndarray  # Bonferroni-corrected CP bound at each grid point


def certify_grid(Rhat_k: np.ndarray, grid_size: int, delta: float) -> LTTResult:
    r"""Certify an evenly-spaced grid of ``grid_size`` prefix lengths with a Bonferroni
    correction ($\delta / \text{grid\_size}$ per point), matching the paper's own
    LTT-B10/B100/B1000 baselines (10/100/1000-point grids).

    Only the grid points themselves are certified -- querying any other prefix length is
    not covered by this result, which is precisely the limitation Section 5.6 discusses
    ("fair, since it certifies nothing between points").
    """
    n = Rhat_k.shape[0]
    grid_k = np.unique(np.linspace(1, n, grid_size, dtype=int))
    delta_per_point = delta / grid_size
    bounds = np.array(
        [clopper_pearson.upper_bound(int(round(k * Rhat_k[k - 1])), k, delta_per_point) for k in grid_k]
    )
    return LTTResult(grid_k=grid_k, upper_bounds=bounds)


def certify_fixed_sequence(Rhat_k: np.ndarray, delta: float, alpha: float) -> int | None:
    r"""The fixed-sequence LTT variant: walk the grid from $k=n$ downward (highest coverage
    first) and stop at the first $k$ whose Clopper-Pearson bound exceeds ``alpha`` -- return
    the *last* $k$ that was still certified at level ``alpha``, spending the full $\delta$ at
    every step (no Bonferroni split, since a fixed-sequence test's validity comes from the
    stopping rule itself, not from splitting the budget across independent tests).

    Returns ``None`` if not even $k=n$ is certifiable at level ``alpha``.
    """
    n = Rhat_k.shape[0]
    best_k = None
    for k in range(n, 0, -1):
        bound = clopper_pearson.upper_bound(int(round(k * Rhat_k[k - 1])), k, delta)
        if bound <= alpha:
            best_k = k
            break
    return best_k
