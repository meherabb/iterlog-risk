r"""A Gaussian-copula sampler for inducing controllable within-block correlation.

Used throughout Section 5.4's block-correlation experiment: fix each item's marginal
conditional risk :math:`\eta_i` exactly, but induce within-block correlation
:math:`\rho \in \{0, 0.2, 0.4, 0.6, 0.8\}` in the *realized losses*, on two independent block
structures (the LLM pool's four source datasets; CIFAR-10's ten class labels) -- this is
what lets the paper (and this module's test) build a failure Theorem 1 actually predicts
(item-level certification should degrade as :math:`\rho` grows) and a fix Theorem 4 actually
delivers (block-level certification should not move at all).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def correlated_losses(
    eta: np.ndarray,
    block_of_item: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Draw losses with exact marginal $\mathbb E[\ell_i] = \eta_i$ and induced within-block
    correlation $\rho$, via a one-factor Gaussian copula.

    Construction: for each block $b$, draw a shared latent $Z_b \sim N(0,1)$; for each item
    $i$ in block $b$, draw an independent $\varepsilon_i \sim N(0,1)$ and form
    $L_i = \sqrt{\rho}\, Z_b + \sqrt{1-\rho}\, \varepsilon_i$, which is marginally
    $N(0,1)$ with $\mathrm{Corr}(L_i, L_j) = \rho$ for any two items $i \ne j$ in the same
    block. Passing $L_i$ through the standard normal CDF, $\Phi(L_i)$, gives a marginally
    $\mathrm{Unif}(0,1)$ variable correlated across the block in the same direction as
    $\rho$; thresholding $\ell_i = \mathbf 1\{\Phi(L_i) \le \eta_i\}$ then gives a Bernoulli
    draw with *exactly* $\Pr[\ell_i = 1] = \eta_i$ (since $\Phi(L_i)$ is exactly uniform,
    marginally, regardless of $\rho$), while correlating the *realizations* within a block.
    Items in different blocks are drawn independently, matching Theorem 4's assumption of
    mutual independence *across* blocks.

    At $\rho=0$ this reduces to independent Bernoulli($\eta_i$) draws (the ordinary Tier-A
    construction in :mod:`src.bands.sorted_filtration`); at $\rho \to 1$, all items in a
    block share (up to sign) the same latent draw and so realize together.

    Parameters
    ----------
    eta:
        Shape ``(n,)``, each item's true (marginal) conditional risk. Unaffected by ``rho``.
    block_of_item:
        Shape ``(n,)``, integer block id for each item (see
        :func:`src.bands.block_robust.assign_blocks`).
    rho:
        Target within-block correlation of the underlying Gaussian factor, in ``[0, 1]``.
    rng:
        Seeded generator, for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``, the drawn losses in ``{0.0, 1.0}``.
    """
    eta = np.asarray(eta, dtype=float)
    block_of_item = np.asarray(block_of_item)
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    n = eta.shape[0]
    n_blocks = int(block_of_item.max()) + 1

    z_block = rng.normal(size=n_blocks)[block_of_item]  # shared per-block latent, broadcast to items
    eps_item = rng.normal(size=n)
    latent = np.sqrt(rho) * z_block + np.sqrt(1.0 - rho) * eps_item
    u = stats.norm.cdf(latent)  # marginally Unif(0,1), correlated within a block

    return (u <= eta).astype(float)


def empirical_within_block_correlation(losses: np.ndarray, block_of_item: np.ndarray) -> float:
    """Diagnostic: the realized (Pearson) correlation between losses of distinct items in
    the same block. Purely descriptive -- used to sanity-check :func:`correlated_losses`
    against its target ``rho``, not part of any certificate.

    Implementation: collect every ordered pair of distinct items sharing a block, pool the
    ``(loss_i, loss_j)`` pairs across *all* blocks, and take the Pearson correlation of the
    pooled sample. Pooling first (rather than averaging a per-block statistic) avoids the
    high variance of estimating a correlation from a handful of items in a single block.
    """
    losses = np.asarray(losses, dtype=float)
    block_of_item = np.asarray(block_of_item)
    n_blocks = int(block_of_item.max()) + 1

    xs, ys = [], []
    for b in range(n_blocks):
        members = losses[block_of_item == b]
        n_b = len(members)
        if n_b < 2:
            continue
        # every ordered pair (i, j), i != j, within this block
        idx = np.arange(n_b)
        i_idx, j_idx = np.meshgrid(idx, idx, indexing="ij")
        mask = i_idx != j_idx
        xs.append(members[i_idx[mask]])
        ys.append(members[j_idx[mask]])

    if not xs:
        return float("nan")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
