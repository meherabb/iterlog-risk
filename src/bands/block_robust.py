r"""The block-robust band (Theorem 4).

Theorem 4 (Block-robust band). Partition the population into blocks
:math:`B_1, B_2, \dots`, fixed in advance and mutually independent conditional on
covariates, with arbitrary dependence permitted *within* a block. Sorting blocks by mean
score and applying Theorem 1 to the block-indexed sequence of within-block mean losses
gives a band valid at every block-count :math:`k`, at the identical functional
form (Eq. "band"), with no correction for whatever dependence lives inside a block.

The whole proof reduces to one step: Theorem 1 only ever uses
:math:`\mathrm{Var}(\ell_{\pi(k)} \mid \mathcal{F}_{k-1}) \le \eta_{\pi(k)}`, which for a
:math:`\{0,1\}`-valued item is immediate. For a block *average* (in :math:`[0,1]`, not
:math:`\{0,1\}`, and possibly an arbitrarily correlated mixture internally), the identical
inequality holds by Bhatia-Davis (:func:`block_variance_bound`) regardless of the
correlation structure inside the block -- so the same construction goes through verbatim on
the block-indexed sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bands import uniform_band


def block_variance_bound(mu: np.ndarray | float) -> np.ndarray | float:
    r"""Bhatia-Davis: for any $Z \in [0,1]$ with mean $\mu$, $\mathrm{Var}(Z) \le \mu(1-\mu)$.

    This is the one property Theorem 1's proof needs and the one thing Theorem 4 has to
    re-establish for a block *average* rather than a single $\{0,1\}$ item. Elementary
    proof (reproduced here since it's a one-liner): $Z(1-Z) \ge 0$ pointwise since
    $Z \in [0,1]$, so $\mathbb E[Z^2] \le \mathbb E[Z] = \mu$, and
    $\mathrm{Var}(Z) = \mathbb E[Z^2] - \mu^2 \le \mu - \mu^2 = \mu(1-\mu)$.

    Tight exactly when a block's internal correlation is perfect (entirely-right-or-
    entirely-wrong behaves as one Bernoulli trial); independence inside a block only
    strictly *helps* (pushes the true variance below this bound), so this is the correct
    worst-case bound to build the martingale argument on, not merely a convenient one.
    """
    mu = np.asarray(mu, dtype=float)
    return mu * (1.0 - mu)


@dataclass(frozen=True)
class BlockAssignment:
    """A fixed-in-advance partition into blocks, plus each block's per-item membership.

    Construct via :func:`assign_blocks`; this is deliberately a thin, explicit structure
    rather than something inferred implicitly, since Theorem 4's guarantee specifically
    requires the block structure to be *fixed in advance* -- committing to it is part of
    what the code should make visible, not something that happens implicitly inside a
    scoring function.
    """

    block_of_item: np.ndarray  # shape (n,), block id for each item
    n_blocks: int


def assign_blocks(block_labels: np.ndarray) -> BlockAssignment:
    """Wrap arbitrary block labels (e.g. source-dataset name, class id) into a
    :class:`BlockAssignment`, remapping labels to ``0..n_blocks-1`` internally."""
    block_labels = np.asarray(block_labels)
    unique_labels, block_of_item = np.unique(block_labels, return_inverse=True)
    return BlockAssignment(block_of_item=block_of_item, n_blocks=len(unique_labels))


def block_level_curves(
    losses: np.ndarray,
    block_scores: np.ndarray,
    assignment: BlockAssignment,
    eta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    r"""Reduce item-level data to the block-indexed sequence Theorem 4 actually certifies.

    Parameters
    ----------
    losses:
        Shape ``(n,)``, item-level losses.
    block_scores:
        Shape ``(n_blocks,)``, one *mean score* per block, used to sort blocks (blocks are
        sorted by mean score, exactly as items are sorted by score in the item-level case).
    assignment:
        From :func:`assign_blocks`.
    eta:
        Optional item-level true conditional risk, for validity checking.

    Returns
    -------
    (block_losses, block_scores_sorted_input, block_eta) where ``block_losses[j]`` is the
    *unweighted* within-block mean loss for block ``j`` (in original block-id order, before
    sorting) -- sorting itself happens inside :func:`compute`, mirroring
    :mod:`src.bands.sorted_filtration` exactly but at the block level.

    Note on estimands (paper's Remark, "What block-count k certifies"): this is the
    unweighted average of each selected block's own internal mean loss, *not* the
    item-weighted risk pooled across all items in those blocks. The two coincide only when
    blocks are equal-sized.
    """
    losses = np.asarray(losses, dtype=float)
    n_blocks = assignment.n_blocks
    block_losses = np.array([losses[assignment.block_of_item == j].mean() for j in range(n_blocks)])
    block_eta = None
    if eta is not None:
        eta = np.asarray(eta, dtype=float)
        block_eta = np.array([eta[assignment.block_of_item == j].mean() for j in range(n_blocks)])
    if block_scores.shape[0] != n_blocks:
        raise ValueError(f"block_scores has {block_scores.shape[0]} entries, expected {n_blocks}")
    return block_losses, block_scores, block_eta


def compute(
    block_losses: np.ndarray,
    block_scores: np.ndarray,
    delta: float = 0.05,
    k_min: int = 1,
    rng: np.random.Generator | None = None,
    block_eta: np.ndarray | None = None,
) -> uniform_band.BandResult:
    r"""Theorem 4: apply Theorem 1 verbatim to the block-indexed sequence.

    This is deliberately *not* a separate reimplementation -- the whole content of the
    theorem is that no new machinery is needed once :func:`block_variance_bound` is
    established, so this function is a thin call into
    :func:`src.bands.uniform_band.compute` with items replaced by blocks. ``k_min`` defaults
    to 1 rather than 100 here since the paper's own block-correlation experiments use as few
    as 4 blocks (the LLM pool's four source datasets) -- far below the item-level
    $k_{\min}=100$ convention.
    """
    if rng is None:
        rng = np.random.default_rng()
    return uniform_band.compute(
        losses=block_losses,
        scores=block_scores,
        delta=delta,
        k_min=k_min,
        rng=rng,
        eta=block_eta,
    )
