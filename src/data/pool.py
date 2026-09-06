r"""Pool construction: combine the four MCQ datasets and cap at the paper's fixed pool size.

Appendix E.1: the pool combines ARC-Easy, ARC-Challenge, OpenBookQA, and CommonsenseQA
(24,706 items total across the four sources) and is capped at 24,000 items per replication
draw. MMLU is deliberately kept as a *separate* held-out group, never merged into the pool,
specifically so its subject labels can supply the group structure the group-conditional
study (Appendix F) needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POOL_CAP_DEFAULT = 24_000

# Item counts as reported in Appendix E.1's dataset table -- used only for the manifest /
# sanity check in fetch_all, not baked into any statistical computation.
DATASET_SIZES = {
    "arc_easy": 5_197,
    "arc_challenge": 2_590,
    "openbookqa": 5_957,
    "commonsense_qa": 10_962,
}


@dataclass(frozen=True)
class Pool:
    item_ids: np.ndarray  # indices into the concatenated source arrays
    source_of_item: np.ndarray  # which of the four datasets each item came from (as a string label)


def build_pool(
    source_sizes: dict[str, int],
    rng: np.random.Generator,
    cap: int = POOL_CAP_DEFAULT,
) -> Pool:
    """Concatenate the four sources and draw a capped, shuffled subset.

    Parameters
    ----------
    source_sizes:
        Mapping from dataset name to its available item count, e.g. ``DATASET_SIZES``
        (or the actual sizes returned by the real loaders in ``src/data/loaders.py`` once
        datasets are fetched -- kept as a parameter, not a hardcoded constant, so tests can
        exercise the capping logic at small sizes without downloading anything).
    rng:
        Seeded generator.
    cap:
        Maximum pool size per replication draw.
    """
    total = sum(source_sizes.values())
    labels = np.concatenate([np.full(size, name) for name, size in source_sizes.items()])
    global_ids = np.arange(total)

    if total <= cap:
        order = rng.permutation(total)
    else:
        order = rng.choice(total, size=cap, replace=False)

    return Pool(item_ids=global_ids[order], source_of_item=labels[order])


def block_labels_for(pool: Pool) -> np.ndarray:
    """Integer block ids (one per source dataset) for the block-robust-band experiments
    (Section 5.4), derived directly from :attr:`Pool.source_of_item` so the block structure
    used there is always the actual pool composition, never a separately-tracked copy that
    could drift out of sync with it."""
    _, block_of_item = np.unique(pool.source_of_item, return_inverse=True)
    return block_of_item
