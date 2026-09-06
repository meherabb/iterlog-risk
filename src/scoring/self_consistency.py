r"""Self-consistency agreement (Wang et al.) -- Tier B only, since it needs genuine repeated
decoding to be meaningful (Section 4: "self-consistency appears only in Tier B, where genuine
repeated decoding makes it meaningful to compute").

Larger score (a higher fraction of samples agreeing with the modal answer) means more
confident.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def score(sampled_answers: list[list[str]]) -> np.ndarray:
    """``sampled_answers``: one list of decoded answer strings per item (length = number of
    samples drawn, e.g. 32 per the paper's Tier B protocol). Returns shape ``(n_items,)``,
    the fraction of samples matching the modal (most common) answer for that item.
    """
    scores = np.empty(len(sampled_answers))
    for i, samples in enumerate(sampled_answers):
        if not samples:
            scores[i] = 0.0
            continue
        counts = Counter(samples)
        scores[i] = counts.most_common(1)[0][1] / len(samples)
    return scores


def modal_answer(sampled_answers: list[str]) -> str:
    """The single most-common decoded answer for one item -- this is also what Tier B's
    leave-one-out loss estimate is checked against (see ``src/eval/tier_protocols.py``)."""
    return Counter(sampled_answers).most_common(1)[0][0]
