r"""Sequence log-probability, length-normalized (Gao et al.'s standard MCQ scoring
convention) -- used both to *rank the answer options themselves* (which option the model
picks) and, separately here, as a *confidence score* on whichever option was picked."""

from __future__ import annotations

import numpy as np


def score(token_logprobs: list[np.ndarray]) -> np.ndarray:
    """``token_logprobs``: one array per item, each the per-token log-probabilities of the
    chosen answer's tokens under the model. Returns the length-normalized total
    log-probability per item, shape ``(n_items,)`` -- larger (less negative) means more
    confident.
    """
    return np.array([lp.mean() for lp in token_logprobs])
