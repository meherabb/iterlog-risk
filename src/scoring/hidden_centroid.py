r"""Hidden-state centroid distance (Azaria & Mitchell; Burns et al.'s line of work) --
one of two hidden-state score families, found in the paper's own Appendix G contrasts to be
the tightest-certifying score on its pool (about 0.038 narrower certified width than
max-softmax-probability, at $10\%$ coverage).

The score is the (negative) distance from a hidden-state representation to a reference
"correct-answer" centroid built from a held-out calibration slice -- larger score (smaller
distance) means more confident.
"""

from __future__ import annotations

import numpy as np


def fit_centroid(hidden_states_correct: np.ndarray) -> np.ndarray:
    """Build the reference centroid from a held-out set of hidden states known to
    correspond to correct answers. ``hidden_states_correct`` shape ``(n_ref, d)`` ->
    returns shape ``(d,)``.
    """
    return hidden_states_correct.mean(axis=0)


def score(hidden_states: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """``hidden_states`` shape ``(n_items, d)``, ``centroid`` shape ``(d,)`` -> shape
    ``(n_items,)``, the negative Euclidean distance to the centroid (larger = closer =
    more confident)."""
    diff = hidden_states - centroid[None, :]
    dist = np.linalg.norm(diff, axis=-1)
    return -dist
