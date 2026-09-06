r"""Maximum softmax probability (Guo et al.) -- the simplest of the seven score families.

Larger score = more confident = the model's own top predicted-class probability. Operates
on already-computed probabilities (or logits, via :func:`from_logits`) so it has no model
dependency and is fully testable on synthetic arrays.
"""

from __future__ import annotations

import numpy as np


def score(probs: np.ndarray) -> np.ndarray:
    """``probs`` shape ``(n_items, n_classes)`` -> shape ``(n_items,)``, the max over classes."""
    return np.max(probs, axis=-1)


def from_logits(logits: np.ndarray) -> np.ndarray:
    """Convenience: soft-max the logits first, then take the max probability."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return score(probs)
