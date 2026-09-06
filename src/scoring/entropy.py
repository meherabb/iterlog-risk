r"""Predictive entropy -- larger (negated) entropy means more confident (a single, peaked
distribution has low entropy, so we negate it to make the score's "larger is more
confident" convention consistent across all seven score families)."""

from __future__ import annotations

import numpy as np


def score(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """``probs`` shape ``(n_items, n_classes)`` -> shape ``(n_items,)``, negative entropy."""
    probs = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    return -entropy  # negate: lower entropy (more peaked/confident) -> higher score
