r"""The ResNet-18 / CIFAR-10 / CIFAR-10-C harness.

Shares the Tier-A *protocol* (:mod:`src.eval.tier_protocols`) with the LLM experiments --
same simulated-Bernoulli-from-exact-predictive-probability construction -- but none of the
scoring code, since a vision classifier's own softmax output plays the role every one of the
seven LLM score families plays there. Kept as a separate module because the CIFAR-10-C
severity sweep (the exchangeability stress test, Section 5.7) has no LLM-side analogue.
"""

from __future__ import annotations

import numpy as np

from src.eval import tier_protocols


def matched_control_arm(softmax_probs_correct_class: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""The CIFAR-10 Tier-A matched-control arm: identical construction to the 32 LLM
    Tier-A arms, applied to the vision model's own predictive probabilities.

    ``softmax_probs_correct_class`` shape ``(n,)``: the model's own softmax probability
    assigned to the *true* class for each image, i.e. $1 - \eta_i$ in the paper's notation
    ($\eta_i = 1 - p(y_i^\star \mid x_i)$, matching Tier A's LLM construction exactly).
    """
    eta = 1.0 - softmax_probs_correct_class
    return tier_protocols.tier_a_loss(eta, rng)


def exchangeability_violation_gap(eta_clean: np.ndarray, eta_corrupted: np.ndarray) -> float:
    """The mean conditional-risk gap the exchangeability stress test reports alongside each
    severity's violation rate (Table 7) -- simply $\\mathbb E[\\eta_{\\text{corrupted}}] -
    \\mathbb E[\\eta_{\\text{clean}}]$, kept as a named function since it's a headline
    diagnostic in its own right, not just an intermediate quantity."""
    return float(np.mean(eta_corrupted) - np.mean(eta_clean))
