r"""The three tiers of ground truth (Section "Experimental design"): three interchangeable
ways of generating the *loss* half of a (score, loss) sequence, sharing everything else
(scoring, sorting, banding) in common.

- **Tier A (exact $\eta$).** The predictor is "sample one answer at temperature $T$", so its
  conditional risk $\eta_i = 1 - p_T(y_i^\star \mid x_i)$ comes straight out of the model's
  own logits, and the loss is a *simulated* Bernoulli($\eta_i$) draw. Ground truth for
  $\bar R_k$ is therefore exact, and any violation falsifies the theorem outright.
- **Tier B (estimated $\eta$).** Genuine repeated decoding (32 samples/question at $T{=}1$
  on TriviaQA); $\hat\eta_i$ is the fraction wrong, checked via a leave-one-out curve built
  from the other 31 samples so no draw is checked against a curve that includes itself.
- **Tier C (held-out measurement).** No $\eta$ at all -- the genuine deployment protocol:
  split calibration/held-out, pick a threshold from the calibration curve, apply it to
  held-out data, and check the *observed* held-out error rate directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def tier_a_loss(eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate the Tier-A loss: a single Bernoulli($\\eta_i$) draw per item, using the
    model's own exact predictive probability. ``eta`` shape ``(n,)`` -> losses shape
    ``(n,)`` in ``{0, 1}``."""
    eta = np.asarray(eta, dtype=float)
    return rng.binomial(1, eta).astype(float)


@dataclass(frozen=True)
class TierBResult:
    eta_hat: np.ndarray  # leave-one-out estimated risk per item
    raw_loss: np.ndarray  # the actual held-back draw being checked, per item


def tier_b_leave_one_out(is_wrong: np.ndarray) -> TierBResult:
    r"""Tier B's leave-one-out construction.

    Parameters
    ----------
    is_wrong:
        Shape ``(n_items, n_samples)``, boolean/0-1 -- whether each of the ``n_samples``
        genuine decodes for an item was wrong.

    Returns
    -------
    TierBResult
        ``eta_hat[i]`` is the fraction wrong among samples *other than* a held-out one
        (here, sample 0 is held out and checked against the estimate from samples 1..n-1,
        matching "each draw is then checked, in turn, against the leave-one-out curve built
        from the other 31" -- callers wanting the full in-turn rotation over all
        ``n_samples`` choices of held-out draw should call this once per held-out index).
    """
    is_wrong = np.asarray(is_wrong, dtype=float)
    n_samples = is_wrong.shape[1]
    if n_samples < 2:
        raise ValueError("Tier B needs at least 2 samples per item for a leave-one-out estimate")
    held_out = is_wrong[:, 0]
    rest = is_wrong[:, 1:]
    eta_hat = rest.mean(axis=1)
    return TierBResult(eta_hat=eta_hat, raw_loss=held_out)


def tier_b_all_rotations(is_wrong: np.ndarray) -> TierBResult:
    """The full in-turn rotation: every one of the ``n_samples`` draws per item is, in
    turn, checked against the leave-one-out curve from the rest -- this is what the paper's
    own Tier-B replication count (320 = 10 arms x 32 genuine decode replicates) reflects,
    rather than holding out a single fixed sample per item.
    """
    is_wrong = np.asarray(is_wrong, dtype=float)
    n_items, n_samples = is_wrong.shape
    eta_hat = np.empty((n_items, n_samples))
    raw_loss = np.empty((n_items, n_samples))
    for h in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[h] = False
        eta_hat[:, h] = is_wrong[:, mask].mean(axis=1)
        raw_loss[:, h] = is_wrong[:, h]
    return TierBResult(eta_hat=eta_hat.ravel(), raw_loss=raw_loss.ravel())


@dataclass(frozen=True)
class DeploymentSplit:
    calibration_idx: np.ndarray
    held_out_idx: np.ndarray


def tier_c_split(n_total: int, n_calibration: int, rng: np.random.Generator) -> DeploymentSplit:
    """Tier C's calibration/held-out split -- a genuine deployment protocol, no $\\eta$
    involved at any point. ``n_calibration`` must leave at least a few thousand items for
    the held-out set (the paper uses $\\ge 3{,}000$ throughout)."""
    if n_calibration >= n_total:
        raise ValueError("calibration set must leave room for a held-out set")
    perm = rng.permutation(n_total)
    return DeploymentSplit(calibration_idx=perm[:n_calibration], held_out_idx=perm[n_calibration:])


def held_out_error_at_threshold(
    losses_held_out: np.ndarray, scores_held_out: np.ndarray, threshold: float
) -> float:
    """Tier C's final check: the observed error rate among held-out items whose score
    clears ``threshold`` -- ground truth by construction, no estimation involved."""
    losses_held_out = np.asarray(losses_held_out, dtype=float)
    scores_held_out = np.asarray(scores_held_out, dtype=float)
    selected = scores_held_out >= threshold
    if not np.any(selected):
        return float("nan")
    return float(losses_held_out[selected].mean())
