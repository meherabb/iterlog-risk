r"""Hidden-state $k$-NN score -- the second hidden-state score family, using a held-out
reference set (Appendix J.1: 8,000 items, $k=10$, on a PCA-reduced 256-dimensional hidden
state) rather than a single centroid.

Larger score (a higher fraction of correct near neighbors) means more confident.
"""

from __future__ import annotations

import numpy as np


def fit_reference(
    hidden_states_ref: np.ndarray,
    is_correct_ref: np.ndarray,
    n_components: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA-reduce the reference hidden states and keep their correctness labels.

    Returns ``(projected_ref, is_correct_ref, components)`` where ``components`` (shape
    ``(d, n_components)``) is the PCA projection matrix, needed to project *new* items into
    the same reduced space in :func:`score`.
    """
    mean = hidden_states_ref.mean(axis=0)
    centered = hidden_states_ref - mean
    # Economy SVD for the projection -- avoids forming the full d x d covariance matrix.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T  # shape (d, n_components)
    projected_ref = centered @ components
    return projected_ref, is_correct_ref, components, mean


def score(
    hidden_states: np.ndarray,
    projected_ref: np.ndarray,
    is_correct_ref: np.ndarray,
    components: np.ndarray,
    ref_mean: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """``hidden_states`` shape ``(n_items, d)`` -> shape ``(n_items,)``, the fraction of the
    ``k`` nearest reference neighbors (in PCA-reduced space) that were themselves correct.
    """
    projected = (hidden_states - ref_mean[None, :]) @ components
    # Pairwise squared distances via the (a-b)^2 = a^2 - 2ab + b^2 expansion, vectorized.
    sq_norms_ref = np.sum(projected_ref**2, axis=1)
    sq_norms_new = np.sum(projected**2, axis=1)
    cross = projected @ projected_ref.T
    dist_sq = sq_norms_new[:, None] - 2.0 * cross + sq_norms_ref[None, :]

    nn_idx = np.argpartition(dist_sq, kth=min(k, dist_sq.shape[1] - 1), axis=1)[:, :k]
    nn_correct = is_correct_ref[nn_idx]  # shape (n_items, k)
    return nn_correct.mean(axis=1)
