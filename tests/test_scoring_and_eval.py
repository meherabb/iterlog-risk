"""Tests for src/scoring/*.py and src/eval/*.py (tier protocols, selection rules, pool,
copula) -- the supporting machinery around the core theorem implementations."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import pool as pool_mod
from src.data.copula import correlated_losses
from src.eval import selection_rules, tier_protocols
from src.scoring import entropy, hidden_centroid, hidden_knn, maxprob, self_consistency, verbalized


def test_maxprob_and_entropy_agree_on_ranking_direction():
    """A sharply peaked distribution should score as more confident than a near-uniform one
    under *both* score families, even though their numeric scales differ."""
    peaked = np.array([[0.97, 0.01, 0.01, 0.01]])
    uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
    probs = np.concatenate([peaked, uniform])
    assert maxprob.score(probs)[0] > maxprob.score(probs)[1]
    assert entropy.score(probs)[0] > entropy.score(probs)[1]


def test_self_consistency_perfect_agreement_scores_one():
    assert self_consistency.score([["A"] * 10])[0] == 1.0


def test_hidden_centroid_scores_closer_points_higher():
    rng = np.random.default_rng(2)
    centroid = np.zeros(8)
    close_point = rng.normal(scale=0.01, size=(1, 8))
    far_point = np.full((1, 8), 10.0)
    hs = np.concatenate([close_point, far_point])
    scores = hidden_centroid.score(hs, centroid)
    assert scores[0] > scores[1], "closer to the centroid should score higher (more confident)"


def test_hidden_centroid_fit_recovers_the_mean():
    rng = np.random.default_rng(3)
    ref = rng.normal(loc=5.0, size=(500, 4))
    centroid = hidden_centroid.fit_centroid(ref)
    assert centroid == pytest.approx(np.full(4, 5.0), abs=0.2)


def test_hidden_knn_all_correct_neighbors_scores_one():
    rng = np.random.default_rng(4)
    hs_ref = rng.normal(size=(300, 20))
    is_correct_ref = np.ones(300, dtype=bool)  # every reference item is "correct"
    projected_ref, labels, components, mean = hidden_knn.fit_reference(hs_ref, is_correct_ref, n_components=6)

    hs_new = rng.normal(size=(10, 20))
    scores = hidden_knn.score(hs_new, projected_ref, labels, components, mean, k=10)
    assert np.allclose(scores, 1.0), "every neighbor is correct, so every score should be 1.0"


def test_hidden_knn_score_reflects_local_neighbor_correctness():
    """Two well-separated clusters, one all-correct and one all-incorrect: a query point
    near the all-correct cluster should score near 1, and near the all-incorrect cluster
    near 0 -- checking the score reflects genuinely *local* structure, not a global average.
    """
    rng = np.random.default_rng(5)
    d = 10
    cluster_correct = rng.normal(loc=0.0, scale=0.1, size=(200, d))
    cluster_wrong = rng.normal(loc=20.0, scale=0.1, size=(200, d))
    hs_ref = np.concatenate([cluster_correct, cluster_wrong])
    is_correct_ref = np.concatenate([np.ones(200, dtype=bool), np.zeros(200, dtype=bool)])
    projected_ref, labels, components, mean = hidden_knn.fit_reference(hs_ref, is_correct_ref, n_components=5)

    query_near_correct = np.zeros((1, d))
    query_near_wrong = np.full((1, d), 20.0)
    queries = np.concatenate([query_near_correct, query_near_wrong])
    scores = hidden_knn.score(queries, projected_ref, labels, components, mean, k=10)
    assert scores[0] > 0.9
    assert scores[1] < 0.1


def test_verbalized_confidence_handles_percentage_and_fraction_forms():
    assert verbalized.parse_confidence("90%") == pytest.approx(0.9)
    assert verbalized.parse_confidence("0.9") == pytest.approx(0.9)
    assert verbalized.parse_confidence("I really don't know") is None


def test_pool_never_exceeds_cap_and_uses_all_items_when_under_cap():
    rng = np.random.default_rng(0)
    small_sizes = {"a": 10, "b": 10}
    pool = pool_mod.build_pool(small_sizes, rng, cap=100)
    assert len(pool.item_ids) == 20  # under cap -> keep everything

    large_sizes = {"a": 5000, "b": 5000}
    pool2 = pool_mod.build_pool(large_sizes, rng, cap=1000)
    assert len(pool2.item_ids) == 1000


def test_copula_preserves_marginal_mean_regardless_of_rho():
    rng = np.random.default_rng(5)
    n_blocks, items_per_block = 500, 8
    eta = np.full(n_blocks * items_per_block, 0.3)
    block_of_item = np.repeat(np.arange(n_blocks), items_per_block)
    for rho in (0.0, 0.5, 0.95):
        losses = correlated_losses(eta, block_of_item, rho, rng)
        assert losses.mean() == pytest.approx(0.3, abs=0.03)


def test_copula_rejects_invalid_rho():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        correlated_losses(np.array([0.5]), np.array([0]), rho=1.5, rng=rng)


def test_copula_empirical_correlation_diagnostic_tracks_rho():
    """The diagnostic used to sanity-check the copula against its own target: empirical
    within-block loss correlation should be near zero at rho=0 and clearly positive and
    increasing as rho grows toward 1 (not equal to rho itself -- the binary threshold
    transformation compresses it -- but monotonically tracking it)."""
    from src.data.copula import empirical_within_block_correlation

    rng = np.random.default_rng(11)
    n_blocks, items_per_block = 3000, 8
    eta = np.full(n_blocks * items_per_block, 0.2)
    block_of_item = np.repeat(np.arange(n_blocks), items_per_block)

    measured = []
    for rho in (0.0, 0.4, 0.8):
        losses = correlated_losses(eta, block_of_item, rho, rng)
        measured.append(empirical_within_block_correlation(losses, block_of_item))

    assert abs(measured[0]) < 0.05, "rho=0 should give near-zero empirical correlation"
    assert measured[0] < measured[1] < measured[2], "empirical correlation should increase with rho"


def test_selection_rule_r3_is_exactly_half_ncal():
    assert selection_rules.r3_fixed(2000) == 1000


def test_selection_rule_r1_picks_the_minimizing_prefix():
    Rhat_k = np.array([0.5, 0.3, 0.1, 0.4, 0.6])
    assert selection_rules.r1_argmin(Rhat_k) == 3  # index 2 (0-indexed) -> prefix length 3


def test_tier_a_loss_matches_bernoulli_mean_at_scale():
    rng = np.random.default_rng(0)
    eta = np.full(20_000, 0.3)
    losses = tier_protocols.tier_a_loss(eta, rng)
    assert losses.mean() == pytest.approx(0.3, abs=0.01)


def test_tier_c_split_has_no_overlap_and_correct_sizes():
    rng = np.random.default_rng(0)
    split = tier_protocols.tier_c_split(n_total=10_000, n_calibration=2_000, rng=rng)
    assert len(split.calibration_idx) == 2_000
    assert len(split.held_out_idx) == 8_000
    assert set(split.calibration_idx.tolist()).isdisjoint(split.held_out_idx.tolist())
