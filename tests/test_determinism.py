"""Tests for src/utils/determinism.py."""

from __future__ import annotations

from src.utils.determinism import MASTER_SEED_DEFAULT, derive_seed, rng_for


def test_same_arm_and_rep_always_gives_same_seed():
    s1 = derive_seed("tierA/qwen/pool/maxprob", 7)
    s2 = derive_seed("tierA/qwen/pool/maxprob", 7)
    assert s1 == s2


def test_different_rep_gives_different_seed():
    s1 = derive_seed("tierA/qwen/pool/maxprob", 7)
    s2 = derive_seed("tierA/qwen/pool/maxprob", 8)
    assert s1 != s2


def test_different_arm_gives_different_seed_even_at_same_rep():
    """Two different arms must not collide at the same rep_index -- this is what makes
    (arm_id, rep_index) jointly identify a replication, not rep_index alone."""
    s1 = derive_seed("tierA/qwen/pool/maxprob", 0)
    s2 = derive_seed("tierA/pythia/pool/maxprob", 0)
    assert s1 != s2


def test_master_seed_matches_paper():
    assert MASTER_SEED_DEFAULT == 20260727


def test_rng_for_produces_a_usable_generator():
    rng = rng_for("some_arm", 0)
    draw = rng.uniform()
    assert 0.0 <= draw <= 1.0
    # Same call again should reproduce the identical draw, since it's a fresh generator
    # seeded deterministically each time.
    rng2 = rng_for("some_arm", 0)
    assert rng2.uniform() == draw
