"""Tests for src/utils/self_checks.py."""

from __future__ import annotations

import numpy as np

from src.utils import self_checks


def test_rng_stream_check_passes():
    assert self_checks.check_rng_stream_reproducibility()


def test_forward_pass_check_passes_for_deterministic_function():
    def deterministic(x):
        return x * 2 + 1

    assert self_checks.check_forward_pass_determinism(deterministic, np.array([1.0, 2.0, 3.0]))


def test_forward_pass_check_catches_nondeterminism():
    """The whole point of a self-check is to fail when it should -- verified directly here,
    not just that it passes on well-behaved input."""

    def nondeterministic(x):
        return x + np.random.default_rng().normal(size=x.shape)

    assert not self_checks.check_forward_pass_determinism(nondeterministic, np.array([1.0, 2.0]))


def test_pipeline_check_catches_nondeterminism():
    def nondeterministic_pipeline(seed):
        # ignores the seed entirely -- a real bug this check is meant to catch
        return np.random.default_rng().binomial(1, 0.5, size=100)

    assert not self_checks.check_full_pipeline_determinism(nondeterministic_pipeline)


def test_run_all_skips_checks_missing_their_inputs():
    results = self_checks.run_all()
    assert "rng_stream" in results
    assert "forward_pass" not in results
    assert "full_pipeline" not in results
