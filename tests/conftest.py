"""Shared pytest fixtures. Keep this file thin -- most tests are self-contained so they
read clearly next to the theorem they check."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def master_seed() -> int:
    """The paper's own master seed (Appendix J.1), for tests that want to match its
    determinism convention rather than an arbitrary literal."""
    return 20260727


@pytest.fixture
def rng(master_seed) -> np.random.Generator:
    return np.random.default_rng(master_seed)
